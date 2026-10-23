# Copyright 2025-2026 RadixArk, Inc. Modified for the RFT cookbook (see THIRD_PARTY_NOTICES.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE-APACHE at the repository root).
# Derived from miles' scripts/run_qwen3_6_35b_a3b_mtp.py at 8e99947 (rft_miles/miles.lock.json).
"""Qwen3.6-35B-A3B on upstream miles, with the training settings of the cookbook's verl recipe
for the same model (recipes/officeqa/verl/qwen3.6-35b-a3b_h100x16).

    python qwen3_6_35b_a3b.py --model-name Qwen3.6-35B-A3B --num-nodes 2 --num-gpus-per-node 8 \
        --hf-checkpoint <hf dir> --ckpt-dir <dir with qwen3.6-35B-A3B_torch_dist> --extra-args "..."

The model arguments are upstream's (scripts/models/qwen3.6-35B-A3B.py, added by execute_train). Its
options are those of upstream's GLM-5.3-Flash launcher (scripts/run_glm5_3_flash.py), which the
adapter drives the same way (rft_miles.render); the adapter's `--extra-args` follow and replace
any flag given here. Against upstream's Qwen3.6 launcher this one:

- trains no MTP layer and decodes without EAGLE speculation, as verl does: upstream turns both on
  with flags that a later flag cannot turn off;
- downloads nothing and converts nothing (the cookbook stages and converts the model);
- accepts any 8-GPU node (upstream's accepts only H200);
- sets the algorithm as verl's defaults have it: token-mean loss, dual-clip 3.0, weight decay
  0.01, Adam beta2 0.999, MoE auxiliary loss 1e-3 over the global batch (Megatron-Bridge's
  Qwen3.5/3.6 provider); the recipe's own settings (learning rate, KL, clip ratios, batch) come
  in `--extra-args`;
- trains and serves on separate GPUs, as verl's separate_async does: the first half of the nodes
  train with verl's parallelism (TP2, CP4, EP8 on each 8 GPUs), the second half serve 8-GPU SGLang
  engines (verl's vLLM rollout is TP8), and the weights go over NCCL after every update;
- computes log-probabilities with miles' fused kernels, without the fp32 copy of the logits that an
  80 GB H100 cannot hold at 24,576 tokens a GPU;
- fills request fields the client leaves out with OpenAI's defaults, never the model's
  generation_config.json (Qwen3.6's top_k=20 would apply silently);
- serves attention and MoE with SGLang's Triton kernels, which run with what the image carries on
  H100 and B300 alike: the image has no FlashAttention 3, and on B300 SGLang would choose FlashInfer's
  TRT-LLM kernels, which it carries no prebuilt copies of (building them at start-up crashed the
  compiler).

Training runs miles' train_async.py: the next rollout is generated while the current one trains,
so every rollout is sampled by the weights one update behind; verl's separate_async rollout may lag
the trainer by up to `train.max_off_policy_threshold` versions. Both recompute the old log-probs
with the trainer's current weights, without an importance-sampling correction (miles' and verl's
defaults).
"""
import os
from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils as U

MEGATRON_MODEL_TYPE = "qwen3.6-35B-A3B"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: str = "Qwen3.6-35B-A3B"
    num_nodes: int = 2                           # half train, half serve
    num_gpus_per_node: int = 8
    run_id: str = U.create_run_id()
    hf_checkpoint: str = "/root/models/Qwen3.6-35B-A3B"
    ckpt_dir: str = "/root/models"               # holds <MEGATRON_MODEL_TYPE>_torch_dist
    megatron_path: str = "/root/Megatron-LM"
    train_offload_dir: str = ""                  # the adapter's option; this model offloads to host memory
    num_rollout: int = 2
    rollout_max_response_len: int = 4096
    max_tokens_per_gpu: int = 24576              # x CP 4: a 98,304-token trajectory in one micro-batch
    extra_args: str = ""


def train_args(args: ScriptArgs) -> str:
    if args.num_nodes % 2 or args.num_gpus_per_node != 8:
        raise SystemExit("this layout (half the nodes train with TP2 x CP4 x EP8, half serve 8-GPU engines) needs "
                         f"an even number of 8-GPU nodes, got {args.num_nodes} x {args.num_gpus_per_node}")
    trainer_nodes = args.num_nodes // 2
    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ckpt_dir}/{MEGATRON_MODEL_TYPE}_torch_dist "
    rollout_args = (
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 1.0 "
        "--rollout-top-k -1 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
    )
    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.0 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--eps-clip-c 3.0 "
        "--calculate-per-token-loss "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.01 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.999 "
        "--clip-grad 1.0 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )
    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 4 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        # log-probs streamed from the logits: the default makes an fp32 copy of them (11 GiB at 24,576
        # tokens a GPU, beyond what an 80 GB H100 has left); over the true vocabulary, which Qwen3.6's
        # needs no padding for at TP2
        "--log-probs-backend fused "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-mem-fraction-static 0.8 "          # the rollout GPUs hold nothing else
        "--sglang-mamba-radix-cache-strategy extra_buffer "
        "--sglang-max-running-requests 256 "
        "--sglang-sampling-defaults openai "
        "--sglang-attention-backend triton "
        "--sglang-moe-runner-backend triton "
    )
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--moe-aux-loss-coeff 0.001 "
        "--moe-router-load-balancing-type global_aux_loss "
        f"--update-weight-buffer-size {2 * 1024**3} "
        f"--actor-num-nodes {trainer_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {(args.num_nodes - trainer_nodes) * args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--distributed-timeout-minutes 60 "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
    )
    return f"{ckpt_args}{rollout_args}{grpo_args}{optimizer_args}{perf_args}{sglang_args}{misc_args}{args.extra_args} "


def execute(args: ScriptArgs) -> None:
    extra_env_vars = {
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        # inductor's subprocess compile pool deadlocks under the torch_memory_saver LD_PRELOAD
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    # ray's runtime_env replaces the actor env; without the JIT cache dirs the kernels recompile every run
    for cache_var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        if os.environ.get(cache_var):
            extra_env_vars[cache_var] = os.environ[cache_var]
    args.create_backend().execute_train(
        train_args=train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train_async.py",   # the next rollout generates while the current one trains
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)

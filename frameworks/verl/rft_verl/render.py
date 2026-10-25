"""Resolve a verl recipe into the files one run needs. The launcher does this at the start of
every job (rft_verl.launch); nothing is written to the repository:

    recipe.yaml        the resolved recipe (with its hash)
    plan.json          every durable path of the run (rft_cookbook.air.jobs.plan)
    verl_args.json     Hydra overrides for `python -m verl.trainer.main_ppo`
    agent_loops.yaml   the agent-loop registry that routes rows to CookbookAgentLoop

Resolving fails on any inconsistency it can see (`check`): GPU splits that do not add up, a model
parallelism that does not divide the trainer GPUs, checkpoint settings the topology cannot use.
The recipe's AI Runtime jobs are checked in next to it
(`jobs/<stage>.yaml`); tests/test_jobs.py keeps them consistent with the recipe.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from rft_cookbook.air.jobs import plan
from rft_cookbook.air.stages import ABORT_FILE, task_resources
from rft_cookbook.recipe import load_recipe, recipes_root
from rft_cookbook.tasks import load_task
from rft_verl.agent_loop import AGENT_NAME

LAUNCHER = "frameworks/verl/launch/node.sh"   # every job's command: bash <snapshot>/<LAUNCHER> <stage>
PARITY_REFERENCE = "parity_reference.json"
GPU_STAGES = {"baseline_eval", "train", "eval"}   # the others run on one A10
GPUS_PER_NODE = {"GPU_8xH100": 8, "GPU_8xH200": 8, "GPU_8xB200": 8, "GPU_1xA10": 1, "GPU_1xH100": 1}
# What a stage's job sets besides the common variables. Megatron wants one CUDA connection for its
# communication/compute overlap; vLLM's symmetric-memory all-reduce is off, as in the verl image's
# qualified runs.
STAGE_ENV = {"train": {"CUDA_DEVICE_MAX_CONNECTIONS": "1", "VLLM_ALLREDUCE_USE_SYMM_MEM": "0"}}


def pools(compute: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    """((nodes, GPUs per node) of the trainer, the same of the rollout pool). verl places each pool
    node by node, so a pool either fits in one node or takes whole nodes."""
    per_node = GPUS_PER_NODE[compute["accelerator"]]

    def pool(gpus: int, name: str) -> tuple[int, int]:
        if gpus <= per_node:
            return 1, gpus
        if gpus % per_node:
            raise ValueError(f"{name} ({gpus}) larger than a node must be whole nodes of {per_node} GPUs")
        return gpus // per_node, per_node

    return pool(compute["trainer_gpus"], "trainer_gpus"), pool(compute["rollout_gpus"], "rollout_gpus")


def check(recipe: dict[str, Any]) -> None:
    compute, train = recipe["compute"], recipe["train"]
    per_node = GPUS_PER_NODE.get(compute["accelerator"])
    if per_node is None:
        raise ValueError(f"unknown accelerator {compute['accelerator']!r}")
    if compute["trainer_gpus"] + compute["rollout_gpus"] != per_node * compute["nodes"]:
        raise ValueError(f"trainer_gpus + rollout_gpus must equal {per_node * compute['nodes']} GPUs")
    (trainer_nodes, _), (rollout_nodes, _) = pools(compute)
    if compute["nodes"] > 1 and trainer_nodes + rollout_nodes != compute["nodes"]:
        raise ValueError("on several nodes the trainer and the rollout pool each take whole nodes")
    if compute["rollout_gpus"] % recipe["rollout"]["tensor_parallel"]:
        raise ValueError("rollout_gpus must be a multiple of rollout.tensor_parallel")
    strategy = recipe["strategy"]
    model_parallel = strategy["tp"] * strategy["pp"] * strategy["cp"]
    if compute["trainer_gpus"] % model_parallel:
        raise ValueError("trainer_gpus must be a multiple of tp x pp x cp")
    if train["save_freq"] <= 0:
        raise ValueError("train.save_freq must be positive: every run publishes checkpoints")
    if strategy["engine"] != "megatron":
        raise ValueError("only the megatron engine is rendered (Qwen3.5 needs Megatron-Bridge)")
    location = recipe["checkpoint"].get("location", "local")
    if location not in ("local", "volume"):
        raise ValueError("checkpoint.location is local or volume")
    if compute["nodes"] > 1 and location != "volume":
        # the trainer may run on another node than the driver, and TransferQueue saves from every node
        raise ValueError("on several nodes checkpoints are saved to the volume (checkpoint.location: volume)")


def verl_args(recipe: dict[str, Any], paths: dict[str, Any], task) -> list[str]:
    t, s, ro, c = recipe["train"], recipe["strategy"], recipe["rollout"], recipe["compute"]
    (trainer_nodes, trainer_per_node), (rollout_nodes, rollout_per_node) = pools(c)
    packed = bool(s.get("remove_padding", False))
    ck = recipe["checkpoint"]
    in_place = ck.get("location", "local") == "volume"
    budgets = task.budgets
    prompt, response = int(recipe["data"]["max_prompt_tokens"]), budgets.max_response_tokens
    episode = prompt + response
    use_kl = float(t["kl_loss_coef"]) > 0
    a, m, r = "actor_rollout_ref.actor", "actor_rollout_ref.actor.megatron", "actor_rollout_ref.rollout"
    args = [
        "model_engine=megatron",
        # algorithm
        f"algorithm.adv_estimator={t['algorithm']}", "algorithm.use_kl_in_reward=False",
        f"algorithm.norm_adv_by_std_in_grpo={bool(t['norm_adv_by_std'])}",
        # data: canonical rows converted by rft_verl.data; agent_name routes them to the cookbook loop
        f"data.train_files={paths['train_parquet']}", f"data.val_files={paths['eval_parquet']}",
        "data.prompt_key=prompt", "data.return_raw_chat=True", f"data.max_prompt_length={prompt}",
        f"data.max_response_length={response}", f"data.train_batch_size={t['parameter_sync_step'] * t['ppo_mini_batch_size']}",
        "data.gen_batch_size=1", "data.filter_overlong_prompts=True", "data.truncation=error", "data.shuffle=True",
        f"data.seed={t['seed']}",
        # model: sequences are packed (THD) only where the recipe says the model supports it
        f"actor_rollout_ref.model.path={paths['model_dir']}", "actor_rollout_ref.model.trust_remote_code=True",
        f"actor_rollout_ref.model.use_remove_padding={packed}", "actor_rollout_ref.hybrid_engine=False",
        f"actor_rollout_ref.model.use_fused_kernels={bool(s.get('fused_kernels', False))}",
        # actor
        f"{a}.optim.lr={t['lr']}", f"{a}.optim.lr_warmup_steps=-1", f"{a}.ppo_mini_batch_size={t['ppo_mini_batch_size']}",
        f"{a}.ppo_micro_batch_size_per_gpu=1", f"{a}.ppo_epochs={t['ppo_epochs']}", f"{a}.use_dynamic_bsz=False",
        f"{a}.ppo_max_token_len_per_gpu={episode}", f"{a}.use_kl_loss={use_kl}", f"{a}.kl_loss_coef={t['kl_loss_coef']}",
        f"{a}.kl_loss_type=low_var_kl", f"{a}.entropy_coeff={t['entropy_coeff']}",
        f"{a}.clip_ratio_low={t['clip_ratio_low']}", f"{a}.clip_ratio_high={t['clip_ratio_high']}",
        f"{a}.checkpoint.async_save=False",
        f"{m}.use_mbridge=True", f"{m}.use_remove_padding={packed}", f"{m}.tensor_model_parallel_size={s['tp']}",
        f"{m}.pipeline_model_parallel_size={s['pp']}", f"{m}.context_parallel_size={s['cp']}",
        f"{m}.expert_model_parallel_size={s['ep']}", f"{m}.expert_tensor_parallel_size={s['etp']}",
        f"{m}.dtype=bfloat16", f"{m}.entropy_from_logits_with_chunking=True", f"{m}.seed={t['seed']}",
        f"{m}.param_offload={bool(s['param_offload'])}", f"{m}.optimizer_offload={bool(s['optimizer_offload'])}",
        f"+{a}.optim.override_optimizer_config.optimizer_cpu_offload={bool(s['optimizer_offload'])}",
        f"+{a}.optim.override_optimizer_config.optimizer_offload_fraction={s['offload_fraction']}",
        f"+{a}.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True",
        f"+{a}.optim.override_optimizer_config.use_precision_aware_optimizer=True",
        f"+{m}.override_transformer_config.recompute_granularity=full",
        f"+{m}.override_transformer_config.recompute_method=uniform",
        f"+{m}.override_transformer_config.recompute_num_layers=1",
        f"++{m}.override_transformer_config.attention_backend=auto",
        # rollout: a standalone vLLM pool; the trainer pushes weights over NCCL
        f"{r}.name={ro['engine']}", f"{r}.mode=async", f"{r}.nnodes={rollout_nodes}", f"{r}.n_gpus_per_node={rollout_per_node}",
        f"{r}.tensor_model_parallel_size={ro['tensor_parallel']}", f"{r}.gpu_memory_utilization={ro['gpu_memory_utilization']}",
        f"{r}.n={t['rollout_n']}", f"{r}.temperature={t['temperature']}", f"{r}.dtype=bfloat16", f"{r}.seed={t['seed']}",
        f"{r}.calculate_log_probs=True", f"{r}.log_prob_micro_batch_size_per_gpu=1", f"{r}.log_prob_use_dynamic_bsz=False",
        f"{r}.log_prob_max_token_len_per_gpu={episode}", f"{r}.prompt_length={prompt}", f"{r}.response_length={response}",
        f"{r}.max_model_len={episode}", f"{r}.max_num_batched_tokens={max(episode, 8192)}", f"{r}.free_cache_engine=True",
        f"{r}.enable_chunked_prefill=True", f"{r}.enable_prefix_caching={bool(ro['prefix_caching'])}",
        f"{r}.enforce_eager={bool(ro['enforce_eager'])}", f"{r}.checkpoint_engine.backend=nccl",
        f"{r}.checkpoint_engine.update_weights_bucket_megabytes={ro['weight_bucket_mb']}",
        # multi-turn: verl parses tool calls; the cookbook loop applies the task's budgets
        f"{r}.multi_turn.enable=True", f"{r}.multi_turn.format={recipe['model']['tool_parser']}",
        f"{r}.multi_turn.max_assistant_turns={budgets.max_turns}", f"{r}.multi_turn.max_user_turns={budgets.max_turns}",
        f"{r}.multi_turn.max_parallel_calls={budgets.max_tool_calls_per_turn}",
        f"{r}.agent.num_workers={ro['agent_workers']}", f"{r}.agent.agent_loop_config_path={paths['agent_loops']}",
        # trainer: V1, separate_async
        "trainer.use_v1=True", "trainer.v1.trainer_mode=separate_async",
        f"trainer.v1.separate_async.parameter_sync_step={t['parameter_sync_step']}",
        f"trainer.v1.separate_async.num_warmup_batches={t['num_warmup_batches']}",
        f"trainer.v1.sampler.max_off_policy_threshold={t['max_off_policy_threshold']}",
        "transfer_queue.enable=True", f"trainer.nnodes={trainer_nodes}", f"trainer.n_gpus_per_node={trainer_per_node}",
        "trainer.critic_warmup=0", "trainer.total_epochs=1000", f"trainer.total_training_steps={t['total_training_steps']}",
        f"trainer.save_freq={t['save_freq']}", "trainer.test_freq=-1", "trainer.val_before_train=False",
        # saved on local disk and published as a copy, or saved straight into the store (in place)
        f"trainer.default_local_dir={paths['checkpoints'] if in_place else paths['local_checkpoints']}",
        "trainer.resume_mode=disable",
        f"trainer.max_actor_ckpt_to_keep={'null' if in_place else t['keep_local_checkpoints']}",
        f"trainer.logger={json.dumps(t['logger'])}", "trainer.project_name=rft-cookbook",
        f"trainer.experiment_name={paths['run_id']}",
        # every save is published to durable storage
        "trainer.checkpoint_callback_class=rft_verl.checkpoint.PublishCheckpoint",
        f"+rft.checkpoint_store={paths['checkpoints']}", f"+rft.checkpoint_hash={bool(ck['hash_files'])}",
        f"+rft.checkpoint_publish_failure={ck['publish_failure']}",
    ]
    if ck.get("keep"):
        args.append(f"+rft.checkpoint_keep={int(ck['keep'])}")
    if use_kl:   # the reference model shares the trainer GPUs and the actor's parallelism
        args += ["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
                 "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False"]
    if ro.get("disable_custom_all_reduce"):
        args.append(f"+{r}.engine_kwargs.vllm.disable_custom_all_reduce=True")
    return args


def agent_loops(recipe: dict[str, Any], paths: dict[str, Any]) -> list[dict[str, Any]]:
    entry = {"name": AGENT_NAME, "_target_": "rft_verl.agent_loop.CookbookAgentLoop", "task": recipe["task"],
             "budgets": dict(recipe.get("agent") or {}), "episode_dir": f"{paths['episodes']}/train",
             "model": {"id": recipe["model"]["id"], "revision": recipe["model"]["revision"]},
             "abort_file": f"{paths['run_dir']}/train/{ABORT_FILE}",
             "max_consecutive_exclusions": int(recipe["train"]["max_consecutive_exclusions"]),
             "judge_outage_wait_s": float(recipe["train"].get("judge_outage_wait_s", 1800))}
    resources = task_resources(recipe, paths)
    if resources and resources.get("judge", {}).get("concurrency"):
        # every agent-loop worker is its own process with its own judge: split the run's limit, rounding
        # down so the workers together stay within it (at least one each)
        workers = int(recipe["rollout"]["agent_workers"])
        judge = resources["judge"]
        resources = {**resources, "judge": {**judge, "concurrency": max(1, int(judge["concurrency"]) // workers)}}
    if resources:
        entry["resources"] = resources
    return [entry]


def resolve_run(ref: str, *, site: dict[str, Any], run_id: str, overrides: list[str] = (), out: Path) -> Path:
    recipe = load_recipe(ref, overrides=list(overrides))
    if recipe["framework"] != "verl":
        raise ValueError(f"{ref} is a {recipe['framework']} recipe")
    check(recipe)
    task = load_task(recipe["task"], **(recipe.get("agent") or {}))
    task.with_schema_view((recipe.get("eval") or {}).get("schema_view", "full"))   # an unknown view fails here
    paths = plan(recipe, site, run_id, task_version=task.version)
    paths.update({
        "train_parquet": f"{paths['data_dir']}/verl/train.parquet",
        "eval_parquet": f"{paths['data_dir']}/verl/eval.parquet",
        "local_checkpoints": f"{paths['local_dir']}/checkpoints",
        # the launcher replaces @RENDERED@ with the directory it resolved the run in
        "agent_loops": "@RENDERED@/agent_loops.yaml",
        "task_version": task.version, "task_settings": task.settings(),
    })
    out.mkdir(parents=True, exist_ok=True)
    (out / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    (out / "plan.json").write_text(json.dumps(paths, indent=1, sort_keys=True))
    (out / "verl_args.json").write_text(json.dumps(verl_args(recipe, paths, task), indent=1))
    (out / "agent_loops.yaml").write_text(yaml.safe_dump(agent_loops(recipe, paths), sort_keys=False))
    if recipe.get("parity"):
        reference = recipes_root() / ref / recipe["parity"]["reference"]
        (out / PARITY_REFERENCE).write_text(reference.read_text())
    return out

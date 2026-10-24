"""Resolve a miles recipe into the files one run needs. The launcher does this at the start of
every job (rft_miles.launch); nothing is written to the repository:

    recipe.yaml   the resolved recipe (with its hash)
    plan.json     every durable path of the run (rft_cookbook.air.jobs.plan)
    miles.json    upstream's model launcher, its own options and the arguments the cookbook adds

Training runs a launcher for the model (`model.launcher`): upstream's, under the pinned miles
source (e.g. scripts/run_glm5_3_flash.py, which knows the model's parallel layout, SGLang and
offload settings), or the cookbook's own under `rft_miles/launchers/` where upstream's cannot be
used as it is. The cookbook adds its hooks through the launcher's `--extra-args`: the prompts, the
agent function on session server v2 and its message matcher (rft_miles.session), the reward and
post-save hooks, the checkpoint directory, the recipe's batch sizes and its algorithm settings. A flag given again in `--extra-args`
replaces the launcher's own.

The stages that do not train (`SERVING_STAGES`, rft_cookbook.air.stages) run the same way for
every framework; evaluation serves the checkpoint with vLLM from the verl image, so a miles run
and a verl run of one model are evaluated alike and can share a baseline (`eval.baseline_run`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from rft_cookbook.air.jobs import plan
from rft_cookbook.air.stages import ABORT_FILE
from rft_cookbook.recipe import load_recipe
from rft_cookbook.tasks import load_task

LAUNCHER = "frameworks/miles/launch/node.sh"   # every job's command: bash <snapshot>/<LAUNCHER> <stage>
GPUS_PER_NODE = {"GPU_8xB300": 8, "GPU_8xB200": 8, "GPU_8xH200": 8, "GPU_8xH100": 8, "GPU_1xA10": 1}
MILES_STAGES = {"stage_model", "train"}   # the stages that need upstream miles (node.sh installs it)
SERVING_STAGES = {"baseline_eval", "eval"}   # vLLM from the verl image, as for every framework
COOKBOOK_LAUNCHERS = "rft_miles/launchers/"   # model.launcher under this prefix is the cookbook's own
GENERATE = "miles.rollout.generate_hub.agentic_tool_call.generate"
AGENT = "rft_miles.agent.run"
AGENT_MODE = "inline"   # on the rollout's event loop, not miles' default (subproc): see rft_miles.agent
REWARD = "rft_miles.reward.reward"
MATCHER = "rft_miles.session.matches"   # a replayed turn as the harness read it: one lineage per episode
POST_SAVE = "rft_miles.checkpoint.publish"
DIAGNOSTICS = "rft_miles.diagnostics.before_train_step"
HF_DIR = "hf"   # a bundle's Hugging Face weights (train.save_hf), next to Megatron's
LOCK = json.loads((Path(__file__).with_name("miles.lock.json")).read_text())   # the pinned upstream miles


def stage_job(recipe: dict[str, Any], stage: str) -> dict[str, Any]:
    """The image, compute and timeout of a stage's job: training takes every node, model staging
    and evaluation one node, the other stages one A10."""
    compute, ev = recipe["compute"], recipe.get("eval") or {}
    image = "<verl-image>" if stage in SERVING_STAGES else "<miles-image>"
    if stage == "train":
        accelerator, nodes = compute["accelerator"], int(compute["nodes"])
        timeout = compute.get("train_timeout_minutes", compute["timeout_minutes"])
    elif stage == "stage_model":
        accelerator, nodes, timeout = compute["accelerator"], 1, compute["timeout_minutes"]
    elif stage in SERVING_STAGES:
        accelerator, nodes = ev.get("accelerator", compute["accelerator"]), 1
        timeout = ev.get("timeout_minutes", compute["timeout_minutes"])
    else:
        accelerator, nodes, timeout = "GPU_1xA10", 1, 60
    return {"image": image, "accelerator": accelerator, "gpus": GPUS_PER_NODE[accelerator] * nodes,
            "timeout_minutes": timeout}


def launcher_path(recipe: dict[str, Any], miles_src: Path) -> Path:
    """`model.launcher`: the cookbook's own (rft_miles/launchers/...) or upstream's, in its source."""
    launcher = recipe["model"]["launcher"]
    if launcher.startswith(COOKBOOK_LAUNCHERS):
        return Path(__file__).parent / launcher.removeprefix("rft_miles/")
    return miles_src / launcher


def check(recipe: dict[str, Any]) -> None:
    compute, train = recipe["compute"], recipe["train"]
    if compute["accelerator"] not in GPUS_PER_NODE:
        raise ValueError(f"unknown accelerator {compute['accelerator']!r}")
    if int(compute["nodes"]) < 1:
        raise ValueError("compute.nodes must be at least 1")
    for stage in set(recipe["stages"]) & SERVING_STAGES:
        if not recipe.get("eval"):
            raise ValueError(f"stage {stage} needs an eval section")
    if "baseline_eval" in recipe["stages"] and (recipe.get("eval") or {}).get("baseline_run"):
        raise ValueError("eval.baseline_run evaluates no baseline of its own: drop baseline_eval from stages")
    if train["save_interval"] <= 0:
        raise ValueError("train.save_interval must be positive: every run publishes checkpoints")
    for key in ("launcher", "model_name", "megatron_type", "tool_parser"):
        if not recipe["model"].get(key):
            raise ValueError(f"model.{key} is required for a miles recipe")
    source = recipe["model"].get("processor_from")
    if source and not (source.get("id") and re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", "")))
                       and source.get("files")):
        raise ValueError("model.processor_from needs an id, a 40-hex revision and its files")


# recipe train settings -> miles flags; a setting the recipe leaves null keeps the launcher's value
ALGORITHM = {"lr": "--lr", "kl_loss_coef": "--kl-loss-coef", "entropy_coeff": "--entropy-coef",
             "clip_ratio_low": "--eps-clip", "clip_ratio_high": "--eps-clip-high"}


def miles_args(recipe: dict[str, Any], paths: dict[str, Any], task=None) -> list[str]:
    """What the cookbook adds to the model's launcher (its `--extra-args`)."""
    t, m = recipe["train"], recipe["model"]
    args = ["--prompt-data", paths["prompts"], "--input-key", "prompt", "--label-key", "label",
            "--metadata-key", "metadata",
            "--rollout-batch-size", str(t["rollout_batch_size"]), "--n-samples-per-prompt", str(t["n_samples_per_prompt"]),
            # one optimizer step per rollout, over every sample of it (verl: train_batch_size groups)
            "--global-batch-size", str(int(t["rollout_batch_size"]) * int(t["n_samples_per_prompt"])),
            "--rollout-temperature", str(t["temperature"]), "--rollout-seed", str(t["seed"]), "--seed", str(t["seed"]),
            "--custom-generate-function-path", GENERATE, "--custom-agent-function-path", AGENT,
            "--use-session-server", "v2", "--session-message-matcher", MATCHER, "--custom-rm-path", REWARD,
            "--sglang-tool-call-parser", m["tool_parser"],
            # a request field the client leaves out takes OpenAI's default, never the model's own
            # generation_config.json (GLM-5.3-Flash: top_p 0.95; Qwen3.6: top_k 20)
            "--sglang-sampling-defaults", "openai",
            "--load", paths["checkpoints"], "--save", paths["checkpoints"], "--save-interval", str(t["save_interval"]),
            # a resume keeps the learning-rate schedule it saved: a run resumed to more rollouts would
            # otherwise fail Megatron's check that the two agree on the number of iterations
            "--use-checkpoint-opt-param-scheduler",
            "--custom-megatron-post-save-hook-path", POST_SAVE,
            # miles appends its audit events to <save>/events by default, and a Volume cannot append; every
            # checkpoint carries a copy of them. Naming the directory would also switch on a full weight
            # checksum of every engine after each update, which miles otherwise leaves off.
            "--save-debug-event-data", f"{paths['local_dir']}/miles_events", "--no-log-inference-engine-weight-checksums"]
    if m.get("reasoning_parser"):
        args += ["--sglang-reasoning-parser", m["reasoning_parser"]]
    if m.get("tito_model"):   # the template and token boundary miles verified for the model family
        args += ["--tito-model", m["tito_model"]]
    args += ["--custom-agent-function-mode", AGENT_MODE]
    if t.get("save_hf"):   # Hugging Face weights inside every saved bundle, for export (CheckpointStore.export)
        args += ["--save-hf", f"{paths['checkpoints']}/iter_{{rollout_id:07d}}/{HF_DIR}"]
    for key, flag in ALGORITHM.items():
        if t.get(key) is not None:
            args += [flag, str(t[key])]
    if t.get("diagnostics"):   # observed, not trained: the entropy is detached when its coefficient is 0
        args += ["--custom-megatron-before-train-step-hook-path", DIAGNOSTICS, "--observe-training-entropy"]
    if task is not None:   # prompts the context cannot hold are filtered out, as verl filters them
        prompt = int(recipe["data"]["max_prompt_tokens"])
        args += ["--rollout-max-prompt-len", str(prompt),
                 "--rollout-max-context-len", str(prompt + task.budgets.max_response_tokens)]
    return args + [str(a) for a in (recipe.get("rollout") or {}).get("extra_args") or []]


def resolve_run(ref: str, *, site: dict[str, Any], run_id: str, overrides: list[str] = (), out: Path) -> Path:
    recipe = load_recipe(ref, overrides=list(overrides))
    if recipe["framework"] != "miles":
        raise ValueError(f"{ref} is a {recipe['framework']} recipe")
    check(recipe)
    task = load_task(recipe["task"], **(recipe.get("agent") or {}))
    paths = plan(recipe, site, run_id, task_version=task.version)
    m, t = recipe["model"], recipe["train"]
    local = paths["local_dir"]
    paths.update({
        # converted by the pinned miles' converter: <megatron_type>_torch_dist inside, as upstream names it
        "megatron_dir": f"{paths['model_dir']}-megatron-{LOCK['commit'][:12]}",
        "local_hf": f"{local}/models/{Path(paths['model_dir']).name}",
        "local_megatron": f"{local}/models/{Path(paths['model_dir']).name}-megatron",
        # model.processor_from, staged without weights (as a judge's tokenizer is)
        "processor_dir": (f"{Path(paths['model_dir']).parent}/{source['id'].split('/')[-1]}@{source['revision'][:12]}-tokenizer"
                          if (source := m.get("processor_from")) else None),
        "abort_file": f"{paths['run_dir']}/train/{ABORT_FILE}",
        "miles_commit": LOCK["commit"],
        "prompts": f"{local}/train/prompts.jsonl",
        "agent_config": f"{local}/train/agent.json",
        "task_version": task.version, "task_settings": task.settings(),
    })
    launcher = {
        "script": m["launcher"],
        "options": ["--model-name", m["model_name"], "--num-nodes", str(recipe["compute"]["nodes"]),
                    "--num-gpus-per-node", str(GPUS_PER_NODE[recipe["compute"]["accelerator"]]),
                    "--hf-checkpoint", paths["local_hf"], "--ckpt-dir", paths["local_megatron"],
                    "--num-rollout", str(t["num_rollout"]),
                    "--rollout-max-response-len", str(task.budgets.max_response_tokens),
                    "--train-offload-dir", f"{local}/train_offload", "--run-id", run_id],
        "extra_args": miles_args(recipe, paths, task),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    (out / "plan.json").write_text(json.dumps(paths, indent=1, sort_keys=True))
    (out / "miles.json").write_text(json.dumps(launcher, indent=1))
    return out

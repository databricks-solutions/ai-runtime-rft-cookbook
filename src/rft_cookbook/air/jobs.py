"""AI Runtime jobs and the run's plan of paths.

A recipe's jobs are checked in next to it, one per stage (`jobs/<stage>.yaml`). A job names its
recipe in `env_variables.RFT_RECIPE`; `databricks air run --override` fills in what belongs to a
workspace and a run (`JOB_ENV`, the image, secrets), one field per `--override`:

    databricks air run -f recipes/math/verl/qwen3.5-2b_h100x8/jobs/train.yaml -p <profile> \
        --override env_variables.RFT_RUN_ID=math-2b-01 --override env_variables.RFT_VOLUME_ROOT=<volume root> \
        --override environment.unity_catalog_image=<catalog>.<schema>.<image>:<tag>

Inside the job the launcher reads these back (`from_job_env`) and resolves the recipe from the
code snapshot. `plan(recipe, site, run_id)` fixes every durable location of a run from the
resolved recipe and the workspace settings, so all stages of one run agree on them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from collections.abc import Mapping
from typing import Any

SNAPSHOT_PATHS = ["pyproject.toml", "README.md", "src", "frameworks", "recipes"]
# filled in per run with `databricks air run --override env_variables.<name>=...`
JOB_ENV = {"RFT_RUN_ID": "the run id", "RFT_VOLUME_ROOT": "the durable root (a Unity Catalog volume)"}
# workspace resources a task may use, filled in the same way where its jobs name them: name -> plan key
SITE_ENV = {"RFT_VS_INDEX": "vector_search_index"}   # the Vector Search index agentic search queries
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,62}")


def data_key(recipe: dict[str, Any], task_version: str) -> str:
    """Identity of the prepared data: the task, its version and the recipe's data settings."""
    body = {"task": recipe["task"], "version": task_version, "data": recipe.get("data", {})}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


def plan(recipe: dict[str, Any], site: dict[str, Any], run_id: str, *, task_version: str) -> dict[str, Any]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"run id {run_id!r}: lowercase letters, digits, '.', '_' or '-', 3-63 characters")
    root = str(site.get("volume_root") or "").rstrip("/")
    if not root or "<" in root:
        raise ValueError("site.volume_root is not set")
    model = recipe.get("model") or {}
    if not model.get("id") or not re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision", ""))):
        raise ValueError("model.id and a 40-hex model.revision are required")
    data_dir = f"{root}/data/{recipe['task']}/{data_key(recipe, task_version)}"
    run_dir = f"{root}/runs/{run_id}"
    local = str(site.get("local_root") or "/local_disk0/rft").rstrip("/")
    # a run compared with another run's baseline evaluation (a replicate, or the same model trained
    # by another framework) reads it there
    baseline_run = (recipe.get("eval") or {}).get("baseline_run") or run_id
    out = {
        "run_id": run_id,
        "recipe": recipe["name"],
        "recipe_sha256": recipe["resolved_sha256"],
        "data_dir": data_dir,
        "train_rows": f"{data_dir}/train.jsonl",
        "eval_rows": f"{data_dir}/eval.jsonl",
        "model_dir": f"{root}/models/{model['id'].split('/')[-1]}@{model['revision'][:12]}",
        "run_dir": run_dir,
        "checkpoints": f"{run_dir}/checkpoints",
        "episodes": f"{run_dir}/episodes",
        "eval_dir": f"{run_dir}/eval",
        "local_dir": f"{local}/{run_id}",
        "baseline_dir": f"{root}/runs/{baseline_run}/eval/baseline",
    }
    out.update({key: site[key] for key in SITE_ENV.values() if site.get(key)})
    tokenizer = ((recipe.get("scoring") or {}).get("judge") or {}).get("tokenizer")
    if tokenizer:   # the judge's chat template counts its prompt tokens; its weights stay elsewhere
        out["judge_tokenizer_dir"] = f"{root}/models/{tokenizer['id'].split('/')[-1]}@{tokenizer['revision'][:12]}-tokenizer"
    return out


def from_job_env(env: Mapping[str, str] | None = None) -> tuple[str, str, dict[str, Any], list[str]]:
    """(recipe, run id, workspace settings, recipe overrides) of the running job.

    `RFT_SET` holds recipe overrides, shell-quoted and space-separated (`train.seed=7 eval.seed=42`);
    `RFT_LOCAL_ROOT` moves node-local scratch from /local_disk0/rft; `SITE_ENV` names the workspace
    resources a task's jobs ask for (`RFT_VS_INDEX`)."""
    env = os.environ if env is None else env
    missing = [name for name in ("RFT_RECIPE", *JOB_ENV) if not env.get(name) or "<" in env[name]]
    if missing:
        raise SystemExit("the job does not say " + ", ".join(JOB_ENV.get(n, "its recipe") for n in missing) +
                         ": submit it with --override " + " ".join(f"env_variables.{n}=..." for n in missing))
    site = {"volume_root": env["RFT_VOLUME_ROOT"], "local_root": env.get("RFT_LOCAL_ROOT") or None}
    site.update({key: env[name] for name, key in SITE_ENV.items() if env.get(name) and "<" not in env[name]})
    return env["RFT_RECIPE"], env["RFT_RUN_ID"], site, shlex.split(env.get("RFT_SET", ""))

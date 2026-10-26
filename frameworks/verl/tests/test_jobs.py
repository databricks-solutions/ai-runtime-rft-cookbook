"""The checked-in AI Runtime jobs of every verl recipe (`recipes/<ref>/jobs/<stage>.yaml`): one per
stage, each running its stage of its own recipe on the compute the recipe asks for, with nothing
workspace-specific in it."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rft_cookbook.air.jobs import JOB_ENV, SITE_ENV, SNAPSHOT_PATHS
from rft_cookbook.recipe import load_recipe, recipes_root
from rft_verl.render import GPU_STAGES, GPUS_PER_NODE, LAUNCHER, STAGE_ENV

ROOT = recipes_root()
RECIPES = sorted(str(p.parent.relative_to(ROOT)) for p in ROOT.glob("*/verl/*/recipe.yaml"))


@pytest.mark.parametrize("ref", RECIPES)
def test_every_stage_has_its_job_and_nothing_else(ref):
    stages = load_recipe(ref)["stages"]
    assert sorted(p.stem for p in (ROOT / ref / "jobs").glob("*.yaml")) == sorted(stages)


@pytest.mark.parametrize("ref", RECIPES)
def test_each_job_runs_its_stage_of_its_recipe(ref):
    recipe = load_recipe(ref)
    compute = recipe["compute"]
    per_node = GPUS_PER_NODE[compute["accelerator"]]
    for stage in recipe["stages"]:
        path = ROOT / ref / "jobs" / f"{stage}.yaml"
        job = yaml.safe_load(path.read_text())
        assert job["command"] == f"bash ${{CODE_SOURCE_PATH}}/{LAUNCHER} {stage}"
        env = job["env_variables"]
        assert env["RFT_RECIPE"] == ref
        assert all(env[name].startswith("<") for name in JOB_ENV)   # filled in with --override
        assert all(env[name].startswith("<") for name in SITE_ENV if name in env)   # a workspace's index too
        assert job["environment"] == {"unity_catalog_image": job["environment"]["unity_catalog_image"]}
        assert job["environment"]["unity_catalog_image"].startswith("<")
        assert all(v == "secret_scope/hf_token" for v in (job.get("secrets") or {}).values())   # filled in too
        assert all(env.get(k) == v for k, v in STAGE_ENV.get(stage, {}).items())
        code = job["code_source"]   # `databricks air`: the code's root, relative to the job's directory
        assert (path.parent / code["root_path"]).resolve() == ROOT.parent
        assert code["include_paths"] == SNAPSHOT_PATHS
        gpu = stage in GPU_STAGES
        want = (per_node * compute["nodes"] if stage == "train" else per_node) if gpu else 1
        assert job["compute"] == {"num_accelerators": want,
                                  "accelerator_type": compute["accelerator"] if gpu else "GPU_1xA10"}, stage
        timeout = compute.get("train_timeout_minutes", compute["timeout_minutes"]) if stage == "train" else (
            compute["timeout_minutes"] if gpu else 60)
        assert job["timeout_minutes"] == timeout and job["max_retries"] == 0, stage

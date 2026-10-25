"""The qualification record (rft_cookbook.evidence.qualification): identities, checks and status."""
from __future__ import annotations

import json
from pathlib import Path

from test_checkpoints import native_save

from rft_cookbook.checkpoints import CheckpointStore
from rft_cookbook.evidence import qualification

RECIPE = {"name": "toy/verl/r", "task": "toy", "framework": "verl", "model": {"id": "m/x", "revision": "a" * 40},
          "train": {"total_training_steps": 4, "seed": 42}, "eval": {"seed": 42}, "compute": {"nodes": 1},
          "scoring": {"judge": {"id": "j", "models": ["judge-a"]}}, "resolved_sha256": "run"}
CHECKED_IN = {**RECIPE, "resolved_sha256": "checked-in"}
COMPARISON = {"summary": {"comparable": True, "incomplete": [], "mismatched_settings": [], "leakage": {"clean": True}},
              "splits": {"heldout": {"baseline": 0.2, "trained": 0.25}}}


def run(tmp_path: Path, *, recipe=RECIPE, comparison=COMPARISON, calibrated=True) -> dict:
    run_dir = tmp_path / "runs" / "r1"
    store = CheckpointStore(run_dir / "checkpoints", hash_files=False)
    store.publish(native_save(tmp_path / "local", 4))
    published = store.latest()
    (run_dir / "export.json").write_text(json.dumps({**store.export(published), "name": published.name}))
    (run_dir / "calibration.json").write_text(json.dumps({"checks": [{"pass": True}, {"pass": calibrated}]}))
    plan = run_dir / "eval" / "step_4" / "plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"task": "toy", "samples": 4, "model": {"checkpoint": "r1/global_step_4"}}))
    paths = {"run_id": "r1", "run_dir": str(run_dir), "checkpoints": str(run_dir / "checkpoints"),
             "eval_dir": str(run_dir / "eval"), "baseline_dir": str(tmp_path / "runs/base/eval/baseline"),
             "model_dir": str(tmp_path / "models/x"), "data_dir": str(tmp_path / "data")}
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True, exist_ok=True)
    (code / "src" / "a.py").write_text("x = 1\n")
    return qualification.record(recipe, paths, comparison, checked_in=CHECKED_IN, code_root=code,
                                env={"RFT_SET": "train.seed=7", "RFT_IMAGE": "img@sha256:abc"})


def test_a_complete_run_of_the_recipe_as_checked_in_is_measured(tmp_path):
    record = run(tmp_path)
    assert record["status"] == "measured" and all(c["passed"] for c in record["checks"])
    assert {c["name"] for c in record["checks"]} == {"calibration", "export", "evaluated_checkpoint",
                                                    "evaluations_complete", "comparable", "leakage_clean"}
    assert record["recipe"]["overrides"] == ["train.seed=7"] and record["image"] == "img@sha256:abc"
    assert record["training"]["published"] == "global_step_4" and record["result"]["heldout"]["trained"] == 0.25


def test_a_reduced_run_is_smoke_tested_and_a_seed_is_not_a_reduction(tmp_path):
    reduced = {**RECIPE, "train": {**RECIPE["train"], "total_training_steps": 2, "seed": 7}}
    record = run(tmp_path, recipe=reduced)
    assert record["status"] == "smoke-tested" and record["recipe"]["deviations"] == ["train.total_training_steps"]


def test_a_failed_check_fails_the_record(tmp_path):
    assert run(tmp_path / "a", calibrated=False)["status"] == "failed"
    incomplete = {**COMPARISON, "summary": {**COMPARISON["summary"], "incomplete": ["trained"]}}
    assert run(tmp_path / "b", comparison=incomplete)["status"] == "failed"


def test_the_code_identity_covers_code_not_documentation(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "m.py").write_text("x = 1\n")
    first = qualification.code_identity(tmp_path)
    (tmp_path / "src" / "pkg" / "README.md").write_text("notes")
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir()
    (tmp_path / "src" / "pkg" / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"\0")
    assert qualification.code_identity(tmp_path) == first
    (tmp_path / "src" / "pkg" / "m.py").write_text("x = 2\n")
    assert qualification.code_identity(tmp_path)["sha256"] != first["sha256"]

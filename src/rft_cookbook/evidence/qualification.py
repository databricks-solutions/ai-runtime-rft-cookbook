"""The qualification record of a run: what ran, on what, and which checks it passed.

The `compare` stage writes it next to the comparison (`<run dir>/qualification.json`). It
identifies the code (a hash of the snapshot's sources, documentation left out, and the commit when
the job was given one: `RFT_CODE_COMMIT`), the recipe as checked in and the overrides the run
applied, the image (`RFT_IMAGE`), the model and judge tokenizer revisions, the data and corpus
(the prepared data's manifest), the evaluation policy, the topology, and the checks:

- same code: every stage of the run (each writes a receipt, `rft_cookbook.air.stages.write_receipt`)
  ran the code this record is written with, and train, export and eval did run (the record lists
  each stage's image and overrides: miles trains on its image and evaluates on verl's, and a run
  continued to more updates trained under two step counts);
- calibration: the judge passed its calibration fixtures (recipes with a judge);
- export: the exported weights are those of the published checkpoint (re-derived from it);
- training budget: the exported step matches the recipe's requested number of updates;
- evaluated checkpoint: the trained evaluation served this run's export and pinned model;
- evaluations complete, comparable (same plan and settings) and free of leakage;
- parity, when the run checked it.

The status follows: `measured` when every check passed and the run is the recipe as checked in (a
training or evaluation seed, the baseline run, the stages and diagnostics may differ), `smoke-tested` when
every check passed on a run that changed anything else (fewer updates, say), `failed` otherwise.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any

SCHEMA = "rft-cookbook/qualification/v1"
# a replicate stays the recipe, and so does a run that only logs more (train.diagnostics: miles' drift logging)
NEUTRAL = ("train.seed", "eval.seed", "eval.baseline_run", "stages", "train.diagnostics")
CODE_DIRS = ("src", "frameworks", "recipes")


def code_identity(root: Path) -> dict[str, Any]:
    """A hash over the code's files (path and content), documentation and caches left out."""
    digest, files = hashlib.sha256(), 0
    for top in CODE_DIRS:
        for path in sorted((root / top).rglob("*")):
            rel = path.relative_to(root)
            if not path.is_file() or path.suffix in (".md", ".pyc") or "__pycache__" in rel.parts \
                    or any(part.endswith(".egg-info") for part in rel.parts):
                continue
            digest.update(f"{rel.as_posix()}\n".encode())
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
            files += 1
    return {"sha256": digest.hexdigest(), "files": files}


def deviations(checked_in: dict[str, Any], run: dict[str, Any], prefix: str = "") -> list[str]:
    """Dotted keys whose value differs between the two recipes, the neutral ones left out."""
    out = []
    for key in sorted(set(checked_in) | set(run)):
        name = f"{prefix}{key}"
        if name in NEUTRAL or name in ("name", "resolved_sha256"):
            continue
        a, b = checked_in.get(key), run.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out += deviations(a, b, f"{name}.")
        elif a != b:
            out.append(name)
    return out


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _check(name: str, passed: bool | None, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _same_code(receipts: list[dict], code: str) -> dict[str, Any]:
    differ = sorted({r["stage"] for r in receipts if r["code"]["sha256"] != code})
    missing = sorted({"train", "export", "eval"} - {r["stage"] for r in receipts})
    detail = "; ".join(filter(None, [f"other code: {', '.join(differ)}" if differ else "",
                                     f"no receipt: {', '.join(missing)}" if missing else ""]))
    return _check("same_code", not differ and not missing, detail or f"{len(receipts)} receipts")


def record(recipe: dict[str, Any], paths: dict[str, Any], comparison: dict[str, Any], *,
           checked_in: dict[str, Any], code_root: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    from rft_cookbook.checkpoints import CheckpointError, CheckpointStore

    env = dict(os.environ) if env is None else env
    run_dir = Path(paths["run_dir"])
    exported = _read(run_dir / "export.json") or {}
    trained_plan = _read(Path(paths["eval_dir"]) / f"step_{exported.get('step')}" / "plan.json") or {}
    summary = comparison.get("summary", {})
    checks = []

    if (recipe.get("scoring") or {}).get("judge"):
        calibration = _read(run_dir / "calibration.json")
        cases = (calibration or {}).get("checks") or []
        checks.append(_check("calibration", bool(cases) and all(c.get("pass") for c in cases),
                             f"{sum(bool(c.get('pass')) for c in cases)}/{len(cases)} fixtures"))
    try:
        published = CheckpointStore(paths["checkpoints"]).get(int(exported["step"]))
        again = CheckpointStore(paths["checkpoints"]).export(published)
        checks.append(_check("export", again["weights_sha256"] == exported.get("weights_sha256"), published.name))
    except (KeyError, TypeError, ValueError, CheckpointError) as error:
        checks.append(_check("export", False, f"{type(error).__name__}: {error}"))
    step_key = {"verl": "total_training_steps", "miles": "num_rollout"}.get(recipe["framework"])
    requested = (recipe.get("train") or {}).get(step_key)
    checks.append(_check("training_budget",
                         isinstance(requested, int) and not isinstance(requested, bool) and requested > 0
                         and exported.get("step") == requested,
                         f"requested {requested} updates, exported {exported.get('step')}"))
    trained_model = trained_plan.get("model") or {}
    served = trained_model.get("checkpoint") or ""
    expected = f"{paths['run_id']}/{exported.get('name')}"
    checks.append(_check("evaluated_checkpoint",
                         bool(exported.get("name")) and served == expected
                         and all(trained_model.get(k) == recipe["model"][k] for k in ("id", "revision")),
                         json.dumps(trained_model, sort_keys=True)))
    checks.append(_check("evaluations_complete", not summary.get("incomplete"),
                         ", ".join(summary.get("incomplete") or [])))
    checks.append(_check("comparable", bool(summary.get("comparable")),
                         ", ".join(summary.get("mismatched_settings") or [])))
    checks.append(_check("leakage_clean", bool((summary.get("leakage") or {}).get("clean"))))
    parity = _read(run_dir / "parity.json")
    if parity is not None:
        checks.append(_check("parity", bool(parity.get("agree"))))
    code = {**code_identity(code_root), "commit": env.get("RFT_CODE_COMMIT")}
    receipts = [r for r in (_read(p) for p in sorted((run_dir / "receipts").glob("*.json"))) if r]
    checks.append(_same_code(receipts, code["sha256"]))

    differs = deviations(checked_in, recipe)
    passed = all(c["passed"] for c in checks)
    status = ("measured" if not differs else "smoke-tested") if passed else "failed"
    staged = _read(Path(paths["model_dir"]) / "STAGED.json") or {}
    judge = (recipe.get("scoring") or {}).get("judge") or {}
    return {
        "schema": SCHEMA, "run_id": paths["run_id"], "status": status,
        "recipe": {"name": recipe["name"], "checked_in_sha256": checked_in["resolved_sha256"],
                   "resolved_sha256": recipe["resolved_sha256"], "deviations": differs,
                   "overrides": shlex.split(env.get("RFT_SET", ""))},
        "code": code,
        "stages": sorted({json.dumps({k: r.get(k) for k in ("stage", "image", "overrides")}, sort_keys=True)
                          for r in receipts}),
        "image": env.get("RFT_IMAGE"),
        "model": {"id": recipe["model"]["id"], "revision": recipe["model"]["revision"],
                  "staged_revision": staged.get("revision")},
        "judge": {k: judge.get(k) for k in ("id", "models", "tokenizer")} if judge else None,
        "data": _read(Path(paths["data_dir"]) / "manifest.json"),
        "evaluation": {**{k: trained_plan.get(k) for k in ("task", "task_version", "rows_sha256", "samples",
                                                           "sampling", "serving", "schema_view", "settings")},
                       "baseline": paths["baseline_dir"], "trained": served},
        "training": {"published": exported.get("name"), "step": exported.get("step"),
                     "weights_sha256": exported.get("weights_sha256"),
                     # "content": a hash of every weight file; absent in records from before it was
                     # (their recipes without file hashes hashed shard names and sizes only)
                     "weights_hash": exported.get("weights_hash")},
        "topology": recipe.get("compute"),
        "checks": checks,
        "result": {"mean_difference": summary.get("mean_difference"), "bootstrap_95": summary.get("bootstrap_95"),
                   "splits": comparison.get("splits")},
    }

"""The stages every framework runs the same way, called by each framework's launcher
(rft_verl.launch, rft_miles.launch) with the resolved recipe and its plan of paths:

  prep           the task's rows on the volume (one copy per task version and data settings)
  stage_judge    the judge's tokenizer, which counts its prompt tokens
  calibrate      the judge grades known cases before it grades anything else
  baseline_eval  the base model, served with vLLM, through the Chat Completions harness
  export         the newest published checkpoint's Hugging Face weights, checked
  eval           the run's exported checkpoint, the same way
  compare        the two evaluations, paired by question

Evaluation does not depend on the framework that trained the checkpoint: every framework's
checkpoint is served by the same vLLM server with the same sampling, harness and scorer, so two
frameworks' runs of one model can share one baseline evaluation (`eval.baseline_run`).

The training stages stay with the frameworks; this module also holds the stop request an agent
function leaves when episodes keep failing (`request_abort`) and the loop that honours it
(`run_until_abort`).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from rft_cookbook.air.cluster import SKEW_S

ABORT_FILE = "ABORT.json"   # under <run_dir>/train: written by an agent, honoured by the launcher


def _write(path: str | Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True, default=str))


def task_resources(recipe: dict[str, Any], paths: dict[str, Any]) -> dict[str, Any] | None:
    """What the task needs at run time (a node-local tool service named after the run, data, a
    judge, an index): its module's `run_resources(recipe, paths)`. None for a task with none."""
    hook = getattr(_task_module(recipe), "run_resources", None)
    return hook(recipe, paths) if hook else None


def start_resources(recipe: dict[str, Any], paths: dict[str, Any], resources: dict[str, Any] | None):
    """Check the task's resources (its module's `preflight`, which raises when one is unusable) and
    start its node-local tool service if it has one; returns the service process or None."""
    from rft_cookbook.tools import service

    check = getattr(_task_module(recipe), "preflight", None)
    if resources and check:
        print(f"[resources] {json.dumps(check(resources), default=str)}", flush=True)
    if not (resources or {}).get("tool_service"):
        return None
    return service.start(recipe["task"], resources, paths["run_id"])


def _task_module(recipe, part: str = ""):
    import importlib

    return importlib.import_module(f"rft_cookbook.tasks.{recipe['task'].replace('-', '_')}{part}")


def scratch(paths) -> Path:
    """Local disk for building large files before they are copied to the volume."""
    local = Path(paths["local_dir"])
    root = Path(local.parts[0], *local.parts[1:2])   # /local_disk0 exists on GPU nodes, not everywhere
    out = local / "scratch" if root.is_dir() else Path("/tmp/rft-scratch")
    out.mkdir(parents=True, exist_ok=True)
    return out


def prepare_data(recipe, paths) -> Path:
    """The task's rows in `data_dir`, prepared once and reused by every run with the same key."""
    from rft_cookbook.data import MANIFEST

    data_dir = Path(paths["data_dir"])
    if (data_dir / MANIFEST).exists():
        return data_dir
    module = _task_module(recipe, ".data")
    if not hasattr(module, "prepare_recipe"):
        raise SystemExit(f"task {recipe['task']!r} has no data preparation (prepare_recipe in its data module)")
    module.prepare_recipe(recipe, data_dir, scratch=lambda: scratch(paths))
    return data_dir


def stage_judge(recipe, paths, rendered) -> None:
    from rft_cookbook.air.stage_model import stage

    tokenizer = recipe["scoring"]["judge"]["tokenizer"]
    stage(tokenizer["id"], tokenizer["revision"], paths["judge_tokenizer_dir"], tokenizer_only=True)


def calibrate(recipe, paths, rendered, out: Path | None = None) -> None:
    import asyncio
    import importlib

    module = importlib.import_module(f"rft_cookbook.tasks.{recipe['task']}.calibration")
    task_module = importlib.import_module(f"rft_cookbook.tasks.{recipe['task']}")
    result = asyncio.run(module.calibrate(task_module.build_scorer(task_resources(recipe, paths)["judge"])))
    _write(out or Path(paths["run_dir"]) / "calibration.json", result)
    passed = sum(c["pass"] for c in result["checks"])
    print(f"[calibrate] {result['status']}: {passed}/{len(result['checks'])} grades as expected ({result['policy']})")
    if result["status"] != "PASS":
        raise SystemExit("the judge failed calibration: it is not used")


def evaluate_model(recipe, paths, *, model_path: str, label: str, checkpoint: str | None) -> None:
    from rft_cookbook.eval.run import evaluate
    from rft_cookbook.tools import service

    ev, model = recipe["eval"], recipe["model"]
    resources = task_resources(recipe, paths)
    if resources and resources.get("judge"):   # a judge proves itself before it grades
        calibrate(recipe, paths, None, out=Path(paths["eval_dir"]) / f"{label}.calibration.json")
    process = start_resources(recipe, paths, resources)
    try:
        evaluate(task_name=recipe["task"], budgets=dict(recipe.get("agent") or {}), rows=paths["eval_rows"],
                 model_path=model_path, out=f"{paths['eval_dir']}/{label}",
                 model_identity={"id": model["id"], "revision": model["revision"], "checkpoint": checkpoint},
                 tool_parser=model["tool_parser"], reasoning_parser=model.get("reasoning_parser"),
                 chat_template_kwargs=model.get("chat_template_kwargs") or {}, samples=int(ev["samples"]),
                 temperature=float(ev["temperature"]), top_p=float(ev["top_p"]), concurrency=int(ev["concurrency"]),
                 gpu_memory_utilization=float(ev["gpu_memory_utilization"]),
                 max_prompt_tokens=int(recipe["data"]["max_prompt_tokens"]),
                 seed=int(recipe["train"]["seed"] if ev.get("seed") is None else ev["seed"]),
                 serve_args=list(model.get("serve_args") or []), tensor_parallel=int(ev.get("tensor_parallel", 1)),
                 resources=resources, request_timeout_s=float(ev.get("request_timeout_s", 600)),
                 local_cache=f"{paths['local_dir']}/models" if model_path.startswith("/Volumes/") else None,
                 top_k=int(ev.get("top_k", -1)), schema_view=ev.get("schema_view", "full"))
    finally:
        if process:
            service.stop(process, paths["run_id"])


def baseline_eval(recipe, paths, rendered) -> None:
    if (recipe["eval"].get("baseline_run") or paths["run_id"]) != paths["run_id"]:
        raise SystemExit(f"this run is compared with {recipe['eval']['baseline_run']}'s baseline: evaluate it there")
    evaluate_model(recipe, paths, model_path=paths["model_dir"], label="baseline", checkpoint=None)


def export(recipe, paths, rendered) -> None:
    """The newest published checkpoint's Hugging Face weights, checked, for evaluation."""
    from rft_cookbook.checkpoints import CheckpointStore

    store = CheckpointStore(paths["checkpoints"])
    latest = store.latest()
    if latest is None:
        raise SystemExit("[export] no published checkpoint")
    store.verify(latest)
    record = {**store.export(latest), "name": latest.name}
    _write(Path(paths["run_dir"]) / "export.json", record)
    print(f"[export] {latest.name}: {record['path']} ({len(record['weights'])} weight files)")


def _exported(paths) -> dict[str, Any]:
    return json.loads((Path(paths["run_dir"]) / "export.json").read_text())


def eval_(recipe, paths, rendered) -> None:
    exported = _exported(paths)
    name = exported.get("name") or f"global_step_{exported['step']}"
    evaluate_model(recipe, paths, model_path=exported["path"], label=f"step_{exported['step']}",
                   checkpoint=f"{paths['run_id']}/{name}")


def compare(recipe, paths, rendered) -> None:
    """The paired comparison with the baseline, and the run's qualification record."""
    from rft_cookbook.evidence import qualification
    from rft_cookbook.evidence.report import compare_runs
    from rft_cookbook.recipe import load_recipe, recipes_root

    exported = _exported(paths)
    report = compare_runs(paths["baseline_dir"], f"{paths['eval_dir']}/step_{exported['step']}",
                          train_rows=paths["train_rows"], probe_splits=tuple(recipe["eval"].get("probe_splits") or ()))
    _write(Path(paths["run_dir"]) / "comparison.json", report)
    print(json.dumps(report["summary"], indent=1))
    root = recipes_root()
    record = qualification.record(recipe, paths, report, checked_in=load_recipe(recipe["name"], root=root),
                                  code_root=root.parent)
    _write(Path(paths["run_dir"]) / "qualification.json", record)
    print(f"[compare] qualification: {record['status']}; "
          + ", ".join(f"{c['name']} {'ok' if c['passed'] else 'FAILED'}" for c in record["checks"]))


def write_receipt(paths, stage: str, overrides: list[str]) -> dict[str, Any]:
    """What this stage runs with (code, image, overrides), under `<run dir>/receipts`: the
    qualification record checks that every stage of the run ran the same."""
    from rft_cookbook.evidence.qualification import code_identity
    from rft_cookbook.recipe import recipes_root

    code = {**code_identity(recipes_root().parent), "commit": os.environ.get("RFT_CODE_COMMIT")}
    receipt = {"stage": stage, "code": code, "image": os.environ.get("RFT_IMAGE"), "overrides": list(overrides),
               "recipe_sha256": paths["recipe_sha256"], "host": socket.gethostname(), "at": time.time()}
    _write(Path(paths["run_dir"]) / "receipts" / f"{stage}-{int(receipt['at'])}-{receipt['host']}.json", receipt)
    return receipt


def request_abort(abort_file: str | None, reason: str) -> None:
    """Ask the launcher to stop the run (atomically: the launcher never reads half a request)."""
    if not abort_file:
        return
    payload = {"reason": reason, "at": time.time(), "host": socket.gethostname(), "pid": os.getpid()}
    tmp = f"{abort_file}.{socket.gethostname()}.{os.getpid()}"
    os.makedirs(os.path.dirname(abort_file), exist_ok=True)
    with open(tmp, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp, abort_file)


def run_until_abort(command: list[str], abort: Path, *, poll_s: float = 15.0, grace_s: float = 120.0,
                    cwd: str | Path | None = None, label: str = "train") -> int:
    """Run the framework; stop it when an agent asks to (`abort`, written after this run started)."""
    abort.unlink(missing_ok=True)
    started = time.time()
    process = subprocess.Popen(command, cwd=cwd)
    while True:
        try:
            return process.wait(timeout=poll_s)
        except subprocess.TimeoutExpired:
            pass
        try:
            request = json.loads(abort.read_text())
        except (OSError, ValueError):
            continue
        if float(request.get("at", 0)) < started - SKEW_S:   # written on another node's clock
            continue
        print(f"[{label}] stopping: {request['reason']} ({request.get('host')}, pid {request.get('pid')})", flush=True)
        process.terminate()   # Ray removes the driver's actors with it
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise SystemExit(f"[{label}] stopped: {request['reason']}")

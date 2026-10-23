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
    """What the task needs at run time: its node-local tool service (named after the run), its data
    and its judge. None for a task without tools or judge of its own."""
    if recipe["task"] != "officeqa":
        return None
    scoring = recipe.get("scoring") or {}
    out = {"tool_service": paths["run_id"], "data_dir": f"{paths['data_dir']}/corpus",
           "sandbox": scoring.get("sandbox", "strict"), "compute": scoring.get("compute") or {}}
    if scoring.get("judge"):
        out["judge"] = {**scoring["judge"], "tokenizer": paths["judge_tokenizer_dir"]}
    return out


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
    data = recipe.get("data") or {}
    if recipe["task"] == "math":
        from rft_cookbook.tasks.math.data import prepare

        prepare(data_dir, levels=set(data["levels"]) if data.get("levels") else None,
                train_limit=int(data.get("train_limit", 0)), eval_limit=int(data.get("eval_limit", 0)))
    elif recipe["task"] == "officeqa":
        from rft_cookbook.tasks.officeqa.data import prepare

        prepare(data_dir, scratch=scratch(paths))
    else:
        raise SystemExit(f"no data preparation is wired for task {recipe['task']!r}")
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
    process = service.start(recipe["task"], resources, paths["run_id"]) if resources else None
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
    from rft_cookbook.evidence.report import compare_runs

    exported = _exported(paths)
    report = compare_runs(paths["baseline_dir"], f"{paths['eval_dir']}/step_{exported['step']}",
                          train_rows=paths["train_rows"], probe_splits=tuple(recipe["eval"].get("probe_splits") or ()))
    _write(Path(paths["run_dir"]) / "comparison.json", report)
    print(json.dumps(report["summary"], indent=1))


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

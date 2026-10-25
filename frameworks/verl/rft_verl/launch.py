"""Run one stage of a verl run inside an AI Runtime job (see launch/node.sh).

    python -m rft_verl.launch <stage>

The job says which recipe and run (rft_cookbook.air.jobs.from_job_env); the launcher resolves the
recipe from the code snapshot (rft_verl.render) and runs the stage.

Stages: prep, stage_model, stage_judge, calibrate, baseline_eval, parity, train, export, eval,
compare. Every stage reads only the resolved run and writes only under the run's durable paths
(plan.json), so a stage can be re-run; prep and the staging stages skip work that is already
done and verified. A task with its own tools (`local_tools`) runs them in a node-local tool
service for the length of a stage; a task with a judge is calibrated before it evaluates.

`train` resumes from the newest published checkpoint of this run if there is one: verl resumes
from it (`trainer.resume_mode=resume_path`), after restoring it to local disk unless verl saves
straight into the store (`checkpoint.location: volume`). Raising `train.total_training_steps`
(`RFT_SET`) continues a finished run. On several nodes every node runs the task's tool service
and joins one Ray cluster (rft_cookbook.air.cluster); the head runs verl.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import yaml

from rft_cookbook.air.stages import (ABORT_FILE, baseline_eval, calibrate, compare, eval_, export, prepare_data,
                                     run_until_abort, stage_judge, write_receipt)
from rft_cookbook.air.stages import task_resources as resources
from rft_cookbook.checkpoints import CheckpointStore
from rft_verl.checkpoint import resume_overrides


def _load(rendered: Path):
    recipe = yaml.safe_load((rendered / "recipe.yaml").read_text())
    paths = json.loads((rendered / "plan.json").read_text())
    return recipe, paths


def _write(path: str | Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True, default=str))


def prep(recipe, paths, rendered) -> None:
    from rft_verl.data import convert

    data_dir = prepare_data(recipe, paths)
    for split, name in (("train", "train_parquet"), ("eval", "eval_parquet")):
        record = convert(recipe["task"], data_dir / f"{split}.jsonl", Path(paths[name]), recipe.get("agent"))
        _write(Path(paths[name]).with_suffix(".json"), record)
        if record["prompt_sha256"] != paths["task_settings"]["prompt_sha256"]:
            raise SystemExit("the task's system prompt changed between rendering and preparation")
    print(f"[prep] data ready in {data_dir}")


def stage_model(recipe, paths, rendered) -> None:
    from rft_cookbook.air.stage_model import stage

    stage(recipe["model"]["id"], recipe["model"]["revision"], paths["model_dir"])


def train(recipe, paths, rendered) -> None:
    from rft_cookbook.air import cluster
    from rft_cookbook.tools import service

    store = CheckpointStore(paths["checkpoints"])
    latest = store.latest()
    if latest is not None and latest.step >= int(recipe["train"]["total_training_steps"]):
        print(f"[train] {latest.name} already reaches total_training_steps; nothing to train")
        return
    nodes = cluster.Nodes.from_env()
    if nodes.count != int(recipe["compute"]["nodes"]):
        raise SystemExit(f"the recipe trains on {recipe['compute']['nodes']} nodes, the job has {nodes.count}")
    task_resources = resources(recipe, paths)
    process = service.start(recipe["task"], task_resources, paths["run_id"]) if task_resources else None
    try:
        if nodes.count == 1:
            _train(recipe, paths, rendered, store, latest)
            return
        per_node = int(recipe["compute"]["trainer_gpus"] + recipe["compute"]["rollout_gpus"]) // nodes.count
        rendezvous = nodes.rendezvous(Path(paths["run_dir"]) / "train" / "cluster")
        if nodes.rank == 0:
            cluster.run_head(nodes, per_node, rendezvous, lambda: _train(recipe, paths, rendered, store, latest))
        else:
            cluster.run_worker(nodes, per_node, rendezvous)
    finally:
        if process:
            service.stop(process, paths["run_id"])


def _train(recipe, paths, rendered, store: CheckpointStore, latest) -> None:
    args = [a.replace("@RENDERED@", str(rendered)) for a in json.loads((rendered / "verl_args.json").read_text())]
    if latest is not None:
        in_place = recipe["checkpoint"].get("location", "local") == "volume"
        store.verify(latest)
        source = Path(latest.location) if in_place else store.restore(latest, Path(paths["local_checkpoints"]) / "restored")
        args = [a for a in args if not a.startswith("trainer.resume_mode=")] + resume_overrides(str(source))
        print(f"[train] resuming from {latest.name} ({source})")
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *args]
    _write(Path(paths["run_dir"]) / "train" / f"command-{int(time.time())}.json", command)
    print("[train] " + " \\\n    ".join(command), flush=True)
    returncode = run_until_abort(command, Path(paths["run_dir"]) / "train" / ABORT_FILE)
    if returncode != 0:
        raise SystemExit(f"[train] verl exited with {returncode}")
    final = store.latest()
    if final is None or final.step < int(recipe["train"]["total_training_steps"]):
        raise SystemExit(f"[train] verl exited 0 but the newest published checkpoint is {final and final.name}")
    print(f"[train] published {final.name} at {final.location}")


def parity_(recipe, paths, rendered) -> None:
    """The base model's evaluation against the recorded reference, paired by question."""
    from rft_cookbook.evidence.report import parity

    spec = recipe["parity"]
    reference = json.loads((rendered / "parity_reference.json").read_text())
    report = parity(paths["baseline_dir"], reference, metrics=dict(spec["metrics"]),
                    max_abs_difference=spec.get("max_abs_difference"), seed=int(recipe["train"]["seed"]))
    _write(Path(paths["run_dir"]) / "parity.json", report)
    for name, metric in report["metrics"].items():
        interval = metric["bootstrap_95"] or [float("nan")] * 2
        print(f"[parity] {name}: ours {metric['ours']:.4f} vs reference {metric['reference']:.4f}, "
              f"difference {metric['mean_difference']:+.4f} [{interval[0]:+.4f}, {interval[1]:+.4f}] "
              f"{'agrees' if metric['agree'] else 'DISAGREES'}")
    if not report["agree"]:
        raise SystemExit("the evaluation does not agree with the reference: no training run")


STAGES = {"prep": prep, "stage_model": stage_model, "stage_judge": stage_judge, "calibrate": calibrate,
          "baseline_eval": baseline_eval, "parity": parity_, "train": train,
          "export": export, "eval": eval_, "compare": compare}


def main(argv=None) -> int:
    from rft_cookbook.air.jobs import from_job_env
    from rft_verl.render import resolve_run

    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0] not in STAGES:
        print(__doc__)
        return 2
    ref, run_id, site, overrides = from_job_env()
    rendered = resolve_run(ref, site=site, run_id=run_id, overrides=overrides,
                           out=Path(os.environ.get("RFT_RUN_DIR", "/tmp/rft-run")) / run_id)
    print(f"[launch] {argv[0]} of {run_id}: {ref} {' '.join(overrides)}".rstrip(), flush=True)
    recipe, paths = _load(rendered)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    write_receipt(paths, argv[0], overrides)
    STAGES[argv[0]](recipe, paths, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

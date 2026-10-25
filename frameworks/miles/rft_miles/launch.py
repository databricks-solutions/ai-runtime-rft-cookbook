"""Run one stage of a miles run inside an AI Runtime job (see launch/node.sh).

    python -m rft_miles.launch <stage>

The job says which recipe and run (rft_cookbook.air.jobs.from_job_env); the launcher resolves the
recipe from the code snapshot (rft_miles.render) and runs the stage. Upstream miles is the
checkout node.sh installed (`$RFT_MILES_SRC`).

Stages:
  prep         the task's rows on the volume (shared with every framework: same data, same key)
  stage_model  the model's Hugging Face weights on the volume, and their Megatron torch_dist
               conversion by upstream's converter (`<model_dir>-megatron/<megatron_type>_torch_dist`)
  train        the model's launcher with the cookbook's hooks (rft_miles.render); it starts Ray
               and runs miles until `train.num_rollout` rollouts are trained. It resumes from the
               newest published checkpoint of the run (`--load`). A task with its own tools runs
               them in a node-local tool service on every node for the length of the stage.
  stage_judge, calibrate, baseline_eval, export, eval, compare
               as for every framework (rft_cookbook.air.stages): evaluation serves the exported
               checkpoint with vLLM from the verl image, so these jobs install no miles.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

import yaml

from rft_cookbook.air.stages import (baseline_eval, calibrate, compare, eval_, export, prepare_data, run_until_abort,
                                     stage_judge, task_resources)
from rft_cookbook.checkpoints import CheckpointStore
from rft_miles.agent import CONFIG_KEY, ROW_KEY
from rft_miles.render import GPUS_PER_NODE, launcher_path

MEGATRON = "/root/Megatron-LM"   # where the miles image keeps Megatron-LM


def _load(rendered: Path):
    return yaml.safe_load((rendered / "recipe.yaml").read_text()), json.loads((rendered / "plan.json").read_text())


def _write(path: str | Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True, default=str))


def _src() -> Path:
    return Path(os.environ.get("RFT_MILES_SRC") or sys.exit("RFT_MILES_SRC is not set: run through launch/node.sh"))


def _model_launcher(recipe):
    """The model's launcher (upstream's, e.g. scripts/run_glm5_3_flash.py, or the cookbook's own), imported."""
    path = launcher_path(recipe, _src())
    spec = importlib.util.spec_from_file_location("rft_miles_model_launcher", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # its dataclasses resolve their annotations through it
    spec.loader.exec_module(module)
    return module


def _copy_dir(source: str, target: str) -> None:
    if Path(target).is_dir():
        return
    partial = Path(f"{target}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    started = time.monotonic()
    shutil.copytree(source, partial)
    partial.rename(target)
    print(f"[launch] copied {source} -> {target} in {time.monotonic() - started:.0f} s", flush=True)


def prep(recipe, paths, rendered) -> None:
    print(f"[prep] data ready in {prepare_data(recipe, paths)}")


def stage_model(recipe, paths, rendered) -> None:
    from rft_cookbook.air.stage_model import stage

    model = recipe["model"]
    stage(model["id"], model["revision"], paths["model_dir"])
    if source := model.get("processor_from"):
        stage(source["id"], source["revision"], paths["processor_dir"], tokenizer_only=True)
        missing = [name for name in source["files"] if not (Path(paths["processor_dir"]) / name).is_file()]
        if missing:
            raise SystemExit(f"[stage_model] {source['id']}@{source['revision'][:12]} has no {', '.join(missing)}")
    kind = model["megatron_type"]
    target = Path(paths["megatron_dir"]) / f"{kind}_torch_dist"
    if (target / "latest_checkpointed_iteration.txt").is_file():
        print(f"[stage_model] {target} is converted")
        return
    scratch = Path(paths["local_dir"]) / "convert"
    os.environ.update({"CONVERT_KEEP_PP1": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1"})
    gpus = GPUS_PER_NODE[recipe["compute"]["accelerator"]]
    script = _model_launcher(recipe).ScriptArgs(model_name=model["model_name"], ckpt_dir=str(scratch), num_nodes=1,
                                                    num_gpus_per_node=gpus, hf_checkpoint=paths["model_dir"])
    script.create_backend().convert_checkpoint(
        model_name=kind, megatron_model_type=kind, num_gpus_per_node=gpus, dir_dst=str(scratch),
        hf_checkpoint=paths["model_dir"], megatron_path=MEGATRON, extra_args=model.get("convert_args", ""))
    _copy_dir(str(scratch / f"{kind}_torch_dist"), str(target))
    print(f"[stage_model] converted {model['id']}@{model['revision'][:12]} -> {target}")


def _agent_inputs(recipe, paths) -> int:
    """The run's agent settings and prompt rows (node-local: one node, read by the rollout process)."""
    from rft_cookbook.agent.harness import initial_messages
    from rft_cookbook.tasks import load_task

    model = recipe["model"]
    _write(paths["agent_config"], {
        # one rollout process runs every episode (inline): the judge's concurrency is the run's
        "task": recipe["task"], "budgets": dict(recipe.get("agent") or {}), "resources": task_resources(recipe, paths),
        "abort_file": paths["abort_file"],
        "max_consecutive_exclusions": int(recipe["train"].get("max_consecutive_exclusions") or 0),
        "judge_outage_wait_s": float(recipe["train"].get("judge_outage_wait_s", 1800)),
        "tokenizer": paths["local_hf"], "chat_template_kwargs": model.get("chat_template_kwargs") or {},
        "episode_dir": f"{paths['episodes']}/train", "model": {"id": model["id"], "revision": model["revision"]},
        "tool_parser": model["tool_parser"], "served_model": "model",
        "request_timeout_s": float((recipe.get("rollout") or {}).get("request_timeout_s", 600))})
    task = load_task(recipe["task"], **(recipe.get("agent") or {}))
    rows = [json.loads(line) for line in Path(paths["train_rows"]).read_text().splitlines() if line.strip()]
    with open(paths["prompts"], "w") as out:
        for row in rows:
            out.write(json.dumps({"prompt": initial_messages(task, row), "label": row["ground_truth"],
                                  "metadata": {ROW_KEY: row, CONFIG_KEY: paths["agent_config"]}}) + "\n")
    return len(rows)


def _add_processor_files(recipe, paths) -> None:
    """Copy model.processor_from's files into the node-local copy of the model; the staged copy on
    the volume stays the checkpoint as published."""
    source = recipe["model"].get("processor_from") or {}
    for name in source.get("files", []):
        target = Path(paths["local_hf"]) / name
        if not target.exists():
            shutil.copyfile(Path(paths["processor_dir"]) / name, target)
            print(f"[train] {name} from {source['id']}@{source['revision'][:12]}", flush=True)


def train(recipe, paths, rendered) -> None:
    """Every node copies the model to local disk and writes the agent inputs (Ray places the
    rollout and the engines on any node); on several nodes, node 0 then runs miles on a Ray
    cluster of all of them (rft_cookbook.air.cluster) and the others serve as its workers."""
    from rft_cookbook.air import cluster
    from rft_cookbook.air.stage_model import free_gb
    from rft_cookbook.tools import service

    store = CheckpointStore(paths["checkpoints"], hash_files=False)
    latest, total = store.latest(), int(recipe["train"]["num_rollout"])
    if latest is not None and latest.step >= total:
        print(f"[train] {latest.name} already reaches train.num_rollout; nothing to train")
        return
    nodes = cluster.Nodes.from_env()
    if nodes.count != int(recipe["compute"]["nodes"]):
        raise SystemExit(f"the recipe trains on {recipe['compute']['nodes']} nodes, the job has {nodes.count}")
    print(f"[train] node {nodes.rank}/{nodes.count}: {free_gb(paths['local_dir']):.0f} GiB free in {paths['local_dir']}",
          flush=True)
    _copy_dir(paths["model_dir"], paths["local_hf"])
    _add_processor_files(recipe, paths)
    _copy_dir(paths["megatron_dir"], paths["local_megatron"])
    rows = _agent_inputs(recipe, paths)
    resources = task_resources(recipe, paths)
    process = service.start(recipe["task"], resources, paths["run_id"]) if resources else None
    try:
        run = lambda: _run_miles(recipe, paths, rendered, store, latest, total, rows)   # noqa: E731
        if nodes.count == 1:
            run()
            return
        rendezvous = nodes.rendezvous(Path(paths["run_dir"]) / "train" / "cluster")
        per_node = GPUS_PER_NODE[recipe["compute"]["accelerator"]]
        if nodes.rank == 0:
            os.environ["MILES_SCRIPT_EXTERNAL_RAY"] = "1"   # miles submits to this cluster instead of starting its own
            cluster.run_head(nodes, per_node, rendezvous, run)
        else:
            cluster.run_worker(nodes, per_node, rendezvous)
    finally:
        if process:
            service.stop(process, paths["run_id"])


def _run_miles(recipe, paths, rendered, store: CheckpointStore, latest, total: int, rows: int) -> None:
    launcher = json.loads((rendered / "miles.json").read_text())
    # the launcher is a one-command typer app: its options follow the script, with no command name
    command = [sys.executable, str(launcher_path(recipe, _src())), *launcher["options"],
               "--extra-args", shlex.join(launcher["extra_args"])]
    _write(Path(paths["run_dir"]) / "train" / f"command-{int(time.time())}.json", command)
    print(f"[train] {rows} prompts; {'resuming from ' + latest.name if latest else 'from the base model'}", flush=True)
    print("[train] " + shlex.join(command), flush=True)
    returncode = run_until_abort(command, Path(paths["abort_file"]), cwd=_src())
    if returncode != 0:
        raise SystemExit(f"[train] miles exited with {returncode}")
    final = store.latest()
    if final is None or final.step < total:
        raise SystemExit(f"[train] miles exited 0 but the newest published checkpoint is {final and final.name}")
    print(f"[train] published {final.name} at {final.location}")

STAGES = {"prep": prep, "stage_model": stage_model, "stage_judge": stage_judge, "calibrate": calibrate,
          "baseline_eval": baseline_eval, "train": train, "export": export, "eval": eval_, "compare": compare}


def main(argv=None) -> int:
    from rft_cookbook.air.jobs import from_job_env
    from rft_miles.render import resolve_run

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
    STAGES[argv[0]](recipe, paths, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

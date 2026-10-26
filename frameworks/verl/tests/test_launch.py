"""The train stage on several nodes: every node runs the tool service; the head runs verl."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from rft_cookbook.air import cluster
from rft_cookbook.tools import service
from rft_verl import launch
from rft_verl.render import resolve_run


@pytest.fixture
def run(tmp_path, monkeypatch):
    out = resolve_run("officeqa/verl/qwen3.5-35b-a3b_h100x16", site={"volume_root": str(tmp_path / "volume")},
                      run_id="officeqa-launch-01", out=tmp_path / "run")
    calls = []
    monkeypatch.setattr(service, "start", lambda task, resources, name: calls.append(("service", name)) or "proc")
    monkeypatch.setattr(service, "stop", lambda process, name: calls.append(("stop", name)))
    monkeypatch.setattr(cluster, "run_head", lambda nodes, gpus, rdv, body: calls.append(("head", gpus, rdv.name)))
    monkeypatch.setattr(cluster, "run_worker", lambda nodes, gpus, rdv: calls.append(("worker", gpus, rdv.name)))
    monkeypatch.setenv("NUM_NODES", "2")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.delenv("MASTER_PORT", raising=False)
    recipe, paths = launch._load(out)
    return recipe, paths, out, calls


@pytest.mark.parametrize("rank, role", [("0", "head"), ("1", "worker")])
def test_each_node_takes_its_role_around_the_tool_service(run, monkeypatch, rank, role):
    recipe, paths, out, calls = run
    monkeypatch.setenv("POD_RANK", rank)
    launch.train(recipe, paths, out)
    assert calls == [("service", "officeqa-launch-01"), (role, 8, "10.0.0.1-6379"), ("stop", "officeqa-launch-01")]


def test_a_job_with_another_node_count_does_not_train(run, monkeypatch):
    recipe, paths, out, calls = run
    monkeypatch.setenv("NUM_NODES", "1")
    with pytest.raises(SystemExit, match="2 nodes"):
        launch.train(recipe, paths, out)
    assert calls == []


def test_a_replicate_evaluates_no_baseline_of_its_own(run):
    recipe, paths, out, calls = run
    recipe["eval"]["baseline_run"] = "officeqa-launch-00"
    with pytest.raises(SystemExit, match="officeqa-launch-00's baseline"):
        launch.baseline_eval(recipe, paths, out)


def test_prep_writes_the_system_prompt_of_the_recipes_budgets(tmp_path, monkeypatch):
    # OfficeQA's prompt names its turn budget, so the converted rows must follow the recipe's
    out = resolve_run("officeqa/verl/qwen3.5-35b-a3b_h100x16", site={"volume_root": str(tmp_path / "volume")},
                      run_id="officeqa-prep-01", overrides=["agent.max_turns=64"], out=tmp_path / "run")
    recipe, paths = launch._load(out)
    rows = tmp_path / "rows"
    rows.mkdir()
    row = {"id": "q1", "split": "train", "messages": [{"role": "user", "content": "Q?"}], "ground_truth": "1",
           "metadata": {}}
    for split in ("train", "eval"):
        (rows / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(launch, "prepare_data", lambda recipe, paths: rows)
    launch.prep(recipe, paths, out)
    assert "64" in pd.read_parquet(paths["train_parquet"])["prompt"][0][0]["content"]


def test_a_recipe_that_changes_the_prompt_does_not_overwrite_another_runs_parquet(tmp_path):
    def paths(run_id, overrides):
        out = resolve_run("search/verl/qwen3.6-35b-a3b_h100x16", site={"volume_root": str(tmp_path / "volume")},
                          run_id=run_id, overrides=overrides, out=tmp_path / run_id)
        return launch._load(out)[1]

    default, more_turns, again = (paths("search-a-01", []), paths("search-b-01", ["agent.max_turns=20"]),
                                  paths("search-c-01", []))
    assert default["data_dir"] == more_turns["data_dir"]   # the task's rows are prepared once
    assert default["train_parquet"] != more_turns["train_parquet"]
    assert default["train_parquet"] == again["train_parquet"]

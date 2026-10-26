"""The stages every framework runs the same way (rft_cookbook.air.stages)."""
from __future__ import annotations

import json
import sys
import time

import pytest

from rft_cookbook.air import stages


def test_the_framework_runs_to_its_exit_code_and_stops_on_an_abort_request(tmp_path):
    abort = tmp_path / "train" / stages.ABORT_FILE
    assert stages.run_until_abort([sys.executable, "-c", "raise SystemExit(3)"], abort, poll_s=0.1) == 3
    writer = ("import json, sys, time; time.sleep(0.5); "
              f"open({str(abort)!r}, 'w').write(json.dumps({{'reason': 'judge down', 'at': time.time()}})); "
              "time.sleep(60)")
    abort.parent.mkdir(parents=True)
    started = time.monotonic()
    with pytest.raises(SystemExit, match="judge down"):
        stages.run_until_abort([sys.executable, "-c", writer], abort, poll_s=0.1, grace_s=5)
    assert time.monotonic() - started < 30


def test_an_abort_request_is_written_whole_and_honoured_only_when_new(tmp_path):
    abort = tmp_path / "train" / stages.ABORT_FILE
    stages.request_abort(str(abort), "32 episodes in a row failed")
    request = json.loads(abort.read_text())
    assert request["reason"] == "32 episodes in a row failed" and request["pid"] and request["host"]
    assert not list(abort.parent.glob("*.json.*"))   # no temporary file left behind
    stages.request_abort(None, "nowhere to write")   # no abort file: nothing to do
    # a request older than this run (beyond the clock skew) is an earlier attempt's
    request["at"] = time.time() - 10 * stages.SKEW_S
    abort.write_text(json.dumps(request))
    abort.unlink()   # run_until_abort clears a stale request before it starts: write it while running
    writer = (f"import json, time; open({str(abort)!r}, 'w').write({json.dumps(json.dumps(request))}); time.sleep(1)")
    assert stages.run_until_abort([sys.executable, "-c", writer], abort, poll_s=0.1) == 0


def test_only_a_task_with_tools_or_a_judge_has_run_time_resources():
    paths = {"run_id": "r-01", "data_dir": "/v/data/officeqa/abc", "judge_tokenizer_dir": "/v/models/judge-tok"}
    officeqa = {"task": "officeqa", "scoring": {"sandbox": "strict", "judge": {"id": "j", "models": ["m"]}}}
    resources = stages.task_resources(officeqa, paths)
    assert resources["tool_service"] == "r-01" and resources["data_dir"] == "/v/data/officeqa/abc/corpus"
    assert resources["judge"] == {"id": "j", "models": ["m"], "tokenizer": "/v/models/judge-tok"}
    assert stages.task_resources({"task": "math"}, paths) is None


def test_a_search_job_names_its_index_and_the_run_searches_it(monkeypatch, tmp_path):
    from rft_cookbook.air.jobs import from_job_env, plan
    from rft_cookbook.recipe import load_recipe

    env = {"RFT_RECIPE": "search/verl/qwen3.6-35b-a3b_h100x16", "RFT_RUN_ID": "search-01",
           "RFT_VOLUME_ROOT": "/Volumes/cat/sch/vol/rft", "RFT_VS_INDEX": "<vs-index>"}
    assert "vector_search_index" not in from_job_env(env)[2]          # a placeholder left in is not an index
    ref, run_id, site, _ = from_job_env({**env, "RFT_VS_INDEX": "cat.sch.passages_index"})
    recipe = load_recipe(ref)
    paths = plan(recipe, site, run_id, task_version="1")
    assert stages.task_resources(recipe, paths) == {"vector_search": {"index": "cat.sch.passages_index"}}
    assert stages.task_resources(load_recipe("math/verl/qwen3.5-2b_h100x8"), paths) is None


@pytest.mark.parametrize("failed", [False, True])
def test_compare_preserves_the_record_and_exits_unsuccessfully_for_failed_qualification(tmp_path, monkeypatch, failed):
    from rft_cookbook.evidence import qualification, report
    from rft_cookbook import recipe

    paths = {"run_dir": str(tmp_path), "eval_dir": str(tmp_path / "eval"), "baseline_dir": "baseline",
             "train_rows": "train.jsonl"}
    record = {"status": "failed" if failed else "measured", "checks": [{"name": "leakage_clean", "passed": not failed}]}
    monkeypatch.setattr(stages, "_exported", lambda _: {"step": 4})
    monkeypatch.setattr(report, "compare_runs", lambda *a, **k: {"summary": {"comparable": not failed}})
    monkeypatch.setattr(qualification, "record", lambda *a, **k: record)
    monkeypatch.setattr(recipe, "load_recipe", lambda *a, **k: {})
    if failed:
        with pytest.raises(SystemExit, match="qualification failed: leakage_clean"):
            stages.compare({"name": "toy", "eval": {}}, paths, None)
    else:
        stages.compare({"name": "toy", "eval": {}}, paths, None)
    assert json.loads((tmp_path / "qualification.json").read_text()) == record
    assert (tmp_path / "comparison.json").is_file()


def test_data_preparation_is_the_tasks_own_and_runs_once(monkeypatch, tmp_path):
    from rft_cookbook.data import MANIFEST
    from rft_cookbook.tasks.search import data

    calls = []
    monkeypatch.setattr(data, "prepare_recipe", lambda recipe, out, scratch: calls.append(out) or
                        (out.mkdir(parents=True), (out / MANIFEST).write_text("{}")))
    paths = {"data_dir": str(tmp_path / "data"), "local_dir": str(tmp_path / "local")}
    assert stages.prepare_data({"task": "search"}, paths) == tmp_path / "data"
    stages.prepare_data({"task": "search"}, paths)                    # prepared: reused
    assert calls == [tmp_path / "data"]
    monkeypatch.delattr(data, "prepare_recipe")
    with pytest.raises(SystemExit, match="no data preparation"):
        stages.prepare_data({"task": "search"}, {**paths, "data_dir": str(tmp_path / "other")})

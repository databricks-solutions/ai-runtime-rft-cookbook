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

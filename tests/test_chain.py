"""The recipe runner (rft_cookbook.air.chain) against a fake `databricks air`: stage order, filled-in jobs,
the baseline beside training, restarts that reuse finished jobs, failures and the GPU smoke."""
from __future__ import annotations

import json
import shlex
import threading

import pytest

from rft_cookbook.air import chain
from rft_cookbook.recipe import recipes_root

SITE = {"RFT_PROFILE": "ws", "RFT_VOLUME_ROOT": "/Volumes/cat/sch/vol/rft", "RFT_VERL_IMAGE": "cat.sch.verl:v1",
        "RFT_MILES_IMAGE": "cat.sch.miles:v1", "RFT_HF_TOKEN_SECRET": "scope/key"}
VERL = "officeqa/verl/qwen3.6-35b-a3b_h100x16"
MILES = "officeqa/miles/qwen3.6-35b-a3b_h100x16"
SEARCH = "search/verl/qwen3.6-35b-a3b_h100x16"


def answer(data: dict) -> str:
    return json.dumps({"v": 1, "ts": "2026-10-26T13:59:04Z", "data": data}, indent=2) + "\n"


class FakeAir:
    """`databricks air run` gives each job an id; `databricks air get` answers RUNNING once, then the
    job's outcome. Both answer as `-o json` does, after a line for each override applied."""

    def __init__(self, fail: str | None = None, statuses: dict[str, list[str]] | None = None):
        self.fail, self.statuses, self.submitted, self.next = fail, dict(statuses or {}), [], 100
        self.lock = threading.Lock()

    def __call__(self, argv):
        if argv[0] == "git":
            return "abc1234\n"
        if argv[:3] == ["databricks", "air", "run"]:
            assert argv[argv.index("-o") + 1] == "json"
            given = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--override"]
            sets = dict(s.split("=", 1) for s in given)
            stage = argv[argv.index("-f") + 1].rsplit("/", 1)[-1][:-len(".yaml")]
            with self.lock:
                self.next += 1
                job = str(self.next)
                self.submitted.append({"stage": stage, "job": job, **sets})
                self.statuses[job] = ["RUNNING", "FAILED" if stage == self.fail else "SUCCESS"]
            overridden = "".join(f"Override: setting {s.split('=', 1)[0]} to {{x}}\n" for s in given)
            return overridden + answer({"run_id": job, "dashboard_url": f"https://host/jobs/runs/{job}"})
        if argv[:3] == ["databricks", "air", "get"]:
            with self.lock:
                steps = self.statuses[argv[3]]
                status = steps.pop(0) if len(steps) > 1 else steps[0]
            return answer({"run_id": argv[3], "status": status})
        raise AssertionError(argv)


def make(recipe, air, tmp_path, overrides=(), run_id="r-01"):
    return chain.Chain(recipe, run_id, SITE, list(overrides), root=recipes_root().parent,
                       state=tmp_path / "state" / f"{run_id}.json", run=air, poll_s=0, log=lambda line: None)


def test_every_stage_runs_in_order_with_its_job_filled_in(tmp_path):
    air = FakeAir()
    make(VERL, air, tmp_path, ["train.seed=7"]).go()
    stages = [s["stage"] for s in air.submitted]
    assert set(stages) == {"prep", "stage_model", "stage_judge", "calibrate", "baseline_eval", "train", "export",
                           "eval", "compare"}
    order = [s for s in stages if s != "baseline_eval"]
    assert order == ["prep", "stage_model", "stage_judge", "calibrate", "train", "export", "eval", "compare"]
    prep = air.submitted[0]
    assert prep["env_variables.RFT_RUN_ID"] == "r-01" and prep["environment.unity_catalog_image"] == "cat.sch.verl:v1"
    assert shlex.split(prep["env_variables.RFT_SET"]) == ["train.seed=7"]
    assert prep["env_variables.RFT_CODE_COMMIT"] == "abc1234" and prep["env_variables.RFT_IMAGE"] == "cat.sch.verl:v1"
    assert prep["secrets.HF_TOKEN"] == "scope/key"
    assert json.loads((tmp_path / "state" / "r-01.json").read_text()).keys() == set(stages)


def test_a_run_compared_with_another_baseline_skips_its_own_and_miles_trains_on_its_image(tmp_path):
    air = FakeAir()
    make(MILES, air, tmp_path, ["eval.baseline_run=oqa-verl-01"]).go()
    by_stage = {s["stage"]: s for s in air.submitted}
    assert "baseline_eval" not in by_stage
    assert by_stage["train"]["environment.unity_catalog_image"] == "cat.sch.miles:v1"
    assert by_stage["eval"]["environment.unity_catalog_image"] == "cat.sch.verl:v1"   # vLLM serves every export


def test_a_restart_reuses_what_succeeded_waits_for_what_runs_and_redoes_what_failed(tmp_path):
    state = tmp_path / "state" / "r-01.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"prep": "1", "stage_model": "2", "stage_judge": "3"}))
    air = FakeAir(statuses={"1": ["SUCCESS"], "2": ["RUNNING", "SUCCESS"], "3": ["FAILED"]})
    make(VERL, air, tmp_path).go()
    submitted = [s["stage"] for s in air.submitted]
    assert "prep" not in submitted and "stage_model" not in submitted   # reused, and waited for
    assert submitted.count("stage_judge") == 1                          # failed before: submitted again
    # without overrides the job keeps its own empty RFT_SET: an empty override would unset it
    assert all("env_variables.RFT_SET" not in s for s in air.submitted)


def test_an_image_run_by_its_tag_is_recorded_by_its_digest(tmp_path):
    air = FakeAir()
    site = {**SITE, "RFT_VERL_IMAGE_DIGEST": "verl@sha256:aa"}
    chain.Chain(VERL, "r-01", site, [], root=recipes_root().parent, state=tmp_path / "r-01.json", run=air,
                poll_s=0, log=lambda line: None).go()
    by_stage = {s["stage"]: s for s in air.submitted}
    assert by_stage["prep"]["environment.unity_catalog_image"] == "cat.sch.verl:v1"
    assert by_stage["prep"]["env_variables.RFT_IMAGE"] == "verl@sha256:aa"
    assert by_stage["train"]["env_variables.RFT_IMAGE"] == "verl@sha256:aa"


def test_the_jobs_that_search_get_the_index_from_the_site_file(tmp_path):
    air = FakeAir()
    site = {**SITE, "RFT_VS_INDEX": "cat.sch.passages_index"}
    chain.Chain(SEARCH, "r-01", site, [], root=recipes_root().parent, state=tmp_path / "r-01.json", run=air,
                poll_s=0, log=lambda line: None).go()
    by_stage = {s["stage"]: s for s in air.submitted}
    assert {stage for stage, s in by_stage.items() if "env_variables.RFT_VS_INDEX" in s} == {
        "baseline_eval", "train", "eval"}
    assert by_stage["train"]["env_variables.RFT_VS_INDEX"] == "cat.sch.passages_index"
    with pytest.raises(RuntimeError, match="needs RFT_VS_INDEX"):
        make(SEARCH, FakeAir(), tmp_path, run_id="r-02").go()


def test_a_failed_stage_stops_the_chain(tmp_path):
    air = FakeAir(fail="train")
    with pytest.raises(RuntimeError, match="train .* ended FAILED"):
        make(VERL, air, tmp_path).go()
    assert "export" not in [s["stage"] for s in air.submitted]


def test_the_smoke_trains_two_updates_resumes_a_third_and_evaluates_that(tmp_path):
    air = FakeAir()
    make(MILES, air, tmp_path, ["eval.baseline_run=oqa-verl-01"]).go(smoke=True)
    trains = [shlex.split(s["env_variables.RFT_SET"]) for s in air.submitted if s["stage"] == "train"]
    assert trains == [["eval.baseline_run=oqa-verl-01", "train.num_rollout=2", "train.save_interval=1"],
                      ["eval.baseline_run=oqa-verl-01", "train.num_rollout=3", "train.save_interval=1"]]
    later = {s["stage"]: shlex.split(s["env_variables.RFT_SET"]) for s in air.submitted}
    assert "train.num_rollout=3" in later["export"] and "train.num_rollout=3" in later["compare"]


def test_a_submission_without_a_run_id_fails_and_an_unanswered_poll_is_asked_again(tmp_path):
    def broken(argv):
        if argv[0] == "git":
            return "abc1234\n"
        if argv[:3] == ["databricks", "air", "run"]:
            return "Error: invalid config: field docker_image not found\n"
        return "Error: request timed out\n"
    runner = make(VERL, broken, tmp_path)
    with pytest.raises(RuntimeError, match="submitting prep failed: Error: invalid config"):
        runner.submit("prep", [])
    assert runner.status("123") == "UNKNOWN"


def test_the_site_file_reads_as_the_shell_would(tmp_path):
    site = tmp_path / "site.env"
    site.write_text("# comment\nRFT_PROFILE=ws   # the profile\nRFT_VERL_IMAGE='r/v:t@sha256:x'\n")
    assert chain.read_site(site) == {"RFT_PROFILE": "ws", "RFT_VERL_IMAGE": "r/v:t@sha256:x"}

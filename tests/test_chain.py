"""The recipe runner (rft_cookbook.air.chain) against a fake `air`: stage order, filled-in jobs,
the baseline beside training, restarts that reuse finished jobs, failures and the GPU smoke."""
from __future__ import annotations

import json
import shlex
import threading

import pytest

from rft_cookbook.air import chain
from rft_cookbook.recipe import recipes_root

SITE = {"RFT_PROFILE": "ws", "RFT_VOLUME_ROOT": "/Volumes/cat/sch/vol/rft", "RFT_VERL_IMAGE": "verl@sha256:aa",
        "RFT_MILES_IMAGE": "miles@sha256:bb", "RFT_HF_TOKEN_SECRET": "scope/key"}
VERL = "officeqa/verl/qwen3.6-35b-a3b_h100x16"
MILES = "officeqa/miles/qwen3.6-35b-a3b_h100x16"
SEARCH = "search/verl/qwen3.6-35b-a3b_h100x16"


class FakeAir:
    """`air run` gives each job an id; `air get run` answers RUNNING once, then the job's outcome."""

    def __init__(self, fail: str | None = None, statuses: dict[str, list[str]] | None = None):
        self.fail, self.statuses, self.submitted, self.next = fail, dict(statuses or {}), [], 100
        self.lock = threading.Lock()

    def __call__(self, argv):
        if argv[0] == "git":
            return "abc1234\n"
        if argv[:2] == ["air", "run"]:
            sets = dict(s.split("=", 1) for s in argv[argv.index("--override") + 1:])
            stage = argv[argv.index("-f") + 1].rsplit("/", 1)[-1][:-len(".yaml")]
            with self.lock:
                self.next += 1
                job = str(self.next)
                self.submitted.append({"stage": stage, "job": job, **sets})
                self.statuses[job] = ["RUNNING", "FAILED" if stage == self.fail else "SUCCESS"]
            return f"Submitted.\nJob Run ID: {job}\n"
        if argv[:3] == ["air", "get", "run"]:
            with self.lock:
                steps = self.statuses[argv[3]]
                status = steps.pop(0) if len(steps) > 1 else steps[0]
            return json.dumps({"data": {"status": status}}) + "\n"
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
    assert prep["env_variables.RFT_RUN_ID"] == "r-01" and prep["environment.docker_image.url"] == "verl@sha256:aa"
    assert shlex.split(prep["env_variables.RFT_SET"]) == ["train.seed=7"]
    assert prep["env_variables.RFT_CODE_COMMIT"] == "abc1234" and prep["env_variables.RFT_IMAGE"] == "verl@sha256:aa"
    assert prep["secrets.HF_TOKEN"] == "scope/key"
    assert json.loads((tmp_path / "state" / "r-01.json").read_text()).keys() == set(stages)


def test_a_run_compared_with_another_baseline_skips_its_own_and_miles_trains_on_its_image(tmp_path):
    air = FakeAir()
    make(MILES, air, tmp_path, ["eval.baseline_run=oqa-verl-01"]).go()
    by_stage = {s["stage"]: s for s in air.submitted}
    assert "baseline_eval" not in by_stage
    assert by_stage["train"]["environment.docker_image.url"] == "miles@sha256:bb"
    assert by_stage["eval"]["environment.docker_image.url"] == "verl@sha256:aa"   # vLLM serves every export


def test_a_restart_reuses_what_succeeded_waits_for_what_runs_and_redoes_what_failed(tmp_path):
    state = tmp_path / "state" / "r-01.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"prep": "1", "stage_model": "2", "stage_judge": "3"}))
    air = FakeAir(statuses={"1": ["SUCCESS"], "2": ["RUNNING", "SUCCESS"], "3": ["FAILED"]})
    make(VERL, air, tmp_path).go()
    submitted = [s["stage"] for s in air.submitted]
    assert "prep" not in submitted and "stage_model" not in submitted   # reused, and waited for
    assert submitted.count("stage_judge") == 1                          # failed before: submitted again


def test_an_image_run_by_its_tag_is_recorded_by_its_digest(tmp_path):
    air = FakeAir()
    site = {**SITE, "RFT_VERL_IMAGE": "verl:v1", "RFT_VERL_IMAGE_DIGEST": "verl@sha256:aa"}
    chain.Chain(VERL, "r-01", site, [], root=recipes_root().parent, state=tmp_path / "r-01.json", run=air,
                poll_s=0, log=lambda line: None).go()
    by_stage = {s["stage"]: s for s in air.submitted}
    assert by_stage["prep"]["environment.docker_image.url"] == "verl:v1"
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


def test_the_site_file_reads_as_the_shell_would(tmp_path):
    site = tmp_path / "site.env"
    site.write_text("# comment\nRFT_PROFILE=ws   # the profile\nRFT_VERL_IMAGE='r/v:t@sha256:x'\n")
    assert chain.read_site(site) == {"RFT_PROFILE": "ws", "RFT_VERL_IMAGE": "r/v:t@sha256:x"}

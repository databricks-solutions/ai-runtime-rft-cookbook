"""Run a recipe's stages as AI Runtime jobs, in order, from a workstation with the `air` CLI.

    python -m rft_cookbook.air.chain <recipe> <run id> [--set "train.seed=7"] [--smoke] [--site site.env]

Each stage is the recipe's checked-in job (`recipes/<recipe>/jobs/<stage>.yaml`), submitted with
its placeholders filled from site.env (profile, volume root, images, Hugging Face secret), the run's
overrides (`RFT_SET`, the same for every stage), and the commit and image for the qualification
record (`RFT_CODE_COMMIT`, `RFT_IMAGE`). The baseline evaluation runs beside training; a run
compared with another run's baseline (`eval.baseline_run`) skips it. The chain waits for each job;
one that failed stops it. Every job it submitted is kept in a state file (`~/.rft-runs/<run id>.json`,
not /tmp: a workstation restart must not lose it): run again, the chain reuses a job that succeeded
and waits for one that still runs, so a chain can be resumed.

`--smoke` is tier 5 of docs/architecture.md "Testing": train two updates with a save each, resume
for a third, export, evaluate the export and compare; its qualification record says smoke-tested.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

FINAL = {"SUCCESS", "FAILED", "FAILURE", "CANCELED", "CANCELLED", "TIMEDOUT", "TIMED_OUT", "ERROR", "INTERNAL_ERROR",
         "TERMINATED", "SKIPPED"}
# the training length and save frequency each framework's recipes name
STEPS = {"verl": ("train.total_training_steps", "train.save_freq"),
         "miles": ("train.num_rollout", "train.save_interval")}


def _run(argv: list[str]) -> str:
    done = subprocess.run(argv, capture_output=True, text=True)
    return done.stdout + done.stderr   # air prints the job id on either


def read_site(path: Path) -> dict[str, str]:
    site = {}
    for line in path.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words and "=" in words[0]:
            key, _, value = words[0].partition("=")
            site[key] = value
    return site


class Chain:
    def __init__(self, recipe: str, run_id: str, site: dict[str, str], overrides: list[str], *, root: Path,
                 state: Path, run: Callable[[list[str]], str] | None = None, poll_s: float = 120.0,
                 log: Callable[[str], None] = print):
        from rft_cookbook.recipe import load_recipe

        self.recipe_ref, self.run_id, self.site, self.overrides = recipe, run_id, site, list(overrides)
        self.recipe = load_recipe(recipe, overrides=self.overrides, root=root / "recipes")
        self.jobs, self.state_path, self.poll_s, self.log = root / "recipes" / recipe / "jobs", state, poll_s, log
        self.run = run or _run
        # "-dirty" when the checkout has uncommitted changes: the record's code hash names them exactly
        describe = ["git", "-C", str(root), "describe", "--always", "--dirty", "--abbrev=7"]
        self.commit = self.run(describe).strip() or None
        self.state = json.loads(state.read_text()) if state.exists() else {}
        self._lock = threading.Lock()

    def stages(self) -> list[str]:
        stages = list(self.recipe["stages"])
        if (self.recipe.get("eval") or {}).get("baseline_run") not in (None, self.run_id):
            stages = [s for s in stages if s != "baseline_eval"]   # evaluated in the run it compares with
        return stages

    def stage(self, stage: str, extra: list[str] = (), key: str | None = None) -> None:
        """Submit (or reuse) the stage's job and wait for it; raises if it did not succeed."""
        key = key or stage
        job = self.state.get(key)
        status = self.status(job) if job else None
        if status == "SUCCESS":
            self.log(f"{key}: reusing {job} (SUCCESS)")
            return
        if job is None or status in FINAL:
            job = self.submit(stage, [*self.overrides, *extra])
            with self._lock:
                self.state[key] = job
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                self.state_path.write_text(json.dumps(self.state, indent=1))
            self.log(f"{key}: submitted {job}")
        while (status := self.status(job)) not in FINAL:
            time.sleep(self.poll_s)
        self.log(f"{key}: {job} {status}")
        if status != "SUCCESS":
            raise RuntimeError(f"{key} ({job}) ended {status}")

    def submit(self, stage: str, overrides: list[str]) -> str:
        path = self.jobs / f"{stage}.yaml"
        text = path.read_text()
        image = self.site["RFT_MILES_IMAGE"] if "<miles-image>" in text else self.site["RFT_VERL_IMAGE"]
        sets = [f"env_variables.RFT_RUN_ID={self.run_id}",
                f"env_variables.RFT_VOLUME_ROOT={self.site['RFT_VOLUME_ROOT']}",
                f"environment.docker_image.url={image}", f"env_variables.RFT_SET={shlex.join(overrides)}",
                f"env_variables.RFT_IMAGE={image}"]
        if self.commit:
            sets.append(f"env_variables.RFT_CODE_COMMIT={self.commit}")
        if re.search(r"^secrets:", text, re.M):
            sets.append(f"secrets.HF_TOKEN={self.site['RFT_HF_TOKEN_SECRET']}")
        out = self.run(["air", "run", "-f", str(path), "-p", self.site["RFT_PROFILE"], "--override", *sets])
        found = re.search(r"Job Run ID: (\d+)", out)
        if not found:
            raise RuntimeError(f"submitting {stage} failed: {out[-500:]}")
        return found.group(1)

    def status(self, job: str) -> str:
        out = self.run(["air", "get", "run", job, "-p", self.site["RFT_PROFILE"], "--json"])
        for line in out.splitlines():
            try:
                return json.loads(line)["data"]["status"]
            except (ValueError, KeyError, TypeError):
                continue
        return "UNKNOWN"   # the API did not answer: asked again at the next poll

    def go(self, *, smoke: bool = False) -> None:
        stages, baseline, failures = self.stages(), None, []

        def beside(stage):
            try:
                self.stage(stage)
            except Exception as error:  # noqa: BLE001 - reported where the chain waits for it
                failures.append(error)

        for stage in stages:
            if stage == "baseline_eval":
                baseline = threading.Thread(target=beside, args=(stage,))
                baseline.start()
                continue
            if stage == "compare" and baseline is not None:
                baseline.join()
                if failures:
                    raise failures[0]
            if stage == "train" and smoke:
                steps, save = STEPS[self.recipe["framework"]]
                self.stage("train", [f"{steps}=2", f"{save}=1"], key="train-2")
                self.stage("train", [f"{steps}=3", f"{save}=1"], key="train-3")
                self.overrides += [f"{steps}=3", f"{save}=1"]   # what the later stages read the run as
                continue
            self.stage(stage)
        if baseline is not None and baseline.is_alive():
            baseline.join()
        if failures:
            raise failures[0]
        self.log(f"{self.run_id}: every stage succeeded")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recipe")
    parser.add_argument("run_id")
    parser.add_argument("--set", default="", help="recipe overrides, shell-quoted (the same for every stage)")
    parser.add_argument("--smoke", action="store_true", help="tier-5 GPU smoke: two updates, resume, export, evaluate")
    parser.add_argument("--site", default="site.env")
    args = parser.parse_args(argv)
    from rft_cookbook.recipe import recipes_root

    root = recipes_root().parent
    chain = Chain(args.recipe, args.run_id, read_site(Path(args.site)), shlex.split(args.set), root=root,
                  state=Path.home() / ".rft-runs" / f"{args.run_id}.json")
    try:
        chain.go(smoke=args.smoke)
    except RuntimeError as error:
        print(f"[chain] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

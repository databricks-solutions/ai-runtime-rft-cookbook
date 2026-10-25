"""The stop guard of a training process: too many excluded episodes, with a pause for a judge outage.

An adapter counts each finished episode. An excluded one (an infrastructure failure, an invalid
score, a truncated terminal turn) adds to a streak; an episode that tells the update something (a
positive score, or one a judge decided) ends it. A valid zero that needed no judge (a wrong or
malformed answer) leaves the streak as it is: in a judge outage those still score while every
episode that needs the judge is excluded, so a streak they reset would never form, and the updates
would train on the judge-free zeros alone.

At `max_consecutive` exclusions the guard stops the run, unless the exclusions are judge failures:
then it is an outage, and while it lasts an invalid score is retried every `interval_s` (which
holds that episode's rollout) until the judge answers again, which ends the streak, or until
`wait_s` has passed since the outage began, which stops the run.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from rft_cookbook.agent.episode import EpisodeRecord, Score
from rft_cookbook.scoring.base import score_episode


def judge_failure(score: Score | None) -> bool:
    """The score is invalid because the judge gave no verdict (worth asking again)."""
    return score is not None and not score.valid and (score.detail or "").startswith("JudgeError")


def informative(record: EpisodeRecord) -> bool:
    score = record.score
    return score is not None and score.valid and (score.value > 0 or score.judge is not None)


class ExclusionGuard:
    def __init__(self, max_consecutive: int, *, wait_s: float = 1800.0, interval_s: float = 60.0,
                 clock=time.monotonic):
        self.max_consecutive, self.wait_s, self.interval_s, self.clock = int(max_consecutive), wait_s, interval_s, clock
        self.streak = 0
        self.outage_since: float | None = None
        self.stopped = False

    @property
    def tripped(self) -> bool:
        return bool(self.max_consecutive) and self.streak >= self.max_consecutive

    def _waited_out(self) -> bool:
        return self.outage_since is not None and self.clock() - self.outage_since >= self.wait_s

    async def score(self, scorer, record: EpisodeRecord, ground_truth: Any, metadata: dict[str, Any] | None) -> Score:
        """`score_episode`; during an outage, again every `interval_s` until the judge answers."""
        score = await score_episode(scorer, record, ground_truth, metadata)
        while self.outage_since is not None and judge_failure(score) and not self._waited_out():
            await asyncio.sleep(self.interval_s)
            score = await score_episode(scorer, record, ground_truth, metadata)
        return score

    def record(self, record: EpisodeRecord | None, excluded: bool, reason: str = "") -> str | None:
        """Count a finished episode (None: one that could not run); returns why to stop, once."""
        if not excluded:
            if record is not None and informative(record):
                self.streak, self.outage_since = 0, None
            return None
        self.streak += 1
        if not self.tripped or self.stopped:
            return None
        if record is not None and judge_failure(record.score) and not self._waited_out():
            if self.outage_since is None:
                self.outage_since = self.clock()
            return None
        self.stopped = True
        if self.outage_since is not None:
            return (f"the judge gave no verdict for {self.wait_s:.0f} s ({self.streak} episodes excluded "
                    f"without a decided one between them); last: {reason}")
        return f"{self.streak} episodes were excluded without a decided one between them; last: {reason}"

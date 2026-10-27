"""The training stop guard (rft_cookbook.agent.guard): what counts, and a judge outage."""
from __future__ import annotations

import asyncio

from rft_cookbook.agent.episode import EpisodeRecord, Score
from rft_cookbook.agent.guard import ExclusionGuard
from rft_cookbook.scoring.base import RewardResult
from rft_cookbook.scoring.judge import JudgeError


def episode(score: Score | None = None, status: str = "ok") -> EpisodeRecord:
    return EpisodeRecord(episode_id="e", task="t", task_version="1", question_id="q", status=status, score=score)


ZERO = Score(0.0, True, "p")                                   # a wrong answer: no judge needed
JUDGED_ZERO = Score(0.0, True, "p", judge={"verdict": "unsupported"})
POSITIVE = Score(1.0, True, "p")
NO_VERDICT = Score(0.0, False, "p", detail="JudgeError: transport: HTTP 503")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_zeros_that_needed_no_judge_do_not_hide_a_run_of_exclusions():
    guard = ExclusionGuard(3)
    for _ in range(2):
        assert guard.record(episode(status="infra_failure"), True, "tool service gone") is None
        assert guard.record(episode(ZERO), False) is None        # before: this reset the count
    stop = guard.record(episode(status="infra_failure"), True, "tool service gone")
    assert stop and "3 episodes were excluded" in stop and "tool service gone" in stop
    assert guard.record(episode(status="infra_failure"), True, "again") is None   # said once


def test_a_decided_episode_ends_the_streak():
    for decided in (JUDGED_ZERO, POSITIVE):
        guard = ExclusionGuard(3)
        guard.record(episode(status="infra_failure"), True, "x")
        guard.record(episode(status="infra_failure"), True, "x")
        guard.record(episode(decided), False)
        assert guard.streak == 0


def test_a_judge_outage_pauses_and_the_run_goes_on_when_the_judge_answers():
    clock = Clock()
    guard = ExclusionGuard(2, wait_s=600, interval_s=0, clock=clock)
    for _ in range(3):   # the judge is down: the streak reaches the limit, but it is an outage, not a stop
        assert guard.record(episode(NO_VERDICT), True, "invalid_score") is None
    assert guard.outage_since == 0.0

    replies = [JudgeError("transport", "HTTP 503")] * 3 + [RewardResult(True, 1.0, judge={"verdict": "supported"})]

    class Scorer:
        policy, calls = "p", 0

        async def score(self, record, ground_truth, metadata):
            self.calls += 1
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

    scorer, record = Scorer(), episode()
    score = asyncio.run(guard.score(scorer, record, "4", {}))   # waits for the judge instead of failing
    assert score.valid and score.value == 1.0 and scorer.calls == 4
    assert guard.record(record, False) is None and guard.streak == 0 and guard.outage_since is None


def test_an_outage_longer_than_the_wait_stops_the_run():
    clock = Clock()
    guard = ExclusionGuard(2, wait_s=600, interval_s=0, clock=clock)
    guard.record(episode(NO_VERDICT), True, "invalid_score")
    assert guard.record(episode(NO_VERDICT), True, "invalid_score") is None   # the outage begins
    clock.now = 601.0

    class DownScorer:
        policy = "p"

        async def score(self, record, ground_truth, metadata):
            raise JudgeError("transport", "HTTP 503")

    record = episode()
    assert not asyncio.run(guard.score(DownScorer(), record, "4", {})).valid   # no retry past the wait
    stop = guard.record(record, True, "invalid_score")
    assert stop and "no verdict for 600 s" in stop


def test_no_limit_means_no_guard():
    guard = ExclusionGuard(0)
    assert all(guard.record(episode(status="infra_failure"), True, "x") is None for _ in range(100))

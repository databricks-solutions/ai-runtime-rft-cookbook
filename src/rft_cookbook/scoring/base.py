"""Scorers read an episode record and return a reward with a validity flag.

`valid=False` means the scorer could not decide (a judge outage, an unparsable verdict); such a
reward never reaches an update. A wrong or missing answer is a valid low score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from rft_cookbook.agent.episode import EpisodeRecord, Score

logger = logging.getLogger(__name__)


@dataclass
class RewardResult:
    valid: bool
    score: float
    components: dict[str, float] = field(default_factory=dict)
    judge: dict[str, Any] | None = None
    detail: str | None = None


class Scorer(Protocol):
    policy: str
    """Versioned scorer policy id; part of every score and qualification record."""

    async def score(self, episode: EpisodeRecord, ground_truth: Any, metadata: dict[str, Any]) -> RewardResult: ...


async def score_episode(scorer: Scorer, episode: EpisodeRecord, ground_truth: Any,
                        metadata: dict[str, Any] | None = None) -> Score:
    """Score and attach the result to the episode. A scorer exception is an invalid score."""
    from rft_cookbook.agent.episode import Score

    try:
        result = await scorer.score(episode, ground_truth, metadata or {})
    except Exception as error:  # noqa: BLE001 - recorded as an invalid score, never a zero reward
        logger.warning("scorer %s failed on %s: %s", scorer.policy, episode.episode_id, error)
        result = RewardResult(valid=False, score=0.0, detail=f"{type(error).__name__}: {error}")
    episode.score = Score(value=float(result.score), valid=bool(result.valid), policy=scorer.policy,
                          components=dict(result.components), judge=result.judge, detail=result.detail)
    return episode.score

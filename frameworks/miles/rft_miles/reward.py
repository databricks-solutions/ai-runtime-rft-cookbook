"""The miles reward hook: the score the agent function left in the sample's metadata.

    --custom-rm-path rft_miles.reward.reward

The cookbook scores the whole episode in the agent function (rft_miles.agent); session server v2
merges what that function returns into every sample of the session. An excluded episode (an
infrastructure failure, an invalid score) has no reward: miles drops a group with a sample
without reward and samples another prompt in its place, as verl's trainer drops and refills a
failed group.
"""
from __future__ import annotations

import logging

from rft_miles.agent import REWARD_KEY

logger = logging.getLogger(__name__)


async def reward(args, sample, **kwargs) -> float | None | list[float | None]:
    """One sample's reward, or a list of them: miles passes a session's samples together
    (`batched_async_rm`) when the generate function returns several, and one at a time otherwise."""
    if isinstance(sample, list):
        return [_reward(s) for s in sample]
    return _reward(sample)


def _reward(sample) -> float | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if REWARD_KEY not in metadata:   # the agent function failed before it returned
        logger.warning("rft: sample %s has no cookbook score; its group is dropped", getattr(sample, "index", None))
        return None
    value = metadata[REWARD_KEY]
    return None if value is None else float(value)

"""Failure types shared by drivers and adapters.

An infrastructure failure ends an episode as `infra_failure`; it is never turned into a reward.
"""
from __future__ import annotations


class InfraError(RuntimeError):
    """Serving, tool or judge infrastructure failed; not the model's fault."""


class EpisodeExcluded(RuntimeError):
    """Raised by a framework adapter to keep an episode out of the update.

    Carries the finished record so the adapter can still write it as evidence.
    """

    def __init__(self, reason: str, record=None):
        super().__init__(reason)
        self.reason = reason
        self.record = record

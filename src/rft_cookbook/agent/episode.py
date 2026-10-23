"""The episode record: one versioned JSON document per training or evaluation episode.

Every driver (verl's native loop, the Chat Completions harness) writes this record, and
scorers, metrics and evidence read only this record. Token arrays stay with the framework;
the record keeps counts and hashes of them in `collection`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "rft-cookbook/episode/v1"

CALL_STATUSES = ("ok", "error", "timeout", "rejected", "skipped")
DELIVERIES = ("delivered", "not_admitted", "not_run")
TERMINATIONS = ("answer", "no_answer", "max_turns", "context_budget", "tool_budget", "length",
                "policy_error", "infra_error")
STATUSES = ("ok", "policy_failure", "infra_failure")
ELIGIBILITY = ("eligible", "zero", "excluded")


@dataclass
class ToolCallRecord:
    name: str
    arguments: str
    """The arguments exactly as the model wrote them."""
    call_id: str | None = None
    parse_error: str | None = None
    status: str = "skipped"
    executed: bool = False
    """The tool implementation ran (a malformed call is answered without running anything)."""
    delivery: str = "not_run"
    observation_id: str | None = None
    delivered_text: str | None = None
    """The text that entered the model's context; None unless `delivery == "delivered"`."""
    delivered_to_turn: int | None = None
    """Index of the first model turn whose context contained the observation."""
    truncated: bool = False
    latency_s: float | None = None
    sandbox: str | None = None
    detail: str | None = None


@dataclass
class TurnRecord:
    index: int
    text: str
    reasoning: str | None = None
    finish_reason: str = "stop"
    """`stop`, `length`, or the driver's own reason when neither applies."""
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    policy_version: int | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


@dataclass
class Termination:
    reason: str
    answer: str | None = None
    terminal_turn: int | None = None
    detail: str | None = None


@dataclass
class Score:
    value: float
    valid: bool
    policy: str
    """Scorer policy id, e.g. `math/rule-v1`."""
    components: dict[str, float] = field(default_factory=dict)
    judge: dict[str, Any] | None = None
    """Judge model identity, verdict and attempts, when a judge was consulted."""
    detail: str | None = None


@dataclass
class Attestation:
    """What one framework training sample made from this episode contains."""

    framework: str
    sample_id: str
    terminal_turn: int | None
    terminal_span_complete: bool
    """The full token span of the terminal turn survived collection (no truncation cut it)."""
    tokens: int
    loss_tokens: int
    token_mask_sha256: str
    logprob_coverage: float | None
    """Fraction of loss tokens with a rollout log-probability; None when none were collected."""
    eligibility: str = "excluded"
    reason: str | None = None


@dataclass
class EpisodeRecord:
    episode_id: str
    task: str
    task_version: str
    question_id: str
    split: str | None = None
    sample_index: int | None = None
    attempt: int = 0
    model: dict[str, Any] = field(default_factory=dict)
    """`id`, `revision`, and `checkpoint` when the policy is a trained checkpoint."""
    settings: dict[str, Any] = field(default_factory=dict)
    """Hashes of prompt, tool schemas and chat template; parser; budgets."""
    turns: list[TurnRecord] = field(default_factory=list)
    termination: Termination | None = None
    score: Score | None = None
    status: str = "ok"
    collection: list[Attestation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    # ---- views -------------------------------------------------------------------------

    @property
    def answer(self) -> str | None:
        return self.termination.answer if self.termination else None

    def calls(self) -> list[ToolCallRecord]:
        return [call for turn in self.turns for call in turn.tool_calls]

    def delivered(self) -> list[ToolCallRecord]:
        return [call for call in self.calls() if call.delivery == "delivered"]

    def policy_versions(self) -> list[int]:
        return sorted({t.policy_version for t in self.turns if t.policy_version is not None})

    # ---- serialization -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeRecord:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported episode schema {data.get('schema')!r}; expected {SCHEMA}")
        data = dict(data)
        data["turns"] = [_build(TurnRecord, {**t, "tool_calls": [_build(ToolCallRecord, c) for c in t.get("tool_calls", [])]})
                         for t in data.get("turns", [])]
        if data.get("termination") is not None:
            data["termination"] = _build(Termination, data["termination"])
        if data.get("score") is not None:
            data["score"] = _build(Score, data["score"])
        data["collection"] = [_build(Attestation, a) for a in data.get("collection", [])]
        record = _build(cls, data)
        record.validate()
        return record

    @classmethod
    def from_json(cls, text: str) -> EpisodeRecord:
        return cls.from_dict(json.loads(text))

    def validate(self) -> None:
        """Reject records no consumer could interpret. Raises ValueError."""
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")
        if self.termination is not None and self.termination.reason not in TERMINATIONS:
            raise ValueError(f"termination {self.termination.reason!r} not in {TERMINATIONS}")
        for i, turn in enumerate(self.turns):
            if turn.index != i:
                raise ValueError(f"turn {i} has index {turn.index}")
            for call in turn.tool_calls:
                if call.status not in CALL_STATUSES:
                    raise ValueError(f"call status {call.status!r} not in {CALL_STATUSES}")
                if call.delivery not in DELIVERIES:
                    raise ValueError(f"delivery {call.delivery!r} not in {DELIVERIES}")
                if (call.delivery == "delivered") != (call.delivered_text is not None):
                    raise ValueError("delivered_text is set exactly when the observation was delivered")
                if call.delivery == "delivered" and call.delivered_to_turn is None:
                    raise ValueError("a delivered observation names the turn that received it")
        for attestation in self.collection:
            if attestation.eligibility not in ELIGIBILITY:
                raise ValueError(f"eligibility {attestation.eligibility!r} not in {ELIGIBILITY}")


def _build(cls, data: dict[str, Any]):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**data)


def token_mask_sha256(token_ids, loss_mask) -> str:
    """Hash of token ids and loss mask together, so a changed mask is a changed sample."""
    digest = hashlib.sha256()
    digest.update(json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode())
    digest.update(b"|")
    digest.update(json.dumps([int(m) for m in loss_mask], separators=(",", ":")).encode())
    return digest.hexdigest()


def eligibility(record: EpisodeRecord, terminal_span_complete: bool) -> tuple[str, str | None]:
    """Reward eligibility for one collected sample (docs/architecture.md, "Episode format")."""
    if record.status == "infra_failure":
        return "excluded", "infra_failure"
    if record.score is None or not record.score.valid:
        return "excluded", "invalid_score"
    if record.answer is not None and not terminal_span_complete:
        return "excluded", "terminal_truncated"
    if record.answer is None:
        return "zero", None
    return "eligible", None

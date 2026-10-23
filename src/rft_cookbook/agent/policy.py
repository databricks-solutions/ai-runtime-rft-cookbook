"""The decision rules of an episode, shared by every driver.

A driver (verl's native loop, the Chat Completions harness) owns generation, parsing and
tokenization. After each model turn it hands the parsed calls and its token counts to
`EpisodePolicy`, which decides which calls run, when the episode ends and whether observations
fit in the context. The rules (docs/architecture.md, "Episode format"):

- calls in one turn are considered in order, up to the per-turn limit; the rest are skipped,
  a terminal call beyond the limit included;
- the first terminal call within the limit ends the episode; every other call in that turn is
  skipped, since its result could never reach the model;
- a malformed call (parse error, unknown tool, arguments not a JSON object) is answered with a
  tool error and counts toward the limits;
- a length stop ends the episode without an answer, even if a call was parsed from the cut text;
  an empty turn ends it without an answer;
- a turn at the turn limit that still calls tools ends the episode with `max_turns`;
- the observations of one turn are admitted together, only if the next turn keeps
  `generation_reserve` tokens of room; otherwise the episode ends with `context_budget`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rft_cookbook.agent.episode import EpisodeRecord, Termination, ToolCallRecord, TurnRecord
from rft_cookbook.agent.spec import AgentTask
from rft_cookbook.tools.base import ToolResult


@dataclass
class ParsedCall:
    """A tool call as the driver's parser produced it."""

    name: str
    arguments: str
    call_id: str | None = None
    parse_error: str | None = None


@dataclass
class PlannedCall:
    record: ToolCallRecord
    action: str
    """`execute`: run the tool with `arguments`; `error`: answer with `error_text` without running."""
    arguments: dict[str, Any] | None = None
    error_text: str | None = None


@dataclass
class TurnDecision:
    calls: list[PlannedCall] = field(default_factory=list)
    termination: Termination | None = None

    @property
    def done(self) -> bool:
        return self.termination is not None


class PolicyStateError(RuntimeError):
    """A driver called the policy out of order: a bug in the driver."""


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def split_reasoning(text: str) -> tuple[str, str | None]:
    """`(visible text, reasoning)` of a turn whose reasoning sits in a `<think>` block.

    Templates that open the block in the generation prompt (Qwen3.5, GLM) leave only the closing
    tag in the output. Every driver applies this, so an answer is read from the same text whether
    or not the server separated the reasoning.
    """
    if THINK_CLOSE not in text:
        return text, None
    reasoning, visible = text.split(THINK_CLOSE, 1)
    reasoning = reasoning.split(THINK_OPEN, 1)[-1] if THINK_OPEN in reasoning else reasoning
    return visible.lstrip("\n"), reasoning.strip()


class EpisodePolicy:
    def __init__(self, task: AgentTask, record: EpisodeRecord):
        self.task = task
        self.budgets = task.budgets
        self.record = record
        self.prompt_tokens = 0
        self.context_tokens = 0
        self.calls_used = 0
        self._pending: list[ToolCallRecord] = []
        self._rendered: dict[str, str] = {}
        self._observations = 0
        self._started = False

    # ---- accounting --------------------------------------------------------------------

    @property
    def response_tokens(self) -> int:
        return self.context_tokens - self.prompt_tokens

    def room(self, context_tokens: int | None = None) -> int:
        """Tokens the conversation may still grow by, under both context budgets."""
        context = self.context_tokens if context_tokens is None else context_tokens
        room = self.budgets.max_response_tokens - (context - self.prompt_tokens)
        if self.budgets.max_context_tokens is not None:
            room = min(room, self.budgets.max_context_tokens - context)
        return room

    @property
    def terminated(self) -> bool:
        return self.record.termination is not None

    def _end(self, reason: str, *, answer: str | None = None, detail: str | None = None) -> Termination:
        if self.terminated:
            raise PolicyStateError("episode already terminated")
        self.record.termination = Termination(
            reason=reason, answer=answer, detail=detail,
            terminal_turn=len(self.record.turns) - 1 if self.record.turns else None)
        return self.record.termination

    # ---- driver protocol ---------------------------------------------------------------

    def start(self, prompt_tokens: int) -> Termination | None:
        if self._started:
            raise PolicyStateError("start called twice")
        self._started = True
        self.prompt_tokens = self.context_tokens = int(prompt_tokens)
        if self.budgets.max_context_tokens is not None and \
                self.room() < self.budgets.generation_reserve:
            return self._end("context_budget", detail="the initial prompt leaves no room for a turn")
        return None

    def next_request(self) -> int | Termination:
        """Max tokens for the next generation, or the termination if no turn fits."""
        if not self._started or self.terminated:
            raise PolicyStateError("next_request outside a running episode")
        room = self.room()
        if room < 1:
            return self._end("context_budget", detail="no room for another turn")
        turn = len(self.record.turns)
        for call in self._pending:
            call.delivery, call.delivered_to_turn = "delivered", turn
            call.delivered_text = self._rendered.pop(call.observation_id)
        self._pending = []
        return min(self.budgets.max_turn_tokens, room)

    def on_turn(self, *, text: str, calls: list[ParsedCall], finish_reason: str, generated_tokens: int,
                context_tokens: int, reasoning: str | None = None, policy_version: int | None = None) -> TurnDecision:
        """Record a model turn (already appended to the driver's context) and decide what follows."""
        if not self._started or self.terminated or self._pending:
            raise PolicyStateError("on_turn outside a running episode or with undelivered observations")
        turn = TurnRecord(index=len(self.record.turns), text=text, reasoning=reasoning,
                          finish_reason=finish_reason, prompt_tokens=self.context_tokens,
                          generated_tokens=int(generated_tokens), policy_version=policy_version)
        turn.tool_calls = [ToolCallRecord(name=c.name, arguments=c.arguments, call_id=c.call_id,
                                          parse_error=c.parse_error) for c in calls]
        self.record.turns.append(turn)
        self.context_tokens = int(context_tokens)

        def skip_all(reason: str, keep: int | None = None):
            for i, call in enumerate(turn.tool_calls):
                if i != keep:
                    call.status, call.detail = "skipped", reason

        if finish_reason == "length":
            skip_all("length_stop")
            return TurnDecision(termination=self._end("length", detail="generation hit its token limit"))
        if not text.strip() and not calls:
            return TurnDecision(termination=self._end("no_answer", detail="empty turn"))

        limit = self.budgets.max_tool_calls_per_turn
        considered = turn.tool_calls[:limit]
        for call in turn.tool_calls[limit:]:
            call.status, call.detail = "skipped", "per_turn_limit"

        rule = self.task.terminal
        if rule.kind == "tool":
            terminal = next((i for i, c in enumerate(considered) if c.name == rule.tool), None)
            if terminal is not None:
                skip_all("terminal_in_turn", keep=terminal)
                considered[terminal].status = "ok"
                return TurnDecision(termination=self._end("answer", answer=considered[terminal].arguments))
        if not considered:
            if rule.kind == "answer":
                answer = rule.extract(text)
                if answer is not None:
                    return TurnDecision(termination=self._end("answer", answer=answer))
                return TurnDecision(termination=self._end("no_answer", detail="no answer in the final turn"))
            return TurnDecision(termination=self._end("no_answer", detail="no tool call"))
        if turn.index + 1 >= self.budgets.max_turns:
            skip_all("max_turns")
            return TurnDecision(termination=self._end("max_turns"))

        planned: list[PlannedCall] = []
        for call in considered:
            if self.budgets.max_tool_calls is not None and self.calls_used >= self.budgets.max_tool_calls:
                call.status, call.detail = "skipped", "tool_budget"
                continue
            self.calls_used += 1
            arguments, error = self._validate(call)
            if error is not None:
                planned.append(PlannedCall(call, "error", error_text=error))
            else:
                planned.append(PlannedCall(call, "execute", arguments=arguments))
        if not planned:
            return TurnDecision(termination=self._end("tool_budget"))
        return TurnDecision(calls=planned)

    def _validate(self, call: ToolCallRecord) -> tuple[dict[str, Any] | None, str | None]:
        if call.parse_error:
            return None, f"Error: the call to '{call.name}' could not be parsed: {call.parse_error}"
        if call.name not in self.task.tools:
            return None, f"Error: unknown tool '{call.name}'. Available tools: {self.task.tools.names()}"
        try:
            arguments = json.loads(call.arguments) if call.arguments.strip() else {}
        except json.JSONDecodeError as error:
            return None, f"Error: arguments for '{call.name}' are not valid JSON: {error}"
        if not isinstance(arguments, dict):
            return None, f"Error: arguments for '{call.name}' must be a JSON object"
        return arguments, None

    def observation(self, planned: PlannedCall, result: ToolResult | None) -> str:
        """Record a call's outcome; returns the observation text to place in the context."""
        call = planned.record
        self._observations += 1
        call.observation_id = f"obs_{self._observations}"
        if planned.action == "error":
            call.status, text = "error", planned.error_text or ""
        else:
            if result is None:
                raise PolicyStateError("an executed call needs its result")
            call.executed = True
            call.status, call.truncated = result.status, result.truncated
            call.latency_s, call.sandbox, text = result.latency_s, result.sandbox, result.text
            if result.detail:
                call.detail = result.detail
        rendered = self.task.render_observation(call.observation_id, call.name, text)
        self._rendered[call.observation_id] = rendered
        return rendered

    def admit(self, planned: list[PlannedCall], context_tokens: int) -> bool:
        """Admit this turn's observations if the context would then be `context_tokens` long."""
        if self.terminated:
            raise PolicyStateError("admit after termination")
        if self.room(context_tokens) >= self.budgets.generation_reserve:
            self.context_tokens = int(context_tokens)
            self._pending = [p.record for p in planned]
            return True
        for p in planned:
            p.record.delivery = "not_admitted"
            self._rendered.pop(p.record.observation_id, None)
        self._end("context_budget", detail=f"observations would leave {self.room(context_tokens)} tokens")
        return False

    def fail(self, reason: str, detail: str) -> Termination:
        """End the episode on a driver-detected failure (`policy_error` or `infra_error`)."""
        if reason not in ("policy_error", "infra_error"):
            raise ValueError(reason)
        if self.terminated:
            earlier = self.record.termination
            earlier.detail = f"{detail} (after {earlier.reason})"
            earlier.reason, earlier.answer = reason, None
            return earlier
        return self._end(reason, detail=detail)

    def finalize(self) -> EpisodeRecord:
        for call in self._pending:
            call.delivery = "not_admitted"
            call.detail = "the episode ended before the observation was delivered"
        self._pending, self._rendered = [], {}
        if not self.terminated:
            self._end("policy_error", detail="the driver stopped without a termination")
        reason = self.record.termination.reason
        self.record.status = {"infra_error": "infra_failure", "policy_error": "policy_failure"}.get(reason, "ok")
        self.record.metadata["tool_calls_used"] = self.calls_used
        self.record.validate()
        return self.record

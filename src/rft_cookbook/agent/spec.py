"""The agentic task specification every task fills in, and every framework adapter runs."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from rft_cookbook.scoring.base import Scorer
from rft_cookbook.tools.base import ToolSet


@dataclass(frozen=True)
class Budgets:
    """Limits of one episode. Token budgets are counted by the driver, in the policy's tokens.

    `max_context_tokens` bounds the whole conversation (initial prompt plus everything appended);
    `max_response_tokens` bounds everything appended after the initial prompt (model turns, tool
    observations and template tokens; verl's `response_length`). The next turn needs at least
    `generation_reserve` free tokens under both; an observation that would leave less is not
    admitted. `max_turn_tokens` caps one generation request.
    """

    max_turns: int
    max_response_tokens: int
    max_turn_tokens: int
    generation_reserve: int = 64
    max_context_tokens: int | None = None
    max_tool_calls_per_turn: int = 1
    max_tool_calls: int | None = None
    tool_output_bytes: int = 16384
    tool_timeout_s: float = 60.0

    def __post_init__(self):
        for name in ("max_turns", "max_response_tokens", "max_turn_tokens", "max_tool_calls_per_turn",
                     "tool_output_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"budget {name} must be positive")
        if self.generation_reserve < 1 or self.generation_reserve > self.max_turn_tokens:
            raise ValueError("generation_reserve must be between 1 and max_turn_tokens")

    def replace(self, **changes) -> Budgets:
        unknown = set(changes) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ValueError(f"unknown budget fields: {sorted(unknown)}")
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TerminalRule:
    """How an episode ends with an answer.

    `tool`: the first call to the named tool ends the episode; its raw arguments are the answer.
    `answer`: the first model turn without tool calls ends the episode; `extract(text)` is the
    answer, or None when the text holds none.
    """

    kind: str
    tool: str | None = None
    extract: Callable[[str], str | None] | None = None

    def __post_init__(self):
        if self.kind == "tool" and not self.tool:
            raise ValueError("a terminal tool rule names the tool")
        if self.kind == "answer" and self.extract is None:
            raise ValueError("an answer rule needs an extractor")
        if self.kind not in ("tool", "answer"):
            raise ValueError(f"unknown terminal rule {self.kind!r}")

    @classmethod
    def terminal_tool(cls, name: str) -> TerminalRule:
        return cls("tool", tool=name)

    @classmethod
    def answer_pattern(cls, extract: Callable[[str], str | None]) -> TerminalRule:
        return cls("answer", extract=extract)


@dataclass(frozen=True)
class AgentTask:
    name: str
    version: str
    system_prompt: str
    tools: ToolSet
    terminal: TerminalRule
    budgets: Budgets
    scorer: Scorer
    observation_format: Callable[[str, str, str], str] | None = None
    """`(observation_id, tool_name, text) -> context text`; default: the tool's text unchanged."""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.terminal.kind == "tool" and self.terminal.tool in self.tools.names():
            raise ValueError("the terminal tool is a schema only: it must not also be an executable tool")

    def render_observation(self, observation_id: str, tool: str, text: str) -> str:
        return self.observation_format(observation_id, tool, text) if self.observation_format else text

    def tool_schemas(self) -> list[dict[str, Any]]:
        """OpenAI function schemas shown to the model, including a terminal tool's schema."""
        schemas = self.tools.schemas() + list(self.metadata.get("terminal_schemas", []))
        view = self.metadata.get("schema_view", "full")
        return schemas if view == "full" else [SCHEMA_VIEWS[view](s) for s in schemas]

    def with_schema_view(self, view: str) -> AgentTask:
        """The task with its tool schemas shown in another view (SCHEMA_VIEWS); for diagnostics only."""
        if view != "full" and view not in SCHEMA_VIEWS:
            raise ValueError(f"unknown schema view {view!r}; known: full, {', '.join(SCHEMA_VIEWS)}")
        return dataclasses.replace(self, metadata={**self.metadata, "schema_view": view})

    def settings(self, **extra: Any) -> dict[str, Any]:
        """The hashed agent settings every episode record carries."""
        return {
            "prompt_sha256": sha256_text(self.system_prompt),
            "tools_sha256": sha256_json(self.tool_schemas()),
            "terminal": {"kind": self.terminal.kind, "tool": self.terminal.tool},
            "budgets": self.budgets.to_dict(),
            "scorer": getattr(self.scorer, "policy", None),   # runs scored by different rules do not compare
            **self.metadata.get("settings", {}),             # what else the task's episodes depend on (an index)
            **extra,
        }

    def with_budgets(self, **changes) -> AgentTask:
        return dataclasses.replace(self, budgets=self.budgets.replace(**changes))


def verl_property_subset(schema: dict[str, Any]) -> dict[str, Any]:
    """A schema as verl 0.10's ToolAgentLoop shows it to the model. Its OpenAIFunctionToolSchema
    keeps only `type`, `description` and `enum` of each parameter, so nested fields (an array's
    `items`) never reach the prompt; the keys come out in the model's field order."""
    function = schema["function"]
    out = {key: function[key] for key in ("name", "description") if function.get(key) is not None}
    if "parameters" in function:
        parameters = function["parameters"]
        view = {"type": parameters["type"],
                "properties": {name: {key: prop[key] for key in ("type", "description", "enum") if prop.get(key) is not None}
                               for name, prop in parameters["properties"].items()}}
        if parameters.get("required") is not None:
            view["required"] = list(parameters["required"])
        out["parameters"] = view
    if function.get("strict") is not None:
        out["strict"] = function["strict"]
    return {"type": schema["type"], "function": out}


SCHEMA_VIEWS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {"verl-0.10-properties": verl_property_subset}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

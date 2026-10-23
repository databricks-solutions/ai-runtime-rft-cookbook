"""Shared helpers for the core tests."""
from __future__ import annotations

from rft_cookbook.agent.harness import ChatTurn
from rft_cookbook.agent.policy import ParsedCall
from rft_cookbook.agent.spec import AgentTask, Budgets, TerminalRule
from rft_cookbook.scoring.base import RewardResult
from rft_cookbook.tools.base import FunctionTool, ToolInfraError, ToolSet, function_schema


def echo(text: str) -> str:
    return f"echo: {text}"


def broken_backend(text: str) -> str:
    raise ToolInfraError("index service unreachable")


ECHO = FunctionTool(echo, function_schema("echo", "Echo text.", {"text": {"type": "string"}}, ["text"]))
BACKEND = FunctionTool(broken_backend, function_schema("lookup", "Look up.", {"text": {"type": "string"}}, ["text"]))
SUBMIT_SCHEMA = function_schema("submit", "Submit the answer.", {"answer": {"type": "string"}}, ["answer"])


class ConstScorer:
    policy = "test/const-v1"

    def __init__(self, value: float = 1.0, valid: bool = True):
        self.value, self.valid = value, valid

    async def score(self, episode, ground_truth, metadata):
        return RewardResult(valid=self.valid, score=self.value if episode.answer is not None else 0.0)


def make_task(*, terminal: str = "tool", tools=(ECHO,), **budgets) -> AgentTask:
    defaults = dict(max_turns=4, max_response_tokens=1000, max_turn_tokens=200, generation_reserve=50,
                    max_tool_calls_per_turn=1, tool_output_bytes=64)
    defaults.update(budgets)
    rule = (TerminalRule.terminal_tool("submit") if terminal == "tool"
            else TerminalRule.answer_pattern(lambda text: text.split("ANSWER:")[-1].strip() if "ANSWER:" in text else None))
    return AgentTask(name="toy", version="1", system_prompt="You are a test agent.", tools=ToolSet(list(tools)),
                     terminal=rule, budgets=Budgets(**defaults), scorer=ConstScorer(),
                     metadata={"terminal_schemas": [SUBMIT_SCHEMA]} if terminal == "tool" else {})


def call(name: str, arguments: str = '{"text": "hi"}', call_id: str | None = None) -> ParsedCall:
    return ParsedCall(name=name, arguments=arguments, call_id=call_id)


class ScriptedClient:
    """A fake Chat Completions server that replays turns and records each request."""

    def __init__(self, turns: list[ChatTurn | Exception]):
        self.turns = list(turns)
        self.requests: list[dict] = []

    async def complete(self, messages, tools, max_tokens):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools, "max_tokens": max_tokens})
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


ROW = {"id": "q1", "split": "eval", "messages": [{"role": "user", "content": "What is 2+2?"}],
       "ground_truth": "4", "metadata": {}}


"""The session message matcher: a replayed assistant turn is the stored one as the harness read it.

    --session-message-matcher rft_miles.session.matches

The cookbook's harness (rft_cookbook.agent.harness) replays each assistant turn as its policy
read it: the reasoning split from the text (`split_reasoning`, unless the server split it) and
only the tool calls it answered (a call over `max_tool_calls_per_turn` or the episode's call
budget is left out). miles' strict matcher compares that replay with SGLang's message as the
session stored it, so a turn whose reading differs opens a new branch of the session tree, and
every branch becomes a training sample of its own, its whole prefix included. This matcher
accepts exactly the harness's reading of the stored message: the session then continues one
lineage, miles keeps the stored message and its token ids for the prefix, and an episode trains
as one sequence with every generated token in context, as verl's loop builds it. Every other
message is compared strictly.
"""
from __future__ import annotations

from typing import Any

from rft_cookbook.agent.policy import split_reasoning


def matches(stored: dict[str, Any], replayed: dict[str, Any]) -> bool:
    from miles.utils.chat_template_utils.message_matcher_hub import strict_message_matches

    if strict_message_matches(stored, replayed):
        return True
    if stored.get("role") != "assistant" or replayed.get("role") != "assistant":
        return False
    text, reasoning = _read(stored)
    if _value(text) != _value(replayed.get("content")):
        return False
    if _value(reasoning) != _value(replayed.get("reasoning_content")):
        return False
    return _answered(stored.get("tool_calls"), replayed.get("tool_calls"))


def _read(message: dict[str, Any]) -> tuple[str, str | None]:
    """`(text, reasoning)` of a server's message, as the harness reads it (OpenAIChatClient, _drive)."""
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return (text, reasoning) if reasoning is not None else split_reasoning(text)


def _answered(stored: list[dict[str, Any]] | None, replayed: list[dict[str, Any]] | None) -> bool:
    """The replayed calls are stored calls, in their order."""
    remaining = iter(_call(c) for c in stored or [])
    return all(any(call == candidate for candidate in remaining) for call in map(_call, replayed or []))


def _call(call: dict[str, Any]) -> tuple:
    function = call.get("function") or {}
    return call.get("id"), call.get("type"), function.get("name"), function.get("arguments")


def _value(value: Any) -> Any:
    """None, "" and [] render alike (miles' strict matcher treats them as one)."""
    return None if value in (None, "", []) else value

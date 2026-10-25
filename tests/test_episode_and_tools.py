"""Episode record schema and reward eligibility; tool execution limits."""
from __future__ import annotations

import asyncio
import json

import pytest
from support import ScriptedClient, call, make_task

from rft_cookbook.agent.episode import SCHEMA, EpisodeRecord, Score, Termination, eligibility, token_mask_sha256
from rft_cookbook.agent.harness import ByteCounter, ChatTurn, run_episode
from rft_cookbook.tools.base import FunctionTool, ToolContext, ToolSet, clip_utf8, function_schema


def recorded(row):
    client = ScriptedClient([ChatTurn(text="", tool_calls=[call("echo")]),
                             ChatTurn(text="", tool_calls=[call("submit", '{"answer": "4"}')])])
    return asyncio.run(run_episode(make_task(), client, row, counter=ByteCounter()))


def test_records_round_trip_and_reject_unknown_fields_and_versions(row):
    record = recorded(row)
    again = EpisodeRecord.from_json(record.to_json())
    assert again.to_dict() == record.to_dict() and again.schema == SCHEMA
    data = json.loads(record.to_json())
    data["turns"][0]["surprise"] = 1
    with pytest.raises(ValueError, match="unknown TurnRecord fields"):
        EpisodeRecord.from_dict(data)
    with pytest.raises(ValueError, match="unsupported episode schema"):
        EpisodeRecord.from_dict({**json.loads(record.to_json()), "schema": "rft-cookbook/episode/v0"})


def test_a_delivered_observation_must_name_its_text_and_turn(row):
    record = recorded(row)
    record.turns[0].tool_calls[0].delivered_text = None
    with pytest.raises(ValueError, match="delivered_text"):
        record.validate()


@pytest.mark.parametrize("status,score,answer,complete,expected", [
    ("ok", Score(1.0, True, "p"), "4", True, ("eligible", None)),
    ("ok", Score(0.0, True, "p"), None, True, ("zero", None)),
    ("ok", Score(1.0, True, "p"), "4", False, ("excluded", "terminal_truncated")),
    ("ok", Score(0.0, False, "p"), "4", True, ("excluded", "invalid_score")),
    ("ok", None, "4", True, ("excluded", "invalid_score")),
    ("infra_failure", Score(0.0, True, "p"), None, True, ("excluded", "infra_failure")),
    ("policy_failure", Score(0.0, True, "p"), None, True, ("excluded", "policy_failure")),   # not the model's zero
])
def test_reward_eligibility_table(status, score, answer, complete, expected):
    record = EpisodeRecord(episode_id="e", task="t", task_version="1", question_id="q", status=status, score=score,
                           termination=Termination("answer" if answer else "no_answer", answer=answer))
    assert eligibility(record, complete) == expected


def test_the_sample_hash_covers_the_mask():
    assert token_mask_sha256([1, 2, 3], [0, 1, 1]) != token_mask_sha256([1, 2, 3], [0, 1, 0])


def test_clip_never_splits_a_character_and_respects_the_cap():
    text, truncated = clip_utf8("ä" * 100, 41)
    assert truncated and len(text.encode()) <= 41 and text.endswith("...(truncated)")
    assert clip_utf8("short", 41) == ("short", False)
    assert clip_utf8("x" * 100, 5) == ("...(t", True)


def test_tool_errors_and_timeouts_are_reported_to_the_model():
    async def slow(text: str) -> str:
        await asyncio.sleep(5)
        return text

    def boom(text: str) -> str:
        raise ValueError("bad input")

    schema = lambda name: function_schema(name, "", {"text": {"type": "string"}}, ["text"])  # noqa: E731
    tools = ToolSet([FunctionTool(slow, schema("slow")), FunctionTool(boom, schema("boom"))])
    context = ToolContext("e", "q")
    slow_result = asyncio.run(tools.run("slow", {"text": "x"}, context, timeout_s=0.05, max_bytes=100))
    boom_result = asyncio.run(tools.run("boom", {"text": "x"}, context, timeout_s=1, max_bytes=100))
    assert slow_result.status == "timeout" and boom_result.status == "error"
    assert "ValueError: bad input" in boom_result.text and boom_result.latency_s is not None


def test_a_terminal_tool_may_not_also_be_executable():
    from support import SUBMIT_SCHEMA, ConstScorer

    from rft_cookbook.agent.spec import AgentTask, Budgets, TerminalRule

    with pytest.raises(ValueError, match="terminal tool"):
        AgentTask(name="t", version="1", system_prompt="s",
                  tools=ToolSet([FunctionTool(lambda answer: answer, SUBMIT_SCHEMA)]),
                  terminal=TerminalRule.terminal_tool("submit"),
                  budgets=Budgets(max_turns=1, max_response_tokens=10, max_turn_tokens=10, generation_reserve=1),
                  scorer=ConstScorer())

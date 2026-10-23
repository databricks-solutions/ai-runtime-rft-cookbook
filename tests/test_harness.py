"""The Chat Completions driver against a scripted fake server."""
from __future__ import annotations

import asyncio

from support import BACKEND, ECHO, ScriptedClient, call, make_task

from rft_cookbook.agent.harness import ByteCounter, ChatTurn, run_episode
from rft_cookbook.scoring.base import score_episode
from rft_cookbook.tasks.math import task as math_task


def run(task, client, row, **kwargs):
    return asyncio.run(run_episode(task, client, row, counter=ByteCounter(), **kwargs))


def test_a_math_episode_with_a_calculator_call(row):
    task = math_task()
    client = ScriptedClient([
        ChatTurn(text="Let me compute.", tool_calls=[call("calculator", '{"expression": "2+2"}', "c1")],
                 prompt_tokens=300, completion_tokens=20),
        ChatTurn(text="So the answer is \\boxed{4}.", prompt_tokens=340, completion_tokens=12),
    ])
    record = run(task, client, row, sample_index=0, model={"id": "toy"})
    assert record.termination.reason == "answer" and record.answer == "4"
    call_record = record.turns[0].tool_calls[0]
    assert call_record.delivered_text == "4" and call_record.delivered_to_turn == 1
    second = client.requests[1]["messages"]
    assert second[0]["role"] == "system" and second[0]["content"] == task.system_prompt
    assert second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "content": "4"}
    score = asyncio.run(score_episode(task.scorer, record, row["ground_truth"]))
    assert score.valid and score.value == 1.0 and score.policy == "math/rule-v1"
    assert record.to_json()   # validates


def test_a_wrong_or_missing_answer_is_a_valid_zero(row):
    task = math_task()
    for text in ("\\boxed{5}", "I am not sure."):
        record = run(task, ScriptedClient([ChatTurn(text=text, prompt_tokens=300, completion_tokens=5)]), row)
        score = asyncio.run(score_episode(task.scorer, record, row["ground_truth"]))
        assert score.valid and score.value == 0.0


def test_a_serving_failure_is_an_infrastructure_failure(row):
    record = run(make_task(), ScriptedClient([ConnectionError("refused")]), row)
    assert record.status == "infra_failure" and record.termination.reason == "infra_error"


def test_a_tool_backend_failure_is_an_infrastructure_failure(row):
    task = make_task(tools=(ECHO, BACKEND))
    record = run(task, ScriptedClient([ChatTurn(text="", tool_calls=[call("lookup")])]), row)
    assert record.status == "infra_failure"


def test_skipped_calls_never_reach_the_next_request(row):
    task = make_task(max_tool_calls_per_turn=1)
    client = ScriptedClient([
        ChatTurn(text="", tool_calls=[call("echo", call_id="a"), call("echo", call_id="b")]),
        ChatTurn(text="", tool_calls=[call("submit", '{"answer": "4"}')]),
    ])
    record = run(task, client, row)
    assistant = client.requests[1]["messages"][-2]
    assert [c["id"] for c in assistant["tool_calls"]] == ["a"]
    assert record.answer == '{"answer": "4"}'


def test_the_tool_output_cap_is_applied_before_delivery(row):
    task = make_task(tool_output_bytes=20)
    client = ScriptedClient([ChatTurn(text="", tool_calls=[call("echo", '{"text": "' + "x" * 100 + '"}')]),
                             ChatTurn(text="", tool_calls=[call("submit", "{}")])])
    record = run(task, client, row)
    delivered = record.turns[0].tool_calls[0]
    assert delivered.truncated and len(delivered.delivered_text.encode()) <= 20

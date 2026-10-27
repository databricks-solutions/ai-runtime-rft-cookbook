"""The shared decision rules (docs/architecture.md, "Episode format"), driven directly."""
from __future__ import annotations

import pytest
from support import call, make_task

from rft_cookbook.agent.episode import EpisodeRecord
from rft_cookbook.agent.policy import EpisodePolicy, ParsedCall, PolicyStateError
from rft_cookbook.tools.base import ToolResult


def start(task, prompt_tokens=100):
    record = EpisodeRecord(episode_id="e1", task=task.name, task_version=task.version, question_id="q1")
    policy = EpisodePolicy(task, record)
    assert policy.start(prompt_tokens) is None
    return policy, record


def turn(policy, text="thinking", calls=(), finish="stop", generated=20):
    return policy.on_turn(text=text, calls=list(calls), finish_reason=finish, generated_tokens=generated,
                          context_tokens=policy.context_tokens + generated)


def deliver(policy, decision, added=10):
    for planned in decision.calls:
        result = ToolResult(text="ok") if planned.action == "execute" else None
        policy.observation(planned, result)
    return policy.admit(decision.calls, policy.context_tokens + added)


def test_calls_beyond_the_per_turn_limit_are_skipped():
    policy, record = start(make_task(max_tool_calls_per_turn=2))
    policy.next_request()
    decision = turn(policy, calls=[call("echo"), call("echo"), call("echo")])
    assert [p.action for p in decision.calls] == ["execute", "execute"]
    statuses = [(c.status, c.detail) for c in record.turns[0].tool_calls]
    assert statuses[2] == ("skipped", "per_turn_limit")


def test_a_terminal_call_beyond_the_limit_does_not_end_the_episode():
    policy, record = start(make_task())
    policy.next_request()
    decision = turn(policy, calls=[call("echo"), call("submit", '{"answer": "4"}')])
    assert not decision.done and record.turns[0].tool_calls[1].status == "skipped"


def test_the_first_terminal_call_ends_the_episode_and_every_other_call_is_skipped():
    policy, record = start(make_task(max_tool_calls_per_turn=3))
    policy.next_request()
    decision = turn(policy, calls=[call("echo"), call("submit", '{"answer": "4"}'), call("submit", '{"answer": "5"}')])
    assert decision.termination.reason == "answer" and decision.termination.answer == '{"answer": "4"}'
    calls = record.turns[0].tool_calls
    assert [c.status for c in calls] == ["skipped", "ok", "skipped"]
    assert calls[1].executed is False and calls[1].delivery == "not_run"
    record = policy.finalize()
    assert record.status == "ok" and record.termination.terminal_turn == 0


def test_a_malformed_call_is_answered_with_an_error_and_counts_toward_the_limits():
    policy, record = start(make_task(max_tool_calls_per_turn=1, max_tool_calls=1))
    policy.next_request()
    decision = turn(policy, calls=[call("echo", "{not json")])
    assert decision.calls[0].action == "error" and "not valid JSON" in decision.calls[0].error_text
    assert deliver(policy, decision)
    policy.next_request()
    assert record.turns[0].tool_calls[0].delivered_text.startswith("Error:")
    assert record.turns[0].tool_calls[0].executed is False
    decision = turn(policy, calls=[call("echo")])
    assert decision.termination.reason == "tool_budget"


@pytest.mark.parametrize("parse_error,arguments", [("unclosed <parameter>", '{"answer": "4"}'), (None, "{not json")])
def test_a_malformed_terminal_call_is_answered_with_an_error_and_the_episode_goes_on(parse_error, arguments):
    policy, record = start(make_task(max_tool_calls_per_turn=2))
    policy.next_request()
    malformed = ParsedCall("submit", arguments, parse_error=parse_error)
    decision = turn(policy, calls=[malformed])
    assert not decision.done and decision.calls[0].action == "error"
    assert "unknown tool" not in decision.calls[0].error_text
    assert deliver(policy, decision)
    policy.next_request()
    decision = turn(policy, calls=[malformed, call("submit", '{"answer": "5"}')])   # a later well-formed one answers
    assert decision.termination.reason == "answer" and decision.termination.answer == '{"answer": "5"}'
    assert [c.status for c in record.turns[1].tool_calls] == ["skipped", "ok"]


@pytest.mark.parametrize("arguments,expected", [
    ("[1, 2]", "must be a JSON object"), ('{"text": "x"}', "unknown tool")])
def test_wrong_shapes_and_unknown_tools_are_errors(arguments, expected):
    policy, _ = start(make_task())
    policy.next_request()
    name = "nope" if expected == "unknown tool" else "echo"
    decision = turn(policy, calls=[call(name, arguments)])
    assert expected in decision.calls[0].error_text


def test_a_length_stop_ends_without_an_answer_even_with_a_parsed_terminal_call():
    policy, record = start(make_task())
    policy.next_request()
    decision = turn(policy, calls=[call("submit", '{"answer": "4"}')], finish="length")
    assert decision.termination.reason == "length" and decision.termination.answer is None
    assert record.turns[0].tool_calls[0].status == "skipped"


def test_an_empty_turn_ends_without_an_answer():
    policy, _ = start(make_task())
    policy.next_request()
    assert turn(policy, text="  ").termination.reason == "no_answer"


def test_answer_pattern_reads_only_the_final_turn():
    policy, _ = start(make_task(terminal="answer"))
    policy.next_request()
    assert turn(policy, text="ANSWER: 4").termination.answer == "4"
    policy, _ = start(make_task(terminal="answer"))
    policy.next_request()
    assert turn(policy, text="no box here").termination.reason == "no_answer"


def test_a_tool_turn_at_the_turn_limit_ends_with_max_turns():
    policy, record = start(make_task(max_turns=2))
    policy.next_request()
    assert deliver(policy, turn(policy, calls=[call("echo")]))
    policy.next_request()
    decision = turn(policy, calls=[call("echo")])
    assert decision.termination.reason == "max_turns"
    assert record.turns[1].tool_calls[0].status == "skipped"


def test_a_terminal_call_in_the_last_turn_still_answers():
    policy, _ = start(make_task(max_turns=1))
    policy.next_request()
    assert turn(policy, calls=[call("submit", '{"answer": "4"}')]).termination.reason == "answer"


def test_observations_are_delivered_to_the_next_turn_only():
    policy, record = start(make_task())
    policy.next_request()
    decision = turn(policy, calls=[call("echo")])
    assert deliver(policy, decision)
    obs = record.turns[0].tool_calls[0]
    assert obs.delivery == "not_run" and obs.delivered_text is None   # admitted, not yet seen
    policy.next_request()
    assert obs.delivery == "delivered" and obs.delivered_to_turn == 1 and obs.delivered_text == "ok"
    assert obs.observation_id == "obs_1" and obs.executed


def test_an_observation_that_leaves_no_reserve_is_not_admitted():
    task = make_task(max_response_tokens=200, generation_reserve=50)
    policy, record = start(task)
    policy.next_request()
    decision = turn(policy, calls=[call("echo")], generated=100)
    assert not deliver(policy, decision, added=60)   # 160 used, 40 left < reserve 50
    obs = record.turns[0].tool_calls[0]
    assert obs.delivery == "not_admitted" and obs.executed and obs.delivered_text is None
    assert policy.finalize().termination.reason == "context_budget"


def test_the_next_request_never_asks_for_more_than_the_room_left():
    policy, _ = start(make_task(max_response_tokens=300, max_turn_tokens=200))
    assert policy.next_request() == 200
    assert deliver(policy, turn(policy, calls=[call("echo")], generated=150), added=20)
    assert policy.next_request() == 130


def test_a_prompt_that_leaves_no_room_ends_before_generation():
    task = make_task(max_context_tokens=120, generation_reserve=50)
    record = EpisodeRecord(episode_id="e", task="toy", task_version="1", question_id="q")
    assert EpisodePolicy(task, record).start(100).reason == "context_budget"


def test_failures_set_the_status():
    policy, _ = start(make_task())
    policy.next_request()
    policy.fail("infra_error", "server gone")
    assert policy.finalize().status == "infra_failure"
    policy, _ = start(make_task())
    assert policy.finalize().status == "policy_failure"   # the driver stopped without a decision


def test_out_of_order_calls_are_driver_bugs():
    policy, _ = start(make_task())
    policy.next_request()
    decision = turn(policy, calls=[call("echo")])
    policy.observation(decision.calls[0], ToolResult(text="ok"))
    policy.admit(decision.calls, policy.context_tokens + 5)
    with pytest.raises(PolicyStateError):
        turn(policy)   # the admitted observation was never delivered

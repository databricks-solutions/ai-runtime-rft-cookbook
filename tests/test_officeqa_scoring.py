"""A recorded OfficeQA episode through the OfficeQA scorer, with frozen judge replies.

The fixture `fixtures/officeqa_episode.json` was produced by the Chat Completions harness from
scripted turns (`officeqa_episode()` below rebuilds it) on synthetic table text; the test checks
that the recording still loads and scores as it did.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from support import ScriptedClient, call

from rft_cookbook.agent.episode import EpisodeRecord
from rft_cookbook.agent.harness import ByteCounter, ChatTurn, run_episode
from rft_cookbook.agent.spec import AgentTask, Budgets, TerminalRule
from rft_cookbook.scoring.base import score_episode
from rft_cookbook.scoring.judge import FrozenJudge, JudgeError, JudgePolicy, JudgeReply
from rft_cookbook.tasks import officeqa
from rft_cookbook.tools.base import FunctionTool, ToolSet

FIXTURE = Path(__file__).parent / "fixtures" / "officeqa_episode.json"
QUESTION = "What were FY1940 national defense expenditures in millions of dollars?"
METADATA = {"question": QUESTION}
TABLE = "Table 3. Expenditures\nNational defense | FY1940 | 100 million dollars\nVeterans | FY1940 | 557 million dollars"
POLICY = JudgePolicy(id="test-judge-v1", models=("judge-a", "judge-b"), max_prompt_tokens=8000, max_output_tokens=512,
                     response_schema=officeqa.VERDICT_SCHEMA)


def count(messages):
    return sum(len(m["content"]) for m in messages) // 4


def supported(model="judge-a"):
    return JudgeReply(content='{"path_status":"supported","path_score":0.9,"issues":[]}', finish_reason="stop",
                      model=model)


def toy_task() -> AgentTask:
    schemas = {s["function"]["name"]: s for s in officeqa.tool_schemas()}
    read = FunctionTool(lambda file_name, start_line=0, num_lines=50: TABLE, schemas["read_document"])
    return AgentTask(name="officeqa", version=officeqa.VERSION, system_prompt=officeqa.system_prompt(4),
                     tools=ToolSet([read]), terminal=TerminalRule.terminal_tool(officeqa.TERMINAL_TOOL),
                     budgets=Budgets(max_turns=4, max_response_tokens=4000, max_turn_tokens=500, generation_reserve=50),
                     scorer=None, observation_format=officeqa.render_observation,
                     metadata={"terminal_schemas": [schemas[officeqa.TERMINAL_TOOL]]})


def report(answer="100", observation="obs_1", claim="FY1940 national defense was 100 million dollars."):
    return json.dumps({"answer": answer, "path": [{"id": "a", "observation": observation, "claim": claim}]})


def officeqa_episode(submission: str | None = None, *, read_first: bool = True) -> EpisodeRecord:
    turns = []
    if read_first:
        turns.append(ChatTurn(text="", tool_calls=[call("read_document", '{"file_name": "treasury_1941_01.txt"}', "r1")]))
    turns.append(ChatTurn(text="", tool_calls=[call("submit_report", submission or report(), "s1")]))
    row = {"id": "UID0001", "split": "eval", "messages": [{"role": "user", "content": QUESTION}],
           "ground_truth": "100", "metadata": METADATA}
    return asyncio.run(run_episode(toy_task(), ScriptedClient(turns), row, counter=ByteCounter(), sample_index=0,
                                   episode_id="officeqa-fixture-0", model={"id": "scripted"}))


def score(record, judge, ground_truth="100"):
    scorer = officeqa.scorer(judge, POLICY, count)
    return asyncio.run(score_episode(scorer, record, ground_truth, METADATA))


def without_timing(record: EpisodeRecord) -> dict:
    for call in record.calls():
        call.latency_s = None
    return record.to_dict()


def test_the_recorded_episode_still_matches_its_driver():
    recorded = EpisodeRecord.from_json(FIXTURE.read_text())
    assert without_timing(recorded) == without_timing(officeqa_episode())


def test_the_recorded_episode_scores_one_when_the_judge_supports_it():
    recorded = EpisodeRecord.from_json(FIXTURE.read_text())
    judge = FrozenJudge([supported()])
    result = score(recorded, judge)
    assert result.valid and result.value == 1.0 and result.policy == "officeqa/path-report-v1+test-judge-v1"
    assert result.components["judge_called"] == 1.0 and result.judge["model"] == "judge-a"
    assert result.judge["fallback"] is False and len(result.judge["request"]["request_sha256"]) == 64
    (messages,) = judge.requests
    assert "National defense | FY1940 | 100 million dollars" in messages[1]["content"]
    assert "[observation obs_1]" not in messages[1]["content"]   # the judge sees the tool text, not our tag


def test_unsupported_wrong_fabricated_and_uncited_reports_score_zero_validly():
    unsupported = JudgeReply(content='{"path_status":"unsupported","path_score":0.0,"issues":["wrong row"]}',
                             finish_reason="stop", model="judge-a")
    assert score(officeqa_episode(), FrozenJudge([unsupported])).value == 0.0
    no_judge = FrozenJudge([])
    for submission, expected in ((report(answer="200"), "wrong"),
                                 (report(claim="FY1940 national defense was 999 million dollars."), "fabricated"),
                                 (report(observation="obs_9"), "invalid_path"),
                                 ('{"answer": "DATA NOT AVAILABLE", "path": []}', "abstention"),
                                 ("not json", "malformed")):
        result = score(officeqa_episode(submission), no_judge)
        assert result.valid and result.value == 0.0 and result.detail.startswith(expected), (submission, result.detail)
    assert no_judge.requests == []   # none of these reached the judge


def test_a_judge_outage_or_bad_verdict_is_an_invalid_score_not_a_zero():
    for reply in (JudgeError("transport", "HTTP 503", retryable=True),
                  JudgeReply(content='{"path_status":"supported","path_score":true,"issues":[]}', finish_reason="stop",
                             model="judge-a"),
                  JudgeReply(content='{"path_status":"supported","path_score":1,"issues":[]}', finish_reason="length",
                             model="judge-a")):
        result = score(officeqa_episode(), FrozenJudge([reply]))
        assert not result.valid and "JudgeError" in result.detail


def test_a_fallback_judge_is_recorded_as_such():
    result = score(officeqa_episode(), FrozenJudge([supported(model="judge-b")]))
    assert result.value == 1.0 and result.judge["fallback"] is True


def test_evidence_the_model_never_received_cannot_support_an_answer():
    result = score(officeqa_episode(read_first=False), FrozenJudge([]))
    assert result.valid and result.value == 0.0 and result.detail.startswith("invalid_path")


@pytest.mark.parametrize("missing", ["question", "gold"])
def test_missing_inputs_are_scorer_failures(missing):
    scorer = officeqa.scorer(FrozenJudge([]), POLICY, count)
    record = officeqa_episode()
    metadata = {} if missing == "question" else METADATA
    gold = "" if missing == "gold" else "100"
    assert not asyncio.run(score_episode(scorer, record, gold, metadata)).valid


def write_fixture() -> None:   # python -c "import test_officeqa_scoring as t; t.write_fixture()"
    FIXTURE.parent.mkdir(exist_ok=True)
    record = officeqa_episode()
    without_timing(record)
    FIXTURE.write_text(json.dumps(record.to_dict(), indent=1, sort_keys=True) + "\n")

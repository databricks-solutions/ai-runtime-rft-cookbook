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
from rft_cookbook.tasks.officeqa import answer_check, path_report
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


def toy_task(*, with_compute: bool = False) -> AgentTask:
    schemas = {s["function"]["name"]: s for s in officeqa.tool_schemas()}
    read = FunctionTool(lambda file_name, start_line=0, num_lines=50: TABLE, schemas["read_document"])
    compute = FunctionTool(lambda code: "Output:\n999", schemas["compute"])   # code can print any figure
    return AgentTask(name="officeqa", version=officeqa.VERSION, system_prompt=officeqa.system_prompt(4),
                     tools=ToolSet([read, compute] if with_compute else [read]),
                     terminal=TerminalRule.terminal_tool(officeqa.TERMINAL_TOOL),
                     budgets=Budgets(max_turns=4, max_response_tokens=4000, max_turn_tokens=500, generation_reserve=50),
                     scorer=None, observation_format=officeqa.render_observation,
                     metadata={"terminal_schemas": [schemas[officeqa.TERMINAL_TOOL]]})


def report(answer="100", observation="obs_1", claim="FY1940 national defense was 100 million dollars."):
    return json.dumps({"answer": answer, "path": [{"id": "a", "observation": observation, "claim": claim}]})


def officeqa_episode(submission: str | None = None, *, read_first: bool = True,
                     compute: bool = False) -> EpisodeRecord:
    turns = []
    if read_first:
        turns.append(ChatTurn(text="", tool_calls=[call("read_document", '{"file_name": "treasury_1941_01.txt"}', "r1")]))
    if compute:
        turns.append(ChatTurn(text="", tool_calls=[call("compute", '{"code": "print(999)"}', "c1")]))
    turns.append(ChatTurn(text="", tool_calls=[call("submit_report", submission or report(), "s1")]))
    row = {"id": "UID0001", "split": "eval", "messages": [{"role": "user", "content": QUESTION}],
           "ground_truth": "100", "metadata": METADATA}
    return asyncio.run(run_episode(toy_task(with_compute=compute), ScriptedClient(turns), row,
                                   counter=ByteCounter(), sample_index=0,
                                   episode_id="officeqa-fixture-0", model={"id": "scripted"}))


def score(record, judge, ground_truth="100", policy=POLICY):
    scorer = officeqa.scorer(judge, policy, count)
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
    assert result.valid and result.value == 1.0 and result.policy == "officeqa/path-report-v2+test-judge-v1"
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


def test_a_value_that_only_compute_printed_is_fabricated():
    claim = "FY1940 national defense was 999 million dollars."
    record = officeqa_episode(report(answer="999", claim=claim), compute=True)
    result = score(record, FrozenJudge([]), ground_truth="999")
    assert result.valid and result.value == 0.0 and result.detail.startswith("fabricated"), result.detail


def value_check(claim, *, code, cited="obs_1"):
    """The value gate on one leaf step, over a read (two table rows) and a compute call."""
    observations = [
        {"observation_id": "obs_1", "order": 1, "tool": "read_document", "args": {}, "delivered_to_request": 1,
         "generated_by_request": 0, "delivered_text": "Bills | 70,000 | 1,805 | 16,942"},
        {"observation_id": "obs_2", "order": 2, "tool": "compute", "args": {"code": code}, "delivered_to_request": 2,
         "generated_by_request": 1, "delivered_text": "Output:\n71805"},
        {"observation_id": "obs_3", "order": 3, "tool": "search_documents", "args": {}, "delivered_to_request": 3,
         "generated_by_request": 2, "delivered_text": "treasury_1941_01.txt: national defense"}]
    ledger = path_report.EpisodeLedger.from_record({"episode_id": "e", "observations": observations})
    report = path_report.parse_report(json.dumps({"answer": "1", "path": [
        {"id": "a", "observation": cited, "claim": claim}]}))
    return path_report.check_claimed_values(report, ledger).valid


def test_the_value_gate_takes_computed_figures_and_rounded_ones_but_not_echoed_ones():
    assert value_check("the bills sum to 71,805", code="print(70000 + 1805)")        # computed from the rows
    assert not value_check("the bills sum to 71,805", code="print(71805)")       # written in and printed back
    assert value_check("the row is 16.9 billion", code="print(1)")                 # rounds the cited 16,942 million
    assert not value_check("the row is 16.9 billion", code="print(1)", cited="obs_3")   # not what obs_3 shows
    assert not value_check("the row is 16.8 billion", code="print(1)")


def test_evidence_too_large_for_the_judge_is_an_invalid_score_not_a_zero():
    small = JudgePolicy(id="test-judge-v1", models=("judge-a",), max_prompt_tokens=10, max_output_tokens=512,
                        response_schema=officeqa.VERDICT_SCHEMA)
    judge = FrozenJudge([])
    result = score(officeqa_episode(), judge, policy=small)
    assert not result.valid and result.detail.startswith("over_budget") and judge.requests == []
    assert result.components["report_over_budget"] == 1.0


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
    assert result.value == 1.0 and result.judge["fallback"] is True and result.judge["headline"] is False
    assert score(officeqa_episode(), FrozenJudge([supported()])).judge["headline"] is True


def test_evidence_the_model_never_received_cannot_support_an_answer():
    result = score(officeqa_episode(read_first=False), FrozenJudge([]))
    assert result.valid and result.value == 0.0 and result.detail.startswith("invalid_path")


@pytest.mark.parametrize("answer,gold,correct", [
    ("Yes", "Yes", True), ("Yes, it rose in 1941", "Yes", True), ("Yes.", "Yes", True),
    ("not yes", "Yes", False), ("Yesterday", "Yes", False), ("eyes", "Yes", False), ("No-one", "No", False),
    ("The Federal Old-Age and Survivors Insurance (OASI) Trust Fund",
     "Federal Old-Age and Survivors Insurance Trust Fund", True),
    ("(5)", "-5", True), ("($1,299 million)", "-1,299 million", True), ("\u20135", "-5", True),
    ("(5)", "5", False), ("\u20135", "5", False), ("1,299 million", "-1,299 million", False),
    ("(5), (3)", "-5, -3", True), ("5 (in millions), 3", "5, 3", True),
])
def test_text_golds_lead_the_answer_and_dashes_and_parentheses_are_negative(answer, gold, correct):
    assert answer_check.check(answer, gold)[0] is correct


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

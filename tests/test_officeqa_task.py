"""OfficeQA end to end on CPU: data split and corpus build, the tools behind the node-local tool
service, the compute sandbox, judge calibration, and one episode driven and scored through all of it.

The corpus here is a tiny synthetic one; the real build is checked against `data.lock.json` by
the preparation job.
"""
from __future__ import annotations

import asyncio
import csv
import json
import shutil
import threading
import time
import zipfile
from pathlib import Path

import pytest
from support import ScriptedClient, call

from rft_cookbook.agent.harness import ByteCounter, ChatTurn, run_episode
from rft_cookbook.scoring.base import score_episode
from rft_cookbook.scoring.judge import JudgeReply
from rft_cookbook.tasks import load_task, officeqa
from rft_cookbook.tasks.officeqa import calibration, corpus, data
from rft_cookbook.tools import sandbox
from rft_cookbook.tools.base import FunctionTool, ToolContext, ToolInfraError, ToolSet
from rft_cookbook.tools.sandbox import PythonSandbox
from rft_cookbook.tools.service import RemoteToolSet, ToolServer

BULLETIN = ("Treasury Bulletin, June 1941\n\nTable 3. Budget expenditures (millions of dollars)\n"
            "| Fiscal year | National defense | Veterans |\n|---|---|---|\n| 1940 | 2,602 | 557 |\n| 1941 | 6,404 | 563 |\n")


@pytest.fixture
def corpus_dir(tmp_path):
    with zipfile.ZipFile(tmp_path / data.ARCHIVE, "w") as archive:
        archive.writestr("treasury_bulletin_1941_06.txt", BULLETIN)
        archive.writestr("treasury_bulletin_1942_06.txt", "Treasury Bulletin, June 1942\nNothing on defense here.\n")
    chunks = [{"content": BULLETIN, "source": "treasury_bulletin_1941_06.txt", "bulletin_date": "1941-06",
               "page_info": "", "chunk_type": "table"}]
    (tmp_path / data.CHUNKS).write_text("".join(json.dumps(c) + "\n" for c in chunks))
    return tmp_path


def resources(corpus_dir, **extra):
    return {"data_dir": str(corpus_dir), "sandbox": "best_effort", "require_bm25": False, **extra}


# ---- data ---------------------------------------------------------------------------------------

def test_the_split_and_probes_follow_the_measured_rule():
    rows = [{"uid": f"UID{i:04d}", "question": f"q{i}", "answer": "GOLD", "difficulty": "easy" if i < 113 else "hard"}
            for i in range(246)]
    train, heldout = data.official_split(rows)
    assert len(train) == 203 and len(heldout) == 43 and not {q.uid for q in train} & {q.uid for q in heldout}
    assert all(q.difficulty == "hard" for q in heldout)
    probe = data.probes(train)
    assert len(probe) == 16 and {q.difficulty for q in probe} == {"easy", "hard"}
    row = data.row(heldout[0], "heldout")
    assert row["messages"] == [{"role": "user", "content": heldout[0].question}] and "GOLD" not in json.dumps(row["messages"])
    assert row["metadata"]["question"] == heldout[0].question


def test_documents_keep_table_cells_and_the_build_is_checked_against_the_lock(tmp_path, monkeypatch):
    parsed = {"document": {"elements": [
        {"content": "Treasury Bulletin\r\nJune 1941"},
        {"content": "<table><tr><th>Year</th><th>Defense</th></tr><tr><td>1940</td><td>2,602</td></tr></table>"}]}}
    text = data.render_document(parsed).decode()
    assert "2,602" in text and "June 1941" in text and "\r" not in text

    release = tmp_path / "release"
    (release / data.PARSED_DIR).mkdir(parents=True)
    (release / data.TRANSFORMED_DIR).mkdir(parents=True)
    (release / data.PARSED_DIR / "treasury_bulletin_1941_06.json").write_text(json.dumps(parsed))
    (release / data.TRANSFORMED_DIR / "treasury_bulletin_1941_06.txt").write_text(BULLETIN)
    with (release / data.QUESTIONS).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uid", "question", "answer", "difficulty", "source_files"])
        writer.writeheader()
        for i in range(246):
            writer.writerow({"uid": f"UID{i:04d}", "question": f"question {i}?", "answer": str(i),
                             "difficulty": "easy" if i < 113 else "hard", "source_files": "treasury_bulletin_1941_06.txt"})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    built = tmp_path / "probe"
    built.mkdir()
    lock = {"source": {"repository": "test/officeqa", "revision": "0" * 40},
            "question_csv": data.file_identity(release / data.QUESTIONS),
            "corpus": data.build_archive(release / data.PARSED_DIR, built / data.ARCHIVE),
            "chunks": data.build_chunks(release / data.TRANSFORMED_DIR, built / data.CHUNKS)}
    monkeypatch.setattr(data, "LOCK", lock)
    out = tmp_path / "out"
    manifest = data.prepare(out, scratch=scratch, source=release)
    assert manifest["split"]["train"] == 203 and manifest["split"]["heldout"] == 43 and manifest["split"]["probe"] == 16
    assert (out / "corpus" / data.ARCHIVE).is_file() and (out / "corpus" / data.CHUNKS).is_file()
    evaluation = [json.loads(line) for line in (out / "eval.jsonl").read_text().splitlines()]
    assert sum(r["split"] == "heldout" for r in evaluation) == 43 and sum(r["split"] == "probe" for r in evaluation) == 16
    (release / data.TRANSFORMED_DIR / "treasury_bulletin_1941_06.txt").write_text(BULLETIN + "changed\n")
    with pytest.raises(ValueError, match="retrieval chunks differs"):
        data.prepare(tmp_path / "out2", scratch=scratch, source=release)


# ---- tools --------------------------------------------------------------------------------------

def run_tool(tools, name, arguments, max_bytes=8000):
    context = ToolContext(episode_id="e", question_id="q")
    return asyncio.run(tools.run(name, arguments, context, timeout_s=30, max_bytes=max_bytes))


def test_the_retrieval_tools_read_the_corpus(corpus_dir):
    tools = officeqa.local_tools(resources(corpus_dir))
    assert tools.names() == ["search_documents", "grep_documents", "read_document", "list_documents", "compute"]
    assert "treasury_bulletin_1941_06.txt" in run_tool(tools, "search_documents", {"query": "national defense"}).text
    grep = run_tool(tools, "grep_documents", {"pattern": "2,602"}).text
    assert grep.startswith("1 match(es)") and "treasury_bulletin_1941_06.txt:5:" in grep
    read = run_tool(tools, "read_document", {"file_name": "treasury_bulletin_1941_06", "start_line": 5, "num_lines": 1})
    assert read.text.endswith("5: | 1940 | 2,602 | 557 |")
    assert run_tool(tools, "list_documents", {"year": "1942"}).text == "1 document(s) for 1942:\ntreasury_bulletin_1942_06.txt"
    assert run_tool(tools, "read_document", {"file_name": "../../etc/passwd"}).text.startswith("Error: no document named")
    assert run_tool(tools, "compute", {"code": "6404 - 2602"}).text == "Result: 3802"
    assert run_tool(tools, "grep_documents", {"pattern": "defense", "bogus": 1}).status == "error"
    assert run_tool(tools, "read_document", {"file_name": "treasury_bulletin_1941_06"}, max_bytes=40).truncated


def test_the_service_serves_the_same_tools_and_its_absence_is_infrastructure(corpus_dir, tmp_path, monkeypatch):
    local = officeqa.local_tools(resources(corpus_dir))
    server = ToolServer(local)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("RFT_TOOL_SERVICE_DIR", str(tmp_path / "services"))
        (tmp_path / "services").mkdir()
        (tmp_path / "services" / "run-1.json").write_text(json.dumps({"url": server.url}))
        remote = load_task("officeqa", resources={"tool_service": "run-1"}).tools
        assert isinstance(remote, RemoteToolSet) and remote.names() == local.names()
        for name, arguments in (("grep_documents", {"pattern": "2,602"}), ("compute", {"code": "print(1+1)"})):
            assert run_tool(remote, name, arguments).text == run_tool(local, name, arguments).text
    finally:
        server.shutdown()
    with pytest.raises(ToolInfraError, match="no tool service"):
        run_tool(RemoteToolSet("not-started", officeqa.tool_schemas()[:1]), "search_documents", {"query": "x y"})
    with pytest.raises(ToolInfraError, match="unreachable"):
        run_tool(RemoteToolSet("http://127.0.0.1:9", officeqa.tool_schemas()[:1]), "search_documents", {"query": "x y"})


def test_a_tool_infrastructure_failure_crosses_the_service(tmp_path):
    def broken(**_):
        raise ToolInfraError("corpus volume is gone")

    schema = officeqa.tool_schemas()[0]
    server = ToolServer(ToolSet([FunctionTool(broken, schema)]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(ToolInfraError, match="corpus volume is gone"):
            run_tool(RemoteToolSet(server.url, [schema]), schema["function"]["name"], {"query": "x"})
    finally:
        server.shutdown()


def test_a_regex_that_backtracks_is_cut_off_and_the_service_keeps_answering(corpus_dir, monkeypatch):
    with zipfile.ZipFile(corpus_dir / data.ARCHIVE, "a") as archive:
        archive.writestr("treasury_bulletin_1943_06.txt", "a" * 60 + "b\n| July | " + " | ".join(["5,089"] * 16) + " | total\n")
    monkeypatch.setattr(corpus, "GREP_REGEX_TIMEOUT_S", 1.0)
    local = officeqa.local_tools(resources(corpus_dir))
    server = ToolServer(local)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    remote, slow = RemoteToolSet(server.url, local.schemas()), []
    try:
        caller = threading.Thread(target=lambda: slow.append(
            run_tool(remote, "grep_documents", {"pattern": "(a|aa)+$", "regex": True})))
        caller.start()
        time.sleep(0.2)
        started = time.monotonic()   # re would hold the GIL for the whole match, and this call would wait for it
        assert run_tool(remote, "grep_documents", {"pattern": "2,602"}).text.startswith("1 match(es)")
        assert time.monotonic() - started < 0.5
        caller.join(timeout=10)
        assert slow[0].text == "Error: regex '(a|aa)+$' took longer than 1 s; use a simpler pattern"
        # the pattern that stalled a training run's service (re: 38 s on one 74-character row)
        stalled = r"^.*\|.*July.*\|.*\d+.*\|.*\d+.*\|.*\d+.*\|.*\d+.*\|.*\d+.*\|.*\d+.*\|.*\d+.*\|$"
        assert run_tool(remote, "grep_documents", {"pattern": stalled, "year": "1943", "regex": True}).text.startswith("No matches")
        assert run_tool(remote, "grep_documents", {"pattern": "(unclosed", "regex": True}).text.startswith(
            "Error: invalid regex '(unclosed': missing ), unterminated subpattern")
    finally:
        server.shutdown()


@pytest.mark.skipif(not sandbox.bwrap_available(), reason="needs bubblewrap and user namespaces")
def test_strict_compute_has_no_network_and_cannot_see_data(tmp_path, monkeypatch):
    secret = tmp_path / "answers.txt"   # the host's /tmp is replaced by a private one inside
    secret.write_text("gold answer")
    strict = PythonSandbox("strict", timeout_s=10)
    result = strict.run(f"print(open({str(secret)!r}).read())")
    assert result.status == "error" and "FileNotFoundError" in result.text and result.sandbox.startswith("bwrap")
    network = strict.run("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)")
    assert network.status == "error" and "unreachable" in network.text
    monkeypatch.setenv("HF_TOKEN", "do-not-leak")
    assert "do-not-leak" not in PythonSandbox("strict").run("import os\nprint(os.environ)").text


def test_strict_without_bubblewrap_runs_nothing(monkeypatch):
    monkeypatch.setattr(sandbox, "bwrap_available", lambda: False)
    with pytest.raises(ToolInfraError, match="unavailable"):
        PythonSandbox("strict").run("1 + 1")
    fallback = PythonSandbox("best_effort").run("1 + 1")
    assert fallback.text == "Result: 2" and fallback.sandbox.startswith("subprocess")


# ---- judge calibration and one full episode -----------------------------------------------------

class ContentJudge:
    """Labels a request supported unless its evidence shows the planted mismatch."""

    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, 0

    async def complete(self, messages, policy):
        self.calls += 1
        if self.fail:
            return JudgeReply(content="not json", finish_reason="stop", model=policy.models[0])
        text = messages[-1]["content"]
        bad = any(marker in text for marker in ("FY1941 | 2,602", "Agriculture", "Computed FY1940"))
        status, score = ("unsupported", 0.0) if bad else ("supported", 0.9)
        return JudgeReply(content=json.dumps({"path_status": status, "path_score": score, "issues": []}),
                          finish_reason="stop", model=policy.models[0])


def test_calibration_passes_a_discriminating_judge_and_fails_a_broken_one():
    policy = officeqa.judge_policy({"models": ["glm"]})
    good = asyncio.run(calibration.calibrate(officeqa.OfficeQAScorer(ContentJudge(), policy, lambda m: 100)))
    assert good["status"] == "PASS" and len(good["checks"]) == 10
    assert sorted({(c["case"], c["expected"]) for c in good["checks"]}) == [
        ("supported_arithmetic", 1.0), ("supported_lookup", 1.0), ("ungrounded_compute", 0.0),
        ("wrong_category", 0.0), ("wrong_period", 0.0)]
    broken = asyncio.run(calibration.calibrate(officeqa.OfficeQAScorer(ContentJudge(fail=True), policy, lambda m: 100)))
    assert broken["status"] == "FAIL" and not any(c["valid"] for c in broken["checks"])


def test_an_episode_through_the_service_is_scored_by_the_judge(corpus_dir):
    server = ToolServer(officeqa.local_tools(resources(corpus_dir)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        task = load_task("officeqa", resources={"tool_service": server.url}, max_turns=4)
        assert "up to 4 steps" in task.system_prompt
        report = {"answer": "2,602", "path": [{"id": "a", "observation": "obs_1",
                                                "claim": "FY1940 national defense was 2,602 million dollars."}]}
        client = ScriptedClient([
            ChatTurn(text="", tool_calls=[call("grep_documents", '{"pattern": "2,602"}', "g1")]),
            ChatTurn(text="", tool_calls=[call("submit_report", json.dumps(report), "s1")])])
        question = "What were national defense expenditures in FY1940, in millions of dollars?"
        row = {"id": "UID0001", "split": "heldout", "messages": [{"role": "user", "content": question}],
               "ground_truth": "2602", "metadata": {"question": question}}
        record = asyncio.run(run_episode(task, client, row, counter=ByteCounter()))
    finally:
        server.shutdown()
    assert record.termination.reason == "answer" and record.turns[0].tool_calls[0].observation_id == "obs_1"
    assert record.turns[0].tool_calls[0].delivered_text.startswith("[observation obs_1]\n1 match(es)")
    judge = ContentJudge()
    scorer = officeqa.OfficeQAScorer(judge, officeqa.judge_policy({"models": ["glm"]}), lambda m: 100)
    score = asyncio.run(score_episode(scorer, record, "2602", row["metadata"]))
    assert score.valid and score.value == 1.0 and judge.calls == 1 and score.components["answer_correct"] == 1.0


def test_the_task_carries_the_measured_budgets_and_needs_a_judge_only_to_score():
    task = load_task("officeqa")
    assert (task.budgets.max_turns, task.budgets.max_turn_tokens, task.budgets.max_context_tokens,
            task.budgets.tool_output_bytes, task.budgets.max_tool_calls_per_turn) == (128, 4096, 98_304, 8000, 1)
    assert [s["function"]["name"] for s in task.tool_schemas()] == [
        "search_documents", "grep_documents", "read_document", "list_documents", "compute", "submit_report"]
    policy = officeqa.judge_policy({"models": ["system.ai.glm-5-3-flash"], "max_model_len": 65536, "max_tokens": 4096})
    assert policy.max_prompt_tokens == 65536 - 4096 - 64 and policy.deadline_s == 600 and policy.retries == 2
    assert load_task("officeqa", resources={"judge": {"id": "glm", "models": ["m"]}}).scorer.policy == \
        "officeqa/path-report-v1+glm"

"""Agentic search (rft_cookbook.tasks.search): the answer and its match, the tools over a fake
Vector Search index, an episode through the harness, and the data split."""
from __future__ import annotations

import asyncio
import json

import pytest
from support import ScriptedClient, call

from rft_cookbook.agent.harness import ByteCounter, ChatTurn, run_episode
from rft_cookbook.scoring.base import score_episode
from rft_cookbook.tasks import load_task, search
from rft_cookbook.tasks.search import data, retrieval, run_resources
from rft_cookbook.tasks.search.answer import extract_answer, match
from rft_cookbook.tools.base import ToolContext, ToolInfraError

INDEX = "cat.sch.passages_index"
PASSAGES = [
    {"id": "p1", "title": "Inception", "text": "Inception is a 2010 film directed by Christopher Nolan.", "score": 0.91},
    {"id": "p2", "title": "Christopher Nolan", "text": "Christopher Nolan was born in London." + " x" * 400,
     "score": 0.87},
    {"id": "p3", "title": "Christopher Nolan", "text": "He studied English literature at UCL.", "score": 0.5},
]


class FakeService:
    """The query REST API of one index: ANN and HYBRID rank by word overlap; a title filter is exact."""

    def __init__(self, fail: bool = False):
        self.fail, self.requests = fail, []

    def __call__(self, method, path, body):
        self.requests.append((method, path, body))
        if self.fail:
            raise ConnectionError("endpoint unreachable")
        if method == "GET":
            return {"name": INDEX, "endpoint_name": "ep", "status": {"ready": True, "indexed_row_count": 3},
                    "delta_sync_index_spec": {"source_table": "cat.sch.passages",
                                              "embedding_source_columns": [{"name": "text",
                                                                            "embedding_model_endpoint_name": "gte"}]}}
        rows = PASSAGES
        if body.get("filters_json"):
            rows = [p for p in rows if p["title"] == json.loads(body["filters_json"])["title"]]
        words = set(body["query_text"].lower().split())
        rows = [p for p in rows if words & set((p["title"] + " " + p["text"]).lower().split())][:body["num_results"]]
        return {"manifest": {"columns": [{"name": c} for c in (*body["columns"], "score")]},
                "result": {"data_array": [[p["id"], p["title"], p["text"], p["score"]] for p in rows]}}


def tool(name, service, **arguments):
    tools = {t.name: t for t in retrieval.tools(retrieval.VectorSearchIndex(INDEX, request=service))}
    return asyncio.run(tools[name].call(arguments, ToolContext("e", "q"))).text


def test_the_answer_is_the_last_answer_block_and_matches_any_alias():
    assert extract_answer("<answer>Paris</answer> then <answer> London </answer>") == "London"
    assert extract_answer("no tags here") is None
    assert match("The Beatles!", ["Beatles"]) == {"em": 1.0, "cover_em": 1.0, "f1": 1.0}
    assert match("London, England", ["Paris", "London"]) == {"em": 0.0, "cover_em": 1.0, "f1": 2 / 3}
    assert match("Rome", "Paris")["em"] == 0.0 and match("", ["x"])["f1"] == 0.0


def test_searches_show_titles_scores_and_clipped_snippets_and_are_cached():
    service = FakeService()
    found = tool("keyword_search", service, query="Christopher Nolan film", top_k=50)
    assert found.startswith("[0] Inception  score=0.910") and "[1] Christopher Nolan" in found
    assert found.count("...") == 1 and len(found) < 1500                         # the long passage is clipped
    method, path, body = service.requests[0]
    assert (method, path) == ("POST", f"/api/2.0/vector-search/indexes/{INDEX}/query")
    assert body["query_type"] == "HYBRID" and body["num_results"] == retrieval.MAX_TOP_K
    assert tool("vector_search", service, query="who directed")[:3] == "[0]"
    assert service.requests[-1][2]["query_type"] == "ANN" and service.requests[-1][2]["num_results"] == retrieval.TOP_K
    index = retrieval.VectorSearchIndex(INDEX, request=service)
    before = len(service.requests)
    index.query("Nolan", 5, "ANN"), index.query("Nolan", 5, "ANN")
    assert len(service.requests) == before + 1                                    # asked once


def test_a_search_without_hits_never_echoes_the_query_and_an_article_is_read_by_exact_title():
    service = FakeService()
    assert tool("vector_search", service, query="zebra quantum") == retrieval.NO_HITS
    article = tool("read_article", service, title="Christopher Nolan")
    assert article.startswith("Article: Christopher Nolan\n\n") and "UCL" in article and "Inception is" not in article
    assert json.loads(service.requests[-1][2]["filters_json"]) == {"title": "Christopher Nolan"}
    assert tool("read_article", service, title="Nolan") == retrieval.NO_ARTICLE
    assert tool("keyword_search", service, query="x").startswith("Error:")       # too short: the model's to fix


def test_a_service_failure_is_an_infrastructure_failure_and_identity_names_the_index():
    with pytest.raises(ToolInfraError, match="unreachable"):
        tool("keyword_search", FakeService(fail=True), query="Nolan")
    with pytest.raises(ToolInfraError, match="no Vector Search index"):
        asyncio.run(retrieval.tools(None)[0].call({"query": "Nolan"}, ToolContext("e", "q")))
    with pytest.raises(ValueError):
        retrieval.VectorSearchIndex("<vs-index>")
    described = retrieval.VectorSearchIndex(INDEX, request=FakeService()).describe()
    assert retrieval.identity(described) == {"index": INDEX, "endpoint": "ep", "ready": True, "rows": 3,
                                             "source_table": "cat.sch.passages", "embedding": [("text", "gte")]}


def test_an_episode_searches_reads_and_answers_and_scores_by_exact_match(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(search, "VectorSearchIndex", lambda index: retrieval.VectorSearchIndex(index, request=service))
    task = load_task("search", resources=run_resources({}, {"vector_search_index": INDEX}))
    assert "up to 12 steps" in task.system_prompt and task.settings()["retrieval"] == {"index": INDEX}
    assert [s["function"]["name"] for s in task.tool_schemas()] == ["vector_search", "keyword_search", "read_article"]
    row = {"id": "2hop__1_2", "split": "heldout", "messages": [{"role": "user", "content": "Question: Where was the "
           "director of Inception born?"}], "ground_truth": ["London", "London, England"], "metadata": {"hops": 2}}
    turns = [ChatTurn("", [call("keyword_search", '{"query": "Inception film director"}')]),
             ChatTurn("", [call("read_article", '{"title": "Christopher Nolan"}')]),
             ChatTurn("He was born in London. <answer> London </answer>")]
    record = asyncio.run(run_episode(task, ScriptedClient(turns), row, counter=ByteCounter()))
    assert record.answer == "London" and [c.name for c in record.calls()] == ["keyword_search", "read_article"]
    result = asyncio.run(score_episode(task.scorer, record, row["ground_truth"], row["metadata"]))
    assert result.valid and result.value == 1.0 and result.policy == "search/em-v1"


def test_no_answer_block_scores_zero_and_every_score_is_valid():
    task = load_task("search")
    record = asyncio.run(run_episode(task, ScriptedClient([ChatTurn("I think it is London.")]),
                                     {"id": "q", "split": "heldout", "messages": [{"role": "user", "content": "Q"}],
                                      "ground_truth": ["London"], "metadata": {}}, counter=ByteCounter()))
    result = asyncio.run(score_episode(task.scorer, record, ["London"], {}))
    assert result.valid and result.value == 0.0 and result.components["answered"] == 0.0


def _example(qid, question, answer="A", aliases=(), answerable=True, hops=2):
    return {"id": qid, "question": question, "answer": answer, "answer_aliases": list(aliases),
            "answerable": answerable, "question_decomposition": [{}] * hops}


def test_the_split_holds_the_recorded_heldout_list_probes_from_training_and_no_overlap():
    held = data.heldout_ids()
    assert len(held) == 500 and len(set(held)) == 500
    validation = [_example(i, f"held question {i}", hops=int(i[0])) for i in held] + [_example("extra", "e")]
    train = [_example(f"t{i}", f"train question {i}", aliases=["a", " B "]) for i in range(40)]
    train.append(_example("t-x", "unanswerable", answerable=False))
    rows, heldout, probe, dropped = data.split(train, validation)
    assert len(rows) == 40 and dropped == {"train: unanswerable or empty": 1}
    assert [r["id"] for r in heldout] == held and {r["split"] for r in heldout} == {"heldout"}
    assert len(probe) == data.PROBES and {r["split"] for r in probe} == {"probe"}
    assert {r["id"] for r in probe} <= {r["id"] for r in rows}
    assert rows[0]["ground_truth"] == ["A", "B"] and rows[0]["messages"][0]["content"].startswith("Question: ")
    assert heldout[0]["metadata"] == {"hops": int(held[0][0])}
    with pytest.raises(SystemExit, match="also a training question"):
        data.split(train + [_example("t-dup", f"held question {held[0]}")], validation)
    with pytest.raises(SystemExit, match="not in the validation split"):
        data.split(train, validation[1:])


def test_the_preflight_names_a_ready_index_and_stops_without_one(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(search, "VectorSearchIndex", lambda index: retrieval.VectorSearchIndex(index, request=service))
    assert search.preflight(run_resources({}, {"vector_search_index": INDEX}))["rows"] == 3
    with pytest.raises(SystemExit, match="RFT_VS_INDEX"):
        search.preflight(run_resources({}, {}))
    empty = FakeService()
    empty_call = empty.__call__
    monkeypatch.setattr(search, "VectorSearchIndex", lambda index: retrieval.VectorSearchIndex(
        index, request=lambda m, p, b: {**empty_call(m, p, b), "status": {"ready": False}}))
    with pytest.raises(SystemExit, match="not ready"):
        search.preflight(run_resources({}, {"vector_search_index": INDEX}))

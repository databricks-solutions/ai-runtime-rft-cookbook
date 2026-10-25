# Derived from verl-on-air (Apache License 2.0), usecases/agentic-search/tool.py at 1c5653c;
# modified for ai-runtime-rft-cookbook: cookbook tools over one index client; a backend failure is an
# infrastructure failure of the episode, in training as in evaluation. See THIRD_PARTY_NOTICES.md.
"""Retrieval tools over a Databricks Vector Search index of encyclopedia passages.

    vector_search(query, top_k)   semantic (ANN) search
    keyword_search(query, top_k)  hybrid keyword and semantic search, for names, titles and numbers
    read_article(title)           every passage of one article, by its exact title

The index is a managed service: nothing is loaded on the GPU nodes, and each agent worker sends
authenticated REST queries (`/api/2.0/vector-search/indexes/<index>/query`) through the
databricks-sdk the images carry, with the job's own workspace credentials. (The
databricks-vectorsearch client is not used: installing it downgrades protobuf below what vLLM
needs.) A query the service cannot answer raises `ToolInfraError`, so the episode is an
infrastructure failure, never a wrong answer. Results are cached per process: the index does not
change during a run, and a group's episodes often repeat a query.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Callable

from rft_cookbook.tools.base import FunctionTool, ToolInfraError, function_schema

TOP_K, MAX_TOP_K = 5, 20
SNIPPET_CHARS = 600
ARTICLE_PASSAGES = 20
CACHE_ENTRIES = 8192
COLUMNS = ("id", "title", "text")
NO_HITS = "No passages found. Try different terms, or the other search tool."
NO_ARTICLE = ("No article with exactly that title was found. Use vector_search or keyword_search to find the "
              "exact title first (titles are shown in each result).")


class VectorSearchIndex:
    """Queries of one index. `request(method, path, body)` defaults to the workspace's REST API."""

    def __init__(self, index: str, *, request: Callable[[str, str, dict | None], dict] | None = None):
        if not index or "<" in index:
            raise ValueError(f"a Vector Search index name (catalog.schema.index) is required, not {index!r}")
        self.index = index
        self._request = request
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple, list[dict[str, Any]]] = OrderedDict()

    def query(self, text: str, k: int, query_type: str, filters: dict | None = None) -> list[dict[str, Any]]:
        key = (query_type, text, k, json.dumps(filters, sort_keys=True))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        body = {"num_results": k, "columns": list(COLUMNS), "query_text": text, "query_type": query_type}
        if filters:
            body["filters_json"] = json.dumps(filters)
        response = self._call("POST", f"/api/2.0/vector-search/indexes/{self.index}/query", body)
        names = [c["name"] for c in (response.get("manifest") or {}).get("columns") or []] or [*COLUMNS, "score"]
        rows = [dict(zip(names, values)) for values in (response.get("result") or {}).get("data_array") or []]
        with self._lock:
            self._cache[key] = rows
            while len(self._cache) > CACHE_ENTRIES:
                self._cache.popitem(last=False)
        return rows

    def describe(self) -> dict[str, Any]:
        """The index as the service reports it (status, row count, source table, embedding model)."""
        return self._call("GET", f"/api/2.0/vector-search/indexes/{self.index}", None)

    def _call(self, method: str, path: str, body: dict | None) -> dict:
        try:
            if self._request is None:
                from databricks.sdk import WorkspaceClient

                api = WorkspaceClient().api_client   # the job's ambient workspace credentials
                self._request = lambda m, p, b: api.do(m, p, body=b) if b is not None else api.do(m, p)
            return self._request(method, path, body)
        except Exception as error:  # noqa: BLE001 - every backend failure is the infrastructure's
            raise ToolInfraError(f"Vector Search {method} {path.rsplit('/', 1)[-1]} failed: "
                                 f"{type(error).__name__}: {error}") from None


def identity(described: dict[str, Any]) -> dict[str, Any]:
    """What an evaluation or a qualification record names the index by."""
    status = described.get("status") or {}
    spec = described.get("delta_sync_index_spec") or {}
    return {"index": described.get("name"), "endpoint": described.get("endpoint_name"),
            "ready": bool(status.get("ready")), "rows": status.get("indexed_row_count"),
            "source_table": spec.get("source_table"),
            "embedding": [(c.get("name"), c.get("embedding_model_endpoint_name"))
                          for c in spec.get("embedding_source_columns") or []]}


def _top_k(value: Any) -> int:
    try:
        return max(1, min(int(str(value).strip()), MAX_TOP_K))
    except ValueError:
        return TOP_K


def _results(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return NO_HITS   # never echo the query: the model must not read its own words back as evidence
    parts = []
    for i, row in enumerate(rows):
        text, score = str(row.get("text") or ""), row.get("score")
        snippet = text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "..."
        shown = f"  score={score:.3f}" if isinstance(score, (int, float)) else ""
        parts.append(f"[{i}] {row.get('title') or '?'}{shown}\n    {snippet}")
    return "\n\n".join(parts)


def tools(index: VectorSearchIndex | None) -> list[FunctionTool]:
    def backend() -> VectorSearchIndex:
        if index is None:
            raise ToolInfraError("no Vector Search index is configured for this run")
        return index

    def search(query_type: str):
        def run(query: str, top_k: int = TOP_K) -> str:
            text = str(query).strip()
            if len(text) < 2:
                return "Error: query must be at least 2 characters"
            return _results(backend().query(text, _top_k(top_k), query_type))
        return run

    def read_article(title: str) -> str:
        wanted = str(title).strip()
        if len(wanted) < 2:
            return "Error: title must be at least 2 characters"
        rows = backend().query(wanted, ARTICLE_PASSAGES, "ANN", filters={"title": wanted})
        # the filter is exact on the server; matching again here keeps a looser index honest
        hits = [r for r in rows if str(r.get("title") or "").strip().lower() == wanted.lower()]
        if not hits:
            return NO_ARTICLE
        return f"Article: {hits[0]['title']}\n\n" + "\n".join(str(r.get("text") or "") for r in hits)

    query_arguments = {"query": {"type": "string", "description": "What to look for."},
                       "top_k": {"type": "integer", "description": f"How many passages to return (default {TOP_K}; "
                                                                   f"at most {MAX_TOP_K})."}}
    return [
        FunctionTool(search("ANN"), function_schema(
            "vector_search",
            "Semantic search over the encyclopedia corpus; returns the most relevant passages, each with its "
            "article title. Use it for conceptual or paraphrased questions; pass a result's title to "
            "read_article to read the whole article.", query_arguments, ["query"])),
        FunctionTool(search("HYBRID"), function_schema(
            "keyword_search",
            "Hybrid keyword and semantic search over the encyclopedia corpus. Prefer it when the query holds "
            "a proper noun, an exact title, a number or a rare term that must appear verbatim.",
            {**query_arguments, "query": {"type": "string", "description": "What to look for, with the exact "
                                                                           "names or terms that must match."}},
            ["query"])),
        FunctionTool(read_article, function_schema(
            "read_article",
            "Read every passage of one article by its EXACT title, as shown in a search result, e.g. to find "
            "the entity for the next hop of a multi-hop question.",
            {"title": {"type": "string", "description": "The exact article title."}}, ["title"])),
    ]

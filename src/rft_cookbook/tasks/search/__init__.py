"""Agentic search: multi-hop questions (MuSiQue) answered by searching an encyclopedia corpus.

    task(resources, **budget_overrides)

The model searches a Databricks Vector Search index of passages (`retrieval.py`) and ends with a
turn without tool calls that holds its answer in `<answer>...</answer>`. The score is exact match
against the question's answer and aliases (`answer.py`); cover-EM and F1 are recorded beside it.
No judge: every score is valid. `resources`:

    vector_search   {"index": "catalog.schema.index"}; without it the tools still render, and a
                    call is an infrastructure failure

Budgets: 12 turns (the earlier prototype's 8 cut off episodes that were still searching), one
tool call per turn, 4,096 generated tokens per turn (Qwen3.6 thinks before each call), 4,000-byte
tool results, a 43,008-token context.

The index the measured prototype searched holds MuSiQue's paragraphs (20 a question) and HotpotQA's
distractor-setting paragraphs (10 a question), training and validation splits, at the revisions
`data.py` pins for MuSiQue and hotpotqa/hotpot_qa@1908d6af (distractor): one row per distinct title and text
of at least 20 characters (id `p_` + the first 20 hex digits of md5(lowercased title, newline,
text)), 603,607 passages in columns id, title and text, a Delta Sync index with managed
embeddings (databricks-gte-large-en) on `text`. It holds the held-out questions' own paragraphs,
gold and distractors, so every answer is retrievable: a fixed, transductive setting that the base
and the trained model search alike.
"""
from __future__ import annotations

from typing import Any

from rft_cookbook.agent.spec import AgentTask, Budgets, TerminalRule
from rft_cookbook.scoring.base import RewardResult
from rft_cookbook.tasks.search.answer import extract_answer, match
from rft_cookbook.tasks.search.retrieval import VectorSearchIndex, identity, tools
from rft_cookbook.tools.base import ToolSet

NAME = "search"
VERSION = "1"

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions by searching an encyclopedic corpus with "
    "tools:\n"
    "- vector_search(query, top_k): semantic search; returns passages, each with its article title.\n"
    "- keyword_search(query, top_k): hybrid keyword and semantic search; use it when exact names, "
    "titles, numbers or rare terms must match.\n"
    "- read_article(title): read the whole article with that EXACT title (as shown in a result).\n\n"
    "Work step by step, one tool call per step; you have up to [[MAX_TURNS]] steps. For a multi-hop "
    "question, find the first entity, read about it to discover the next one, then search again. "
    "Ground every claim in retrieved passages -- do not answer from memory alone. When you have enough "
    "evidence, stop calling tools and give your final answer and NOTHING else, wrapped in <answer> "
    "and </answer> tags. The answer is a short span -- a name, entity, number, date, or yes/no -- with "
    "no extra words. For example: <answer> Beijing </answer>."
)

BUDGETS = Budgets(max_turns=12, max_response_tokens=40_960, max_turn_tokens=4096, max_context_tokens=43_008,
                  max_tool_calls_per_turn=1, tool_output_bytes=4000, tool_timeout_s=60.0)


def system_prompt(max_turns: int) -> str:
    return SYSTEM_PROMPT.replace("[[MAX_TURNS]]", str(max_turns))


class ExactMatchScorer:
    policy = "search/em-v1"
    components = ("answered", "em", "cover_em", "f1")

    async def score(self, episode, ground_truth: Any, metadata: dict[str, Any]) -> RewardResult:
        if episode.answer is None:
            return RewardResult(valid=True, score=0.0, components={"answered": 0.0, "em": 0.0, "cover_em": 0.0,
                                                                   "f1": 0.0})
        found = match(episode.answer, ground_truth)
        return RewardResult(valid=True, score=found["em"], components={"answered": 1.0, **found})


def run_resources(recipe: dict[str, Any], paths: dict[str, Any]) -> dict[str, Any]:
    """The index the run searches: the workspace's, which the job names (`RFT_VS_INDEX`)."""
    return {"vector_search": {"index": paths.get("vector_search_index")}}


def preflight(resources: dict[str, Any]) -> dict[str, Any]:
    """The index as the service reports it; stops the stage unless it is ready and holds passages."""
    index = (resources.get("vector_search") or {}).get("index")
    if not index:
        raise SystemExit("agentic search needs its Vector Search index: submit the job with "
                         "--override env_variables.RFT_VS_INDEX=<catalog>.<schema>.<index>")
    found = identity(VectorSearchIndex(index).describe())
    if not found["ready"] or not found["rows"]:
        raise SystemExit(f"the Vector Search index is not ready to search: {found}")
    return found


def task(resources: dict[str, Any] | None = None, **budget_overrides: Any) -> AgentTask:
    config = (resources or {}).get("vector_search") or {}
    index = VectorSearchIndex(config["index"]) if config.get("index") else None
    budgets = BUDGETS.replace(**budget_overrides)
    return AgentTask(name=NAME, version=VERSION, system_prompt=system_prompt(budgets.max_turns),
                     tools=ToolSet(tools(index)), terminal=TerminalRule.answer_pattern(extract_answer),
                     budgets=budgets, scorer=ExactMatchScorer(),
                     metadata={"settings": {"retrieval": {"index": index.index if index else None}}})

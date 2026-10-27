"""OfficeQA: questions about U.S. Treasury Bulletins, answered with retrieval and compute tools and
a cited path of evidence.

    task(resources, **budget_overrides)

`resources` says where the task's state lives (all optional, so rendering and data conversion
work without any):

    tool_service   name of the node-local tool service episodes call (rft_cookbook.tools.service)
    data_dir       directory with the corpus archive and chunks (the service, or in-process tools)
    sandbox        compute isolation: strict (default) | best_effort | off
    compute        {"timeout_s": 15, "mem_mb": 4096}
    judge          the support judge: {"id", "provider": databricks|openai, "base_url", "models",
                   "tokenizer", "max_model_len", "max_tokens", "timeout_s", "retries", "deadline_s",
                   "reasoning_effort", "concurrency", "profile"}

Budgets are those of the measured verl-on-air run: 128 assistant turns, one tool call per turn,
4,096 generated tokens per turn, 8,000-byte tool results, a 98,304-token context.
"""
from __future__ import annotations

import json
import threading
from importlib import resources as package_files
from pathlib import Path
from typing import Any

from rft_cookbook.agent.spec import AgentTask, Budgets, TerminalRule
from rft_cookbook.scoring.base import RewardResult
from rft_cookbook.scoring.judge import JudgeError, JudgePolicy
from rft_cookbook.tasks.officeqa.scoring import COMPONENTS, VERDICT_SCHEMA, OfficeQAScorer, render_observation
from rft_cookbook.tools.base import FunctionTool, ToolSet

NAME = "officeqa"
VERSION = "1"
TERMINAL_TOOL = "submit_report"

SYSTEM_PROMPT = (
    "You are a meticulous financial-data research agent answering questions about U.S. "
    "Treasury Bulletins. Ground every answer in figures you actually retrieve.\n\n"
    "Tools (each result is tagged on its first line as [observation obs_N] -- an id you must "
    "cite later):\n"
    "- search_documents(query, top_k): find the file/table for a figure.\n"
    "- grep_documents(pattern, file_name, year, regex): pin an exact row label/value.\n"
    "- read_document(file_name, start_line, num_lines): read the table cells + headers.\n"
    "- list_documents(year): list available bulletins.\n"
    "- compute(code): run Python (numpy/pandas) for ALL arithmetic.\n"
    "- submit_report(answer, path): FINISH the task. This is the ONLY way to end.\n\n"
    "How this task runs -- plan your steps around it:\n"
    "- You have up to [[MAX_TURNS]] steps, but you need not use them all: call submit_report as "
    "soon as you are confident in your answer.\n"
    "- If you have retrieved figures relevant to the question, commit your BEST supported answer "
    'and cite the observations behind it. Reserve answer="DATA NOT AVAILABLE" for when nothing you '
    "retrieved is relevant -- do not decline merely because you are unsure.\n\n"
    "FINISH by calling submit_report with your answer and the supporting path as its arguments. "
    "Do NOT print the JSON as text -- call the tool. Its arguments:\n"
    '  answer: "<ONLY the final value, in the units the question asks -- no working or other '
    'figures>"\n'
    '  path: a list of {"id","observation":"obs_N you received","claim":"row/period/units/value '
    'or the calculation","depends_on":["earlier id",...]}\n'
    "Rules for the path: cite ONLY observation ids that were delivered to you; a compute step "
    "must list in depends_on the read/search observations whose figures it used; report the path "
    "that actually supports the answer, not every dead end."
)


def system_prompt(max_turns: int) -> str:
    return SYSTEM_PROMPT.replace("[[MAX_TURNS]]", str(max_turns))


BUDGETS = Budgets(max_turns=128, max_response_tokens=96_256, max_turn_tokens=4096, max_context_tokens=98_304,
                  max_tool_calls_per_turn=1, tool_output_bytes=8000, tool_timeout_s=60.0)
RETRIEVAL = ("search_documents", "grep_documents", "read_document", "list_documents")


def tool_schemas() -> list[dict]:
    """All six schemas the model sees, `submit_report` (the terminal tool) included."""
    return json.loads(package_files.files(__package__).joinpath("tools.json").read_text())


def _executable_schemas() -> list[dict]:
    return [s for s in tool_schemas() if s["function"]["name"] != TERMINAL_TOOL]


def local_tools(resources: dict[str, Any]) -> ToolSet:
    """The five executable tools in this process: what the tool service hosts."""
    from rft_cookbook.tasks.officeqa import corpus
    from rft_cookbook.tools.sandbox import PythonSandbox

    data_dir = Path(resources["data_dir"])
    compute = resources.get("compute") or {}
    sandbox = PythonSandbox(resources.get("sandbox", "strict"), timeout_s=float(compute.get("timeout_s", 15)),
                            mem_mb=int(compute.get("mem_mb", 4096)))
    schemas = {s["function"]["name"]: s for s in _executable_schemas()}

    def retrieval(name):
        function = getattr(corpus, name)
        return lambda **arguments: function(corpus.load_corpus(data_dir), **arguments)

    tools = [FunctionTool(retrieval(name), schemas[name]) for name in RETRIEVAL]
    tools.append(FunctionTool(lambda code: sandbox.run(code), schemas["compute"]))
    return ToolSet(tools)


def warm(resources: dict[str, Any]) -> None:
    """Load the corpus and build the search index before the service reports ready."""
    from rft_cookbook.tasks.officeqa import corpus

    loaded = corpus.load_corpus(Path(resources["data_dir"]))
    if resources.get("require_bm25", True) and loaded.retriever is None:
        raise RuntimeError("bm25s is not importable: search would fall back to keyword overlap")
    try:   # grep's regex deadline; without it one policy pattern can stall the service
        corpus.regex_engine.compile("ok").search("ok", timeout=1.0, concurrent=True)
    except TypeError as error:
        raise RuntimeError(f"regex {corpus.regex_engine.__version__} cannot time out a match") from error


def judge_policy(config: dict[str, Any]) -> JudgePolicy:
    """The support judge's policy. Its prompt budget leaves room for the verdict, as measured."""
    max_tokens = int(config.get("max_tokens", 4096))
    return JudgePolicy(
        id=config.get("id", "officeqa-support-v1"), models=tuple(config["models"]),
        max_prompt_tokens=int(config.get("max_model_len", 65_536)) - max_tokens - 64, max_output_tokens=max_tokens,
        temperature=0.0, timeout_s=float(config.get("timeout_s", 180)), retries=int(config.get("retries", 2)),
        deadline_s=float(config.get("deadline_s", 600)), reasoning_effort=config.get("reasoning_effort", "low"),
        response_schema=VERDICT_SCHEMA, fallback_in_headline=False, extra={"schema_name": "support"})


class _TemplateTokens:
    """Judge prompt tokens, counted with the judge's own chat template (thread-safe)."""

    def __init__(self, path: str, reasoning_effort: str | None):
        self.path, self.reasoning_effort = path, reasoning_effort
        self.lock = threading.Lock()
        self.tokenizer = None

    def __call__(self, messages: list[dict]) -> int:
        with self.lock:
            if self.tokenizer is None:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
            kwargs = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
            return len(self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                          add_generation_prompt=True, **kwargs))


class TaskScorer:
    """OfficeQA's scorer with its judge built on first use, so a task can be loaded anywhere."""

    components = COMPONENTS

    def __init__(self, config: dict[str, Any] | None):
        self.config = config
        self.policy = f"officeqa/path-report-v2+{judge_policy(config).id}" if config else "officeqa/path-report-v2"
        self._scorer: OfficeQAScorer | None = None
        self._lock = threading.Lock()

    def scorer(self) -> OfficeQAScorer:
        with self._lock:
            if self._scorer is None:
                if not self.config:
                    raise JudgeError("configuration", "OfficeQA scoring needs resources['judge']")
                self._scorer = build_scorer(self.config)
            return self._scorer

    async def score(self, episode, ground_truth, metadata) -> RewardResult:
        return await self.scorer().score(episode, ground_truth, metadata)


def build_scorer(config: dict[str, Any]) -> OfficeQAScorer:
    from rft_cookbook.scoring.judge import OpenAICompatibleJudge, databricks_gateway

    policy = judge_policy(config)
    if config.get("provider", "databricks") == "databricks":
        base_url, headers = databricks_gateway(config.get("profile"))
        judge = OpenAICompatibleJudge(base_url=base_url, headers=headers, concurrency=config.get("concurrency"))
    else:
        judge = OpenAICompatibleJudge(base_url=config["base_url"], concurrency=config.get("concurrency"))
    return OfficeQAScorer(judge, policy, _TemplateTokens(config["tokenizer"], policy.reasoning_effort))


def scorer(judge, policy, count_tokens) -> OfficeQAScorer:
    return OfficeQAScorer(judge, policy, count_tokens)


def run_resources(recipe: dict[str, Any], paths: dict[str, Any]) -> dict[str, Any]:
    """The tool service (named after the run) over the run's corpus, the sandbox and the judge."""
    scoring = recipe.get("scoring") or {}
    out = {"tool_service": paths["run_id"], "data_dir": f"{paths['data_dir']}/corpus",
           "sandbox": scoring.get("sandbox", "strict"), "compute": scoring.get("compute") or {}}
    if scoring.get("judge"):
        out["judge"] = {**scoring["judge"], "tokenizer": paths["judge_tokenizer_dir"]}
    return out


def task(resources: dict[str, Any] | None = None, **budget_overrides) -> AgentTask:
    from rft_cookbook.tools.service import RemoteToolSet

    resources = dict(resources or {})
    budgets = BUDGETS.replace(**budget_overrides)
    if resources.get("tool_service") or not resources.get("data_dir"):
        tools = RemoteToolSet(resources.get("tool_service") or NAME, _executable_schemas())
    else:
        tools = local_tools(resources)
    terminal = next(s for s in tool_schemas() if s["function"]["name"] == TERMINAL_TOOL)
    return AgentTask(name=NAME, version=VERSION, system_prompt=system_prompt(budgets.max_turns), tools=tools,
                     terminal=TerminalRule.terminal_tool(TERMINAL_TOOL), budgets=budgets,
                     scorer=TaskScorer(resources.get("judge")), observation_format=render_observation,
                     metadata={"terminal_schemas": [terminal]})


__all__ = ["NAME", "VERSION", "TERMINAL_TOOL", "SYSTEM_PROMPT", "VERDICT_SCHEMA", "BUDGETS", "render_observation",
           "system_prompt", "tool_schemas", "local_tools", "warm", "judge_policy", "build_scorer", "scorer", "task"]

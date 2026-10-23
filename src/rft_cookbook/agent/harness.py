"""The Chat Completions driver: runs an `AgentTask` against an OpenAI-compatible endpoint.

Used for evaluation against a served checkpoint, and as the agent function behind token-
capturing endpoints (miles' session server). Generation and tool-call parsing happen on the
server; the decisions come from `agent.policy`, so an episode follows the same rules here as
in verl's native loop.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from rft_cookbook.agent.episode import EpisodeRecord
from rft_cookbook.agent.errors import InfraError
from rft_cookbook.agent.policy import EpisodePolicy, ParsedCall, split_reasoning
from rft_cookbook.agent.spec import AgentTask
from rft_cookbook.tools.base import ToolContext


@dataclass
class ChatTurn:
    text: str
    tool_calls: list[ParsedCall] = field(default_factory=list)
    finish_reason: str = "stop"
    reasoning: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                       max_tokens: int) -> ChatTurn: ...


class TokenCounter(Protocol):
    def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
        """Tokens of the next request's prompt, rendered as the server will render it."""


class ByteCounter:
    """Approximate counter for tests and servers without a local tokenizer: one token per
    `bytes_per_token` bytes of JSON. Overestimates for typical text, so budgets stay safe."""

    def __init__(self, bytes_per_token: float = 3.0):
        self.bytes_per_token = bytes_per_token

    def count(self, messages, tools) -> int:
        size = len(json.dumps(messages, ensure_ascii=False).encode()) + len(json.dumps(tools).encode())
        return int(size / self.bytes_per_token) + 1


class TemplateCounter:
    """Exact counter: the model's own chat template, with the serving template arguments."""

    def __init__(self, tokenizer, chat_template_kwargs: dict[str, Any] | None = None):
        self.tokenizer = tokenizer
        self.kwargs = dict(chat_template_kwargs or {})

    def count(self, messages, tools) -> int:
        ids = self.tokenizer.apply_chat_template(template_messages(messages), tools=tools or None, tokenize=True,
                                                 add_generation_prompt=True, **self.kwargs)
        if isinstance(ids, Mapping):   # transformers 5 returns a BatchEncoding, which is not a dict
            ids = ids["input_ids"]
        return len(ids)


_COUNTING = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="token-count")


async def _count(counter: TokenCounter, messages, tools) -> int:
    """Count off the event loop that drives every episode's generation, one count at a time: a
    long conversation takes a while to template and tokenize, and a fast tokenizer must not be
    used from two threads at once."""
    return await asyncio.get_running_loop().run_in_executor(_COUNTING, counter.count, messages, tools)


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages as a server hands them to the chat template: tool-call arguments that are JSON
    objects become mappings (vLLM does this; Qwen and GLM templates iterate over them)."""
    out = []
    for message in messages:
        if message.get("tool_calls"):
            message = {**message, "tool_calls": [_with_mapping_arguments(c) for c in message["tool_calls"]]}
        out.append(message)
    return out


def _with_mapping_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    arguments = tool_call.get("function", {}).get("arguments")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return tool_call
        if isinstance(parsed, dict):
            return {**tool_call, "function": {**tool_call["function"], "arguments": parsed}}
    return tool_call


def initial_messages(task: AgentTask, row: dict[str, Any]) -> list[dict[str, Any]]:
    """System prompt of the task, then the row's own messages (which carry no system prompt)."""
    messages = [dict(m) for m in row["messages"]]
    if any(m.get("role") == "system" for m in messages):
        raise ValueError(f"row {row.get('id')} carries a system message; the task owns the system prompt")
    return [{"role": "system", "content": task.system_prompt}, *messages]


def new_record(task: AgentTask, row: dict[str, Any], *, sample_index: int | None = None, attempt: int = 0,
               episode_id: str | None = None, model: dict[str, Any] | None = None,
               settings: dict[str, Any] | None = None) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=episode_id or uuid.uuid4().hex, task=task.name, task_version=task.version,
        question_id=str(row["id"]), split=row.get("split"), sample_index=sample_index, attempt=attempt,
        model=dict(model or {}), settings=task.settings(**(settings or {})))


async def run_episode(task: AgentTask, client: ChatClient, row: dict[str, Any], *, counter: TokenCounter,
                      sample_index: int | None = None, attempt: int = 0, episode_id: str | None = None,
                      model: dict[str, Any] | None = None, settings: dict[str, Any] | None = None) -> EpisodeRecord:
    """Run one episode to its end and return the finalized (unscored) record."""
    record = new_record(task, row, sample_index=sample_index, attempt=attempt, episode_id=episode_id,
                        model=model, settings=settings)
    policy = EpisodePolicy(task, record)
    tools = task.tool_schemas()
    messages = initial_messages(task, row)
    context = ToolContext(episode_id=record.episode_id, question_id=record.question_id,
                          metadata=dict(row.get("metadata") or {}))
    try:
        if policy.start(await _count(counter, messages, tools)) is None:
            await _drive(task, client, counter, policy, messages, tools, context)
    except InfraError as error:
        policy.fail("infra_error", f"{type(error).__name__}: {error}")
    record.metadata["messages"] = len(messages)
    return policy.finalize()


async def _drive(task, client, counter, policy, messages, tools, context) -> None:
    budgets = task.budgets
    while True:
        max_tokens = policy.next_request()
        if not isinstance(max_tokens, int):
            return
        try:
            turn = await client.complete(messages, tools, max_tokens)
        except Exception as error:  # noqa: BLE001 - any serving failure is infrastructure
            raise InfraError(f"generation failed: {type(error).__name__}: {error}") from error
        before = policy.context_tokens
        generated = turn.completion_tokens if turn.completion_tokens is not None else 0
        after_turn = (turn.prompt_tokens + turn.completion_tokens
                      if turn.prompt_tokens is not None and turn.completion_tokens is not None
                      else before + generated)
        calls = [ParsedCall(c.name, c.arguments, c.call_id or f"call_{len(policy.record.turns)}_{i}", c.parse_error)
                 for i, c in enumerate(turn.tool_calls)]
        text, reasoning = (turn.text, turn.reasoning) if turn.reasoning is not None else split_reasoning(turn.text)
        decision = policy.on_turn(text=text, calls=calls, finish_reason=turn.finish_reason,
                                  generated_tokens=generated, context_tokens=after_turn, reasoning=reasoning)
        # Reasoning goes back to the server: templates render it for the turns of the current
        # episode, as verl's native loop keeps it in the token context.
        assistant: dict[str, Any] = {"role": "assistant", "content": text}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        if decision.calls:
            assistant["tool_calls"] = [{"id": p.record.call_id, "type": "function",
                                        "function": {"name": p.record.name, "arguments": p.record.arguments}}
                                       for p in decision.calls]
        messages.append(assistant)
        if decision.done:
            return
        replies = []
        for planned in decision.calls:
            result = None
            if planned.action == "execute":  # a ToolInfraError propagates: the episode is infra_failure
                result = await task.tools.run(planned.record.name, planned.arguments, context,
                                              timeout_s=budgets.tool_timeout_s, max_bytes=budgets.tool_output_bytes)
            text = policy.observation(planned, result)
            replies.append({"role": "tool", "tool_call_id": planned.record.call_id, "content": text})
        if not policy.admit(decision.calls, await _count(counter, messages + replies, tools)):
            return
        messages.extend(replies)


class OpenAIChatClient:
    """`ChatClient` for an OpenAI-compatible server (vLLM, SGLang, Databricks serving)."""

    def __init__(self, *, base_url: str, model: str, api_key: str = "EMPTY", temperature: float = 1.0,
                 top_p: float = 1.0, top_k: int | None = None, timeout_s: float = 600.0, seed: int | None = None,
                 extra_body: dict[str, Any] | None = None):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=2)
        self.model, self.temperature, self.top_p, self.seed = model, temperature, top_p, seed
        # top_k is not an OpenAI parameter; vLLM reads it from the body. Sending it explicitly
        # keeps a server default (e.g. a model's generation_config.json) from changing the policy.
        self.extra_body = {**({"top_k": top_k} if top_k is not None else {}), **dict(extra_body or {})}

    def with_seed(self, seed: int | None) -> OpenAIChatClient:
        """The same client and settings with another sampling seed (one per episode)."""
        clone = object.__new__(OpenAIChatClient)
        clone.__dict__.update(self.__dict__)
        clone.seed = seed
        return clone

    async def complete(self, messages, tools, max_tokens) -> ChatTurn:
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools or None, max_tokens=max_tokens,
            temperature=self.temperature, top_p=self.top_p, seed=self.seed, extra_body=self.extra_body or None)
        choice = response.choices[0]
        message = choice.message
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        calls = [ParsedCall(name=c.function.name, arguments=c.function.arguments or "", call_id=c.id)
                 for c in (message.tool_calls or [])]
        finish = "length" if choice.finish_reason == "length" else "stop"
        usage = response.usage
        return ChatTurn(text=message.content or "", tool_calls=calls, finish_reason=finish, reasoning=reasoning,
                        prompt_tokens=usage.prompt_tokens if usage else None,
                        completion_tokens=usage.completion_tokens if usage else None)

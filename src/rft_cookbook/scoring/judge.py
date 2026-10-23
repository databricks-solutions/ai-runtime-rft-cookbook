"""LLM judges behind a versioned policy.

A `JudgePolicy` fixes what a judge is allowed to be and do: which models (in order), the
request budget, sampling, timeouts and retries, and whether a fallback model's labels may enter
headline evaluation. A judge that cannot produce a verdict raises `JudgeError`, an
infrastructure error: the episode's score becomes invalid and never a zero reward.

`OpenAICompatibleJudge` calls any OpenAI-compatible endpoint (a vLLM server, Databricks model
serving); `databricks_gateway()` points it at the workspace's AI gateway with the job's ambient
credentials. `FrozenJudge` replays recorded replies, for tests and for re-scoring recorded episodes.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from rft_cookbook.agent.errors import InfraError


@dataclass(frozen=True)
class JudgePolicy:
    id: str
    models: tuple[str, ...]
    max_prompt_tokens: int
    max_output_tokens: int
    temperature: float = 0.0
    timeout_s: float = 90.0
    retries: int = 1
    deadline_s: float | None = None
    """Bound on one verdict across all models and retries."""
    reasoning_effort: str | None = "low"
    response_schema: dict[str, Any] | None = None
    fallback_in_headline: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.models:
            raise ValueError("a judge policy names at least one model")
        if self.max_prompt_tokens <= 0 or self.max_output_tokens <= 0:
            raise ValueError("judge budgets must be positive")


class JudgeError(InfraError):
    def __init__(self, kind: str, detail: str, *, retryable: bool = False):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.retryable = kind, retryable


@dataclass
class JudgeReply:
    content: str | None
    finish_reason: str | None
    model: str
    usage: dict[str, Any] | None = None
    attempts: int = 1


class Judge(Protocol):
    async def complete(self, messages: list[dict[str, str]], policy: JudgePolicy) -> JudgeReply: ...


class OpenAICompatibleJudge:
    """Tries the policy's models in order; each with the policy's retries for retryable errors."""

    def __init__(self, *, base_url: str, api_key: str | None = None,
                 headers: Callable[[], dict[str, str]] | None = None, concurrency: int | None = None):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY", max_retries=0)
        self.headers = headers
        self.concurrency = concurrency
        self._gates: dict[int, asyncio.Semaphore] = {}   # one per event loop

    async def complete(self, messages, policy) -> JudgeReply:
        gate = None
        if self.concurrency:
            gate = self._gates.setdefault(id(asyncio.get_running_loop()), asyncio.Semaphore(self.concurrency))
        try:
            if gate:
                async with gate:
                    return await asyncio.wait_for(self._complete(messages, policy), policy.deadline_s)
            return await asyncio.wait_for(self._complete(messages, policy), policy.deadline_s)
        except asyncio.TimeoutError:
            raise JudgeError("timeout", f"no verdict within the {policy.deadline_s:g} s deadline") from None

    async def aclose(self) -> None:
        """Close the HTTP client on the loop that used it (before `asyncio.run` closes that loop)."""
        await self.client.close()

    async def _complete(self, messages, policy) -> JudgeReply:
        failure: JudgeError | None = None
        attempts = 0
        for model in policy.models:
            for attempt in range(policy.retries + 1):
                attempts += 1
                try:
                    return await asyncio.wait_for(self._once(model, messages, policy, attempts), policy.timeout_s)
                except asyncio.TimeoutError:
                    failure = JudgeError("timeout", f"{model} gave no reply in {policy.timeout_s:g} s", retryable=True)
                except JudgeError as error:
                    failure = error
                except Exception as error:  # noqa: BLE001 - transport errors of the client library
                    status = getattr(error, "status_code", None)
                    failure = JudgeError("transport", f"{model}: {type(error).__name__}: {str(error)[:200]}",
                                         retryable=status is None or status == 429 or status >= 500)
                if not failure.retryable:
                    break
                if attempt < policy.retries:
                    await asyncio.sleep(min(30, 2 ** (attempt + 1)))
        raise failure or JudgeError("configuration", "no judge model was tried")

    async def _once(self, model, messages, policy, attempts) -> JudgeReply:
        body: dict[str, Any] = {}
        if policy.reasoning_effort:
            body["reasoning_effort"] = policy.reasoning_effort
        if policy.response_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": policy.extra.get("schema_name", "verdict"), "schema": policy.response_schema, "strict": True}}
        headers = await asyncio.to_thread(self.headers) if self.headers else None   # may refresh a token
        response = await self.client.chat.completions.create(
            model=model, messages=messages, temperature=policy.temperature, max_tokens=policy.max_output_tokens,
            extra_body=body or None, extra_headers=headers)
        choice = response.choices[0]
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        return JudgeReply(content=choice.message.content, finish_reason=choice.finish_reason, model=model,
                          usage=usage, attempts=attempts)


def databricks_gateway(profile: str | None = None) -> tuple[str, Callable[[], dict[str, str]]]:
    """`(base_url, headers)` for the workspace AI gateway's OpenAI-compatible API.

    Inside an AI Runtime job the SDK finds the job's ambient credentials; locally, pass a profile.
    `headers()` refreshes them when needed. Credentials go only to this workspace's host.
    """
    try:
        from databricks.sdk.core import Config

        config = Config(**({"profile": profile} if profile else {}))
        host = config.host.rstrip("/")
    except Exception as error:   # SDK messages can include credentials: keep only the type
        raise JudgeError("configuration", f"workspace authentication failed ({type(error).__name__})") from None

    def headers() -> dict[str, str]:
        try:
            return config.authenticate()
        except Exception as error:  # noqa: BLE001
            raise JudgeError("transport", f"workspace authentication failed ({type(error).__name__})",
                             retryable=True) from None

    return f"{host}/ai-gateway/mlflow/v1", headers


class FrozenJudge:
    """Replays recorded replies in order; records the requests it was given."""

    def __init__(self, replies: list[JudgeReply | Exception]):
        self.replies = list(replies)
        self.requests: list[list[dict[str, str]]] = []

    async def complete(self, messages, policy) -> JudgeReply:
        self.requests.append(messages)
        if not self.replies:
            raise JudgeError("transport", "no recorded reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

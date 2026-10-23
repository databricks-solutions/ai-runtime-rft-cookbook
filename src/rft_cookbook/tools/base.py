"""Tools: an OpenAI function schema plus `call(arguments, context) -> ToolResult`.

`ToolSet.run` is the one place a call is executed: it enforces the timeout and the output
cap, and turns tool exceptions into tool errors the model sees. A tool that cannot reach its
own infrastructure raises `ToolInfraError`; that ends the episode as an infrastructure failure
instead of teaching the model that the tool is broken.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from rft_cookbook.agent.errors import InfraError

TRUNCATION_MARK = "...(truncated)"


class ToolInfraError(InfraError):
    """The tool's backing service failed; not the model's fault."""


@dataclass
class ToolContext:
    episode_id: str
    question_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    """The data row's metadata, for tools that depend on the question (e.g. a per-question corpus)."""


@dataclass
class ToolResult:
    text: str
    status: str = "ok"
    """`ok`, `error` (the tool reported a problem to the model) or `timeout`."""
    truncated: bool = False
    latency_s: float | None = None
    sandbox: str | None = None
    detail: str | None = None


class Tool(Protocol):
    name: str
    schema: dict[str, Any]

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult: ...


class FunctionTool:
    """A tool from a plain function `fn(**arguments) -> str | ToolResult` (sync or async).

    Sync functions run in the default executor so a slow tool cannot stall the event loop that
    also drives generation.
    """

    def __init__(self, fn: Callable[..., Any], schema: dict[str, Any]):
        function = schema.get("function", {})
        if schema.get("type") != "function" or not function.get("name"):
            raise ValueError("schema must be an OpenAI function schema with a name")
        self.fn = fn
        self.schema = schema
        self.name = function["name"]

    async def call(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if inspect.iscoroutinefunction(self.fn):
            result = await self.fn(**arguments)
        else:
            result = await asyncio.get_running_loop().run_in_executor(None, lambda: self.fn(**arguments))
        return result if isinstance(result, ToolResult) else ToolResult(text=str(result))


def function_schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


class ToolSet:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool {tool.name!r}")
            self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    async def run(self, name: str, arguments: dict[str, Any], context: ToolContext, *,
                  timeout_s: float, max_bytes: int) -> ToolResult:
        """Execute one call. The caller has already validated the name and arguments."""
        tool = self._tools[name]
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(tool.call(arguments, context), timeout=timeout_s)
        except asyncio.TimeoutError:
            result = ToolResult(text=f"Error: tool '{name}' timed out after {timeout_s:g} s", status="timeout")
        except ToolInfraError:
            raise
        except Exception as error:  # noqa: BLE001 - a tool's own failure is reported to the model
            result = ToolResult(text=f"Error: tool '{name}' failed: {type(error).__name__}: {error}", status="error")
        result.latency_s = time.monotonic() - started
        text, truncated = clip_utf8(result.text, max_bytes)
        result.text, result.truncated = text, result.truncated or truncated
        return result


def clip_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    """Clip to at most `max_bytes` UTF-8 bytes, marker included, never splitting a character."""
    data = text.encode()
    if len(data) <= max_bytes:
        return text, False
    keep = max_bytes - len(TRUNCATION_MARK)  # the mark is ASCII: one byte per character
    if keep <= 0:
        return TRUNCATION_MARK[:max_bytes], True
    return data[:keep].decode(errors="ignore") + TRUNCATION_MARK, True

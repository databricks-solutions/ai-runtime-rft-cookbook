"""A node-local tool service: one process per node holds a task's tools and their heavy state
(corpora, indexes), and every episode on that node calls it over HTTP.

    python -m rft_cookbook.tools.service --task officeqa --resources resources.json --name <run id>

The service loads the task's tools (`local_tools(resources)` in the task module), warms them
(`warm(resources)`, e.g. builds a search index), then writes a discovery file
`<RFT_TOOL_SERVICE_DIR or /tmp/rft-tools>/<name>.json` with its URL. Clients find the service
through the node they run on, by reading that file, so episodes always use their own node's
service. The launcher starts the service before the framework and stops it after.

`RemoteToolSet` has `ToolSet`'s interface. The service applies the timeout and the output cap; a
service that cannot be reached, or a tool that reports an infrastructure failure, raises
`ToolInfraError` in the episode: an infrastructure failure, never a tool error the model sees.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rft_cookbook.tools.base import ToolContext, ToolInfraError, ToolResult, ToolSet


def discovery_path(name: str) -> Path:
    return Path(os.environ.get("RFT_TOOL_SERVICE_DIR", "/tmp/rft-tools")) / f"{name}.json"


class ToolServer:
    """Serves a `ToolSet` on 127.0.0.1. Calls run on one event loop in a background thread."""

    def __init__(self, tools: ToolSet, *, host: str = "127.0.0.1", port: int = 0):
        self.tools = tools
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="tool-service-loop", daemon=True).start()
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.url = f"http://{host}:{self.httpd.server_address[1]}"

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):   # one line per call would drown the job log
                pass

            def _reply(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._reply(200, {"tools": server.tools.names(), "pid": os.getpid()})
                else:
                    self._reply(404, {"error": "not found"})

            def do_POST(self):
                if self.path != "/call":
                    return self._reply(404, {"error": "not found"})
                try:
                    request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                    name = request["name"]
                    if name not in server.tools:
                        return self._reply(400, {"error": f"unknown tool {name!r}"})
                    context = ToolContext(**request.get("context") or {"episode_id": "", "question_id": ""})
                    call = server.tools.run(name, request.get("arguments") or {}, context,
                                            timeout_s=float(request["timeout_s"]), max_bytes=int(request["max_bytes"]))
                    result = asyncio.run_coroutine_threadsafe(call, server.loop).result()
                    self._reply(200, {"result": asdict(result)})
                except ToolInfraError as error:
                    self._reply(503, {"infra": str(error)})
                except (KeyError, TypeError, ValueError) as error:
                    self._reply(400, {"error": f"bad request: {type(error).__name__}: {error}"})

        return Handler

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop)


class RemoteToolSet:
    """`ToolSet`'s interface over a node-local service. `schemas` are the task's own, so rendering
    and data conversion never need a running service."""

    def __init__(self, service: str, schemas: list[dict[str, Any]]):
        self.service = service   # a URL, or the name of a service to discover on this node
        self._schemas = list(schemas)
        self._names = [schema["function"]["name"] for schema in self._schemas]
        self._url: str | None = service if service.startswith("http") else None

    def names(self) -> list[str]:
        return list(self._names)

    def schemas(self) -> list[dict[str, Any]]:
        return list(self._schemas)

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def url(self) -> str:
        if self._url is None:
            path = discovery_path(self.service)
            try:
                self._url = json.loads(path.read_text())["url"]
            except (OSError, ValueError, KeyError) as error:
                raise ToolInfraError(f"no tool service {self.service!r} on this node ({path})") from error
        return self._url

    async def run(self, name: str, arguments: dict[str, Any], context: ToolContext, *,
                  timeout_s: float, max_bytes: int) -> ToolResult:
        body = {"name": name, "arguments": arguments, "timeout_s": timeout_s, "max_bytes": max_bytes,
                "context": {"episode_id": context.episode_id, "question_id": context.question_id,
                            "metadata": context.metadata}}
        reply = await asyncio.to_thread(self._post, body, timeout_s + 30)
        return ToolResult(**reply["result"])

    def _post(self, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        request = urllib.request.Request(self.url() + "/call", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            raise ToolInfraError(f"tool service: HTTP {error.code}: {detail}") from error
        except (OSError, ValueError) as error:   # refused, reset, timed out, unreadable reply
            raise ToolInfraError(f"tool service unreachable: {type(error).__name__}: {error}") from error


def start(task: str, resources: dict[str, Any], name: str, *, timeout_s: float = 1800) -> subprocess.Popen:
    """Start the service for `task` as a child process; return once it is ready."""
    path = discovery_path(name)
    path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = path.with_suffix(".resources.json")
    spec.write_text(json.dumps(resources))
    process = subprocess.Popen([sys.executable, "-m", "rft_cookbook.tools.service", "--task", task,
                                "--resources", str(spec), "--name", name])
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"tool service {name!r} exited with {process.returncode} before it was ready")
        if time.monotonic() > deadline:
            process.kill()
            raise TimeoutError(f"tool service {name!r} not ready after {timeout_s:g} s")
        time.sleep(1)
    print(f"[tools] service {name!r} ready at {json.loads(path.read_text())['url']}", flush=True)
    return process


def stop(process: subprocess.Popen, name: str) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    discovery_path(name).unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--resources", type=Path, required=True, help="JSON file with the task's resources")
    parser.add_argument("--name", required=True, help="discovery name (the run id)")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    module = importlib.import_module(f"rft_cookbook.tasks.{args.task.replace('-', '_')}")
    resources = json.loads(args.resources.read_text())
    started = time.monotonic()
    tools = module.local_tools(resources)
    if hasattr(module, "warm"):
        module.warm(resources)
    server = ToolServer(tools, port=args.port)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    path = discovery_path(args.name)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps({"url": server.url, "pid": os.getpid(), "tools": tools.names()}))
    partial.replace(path)
    print(f"[tool-service] {args.task}: {len(tools.names())} tools at {server.url} "
          f"(ready in {time.monotonic() - started:.0f} s)", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

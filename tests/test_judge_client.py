"""The judge client (OpenAICompatibleJudge) against a fake OpenAI-compatible server: model order,
retries of retryable errors only, per-attempt timeout, the verdict deadline, the concurrency limit,
credential refresh and the request it sends."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rft_cookbook.scoring import judge as judge_module
from rft_cookbook.scoring.judge import JudgeError, JudgePolicy, OpenAICompatibleJudge

pytest.importorskip("openai")


class FakeServer:
    """`/v1/chat/completions` that plays a script per model: an HTTP status, a delay, or a reply."""

    def __init__(self, script: dict[str, list]):
        self.script = {model: list(steps) for model, steps in script.items()}
        self.requests: list[dict] = []
        self.in_flight = self.most_in_flight = 0
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with server.lock:
                    server.requests.append({"body": body, "authorization": self.headers.get("Authorization")})
                    server.in_flight += 1
                    server.most_in_flight = max(server.most_in_flight, server.in_flight)
                    steps = server.script[body["model"]]
                    step = steps.pop(0) if len(steps) > 1 else steps[0]
                try:
                    kind, value = step
                    if kind == "delay":
                        time.sleep(value)
                        kind, value = "reply", "late"
                    if kind == "status":
                        self.send_response(value)
                        payload = {"error": {"message": f"status {value}"}}
                    else:
                        self.send_response(200)
                        payload = {"id": "c", "object": "chat.completion", "created": 0, "model": body["model"],
                                   "choices": [{"index": 0, "finish_reason": "stop",
                                                "message": {"role": "assistant", "content": value}}],
                                   "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
                    data = json.dumps(payload).encode()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                finally:
                    with server.lock:
                        server.in_flight -= 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def models(self) -> list[str]:
        return [r["body"]["model"] for r in self.requests]

    def close(self):
        self.httpd.shutdown()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    real = asyncio.sleep
    monkeypatch.setattr(judge_module.asyncio, "sleep", lambda seconds, *a, **k: real(0))


def policy(**changes) -> JudgePolicy:
    values = dict(id="p", models=("judge-a", "judge-b"), max_prompt_tokens=100, max_output_tokens=50, retries=2,
                  timeout_s=5.0, deadline_s=30.0, reasoning_effort="low",
                  response_schema={"type": "object", "properties": {}}, extra={"schema_name": "support"})
    values.update(changes)
    return JudgePolicy(**values)


def ask(server: FakeServer, judge_policy: JudgePolicy, **client):
    judge = OpenAICompatibleJudge(base_url=server.url, api_key="k", **client)

    async def go():
        try:
            return await judge.complete([{"role": "user", "content": "q"}], judge_policy)
        finally:
            await judge.aclose()
    return asyncio.run(go())


def test_a_retryable_error_is_retried_on_the_same_model():
    for status in (503, 429):
        server = FakeServer({"judge-a": [("status", status), ("reply", "ok")], "judge-b": [("reply", "b")]})
        reply = ask(server, policy())
        assert (reply.model, reply.content, reply.attempts) == ("judge-a", "ok", 2)
        assert server.models() == ["judge-a"] * 2
        server.close()


def test_a_client_error_moves_to_the_next_model_without_a_retry():
    server = FakeServer({"judge-a": [("status", 400)], "judge-b": [("reply", "b")]})
    reply = ask(server, policy())
    assert (reply.model, reply.attempts) == ("judge-b", 2) and server.models() == ["judge-a", "judge-b"]
    server.close()


def test_an_attempt_that_takes_too_long_is_retried_then_falls_back():
    server = FakeServer({"judge-a": [("delay", 1.0)], "judge-b": [("reply", "b")]})
    reply = ask(server, policy(timeout_s=0.2, retries=1))
    assert reply.model == "judge-b" and server.models() == ["judge-a", "judge-a", "judge-b"]
    server.close()


def test_no_verdict_within_the_deadline_is_a_judge_error():
    server = FakeServer({"judge-a": [("delay", 1.0)], "judge-b": [("delay", 1.0)]})
    with pytest.raises(JudgeError, match="deadline"):
        ask(server, policy(timeout_s=5.0, deadline_s=0.3))
    server.close()


def test_when_every_model_fails_the_last_failure_is_raised():
    server = FakeServer({"judge-a": [("status", 500)], "judge-b": [("status", 404)]})
    with pytest.raises(JudgeError, match="judge-b") as error:
        ask(server, policy(retries=1))
    assert error.value.kind == "transport" and not error.value.retryable
    assert server.models() == ["judge-a", "judge-a", "judge-b"]   # 500 retried once, 404 not at all
    server.close()


def test_the_request_carries_the_policy_and_fresh_credentials():
    server = FakeServer({"judge-a": [("status", 503), ("reply", "ok")], "judge-b": [("reply", "b")]})
    tokens = iter(["t1", "t2", "t3"])
    ask(server, policy(), headers=lambda: {"Authorization": f"Bearer {next(tokens)}"})
    assert [r["authorization"] for r in server.requests] == ["Bearer t1", "Bearer t2"]   # refreshed per attempt
    body = server.requests[0]["body"]
    assert body["max_tokens"] == 50 and body["temperature"] == 0.0 and body["reasoning_effort"] == "low"
    schema = body["response_format"]
    assert schema["type"] == "json_schema" and schema["json_schema"]["name"] == "support"


def test_the_concurrency_limit_holds_across_concurrent_verdicts():
    server = FakeServer({"judge-a": [("delay", 0.2)], "judge-b": [("reply", "b")]})
    judge = OpenAICompatibleJudge(base_url=server.url, api_key="k", concurrency=2)

    async def go():
        question = [{"role": "user", "content": "q"}]
        replies = await asyncio.gather(*(judge.complete(question, policy()) for _ in range(6)))
        await judge.aclose()
        return replies
    replies = asyncio.run(go())
    assert len(replies) == 6 and server.most_in_flight == 2
    server.close()

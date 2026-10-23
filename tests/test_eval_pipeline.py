"""The evaluation pipeline without a GPU: every planned episode runs, is scored and is accounted for."""
from __future__ import annotations

import asyncio
from pathlib import Path

from rft_cookbook.agent.harness import ByteCounter, ChatTurn
from rft_cookbook.eval.run import VllmServer, run_plan
from rft_cookbook.evidence.report import summarize
from rft_cookbook.evidence.sink import EpisodeSink, read_episodes
from rft_cookbook.tasks import load_task


class Answering:
    """Answers \\boxed{4} to every question; fails the request for the question 'down'."""

    async def complete(self, messages, tools, max_tokens):
        if messages[-1]["content"] == "down":
            raise ConnectionError("replica restarted")
        return ChatTurn(text="\\boxed{4}", prompt_tokens=300, completion_tokens=5)


def test_every_planned_episode_is_scored_or_accounted_for(tmp_path):
    task = load_task("math")
    rows = [{"id": q, "split": "eval", "messages": [{"role": "user", "content": q}], "ground_truth": gold,
             "metadata": {}} for q, gold in (("2+2", "4"), ("3+3", "6"), ("down", "1"))]
    sink = EpisodeSink(tmp_path, prefix="eval")
    asyncio.run(run_plan(task, Answering(), ByteCounter(), rows, samples=2, concurrency=4, sink=sink,
                         model_identity={"id": "toy"}, settings={"driver": "test"}))
    sink.flush()
    records, damaged = read_episodes(tmp_path)
    plan = {"ids": [r["id"] for r in rows], "samples": 2}
    head = summarize(records, plan, damaged=damaged)["headline"]
    assert head["planned"] == head["present"] == 6 and head["valid"] == 4 and head["infra_failures"] == 2
    assert head["score_mean_valid"] == 0.5 and head["score_mean_strict"] == 2 / 6 and not head["complete"]


def test_the_server_command_carries_the_parsers_and_the_context():
    server = VllmServer("/models/m", max_model_len=5120, tool_parser="qwen3_coder", reasoning_parser="qwen3", gpus=8,
                        gpu_memory_utilization=0.85, seed=42, extra_args=["--language-model-only"])
    command = " ".join(server.command)
    assert "--tensor-parallel-size 1" in command and "--disable-custom-all-reduce" in command
    assert "--generation-config vllm" in command   # a model's sampling defaults never apply silently
    tp = " ".join(VllmServer("/m", max_model_len=8, tool_parser="p", reasoning_parser=None, gpus=8,
                             gpu_memory_utilization=0.8, seed=0, tensor_parallel=8).command)
    assert "--tensor-parallel-size 8 --data-parallel-size 1" in tp
    for flag in ("--tool-call-parser qwen3_coder", "--reasoning-parser qwen3", "--max-model-len 5120",
                 "--data-parallel-size 8", "--enable-auto-tool-choice", "--served-model-name policy",
                 "--language-model-only"):
        assert flag in command


def test_each_sample_of_a_question_has_its_own_seed(tmp_path):
    class Seeded(Answering):
        def __init__(self, seed=None, log=None):
            self.seed, self.log = seed, log if log is not None else []

        def with_seed(self, seed):
            return Seeded(seed, self.log)

        async def complete(self, messages, tools, max_tokens):
            self.log.append((messages[-1]["content"], self.seed))
            return await super().complete(messages, tools, max_tokens)

    rows = [{"id": q, "split": "eval", "messages": [{"role": "user", "content": q}], "ground_truth": "4",
             "metadata": {}} for q in ("2+2", "1+3")]
    client = Seeded()
    asyncio.run(run_plan(load_task("math"), client, ByteCounter(), rows, samples=2, concurrency=4,
                         sink=EpisodeSink(tmp_path, prefix="eval"), model_identity={}, settings={}, seed=42))
    assert sorted(client.log) == [("1+3", 44), ("1+3", 45), ("2+2", 42), ("2+2", 43)]


def test_a_model_is_served_from_a_complete_local_copy(tmp_path):
    from rft_cookbook.eval.run import local_copy

    model = tmp_path / "volume" / "model"
    (model / "sub").mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "sub" / "weights.safetensors").write_bytes(b"w" * 1000)
    (model / ".cache").mkdir()
    (model / ".cache" / "lock").write_text("x")
    served = Path(local_copy(str(model), str(tmp_path / "local")))
    assert (served / "sub" / "weights.safetensors").stat().st_size == 1000 and not (served / ".cache").exists()
    (model / "config.json").write_text('{"changed": true}')
    assert local_copy(str(model), str(tmp_path / "local")) == str(served)   # a finished copy is reused
    assert (served / "config.json").read_text() == "{}"


def test_the_client_states_top_k_so_no_server_default_applies():
    import pytest

    pytest.importorskip("openai")
    from rft_cookbook.agent.harness import OpenAIChatClient

    client = OpenAIChatClient(base_url="http://127.0.0.1:9/v1", model="policy", top_k=-1, seed=1)
    assert client.extra_body == {"top_k": -1} and client.with_seed(2).extra_body == {"top_k": -1}

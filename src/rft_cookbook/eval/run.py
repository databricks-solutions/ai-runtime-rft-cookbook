"""Evaluate a model on a task: serve it with vLLM, run every planned episode through the Chat
Completions harness, score each record, and write the evidence.

    <out>/plan.json       question ids x samples, written before any episode runs
    <out>/episodes/       one EpisodeRecord per planned episode (chunked JSONL)
    <out>/summary.json    rft_cookbook.evidence.report.summarize over the records

The same harness and decision rules run in training (verl's native loop calls the same
policy), and the same scorer scores both.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from rft_cookbook.agent.harness import OpenAIChatClient, TemplateCounter, run_episode
from rft_cookbook.data import file_sha256, read_rows
from rft_cookbook.evidence.report import summarize
from rft_cookbook.evidence.sink import EpisodeSink, read_episodes
from rft_cookbook.scoring.base import score_episode
from rft_cookbook.tasks import load_task


class VllmServer:
    """`vllm serve` in a child process over the visible GPUs: tensor-parallel groups, data-parallel across them."""

    def __init__(self, model_path: str, *, max_model_len: int, tool_parser: str, reasoning_parser: str | None,
                 gpus: int, gpu_memory_utilization: float, seed: int, port: int = 8000, log: Path | None = None,
                 extra_args: list[str] = (), tensor_parallel: int = 1):
        if gpus % tensor_parallel:
            raise ValueError(f"{gpus} GPUs do not split into tensor-parallel groups of {tensor_parallel}")
        self.port, self.log = port, log
        script = Path(sys.executable).with_name("vllm")   # the console script beside this interpreter
        launcher = [str(script)] if script.exists() else [sys.executable, "-m", "vllm.entrypoints.cli.main"]
        self.command = [*launcher, "serve", model_path,
                        "--served-model-name", "policy", "--port", str(port), "--max-model-len", str(max_model_len),
                        "--gpu-memory-utilization", str(gpu_memory_utilization),
                        "--tensor-parallel-size", str(tensor_parallel), "--data-parallel-size", str(gpus // tensor_parallel),
                        "--seed", str(seed), "--enable-auto-tool-choice", "--tool-call-parser", tool_parser,
                        # vLLM's custom all-reduce crashes CUDA-graph capture on H100 with TP <= 8
                        "--disable-custom-all-reduce", "--trust-remote-code", "--enable-prefix-caching",
                        # sampling is exactly what each request says: never the model's own
                        # generation_config.json defaults (Qwen3.5's top_k=20 would apply silently)
                        "--generation-config", "vllm"]
        if reasoning_parser:
            self.command += ["--reasoning-parser", reasoning_parser]
        self.command += list(extra_args)
        self.process = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        handle = open(self.log, "w") if self.log else None
        self.process = subprocess.Popen(self.command, stdout=handle, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                print(self.tail(), file=sys.stderr)
                raise RuntimeError(f"vllm serve exited with {self.process.returncode}; see {self.log}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/models", timeout=5) as response:
                    if response.status == 200:
                        return self
            except OSError:
                time.sleep(5)
        raise TimeoutError("vllm serve did not become ready in 30 minutes")

    def tail(self, lines: int = 60) -> str:
        if not self.log or not Path(self.log).exists():
            return ""
        return "\n".join(Path(self.log).read_text(errors="replace").splitlines()[-lines:])

    def __exit__(self, *exc):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()


def local_copy(source: str, cache_root: str) -> str:
    """A complete copy of a model directory on local disk, to serve from.

    Serving memory-maps the weights; over a volume's FUSE mount those reads are slow for a large
    model. Files are copied in parallel and their sizes checked; a finished copy is marked and
    reused; a copy is per source path, so another model never reuses it.
    """
    src = Path(source)
    dest = Path(cache_root) / hashlib.sha256(str(src).encode()).hexdigest()[:16]
    done = dest / ".rft_complete"
    if done.exists():
        return str(dest)
    files = [f for f in src.rglob("*") if f.is_file() and not any(part.startswith(".") for part in f.relative_to(src).parts)]
    started = time.monotonic()

    def copy(f: Path) -> None:
        target = dest / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
        if target.stat().st_size != f.stat().st_size:
            raise OSError(f"incomplete copy of {f}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy, files))
    done.write_text(str(src))
    size = sum(f.stat().st_size for f in files) / 1e9
    print(f"[eval] copied {len(files)} files ({size:.1f} GB) to {dest} in {time.monotonic() - started:.0f} s", flush=True)
    return str(dest)


def visible_gpus() -> int:
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        return len([d for d in devices.split(",") if d.strip()])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return max(1, len([line for line in out.splitlines() if line.startswith("GPU ")]))
    except (OSError, subprocess.CalledProcessError):
        return 1


async def run_plan(task, client, counter, rows: list[dict[str, Any]], *, samples: int, concurrency: int,
                   sink: EpisodeSink, model_identity: dict[str, Any], settings: dict[str, Any],
                   seed: int | None = None) -> None:
    """Every (row, sample) once. With a seed, episode `i * samples + k` samples with `seed + i * samples + k`,
    so the samples of one question are independent and a rerun reproduces them."""
    gate = asyncio.Semaphore(concurrency)
    total, done, started = len(rows) * samples, [], time.monotonic()

    async def one(index, row, sample):
        episode_client = client
        if seed is not None and hasattr(client, "with_seed"):
            episode_client = client.with_seed(seed + index * samples + sample)
        async with gate:
            record = await run_episode(task, episode_client, row, counter=counter, sample_index=sample,
                                       episode_id=f"{row['id']}_{sample}", model=model_identity, settings=settings)
            if record.status == "ok":   # a failed episode is never scored
                await score_episode(task.scorer, record, row["ground_truth"], row.get("metadata"))
            sink.write(record)
        done.append(record.score.value if record.score and record.score.valid else None)
        if len(done) % 10 == 0 or len(done) == total:
            scored = [v for v in done if v is not None]
            print(f"[eval] {len(done)}/{total} episodes in {time.monotonic() - started:.0f} s; "
                  f"{len(scored)} scored, mean {sum(scored) / max(1, len(scored)):.3f}", flush=True)

    await asyncio.gather(*(one(i, row, k) for i, row in enumerate(rows) for k in range(samples)))


def _prepare_output(out: Path, plan: dict[str, Any]) -> None:
    """Retry an incomplete evaluation without mixing attempts; preserve the earlier evidence."""
    previous_plan, summary = out / "plan.json", out / "summary.json"
    if summary.exists() and json.loads(summary.read_text()).get("headline", {}).get("complete") is not False:
        raise SystemExit(f"{out} already holds an evaluation; use a new label")
    chunks = list((out / "episodes").glob("*.jsonl"))
    if previous_plan.exists():
        if json.loads(previous_plan.read_text()) != plan:
            raise SystemExit(f"{out} holds a different evaluation plan; use a new label")
        archive = out / "attempts" / str(time.time_ns())
        archive.mkdir(parents=True)
        # Move files individually: the volume supports file rename, not every directory operation.
        for path in (previous_plan, summary, out / "vllm.log", *chunks):
            if path.exists():
                target = archive / path.relative_to(out)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
        print(f"[eval] preserved the incomplete attempt in {archive}", flush=True)
    elif chunks or summary.exists():
        raise SystemExit(f"{out} has evaluation evidence without its plan; use a new label")
    out.mkdir(parents=True, exist_ok=True)
    previous_plan.write_text(json.dumps(plan, indent=1, sort_keys=True))


def evaluate(*, task_name: str, budgets: dict[str, Any], rows: str, model_path: str, out: str,
             model_identity: dict[str, Any], tool_parser: str, reasoning_parser: str | None,
             chat_template_kwargs: dict[str, Any], samples: int, temperature: float, top_p: float,
             concurrency: int, gpu_memory_utilization: float, max_prompt_tokens: int, seed: int,
             gpus: int | None = None, serve_args: list[str] = (), tensor_parallel: int = 1,
             resources: dict[str, Any] | None = None, request_timeout_s: float = 600.0,
             local_cache: str | None = None, top_k: int = -1, schema_view: str = "full") -> dict[str, Any]:
    from transformers import AutoTokenizer

    out_dir = Path(out)
    task = load_task(task_name, resources, **budgets).with_schema_view(schema_view)
    data = list(read_rows(Path(rows)))
    plan = {"task": task.name, "task_version": task.version, "rows": rows, "rows_sha256": file_sha256(Path(rows)),
            "samples": samples, "ids": [r["id"] for r in data], "model": model_identity,
            "sampling": {"temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed,
                         "seeds": "seed + index * samples + k"},
            "serving": {"engine": "vllm", "tool_parser": tool_parser, "reasoning_parser": reasoning_parser,
                        "chat_template_kwargs": chat_template_kwargs, "args": list(serve_args),
                        "tensor_parallel": tensor_parallel},
            "schema_view": schema_view, "settings": task.settings()}
    _prepare_output(out_dir, plan)
    max_model_len = max_prompt_tokens + task.budgets.max_response_tokens
    sink = EpisodeSink(out_dir / "episodes", prefix="eval", flush_every=64)
    # The server log stays on local disk while it is open (a volume publishes a file only once it
    # is closed) and is copied next to the evidence afterwards.
    local_log = Path(tempfile.mkdtemp(prefix="rft-eval-")) / "vllm.log"
    serve_path = local_copy(model_path, local_cache) if local_cache else model_path
    server = VllmServer(serve_path, max_model_len=max_model_len, tool_parser=tool_parser,
                        reasoning_parser=reasoning_parser, gpus=gpus or visible_gpus(),
                        gpu_memory_utilization=gpu_memory_utilization, seed=seed, log=local_log,
                        extra_args=serve_args, tensor_parallel=tensor_parallel)
    try:
        with server:
            client = OpenAIChatClient(base_url=server.base_url, model="policy", temperature=temperature, top_p=top_p,
                                      top_k=top_k, seed=seed, timeout_s=request_timeout_s,
                                      extra_body={"chat_template_kwargs": chat_template_kwargs}
                                      if chat_template_kwargs else None)
            counter = TemplateCounter(AutoTokenizer.from_pretrained(model_path), chat_template_kwargs)
            settings = {"driver": "chat_completions", "parser": tool_parser, "reasoning_parser": reasoning_parser}
            asyncio.run(run_plan(task, client, counter, data, samples=samples, concurrency=concurrency, sink=sink,
                                 model_identity=model_identity, settings=settings, seed=seed))
    finally:
        sink.flush()
        if local_log.exists():
            shutil.copyfile(local_log, out_dir / "vllm.log")
    records, damaged = read_episodes(out_dir / "episodes")
    summary = summarize(records, plan, damaged=damaged)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary["headline"], indent=1))
    return summary

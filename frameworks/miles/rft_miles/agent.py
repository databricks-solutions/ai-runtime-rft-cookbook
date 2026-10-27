"""The miles agent function: one cookbook episode against miles' session server.

    --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path rft_miles.agent.run --use-session-server v2

miles' session server records every model call (token ids, log-probabilities) and builds the
training samples from them. This function only drives the episode: the cookbook's Chat
Completions harness (rft_cookbook.agent.harness) against the session's OpenAI-compatible
endpoint, under the same decision rules as every other driver, then the task's scorer. It writes
the `EpisodeRecord` to the run's episodes and returns the score, which miles merges into the
samples' metadata for the reward hook (rft_miles.reward). It runs inline
(`--custom-agent-function-mode inline`), so the task, tokenizer and episode sink stay in the
rollout process across episodes. miles' default runs every call in a fresh child process, which
would load them again for every episode and exit before the sink writes its buffered records. The
harness awaits every model and tool call, so no episode holds the shared event loop for long.

Each prompt row carries its cookbook row (`ROW_KEY`) and the path of the run's agent settings
(`CONFIG_KEY`, written by the train stage): the agent function sees only a row's metadata.

An excluded episode (an infrastructure failure, an invalid score, no turn at all) returns no
reward, under the same rules as verl's loop (rft_cookbook.agent.episode.eligibility): miles then
drops the whole group and samples another prompt, as verl does, so an excluded episode never
reaches the update and never biases its group's baseline. An episode the function cannot run at
all (it raises: a setup or driver error) is excluded too, since miles then finds no score
(rft_miles.reward). After `max_consecutive_exclusions` excluded episodes without a decided one
between them (rft_cookbook.agent.guard; a judge outage first pauses for `judge_outage_wait_s`) the
function asks the launcher to stop the run (`abort_file`).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from rft_cookbook.agent.episode import eligibility
from rft_cookbook.agent.guard import ExclusionGuard
from rft_cookbook.agent.harness import OpenAIChatClient, TemplateCounter, run_episode
from rft_cookbook.air.stages import request_abort
from rft_cookbook.evidence.sink import shared_sink
from rft_cookbook.tasks import load_task

ROW_KEY = "rft_row"
CONFIG_KEY = "rft_config"
REWARD_KEY = "rft_reward"
ATTESTATION_DIR_KEY = "rft_attestation_dir"
_GUARD: list[ExclusionGuard] = []   # one per process: every episode runs in it (inline)


@lru_cache(maxsize=4)
def _config(config_path: str) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text())


@lru_cache(maxsize=4)
def _setup(config_path: str):
    from transformers import AutoTokenizer

    config = _config(config_path)
    task = load_task(config["task"], config.get("resources"), **config["budgets"])
    counter = TemplateCounter(AutoTokenizer.from_pretrained(config["tokenizer"]), config.get("chat_template_kwargs") or {})
    return config, task, counter, shared_sink(config["episode_dir"], prefix="train", flush_after_s=30.0)


async def run(base_url: str, prompt: Any, request_kwargs: dict[str, Any], metadata: dict[str, Any],
              **kwargs) -> dict[str, Any]:
    try:
        return await _episode(base_url, request_kwargs, metadata)
    except Exception as error:   # no episode at all: excluded like any other, so a broken setup stops the run
        _count(_config(metadata[CONFIG_KEY]), None, True, f"{type(error).__name__}: {error}")
        raise


async def _episode(base_url: str, request_kwargs: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    config, task, counter, sink = _setup(metadata[CONFIG_KEY])
    row = metadata[ROW_KEY]
    # every sampling field is sent, so no server default (a model's generation_config.json) applies
    top_k = request_kwargs.get("top_k")
    client = OpenAIChatClient(base_url=f"{base_url}/v1", model=config["served_model"],
                              temperature=float(request_kwargs.get("temperature", 1.0)),
                              top_p=float(request_kwargs.get("top_p", 1.0)), top_k=-1 if top_k is None else int(top_k),
                              seed=request_kwargs.get("seed"), timeout_s=float(config["request_timeout_s"]))
    record = await run_episode(task, client, row, counter=counter, model=config["model"],
                               settings={"driver": "miles", "parser": config["tool_parser"]})
    if record.status == "ok":   # a failed episode is never scored
        await _guard(config).score(task.scorer, record, row["ground_truth"], row.get("metadata"))
    if sink:
        sink.write(record)
    # the session server records every token it served; whether collection keeps the terminal
    # turn whole is attested per sample, after this returns (rft_miles.attest)
    verdict, reason = eligibility(record, True)
    if verdict != "excluded" and not record.turns:   # no model call: the session holds no sample
        verdict, reason = "excluded", "no_turn"
    episode = {"rft_episode_id": record.episode_id, "rft_status": record.status,
               "rft_answered": record.answer is not None,
               "rft_terminal_turn": record.termination.terminal_turn if record.termination else None,
               ATTESTATION_DIR_KEY: f"{config['episode_dir']}/attestations" if config.get("episode_dir") else None}
    if verdict == "excluded":
        _count(config, record, True, f"{record.status}: {reason}")
        return {REWARD_KEY: None, "rft_scored": False, "rft_excluded": reason, **episode}
    _count(config, record, False)
    return {REWARD_KEY: float(record.score.value), "rft_scored": True, **episode}


def _guard(config: dict[str, Any]) -> ExclusionGuard:
    if not _GUARD:
        _GUARD.append(ExclusionGuard(int(config.get("max_consecutive_exclusions") or 0),
                                     wait_s=float(config.get("judge_outage_wait_s", 1800))))
    return _GUARD[0]


def _count(config: dict[str, Any], record, excluded: bool, reason: str = "") -> None:
    stop = _guard(config).record(record, excluded, reason)
    if stop:
        request_abort(config.get("abort_file"), stop)

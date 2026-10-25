"""verl's native tool agent loop, driven by the cookbook's shared episode policy.

Registered in the agent-loop config the launcher resolves (rft_verl.render):

    - name: rft_agent
      _target_: rft_verl.agent_loop.CookbookAgentLoop
      task: math
      budgets: {max_turns: 4}
      episode_dir: /path/for/episode/records

verl keeps generation, tokenization (continuous tokens) and the token/mask/log-prob arrays.
The loop hands each turn's parsed calls and token counts to `EpisodePolicy`, runs the task's
tools, and scores the finished `EpisodeRecord` itself, so training and evaluation score the same
record with the same scorer. The reward is set on the output, which bypasses verl's reward loop.

An excluded episode (infrastructure failure, invalid score, truncated terminal turn) raises.
verl's V1 trainer marks the prompt group failed, drops the whole group and refills it, so an
excluded sample never reaches the update and never biases its group's baseline. verl refills
forever, so a broken dependency (a judge outage, a loop that cannot be built) would spin without
training: after `max_consecutive_exclusions` episodes in one process that failed without a decided
one between them (rft_cookbook.agent.guard; a judge outage first pauses for
`judge_outage_wait_s`), the loop writes `abort_file`, and the launcher stops the run
(rft_verl.launch).
"""
from __future__ import annotations

import json
import logging
import uuid
from functools import lru_cache
from typing import Any

from omegaconf import OmegaConf
from verl.experimental.agent_loop.tool_agent_loop import SPEC_DECODE_EXTRA_KEYS, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.profiler import simple_timer

from rft_cookbook.agent.episode import TERMINATIONS, Attestation, eligibility, token_mask_sha256
from rft_cookbook.agent.errors import EpisodeExcluded, InfraError
from rft_cookbook.agent.guard import ExclusionGuard
from rft_cookbook.agent.harness import new_record
from rft_cookbook.agent.policy import EpisodePolicy, ParsedCall, split_reasoning
from rft_cookbook.agent.spec import AgentTask
from rft_cookbook.air.stages import request_abort
from rft_cookbook.evidence.sink import shared_sink
from rft_cookbook.tasks import load_task
from rft_cookbook.tools.base import ToolContext

logger = logging.getLogger(__name__)

AGENT_NAME = "rft_agent"
_GUARD: list[ExclusionGuard] = []   # one per process: agent loops are created per episode


@lru_cache(maxsize=8)
def _task(name: str, budgets_json: str, resources_json: str = "null") -> AgentTask:
    """One task per process (agent loops are created per episode): its tools client and judge are shared."""
    return load_task(name, json.loads(resources_json), **json.loads(budgets_json))



def row_ground_truth(kwargs: dict) -> Any:
    """The row's ground truth as the task wrote it: `extra_info.ground_truth_json` (rft_verl.data), else
    verl's `reward_model.ground_truth` text (parquet written before the JSON copy was kept)."""
    extra = kwargs.get("extra_info") or {}
    if extra.get("ground_truth_json") is not None:
        return json.loads(extra["ground_truth_json"])
    return (kwargs.get("reward_model") or {}).get("ground_truth")

class _Episode:
    def __init__(self, task: AgentTask, policy: EpisodePolicy, context: ToolContext, sample_id: str):
        self.task, self.policy, self.context, self.sample_id = task, policy, context, sample_id
        self.decision = None
        self.agent_data = None


def _guard(limit: int, wait_s: float) -> ExclusionGuard:
    if not _GUARD:
        _GUARD.append(ExclusionGuard(limit, wait_s=wait_s))
    return _GUARD[0]


def _count(abort_file: str | None, guard: ExclusionGuard, record, excluded: bool, reason: str = "") -> None:
    stop = guard.record(record, excluded, reason)
    if stop and abort_file:
        logger.error("rft: %s; asking the launcher to stop", stop)
        request_abort(abort_file, f"in one agent-loop process, {stop}")


class CookbookAgentLoop(ToolAgentLoop):
    def __init__(self, *args, task: str, budgets: dict[str, Any] | None = None, episode_dir: str | None = None,
                 model: dict[str, Any] | None = None, resources: dict[str, Any] | None = None,
                 abort_file: str | None = None, max_consecutive_exclusions: int = 32,
                 judge_outage_wait_s: float = 1800.0, **kwargs):
        self.abort_file = abort_file
        self.guard = _guard(int(max_consecutive_exclusions), float(judge_outage_wait_s))
        try:
            self._setup(args, kwargs, task, budgets, episode_dir, model, resources)
        except Exception as error:
            _count(self.abort_file, self.guard, None, True, f"building the loop: {type(error).__name__}: {error}")
            raise

    def _setup(self, args, kwargs, task, budgets, episode_dir, model, resources) -> None:
        super().__init__(*args, **kwargs)
        # verl instantiates the loop from the registry with Hydra: nested values arrive as DictConfig
        budgets, resources, model = (OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
                                     for v in (budgets, resources, model))
        budgets = dict(budgets or {})
        base = _task(task, json.dumps(budgets, sort_keys=True), json.dumps(resources, sort_keys=True))
        if base.budgets.max_response_tokens > self.response_length:
            raise ValueError(f"task budget max_response_tokens={base.budgets.max_response_tokens} exceeds "
                             f"verl's response_length={self.response_length}; resolve them together")
        self.task = base
        self.tool_schemas = base.tool_schemas()
        self.parser_schemas = [OpenAIFunctionToolSchema.model_validate(s) for s in self.tool_schemas]
        self.max_parallel_calls = base.budgets.max_tool_calls_per_turn
        self.sink = shared_sink(episode_dir, prefix="train", flush_after_s=30.0)
        self.model = dict(model or {})
        self._episode: _Episode | None = None

    # ---- episode -----------------------------------------------------------------------

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        try:
            return await self._run(sampling_params, **kwargs)
        except EpisodeExcluded:
            raise   # counted where it was decided
        except Exception as error:
            _count(self.abort_file, self.guard, None, True, f"{type(error).__name__}: {error}")
            raise

    async def _run(self, sampling_params: dict[str, Any], **kwargs):
        extra = dict(kwargs.get("extra_info") or {})
        uid, session = kwargs.get("uid"), kwargs.get("session_id")
        row = {"id": extra.get("id", uid), "split": extra.get("split")}
        sample_id = f"{uid}_{session}"
        record = new_record(self.task, row, sample_index=int(session) if session is not None else None,
                            episode_id=f"{sample_id}_{uuid.uuid4().hex[:8]}", model=self.model,
                            settings={"driver": "verl", "parser": self.tool_parser_name})
        record.metadata["group"] = str(uid)
        metadata = json.loads(extra.get("metadata_json") or "{}")
        context = ToolContext(episode_id=record.episode_id, question_id=record.question_id, metadata=metadata)
        episode = self._episode = _Episode(self.task, EpisodePolicy(self.task, record), context, sample_id)
        output, excluded = None, None
        try:
            output = await super().run(sampling_params, **kwargs)
        except InfraError as error:
            episode.policy.fail("infra_error", f"{type(error).__name__}: {error}")
        except EpisodeExcluded as error:   # decided before the first turn: recorded and counted below
            excluded = str(error)
        record = episode.policy.finalize()
        ground_truth = row_ground_truth(kwargs)
        if record.status == "ok":   # a failed episode is never scored
            await self.guard.score(self.task.scorer, record, ground_truth, metadata)
        record.collection = [self._attest(record, episode.agent_data)]
        if self.sink:
            self.sink.write(record)
        attestation = record.collection[0]
        if attestation.eligibility == "excluded" or output is None:
            _count(self.abort_file, self.guard, record, True, excluded or f"{record.status}: {attestation.reason}")
            raise EpisodeExcluded(excluded or attestation.reason or "no output", record)
        _count(self.abort_file, self.guard, record, False)
        output.reward_score = record.score.value
        output.extra_fields["reward_extra_info"] = self._metrics(record)
        output.extra_fields["rft_episode_id"] = record.episode_id
        return output

    def _attest(self, record, agent_data) -> Attestation:
        if agent_data is None or not agent_data.response_mask:
            return Attestation(framework="verl", sample_id=self._episode.sample_id, terminal_turn=None,
                               terminal_span_complete=False, tokens=0, loss_tokens=0,
                               token_mask_sha256=token_mask_sha256([], []), logprob_coverage=None,
                               eligibility="excluded", reason=record.status if record.status != "ok" else "no_response")
        mask = agent_data.response_mask
        complete = len(mask) <= self.response_length
        verdict, reason = eligibility(record, complete)
        logprobs = agent_data.response_logprobs
        return Attestation(
            framework="verl", sample_id=self._episode.sample_id,
            terminal_turn=record.termination.terminal_turn if record.termination else None,
            terminal_span_complete=complete, tokens=len(agent_data.prompt_ids), loss_tokens=int(sum(mask)),
            token_mask_sha256=token_mask_sha256(agent_data.prompt_ids, mask),
            logprob_coverage=min(1.0, len(logprobs) / len(mask)) if logprobs else None,
            eligibility=verdict, reason=reason)

    def _metrics(self, record) -> dict[str, float]:
        """Same keys for every sample: verl stacks these per key across the batch."""
        metrics = {"rft/score": float(record.score.value), "rft/turns": float(len(record.turns)),
                   "rft/tool_calls": float(sum(c.executed for c in record.calls())),
                   "rft/answered": float(record.answer is not None)}
        for name in getattr(self.task.scorer, "components", ()):
            metrics[f"rft/{name}"] = float(record.score.components.get(name, 0.0))
        for reason in TERMINATIONS:
            metrics[f"rft/end_{reason}"] = float(record.termination.reason == reason)
        return metrics

    # ---- state machine -----------------------------------------------------------------

    async def _handle_pending_state(self, agent_data, sampling_params):
        episode = self._episode
        episode.agent_data = agent_data
        system = next((m.get("content") for m in agent_data.messages if m.get("role") == "system"), None)
        if system != episode.task.system_prompt:
            raise ValueError("the row's system prompt is not the task's: convert the data with this task version")
        state = await super()._handle_pending_state(agent_data, sampling_params)
        if episode.policy.start(len(agent_data.prompt_ids)) is not None:
            raise EpisodeExcluded("the prompt leaves no room for a turn")
        return state

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        policy = self._episode.policy
        max_tokens = policy.next_request()
        if not isinstance(max_tokens, int):
            return AgentState.TERMINATED
        params = {**sampling_params, "max_tokens": max_tokens}
        if self.tool_parser.stop_token_ids:
            params["stop_token_ids"] = sorted(set((params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
        with simple_timer("generate_sequences", agent_data.metrics):
            try:
                output = await self.server_manager.generate(
                    request_id=agent_data.request_id, prompt_ids=agent_data.prompt_ids, sampling_params=params,
                    image_data=agent_data.image_data, video_data=agent_data.video_data,
                    mm_processor_output=agent_data.mm_processor_output, audio_data=agent_data.audio_data,
                    mm_processor_kwargs=agent_data.mm_processor_kwargs)
            except Exception as error:  # noqa: BLE001 - a failed generation is infrastructure
                raise InfraError(f"generation failed: {type(error).__name__}: {error}") from error
        if output.stop_reason in ("aborted", "abort"):
            raise InfraError("generation was aborted and not resumed")
        self._merge_output_fields(agent_data, output)
        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        merged, mask, logprobs = await self.ct_merge_assistant_token(
            agent_data.prompt_ids, output.token_ids, agent_data.response_mask,
            agent_data.response_logprobs if (agent_data.response_logprobs or output.log_probs) else None,
            assistant_logprobs=output.log_probs if output.log_probs else None)
        agent_data.prompt_ids, agent_data.response_mask = merged.token_ids, mask
        if logprobs is not None:
            agent_data.response_logprobs = logprobs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        content, parsed = await self.tool_parser.extract_tool_calls(output.token_ids, self.parser_schemas)
        text, reasoning = split_reasoning(self._strip_end_tokens(content or ""))
        calls = [ParsedCall(name=c.name, arguments=c.arguments, call_id=c.tool_call_id or f"call_{agent_data.assistant_turns}_{i}")
                 for i, c in enumerate(parsed)]
        finish = "length" if output.stop_reason == "length" or len(output.token_ids) >= max_tokens else "stop"
        decision = policy.on_turn(text=text, calls=calls, finish_reason=finish, generated_tokens=len(output.token_ids),
                                  context_tokens=len(agent_data.prompt_ids), reasoning=reasoning,
                                  policy_version=output.extra_fields.get("max_global_steps"))
        agent_data.tool_calls = [FunctionCall(name=p.record.name, arguments=p.record.arguments, tool_call_id=p.record.call_id)
                                 for p in decision.calls]
        agent_data.messages.append(self._build_assistant_message(content, agent_data))
        if decision.done:
            return AgentState.TERMINATED
        self._episode.decision = decision
        return AgentState.PROCESSING_TOOLS

    async def _handle_processing_tools_state(self, agent_data):
        episode = self._episode
        decision, budgets = episode.decision, episode.task.budgets
        previous = list(agent_data.messages)
        replies = []
        with simple_timer("tool_calls", agent_data.metrics):
            for planned in decision.calls:
                result = None
                if planned.action == "execute":
                    result = await episode.task.tools.run(planned.record.name, planned.arguments, episode.context,
                                                          timeout_s=budgets.tool_timeout_s,
                                                          max_bytes=budgets.tool_output_bytes)
                text = episode.policy.observation(planned, result)
                replies.append({"role": "tool", "content": text, "tool_call_id": planned.record.call_id})
        agent_data.messages.extend(replies)
        merged, mask, logprobs = await self.ct_merge_context_msg(
            previous, agent_data.messages, agent_data.prompt_ids, agent_data.response_mask,
            agent_data.response_logprobs if agent_data.response_logprobs else None, tools=self.tool_schemas)
        if not episode.policy.admit(decision.calls, len(merged.token_ids)):
            del agent_data.messages[len(previous):]
            return AgentState.TERMINATED
        agent_data.prompt_ids, agent_data.response_mask = merged.token_ids, mask
        if agent_data.response_logprobs:
            agent_data.response_logprobs = logprobs or []
        agent_data.user_turns += 1
        return AgentState.GENERATING

    def _strip_end_tokens(self, text: str) -> str:
        """verl's parsers decode with special tokens; the server's text has no trailing end-of-turn."""
        specials = [t for t in getattr(self.tokenizer, "all_special_tokens", []) if t]
        stripped = True
        while stripped:
            stripped = False
            text = text.rstrip()
            for token in specials:
                if text.endswith(token):
                    text, stripped = text[: -len(token)], True
        return text

    @staticmethod
    def _merge_output_fields(agent_data, output) -> None:
        """verl's own bookkeeping of preemption, policy version and speculative decoding stats."""
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted or 0
        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
            return
        if output.extra_fields.get("max_global_steps") is not None:
            agent_data.extra_fields["max_global_steps"] = output.extra_fields["max_global_steps"]
        for key in SPEC_DECODE_EXTRA_KEYS:
            if key in output.extra_fields and key in agent_data.extra_fields:
                agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(output.extra_fields[key])

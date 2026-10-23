# Derived from verl-on-air (Apache License 2.0), usecases/officeqa/reward.py and protocol.py at 495b583;
# modified for ai-runtime-rft-cookbook: scores an EpisodeRecord through a judge policy instead of
# a verl reward function; an unresolved judge makes the score invalid instead of aborting the run.
# See THIRD_PARTY_NOTICES.md.
"""OfficeQA's binary answer-and-evidence score.

1. The terminal `submit_report` arguments must parse as a report (answer + path of steps that
   cite observation ids). An explicit "DATA NOT AVAILABLE" is an abstention.
2. Every cited observation must exist and have been delivered, and every value a step claims
   must appear in what was delivered (fabricated-value gate). A compute step whose input arrived
   after the compute call is counted (`late_inputs`), not rejected, as in the measured reward.
3. The answer is checked against gold (scale, sign and list length included).
4. Only a correct answer with a valid path goes to the support judge, which sees the question,
   the report and the cited evidence intact. Score 1 only if the judge says supported.

Everything up to the judge is deterministic. A judge failure or unparsable verdict raises
`JudgeError`, so the score is invalid (infrastructure), never a zero.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import re
from typing import Any, Callable

from rft_cookbook.agent.episode import EpisodeRecord
from rft_cookbook.scoring.base import RewardResult
from rft_cookbook.scoring.judge import Judge, JudgeError, JudgePolicy
from rft_cookbook.tasks.officeqa import answer_check, path_report

OBSERVATION_PREFIX = "[observation {oid}]\n"
COMPONENTS = ("answer_correct", "report_valid", "abstention", "reference_valid", "value_valid", "judge_called",
              "judge_supported", "judge_path_score", "late_inputs", "observations", "delivered_observations",
              "path_steps", "report_over_budget")
VERDICT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "path_status": {"type": "string", "enum": ["supported", "unsupported", "unknown"]},
        "path_score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["path_status", "path_score", "issues"],
}


def render_observation(observation_id: str, tool: str, text: str) -> str:
    """What the model sees: the runtime-issued id on the first line, then the tool's text."""
    return OBSERVATION_PREFIX.format(oid=observation_id) + text


def _tool_text(call) -> str | None:
    if call.delivered_text is None:
        return None
    prefix = OBSERVATION_PREFIX.format(oid=call.observation_id)
    return call.delivered_text[len(prefix):] if call.delivered_text.startswith(prefix) else call.delivered_text


def outcome(text: str) -> str:
    if not text.strip():
        return "empty"
    stripped = text.lstrip()
    if stripped.startswith(("Error:", "Error executing", "Unknown function", "Invalid JSON")):
        return "error"
    if stripped.startswith("No ") and re.search(r"found|match|document", stripped[:80]):
        return "empty"
    return "ok"


def ledger_record(episode: EpisodeRecord, *, question: str, requirements: str = "") -> dict[str, Any]:
    """The capture record path_report reads, built from the runtime-owned episode record."""
    observations = []
    for turn in episode.turns:
        for call in turn.tool_calls:
            if call.observation_id is None:
                continue
            try:
                arguments = path_report.loads_strict(call.arguments)
            except (TypeError, ValueError):
                arguments = {}
            text = _tool_text(call)
            observations.append({
                "observation_id": call.observation_id, "order": int(call.observation_id.split("_")[-1]),
                "tool": call.name, "args": arguments if isinstance(arguments, dict) else {},
                "outcome": outcome(text) if text is not None else ("error" if call.status != "ok" else "ok"),
                "generated_by_request": turn.index, "delivered_to_request": call.delivered_to_turn,
                "delivered_text": text})
    terminal_seen = episode.termination is not None and episode.termination.reason == "answer"
    return {"episode_id": episode.episode_id, "question": question, "question_requirements": requirements,
            "observations": observations, "terminal_seen": terminal_seen,
            "terminal_text": _report_text(episode.answer) if terminal_seen else ""}


def _report_text(arguments: str | None) -> str:
    try:
        report = path_report.loads_strict(arguments or "")
        if not isinstance(report, dict):
            return ""
        if isinstance(report.get("path"), str):
            report = {**report, "path": path_report.loads_strict(report["path"])}
        return json.dumps(report, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return ""


def parse_verdict(content, finish_reason) -> dict:
    if finish_reason != "stop" or not isinstance(content, str):
        raise JudgeError("verdict", f"missing complete final-content verdict (finish={finish_reason})")
    try:
        obj = path_report.loads_strict(content)
    except (TypeError, ValueError) as error:
        raise JudgeError("verdict", str(error)) from error
    if not isinstance(obj, dict) or set(obj) != {"path_status", "path_score", "issues"}:
        raise JudgeError("verdict", "unexpected verdict fields")
    score = obj["path_score"]
    if isinstance(score, bool) or (score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score))):
        raise JudgeError("verdict", "non-finite/non-numeric path score")
    if obj["path_status"] != "unknown" and score is None:
        raise JudgeError("verdict", "resolved verdict requires a numeric path score")
    if not isinstance(obj["issues"], list) or any(not isinstance(issue, str) for issue in obj["issues"]):
        raise JudgeError("verdict", "issues must be strings")
    verdict = path_report.parse_support_verdict(content)
    if verdict["status"] != "ok":
        raise JudgeError("verdict", verdict.get("reason", "unresolved support verdict"))
    return verdict


class ReportBudgetExceeded(ValueError):
    """The report cites more evidence than the judge can accept intact."""


def support_messages(record, report, ledger, *, budget: int, count: Callable[[list[dict]], int]):
    """The judge request: cited evidence first and intact; uncited history only while it fits."""
    request = path_report.build_support_request(record["question"], record.get("question_requirements", ""),
                                                report, ledger)
    rows = sorted(ledger.observations.values(), key=lambda obs: obs.order)
    cited = {step.observation for step in report.steps}

    def row(obs):
        return {"observation_id": obs.observation_id, "tool": obs.tool, "args": obs.args, "outcome": obs.outcome,
                "delivered": obs.delivered, "delivered_text": obs.delivered_text or ""}

    history = [row(obs) for obs in rows if obs.observation_id in cited]
    request["tool_history"] = history

    def rendered():
        system, user = path_report.render_support_prompt(request)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    messages = rendered()
    tokens = count(messages)
    if tokens > budget:
        raise ReportBudgetExceeded(f"all cited evidence needs {tokens} prompt tokens; budget={budget}")
    for obs in rows:
        if obs.observation_id in cited:
            continue
        history.append(row(obs))
        candidate = rendered()
        candidate_tokens = count(candidate)
        if candidate_tokens > budget:
            history.pop()
        else:
            messages, tokens = candidate, candidate_tokens
    receipt = {"prompt_tokens": tokens, "prompt_budget": budget, "cited_observations": len(cited),
               "omitted_observations": len(rows) - len(history),
               "request_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()}
    return messages, receipt


class OfficeQAScorer:
    components = COMPONENTS

    def __init__(self, judge: Judge, policy: JudgePolicy, count_tokens: Callable[[list[dict]], int]):
        self.judge, self.judge_policy, self.count_tokens = judge, policy, count_tokens
        self.policy = f"officeqa/path-report-v1+{policy.id}"

    async def score(self, episode: EpisodeRecord, ground_truth: Any, metadata: dict[str, Any]) -> RewardResult:
        question = metadata.get("question")
        if not question:
            raise ValueError("OfficeQA rows carry the question text in metadata['question']")
        record = ledger_record(episode, question=question, requirements=metadata.get("question_requirements", ""))
        return await self.score_ledger(record, ground_truth)

    async def score_ledger(self, record: dict[str, Any], ground_truth: Any) -> RewardResult:
        """Score an evidence record (`ledger_record`'s form, which verl-on-air's recorded episodes
        also use), so recorded episodes of another driver can be re-scored the same way."""
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            raise ValueError("missing gold answer")
        c = dict.fromkeys(COMPONENTS, 0.0)
        ledger = path_report.EpisodeLedger.from_record(record)
        c.update(observations=float(len(ledger.observations)),
                 delivered_observations=float(sum(obs.delivered for obs in ledger.observations.values())))
        report = path_report.parse_report(record["terminal_text"])
        c.update(report_valid=float(report.ok), abstention=float(report.is_abstention), path_steps=float(len(report.steps)))
        refs = path_report.check_references(report, ledger)
        c["late_inputs"] = float(sum(issue.code == "impossible_chronology" for issue in refs.issues))
        kept = [issue for issue in refs.issues if issue.code != "impossible_chronology"]
        refs = path_report.RefCheck(not kept, kept)
        # Lookup values are checked even where the report declares dependencies.
        steps = [dataclasses.replace(step, depends_on=[]) for step in report.steps]
        values = path_report.check_claimed_values(dataclasses.replace(report, steps=steps), ledger)
        c.update(reference_valid=float(report.ok and refs.valid), value_valid=float(report.ok and values.valid))
        correct, why = False, "no answer"
        if report.ok:
            correct, why = answer_check.check(report.answer, ground_truth, tolerance=0.0)
            c["answer_correct"] = float(correct)
        if report.is_abstention:
            return RewardResult(True, 0.0, c, detail="abstention")
        if not report.ok:
            return RewardResult(True, 0.0, c, detail="malformed: " + "; ".join(report.errors))
        if not refs.valid:
            return RewardResult(True, 0.0, c, detail="invalid_path: " + ", ".join(refs.codes))
        if not values.valid:
            return RewardResult(True, 0.0, c, detail="fabricated: " + ", ".join(values.codes))
        if not correct:
            return RewardResult(True, 0.0, c, detail=f"wrong: {why}")
        budget = self.judge_policy.max_prompt_tokens
        try:
            messages, receipt = await asyncio.to_thread(support_messages, record, report, ledger, budget=budget,
                                                        count=self.count_tokens)
        except ReportBudgetExceeded as error:
            c["report_over_budget"] = 1.0
            return RewardResult(True, 0.0, c, detail=f"over_budget: {error}")
        c["judge_called"] = 1.0
        reply = await self.judge.complete(messages, self.judge_policy)
        verdict = parse_verdict(reply.content, reply.finish_reason)
        supported = verdict["path_status"] == path_report.SUPPORTED
        c.update(judge_supported=float(supported), judge_path_score=float(verdict["path_score"]))
        judge = {"policy": self.judge_policy.id, "model": reply.model, "attempts": reply.attempts,
                 "fallback": reply.model != self.judge_policy.models[0], "verdict": verdict["path_status"],
                 "issues": verdict.get("issues", []), "request": receipt, "usage": reply.usage}
        return RewardResult(True, float(supported), c, judge=judge, detail="correct" if supported else "unsupported")

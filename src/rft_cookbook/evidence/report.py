"""Evidence shared by every task: completeness, accounting, leakage and paired comparison.

`summarize` checks an evaluation against its frozen plan: every planned (question, sample) is
present exactly once and scored validly. Missing, invalid and failed episodes (an infrastructure
failure, or a driver that stopped without a decision) are counted, never silently dropped, and
leave the evaluation incomplete; the strict mean counts them as zero. So does a label from a
fallback judge whose policy keeps such labels out of the headline. Results are also reported per
split (e.g. held-out questions and training probes). `compare_runs` pairs two evaluations of the
same plan by question: bootstrap intervals resample questions (several samples of one question
are not independent), and McNemar's exact test is reported only with one binary sample per
question. Pairing uses the valid samples only, so two evaluations are comparable only when both
are complete. `parity` pairs an evaluation with recorded per-question results of an earlier one,
the same way.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rft_cookbook.agent.episode import EpisodeRecord
from rft_cookbook.data import read_rows
from rft_cookbook.evidence.sink import read_episodes


def summarize(records: list[EpisodeRecord], plan: dict[str, Any], *, damaged: int = 0) -> dict[str, Any]:
    planned = {(qid, k) for qid in plan["ids"] for k in range(plan["samples"])}
    seen: Counter = Counter((r.question_id, r.sample_index) for r in records)
    by_key = {(r.question_id, r.sample_index): r for r in records}
    missing = sorted(planned - set(seen))
    unplanned = sorted(set(seen) - planned)
    duplicates = sorted(k for k, n in seen.items() if n > 1)
    scored = [r for k, r in by_key.items() if k in planned and r.status == "ok" and r.score and r.score.valid]
    valid = [r for r in scored if (r.score.judge or {}).get("headline", True)]
    withheld = [r for r in scored if not (r.score.judge or {}).get("headline", True)]   # a fallback judge's label
    infra = [r for k, r in by_key.items() if k in planned and r.status == "infra_failure"]
    driver = [r for k, r in by_key.items() if k in planned and r.status == "policy_failure"]
    invalid = [r for k, r in by_key.items() if k in planned and r.status == "ok" and not (r.score and r.score.valid)]
    per_question: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        per_question[r.question_id].append(r.score.value)
    components = sorted({c for r in valid for c in r.score.components})
    by_question: dict[str, list[EpisodeRecord]] = defaultdict(list)
    by_split: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for r in valid:
        by_question[r.question_id].append(r)
        by_split[r.split].append(r)
    strict_total = sum(r.score.value for r in valid)
    headline = {
        "planned": len(planned), "present": len(planned & set(seen)), "valid": len(valid),
        "missing": len(missing), "infra_failures": len(infra), "policy_failures": len(driver),
        "invalid_scores": len(invalid), "fallback_withheld": len(withheld),
        "fallback_labels": sum(bool((r.score.judge or {}).get("fallback")) for r in scored),
        "duplicates": len(duplicates), "unplanned": len(unplanned), "damaged_lines": damaged,
        "complete": not (missing or infra or driver or invalid or withheld or duplicates or unplanned or damaged),
        "score_mean_valid": strict_total / len(valid) if valid else None,
        "score_mean_strict": strict_total / len(planned) if planned else None,
    }
    return {
        "headline": headline,
        "terminations": dict(Counter(r.termination.reason for r in by_key.values() if r.termination)),
        "mean_turns": _mean([len(r.turns) for r in valid]),
        "mean_tool_calls": _mean([sum(c.executed for c in r.calls()) for r in valid]),
        "sandbox_levels": dict(Counter(c.sandbox for r in valid for c in r.calls() if c.sandbox)),
        "per_question": {q: _mean(v) for q, v in sorted(per_question.items())},
        "per_question_components": {q: {c: _mean([r.score.components.get(c, 0.0) for r in rs]) for c in components}
                                    for q, rs in sorted(by_question.items())},
        "question_split": {q: rs[0].split for q, rs in sorted(by_question.items())},
        "splits": {name: _split_summary(rs, components) for name, rs in sorted(by_split.items())},
        "missing": [list(k) for k in missing[:100]],
        "plan": {k: plan[k] for k in ("task", "task_version", "rows_sha256", "samples", "model") if k in plan},
    }


def _split_summary(records: list[EpisodeRecord], components: list[str]) -> dict[str, Any]:
    return {"questions": len({r.question_id for r in records}), "valid": len(records),
            "score_mean": _mean([r.score.value for r in records]),
            "answered": _mean([float(r.termination is not None and r.termination.reason == "answer") for r in records]),
            "components": {c: _mean([r.score.components.get(c, 0.0) for r in records]) for c in components},
            "terminations": dict(Counter(r.termination.reason for r in records if r.termination))}


def compare_runs(baseline_dir: str, trained_dir: str, *, train_rows: str | None = None, seed: int = 0,
                 resamples: int = 10000, probe_splits: tuple[str, ...] = ()) -> dict[str, Any]:
    plans = [json.loads((Path(d) / "plan.json").read_text()) for d in (baseline_dir, trained_dir)]
    mismatched = [k for k in ("task", "task_version", "rows_sha256", "samples", "sampling", "serving", "settings")
                  if plans[0].get(k) != plans[1].get(k)]
    summaries = []
    for directory, plan in zip((baseline_dir, trained_dir), plans):
        records, damaged = read_episodes(Path(directory) / "episodes")
        summaries.append(summarize(records, plan, damaged=damaged))
    base, trained = (s["per_question"] for s in summaries)
    paired = sorted(set(base) & set(trained))
    diffs = [trained[q] - base[q] for q in paired]
    incomplete = [name for name, s in zip(("baseline", "trained"), summaries) if not s["headline"]["complete"]]
    report = {
        "summary": {
            "comparable": not (mismatched or incomplete), "mismatched_settings": mismatched, "incomplete": incomplete,
            "baseline": summaries[0]["headline"], "trained": summaries[1]["headline"],
            "paired_questions": len(paired), "mean_difference": _mean(diffs),
            "bootstrap_95": bootstrap_ci(diffs, seed=seed, resamples=resamples),
        },
        "terminations": {"baseline": summaries[0]["terminations"], "trained": summaries[1]["terminations"]},
    }
    if plans[0]["samples"] == 1:
        report["summary"]["mcnemar_exact_p"] = mcnemar_exact(base, trained, paired)
    splits = summaries[0]["question_split"]
    if len(set(splits.values())) > 1:
        report["splits"] = {}
        for name in sorted(set(splits.values())):
            questions = [q for q in paired if splits.get(q) == name]
            split_diffs = [trained[q] - base[q] for q in questions]
            report["splits"][name] = {"paired_questions": len(questions), "baseline": _mean([base[q] for q in questions]),
                                      "trained": _mean([trained[q] for q in questions]),
                                      "mean_difference": _mean(split_diffs),
                                      "bootstrap_95": bootstrap_ci(split_diffs, seed=seed, resamples=resamples)}
    if train_rows:
        report["summary"]["leakage"] = leakage(train_rows, plans[0]["rows"], exempt_splits=probe_splits)
    return report


def parity(eval_dir: str, reference: dict[str, Any], *, metrics: dict[str, str], max_abs_difference: float | None = None,
           seed: int = 0, resamples: int = 10000) -> dict[str, Any]:
    """An evaluation against recorded per-question results of an earlier one, per split.

    `reference["per_question"]` maps question id to `{"split": ..., <reference metric>: value}`;
    `metrics` maps each reference metric to this evaluation's measure: `score` (the episode score)
    or a score component. A metric agrees when every reference question was evaluated, the paired
    question-bootstrap 95% interval of the difference contains zero, and, if given, the mean
    difference is at most `max_abs_difference`.
    """
    plan = json.loads((Path(eval_dir) / "plan.json").read_text())
    records, damaged = read_episodes(Path(eval_dir) / "episodes")
    summary = summarize(records, plan, damaged=damaged)
    ours = {q: {"score": summary["per_question"][q], **summary["per_question_components"][q]}
            for q in summary["per_question"]}
    rows = reference["per_question"]
    results, agree = {}, summary["headline"]["complete"]
    for split in sorted({row["split"] for row in rows.values()}):
        questions = sorted(q for q, row in rows.items() if row["split"] == split)
        missing = [q for q in questions if q not in ours]
        present = [q for q in questions if q in ours]
        for ref_metric, our_metric in metrics.items():
            diffs = [ours[q][our_metric] - rows[q][ref_metric] for q in present]
            interval = bootstrap_ci(diffs, seed=seed, resamples=resamples)
            difference = _mean(diffs)
            ok = (not missing and interval is not None and interval[0] <= 0.0 <= interval[1]
                  and (max_abs_difference is None or abs(difference) <= max_abs_difference))
            results[f"{split}/{ref_metric}"] = {
                "questions": len(questions), "missing": missing, "reference": _mean([rows[q][ref_metric] for q in present]),
                "ours": _mean([ours[q][our_metric] for q in present]), "mean_difference": difference,
                "bootstrap_95": interval, "agree": ok}
            agree = agree and ok
    return {"agree": agree, "complete": summary["headline"]["complete"], "headline": summary["headline"],
            "reference": {k: v for k, v in reference.items() if k != "per_question"},
            "max_abs_difference": max_abs_difference, "metrics": results}


def bootstrap_ci(diffs: list[float], *, seed: int = 0, resamples: int = 10000) -> list[float] | None:
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]]


def mcnemar_exact(base: dict[str, float], trained: dict[str, float], paired: list[str]) -> float | None:
    """Two-sided exact McNemar test on binary per-question outcomes."""
    if any(base[q] not in (0.0, 1.0) or trained[q] not in (0.0, 1.0) for q in paired):
        return None
    gained = sum(1 for q in paired if trained[q] > base[q])
    lost = sum(1 for q in paired if trained[q] < base[q])
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(gained, lost) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def leakage(train_rows: str, eval_rows: str, *, exempt_splits: tuple[str, ...] = ()) -> dict[str, Any]:
    """Training and evaluation must not share ids or question text. Evaluation rows of an
    `exempt_splits` split (training probes, by design training questions) are counted, not checked."""
    def index(path, skip=()):
        ids, texts, skipped = set(), set(), 0
        for row in read_rows(Path(path)):
            if row.get("split") in skip:
                skipped += 1
                continue
            ids.add(row["id"])
            texts.add(hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest())
        return ids, texts, skipped

    train_ids, train_texts, _ = index(train_rows)
    eval_ids, eval_texts, probes = index(eval_rows, exempt_splits)
    return {"shared_ids": len(train_ids & eval_ids), "shared_questions": len(train_texts & eval_texts),
            "exempt_probe_rows": probes, "clean": not (train_ids & eval_ids or train_texts & eval_texts)}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None

"""Recipe resolution; evidence completeness, comparison and leakage; the episode sink."""
from __future__ import annotations

import json

import pytest
import yaml
from support import ScriptedClient, call, make_task

from rft_cookbook.agent.episode import EpisodeRecord, Score, Termination
from rft_cookbook.data import write_rows
from rft_cookbook.evidence.report import bootstrap_ci, compare_runs, mcnemar_exact, parity, summarize
from rft_cookbook.evidence.sink import EpisodeSink, read_episodes
from rft_cookbook.recipe import RecipeError, load_recipe, recipes_root


def write(root, name, data):
    path = root / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def test_extends_merges_mappings_and_replaces_lists(tmp_path):
    write(tmp_path, "base", {"framework": "verl", "train": {"lr": 1, "logger": ["console", "mlflow"]},
                             "stages": ["prep", "train"]})
    write(tmp_path, "child", {"extends": "base", "task": "math", "train": {"logger": ["console"]}})
    recipe = load_recipe("child", root=tmp_path, overrides=["train.lr=0.5", "compute.nodes=2"])
    assert recipe["train"] == {"lr": 0.5, "logger": ["console"]}
    assert recipe["stages"] == ["prep", "train"] and recipe["compute"] == {"nodes": 2}
    again = load_recipe("child", root=tmp_path, overrides=["train.lr=0.5", "compute.nodes=2"])
    assert again["resolved_sha256"] == recipe["resolved_sha256"]
    assert load_recipe("child", root=tmp_path)["resolved_sha256"] != recipe["resolved_sha256"]


def test_cycles_unknown_keys_and_incomplete_recipes_are_errors(tmp_path):
    write(tmp_path, "a", {"extends": "b", "task": "t", "framework": "f"})
    write(tmp_path, "b", {"extends": "a"})
    with pytest.raises(RecipeError, match="cycle"):
        load_recipe("a", root=tmp_path)
    write(tmp_path, "typo", {"task": "t", "framework": "f", "trian": {}})
    with pytest.raises(RecipeError, match="unknown keys"):
        load_recipe("typo", root=tmp_path)
    write(tmp_path, "half", {"framework": "f"})
    with pytest.raises(RecipeError, match="without 'task'"):
        load_recipe("half", root=tmp_path)
    write(tmp_path, "ok", {"task": "t", "framework": "f"})
    with pytest.raises(RecipeError, match="unknown top-level"):
        load_recipe("ok", root=tmp_path, overrides=["nope.x=1"])


def test_the_shipped_recipes_resolve():
    recipe = load_recipe("math/verl/qwen3.5-2b_h100x8", root=recipes_root())
    assert recipe["framework"] == "verl" and recipe["task"] == "math"
    assert recipe["train"]["algorithm"] == "grpo" and recipe["compute"]["trainer_gpus"] == 4


def record(qid, sample, value, *, status="ok", valid=True, split="eval", components=None, judge=None):
    return EpisodeRecord(episode_id=f"{qid}_{sample}", task="toy", task_version="1", question_id=qid, split=split,
                         sample_index=sample, status=status, termination=Termination("answer", answer="x"),
                         score=None if status != "ok" else Score(value, valid, "p", components=components or {},
                                                                 judge=judge))


PLAN = {"task": "toy", "task_version": "1", "rows_sha256": "h", "samples": 2, "ids": ["a", "b"], "model": {}}


def test_completeness_counts_every_kind_of_gap():
    records = [record("a", 0, 1.0), record("a", 1, 0.0), record("b", 0, 1.0, status="infra_failure")]
    head = summarize(records, PLAN)["headline"]
    assert head["planned"] == 4 and head["missing"] == 1 and head["infra_failures"] == 1
    assert not head["complete"] and head["score_mean_valid"] == 0.5 and head["score_mean_strict"] == 0.25
    full = summarize([record(q, k, 1.0) for q in "ab" for k in range(2)], PLAN)["headline"]
    assert full["complete"] and full["score_mean_strict"] == 1.0
    driver = summarize([record("a", 0, 1.0), record("a", 1, 1.0), record("b", 0, 1.0),
                        record("b", 1, 0.0, status="policy_failure")], PLAN)["headline"]
    assert driver["policy_failures"] == 1 and driver["valid"] == 3 and not driver["complete"]


def test_a_fallback_judges_label_enters_the_headline_only_if_its_policy_allows():
    def run(headline):
        judge = {"model": "judge-b", "fallback": True, "headline": headline}
        return summarize([record("a", 0, 1.0), record("a", 1, 1.0), record("b", 0, 1.0),
                          record("b", 1, 1.0, judge=judge)], PLAN)["headline"]
    allowed, withheld = run(True), run(False)
    assert allowed["complete"] and allowed["valid"] == 4 and allowed["fallback_labels"] == 1
    assert not withheld["complete"] and withheld["valid"] == 3 and withheld["fallback_withheld"] == 1


def test_paired_comparison_and_leakage(tmp_path):
    rows = [{"id": q, "split": "eval", "messages": [{"role": "user", "content": q}], "ground_truth": "1",
             "metadata": {}} for q in ("a", "b", "c")]
    write_rows(tmp_path / "eval.jsonl", rows)
    write_rows(tmp_path / "train.jsonl", [{**rows[0], "id": "train-0", "split": "train"}])   # same question text
    for label, scores in (("base", {"a": 0, "b": 0, "c": 1}), ("trained", {"a": 1, "b": 1, "c": 1})):
        sink = EpisodeSink(tmp_path / label / "episodes")
        for q, v in scores.items():
            sink.write(record(q, 0, float(v)))
        sink.flush()
        plan = {**PLAN, "samples": 1, "ids": list(scores), "rows": str(tmp_path / "eval.jsonl")}
        (tmp_path / label / "plan.json").write_text(json.dumps(plan))
    report = compare_runs(str(tmp_path / "base"), str(tmp_path / "trained"), train_rows=str(tmp_path / "train.jsonl"))
    summary = report["summary"]
    assert summary["comparable"] and summary["paired_questions"] == 3 and summary["incomplete"] == []
    assert summary["mean_difference"] == pytest.approx(2 / 3) and summary["mcnemar_exact_p"] == 0.5
    assert summary["leakage"] == {"shared_ids": 0, "shared_questions": 1, "exempt_probe_rows": 0, "clean": False}


def test_an_incomplete_evaluation_is_not_comparable(tmp_path):
    """Pairing drops a question without a valid sample, which the strict mean counts as zero."""
    for label, records in (("base", [record("a", 0, 0.0), record("b", 0, 1.0)]),
                           ("trained", [record("a", 0, 1.0), record("b", 0, 1.0, status="infra_failure")])):
        sink = EpisodeSink(tmp_path / label / "episodes")
        for r in records:
            sink.write(r)
        sink.flush()
        (tmp_path / label / "plan.json").write_text(json.dumps({**PLAN, "samples": 1}))
    summary = compare_runs(str(tmp_path / "base"), str(tmp_path / "trained"))["summary"]
    assert summary["paired_questions"] == 1 and summary["incomplete"] == ["trained"] and not summary["comparable"]


def test_probes_are_reported_per_split_and_exempt_from_leakage(tmp_path):
    """Training probes are training questions on purpose: compared per split, not counted as leakage."""
    rows = [{"id": q, "split": split, "messages": [{"role": "user", "content": q}], "ground_truth": "1",
             "metadata": {}} for q, split in (("h1", "heldout"), ("h2", "heldout"), ("p1", "probe"))]
    write_rows(tmp_path / "eval.jsonl", rows)
    write_rows(tmp_path / "train.jsonl", [{**rows[2], "split": "train"}])
    for label, scores in (("base", {"h1": 0, "h2": 0, "p1": 1}), ("trained", {"h1": 1, "h2": 0, "p1": 1})):
        sink = EpisodeSink(tmp_path / label / "episodes")
        for q, v in scores.items():
            sink.write(record(q, 0, float(v), split=next(r["split"] for r in rows if r["id"] == q)))
        sink.flush()
        plan = {**PLAN, "samples": 1, "ids": list(scores), "rows": str(tmp_path / "eval.jsonl")}
        (tmp_path / label / "plan.json").write_text(json.dumps(plan))
    report = compare_runs(str(tmp_path / "base"), str(tmp_path / "trained"), train_rows=str(tmp_path / "train.jsonl"),
                          probe_splits=("probe",))
    assert report["splits"]["heldout"]["mean_difference"] == 0.5 and report["splits"]["probe"]["mean_difference"] == 0
    assert report["summary"]["leakage"] == {"shared_ids": 0, "shared_questions": 0, "exempt_probe_rows": 1, "clean": True}


def test_parity_with_recorded_per_question_results(tmp_path):
    sink = EpisodeSink(tmp_path / "episodes")
    for q, split, values in (("h1", "heldout", (1, 0)), ("h2", "heldout", (0, 0)), ("p1", "probe", (1, 1))):
        for k, v in enumerate(values):
            sink.write(record(q, k, float(v), split=split, components={"answer_correct": float(v)}))
    sink.flush()
    (tmp_path / "plan.json").write_text(json.dumps({**PLAN, "ids": ["h1", "h2", "p1"]}))
    reference = {"source": "earlier run", "per_question": {
        "h1": {"split": "heldout", "pass_at_1": 0.5, "answer_accuracy": 0.5},
        "h2": {"split": "heldout", "pass_at_1": 0.0, "answer_accuracy": 0.5},
        "p1": {"split": "probe", "pass_at_1": 1.0, "answer_accuracy": 1.0}}}
    metrics = {"pass_at_1": "score", "answer_accuracy": "answer_correct"}
    report = parity(str(tmp_path), reference, metrics=metrics)
    assert report["agree"] and report["metrics"]["heldout/pass_at_1"]["mean_difference"] == 0.0
    assert report["metrics"]["heldout/answer_accuracy"]["mean_difference"] == -0.25
    strict = parity(str(tmp_path), reference, metrics=metrics, max_abs_difference=0.1)
    assert not strict["agree"] and not strict["metrics"]["heldout/answer_accuracy"]["agree"]
    reference["per_question"]["h3"] = {"split": "heldout", "pass_at_1": 0.0, "answer_accuracy": 0.0}
    assert not parity(str(tmp_path), reference, metrics=metrics)["agree"]   # a reference question was not run


def test_statistics():
    assert bootstrap_ci([1.0] * 10) == [1.0, 1.0]
    assert mcnemar_exact({"a": 0.0}, {"a": 1.0}, ["a"]) == 1.0
    assert mcnemar_exact({"a": 0.5}, {"a": 1.0}, ["a"]) is None   # not binary: several samples


def test_the_sink_writes_closed_chunks_and_readers_skip_damage(tmp_path, row):
    import asyncio

    from rft_cookbook.agent.harness import ByteCounter, ChatTurn, run_episode

    sink = EpisodeSink(tmp_path, flush_every=2)
    for i in range(3):
        client = ScriptedClient([ChatTurn(text="", tool_calls=[call("submit", "{}")])])
        sink.write(asyncio.run(run_episode(make_task(), client, {**row, "id": f"q{i}"}, counter=ByteCounter())))
    assert len(list(tmp_path.glob("*.jsonl"))) == 1   # the first two, closed; the third still buffered
    sink.flush()
    (tmp_path / "zz-damaged.jsonl").write_text('{"schema": "rft-cookbook/episode/v1", "tru\n')
    records, damaged = read_episodes(tmp_path)
    assert len(records) == 3 and damaged == 1


def test_buffered_records_reach_storage_without_another_write_or_exit(tmp_path, row):
    """Ray ends agent-loop workers without exit handlers: a quiet buffer still gets written."""
    import asyncio
    import time

    from rft_cookbook.agent.harness import ByteCounter, ChatTurn, run_episode

    first, second = EpisodeSink(tmp_path, flush_after_s=0.2), EpisodeSink(tmp_path, flush_after_s=0.2)
    for sink in (first, second):
        client = ScriptedClient([ChatTurn(text="", tool_calls=[call("submit", "{}")])])
        sink.write(asyncio.run(run_episode(make_task(), client, row, counter=ByteCounter())))
    deadline = time.monotonic() + 5
    while len(list(tmp_path.glob("*.jsonl"))) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(list(tmp_path.glob("*.jsonl"))) == 2   # two sinks in one process never share a name
    assert len(read_episodes(tmp_path)[0]) == 2

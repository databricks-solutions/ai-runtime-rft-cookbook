# Derived from verl-on-air (Apache License 2.0), usecases/math/prep_data.py and eval.py at 8816f63;
# modified for ai-runtime-rft-cookbook: writes canonical rows instead of verl parquet.
"""Prepare MATH (train, by level) and MATH-500 (evaluation) as canonical rows.

MATH is read at a pinned commit and must hold the recorded content (row counts and an
order-independent digest of its problem/solution pairs); the one known mirror is accepted only
with `--allow-mirror` and only if it holds that same content. MATH-500 is 500 problems of the
MATH test split, so training on the train split does not touch it.

    python -m rft_cookbook.tasks.math.data --out <dir> --levels 1,2,3
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from rft_cookbook.data import HubSource, content_digest, load_verified, write_manifest, write_rows
from rft_cookbook.tasks.math.grading import last_boxed

MATH = HubSource("DigitalLearningGmbH/MATH-lighteval", "0530c78699ea5e8eb5530600900e1f328b48acad")
MATH_CONTENT = {
    "train": (7500, "fbdc8b6908196fdde6f076560b176af917c439bbe85fe0bf6ea5db274ea60ccc"),
    "test": (5000, "13d5c4873d6a6b184618aad7d1ff5b748c0adcbc56656f916d2b0a8bf8de08b5"),
}
# Its seven subject configs, concatenated in this order, are the same rows in the same order.
MATH_MIRROR = HubSource("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3")
_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
             "prealgebra", "precalculus"]
MATH500 = HubSource("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")


def _level(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit()) if value is not None else ""
    return int(digits) if digits else None


def _ground_truth(example: dict[str, Any]) -> str | None:
    answer = example.get("answer")
    gt = answer if (answer and "\\boxed" not in str(answer)) else last_boxed(example.get("solution") or "")
    return str(gt).strip() if gt else None


def _row(prefix: str, split: str, index: int, example: dict[str, Any]) -> dict[str, Any] | None:
    problem = example.get("problem") or ""
    gt = _ground_truth(example)
    if not problem or not gt:
        return None
    return {"id": f"{prefix}-{index:05d}", "split": split, "messages": [{"role": "user", "content": problem}],
            "ground_truth": gt,
            "metadata": {"level": _level(example.get("level")), "subject": example.get("type") or example.get("subject") or "",
                         "source_index": index}}


def _by_subject(source: HubSource):
    import datasets

    parts = [HubSource(source.hf_id, source.revision, subject).load() for subject in _SUBJECTS]
    return datasets.DatasetDict(train=datasets.concatenate_datasets([p["train"] for p in parts]),
                                test=datasets.concatenate_datasets([p["test"] for p in parts]))


def _is_math(ds) -> str | None:
    for split, (n, digest) in MATH_CONTENT.items():
        if split not in ds:
            return f"no {split!r} split"
        got = (len(ds[split]), content_digest(ds[split], ["problem", "solution"]))
        if got != (n, digest):
            return f"{split}: {got[0]} rows, digest {got[1][:12]}; expected {n}, {digest[:12]}"
    return None


def prepare_recipe(recipe: dict[str, Any], out: Path, **_: Any) -> dict[str, Any]:
    data = recipe.get("data") or {}
    return prepare(out, levels=set(data["levels"]) if data.get("levels") else None,
                   train_limit=int(data.get("train_limit", 0)), eval_limit=int(data.get("eval_limit", 0)))


def prepare(out: Path, *, levels: set[int] | None, train_limit: int = 0, eval_limit: int = 0,
            allow_mirror: bool = False) -> dict[str, Any]:
    math, math_record = load_verified([(MATH, lambda s: s.load()), (MATH_MIRROR, _by_subject)], _is_math,
                                      allow_mirror=allow_mirror)
    train = [r for i, ex in enumerate(math["train"]) if (r := _row("math-train", "train", i, ex))
             and (levels is None or r["metadata"]["level"] in levels)]
    if train_limit:
        train = train[:train_limit]
    test = MATH500.load(split="test")
    evaluation = [r for i, ex in enumerate(test) if (r := _row("math500", "eval", i, ex))]
    if eval_limit:
        evaluation = evaluation[:eval_limit]
    outputs = [write_rows(out / "train.jsonl", train), write_rows(out / "eval.jsonl", evaluation)]
    manifest = {
        "task": "math", "tool": "rft_cookbook.tasks.math.data",
        "sources": [{**math_record, "content": MATH_CONTENT}, MATH500.record()],
        "filters": {"levels": sorted(levels) if levels else "all", "train_limit": train_limit,
                    "eval_limit": eval_limit,
                    "ground_truth": "a clean `answer` field, else the solution's last \\boxed{}"},
        "levels": {"train": dict(sorted(Counter(r["metadata"]["level"] for r in train).items()))},
        "outputs": outputs,
    }
    write_manifest(out, **manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--levels", default="", help="comma-separated MATH levels for training; empty = all")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--allow-mirror", action="store_true")
    args = parser.parse_args(argv)
    levels = {int(x) for x in args.levels.split(",") if x.strip()} or None
    manifest = prepare(args.out, levels=levels, train_limit=args.train_limit, eval_limit=args.eval_limit,
                       allow_mirror=args.allow_mirror)
    for output in manifest["outputs"]:
        print(f"wrote {output['rows']} rows -> {args.out / output['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

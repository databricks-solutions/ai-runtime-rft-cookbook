# Derived from verl-on-air (Apache License 2.0), usecases/agentic-search/prep_data.py and
# make_splits.py at 1c5653c; modified for ai-runtime-rft-cookbook: writes canonical rows, and reads
# the held-out list from heldout.ids instead of drawing it again. See THIRD_PARTY_NOTICES.md.
"""Prepare MuSiQue-Ans for agentic search: training questions, a held-out set and probes.

    train.jsonl   every answerable training question (split `train`)
    eval.jsonl    500 held-out validation questions (split `heldout`) and 32 fixed training
                  questions (split `probe`)

The held-out questions are `heldout.ids`: validation rows 500 onwards (past the rows the earlier
prototype developed on), stratified by hop count with seed 20260923 (196 two-hop, 198 three-hop,
106 four-hop); the list is checked against its recorded sha256. Probes are the first 32 training
questions by the sha256 of their id. A row's ground truth is the answer and its aliases; its
metadata records the hop count. Unanswerable questions are dropped.

    python -m rft_cookbook.tasks.search.data --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from importlib import resources as package_files
from pathlib import Path
from typing import Any

from rft_cookbook.data import HubSource, write_manifest, write_rows

MUSIQUE = HubSource("dgslibisey/MuSiQue", "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321")
VALIDATION_ROWS = 2417
HELDOUT_SHA256 = "12b4f232aa61a028f684c690e91020e1836eefc5ede96f237de8c340d1fe417c"
HELDOUT_RULE = "validation rows 500.. stratified by hop count, proportional, seed 20260923"
PROBES = 32


def heldout_ids() -> list[str]:
    text = package_files.files(__package__).joinpath("heldout.ids").read_text()
    if hashlib.sha256(text.encode()).hexdigest() != HELDOUT_SHA256:
        raise SystemExit("heldout.ids does not hold the recorded held-out list")
    return text.split()


def golds(example: dict[str, Any]) -> list[str]:
    seen, out = set(), []
    for answer in [example.get("answer"), *(example.get("answer_aliases") or [])]:
        answer = str(answer or "").strip()
        if answer and answer.lower() not in seen:
            seen.add(answer.lower())
            out.append(answer)
    return out


def row(example: dict[str, Any], split: str) -> dict[str, Any] | None:
    question = (example.get("question") or "").strip()
    answers = golds(example)
    if example.get("answerable") is False or not question or not answers:
        return None
    return {"id": str(example["id"]), "split": split,
            "messages": [{"role": "user", "content": f"Question: {question}"}], "ground_truth": answers,
            "metadata": {"hops": len(example.get("question_decomposition") or [])}}


def split(train_examples, validation_examples) -> tuple[list[dict], list[dict], list[dict], Counter]:
    """(train, heldout, probe, dropped) from MuSiQue's training and validation examples."""
    dropped: Counter = Counter()
    train = []
    for example in train_examples:
        got = row(example, "train")
        if got is None:
            dropped["train: unanswerable or empty"] += 1
        else:
            train.append(got)
    wanted = heldout_ids()
    by_id = {str(e["id"]): e for e in validation_examples}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} held-out questions are not in the validation split, e.g. {missing[:3]}")
    heldout = [row(by_id[i], "heldout") for i in wanted]
    if any(r is None for r in heldout):
        raise SystemExit("a held-out question is unanswerable: the source is not what it should be")
    probe = [{**r, "split": "probe"} for r in sorted(train, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
             [:PROBES]]
    train_ids, train_questions = {r["id"] for r in train}, {r["messages"][0]["content"] for r in train}
    if any(r["id"] in train_ids or r["messages"][0]["content"] in train_questions for r in heldout):
        raise SystemExit("a held-out question is also a training question")
    return train, heldout, probe, dropped


def prepare_recipe(recipe: dict[str, Any], out: Path, **_: Any) -> dict[str, Any]:
    return prepare(out)


def prepare(out: Path) -> dict[str, Any]:
    data = MUSIQUE.load()
    if len(data["validation"]) != VALIDATION_ROWS:
        raise SystemExit(f"MuSiQue validation has {len(data['validation'])} rows, not {VALIDATION_ROWS}")
    train, heldout, probe, dropped = split(data["train"], data["validation"])
    outputs = [write_rows(out / "train.jsonl", train), write_rows(out / "eval.jsonl", probe + heldout)]
    manifest = {
        "task": "search", "tool": "rft_cookbook.tasks.search.data",
        "source": {"repository": MUSIQUE.hf_id, "revision": MUSIQUE.revision, "train": len(data["train"]),
                   "validation": len(data["validation"])},
        "dropped": dict(dropped),
        "split": {"heldout_rule": HELDOUT_RULE, "heldout_sha256": HELDOUT_SHA256, "train": len(train),
                  "heldout": len(heldout), "probe": len(probe),
                  "heldout_by_hops": dict(sorted(Counter(r["metadata"]["hops"] for r in heldout).items())),
                  "probe_rule": f"the first {PROBES} training questions by sha256(id)",
                  "probe_ids": [r["id"] for r in probe]},
        "outputs": outputs,
    }
    write_manifest(out, **manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = prepare(args.out)
    print({k: manifest["split"][k] for k in ("train", "heldout", "probe", "heldout_by_hops")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

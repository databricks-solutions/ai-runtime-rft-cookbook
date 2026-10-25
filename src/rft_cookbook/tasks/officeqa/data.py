# Derived from verl-on-air (Apache License 2.0), usecases/officeqa/build_corpus.py, prep_data.py and
# task.py at b088a42; modified for ai-runtime-rft-cookbook: writes canonical rows and the corpus in
# one step. The document rendering, chunking and split are unchanged. See THIRD_PARTY_NOTICES.md.
"""Prepare OfficeQA: the pinned question set, the document corpus, the BM25 chunks and the split.

    python -m rft_cookbook.tasks.officeqa.data --out <dir> [--scratch <local dir>]

Downloads the creators' release (`databricks/officeqa`) at the revision in `data.lock.json`, and
rebuilds what the measured verl-on-air run used, checking each against the lock:

    corpus/treasury_bulletins_clean.zip   697 documents rendered from the parsed JSON, tables kept
                                          lossless (cell text unchanged); what grep and read see
    corpus/chunks.jsonl                   131,113 chunks of the creators' transformed text; what
                                          search ranks
    train.jsonl                           every easy question and 90 hard ones (203)
    eval.jsonl                            the other 43 hard questions (split `heldout`) and 16
                                          fixed training questions (split `probe`)

Hard questions are ordered by the SHA-256 of their uid, so the split is fixed without a seed and
without model outcomes. Gold answers and source-file hints never enter a prompt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from rft_cookbook.data import write_manifest, write_rows

LOCK = json.loads(resources.files(__package__).joinpath("data.lock.json").read_text())
PARSED_DIR = "treasury_bulletins_parsed/jsons"
TRANSFORMED_DIR = "treasury_bulletins_parsed/transformed"
ARCHIVE = "treasury_bulletins_clean.zip"
CHUNKS = "chunks.jsonl"
QUESTIONS = "officeqa_full.csv"
HARD_TRAIN = 90
PROBES_PER_DIFFICULTY = 8


@dataclass(frozen=True)
class Question:
    uid: str
    question: str
    answer: str
    difficulty: str


def official_split(rows: list[dict], *, hard_train: int = HARD_TRAIN) -> tuple[list[Question], list[Question]]:
    """(train, held out): every easy question plus `hard_train` hard ones in uid-hash order."""
    by_difficulty: dict[str, list[Question]] = {"easy": [], "hard": []}
    for row in sorted(rows, key=lambda row: row["uid"]):
        if row["difficulty"] not in by_difficulty:
            raise ValueError(f"{row['uid']}: unknown difficulty {row['difficulty']!r}")
        by_difficulty[row["difficulty"]].append(
            Question(row["uid"], row["question"].strip(), row["answer"].strip(), row["difficulty"]))
    hard = sorted(by_difficulty["hard"], key=lambda q: hashlib.sha256(q.uid.encode()).hexdigest())
    if not 0 < hard_train < len(hard):
        raise ValueError("hard_train must leave hard questions in both training and the held-out set")
    return by_difficulty["easy"] + hard[:hard_train], hard[hard_train:]


def probes(train: list[Question], per_difficulty: int = PROBES_PER_DIFFICULTY) -> list[Question]:
    """Fixed training questions evaluated alongside the held-out set (the measured run's 16)."""
    ordered = sorted(train, key=lambda q: hashlib.sha256(q.uid.encode()).hexdigest())
    return [q for difficulty in ("easy", "hard") for q in [q for q in ordered if q.difficulty == difficulty][:per_difficulty]]


def row(question: Question, split: str) -> dict[str, Any]:
    return {"id": question.uid, "split": split, "messages": [{"role": "user", "content": question.question}],
            "ground_truth": question.answer,
            "metadata": {"question": question.question, "question_requirements": "", "difficulty": question.difficulty}}


def render_document(document: dict) -> bytes:
    """The creators' element order and text layout, with lossless tables."""
    from rft_cookbook.tasks.officeqa.html_table_parser import parse_html_tables

    elements = (document.get("document") or {}).get("elements")
    if not isinstance(elements, list):
        raise ValueError("parsed document has no elements list")
    lines = []
    for element in elements:
        content = element.get("content") if isinstance(element, dict) else None
        if not isinstance(content, str) or not content.strip():
            continue
        if "<table" in content.lower():
            for table in parse_html_tables(content):
                lines.extend([table, ""])
        else:
            lines.extend(re.sub(r"\r\n?", "\n", content.strip()).split("\n"))
            lines.append("")
    if not lines:
        raise ValueError("parsed document has no usable text")
    return "".join(line + "\n" for line in lines).encode("utf-8")


def archive_identity(path: Path) -> dict[str, Any]:
    """SHA-256 of the lines '<basename>\\0<text SHA-256>\\n', sorted by basename: independent of ZIP metadata."""
    digest, text_bytes = hashlib.sha256(), 0
    with zipfile.ZipFile(path) as archive:
        members = sorted((Path(name).name, name) for name in archive.namelist() if name.endswith(".txt"))
        if not members or len({base for base, _ in members}) != len(members):
            raise ValueError("corpus is empty or has duplicate document basenames")
        for basename, member in members:
            content = archive.read(member)
            text_bytes += len(content)
            digest.update(f"{basename}\0{hashlib.sha256(content).hexdigest()}\n".encode())
    return {"documents": len(members), "text_bytes": text_bytes, "content_sha256": digest.hexdigest()}


def file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def check_identity(label: str, actual: dict, expected: dict) -> None:
    mismatched = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatched:
        raise ValueError(f"{label} differs from the pinned snapshot ({', '.join(mismatched)})")


def build_archive(parsed_dir: Path, destination: Path) -> dict[str, Any]:
    files = sorted(parsed_dir.glob("treasury_bulletin_*.json"))
    if not files:
        raise FileNotFoundError(f"no parsed Treasury Bulletins in {parsed_dir}")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for count, source in enumerate(files, 1):
            member = zipfile.ZipInfo(source.with_suffix(".txt").name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type, member.create_system, member.external_attr = zipfile.ZIP_DEFLATED, 3, 0o644 << 16
            archive.writestr(member, render_document(json.loads(source.read_text(encoding="utf-8"))), compresslevel=6)
            if count % 100 == 0:
                print(f"[officeqa-data] rendered {count}/{len(files)} documents", flush=True)
    return archive_identity(destination)


def build_chunks(text_dir: Path, destination: Path) -> dict[str, Any]:
    """The measured index's rendering, order and JSON serialization."""
    from rft_cookbook.tasks.officeqa.chunker import chunk_file

    files = sorted(text_dir.glob("treasury_bulletin_*.txt"))
    if not files:
        raise FileNotFoundError(f"no transformed Treasury Bulletins in {text_dir}")
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for documents, source in enumerate(files, 1):
            for chunk in chunk_file(source):
                if chunk.content.strip():
                    output.write(json.dumps({"content": chunk.content, "source": chunk.file_name,
                                             "bulletin_date": chunk.bulletin_date, "page_info": chunk.page_info,
                                             "chunk_type": chunk.chunk_type}) + "\n")
                    count += 1
            if documents % 100 == 0:
                print(f"[officeqa-data] indexed {documents} documents, {count} chunks", flush=True)
    if not count:
        raise ValueError("empty OfficeQA retrieval index")
    return {**file_identity(destination), "count": count}


def download(scratch: Path) -> Path:
    from huggingface_hub import snapshot_download

    source = LOCK["source"]
    return Path(snapshot_download(source["repository"], repo_type="dataset", revision=source["revision"],
                                  allow_patterns=[QUESTIONS, f"{PARSED_DIR}/*.json", f"{TRANSFORMED_DIR}/*.txt"],
                                  local_dir=scratch / "officeqa"))


def prepare_recipe(recipe: dict[str, Any], out: Path, *, scratch) -> dict[str, Any]:
    return prepare(out, scratch=scratch())


def prepare(out: Path, *, scratch: Path, source: Path | None = None) -> dict[str, Any]:
    """Build everything in `scratch` (local disk), check it against the lock, then copy closed files to `out`."""
    source = source or download(scratch)
    questions_csv = source / QUESTIONS
    check_identity("question CSV", file_identity(questions_csv), LOCK["question_csv"])
    with questions_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    difficulties = {r["uid"]: r["difficulty"] for r in rows}
    if len(difficulties) != len(rows):
        raise ValueError("duplicate question uid")
    counts = {d: sum(v == d for v in difficulties.values()) for d in ("easy", "hard")}
    if counts != {"easy": 113, "hard": 133} or len(rows) != 246:
        raise ValueError(f"expected the 246-question benchmark (113 easy, 133 hard), got {counts}")
    train, heldout = official_split(rows)
    probe = probes(train)
    if {q.uid for q in train} & {q.uid for q in heldout} or {q.question for q in train} & {q.question for q in heldout}:
        raise ValueError("training and held-out questions overlap")

    build = Path(tempfile.mkdtemp(prefix="officeqa-build-", dir=scratch))
    corpus = build_archive(source / PARSED_DIR, build / ARCHIVE)
    check_identity("rendered corpus", corpus, LOCK["corpus"])
    chunks = build_chunks(source / TRANSFORMED_DIR, build / CHUNKS)
    check_identity("retrieval chunks", chunks, LOCK["chunks"])
    (out / "corpus").mkdir(parents=True, exist_ok=True)
    for name in (ARCHIVE, CHUNKS):   # a volume publishes a file once it is closed: copy, then rename
        partial = out / "corpus" / f".{name}.partial"
        shutil.copyfile(build / name, partial)
        partial.replace(out / "corpus" / name)
    shutil.rmtree(build, ignore_errors=True)

    outputs = [write_rows(out / "train.jsonl", [row(q, "train") for q in train]),
               write_rows(out / "eval.jsonl", [row(q, "probe") for q in probe] + [row(q, "heldout") for q in heldout])]
    manifest = {
        "task": "officeqa", "tool": "rft_cookbook.tasks.officeqa.data", "source": LOCK["source"],
        "question_csv": LOCK["question_csv"], "corpus": {**corpus, "file": f"corpus/{ARCHIVE}"},
        "chunks": {**chunks, "file": f"corpus/{CHUNKS}"},
        "split": {"rule": "all easy + the first 90 hard questions by sha256(uid); the rest held out",
                  "train": len(train), "heldout": len(heldout), "probe": len(probe),
                  "probe_rule": "the first 8 easy and 8 hard training questions by sha256(uid)",
                  "heldout_ids": [q.uid for q in heldout], "probe_ids": [q.uid for q in probe]},
        "outputs": outputs,
    }
    write_manifest(out, **manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--source", type=Path, help="an existing download of the pinned release")
    args = parser.parse_args(argv)
    manifest = prepare(args.out, scratch=args.scratch, source=args.source)
    print(json.dumps({k: manifest[k] for k in ("corpus", "chunks", "outputs")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

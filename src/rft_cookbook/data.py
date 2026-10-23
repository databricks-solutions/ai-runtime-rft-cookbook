"""Canonical task data: JSONL rows plus a manifest that pins where they came from.

A row is `{"id", "split", "messages", "ground_truth", "metadata"}`. `messages` holds the
question as chat messages without a system prompt (the task owns that). Framework adapters
convert these files to their own formats; evaluation reads them directly.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

MANIFEST = "manifest.json"
_COMMIT = re.compile(r"[0-9a-f]{40}")
ROW_FIELDS = ("id", "split", "messages", "ground_truth", "metadata")


class SourceError(RuntimeError):
    """A dataset could not be read as pinned, or does not hold the pinned content."""


@dataclass(frozen=True)
class HubSource:
    """One Hugging Face dataset at one commit."""

    hf_id: str
    revision: str
    config: str | None = None

    def __post_init__(self):
        if not _COMMIT.fullmatch(self.revision or ""):
            raise SourceError(f"{self.hf_id}: pin a 40-hex commit, not {self.revision!r}")

    def load(self, split: str | None = None):
        import datasets

        return datasets.load_dataset(self.hf_id, self.config, split=split, revision=self.revision)

    def record(self, **extra: Any) -> dict[str, Any]:
        return {"hf_id": self.hf_id, "config": self.config, "revision": self.revision, **extra}


def content_digest(rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> str:
    """sha256 over `fields` of every row, independent of row order: equal digests, same rows."""
    per_row = sorted(hashlib.sha256(json.dumps([r[f] for f in fields], default=str).encode()).hexdigest()
                     for r in rows)
    return hashlib.sha256("".join(per_row).encode()).hexdigest()


def load_verified(candidates: Sequence[tuple[HubSource, Callable[[HubSource], Any]]],
                  verify: Callable[[Any], str | None], *, allow_mirror: bool = False) -> tuple[Any, dict[str, Any]]:
    """Load the pinned source (candidates[0]); mirrors only with `allow_mirror`, and only if
    `verify` accepts their content too. Returns (data, source record)."""
    tried = []
    for i, (source, loader) in enumerate(candidates):
        if i and not allow_mirror:
            break
        try:
            data = loader(source)
            why = verify(data)
        except Exception as error:  # noqa: BLE001 - a candidate that cannot load is a reason
            why = f"{type(error).__name__}: {error}"
        if why is None:
            return data, source.record(mirror_of=candidates[0][0].hf_id if i else None)
        tried.append(f"{source.hf_id}@{source.revision[:12]}: {why}")
    raise SourceError("no candidate holds the pinned data: " + "; ".join(tried))


def check_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = [f for f in ROW_FIELDS if f not in row]
    if missing:
        raise ValueError(f"row {row.get('id')!r} lacks {missing}")
    if not row["messages"] or any(m.get("role") == "system" for m in row["messages"]):
        raise ValueError(f"row {row['id']!r}: messages must be non-empty and carry no system prompt")
    return row


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Write rows as JSONL; returns the output record (path, rows, sha256, ids digest)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count, ids, seen = 0, hashlib.sha256(), set()
    with open(path, "w") as handle:
        for row in rows:
            check_row(row)
            if row["id"] in seen:
                raise ValueError(f"duplicate row id {row['id']!r} in {path.name}")
            seen.add(row["id"])
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    for row_id in sorted(seen):
        ids.update(f"{row_id}\n".encode())
    return {"file": path.name, "rows": count, "sha256": file_sha256(path), "ids_sha256": ids.hexdigest()}


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield check_row(json.loads(line))


def write_manifest(directory: Path, **fields: Any) -> Path:
    path = directory / MANIFEST
    path.write_text(json.dumps(fields, indent=1, sort_keys=True))
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

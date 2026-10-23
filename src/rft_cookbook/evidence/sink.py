"""Write episode records from many worker processes without long-open files.

Unity Catalog volumes upload a file when it is closed, so a process never appends to a shared
file: it writes closed chunks `<prefix>-<host>-<pid>-<token>-<n>.jsonl`, each once, when
`flush_every` records are buffered or, from a background thread, once the oldest buffered record
is `flush_after_s` old. The thread matters: Ray ends worker processes without running exit
handlers, so a buffer that waits for the next write or for exit is lost. The random token keeps
two jobs from writing the same names (containers reuse host names and pids). Readers skip a
damaged line and count it.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

from rft_cookbook.agent.episode import EpisodeRecord


class EpisodeSink:
    def __init__(self, directory: str | os.PathLike, *, prefix: str = "episodes", flush_every: int = 32,
                 flush_after_s: float = 60.0):
        self.directory = Path(directory)
        self.prefix = f"{prefix}-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.flush_every, self.flush_after_s = flush_every, flush_after_s
        self._buffer: list[str] = []
        self._oldest: float | None = None
        self._chunks = 0
        self._lock = threading.Lock()
        atexit.register(self.flush)
        threading.Thread(target=self._flush_when_old, name="episode-sink", daemon=True).start()

    def _flush_when_old(self) -> None:
        while True:
            time.sleep(min(1.0, self.flush_after_s))
            with self._lock:
                due = self._oldest is not None and time.monotonic() - self._oldest >= self.flush_after_s
            if due:
                try:
                    self.flush()
                except OSError as error:   # kept in the buffer; the next attempt retries it
                    print(f"[episode-sink] {self.directory}: {error}", file=sys.stderr)

    def write(self, record: EpisodeRecord) -> None:
        line = record.to_json()
        with self._lock:
            self._buffer.append(line)
            self._oldest = self._oldest or time.monotonic()
            due = len(self._buffer) >= self.flush_every or time.monotonic() - self._oldest >= self.flush_after_s
        if due:
            self.flush()

    def flush(self) -> Path | None:
        with self._lock:
            if not self._buffer:
                return None
            lines, self._buffer, self._oldest = self._buffer, [], None
            self._chunks += 1
            path = self.directory / f"{self.prefix}-{self._chunks:05d}.jsonl"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n")
        except OSError:
            with self._lock:
                self._buffer[:0] = lines
                self._oldest = time.monotonic()
            raise
        return path


def read_episodes(directory: str | os.PathLike) -> tuple[list[EpisodeRecord], int]:
    """All records under `directory`, and the number of lines that could not be read."""
    records, damaged = [], 0
    for record in _lines(Path(directory)):
        if record is None:
            damaged += 1
        else:
            records.append(record)
    return records, damaged


def _lines(directory: Path) -> Iterator[EpisodeRecord | None]:
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    yield EpisodeRecord.from_json(line)
                except (ValueError, TypeError):
                    yield None


_SINKS: dict[str, EpisodeSink] = {}


def shared_sink(directory: str | os.PathLike | None, **kwargs) -> EpisodeSink | None:
    """One sink per directory per process (agent loops are created per episode)."""
    if not directory:
        return None
    key = str(directory)
    if key not in _SINKS:
        _SINKS[key] = EpisodeSink(directory, **kwargs)
    return _SINKS[key]

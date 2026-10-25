"""Publish a run's evidence: its qualification record, comparison, calibration, export record and
stage receipts, with workspace details replaced by placeholders (docs/architecture.md, "Public
repository rules").

    python -m rft_cookbook.evidence.publish <run dir> <out dir> --replace <your volume root>=<volume-root> ...

Each file keeps every value except the replaced text. `SANITIZED.json` lists, per file, the sha256
of the original and how often each replacement applied, so a published file can be checked
against the run's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "rft-cookbook/sanitized/v1"
FILES = ("qualification.json", "comparison.json", "calibration.json", "export.json")


def publish(run_dir: Path, out: Path, replace: dict[str, str]) -> dict[str, Any]:
    if not (run_dir / "qualification.json").is_file():
        raise FileNotFoundError(f"{run_dir} has no qualification.json: its compare stage has not run")
    sources = [run_dir / name for name in FILES if (run_dir / name).is_file()]
    sources += sorted((run_dir / "receipts").glob("*.json"))
    report: dict[str, Any] = {"schema": SCHEMA, "files": {}}
    for source in sources:
        raw = source.read_bytes()
        text, counts = raw.decode(), {}
        for old, new in replace.items():
            if old in text:
                counts[new] = text.count(old)
                text = text.replace(old, new)
        json.loads(text)   # placeholders never break the JSON
        target = out / source.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        report["files"][source.relative_to(run_dir).as_posix()] = {
            "original_sha256": hashlib.sha256(raw).hexdigest(), "replaced": counts}
    (out / "SANITIZED.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--replace", action="append", default=[], metavar="TEXT=PLACEHOLDER")
    args = parser.parse_args(argv)
    replace = dict(item.split("=", 1) for item in args.replace)
    report = publish(args.run_dir, args.out, replace)
    print(f"{len(report['files'])} files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

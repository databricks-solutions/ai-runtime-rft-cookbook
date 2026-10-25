"""Publishing a run's evidence (rft_cookbook.evidence.publish): placeholders for workspace details,
every other value kept, the original's hash recorded."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from rft_cookbook.evidence.publish import publish

REPO = Path(__file__).resolve().parents[1]
ROOT = "/Volumes/cat/sch/vol/rft"


def test_workspace_details_become_placeholders_and_the_originals_are_named_by_hash(tmp_path):
    run = tmp_path / "run"
    (run / "receipts").mkdir(parents=True)
    record = {"status": "measured", "image": "someone/verl@sha256:aa",
              "evaluation": {"baseline": f"{ROOT}/runs/r/eval/baseline"},
              "stages": [json.dumps({"image": "someone/verl@sha256:aa", "stage": "train"})]}
    (run / "qualification.json").write_text(json.dumps(record))
    (run / "receipts" / "train-1-host.json").write_text(json.dumps({"image": "someone/verl@sha256:aa"}))
    (run / "notes.txt").write_text(ROOT)   # not evidence: not published

    report = publish(run, tmp_path / "out", {ROOT: "<volume-root>", "someone/": "<registry>/"})
    published = json.loads((tmp_path / "out" / "qualification.json").read_text())
    assert published["status"] == "measured" and published["image"] == "<registry>/verl@sha256:aa"
    assert published["evaluation"]["baseline"] == "<volume-root>/runs/r/eval/baseline"
    assert json.loads(published["stages"][0])["image"] == "<registry>/verl@sha256:aa"   # inside encoded JSON too
    entry = report["files"]["qualification.json"]
    assert entry["original_sha256"] == hashlib.sha256((run / "qualification.json").read_bytes()).hexdigest()
    assert entry["replaced"] == {"<volume-root>": 1, "<registry>/": 2}
    assert set(report["files"]) == {"qualification.json", "receipts/train-1-host.json"}
    assert json.loads((tmp_path / "out" / "SANITIZED.json").read_text()) == report


def test_a_run_without_a_record_is_not_published(tmp_path):
    with pytest.raises(FileNotFoundError, match="compare stage"):
        publish(tmp_path, tmp_path / "out", {})


def test_every_published_record_is_sanitized_whole_and_every_readme_link_resolves():
    records = REPO / "docs" / "records"
    for run in sorted(p for p in records.iterdir() if p.is_dir()):
        listed = set(json.loads((run / "SANITIZED.json").read_text())["files"])
        present = {p.relative_to(run).as_posix() for p in run.rglob("*.json") if p.name != "SANITIZED.json"}
        assert listed == present, run.name
        assert json.loads((run / "qualification.json").read_text())["status"] in ("measured", "smoke-tested")
    links = re.findall(r"\]\((docs/records/[^)]+)\)", (REPO / "README.md").read_text())
    assert links and all((REPO / link).exists() for link in links)

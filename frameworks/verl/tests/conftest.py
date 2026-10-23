"""Fixtures for adapter tests against the real pinned verl (CPU only).

They need verl importable and a cached Qwen3.5 tokenizer; otherwise they are skipped. Generation
is the only fake: a scripted server returns token ids for scripted text.
"""
from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path

import pytest

if importlib.util.find_spec("verl") is None:   # the core tests run without any framework
    collect_ignore_glob = ["test_*.py"]


def _tokenizer_dir() -> str | None:
    explicit = os.environ.get("RFT_TEST_TOKENIZER")
    if explicit:
        return explicit
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for pattern in ("models--Qwen--Qwen3.5-2B", "models--Qwen--Qwen3.5-0.8B"):
        hits = sorted(glob.glob(str(hub / pattern / "snapshots" / "*" / "tokenizer_config.json")))
        if hits:
            return str(Path(hits[-1]).parent)
    return None


@pytest.fixture(scope="session")
def tokenizer_dir():
    path = _tokenizer_dir()
    if not path:
        pytest.skip("no cached Qwen3.5 tokenizer (set RFT_TEST_TOKENIZER)")
    return path


@pytest.fixture(scope="session")
def tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir)

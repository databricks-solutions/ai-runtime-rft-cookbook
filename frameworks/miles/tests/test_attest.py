"""The retention attestation (rft_miles.attest), on samples shaped as session server v2 builds them."""
from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from rft_cookbook.agent.episode import token_mask_sha256
from rft_cookbook.evidence.sink import shared_sink
from rft_miles import attest
from rft_miles.agent import ATTESTATION_DIR_KEY, REWARD_KEY

TREE = {"nodes": [{"id": 1, "completion_span": [5, 12]}, {"id": 2, "completion_span": [20, 30]}]}


@pytest.fixture(autouse=True)
def miles_default(monkeypatch):
    """miles' default post-processing, or a stand-in (one leaf: nothing shared to mask) without miles."""
    try:
        from miles.rollout.session.v2.postprocessor_hub import default_postprocess  # noqa: F401
    except ImportError:
        hub = types.ModuleType("miles.rollout.session.v2.postprocessor_hub")
        hub.default_postprocess = lambda samples, metadata: samples
        for name in ("miles", "miles.rollout", "miles.rollout.session", "miles.rollout.session.v2"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, hub.__name__, hub)


def sample(length: int, metadata: dict) -> SimpleNamespace:
    response = length - 10
    return SimpleNamespace(tokens=list(range(length)), response_length=response, loss_mask=[1] * response,
                           rollout_log_probs=[-0.1] * response,
                           metadata={"leaf": {"node_id": 2, "path_node_ids": [1, 2]}, **metadata})


def run(tmp_path, length: int, **agent):
    agent = {REWARD_KEY: 1.0, "rft_episode_id": "e1", "rft_answered": True, "rft_terminal_turn": 1,
             ATTESTATION_DIR_KEY: str(tmp_path / "attestations"), **agent}
    kept = sample(length, {REWARD_KEY: agent[REWARD_KEY]})
    (out,) = attest.postprocess([kept], {"tree": TREE, "agent": agent})
    shared_sink(tmp_path / "attestations").flush()
    lines = [json.loads(line) for path in sorted((tmp_path / "attestations").glob("*.jsonl"))
             for line in path.read_text().splitlines()]
    return out, lines[-1]


def test_a_whole_sample_is_attested_and_keeps_its_reward(tmp_path):
    out, line = run(tmp_path, 30)
    assert out.metadata[REWARD_KEY] == 1.0
    assert line["episode_id"] == "e1" and line["schema"] == attest.SCHEMA and line["framework"] == "miles"
    assert (line["eligibility"], line["terminal_span_complete"], line["terminal_turn"]) == ("eligible", True, 1)
    assert line["tokens"] == 30 and line["loss_tokens"] == 20 and line["logprob_coverage"] == 1.0
    assert line["token_mask_sha256"] == token_mask_sha256(list(range(30)), [1] * 20)


def test_a_sample_cut_inside_its_terminal_answer_loses_its_reward(tmp_path):
    out, line = run(tmp_path, 25)   # collection cut the sequence at 25 tokens; the answer ends at 30
    assert out.metadata[REWARD_KEY] is None   # rft_miles.reward gives none: miles drops the group
    assert (line["eligibility"], line["reason"], line["terminal_span_complete"]) == ("excluded", "terminal_truncated",
                                                                                     False)


def test_an_episode_without_an_answer_keeps_its_zero_however_it_was_cut(tmp_path):
    out, line = run(tmp_path, 25, **{REWARD_KEY: 0.0, "rft_answered": False})
    assert out.metadata[REWARD_KEY] == 0.0 and line["eligibility"] == "zero"


def test_an_excluded_episode_is_attested_as_excluded(tmp_path):
    out, line = run(tmp_path, 30, **{REWARD_KEY: None, "rft_excluded": "invalid_score"})
    assert out.metadata[REWARD_KEY] is None and (line["eligibility"], line["reason"]) == ("excluded", "invalid_score")


def test_a_session_whose_agent_function_failed_passes_through(tmp_path):
    kept = sample(30, {})
    assert attest.postprocess([kept], {"tree": TREE, "agent": None}) == [kept]
    assert not (tmp_path / "attestations").exists()

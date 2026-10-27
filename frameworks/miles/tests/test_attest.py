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


def test_a_session_with_two_leaves_rewards_only_the_one_that_holds_the_terminal_answer(tmp_path):
    """A branched session, or a retried turn the picker kept: leaf 2 ends in an earlier attempt of the
    final turn, leaf 3 in the answer. miles' default post-processing trains their shared turn once."""
    tree = {"nodes": [{"id": 1, "completion_span": [12, 16]}, {"id": 2, "completion_span": [20, 30]},
                      {"id": 3, "completion_span": [20, 28]}]}   # a sample's response starts at token 10
    agent = {REWARD_KEY: 1.0, "rft_episode_id": "e2", "rft_answered": True, "rft_terminal_turn": 2,
             ATTESTATION_DIR_KEY: str(tmp_path / "attestations")}
    earlier, answer = sample(30, {REWARD_KEY: 1.0}), sample(28, {REWARD_KEY: 1.0})
    earlier.metadata["leaf"] = {"node_id": 2, "path_node_ids": [1, 2]}
    answer.metadata["leaf"] = {"node_id": 3, "path_node_ids": [1, 3]}
    out = attest.postprocess([earlier, answer], {"tree": tree, "agent": agent})
    shared_trained = sum(out[1].loss_mask[12 - 10:16 - 10])   # turn 1 in the answer's leaf
    shared_sink(tmp_path / "attestations").flush()
    lines = {json.loads(line)["sample_id"]: json.loads(line) for path in (tmp_path / "attestations").glob("*.jsonl")
             for line in path.read_text().splitlines()}
    first, last = lines["e2/leaf-2"], lines["e2/leaf-3"]
    assert (first["eligibility"], first["reason"], first["terminal_turn"]) == ("excluded", "terminal_not_in_sample",
                                                                             None)
    assert (last["eligibility"], last["terminal_span_complete"], last["terminal_turn"]) == ("eligible", True, 2)
    assert out[0].metadata[REWARD_KEY] is None and out[1].metadata[REWARD_KEY] == 1.0   # the group is dropped
    assert last["loss_tokens"] == sum(out[1].loss_mask)   # the attestation counts what trains
    assert shared_trained == (0 if _real_miles() else 4)      # miles trains the shared turn in leaf 2 only


def _real_miles() -> bool:
    """Whether miles' own post-processing ran (it masks a shared turn in all but its first leaf)."""
    from miles.rollout.session.v2.postprocessor_hub import default_postprocess
    return not getattr(default_postprocess, "__name__", "") == "<lambda>"

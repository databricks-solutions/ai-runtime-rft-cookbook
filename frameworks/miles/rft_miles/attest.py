"""The retention attestation: what each miles training sample keeps of its episode.

    --session-sample-postprocessor-path rft_miles.attest.postprocess

Session server v2 builds a session's samples after the agent function has returned (and written
the episode record), and passes them through this hook. It runs miles' default post-processing
(each shared completion trains once) and then attests every sample, as verl's loop attests its one
sample in the episode record (rft_cookbook.agent.episode.Attestation): token and loss-token counts,
the hash of tokens and loss mask together, the rollout log-probability coverage, and whether the
terminal turn's completion survived collection, which cuts a sample at the maximum sequence length.
A sample whose terminal answer was cut off or left out loses its reward, so miles drops its group
(rft_miles.reward), as verl's loop excludes a truncated terminal turn. Each attestation is written
with its episode id under the episode records (`<episode dir>/attestations`).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from rft_cookbook.agent.episode import Attestation, token_mask_sha256
from rft_cookbook.evidence.sink import shared_sink

from rft_miles.agent import ATTESTATION_DIR_KEY, REWARD_KEY

SCHEMA = "rft-cookbook/attestation/v1"


@dataclass
class AttestationLine:
    episode_id: str
    attestation: Attestation

    def to_json(self) -> str:
        return json.dumps({"schema": SCHEMA, "episode_id": self.episode_id, **asdict(self.attestation)},
                          sort_keys=True)


def postprocess(leaf_samples: list, session_metadata: dict[str, Any]) -> list:
    from miles.rollout.session.v2.postprocessor_hub import default_postprocess

    samples = default_postprocess(leaf_samples, session_metadata)
    agent = session_metadata.get("agent") or {}
    episode_id = agent.get("rft_episode_id")
    if not episode_id:
        return samples   # the agent function failed: no record to attest, and no reward (rft_miles.reward)
    nodes = {node["id"]: node for node in session_metadata["tree"]["nodes"]}
    terminal = max(nodes) if nodes else None   # node ids follow commit order: the last generation
    lines = []
    for sample in samples:
        attestation = attest(sample, nodes, terminal, agent)
        if attestation.eligibility == "excluded" and sample.metadata.get(REWARD_KEY) is not None:
            sample.metadata[REWARD_KEY] = None
        lines.append(AttestationLine(episode_id, attestation))
    sink = shared_sink(agent.get(ATTESTATION_DIR_KEY), prefix="attest", flush_after_s=30.0)
    for line in lines if sink else ():
        sink.write(line)
    return samples


def attest(sample, nodes: dict[int, dict], terminal: int | None, agent: dict[str, Any]) -> Attestation:
    leaf = sample.metadata["leaf"]
    covered = terminal is not None and terminal in leaf["path_node_ids"]
    complete = covered and nodes[terminal]["completion_span"][1] <= len(sample.tokens)
    if agent.get(REWARD_KEY) is None:
        verdict, reason = "excluded", agent.get("rft_excluded") or "no_reward"
    elif agent.get("rft_answered") and not complete:
        verdict, reason = "excluded", "terminal_truncated" if covered else "terminal_not_in_sample"
    else:
        verdict, reason = ("eligible" if agent.get("rft_answered") else "zero"), None
    logprobs = getattr(sample, "rollout_log_probs", None)
    return Attestation(
        framework="miles", sample_id=f"{agent['rft_episode_id']}/leaf-{leaf['node_id']}",
        terminal_turn=agent.get("rft_terminal_turn") if covered else None, terminal_span_complete=complete,
        tokens=len(sample.tokens), loss_tokens=int(sum(sample.loss_mask)),
        token_mask_sha256=token_mask_sha256(sample.tokens, sample.loss_mask),
        logprob_coverage=min(1.0, len(logprobs) / sample.response_length) if logprobs and sample.response_length
        else None,
        eligibility=verdict, reason=reason)

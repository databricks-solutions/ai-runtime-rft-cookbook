"""CookbookAgentLoop on the real pinned verl loop and a real Qwen3.5 tokenizer and chat template."""
from __future__ import annotations

import asyncio
import json

import pytest
from verl_support import compose_verl

from rft_cookbook.agent.errors import EpisodeExcluded
from rft_cookbook.evidence.sink import read_episodes
from rft_cookbook.tasks import load_task
from rft_verl.data import to_verl_rows

RESPONSE_LENGTH = 1024


class ScriptedServer:
    """Stands in for verl's LLM server client: replays text as token ids."""

    def __init__(self, tokenizer, turns: list[str], *, truncate_at: int | None = None):
        self.tokenizer, self.turns, self.truncate_at = tokenizer, list(turns), truncate_at
        self.requests = []

    async def generate(self, *, request_id, prompt_ids, sampling_params, **_):
        from verl.workers.rollout.replica import TokenOutput

        self.requests.append({"prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)})
        text = self.turns.pop(0)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        max_tokens = sampling_params["max_tokens"]
        stop = "completed"
        if len(ids) >= max_tokens:
            ids, stop = ids[:max_tokens], "length"
        else:
            ids = ids + [self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
        return TokenOutput(token_ids=ids, log_probs=[-0.5] * len(ids), stop_reason=stop,
                           extra_fields={"global_steps": 3, "min_global_steps": 3, "max_global_steps": 3})


def calculator_call(expression: str) -> str:
    return ("I will compute it.\n</think>\n\n<tool_call>\n<function=calculator>\n<parameter=expression>\n"
            f"{expression}\n</parameter>\n</function>\n</tool_call>")


def make_loop(tokenizer, tokenizer_dir, server, tmp_path, **budgets):
    return lambda: _make_loop(tokenizer, tokenizer_dir, server, tmp_path, **budgets)


def _make_loop(tokenizer, tokenizer_dir, server, tmp_path, entry=None, **budgets):
    """The loop as verl builds it: Hydra instantiates the registry entry (agent_loops.yaml)."""
    import hydra
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset

    config = compose_verl([
        f"actor_rollout_ref.model.path={tokenizer_dir}", "data.max_prompt_length=1024",
        f"data.max_response_length={RESPONSE_LENGTH}", "actor_rollout_ref.rollout.multi_turn.enable=True",
        "actor_rollout_ref.rollout.multi_turn.format=qwen3_coder"])
    entry = entry or {"_target_": "rft_verl.agent_loop.CookbookAgentLoop", "task": "math",
                      "budgets": {"max_response_tokens": RESPONSE_LENGTH, **budgets},
                      "episode_dir": str(tmp_path / "episodes"), "model": {"id": "Qwen/Qwen3.5-test"}}
    return hydra.utils.instantiate(
        config=OmegaConf.create(entry), trainer_config=DictConfigWrap(config), server_manager=server,
        tokenizer=tokenizer, processor=None, dataset_cls=RLHFDataset, data_config=DictConfigWrap(config.data),
        hf_model_type="qwen3_5", tools=ToolListWrap([]))


def run(make, question="What is 2+2?", ground_truth="4", system=None):
    """Build the loop inside the running event loop (verl binds it at construction), then run."""
    async def main():
        loop = make()
        return loop, await loop.run({"temperature": 1.0}, **kwargs)

    task = load_task("math")
    row = {"id": "q1", "split": "train", "messages": [{"role": "user", "content": question}],
           "ground_truth": ground_truth, "metadata": {"level": 1}}
    verl_row = next(to_verl_rows(task, [row]))
    if system is not None:
        verl_row["prompt"][0]["content"] = system
    kwargs = {"raw_prompt": verl_row["prompt"], "extra_info": verl_row["extra_info"],
              "reward_model": verl_row["reward_model"], "uid": "u1", "session_id": 0}
    return asyncio.run(main())


def episodes(tmp_path):
    from rft_cookbook.evidence.sink import _SINKS

    for sink in _SINKS.values():
        sink.flush()
    records, damaged = read_episodes(tmp_path / "episodes")
    assert damaged == 0
    return records


def test_a_tool_episode_scores_inside_the_loop(tokenizer, tokenizer_dir, tmp_path):
    server = ScriptedServer(tokenizer, [calculator_call("2+2"), "The sum is 4.\n</think>\n\n\\boxed{4}"])
    loop = make_loop(tokenizer, tokenizer_dir, server, tmp_path)
    _, output = run(loop)
    assert output.reward_score == 1.0
    assert output.extra_fields["reward_extra_info"]["rft/end_answer"] == 1.0
    # the tool observation is in the context, masked out of the loss
    tool_text = tokenizer.decode([t for t, m in zip(output.response_ids, output.response_mask) if m == 0])
    assert "4" in tool_text and "<tool_response>" in tool_text
    model_text = tokenizer.decode([t for t, m in zip(output.response_ids, output.response_mask) if m == 1])
    assert "\\boxed{4}" in model_text and "<tool_response>" not in model_text
    (record,) = episodes(tmp_path)
    assert record.answer == "4" and record.status == "ok" and record.score.valid
    call = record.turns[0].tool_calls[0]
    assert call.name == "calculator" and json.loads(call.arguments) == {"expression": "2+2"}
    assert call.delivered_text == "4" and call.delivered_to_turn == 1
    assert record.turns[0].reasoning == "I will compute it." and record.turns[0].policy_version == 3
    (attestation,) = record.collection
    assert attestation.eligibility == "eligible" and attestation.terminal_span_complete
    assert attestation.tokens == len(output.prompt_ids) + len(output.response_ids)
    assert attestation.loss_tokens == sum(output.response_mask) and attestation.logprob_coverage == 1.0
    # the policy's room, not verl's default, bounds each request
    assert server.requests[0]["sampling_params"]["max_tokens"] == min(load_task("math").budgets.max_turn_tokens,
                                                                      RESPONSE_LENGTH)
    assert server.requests[1]["sampling_params"]["max_tokens"] == RESPONSE_LENGTH - len(server.requests[1]["prompt_ids"]) \
        + len(server.requests[0]["prompt_ids"])


def test_a_wrong_answer_is_a_valid_zero(tokenizer, tokenizer_dir, tmp_path):
    loop = make_loop(tokenizer, tokenizer_dir, ScriptedServer(tokenizer, ["\\boxed{5}"]), tmp_path)
    assert run(loop)[1].reward_score == 0.0


def test_a_length_stop_is_a_valid_zero_without_an_answer(tokenizer, tokenizer_dir, tmp_path):
    loop = make_loop(tokenizer, tokenizer_dir, ScriptedServer(tokenizer, ["\\boxed{4} " + "and more " * 300]),
                     tmp_path, max_turn_tokens=64, generation_reserve=32)
    _, output = run(loop)
    assert output.reward_score == 0.0
    (record,) = episodes(tmp_path)
    assert record.termination.reason == "length" and record.answer is None


def test_an_observation_that_does_not_fit_is_not_committed(tokenizer, tokenizer_dir, tmp_path):
    server = ScriptedServer(tokenizer, [calculator_call("factorial(400)")])
    loop = make_loop(tokenizer, tokenizer_dir, server, tmp_path, max_response_tokens=120, max_turn_tokens=100,
                     generation_reserve=60)
    _, output = run(loop)
    (record,) = episodes(tmp_path)
    assert record.termination.reason == "context_budget"
    assert record.turns[0].tool_calls[0].delivery == "not_admitted"
    assert all(m == 1 for m in output.response_mask)   # no observation tokens were appended
    assert output.reward_score == 0.0


def test_an_infrastructure_failure_excludes_the_episode(tokenizer, tokenizer_dir, tmp_path):
    class DeadServer:
        async def generate(self, **_):
            raise ConnectionError("rollout replica gone")

    loop = make_loop(tokenizer, tokenizer_dir, DeadServer(), tmp_path)
    with pytest.raises(EpisodeExcluded, match="infra_failure"):
        run(loop)
    (record,) = episodes(tmp_path)
    assert record.status == "infra_failure" and record.score is None
    assert record.collection[0].eligibility == "excluded"


def test_a_row_converted_for_another_prompt_is_refused(tokenizer, tokenizer_dir, tmp_path):
    loop = make_loop(tokenizer, tokenizer_dir, ScriptedServer(tokenizer, ["\\boxed{4}"]), tmp_path)
    with pytest.raises(ValueError, match="system prompt"):
        run(loop, system="An older prompt.")


def test_budgets_beyond_verls_response_length_are_refused(tokenizer, tokenizer_dir, tmp_path):
    from hydra.errors import InstantiationException

    with pytest.raises(InstantiationException, match="response_length"):   # as verl's worker sees it
        run(make_loop(tokenizer, tokenizer_dir, ScriptedServer(tokenizer, []), tmp_path,
                      max_response_tokens=RESPONSE_LENGTH + 1))


def test_the_verl_schema_view_is_what_verls_tool_loop_renders():
    from verl.tools.schemas import OpenAIFunctionToolSchema

    from rft_cookbook.agent.spec import verl_property_subset
    from rft_cookbook.tasks.officeqa import tool_schemas

    for schema in tool_schemas():
        rendered = OpenAIFunctionToolSchema.model_validate(schema).model_dump(exclude_unset=True, exclude_none=True)
        assert json.dumps(verl_property_subset(schema)) == json.dumps(rendered)
    (report,) = [s for s in tool_schemas() if s["function"]["name"] == "submit_report"]
    assert "items" in report["function"]["parameters"]["properties"]["path"]
    assert "items" not in verl_property_subset(report)["function"]["parameters"]["properties"]["path"]


def test_registry_values_reach_the_task_as_plain_data(tokenizer, tokenizer_dir, tmp_path, monkeypatch):
    """Hydra hands nested registry values over as DictConfig; the task gets them as plain JSON."""
    from rft_verl import agent_loop

    seen = []
    real = agent_loop._task.__wrapped__
    monkeypatch.setattr(agent_loop, "_task", lambda name, budgets, resources: seen.append(json.loads(resources))
                        or real(name, budgets, "null"))
    resources = {"tool_service": "run-1", "judge": {"models": ["m"], "concurrency": 4}, "compute": {"mem_mb": 4096}}
    entry = {"_target_": "rft_verl.agent_loop.CookbookAgentLoop", "task": "math",
             "budgets": {"max_response_tokens": RESPONSE_LENGTH}, "episode_dir": str(tmp_path / "episodes"),
             "model": {"id": "Qwen/Qwen3.5-test"}, "resources": resources}
    loop = _make_loop(tokenizer, tokenizer_dir, ScriptedServer(tokenizer, []), tmp_path, entry=entry)
    assert seen == [resources] and loop.model == {"id": "Qwen/Qwen3.5-test"}



def test_failures_in_a_row_ask_the_launcher_to_stop(tokenizer, tokenizer_dir, tmp_path, monkeypatch):
    from hydra.errors import InstantiationException

    from rft_verl import agent_loop

    class DeadServer:
        async def generate(self, **_):
            raise ConnectionError("rollout replica gone")

    monkeypatch.setattr(agent_loop, "_FAILED_IN_A_ROW", [0])
    abort = tmp_path / "train" / "ABORT.json"
    entry = {"_target_": "rft_verl.agent_loop.CookbookAgentLoop", "task": "math",
             "budgets": {"max_response_tokens": RESPONSE_LENGTH}, "episode_dir": str(tmp_path / "episodes"),
             "abort_file": str(abort), "max_consecutive_exclusions": 3}
    with pytest.raises(InstantiationException):   # a loop that cannot be built counts too
        _make_loop(tokenizer, tokenizer_dir, DeadServer(), tmp_path,
                   entry={**entry, "budgets": {"max_response_tokens": RESPONSE_LENGTH + 1}})
    with pytest.raises(EpisodeExcluded):
        run(lambda: _make_loop(tokenizer, tokenizer_dir, DeadServer(), tmp_path, entry=entry))
    assert not abort.exists()
    with pytest.raises(EpisodeExcluded):
        run(lambda: _make_loop(tokenizer, tokenizer_dir, DeadServer(), tmp_path, entry=entry))
    assert "3 episodes in a row failed" in json.loads(abort.read_text())["reason"]

    abort.unlink()
    monkeypatch.setattr(agent_loop, "_FAILED_IN_A_ROW", [2])   # a prompt with no room for a turn counts too
    with pytest.raises(EpisodeExcluded, match="no room"):
        run(lambda: _make_loop(tokenizer, tokenizer_dir, DeadServer(), tmp_path,
                               entry={**entry, "budgets": {**entry["budgets"], "max_context_tokens": 128}}))
    assert "no room for a turn" in json.loads(abort.read_text())["reason"]
    assert any(r.termination and r.termination.reason == "context_budget" for r in episodes(tmp_path))   # recorded

    abort.unlink()
    monkeypatch.setattr(agent_loop, "_FAILED_IN_A_ROW", [2])
    run(lambda: _make_loop(tokenizer, tokenizer_dir, ScriptedServer(tokenizer, ["\\boxed{4}"]), tmp_path, entry=entry))
    assert agent_loop._FAILED_IN_A_ROW == [0] and not abort.exists()   # a scored episode resets the count

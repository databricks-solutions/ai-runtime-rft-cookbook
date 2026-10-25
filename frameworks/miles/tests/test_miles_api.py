"""The adapter against the pinned miles itself (tier 3 of docs/architecture.md "Testing"): every
hook loads with miles' own loader and fits the call miles makes, and every flag the recipes pass
is one miles (or Megatron) defines. Runs where the pinned miles source is importable (CI's
miles-api job); the rest of the adapter's tests need no miles.

miles' generate, matcher and argument modules import SGLang, so their call sites are read from
the pinned source instead of being imported."""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
miles = pytest.importorskip("miles.utils.function_registry")

from miles.utils.function_registry import load_function  # noqa: E402

from rft_miles import render  # noqa: E402

SOURCE = Path(miles.__file__).resolve().parents[1]   # the pinned miles package
# Megatron's own arguments, which miles passes through: the only ones miles' parser does not define
MEGATRON = {"--global-batch-size", "--load", "--lr", "--save", "--save-interval", "--seed",
            "--use-checkpoint-opt-param-scheduler"}


def _calls(relative: str, name: str) -> list[ast.Call]:
    tree = ast.parse((SOURCE / relative).read_text())
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and (getattr(node.func, "id", None) == name or getattr(node.func, "attr", None) == name)]


def _binds(fn, call: ast.Call, *, drop: tuple[str, ...] = ()) -> None:
    """`fn` accepts the arguments of `call` (positional count, keyword names)."""
    keywords = {k.arg: None for k in call.keywords if k.arg and k.arg not in drop}
    inspect.signature(fn).bind(*([None] * len(call.args)), **keywords)


def test_the_hooks_fit_the_calls_the_pinned_miles_makes():
    (save,) = _calls("backends/megatron_utils/actor.py", "post_save_hook")
    _binds(load_function(render.POST_SAVE), save)                    # (args, rollout_id, checkpoint_dir, hf_dir)
    (step,) = _calls("backends/megatron_utils/model.py", "custom_before_train_step_hook")
    _binds(load_function(render.DIAGNOSTICS), step)
    (agent,) = _calls("rollout/generate_hub/agentic_tool_call.py", "call_agent_function")
    _binds(load_function(render.AGENT), ast.Call(func=agent.func, args=[], keywords=agent.keywords),
           drop=("mode", "on_child_reports"))                       # call_agent_function's own; fn(**the rest)
    (match,) = _calls("rollout/session/linear_trajectory.py", "message_matcher")
    _binds(load_function(render.MATCHER), match)
    # the session server loads its post-processor for a synchronous call
    postprocess = load_function(render.POSTPROCESS, sync_required=True)
    assert list(inspect.signature(postprocess).parameters) == ["leaf_samples", "session_metadata"]


def test_the_reward_hook_answers_through_miles_reward_dispatch():
    from miles.rollout.rm_hub import async_rm, batched_async_rm
    from miles.utils.types import Sample

    from rft_miles.agent import REWARD_KEY

    args = SimpleNamespace(custom_rm_path=render.REWARD, rm_type=None)
    scored, excluded = Sample(metadata={REWARD_KEY: 1.0}), Sample(metadata={REWARD_KEY: None})
    assert asyncio.run(async_rm(args, scored)) == 1.0
    assert asyncio.run(batched_async_rm(args, [scored, excluded])) == [1.0, None]   # a session's samples together


def test_the_postprocessor_runs_on_miles_samples(tmp_path):
    from miles.utils.types import Sample

    from rft_miles import attest
    from rft_miles.agent import ATTESTATION_DIR_KEY, REWARD_KEY

    kept = Sample(tokens=list(range(30)), response_length=20, loss_mask=[1] * 20, rollout_log_probs=[-0.1] * 20,
                  metadata={"leaf": {"node_id": 2, "path_node_ids": [1, 2]}})
    agent = {REWARD_KEY: 1.0, "rft_episode_id": "e1", "rft_answered": True, "rft_terminal_turn": 1,
             ATTESTATION_DIR_KEY: str(tmp_path)}
    tree = {"nodes": [{"id": 1, "completion_span": [12, 16]}, {"id": 2, "completion_span": [20, 30]}]}
    (out,) = attest.postprocess([kept], {"tree": tree, "agent": agent})
    assert out.metadata[REWARD_KEY] == 1.0 and out.metadata["rft_episode_id"] == "e1"   # miles merged the agent's


def _defined_flags() -> set[str]:
    flags = set()
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
                flags |= {a.value for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")}
    return flags


@pytest.mark.parametrize("recipe", sorted(str(p.parent.relative_to(p.parents[3])) for p in
                                          Path(__file__).resolve().parents[3].glob("recipes/*/miles/*/recipe.yaml")))
def test_every_flag_a_recipe_passes_is_one_miles_or_megatron_defines(recipe, tmp_path):
    render.resolve_run(recipe, site={"volume_root": "/Volumes/cat/sch/vol/rft"}, run_id="api-check", overrides=[],
                       out=tmp_path)
    passed = {a.split("=")[0] for a in json.loads((tmp_path / "miles.json").read_text())["extra_args"]
              if a.startswith("--")}
    defined = _defined_flags()
    unknown = {flag for flag in passed - defined - MEGATRON
               if not (flag.startswith("--no-") and f"--{flag[5:]}" in defined)   # a boolean option's negation
               and not flag.startswith("--sglang-")}                               # from SGLang's own arguments
    assert not unknown, f"{recipe} passes flags the pinned miles does not define: {sorted(unknown)}"

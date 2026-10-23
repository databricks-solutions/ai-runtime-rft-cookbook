"""Resolving a verl recipe: every verl override composes against verl's real Hydra config."""
from __future__ import annotations

import json

import pytest
import yaml
from verl_support import compose_verl

from rft_verl.render import resolve_run

SITE = {"volume_root": "/Volumes/cat/sch/vol/rft"}
RECIPE = "math/verl/qwen3.5-2b_h100x8"
OFFICEQA = "officeqa/verl/qwen3.5-35b-a3b_h100x16"


def resolve(tmp_path, ref=RECIPE, run_id="math-test-01", overrides=()):
    return resolve_run(ref, site=SITE, run_id=run_id, overrides=list(overrides), out=tmp_path / "runs" / run_id)


@pytest.fixture
def rendered(tmp_path):
    return resolve(tmp_path)


def test_the_rendered_arguments_compose_with_the_pinned_verl(rendered):
    args = [a.replace("@RENDERED@", str(rendered)) for a in json.loads((rendered / "verl_args.json").read_text())]
    config = compose_verl(args)
    assert config.trainer.use_v1 and config.trainer.v1.trainer_mode == "separate_async"
    assert config.actor_rollout_ref.hybrid_engine is False
    assert config.data.train_batch_size == (config.trainer.v1.separate_async.parameter_sync_step
                                            * config.actor_rollout_ref.actor.ppo_mini_batch_size)
    assert config.actor_rollout_ref.rollout.checkpoint_engine.backend == "nccl"
    assert config.trainer.n_gpus_per_node + config.actor_rollout_ref.rollout.n_gpus_per_node == 8
    assert config.trainer.checkpoint_callback_class == "rft_verl.checkpoint.PublishCheckpoint"
    assert config.rft.checkpoint_store == "/Volumes/cat/sch/vol/rft/runs/math-test-01/checkpoints"
    assert config.actor_rollout_ref.rollout.response_length == 4096
    assert config.actor_rollout_ref.rollout.max_model_len == 1024 + 4096
    assert config.actor_rollout_ref.actor.checkpoint.async_save is False


def test_the_agent_loop_registry_and_plan(rendered):
    (entry,) = yaml.safe_load((rendered / "agent_loops.yaml").read_text())
    assert entry["_target_"] == "rft_verl.agent_loop.CookbookAgentLoop" and entry["task"] == "math"
    assert entry["episode_dir"].endswith("/runs/math-test-01/episodes/train")
    plan = json.loads((rendered / "plan.json").read_text())
    assert plan["model_dir"] == "/Volumes/cat/sch/vol/rft/models/Qwen3.5-2B@15852e8c1636"
    assert plan["local_dir"].startswith("/local_disk0/")


@pytest.mark.parametrize("override,message", [
    ("compute.trainer_gpus=6", "must equal 8"),
    ("train.save_freq=-1", "save_freq"),
    ("model.revision=main", "40-hex"),
])
def test_inconsistent_recipes_do_not_resolve(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        resolve(tmp_path, run_id="math-test-02", overrides=[override])


def test_a_job_resolves_its_run_from_its_environment(tmp_path, monkeypatch):
    """A checked-in job names its recipe; `air run --override` gives the run id and volume root."""
    from rft_verl import launch

    monkeypatch.setenv("RFT_RECIPE", RECIPE)
    monkeypatch.setenv("RFT_RUN_ID", "<run-id>")
    monkeypatch.setenv("RFT_VOLUME_ROOT", "<volume-root>")
    with pytest.raises(SystemExit, match="env_variables.RFT_RUN_ID=... env_variables.RFT_VOLUME_ROOT=..."):
        launch.main(["prep"])
    seen = {}
    monkeypatch.setenv("RFT_RUN_ID", "math-test-03")
    monkeypatch.setenv("RFT_VOLUME_ROOT", SITE["volume_root"])
    monkeypatch.setenv("RFT_SET", "train.seed=7 'eval.temperature=0.5'")
    monkeypatch.setenv("RFT_RUN_DIR", str(tmp_path / "job"))
    monkeypatch.setitem(launch.STAGES, "prep", lambda recipe, paths, run: seen.update(
        run=run, id=paths["run_id"], seed=recipe["train"]["seed"], temperature=recipe["eval"]["temperature"]))
    assert launch.main(["prep"]) == 0
    assert seen == {"run": tmp_path / "job" / "math-test-03", "id": "math-test-03", "seed": 7, "temperature": 0.5}


def test_officeqa_renders_its_evaluation_gate(tmp_path):
    """The OfficeQA recipe renders the parity gate before training: staging, judge calibration, the
    base model's evaluation on one node with its tool service and judge, and the comparison with
    the reference; training takes both nodes."""
    from rft_verl.launch import resources

    out = resolve(tmp_path, OFFICEQA, "officeqa-test-01")
    recipe = yaml.safe_load((out / "recipe.yaml").read_text())
    assert recipe["stages"][:7] == ["prep", "stage_model", "stage_judge", "calibrate", "baseline_eval", "parity", "train"]
    reference = json.loads((out / "parity_reference.json").read_text())
    assert sum(r["split"] == "heldout" for r in reference["per_question"].values()) == 43
    plan = json.loads((out / "plan.json").read_text())
    assert plan["judge_tokenizer_dir"].endswith("/models/GLM-5.3-Flash@eb9eb208eb0d-tokenizer")
    assert plan["baseline_dir"] == f"{plan['eval_dir']}/baseline"
    task = resources(recipe, plan)
    assert task["tool_service"] == "officeqa-test-01" and task["data_dir"].endswith("/corpus")
    assert task["sandbox"] == "strict" and task["judge"]["models"] == ["system.ai.glm-5-3-flash"]
    assert task["judge"]["tokenizer"] == plan["judge_tokenizer_dir"]


def test_two_nodes_give_the_trainer_and_the_rollout_one_node_each(tmp_path):
    out = resolve(tmp_path, OFFICEQA, "officeqa-test-02")
    config = compose_verl([a.replace("@RENDERED@", str(out)) for a in json.loads((out / "verl_args.json").read_text())])
    assert (config.trainer.nnodes, config.trainer.n_gpus_per_node) == (1, 8)
    rollout = config.actor_rollout_ref.rollout
    assert (rollout.nnodes, rollout.n_gpus_per_node, rollout.tensor_model_parallel_size) == (1, 8, 8)
    megatron = config.actor_rollout_ref.actor.megatron
    assert (megatron.tensor_model_parallel_size, megatron.context_parallel_size, megatron.expert_model_parallel_size) == (2, 4, 8)
    assert config.actor_rollout_ref.model.use_remove_padding is True
    assert config.actor_rollout_ref.model.use_fused_kernels is True
    assert config.actor_rollout_ref.rollout.max_model_len == 2048 + 96_256
    # a KL loss to the base model: the reference shares the trainer GPUs and the actor's parallelism
    actor, ref = config.actor_rollout_ref.actor, config.actor_rollout_ref.ref
    assert actor.use_kl_loss and actor.kl_loss_coef == 0.01
    assert (ref.megatron.tensor_model_parallel_size, ref.megatron.context_parallel_size) == (2, 4)
    assert ref.log_prob_micro_batch_size_per_gpu == 1 and ref.log_prob_max_token_len_per_gpu == 98_304
    # verl saves straight into the store, deletes nothing, and the callback keeps the newest two
    plan = json.loads((out / "plan.json").read_text())
    assert config.trainer.default_local_dir == plan["checkpoints"] == config.rft.checkpoint_store
    assert config.trainer.max_actor_ckpt_to_keep is None and config.rft.checkpoint_keep == 2
    assert config.rft.checkpoint_hash is False
    # every agent-loop worker gets its share of the judge's concurrency
    (loop,) = yaml.safe_load((out / "agent_loops.yaml").read_text())
    assert loop["resources"]["judge"]["concurrency"] == 16 // config.actor_rollout_ref.rollout.agent.num_workers
    with pytest.raises(ValueError, match="checkpoint.location: volume"):
        resolve(tmp_path, OFFICEQA, "officeqa-test-04", ["checkpoint.location=local"])
    with pytest.raises(ValueError, match="whole nodes"):
        resolve(tmp_path, OFFICEQA, "officeqa-test-03", ["compute.trainer_gpus=12", "compute.rollout_gpus=4"])


def test_a_replicate_trains_with_its_own_seed_against_another_runs_baseline(tmp_path):
    out = resolve(tmp_path, OFFICEQA, "officeqa-test-05",
                  ["train.seed=7", "eval.seed=42", "eval.baseline_run=officeqa-test-01"])
    plan = json.loads((out / "plan.json").read_text())
    assert plan["baseline_dir"] == "/Volumes/cat/sch/vol/rft/runs/officeqa-test-01/eval/baseline"
    assert "data.seed=7" in json.loads((out / "verl_args.json").read_text())
    assert yaml.safe_load((out / "recipe.yaml").read_text())["eval"]["seed"] == 42

"""PublishCheckpoint, loaded the way verl's V1 trainer loads it, on a verl-shaped save."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from verl_support import compose_verl

from rft_cookbook.checkpoints import CheckpointStore
from rft_verl.checkpoint import resume_overrides, verl_bundle


def fake_save(root: Path, step: int) -> Path:
    """The layout verl 8718ca30 writes for a Megatron actor with Bridge HF weights."""
    folder = root / f"global_step_{step}"
    actor = folder / "actor"
    hf = actor / "model" / "huggingface"
    hf.mkdir(parents=True)
    (actor / "optimizer" / "dist_ckpt").mkdir(parents=True)
    (actor / "optimizer" / "dist_ckpt" / "__0_0.distcp").write_bytes(b"o" * 64)
    (actor / "extra" / "dist_ckpt").mkdir(parents=True)
    (actor / "extra" / "dist_ckpt" / "__0_0.distcp").write_bytes(b"r" * 8)
    (hf / "config.json").write_text("{}")
    (hf / "model.safetensors").write_bytes(b"w" * 128)
    (actor / "ckpt_contents.json").write_text(json.dumps({
        "schema_version": 2, "save_contents": ["model", "optimizer", "extra"], "world_size": 4,
        "contents": {"model": {"path": "model/huggingface", "format": "huggingface"},
                     "optimizer": {"path": "optimizer/dist_ckpt"}, "rng_state": {"path": "extra/dist_ckpt"}}}))
    (folder / "data.pt").write_bytes(b"d")
    (folder / "transfer_queue").mkdir()
    (folder / "transfer_queue" / "state.pt").write_bytes(b"q")
    return folder


def callback(store: Path, *extra: str):
    from verl.trainer.ppo.checkpoint_callback import build_checkpoint_callback

    config = compose_verl(["trainer.checkpoint_callback_class=rft_verl.checkpoint.PublishCheckpoint",
                           "trainer.v1.trainer_mode=separate_async", f"+rft.checkpoint_store={store}", *extra])
    return build_checkpoint_callback(config)


def test_verl_builds_the_callback_and_it_publishes_a_resumable_bundle(tmp_path):
    hook = callback(tmp_path / "volume")
    folder = fake_save(tmp_path / "local", 2)
    hook.on_save(trainer=None, global_step=2, checkpoint_dir=str(folder), async_save=False)
    published = CheckpointStore(tmp_path / "volume").latest()
    assert published.name == "global_step_2"
    assert set(published.manifest["bundle"]["resumes"]) == {"model", "data_position", "rollout_state", "optimizer", "rng"}
    assert (Path(published.location) / "transfer_queue" / "state.pt").exists()
    assert CheckpointStore(tmp_path / "volume").export(published)["weights"] == ["model.safetensors"]
    assert resume_overrides(published.location)[1] == f"trainer.resume_from_path={published.location}"


def test_an_unfinished_save_stops_the_run(tmp_path):
    hook = callback(tmp_path / "volume")
    folder = fake_save(tmp_path / "local", 2)
    (folder / "actor" / "ckpt_contents.json").unlink()
    with pytest.raises(RuntimeError, match="completion marker"):
        hook.on_save(trainer=None, global_step=2, checkpoint_dir=str(folder))


def test_async_mode_without_transfer_queue_state_is_incomplete(tmp_path):
    folder = fake_save(tmp_path / "local", 2)
    import shutil

    shutil.rmtree(folder / "transfer_queue")
    with pytest.raises(Exception, match="transfer_queue"):
        verl_bundle(str(folder), 2, trainer_mode="separate_async").check_local()
    verl_bundle(str(folder), 2, trainer_mode="sync").check_local()


def test_asynchronous_saves_are_refused(tmp_path):
    with pytest.raises(ValueError, match="synchronous"):
        callback(tmp_path / "volume", "actor_rollout_ref.actor.checkpoint.async_save=True")
    hook = callback(tmp_path / "volume")
    with pytest.raises(RuntimeError, match="not durable"):
        hook.on_save(trainer=None, global_step=2, checkpoint_dir=str(tmp_path), async_save=True)


def test_a_save_into_the_store_is_published_in_place_and_retained(tmp_path):
    hook = callback(tmp_path / "volume", "+rft.checkpoint_keep=2", "+rft.checkpoint_hash=False")
    for step in (25, 50, 75):
        folder = fake_save(tmp_path / "volume", step)
        hook.on_save(trainer=None, global_step=step, checkpoint_dir=str(folder), async_save=False)
    store = CheckpointStore(tmp_path / "volume")
    assert [p.step for p in store.list()] == [50, 75]
    assert not (tmp_path / "volume" / "global_step_25").exists()
    assert store.latest().location == str(tmp_path / "volume" / "global_step_75")

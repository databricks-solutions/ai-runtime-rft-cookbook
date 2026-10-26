"""CheckpointStore: publish whole bundles only, find the latest complete one, verify, restore, export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rft_cookbook.checkpoints import Artifact, CheckpointBundle, CheckpointError, CheckpointStore
from rft_cookbook.checkpoints.store import COMPLETE


def native_save(root: Path, step: int, *, marker: bool = True, shards: int = 2) -> CheckpointBundle:
    """A verl-shaped save: actor (with its completion marker and HF weights) plus data.pt."""
    folder = root / f"global_step_{step}"
    hf = folder / "actor" / "model" / "huggingface"
    hf.mkdir(parents=True)
    (folder / "actor" / "optim").mkdir()
    (folder / "actor" / "optim" / "shard_0.pt").write_bytes(b"o" * 100)
    (hf / "config.json").write_text(json.dumps({"model_type": "toy"}))
    names = [f"model-{i:05d}-of-{shards:05d}.safetensors" for i in range(1, shards + 1)]
    for name in names:
        (hf / name).write_bytes(name.encode() * 10)
    (hf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{i}": n for i, n in enumerate(names)}}))
    if marker:
        (folder / "actor" / "ckpt_contents.json").write_text("{}")
    (folder / "data.pt").write_bytes(b"d" * 10)
    return CheckpointBundle(
        framework="verl", name=folder.name, step=step, root=str(folder),
        artifacts=[Artifact("actor", "actor", marker="ckpt_contents.json"), Artifact("data", "data.pt"),
                   Artifact("transfer_queue", "transfer_queue", required=False)],
        resumes=["model", "optimizer", "rng", "data_position"], hf_path="actor/model/huggingface")


def test_publish_then_latest_verify_export_restore(tmp_path):
    store = CheckpointStore(tmp_path / "volume")
    for step in (2, 4):
        result = store.publish(native_save(tmp_path / "local", step))
        assert result.status == "published", result.reason
    latest = store.latest()
    assert latest.step == 4 and latest.name == "global_step_4"
    store.verify(latest)
    export = store.export(latest)
    assert len(export["weights"]) == 2 and len(export["weights_sha256"]) == 64
    restored = store.restore(latest, tmp_path / "restore")
    assert (restored / "actor" / "ckpt_contents.json").exists() and (restored / "data.pt").read_bytes() == b"d" * 10


def test_a_save_without_its_native_marker_is_not_published(tmp_path):
    store = CheckpointStore(tmp_path / "volume")
    result = store.publish(native_save(tmp_path / "local", 2, marker=False))
    assert result.status == "failed" and "completion marker" in result.reason
    assert store.latest() is None


def test_a_missing_required_artifact_fails_the_publish(tmp_path):
    bundle = native_save(tmp_path / "local", 2)
    (Path(bundle.root) / "data.pt").unlink()
    assert CheckpointStore(tmp_path / "volume").publish(bundle).status == "failed"


def test_an_unfinished_copy_is_ignored_and_replaced(tmp_path):
    store = CheckpointStore(tmp_path / "volume")
    partial = tmp_path / "volume" / "global_step_2"
    partial.mkdir(parents=True)
    (partial / "junk").write_text("left by a crashed copy")
    assert store.latest() is None
    assert store.publish(native_save(tmp_path / "local", 2)).status == "published"
    assert not (partial / "junk").exists() and store.latest().step == 2


def test_corruption_after_publish_is_detected(tmp_path):
    store = CheckpointStore(tmp_path / "volume")
    store.publish(native_save(tmp_path / "local", 2))
    published = store.latest()
    shard = Path(published.location) / "actor" / "optim" / "shard_0.pt"
    shard.write_bytes(b"x" * 100)   # same size, different bytes
    with pytest.raises(CheckpointError, match="hash differs"):
        store.verify(published)
    shard.write_bytes(b"x")
    with pytest.raises(CheckpointError, match="size differs"):
        store.verify(published, hashes=False)


def test_an_incomplete_hf_export_is_rejected(tmp_path):
    store = CheckpointStore(tmp_path / "volume")
    store.publish(native_save(tmp_path / "local", 2))
    published = store.latest()
    next(Path(published.hf_path).glob("*.safetensors")).unlink()
    with pytest.raises(CheckpointError, match="missing shards"):
        store.export(published)


def test_publishing_the_same_step_twice_is_refused(tmp_path):
    store = CheckpointStore(tmp_path / "volume")
    store.publish(native_save(tmp_path / "a", 2))
    assert "already published" in store.publish(native_save(tmp_path / "b", 2)).reason


def test_retention_keeps_the_newest_and_the_pinned(tmp_path):
    store = CheckpointStore(tmp_path / "volume", hash_files=False)
    for step in (1, 2, 3, 4):
        store.publish(native_save(tmp_path / "local", step))
    assert store.retain(keep_last=1, keep_steps={2}) == [1, 3]
    assert [p.step for p in store.list()] == [2, 4]
    assert not any((tmp_path / "volume" / f"global_step_{s}" / COMPLETE).exists() for s in (1, 3))


def test_incomplete_hf_weights_are_recorded_at_publish_and_retention_keeps_the_last_complete_ones(tmp_path):
    store = CheckpointStore(tmp_path / "volume", hash_files=False)
    for step in (1, 2, 3):
        bundle = native_save(tmp_path / "local", step)
        if step > 1:   # the framework's export broke after the first save; the bundles still resume
            next((Path(bundle.root) / bundle.hf_path).glob("*.safetensors")).unlink()
        result = store.publish(bundle)
        assert result.status == "published" and bool(result.hf_problems) is (step > 1)
    assert [p.exportable for p in store.list()] == [True, False, False]
    assert store.retain(keep_last=1) == [2]   # step 1 is kept: no newer bundle has complete weights
    assert [p.step for p in store.list()] == [1, 3]


def test_the_export_hash_covers_the_weights_content_when_the_bundle_was_published_without_hashes(tmp_path):
    def exported(root: Path, hash_files: bool, fill: bytes) -> dict:
        store = CheckpointStore(root, hash_files=hash_files)
        bundle = native_save(root, 100)
        for shard in (Path(bundle.root) / "actor" / "model" / "huggingface").glob("*.safetensors"):
            shard.write_bytes(fill * shard.stat().st_size)   # same names and sizes, other weights
        assert store.publish(bundle).status == "published"
        return store.export(store.latest())

    one, other = exported(tmp_path / "a", False, b"a"), exported(tmp_path / "b", False, b"b")
    assert one["weights_sha256"] != other["weights_sha256"]
    # hashed when published or when exported: the same weights give the same hash
    assert exported(tmp_path / "c", True, b"a")["weights_sha256"] == one["weights_sha256"]


def test_the_store_records_its_retention(tmp_path):
    store = CheckpointStore(tmp_path / "volume")
    assert store.retention() is None
    store.set_retention(2)
    assert CheckpointStore(tmp_path / "volume").retention() == 2
    store.set_retention(None)
    assert store.retention() is None
    with pytest.raises(ValueError):
        store.set_retention(0)


def test_a_save_inside_the_store_is_published_in_place(tmp_path):
    store = CheckpointStore(tmp_path / "volume", hash_files=False)
    bundle = native_save(tmp_path / "volume", 25)
    weights = Path(bundle.root) / "actor" / "model" / "huggingface" / "model-00001-of-00002.safetensors"
    inode = weights.stat().st_ino
    result = store.publish(bundle)
    assert result.status == "published" and result.files == 7
    published = store.latest()
    assert published.location == bundle.root and weights.stat().st_ino == inode   # nothing copied
    store.verify(published)
    assert store.export(published)["weights"] == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    assert store.publish(native_save(tmp_path / "volume", 50)).status == "published"
    assert store.retain(1) == [25] and not Path(bundle.root).exists()

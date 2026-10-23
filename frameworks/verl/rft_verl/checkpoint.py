"""Publish every verl V1 checkpoint through the cookbook's CheckpointStore.

    trainer.checkpoint_callback_class=rft_verl.checkpoint.PublishCheckpoint
    +rft.checkpoint_store=/Volumes/<catalog>/<schema>/<volume>/<run>/checkpoints

verl calls `on_save` on the driver after `_save_checkpoint`. With synchronous saves (the only
mode this callback accepts) every worker has finished writing and verl has written its own
`latest_checkpointed_iteration.txt`; the actor's `ckpt_contents.json`, written last by rank 0,
is the native completion marker. The bundle adds the driver's `data.pt` and, in async trainer
modes, the TransferQueue state that resume needs. A failed publish stops the run unless
`+rft.checkpoint_publish_failure=warn`.

When verl saves straight into the store (`trainer.default_local_dir` is the store, as on several
nodes, where the trainer may run on another node than the driver) the bundle is published in
place; the marker, written on the trainer's node, is waited for until `+rft.checkpoint_visible_s`.
`+rft.checkpoint_keep=N` then keeps the newest N published bundles; verl itself deletes none.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback

from rft_cookbook.checkpoints import Artifact, CheckpointBundle, CheckpointStore

logger = logging.getLogger(__name__)

ACTOR_MARKER = "ckpt_contents.json"


def verl_bundle(checkpoint_dir: str, step: int, *, trainer_mode: str) -> CheckpointBundle:
    root = Path(checkpoint_dir)
    artifacts = [Artifact("actor", "actor", marker=ACTOR_MARKER), Artifact("data", "data.pt")]
    resumes = ["model", "data_position"]
    if trainer_mode != "sync":
        artifacts.append(Artifact("transfer_queue", "transfer_queue"))
        resumes.append("rollout_state")
    if (root / "critic").exists():
        artifacts.append(Artifact("critic", "critic", marker=ACTOR_MARKER))
    hf_path, manifest = None, {}
    marker = root / "actor" / ACTOR_MARKER
    if marker.is_file():
        manifest = json.loads(marker.read_text())
        contents = manifest.get("contents", {})
        hf = contents.get("hf_model") or contents.get("model")
        if hf and hf.get("format") == "huggingface":
            hf_path = str(Path("actor") / hf["path"])
        if "optimizer" in contents:
            resumes.append("optimizer")
        if "rng_state" in contents:
            resumes.append("rng")
    return CheckpointBundle(framework="verl", name=root.name, step=int(step), root=str(root), artifacts=artifacts,
                            resumes=resumes, hf_path=hf_path,
                            metadata={"save_contents": manifest.get("save_contents"),
                                      "world_size": manifest.get("world_size")})


def resume_overrides(published_location: str) -> list[str]:
    """verl arguments that resume from a published bundle (a native `global_step_N` directory)."""
    if "global_step_" not in Path(published_location).name:
        raise ValueError(f"{published_location} is not a verl global_step_N directory")
    return ["trainer.resume_mode=resume_path", f"trainer.resume_from_path={published_location}"]


class PublishCheckpoint(CheckpointCallback):
    def __init__(self, config=None):
        super().__init__(config)
        rft = (config.get("rft") if config is not None else None) or {}
        if not rft.get("checkpoint_store"):
            raise ValueError("PublishCheckpoint needs +rft.checkpoint_store=<durable directory>")
        if config.actor_rollout_ref.actor.get("checkpoint", {}).get("async_save", False):
            raise ValueError("PublishCheckpoint supports synchronous saves only (checkpoint.async_save=False)")
        self.store = CheckpointStore(rft["checkpoint_store"], hash_files=bool(rft.get("checkpoint_hash", True)))
        self.on_failure = rft.get("checkpoint_publish_failure", "fail")
        self.trainer_mode = config.trainer.get("v1", {}).get("trainer_mode", "sync")
        self.keep = rft.get("checkpoint_keep")
        self.visible_s = float(rft.get("checkpoint_visible_s", 300))
        self.results = []

    def on_save(self, trainer, global_step: int, checkpoint_dir: str, async_save: bool = False, **kwargs) -> None:
        if async_save:
            raise RuntimeError("an asynchronous save is not durable when on_save fires; refusing to publish")
        marker = Path(checkpoint_dir) / "actor" / ACTOR_MARKER
        in_place = Path(checkpoint_dir).resolve().parent == self.store.root.resolve()
        deadline = time.monotonic() + (self.visible_s if in_place else 0)
        while not marker.is_file() and time.monotonic() < deadline:
            time.sleep(5)
        bundle = verl_bundle(checkpoint_dir, global_step, trainer_mode=self.trainer_mode)
        result = self.store.publish(bundle)
        self.results.append(result)
        logger.warning("rft checkpoint %s: %s in %.1fs (%d files, %.2f GB) -> %s %s", bundle.name, result.status,
                       result.seconds, result.files, result.bytes / 1e9, result.location, result.reason or "")
        if result.status != "published" and self.on_failure != "warn":
            raise RuntimeError(f"publishing {bundle.name} failed: {result.reason}")
        if result.status == "published" and self.keep:
            deleted = self.store.retain(int(self.keep))
            if deleted:
                logger.warning("rft checkpoint retention: deleted steps %s", deleted)

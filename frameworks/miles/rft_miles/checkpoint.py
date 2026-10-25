"""Publish every miles (Megatron) checkpoint through the cookbook's CheckpointStore.

    --save <run>/checkpoints --custom-megatron-post-save-hook-path rft_miles.checkpoint.publish

miles calls the hook on rank 0 after `save(rollout_id, ...)` has finished, with Megatron's
`iter_<rollout_id>` directory under `--save`. `--save` is the store itself, so the bundle is
published in place: nothing is copied; the manifest and completion marker are added to the
native directory. Megatron's `metadata.json`, written when the distributed save completes, is the
native completion marker. The bundle's step counts updates (`rollout_id + 1`), as verl's
`global_step_N` does. A failed publish stops the run; incomplete Hugging Face weights are reported
and recorded, not fatal. After a publish the hook keeps the newest bundles the store's retention
names (`CheckpointStore.set_retention`, written by the train stage from `checkpoint.keep`).
"""
from __future__ import annotations

from pathlib import Path

from rft_cookbook.checkpoints import Artifact, CheckpointBundle, CheckpointStore

MARKER = "metadata.json"


def megatron_bundle(checkpoint_dir: str, rollout_id: int, hf_checkpoint_dir: str | None) -> CheckpointBundle:
    root = Path(checkpoint_dir)
    hf = Path(hf_checkpoint_dir) if hf_checkpoint_dir else None
    hf_path = str(hf.relative_to(root)) if hf and hf.is_relative_to(root) else None
    return CheckpointBundle(framework="miles", name=root.name, step=int(rollout_id) + 1, root=str(root),
                            artifacts=[Artifact("model", ".", marker=MARKER)], resumes=["model", "optimizer", "rng"],
                            hf_path=hf_path, metadata={"rollout_id": int(rollout_id), "hf_checkpoint_dir": hf_checkpoint_dir})


def publish(args, rollout_id: int, checkpoint_dir: str, hf_checkpoint_dir: str | None) -> None:
    bundle = megatron_bundle(checkpoint_dir, rollout_id, hf_checkpoint_dir)
    store = CheckpointStore(args.save, hash_files=False)
    result = store.publish(bundle)
    if result.status != "published":
        raise RuntimeError(f"rft: publishing {bundle.name} failed: {result.reason}")
    print(f"[rft] published {bundle.name} (step {bundle.step}): {result.files} files, "
          f"{result.bytes / 1e9:.1f} GB in {result.seconds:.0f} s", flush=True)
    if result.hf_problems:
        print(f"[rft] {bundle.name}: its Hugging Face weights are incomplete: {'; '.join(result.hf_problems)}",
              flush=True)
    keep = store.retention()
    if keep:
        deleted = store.retain(keep)
        if deleted:
            print(f"[rft] checkpoint retention: deleted steps {deleted}", flush=True)

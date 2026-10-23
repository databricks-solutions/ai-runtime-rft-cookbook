"""Publish, find, verify, restore and export framework checkpoints the same way for every framework.

A framework adapter describes one saved checkpoint as a `CheckpointBundle`: the artifacts that
together make it resumable, where each lives locally, and the native marker that says each
artifact is complete. `CheckpointStore.publish` copies a bundle to durable storage (a Unity
Catalog volume) under the bundle's native directory name, so a published copy is itself a
native checkpoint the framework can load. The completion marker is written last, only after
every required artifact verified; `latest` ignores anything without it.

Publication is synchronous. The adapter calls it from the framework's post-save hook, after the
framework reported the save durable. A framework that saves straight into the store (the bundle's
root is its published location) is published in place: nothing is copied, and the manifest and
marker are added to the native directory. Large multi-node checkpoints are saved that way.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST = "rft_checkpoint.json"
COMPLETE = "RFT_COMPLETE"
MANIFEST_SCHEMA = "rft-cookbook/checkpoint/v1"


class CheckpointError(RuntimeError):
    pass


@dataclass
class Artifact:
    name: str
    path: str
    """Relative to the bundle root: a file or a directory."""
    required: bool = True
    marker: str | None = None
    """Relative to the artifact: a file whose presence is the framework's own completion signal."""


@dataclass
class CheckpointBundle:
    framework: str
    name: str
    """Native directory name, e.g. verl's `global_step_8`."""
    step: int
    root: str
    """Local directory that contains every artifact."""
    artifacts: list[Artifact]
    resumes: list[str]
    """What loading this bundle restores: model, optimizer, rng, data_position, rollout_state."""
    hf_path: str | None = None
    """Relative path of the Hugging Face weights inside the bundle, when the framework wrote them."""
    metadata: dict[str, Any] = field(default_factory=dict)

    def check_local(self) -> None:
        """Every required artifact exists and carries its native completion marker."""
        root = Path(self.root)
        for artifact in self.artifacts:
            path = root / artifact.path
            if not path.exists():
                if artifact.required:
                    raise CheckpointError(f"{self.name}: required artifact {artifact.name} missing at {path}")
                continue
            if artifact.marker and not (path / artifact.marker).exists():
                raise CheckpointError(f"{self.name}: {artifact.name} has no completion marker {artifact.marker}")


@dataclass
class PublishResult:
    status: str
    """`published` or `failed`."""
    step: int
    location: str | None = None
    files: int = 0
    bytes: int = 0
    seconds: float = 0.0
    reason: str | None = None


@dataclass
class Published:
    step: int
    name: str
    location: str
    manifest: dict[str, Any]

    @property
    def hf_path(self) -> str | None:
        rel = self.manifest["bundle"].get("hf_path")
        return str(Path(self.location) / rel) if rel else None


class CheckpointStore:
    def __init__(self, root: str | os.PathLike, *, hash_files: bool = True):
        self.root = Path(root)
        self.hash_files = hash_files

    # ---- publish -----------------------------------------------------------------------

    def publish(self, bundle: CheckpointBundle) -> PublishResult:
        started = time.monotonic()
        target = self.root / bundle.name
        in_place = Path(bundle.root).resolve() == target.resolve()
        try:
            bundle.check_local()
            if (target / COMPLETE).exists():
                raise CheckpointError(f"{bundle.name} is already published at {target}")
            if target.exists() and not in_place:
                shutil.rmtree(target)  # an earlier, unfinished copy
            files = []
            for artifact in bundle.artifacts:
                source = Path(bundle.root) / artifact.path
                if not source.exists():
                    continue
                files += _entries(source, artifact.path) if in_place else _copy(source, target / artifact.path, artifact.path)
            manifest = {"schema": MANIFEST_SCHEMA, "bundle": asdict(bundle), "files": files,
                        "hashed": self.hash_files, "published_at": time.time()}
            self._verify_files(target, files)
            if self.hash_files:
                for entry in files:
                    entry["sha256"] = _sha256(target / entry["path"])
            _write_json(target / MANIFEST, manifest)
            (target / COMPLETE).write_text(f"{bundle.step}\n")
        except Exception as error:  # noqa: BLE001 - the caller decides whether a failed publish stops training
            return PublishResult("failed", bundle.step, str(target), seconds=time.monotonic() - started,
                                 reason=f"{type(error).__name__}: {error}")
        return PublishResult("published", bundle.step, str(target), files=len(files),
                             bytes=sum(f["size"] for f in files), seconds=time.monotonic() - started)

    # ---- query -------------------------------------------------------------------------

    def list(self) -> list[Published]:
        if not self.root.exists():
            return []
        found = []
        for entry in self.root.iterdir():
            if (entry / COMPLETE).exists() and (entry / MANIFEST).exists():
                manifest = json.loads((entry / MANIFEST).read_text())
                found.append(Published(int(manifest["bundle"]["step"]), entry.name, str(entry), manifest))
        return sorted(found, key=lambda p: p.step)

    def latest(self) -> Published | None:
        published = self.list()
        return published[-1] if published else None

    def get(self, step: int) -> Published:
        for published in self.list():
            if published.step == step:
                return published
        raise CheckpointError(f"no published checkpoint for step {step} under {self.root}")

    def verify(self, published: Published, *, hashes: bool = True) -> None:
        """Every file listed in the manifest is present with its size (and hash, if recorded)."""
        files = published.manifest["files"]
        self._verify_files(Path(published.location), files)
        if hashes and published.manifest.get("hashed"):
            for entry in files:
                if _sha256(Path(published.location) / entry["path"]) != entry["sha256"]:
                    raise CheckpointError(f"{published.name}: {entry['path']} hash differs from the manifest")

    # ---- restore / export / retention --------------------------------------------------

    def restore(self, published: Published, local_root: str | os.PathLike) -> Path:
        """Copy a published bundle back to local disk under its native name; returns that directory."""
        self.verify(published, hashes=False)
        target = Path(local_root) / published.name
        if target.exists():
            shutil.rmtree(target)
        for artifact in published.manifest["bundle"]["artifacts"]:
            source = Path(published.location) / artifact["path"]
            if source.exists():
                _copy(source, target / artifact["path"], artifact["path"])
        return target

    def export(self, published: Published) -> dict[str, Any]:
        """Locate and check the Hugging Face weights of a published bundle."""
        if not published.hf_path:
            raise CheckpointError(f"{published.name}: the framework wrote no Hugging Face weights")
        hf = Path(published.hf_path)
        problems = hf_problems(hf)
        if problems:
            raise CheckpointError(f"{published.name}: Hugging Face export incomplete: {'; '.join(problems)}")
        weights = sorted(p for p in hf.iterdir() if p.suffix == ".safetensors")
        rel = Path(published.manifest["bundle"]["hf_path"])
        by_path = {f["path"]: f for f in published.manifest["files"]}
        digest = hashlib.sha256()
        for weight in weights:
            entry = by_path.get(str(rel / weight.name))
            digest.update(f"{weight.name}:{entry.get('sha256') if entry else weight.stat().st_size}\n".encode())
        return {"step": published.step, "path": str(hf), "weights": [w.name for w in weights],
                "weights_sha256": digest.hexdigest()}

    def retain(self, keep_last: int, keep_steps: set[int] | None = None) -> list[int]:
        """Delete published bundles beyond the newest `keep_last`, except `keep_steps`. Returns deleted steps."""
        if keep_last < 1:
            raise ValueError("keep_last must be at least 1")
        keep_steps = keep_steps or set()
        doomed = [p for p in self.list()[:-keep_last] if p.step not in keep_steps]
        for p in doomed:
            (Path(p.location) / COMPLETE).unlink()  # first, so a partial delete is never "latest"
            shutil.rmtree(p.location)
        return [p.step for p in doomed]

    @staticmethod
    def _verify_files(root: Path, files: list[dict[str, Any]]) -> None:
        for entry in files:
            path = root / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise CheckpointError(f"{path} missing or size differs from the source")


def hf_problems(path: Path) -> list[str]:
    """Why `path` is not a loadable Hugging Face model directory; empty when it is."""
    if not path.is_dir():
        return [f"{path} is not a directory"]
    problems = []
    if not (path / "config.json").is_file():
        problems.append("no config.json")
    weights = [p for p in path.iterdir() if p.suffix == ".safetensors"]
    if not weights:
        problems.append("no .safetensors weights")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        missing = sorted(s for s in shards if not (path / s).is_file())
        if missing:
            problems.append(f"index names missing shards {missing}")
    elif len(weights) > 1:
        problems.append("several weight files but no model.safetensors.index.json")
    return problems


def _copy(source: Path, target: Path, rel: str) -> list[dict[str, Any]]:
    entries = []
    sources = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    for path in sources:
        rel_path = Path(rel) if source.is_file() else Path(rel) / path.relative_to(source)
        destination = target if source.is_file() else target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        entries.append({"path": str(rel_path), "size": path.stat().st_size})
    return entries


def _entries(source: Path, rel: str) -> list[dict[str, Any]]:
    sources = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    return [{"path": str(Path(rel) if source.is_file() else Path(rel) / p.relative_to(source)), "size": p.stat().st_size}
            for p in sources]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    # Not renamed into place: the completion marker, written after it, is what makes it count.
    path.write_text(json.dumps(value, indent=1, sort_keys=True))

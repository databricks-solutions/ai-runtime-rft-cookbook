# Derived from verl-on-air (Apache License 2.0), engine/stage_model.py at 495b583;
# modified for ai-runtime-rft-cookbook: arguments instead of environment defaults; returns the
# staging record; the lock is created exclusively, read back and kept alive. See THIRD_PARTY_NOTICES.md.
"""Stage a Hugging Face checkpoint into a Unity Catalog volume once, instead of on every run.

UC volumes are a FUSE mount that supports sequential writes but not the random-access writes
Hugging Face's accelerated downloaders use (Xet/CAS and hf_transfer fail with "Operation not
supported (os error 95)"; set HF_HUB_DISABLE_XET=1). So each file is downloaded to local scratch
and copied to the volume with a sequential write; local disk use stays at one shard.

One revision, verified: the requested revision is resolved once to a commit and every file is
downloaded at that commit. Each file is checked against the Hub's own metadata (LFS sha256 for
weights, git blob hash for small files) before it is copied; the finished directory records
what it holds in STAGED.json. A directory that already holds a different revision is refused.
Resumable: a file already on the volume counts only if its content hash matches; a lock file
keeps two staging jobs from writing the same directory at once. It is created exclusively and
then read back, since a volume shared by several nodes may let two creations both succeed (the
last writer's token is then what both read, and the other job stops); it is rewritten after every
file (a volume ignores `utime`: a file's time is when it was last written), and a lock not
rewritten for STAGE_LOCK_STALE_H is taken over.

    python -m rft_cookbook.air.stage_model --model-id Qwen/Qwen3.5-2B --revision <sha> --dest <dir>
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sys
import time
import uuid

# Weights only. Skipping duplicate .bin/.pth formats matters: several Qwen repos
# ship both safetensors and consolidated files, doubling the transfer for nothing.
ALLOW_SUFFIXES = (".safetensors", ".json", ".txt", ".model", ".py", ".jinja")
IGNORE_SUFFIXES = (".bin", ".pth", ".pt", ".gguf", ".onnx")
IGNORE_PREFIXES = ("original/",)

COPY_BUF = 32 * 1024 * 1024   # 32 MiB sequential writes to the FUSE mount
STAGED, PROGRESS, LOCK = "STAGED.json", ".staging.json", ".staging.lock"
LOCK_STALE_S = float(os.environ.get("STAGE_LOCK_STALE_H", "12")) * 3600
LOCK_SETTLE_S = 2.0   # before the lock is read back


def pick_scratch(explicit: str | None = None) -> str:
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    for cand in ("/local_disk0", "/tmp"):
        if os.path.isdir(cand):
            path = os.path.join(cand, "hf_stage")
            try:
                os.makedirs(path, exist_ok=True)
                return path
            except OSError:
                continue
    raise RuntimeError("no writable scratch directory found")


def free_gb(path: str) -> float:
    probe = path
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    return shutil.disk_usage(probe or "/").free / 1024**3


def wanted(name: str) -> bool:
    if name.startswith(IGNORE_PREFIXES) or name.endswith(IGNORE_SUFFIXES):
        return False
    return name.endswith(ALLOW_SUFFIXES)


def stream_copy(src: str, dst: str) -> None:
    """Sequential copy onto the FUSE mount, via a temp name then rename."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".partial"
    with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, COPY_BUF)
        fdst.flush()
        os.fsync(fdst.fileno())
    os.replace(tmp, dst)


def expected_digest(sibling) -> tuple[str, str] | None:
    """The content hash the Hub publishes for a file: LFS sha256, else the git blob sha1."""
    lfs = getattr(sibling, "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if sha:
        return "sha256", sha
    blob = getattr(sibling, "blob_id", None)
    return ("git-blob-sha1", blob) if blob else None


def file_digest(path: str, kind: str) -> str:
    size = os.path.getsize(path)
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(COPY_BUF), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: str, obj: dict) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def take_lock(model_dir: str) -> tuple[str, str]:
    """`(lock path, this job's token)`; exits if another staging job holds the lock."""
    lock = os.path.join(model_dir, LOCK)
    token = (f"host={socket.gethostname()} pid={os.getpid()} since={time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
             f"id={uuid.uuid4().hex}")
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock)
                holder = open(lock).read().strip()
            except FileNotFoundError:
                continue   # released meanwhile
            if age < LOCK_STALE_S:
                sys.exit(f"FATAL: {lock} is held ({holder}, renewed {age / 60:.0f} min ago): another staging job "
                         "is writing this directory. Delete the lock only if none is running.")
            print(f"WARNING: taking over a stale lock ({age / 3600:.1f} h old): {lock}")
            try:   # whichever job renames it first takes over; the other finds a fresh lock
                os.replace(lock, f"{lock}.stale-{uuid.uuid4().hex[:8]}")
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(token)
        time.sleep(LOCK_SETTLE_S)
        if _lock_holder(lock) != token:
            sys.exit(f"FATAL: another staging job took {lock} at the same time ({_lock_holder(lock)})")
        return lock, token
    sys.exit(f"FATAL: could not take {lock}")


def _lock_holder(lock: str) -> str | None:
    try:
        return open(lock).read().strip()
    except FileNotFoundError:
        return None


def renew_lock(lock: str, token: str) -> None:
    """Rewrite the lock, which renews its modification time, if this job still holds it."""
    if _lock_holder(lock) != token:
        return
    tmp = f"{lock}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "w") as fh:
            fh.write(token)
        os.replace(tmp, lock)
    except OSError:
        pass


def release_lock(lock: str, token: str) -> None:
    if _lock_holder(lock) == token:   # a job that took over a lock it thought stale now owns it
        try:
            os.remove(lock)
        except OSError:
            pass


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".msgpack", ".h5")


def stage(MODEL_ID: str, MODEL_REVISION: str, MODEL_DIR: str, scratch_dir: str | None = None,
          tokenizer_only: bool = False) -> dict:
    """Stage `MODEL_ID@MODEL_REVISION` into `MODEL_DIR`; returns the STAGED.json record.

    `tokenizer_only` stages the tokenizer, chat template and configuration without weights (a
    judge served elsewhere still needs its own chat template to count prompt tokens)."""
    from huggingface_hub import HfApi, hf_hub_download

    scratch = pick_scratch(scratch_dir)
    os.makedirs(MODEL_DIR, exist_ok=True)

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    info = api.model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    commit = info.sha
    print(f"model    : {MODEL_ID} @ {MODEL_REVISION} = commit {commit}")
    print(f"dest     : {MODEL_DIR}   ({free_gb(MODEL_DIR):.0f} GiB free)")
    print(f"scratch  : {scratch}     ({free_gb(scratch):.0f} GiB free)")
    print(f"hf_xfer  : {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')}   "
          f"xet_disabled: {os.environ.get('HF_HUB_DISABLE_XET', '0')}")

    # One directory, one revision: never mix a new commit's files into an older snapshot.
    for rec_name in (STAGED, PROGRESS):
        held = read_json(os.path.join(MODEL_DIR, rec_name)).get("revision")
        if held and held != commit:
            sys.exit(f"FATAL: {MODEL_DIR} holds {MODEL_ID}@{held} ({rec_name}), not {commit}. Stage this "
                     "revision into its own directory.")
    if read_json(os.path.join(MODEL_DIR, STAGED)).get("revision") == commit:
        print(f"already staged: {MODEL_ID}@{commit} ({STAGED}) -- nothing to do")
        return read_json(os.path.join(MODEL_DIR, STAGED))

    lock, token = take_lock(MODEL_DIR)
    try:
        siblings = sorted((s for s in info.siblings if wanted(s.rfilename)
                           and not (tokenizer_only and s.rfilename.endswith(WEIGHT_SUFFIXES))),
                          key=lambda s: s.rfilename)
        total = sum(s.size or 0 for s in siblings)
        print(f"\n{len(siblings)} file(s), {total / 1024**3:.1f} GiB to transfer\n")
        if free_gb(scratch) < 12:
            print(f"WARNING: only {free_gb(scratch):.0f} GiB of scratch; largest shard "
                  "may not fit. Set SCRATCH_DIR to a bigger mount if this fails.")

        progress_path = os.path.join(MODEL_DIR, PROGRESS)
        progress = read_json(progress_path) or {"model_id": MODEL_ID, "revision": commit, "files": {}}
        started = time.time()
        done_bytes = skipped = 0
        for idx, sib in enumerate(siblings, 1):
            name, size = sib.rfilename, sib.size or 0
            digest = expected_digest(sib)
            dst = os.path.join(MODEL_DIR, name)
            rec = {"size": size, "digest": list(digest) if digest else None}
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                # resume: trust a file only if its CONTENT matches the Hub's hash
                if progress["files"].get(name) == rec or (digest and file_digest(dst, digest[0]) == digest[1]):
                    progress["files"][name] = rec
                    write_json(progress_path, progress)
                    renew_lock(lock, token)
                    skipped += 1
                    done_bytes += size
                    print(f"[{idx}/{len(siblings)}] skip  {name}  ({size / 1024**2:.0f} MiB, verified)")
                    continue

            t0 = time.time()
            local = hf_hub_download(repo_id=MODEL_ID, filename=name, revision=commit, local_dir=scratch,
                                    token=os.environ.get("HF_TOKEN") or None)
            dl = time.time() - t0
            if digest and file_digest(local, digest[0]) != digest[1]:
                sys.exit(f"FATAL: {name} downloaded from {MODEL_ID}@{commit} does not match its {digest[0]}")
            t1 = time.time()
            stream_copy(local, dst)
            cp = time.time() - t1
            if os.path.getsize(dst) != size:
                sys.exit(f"FATAL: {dst} is {os.path.getsize(dst)} bytes after the copy, expected {size}")
            try:  # free scratch immediately so the peak stays at one shard
                os.remove(local)
            except OSError:
                pass
            progress["files"][name] = rec
            write_json(progress_path, progress)
            renew_lock(lock, token)
            done_bytes += size
            mib = size / 1024**2
            pct = 100 * done_bytes / total if total else 100
            print(f"[{idx}/{len(siblings)}] ok    {name}  {mib:.0f} MiB  "
                  f"dl {dl:.1f}s ({mib / max(dl, 1e-3):.0f} MiB/s)  "
                  f"cp {cp:.1f}s ({mib / max(cp, 1e-3):.0f} MiB/s)  [{pct:.0f}%]")

        elapsed = time.time() - started
        print(f"\ntransferred in {elapsed / 60:.1f} min ({skipped} verified in place, "
              f"{len(siblings) - skipped} copied)")

        # A truncated transfer otherwise surfaces much later as a cryptic safetensors
        # error inside a Ray worker, so verify the shard set here where it is obvious.
        index = os.path.join(MODEL_DIR, "model.safetensors.index.json")
        if tokenizer_only:
            print("tokenizer only: no weights staged")
        elif os.path.exists(index):
            shards = set(read_json(index).get("weight_map", {}).values())
            missing = [s for s in sorted(shards) if not os.path.exists(os.path.join(MODEL_DIR, s))]
            if missing:
                sys.exit(f"INCOMPLETE: {len(missing)} shard(s) missing, e.g. {missing[:3]}")
            print(f"verified all {len(shards)} safetensors shards present")
        else:
            print("note: no safetensors index (single-shard model?) — skipping shard check")

        staged = {
            "model_id": MODEL_ID, "requested_revision": MODEL_REVISION, "revision": commit,
            "files": progress["files"], "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        write_json(os.path.join(MODEL_DIR, STAGED), staged)
        os.remove(progress_path)
    finally:
        release_lock(lock, token)

    on_disk = sum(os.path.getsize(os.path.join(root, f)) for root, _, fs in os.walk(MODEL_DIR) for f in fs)
    print(f"\n{on_disk / 1024**3:.1f} GiB staged at {MODEL_DIR}  ({MODEL_ID}@{commit}, see {STAGED})")
    return staged


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Stage a Hugging Face model into a Unity Catalog volume.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True, help="a commit sha (a branch is resolved once and recorded)")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--scratch", default=None, help="local scratch (default /local_disk0/hf_stage, else /tmp)")
    parser.add_argument("--tokenizer-only", action="store_true", help="tokenizer, template and config; no weights")
    args = parser.parse_args(argv)
    stage(args.model_id, args.revision, args.dest, args.scratch, tokenizer_only=args.tokenizer_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

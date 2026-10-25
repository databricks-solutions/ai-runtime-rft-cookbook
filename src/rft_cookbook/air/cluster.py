"""One Ray cluster across the nodes of an AI Runtime job.

AI Runtime runs a job's command once per node and sets NUM_NODES, POD_RANK and MASTER_ADDR; it
starts no Ray. A Ray application (verl, miles) on several nodes therefore runs as

    nodes = Nodes.from_env()
    if nodes.rank == 0:
        run_head(nodes, gpus_per_node, rendezvous, body)   # body attaches through RAY_ADDRESS
    else:
        run_worker(nodes, gpus_per_node, rendezvous)

The head starts Ray, waits until every node's GPUs have registered, runs the body, and records
its outcome in the rendezvous directory (on storage every node sees) before it stops Ray. A
worker joins, then waits for that outcome and leaves with it. A worker also leaves, with a
failure, when the head stops writing its heartbeat (killed without cleaning up) or disappears
without an outcome, so no node idles until the job's timeout.

Rendezvous files carry the head's clock; a worker ignores files from before its own start (minus
`SKEW_S`), so a resumed job that reuses the directory never reads the last attempt's outcome.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PORT = 6379
DONE = "head_done.json"
ALIVE = "head_alive.json"
SKEW_S = 300.0


@dataclass(frozen=True)
class Nodes:
    count: int
    rank: int
    head: str

    @classmethod
    def from_env(cls) -> Nodes:
        count = int(os.environ.get("NUM_NODES", "1"))
        rank = int(os.environ.get("POD_RANK", os.environ.get("NODE_RANK", "0")))
        head = os.environ.get("MASTER_ADDR", "127.0.0.1")
        if count > 1 and not os.environ.get("MASTER_ADDR"):
            raise SystemExit("a job on several nodes needs MASTER_ADDR")
        return cls(count, rank, head)

    def rendezvous(self, root: str | os.PathLike) -> Path:
        """This job's rendezvous directory under `root`: the head's address is new for every job."""
        return Path(root) / f"{self.head}-{os.environ.get('MASTER_PORT', PORT)}"


def run_head(nodes: Nodes, gpus_per_node: int, rendezvous: Path, body: Callable[[], None], *,
             port: int = PORT, register_timeout_s: float = 900.0, heartbeat_s: float = 30.0) -> None:
    rendezvous.mkdir(parents=True, exist_ok=True)
    (rendezvous / DONE).unlink(missing_ok=True)
    stop = threading.Event()
    _put(rendezvous / ALIVE, {"at": time.time()})   # the first beat before anything can fail
    beat = threading.Thread(target=_heartbeat, args=(rendezvous, stop, heartbeat_s), daemon=True)
    beat.start()
    rc = 1
    try:
        _ray("start", "--head", f"--node-ip-address={nodes.head}", f"--port={port}", f"--num-gpus={gpus_per_node}",
             "--disable-usage-stats")
        _wait_for_gpus(nodes.count * gpus_per_node, register_timeout_s)
        os.environ["RAY_ADDRESS"] = f"{nodes.head}:{port}"
        body()
        rc = 0
    except SystemExit as error:
        rc = error.code if isinstance(error.code, int) else 1
        raise
    finally:
        _put(rendezvous / DONE, {"rc": rc, "at": time.time()})
        stop.set()
        _stop_ray()
        print(f"[cluster] head done (rc={rc}); workers released", flush=True)


def run_worker(nodes: Nodes, gpus_per_node: int, rendezvous: Path, *, port: int = PORT, join_delay_s: float = 15.0,
               join_attempts: int = 30, join_interval_s: float = 10.0, poll_s: float = 15.0,
               stale_s: float = 600.0, outcome_grace_s: float = 120.0) -> None:
    """Join the head and return once it finished cleanly; SystemExit when it failed or vanished."""
    started = time.time()
    time.sleep(join_delay_s)
    for attempt in range(1, join_attempts + 1):
        if _ray("start", f"--address={nodes.head}:{port}", f"--num-gpus={gpus_per_node}", "--disable-usage-stats",
                check=False) == 0:
            break
        print(f"[cluster] node {nodes.rank}: head not up yet ({attempt}/{join_attempts})", flush=True)
        time.sleep(join_interval_s)
    else:
        raise SystemExit(f"node {nodes.rank} could not join the Ray head at {nodes.head}:{port}")
    print(f"[cluster] node {nodes.rank} joined {nodes.head}:{port}; waiting for the head", flush=True)
    misses = 0
    try:
        while True:
            outcome = _fresh(rendezvous / DONE, started)
            if outcome is not None:
                _leave(nodes, outcome)
                return
            alive = _fresh(rendezvous / ALIVE, started)
            if alive is not None and time.time() - alive["at"] > stale_s:
                raise SystemExit(f"node {nodes.rank}: the head's heartbeat is {time.time() - alive['at']:.0f}s old")
            misses = 0 if _reachable(nodes.head, port) else misses + 1
            if misses >= 3:
                deadline = time.time() + outcome_grace_s   # the outcome can show up after the port closed
                while time.time() < deadline:
                    outcome = _fresh(rendezvous / DONE, started)
                    if outcome is not None:
                        _leave(nodes, outcome)
                        return
                    time.sleep(5)
                raise SystemExit(f"node {nodes.rank}: the head is gone without an outcome")
            time.sleep(poll_s)
    finally:
        _stop_ray()


def _leave(nodes: Nodes, outcome: dict) -> None:
    print(f"[cluster] node {nodes.rank}: head finished with rc={outcome['rc']}", flush=True)
    if outcome["rc"] != 0:
        raise SystemExit(f"the head failed (rc={outcome['rc']})")


def _stop_ray(exit_wait_s: float = 600.0, poll_s: float = 5.0) -> None:
    """Stop Ray, then wait for its processes to exit: a killed actor can take minutes to release
    its memory (an optimizer offloaded to pinned host memory), and a node that ends its command
    before then is reported as failed."""
    _ray("stop", "--force", check=False)
    deadline = time.time() + exit_wait_s
    left = _ray_processes()
    if left:
        print(f"[cluster] waiting for {len(left)} Ray processes to exit", flush=True)
    while left and time.time() < deadline:
        time.sleep(poll_s)
        left = _ray_processes()
    if left:
        print(f"[cluster] {len(left)} Ray processes still alive after {exit_wait_s:.0f}s: {left[:4]}", flush=True)


def _ray_processes() -> list[str]:
    """Live Ray worker processes on this node (`ray::<name>`; zombies are already gone)."""
    out = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            name = (proc / "cmdline").read_bytes().split(b"\0", 1)[0].decode(errors="replace")
            state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            continue
        if name.startswith("ray::") and state != "Z" and int(proc.name) != os.getpid():
            out.append(f"{proc.name} {name} ({state})")
    return out


def _ray(*args: str, check: bool = True) -> int:
    ray = Path(sys.executable).with_name("ray")
    result = subprocess.run([str(ray) if ray.exists() else "ray", *args])
    if check and result.returncode != 0:
        raise SystemExit(f"ray {args[0]} exited with {result.returncode}")
    return result.returncode


def _wait_for_gpus(want: int, timeout_s: float) -> None:
    probe = "import ray; ray.init(address='auto', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))"
    deadline = time.time() + timeout_s
    have = 0
    while time.time() < deadline:
        try:   # a probe that hangs counts as no GPUs, so the deadline holds
            result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            result = None
        have = int(result.stdout.strip().splitlines()[-1]) if result and result.returncode == 0 and result.stdout.strip() else 0
        print(f"[cluster] GPUs registered: {have}/{want}", flush=True)
        if have >= want:
            return
        time.sleep(10)
    raise SystemExit(f"only {have}/{want} GPUs registered within {timeout_s:.0f}s")


def _heartbeat(rendezvous: Path, stop: threading.Event, period_s: float) -> None:
    while not stop.is_set():
        try:
            _put(rendezvous / ALIVE, {"at": time.time()})
        except OSError:
            pass   # one missed beat is not a failure; a long silence is
        stop.wait(period_s)


def _put(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")   # a heartbeat may write at once
    tmp.write_text(json.dumps(value))
    os.replace(tmp, path)


def _fresh(path: Path, since: float) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if float(value.get("at", 0)) >= since - SKEW_S else None


def _reachable(host: str, port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(5)
        return probe.connect_ex((host, port)) == 0

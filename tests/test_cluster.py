"""The Ray cluster bootstrap across AI Runtime nodes, with `ray` and the network faked."""
from __future__ import annotations

import json
import threading
import time

import pytest

from rft_cookbook.air import cluster

NODES = cluster.Nodes(count=2, rank=1, head="10.0.0.1")
FAST = {"join_delay_s": 0, "join_interval_s": 0, "poll_s": 0.01, "outcome_grace_s": 0.2}


@pytest.fixture
def ray(monkeypatch):
    calls = []
    monkeypatch.setattr(cluster, "_ray", lambda *args, check=True: calls.append(args) or 0)
    monkeypatch.setattr(cluster, "_wait_for_gpus", lambda want, timeout_s: calls.append(("wait", want)))
    monkeypatch.setattr(cluster, "_reachable", lambda host, port: True)
    monkeypatch.setattr(cluster, "_ray_processes", lambda: [])
    return calls


def put_later(path, value, delay=0.05):
    threading.Timer(delay, lambda: path.write_text(json.dumps(value))).start()


def test_a_worker_leaves_with_the_heads_outcome(ray, tmp_path):
    put_later(tmp_path / cluster.DONE, {"rc": 0, "at": time.time()})
    cluster.run_worker(NODES, 8, tmp_path, **FAST)
    assert ray[0][:2] == ("start", "--address=10.0.0.1:6379") and ray[-1] == ("stop", "--force")

    (tmp_path / cluster.DONE).unlink()
    put_later(tmp_path / cluster.DONE, {"rc": 1, "at": time.time()})
    with pytest.raises(SystemExit, match="head failed"):
        cluster.run_worker(NODES, 8, tmp_path, **FAST)


def test_a_worker_ignores_the_last_jobs_outcome(ray, tmp_path, monkeypatch):
    (tmp_path / cluster.DONE).write_text(json.dumps({"rc": 0, "at": time.time() - 2 * cluster.SKEW_S}))
    monkeypatch.setattr(cluster, "_reachable", lambda host, port: False)
    with pytest.raises(SystemExit, match="without an outcome"):
        cluster.run_worker(NODES, 8, tmp_path, **FAST)


def test_a_worker_does_not_wait_for_a_dead_head(ray, tmp_path):
    (tmp_path / cluster.ALIVE).write_text(json.dumps({"at": time.time() - 100}))
    with pytest.raises(SystemExit, match="heartbeat"):
        cluster.run_worker(NODES, 8, tmp_path, **FAST, stale_s=50)


def test_a_worker_that_cannot_join_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster, "_ray", lambda *args, check=True: 1)
    with pytest.raises(SystemExit, match="could not join"):
        cluster.run_worker(NODES, 8, tmp_path, **FAST, join_attempts=2)


def test_the_head_records_its_outcome_and_exports_its_address(ray, tmp_path, monkeypatch):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    head = cluster.Nodes(count=2, rank=0, head="10.0.0.1")
    seen = []
    cluster.run_head(head, 8, tmp_path, lambda: seen.append(cluster.os.environ["RAY_ADDRESS"]), heartbeat_s=0.01)
    assert seen == ["10.0.0.1:6379"] and ("wait", 16) in ray and ray[-1] == ("stop", "--force")
    assert json.loads((tmp_path / cluster.DONE).read_text())["rc"] == 0
    assert json.loads((tmp_path / cluster.ALIVE).read_text())["at"] > 0

    def fail():
        raise SystemExit("verl exited with 1")

    with pytest.raises(SystemExit):
        cluster.run_head(head, 8, tmp_path, fail, heartbeat_s=0.01)
    assert json.loads((tmp_path / cluster.DONE).read_text())["rc"] == 1


def test_nodes_come_from_the_job_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("NUM_NODES", "2")
    monkeypatch.setenv("POD_RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.7")
    monkeypatch.setenv("MASTER_PORT", "29500")
    nodes = cluster.Nodes.from_env()
    assert nodes == cluster.Nodes(2, 1, "10.0.0.7")
    assert nodes.rendezvous(tmp_path) == tmp_path / "10.0.0.7-29500"
    monkeypatch.delenv("MASTER_ADDR")
    with pytest.raises(SystemExit, match="MASTER_ADDR"):
        cluster.Nodes.from_env()


def test_a_node_waits_for_rays_processes_to_exit_before_it_leaves(ray, monkeypatch, capsys):
    """An actor freeing its offloaded optimizer outlives `ray stop`; the node waits for it."""
    alive = [["21121 ray::MegatronTrainRayActor (D)"]] * 2 + [[]]
    monkeypatch.setattr(cluster, "_ray_processes", lambda: alive.pop(0))
    cluster._stop_ray(exit_wait_s=5, poll_s=0)
    assert ray == [("stop", "--force")] and not alive
    assert "waiting for 1 Ray processes" in capsys.readouterr().out

    monkeypatch.setattr(cluster, "_ray_processes", lambda: ["7 ray::IDLE (D)"])
    cluster._stop_ray(exit_wait_s=0.05, poll_s=0.01)   # bounded: the node leaves anyway
    assert "still alive after 0s" in capsys.readouterr().out


def test_live_ray_processes_are_read_from_proc():
    import subprocess

    actor = subprocess.Popen(["bash", "-c", "exec -a ray::RftTestActor sleep 30"])
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not any("ray::RftTestActor" in p for p in cluster._ray_processes()):
            time.sleep(0.05)
        assert f"{actor.pid} ray::RftTestActor (S)" in cluster._ray_processes()
    finally:
        actor.kill()
        actor.wait()
    assert not any("ray::RftTestActor" in p for p in cluster._ray_processes())   # reaped: gone

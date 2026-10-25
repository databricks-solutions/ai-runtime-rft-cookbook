"""The staging lock (rft_cookbook.air.stage_model): one staging job per model directory."""
from __future__ import annotations

import os
import time

import pytest

from rft_cookbook.air import stage_model


@pytest.fixture(autouse=True)
def no_settle(monkeypatch):
    monkeypatch.setattr(stage_model, "LOCK_SETTLE_S", 0.0)


def test_a_held_lock_stops_a_second_job_and_is_released_by_its_holder(tmp_path):
    lock, token = stage_model.take_lock(str(tmp_path))
    with pytest.raises(SystemExit, match="another staging job"):
        stage_model.take_lock(str(tmp_path))
    stage_model.release_lock(lock, token)
    assert not os.path.exists(lock)
    stage_model.release_lock(*stage_model.take_lock(str(tmp_path)))   # free again


def test_a_stale_lock_is_taken_over_and_a_renewed_one_is_not(tmp_path, monkeypatch):
    lock, token = stage_model.take_lock(str(tmp_path))
    old = time.time() - stage_model.LOCK_STALE_S - 60
    os.utime(lock, (old, old))
    stage_model.renew_lock(lock, token)   # a live job rewrites it after every file
    with pytest.raises(SystemExit):
        stage_model.take_lock(str(tmp_path))
    os.utime(lock, (old, old))     # a job that died stops rewriting it
    _, token = stage_model.take_lock(str(tmp_path))
    assert stage_model._lock_holder(lock) == token


def test_a_job_that_loses_a_simultaneous_creation_stops(tmp_path, monkeypatch):
    lock = os.path.join(str(tmp_path), stage_model.LOCK)

    def the_other_job_writes_last(seconds):
        with open(lock, "w") as fh:
            fh.write("host=other pid=1 id=theirs")

    monkeypatch.setattr(stage_model.time, "sleep", the_other_job_writes_last)
    with pytest.raises(SystemExit, match="at the same time"):
        stage_model.take_lock(str(tmp_path))


def test_a_job_never_removes_a_lock_it_no_longer_holds(tmp_path):
    lock, token = stage_model.take_lock(str(tmp_path))
    with open(lock, "w") as fh:
        fh.write("host=other pid=1 id=theirs")   # taken over while this job looked dead
    stage_model.release_lock(lock, token)
    assert stage_model._lock_holder(lock) == "host=other pid=1 id=theirs"

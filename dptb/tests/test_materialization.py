"""Materialization package: fail-safe JSON persistence, the advisory work-root
lock (live-lock refusal, stale takeover by age or by a dead owner pid, clean-
exit release), CLI wiring, and the bounded LMDB environment cache.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import pytest

import dptb.data.materialization.workroot as workroot_mod
import tools.materialize_nonsoc_dual_prior_cache as dual_tool
import tools.materialize_nonsoc_p2_cache as p2_tool
from dptb.data.materialization import (
    DEFAULT_LOCK_STALE_SECONDS,
    LOCK_FILE_NAME,
    WorkRootLock,
    WorkRootLockError,
    check_work_root_lock,
    guard_work_root,
    write_json,
)
from dptb.data.materialization.lmdb_io import BoundedEnvCache, TransactionalLMDBWriter
from dptb.data.materialization.workroot import LOCK_SCHEMA


# ===========================================================================
# write_json durability
# ===========================================================================
def test_write_json_preserves_legacy_byte_layout(tmp_path):
    target = tmp_path / "manifest.json"
    payload = {"b": 2, "a": [1, 2], "nested": {"y": 1, "x": 0}}
    write_json(target, payload)

    expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert target.read_text(encoding="utf-8") == expected
    # No temp files linger after a successful swap.
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("preexisting", [False, True], ids=["no_previous_file", "keeps_previous_file"])
def test_write_json_crash_during_swap_leaves_no_partial_file(tmp_path, monkeypatch, preexisting):
    target = tmp_path / "manifest.json"
    before = None
    if preexisting:
        write_json(target, {"generation": 1})
        before = target.read_bytes()

    def _boom(src, dst):
        raise OSError("injected crash during os.replace")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="injected crash"):
        write_json(target, {"generation": 2})

    if preexisting:
        # Old complete file survives byte-for-byte; no torn/partial replacement.
        assert target.read_bytes() == before
    else:
        assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_temp_names_are_unique(tmp_path, monkeypatch):
    target = tmp_path / "manifest.json"
    seen = []

    real_replace = os.replace

    def _spy(src, dst):
        seen.append(os.path.basename(str(src)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _spy)
    write_json(target, {"n": 1})
    write_json(target, {"n": 2})

    assert len(seen) == 2 and seen[0] != seen[1]


# ===========================================================================
# WorkRootLock: acquire / release / refusal / takeover
# ===========================================================================
def _write_foreign_lock(work_root, *, host, pid, age_seconds):
    payload = {
        "schema": LOCK_SCHEMA, "run_uuid": "f" * 32, "pid": pid, "host": host,
        "acquired_unix": time.time() - float(age_seconds),
        "stale_after_seconds": DEFAULT_LOCK_STALE_SECONDS,
    }
    lock_path = work_root / LOCK_FILE_NAME
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock_path


def test_lock_acquire_writes_identity_and_release_removes_it(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    lock = WorkRootLock(work)
    with lock:
        assert lock.acquired
        payload = json.loads((work / LOCK_FILE_NAME).read_text(encoding="utf-8"))
        assert payload["schema"] == LOCK_SCHEMA
        assert payload["run_uuid"] == lock.run_uuid
        assert payload["pid"] == os.getpid()
        assert payload["host"] == socket.gethostname()
        assert isinstance(payload["acquired_unix"], float)
    # Clean exit releases the lock.
    assert not (work / LOCK_FILE_NAME).exists()
    assert not lock.acquired


def test_lock_released_even_when_body_raises(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(RuntimeError, match="body failed"):
        with WorkRootLock(work):
            raise RuntimeError("body failed")
    assert not (work / LOCK_FILE_NAME).exists()


def test_live_lock_refuses_second_acquire_and_checks(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    with WorkRootLock(work):
        # Same-process pid is alive -> the lock is live.
        with pytest.raises(WorkRootLockError, match="locked by a live"):
            WorkRootLock(work).acquire()
        with pytest.raises(WorkRootLockError, match="locked by a live"):
            check_work_root_lock(work)
    # After release, both pass again.
    check_work_root_lock(work)
    WorkRootLock(work).acquire().release()


@pytest.mark.parametrize(
    ("host_is_foreign", "kill_pid", "age_seconds", "stale_after_seconds", "expect_takeover"),
    [
        (True, False, 1000.0, 100000.0, False),  # foreign host, still within a large stale window -> refused
        (True, False, 1000.0, 10.0, True),       # same foreign lock, past a smaller stale window -> taken over
        (False, True, 0.0, None, True),          # same host, dead owner pid, default age -> liveness beats age
    ],
    ids=["refused_within_stale_window", "taken_over_by_age_on_foreign_host", "taken_over_by_dead_pid_same_host"],
)
def test_stale_lock_refusal_and_takeover(tmp_path, monkeypatch, host_is_foreign, kill_pid, age_seconds,
                                         stale_after_seconds, expect_takeover):
    work = tmp_path / "work"
    work.mkdir()
    host = "definitely-not-" + socket.gethostname() if host_is_foreign else socket.gethostname()
    pid = 1 if host_is_foreign else os.getpid()
    _write_foreign_lock(work, host=host, pid=pid, age_seconds=age_seconds)
    if kill_pid:
        monkeypatch.setattr(workroot_mod, "_pid_alive", lambda pid: False)

    kwargs = {} if stale_after_seconds is None else {"stale_after_seconds": stale_after_seconds}
    lock = WorkRootLock(work, **kwargs)
    if not expect_takeover:
        with pytest.raises(WorkRootLockError):
            lock.acquire()
        return

    # Takeover succeeds and re-stamps our identity.
    lock.acquire()
    try:
        payload = json.loads((work / LOCK_FILE_NAME).read_text(encoding="utf-8"))
        assert payload["run_uuid"] == lock.run_uuid
        assert payload["pid"] == os.getpid()
    finally:
        lock.release()
    assert not (work / LOCK_FILE_NAME).exists()


def test_release_never_clobbers_a_takeover_by_another_run(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    lock = WorkRootLock(work)
    lock.acquire()

    # Simulate another run taking the lock over (e.g. after wrongly judging
    # ours stale): the file now carries a different run UUID.
    (work / LOCK_FILE_NAME).write_text(
        json.dumps({"schema": LOCK_SCHEMA, "run_uuid": "0" * 32}) + "\n", encoding="utf-8",
    )
    lock.release()
    # The foreign lock file must survive our release.
    assert (work / LOCK_FILE_NAME).exists()


# ===========================================================================
# guard_work_root wiring: the lock check runs BEFORE any destructive step
# ===========================================================================
def _guard(protected, work, *, overwrite=False, resume=False, stale=None):
    guard_work_root(
        protected, work, overwrite=overwrite, resume=resume,
        disjoint_message="disjoint", mutually_exclusive_message="exclusive",
        missing_work_root_message="missing", lock_stale_after_seconds=stale,
    )


def test_guard_overwrite_refuses_live_lock_and_preserves_tree(tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    sentinel = work / "committed.data"
    sentinel.write_text("do not delete", encoding="utf-8")

    with WorkRootLock(work):
        with pytest.raises(WorkRootLockError):
            _guard(protected, work, overwrite=True)
        # Refusal happened BEFORE rmtree: the concurrent run's tree survives.
        assert sentinel.read_text(encoding="utf-8") == "do not delete"


def test_guard_resume_refuses_live_lock_but_accepts_stale(tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    work = tmp_path / "work"
    work.mkdir()

    with WorkRootLock(work):
        with pytest.raises(WorkRootLockError):
            _guard(protected, work, resume=True)

    # Foreign stale lock: resume proceeds (takeover is acquire()'s job).
    _write_foreign_lock(work, host="definitely-not-" + socket.gethostname(), pid=1, age_seconds=1000.0)
    _guard(protected, work, resume=True, stale=10.0)


def test_guard_overwrite_removes_stale_locked_tree_and_relocks(tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    _write_foreign_lock(work, host="definitely-not-" + socket.gethostname(), pid=1, age_seconds=1000.0)
    (work / "old.data").write_text("stale run output", encoding="utf-8")

    _guard(protected, work, overwrite=True, stale=10.0)
    # Stale tree (lock included) was removed and the root recreated empty.
    assert work.is_dir()
    assert list(work.iterdir()) == []
    with WorkRootLock(work) as lock:
        assert (work / LOCK_FILE_NAME).exists()
        assert lock.acquired


def test_guard_without_lock_file_behaves_as_before(tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    work = tmp_path / "work"

    _guard(protected, work)  # creates the work root
    assert work.is_dir()
    with pytest.raises(FileExistsError):
        _guard(protected, work)  # refuses clobber without overwrite
    _guard(protected, work, overwrite=True)
    assert work.is_dir()
    with pytest.raises(ValueError, match="disjoint"):
        _guard(work, work / "inner")


# ===========================================================================
# CLI entry wiring: a live lock refuses both materializers up front
# ===========================================================================
def test_p2_cli_entry_refuses_live_lock_on_resume(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    work = tmp_path / "work"
    work.mkdir()

    with WorkRootLock(work):
        with pytest.raises(WorkRootLockError, match="locked by a live"):
            p2_tool.main([
                "--dataset-root", str(dataset), "--p2-root", str(tmp_path / "p2"),
                "--work-root", str(work), "--input-json", str(tmp_path / "input.json"),
                "--resume-raw-staging",
            ])


def test_dual_cli_entry_refuses_live_lock_on_resume(tmp_path):
    for name in ("p2_cache", "p23_table", "dataset"):
        (tmp_path / name).mkdir()
    for name in ("p2_manifest.json", "p23_audit.json", "gate1.py", "input.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()

    with WorkRootLock(work):
        with pytest.raises(WorkRootLockError, match="locked by a live"):
            dual_tool.main([
                "--p2-cache-root", str(tmp_path / "p2_cache"),
                "--p2-cache-manifest", str(tmp_path / "p2_manifest.json"),
                "--p23-table-root", str(tmp_path / "p23_table"),
                "--p23-table-audit", str(tmp_path / "p23_audit.json"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--gate1-script", str(tmp_path / "gate1.py"),
                "--input-json", str(tmp_path / "input.json"),
                "--work-root", str(work), "--resume",
            ])


def test_cli_lock_released_after_failed_run_beyond_lock_acquisition(tmp_path):
    # Drive the p2 CLI past guard+lock acquisition into the run body (which
    # fails on the empty dataset); the advisory lock must be released so a
    # follow-up attempt is not refused as concurrent.
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    work = tmp_path / "work"

    argv = [
        "--dataset-root", str(dataset), "--p2-root", str(tmp_path / "p2"),
        "--work-root", str(work), "--input-json", str(tmp_path / "input.json"),
    ]
    # Fails inside the run body: the empty dataset has no ordered_paths.txt.
    with pytest.raises(FileNotFoundError, match="ordered_paths"):
        p2_tool.main(argv)
    assert not (work / LOCK_FILE_NAME).exists()
    # The work root itself was created by the guard; a retry passes the lock
    # gate again (and fails later for the same non-lock reason).
    with pytest.raises(FileNotFoundError, match="ordered_paths"):
        p2_tool.main(["--overwrite", *argv])
    assert not (work / LOCK_FILE_NAME).exists()


# ===========================================================================
# BoundedEnvCache: never holds more than max_open environments open
# ===========================================================================
class _FakeEnv:
    def __init__(self, path, live: set):
        self.path = path
        self.closed = False
        self._live = live
        live.add(path)

    def close(self):
        if not self.closed:
            self.closed = True
            self._live.discard(self.path)


def test_bounded_env_cache_caps_concurrently_open_envs():
    """The cache never holds more than ``max_open`` environments open at once."""
    live: set = set()
    opened: list = []

    def opener(path):
        env = _FakeEnv(path, live)
        opened.append(path)
        return env

    def shard(index: int) -> Path:
        return Path(f"/shard/{index}")

    cache = BoundedEnvCache(max_open=2, opener=opener)

    peak = 0
    for index in range(5):
        env = cache.get(f"/shard/{index}")
        peak = max(peak, cache.open_count)
        assert cache.open_count <= 2, "cache exceeded its concurrency cap"
        assert len(live) <= 2, "more than max_open real envs are alive"
        assert env.path == shard(index)

    # Five distinct shards were requested but only two are ever kept alive; the
    # three least-recently-used environments must have been closed on eviction.
    assert opened == [shard(i) for i in range(5)]
    assert peak == 2
    assert cache.open_count == 2
    assert len(live) == 2
    assert live == {shard(3), shard(4)}

    # A cache hit must not reopen the environment.
    before = len(opened)
    again = cache.get(shard(4))
    assert again.path == shard(4)
    assert len(opened) == before

    # Re-requesting an evicted shard reopens it, still within the cap.
    reopened = cache.get(shard(0))
    assert reopened.path == shard(0)
    assert opened[-1] == shard(0)
    assert cache.open_count == 2
    assert len(live) == 2

    cache.close_all()
    assert cache.open_count == 0
    assert live == set()


def test_bounded_env_cache_rejects_non_positive_cap():
    with pytest.raises(ValueError, match="at least 1"):
        BoundedEnvCache(max_open=0)


def test_transactional_writer_put_new_rejects_duplicate_key(tmp_path):
    writer = TransactionalLMDBWriter(tmp_path / "out.lmdb", map_size=1 << 20)
    try:
        key = (0).to_bytes(4, "big")
        writer.put_new(key, b"first", exists_message="row already exists")
        with pytest.raises(ValueError, match="row already exists"):
            writer.put_new(key, b"second", exists_message="row already exists")
        with writer.env.begin() as txn:
            assert txn.get(key) == b"first"
    finally:
        writer.close()

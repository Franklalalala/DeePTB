from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from dptb.data.materialization.lmdb_io import (
    BoundedEnvCache,
    TransactionalLMDBWriter,
    open_read_env,
    open_write_env,
)


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
        # BoundedEnvCache normalizes the requested path through Path() before
        # invoking the opener, so the fake records that normalized key.
        env = _FakeEnv(path, live)
        opened.append(path)
        return env

    def shard(index: int) -> Path:
        return Path(f"/shard/{index}")

    cache = BoundedEnvCache(max_open=2, opener=opener)
    assert TransactionalLMDBWriter.env_cache_cls is BoundedEnvCache

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


def test_bounded_env_cache_caps_real_lmdb_file_descriptors(tmp_path):
    """Same cap guarantee, exercised against real read-only LMDB environments."""

    shard_paths = []
    for index in range(5):
        path = tmp_path / f"data.{index:04d}.lmdb"
        # Small map so the test does not need a 1 TiB reservation on Windows.
        env = open_write_env(path, map_size=1 << 20)
        with env.begin(write=True) as txn:
            txn.put((0).to_bytes(4, "big"), pickle.dumps({"shard": index}))
        env.close()
        shard_paths.append(path)

    cache = BoundedEnvCache(max_open=2, opener=open_read_env)
    try:
        for path in shard_paths:
            env = cache.get(path.resolve())
            with env.begin() as txn:
                record = pickle.loads(txn.get((0).to_bytes(4, "big")))
            assert "shard" in record
            assert cache.open_count <= 2
        assert cache.open_count == 2
    finally:
        cache.close_all()
    assert cache.open_count == 0


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

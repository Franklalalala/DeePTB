"""LMDB environment helpers for the durable non-SOC materializers.

Two concerns are extracted here from the two ``tools/materialize_*`` CLIs:

* how a durable materialization *opens* its read/write environments
  (:data:`MAP_SIZE`, :func:`open_write_env`, :func:`open_read_env`), and
* the two access patterns that protect against corruption/FD exhaustion:
  :class:`TransactionalLMDBWriter` (append-only, put-if-absent) and
  :class:`BoundedEnvCache` (lazy, LRU-bounded read environments so a run does
  not hold one file descriptor per source shard for its whole lifetime).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable

import lmdb


# 1 TiB sparse map.  Both materializers used this identical value; keep it as
# the single source of truth so the re-export shims stay byte-compatible.
MAP_SIZE = 1 << 40

# Default cap on concurrently-open source-shard read environments.  Source
# rows are visited in shard order, so even a small cap avoids reopen thrash
# while bounding the number of open file descriptors regardless of shard count.
DEFAULT_MAX_OPEN_READ_ENVS = 16


def open_write_env(path: Path | str, *, map_size: int = MAP_SIZE) -> lmdb.Environment:
    """Open (creating the directory) a single-DB writable LMDB environment.

    ``map_size`` defaults to the shared 1 TiB sparse reservation used by the
    production materializers; it is exposed only so that tests can request a
    small map on platforms (e.g. Windows) that cannot back a 1 TiB mmap.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(path),
        map_size=map_size,
        subdir=True,
        lock=True,
        readahead=False,
        max_dbs=1,
    )


def open_read_env(path: Path | str) -> lmdb.Environment:
    """Open a read-only, lock-free LMDB environment for a committed shard."""

    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=512,
        subdir=True,
    )


class TransactionalLMDBWriter:
    """Append-only writer with a put-if-absent guard.

    The durable materializers only ever *append* new rows beyond an already
    validated committed prefix.  Writing a key that already exists therefore
    signals that the resume bookkeeping and the physical LMDB disagree, which
    must fail loudly instead of silently overwriting a committed record.
    """

    #: The lazy/bounded read-env cache companion lives in the same module and
    #: is the mechanism the writer's driver uses for source shards.
    env_cache_cls = None  # set to BoundedEnvCache after its definition below

    def __init__(self, path: Path | str, *, map_size: int = MAP_SIZE):
        self.path = Path(path)
        self.env = open_write_env(self.path, map_size=map_size)

    def put_new(self, key: bytes, payload: bytes, *, exists_message: str) -> None:
        """Store ``payload`` at ``key``, failing if the key already exists."""

        with self.env.begin(write=True) as txn:
            if txn.get(key) is not None:
                raise ValueError(exists_message)
            txn.put(key, payload)

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "TransactionalLMDBWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class BoundedEnvCache:
    """Lazily open read environments, keeping at most ``max_open`` alive.

    Environments are opened on first request and cached in LRU order.  When the
    cache would exceed its cap the least-recently-used environment is closed and
    evicted.  Callers must not hold an open transaction across a subsequent
    :meth:`get` for a *different* path, which is the case for the source-shard
    readers here (each ``env.begin()`` transaction is opened and closed around a
    single ``txn.get``).
    """

    def __init__(
        self,
        *,
        max_open: int = DEFAULT_MAX_OPEN_READ_ENVS,
        opener: Callable[[Path], lmdb.Environment] = open_read_env,
    ):
        if int(max_open) < 1:
            raise ValueError("max_open must be at least 1.")
        self._max_open = int(max_open)
        self._opener = opener
        self._envs: "OrderedDict[Path, lmdb.Environment]" = OrderedDict()

    @property
    def max_open(self) -> int:
        return self._max_open

    @property
    def open_count(self) -> int:
        """Number of environments currently held open by the cache."""

        return len(self._envs)

    def get(self, path: Path | str) -> lmdb.Environment:
        key = Path(path)
        env = self._envs.get(key)
        if env is not None:
            self._envs.move_to_end(key)
            return env
        env = self._opener(key)
        self._envs[key] = env
        while len(self._envs) > self._max_open:
            _evicted_path, evicted_env = self._envs.popitem(last=False)
            evicted_env.close()
        return env

    def close_all(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()

    def __enter__(self) -> "BoundedEnvCache":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close_all()


TransactionalLMDBWriter.env_cache_cls = BoundedEnvCache

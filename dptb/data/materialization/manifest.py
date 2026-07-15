"""Atomic manifest persistence for the durable materializers.

The materializers protect against corrupt/partial output by only ever writing
manifests through a temp-file + ``os.replace`` swap, and by keeping a
``manifest.partial.json`` alongside the final ``manifest.json``.  The exact byte
layout (``indent=2``, ``sort_keys=True``, trailing newline) is part of the
on-disk contract that resume validation and downstream audits depend on, so it
must not change here.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


# The non-SOC serial P2 cache schema string.  It is the single source of truth
# shared between the P2 materializer (its ``SCHEMA``/heartbeat schema) and the
# dual-prior materializer (which validates it as the parent-cache lineage).
NONSOC_P2_CACHE_SCHEMA = "deeptb.nonsoc_p2_cache/v1"


def _fsync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory entry so a rename survives power loss.

    Directory fsync is a POSIX durability idiom; Windows cannot open a
    directory with :func:`os.open`, and some filesystems refuse to fsync one.
    Both cases degrade gracefully to a no-op -- the data file itself has
    already been fsynced by the caller.
    """

    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomically and durably write ``payload`` as canonical pretty JSON.

    The payload is serialized with ``indent=2`` and ``sort_keys=True`` plus a
    trailing newline (exact legacy bytes, including platform newline
    translation) into a UUID-suffixed sibling temp file, flushed and fsynced,
    then swapped into place with :func:`os.replace`, followed by a best-effort
    fsync of the parent directory.  A crash at any point leaves either the old
    complete file or the new complete file -- never a partial final file; the
    temp file is removed if the write or swap fails.  Preserving the exact
    serialization keeps existing manifests/heartbeats/identity files
    byte-compatible.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # UUID suffix: concurrent writers (or a takeover after a crash) never
    # collide on the temp name, so an interrupted write cannot corrupt a
    # later one's in-flight temp file.
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # newline translation intentionally matches the legacy
        # ``Path.write_text`` behaviour so on-disk bytes are unchanged.
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)


class ManifestStore:
    """Own the ``manifest.partial.json`` / ``manifest.json`` pair for a run."""

    PARTIAL_NAME = "manifest.partial.json"
    FINAL_NAME = "manifest.json"

    def __init__(self, work_root: Path | str):
        self.work_root = Path(work_root)

    @property
    def partial_path(self) -> Path:
        return self.work_root / self.PARTIAL_NAME

    @property
    def final_path(self) -> Path:
        return self.work_root / self.FINAL_NAME

    def write_partial(self, manifest: Mapping[str, Any]) -> None:
        write_json(self.partial_path, manifest)

    def write_final(self, manifest: Mapping[str, Any]) -> None:
        write_json(self.final_path, manifest)

    def read_partial(self) -> dict[str, Any]:
        return json.loads(self.partial_path.read_text(encoding="utf-8"))

    def partial_exists(self) -> bool:
        return self.partial_path.is_file()

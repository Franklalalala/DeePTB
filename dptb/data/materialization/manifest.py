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
from pathlib import Path
from typing import Any, Mapping


# The non-SOC serial P2 cache schema string.  It is the single source of truth
# shared between the P2 materializer (its ``SCHEMA``/heartbeat schema) and the
# dual-prior materializer (which validates it as the parent-cache lineage).
NONSOC_P2_CACHE_SCHEMA = "deeptb.nonsoc_p2_cache/v1"


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomically write ``payload`` as canonical pretty JSON.

    The payload is serialized with ``indent=2`` and ``sort_keys=True`` plus a
    trailing newline, written to a sibling ``*.tmp`` file and then swapped into
    place with :func:`os.replace`.  Preserving these exact bytes keeps existing
    manifests/heartbeats/identity files byte-compatible.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


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

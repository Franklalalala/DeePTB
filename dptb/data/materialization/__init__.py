"""Durable-materialization glue extracted from the ``tools/materialize_*`` CLIs.

This package hosts the reusable, behaviour-preserving components that the two
non-SOC cache materializers previously duplicated (or cross-imported as private
helpers): LMDB IO with a bounded read-env cache, atomic manifest persistence,
resume journalling, geometry identity, the disjoint work-root guard, and the
strict-reload dataset auditor.

Fingerprint primitives (RME/graph/basis) intentionally are *not* here -- they
remain in :mod:`dptb.data.interfaces.p2_contract`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .audit import DatasetAuditor
from .geometry import (
    GEOMETRY_TOLERANCE_ANGSTROM,
    reverse_edge_rows,
    structure_geometry_bohr,
)
from .hashing import bytes_sha256, json_sha256, sha256_file
from .lmdb_io import (
    DEFAULT_MAX_OPEN_READ_ENVS,
    MAP_SIZE,
    BoundedEnvCache,
    TransactionalLMDBWriter,
    open_read_env,
    open_write_env,
)
from .manifest import NONSOC_P2_CACHE_SCHEMA, ManifestStore, write_json
from .resume import (
    ResumeJournal,
    drop_uncommitted_tail,
    identity_signature_matches,
    sign_identity,
    write_heartbeat,
)
from .workroot import guard_work_root


def load_module(path: Path, name: str) -> ModuleType:
    """Import a standalone Python file (e.g. a gate1 STRU parser) by path."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper module {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


__all__ = [
    # lmdb_io
    "MAP_SIZE",
    "DEFAULT_MAX_OPEN_READ_ENVS",
    "BoundedEnvCache",
    "TransactionalLMDBWriter",
    "open_read_env",
    "open_write_env",
    # manifest
    "NONSOC_P2_CACHE_SCHEMA",
    "ManifestStore",
    "write_json",
    # hashing
    "bytes_sha256",
    "json_sha256",
    "sha256_file",
    # resume
    "ResumeJournal",
    "drop_uncommitted_tail",
    "identity_signature_matches",
    "sign_identity",
    "write_heartbeat",
    # geometry
    "GEOMETRY_TOLERANCE_ANGSTROM",
    "reverse_edge_rows",
    "structure_geometry_bohr",
    # workroot
    "guard_work_root",
    # audit
    "DatasetAuditor",
    # utility
    "load_module",
]

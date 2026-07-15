"""Small content-hashing primitives shared by the durable materializers.

These are deliberately *not* the RME/graph/basis fingerprint helpers, which
live in :mod:`dptb.data.interfaces.p2_contract` and must not be re-hosted here.
The helpers below hash raw bytes, whole files and canonical-JSON payloads, and
are the building blocks the manifest / resume / geometry glue relies on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path | str) -> str:
    """Stream a file through SHA256 in 1 MiB chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    """SHA256 of an in-memory byte string (e.g. a pickled LMDB record)."""

    return hashlib.sha256(payload).hexdigest()


def json_sha256(payload: Mapping[str, Any]) -> str:
    """Canonical-JSON SHA256 used for identity signatures.

    The serialization is fixed (``sort_keys``, compact separators, ASCII) so
    that the digest is stable across processes and platforms.  This is the
    exact encoding both durable materializers used for their identity hashes.
    """

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()

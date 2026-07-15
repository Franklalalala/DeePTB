"""Resume/journalling primitives shared by the durable materializers.

A durable materialization protects itself against interruption with three
mechanisms, all extracted here:

* a **heartbeat** file rewritten after every unit of work;
* a canonical-JSON **identity signature** so a partial run can only be resumed
  against byte-identical bound inputs; and
* **tail truncation**, which drops a single uncommitted LMDB row written ahead
  of the partial manifest before the committed prefix is trusted.

The tool-specific identity *payloads* and migration policies stay in their
CLIs; this module owns the reusable signing / verification / heartbeat / tail
mechanics so both tools share one implementation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import lmdb

from .hashing import json_sha256
from .manifest import write_json


def write_heartbeat(work_root: Path | str, *, schema: str, **payload: Any) -> None:
    """Rewrite ``heartbeat.json`` with a schema tag and wall-clock timestamp."""

    write_json(
        Path(work_root) / "heartbeat.json",
        {"schema": schema, "updated_unix": time.time(), **payload},
    )


def sign_identity(
    payload: Mapping[str, Any], *, digest_key: str = "identity_sha256"
) -> dict[str, Any]:
    """Return a copy of ``payload`` with ``digest_key`` set to its JSON digest.

    The digest is computed over the payload *without* the digest key, matching
    the way both materializers stamped their identity documents.
    """

    unsigned = {key: value for key, value in payload.items() if key != digest_key}
    signed = dict(unsigned)
    signed[digest_key] = json_sha256(unsigned)
    return signed


def identity_signature_matches(
    payload: Mapping[str, Any], *, digest_key: str = "identity_sha256"
) -> bool:
    """Whether ``payload[digest_key]`` equals the recomputed JSON digest."""

    declared = payload.get(digest_key)
    if not isinstance(declared, str):
        return False
    unsigned = {key: value for key, value in payload.items() if key != digest_key}
    return declared.strip().lower() == json_sha256(unsigned)


def drop_uncommitted_tail(
    txn: "lmdb.Transaction",
    *,
    committed_rows: int,
    key_width: int = 4,
    tail_error: str = "Cannot resume: failed to remove uncommitted LMDB tail.",
    length_error: str | None = None,
) -> int:
    """Reconcile a physical LMDB with a partial manifest's committed row count.

    A single row may have been written to LMDB immediately before the process
    was interrupted, ahead of the partial-manifest commit.  If exactly one such
    tail row exists it is deleted; any other divergence between the physical
    entry count and ``committed_rows`` is fatal.
    """

    entries = int(txn.stat()["entries"])
    if entries == committed_rows + 1:
        tail_key = committed_rows.to_bytes(length=key_width, byteorder="big")
        if not txn.delete(tail_key):
            raise KeyError(tail_error)
        entries -= 1
    if entries != committed_rows:
        if length_error is None:
            length_error = (
                f"Cannot resume: output has {entries} rows but partial manifest "
                f"commits {committed_rows}."
            )
        raise ValueError(length_error)
    return entries


class ResumeJournal:
    """Bundle the resume mechanics for one work root / schema.

    The class is a thin, stateless-per-call convenience over the module-level
    functions so a CLI can hold a single object; the functions remain available
    for call sites (e.g. monkeypatchable module-level ``_heartbeat`` shims) that
    need to route through their own module namespace.
    """

    def __init__(self, *, schema: str):
        self.schema = schema

    def heartbeat(self, work_root: Path | str, **payload: Any) -> None:
        write_heartbeat(work_root, schema=self.schema, **payload)

    @staticmethod
    def sign_identity(
        payload: Mapping[str, Any], *, digest_key: str = "identity_sha256"
    ) -> dict[str, Any]:
        return sign_identity(payload, digest_key=digest_key)

    @staticmethod
    def identity_signature_matches(
        payload: Mapping[str, Any], *, digest_key: str = "identity_sha256"
    ) -> bool:
        return identity_signature_matches(payload, digest_key=digest_key)

    @staticmethod
    def drop_uncommitted_tail(
        txn: "lmdb.Transaction", *, committed_rows: int, **kwargs: Any
    ) -> int:
        return drop_uncommitted_tail(txn, committed_rows=committed_rows, **kwargs)

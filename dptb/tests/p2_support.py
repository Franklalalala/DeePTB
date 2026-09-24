"""Shared helpers for the P2/P23 table tests (not collected; import from dptb.tests.p2_support)."""
import hashlib
from pathlib import Path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

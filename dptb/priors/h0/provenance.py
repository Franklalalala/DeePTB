from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ARTIFACT_SCHEMA = "h0rebuild.artifact/v1"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot canonicalize {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def array_sha256(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(arr.shape)))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def structure_fingerprint(cell_bohr: np.ndarray, species: list[str], frac: np.ndarray) -> str:
    payload = {
        "cell_bohr": np.asarray(cell_bohr, dtype=np.float64).round(14).tolist(),
        "species": [str(item) for item in species],
        "fractional_coordinates": np.asarray(frac, dtype=np.float64).round(14).tolist(),
    }
    return canonical_sha256(payload)


def input_file_manifest(species_data: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for symbol in sorted(species_data):
        data = species_data[symbol]
        item: dict[str, str] = {}
        for name, obj in (("orb", data.orb), ("upf", data.upf)):
            source = getattr(obj, "source", None)
            if source is None:
                continue
            path = Path(source).resolve()
            item[f"{name}_path"] = str(path)
            item[f"{name}_sha256"] = obj.metadata.get('offline_source_sha256') or sha256_file(path)
        manifest[str(symbol)] = item
    return manifest

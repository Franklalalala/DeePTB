from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .assemble import AssemblyResult
from .constants import energy_from_ry
from .models import BlockKey
from .provenance import ARTIFACT_SCHEMA, canonical_sha256, sha256_file


def _pack_blocks(blocks: Mapping[BlockKey, np.ndarray], scale: float = 1.0):
    keys = sorted(blocks, key=lambda k: k.as_tuple())
    key_arr = np.asarray([k.as_tuple() for k in keys], dtype=np.int64)
    shapes = np.asarray([blocks[k].shape for k in keys], dtype=np.int64)
    offsets = [0]
    data = []
    for key in keys:
        flat = np.asarray(blocks[key] * scale, dtype=np.complex128).ravel()
        data.append(flat)
        offsets.append(offsets[-1] + flat.size)
    packed = np.concatenate(data) if data else np.empty(0, dtype=np.complex128)
    return key_arr, shapes, np.asarray(offsets, dtype=np.int64), packed


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def save_result(
    result: AssemblyResult,
    path: str | Path,
    energy_unit: str = "eV",
) -> tuple[Path, Path]:
    """Write a versioned sparse AO-block artifact and its provenance JSON."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = float(energy_from_ry(1.0, energy_unit))
    h_keys, h_shapes, h_offsets, h_data = _pack_blocks(result.h_blocks_ry, scale)
    s_keys, s_shapes, s_offsets, s_data = _pack_blocks(result.s_blocks, 1.0)
    arrays = {
        "artifact_schema": np.asarray(ARTIFACT_SCHEMA),
        "h_keys": h_keys,
        "h_shapes": h_shapes,
        "h_offsets": h_offsets,
        "h_data_real": h_data.real,
        "h_data_imag": h_data.imag,
        "s_keys": s_keys,
        "s_shapes": s_shapes,
        "s_offsets": s_offsets,
        "s_data_real": s_data.real,
        "s_data_imag": s_data.imag,
        "orbital_counts": np.asarray(result.orbital_counts, dtype=np.int64),
    }
    np.savez_compressed(path, **arrays)

    meta = dict(result.metadata)
    meta.update(
        {
            "artifact_schema": ARTIFACT_SCHEMA,
            "output_energy_unit": str(energy_unit),
            "block_key": "(atom_i, atom_j, R1, R2, R3), ket center is r_j + R@cell",
            "atom_index_base": 0,
            "npz_layout": {
                "keys": "N x 5 int64",
                "shapes": "N x 2 int64",
                "offsets": "N+1 int64 offsets into flat complex128 data",
            },
            "number_of_h_blocks": len(result.h_blocks_ry),
            "number_of_s_blocks": len(result.s_blocks),
            "npz_sha256": sha256_file(path),
        }
    )
    # Hash the semantic metadata before adding the hash field itself.
    meta["metadata_content_sha256"] = canonical_sha256(meta)
    meta_path = _metadata_path(path)
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return path, meta_path


def load_metadata(path: str | Path) -> dict:
    path = Path(path)
    meta_path = path if path.suffix.lower() == ".json" else _metadata_path(path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def validate_artifact(
    path: str | Path,
    *,
    require_schema: str = ARTIFACT_SCHEMA,
    verify_hash: bool = True,
) -> dict:
    """Fail closed on schema/hash/packing mismatches and return metadata."""
    path = Path(path)
    meta = load_metadata(path)
    if meta.get("artifact_schema") != require_schema:
        raise ValueError(
            f"Unsupported artifact schema {meta.get('artifact_schema')!r}; "
            f"expected {require_schema!r}."
        )
    if verify_hash:
        actual = sha256_file(path)
        if actual != meta.get("npz_sha256"):
            raise ValueError(f"NPZ SHA256 mismatch for {path}: {actual} != {meta.get('npz_sha256')}")
        semantic = dict(meta)
        recorded = semantic.pop("metadata_content_sha256", None)
        actual_meta = canonical_sha256(semantic)
        if recorded != actual_meta:
            raise ValueError(
                f"Metadata content SHA256 mismatch for {_metadata_path(path)}: "
                f"{actual_meta} != {recorded}"
            )
    with np.load(path, allow_pickle=False) as archive:
        schema = str(archive["artifact_schema"].item())
        if schema != require_schema:
            raise ValueError(f"NPZ embedded schema {schema!r} != {require_schema!r}")
        for prefix in ("h", "s"):
            keys = archive[f"{prefix}_keys"]
            shapes = archive[f"{prefix}_shapes"]
            offsets = archive[f"{prefix}_offsets"]
            real = archive[f"{prefix}_data_real"]
            imag = archive[f"{prefix}_data_imag"]
            if keys.ndim != 2 or keys.shape[1] != 5:
                raise ValueError(f"{prefix}_keys must have shape [N,5]")
            if shapes.shape != (keys.shape[0], 2):
                raise ValueError(f"{prefix}_shapes is inconsistent with {prefix}_keys")
            if offsets.shape != (keys.shape[0] + 1,) or offsets[0] != 0:
                raise ValueError(f"{prefix}_offsets is malformed")
            if offsets[-1] != real.size or real.shape != imag.shape:
                raise ValueError(f"{prefix} packed data lengths are inconsistent")
            expected = np.prod(shapes, axis=1, dtype=np.int64)
            if not np.array_equal(np.diff(offsets), expected):
                raise ValueError(f"{prefix} offsets do not match block shapes")
    return meta


def load_blocks(
    path: str | Path,
    prefix: str = "h",
    *,
    validate: bool = True,
) -> dict[BlockKey, np.ndarray]:
    if validate:
        validate_artifact(path)
    with np.load(path, allow_pickle=False) as archive:
        keys = archive[f"{prefix}_keys"]
        shapes = archive[f"{prefix}_shapes"]
        offsets = archive[f"{prefix}_offsets"]
        data = archive[f"{prefix}_data_real"] + 1j * archive[f"{prefix}_data_imag"]
        out: dict[BlockKey, np.ndarray] = {}
        for n, raw in enumerate(keys):
            key = BlockKey(int(raw[0]), int(raw[1]), tuple(int(x) for x in raw[2:5]))
            out[key] = data[offsets[n] : offsets[n + 1]].reshape(tuple(shapes[n]))
        return out

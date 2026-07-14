#!/usr/bin/env python3
"""Build paired Full-H/P2 and H0->dH cache views for non-SOC ABACUS data.

The authoritative source remains the raw ABACUS H/H0 CSR quartet.  P2 comes
either from a preserved direct-oracle ``r_keys/P2`` NPZ artifact or from a
completed radial table.  Both paths are rotated from ABACUS AO order to DeePTB
order, converted from Ry to eV, and then passed through DeePTB's own
``LMDBDataset``/``OrbitalMapper`` conversion path.

Two compact LMDB views are emitted:

``full_h``
    absolute Full-H targets plus cached H0/P2 RME features and H0/P2 AO blocks;
``h0_delta``
    cached ``H-H0`` targets plus cached H0 RME/AO blocks for the traditional
    H0-conditioned residual baseline.

The split is deterministic and a heartbeat is updated after every case.  Raw
staging LMDBs are removed only after a successful build unless explicitly kept.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import lmdb
import numpy as np
import torch

import dptb.data.AtomicDataDict as AtomicDataDict
from dptb.data import AtomicData
from dptb.data.interfaces.abacus import OrbAbacus2DeepTB, _abacus_parse
from dptb.data.interfaces.h0_lmdb_helper import (
    _build_context_dataset,
    _dataset_args_from_input,
)
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    BASIS_FINGERPRINT_KEY,
    EDGE_GRAPH_FINGERPRINT_KEY,
    FULL_H_TARGET_FINGERPRINT_KEY,
    H0_RESIDUAL_SEMANTICS,
    P2_BLOCK_FINGERPRINT_KEY,
    P2_BUNDLE_FINGERPRINT_KEY,
    P2_RME_FINGERPRINT_KEY,
    P2_SAMPLE_SCHEMA,
    P2_SOURCE_FINGERPRINT_KEY,
    ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
    ROW_ALIGNED_DATA_FINGERPRINT_KEY,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
    canonical_edge_graph,
    edge_graph_fingerprint,
    fingerprint_fields,
    fingerprint_present_row_aligned_fields,
    fingerprint_text_fields,
    mapper_basis_fingerprint,
)
from dptb.data.interfaces.p2_table import (
    P2_COMPONENT_NUMERICAL_GATE_SCHEMA,
    P2TableAssembler,
    P2TableStore,
)


RY_TO_EV = 13.605698
SCHEMA = "deeptb.nonsoc_p2_cache/v1"
RAW_STAGING_IDENTITY_SCHEMA = "deeptb.nonsoc_p2_raw_staging_identity/v2"
RAW_DATASET_SOURCE_IDENTITY_SCHEMA = "deeptb.nonsoc_p2_dataset_source/v1"
RAW_CASE_SOURCE_IDENTITY_SCHEMA = "deeptb.nonsoc_p2_case_source/v1"
RAW_STAGING_IDENTITY_NAME = "identity.json"
RAW_CASE_SOURCE_FINGERPRINT_KEY = "raw_case_source_fingerprint"
MAP_SIZE = 1 << 40
DEFAULT_P2_HERMITIAN_MISMATCH_TOLERANCE_EV = 5.0e-3

_RAW_ABACUS_SOURCE_FILES = {
    "structure_log": Path("OUT.ABACUS") / "running_scf.log",
    "full_h": Path("OUT.ABACUS") / "data-HR-sparse_SPIN0.csr",
    "h0": Path("OUT.ABACUS") / "data-HR0_SPIN0.csr",
}

_BASE_FIELDS = (
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.ATOMIC_NUMBERS_KEY,
    AtomicDataDict.PBC_KEY,
)
_GRAPH_FIELDS = (
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
)
_TARGET_FIELDS = (
    AtomicDataDict.NODE_DELTA_HAMIL_BLOCKS_KEY,
    AtomicDataDict.EDGE_DELTA_HAMIL_BLOCKS_KEY,
    AtomicDataDict.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
)
_FULL_H_TARGET_FIELDS = (
    AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY,
    AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY,
    AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
)
_H0_BLOCK_FIELDS = (
    AtomicDataDict.NODE_H0_BLOCKS_KEY,
    AtomicDataDict.EDGE_H0_BLOCKS_KEY,
    AtomicDataDict.NODE_H0_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_H0_BLOCK_SHAPE_KEY,
)
_P2_BLOCK_FIELDS = (
    AtomicDataDict.NODE_P2_BLOCKS_KEY,
    AtomicDataDict.EDGE_P2_BLOCKS_KEY,
    AtomicDataDict.NODE_P2_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_P2_BLOCK_SHAPE_KEY,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper module {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _required_block_keys_from_graph(
    atomic_numbers: Any,
    edge_index: Any,
    edge_cell_shift: Any,
) -> list[tuple[int, int, int, int, int]]:
    """Return onsite plus the exact ordered main-graph ``(i,j,R)`` keys."""

    edge, shift = canonical_edge_graph(
        atomic_numbers, edge_index, edge_cell_shift
    )
    n_atoms = int(np.asarray(atomic_numbers).reshape(-1).shape[0])
    keys = {(index, index, 0, 0, 0) for index in range(n_atoms)}
    keys.update(
        (int(u), int(v), int(r0), int(r1), int(r2))
        for (u, v), (r0, r1, r2) in zip(edge.T.tolist(), shift.tolist())
    )
    return sorted(keys)


def complete_sparse_zero_blocks(
    blocks: dict[str, np.ndarray],
    *,
    required_keys: Iterable[tuple[int, int, int, int, int]],
    basis_lines: bytes | str,
    atom_count: int,
    label: str,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int, int, int, int]]]:
    """Make ABACUS sparse-zero semantics explicit on the model graph.

    ABACUS omits blocks below its sparse-output threshold.  A graph edge can
    therefore be absent in both directed H(R) dictionaries even though it is
    inside the model cutoff.  We materialize those *source-declared sparse
    zeros* before block conversion and record their keys in provenance.  This
    helper is intentionally not used for P2: every requested P2 block must be
    assembled or loaded explicitly.
    """

    if not isinstance(blocks, dict) or not blocks:
        raise ValueError(f"{label} must be a non-empty AO-block dictionary.")
    atom_shells = parse_basis_lines(basis_lines, atom_count)
    atom_norb = [sum(2 * l + 1 for l in shells) for shells in atom_shells]
    sample = np.asarray(next(iter(blocks.values())))
    if np.iscomplexobj(sample):
        raise NotImplementedError(
            f"{label} sparse-zero completion is non-SOC only; got {sample.dtype}."
        )
    if not np.issubdtype(sample.dtype, np.floating):
        raise TypeError(f"{label} blocks must be floating point, got {sample.dtype}.")

    completed = dict(blocks)
    filled: list[tuple[int, int, int, int, int]] = []
    for key in required_keys:
        i, j, rx, ry, rz = (int(value) for value in key)
        if i < 0 or i >= atom_count or j < 0 or j >= atom_count:
            raise ValueError(f"{label} graph key {key} has invalid atom indices.")
        direct = f"{i}_{j}_{rx}_{ry}_{rz}"
        reverse = f"{j}_{i}_{-rx}_{-ry}_{-rz}"
        expected = (atom_norb[i], atom_norb[j])
        if direct in completed:
            value = np.asarray(completed[direct])
            if value.shape != expected:
                raise ValueError(
                    f"{label} block {direct} shape {value.shape} != {expected}."
                )
            if np.iscomplexobj(value) or not np.isfinite(value).all():
                raise ValueError(f"{label} block {direct} is not finite real non-SOC data.")
            continue
        if reverse in completed:
            value = np.asarray(completed[reverse])
            reverse_expected = (atom_norb[j], atom_norb[i])
            if value.shape != reverse_expected:
                raise ValueError(
                    f"{label} reverse block {reverse} shape {value.shape} "
                    f"!= {reverse_expected}."
                )
            if np.iscomplexobj(value) or not np.isfinite(value).all():
                raise ValueError(
                    f"{label} reverse block {reverse} is not finite real non-SOC data."
                )
            continue
        completed[direct] = np.zeros(expected, dtype=sample.dtype)
        filled.append(key)
    return completed, filled


def _block_key_set_fingerprint(
    keys: Iterable[tuple[int, int, int, int, int]],
) -> str:
    digest = hashlib.sha256()
    for key in sorted(tuple(int(value) for value in item) for item in keys):
        digest.update("_".join(str(value) for value in key).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _zero_completion_metadata(
    filled: Iterable[tuple[int, int, int, int, int]],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    keys = {tuple(int(value) for value in item) for item in filled}
    for encoded in (previous or {}).get("filled_keys", []):
        parsed = tuple(int(value) for value in str(encoded).split("_"))
        if len(parsed) != 5:
            raise ValueError(f"Invalid prior sparse-zero key {encoded!r}.")
        keys.add(parsed)
    ordered = sorted(keys)
    return {
        "semantics": "abacus_sparse_omission_is_zero",
        "filled_count": len(ordered),
        "filled_key_set_sha256": _block_key_set_fingerprint(ordered),
        "filled_keys": ["_".join(str(value) for value in key) for key in ordered],
    }


def project_non_soc_blocks_hermitian(
    blocks: dict[str, np.ndarray],
    *,
    required_keys: Iterable[tuple[int, int, int, int, int]],
    label: str = "P2",
    mismatch_tolerance: float = DEFAULT_P2_HERMITIAN_MISMATCH_TOLERANCE_EV,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Project graph-aligned real AO blocks onto exact Hermitian pairs.

    Radial interpolation, frame rotation, and the final float32 cast can leave
    tiny numerical differences between ``H_ij(R)`` and ``H_ji(-R)^T`` even
    when the radial table itself satisfies reciprocity.  Cache that noise only
    after an explicit pair projection, so later loaders see an exactly
    Hermitian prior instead of relying on a relaxed validation tolerance.  A
    mismatch above ``mismatch_tolerance`` is evidence of a convention/source
    error and fails before any averaging can hide it.
    """

    tolerance = float(mismatch_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Hermitian mismatch tolerance must be finite and non-negative.")
    output = {key: np.asarray(value) for key, value in blocks.items()}
    required = sorted(set(required_keys))
    required_set = set(required)
    missing = [key for key in required if "_".join(map(str, key)) not in output]
    if missing:
        raise ValueError(f"{label} is missing required blocks {missing[:5]}.")

    seen: set[tuple[int, int, int, int, int]] = set()
    pair_rows: list[tuple[str, str, np.ndarray, np.ndarray, float]] = []
    for key in required:
        if key in seen:
            continue
        i, j, rx, ry, rz = key
        reverse = (j, i, -rx, -ry, -rz)
        if reverse not in required_set:
            raise ValueError(f"{label} block {key} has no reverse graph block {reverse}.")
        forward_name = "_".join(map(str, key))
        reverse_name = "_".join(map(str, reverse))
        forward = np.asarray(output[forward_name])
        backward = np.asarray(output[reverse_name])
        if forward.dtype != backward.dtype:
            raise ValueError(
                f"{label} Hermitian pair {key}/{reverse} has dtype mismatch "
                f"{forward.dtype} != {backward.dtype}."
            )
        if forward.shape != backward.T.shape:
            raise ValueError(
                f"{label} Hermitian pair {key}/{reverse} has shape mismatch "
                f"{forward.shape} != transpose({backward.shape})."
            )
        reverse_t = backward.T.conj()
        mismatch = float(np.max(np.abs(forward - reverse_t), initial=0.0))
        pair_rows.append(
            (forward_name, reverse_name, forward, backward, mismatch)
        )
        seen.add(key)
        seen.add(reverse)

    mismatches = np.asarray([row[-1] for row in pair_rows], dtype=np.float64)
    mismatch_stats = {
        "pair_count": len(pair_rows),
        "max_pre_projection_mismatch": float(
            np.max(mismatches, initial=0.0)
        ),
        "mean_pre_projection_mismatch": float(
            np.mean(mismatches) if mismatches.size else 0.0
        ),
        "p95_pre_projection_mismatch": float(
            np.quantile(mismatches, 0.95) if mismatches.size else 0.0
        ),
        "p99_pre_projection_mismatch": float(
            np.quantile(mismatches, 0.99) if mismatches.size else 0.0
        ),
        "mismatch_tolerance": tolerance,
        "above_tolerance_count": int(np.count_nonzero(mismatches > tolerance)),
        "unit": "eV",
    }
    if mismatch_stats["above_tolerance_count"]:
        raise ValueError(
            f"{label} pre-projection Hermitian mismatch exceeds tolerance: "
            + json.dumps(mismatch_stats, sort_keys=True)
        )

    max_correction = 0.0
    for forward_name, reverse_name, forward, backward, _ in pair_rows:
        reverse_t = backward.T.conj()
        calculation_dtype = np.complex128 if np.iscomplexobj(forward) else np.float64
        average = 0.5 * (
            forward.astype(calculation_dtype, copy=False)
            + reverse_t.astype(calculation_dtype, copy=False)
        )
        projected = np.asarray(average, dtype=forward.dtype)
        correction = float(np.max(np.abs(projected - forward), initial=0.0))
        reverse_correction = float(
            np.max(np.abs(projected.T.conj() - backward), initial=0.0)
        )
        output[forward_name] = projected
        # Derive reverse from the already-cast forward block.  This makes
        # equality exact in storage, including for float32.
        output[reverse_name] = projected.T.conj().copy()
        max_correction = max(max_correction, correction, reverse_correction)
    return output, {
        **mismatch_stats,
        "max_projection_correction": max_correction,
        "method": "pair_average_exact_reverse",
    }


def _source_set_fingerprint(cases: list[Path], p2_root: Path) -> str:
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda item: item.name):
        path = p2_root / f"{case.name}.npz"
        digest.update(case.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _identity_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_qualified_table_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> None:
    if manifest.get("complete") is not True:
        raise ValueError(f"P2 table is not complete: {manifest_path}")
    gate = manifest.get("component_numerical_gate")
    if (
        not isinstance(gate, Mapping)
        or gate.get("schema") != P2_COMPONENT_NUMERICAL_GATE_SCHEMA
        or gate.get("status") != "pass"
    ):
        raise ValueError(
            "P2 table has not passed its persisted component numerical gate: "
            f"{manifest_path}"
        )


def _stable_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    before = resolved.stat()
    checksum = _sha256(resolved)
    after = resolved.stat()
    before_signature = (int(before.st_size), int(before.st_mtime_ns))
    after_signature = (int(after.st_size), int(after.st_mtime_ns))
    if before_signature != after_signature:
        raise ValueError(f"source file changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "size": int(after.st_size),
        "sha256": checksum,
    }


def _case_source_identity(
    case: Path,
    *,
    require_stru: bool,
) -> dict[str, Any]:
    resolved_case = case.resolve()
    if not resolved_case.is_dir():
        raise NotADirectoryError(resolved_case)
    files: dict[str, Any] = {}
    for role, relative in _RAW_ABACUS_SOURCE_FILES.items():
        file_identity = _stable_file_identity(resolved_case / relative)
        file_identity["relative_path"] = relative.as_posix()
        files[role] = file_identity
    stru_path = resolved_case / "STRU"
    if require_stru or stru_path.is_file():
        stru_identity = _stable_file_identity(stru_path)
        stru_identity["relative_path"] = "STRU"
        files["stru"] = stru_identity
    payload: dict[str, Any] = {
        "schema": RAW_CASE_SOURCE_IDENTITY_SCHEMA,
        "case_id": resolved_case.name,
        "case_path": str(resolved_case),
        "files": files,
    }
    payload["source_fingerprint"] = _identity_sha256(payload)
    return payload


def _dataset_source_identity(
    dataset_root: Path,
    cases: Iterable[Path],
    *,
    require_stru: bool,
) -> dict[str, Any]:
    resolved_root = dataset_root.resolve()
    ordered_path = resolved_root / "ordered_paths.txt"
    ordered_identity = _stable_file_identity(ordered_path)
    case_identities = [
        _case_source_identity(case, require_stru=require_stru) for case in cases
    ]
    paths = [str(item["case_path"]) for item in case_identities]
    if len(paths) != len(set(paths)):
        raise ValueError("selected P2 cases contain duplicate resolved paths")
    payload: dict[str, Any] = {
        "schema": RAW_DATASET_SOURCE_IDENTITY_SCHEMA,
        "dataset_root": str(resolved_root),
        "ordered_paths": ordered_identity,
        "selected_cases": case_identities,
    }
    payload["identity_sha256"] = _identity_sha256(payload)
    return payload


def _validate_dataset_source_identity(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != RAW_DATASET_SOURCE_IDENTITY_SCHEMA:
        raise ValueError("dataset source identity has the wrong schema")
    declared = str(payload.get("identity_sha256", "")).strip().lower()
    if len(declared) != 64:
        raise ValueError("dataset source identity lacks a SHA256 checksum")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    if _identity_sha256(unsigned) != declared:
        raise ValueError("dataset source identity SHA256 is inconsistent")


def _case_source_fingerprint_map(
    dataset_source_identity: Mapping[str, Any],
) -> dict[str, str]:
    _validate_dataset_source_identity(dataset_source_identity)
    output: dict[str, str] = {}
    for item in dataset_source_identity.get("selected_cases", []):
        if not isinstance(item, Mapping):
            raise ValueError("dataset source identity case entry is not a mapping")
        path = str(item.get("case_path", ""))
        fingerprint = str(item.get("source_fingerprint", "")).strip().lower()
        if not path or len(fingerprint) != 64:
            raise ValueError("dataset source identity has an invalid case entry")
        if path in output:
            raise ValueError(f"duplicate case path in dataset identity: {path}")
        output[path] = fingerprint
    if not output:
        raise ValueError("dataset source identity has no selected cases")
    return output


def _raw_staging_identity(
    *,
    input_json: Path,
    gate1_script: Path | None,
    p2_source_fingerprint: str,
    p2_source_kind: str,
    dataset_source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Content identity for raw staging reuse, including raw H/H0 sources."""

    _validate_dataset_source_identity(dataset_source_identity)
    payload: dict[str, Any] = {
        "schema": RAW_STAGING_IDENTITY_SCHEMA,
        "materializer_script_sha256": _sha256(Path(__file__).resolve()),
        "gate1_script_sha256": (
            _sha256(gate1_script.resolve()) if gate1_script is not None else None
        ),
        "input_json_sha256": _sha256(input_json.resolve()),
        "p2_source_fingerprint": str(p2_source_fingerprint).strip().lower(),
        "p2_source_kind": str(p2_source_kind),
        "dataset_source_identity": dict(dataset_source_identity),
    }
    if len(payload["p2_source_fingerprint"]) != 64:
        raise ValueError("p2_source_fingerprint must be a SHA256 hex string.")
    payload["identity_sha256"] = _identity_sha256(payload)
    return payload


def _validate_raw_staging_identity_payload(
    payload: Any,
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Cannot resume raw staging: {label} is not a mapping.")
    if payload.get("schema") != RAW_STAGING_IDENTITY_SCHEMA:
        raise ValueError(
            f"Cannot resume raw staging: {label} lacks raw staging identity "
            f"schema {RAW_STAGING_IDENTITY_SCHEMA!r}; rebuild raw staging."
        )
    declared = str(payload.get("identity_sha256", "")).strip().lower()
    if len(declared) != 64:
        raise ValueError(
            f"Cannot resume raw staging: {label} lacks raw staging identity "
            "SHA256; rebuild raw staging."
        )
    without_digest = dict(payload)
    without_digest.pop("identity_sha256", None)
    actual = _identity_sha256(without_digest)
    if declared != actual:
        raise ValueError(
            f"Cannot resume raw staging: {label} identity SHA256 is inconsistent; "
            "rebuild raw staging."
        )
    if payload != expected:
        mismatched = [
            key
            for key in sorted(set(payload) | set(expected))
            if payload.get(key) != expected.get(key)
        ]
        raise ValueError(
            "Cannot resume raw staging: raw staging identity mismatch for "
            f"{label}: {mismatched}. Rebuild raw staging or use a new work root."
        )


def _write_raw_staging_identity(work_root: Path, identity: dict[str, Any]) -> None:
    raw_root = work_root / "raw_staging"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_json(raw_root / RAW_STAGING_IDENTITY_NAME, identity)


def _validate_raw_staging_identity(
    work_root: Path,
    expected: dict[str, Any],
) -> None:
    identity_path = work_root / "raw_staging" / RAW_STAGING_IDENTITY_NAME
    if not identity_path.is_file():
        raise ValueError(
            "Cannot resume raw staging: raw staging identity is missing; "
            "legacy or manually edited raw staging must be rebuilt."
        )
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    _validate_raw_staging_identity_payload(
        payload, expected, label=identity_path.name
    )

    manifest_paths = [
        path
        for path in (work_root / "manifest.partial.json", work_root / "manifest.json")
        if path.is_file()
    ]
    if not manifest_paths:
        raise ValueError(
            "Cannot resume raw staging: no manifest.partial.json or manifest.json "
            "with raw staging identity is present; rebuild raw staging."
        )
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "raw_staging_identity" not in manifest:
            raise ValueError(
                f"Cannot resume raw staging: {manifest_path.name} lacks raw "
                "staging identity; rebuild raw staging."
            )
        _validate_raw_staging_identity_payload(
            manifest["raw_staging_identity"],
            expected,
            label=manifest_path.name,
        )


def _validate_raw_record_source(
    record: Mapping[str, Any],
    case: Path,
    *,
    case_source_fingerprints: Mapping[str, str],
    label: str,
) -> str:
    expected_path = str(case.resolve())
    expected_fingerprint = case_source_fingerprints.get(expected_path)
    if expected_fingerprint is None:
        raise ValueError(
            f"Cannot resume {label}: current dataset identity has no source "
            f"fingerprint for {expected_path}."
        )
    if record.get("source") != expected_path:
        raise ValueError(
            f"Cannot resume {label}: source path {record.get('source')!r} != "
            f"{expected_path!r}."
        )
    recorded_fingerprint = str(
        record.get(RAW_CASE_SOURCE_FINGERPRINT_KEY, "")
    ).strip().lower()
    if recorded_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Cannot resume {label}: raw H/H0/source fingerprint mismatch "
            f"for {case.name}."
        )
    return expected_fingerprint


def _heartbeat(work_root: Path, **payload: Any) -> None:
    _write_json(
        work_root / "heartbeat.json",
        {"schema": SCHEMA, "updated_unix": time.time(), **payload},
    )


def parse_basis_lines(raw: bytes | str, expected_atoms: int) -> list[list[int]]:
    """Expand per-atom strings such as ``4s2p2d1f`` into radial-shell l lists."""
    import re

    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    letters = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
    result: list[list[int]] = []
    for line in (item.strip() for item in text.splitlines()):
        if not line:
            continue
        shells: list[int] = []
        tokens = re.findall(r"(\d+)([spdfgh])", line.lower())
        if not tokens or "".join(f"{n}{symbol}" for n, symbol in tokens) != line.lower():
            raise ValueError(f"Unsupported ABACUS basis line {line!r}.")
        for count, symbol in tokens:
            shells.extend([letters[symbol]] * int(count))
        result.append(shells)
    if len(result) != int(expected_atoms):
        raise ValueError(
            f"Basis lines ({len(result)}) do not match atoms ({expected_atoms})."
        )
    return result


def _sample_source_files(parsed: Any, pp_orb_root: Path) -> dict[str, dict[str, Path]]:
    """Resolve the exact ORB/UPF files selected by one STRU."""

    files: dict[str, dict[str, Path]] = {}
    symbols = {str(atom.species) for atom in parsed.structure.atoms}
    for symbol in sorted(symbols):
        if symbol not in parsed.specs:
            raise KeyError(f"STRU has no source-file specification for {symbol!r}")
        spec = parsed.specs[symbol]
        orbital = (pp_orb_root / str(spec.orbital)).resolve()
        upf = (pp_orb_root / str(spec.pseudo)).resolve()
        files[symbol] = {"orbital": orbital, "upf": upf}
    return files


def dense_p2_to_deeptb_blocks(
    *,
    r_keys: np.ndarray,
    p2: np.ndarray,
    basis_lines: bytes | str,
    atom_count: int,
    imag_tolerance: float = 1.0e-10,
) -> dict[str, np.ndarray]:
    """Rotate dense non-SOC P2 H(R) from ABACUS AO order into DeePTB blocks."""
    r_keys = np.asarray(r_keys)
    p2 = np.asarray(p2)
    if r_keys.ndim != 2 or r_keys.shape[1] != 3:
        raise ValueError(f"r_keys must have shape [nR,3], got {r_keys.shape}.")
    if not np.isfinite(r_keys).all():
        raise ValueError("r_keys contains NaN or infinity.")
    integer_r_keys = r_keys.astype(np.int64)
    if not np.array_equal(r_keys, integer_r_keys):
        raise ValueError("r_keys must contain exact integer lattice translations.")
    if len({tuple(row) for row in integer_r_keys.tolist()}) != len(integer_r_keys):
        raise ValueError("r_keys contains duplicate lattice translations.")
    if p2.ndim != 3 or p2.shape[0] != r_keys.shape[0] or p2.shape[1] != p2.shape[2]:
        raise ValueError(f"P2 must have shape [nR,n,n], got {p2.shape}.")
    if not np.isfinite(p2).all():
        raise ValueError("P2 contains NaN or infinity.")
    max_imag = float(np.max(np.abs(p2.imag), initial=0.0)) if np.iscomplexobj(p2) else 0.0
    if max_imag > imag_tolerance:
        raise ValueError(
            f"Non-SOC P2 has max imaginary magnitude {max_imag:.3e} > {imag_tolerance:.3e}."
        )

    atom_shells = parse_basis_lines(basis_lines, atom_count)
    atom_norb = [sum(2 * l + 1 for l in shells) for shells in atom_shells]
    offsets = np.concatenate(([0], np.cumsum(atom_norb))).astype(int)
    if int(offsets[-1]) != int(p2.shape[1]):
        raise ValueError(
            f"P2 AO dimension {p2.shape[1]} does not match basis dimension {offsets[-1]}."
        )

    converter = OrbAbacus2DeepTB()
    output: dict[str, np.ndarray] = {}
    real_p2 = np.asarray(p2.real, dtype=np.float64)
    for r_index, r_key in enumerate(integer_r_keys):
        rx, ry, rz = (int(value) for value in r_key)
        dense = real_p2[r_index]
        for i in range(atom_count):
            rows = slice(int(offsets[i]), int(offsets[i + 1]))
            for j in range(atom_count):
                cols = slice(int(offsets[j]), int(offsets[j + 1]))
                block = converter.transform(
                    dense[rows, cols], atom_shells[i], atom_shells[j]
                )
                output[f"{i}_{j}_{rx}_{ry}_{rz}"] = np.asarray(
                    block * RY_TO_EV, dtype=np.float32
                )
    return output


def table_p2_to_deeptb_blocks(
    *,
    assembler: P2TableAssembler,
    symbols: list[str],
    positions_bohr: np.ndarray,
    cell_bohr: np.ndarray,
    required_keys: Iterable[tuple[int, int, int, int, int]],
    basis_lines: bytes | str,
    atom_count: int,
    source_files: dict[str, dict[str, Path]] | None = None,
    require_source_files: bool = False,
) -> dict[str, np.ndarray]:
    """Assemble only sparse H/H0 blocks and rotate them to DeePTB AO order.

    A dense ``nR x nao x nao`` reconstruction scales as ``nR * n_atoms^2``
    even though training consumes onsite rows and graph edges only.  The exact
    canonical model graph is therefore the sparse contract used here; the
    later ``missing_block_policy='error'`` conversion independently verifies
    that every requested edge was assembled.
    """

    symbols = [str(symbol) for symbol in symbols]
    if len(symbols) != int(atom_count):
        raise ValueError(
            f"Structure symbols ({len(symbols)}) do not match atoms ({atom_count})."
        )
    atom_shells = parse_basis_lines(basis_lines, atom_count)
    atom_norb = [sum(2 * l + 1 for l in shells) for shells in atom_shells]
    assembler.store.validate_sample_contract(
        symbols=symbols,
        atom_shells=atom_shells,
        source_files=source_files,
        require_source_files=require_source_files,
    )
    for index, (symbol, norb) in enumerate(zip(symbols, atom_norb)):
        table_norb = int(assembler.store.species[symbol]["orbital_norb"])
        if table_norb != int(norb):
            raise ValueError(
                f"Atom {index} ({symbol}) table AO dimension {table_norb} "
                f"does not match ABACUS basis dimension {norb}."
            )

    required = sorted(
        {
            tuple(int(value) for value in key)
            for key in required_keys
        }
    )
    converter = OrbAbacus2DeepTB()
    assembled = assembler.assemble_sparse_blocks(
        symbols=symbols,
        positions_bohr=positions_bohr,
        cell_bohr=cell_bohr,
        block_keys=required,
    )
    output: dict[str, np.ndarray] = {}
    for i, j, rx, ry, rz in required:
        if i < 0 or i >= atom_count or j < 0 or j >= atom_count:
            raise ValueError(
                f"P2 block atom indices {(i, j)} are outside [0,{atom_count - 1}]."
            )
        block = assembled[(i, j, rx, ry, rz)]
        expected = (atom_norb[i], atom_norb[j])
        if tuple(block.shape) != expected:
            raise ValueError(
                f"Table P2 block {(i, j, rx, ry, rz)} shape {block.shape} != {expected}."
            )
        transformed = converter.transform(block, atom_shells[i], atom_shells[j])
        output[f"{i}_{j}_{rx}_{ry}_{rz}"] = np.asarray(
            transformed * RY_TO_EV, dtype=np.float32
        )
    return output


def _ordered_cases(dataset_root: Path, selected: Iterable[str]) -> list[Path]:
    ordered_file = dataset_root / "ordered_paths.txt"
    if not ordered_file.is_file():
        raise FileNotFoundError(ordered_file)
    wanted = set(selected)
    cases = [Path(line.strip()) for line in ordered_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if wanted:
        cases = [case for case in cases if case.name in wanted]
        missing = sorted(wanted - {case.name for case in cases})
        if missing:
            raise KeyError(f"Requested cases are absent from ordered_paths.txt: {missing}")
    if not cases:
        raise ValueError("No cases selected.")
    for case in cases:
        if not case.is_dir():
            raise NotADirectoryError(case)
    return cases


def _split_cases(
    cases: list[Path], *, valid_count: int, seed: str
) -> dict[str, list[Path]]:
    if valid_count < 0 or valid_count >= len(cases):
        if not (valid_count == 0 and len(cases) == 1):
            raise ValueError(
                f"valid_count must be in [0,{len(cases) - 1}], got {valid_count}."
            )
    ranks = {
        case.name: hashlib.sha256(f"{seed}\0{case.name}".encode("utf-8")).hexdigest()
        for case in cases
    }
    validation_names = {
        case.name for case in sorted(cases, key=lambda item: ranks[item.name])[:valid_count]
    }
    return {
        "train": [case for case in cases if case.name not in validation_names],
        "validation": [case for case in cases if case.name in validation_names],
    }


def _open_write(path: Path) -> lmdb.Environment:
    path.mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(path), map_size=MAP_SIZE, subdir=True, lock=True, readahead=False, max_dbs=1
    )


def _open_read(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path), readonly=True, lock=False, readahead=False, max_readers=512, subdir=True
    )


def _raw_split(
    *,
    cases: list[Path],
    raw_split: Path,
    p2_root: Path | None,
    p2_assembler: P2TableAssembler | None,
    gate1: Any | None,
    p2_pp_orb_root: Path | None,
    p2_source_fingerprint: str,
    case_source_fingerprints: Mapping[str, str],
    p2_hermitian_mismatch_tolerance: float,
    input_json: Path,
    work_root: Path,
    split: str,
    start_index: int = 0,
    prefix_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cutoff_options, _, _ = _dataset_args_from_input(
        input_json=str(input_json),
        root_override=str(raw_split),
        get_h0=True,
    )
    env = _open_write(raw_split / "data.0000.lmdb")
    rows = list(prefix_rows or [])
    if start_index != len(rows) or not 0 <= start_index <= len(cases):
        raise ValueError(
            f"Invalid raw resume prefix: start_index={start_index}, rows={len(rows)}, "
            f"cases={len(cases)}."
        )
    try:
        for index, case in enumerate(cases[start_index:], start=start_index):
            resolved_case = case.resolve()
            case_source_fingerprint = case_source_fingerprints.get(
                str(resolved_case)
            )
            if case_source_fingerprint is None:
                raise ValueError(
                    f"Current dataset identity has no source fingerprint for "
                    f"{resolved_case}."
                )
            _abacus_parse(
                str(resolved_case),
                str(raw_split),
                "OUT.ABACUS",
                output_mode="lmdb",
                idx=index,
                lmdb_env=env,
                get_Ham=True,
                get_H0=True,
                get_DM=False,
                get_overlap=False,
                get_eigenvalues=False,
            )
            key = index.to_bytes(length=4, byteorder="big")
            with env.begin(write=True) as txn:
                raw = txn.get(key)
                if raw is None:
                    raise KeyError(f"ABACUS parser did not write {split}[{index}].")
                record = pickle.loads(raw)
                graph = AtomicData.from_points(
                    pos=np.asarray(record[AtomicDataDict.POSITIONS_KEY]).reshape(-1, 3),
                    cell=np.asarray(record[AtomicDataDict.CELL_KEY]).reshape(3, 3),
                    atomic_numbers=np.asarray(record[AtomicDataDict.ATOMIC_NUMBERS_KEY]),
                    pbc=record[AtomicDataDict.PBC_KEY],
                    r_max=cutoff_options["r_max"],
                )
                required_keys = _required_block_keys_from_graph(
                    record[AtomicDataDict.ATOMIC_NUMBERS_KEY],
                    graph[AtomicDataDict.EDGE_INDEX_KEY],
                    graph[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
                )
                atom_count = len(record[AtomicDataDict.ATOMIC_NUMBERS_KEY])
                zero_completion: dict[str, Any] = {}
                for source_key in ("hamiltonian", "hamiltonian_0"):
                    completed, filled = complete_sparse_zero_blocks(
                        record.get(source_key),
                        required_keys=required_keys,
                        basis_lines=record["basis"],
                        atom_count=atom_count,
                        label=source_key,
                    )
                    record[source_key] = completed
                    zero_completion[source_key] = _zero_completion_metadata(filled)
                record["sparse_zero_completion"] = zero_completion
                if p2_assembler is not None:
                    if gate1 is None:
                        raise RuntimeError("Table P2 assembly requires the STRU parser.")
                    if p2_pp_orb_root is None:
                        raise RuntimeError(
                            "Table P2 assembly requires a validated PP_ORB root."
                        )
                    parsed = gate1.parse_stru(resolved_case / "STRU")
                    structure = parsed.structure
                    symbols = [atom.species for atom in structure.atoms]
                    if len(symbols) != atom_count:
                        raise ValueError(
                            f"{case.name}: STRU atoms {len(symbols)} != ABACUS record {atom_count}."
                        )
                    p2_blocks = table_p2_to_deeptb_blocks(
                        assembler=p2_assembler,
                        symbols=symbols,
                        positions_bohr=np.asarray(
                            structure.cart_positions, dtype=np.float64
                        ),
                        cell_bohr=np.asarray(structure.cell_bohr, dtype=np.float64),
                        required_keys=required_keys,
                        basis_lines=record["basis"],
                        atom_count=atom_count,
                        source_files=_sample_source_files(parsed, p2_pp_orb_root),
                        require_source_files=True,
                    )
                    p2_batch_stats = p2_assembler.batch_stats_snapshot()
                    p2_block_keys = set(required_keys)
                    p2_path = None
                    source_metadata = {
                        "kind": "radial_table",
                        "schema": SCHEMA,
                        "table_manifest_sha256": p2_source_fingerprint,
                        "species_source_identity_sha256": (
                            p2_assembler.store.species_source_identity_sha256
                        ),
                        "sample_source_identity_verified": True,
                        "energy_unit": "eV",
                        "ao_order": "DeePTB",
                        "batch_stats": p2_batch_stats,
                    }
                else:
                    if p2_root is None:
                        raise RuntimeError("Direct P2 source root is missing.")
                    p2_path = p2_root / f"{resolved_case.name}.npz"
                    if not p2_path.is_file():
                        raise FileNotFoundError(p2_path)
                    with np.load(p2_path, allow_pickle=False) as payload:
                        if "r_keys" not in payload or "P2" not in payload:
                            raise KeyError(f"{p2_path} must contain r_keys and P2.")
                        p2_blocks = dense_p2_to_deeptb_blocks(
                            r_keys=payload["r_keys"],
                            p2=payload["P2"],
                            basis_lines=record["basis"],
                            atom_count=len(record[AtomicDataDict.ATOMIC_NUMBERS_KEY]),
                        )
                        p2_block_keys = {
                            tuple(int(value) for value in key.split("_"))
                            for key in p2_blocks
                        }
                    source_metadata = {
                        "kind": "direct_npz",
                        "schema": SCHEMA,
                        "npz": str(p2_path),
                        "npz_sha256": _sha256(p2_path),
                        "source_set_sha256": p2_source_fingerprint,
                        "energy_unit": "eV",
                        "ao_order": "DeePTB",
                    }
                missing_required = sorted(set(required_keys) - p2_block_keys)
                if missing_required:
                    raise ValueError(
                        f"{case.name}: main graph has P2 block keys absent from the "
                        f"source: {missing_required[:5]}"
                    )
                p2_blocks, projection = project_non_soc_blocks_hermitian(
                    p2_blocks,
                    required_keys=required_keys,
                    label=f"{case.name} P2",
                    mismatch_tolerance=p2_hermitian_mismatch_tolerance,
                )
                record["hamiltonian_p2"] = p2_blocks
                record["p2_hermitian_projection"] = projection
                record["case_id"] = resolved_case.name
                record["source"] = str(resolved_case)
                record[RAW_CASE_SOURCE_FINGERPRINT_KEY] = case_source_fingerprint
                record["p2_cache_source"] = source_metadata
                record[SAMPLE_SCHEMA_KEY] = P2_SAMPLE_SCHEMA
                record[TARGET_SEMANTICS_KEY] = ABSOLUTE_FULL_H_SEMANTICS
                record[TARGET_SOURCE_KEY] = "raw_hamiltonian"
                record[P2_SOURCE_FINGERPRINT_KEY] = p2_source_fingerprint
                txn.put(key, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
            row = _raw_row(record, index=index)
            rows.append(row)
            _heartbeat(
                work_root,
                stage="raw",
                split=split,
                completed=index + 1,
                total=len(cases),
                case_id=case.name,
            )
            print(json.dumps({"stage": "raw", "split": split, **row}), flush=True)
    finally:
        env.close()
    (raw_split / "data.0000.paths.txt").write_text(
        "".join(f"{case.resolve()}\n" for case in cases), encoding="utf-8"
    )
    return rows


def _raw_row(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    source = record["p2_cache_source"]
    zero_completion = record.get("sparse_zero_completion", {})
    return {
        "index": int(index),
        "case_id": record["case_id"],
        "source": record["source"],
        RAW_CASE_SOURCE_FINGERPRINT_KEY: record[RAW_CASE_SOURCE_FINGERPRINT_KEY],
        "p2_source_kind": source["kind"],
        "p2_source": source.get("npz", "radial_table"),
        "p2_source_sha256": (
            source.get("npz_sha256") or source.get("table_manifest_sha256")
        ),
        "p2_blocks": len(record["hamiltonian_p2"]),
        "p2_max_pre_projection_mismatch": record.get(
            "p2_hermitian_projection", {}
        ).get("max_pre_projection_mismatch", 0.0),
        "p2_p99_pre_projection_mismatch": record.get(
            "p2_hermitian_projection", {}
        ).get("p99_pre_projection_mismatch", 0.0),
        "p2_hermitian_mismatch_tolerance": record.get(
            "p2_hermitian_projection", {}
        ).get("mismatch_tolerance"),
        "p2_hermitian_above_tolerance_count": record.get(
            "p2_hermitian_projection", {}
        ).get("above_tolerance_count", 0),
        "p2_max_projection_correction": record.get(
            "p2_hermitian_projection", {}
        ).get("max_projection_correction", 0.0),
        "full_h_sparse_zero_filled": zero_completion.get("hamiltonian", {}).get(
            "filled_count", 0
        ),
        "h0_sparse_zero_filled": zero_completion.get("hamiltonian_0", {}).get(
            "filled_count", 0
        ),
    }


def _reuse_raw_split(
    *,
    cases: list[Path],
    raw_split: Path,
    p2_root: Path | None,
    p2_assembler: P2TableAssembler | None,
    gate1: Any | None,
    p2_pp_orb_root: Path | None,
    p2_source_fingerprint: str,
    case_source_fingerprints: Mapping[str, str],
    p2_hermitian_mismatch_tolerance: float,
    input_json: Path,
    work_root: Path,
    split: str,
) -> list[dict[str, Any]]:
    """Validate a committed raw prefix and append any missing cases."""

    lmdb_path = raw_split / "data.0000.lmdb"
    paths_path = raw_split / "data.0000.paths.txt"
    if not lmdb_path.is_dir():
        raise FileNotFoundError(f"Cannot resume {split}: raw staging LMDB is missing.")
    if paths_path.is_file():
        staged_paths = [
            Path(line.strip())
            for line in paths_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [str(path.resolve()) for path in staged_paths] != [
            str(case.resolve()) for case in cases
        ]:
            raise ValueError(
                f"Cannot resume {split}: staged case order does not match requested split."
            )

    cutoff_options, _, _ = _dataset_args_from_input(
        input_json=str(input_json),
        root_override=str(raw_split),
        get_h0=True,
    )
    env = lmdb.open(
        str(lmdb_path),
        map_size=MAP_SIZE,
        subdir=True,
        lock=True,
        readahead=False,
        max_dbs=1,
    )
    rows: list[dict[str, Any]] = []
    try:
        with env.begin(write=True) as txn:
            entry_count = int(txn.stat()["entries"])
            if entry_count > len(cases):
                raise ValueError(
                    f"Cannot resume {split}: LMDB has {entry_count} entries, "
                    f"more than requested {len(cases)}."
                )
            committed = 0
            parser_placeholder = False
            for index, case in enumerate(cases):
                key = index.to_bytes(length=4, byteorder="big")
                value = txn.get(key)
                if value is None:
                    if index < entry_count:
                        raise KeyError(
                            f"Cannot resume {split}: raw LMDB is missing row {index} "
                            f"inside {entry_count} entries."
                        )
                    break
                record = pickle.loads(value)
                case_id = record.get("case_id")
                if case_id is None and record.get(SAMPLE_SCHEMA_KEY) is None:
                    # _abacus_parse commits H/H0 first.  A TERM between that
                    # commit and P2 enrichment leaves one safe-to-rebuild tail
                    # row that must not count as a completed cache record.
                    parser_placeholder = True
                    break
                if case_id != case.name:
                    raise ValueError(
                        f"Cannot resume {split}[{index}]: case_id "
                        f"{case_id!r} != {case.name!r}."
                    )
                if record.get(SAMPLE_SCHEMA_KEY) != P2_SAMPLE_SCHEMA:
                    raise ValueError(f"Cannot resume {case.name}: schema mismatch.")
                if record.get(P2_SOURCE_FINGERPRINT_KEY) != p2_source_fingerprint:
                    raise ValueError(
                        f"Cannot resume {case.name}: P2 source fingerprint mismatch."
                    )
                _validate_raw_record_source(
                    record,
                    case,
                    case_source_fingerprints=case_source_fingerprints,
                    label=f"{split}[{index}]",
                )
                committed += 1

            if entry_count != committed:
                if not parser_placeholder or entry_count != committed + 1:
                    raise ValueError(
                        f"Cannot resume {split}: {entry_count} LMDB entries do not "
                        f"form a complete prefix of {committed} plus one parser tail."
                    )
                tail_key = committed.to_bytes(length=4, byteorder="big")
                if not txn.delete(tail_key):
                    raise KeyError(
                        f"Cannot resume {split}: failed to remove parser tail {committed}."
                    )
                _heartbeat(
                    work_root,
                    stage="truncate_raw_parser_tail",
                    split=split,
                    completed=committed,
                    total=len(cases),
                    discarded_index=committed,
                )
        for index, case in enumerate(cases[:committed]):
            key = index.to_bytes(length=4, byteorder="big")
            with env.begin(write=True) as txn:
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"Cannot resume {split}: missing raw row {index}.")
                record = pickle.loads(value)
                if record.get("case_id") != case.name:
                    raise ValueError(
                        f"Cannot resume {split}[{index}]: case_id "
                        f"{record.get('case_id')!r} != {case.name!r}."
                    )
                if record.get(SAMPLE_SCHEMA_KEY) != P2_SAMPLE_SCHEMA:
                    raise ValueError(f"Cannot resume {case.name}: schema mismatch.")
                if record.get(P2_SOURCE_FINGERPRINT_KEY) != p2_source_fingerprint:
                    raise ValueError(
                        f"Cannot resume {case.name}: P2 source fingerprint mismatch."
                    )
                _validate_raw_record_source(
                    record,
                    case,
                    case_source_fingerprints=case_source_fingerprints,
                    label=f"{split}[{index}]",
                )
                if p2_assembler is not None:
                    if gate1 is None or p2_pp_orb_root is None:
                        raise RuntimeError(
                            "Table P2 resume requires the STRU parser and PP_ORB root."
                        )
                    parsed = gate1.parse_stru(case / "STRU")
                    symbols = [str(atom.species) for atom in parsed.structure.atoms]
                    atom_count_for_contract = len(
                        record[AtomicDataDict.ATOMIC_NUMBERS_KEY]
                    )
                    if len(symbols) != atom_count_for_contract:
                        raise ValueError(
                            f"Cannot resume {case.name}: STRU atoms {len(symbols)} != "
                            f"record {atom_count_for_contract}."
                        )
                    p2_assembler.store.validate_sample_contract(
                        symbols=symbols,
                        atom_shells=parse_basis_lines(
                            record["basis"], atom_count_for_contract
                        ),
                        source_files=_sample_source_files(parsed, p2_pp_orb_root),
                        require_source_files=True,
                    )
                    source_metadata = dict(record.get("p2_cache_source", {}))
                    source_metadata.update(
                        {
                            "species_source_identity_sha256": (
                                p2_assembler.store.species_source_identity_sha256
                            ),
                            "sample_source_identity_verified": True,
                        }
                    )
                    record["p2_cache_source"] = source_metadata
                graph = AtomicData.from_points(
                    pos=np.asarray(record[AtomicDataDict.POSITIONS_KEY]).reshape(-1, 3),
                    cell=np.asarray(record[AtomicDataDict.CELL_KEY]).reshape(3, 3),
                    atomic_numbers=np.asarray(
                        record[AtomicDataDict.ATOMIC_NUMBERS_KEY]
                    ),
                    pbc=record[AtomicDataDict.PBC_KEY],
                    r_max=cutoff_options["r_max"],
                )
                required_keys = _required_block_keys_from_graph(
                    record[AtomicDataDict.ATOMIC_NUMBERS_KEY],
                    graph[AtomicDataDict.EDGE_INDEX_KEY],
                    graph[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
                )
                atom_count = len(record[AtomicDataDict.ATOMIC_NUMBERS_KEY])
                zero_completion: dict[str, Any] = {}
                for source_key in ("hamiltonian", "hamiltonian_0"):
                    completed, filled = complete_sparse_zero_blocks(
                        record.get(source_key),
                        required_keys=required_keys,
                        basis_lines=record["basis"],
                        atom_count=atom_count,
                        label=source_key,
                    )
                    record[source_key] = completed
                    zero_completion[source_key] = _zero_completion_metadata(
                        filled,
                        record.get("sparse_zero_completion", {}).get(source_key),
                    )
                p2_keys = {
                    tuple(int(value) for value in block_key.split("_"))
                    for block_key in record["hamiltonian_p2"]
                }
                missing_p2 = sorted(set(required_keys) - p2_keys)
                if missing_p2:
                    raise ValueError(
                        f"Cannot resume {case.name}: P2 is missing exact graph "
                        f"blocks {missing_p2[:5]}."
                    )
                previous_projection = record.get("p2_hermitian_projection")
                if not isinstance(previous_projection, dict) or (
                    "max_pre_projection_mismatch" not in previous_projection
                ):
                    raise ValueError(
                        f"Cannot resume {case.name}: prior P2 Hermitian projection "
                        "statistics are missing. Rebuild this raw row."
                    )
                previous_max_mismatch = float(
                    previous_projection["max_pre_projection_mismatch"]
                )
                if (
                    not np.isfinite(previous_max_mismatch)
                    or previous_max_mismatch > p2_hermitian_mismatch_tolerance
                ):
                    raise ValueError(
                        f"Cannot resume {case.name}: recorded pre-projection P2 "
                        f"Hermitian mismatch {previous_max_mismatch:.6e} eV exceeds "
                        f"{p2_hermitian_mismatch_tolerance:.6e} eV."
                    )
                projected, resume_projection = project_non_soc_blocks_hermitian(
                    record["hamiltonian_p2"],
                    required_keys=required_keys,
                    label=f"{case.name} P2",
                    mismatch_tolerance=p2_hermitian_mismatch_tolerance,
                )
                projection = dict(previous_projection)
                projection.setdefault(
                    "mean_pre_projection_mismatch", previous_max_mismatch
                )
                projection.setdefault(
                    "p95_pre_projection_mismatch", previous_max_mismatch
                )
                projection.setdefault(
                    "p99_pre_projection_mismatch", previous_max_mismatch
                )
                projection["mismatch_tolerance"] = float(
                    p2_hermitian_mismatch_tolerance
                )
                projection["above_tolerance_count"] = 0
                projection["unit"] = "eV"
                projection["resume_exact_revalidation"] = resume_projection
                record["hamiltonian_p2"] = projected
                record["p2_hermitian_projection"] = projection
                record["sparse_zero_completion"] = zero_completion
                txn.put(key, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
            row = _raw_row(record, index=index)
            rows.append(row)
            _heartbeat(
                work_root,
                stage="reuse_raw",
                split=split,
                completed=index + 1,
                total=len(cases),
                case_id=case.name,
            )
            print(json.dumps({"stage": "reuse_raw", "split": split, **row}), flush=True)
    finally:
        env.close()
    if committed < len(cases):
        return _raw_split(
            cases=cases,
            raw_split=raw_split,
            p2_root=p2_root,
            p2_assembler=p2_assembler,
            gate1=gate1,
            p2_pp_orb_root=p2_pp_orb_root,
            p2_source_fingerprint=p2_source_fingerprint,
            case_source_fingerprints=case_source_fingerprints,
            p2_hermitian_mismatch_tolerance=p2_hermitian_mismatch_tolerance,
            input_json=input_json,
            work_root=work_root,
            split=split,
            start_index=committed,
            prefix_rows=rows,
        )
    if not paths_path.is_file():
        paths_path.write_text(
            "".join(f"{case.resolve()}\n" for case in cases), encoding="utf-8"
        )
    return rows


def _numpy(value: Any) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu()
    if torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    return tensor.numpy()


def _require(data: Any, fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise KeyError(f"Materialized AtomicData is missing fields {missing}.")


def _base_record(
    raw: dict[str, Any],
    data: Any,
    index: int,
    *,
    basis_fingerprint: str,
) -> dict[str, Any]:
    record = {field: np.asarray(raw[field]) for field in _BASE_FIELDS}
    for field in _GRAPH_FIELDS:
        record[field] = _numpy(data[field])
    record.update(
        {
            "idx": index,
            "case_id": raw["case_id"],
            "source": raw["source"],
            "p2_cache_source": raw["p2_cache_source"],
            "p2_hermitian_projection": raw.get("p2_hermitian_projection", {}),
            "sparse_zero_completion": raw.get("sparse_zero_completion", {}),
            SAMPLE_SCHEMA_KEY: P2_SAMPLE_SCHEMA,
            BASIS_FINGERPRINT_KEY: basis_fingerprint,
            P2_SOURCE_FINGERPRINT_KEY: raw[P2_SOURCE_FINGERPRINT_KEY],
        }
    )
    record[EDGE_GRAPH_FINGERPRINT_KEY] = edge_graph_fingerprint(
        record[AtomicDataDict.ATOMIC_NUMBERS_KEY],
        record[AtomicDataDict.EDGE_INDEX_KEY],
        record[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
        basis_fingerprint=basis_fingerprint,
    )
    return record


def _materialize_split(
    *,
    raw_split: Path,
    full_split: Path,
    delta_split: Path,
    input_json: Path,
    work_root: Path,
    split: str,
) -> dict[str, Any]:
    dataset, _ = _build_context_dataset(str(input_json), str(raw_split))
    raw_env = _open_read(raw_split / "data.0000.lmdb")
    full_env = _open_write(full_split / "data.0000.lmdb")
    delta_env = _open_write(delta_split / "data.0000.lmdb")
    required = (
        AtomicDataDict.NODE_FEATURES_KEY,
        AtomicDataDict.EDGE_FEATURES_KEY,
        AtomicDataDict.NODE_H0_KEY,
        AtomicDataDict.EDGE_H0_KEY,
        AtomicDataDict.NODE_P2_KEY,
        AtomicDataDict.EDGE_P2_KEY,
        *_FULL_H_TARGET_FIELDS,
        *_H0_BLOCK_FIELDS,
        *_P2_BLOCK_FIELDS,
        *_GRAPH_FIELDS,
    )
    feature_widths: set[int] = set()
    max_identity_error = 0.0
    basis_fingerprint = mapper_basis_fingerprint(dataset.type_mapper)
    try:
        for index in range(len(dataset)):
            key = index.to_bytes(length=4, byteorder="big")
            with raw_env.begin() as txn:
                raw_value = txn.get(key)
                if raw_value is None:
                    raise KeyError(f"Missing raw record {split}[{index}].")
                raw = pickle.loads(raw_value)
            data = dataset.get(index)
            _require(data, required)

            full = _base_record(
                raw, data, index, basis_fingerprint=basis_fingerprint
            )
            for field in (
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.EDGE_FEATURES_KEY,
                AtomicDataDict.NODE_H0_KEY,
                AtomicDataDict.EDGE_H0_KEY,
                AtomicDataDict.NODE_P2_KEY,
                AtomicDataDict.EDGE_P2_KEY,
                *_FULL_H_TARGET_FIELDS,
                *_H0_BLOCK_FIELDS,
                *_P2_BLOCK_FIELDS,
            ):
                full[field] = _numpy(data[field])

            feature_widths.update(
                {
                    int(full[AtomicDataDict.NODE_FEATURES_KEY].shape[-1]),
                    int(full[AtomicDataDict.NODE_H0_KEY].shape[-1]),
                    int(full[AtomicDataDict.NODE_P2_KEY].shape[-1]),
                }
            )
            for field in (
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.EDGE_FEATURES_KEY,
                AtomicDataDict.NODE_H0_KEY,
                AtomicDataDict.EDGE_H0_KEY,
                AtomicDataDict.NODE_P2_KEY,
                AtomicDataDict.EDGE_P2_KEY,
                *_FULL_H_TARGET_FIELDS[:2],
                *_H0_BLOCK_FIELDS[:2],
                *_P2_BLOCK_FIELDS[:2],
            ):
                if not np.isfinite(full[field]).all():
                    raise ValueError(f"{raw['case_id']}: non-finite values in {field}.")
            full[TARGET_SEMANTICS_KEY] = ABSOLUTE_FULL_H_SEMANTICS
            full[TARGET_SOURCE_KEY] = "dedicated_full_h_blocks"
            full[P2_RME_FINGERPRINT_KEY] = fingerprint_fields(
                full, (AtomicDataDict.NODE_P2_KEY, AtomicDataDict.EDGE_P2_KEY)
            )
            full[P2_BLOCK_FINGERPRINT_KEY] = fingerprint_fields(
                full, _P2_BLOCK_FIELDS
            )
            full[P2_BUNDLE_FINGERPRINT_KEY] = fingerprint_text_fields(
                full,
                (
                    BASIS_FINGERPRINT_KEY,
                    EDGE_GRAPH_FINGERPRINT_KEY,
                    P2_SOURCE_FINGERPRINT_KEY,
                    P2_RME_FINGERPRINT_KEY,
                    P2_BLOCK_FINGERPRINT_KEY,
                ),
            )
            full[FULL_H_TARGET_FINGERPRINT_KEY] = fingerprint_fields(
                full, _FULL_H_TARGET_FIELDS
            )

            delta = _base_record(
                raw, data, index, basis_fingerprint=basis_fingerprint
            )
            delta[TARGET_SEMANTICS_KEY] = H0_RESIDUAL_SEMANTICS
            delta[TARGET_SOURCE_KEY] = "dedicated_h0_residual_blocks"
            delta[AtomicDataDict.NODE_FEATURES_KEY] = (
                full[AtomicDataDict.NODE_FEATURES_KEY]
                - full[AtomicDataDict.NODE_H0_KEY]
            ).astype(np.float32, copy=False)
            delta[AtomicDataDict.EDGE_FEATURES_KEY] = (
                full[AtomicDataDict.EDGE_FEATURES_KEY]
                - full[AtomicDataDict.EDGE_H0_KEY]
            ).astype(np.float32, copy=False)
            for field in (
                AtomicDataDict.NODE_H0_KEY,
                AtomicDataDict.EDGE_H0_KEY,
                *_H0_BLOCK_FIELDS,
            ):
                delta[field] = full[field]

            full_target_node, full_target_edge, node_shape, edge_shape = (
                _FULL_H_TARGET_FIELDS
            )
            h0_node, h0_edge, h0_node_shape, h0_edge_shape = _H0_BLOCK_FIELDS
            _, _, p2_node_shape, p2_edge_shape = _P2_BLOCK_FIELDS
            if not np.array_equal(full[node_shape], full[h0_node_shape]):
                raise ValueError(f"{raw['case_id']}: Full-H/H0 node block shapes differ.")
            if not np.array_equal(full[edge_shape], full[h0_edge_shape]):
                raise ValueError(f"{raw['case_id']}: Full-H/H0 edge block shapes differ.")
            if not np.array_equal(full[node_shape], full[p2_node_shape]):
                raise ValueError(f"{raw['case_id']}: Full-H/P2 node block shapes differ.")
            if not np.array_equal(full[edge_shape], full[p2_edge_shape]):
                raise ValueError(f"{raw['case_id']}: Full-H/P2 edge block shapes differ.")
            delta_target_node, delta_target_edge, delta_node_shape, delta_edge_shape = (
                _TARGET_FIELDS
            )
            delta[delta_target_node] = (full[full_target_node] - full[h0_node]).astype(
                np.float32, copy=False
            )
            delta[delta_target_edge] = (full[full_target_edge] - full[h0_edge]).astype(
                np.float32, copy=False
            )
            delta[delta_node_shape] = full[node_shape]
            delta[delta_edge_shape] = full[edge_shape]

            node_error = np.max(
                np.abs(delta[delta_target_node] + full[h0_node] - full[full_target_node]),
                initial=0.0,
            )
            edge_error = np.max(
                np.abs(delta[delta_target_edge] + full[h0_edge] - full[full_target_edge]),
                initial=0.0,
            )
            max_identity_error = max(max_identity_error, float(node_error), float(edge_error))
            if max_identity_error > 5.0e-6:
                raise ValueError(
                    f"{raw['case_id']}: cached H0+dH identity error {max_identity_error:.3e}."
                )

            for compact in (full, delta):
                compact[ROW_ALIGNED_DATA_FINGERPRINT_KEY] = (
                    fingerprint_present_row_aligned_fields(compact)
                )
                compact[ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY] = (
                    fingerprint_text_fields(
                        compact,
                        (
                            BASIS_FINGERPRINT_KEY,
                            EDGE_GRAPH_FINGERPRINT_KEY,
                            ROW_ALIGNED_DATA_FINGERPRINT_KEY,
                        ),
                    )
                )

            with full_env.begin(write=True) as txn:
                txn.put(key, pickle.dumps(full, protocol=pickle.HIGHEST_PROTOCOL))
            with delta_env.begin(write=True) as txn:
                txn.put(key, pickle.dumps(delta, protocol=pickle.HIGHEST_PROTOCOL))
            _heartbeat(
                work_root,
                stage="materialize",
                split=split,
                completed=index + 1,
                total=len(dataset),
                case_id=raw["case_id"],
                max_h0_delta_identity_error=max_identity_error,
            )
            print(
                json.dumps(
                    {
                        "stage": "materialize",
                        "split": split,
                        "index": index,
                        "case_id": raw["case_id"],
                        "nodes": int(full[AtomicDataDict.NODE_FEATURES_KEY].shape[0]),
                        "edges": int(full[AtomicDataDict.EDGE_FEATURES_KEY].shape[0]),
                        "feature_width": int(full[AtomicDataDict.NODE_FEATURES_KEY].shape[-1]),
                    }
                ),
                flush=True,
            )
    finally:
        raw_env.close()
        full_env.close()
        delta_env.close()
        for env in getattr(dataset, "_lmdb_env_cache", {}).values():
            env.close()
        dataset._lmdb_env_cache = {}

    paths = raw_split / "data.0000.paths.txt"
    shutil.copy2(paths, full_split / paths.name)
    shutil.copy2(paths, delta_split / paths.name)
    return {
        "entries": len(dataset),
        "feature_widths": sorted(feature_widths),
        "max_h0_delta_identity_error": max_identity_error,
    }


def _guard_output(
    dataset_root: Path,
    work_root: Path,
    overwrite: bool,
    *,
    resume_raw_staging: bool = False,
) -> None:
    dataset_root = dataset_root.resolve()
    work_root = work_root.resolve()
    if (
        dataset_root == work_root
        or dataset_root in work_root.parents
        or work_root in dataset_root.parents
    ):
        raise ValueError(
            "dataset_root and work_root must be disjoint; neither may contain the other."
        )
    if resume_raw_staging:
        if overwrite:
            raise ValueError("--resume-raw-staging and --overwrite are mutually exclusive.")
        if not work_root.is_dir():
            raise FileNotFoundError(
                f"--resume-raw-staging requires an existing work root: {work_root}"
            )
        return
    if work_root.exists():
        if not overwrite:
            raise FileExistsError(work_root)
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--p2-root", type=Path)
    source_group.add_argument("--table-root", type=Path)
    parser.add_argument("--gate1-script", type=Path)
    parser.add_argument(
        "--pp-orb",
        type=Path,
        help=(
            "ORB/UPF root used by selected STRU files; with --table-root the "
            "builder manifest path is used only when it still exists"
        ),
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--valid-count", type=int, default=20)
    parser.add_argument("--split-seed", default="nonsoc-p2-raw200-v1")
    parser.add_argument("--require-count", type=int, default=0)
    parser.add_argument(
        "--p2-hermitian-mismatch-tolerance-ev",
        type=float,
        default=DEFAULT_P2_HERMITIAN_MISMATCH_TOLERANCE_EV,
        help=(
            "fail before pair averaging when any raw P2/reverse mismatch exceeds "
            "this value"
        ),
    )
    parser.add_argument(
        "--keep-raw-staging",
        action="store_true",
        help="retain the large intermediate raw H/H0/P2 block LMDB after success",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume-raw-staging",
        action="store_true",
        help=(
            "reuse and revalidate existing raw_staging splits after an interrupted "
            "materialization; partial compact outputs are rebuilt"
        ),
    )
    args = parser.parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    p2_root = args.p2_root.resolve() if args.p2_root is not None else None
    table_root = args.table_root.resolve() if args.table_root is not None else None
    work_root = args.work_root.resolve()
    input_json = args.input_json.resolve()
    _guard_output(
        dataset_root,
        work_root,
        args.overwrite,
        resume_raw_staging=args.resume_raw_staging,
    )
    cases = _ordered_cases(dataset_root, args.case)
    if args.require_count and len(cases) != args.require_count:
        raise ValueError(f"Selected {len(cases)} cases; required {args.require_count}.")
    split_cases = _split_cases(cases, valid_count=args.valid_count, seed=args.split_seed)
    p2_assembler = None
    gate1 = None
    p2_pp_orb_root = None
    if table_root is not None:
        manifest_path = table_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        table_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require_qualified_table_manifest(
            table_manifest,
            manifest_path=manifest_path,
        )
        if args.gate1_script is None:
            raise ValueError("--gate1-script is required with --table-root.")
        p2_source_fingerprint = _sha256(manifest_path)
        declared_pp_orb = table_manifest.get("source", {}).get("pp_orb")
        pp_orb_candidate = args.pp_orb
        if pp_orb_candidate is None and declared_pp_orb:
            pp_orb_candidate = Path(str(declared_pp_orb))
        if pp_orb_candidate is None or not pp_orb_candidate.resolve().is_dir():
            raise ValueError(
                "--table-root production materialization must resolve the STRU "
                "ORB/UPF bundle; pass --pp-orb explicitly"
            )
        p2_pp_orb_root = pp_orb_candidate.resolve()
        p2_store = P2TableStore(
            table_root,
            require_explicit_source_identity=True,
        )
        p2_assembler = P2TableAssembler(p2_store)
        gate1 = _load_module(args.gate1_script.resolve(), "deeptb_p2_cache_gate1")
        p2_source_manifest = {
            "kind": "radial_table",
            "root": str(table_root),
            "manifest_sha256": p2_source_fingerprint,
            "pp_orb": str(p2_pp_orb_root),
            "species_source_identity_sha256": (
                p2_store.species_source_identity_sha256
            ),
        }
    else:
        assert p2_root is not None
        for case in cases:
            if not (p2_root / f"{case.name}.npz").is_file():
                raise FileNotFoundError(p2_root / f"{case.name}.npz")
        p2_source_fingerprint = _source_set_fingerprint(cases, p2_root)
        p2_source_manifest = {
            "kind": "direct_npz_set",
            "root": str(p2_root),
            "source_set_sha256": p2_source_fingerprint,
        }

    dataset_source_identity = _dataset_source_identity(
        dataset_root,
        cases,
        require_stru=table_root is not None,
    )
    case_source_fingerprints = _case_source_fingerprint_map(
        dataset_source_identity
    )
    raw_identity = _raw_staging_identity(
        input_json=input_json,
        gate1_script=args.gate1_script.resolve()
        if args.gate1_script is not None
        else None,
        p2_source_fingerprint=p2_source_fingerprint,
        p2_source_kind=str(p2_source_manifest["kind"]),
        dataset_source_identity=dataset_source_identity,
    )
    torch.set_default_dtype(torch.float32)
    raw_root = work_root / "raw_staging"
    full_root = work_root / "full_h"
    delta_root = work_root / "h0_delta"
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "dataset_root": str(dataset_root),
        "p2_source": p2_source_manifest,
        "p2_source_fingerprint": p2_source_fingerprint,
        "input_json": str(input_json),
        "input_json_sha256": _sha256(input_json),
        "script_sha256": _sha256(Path(__file__)),
        "raw_staging_identity": raw_identity,
        "p2_hermitian_mismatch_tolerance_ev": float(
            args.p2_hermitian_mismatch_tolerance_ev
        ),
        "split_seed": args.split_seed,
        "splits": {},
    }
    if args.resume_raw_staging:
        _validate_raw_staging_identity(work_root, raw_identity)
    else:
        _write_raw_staging_identity(work_root, raw_identity)
        _write_json(work_root / "manifest.partial.json", manifest)
    for split, selected in split_cases.items():
        if not selected:
            continue
        raw_split = raw_root / split
        if args.resume_raw_staging and (raw_split / "data.0000.lmdb").is_dir():
            raw_rows = _reuse_raw_split(
                cases=selected,
                raw_split=raw_split,
                p2_root=p2_root,
                p2_assembler=p2_assembler,
                gate1=gate1,
                p2_pp_orb_root=p2_pp_orb_root,
                p2_source_fingerprint=p2_source_fingerprint,
                case_source_fingerprints=case_source_fingerprints,
                p2_hermitian_mismatch_tolerance=float(
                    args.p2_hermitian_mismatch_tolerance_ev
                ),
                input_json=input_json,
                work_root=work_root,
                split=split,
            )
        else:
            raw_rows = _raw_split(
                cases=selected,
                raw_split=raw_split,
                p2_root=p2_root,
                p2_assembler=p2_assembler,
                gate1=gate1,
                p2_pp_orb_root=p2_pp_orb_root,
                p2_source_fingerprint=p2_source_fingerprint,
                case_source_fingerprints=case_source_fingerprints,
                p2_hermitian_mismatch_tolerance=float(
                    args.p2_hermitian_mismatch_tolerance_ev
                ),
                input_json=input_json,
                work_root=work_root,
                split=split,
            )
        for compact_split in (full_root / split, delta_root / split):
            if compact_split.exists():
                if not args.resume_raw_staging:
                    raise FileExistsError(compact_split)
                resolved = compact_split.resolve()
                if work_root not in resolved.parents:
                    raise ValueError(f"Refusing to remove output outside work root: {resolved}")
                shutil.rmtree(resolved)
        stats = _materialize_split(
            raw_split=raw_split,
            full_split=full_root / split,
            delta_split=delta_root / split,
            input_json=input_json,
            work_root=work_root,
            split=split,
        )
        manifest["splits"][split] = {
            "cases": [case.name for case in selected],
            "records": raw_rows,
            **stats,
        }
        _write_json(work_root / "manifest.partial.json", manifest)

    manifest["completed_unix"] = time.time()
    manifest["full_h_root"] = str(full_root)
    manifest["h0_delta_root"] = str(delta_root)
    manifest["raw_staging_retained"] = bool(args.keep_raw_staging)
    if not args.keep_raw_staging:
        shutil.rmtree(raw_root)
    _write_json(work_root / "manifest.json", manifest)
    _heartbeat(work_root, stage="complete", manifest=str(work_root / "manifest.json"))
    print(json.dumps({"status": "complete", "manifest": str(work_root / "manifest.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed metadata and fingerprint helpers for non-SOC physical priors.

The compact training LMDB stores several row-aligned views of the same
structure: a canonical directed edge graph, P2/P23 RME features, packed prior
AO blocks, and (for Full-H training) packed absolute-H targets.  Row counts
alone cannot prove that those views still refer to the same graph.  This module
keeps the small, deterministic contract used both by cache materializers and
the LMDB loader while retaining the historical P2-v2 API.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


P2_SAMPLE_SCHEMA = "deeptb.p2_training_sample/v2"
DUAL_PRIOR_SAMPLE_SCHEMA = "deeptb.physical_prior_training_sample/v3"
ABSOLUTE_FULL_H_SEMANTICS = "absolute_full_h"
H0_RESIDUAL_SEMANTICS = "h0_residual"

SAMPLE_SCHEMA_KEY = "hamiltonian_schema"
TARGET_SEMANTICS_KEY = "hamiltonian_target_semantics"
TARGET_SOURCE_KEY = "hamiltonian_target_source"
BASIS_FINGERPRINT_KEY = "basis_fingerprint"
EDGE_GRAPH_FINGERPRINT_KEY = "edge_graph_fingerprint"
P2_SOURCE_FINGERPRINT_KEY = "p2_source_fingerprint"
P2_RME_FINGERPRINT_KEY = "p2_rme_fingerprint"
P2_BLOCK_FINGERPRINT_KEY = "p2_blocks_fingerprint"
P2_BUNDLE_FINGERPRINT_KEY = "p2_bundle_fingerprint"
P23_SOURCE_FINGERPRINT_KEY = "p23_source_fingerprint"
P23_RME_FINGERPRINT_KEY = "p23_rme_fingerprint"
P23_BLOCK_FINGERPRINT_KEY = "p23_blocks_fingerprint"
P23_BUNDLE_FINGERPRINT_KEY = "p23_bundle_fingerprint"
P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY = "p23_parent_p2_bundle_fingerprint"
FULL_H_TARGET_FINGERPRINT_KEY = "full_h_target_fingerprint"
ROW_ALIGNED_DATA_FINGERPRINT_KEY = "row_aligned_data_fingerprint"
ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY = "row_aligned_bundle_fingerprint"

# Ordered once for schema v2.  The row-data digest covers every tensor that is
# interpreted by atom/edge row rather than merely checking its row count.  The
# companion bundle digest binds this content digest to the basis and canonical
# graph fingerprints.
ROW_ALIGNED_FIELD_CANDIDATES = (
    "node_features",
    "edge_features",
    "node_overlap",
    "edge_overlap",
    "node_h0",
    "edge_h0",
    "node_physical_h0",
    "edge_physical_h0",
    "node_p2",
    "edge_p2",
    "node_p23",
    "edge_p23",
    "node_delta_hamil_blocks",
    "edge_delta_hamil_blocks",
    "node_delta_hamil_block_shape",
    "edge_delta_hamil_block_shape",
    "node_full_hamil_target_blocks",
    "edge_full_hamil_target_blocks",
    "node_full_hamil_target_block_shape",
    "edge_full_hamil_target_block_shape",
    "node_h0_blocks",
    "edge_h0_blocks",
    "node_h0_block_shape",
    "edge_h0_block_shape",
    "node_p2_blocks",
    "edge_p2_blocks",
    "node_p2_block_shape",
    "edge_p2_block_shape",
    "node_p23_blocks",
    "edge_p23_blocks",
    "node_p23_block_shape",
    "edge_p23_block_shape",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PriorFieldSpec:
    """All record fields that must move together for one physical prior."""

    kind: str
    raw_key: str
    node_rme_key: str
    edge_rme_key: str
    node_blocks_key: str
    edge_blocks_key: str
    node_shape_key: str
    edge_shape_key: str
    source_fingerprint_key: str
    rme_fingerprint_key: str
    block_fingerprint_key: str
    bundle_fingerprint_key: str
    allowed_sample_schemas: tuple[str, ...]
    bundle_dependency_fields: tuple[str, ...] = ()

    @property
    def rme_fields(self) -> tuple[str, str]:
        return self.node_rme_key, self.edge_rme_key

    @property
    def block_fields(self) -> tuple[str, str, str, str]:
        return (
            self.node_blocks_key,
            self.edge_blocks_key,
            self.node_shape_key,
            self.edge_shape_key,
        )


PRIOR_FIELD_SPECS = {
    "p2": PriorFieldSpec(
        kind="p2",
        raw_key="hamiltonian_p2",
        node_rme_key="node_p2",
        edge_rme_key="edge_p2",
        node_blocks_key="node_p2_blocks",
        edge_blocks_key="edge_p2_blocks",
        node_shape_key="node_p2_block_shape",
        edge_shape_key="edge_p2_block_shape",
        source_fingerprint_key=P2_SOURCE_FINGERPRINT_KEY,
        rme_fingerprint_key=P2_RME_FINGERPRINT_KEY,
        block_fingerprint_key=P2_BLOCK_FINGERPRINT_KEY,
        bundle_fingerprint_key=P2_BUNDLE_FINGERPRINT_KEY,
        allowed_sample_schemas=(P2_SAMPLE_SCHEMA, DUAL_PRIOR_SAMPLE_SCHEMA),
    ),
    "p23": PriorFieldSpec(
        kind="p23",
        raw_key="hamiltonian_p23",
        node_rme_key="node_p23",
        edge_rme_key="edge_p23",
        node_blocks_key="node_p23_blocks",
        edge_blocks_key="edge_p23_blocks",
        node_shape_key="node_p23_block_shape",
        edge_shape_key="edge_p23_block_shape",
        source_fingerprint_key=P23_SOURCE_FINGERPRINT_KEY,
        rme_fingerprint_key=P23_RME_FINGERPRINT_KEY,
        block_fingerprint_key=P23_BLOCK_FINGERPRINT_KEY,
        bundle_fingerprint_key=P23_BUNDLE_FINGERPRINT_KEY,
        allowed_sample_schemas=(DUAL_PRIOR_SAMPLE_SCHEMA,),
        bundle_dependency_fields=(P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,),
    ),
}


def resolve_prior_field_spec(kind: Any) -> PriorFieldSpec:
    normalized = str(kind).strip().lower()
    try:
        return PRIOR_FIELD_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"prior kind must be one of {tuple(PRIOR_FIELD_SPECS)}, got {kind!r}."
        ) from exc


def resolve_prior_field_spec_from_raw_key(raw_key: Any) -> PriorFieldSpec:
    normalized = str(raw_key).strip()
    for spec in PRIOR_FIELD_SPECS.values():
        if normalized == spec.raw_key:
            return spec
    raise ValueError(
        "physical-prior raw key must select exactly 'hamiltonian_p2' or "
        f"'hamiltonian_p23'; got {raw_key!r}."
    )


def require_sha256(value: Any, *, field: str) -> str:
    """Return a normalized SHA256 hex string or fail closed."""

    if isinstance(value, bytes):
        value = value.decode("ascii", "strict")
    normalized = str(value).strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA256 hex string.")
    return normalized


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _update_array(digest: "hashlib._Hash", label: str, value: Any) -> None:
    array = _numpy(value)
    if array.dtype.hasobject:
        raise TypeError(f"Cannot fingerprint object array {label!r}.")
    contiguous = np.ascontiguousarray(array)
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes(order="C"))


def fingerprint_fields(mapping: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Fingerprint ordered tensor/array fields including dtype, shape, and rows."""

    digest = hashlib.sha256()
    for field in fields:
        if field not in mapping:
            raise KeyError(f"Cannot fingerprint missing field {field!r}.")
        _update_array(digest, field, mapping[field])
    return digest.hexdigest()


def fingerprint_text_fields(mapping: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Fingerprint an ordered set of already-normalized textual provenance fields."""

    digest = hashlib.sha256()
    for field in fields:
        value = require_sha256(mapping.get(field), field=field)
        digest.update(field.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def fingerprint_present_row_aligned_fields(mapping: Mapping[str, Any]) -> str:
    """Fingerprint every schema-v2 row-aligned field present in ``mapping``."""

    fields = tuple(
        field for field in ROW_ALIGNED_FIELD_CANDIDATES if field in mapping
    )
    if not fields:
        raise ValueError("No row-aligned fields are present to fingerprint.")
    return fingerprint_fields(mapping, fields)


def mapper_basis_fingerprint(idp: Any) -> str:
    """Fingerprint the exact non-SOC OrbitalMapper basis contract."""

    basis = getattr(idp, "basis", None)
    norbs = getattr(idp, "norbs", None)
    if not isinstance(basis, Mapping):
        raise TypeError("OrbitalMapper must expose a basis mapping.")
    normalized_basis = {
        str(symbol): [str(orbital) for orbital in basis[symbol]]
        for symbol in sorted(basis)
    }
    if isinstance(norbs, Mapping):
        normalized_norbs = {
            str(symbol): int(norbs[symbol]) for symbol in sorted(basis)
        }
    else:
        # The production OrbitalMapper creates ``norbs`` lazily in
        # get_orbital_maps(), while the upper-triangle mapper exposes it at
        # construction.  Derive the same value without mutating mapper state.
        angular_momentum = {letter: l for l, letter in enumerate("spdfgh")}
        normalized_norbs = {}
        for symbol, orbitals in normalized_basis.items():
            count = 0
            for orbital in orbitals:
                match = re.search(r"([A-Za-z])$", orbital)
                if match is None or match.group(1).lower() not in angular_momentum:
                    raise ValueError(
                        f"Cannot infer AO dimension from basis orbital {orbital!r}."
                    )
                count += 2 * angular_momentum[match.group(1).lower()] + 1
            normalized_norbs[symbol] = count
    payload = {
        "has_soc": bool(getattr(idp, "has_soc", False)),
        "basis": normalized_basis,
        "norbs": normalized_norbs,
        "full_basis": [str(x) for x in getattr(idp, "full_basis", ())],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def canonical_edge_graph(
    atomic_numbers: Any,
    edge_index: Any,
    edge_cell_shift: Any,
    *,
    shift_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and canonicalize a unique directed periodic graph.

    Edge indices and lattice translations are returned as int64 arrays.  A
    near-integer shift is rejected instead of being rounded differently by
    separate feature/block conversion paths.
    """

    numbers = _numpy(atomic_numbers)
    if numbers.ndim not in {1, 2}:
        raise ValueError(f"atomic_numbers must be rank 1/2, got {numbers.shape}.")
    n_atoms = int(numbers.reshape(-1).shape[0])

    edge_raw = _numpy(edge_index)
    if edge_raw.ndim != 2 or edge_raw.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,E], got {edge_raw.shape}.")
    if not np.issubdtype(edge_raw.dtype, np.number) or not np.isfinite(edge_raw).all():
        raise ValueError("edge_index must contain finite integer indices.")
    edge = edge_raw.astype(np.int64)
    if not np.array_equal(edge_raw, edge):
        raise ValueError("edge_index contains non-integer values.")
    if edge.size and (edge.min() < 0 or edge.max() >= n_atoms):
        raise ValueError(
            f"edge_index contains atom indices outside [0,{max(n_atoms - 1, 0)}]."
        )

    shift_raw = _numpy(edge_cell_shift)
    if shift_raw.ndim != 2 or shift_raw.shape != (edge.shape[1], 3):
        raise ValueError(
            "edge_cell_shift must have shape [E,3] matching edge_index; got "
            f"{shift_raw.shape} for E={edge.shape[1]}."
        )
    if not np.issubdtype(shift_raw.dtype, np.number) or not np.isfinite(shift_raw).all():
        raise ValueError("edge_cell_shift must contain finite lattice translations.")
    rounded = np.rint(shift_raw)
    max_error = float(np.max(np.abs(shift_raw - rounded), initial=0.0))
    if max_error > float(shift_tolerance):
        raise ValueError(
            "edge_cell_shift must be integer-valued within tolerance "
            f"{shift_tolerance:.1e}; max error is {max_error:.3e}."
        )
    shift = rounded.astype(np.int64)

    keys = [
        (int(u), int(v), int(r0), int(r1), int(r2))
        for (u, v), (r0, r1, r2) in zip(edge.T.tolist(), shift.tolist())
    ]
    if len(set(keys)) != len(keys):
        seen: set[tuple[int, int, int, int, int]] = set()
        duplicate = None
        for key in keys:
            if key in seen:
                duplicate = key
                break
            seen.add(key)
        raise ValueError(f"stored edge graph contains duplicate directed edge {duplicate}.")
    return edge, shift


def edge_graph_fingerprint(
    atomic_numbers: Any,
    edge_index: Any,
    edge_cell_shift: Any,
    *,
    basis_fingerprint: str,
) -> str:
    """Fingerprint atom identities, ordered directed edges, shifts, and basis."""

    edge, shift = canonical_edge_graph(
        atomic_numbers, edge_index, edge_cell_shift
    )
    digest = hashlib.sha256()
    _update_array(digest, "atomic_numbers", _numpy(atomic_numbers).reshape(-1).astype(np.int64))
    _update_array(digest, "edge_index", edge)
    _update_array(digest, "edge_cell_shift", shift)
    digest.update(require_sha256(basis_fingerprint, field=BASIS_FINGERPRINT_KEY).encode("ascii"))
    return digest.hexdigest()


def assert_record_fingerprint(
    record: Mapping[str, Any],
    *,
    field: str,
    actual: str,
) -> None:
    expected = require_sha256(record.get(field), field=field)
    if expected != actual:
        raise ValueError(
            f"{field} mismatch: stored={expected}, recomputed={actual}. "
            "The cached representation is not aligned with its declared provenance."
        )


__all__ = [
    "ABSOLUTE_FULL_H_SEMANTICS",
    "BASIS_FINGERPRINT_KEY",
    "DUAL_PRIOR_SAMPLE_SCHEMA",
    "EDGE_GRAPH_FINGERPRINT_KEY",
    "FULL_H_TARGET_FINGERPRINT_KEY",
    "H0_RESIDUAL_SEMANTICS",
    "P2_BLOCK_FINGERPRINT_KEY",
    "P2_BUNDLE_FINGERPRINT_KEY",
    "P2_RME_FINGERPRINT_KEY",
    "P2_SAMPLE_SCHEMA",
    "P2_SOURCE_FINGERPRINT_KEY",
    "P23_BLOCK_FINGERPRINT_KEY",
    "P23_BUNDLE_FINGERPRINT_KEY",
    "P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY",
    "P23_RME_FINGERPRINT_KEY",
    "P23_SOURCE_FINGERPRINT_KEY",
    "PRIOR_FIELD_SPECS",
    "PriorFieldSpec",
    "ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY",
    "ROW_ALIGNED_DATA_FINGERPRINT_KEY",
    "ROW_ALIGNED_FIELD_CANDIDATES",
    "SAMPLE_SCHEMA_KEY",
    "TARGET_SEMANTICS_KEY",
    "TARGET_SOURCE_KEY",
    "assert_record_fingerprint",
    "canonical_edge_graph",
    "edge_graph_fingerprint",
    "fingerprint_fields",
    "fingerprint_present_row_aligned_fields",
    "fingerprint_text_fields",
    "mapper_basis_fingerprint",
    "require_sha256",
    "resolve_prior_field_spec",
    "resolve_prior_field_spec_from_raw_key",
]

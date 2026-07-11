"""Fail-closed adapter for offline ``h0rebuild`` H[rho0] artifacts.

This module keeps the numerical reconstruction outside the training loop.  It
verifies the artifact and graph first, then delegates the SOC AO-to-RME packing
to DeePTB's audited :func:`block_to_feature` implementation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import ase.data
import numpy as np
import torch

from dptb.data import AtomicDataDict
from dptb.data.interfaces.ham_to_feature import block_to_feature
PHYSICAL_H0_SIDECAR_SCHEMA = "emolflow.physical_h0_sidecar/v1"
PHYSICAL_H0_META_KEY = "physical_h0_meta"
DEEPNET_RME_REPRESENTATION = "deeptb.rme_soc_real_imag/v1"
_SUPPORTED_ENERGY_UNITS = frozenset({"eV", "Ry", "Ha"})


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(_numpy(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes())
    return digest.hexdigest()


def structure_signature(record: Mapping[str, Any]) -> str:
    required = (
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
        AtomicDataDict.POSITIONS_KEY,
        AtomicDataDict.CELL_KEY,
    )
    missing = [key for key in required if record.get(key) is None]
    if missing:
        raise KeyError(f"Physical H0 record is missing structure fields {missing}")
    atomic_numbers = _numpy(record[AtomicDataDict.ATOMIC_NUMBERS_KEY]).reshape(-1)
    positions = _numpy(record[AtomicDataDict.POSITIONS_KEY]).reshape(-1, 3)
    cell = _numpy(record[AtomicDataDict.CELL_KEY]).reshape(3, 3)
    if positions.shape[0] != atomic_numbers.shape[0]:
        raise ValueError("Position rows do not match atomic_numbers")
    digest = hashlib.sha256()
    for value in (atomic_numbers, positions, cell):
        digest.update(array_sha256(value).encode("ascii"))
    return digest.hexdigest()


def edge_signature(record: Mapping[str, Any]) -> str:
    edge_index = _numpy(record[AtomicDataDict.EDGE_INDEX_KEY])
    edge_shift = _numpy(record[AtomicDataDict.EDGE_CELL_SHIFT_KEY])
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,n_edge], got {edge_index.shape}")
    if edge_shift.shape != (edge_index.shape[1], 3):
        raise ValueError(
            f"edge_cell_shift must have shape [{edge_index.shape[1]},3], got {edge_shift.shape}"
        )
    digest = hashlib.sha256()
    digest.update(array_sha256(edge_index).encode("ascii"))
    digest.update(array_sha256(edge_shift).encode("ascii"))
    return digest.hexdigest()


def build_physical_h0_meta(
    record: Mapping[str, Any],
    *,
    node_key: str = AtomicDataDict.NODE_PHYSICAL_H0_KEY,
    edge_key: str = AtomicDataDict.EDGE_PHYSICAL_H0_KEY,
    energy_unit: str = "eV",
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the order-sensitive sidecar contract consumed by EMolFlow."""
    if energy_unit not in _SUPPORTED_ENERGY_UNITS:
        raise ValueError(f"Unsupported physical H0 energy unit {energy_unit!r}")
    node = _numpy(record[node_key])
    edge = _numpy(record[edge_key])
    for name, value in ((node_key, node), (edge_key, edge)):
        if value.ndim != 2 or np.iscomplexobj(value) or value.dtype.kind != "f":
            raise TypeError(f"{name} must be a real floating rank-2 RME array")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinity")
    meta: dict[str, Any] = {
        "schema": PHYSICAL_H0_SIDECAR_SCHEMA,
        "representation": DEEPNET_RME_REPRESENTATION,
        "energy_unit": energy_unit,
        "node_key": node_key,
        "edge_key": edge_key,
        "node_shape": list(node.shape),
        "edge_shape": list(edge.shape),
        "node_sha256": array_sha256(node),
        "edge_sha256": array_sha256(edge),
        "structure_signature": structure_signature(record),
        "edge_signature": edge_signature(record),
    }
    if source:
        meta["source"] = dict(source)
    return meta


def materialize_h0rebuild_features(
    data,
    idp,
    artifact_path: str | Path,
    *,
    target_energy_unit: str = "eV",
    data_length_unit: str = "angstrom",
    orthogonal: bool = False,
    verify_hash: bool = True,
    feature_output_dtype: torch.dtype | None = torch.float64,
    hermitian_atol: float = 1.0e-8,
    hermitian_rtol: float = 1.0e-7,
) -> tuple[Any, dict[str, Any]]:
    """Validate and materialize dedicated node/edge physical-H0 RME fields.

    ``artifact_path`` is an ``h0rebuild.artifact/v1`` NPZ plus adjacent JSON.
    No missing edge is allowed, no atom/graph reorder is inferred, and raw
    complex SOC blocks never enter the flow prior directly.
    """
    try:
        from h0rebuild.deeptb import (
            load_deeptb_artifact,
            preflight_deeptb_graph,
            validate_structure_against_artifact,
        )
    except ImportError as exc:
        raise ImportError(
            "Materializing physical H0 features requires a compatible h0rebuild "
            "installation; install it in the preprocessing environment."
        ) from exc

    if target_energy_unit not in _SUPPORTED_ENERGY_UNITS:
        raise ValueError(f"Unsupported target_energy_unit={target_energy_unit!r}")
    if not bool(getattr(idp, "has_soc", False)):
        raise ValueError("h0rebuild physical H0 integration requires a SOC OrbitalMapper")
    if not bool(getattr(idp, "soc_complex_doubling", True)):
        raise ValueError(
            "Physical H0 sidecars require soc_complex_doubling=True so complex AO "
            "blocks become real Re/Im feature channels."
        )

    idp.get_orbital_maps()
    idp.get_orbpair_maps()
    artifact = load_deeptb_artifact(
        artifact_path,
        target_energy_unit=target_energy_unit,
        verify_hash=verify_hash,
    )

    atomic_numbers = _numpy(data[AtomicDataDict.ATOMIC_NUMBERS_KEY]).reshape(-1)
    species: list[str] = []
    for number in atomic_numbers:
        z = int(number)
        if z <= 0 or z >= len(ase.data.chemical_symbols):
            raise ValueError(f"Invalid atomic number {z}")
        species.append(ase.data.chemical_symbols[z])
    expected_counts = tuple(int(idp.norbs[symbol]) for symbol in species)
    if artifact.orbital_counts != expected_counts:
        raise ValueError(
            "Artifact/DeePTB spatial-orbital counts differ: "
            f"artifact={artifact.orbital_counts}, idp={expected_counts}"
        )

    structure_report = validate_structure_against_artifact(
        artifact.metadata,
        cell=data[AtomicDataDict.CELL_KEY],
        positions=data[AtomicDataDict.POSITIONS_KEY],
        species=species,
        data_length_unit=data_length_unit,
    )
    graph_report = preflight_deeptb_graph(
        artifact.blocks,
        edge_index=data[AtomicDataDict.EDGE_INDEX_KEY],
        edge_cell_shift=data[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
        orbital_counts=artifact.orbital_counts,
        has_soc=True,
        hermitian_atol=hermitian_atol,
        hermitian_rtol=hermitian_rtol,
    )

    block_to_feature(
        data,
        idp,
        artifact.blocks,
        False,
        orthogonal,
        node_field=AtomicDataDict.NODE_PHYSICAL_H0_KEY,
        edge_field=AtomicDataDict.EDGE_PHYSICAL_H0_KEY,
        missing_block_policy="error",
        output_dtype=feature_output_dtype,
    )
    node = data[AtomicDataDict.NODE_PHYSICAL_H0_KEY]
    edge = data[AtomicDataDict.EDGE_PHYSICAL_H0_KEY]
    if torch.is_complex(node) or torch.is_complex(edge):
        raise TypeError("Physical H0 RME features must be real after SOC Re/Im packing")
    if node.ndim != 2 or edge.ndim != 2:
        raise ValueError(
            f"Physical H0 features must be rank 2, got node={tuple(node.shape)}, "
            f"edge={tuple(edge.shape)}"
        )
    if node.shape[0] != len(species):
        raise ValueError("Physical H0 node rows do not match the atom count")
    if edge.shape[0] != int(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]):
        raise ValueError("Physical H0 edge rows do not match edge_index")
    if node.shape[1] != edge.shape[1]:
        raise ValueError("Physical H0 node/edge feature widths differ")
    if not torch.isfinite(node).all() or not torch.isfinite(edge).all():
        raise ValueError("Physical H0 RME features contain NaN or infinity")

    report = {
        "schema": PHYSICAL_H0_SIDECAR_SCHEMA,
        "representation": DEEPNET_RME_REPRESENTATION,
        "artifact_path": str(Path(artifact_path).resolve()),
        "artifact_npz_sha256": artifact.metadata.get("npz_sha256"),
        "artifact_metadata_content_sha256": artifact.metadata.get(
            "metadata_content_sha256"
        ),
        "source_energy_unit": artifact.source_energy_unit,
        "target_energy_unit": artifact.target_energy_unit,
        "energy_scale": artifact.energy_scale,
        "orbital_counts": list(artifact.orbital_counts),
        "node_shape": list(node.shape),
        "edge_shape": list(edge.shape),
        "feature_output_dtype": str(node.dtype),
        "structure": structure_report,
        "graph": graph_report,
    }
    return data, report

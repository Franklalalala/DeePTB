#!/usr/bin/env python3
"""Integrity and numerical spot audit for a completed non-SOC P2 table."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from dptb.data.interfaces.p2_table import (
    P2_TABLE_SCHEMA,
    P2TableStore,
    validate_p2_component_numerical_contract,
)


SCHEMA = "deeptb.p2_radial_table_audit/v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sample(keys: Sequence[str], count: int, rng: random.Random) -> list[str]:
    ordered = sorted(keys)
    if count <= 0 or count >= len(ordered):
        return ordered
    return sorted(rng.sample(ordered, count))


def _metadata_inventory(root: Path, entries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing: list[str] = []
    invalid_sha: list[str] = []
    paths: list[str] = []
    for key, metadata in entries.items():
        relative = str(metadata.get("path", ""))
        paths.append(relative)
        if not (root / relative).is_file():
            missing.append(key)
        checksum = str(metadata.get("sha256", ""))
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            invalid_sha.append(key)
    duplicate_paths = len(paths) - len(set(paths))
    if missing or invalid_sha or duplicate_paths:
        raise ValueError(
            "Table inventory failed: "
            f"missing={missing[:5]}, invalid_sha={invalid_sha[:5]}, "
            f"duplicate_paths={duplicate_paths}."
        )
    return {
        "entries": len(entries),
        "missing": 0,
        "invalid_sha": 0,
        "duplicate_paths": 0,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    root = args.table_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != P2_TABLE_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("P2 table manifest is not a completed v1 table.")
    rng = random.Random(int(args.seed))
    store = P2TableStore(
        root,
        verify_checksums=True,
        require_explicit_source_identity=True,
    )
    base_inventory = _metadata_inventory(root, manifest["base_tables"])
    projector_inventory = _metadata_inventory(root, manifest["projector_tables"])

    species_checks = []
    for symbol in sorted(store.species):
        onsite = store.onsite(symbol)
        d_eff = store.d_eff(symbol)
        onsite_components = {
            component: float(
                np.max(
                    np.abs(store.onsite_component(symbol, component)),
                    initial=0.0,
                )
            )
            for component in store.onsite_component_arrays
        }
        species_checks.append(
            {
                "symbol": symbol,
                "orbital_norb": int(store.species[symbol]["orbital_norb"]),
                "projector_norb": int(store.species[symbol]["projector_norb"]),
                "onsite_max_abs_ry": float(
                    np.max(np.abs(onsite), initial=0.0)
                ),
                "d_eff_max_abs_ry": float(np.max(np.abs(d_eff), initial=0.0)),
                "onsite_component_max_abs": onsite_components,
            }
        )

    selected_base = set(
        _sample(list(store.base_tables), int(args.base_samples), rng)
    )
    # Always include every self pair and several chemically asymmetric pairs.
    for symbol in sorted(store.species):
        selected_base.add(f"{symbol}|{symbol}")
    base_checks = []
    worst_reciprocity = 0.0
    for key in sorted(selected_base):
        left, right = key.split("|", 1)
        table = store.base(left, right)
        reverse = store.base(right, left)
        distance = 0.37 * min(table.support_bohr, reverse.support_bohr)
        direction = np.asarray([0.31, -0.47, 0.826], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        displacement = direction * distance
        direct = table.evaluate(displacement)
        opposite = reverse.evaluate(-displacement).T
        component_max_abs = {
            component: float(
                np.max(
                    np.abs(
                        store.base_component(left, right, component).evaluate(
                            displacement
                        )
                    ),
                    initial=0.0,
                )
            )
            for component in store.base_component_arrays
        }
        error = float(np.max(np.abs(direct - opposite), initial=0.0))
        worst_reciprocity = max(worst_reciprocity, error)
        base_checks.append(
            {
                "pair": key,
                "shape": list(direct.shape),
                "distance_bohr": distance,
                "reciprocity_max_abs_ry": error,
                "component_max_abs": component_max_abs,
            }
        )

    selected_projector = _sample(
        list(store.projector_tables), int(args.projector_samples), rng
    )
    projector_checks = []
    for key in selected_projector:
        projector, orbital = key.split("|", 1)
        table = store.projector(projector, orbital)
        distance = 0.41 * table.support_bohr
        value = table.evaluate(np.asarray([distance, 0.0, 0.0]))
        projector_checks.append(
            {
                "pair": key,
                "shape": list(value.shape),
                "max_abs": float(np.max(np.abs(value), initial=0.0)),
            }
        )

    if worst_reciprocity > float(args.reciprocity_tolerance):
        raise ValueError(
            f"Base-table reciprocity error {worst_reciprocity:.3e} exceeds "
            f"{float(args.reciprocity_tolerance):.3e}."
        )
    component_gate = validate_p2_component_numerical_contract(
        store,
        reciprocity_tolerance=float(args.component_reciprocity_tolerance),
        reconstruction_tolerance=float(args.component_reconstruction_tolerance),
        distance_samples=int(args.component_gate_distance_samples),
    )
    if component_gate["status"] != "pass":
        raise ValueError(
            "P2 component numerical gate failed: "
            + "; ".join(component_gate.get("failures", []))
        )
    full_base_scope = len(selected_base) == len(store.base_tables)
    full_projector_scope = len(selected_projector) == len(store.projector_tables)
    production_eligible = full_base_scope and full_projector_scope
    audit_mode = "full_production" if production_eligible else "sampled_non_production"
    expected_checksum_files = (
        len(store.species) + len(store.base_tables) + len(store.projector_tables)
    )
    if production_eligible and store.verified_path_count != expected_checksum_files:
        raise ValueError(
            "Full P2 audit did not checksum one distinct file per species/base/"
            "projector entry; cross-category duplicate paths or an unloaded shard "
            f"exist (verified={store.verified_path_count}, "
            f"expected={expected_checksum_files})."
        )
    report = {
        "schema": SCHEMA,
        "table_root": str(root),
        "audit_mode": audit_mode,
        "production_eligible": production_eligible,
        "requested_base_samples": int(args.base_samples),
        "requested_projector_samples": int(args.projector_samples),
        "species": len(store.species),
        "species_source_identity_sha256": store.species_source_identity_sha256,
        "base_component_contract": store.base_component_contract,
        "component_numerical_gate": component_gate,
        "base_inventory": base_inventory,
        "projector_inventory": projector_inventory,
        "species_checks": species_checks,
        "base_samples": base_checks,
        "projector_samples": projector_checks,
        "worst_base_reciprocity_max_abs_ry": worst_reciprocity,
        "verified_file_checksums": store.verified_path_count,
        "expected_full_file_checksums": expected_checksum_files,
        "elapsed_seconds": time.time() - started,
        "status": "pass" if production_eligible else "pass_non_production",
    }
    _atomic_json(args.output.resolve(), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-samples",
        type=int,
        default=0,
        help=(
            "explicit non-production sample count; 0 (default) loads and checks "
            "every base shard"
        ),
    )
    parser.add_argument(
        "--projector-samples",
        type=int,
        default=0,
        help=(
            "explicit non-production sample count; 0 (default) loads and checks "
            "every projector shard"
        ),
    )
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--reciprocity-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--component-reciprocity-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--component-reconstruction-tolerance", type=float, default=1.0e-6
    )
    parser.add_argument("--component-gate-distance-samples", type=int, default=33)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = audit(parse_args(argv))
    if not result["production_eligible"]:
        print(
            "WARNING: sampled P2 audit is diagnostic only and is not a production gate.",
            file=sys.stderr,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "species": result["species"],
                "base_samples": len(result["base_samples"]),
                "projector_samples": len(result["projector_samples"]),
                "worst_base_reciprocity_max_abs_ry": result[
                    "worst_base_reciprocity_max_abs_ry"
                ],
                "component_numerical_gate": result["component_numerical_gate"][
                    "status"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

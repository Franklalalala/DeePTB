#!/usr/bin/env python3
"""Fail-closed integrity and numerical audit for a completed P23 table.

This audit is intentionally separate from the builder.  A production pass
checks every declared species/projector and factor shard, validates the NPZ
payload contracts, and verifies the immutable P2/census/code provenance bound
into the P23 manifest.  Small interpolation probes exercise the runtime table
reader, but they are not a replacement for a structure-level direct oracle.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np

from dptb.data.interfaces.p23_table import (
    P23_FACTOR_SUPPORT_SEMANTICS,
    P23_VNA_TABLE_SCHEMA,
    P23VNAFactorTableStore,
)


SCHEMA = "deeptb.p23_factorized_vna_table_audit/v1"
PARENT_P2_OPERATOR = "T + unique(VNA_i,VNA_j) + factorized all-K Vnl"
PARENT_P2_SOC_MODE = "non-SOC scalar spin trace"
ORACLE_SCHEMA = "deeptb.openmx_inspired_vna3c_rank_sweep/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _resolved_member(root: Path, relative: Any, *, label: str) -> Path:
    text = str(relative or "")
    if not text or Path(text).is_absolute():
        raise ValueError(f"{label}: table path must be non-empty and relative")
    path = (root / text).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"{label}: table path escapes root: {text!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _valid_sha256(value: Any, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label}: invalid SHA256 {value!r}")
    return text


def _audit_species(
    root: Path, symbol: str, row: Mapping[str, Any]
) -> dict[str, Any]:
    path = _resolved_member(root, row.get("array_path"), label=f"species:{symbol}")
    expected_sha = _valid_sha256(
        row.get("array_sha256"), label=f"species:{symbol}"
    )
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"species:{symbol}: checksum mismatch {actual_sha} != {expected_sha}"
        )
    with np.load(path, allow_pickle=False) as payload:
        required = {"epsilon_ao", "radial_epsilon", "radial_l", "radial_n"}
        missing = required.difference(payload.files)
        if missing:
            raise KeyError(f"species:{symbol}: missing arrays {sorted(missing)}")
        epsilon = np.asarray(payload["epsilon_ao"])
        radial_epsilon = np.asarray(payload["radial_epsilon"])
        radial_l = np.asarray(payload["radial_l"])
        radial_n = np.asarray(payload["radial_n"])
    expected_norb = int(row["vna_projector_norb"])
    if epsilon.shape != (expected_norb,):
        raise ValueError(
            f"species:{symbol}: epsilon shape {epsilon.shape} != {(expected_norb,)}"
        )
    if not np.issubdtype(epsilon.dtype, np.floating) or not np.isfinite(epsilon).all():
        raise ValueError(f"species:{symbol}: epsilon is not finite floating data")
    if not (
        radial_epsilon.ndim == radial_l.ndim == radial_n.ndim == 1
        and len(radial_epsilon) == len(radial_l) == len(radial_n)
    ):
        raise ValueError(f"species:{symbol}: radial projector arrays are misaligned")
    if len(radial_epsilon) == 0 or not np.isfinite(radial_epsilon).all():
        raise ValueError(f"species:{symbol}: invalid radial epsilon")
    if not np.issubdtype(radial_l.dtype, np.integer) or not np.issubdtype(
        radial_n.dtype, np.integer
    ):
        raise TypeError(f"species:{symbol}: radial l/n arrays must be integer")
    return {
        "symbol": symbol,
        "path": str(row["array_path"]),
        "sha256": actual_sha,
        "vna_projector_norb": expected_norb,
        "radial_projectors": int(len(radial_epsilon)),
        "epsilon_max_abs": float(np.max(np.abs(epsilon), initial=0.0)),
    }


def _audit_factor(
    root: Path,
    key: str,
    row: Mapping[str, Any],
    species: Mapping[str, Mapping[str, Any]],
    distance_step: float,
) -> dict[str, Any]:
    if "|" not in key:
        raise ValueError(f"factor:{key}: invalid key")
    vna, ao = key.split("|", 1)
    if vna not in species or ao not in species:
        raise KeyError(f"factor:{key}: unknown species")
    path = _resolved_member(root, row.get("path"), label=f"factor:{key}")
    expected_sha = _valid_sha256(row.get("sha256"), label=f"factor:{key}")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"factor:{key}: checksum mismatch {actual_sha} != {expected_sha}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "distances",
            "values",
            "left_shells",
            "right_shells",
            "support_bohr",
        }
        missing = required.difference(payload.files)
        if missing:
            raise KeyError(f"factor:{key}: missing arrays {sorted(missing)}")
        distances = np.asarray(payload["distances"])
        values = np.asarray(payload["values"])
        left_shells = tuple(int(value) for value in payload["left_shells"])
        right_shells = tuple(int(value) for value in payload["right_shells"])
        support = float(np.asarray(payload["support_bohr"]))
    expected_shape = tuple(int(value) for value in row["shape"])
    if values.shape != expected_shape or len(distances) != expected_shape[0]:
        raise ValueError(
            f"factor:{key}: payload shape {values.shape}/{distances.shape} "
            f"!= {expected_shape}"
        )
    if values.dtype != np.dtype(str(row.get("value_dtype", ""))):
        raise TypeError(
            f"factor:{key}: dtype {values.dtype} != {row.get('value_dtype')!r}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"factor:{key}: values contain NaN or infinity")
    if distances.ndim != 1 or len(distances) < 2:
        raise ValueError(f"factor:{key}: invalid distance grid")
    if not np.isfinite(distances).all() or float(distances[0]) != 0.0:
        raise ValueError(f"factor:{key}: distance grid must be finite and start at zero")
    if np.any(np.diff(distances) <= 0.0):
        raise ValueError(f"factor:{key}: distance grid is not strictly increasing")
    if not np.allclose(
        np.diff(distances), distance_step, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"factor:{key}: distance grid does not use the build step")
    declared_support = float(row["support_bohr"])
    if not np.isclose(support, declared_support, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"factor:{key}: support payload/manifest mismatch")
    # The builder deliberately leaves a uniform-grid sentinel beyond physical
    # support so cubic interpolation never needs to extrapolate at the cutoff.
    expected_end = math.ceil(support / distance_step) * distance_step
    if expected_end < support + 0.25 * distance_step:
        expected_end += distance_step
    if not np.isclose(
        float(distances[-1]), expected_end, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(
            f"factor:{key}: distance endpoint {distances[-1]} != {expected_end}"
        )
    expected_left = tuple(int(x) for x in species[vna]["vna_projector_shells"])
    expected_right = tuple(int(x) for x in species[ao]["orbital_shells"])
    if left_shells != expected_left or right_shells != expected_right:
        raise ValueError(f"factor:{key}: shell contract mismatch")
    return {
        "pair": key,
        "path": str(row["path"]),
        "sha256": actual_sha,
        "shape": list(values.shape),
        "support_bohr": support,
        "max_abs": float(np.max(np.abs(values), initial=0.0)),
    }


def _audit_physical_oracle(
    table_manifest: Mapping[str, Any], oracle_path: Path
) -> dict[str, Any]:
    oracle_path = oracle_path.resolve()
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if oracle.get("schema") != ORACLE_SCHEMA:
        raise ValueError("P23 physical-oracle schema mismatch")
    if int(oracle.get("geometry_count", 0)) < 9:
        raise ValueError("P23 physical oracle must contain at least nine geometries")
    source = oracle.get("source", {})
    if source.get("kind") != "independent_direct_field_quadrature":
        raise ValueError("P23 oracle is not independent direct field quadrature")
    quadrature = source.get("quadrature", {})
    if any(
        int(quadrature.get(key, 0)) < minimum
        for key, minimum in (("n_r", 192), ("n_mu", 48), ("n_phi", 96))
    ):
        raise ValueError("P23 independent direct oracle quadrature is too coarse")

    species = table_manifest["species"]
    if "Si" not in species:
        raise KeyError("P23 table lacks the Si species used by the physical oracle")
    si = species["Si"]
    if str(oracle.get("orb", {}).get("sha256")) != str(si["orbital_sha256"]):
        raise ValueError("P23 oracle Si ORB differs from the table Si ORB")
    if str(oracle.get("upf", {}).get("sha256")) != str(si["upf_sha256"]):
        raise ValueError("P23 oracle Si UPF differs from the table Si UPF")

    table_settings = table_manifest["build_settings"]
    sbt = oracle.get("sbt_settings", {})
    for oracle_key, table_key in (
        ("n_k", "n_k"),
        ("kmax", "kmax_bohr_inv"),
        ("n_mu", "n_mu"),
        ("n_phi", "n_phi"),
    ):
        if float(sbt.get(oracle_key, -1)) != float(table_settings[table_key]):
            raise ValueError(
                f"P23 oracle/table SBT setting differs: {oracle_key}/{table_key}"
            )
    candidates = [
        row
        for row in oracle.get("results", [])
        if int(row.get("rank", -1)) == int(table_settings["radial_rank"])
        and int(row.get("l_buffer", -1)) == int(table_settings["l_buffer"])
        and row.get("status") == "ok"
    ]
    if len(candidates) != 1:
        raise ValueError("P23 oracle lacks one exact rank/l-buffer result")
    row = candidates[0]
    metrics = row.get("metrics", {})
    gates = {
        "relative_frobenius": 5.0e-2,
        "mae_ev": 3.0e-3,
        "max_abs_ev": 4.0e-2,
    }
    failures = {
        key: {"actual": float(metrics.get(key, float("inf"))), "limit": limit}
        for key, limit in gates.items()
        if not np.isfinite(float(metrics.get(key, float("inf"))))
        or float(metrics.get(key, float("inf"))) > limit
    }
    if failures:
        raise ValueError(f"P23 physical-oracle accuracy gate failed: {failures}")
    return {
        "status": "pass",
        "scope": "Si-only rank/l-buffer projector approximation on 9 geometries",
        "universal_multi_species_accuracy_established": False,
        "oracle_path": str(oracle_path),
        "oracle_sha256": _sha256(oracle_path),
        "schema": ORACLE_SCHEMA,
        "geometry_count": int(oracle["geometry_count"]),
        "source_kind": source["kind"],
        "source_sha256": str(source.get("sha256", "")),
        "quadrature": dict(quadrature),
        "rank": int(row["rank"]),
        "l_buffer": int(row["l_buffer"]),
        "metrics": {key: float(metrics[key]) for key in gates},
        "acceptance_limits": gates,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    root = args.table_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != P23_VNA_TABLE_SCHEMA:
        raise ValueError("P23 table schema mismatch")
    if manifest.get("complete") is not True:
        raise ValueError("P23 table is not complete")
    if manifest.get("factor_support_semantics") != P23_FACTOR_SUPPORT_SEMANTICS:
        raise ValueError("P23 factor support semantics mismatch")
    species: Mapping[str, Mapping[str, Any]] = manifest["species"]
    factors: Mapping[str, Mapping[str, Any]] = manifest["factor_tables"]
    distance_step = float(manifest["build_settings"]["distance_step_bohr"])
    if not np.isfinite(distance_step) or distance_step <= 0.0:
        raise ValueError("P23 manifest has an invalid distance step")
    active = [str(value) for value in manifest["active_factor_pairs"]]
    if len(active) != len(set(active)) or set(active) != set(factors):
        raise ValueError("active factor-pair inventory does not match factor tables")
    if int(manifest.get("factor_census_case_count", 0)) <= 0:
        raise ValueError("P23 factor census has no cases")

    declared_paths = [str(row.get("array_path", "")) for row in species.values()]
    declared_paths.extend(str(row.get("path", "")) for row in factors.values())
    if len(declared_paths) != len(set(declared_paths)):
        raise ValueError("P23 manifest contains duplicate shard paths")

    external_checks: dict[str, str] = {}
    external_payloads: dict[str, Mapping[str, Any]] = {}
    for label, path_key, sha_key in (
        ("base_p2_manifest", "base_p2_table_manifest", "base_p2_table_manifest_sha256"),
        ("factor_census", "factor_census", "factor_census_sha256"),
    ):
        path = Path(str(manifest[path_key])).resolve()
        expected = _valid_sha256(manifest[sha_key], label=label)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{label}: checksum mismatch {actual} != {expected}")
        external_checks[label] = actual
        external_payloads[label] = json.loads(path.read_text(encoding="utf-8"))
    base_p2 = external_payloads["base_p2_manifest"]
    base_settings = base_p2.get("build_settings", {})
    expected_parent_contract = {
        "operator": PARENT_P2_OPERATOR,
        "soc_mode": PARENT_P2_SOC_MODE,
        "energy_unit": "Ry",
    }
    for key, expected in expected_parent_contract.items():
        actual = (
            base_p2.get("energy_unit") if key == "energy_unit" else base_settings.get(key)
        )
        if actual != expected:
            raise ValueError(f"base P2 {key} contract differs: {actual!r} != {expected!r}")
    declared_parent_contract = manifest.get("base_p2_operator_contract", {})
    for key, expected in expected_parent_contract.items():
        if declared_parent_contract.get(key) != expected:
            raise ValueError(f"P23 manifest does not bind base P2 {key}")
    code_identity = manifest["code_identity"]
    if _json_sha256(code_identity) != str(manifest["code_identity_sha256"]):
        raise ValueError("P23 code identity digest is inconsistent")
    for label, path, sha_key in (
        ("builder", Path(__file__).with_name("build_nonsoc_p23_tables.py"), "builder_script_sha256"),
        (
            "p23_interface",
            Path(__file__).resolve().parents[1] / "dptb/data/interfaces/p23_table.py",
            "p23_interface_sha256",
        ),
        ("gate1", Path(str(manifest["source"]["gate1_script"])), "gate1_script_sha256"),
    ):
        actual = _sha256(path.resolve())
        expected = _valid_sha256(code_identity[sha_key], label=label)
        if actual != expected:
            raise ValueError(f"{label}: current source differs from table build source")
        external_checks[label] = actual

    tasks: list[tuple[str, str, Mapping[str, Any]]] = []
    tasks.extend(("species", key, row) for key, row in species.items())
    tasks.extend(("factor", key, row) for key, row in factors.items())
    species_results: list[dict[str, Any]] = []
    factor_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {}
        for kind, key, row in tasks:
            function = _audit_species if kind == "species" else _audit_factor
            positional = (
                (root, key, row)
                if kind == "species"
                else (root, key, row, species, distance_step)
            )
            futures[executor.submit(function, *positional)] = kind
        for future in as_completed(futures):
            result = future.result()
            if futures[future] == "species":
                species_results.append(result)
            else:
                factor_results.append(result)

    store = P23VNAFactorTableStore(root, verify_checksums=False)
    rng = random.Random(int(args.seed))
    selected = sorted(factors)
    if 0 < int(args.spot_samples) < len(selected):
        selected = sorted(rng.sample(selected, int(args.spot_samples)))
    spot_results = []
    for key in selected:
        vna, ao = key.split("|", 1)
        table = store.factor(vna, ao)
        direction = np.asarray([0.31, -0.47, 0.826], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        displacement = direction * (0.413 * table.support_bohr)
        value = table.evaluate(displacement)
        outside = table.evaluate(direction * (table.support_bohr + 1.0e-6))
        if not np.isfinite(value).all() or np.any(outside != 0.0):
            raise ValueError(f"factor:{key}: runtime interpolation/support probe failed")
        spot_results.append(
            {
                "pair": key,
                "shape": list(value.shape),
                "max_abs": float(np.max(np.abs(value), initial=0.0)),
            }
        )

    physical_oracle = _audit_physical_oracle(
        manifest, args.oracle_report.resolve()
    )

    report = {
        "schema": SCHEMA,
        "status": "pass",
        "integrity_eligible": True,
        "experimental_training_eligible": True,
        "production_eligible": True,
        "qualification_scope": (
            "raw200 non-SOC experimental prior ablation; table integrity and "
            "Si rank-4 physical approximation are qualified, but universal "
            "62-species direct-oracle accuracy is not established"
        ),
        "table_root": str(root),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "species": len(species),
        "factor_tables": len(factors),
        "factor_census_case_count": int(manifest["factor_census_case_count"]),
        "verified_shards": len(species_results) + len(factor_results),
        "external_checks": external_checks,
        "species_checks": sorted(species_results, key=lambda row: row["symbol"]),
        "factor_checks": sorted(factor_results, key=lambda row: row["pair"]),
        "runtime_spot_checks": spot_results,
        "base_p2_operator_contract": expected_parent_contract,
        "physical_oracle": physical_oracle,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(args.output.resolve(), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--spot-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=713)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = audit(parse_args(argv))
    print(
        json.dumps(
            {
                "status": result["status"],
                "species": result["species"],
                "factor_tables": result["factor_tables"],
                "verified_shards": result["verified_shards"],
                "runtime_spot_checks": len(result["runtime_spot_checks"]),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

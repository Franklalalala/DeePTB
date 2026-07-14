#!/usr/bin/env python3
"""Compare fast radial-table P2 assembly with a preserved direct P2 oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from dptb.data.interfaces.p2_table import (
    P2_COMPONENT_NUMERICAL_GATE_SCHEMA,
    P2TableAssembler,
    P2TableStore,
)


SCHEMA = "deeptb.p2_table_oracle_smoke/v1"

# Conservative catastrophe gates for the production-default profile.  A
# campaign may tighten these from a validated oracle distribution, but it must
# never silently remove every numerical acceptance condition.
DEFAULT_MAX_ACTIVE_MAE_RY = 5.0e-3
DEFAULT_MAX_ACTIVE_RMSE_RY = 1.0e-2
DEFAULT_MAX_ACTIVE_P99_ABS_RY = 5.0e-2
DEFAULT_MAX_ACTIVE_ABS_RY = 2.0e-1
DEFAULT_MIN_ACTIVE_R2 = 0.99
DEFAULT_MAX_TABLE_HERMITICITY_RY = 5.0e-4
DEFAULT_MAX_ORACLE_HERMITICITY_RY = 5.0e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


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


def _load_gate1(path: Path):
    spec = importlib.util.spec_from_file_location("deeptb_gate1_p2_smoke", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import Gate-1 script {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    pred = np.asarray(prediction, dtype=np.float64).ravel()
    if ref.shape != pred.shape or ref.size == 0:
        raise ValueError(f"metric arrays are incompatible: {ref.shape}, {pred.shape}")
    error = pred - ref
    absolute_error = np.abs(error)
    centered = ref - np.mean(ref)
    denominator = float(np.dot(centered, centered))
    return {
        "count": int(ref.size),
        "mae_ry": float(np.mean(absolute_error)),
        "rmse_ry": float(np.sqrt(np.mean(error * error))),
        "p95_abs_ry": float(np.quantile(absolute_error, 0.95)),
        "p99_abs_ry": float(np.quantile(absolute_error, 0.99)),
        "p999_abs_ry": float(np.quantile(absolute_error, 0.999)),
        "max_abs_ry": float(np.max(absolute_error, initial=0.0)),
        "reference_max_abs_ry": float(np.max(np.abs(ref), initial=0.0)),
        "r2": float(1.0 - np.dot(error, error) / denominator)
        if denominator > 0.0
        else float("nan"),
    }


def _block_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    r_keys: np.ndarray,
    norb: Sequence[int],
) -> dict[str, Any]:
    offsets = np.concatenate(([0], np.cumsum(norb))).astype(int)
    buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "onsite": [],
        "hopping": [],
        "active_total": [],
    }
    active_blocks = 0
    for r_index, r_key in enumerate(np.asarray(r_keys, dtype=np.int64)):
        for i in range(len(norb)):
            rows = slice(int(offsets[i]), int(offsets[i + 1]))
            for j in range(len(norb)):
                cols = slice(int(offsets[j]), int(offsets[j + 1]))
                ref = reference[r_index, rows, cols]
                pred = prediction[r_index, rows, cols]
                if not (np.any(np.abs(ref) > 1.0e-14) or np.any(np.abs(pred) > 1.0e-14)):
                    continue
                active_blocks += 1
                kind = "onsite" if i == j and np.all(r_key == 0) else "hopping"
                buckets[kind].append((ref, pred))
                buckets["active_total"].append((ref, pred))
    output: dict[str, Any] = {"active_blocks": active_blocks}
    for name, blocks in buckets.items():
        if not blocks:
            output[name] = None
            continue
        output[name] = _metrics(
            np.concatenate([ref.ravel() for ref, _ in blocks]),
            np.concatenate([pred.ravel() for _, pred in blocks]),
        )
        output[name]["blocks"] = len(blocks)
    return output


def _hermiticity(array: np.ndarray, r_keys: np.ndarray) -> dict[str, Any]:
    key_to_index = {tuple(int(x) for x in row): i for i, row in enumerate(r_keys)}
    worst = 0.0
    compared = 0
    nonzero_pairs = 0
    missing_reverse: list[tuple[int, int, int]] = []
    seen_pairs: set[
        tuple[tuple[int, int, int], tuple[int, int, int]]
    ] = set()
    for key, index in key_to_index.items():
        opposite = tuple(-x for x in key)
        if opposite not in key_to_index:
            missing_reverse.append(key)
            continue
        pair = tuple(sorted((key, opposite)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        error = np.asarray(array[index]) - np.asarray(array[key_to_index[opposite]]).T
        worst = max(worst, float(np.max(np.abs(error), initial=0.0)))
        compared += 1
        if any(key):
            nonzero_pairs += 1
    return {
        "paired_r_keys": compared,
        "nonzero_paired_r_key_pairs": nonzero_pairs,
        "missing_reverse_r_keys": len(missing_reverse),
        "missing_reverse_examples": [list(key) for key in missing_reverse[:8]],
        "reverse_closed": not missing_reverse,
        "max_abs_ry": worst,
    }


def _numerical_gate(
    metrics: dict[str, Any],
    *,
    enabled: bool,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    for name, value in thresholds.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"numerical threshold {name} must be finite")
        if name != "min_active_r2" and float(value) < 0.0:
            raise ValueError(f"numerical threshold {name} must be non-negative")
    active = metrics["blocks"].get("active_total")
    if active is None:
        raise ValueError("direct-oracle smoke has no active AO blocks to qualify")
    checks = [
        {
            "metric": "active_total.mae_ry",
            "value": float(active["mae_ry"]),
            "operator": "<=",
            "threshold": float(thresholds["max_active_mae_ry"]),
            "pass": float(active["mae_ry"])
            <= float(thresholds["max_active_mae_ry"]),
        },
        {
            "metric": "active_total.rmse_ry",
            "value": float(active["rmse_ry"]),
            "operator": "<=",
            "threshold": float(thresholds["max_active_rmse_ry"]),
            "pass": float(active["rmse_ry"])
            <= float(thresholds["max_active_rmse_ry"]),
        },
        {
            "metric": "active_total.p99_abs_ry",
            "value": float(active["p99_abs_ry"]),
            "operator": "<=",
            "threshold": float(thresholds["max_active_p99_abs_ry"]),
            "pass": float(active["p99_abs_ry"])
            <= float(thresholds["max_active_p99_abs_ry"]),
        },
        {
            "metric": "active_total.max_abs_ry",
            "value": float(active["max_abs_ry"]),
            "operator": "<=",
            "threshold": float(thresholds["max_active_abs_ry"]),
            "pass": float(active["max_abs_ry"])
            <= float(thresholds["max_active_abs_ry"]),
        },
        {
            "metric": "active_total.r2",
            "value": float(active["r2"]),
            "operator": ">=",
            "threshold": float(thresholds["min_active_r2"]),
            "pass": math.isfinite(float(active["r2"]))
            and float(active["r2"]) >= float(thresholds["min_active_r2"]),
        },
        {
            "metric": "table_hermiticity.max_abs_ry",
            "value": float(metrics["table_hermiticity"]["max_abs_ry"]),
            "operator": "<=",
            "threshold": float(thresholds["max_table_hermiticity_ry"]),
            "pass": float(metrics["table_hermiticity"]["max_abs_ry"])
            <= float(thresholds["max_table_hermiticity_ry"]),
        },
        {
            "metric": "table_hermiticity.reverse_closed",
            "value": bool(metrics["table_hermiticity"]["reverse_closed"]),
            "operator": "==",
            "threshold": True,
            "pass": bool(metrics["table_hermiticity"]["reverse_closed"]),
        },
        {
            "metric": "table_hermiticity.nonzero_paired_r_key_pairs",
            "value": int(
                metrics["table_hermiticity"]["nonzero_paired_r_key_pairs"]
            ),
            "operator": ">=",
            "threshold": 1,
            "pass": int(
                metrics["table_hermiticity"]["nonzero_paired_r_key_pairs"]
            )
            >= 1,
        },
        {
            "metric": "oracle_hermiticity.max_abs_ry",
            "value": float(metrics["oracle_hermiticity"]["max_abs_ry"]),
            "operator": "<=",
            "threshold": float(thresholds["max_oracle_hermiticity_ry"]),
            "pass": float(metrics["oracle_hermiticity"]["max_abs_ry"])
            <= float(thresholds["max_oracle_hermiticity_ry"]),
        },
        {
            "metric": "oracle_hermiticity.reverse_closed",
            "value": bool(metrics["oracle_hermiticity"]["reverse_closed"]),
            "operator": "==",
            "threshold": True,
            "pass": bool(metrics["oracle_hermiticity"]["reverse_closed"]),
        },
        {
            "metric": "oracle_hermiticity.nonzero_paired_r_key_pairs",
            "value": int(
                metrics["oracle_hermiticity"]["nonzero_paired_r_key_pairs"]
            ),
            "operator": ">=",
            "threshold": 1,
            "pass": int(
                metrics["oracle_hermiticity"]["nonzero_paired_r_key_pairs"]
            )
            >= 1,
        },
    ]
    passed = all(bool(check["pass"]) for check in checks)
    if not enabled:
        return {
            "profile": "disabled_explicit_non_production",
            "enabled": False,
            "production_eligible": False,
            "status": "not_run",
            "thresholds": thresholds,
            "checks": checks,
        }
    return {
        "profile": "production_default",
        "enabled": True,
        "production_eligible": passed,
        "status": "pass" if passed else "fail",
        "thresholds": thresholds,
        "checks": checks,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gate1 = _load_gate1(args.gate1_script.resolve())
    parsed = gate1.parse_stru(args.case.resolve() / "STRU")
    structure = parsed.structure
    symbols = [atom.species for atom in structure.atoms]
    positions = np.asarray(structure.cart_positions, dtype=np.float64)
    cell = np.asarray(structure.cell_bohr, dtype=np.float64)
    with np.load(args.oracle_npz.resolve(), allow_pickle=False) as payload:
        r_keys_raw = np.asarray(payload["r_keys"])
        oracle_raw = np.asarray(payload["P2"])
    if r_keys_raw.ndim != 2 or r_keys_raw.shape[1] != 3:
        raise ValueError(f"oracle r_keys must be [nR,3], got {r_keys_raw.shape}")
    if not np.isfinite(r_keys_raw).all():
        raise ValueError("oracle r_keys contains NaN or infinity")
    r_keys = r_keys_raw.astype(np.int64)
    if not np.array_equal(r_keys_raw, r_keys):
        raise ValueError("oracle r_keys must contain exact integer translations")
    if len({tuple(row) for row in r_keys.tolist()}) != len(r_keys):
        raise ValueError("oracle r_keys contains duplicate translations")
    oracle_imag = (
        float(np.max(np.abs(oracle_raw.imag), initial=0.0))
        if np.iscomplexobj(oracle_raw)
        else 0.0
    )
    if oracle_imag > float(args.imag_tolerance):
        raise ValueError(
            f"non-SOC direct oracle has imaginary part {oracle_imag:.3e}"
        )
    oracle = np.asarray(oracle_raw.real, dtype=np.float64)

    cold_started = perf_counter()
    table_root = args.table_root.resolve()
    store = P2TableStore(table_root, require_explicit_source_identity=True)
    _require_qualified_table_manifest(
        store.manifest,
        manifest_path=table_root / "manifest.json",
    )
    declared_pp_orb = store.manifest.get("source", {}).get("pp_orb")
    pp_orb_candidate = args.pp_orb
    if pp_orb_candidate is None and declared_pp_orb:
        pp_orb_candidate = Path(str(declared_pp_orb))
    if pp_orb_candidate is None or not pp_orb_candidate.resolve().is_dir():
        raise ValueError(
            "direct-oracle production smoke must resolve the case ORB/UPF bundle; "
            "pass --pp-orb explicitly"
        )
    pp_orb_root = pp_orb_candidate.resolve()
    source_files: dict[str, dict[str, Path]] = {}
    shells_by_symbol: dict[str, tuple[int, ...]] = {}
    for symbol in sorted(set(symbols)):
        if symbol not in parsed.specs:
            raise KeyError(f"case STRU has no ORB/UPF specification for {symbol!r}")
        source_spec = parsed.specs[symbol]
        orbital_path = (pp_orb_root / str(source_spec.orbital)).resolve()
        upf_path = (pp_orb_root / str(source_spec.pseudo)).resolve()
        source_files[symbol] = {"orbital": orbital_path, "upf": upf_path}
        orbital = gate1.read_abacus_orb(orbital_path)
        shells_by_symbol[symbol] = tuple(
            int(channel.l) for channel in orbital.channels
        )
    sample_contract = store.validate_sample_contract(
        symbols=symbols,
        atom_shells=[shells_by_symbol[symbol] for symbol in symbols],
        source_files=source_files,
        require_source_files=True,
    )
    assembler = P2TableAssembler(store)
    table_p2 = assembler.assemble_dense_rkeys(
        symbols=symbols,
        positions_bohr=positions,
        cell_bohr=cell,
        r_keys=r_keys,
    )
    cold_batch_stats = assembler.batch_stats_snapshot()
    cold_seconds = perf_counter() - cold_started
    warm_started = perf_counter()
    table_p2_warm = assembler.assemble_dense_rkeys(
        symbols=symbols,
        positions_bohr=positions,
        cell_bohr=cell,
        r_keys=r_keys,
    )
    warm_batch_stats = assembler.batch_stats_snapshot()
    warm_seconds = perf_counter() - warm_started
    warm_repeat_error = float(
        np.max(np.abs(table_p2_warm - table_p2), initial=0.0)
    )
    if warm_repeat_error != 0.0:
        raise ValueError(f"table assembly is not deterministic: {warm_repeat_error}")
    if oracle.shape != table_p2.shape:
        raise ValueError(f"table P2 shape {table_p2.shape} != oracle {oracle.shape}")

    norb = [int(store.species[symbol]["orbital_norb"]) for symbol in symbols]
    direct_seconds = None
    if args.oracle_state is not None:
        state = json.loads(args.oracle_state.read_text(encoding="utf-8"))
        direct_seconds = float(state["wall_seconds"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "case": args.case.name,
        "symbols": symbols,
        "dimension": int(table_p2.shape[-1]),
        "r_key_count": int(len(r_keys)),
        "oracle": {
            "path": str(args.oracle_npz.resolve()),
            "sha256": _sha256(args.oracle_npz.resolve()),
            "max_imag_ry": oracle_imag,
            "direct_wall_seconds": direct_seconds,
        },
        "table": {
            "root": str(args.table_root.resolve()),
            "manifest_sha256": _sha256(args.table_root.resolve() / "manifest.json"),
            "pp_orb": str(pp_orb_root),
            "sample_contract": sample_contract,
            "batch_stats": {
                "cold": cold_batch_stats,
                "warm": warm_batch_stats,
            },
        },
        "metrics": {
            "dense_all": _metrics(oracle, table_p2),
            "blocks": _block_metrics(oracle, table_p2, r_keys, norb),
            "oracle_hermiticity": _hermiticity(oracle, r_keys),
            "table_hermiticity": _hermiticity(table_p2, r_keys),
        },
        "timing": {
            "cold_seconds": cold_seconds,
            "warm_seconds": warm_seconds,
            "warm_repeat_max_abs_ry": warm_repeat_error,
            "direct_to_warm_speedup": direct_seconds / warm_seconds
            if direct_seconds is not None
            else None,
        },
    }
    thresholds = {
        "max_active_mae_ry": float(args.max_active_mae_ry),
        "max_active_rmse_ry": float(args.max_active_rmse_ry),
        "max_active_p99_abs_ry": float(args.max_active_p99_abs_ry),
        "max_active_abs_ry": float(args.max_active_abs_ry),
        "min_active_r2": float(args.min_active_r2),
        "max_table_hermiticity_ry": float(args.max_table_hermiticity_ry),
        "max_oracle_hermiticity_ry": float(args.max_oracle_hermiticity_ry),
    }
    report["numerical_gate"] = _numerical_gate(
        report["metrics"],
        enabled=not bool(args.disable_numerical_gate),
        thresholds=thresholds,
    )
    _atomic_npz(
        args.output_npz.resolve(),
        r_keys=r_keys,
        P2=table_p2.astype(np.float32),
        oracle_sha256=np.asarray(report["oracle"]["sha256"]),
        table_manifest_sha256=np.asarray(report["table"]["manifest_sha256"]),
    )
    report["output_npz"] = str(args.output_npz.resolve())
    report["output_npz_sha256"] = _sha256(args.output_npz.resolve())
    _atomic_json(args.output_json.resolve(), report)
    if report["numerical_gate"]["enabled"] and report["numerical_gate"]["status"] != "pass":
        failed = [
            check for check in report["numerical_gate"]["checks"] if not check["pass"]
        ]
        raise ValueError(
            "P2 table direct-oracle numerical gate failed: "
            + json.dumps(failed, sort_keys=True)
        )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--oracle-npz", type=Path, required=True)
    parser.add_argument("--oracle-state", type=Path)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument(
        "--pp-orb",
        type=Path,
        help=(
            "ORB/UPF root selected by the case STRU; the table builder path is "
            "used only when it still exists"
        ),
    )
    parser.add_argument("--gate1-script", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--imag-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--max-active-mae-ry", type=float, default=DEFAULT_MAX_ACTIVE_MAE_RY
    )
    parser.add_argument(
        "--max-active-rmse-ry", type=float, default=DEFAULT_MAX_ACTIVE_RMSE_RY
    )
    parser.add_argument(
        "--max-active-p99-abs-ry",
        type=float,
        default=DEFAULT_MAX_ACTIVE_P99_ABS_RY,
    )
    parser.add_argument(
        "--max-active-abs-ry", type=float, default=DEFAULT_MAX_ACTIVE_ABS_RY
    )
    parser.add_argument("--min-active-r2", type=float, default=DEFAULT_MIN_ACTIVE_R2)
    parser.add_argument(
        "--max-table-hermiticity-ry",
        type=float,
        default=DEFAULT_MAX_TABLE_HERMITICITY_RY,
    )
    parser.add_argument(
        "--max-oracle-hermiticity-ry",
        type=float,
        default=DEFAULT_MAX_ORACLE_HERMITICITY_RY,
    )
    parser.add_argument(
        "--disable-numerical-gate",
        action="store_true",
        help=(
            "explicit diagnostic-only compatibility mode; the report is marked "
            "non-production and must not qualify a table"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "case": report["case"],
                "active_total": report["metrics"]["blocks"]["active_total"],
                "numerical_gate": report["numerical_gate"],
                "timing": report["timing"],
                "output_npz": report["output_npz"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

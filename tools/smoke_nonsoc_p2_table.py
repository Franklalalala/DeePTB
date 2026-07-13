#!/usr/bin/env python3
"""Compare fast radial-table P2 assembly with a preserved direct P2 oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from dptb.data.interfaces.p2_table import P2TableAssembler, P2TableStore


SCHEMA = "deeptb.p2_table_oracle_smoke/v1"


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
    for key, index in key_to_index.items():
        opposite = tuple(-x for x in key)
        if opposite not in key_to_index:
            continue
        error = np.asarray(array[index]) - np.asarray(array[key_to_index[opposite]]).T
        worst = max(worst, float(np.max(np.abs(error), initial=0.0)))
        compared += 1
    return {"paired_r_keys": compared, "max_abs_ry": worst}


def run(args: argparse.Namespace) -> dict[str, Any]:
    gate1 = _load_gate1(args.gate1_script.resolve())
    parsed = gate1.parse_stru(args.case.resolve() / "STRU")
    structure = parsed.structure
    symbols = [atom.species for atom in structure.atoms]
    positions = np.asarray(structure.cart_positions, dtype=np.float64)
    cell = np.asarray(structure.cell_bohr, dtype=np.float64)
    with np.load(args.oracle_npz.resolve(), allow_pickle=False) as payload:
        r_keys = np.asarray(payload["r_keys"], dtype=np.int64)
        oracle_raw = np.asarray(payload["P2"])
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
    store = P2TableStore(args.table_root.resolve())
    assembler = P2TableAssembler(store)
    table_p2 = assembler.assemble_dense_rkeys(
        symbols=symbols,
        positions_bohr=positions,
        cell_bohr=cell,
        r_keys=r_keys,
    )
    cold_seconds = perf_counter() - cold_started
    warm_started = perf_counter()
    table_p2_warm = assembler.assemble_dense_rkeys(
        symbols=symbols,
        positions_bohr=positions,
        cell_bohr=cell,
        r_keys=r_keys,
    )
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
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--oracle-npz", type=Path, required=True)
    parser.add_argument("--oracle-state", type=Path)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--gate1-script", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--imag-tolerance", type=float, default=1.0e-10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "case": report["case"],
                "active_total": report["metrics"]["blocks"]["active_total"],
                "timing": report["timing"],
                "output_npz": report["output_npz"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

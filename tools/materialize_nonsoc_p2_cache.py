#!/usr/bin/env python3
"""Build paired Full-H/P2 and H0->dH cache views for non-SOC ABACUS data.

The authoritative source remains the raw ABACUS H/H0 CSR quartet.  P2 is read
from the direct-oracle ``r_keys/P2`` NPZ artifact, rotated from ABACUS AO order
to DeePTB order, converted from Ry to eV, and then passed through DeePTB's own
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
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import lmdb
import numpy as np
import torch

import dptb.data.AtomicDataDict as AtomicDataDict
from dptb.data.interfaces.abacus import OrbAbacus2DeepTB, _abacus_parse
from dptb.data.interfaces.h0_lmdb_helper import _build_context_dataset


RY_TO_EV = 13.605698
SCHEMA = "deeptb.nonsoc_p2_cache/v1"
MAP_SIZE = 1 << 40

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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


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
    p2_root: Path,
    work_root: Path,
    split: str,
) -> list[dict[str, Any]]:
    env = _open_write(raw_split / "data.0000.lmdb")
    rows: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases):
            p2_path = p2_root / f"{case.name}.npz"
            if not p2_path.is_file():
                raise FileNotFoundError(p2_path)
            _abacus_parse(
                str(case),
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
                with np.load(p2_path, allow_pickle=False) as payload:
                    if "r_keys" not in payload or "P2" not in payload:
                        raise KeyError(f"{p2_path} must contain r_keys and P2.")
                    p2_blocks = dense_p2_to_deeptb_blocks(
                        r_keys=payload["r_keys"],
                        p2=payload["P2"],
                        basis_lines=record["basis"],
                        atom_count=len(record[AtomicDataDict.ATOMIC_NUMBERS_KEY]),
                    )
                    p2_r_keys = {tuple(map(int, value)) for value in payload["r_keys"]}
                for source_key in ("hamiltonian", "hamiltonian_0"):
                    source_r_keys = {
                        tuple(int(value) for value in key_string.split("_")[-3:])
                        for key_string in record[source_key]
                    }
                    if not source_r_keys.issubset(p2_r_keys):
                        missing = sorted(source_r_keys - p2_r_keys)
                        raise ValueError(
                            f"{case.name}: {source_key} has R keys absent from P2: {missing[:5]}"
                        )
                record["hamiltonian_p2"] = p2_blocks
                record["case_id"] = case.name
                record["source"] = str(case)
                record["p2_cache_source"] = {
                    "schema": SCHEMA,
                    "npz": str(p2_path),
                    "npz_sha256": _sha256(p2_path),
                    "energy_unit": "eV",
                    "ao_order": "DeePTB",
                }
                txn.put(key, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
            row = {
                "index": index,
                "case_id": case.name,
                "source": str(case),
                "p2_npz": str(p2_path),
                "p2_npz_sha256": record["p2_cache_source"]["npz_sha256"],
                "p2_blocks": len(p2_blocks),
            }
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
        "".join(f"{case}\n" for case in cases), encoding="utf-8"
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


def _base_record(raw: dict[str, Any], data: Any, index: int) -> dict[str, Any]:
    record = {field: np.asarray(raw[field]) for field in _BASE_FIELDS}
    for field in _GRAPH_FIELDS:
        record[field] = _numpy(data[field])
    record.update(
        {
            "idx": index,
            "case_id": raw["case_id"],
            "source": raw["source"],
            "p2_cache_source": raw["p2_cache_source"],
        }
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
        *_TARGET_FIELDS,
        *_H0_BLOCK_FIELDS,
        *_P2_BLOCK_FIELDS,
        *_GRAPH_FIELDS,
    )
    feature_widths: set[int] = set()
    max_identity_error = 0.0
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

            full = _base_record(raw, data, index)
            for field in (
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.EDGE_FEATURES_KEY,
                AtomicDataDict.NODE_H0_KEY,
                AtomicDataDict.EDGE_H0_KEY,
                AtomicDataDict.NODE_P2_KEY,
                AtomicDataDict.EDGE_P2_KEY,
                *_TARGET_FIELDS,
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
                *_TARGET_FIELDS[:2],
                *_H0_BLOCK_FIELDS[:2],
                *_P2_BLOCK_FIELDS[:2],
            ):
                if not np.isfinite(full[field]).all():
                    raise ValueError(f"{raw['case_id']}: non-finite values in {field}.")
            delta = _base_record(raw, data, index)
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

            full_target_node, full_target_edge, node_shape, edge_shape = _TARGET_FIELDS
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
            delta[full_target_node] = (full[full_target_node] - full[h0_node]).astype(
                np.float32, copy=False
            )
            delta[full_target_edge] = (full[full_target_edge] - full[h0_edge]).astype(
                np.float32, copy=False
            )
            delta[node_shape] = full[node_shape]
            delta[edge_shape] = full[edge_shape]

            node_error = np.max(
                np.abs(delta[full_target_node] + full[h0_node] - full[full_target_node]),
                initial=0.0,
            )
            edge_error = np.max(
                np.abs(delta[full_target_edge] + full[h0_edge] - full[full_target_edge]),
                initial=0.0,
            )
            max_identity_error = max(max_identity_error, float(node_error), float(edge_error))
            if max_identity_error > 5.0e-6:
                raise ValueError(
                    f"{raw['case_id']}: cached H0+dH identity error {max_identity_error:.3e}."
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


def _guard_output(dataset_root: Path, work_root: Path, overwrite: bool) -> None:
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
    if work_root.exists():
        if not overwrite:
            raise FileExistsError(work_root)
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--p2-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--valid-count", type=int, default=20)
    parser.add_argument("--split-seed", default="nonsoc-p2-raw200-v1")
    parser.add_argument("--require-count", type=int, default=0)
    parser.add_argument(
        "--keep-raw-staging",
        action="store_true",
        help="retain the large intermediate raw H/H0/P2 block LMDB after success",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    p2_root = args.p2_root.resolve()
    work_root = args.work_root.resolve()
    input_json = args.input_json.resolve()
    _guard_output(dataset_root, work_root, args.overwrite)
    cases = _ordered_cases(dataset_root, args.case)
    if args.require_count and len(cases) != args.require_count:
        raise ValueError(f"Selected {len(cases)} cases; required {args.require_count}.")
    split_cases = _split_cases(cases, valid_count=args.valid_count, seed=args.split_seed)
    for case in cases:
        if not (p2_root / f"{case.name}.npz").is_file():
            raise FileNotFoundError(p2_root / f"{case.name}.npz")

    torch.set_default_dtype(torch.float32)
    raw_root = work_root / "raw_staging"
    full_root = work_root / "full_h"
    delta_root = work_root / "h0_delta"
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "dataset_root": str(dataset_root),
        "p2_root": str(p2_root),
        "input_json": str(input_json),
        "input_json_sha256": _sha256(input_json),
        "script_sha256": _sha256(Path(__file__)),
        "split_seed": args.split_seed,
        "splits": {},
    }
    for split, selected in split_cases.items():
        if not selected:
            continue
        raw_rows = _raw_split(
            cases=selected,
            raw_split=raw_root / split,
            p2_root=p2_root,
            work_root=work_root,
            split=split,
        )
        stats = _materialize_split(
            raw_split=raw_root / split,
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

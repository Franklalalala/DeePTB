#!/usr/bin/env python3
"""Materialize validated h0rebuild artifacts into a DeePTB LMDB tree.

Manifest schema::

    {
      "schema": "deeptb.h0rebuild_manifest/v1",
      "data_length_unit": "angstrom",
      "target_energy_unit": "eV",
      "records": [
        {"lmdb": "data.0.lmdb", "key": 0, "artifact": "/abs/0.npz"}
      ]
    }

Every LMDB entry must have exactly one manifest row.  The output stores the
active edge graph together with ``node_physical_h0``/``edge_physical_h0`` and an
order-sensitive sidecar hash contract.
"""
from __future__ import annotations

import argparse
import copy
import json
import os.path as osp
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

import dptb.data.AtomicDataDict as AtomicDataDict
from dptb.data.AtomicData import AtomicData
from dptb.data.interfaces.h0_lmdb_helper import (
    _build_context_dataset,
    _copy_non_lmdb_tree,
    _iter_lmdb_dirs,
    _open_read_env,
    _open_write_env,
    _relative_lmdb_dir,
    path_size_bytes,
)
from dptb.data.interfaces.h0rebuild_adapter import (
    PHYSICAL_H0_META_KEY,
    build_physical_h0_meta,
    materialize_h0rebuild_features,
)


MANIFEST_SCHEMA = "deeptb.h0rebuild_manifest/v1"
_ATOMICDATA_CONSTRUCTOR_OPTIONS = {"r_max", "er_max", "oer_max", "self_interaction"}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validated_roots(input_root: str, output_root: str) -> tuple[Path, Path]:
    input_path = Path(input_root).resolve()
    output_path = Path(output_root).resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(input_path)
    if (
        input_path == output_path
        or input_path in output_path.parents
        or output_path in input_path.parents
    ):
        raise ValueError(
            "input_root and output_root must be separate, non-nested directories; "
            f"got input_root={input_path}, output_root={output_path}"
        )
    return input_path, output_path


def _manifest_index(manifest: dict[str, Any], manifest_path: Path) -> dict[tuple[str, int], Path]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported manifest schema {manifest.get('schema')!r}; expected {MANIFEST_SCHEMA!r}"
        )
    out: dict[tuple[str, int], Path] = {}
    for row in manifest.get("records", []):
        rel = osp.normpath(str(row["lmdb"]))
        key = int(row["key"])
        artifact = Path(row["artifact"])
        if not artifact.is_absolute():
            artifact = (manifest_path.parent / artifact).resolve()
        token = (rel, key)
        if token in out:
            raise ValueError(f"Duplicate manifest row for {token}")
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        out[token] = artifact
    if not out:
        raise ValueError("Manifest contains no records")
    return out


def _record_atomicdata(record: dict[str, Any], info: dict[str, Any]) -> AtomicData:
    pos = np.asarray(record[AtomicDataDict.POSITIONS_KEY]).reshape(-1, 3)
    cell = np.asarray(record[AtomicDataDict.CELL_KEY]).reshape(3, 3)
    atomic_numbers = np.asarray(record[AtomicDataDict.ATOMIC_NUMBERS_KEY]).reshape(-1)
    pbc = np.asarray(record[AtomicDataDict.PBC_KEY])
    edge_index = record.get(AtomicDataDict.EDGE_INDEX_KEY)
    edge_shift = record.get(AtomicDataDict.EDGE_CELL_SHIFT_KEY)
    if (edge_index is None) != (edge_shift is None):
        raise ValueError("Stored edge_index and edge_cell_shift must appear together")
    if edge_index is None:
        return AtomicData.from_points(
            pos=pos,
            cell=cell,
            atomic_numbers=atomic_numbers,
            pbc=pbc,
            **info,
        )
    kwargs = {
        key: value
        for key, value in info.items()
        if key not in _ATOMICDATA_CONSTRUCTOR_OPTIONS
    }
    kwargs[AtomicDataDict.EDGE_INDEX_KEY] = torch.as_tensor(edge_index, dtype=torch.long)
    kwargs[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = torch.as_tensor(
        edge_shift, dtype=torch.get_default_dtype()
    )
    return AtomicData(
        pos=torch.as_tensor(pos, dtype=torch.get_default_dtype()),
        cell=torch.as_tensor(cell, dtype=torch.get_default_dtype()),
        atomic_numbers=torch.as_tensor(atomic_numbers, dtype=torch.long),
        pbc=torch.as_tensor(pbc, dtype=torch.bool),
        **kwargs,
    )


def materialize_root(
    *,
    input_root: str,
    output_root: str,
    input_json: str,
    manifest_path: str,
    overwrite: bool = False,
    feature_dtype: str = "float64",
) -> dict[str, Any]:
    input_path, output = _validated_roots(input_root, output_root)
    input_root = str(input_path)
    output_root = str(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    manifest_file = Path(manifest_path).resolve()
    manifest = _load_json(manifest_file)
    index = _manifest_index(manifest, manifest_file)
    data_length_unit = str(manifest.get("data_length_unit", ""))
    if not data_length_unit:
        raise ValueError("Manifest must define data_length_unit explicitly")
    target_energy_unit = str(manifest.get("target_energy_unit", "eV"))
    dtype = {"float32": torch.float32, "float64": torch.float64}[feature_dtype]

    dataset, _ = _build_context_dataset(input_json=input_json, root_override=input_root)
    _copy_non_lmdb_tree(input_root, output_root)
    seen: set[tuple[str, int]] = set()
    total = 0
    shards: list[dict[str, Any]] = []
    for lmdb_dir in _iter_lmdb_dirs(input_root):
        rel = osp.normpath(_relative_lmdb_dir(input_root, lmdb_dir))
        out_lmdb_dir = osp.join(output_root, rel)
        map_size = max(1 << 30, int(path_size_bytes(lmdb_dir) * 5))
        in_env = _open_read_env(lmdb_dir)
        out_env = _open_write_env(out_lmdb_dir, map_size=map_size)
        folder_name = osp.basename(lmdb_dir)
        info = copy.deepcopy(dataset.info_files[folder_name])
        shard_count = 0
        with in_env.begin() as in_txn, out_env.begin(write=True) as out_txn:
            for raw_key, raw_value in in_txn.cursor():
                key_int = int.from_bytes(raw_key, byteorder="big")
                token = (rel, key_int)
                artifact = index.get(token)
                if artifact is None:
                    raise KeyError(f"Manifest lacks artifact for LMDB entry {token}")
                record = pickle.loads(raw_value)
                for field in (
                    AtomicDataDict.NODE_PHYSICAL_H0_KEY,
                    AtomicDataDict.EDGE_PHYSICAL_H0_KEY,
                    PHYSICAL_H0_META_KEY,
                ):
                    if field in record:
                        raise KeyError(
                            f"{token}: destination field {field!r} already exists; "
                            "materialization is intentionally non-overwriting"
                        )
                atomicdata = _record_atomicdata(record, info)
                atomicdata, report = materialize_h0rebuild_features(
                    atomicdata,
                    dataset.type_mapper,
                    artifact,
                    target_energy_unit=target_energy_unit,
                    data_length_unit=data_length_unit,
                    orthogonal=dataset.orthogonal,
                    feature_output_dtype=dtype,
                )
                node = atomicdata[AtomicDataDict.NODE_PHYSICAL_H0_KEY].detach().cpu().numpy()
                edge = atomicdata[AtomicDataDict.EDGE_PHYSICAL_H0_KEY].detach().cpu().numpy()
                record[AtomicDataDict.NODE_PHYSICAL_H0_KEY] = node
                record[AtomicDataDict.EDGE_PHYSICAL_H0_KEY] = edge
                # Persist exactly the graph row order used for feature packing.
                record[AtomicDataDict.EDGE_INDEX_KEY] = (
                    atomicdata[AtomicDataDict.EDGE_INDEX_KEY].detach().cpu().numpy()
                )
                record[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = (
                    atomicdata[AtomicDataDict.EDGE_CELL_SHIFT_KEY].detach().cpu().numpy()
                )
                record[PHYSICAL_H0_META_KEY] = build_physical_h0_meta(
                    record,
                    energy_unit=target_energy_unit,
                    source=report,
                )
                out_txn.put(raw_key, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                seen.add(token)
                total += 1
                shard_count += 1
        in_env.close()
        out_env.close()
        shards.append({"lmdb": rel, "entries": shard_count})

    unused = sorted(set(index) - seen)
    if unused:
        raise ValueError(f"Manifest has {len(unused)} rows not found in the input LMDB: {unused[:5]}")
    summary = {
        "schema": "deeptb.h0rebuild_materialization/v1",
        "input_root": str(Path(input_root).resolve()),
        "output_root": str(output.resolve()),
        "input_json": str(Path(input_json).resolve()),
        "manifest": str(manifest_file),
        "data_length_unit": data_length_unit,
        "target_energy_unit": target_energy_unit,
        "feature_dtype": feature_dtype,
        "entries": total,
        "shards": shards,
    }
    (output / "physical_h0_materialization.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = materialize_root(
        input_root=args.input_root,
        output_root=args.output_root,
        input_json=args.input_json,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
        feature_dtype=args.feature_dtype,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert precomputed DeePTB RME labels into padded spatial AO blocks.

The converter is deliberately explicit about target provenance.  For an
H0->delta-H run, choose either ``already-delta`` (the feature fields already
store delta-H) or ``subtract-h0`` (form delta-H from feature-H minus feature-H0).

Reduced SOC is accepted only for the single directed real ``uu`` contract.
Full spinor SOC remains unsupported by the block-wise spatial tensor schema.
Every record is checked by an exact feature -> block -> feature round trip on
the mapper-valid slots before it is committed to the output LMDB.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Iterable

import lmdb
import numpy as np
import torch

from dptb.data.interfaces.blockwise_tensor import (
    EDGE_DELTA_HAMIL_BLOCKS_KEY,
    EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    EDGE_H0_BLOCKS_KEY,
    EDGE_H0_BLOCK_SHAPE_KEY,
    NODE_DELTA_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    NODE_H0_BLOCKS_KEY,
    NODE_H0_BLOCK_SHAPE_KEY,
    atom_types_from_data,
    block_tensors_to_feature_tensors,
    edge_types_from_data,
    ensure_spatial_block_mapper,
    feature_tensors_to_block_tensors,
    is_soc_uureal_mapper,
)
from dptb.data.transforms import OrbitalMapper


SCHEMA = "deeptb.blockwise_spatial/v1"
TARGET_NODE_KEY = "node_features"
TARGET_EDGE_KEY = "edge_features"
H0_NODE_KEY = "node_h0"
H0_EDGE_KEY = "edge_h0"


def _validated_roots(input_root: str | Path, output_root: str | Path) -> tuple[Path, Path]:
    source = Path(input_root).resolve()
    destination = Path(output_root).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError(
            "input_root and output_root must be separate, non-nested directories; "
            f"got input_root={source}, output_root={destination}"
        )
    return source, destination


def _iter_lmdb_dirs(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        path = Path(dirpath)
        if path.name.endswith(".lmdb"):
            found.append(path)
            dirnames[:] = []
    return sorted(found)


def _copy_non_lmdb_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(source):
        src_dir = Path(dirpath)
        rel = src_dir.relative_to(source)
        dst_dir = destination / rel
        dst_dir.mkdir(parents=True, exist_ok=True)
        dirnames[:] = [name for name in dirnames if not name.endswith(".lmdb")]
        for filename in filenames:
            shutil.copy2(src_dir / filename, dst_dir / filename)


def _path_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _mapper_from_input(input_json: str | Path) -> OrbitalMapper:
    config = json.loads(Path(input_json).read_text(encoding="utf-8"))
    common = config["common_options"]
    mapper = OrbitalMapper(
        basis=common["basis"],
        method="e3tb",
        device="cpu",
        has_soc=bool(common.get("has_soc", False)),
        soc_complex_doubling=bool(common.get("soc_complex_doubling", True)),
        nextham_uureal_mask=bool(common.get("nextham_uureal_mask", False)),
        full_soc_prediction=bool(common.get("full_soc_prediction", False)),
    )
    return ensure_spatial_block_mapper(mapper)


def _require_pair(record: dict[str, Any], node_key: str, edge_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    present = (node_key in record, edge_key in record)
    if any(present) and not all(present):
        raise KeyError(f"{node_key!r} and {edge_key!r} must appear together")
    if not all(present):
        raise KeyError(f"record lacks required fields {node_key!r}/{edge_key!r}")
    return torch.as_tensor(record[node_key]), torch.as_tensor(record[edge_key])


def _target_features(
    record: dict[str, Any], target_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    node, edge = _require_pair(record, TARGET_NODE_KEY, TARGET_EDGE_KEY)
    semantics = str(record.get("hamiltonian_semantics", "")).lower()
    if target_mode == "already-delta":
        if semantics and "delta" not in semantics and "h-h0" not in semantics:
            raise ValueError(
                "--target-mode already-delta conflicts with record "
                f"hamiltonian_semantics={semantics!r}"
            )
        return node, edge
    if target_mode == "subtract-h0":
        if "delta" in semantics or "h-h0" in semantics:
            raise ValueError(
                "--target-mode subtract-h0 would double-subtract a record marked "
                f"as residual: hamiltonian_semantics={semantics!r}"
            )
        node_h0, edge_h0 = _require_pair(record, H0_NODE_KEY, H0_EDGE_KEY)
        if node.shape != node_h0.shape or edge.shape != edge_h0.shape:
            raise ValueError(
                "Cannot form delta-H: H/H0 feature shapes differ: "
                f"node={tuple(node.shape)}/{tuple(node_h0.shape)}, "
                f"edge={tuple(edge.shape)}/{tuple(edge_h0.shape)}"
            )
        return node - node_h0.to(dtype=node.dtype), edge - edge_h0.to(dtype=edge.dtype)
    raise ValueError(f"unknown target_mode={target_mode!r}")


def _check_feature_width(node: torch.Tensor, edge: torch.Tensor, mapper: OrbitalMapper) -> None:
    width = int(mapper.reduced_matrix_element)
    if node.ndim != 2 or edge.ndim != 2 or node.shape[1] != width or edge.shape[1] != width:
        raise ValueError(
            f"Feature tensors must be [N,{width}]/[E,{width}], got "
            f"{tuple(node.shape)}/{tuple(edge.shape)}"
        )


def _full_soc_mapper_for(reduced: OrbitalMapper) -> OrbitalMapper:
    return OrbitalMapper(
        basis=reduced.basis,
        chemical_symbol_to_type=reduced.chemical_symbol_to_type,
        method="e3tb",
        device="cpu",
        has_soc=True,
        soc_complex_doubling=bool(reduced.soc_complex_doubling),
        nextham_uureal_mask=False,
        full_soc_prediction=True,
    )


def _project_full_soc_to_uureal(
    tensor: torch.Tensor,
    reduced: OrbitalMapper,
    full: OrbitalMapper,
) -> torch.Tensor:
    output = tensor.new_zeros((*tensor.shape[:-1], int(reduced.reduced_matrix_element)))
    for pair, reduced_slice in reduced.orbpair_maps.items():
        full_slice = full.orbpair_maps.get(pair)
        if full_slice is None:
            raise RuntimeError(f"Full SOC mapper lacks orbpair slice {pair!r}")
        spatial_width = int(reduced_slice.stop - reduced_slice.start)
        full_width = int(full_slice.stop - full_slice.start)
        expected_full_width = spatial_width * full._e3tb_soc_feature_factor()
        if full_width != expected_full_width:
            raise RuntimeError(
                f"Unexpected full SOC layout for {pair}: width={full_width}, "
                f"expected={expected_full_width}"
            )
        # Full SOC feature layout is channel-major within every orbpair slice;
        # the first spatial_width entries are uu_real.
        output[..., reduced_slice] = tensor[
            ..., full_slice.start : full_slice.start + spatial_width
        ]
    return output


def _coerce_uureal_width(
    record: dict[str, Any],
    node: torch.Tensor,
    edge: torch.Tensor,
    mapper: OrbitalMapper,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if node.ndim != 2 or edge.ndim != 2 or node.shape[1] != edge.shape[1]:
        raise ValueError(
            "Node/edge feature fields must be rank-2 with the same width; got "
            f"{tuple(node.shape)}/{tuple(edge.shape)}"
        )
    source_width = int(node.shape[1])
    reduced_width = int(mapper.reduced_matrix_element)
    if source_width == reduced_width:
        return node, edge, source_width
    if not is_soc_uureal_mapper(mapper):
        raise ValueError(
            f"Input feature width is {source_width}, mapper width is {reduced_width}; "
            "automatic full-SOC projection is allowed only for SOC uu_real"
        )

    channel_order = record.get("soc_real_channel_order", None)
    if channel_order is None:
        raise ValueError(
            "Full SOC source requires explicit soc_real_channel_order metadata; "
            "refusing to guess which 729-wide channel is uu_real"
        )
    order = [str(value) for value in np.asarray(channel_order).reshape(-1).tolist()]
    if not order or order[0].lower() not in {"uu_re", "uu_real"}:
        raise ValueError(
            "Full SOC source does not declare uu_real as its first channel: "
            f"soc_real_channel_order={order}"
        )
    declared_width = record.get("full_soc_feature_width", None)
    if declared_width is not None and int(np.asarray(declared_width).item()) != source_width:
        raise ValueError(
            "full_soc_feature_width metadata disagrees with tensors: "
            f"metadata={declared_width}, tensor={source_width}"
        )

    full_mapper = getattr(mapper, "_blockwise_full_soc_mapper", None)
    if full_mapper is None:
        full_mapper = _full_soc_mapper_for(mapper)
        # Building the 5832-wide mapper walks all element/bond masks.  Cache it
        # once per conversion instead of repeating that work for every record
        # and again for its H0 fields.
        mapper._blockwise_full_soc_mapper = full_mapper
    if source_width != int(full_mapper.reduced_matrix_element):
        raise ValueError(
            f"Input width {source_width} is neither reduced uu_real width "
            f"{reduced_width} nor full SOC width {full_mapper.reduced_matrix_element}"
        )
    return (
        _project_full_soc_to_uureal(node, mapper, full_mapper),
        _project_full_soc_to_uureal(edge, mapper, full_mapper),
        source_width,
    )


def _roundtrip_check(
    record: dict[str, Any],
    mapper: OrbitalMapper,
    node_features: torch.Tensor,
    edge_features: torch.Tensor,
    node_blocks: torch.Tensor,
    edge_blocks: torch.Tensor,
) -> float:
    back_node, back_edge = block_tensors_to_feature_tensors(
        record,
        mapper,
        node_blocks=node_blocks,
        edge_blocks=edge_blocks,
    )
    atom_types = atom_types_from_data(record, mapper)
    edge_types = edge_types_from_data(record, mapper)
    node_mask = mapper.mask_to_nrme.index_select(0, atom_types.cpu())
    edge_mask = mapper.mask_to_erme.index_select(0, edge_types.cpu())
    node_diff = (back_node.cpu() - node_features.cpu()).abs()[node_mask]
    edge_diff = (back_edge.cpu() - edge_features.cpu()).abs()[edge_mask]
    maxima = [
        float(node_diff.max().item()) if node_diff.numel() else 0.0,
        float(edge_diff.max().item()) if edge_diff.numel() else 0.0,
    ]
    maximum = max(maxima)
    if maximum != 0.0:
        raise RuntimeError(
            "feature -> block -> feature round trip changed a mapper-valid slot; "
            f"max_abs_diff={maximum:.9g}"
        )
    return maximum


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _convert_record(
    record: dict[str, Any],
    mapper: OrbitalMapper,
    *,
    target_mode: str,
    include_h0_blocks: bool,
    replace_existing: bool,
) -> tuple[dict[str, Any], float]:
    required_graph = ("atomic_numbers", "edge_index", "edge_cell_shift")
    missing_graph = [key for key in required_graph if key not in record]
    if missing_graph:
        raise KeyError(
            f"record lacks stored active graph fields {missing_graph}; rebuild/materialize "
            "the feature LMDB with edge_index/edge_cell_shift before block conversion"
        )

    output_fields = [
        NODE_DELTA_HAMIL_BLOCKS_KEY,
        EDGE_DELTA_HAMIL_BLOCKS_KEY,
        NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
        EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    ]
    if include_h0_blocks:
        output_fields += [
            NODE_H0_BLOCKS_KEY,
            EDGE_H0_BLOCKS_KEY,
            NODE_H0_BLOCK_SHAPE_KEY,
            EDGE_H0_BLOCK_SHAPE_KEY,
        ]
    occupied = [key for key in output_fields if key in record]
    if occupied and not replace_existing:
        raise KeyError(f"destination block fields already exist: {occupied}")

    node_features, edge_features = _target_features(record, target_mode)
    node_features, edge_features, source_target_width = _coerce_uureal_width(
        record, node_features, edge_features, mapper
    )
    # Keep the conventional feature side channel loader-compatible, but compact
    # it to the same 729-wide contract instead of retaining a misleading
    # 5832-wide copy.  The current LMDB loader still uses this side channel when
    # get_Hamiltonian=True, while the block loss reads the AO-block fields.
    record[TARGET_NODE_KEY] = _numpy(node_features)
    record[TARGET_EDGE_KEY] = _numpy(edge_features)
    _check_feature_width(node_features, edge_features, mapper)
    packed = feature_tensors_to_block_tensors(
        record,
        mapper,
        node_features=node_features,
        edge_features=edge_features,
        complete_edges=True,
        strict_complete_edges=True,
    )
    maximum = _roundtrip_check(
        record,
        mapper,
        node_features,
        edge_features,
        packed.node_blocks,
        packed.edge_blocks,
    )
    record[NODE_DELTA_HAMIL_BLOCKS_KEY] = _numpy(packed.node_blocks)
    record[EDGE_DELTA_HAMIL_BLOCKS_KEY] = _numpy(packed.edge_blocks)
    record[NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = _numpy(packed.node_shapes)
    record[EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = _numpy(packed.edge_shapes)

    h0_present = (H0_NODE_KEY in record, H0_EDGE_KEY in record)
    if any(h0_present) and not all(h0_present):
        raise KeyError(f"{H0_NODE_KEY!r} and {H0_EDGE_KEY!r} must appear together")
    if all(h0_present):
        node_h0, edge_h0 = _require_pair(record, H0_NODE_KEY, H0_EDGE_KEY)
        node_h0, edge_h0, source_h0_width = _coerce_uureal_width(
            record, node_h0, edge_h0, mapper
        )
        _check_feature_width(node_h0, edge_h0, mapper)
        # H0 is an embedding input in the H0->delta-H production model, so it
        # must follow the reduced mapper even when H0 AO blocks are not needed.
        record[H0_NODE_KEY] = _numpy(node_h0)
        record[H0_EDGE_KEY] = _numpy(edge_h0)
    else:
        node_h0 = edge_h0 = None
        source_h0_width = None

    if include_h0_blocks:
        if node_h0 is None or edge_h0 is None:
            raise KeyError(
                f"--include-h0-blocks requires {H0_NODE_KEY!r}/{H0_EDGE_KEY!r}"
            )
        h0_packed = feature_tensors_to_block_tensors(
            record,
            mapper,
            node_features=node_h0,
            edge_features=edge_h0,
            complete_edges=True,
            strict_complete_edges=True,
        )
        maximum = max(
            maximum,
            _roundtrip_check(
                record,
                mapper,
                node_h0,
                edge_h0,
                h0_packed.node_blocks,
                h0_packed.edge_blocks,
            ),
        )
        record[NODE_H0_BLOCKS_KEY] = _numpy(h0_packed.node_blocks)
        record[EDGE_H0_BLOCKS_KEY] = _numpy(h0_packed.edge_blocks)
        record[NODE_H0_BLOCK_SHAPE_KEY] = _numpy(h0_packed.node_shapes)
        record[EDGE_H0_BLOCK_SHAPE_KEY] = _numpy(h0_packed.edge_shapes)

    record["blockwise_spatial_schema"] = SCHEMA
    record["blockwise_target_mode"] = target_mode
    record["blockwise_source_target_feature_width"] = source_target_width
    source_contract_width = max(source_target_width, source_h0_width or 0)
    if source_contract_width != int(mapper.reduced_matrix_element):
        record["soc_uureal_compact"] = True
        record["soc_uureal_full_rme"] = source_contract_width
        record["soc_uureal_keep"] = int(mapper.reduced_matrix_element)
        record["blockwise_source_soc_target_projection"] = record.get(
            "soc_target_projection", None
        )
        record["blockwise_source_nextham_uureal_mask"] = record.get(
            "nextham_uureal_mask", None
        )
        record["soc_target_projection"] = "uu_real"
        record["nextham_uureal_mask"] = True
    if source_h0_width is not None:
        record["blockwise_source_h0_feature_width"] = source_h0_width
    return record, maximum


def convert_root(
    *,
    input_root: str | Path,
    output_root: str | Path,
    input_json: str | Path,
    target_mode: str,
    include_h0_blocks: bool = False,
    overwrite: bool = False,
    replace_existing: bool = False,
) -> dict[str, Any]:
    source, destination = _validated_roots(input_root, output_root)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        shutil.rmtree(destination)
    mapper = _mapper_from_input(input_json)
    shards = _iter_lmdb_dirs(source)
    if not shards:
        raise FileNotFoundError(f"No *.lmdb directories found under {source}")
    _copy_non_lmdb_tree(source, destination)

    total = 0
    max_roundtrip = 0.0
    shard_rows: list[dict[str, Any]] = []
    for shard in shards:
        relative = shard.relative_to(source)
        output_shard = destination / relative
        output_shard.mkdir(parents=True, exist_ok=True)
        map_size = max(1 << 30, int(_path_size(shard) * 4))
        input_env = lmdb.open(
            str(shard), readonly=True, lock=False, readahead=False, max_readers=512
        )
        output_env = lmdb.open(
            str(output_shard), map_size=map_size, subdir=True, lock=True, max_dbs=1
        )
        shard_count = 0
        with input_env.begin() as input_txn, output_env.begin(write=True) as output_txn:
            for raw_key, raw_value in input_txn.cursor():
                record = pickle.loads(raw_value)
                record, roundtrip = _convert_record(
                    record,
                    mapper,
                    target_mode=target_mode,
                    include_h0_blocks=include_h0_blocks,
                    replace_existing=replace_existing,
                )
                output_txn.put(
                    raw_key,
                    pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
                )
                max_roundtrip = max(max_roundtrip, roundtrip)
                shard_count += 1
                total += 1
        input_env.close()
        output_env.close()
        shard_rows.append({"lmdb": str(relative), "entries": shard_count})

    summary = {
        "schema": SCHEMA,
        "input_root": str(source),
        "output_root": str(destination),
        "input_json": str(Path(input_json).resolve()),
        "has_soc": bool(mapper.has_soc),
        "nextham_uureal_mask": bool(mapper.nextham_uureal_mask),
        "full_soc_prediction": bool(mapper.full_soc_prediction),
        "full_basis_norb": int(mapper.full_basis_norb),
        "reduced_matrix_element": int(mapper.reduced_matrix_element),
        "target_mode": target_mode,
        "include_h0_blocks": bool(include_h0_blocks),
        "entries": total,
        "max_valid_roundtrip_abs_diff": max_roundtrip,
        "shards": shard_rows,
    }
    (destination / "blockwise_conversion.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument(
        "--target-mode",
        required=True,
        choices=("already-delta", "subtract-h0"),
        help="State explicitly whether node/edge_features already contain delta-H.",
    )
    parser.add_argument("--include-h0-blocks", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = convert_root(
        input_root=args.input_root,
        output_root=args.output_root,
        input_json=args.input_json,
        target_mode=args.target_mode,
        include_h0_blocks=args.include_h0_blocks,
        replace_existing=args.replace_existing,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

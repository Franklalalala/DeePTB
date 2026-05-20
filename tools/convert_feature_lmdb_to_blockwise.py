#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Convert a non-SOC DeePTB/NexTHam feature LMDB into block-wise LMDB.

Input sample schema is the current feature workflow:
    node_features / edge_features   -> delta-H feature labels
    node_h0 / edge_h0               -> H0 feature inputs
    edge_index / edge_cell_shift / atomic_numbers / ...

Output sample adds block tensors:
    node_delta_hamil_blocks / edge_delta_hamil_blocks
    node_h0_blocks / edge_h0_blocks
    corresponding *_block_shape fields

By default the original feature labels and feature H0 are dropped after strict
feature -> block -> feature roundtrip validation to reduce IO.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

try:
    import lmdb
except Exception as exc:  # pragma: no cover
    raise ImportError("python-lmdb is required in the DeePTB environment.") from exc

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from dptb.data.interfaces.ham_to_feature import block_to_feature, feature_to_block
from dptb.data.interfaces.blockwise_tensor import (
    EDGE_DELTA_HAMIL_BLOCKS_KEY,
    EDGE_H0_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCKS_KEY,
    NODE_H0_BLOCKS_KEY,
    attach_block_tensors,
    atom_types_from_data,
    block_dict_to_ordered_tensors,
    edge_types_from_data,
)
from dptb.data.transforms import OrbitalMapper

NODE_FEATURES_KEY = "node_features"
EDGE_FEATURES_KEY = "edge_features"
NODE_H0_KEY = "node_h0"
EDGE_H0_KEY = "edge_h0"


def load_json_or_yaml(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        if yaml is None:
            raise ImportError("PyYAML is required for YAML input configs.")
        return yaml.safe_load(text)
    return json.loads(text)


def find_basis(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        if isinstance(obj.get("basis"), dict):
            return obj["basis"]
        for value in obj.values():
            found = find_basis(value)
            if found is not None:
                return found
    return None


def load_basis(args) -> Dict[str, Any]:
    if args.basis_json:
        return json.loads(args.basis_json)
    if args.basis_file:
        obj = load_json_or_yaml(args.basis_file)
        return obj.get("basis", obj) if isinstance(obj, dict) else obj
    if args.input_config:
        basis = find_basis(load_json_or_yaml(args.input_config))
        if basis is not None:
            return basis
        raise ValueError(f"Could not find a basis dict in {args.input_config}")
    raise ValueError("Provide --input-config, --basis-file, or --basis-json.")


def build_mapper(args):
    if args.has_soc:
        raise ValueError("This converter is for the current non-SOC workflow; SOC is intentionally disabled.")
    basis = load_basis(args)
    kwargs = dict(method=args.mapper_method, device="cpu", has_soc=False)
    try:
        idp = OrbitalMapper(basis, nextham_uureal_mask=bool(args.nextham_uureal_mask), **kwargs)
    except TypeError:
        idp = OrbitalMapper(basis, **kwargs)
        if args.nextham_uureal_mask:
            setattr(idp, "nextham_uureal_mask", True)
    if hasattr(idp, "get_orbital_maps"):
        idp.get_orbital_maps()
    if hasattr(idp, "get_orbpair_maps"):
        idp.get_orbpair_maps()
    return idp


def find_lmdb_paths(input_root: str, split: Optional[str]) -> Iterable[Path]:
    root = Path(input_root)
    if root.is_dir() and root.suffix == ".lmdb":
        yield root
        return
    patterns = [str(root / split / "*.lmdb"), str(root / split / "**" / "*.lmdb")] if split else [str(root / "*.lmdb"), str(root / "**" / "*.lmdb")]
    seen = set()
    for pattern in patterns:
        for item in sorted(glob.glob(pattern, recursive=True)):
            path = Path(item)
            if path.is_dir() and path not in seen:
                seen.add(path)
                yield path


def mirror_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    try:
        rel = input_path.relative_to(input_root)
    except ValueError:
        rel = Path(input_path.name)
    return output_root / rel


def feature_masks(data: Mapping[str, Any], idp, node_feat: torch.Tensor, edge_feat: torch.Tensor):
    atom_types = atom_types_from_data(data, idp, device=node_feat.device)
    edge_types = edge_types_from_data(data, idp, device=edge_feat.device)
    return (
        idp.mask_to_nrme[atom_types].to(device=node_feat.device, dtype=torch.bool),
        idp.mask_to_erme[edge_types].to(device=edge_feat.device, dtype=torch.bool),
    )


def tensorize_feature_sample(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)
    for key in (
        "atomic_numbers",
        "atom_types",
        "edge_index",
        "edge_cell_shift",
        "edge_types",
        NODE_FEATURES_KEY,
        EDGE_FEATURES_KEY,
        NODE_H0_KEY,
        EDGE_H0_KEY,
    ):
        if key in out:
            out[key] = torch.as_tensor(out[key])
    return out


def strict_roundtrip_check(
    sample: Dict[str, Any],
    idp,
    blocks: Mapping[str, Any],
    *,
    node_key: str,
    edge_key: str,
    atol: float,
    rtol: float,
) -> float:
    """Assert that feature_to_block followed by block_to_feature reproduces masked features."""
    check_data = dict(sample)
    check_data[NODE_FEATURES_KEY] = sample[node_key]
    check_data[EDGE_FEATURES_KEY] = sample[edge_key]
    check_data = tensorize_feature_sample(check_data)
    out_data = dict(check_data)
    out_data.pop(NODE_FEATURES_KEY, None)
    out_data.pop(EDGE_FEATURES_KEY, None)
    block_to_feature(out_data, idp, blocks=blocks, overlap_blocks=False, orthogonal=False)

    node_old = torch.as_tensor(sample[node_key])
    edge_old = torch.as_tensor(sample[edge_key])
    node_new = torch.as_tensor(out_data[NODE_FEATURES_KEY]).to(dtype=node_old.dtype)
    edge_new = torch.as_tensor(out_data[EDGE_FEATURES_KEY]).to(dtype=edge_old.dtype)
    node_mask, edge_mask = feature_masks(sample, idp, node_old, edge_old)

    node_ok = torch.allclose(node_new[node_mask], node_old[node_mask], atol=atol, rtol=rtol)
    edge_ok = torch.allclose(edge_new[edge_mask], edge_old[edge_mask], atol=atol, rtol=rtol)
    max_vals = []
    if node_mask.any():
        max_vals.append((node_new - node_old).abs()[node_mask].max())
    if edge_mask.any():
        max_vals.append((edge_new - edge_old).abs()[edge_mask].max())
    max_abs = float(torch.stack([v.detach().cpu() for v in max_vals]).max()) if max_vals else 0.0
    if not (node_ok and edge_ok):
        raise RuntimeError(
            f"strict roundtrip failed for {node_key}/{edge_key}; masked max_abs={max_abs:g}. "
            "Check basis, mapper_method, or feature schema."
        )
    return max_abs


def feature_to_block_dict_for_keys(sample: Dict[str, Any], idp, node_key: str, edge_key: str):
    work = dict(sample)
    work[NODE_FEATURES_KEY] = torch.as_tensor(sample[node_key])
    work[EDGE_FEATURES_KEY] = torch.as_tensor(sample[edge_key])
    work = tensorize_feature_sample(work)
    return feature_to_block(work, idp, overlap=False)


def cast_block_tensors(sample: Dict[str, Any], dtype_policy: str) -> None:
    if dtype_policy == "auto":
        return
    dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }[dtype_policy]
    for key in (NODE_DELTA_HAMIL_BLOCKS_KEY, EDGE_DELTA_HAMIL_BLOCKS_KEY, NODE_H0_BLOCKS_KEY, EDGE_H0_BLOCKS_KEY):
        if key in sample:
            sample[key] = torch.as_tensor(sample[key]).to(dtype=dtype)


def apply_feature_policy(sample: Dict[str, Any], *, node_key: str, edge_key: str, policy: str, shadow_prefix: str) -> None:
    if policy == "keep":
        return
    if policy == "drop":
        sample.pop(node_key, None)
        sample.pop(edge_key, None)
        return
    if policy == "rename_shadow":
        if node_key in sample:
            sample[f"node_{shadow_prefix}_features_shadow"] = sample.pop(node_key)
        if edge_key in sample:
            sample[f"edge_{shadow_prefix}_features_shadow"] = sample.pop(edge_key)
        return
    raise ValueError(f"Unknown feature policy: {policy}")


def convert_one_feature_pair(
    out: Dict[str, Any],
    idp,
    args,
    *,
    node_key: str,
    edge_key: str,
    prefix: str,
) -> Tuple[float, list, list]:
    blocks = feature_to_block_dict_for_keys(out, idp, node_key, edge_key)
    max_abs = None
    if args.strict_roundtrip:
        max_abs = strict_roundtrip_check(out, idp, blocks, node_key=node_key, edge_key=edge_key, atol=args.atol, rtol=args.rtol)
    packed = block_dict_to_ordered_tensors(
        out,
        idp,
        blocks,
        start_id=args.start_id,
        complete_edges=(args.edge_complete_policy == "hermitian"),
        strict_complete_edges=args.strict_edge_completion,
    )
    attach_block_tensors(out, packed, prefix=prefix)
    node_shape = list(torch.as_tensor(packed.node_blocks).shape) if packed.node_blocks is not None else None
    edge_shape = list(torch.as_tensor(packed.edge_blocks).shape) if packed.edge_blocks is not None else None
    return max_abs, node_shape, edge_shape


def convert_sample(sample: Dict[str, Any], idp, args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for required in (NODE_FEATURES_KEY, EDGE_FEATURES_KEY):
        if required not in sample:
            raise KeyError(f"Sample lacks delta-H feature key {required}")
    if args.convert_h0_blocks:
        for required in (NODE_H0_KEY, EDGE_H0_KEY):
            if required not in sample:
                raise KeyError(f"--convert-h0-blocks requested but sample lacks {required}")

    out = dict(sample)
    delta_max, delta_node_shape, delta_edge_shape = convert_one_feature_pair(
        out, idp, args, node_key=NODE_FEATURES_KEY, edge_key=EDGE_FEATURES_KEY, prefix="delta_hamil"
    )
    h0_max = h0_node_shape = h0_edge_shape = None
    if args.convert_h0_blocks:
        h0_max, h0_node_shape, h0_edge_shape = convert_one_feature_pair(
            out, idp, args, node_key=NODE_H0_KEY, edge_key=EDGE_H0_KEY, prefix="h0"
        )

    cast_block_tensors(out, args.block_dtype)
    apply_feature_policy(out, node_key=NODE_FEATURES_KEY, edge_key=EDGE_FEATURES_KEY, policy=args.target_feature_policy, shadow_prefix="delta")
    if args.convert_h0_blocks:
        apply_feature_policy(out, node_key=NODE_H0_KEY, edge_key=EDGE_H0_KEY, policy=args.h0_feature_policy, shadow_prefix="h0")

    return out, {
        "delta_roundtrip_max_abs": delta_max,
        "h0_roundtrip_max_abs": h0_max,
        "node_delta_hamil_blocks_shape": delta_node_shape,
        "edge_delta_hamil_blocks_shape": delta_edge_shape,
        "node_h0_blocks_shape": h0_node_shape,
        "edge_h0_blocks_shape": h0_edge_shape,
        "edge_complete_policy": args.edge_complete_policy,
        "kept_target_features": NODE_FEATURES_KEY in out and EDGE_FEATURES_KEY in out,
        "kept_h0_features": NODE_H0_KEY in out and EDGE_H0_KEY in out,
    }


def convert_lmdb(src: Path, dst: Path, idp, args) -> Dict[str, Any]:
    if dst.exists():
        if args.overwrite:
            shutil.rmtree(dst)
        else:
            raise FileExistsError(f"Output exists: {dst}; use --overwrite.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_env = lmdb.open(str(src), readonly=True, lock=False, readahead=False, max_readers=2048)
    dst_env = lmdb.open(str(dst), map_size=args.map_size, subdir=True, lock=True, readahead=False, meminit=False)
    t0 = time.time()
    count = 0
    max_delta = 0.0
    max_h0 = 0.0
    first_meta = None
    try:
        with src_env.begin(buffers=True) as src_txn:
            total_entries = src_txn.stat()["entries"]
            cursor = src_txn.cursor()
            txn = dst_env.begin(write=True)
            try:
                for raw_key, raw_val in cursor:
                    sample = pickle.loads(bytes(raw_val))
                    converted, meta = convert_sample(sample, idp, args)
                    first_meta = first_meta or meta
                    if meta["delta_roundtrip_max_abs"] is not None:
                        max_delta = max(max_delta, float(meta["delta_roundtrip_max_abs"]))
                    if meta["h0_roundtrip_max_abs"] is not None:
                        max_h0 = max(max_h0, float(meta["h0_roundtrip_max_abs"]))
                    txn.put(bytes(raw_key), pickle.dumps(converted, protocol=pickle.HIGHEST_PROTOCOL))
                    count += 1
                    if count % args.commit_every == 0:
                        txn.commit()
                        txn = dst_env.begin(write=True)
                    if args.max_entries and count >= args.max_entries:
                        break
                txn.commit()
            except Exception:
                txn.abort()
                raise
    finally:
        src_env.close()
        dst_env.sync()
        dst_env.close()
    return {
        "src": str(src),
        "dst": str(dst),
        "entries_seen": total_entries,
        "entries_written": count,
        "max_delta_roundtrip_abs": max_delta,
        "max_h0_roundtrip_abs": max_h0,
        "first_entry_meta": first_meta,
        "elapsed_sec": time.time() - t0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-root", required=True, help="Source LMDB root or a single .lmdb directory.")
    parser.add_argument("--output-root", required=True, help="Destination root; shard paths are mirrored.")
    parser.add_argument("--split", default=None, help="Optional split subdir such as train/valid/test.")
    parser.add_argument("--input-config", default=None, help="DeePTB JSON/YAML input file containing basis.")
    parser.add_argument("--basis-file", default=None, help="JSON/YAML basis dict or object containing basis.")
    parser.add_argument("--basis-json", default=None, help="Inline JSON basis dict.")
    parser.add_argument("--mapper-method", default="e3tb", help="OrbitalMapper method used by the feature dataset.")
    parser.add_argument("--has-soc", action="store_true", help="Fail-fast guard; SOC is not supported in this package.")
    parser.add_argument("--nextham-uureal-mask", action="store_true", help="Pass nextham_uureal_mask to OrbitalMapper when supported.")
    parser.add_argument("--strict-roundtrip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--start-id", type=int, default=0, help="Block key atom index base used by feature_to_block.")
    parser.add_argument("--edge-complete-policy", choices=["hermitian", "none"], default="hermitian")
    parser.add_argument(
        "--strict-edge-completion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When hermitian completion is enabled, fail if reverse edges are "
            "missing and full AO blocks would contain unresolved entries."
        ),
    )
    parser.add_argument("--block-dtype", choices=["auto", "float32", "float64", "complex64", "complex128"], default="float32")
    parser.add_argument("--convert-h0-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-feature-policy", choices=["drop", "keep", "rename_shadow"], default="drop")
    parser.add_argument("--h0-feature-policy", choices=["drop", "keep", "rename_shadow"], default="drop")
    parser.add_argument("--map-size", type=int, default=1 << 40)
    parser.add_argument("--commit-every", type=int, default=256)
    parser.add_argument("--max-entries", type=int, default=0, help="Debug: stop after N entries; 0 means all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default=None, help="Write conversion report JSON.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    idp = build_mapper(args)
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    paths = list(find_lmdb_paths(str(input_root), args.split))
    if not paths:
        raise FileNotFoundError(f"No .lmdb directories found under {input_root}, split={args.split}")

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "split": args.split,
        "basis": load_basis(args),
        "mapper_method": args.mapper_method,
        "strict_roundtrip": bool(args.strict_roundtrip),
        "strict_edge_completion": bool(args.strict_edge_completion),
        "edge_complete_policy": args.edge_complete_policy,
        "target_feature_policy": args.target_feature_policy,
        "h0_feature_policy": args.h0_feature_policy,
        "convert_h0_blocks": bool(args.convert_h0_blocks),
        "shards": [],
    }
    for src in paths:
        dst = mirror_output_path(src.resolve(), input_root, output_root)
        print(f"[convert] {src} -> {dst}", flush=True)
        stat = convert_lmdb(src, dst, idp, args)
        report["shards"].append(stat)
        print(
            f"  wrote {stat['entries_written']} entries; "
            f"delta max={stat['max_delta_roundtrip_abs']:.3e}; "
            f"h0 max={stat['max_h0_roundtrip_abs']:.3e}; "
            f"elapsed={stat['elapsed_sec']:.1f}s",
            flush=True,
        )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

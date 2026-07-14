#!/usr/bin/env python3
"""Benchmark the four distinct dual-P2/P23 LMDB data contracts.

This is a CPU/data-loader benchmark only.  It deliberately does not build a
model.  The eight training configs collapse to four loader contracts because
the memory on/off axis does not change ``data_options``:

* P2 residual (RME + P2 AO blocks + absolute Full-H target);
* P2 direct (RME + absolute Full-H target);
* P23 residual (RME + P23 AO blocks + absolute Full-H target);
* P23 direct (RME + absolute Full-H target).

"Cold" below means the first immutable-record contract gate in a fresh dataset
instance.  It does not claim that the operating-system page cache is cold.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dptb.data.build import build_dataset
from dptb.utils.argcheck import collect_cutoffs, normalize


SCHEMA = "deeptb.dual_prior_loader_benchmark/v1"
EXPECTED_AXES = {
    (prior, head, memory)
    for prior in ("p2", "p23")
    for head in ("residual", "direct")
    for memory in (False, True)
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_production_manifest(
    manifest: Mapping[str, Any], *, allow_dev_configs: bool
) -> None:
    binding = manifest.get("source_binding_contract")
    if not isinstance(binding, Mapping):
        raise ValueError(
            "dual-prior config manifest lacks source_binding_contract; "
            "refusing an unversioned loader benchmark"
        )
    production_qualified = binding.get("production_qualified") is True
    dev_override = binding.get("allow_unbound_source_fingerprints") is True
    if not allow_dev_configs and (not production_qualified or dev_override):
        raise ValueError(
            "production loader benchmark requires production_qualified=true "
            "and no unbound-source dev override"
        )


def _manifest_config_path(config_dir: Path, name: str, entry: Mapping[str, Any]) -> Path:
    candidates = [
        config_dir / f"{name}.json",
        config_dir / Path(str(entry.get("path", ""))).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"manifest config {name!r} is absent from {config_dir}; tried {candidates}"
    )


def load_four_data_contracts(
    config_dir: Path,
    *,
    split: str,
    allow_dev_configs: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load, normalize, and collapse the eight configs to four data arms."""

    config_dir = config_dir.resolve()
    manifest_path = config_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    _require_production_manifest(manifest, allow_dev_configs=allow_dev_configs)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or len(files) != 8:
        raise ValueError("dual-prior benchmark requires exactly eight manifest configs")

    matrix: dict[tuple[str, str], dict[bool, dict[str, Any]]] = {}
    observed_axes: set[tuple[str, str, bool]] = set()
    for name, entry_value in files.items():
        if not isinstance(entry_value, Mapping):
            raise TypeError(f"manifest files[{name!r}] must be an object")
        path = _manifest_config_path(config_dir, str(name), entry_value)
        expected_sha = str(entry_value.get("sha256", "")).strip().lower()
        if len(expected_sha) != 64 or _sha256(path) != expected_sha:
            raise ValueError(f"config SHA256 mismatch for {path}")

        config = normalize(copy.deepcopy(_read_json(path)))
        embedding = config["model_options"]["embedding"]
        prediction = config["model_options"]["prediction"]
        prior = str(embedding["prior_kind"]).strip().lower()
        head = "residual" if bool(prediction.get("add_prior", False)) else "direct"
        memory = bool(embedding.get("use_soft_edge_memory", False))
        axis = (prior, head, memory)
        if axis in observed_axes:
            raise ValueError(f"duplicate dual-prior config axis {axis}")
        observed_axes.add(axis)
        if split not in config.get("data_options", {}):
            raise KeyError(f"config {path} lacks data_options.{split}")
        matrix.setdefault((prior, head), {})[memory] = {
            "name": str(name),
            "path": str(path),
            "config": config,
        }

    if observed_axes != EXPECTED_AXES:
        raise ValueError(
            "dual-prior config axes are incomplete: "
            f"missing={sorted(EXPECTED_AXES - observed_axes)}, "
            f"extra={sorted(observed_axes - EXPECTED_AXES)}"
        )

    selected: dict[str, dict[str, Any]] = {}
    shared_roots: set[str] = set()
    shared_storage_contracts: set[tuple[str, str, str]] = set()
    for prior, head in (("p2", "residual"), ("p2", "direct"), ("p23", "residual"), ("p23", "direct")):
        pair = matrix[(prior, head)]
        off = pair[False]
        on = pair[True]
        off_data = off["config"]["data_options"][split]
        on_data = on["config"]["data_options"][split]
        if off_data != on_data:
            raise ValueError(
                f"memory axis changes data_options.{split} for {prior}/{head}"
            )
        if off["config"]["common_options"] != on["config"]["common_options"]:
            raise ValueError(f"memory axis changes common_options for {prior}/{head}")
        if collect_cutoffs(off["config"]) != collect_cutoffs(on["config"]):
            raise ValueError(f"memory axis changes cutoffs for {prior}/{head}")

        expected_source = str(
            off_data.get("expected_p2_source_fingerprint", "")
        ).strip()
        if not allow_dev_configs and len(expected_source) != 64:
            raise ValueError(
                f"{prior}/{head} lacks a bound 64-hex source fingerprint"
            )
        root = os.path.realpath(str(off_data["root"]))
        shared_roots.add(root)
        shared_storage_contracts.add(
            (
                str(off_data.get("type")),
                str(off_data.get("prefix")),
                str(off_data.get("separator", ".")),
            )
        )
        arm = f"{prior}_{head}"
        selected[arm] = {
            **off,
            "prior_kind": prior,
            "head_mode": head,
            "memory_config_equivalent": on["path"],
            "root": root,
        }

    if len(shared_roots) != 1 or len(shared_storage_contracts) != 1:
        raise ValueError(
            "all four loader arms must reference the same LMDB root, type, "
            "prefix, and separator"
        )
    return selected


def _build_from_config(config: Mapping[str, Any], *, split: str):
    payload = copy.deepcopy(dict(config))
    return build_dataset(
        **collect_cutoffs(payload),
        **payload["data_options"][split],
        **payload["common_options"],
    )


def _close_dataset(dataset: Any) -> None:
    for env in getattr(dataset, "_lmdb_env_cache", {}).values():
        try:
            env.close()
        except Exception:
            pass
    if hasattr(dataset, "_lmdb_env_cache"):
        dataset._lmdb_env_cache = {}


def _cache_size(dataset: Any) -> int:
    cache = getattr(dataset, "_validated_record_contracts", None)
    if not isinstance(cache, dict):
        raise RuntimeError(
            "LMDBDataset does not expose the worker-local runtime contract cache"
        )
    return len(cache)


def _last_pickle_bytes(dataset: Any) -> int:
    value = getattr(dataset, "_last_lmdb_pickle_bytes", None)
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            "LMDBDataset did not report positive _last_lmdb_pickle_bytes"
        )
    return value


def _metric(*, records: int, pickle_bytes: int, seconds: float) -> dict[str, Any]:
    seconds = max(float(seconds), 1.0e-12)
    mib = float(pickle_bytes) / (1024.0 * 1024.0)
    return {
        "records": int(records),
        "seconds": seconds,
        "milliseconds": seconds * 1000.0,
        "pickle_bytes": int(pickle_bytes),
        "pickle_mib": mib,
        "records_per_second": float(records) / seconds,
        "mib_per_second": mib / seconds,
    }


def benchmark_one_arm(
    config: Mapping[str, Any],
    *,
    split: str,
    warm_repeats: int,
    probe_index: int,
    max_records: int,
) -> dict[str, Any]:
    if warm_repeats <= 0:
        raise ValueError("warm_repeats must be positive")

    probe_dataset = _build_from_config(config, split=split)
    try:
        available = len(probe_dataset)
        if available <= 0:
            raise ValueError("selected LMDB split is empty")
        if not 0 <= probe_index < available:
            raise IndexError(
                f"probe_index={probe_index} is outside [0,{available - 1}]"
            )
        cache_before = _cache_size(probe_dataset)
        if cache_before != 0:
            raise RuntimeError("fresh probe dataset already has validated records")

        gc.collect()
        started = time.perf_counter()
        probe_dataset.get(probe_index)
        cold_seconds = time.perf_counter() - started
        cold_bytes = _last_pickle_bytes(probe_dataset)
        cache_after_cold = _cache_size(probe_dataset)
        if cache_after_cold != 1:
            raise RuntimeError(
                "first dual-prior get did not add exactly one validated-record "
                f"cache entry: before={cache_before}, after={cache_after_cold}"
            )

        warm_bytes = 0
        gc.collect()
        started = time.perf_counter()
        for _ in range(warm_repeats):
            probe_dataset.get(probe_index)
            warm_bytes += _last_pickle_bytes(probe_dataset)
        warm_seconds = time.perf_counter() - started
        cache_after_warm = _cache_size(probe_dataset)
        if cache_after_warm != cache_after_cold:
            raise RuntimeError(
                "same-record warm reads changed the validation cache size: "
                f"cold={cache_after_cold}, warm={cache_after_warm}"
            )
        last_identity = getattr(probe_dataset, "_last_lmdb_record_identity", None)
    finally:
        _close_dataset(probe_dataset)

    sequential_dataset = _build_from_config(config, split=split)
    try:
        sequential_available = len(sequential_dataset)
        if sequential_available != available:
            raise RuntimeError("fresh sequential dataset changed split length")
        record_count = (
            sequential_available
            if max_records <= 0
            else min(int(max_records), sequential_available)
        )
        if record_count <= 0:
            raise ValueError("sequential benchmark selected zero records")
        sequential_cache_before = _cache_size(sequential_dataset)
        if sequential_cache_before != 0:
            raise RuntimeError("fresh sequential dataset already has validated records")

        sequential_bytes = 0
        gc.collect()
        started = time.perf_counter()
        for index in range(record_count):
            sequential_dataset.get(index)
            sequential_bytes += _last_pickle_bytes(sequential_dataset)
        sequential_seconds = time.perf_counter() - started
        sequential_cache_after = _cache_size(sequential_dataset)
        if sequential_cache_after != record_count:
            raise RuntimeError(
                "sequential first pass did not validate exactly one cache entry "
                f"per record: expected={record_count}, got={sequential_cache_after}"
            )
    finally:
        _close_dataset(sequential_dataset)

    return {
        "records_available": available,
        "probe_index": probe_index,
        "probe_record_identity": last_identity,
        "cold_first_get": {
            **_metric(records=1, pickle_bytes=cold_bytes, seconds=cold_seconds),
            "semantics": "fresh dataset; first immutable-record contract gate",
            "validation_cache_entries_before": cache_before,
            "validation_cache_entries_after": cache_after_cold,
            "first_gate_executed": True,
        },
        "warm_same_record": {
            **_metric(
                records=warm_repeats,
                pickle_bytes=warm_bytes,
                seconds=warm_seconds,
            ),
            "semantics": "same record; runtime contract cache already populated",
            "validation_cache_entries_before": cache_after_cold,
            "validation_cache_entries_after": cache_after_warm,
            "full_array_gates_reused": True,
        },
        "sequential_first_pass": {
            **_metric(
                records=record_count,
                pickle_bytes=sequential_bytes,
                seconds=sequential_seconds,
            ),
            "semantics": "fresh dataset; one first-gate get per sequential record",
            "complete_split": record_count == sequential_available,
            "validation_cache_entries_before": sequential_cache_before,
            "validation_cache_entries_after": sequential_cache_after,
        },
    }


def benchmark_matrix(
    *,
    config_dir: Path,
    split: str = "train",
    warm_repeats: int = 10,
    probe_index: int = 0,
    max_records: int = 0,
    allow_dev_configs: bool = False,
) -> dict[str, Any]:
    contracts = load_four_data_contracts(
        config_dir,
        split=split,
        allow_dev_configs=allow_dev_configs,
    )
    results: dict[str, Any] = {}
    for arm, entry in contracts.items():
        result = benchmark_one_arm(
            entry["config"],
            split=split,
            warm_repeats=warm_repeats,
            probe_index=probe_index,
            max_records=max_records,
        )
        results[arm] = {
            "config_name": entry["name"],
            "config_path": entry["path"],
            "memory_config_with_identical_data_options": entry[
                "memory_config_equivalent"
            ],
            "prior_kind": entry["prior_kind"],
            "head_mode": entry["head_mode"],
            "require_prior_ao_blocks": bool(
                entry["config"]["data_options"][split].get(
                    "require_p2_blocks", False
                )
            ),
            **result,
        }

    roots = {entry["root"] for entry in contracts.values()}
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
        },
        "config_dir": str(config_dir.resolve()),
        "split": split,
        "shared_lmdb_root": next(iter(roots)),
        "warm_repeats": warm_repeats,
        "probe_index": probe_index,
        "max_records": max_records,
        "allow_dev_configs": allow_dev_configs,
        "cold_semantics": (
            "contract-cache cold in a fresh dataset instance; OS page cache is "
            "not flushed or claimed cold"
        ),
        "arms": results,
    }


def _print_summary(report: Mapping[str, Any]) -> None:
    print(
        "arm cold_ms warm_records/s sequential_records/s sequential_MiB/s "
        "cache_after"
    )
    for arm, result in report["arms"].items():
        cold = result["cold_first_get"]
        warm = result["warm_same_record"]
        sequential = result["sequential_first_pass"]
        print(
            f"{arm} {cold['milliseconds']:.3f} "
            f"{warm['records_per_second']:.3f} "
            f"{sequential['records_per_second']:.3f} "
            f"{sequential['mib_per_second']:.3f} "
            f"{sequential['validation_cache_entries_after']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "reference", "test"), default="train"
    )
    parser.add_argument("--warm-repeats", type=int, default=10)
    parser.add_argument("--probe-index", type=int, default=0)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="0 benchmarks the complete split; positive values cap the first pass",
    )
    parser.add_argument(
        "--allow-dev-configs",
        action="store_true",
        help="Allow a manifest explicitly marked non-production/dev-only",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = benchmark_matrix(
        config_dir=args.config_dir,
        split=args.split,
        warm_repeats=args.warm_repeats,
        probe_index=args.probe_index,
        max_records=args.max_records,
        allow_dev_configs=args.allow_dev_configs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _print_summary(report)
    print(f"OUTPUT={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

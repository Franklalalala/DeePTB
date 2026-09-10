"""Aggregate matched DeePTB LoopSCF evaluator JSON into a units-labeled report.

This is supporting tooling. It does not evaluate Hamiltonians; it only
aggregates already-computed per-structure metrics.

Usage:
  python summarize_experiment.py --inputs JSON [JSON ...] --output report.json
  python summarize_experiment.py --inputs JSON [JSON ...] --output report.json --closure-only

MAE fields are eV. forward_seconds is aggregated separately in seconds.
Paired deltas are (new - reference); a positive difference is worse.
Pair stats require identical index sets. If sets differ, missing IDs are
listed and pair stats are omitted (no silent intersection). Failures are
preserved and counted. Duplicate (variant, index) pairs, nonfinite numbers,
negative counts, and inconsistent count fields are rejected.

Default includes every parsed result row and labels closure-mismatch
count/indices. --closure-only is an explicit policy that drops rows with
label_closure_pass=false or label_closure_fw10_ev>=0.001 from aggregate
stats only; excluded IDs are reported per input and original failures are
kept in full.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

METRIC_FIELDS = (
    "onsite_mae_ev",
    "hopping_mae_ev",
    "packed_valid_element_mae_ev",
    "legacy_vbm_aligned_fw10_ev",
)
TIME_FIELD = "forward_seconds"
COUNT_FIELDS = ("n_onsite_elements", "n_hopping_elements")
REQUIRED_VARIANT_FIELDS = METRIC_FIELDS + COUNT_FIELDS + (TIME_FIELD,)
DIAGNOSTIC_FIELDS = (
    "label_closure_fw10_ev",
    "label_closure_pass",
    "path_overlap",
    "occupation_overlap",
    "rotation_relative_errors",
)
CLOSURE_THRESHOLD_EV = 0.001
PACKED_REL_TOL = 1e-5
PACKED_ABS_TOL = 1e-9

UNITS = {
    "onsite_mae_ev": "eV",
    "hopping_mae_ev": "eV",
    "packed_valid_element_mae_ev": "eV",
    "legacy_vbm_aligned_fw10_ev": "eV",
    "label_closure_fw10_ev": "eV",
    "forward_seconds": "s",
}


class SummarizeError(ValueError):
    """Fail-closed input or aggregation error."""


def _ctx(path: str, index: Any = None, variant: Any = None) -> str:
    parts = [f"input={path}"]
    if index is not None:
        parts.append(f"index={index}")
    if variant is not None:
        parts.append(f"variant={variant!r}")
    return " ".join(parts)


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value) and value.is_integer()


def require_index(value: Any, ctx: str) -> int:
    if not _is_int_like(value):
        raise SummarizeError(f"index must be an integer ({ctx}); got {value!r}")
    return int(value)


def require_nonneg_int(value: Any, field: str, ctx: str) -> int:
    if not _is_int_like(value):
        raise SummarizeError(
            f"inconsistent count field {field} ({ctx}): expected an integer, got {value!r}"
        )
    number = int(value)
    if number < 0:
        raise SummarizeError(f"negative count field {field} ({ctx})")
    return number


def require_finite_float(value: Any, field: str, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummarizeError(f"{field} must be a finite number ({ctx}); got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SummarizeError(f"nonfinite {field} ({ctx}): {value!r}")
    return number


def require_mapping(value: Any, field: str, ctx: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SummarizeError(
            f"{field} must be an object ({ctx}); got {type(value).__name__}"
        )
    return value


def require_list(value: Any, field: str, ctx: str) -> list[Any]:
    if not isinstance(value, list):
        raise SummarizeError(
            f"{field} must be a list ({ctx}); got {type(value).__name__}"
        )
    return value


def implied_packed_mae(
    onsite_mae: float, n_onsite: int, hopping_mae: float, n_hopping: int
) -> float:
    total_n = n_onsite + n_hopping
    if total_n <= 0:
        raise SummarizeError("n_onsite_elements + n_hopping_elements must be > 0")
    return (onsite_mae * n_onsite + hopping_mae * n_hopping) / total_n


def percentile(values: Sequence[float], q: float) -> float:
    """Inclusive linear interpolation; q in [0, 1]. Hyndman-Fan R7."""
    if not values:
        raise SummarizeError("cannot compute percentile of an empty series")
    if not 0.0 <= q <= 1.0:
        raise SummarizeError(f"percentile q must be in [0, 1]; got {q}")
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * q
    lo = int(math.floor(h)) - 1
    hi = int(math.ceil(h)) - 1
    if lo == hi:
        return ordered[lo]
    frac = h - math.floor(h)
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def series_stats(values: Sequence[float], unit: str | None = None) -> dict[str, Any]:
    if not values:
        raise SummarizeError("cannot aggregate an empty series")
    ordered = sorted(values)
    out = {
        "mean": sum(ordered) / len(ordered),
        "median": statistics.median(ordered),
        "p90": percentile(ordered, 0.90),
        "max": ordered[-1],
        "count": len(ordered),
    }
    if unit is not None:
        out["unit"] = unit
    return out


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SummarizeError(f"input not found: {path}") from exc
    except OSError as exc:
        raise SummarizeError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SummarizeError(f"invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, ensure_ascii=False)
        handle.write("\n")


def parse_diagnostics(raw: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "label_closure_fw10_ev" in raw:
        out["label_closure_fw10_ev"] = require_finite_float(
            raw["label_closure_fw10_ev"], "label_closure_fw10_ev", ctx
        )
    if "label_closure_pass" in raw:
        value = raw["label_closure_pass"]
        if not isinstance(value, bool):
            raise SummarizeError(
                f"label_closure_pass must be a boolean ({ctx}); got {value!r}"
            )
        out["label_closure_pass"] = value
    for field in ("path_overlap", "occupation_overlap", "rotation_relative_errors"):
        if field in raw:
            out[field] = raw[field]
    return out


def present_diagnostics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in DIAGNOSTIC_FIELDS if field in row}


def closure_mismatch(diag: Mapping[str, Any]) -> bool:
    if diag.get("label_closure_pass") is False:
        return True
    fw10 = diag.get("label_closure_fw10_ev")
    return fw10 is not None and fw10 >= CLOSURE_THRESHOLD_EV


def closure_reasons(diag: Mapping[str, Any]) -> list[str]:
    reasons = []
    if diag.get("label_closure_pass") is False:
        reasons.append("label_closure_pass=false")
    fw10 = diag.get("label_closure_fw10_ev")
    if fw10 is not None and fw10 >= CLOSURE_THRESHOLD_EV:
        reasons.append(f"label_closure_fw10_ev>={CLOSURE_THRESHOLD_EV}")
    return reasons


def parse_variant_metrics(raw: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    missing = [field for field in REQUIRED_VARIANT_FIELDS if field not in raw]
    if missing:
        raise SummarizeError(f"missing fields {missing} ({ctx})")
    metrics = {
        field: require_finite_float(raw[field], field, ctx) for field in METRIC_FIELDS
    }
    if any(value < 0 for value in metrics.values()):
        raise SummarizeError(f"negative absolute error ({ctx})")
    n_onsite = require_nonneg_int(raw["n_onsite_elements"], "n_onsite_elements", ctx)
    n_hopping = require_nonneg_int(raw["n_hopping_elements"], "n_hopping_elements", ctx)
    if n_onsite + n_hopping <= 0:
        raise SummarizeError(
            f"inconsistent count fields n_onsite_elements/n_hopping_elements ({ctx}): both are 0"
        )
    implied = implied_packed_mae(
        metrics["onsite_mae_ev"], n_onsite, metrics["hopping_mae_ev"], n_hopping
    )
    packed = metrics["packed_valid_element_mae_ev"]
    if not math.isclose(
        packed, implied, rel_tol=PACKED_REL_TOL, abs_tol=PACKED_ABS_TOL
    ):
        raise SummarizeError(
            "inconsistent count fields "
            f"({ctx}): packed_valid_element_mae_ev={packed} but onsite/hopping "
            f"weighted by n_onsite_elements={n_onsite} and n_hopping_elements="
            f"{n_hopping} gives {implied}"
        )
    return {
        **metrics,
        "n_onsite_elements": n_onsite,
        "n_hopping_elements": n_hopping,
        "forward_seconds": require_finite_float(raw[TIME_FIELD], TIME_FIELD, ctx),
    }


def parse_document(path: str, data: Any) -> dict[str, Any]:
    ctx = _ctx(path)
    root = require_mapping(data, "document", ctx)
    if "protocol" not in root or "results" not in root or "failures" not in root:
        raise SummarizeError(
            f"document must contain protocol, results, and failures ({ctx})"
        )
    protocol = require_mapping(root["protocol"], "protocol", ctx)
    results = require_list(root["results"], "results", ctx)
    failures_raw = require_list(root["failures"], "failures", ctx)

    records: list[dict[str, Any]] = []
    for row_i, row in enumerate(results):
        row_ctx = f"{ctx} results[{row_i}]"
        row_map = require_mapping(row, "result", row_ctx)
        if "index" not in row_map or "variants" not in row_map:
            raise SummarizeError(f"result requires index and variants ({row_ctx})")
        index = require_index(row_map["index"], row_ctx)
        variants = require_mapping(
            row_map["variants"], "variants", f"{row_ctx} index={index}"
        )
        if not variants:
            raise SummarizeError(
                f"variants must not be empty ({row_ctx} index={index})"
            )
        row_diag = parse_diagnostics(row_map, f"{row_ctx} index={index}")
        for name, payload in variants.items():
            if not isinstance(name, str) or not name:
                raise SummarizeError(
                    f"variant name must be a non-empty string ({row_ctx})"
                )
            item_ctx = _ctx(path, index, name)
            payload_map = require_mapping(payload, "variant", item_ctx)
            metrics = parse_variant_metrics(payload_map, item_ctx)
            diag = {**row_diag, **parse_diagnostics(payload_map, item_ctx)}
            records.append(
                {"index": index, "variant": name, "source": path, **metrics, **diag}
            )

    failures: list[dict[str, Any]] = []
    for fail_i, failure in enumerate(failures_raw):
        fail_ctx = f"{ctx} failures[{fail_i}]"
        fail_map = require_mapping(failure, "failure", fail_ctx)
        if "index" not in fail_map:
            raise SummarizeError(f"failure requires index ({fail_ctx})")
        item = dict(fail_map)
        item["index"] = require_index(fail_map["index"], fail_ctx)
        item["report_input_path"] = path
        failures.append(item)

    return {
        "path": path,
        "protocol": protocol,
        "records": records,
        "failures": failures,
    }


def _reject_duplicates(records: Sequence[Mapping[str, Any]]) -> None:
    seen: dict[tuple[int, str], str] = {}
    for row in records:
        key = (row["index"], row["variant"])
        if key in seen:
            raise SummarizeError(
                "duplicate input variant/index pair: "
                f"variant={row['variant']!r} index={row['index']} "
                f"(first source: {seen[key]}, duplicate source: {row['source']})"
            )
        seen[key] = row["source"]


def _reject_inconsistent_counts(records: Sequence[Mapping[str, Any]]) -> None:
    by_index: dict[int, tuple[int, int, str, str]] = {}
    for row in records:
        counts = (row["n_onsite_elements"], row["n_hopping_elements"])
        previous = by_index.get(row["index"])
        if previous is None:
            by_index[row["index"]] = (*counts, row["variant"], row["source"])
            continue
        n_on, n_hop, other_variant, source = previous
        if counts != (n_on, n_hop):
            raise SummarizeError(
                "inconsistent count fields for "
                f"index={row['index']}: variant {other_variant!r} ({source}) has "
                f"n_onsite_elements={n_on}, n_hopping_elements={n_hop}; "
                f"variant {row['variant']!r} ({row['source']}) has "
                f"n_onsite_elements={counts[0]}, n_hopping_elements={counts[1]}"
            )


def _weighted_mae(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    on_abs = sum(row["onsite_mae_ev"] * row["n_onsite_elements"] for row in rows)
    hop_abs = sum(row["hopping_mae_ev"] * row["n_hopping_elements"] for row in rows)
    n_on = sum(row["n_onsite_elements"] for row in rows)
    n_hop = sum(row["n_hopping_elements"] for row in rows)
    n_valid = n_on + n_hop
    if n_valid <= 0:
        raise SummarizeError(
            "valid-element weighted MAE requires a positive element count"
        )
    return {
        "onsite": (on_abs / n_on) if n_on else None,
        "hopping": (hop_abs / n_hop) if n_hop else None,
        "packed_valid_element": (on_abs + hop_abs) / n_valid,
        "n_onsite_elements": n_on,
        "n_hopping_elements": n_hop,
        "n_valid_elements": n_valid,
        "unit": "eV",
        "note": (
            "element-weighted from onsite and hopping MAE*count sums; "
            "not the unweighted mean of per-structure packed_valid_element_mae_ev"
        ),
    }


def _variant_summary(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["index"])
    return {
        "name": name,
        "count": len(ordered),
        "indices": [row["index"] for row in ordered],
        "metrics": {
            field: series_stats([row[field] for row in ordered], UNITS[field])
            for field in METRIC_FIELDS
        },
        "forward_seconds": series_stats(
            [row[TIME_FIELD] for row in ordered], UNITS[TIME_FIELD]
        ),
        "weighted_mae_ev": _weighted_mae(ordered),
        "per_structure_diagnostics": [
            {"index": row["index"], **present_diagnostics(row)} for row in ordered
        ],
    }


def _delta_block(
    field: str,
    ref_by_index: Mapping[int, Mapping[str, Any]],
    new_by_index: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
) -> dict[str, Any]:
    per_structure = []
    deltas = []
    for index in indices:
        reference = ref_by_index[index][field]
        new = new_by_index[index][field]
        delta = new - reference
        deltas.append(delta)
        per_structure.append(
            {"index": index, "reference": reference, "new": new, "delta": delta}
        )
    stats = series_stats(deltas, UNITS[field])
    stats["worst_increase"] = stats.pop("max")
    stats["improvement_count"] = sum(1 for delta in deltas if delta < 0.0)
    stats["per_structure"] = per_structure
    return stats


def _pair_entry(
    reference: str,
    new: str,
    by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    ref_rows = by_variant[reference]
    new_rows = by_variant[new]
    ref_ids = [row["index"] for row in sorted(ref_rows, key=lambda row: row["index"])]
    new_ids = [row["index"] for row in sorted(new_rows, key=lambda row: row["index"])]
    missing_in_reference = sorted(set(new_ids) - set(ref_ids))
    missing_in_new = sorted(set(ref_ids) - set(new_ids))
    identical = not missing_in_reference and not missing_in_new
    entry: dict[str, Any] = {
        "reference": reference,
        "new": new,
        "identical_index_set": identical,
        "count_reference": len(ref_ids),
        "count_new": len(new_ids),
        "indices_reference": ref_ids,
        "indices_new": new_ids,
        "missing_in_reference": missing_in_reference,
        "missing_in_new": missing_in_new,
        "delta_convention": "new minus reference; positive difference = worse",
        "metrics": None,
        "forward_seconds": None,
    }
    if not identical:
        entry["note"] = (
            "index sets differ; missing IDs listed explicitly; "
            "pair stats omitted (no silent intersection, failures not dropped)"
        )
        return entry
    ref_by_index = {row["index"]: row for row in ref_rows}
    new_by_index = {row["index"]: row for row in new_rows}
    entry["count"] = len(ref_ids)
    entry["indices"] = ref_ids
    entry["metrics"] = {
        field: _delta_block(field, ref_by_index, new_by_index, ref_ids)
        for field in METRIC_FIELDS
    }
    entry["forward_seconds"] = _delta_block(
        TIME_FIELD, ref_by_index, new_by_index, ref_ids
    )
    entry["note"] = (
        "pair stats use the identical index set; time is separate from MAE; "
        "positive difference = worse"
    )
    return entry


def _mismatch_groups(
    records: Sequence[Mapping[str, Any]]
) -> dict[str, dict[int, list[str]]]:
    grouped: dict[str, dict[int, list[str]]] = {}
    for row in records:
        if not closure_mismatch(row):
            continue
        reasons = grouped.setdefault(row["source"], {}).setdefault(row["index"], [])
        for reason in closure_reasons(row):
            if reason not in reasons:
                reasons.append(reason)
    return grouped


def _closure_block(
    parsed: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    included: Sequence[Mapping[str, Any]],
    closure_only: bool,
) -> dict[str, Any]:
    mismatches = _mismatch_groups(records)
    mismatch_by_input = []
    excluded_ids_by_input = []
    all_ids: list[int] = []
    for item in parsed:
        path = item["path"]
        reasons = mismatches.get(path, {})
        ids = sorted(reasons)
        all_ids.extend(ids)
        mismatch_by_input.append(
            {"path": path, "ids": ids, "count": len(ids), "reasons": reasons}
        )
        excluded_ids_by_input.append(
            {
                "path": path,
                "ids": ids if closure_only else [],
                "count": len(ids) if closure_only else 0,
            }
        )
    return {
        "policy": "closure_only" if closure_only else "all",
        "threshold_ev": CLOSURE_THRESHOLD_EV,
        "rule": (
            "mismatch if label_closure_pass is false or "
            f"label_closure_fw10_ev>={CLOSURE_THRESHOLD_EV}"
        ),
        "mismatch_count": len(all_ids),
        "mismatch_indices": sorted(set(all_ids)),
        "mismatch_by_input": mismatch_by_input,
        "excluded_ids_by_input": excluded_ids_by_input,
        "n_records_parsed": len(records),
        "n_records_included": len(included),
    }


def summarize_documents(
    documents: Sequence[tuple[str, Any]],
    closure_only: bool = False,
) -> dict[str, Any]:
    if not documents:
        raise SummarizeError("at least one input JSON is required")
    parsed = [parse_document(path, data) for path, data in documents]
    records = [row for item in parsed for row in item["records"]]
    roots = {item["protocol"].get("dataset_root") for item in parsed} - {None}
    if len(roots) > 1:
        raise SummarizeError("cannot pair indices from different dataset roots")
    if closure_only and any(
        "label_closure_pass" not in row and "label_closure_fw10_ev" not in row
        for row in records
    ):
        raise SummarizeError("closure-only requires closure evidence for every record")
    failures = [row for item in parsed for row in item["failures"]]
    _reject_duplicates(records)
    _reject_inconsistent_counts(records)

    excluded_keys = {
        (row["source"], row["index"])
        for row in records
        if closure_only and closure_mismatch(row)
    }
    included = [
        row for row in records if (row["source"], row["index"]) not in excluded_keys
    ]

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in included:
        by_variant.setdefault(row["variant"], []).append(row)
    variant_names = sorted(by_variant)
    paired = [
        _pair_entry(reference, new, by_variant)
        for reference in variant_names
        for new in variant_names
        if reference != new
    ]

    diagnostics = []
    for row in sorted(
        records, key=lambda item: (item["source"], item["index"], item["variant"])
    ):
        diagnostics.append(
            {
                "index": row["index"],
                "variant": row["variant"],
                "source": row["source"],
                "included_in_aggregate": (row["source"], row["index"])
                not in excluded_keys,
                **present_diagnostics(row),
            }
        )

    inputs_out = []
    for item in parsed:
        path = item["path"]
        n_excluded = sum(1 for entry in excluded_keys if entry[0] == path)
        inputs_out.append(
            {
                "path": path,
                "protocol": item["protocol"],
                "n_results": len({row["index"] for row in item["records"]}),
                "n_result_rows": len(item["records"]),
                "n_failures": len(item["failures"]),
                "n_excluded_by_closure": n_excluded,
            }
        )

    return {
        "units": dict(UNITS),
        "difference_convention": (
            "paired deltas are new minus reference; positive difference = worse"
        ),
        "time_note": (
            "forward_seconds is aggregated separately from MAE/FW10 and is in seconds"
        ),
        "inputs": inputs_out,
        "closure": _closure_block(parsed, records, included, closure_only),
        "failure_counts": {
            "total": len(failures),
            "by_input": [
                {"path": item["path"], "n_failures": len(item["failures"])}
                for item in parsed
            ],
        },
        "failures": failures,
        "per_structure_diagnostics": diagnostics,
        "variants": {
            name: _variant_summary(name, by_variant[name]) for name in variant_names
        },
        "paired_differences": paired,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarize_experiment.py",
        description=(
            "Aggregate matched DeePTB LoopSCF per-structure evaluator JSON. "
            "MAE in eV; time separate; positive paired difference = worse."
        ),
        epilog=(
            "example:\n"
            "  python summarize_experiment.py --inputs arm_a.json arm_b.json "
            "--output report.json\n"
            "  python summarize_experiment.py --inputs arm_a.json --output "
            "report.json --closure-only\n\n"
            "Default includes all rows and labels closure-mismatch IDs. "
            "--closure-only drops those rows from stats only; excluded IDs are "
            "reported and original failures are preserved."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        metavar="JSON",
        nargs="+",
        required=True,
        help="one or more evaluator JSON files",
    )
    parser.add_argument(
        "--output",
        metavar="report.json",
        required=True,
        help="output report path",
    )
    parser.add_argument(
        "--closure-only",
        action="store_true",
        help=(
            "exclude rows with label_closure_pass=false or "
            f"label_closure_fw10_ev>={CLOSURE_THRESHOLD_EV} from aggregate stats; "
            "report excluded IDs; keep original failures"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        documents = []
        for raw in args.inputs:
            path = Path(raw)
            documents.append((str(path), _read_json(path)))
        report = summarize_documents(documents, closure_only=args.closure_only)
        _write_json(Path(args.output), report)
    except SummarizeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

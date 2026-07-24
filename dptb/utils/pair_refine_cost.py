#!/usr/bin/env python3
"""Estimate :class:`PairSO3RefineTP` path-space and memory cost.

Adapted from ``inputs/pair_refine_cost.py`` supplied with the independent
Review B of the 2026-07-24 ``lem_pair`` iteration.  The parser and counting
logic intentionally do not import e3nn, so this module remains useful for
cheap configuration checks.  :func:`validate_weight_numel_with_e3nn` is the
explicit optional cross-check against the installed e3nn implementation.

For an e3nn fully connected tensor product, every parity/triangle-compatible
``(input term 1, input term 2, output term)`` triple is one instruction.  Its
external-weight slice contains ``mul1 * mul2 * mulout`` scalar weights.  The
QHFlow mode retains only equal-multiplicity ``uuu`` instructions, whose slices
contain ``mul`` scalars.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Optional, Union


_TERM_RE = re.compile(r"^\s*(?:(\d+)x)?(\d+)([eo])\s*$")


@dataclass(frozen=True)
class Term:
    mul: int
    ell: int
    parity: int


def parse_irreps(text: str) -> list[Term]:
    """Parse the explicit ``mul x l parity`` subset used by DeePTB configs."""
    terms: list[Term] = []
    for raw in text.split("+"):
        match = _TERM_RE.match(raw)
        if match is None:
            raise ValueError(f"Cannot parse irrep term {raw!r} in {text!r}")
        mul_text, ell_text, parity_text = match.groups()
        mul = 1 if mul_text is None else int(mul_text)
        ell = int(ell_text)
        parity = 1 if parity_text == "e" else -1
        if mul <= 0:
            raise ValueError(f"Multiplicity must be positive, got {mul}")
        terms.append(Term(mul=mul, ell=ell, parity=parity))
    if not terms:
        raise ValueError("Irreps string is empty")
    return terms


def compatible(a: Term, b: Term, out: Term) -> bool:
    """Whether one term triple contributes an SO(3) tensor-product path."""
    return (
        abs(a.ell - b.ell) <= out.ell <= a.ell + b.ell
        and a.parity * b.parity == out.parity
    )


def fctp_path_count(node: list[Term], edge: list[Term]) -> int:
    """Count compatible FCTP instructions without importing e3nn."""
    return sum(
        1
        for a in node
        for b in node
        for out in edge
        if compatible(a, b, out)
    )


def fctp_weight_numel(node: list[Term], edge: list[Term]) -> int:
    """Count external FCTP scalars without importing e3nn."""
    return sum(
        a.mul * b.mul * out.mul
        for a in node
        for b in node
        for out in edge
        if compatible(a, b, out)
    )


def uuu_path_count(node: list[Term], edge: list[Term]) -> int:
    """Count compatible equal-multiplicity ``uuu`` instructions."""
    return sum(
        1
        for a in node
        for b in node
        for out in edge
        if a.mul == b.mul == out.mul and compatible(a, b, out)
    )


def uuu_weight_numel(node: list[Term], edge: list[Term]) -> int:
    """Count per-edge dynamic scalars in the QHFlow ``uuu`` tensor product."""
    return sum(
        out.mul
        for a in node
        for b in node
        for out in edge
        if a.mul == b.mul == out.mul and compatible(a, b, out)
    )


def shared_linear_params(
    irreps_in: list[Term],
    irreps_out: list[Term],
    *,
    biases: bool = True,
) -> int:
    """Count parameters in an e3nn shared ``o3.Linear``."""
    weights = sum(
        term_in.mul * term_out.mul
        for term_in in irreps_in
        for term_out in irreps_out
        if (
            term_in.ell == term_out.ell
            and term_in.parity == term_out.parity
        )
    )
    bias_params = (
        sum(
            term.mul
            for term in irreps_out
            if term.ell == 0 and term.parity == 1
        )
        if biases
        else 0
    )
    return weights + bias_params


def e3nn_fctp_weight_numel(
    node_irreps: Union[str, object],
    edge_irreps: Optional[Union[str, object]] = None,
) -> int:
    """Return e3nn's external-weight count for the same FCTP configuration."""
    from e3nn import o3

    node = o3.Irreps(node_irreps)
    edge = node if edge_irreps is None else o3.Irreps(edge_irreps)
    tensor_product = o3.FullyConnectedTensorProduct(
        node,
        node,
        edge,
        shared_weights=False,
        internal_weights=False,
    )
    return int(tensor_product.weight_numel)


def validate_weight_numel_with_e3nn(
    node_irreps: str,
    edge_irreps: Optional[str] = None,
) -> int:
    """Cross-check the dependency-free count and return the agreed value."""
    edge_text = node_irreps if edge_irreps is None else edge_irreps
    independent = fctp_weight_numel(
        parse_irreps(node_irreps),
        parse_irreps(edge_text),
    )
    e3nn_value = e3nn_fctp_weight_numel(node_irreps, edge_text)
    if independent != e3nn_value:
        raise AssertionError(
            "Pair-refine external-weight count disagrees with e3nn: "
            f"independent={independent}, e3nn={e3nn_value}, "
            f"node_irreps={node_irreps!r}, edge_irreps={edge_text!r}."
        )
    return independent


def scalar_0e_dim(terms: list[Term]) -> int:
    return sum(term.mul for term in terms if term.ell == 0 and term.parity == 1)


def human_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.3f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def estimate(
    irreps: str,
    edges: int,
    rank: int,
    dtype_bytes: int,
    internal_weights: bool,
    weight_mode: str = "full",
) -> dict[str, int | str]:
    if weight_mode not in {"full", "per_path", "qhflow"}:
        raise ValueError(
            "weight_mode must be 'full', 'per_path', or 'qhflow', "
            f"got {weight_mode!r}."
        )
    node = parse_irreps(irreps)
    edge = node
    weight_numel = fctp_weight_numel(node, edge)
    path_count = fctp_path_count(node, edge)
    qhflow_path_count = uuu_path_count(node, edge)
    qhflow_weight_numel = uuu_weight_numel(node, edge)
    condition_dim = 2 * scalar_0e_dim(node) + scalar_0e_dim(edge)
    condition_down_params = condition_dim * rank + rank
    dynamic_weight_numel = {
        "full": weight_numel,
        "per_path": path_count,
        "qhflow": qhflow_weight_numel,
    }[weight_mode]
    dynamic_up_params = (
        rank * dynamic_weight_numel + dynamic_weight_numel
    )
    static_params = (
        weight_numel
        if internal_weights and weight_mode != "qhflow"
        else 0
    )
    linear_pre_params = (
        shared_linear_params(node, node) if weight_mode == "qhflow" else 0
    )
    linear_post_params = (
        shared_linear_params(edge, edge) if weight_mode == "qhflow" else 0
    )
    total_params = (
        condition_down_params
        + dynamic_up_params
        + static_params
        + linear_pre_params
        + linear_post_params
    )
    per_edge_bytes = dynamic_weight_numel * dtype_bytes
    batch_bytes = per_edge_bytes * edges
    return {
        "irreps": irreps,
        "weight_mode": weight_mode,
        "weight_numel_per_edge": weight_numel,
        "path_count": path_count,
        "qhflow_path_count": qhflow_path_count,
        "qhflow_weight_numel_per_edge": qhflow_weight_numel,
        "dynamic_weight_numel_per_edge": dynamic_weight_numel,
        "condition_dim": condition_dim,
        "rank": rank,
        "condition_down_params": condition_down_params,
        "dynamic_up_params": dynamic_up_params,
        "static_params": static_params,
        "linear_pre_params": linear_pre_params,
        "linear_post_params": linear_post_params,
        "total_refiner_params": total_params,
        "dynamic_weight_buffer_per_edge": human_bytes(per_edge_bytes),
        "dynamic_weight_buffer_for_batch": human_bytes(batch_bytes),
        "per_path_buffer_per_edge": human_bytes(path_count * dtype_bytes),
        "per_path_buffer_for_batch": human_bytes(
            path_count * dtype_bytes * edges
        ),
        "qhflow_buffer_per_edge": human_bytes(
            qhflow_weight_numel * dtype_bytes
        ),
        "qhflow_buffer_for_batch": human_bytes(
            qhflow_weight_numel * dtype_bytes * edges
        ),
        "edges": edges,
        "dtype_bytes": dtype_bytes,
    }


def print_estimate(result: dict[str, int | str]) -> None:
    print(f"irreps: {result['irreps']}")
    print(f"weight mode: {result['weight_mode']}")
    print(f"FCTP instructions / edge: {result['path_count']:,}")
    print(f"FCTP external weights / edge: {result['weight_numel_per_edge']:,}")
    print(f"QHFlow uuu instructions / edge: {result['qhflow_path_count']:,}")
    print(
        "QHFlow uuu dynamic weights / edge: "
        f"{result['qhflow_weight_numel_per_edge']:,}"
    )
    print(
        "selected dynamic weights / edge: "
        f"{result['dynamic_weight_numel_per_edge']:,}"
    )
    print(f"condition dim -> rank: {result['condition_dim']} -> {result['rank']}")
    print(f"condition_down params: {result['condition_down_params']:,}")
    print(f"dynamic_up params: {result['dynamic_up_params']:,}")
    print(f"static TP params: {result['static_params']:,}")
    print(f"Linear_pre params: {result['linear_pre_params']:,}")
    print(f"Linear_post params: {result['linear_post_params']:,}")
    print(f"total refiner params: {result['total_refiner_params']:,}")
    print(
        "materialized selected dynamic-weight buffer: "
        f"{result['dynamic_weight_buffer_per_edge']} / edge; "
        f"{result['dynamic_weight_buffer_for_batch']} for "
        f"{result['edges']:,} edges"
    )
    print(
        "materialized per-path gate buffer: "
        f"{result['per_path_buffer_per_edge']} / edge; "
        f"{result['per_path_buffer_for_batch']} for "
        f"{result['edges']:,} edges"
    )
    print(
        "materialized QHFlow uuu buffer: "
        f"{result['qhflow_buffer_per_edge']} / edge; "
        f"{result['qhflow_buffer_for_batch']} for "
        f"{result['edges']:,} edges"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--irreps", help="e.g. '4x0e+4x1o+4x1e+4x2e'")
    parser.add_argument("--edges", type=int, default=5000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--no-static", action="store_true")
    parser.add_argument(
        "--weight-mode",
        choices=("full", "per_path", "qhflow"),
        default="full",
    )
    parser.add_argument(
        "--validate-e3nn",
        action="store_true",
        help="cross-check each dependency-free weight count with e3nn",
    )
    args = parser.parse_args()
    if args.edges < 0 or args.rank <= 0 or args.dtype_bytes <= 0:
        parser.error("edges must be non-negative; rank and dtype-bytes must be positive")

    examples = [
        "2x0e+2x1o+2x1e+2x2e",
        "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e",
        "8x0e+8x1o+8x2e+8x3o+8x4e",
        "32x0e+32x1o+32x2e+32x3o+32x4e+32x5o+32x6e",
    ]
    irreps_values = [args.irreps] if args.irreps else examples
    for index, irreps in enumerate(irreps_values):
        if index:
            print()
        if args.validate_e3nn:
            validate_weight_numel_with_e3nn(irreps)
        print_estimate(
            estimate(
                irreps,
                edges=args.edges,
                rank=args.rank,
                dtype_bytes=args.dtype_bytes,
                internal_weights=not args.no_static,
                weight_mode=args.weight_mode,
            )
        )


if __name__ == "__main__":
    main()

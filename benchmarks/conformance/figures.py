"""Generate the seven C1.2 conformance figures from ``cases.csv``."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_FILENAMES = (
    "01_condition_residual.png",
    "02_acceptance_domain_heatmap.png",
    "03_fixed_mu_gauge_curve.png",
    "04_qeq_alpha_gauge_scan.png",
    "05_scan_point_parity.png",
    "06_analytic_root_parity.png",
    "07_mutation_confusion_matrix.png",
)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return np.nan


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _condition_residual(rows: List[Dict[str, str]], path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["operator"] == "fixed_mu"
        and row["family"] != "mutation"
        and row["actual_status"] == "accept"
    ]
    condition = np.asarray([_number(r, "actual_condition") for r in selected])
    eigen = np.asarray(
        [_number(r, "generalized_eigen_residual") for r in selected]
    )
    trace = np.asarray([_number(r, "trace_residual") for r in selected])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog(condition, np.maximum(eigen, 1e-18), ".", ms=2.5, alpha=0.5, label="generalized eigen residual")
    ax.loglog(condition, np.maximum(trace, 1e-18), ".", ms=2.5, alpha=0.5, label="|Tr(DS)-N|")
    ax.axvline(1e12, color="black", ls="--", lw=1, label="certification ceiling")
    ax.set(xlabel="cond(S)", ylabel="absolute residual", title="Conditioning versus fixed-mu residual")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    _finish(fig, path)


def _acceptance_heatmap(rows: List[Dict[str, str]], path: Path) -> None:
    selected = [
        r
        for r in rows
        if r["family"] != "mutation"
        and np.isfinite(_number(r, "actual_condition"))
        and np.isfinite(_number(r, "declared_tolerance"))
    ]
    cond_edges = np.arange(0.0, 14.01, 2.0)
    tol_edges = np.arange(-12.0, -1.99, 2.0)
    total = np.zeros((len(tol_edges) - 1, len(cond_edges) - 1))
    accepted = np.zeros_like(total)
    for row in selected:
        x = np.log10(max(_number(row, "actual_condition"), 1.0))
        y = np.log10(_number(row, "declared_tolerance"))
        ix = np.searchsorted(cond_edges, x, side="right") - 1
        iy = np.searchsorted(tol_edges, y, side="right") - 1
        if 0 <= ix < total.shape[1] and 0 <= iy < total.shape[0]:
            total[iy, ix] += 1
            accepted[iy, ix] += row["actual_status"] == "accept"
    rate = np.divide(
        accepted,
        total,
        out=np.full_like(total, np.nan),
        where=total > 0,
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    image = ax.imshow(rate, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(total.shape[1]), [f"{a:.0f}-{b:.0f}" for a, b in zip(cond_edges[:-1], cond_edges[1:])])
    ax.set_yticks(np.arange(total.shape[0]), [f"{a:.0f}-{b:.0f}" for a, b in zip(tol_edges[:-1], tol_edges[1:])])
    ax.set(xlabel="log10 input condition bin", ylabel="log10 declared tolerance bin", title="Fail-closed acceptance domain")
    fig.colorbar(image, ax=ax, label="acceptance rate")
    _finish(fig, path)


def _gauge_curve(rows: List[Dict[str, str]], path: Path) -> None:
    selected = [
        r
        for r in rows
        if r["operator"] == "fixed_mu"
        and r["family"] != "mutation"
        and r["actual_status"] == "accept"
    ]
    c = np.asarray([_number(r, "gauge_c") for r in selected])
    dn = np.asarray([_number(r, "gauge_delta_N") for r in selected])
    dd = np.asarray([_number(r, "gauge_delta_D") for r in selected])
    order = np.argsort(c)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.semilogy(c[order], np.maximum(dn[order], 1e-18), ".", ms=3, alpha=0.6, label="max |Delta N|")
    ax.semilogy(c[order], np.maximum(dd[order], 1e-18), ".", ms=3, alpha=0.6, label="max |Delta D|")
    ax.set(xlabel="uniform shift c in H+cS, mu+c", ylabel="gauge difference", title="Fixed-mu energy-zero covariance")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    _finish(fig, path)


def _qeq_gauge(rows: List[Dict[str, str]], path: Path) -> None:
    selected = [
        r
        for r in rows
        if r["operator"] == "qeq" and r["family"] != "mutation"
    ]
    accepted = [r for r in selected if r["actual_status"] == "accept"]
    rejected = [r for r in selected if r["actual_status"] == "reject"]
    alpha = np.asarray([_number(r, "alpha") for r in accepted])
    delta = np.asarray([_number(r, "qeq_gauge_delta_q") for r in accepted])
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.loglog(alpha, np.maximum(delta, 1e-18), ".", ms=3, alpha=0.6, label="accepted: max |q(alpha)-q(0)|")
    if rejected:
        rejected_alpha = np.asarray([_number(r, "alpha") for r in rejected])
        ax.scatter(rejected_alpha, np.full_like(rejected_alpha, 3e-1), marker="x", s=12, color="tab:red", label="rejected input")
    ax.axvspan(1e12, 1e16, color="tab:red", alpha=0.12, label="documented float64 loss region")
    ax.set(xlabel="uniform gauge alpha in J+alpha 11^T", ylabel="charge drift / rejection marker", title="QEq uniform-gauge scan")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    _finish(fig, path)


def _scan_parity(rows: List[Dict[str, str]], path: Path) -> None:
    selected = [
        r
        for r in rows
        if r["operator"] == "fixed_mu"
        and r["family"] != "mutation"
        and r["actual_status"] == "accept"
    ]
    pairs = (
        ("electron_count", "scan_electron_count", "N"),
        ("dos_like_response", "scan_dos_like_response", "dN/dmu"),
        ("band_energy", "scan_band_energy", "band E"),
        ("band_grand_energy", "scan_band_grand_energy", "band grand ledger"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    for ax, (x_key, y_key, label) in zip(axes.ravel(), pairs):
        x = np.asarray([_number(r, x_key) for r in selected])
        y = np.asarray([_number(r, y_key) for r in selected])
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        ax.plot(x, y, ".", ms=2.5, alpha=0.5)
        if x.size:
            lower = min(float(np.min(x)), float(np.min(y)))
            upper = max(float(np.max(x)), float(np.max(y)))
            ax.plot([lower, upper], [lower, upper], "k--", lw=0.8)
        ax.set(xlabel=f"point {label}", ylabel=f"scan {label}")
        ax.grid(True, alpha=0.2)
    fig.suptitle("fixed_mu_scan versus point evaluation parity")
    _finish(fig, path)


def _analytic_parity(rows: List[Dict[str, str]], path: Path) -> None:
    selected = [
        r
        for r in rows
        if r["actual_status"] == "accept"
        and np.isfinite(_number(r, "analytic_reference"))
        and np.isfinite(_number(r, "numeric_value"))
    ]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for operator, marker in (("qeq", "."), ("scf", "x")):
        group = [r for r in selected if r["operator"] == operator]
        x = np.asarray([_number(r, "analytic_reference") for r in group])
        y = np.asarray([_number(r, "numeric_value") for r in group])
        ax.plot(x, y, marker, ls="none", ms=3.5, alpha=0.6, label=operator)
    if selected:
        values = np.asarray(
            [
                _number(r, key)
                for r in selected
                for key in ("analytic_reference", "numeric_value")
            ]
        )
        ax.plot([np.min(values), np.max(values)], [np.min(values), np.max(values)], "k--", lw=0.8)
    ax.set(xlabel="analytic / independent reference", ylabel="operator result", title="Independent-root parity")
    ax.grid(True, alpha=0.2)
    ax.legend()
    _finish(fig, path)


def _confusion(rows: List[Dict[str, str]], path: Path) -> None:
    selected = [r for r in rows if r["family"] == "mutation"]
    labels = ("accept", "reject")
    matrix = np.zeros((2, 2), dtype=int)
    for row in selected:
        matrix[labels.index(row["expected_status"]), labels.index(row["actual_status"])] += 1
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    for iy in range(2):
        for ix in range(2):
            ax.text(ix, iy, str(matrix[iy, ix]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(2), labels)
    ax.set_yticks(range(2), labels)
    ax.set(xlabel="validator verdict", ylabel="expected status", title="Mutation confusion matrix")
    fig.colorbar(image, ax=ax, label="cases")
    _finish(fig, path)


def generate_figures(
    cases_csv: str | Path, output_dir: str | Path
) -> List[Path]:
    """Write all seven C1.2 PNG figures and a compact manifest."""

    rows = _read_rows(Path(cases_csv))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    builders = (
        _condition_residual,
        _acceptance_heatmap,
        _gauge_curve,
        _qeq_gauge,
        _scan_parity,
        _analytic_parity,
        _confusion,
    )
    paths = []
    for filename, builder in zip(FIGURE_FILENAMES, builders):
        path = output / filename
        builder(rows, path)
        paths.append(path)
    (output / "figures.json").write_text(
        json.dumps(
            {"cases_csv": str(cases_csv), "figures": [p.name for p in paths]},
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in generate_figures(args.cases, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()

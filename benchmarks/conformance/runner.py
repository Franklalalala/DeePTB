"""Run the C1 numerical conformance and adversarial benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from dptb.nnops.fixed_mu_operator import (
    fixed_mu_observables,
    fixed_mu_scan,
    validate_conservation,
)
from dptb.nnops.fixed_mu_scf_operator import fixed_mu_electrostatic_scf
from dptb.nnops.qeq_operator import solve_qeq, validate_qeq_result

try:
    from .generators import (
        GENERATOR_VERSION,
        ConformanceCase,
        generate_cases,
        sum_zero_basis,
    )
except ImportError:  # pragma: no cover - direct script execution
    from generators import (  # type: ignore
        GENERATOR_VERSION,
        ConformanceCase,
        generate_cases,
        sum_zero_basis,
    )


CSV_FIELDS = (
    "case_id",
    "seed",
    "operator",
    "family",
    "mutation_kind",
    "generator_version",
    "code_commit",
    "dtype",
    "expected_status",
    "actual_status",
    "validator_pass",
    "verdict_match",
    "exception_class",
    "reject_reason",
    "matrix_path",
    "matrix_prefix",
    "n_orb",
    "n_site",
    "n_k",
    "mu",
    "kT",
    "spin_degeneracy",
    "mixing",
    "mixing_step",
    "max_iter",
    "charge_tol",
    "total_charge",
    "target_condition",
    "actual_condition",
    "alpha",
    "declared_tolerance",
    "generalized_eigen_residual",
    "trace_residual",
    "density_hermiticity_error",
    "electron_count",
    "dos_like_response",
    "band_energy",
    "band_free_energy",
    "band_grand_energy",
    "scan_electron_count",
    "scan_dos_like_response",
    "scan_band_energy",
    "scan_band_free_energy",
    "scan_band_grand_energy",
    "scan_parity_max",
    "gauge_c",
    "gauge_delta_N",
    "gauge_delta_D",
    "qeq_charge_residual",
    "qeq_tangent_residual",
    "qeq_gauge_delta_q",
    "qeq_energy",
    "qeq_lambda",
    "scf_iterations",
    "scf_final_residual",
    "analytic_reference",
    "numeric_value",
    "max_abs_error",
    "runtime_ms",
)


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _float(value: Any) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def _max_abs(value: Any) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _base_row(case: ConformanceCase, commit: str) -> Dict[str, Any]:
    payload = case.payload
    arrays = [
        value
        for value in payload.values()
        if isinstance(value, np.ndarray) and value.size
    ]
    dtype = str(np.result_type(*[x.dtype for x in arrays])) if arrays else ""
    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "case_id": case.case_id,
            "seed": case.seed,
            "operator": case.operator,
            "family": case.family,
            "mutation_kind": case.mutation_kind,
            "generator_version": GENERATOR_VERSION,
            "code_commit": commit,
            "dtype": dtype,
            "expected_status": case.expected_status,
            "target_condition": payload.get("target_condition", ""),
            "actual_condition": payload.get("actual_condition", ""),
            "alpha": payload.get("alpha", ""),
            "declared_tolerance": payload.get("declared_tolerance", ""),
            "gauge_c": payload.get("gauge_c", ""),
        }
    )
    return row


def _fixed_matrices(payload: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    return {
        "H": np.asarray(payload["h"]),
        "S": np.asarray(payload["s"]),
        "mu_grid": np.asarray(payload["mu_grid"]),
        **(
            {"k_weights": np.asarray(payload["k_weights"])}
            if payload.get("k_weights") is not None
            else {}
        ),
    }


def _generalized_residual(
    h: np.ndarray, s: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray
) -> float:
    lhs = h @ eigvecs
    rhs = (s @ eigvecs) * eigvals[..., None, :]
    scale = max(_max_abs(lhs), _max_abs(rhs), 1.0)
    return _max_abs(lhs - rhs) / scale


def _scan_parity(point: Any, scan: Any, center: int = 1) -> float:
    differences = [
        _max_abs(np.asarray(scan.electron_count)[center] - point.electron_count),
        _max_abs(
            np.asarray(scan.dos_like_response)[center]
            - point.dos_like_response
        ),
    ]
    for name in (
        "band_energy",
        "minus_t_s",
        "band_free_energy",
        "band_grand_energy",
    ):
        differences.append(
            _max_abs(
                np.asarray(getattr(scan.energies, name))[center]
                - getattr(point.energies, name)
            )
        )
    return max(differences)


def _run_fixed(
    case: ConformanceCase, row: MutableMapping[str, Any]
) -> tuple[Dict[str, np.ndarray], bool]:
    p = case.payload
    h = np.asarray(p["h"])
    s = np.asarray(p["s"])
    kwargs = {
        "mu": p["mu"],
        "kT": p["kT"],
        "spin_degeneracy": p["spin_degeneracy"],
        "k_weights": p["k_weights"],
        "k_axis": p["k_axis"],
        "normalize_k_weights": p["normalize_k_weights"],
    }
    result = fixed_mu_observables(h, s, **kwargs)
    validate_conservation(
        result,
        h=h,
        s=s,
        atol=p["declared_tolerance"],
        expected_mu=p["mu"],
        expected_kT=p["kT"],
        expected_spin_degeneracy=p["spin_degeneracy"],
        expected_k_weights=p["k_weights"],
        expected_k_axis=p["k_axis"],
        expected_normalize_k_weights=p["normalize_k_weights"],
    )
    scan = fixed_mu_scan(
        h,
        s,
        p["mu_grid"],
        kT=p["kT"],
        spin_degeneracy=p["spin_degeneracy"],
        k_weights=p["k_weights"],
        k_axis=p["k_axis"],
        normalize_k_weights=p["normalize_k_weights"],
    )
    shifted = fixed_mu_observables(
        h + p["gauge_c"] * s,
        s,
        **{**kwargs, "mu": p["mu"] + p["gauge_c"]},
    )
    row.update(
        {
            "n_orb": h.shape[-1],
            "n_k": h.shape[0] if h.ndim == 3 else 1,
            "mu": p["mu"],
            "kT": p["kT"],
            "spin_degeneracy": p["spin_degeneracy"],
            "actual_condition": _max_abs(result.overlap_condition),
            "generalized_eigen_residual": _generalized_residual(
                h, s, result.eigvals, result.eigvecs
            ),
            "trace_residual": _max_abs(
                result.conservation.electron_count_residual
            ),
            "density_hermiticity_error": _max_abs(
                result.conservation.density_hermiticity_error
            ),
            "electron_count": _float(result.electron_count),
            "dos_like_response": _float(result.dos_like_response),
            "band_energy": _float(result.energies.band_energy),
            "band_free_energy": _float(result.energies.band_free_energy),
            "band_grand_energy": _float(result.energies.band_grand_energy),
            "scan_electron_count": _float(
                np.asarray(scan.electron_count)[1]
            ),
            "scan_dos_like_response": _float(
                np.asarray(scan.dos_like_response)[1]
            ),
            "scan_band_energy": _float(
                np.asarray(scan.energies.band_energy)[1]
            ),
            "scan_band_free_energy": _float(
                np.asarray(scan.energies.band_free_energy)[1]
            ),
            "scan_band_grand_energy": _float(
                np.asarray(scan.energies.band_grand_energy)[1]
            ),
            "scan_parity_max": _scan_parity(result, scan),
            "gauge_delta_N": _max_abs(
                shifted.electron_count - result.electron_count
            ),
            "gauge_delta_D": _max_abs(shifted.density - result.density),
        }
    )
    matrices = _fixed_matrices(p)
    matrices.update(
        {
            "eps": np.asarray(result.eigvals),
            "C": np.asarray(result.eigvecs),
            "f": np.asarray(result.occupations),
            "df_dmu": np.asarray(result.occupation_response),
            "D": np.asarray(result.density),
            "dD_dmu": np.asarray(result.density_response),
        }
    )
    return matrices, True


def _independent_qeq(
    chi: np.ndarray, kernel: np.ndarray, total_charge: float
) -> np.ndarray:
    n = chi.shape[-1]
    if n == 1:
        return np.asarray([total_charge], dtype=np.float64)
    z = sum_zero_basis(n)
    q0 = np.full(n, total_charge / n, dtype=np.float64)
    tangent = z.T @ kernel @ z
    reduced_gradient = z.T @ (chi + kernel @ q0)
    return q0 - z @ np.linalg.solve(tangent, reduced_gradient)


def _run_qeq(
    case: ConformanceCase, row: MutableMapping[str, Any]
) -> tuple[Dict[str, np.ndarray], bool]:
    p = case.payload
    chi = np.asarray(p["electronegativity"])
    kernel = np.asarray(p["hardness_kernel"])
    total_charge = p["total_charge"]
    result = solve_qeq(
        chi,
        kernel,
        total_charge=total_charge,
        residual_tol=p["declared_tolerance"],
    )
    validate_qeq_result(
        result,
        atol=p["declared_tolerance"],
        expected_electronegativity=chi,
        expected_hardness_kernel=kernel,
        expected_total_charge=total_charge,
    )
    reference = _independent_qeq(
        chi, np.asarray(p["base_hardness_kernel"]), total_charge
    )
    baseline = solve_qeq(
        chi,
        np.asarray(p["base_hardness_kernel"]),
        total_charge=total_charge,
        residual_tol=p["declared_tolerance"],
    )
    row.update(
        {
            "n_site": chi.shape[-1],
            "total_charge": total_charge,
            "actual_condition": _max_abs(
                result.diagnostics.constrained_condition
            ),
            "qeq_charge_residual": _max_abs(
                result.diagnostics.charge_residual
            ),
            "qeq_tangent_residual": _max_abs(
                result.diagnostics.stationarity_tangent_max_abs
            ),
            "qeq_gauge_delta_q": _max_abs(
                result.charges - baseline.charges
            ),
            "qeq_energy": _float(result.energy),
            "qeq_lambda": _float(result.lagrange_multiplier),
            "analytic_reference": _float(reference[0]),
            "numeric_value": _float(result.charges[0]),
            "max_abs_error": _max_abs(result.charges - reference),
        }
    )
    return {
        "chi": chi,
        "J": kernel,
        "J_base": np.asarray(p["base_hardness_kernel"]),
        "q": np.asarray(result.charges),
        "q_independent": reference,
    }, True


def _one_level_reference(payload: Mapping[str, Any]) -> float:
    epsilon = float(np.asarray(payload["h0"])[0, 0])
    gamma = float(np.asarray(payload["coulomb_kernel"])[0, 0])
    n_ref = float(np.asarray(payload["reference_populations"])[0])
    mu = float(payload["mu"])
    kT = float(payload["kT"])
    degeneracy = float(payload["spin_degeneracy"])

    def equation(population: float) -> float:
        effective_energy = epsilon + gamma * (population - n_ref)
        return population - degeneracy * expit(
            -(effective_energy - mu) / kT
        )

    return float(brentq(equation, 0.0, degeneracy, xtol=1.0e-14))


def _run_scf(
    case: ConformanceCase, row: MutableMapping[str, Any]
) -> tuple[Dict[str, np.ndarray], bool]:
    p = case.payload
    result = fixed_mu_electrostatic_scf(
        p["h0"],
        p["s"],
        mu=p["mu"],
        kT=p["kT"],
        ao_atom_index=p["ao_atom_index"],
        reference_populations=p["reference_populations"],
        coulomb_kernel=p["coulomb_kernel"],
        mixing=p["mixing"],
        mixing_step=p["mixing_step"],
        n_history=p["n_history"],
        mixing_period=p["mixing_period"],
        max_iter=p["max_iter"],
        charge_tol=p["charge_tol"],
        spin_degeneracy=p["spin_degeneracy"],
    )
    validate_conservation(
        result.fixed_mu_result,
        atol=max(float(p["charge_tol"]), 1.0e-10),
        expected_mu=p["mu"],
        expected_kT=p["kT"],
        expected_spin_degeneracy=p["spin_degeneracy"],
    )
    row.update(
        {
            "n_orb": np.asarray(p["h0"]).shape[-1],
            "n_site": np.asarray(p["reference_populations"]).size,
            "mu": p["mu"],
            "kT": p["kT"],
            "spin_degeneracy": p["spin_degeneracy"],
            "mixing": p["mixing"],
            "mixing_step": p["mixing_step"],
            "max_iter": p["max_iter"],
            "charge_tol": p["charge_tol"],
            "scf_iterations": result.iterations,
            "scf_final_residual": float(result.residual_history[-1]),
            "electron_count": _float(
                result.fixed_mu_result.electron_count
            ),
        }
    )
    if case.family == "one_level":
        reference = _one_level_reference(p)
        row["analytic_reference"] = reference
        row["numeric_value"] = _float(
            result.fixed_mu_result.electron_count
        )
        row["max_abs_error"] = abs(
            _float(result.fixed_mu_result.electron_count) - reference
        )
    return {
        "H0": np.asarray(p["h0"]),
        "S": np.asarray(p["s"]),
        "ao_atom_index": np.asarray(p["ao_atom_index"]),
        "n_ref": np.asarray(p["reference_populations"]),
        "K": np.asarray(p["coulomb_kernel"]),
        "q": np.asarray(result.q),
        "phi": np.asarray(result.phi),
        "residual_history": np.asarray(result.residual_history),
        "D": np.asarray(result.fixed_mu_result.density),
    }, True


def _mutate_fixed(kind: str) -> Dict[str, np.ndarray]:
    h = np.diag([-0.2, 0.3])
    s = np.eye(2)
    if kind == "wrong_shape":
        fixed_mu_observables(np.zeros((2, 3)), s, mu=0.0)
    elif kind == "nan_inf":
        h[0, 0] = np.nan
        fixed_mu_observables(h, s, mu=0.0)
    elif kind == "non_hermitian":
        h[0, 1] = 1.0
        fixed_mu_observables(h, s, mu=0.0)
    elif kind == "non_spd":
        s[1, 1] = -1.0
        fixed_mu_observables(h, s, mu=0.0)
    else:
        result = fixed_mu_observables(h, s, mu=0.0, kT=0.1)
        if kind == "truncated_eigenbasis":
            bad = replace(
                result,
                eigvals=np.asarray(result.eigvals)[:1],
                eigvecs=np.asarray(result.eigvecs)[:, :1],
            )
            validate_conservation(bad, h=h, s=s)
        elif kind == "wrong_request":
            validate_conservation(result, expected_mu=1.0)
        elif kind == "serialized_field_rewrite":
            bad = replace(
                result, electron_count=result.electron_count + 0.25
            )
            restored = pickle.loads(pickle.dumps(bad))
            validate_conservation(restored, h=h, s=s)
        elif kind == "self_reported_loose_tolerance":
            bad = replace(result, hermitian_tol=1.0e-2)
            validate_conservation(bad, h=h, s=s)
        else:  # pragma: no cover
            raise ValueError(f"unknown fixed-mu mutation {kind}")
    return {"H": h, "S": s}


def _mutate_qeq(kind: str) -> Dict[str, np.ndarray]:
    chi = np.asarray([1.0, 3.0])
    kernel = np.asarray([[5.0, 1.0], [1.0, 7.0]])
    if kind == "wrong_shape":
        solve_qeq(chi, np.zeros((2, 3)))
    elif kind == "nan_inf":
        chi[0] = np.inf
        solve_qeq(chi, kernel)
    elif kind == "non_hermitian":
        kernel[0, 1] += 0.5
        solve_qeq(chi, kernel)
    elif kind == "non_spd":
        solve_qeq(chi, -np.eye(2))
    else:
        result = solve_qeq(chi, kernel, total_charge=0.5)
        if kind == "wrong_request":
            validate_qeq_result(
                result, expected_electronegativity=chi + 1.0
            )
        elif kind == "serialized_field_rewrite":
            bad = replace(result, energy=result.energy + 0.25)
            restored = pickle.loads(pickle.dumps(bad))
            validate_qeq_result(restored)
        elif kind == "self_reported_loose_tolerance":
            bad = replace(result, residual_tol=1.0e-2)
            validate_qeq_result(bad)
        else:  # pragma: no cover
            raise ValueError(f"unknown QEq mutation {kind}")
    return {"chi": chi, "J": kernel}


def _mutate_scf(kind: str) -> Dict[str, np.ndarray]:
    h = np.diag([-0.2, 0.3])
    s = np.eye(2)
    ao = np.arange(2)
    kernel = 0.1 * np.eye(2)
    if kind == "wrong_shape":
        ao = np.asarray([0])
    elif kind == "nan_inf":
        kernel[0, 0] = np.nan
    elif kind == "non_hermitian":
        kernel[0, 1] = 0.2
    else:  # pragma: no cover
        raise ValueError(f"mutation {kind} is not an SCF mutation")
    fixed_mu_electrostatic_scf(
        h,
        s,
        mu=0.0,
        kT=0.1,
        ao_atom_index=ao,
        reference_populations=np.ones(2),
        coulomb_kernel=kernel,
    )
    return {"H0": h, "S": s, "ao_atom_index": ao, "K": kernel}


def _run_mutation(
    case: ConformanceCase, row: MutableMapping[str, Any]
) -> tuple[Dict[str, np.ndarray], bool]:
    if case.operator == "fixed_mu":
        return _mutate_fixed(case.mutation_kind), False
    if case.operator == "qeq":
        return _mutate_qeq(case.mutation_kind), False
    return _mutate_scf(case.mutation_kind), False


def _mutation_matrices(case: ConformanceCase) -> Dict[str, np.ndarray]:
    if case.operator == "qeq":
        chi = np.asarray([1.0, 3.0])
        kernel = np.asarray([[5.0, 1.0], [1.0, 7.0]])
        if case.mutation_kind == "wrong_shape":
            kernel = np.zeros((2, 3))
        elif case.mutation_kind == "nan_inf":
            chi[0] = np.inf
        elif case.mutation_kind == "non_hermitian":
            kernel[0, 1] += 0.5
        elif case.mutation_kind == "non_spd":
            kernel = -np.eye(2)
        return {"chi": chi, "J": kernel}
    h = np.diag([-0.2, 0.3])
    s = np.eye(2)
    if case.operator == "scf":
        ao = np.arange(2)
        kernel = 0.1 * np.eye(2)
        if case.mutation_kind == "wrong_shape":
            ao = np.asarray([0])
        elif case.mutation_kind == "nan_inf":
            kernel[0, 0] = np.nan
        elif case.mutation_kind == "non_hermitian":
            kernel[0, 1] = 0.2
        return {"H0": h, "S": s, "ao_atom_index": ao, "K": kernel}
    if case.mutation_kind == "wrong_shape":
        h = np.zeros((2, 3))
    elif case.mutation_kind == "nan_inf":
        h[0, 0] = np.nan
    elif case.mutation_kind == "non_hermitian":
        h[0, 1] = 1.0
    elif case.mutation_kind == "non_spd":
        s[1, 1] = -1.0
    return {"H": h, "S": s}


def _case_matrices(case: ConformanceCase) -> Dict[str, np.ndarray]:
    if case.family == "mutation":
        return _mutation_matrices(case)
    payload = case.payload
    return {
        key: np.asarray(value)
        for key, value in payload.items()
        if isinstance(value, np.ndarray)
    }


def _execute_case(
    case: ConformanceCase, commit: str
) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    row = _base_row(case, commit)
    matrices: Dict[str, np.ndarray] = _case_matrices(case)
    started = time.perf_counter()
    validator_pass = False
    try:
        if case.family == "mutation":
            matrices, validator_pass = _run_mutation(case, row)
        elif case.operator == "fixed_mu":
            matrices, validator_pass = _run_fixed(case, row)
        elif case.operator == "qeq":
            matrices, validator_pass = _run_qeq(case, row)
        elif case.operator == "scf":
            matrices, validator_pass = _run_scf(case, row)
        else:  # pragma: no cover
            raise ValueError(f"unknown operator {case.operator}")
        actual_status = "accept"
    except Exception as exc:
        actual_status = "reject"
        row["exception_class"] = type(exc).__name__
        row["reject_reason"] = str(exc).replace("\n", " ")[:1000]
    row["actual_status"] = actual_status
    row["validator_pass"] = int(validator_pass)
    row["verdict_match"] = int(actual_status == case.expected_status)
    row["runtime_ms"] = 1000.0 * (time.perf_counter() - started)
    return row, matrices


def _write_npz_shard(
    output_dir: Path,
    shard_index: int,
    records: Iterable[tuple[Dict[str, Any], Dict[str, np.ndarray]]],
) -> None:
    arrays: Dict[str, np.ndarray] = {}
    records = list(records)
    filename = f"matrices-{shard_index:05d}.npz"
    for row, matrices in records:
        prefix = str(row["case_id"]).replace("-", "_")
        for name, value in matrices.items():
            arrays[f"{prefix}__{name}"] = np.asarray(value)
        row["matrix_path"] = filename
        row["matrix_prefix"] = prefix
    np.savez_compressed(output_dir / filename, **arrays)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_reproducer(
    output_dir: Path,
    case: ConformanceCase,
    row: Mapping[str, Any],
    matrices: Mapping[str, np.ndarray],
) -> None:
    repro_dir = output_dir / "bug_reproducers" / case.case_id
    repro_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(repro_dir / "payload.npz", **matrices)
    metadata = {
        "case": {
            key: value
            for key, value in asdict(case).items()
            if key != "payload"
        },
        "row": dict(row),
        "note": (
            "Minimal dimension-two mutation accepted contrary to the "
            "fail-closed contract. Re-run this case with runner.py --case-id."
        ),
    }
    (repro_dir / "README.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def run_conformance(
    n_cases: int,
    seed: int,
    output_dir: str | Path,
    *,
    shard_size: int = 500,
    case_id: str | None = None,
) -> Dict[str, Any]:
    """Run cases, write the C1 CSV/NPZ dataset, and return summary counts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    commit = _git_commit(repo_root)
    rows: List[Dict[str, Any]] = []
    shard: List[tuple[Dict[str, Any], Dict[str, np.ndarray]]] = []
    selected_cases: List[ConformanceCase] = []
    for case in generate_cases(n_cases, seed):
        if case_id is None or case.case_id == case_id:
            selected_cases.append(case)
    if case_id is not None and not selected_cases:
        raise ValueError(
            f"{case_id!r} is not in the first {n_cases} generated cases"
        )

    started = time.perf_counter()
    shard_index = 0
    shard_count = 0
    bugs = 0
    for case in selected_cases:
        row, matrices = _execute_case(case, commit)
        rows.append(row)
        shard.append((row, matrices))
        if (
            case.expected_status == "reject"
            and row["actual_status"] == "accept"
        ):
            bugs += 1
            _write_reproducer(output, case, row, matrices)
        if len(shard) >= shard_size:
            _write_npz_shard(output, shard_index, shard)
            shard.clear()
            shard_index += 1
            shard_count += 1
    if shard:
        _write_npz_shard(output, shard_index, shard)
        shard_count += 1

    csv_path = output / "cases.csv"
    _write_csv(csv_path, rows)
    matches = sum(int(row["verdict_match"]) for row in rows)
    accepts = sum(row["actual_status"] == "accept" for row in rows)
    summary = {
        "n_cases": len(rows),
        "seed": int(seed),
        "generator_version": GENERATOR_VERSION,
        "code_commit": commit,
        "accept_count": accepts,
        "reject_count": len(rows) - accepts,
        "verdict_match_count": matches,
        "verdict_mismatch_count": len(rows) - matches,
        "expected_reject_accepted_count": bugs,
        "elapsed_seconds": time.perf_counter() - started,
        "cases_csv": str(csv_path),
        "matrix_shards": shard_count,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=730)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/conformance")
    )
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--case-id")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_conformance(
        args.n_cases,
        args.seed,
        args.output_dir,
        shard_size=args.shard_size,
        case_id=args.case_id,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

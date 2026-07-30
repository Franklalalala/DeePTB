"""Deterministic generators for the C1 operator-conformance dataset.

The generated matrices are deliberately synthetic.  They exercise numerical
and serialization contracts; they are not models of materials or
electrochemical interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator

import numpy as np


GENERATOR_VERSION = "c1.1-v1"
_KTS = (0.0, 0.02585, 0.1)
_DEGENERACIES = (1.0, 2.0)
_MUTATIONS = (
    "wrong_shape",
    "nan_inf",
    "non_hermitian",
    "non_spd",
    "truncated_eigenbasis",
    "wrong_request",
    "serialized_field_rewrite",
    "self_reported_loose_tolerance",
)


@dataclass(frozen=True)
class ConformanceCase:
    """One reproducible operator request or adversarial mutation."""

    case_id: str
    seed: int
    operator: str
    family: str
    expected_status: str
    payload: Dict[str, Any]
    mutation_kind: str = ""


def _case_rng(seed: int, index: int) -> tuple[np.random.Generator, int]:
    state = np.random.SeedSequence([int(seed), int(index)]).generate_state(
        2, dtype=np.uint32
    )
    case_seed = (int(state[0]) << 32) | int(state[1])
    return np.random.default_rng(case_seed), case_seed


def _random_unitary(
    rng: np.random.Generator, n: int, *, complex_: bool
) -> np.ndarray:
    a = rng.normal(size=(n, n))
    if complex_:
        a = a + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(a)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return q * phases.conj()[None, :]


def _spd_with_condition(
    rng: np.random.Generator,
    n: int,
    condition: float,
    *,
    complex_: bool = False,
) -> np.ndarray:
    if n == 1:
        return np.ones((1, 1), dtype=np.complex128 if complex_ else np.float64)
    u = _random_unitary(rng, n, complex_=complex_)
    eigvals = np.geomspace(1.0, float(condition), n)
    return (u * eigvals[None, :]) @ u.conj().T


def _hermitian_from_generalized_spectrum(
    rng: np.random.Generator,
    overlap: np.ndarray,
    eigvals: np.ndarray,
    *,
    complex_: bool,
) -> np.ndarray:
    s_eig, s_vec = np.linalg.eigh(overlap)
    s_half = (s_vec * np.sqrt(s_eig)[None, :]) @ s_vec.conj().T
    u = _random_unitary(rng, overlap.shape[0], complex_=complex_)
    orthogonal_h = (u * eigvals[None, :]) @ u.conj().T
    h = s_half @ orthogonal_h @ s_half
    return 0.5 * (h + h.conj().T)


def _soc_style_spectrum(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, str]:
    if n < 2:
        return rng.uniform(-1.5, 1.5, size=n), "complex"
    pairs = n // 2
    base = np.sort(rng.uniform(-1.5, 1.5, size=pairs))
    split = rng.uniform(0.0, 0.15, size=pairs)
    eps = np.column_stack((base - split, base + split)).ravel()
    if n % 2:
        eps = np.concatenate((eps, rng.uniform(-1.5, 1.5, size=1)))
    return np.sort(eps), "soc"


def _fixed_mu_case(
    rng: np.random.Generator, case_id: str, case_seed: int, index: int
) -> ConformanceCase:
    n = 1 + int((rng.random() ** 2) * 32)
    nk = 1 + int(rng.integers(0, 8))
    matrix_style = ("real", "complex", "soc")[index % 3]
    complex_ = matrix_style != "real"
    outside_gate = index % 11 == 0
    log_condition = rng.uniform(12.05, 14.0) if outside_gate else rng.uniform(0.0, 11.85)
    target_condition = 10.0**log_condition

    hs = []
    ss = []
    for _ in range(nk):
        s = _spd_with_condition(rng, n, target_condition, complex_=complex_)
        if matrix_style == "soc":
            eps, _ = _soc_style_spectrum(rng, n)
        else:
            eps = np.sort(rng.uniform(-1.5, 1.5, size=n))
        h = _hermitian_from_generalized_spectrum(
            rng, s, eps, complex_=complex_
        )
        hs.append(h)
        ss.append(s)

    h_arr = np.stack(hs) if nk > 1 else hs[0]
    s_arr = np.stack(ss) if nk > 1 else ss[0]
    actual_condition = float(
        max(np.linalg.cond(s_one) for s_one in ss)
    )
    k_weights = rng.uniform(0.1, 2.0, size=nk) if nk > 1 else None
    k_axis = 0 if nk > 1 else None
    mu = float(rng.uniform(-2.0, 2.0))
    declared_tolerance = float(10.0 ** rng.uniform(-12.0, -6.0))
    expected = "reject" if actual_condition > 1.0e12 else "accept"
    return ConformanceCase(
        case_id=case_id,
        seed=case_seed,
        operator="fixed_mu",
        family=f"random_{matrix_style}",
        expected_status=expected,
        payload={
            "h": h_arr,
            "s": s_arr,
            "mu": mu,
            "kT": float(_KTS[index % len(_KTS)]),
            "spin_degeneracy": float(
                _DEGENERACIES[(index // len(_KTS)) % 2]
            ),
            "k_weights": k_weights,
            "k_axis": k_axis,
            "normalize_k_weights": True,
            "target_condition": target_condition,
            "actual_condition": actual_condition,
            "matrix_style": matrix_style,
            "declared_tolerance": declared_tolerance,
            "gauge_c": float(rng.uniform(-10.0, 10.0)),
            "mu_grid": np.asarray(
                [mu - 0.25, mu, mu + 0.25], dtype=np.float64
            ),
        },
    )


def sum_zero_basis(n: int) -> np.ndarray:
    """Return a deterministic orthonormal basis for ``sum(q) == 0``."""

    if n <= 1:
        return np.zeros((n, 0), dtype=np.float64)
    basis = np.zeros((n, n - 1), dtype=np.float64)
    for j in range(1, n):
        scale = np.sqrt(j * (j + 1.0))
        basis[:j, j - 1] = 1.0 / scale
        basis[j, j - 1] = -j / scale
    return basis


def _qeq_kernel(
    rng: np.random.Generator, n: int, condition: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    z = sum_zero_basis(n)
    if n == 1:
        base = np.zeros((1, 1), dtype=np.float64)
    else:
        u = _random_unitary(rng, n - 1, complex_=False)
        eigvals = np.geomspace(1.0, condition, n - 1)
        tangent = (u * eigvals[None, :]) @ u.T
        base = z @ tangent @ z.T
    base = 0.5 * (base + base.T)
    kernel = base + alpha * np.ones((n, n), dtype=np.float64)
    return base, 0.5 * (kernel + kernel.T)


def _formed_tangent_condition(kernel: np.ndarray) -> tuple[float, float]:
    n = kernel.shape[0]
    if n == 1:
        return np.inf, 1.0
    # Match the public operator's certification coordinates exactly.  This is
    # intentionally not the Helmert basis used by the independent solve:
    # at alpha~1e16, choosing a different valid basis changes where already
    # lost float64 digits appear.
    projector = np.eye(n) - np.ones((n, n)) / float(n)
    projector_eigvals, projector_eigvecs = np.linalg.eigh(projector)
    z = projector_eigvecs[:, projector_eigvals > 0.5]
    physical_kernel = kernel - np.mean(kernel)
    eigvals = np.linalg.eigvalsh(z.T @ physical_kernel @ z)
    if eigvals[0] <= 0.0:
        return float(eigvals[0]), np.inf
    return float(eigvals[0]), float(eigvals[-1] / eigvals[0])


def _qeq_case(
    rng: np.random.Generator, case_id: str, case_seed: int, index: int
) -> ConformanceCase:
    n = 1 + int((rng.random() ** 2) * 64)
    outside_gate = index % 13 == 0 and n > 1
    if outside_gate:
        n = max(n, 3)
    log_condition = rng.uniform(12.05, 14.0) if outside_gate else rng.uniform(0.0, 11.7)
    target_condition = 10.0**log_condition
    alpha = float(10.0 ** rng.uniform(-6.0, 16.0))
    base, kernel = _qeq_kernel(rng, n, target_condition, alpha)
    min_eig, actual_condition = _formed_tangent_condition(kernel)
    expected = "accept"
    if n > 1 and (
        min_eig <= 1.0e-12
        or not np.isfinite(actual_condition)
        or actual_condition > 1.0e12
    ):
        expected = "reject"
    return ConformanceCase(
        case_id=case_id,
        seed=case_seed,
        operator="qeq",
        family="tangent_spd_uniform_gauge",
        expected_status=expected,
        payload={
            "electronegativity": rng.uniform(-5.0, 5.0, size=n),
            "hardness_kernel": kernel,
            "base_hardness_kernel": base,
            "total_charge": float(rng.uniform(-2.0, 2.0)),
            "target_condition": target_condition,
            "actual_condition": actual_condition,
            "alpha": alpha,
            "declared_tolerance": float(
                10.0 ** rng.uniform(-12.0, -6.0)
            ),
        },
    )


def _random_spd_kernel(
    rng: np.random.Generator, n: int, strength: float
) -> np.ndarray:
    if n == 1:
        return np.asarray([[strength]], dtype=np.float64)
    a = rng.normal(size=(n, n))
    kernel = a @ a.T
    kernel /= max(float(np.linalg.eigvalsh(kernel)[-1]), 1.0)
    return strength * (kernel + 0.1 * np.eye(n))


def _scf_case(
    rng: np.random.Generator, case_id: str, case_seed: int, index: int
) -> ConformanceCase:
    kind = ("one_level", "symmetric_dimer", "asymmetric_dimer", "random_small")[
        index % 4
    ]
    mixing = ("linear", "pdiis")[(index // 4) % 2]
    mixing_step = (0.05, 0.1, 0.2, 0.5)[(index // 8) % 4]
    charge_tol = (1.0e-6, 1.0e-8, 1.0e-10)[(index // 32) % 3]
    max_iter = 1000
    g = float(_DEGENERACIES[index % 2])
    mu = float(rng.uniform(-0.5, 0.5))
    kT = 0.1

    if kind == "one_level":
        n = 1
        h = np.asarray([[rng.uniform(-0.5, 0.5)]])
        ao = np.asarray([0])
        reference = np.asarray([0.5 * g])
        kernel = np.asarray([[rng.uniform(0.05, 0.5)]])
    elif kind.endswith("dimer"):
        n = 2
        if kind == "symmetric_dimer":
            onsite = np.repeat(rng.uniform(-0.3, 0.3), 2)
        else:
            center = rng.uniform(-0.2, 0.2)
            split = rng.uniform(0.1, 0.5)
            onsite = np.asarray([center - split, center + split])
        hopping = rng.uniform(-0.1, 0.1)
        h = np.asarray(
            [[onsite[0], hopping], [hopping, onsite[1]]],
            dtype=np.float64,
        )
        ao = np.arange(2)
        reference = np.repeat(0.5 * g, 2)
        gamma = rng.uniform(0.05, 0.35)
        kernel = np.asarray(
            [[gamma, 0.2 * gamma], [0.2 * gamma, gamma]]
        )
    else:
        n = 1 + int((rng.random() ** 2) * 16)
        a = rng.normal(scale=0.15, size=(n, n))
        h = 0.5 * (a + a.T)
        ao = np.arange(n)
        reference = np.repeat(0.5 * g, n)
        kernel_kind = ("zero", "hubbard", "spd")[(index // 3) % 3]
        if kernel_kind == "zero":
            kernel = np.zeros((n, n))
        elif kernel_kind == "hubbard":
            kernel = np.diag(rng.uniform(0.01, 0.08, size=n))
        else:
            kernel = _random_spd_kernel(rng, n, 0.05)
        if not np.any(kernel):
            max_iter = 100

    return ConformanceCase(
        case_id=case_id,
        seed=case_seed,
        operator="scf",
        family=kind,
        expected_status="accept",
        payload={
            "h0": h,
            "s": np.eye(n),
            "mu": mu,
            "kT": kT,
            "ao_atom_index": ao,
            "reference_populations": reference,
            "coulomb_kernel": kernel,
            "mixing": mixing,
            "mixing_step": mixing_step,
            "n_history": 6,
            "mixing_period": 3,
            "max_iter": max_iter,
            "charge_tol": charge_tol,
            "spin_degeneracy": g,
            "declared_tolerance": charge_tol,
        },
    )


def _mutation_case(
    rng: np.random.Generator, case_id: str, case_seed: int, index: int
) -> ConformanceCase:
    mutation = _MUTATIONS[index % len(_MUTATIONS)]
    target = "qeq" if index % 2 else "fixed_mu"
    if mutation in {"wrong_shape", "nan_inf", "non_hermitian"}:
        target = ("fixed_mu", "qeq", "scf")[index % 3]
    elif mutation == "non_spd":
        target = ("fixed_mu", "qeq")[index % 2]
    elif mutation == "truncated_eigenbasis":
        target = "fixed_mu"
    return ConformanceCase(
        case_id=case_id,
        seed=case_seed,
        operator=target,
        family="mutation",
        expected_status="reject",
        mutation_kind=mutation,
        payload={
            "declared_tolerance": 1.0e-2
            if mutation == "self_reported_loose_tolerance"
            else 1.0e-8,
            "mutation_dimension": 2,
            "selector": int(rng.integers(0, 2**31 - 1)),
        },
    )


def generate_cases(n_cases: int, seed: int = 730) -> Iterator[ConformanceCase]:
    """Yield a deterministic prefix of the controlled C1.1 case family."""

    if isinstance(n_cases, bool) or int(n_cases) != n_cases or n_cases < 1:
        raise ValueError("n_cases must be a positive integer")
    for index in range(int(n_cases)):
        rng, case_seed = _case_rng(seed, index)
        case_id = f"c1-{index:06d}"
        slot = index % 20
        if slot < 8:
            yield _fixed_mu_case(rng, case_id, case_seed, index)
        elif slot < 14:
            yield _qeq_case(rng, case_id, case_seed, index)
        elif slot < 17:
            scf_ordinal = (index // 20) * 3 + (slot - 14)
            yield _scf_case(
                rng, case_id, case_seed, scf_ordinal
            )
        else:
            mutation_ordinal = (index // 20) * 3 + (slot - 17)
            yield _mutation_case(
                rng, case_id, case_seed, mutation_ordinal
            )

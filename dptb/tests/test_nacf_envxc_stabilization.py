"""McWEDA stabilization versions (F1 of the 2026-09-20 independent audit): v1 (fixed100 recipe) and v2 (composed projected residuals).

CPU only. The synthetic forward below is the audit's reproduction: the real ``NACFEnvXCPlan.forward`` and ``_contract`` run on
manufactured rank-three factor tables (positive epsilon, envelope/AO moments deliberately inconsistent) for one p-p shell pair
with kappa = 3; only the envelope factor scales with N, the AO moment leak De stays. It establishes the stabilization
counterexample, not an occurrence in real geometry.
"""
import math
import types

import numpy as np
import pytest
import torch

from dptb.nacf.envxc import EnvXCBank, EnvXCStore, NACFEnvXCPlan, STABILIZATIONS, mcweda_composition, reference_edge_envxc
from dptb.tests.test_nacf_envxc import build_root, edge_overlap, make_structure

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    path = tmp_path_factory.mktemp("envxc_stab")
    sources, proj = build_root(path)
    return path, sources, proj


def synthetic_plan(N_value, stabilization, *, Sw=1e-2, P=1e-7, rotation=None):
    p = NACFEnvXCPlan.__new__(NACFEnvXCPlan); torch.nn.Module.__init__(p)
    p.positions = torch.zeros((2, 3), dtype=torch.float64); p.profile = False; p.fused = None; p.nedges = 1; p.width = 3; p.nsmax = 1
    p.need_moment = True; p.need_pairmom = True; p.need_layers = False; p.arms = ("mcweda",); p.overlap_floor = 1e-8
    p.moment_density_floor = 0.0; p.nterms = 1; p.nqueries = 2; p.topology_stats = []; p.edge_ptr = torch.tensor([0, 1])
    p.representative = torch.tensor([True]); p.reverse = torch.tensor([0]); p.stabilization = stabilization
    R = torch.eye(3, dtype=torch.float64) if rotation is None else torch.as_tensor(rotation, dtype=torch.float64)
    rp = P / Sw
    S = (R @ (torch.eye(3, dtype=torch.float64) * Sw) @ R.T).unsqueeze(0)
    Dp = rp * S

    def fixed(t):
        return lambda v: t
    er0 = torch.tensor([[[1e-3], [0.], [0.]]], dtype=torch.float64)
    er1 = torch.tensor([[[N_value / 1e-3], [0.], [0.]]], dtype=torch.float64)
    qr0 = (torch.eye(3, dtype=torch.float64) * math.sqrt(1e-7) @ R.T).unsqueeze(0)
    qr1 = (torch.diag(torch.tensor([1., -1., 0.], dtype=torch.float64)) * math.sqrt(1e-7) @ R.T).unsqueeze(0)
    p.bank = types.SimpleNamespace(tables={"envfac0": fixed(er0), "envfac1": fixed(er1), "rhofac0": fixed(qr0), "rhofac1": fixed(qr1),
                                           "sw": fixed(torch.tensor([[[Sw]]], dtype=torch.float64)), "pw": fixed(torch.tensor([[[P]]], dtype=torch.float64)),
                                           "dp": fixed(Dp)}, epsilon=lambda sk: torch.ones(3, dtype=torch.float64))
    p.factor_specs = [(0, "K", "P", "envfac0", "rhofac0"), (1, "K", "P", "envfac1", "rhofac1")]
    p.contraction_specs = [(0, 0, 1, "K", 3, 3, 1, 1, 1)]; p.terms_0 = torch.tensor([[0, 0, 0]])
    p._factor_delta = lambda i: torch.zeros((1, 3), dtype=torch.float64)
    p.pair_specs = [(0, "P", "P", False, dict(envnorm="sw", envpair="pw", pairmom="dp"))]
    p.pe_0 = torch.tensor([0]); p.pv_0 = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    p.species_meta = {"P": dict(nshells=1, norb=3)}; p.shell_of_ao_P = torch.tensor([0, 0, 0]); p.kappa_P_P = torch.tensor([[3.]], dtype=torch.float64)
    return p, S


def test_v2_is_continuous_in_the_weak_environment_limit_where_v1_is_not():
    """At fixed pair moment P a finite-rank leak of Denv survives the v1 total projection (bound 2 kappa P), so the v1
    correction stays at about 2.66 meV as N -> 0+ although the N <= 0 branch is exactly zero. v2 decays with N."""
    rows = {}
    for N in (1e-8, 1e-12, 1e-16, 1e-20, 1e-24, 0.0, -1e-24):
        rows[N] = {}
        for version in STABILIZATIONS:
            plan, S = synthetic_plan(N, version)
            out = plan.forward(edge_overlap_ao=S)
            assert out["diagnostics"]["stabilization"] == version
            rows[N][version] = float(out["mcweda"].abs().max())
    assert rows[1e-20]["v1"] > 1e-3 and rows[1e-24]["v1"] > 1e-3          # the audited defect is reproduced
    for N in (0.0, -1e-24):
        assert rows[N]["v1"] == 0.0 and rows[N]["v2"] == 0.0                # exact zero-environment branch
    # O(N) decay towards the zero branch: the audit measured 1.40e-7 eV at N = 1e-12 and 1.39e-15 eV at N = 1e-20,
    # i.e. a stable ratio of about 1.4e5 eV Bohr^3 (v'_t times the 2 kappa N envelope bound)
    ratios = [rows[N]["v2"] / N for N in (1e-12, 1e-16, 1e-20, 1e-24)]
    assert all(r < 1e6 for r in ratios), ratios
    assert max(ratios) < 3 * min(ratios)
    assert rows[1e-12]["v2"] < 1e-6 and rows[1e-20]["v2"] < 1e-12
    values = [rows[N]["v2"] for N in (1e-8, 1e-12, 1e-16, 1e-20, 1e-24)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_versions_agree_algebraically_without_rescaling(monkeypatch):
    """With every Frobenius projection disabled the two versions are one formula: Dp + De - rho_t S = (Dp - rho_p S) + (De - b S)."""
    import dptb.nacf.envxc as envxc
    monkeypatch.setattr(envxc, "residual_rescale", lambda M, bound, ia, ib: (M, 0, 0.0))
    for N in (1e-8, 1e-10, 3e-7):
        outs = {}
        for version in STABILIZATIONS:
            plan, S = synthetic_plan(N, version)
            outs[version] = plan.forward(edge_overlap_ao=S)["mcweda"]
        assert outs["v1"].abs().max() > 0
        assert torch.allclose(outs["v1"], outs["v2"], rtol=1e-12, atol=1e-15)


def test_composition_is_shell_covariant_for_both_versions():
    """Rotating the AO frame of the p shell rotates the correction: Delta(R S R^T, R D R^T) = R Delta R^T (both versions)."""
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R = torch.as_tensor(Q, dtype=torch.float64)
    for version in STABILIZATIONS:
        for N in (1e-8, 1e-11):
            plain, S0 = synthetic_plan(N, version)
            rotated, S1 = synthetic_plan(N, version, rotation=Q)
            a = plain.forward(edge_overlap_ao=S0)["mcweda"][0]
            b = rotated.forward(edge_overlap_ao=S1)["mcweda"][0]
            assert a.abs().max() > 0
            assert torch.allclose(R @ a @ R.T, b, rtol=1e-10, atol=1e-10 * float(a.abs().max()))
    # the pure composition on random blocks, NumPy and torch, with active projections
    ia = np.repeat(np.arange(2), [1, 3]); ib = np.repeat(np.arange(2), [3, 1])
    kappa = np.sqrt(np.array([[1, 3], [3, 9]], dtype=float) * np.array([[3, 1], [9, 3]], dtype=float))
    for version in STABILIZATIONS:
        S, De, Dp, M_env = (rng.normal(size=(2, 4, 4)) * 1e-2 for _ in range(4))
        rho_p, rho_t = np.abs(rng.normal(size=(2, 4, 4))) * 1e-3 + 1e-4, np.abs(rng.normal(size=(2, 4, 4))) * 1e-3 + 2e-4
        Nsh, Psh = np.abs(rng.normal(size=(2, 2, 2))) * 1e-4, np.abs(rng.normal(size=(2, 2, 2))) * 1e-4
        v = [rng.normal(size=(2, 4, 4)) for _ in range(4)]
        term_np, counters = mcweda_composition(version, S, De, Dp, M_env, rho_p, rho_t, Nsh, Psh, kappa, ia, ib, *v)
        term_t, _ = mcweda_composition(version, *(torch.as_tensor(x) for x in (S, De, Dp, M_env, rho_p, rho_t, Nsh, Psh, kappa)),
                                       torch.as_tensor(ia), torch.as_tensor(ib), *(torch.as_tensor(x) for x in v))
        assert np.allclose(term_np, term_t.numpy(), atol=1e-14)
        assert counters["mcweda_pair_rescaled_shell_pairs"] > 0
        assert (counters["mcweda_total_rescaled_shell_pairs"] > 0) == (version == "v1")
    with pytest.raises(ValueError):
        mcweda_composition("v3", S, De, Dp, M_env, rho_p, rho_t, Nsh, Psh, kappa, ia, ib, *v)


def test_plan_matches_numpy_reference_for_both_versions_and_default_is_v1(root):
    path, sources, _ = root
    store = EnvXCStore(path)
    cut = {s: src.orbital_cutoff_bohr for s, src in sources.items()}
    g = make_structure(["Xa", "Yb", "Xa"], [[0.3, 0.2, 0.1], [2.9, 0.4, 1.7], [0.8, 3.2, 2.6]], np.diag([7.0, 7.5, 8.0]), cut)
    S = edge_overlap(sources, g)
    bank = EnvXCBank(store, device="cpu", backend="torch")
    results = {}
    for version in STABILIZATIONS:
        plan = bank.prepare(**g, arms=("mcweda",), topology="python", stabilization=version)
        out = plan(edge_overlap_ao=torch.tensor(S))
        ref = reference_edge_envxc(store, g, arms=("mcweda",), edge_overlap_ao=S, stabilization=version)
        got = out["mcweda"].numpy()
        assert np.isfinite(got).all()
        assert np.max(np.abs(got - ref["mcweda"])) < 1e-9 * max(1.0, np.max(np.abs(ref["mcweda"]))), version
        assert torch.equal(out["mcweda"], out["mcweda"][plan.reverse].transpose(-1, -2))
        diag = out["diagnostics"]
        assert diag["stabilization"] == version
        assert diag["mcweda_pair_removed_frobenius_bohr_minus3"] == diag["mcweda_pair_removed_frobenius_bohr3"]
        if version == "v2":
            assert diag["mcweda_total_rescaled_shell_pairs"] == 0
        results[version] = got
    default = bank.prepare(**g, arms=("mcweda",), topology="python")(edge_overlap_ao=torch.tensor(S))
    assert default["diagnostics"]["stabilization"] == "v1"
    assert np.array_equal(default["mcweda"].numpy(), results["v1"])
    with pytest.raises(ValueError):
        bank.prepare(**g, arms=("mcweda",), topology="python", stabilization="v3")
    with pytest.raises(ValueError):
        reference_edge_envxc(store, g, arms=("mcweda",), edge_overlap_ao=S, stabilization="v3")

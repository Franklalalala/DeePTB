"""Physical counterexamples for AO feedback, overlap and spectral alignment."""

from types import SimpleNamespace
import pytest
import torch
from e3nn import o3
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.loopscf.representation import AOScalarFeedback
from dptb.nnops.loopscf.adapters import _inject, ZeroInitWM
from dptb.nnops.loopscf.kspace import build_k_plan, assemble_flat
from dptb.nnops.loopscf.occupations import factor_overlap_robust
from dptb.nnops.loopscf.spectral import eigvals_from_factor
from dptb.nnops.loopscf.metrics import fermi_level, mu_aligned_band_error


def test_physical_h0_roundtrip_through_network_rme_without_input_mutation():
    from dptb.nnops.loopscf.representation import AOPriorToRME
    from dptb.nn.hamiltonian import E3Hamiltonian
    from dptb.data import AtomicDataDict as A

    mapper = OrbitalMapper({"C": "2s2p1d"}, method="e3tb")
    codec = AOPriorToRME(mapper, dtype=torch.float64, device="cpu")
    node = torch.randn(2, mapper.reduced_matrix_element, dtype=torch.float64)
    edge = torch.randn(3, mapper.reduced_matrix_element, dtype=torch.float64)
    saved_node, saved_edge = node.clone(), edge.clone()
    edges = torch.tensor([[0, 0, 1], [1, 0, 0]])
    data = {
        A.NODE_H0_KEY: node,
        A.EDGE_H0_KEY: edge,
        A.EDGE_INDEX_KEY: edges,
        A.POSITIONS_KEY: torch.randn(2, 3, dtype=torch.float64),
    }
    rn, re = codec(data)
    reconstruct = E3Hamiltonian(
        idp=mapper,
        dtype=torch.float64,
        device="cpu",
        node_field=A.NODE_H0_KEY,
        edge_field=A.EDGE_H0_KEY,
    )
    recovered = reconstruct({**data, A.NODE_H0_KEY: rn, A.EDGE_H0_KEY: re})
    torch.testing.assert_close(recovered[A.NODE_H0_KEY], node, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(recovered[A.EDGE_H0_KEY], edge, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(saved_node, node)
    torch.testing.assert_close(saved_edge, edge)


def test_ao_scalar_rotation_and_gradient():
    mapper = OrbitalMapper({"C": "2s2p1d1f"}, method="e3tb")
    scalar = AOScalarFeedback(mapper)
    torch.manual_seed(74)
    x = torch.randn(3, scalar.width, dtype=torch.float64, requires_grad=True)
    R = o3.rand_matrix(dtype=torch.float64)
    rotated = x.clone()
    from dptb.utils.constants import anglrMId
    import re

    for pair, sl in mapper.orbpair_maps.items():
        ls = [anglrMId[re.findall(r"[a-zA-Z]", shell)[0]] for shell in pair.split("-")]
        a, b = [o3.Irrep(l, (-1) ** l).D_from_matrix(R) for l in ls]
        block = x[:, sl].reshape(-1, a.shape[0], b.shape[0])
        rotated[:, sl] = (a @ block @ b.T).flatten(1)
    torch.testing.assert_close(scalar(x), scalar(rotated), atol=1e-10, rtol=1e-10)
    scalar(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    # A traceless p block has zero scalar despite a nonzero first coordinate.
    pp = mapper.orbpair_maps["1p-1p"]
    y = torch.zeros_like(x)
    y[:, pp] = torch.diag(torch.tensor([-2.0, 0.0, 2.0])).flatten()
    torch.testing.assert_close(scalar(y), torch.zeros_like(scalar(y)))


@pytest.mark.parametrize("indices", [[1, 2], [2, 0, 1]])
def test_feedback_uses_active_order_even_at_equal_count(indices):
    emb = SimpleNamespace(
        wm_node=torch.nn.Identity(),
        wm_edge=torch.nn.Identity(),
        _wm_hid_slices=[slice(0, 1)],
    )
    wm = torch.tensor([[10.0], [20.0], [30.0]])
    _, edge = _inject(
        emb,
        {"wm_n": torch.zeros(1, 1), "wm_e": wm, "active_edges": torch.tensor(indices)},
        torch.zeros(1, 1),
        torch.zeros(len(indices), 1),
    )
    torch.testing.assert_close(edge, wm[indices])
    if len(indices) != 3:
        with pytest.raises(RuntimeError, match="mapping"):
            _inject(
                emb,
                {"wm_n": torch.zeros(1, 1), "wm_e": wm},
                torch.zeros(1, 1),
                torch.zeros(len(indices), 1),
            )


def test_same_shape_legacy_feedback_checkpoint_is_rejected():
    module = ZeroInitWM(3, 1)
    state = module.state_dict()
    module.load_state_dict(state)
    del state["feedback_version"]
    with pytest.raises(RuntimeError, match="incompatible"):
        module.load_state_dict(state, strict=False)


@pytest.mark.parametrize("mode", ["head", "moe"])
def test_injection_is_equivariant_at_its_actual_layer(mode):
    from dptb.nnops.loopscf.adapters import _attach_adapters

    emb = torch.nn.Module()
    emb.idp = OrbitalMapper({"H": "1s1p"}, method="e3tb")
    emb.init_layer = torch.nn.Linear(5, 5, dtype=torch.float64)
    emb.init_layer.irreps_out = o3.Irreps("2x0e+1x1o")
    final = torch.nn.Linear(5, 5, dtype=torch.float64)
    final.irreps_out = o3.Irreps("1x0e+1x1o+1x0e")
    emb.layers = torch.nn.ModuleList([final])
    _attach_adapters(emb, mode)
    with torch.no_grad():
        emb.wm_node.proj.weight.fill_(0.2)
        emb.wm_edge.proj.weight.fill_(0.3)
    ir = emb.init_layer.irreps_out if mode == "moe" else final.irreps_out
    D = ir.D_from_matrix(o3.rand_matrix(dtype=torch.float64))
    h = torch.randn(2, 5, dtype=torch.float64)
    wm = torch.ones(2, emb.wm_node.proj.in_features, dtype=torch.float64)
    ctx = {"wm_n": wm, "wm_e": wm}
    original = _inject(emb, ctx, h, h)
    rotated = _inject(emb, ctx, h @ D.T, h @ D.T)
    for before, after in zip(original, rotated):
        torch.testing.assert_close(before @ D.T, after, atol=1e-10, rtol=1e-10)


def test_isolated_atom_assembly_preserves_double_and_gradient():
    mapper = OrbitalMapper({"H": "1s"}, method="e3tb")
    plan = build_k_plan(
        mapper,
        torch.tensor([0]),
        torch.empty(2, 0, dtype=torch.long),
        torch.tensor([0]),
        torch.tensor([0, 1]),
        "cpu",
    )
    x = torch.tensor([[1.000000001]], dtype=torch.float64, requires_grad=True)
    H = plan.block(
        assemble_flat(
            plan,
            x,
            x.new_empty(0, 1),
            torch.empty(1, 0, dtype=torch.complex128),
            torch.complex128,
        ),
        0,
    )
    torch.testing.assert_close(H.real.reshape_as(x), x, atol=1e-14, rtol=0)
    H.real.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_positive_but_ill_conditioned_overlap_is_projected_and_reported():
    S = torch.diag(torch.tensor([1.0, 1e-12], dtype=torch.float64)).unsqueeze(0)
    diagnostics = {}
    factors = factor_overlap_robust(S, diagnostics=diagnostics)
    assert diagnostics["dropped_modes"].tolist() == [1]
    assert diagnostics["spin_degenerate_capacity"].tolist() == [2]
    assert diagnostics["condition_S"].item() == pytest.approx(1e12)
    H = torch.diag(torch.tensor([2.0, 1e-8], dtype=torch.float64)).unsqueeze(0)
    assert eigvals_from_factor(H, *factors)[0, 0].item() == pytest.approx(2.0)


@pytest.mark.parametrize("smearing", [0.0, 0.1])
def test_metal_mu_from_bz_and_shift_invariant_path_metric(smearing):
    bz = torch.tensor([[-3.0, -2.0], [-1.0, 3.0]], dtype=torch.float64)
    mu = fermi_level(bz, 2, smearing=smearing)
    assert mu.item() == pytest.approx(-1.5, abs=2e-5)
    shifted = fermi_level(bz + 7, 2, smearing=smearing)
    path = torch.tensor([[-4.0, 2.0], [-0.4, 0.8]])
    error, count = mu_aligned_band_error(path + 7, path, mu_pred=shifted, mu_ref=mu)
    assert count == 4
    assert error.item() == pytest.approx(0, abs=1e-6)


def test_fractional_degenerate_mu_and_weighted_filling():
    assert fermi_level(torch.zeros(2, 2), 1.5).item() == 0
    bz = torch.tensor([[-3.0, -2.0], [-1.0, 3.0]])
    assert fermi_level(bz, 2, k_weights=torch.tensor([0.25, 0.75])).item() == -1

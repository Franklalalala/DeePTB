"""Selective routed layers keep a dense shared path and exact upcycling start."""
from __future__ import annotations

import copy

import pytest
import torch
from e3nn import o3

from dptb.nn.build import build_model
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear, SO2_Linear
from dptb.tests.model_helpers import _data
from dptb.tests._requires import requires_so2_cuda


def _routing(n, *, branch="all", device="cpu"):
    idx = torch.full((n, 2), 3, dtype=torch.long, device=device)
    idx[:, 1] = 2
    val = torch.full((n, 2), 0.2, device=device)
    return MOLEGlobals(coefficients=torch.zeros(n, 4, device=device).scatter(1, idx, val),
                       topk_indices=idx, topk_values=val, activation_space=True,
                       coefficients_sum_to_one=False, branch=branch)


def _model(layers=None, *, dense=False):
    torch.manual_seed(29)
    return build_model(
        common_options=dict(basis={"H": "1s", "O": "1s1p"}, overlap=False,
                            dtype="float32", device="cpu"),
        model_options=dict(
            embedding=dict(method="lem_moe_v3_edge_h0", n_layers=3,
                           avg_num_neighbors=2.0, r_max=4.0,
                           irreps_hidden="4x0e+4x1o+4x1e+4x2e", env_embed_multiplicity=2,
                           latent_dim=8, latent_channels=[8], edge_one_hot_dim=4,
                           num_experts=1 if dense else 4, num_shared_experts=0 if dense else 1,
                           top_k=1 if dense else 2, universal=True,
                           use_layer_onehot_tp=False, use_out_onehot_tp=False,
                           use_interpolation_out=True, tp_radial_emb=True, tp_radial_channels=[8],
                           mole_linear_mode="indexed_ref", so2_fusion_mode="staged",
                           equivariant_norm_type="none", edge_router_prior_activate=not dense,
                           edge_router_logit="cosine", edge_router_select="logit",
                           edge_router_bias_at_eval=True, edge_router_gate="full_softmax",
                           so2_expert_mixing_mode="pre_activation" if dense else "post_activation_shared",
                           **({} if layers is None else {"so2_moe_layers": layers})),
            prediction=dict(method="e3tb", scale_type="no_scale")), train_options={}, no_check=False)


def _h0_data(model):
    data = _data(model)
    data["node_h0"] = data.pop("node_p23")
    data["edge_h0"] = data.pop("edge_p2")
    return data


def _clone(data):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}


def _seed_from_dense(target, dense):
    """v20's dense->shared mapping, with only actual router buffers left fresh."""
    src, dst = dense.state_dict(), target.state_dict()
    used = set()
    for key in dst:
        if key.endswith(("weight_shared", "bias_shared")):
            source = key.replace("_shared", "_experts")
            assert src[source].shape[0] == 1 and dst[key].shape == src[source].shape
            dst[key] = src[source].clone()
            used.add(source)
        elif key.endswith(("weight_experts", "bias_experts")):
            assert key.replace("_experts", "_shared") in dst
            dst[key] = torch.zeros_like(dst[key])
        elif key in src and src[key].shape == dst[key].shape:
            dst[key] = src[key].clone()
            used.add(key)
        else:
            assert ".router." in key or "._prior_" in key, key
    assert all(".router." in key or "._prior_" in key for key in src.keys() - used)
    target.load_state_dict(dst, strict=True)


def test_default_all_is_the_same_model_and_checkpoint():
    default = _model()
    explicit = _model([0, 1, 2])
    a, b = default.state_dict(), explicit.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    default.eval(), explicit.eval()
    data = _h0_data(default)
    out_a, out_b = default(_clone(data)), explicit(_clone(data))
    for key in ("node_features", "edge_features"):
        torch.testing.assert_close(out_a[key], out_b[key], rtol=0, atol=0)


@pytest.mark.parametrize("interpolation", [False, True])
def test_shared_only_so2_matches_shared_reference_and_ignores_routed_ids(interpolation):
    torch.manual_seed(3)
    options = dict(irreps_in=o3.Irreps("3x0e+2x1o+2x2e"),
                   irreps_out=o3.Irreps("4x0e+2x1o+1x2e"),
                   num_shared_experts=1, use_interpolation=interpolation, so2_fusion_mode="staged")
    reference = SO2_Linear(num_experts=4, **options)
    shared = SO2_Linear(num_experts=0, **options)
    shared.load_state_dict({k: v for k, v in reference.state_dict().items()
                            if not k.endswith(("weight_experts", "bias_experts"))}, strict=True)
    assert not any("experts" in k for k in shared.state_dict())
    x = torch.randn(7, shared.irreps_in.dim, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    r = torch.randn(7, 3)
    out, _ = shared(x, r, _routing(7))  # ids 2 and 3 must never index nonexistent expert weights
    expected, _ = reference(x_ref, r, _routing(7, branch="shared"))
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
    out.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-6)
    for name, value in shared.named_parameters():
        ref_value = dict(reference.named_parameters())[name]
        torch.testing.assert_close(value.grad, ref_value.grad, rtol=1e-5, atol=1e-6)


@requires_so2_cuda
def test_shared_only_fused_p0_interpolation_forward_backward(monkeypatch):
    from dptb.nn import so2_activation_routes as routes

    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "1")
    torch.manual_seed(7)
    reference = SO2_Linear(
        irreps_in=o3.Irreps("3x0e+2x1o+2x2e"), irreps_out=o3.Irreps("4x0e+2x1o+1x2e"),
        num_experts=0, num_shared_experts=1, use_interpolation=True,
        so2_fusion_mode="staged").cuda()
    fused = copy.deepcopy(reference)
    fused.so2_fusion_mode = "streamed_m_major_fused_p0"
    x = torch.randn(11, fused.irreps_in.dim, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    r = torch.randn(11, 3, device="cuda")
    before = routes.STATS.calls.get(routes.FUSED_P0, 0)
    actual, _ = fused(x, r, _routing(11, device="cuda"))
    expected, _ = reference(xr, r, _routing(11, device="cuda"))
    assert routes.STATS.calls.get(routes.FUSED_P0, 0) == before + 1
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=5e-4, atol=5e-5)
    for name, value in fused.named_parameters():
        torch.testing.assert_close(value.grad, dict(reference.named_parameters())[name].grad,
                                   rtol=5e-4, atol=5e-5)


def test_late_hidden_upcycling_matches_dense_then_learns_and_roundtrips():
    dense, late, all_layers = _model(dense=True), _model([1]), _model()
    for model in (late, all_layers):
        _seed_from_dense(model, dense)
    # Removing experts changes constructor RNG consumption. Pair the new router
    # with the all-layer arm's iteration-zero state, not merely its common seed.
    late.embedding.router.load_state_dict(all_layers.embedding.router.state_dict(), strict=True)
    assert all(torch.equal(v, all_layers.embedding.router.state_dict()[k])
               for k, v in late.embedding.router.state_dict().items())
    for model in (dense, late, all_layers):
        model.eval()
    data = _h0_data(dense)
    expected = dense(_clone(data))
    for model in (late, all_layers):
        actual = model(_clone(data))
        for key in ("node_features", "edge_features"):
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)
    for index, layer in enumerate(late.embedding.layers):
        linears = [m for m in layer.modules() if isinstance(m, MOLELinear)]
        assert linears
        assert all(m.num_experts == (4 if index == 1 else 0) for m in linears)
        if index != 1:
            assert all(m.weight_experts is None and m.bias_experts is None for m in linears)
            assert not any(n.endswith(("weight_experts", "bias_experts")) for n, _ in layer.named_parameters())
    assert sum(p.numel() for p in late.parameters()) < sum(p.numel() for p in all_layers.parameters())
    opt = torch.optim.SGD(late.parameters(), lr=0.01)
    for _ in range(2):
        opt.zero_grad()
        out = late(_clone(data))
        (out["node_features"].square().mean() + out["edge_features"].square().mean()).backward()
        routed = [m for m in late.embedding.layers[1].modules() if isinstance(m, MOLELinear)]
        assert all(m.weight_experts.grad is not None and torch.isfinite(m.weight_experts.grad).all() for m in routed)
        assert any(m.weight_experts.grad.abs().sum() > 0 for m in routed)
        opt.step()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in late.embedding.router.parameters())
    restored = _model([1]).eval()
    restored.load_state_dict(copy.deepcopy(late.state_dict()), strict=True)
    a, b = late(_clone(data)), restored(_clone(data))
    for key in ("node_features", "edge_features"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


@pytest.mark.parametrize("layers", [[], [3], [-1], [1, 1], [True], "last"])
def test_invalid_placement_is_rejected(layers):
    with pytest.raises(ValueError):
        _model(layers)

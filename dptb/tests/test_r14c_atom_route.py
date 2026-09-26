import copy
import logging

import pytest
import torch
from e3nn import o3
from torch.nn import functional as F

from dptb.data import _keys
from dptb.nn.atom_route import node_invariants, edge_coefficients, pool_prior, record_atom_routes
from dptb.nn.build import build_model
from dptb.nn.tensor_product_moe_v3 import MOLELinear
from dptb.tests.atom_route_helpers import atom_config, atom_model, atom_batch
from dptb.tests._requires import requires_so2_cuda


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
def test_timing_placement_reverse_and_backbone_gradient(parameterization):
    model = atom_model(parameterization)
    emb = model.embedding
    seen, hidden, alphas = {}, [], []
    def capture_hidden(module, args, result):
        hidden.append(result[1])
    hooks = [emb.layers[0].register_forward_hook(capture_hidden),
             emb.router.register_forward_hook(lambda m, a, r: alphas.append(r[0]))]
    for i, layer in enumerate(emb.layers):
        hooks.append(layer.register_forward_pre_hook(lambda m, a, i=i: seen.update({i: a[-1]})))
        linears = [m for m in layer.modules() if isinstance(m, MOLELinear)]
        assert linears and all(m.num_experts == (4 if i == 1 else 0) for m in linears)
    out = model(atom_batch(model))
    for h in hooks:
        h.remove()
    assert seen[0] is None and seen[2] is None
    route = seen[1]
    assert route.activation_space and route.topk_values.shape == (6, 4)
    assert torch.equal(route.topk_values[::2], route.topk_values[1::2])
    assert torch.equal(route.topk_indices, torch.arange(4).expand(6, 4))
    torch.testing.assert_close(route.coefficients.sum(-1), torch.ones(6))
    # Routing-only objective proves the gradient to layer 0, independently of
    # the ordinary message path to the prediction loss.
    g = torch.autograd.grad((alphas[0] * torch.arange(4.)).sum(), hidden[0], retain_graph=True)[0]
    assert g.isfinite().all() and g.norm() > 0
    task = out["node_features"].square().sum() + out["edge_features"].square().sum()
    task.backward()
    assert emb.router.net[0].weight.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in emb.layers[0].parameters())
    assert set(emb.last_atom_route_stats) == {"1"}


@pytest.mark.parametrize("layers", [[1], [2], [1, 2]])
def test_each_requested_layer_reads_current_environment(layers):
    cfg = atom_config(); cfg["model_options"]["embedding"]["so2_moe_layers"] = layers
    m = build_model(**cfg)
    calls = []
    hook = m.embedding.router.register_forward_pre_hook(lambda mod, args: calls.append(args[0].detach().clone()))
    m(atom_batch(m)); hook.remove()
    assert len(calls) == len(layers)
    assert set(m.embedding.last_atom_route_stats) == set(map(str, layers))
    if len(calls) == 2:
        assert not torch.equal(calls[0], calls[1])


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
def test_atom_and_edge_permutation(parameterization):
    m = atom_model(parameterization).eval()
    data = atom_batch(m)
    p = torch.tensor([2, 0, 1]); inv = torch.argsort(p)
    ep = torch.tensor([4, 0, 3, 1, 5, 2])
    perm = {k: (v[p] if v.ndim and v.shape[0] == 3 else
                v[ep] if v.ndim and v.shape[0] == 6 else v.clone()) for k, v in data.items()}
    perm["edge_index"] = inv[data["edge_index"][:, ep]]
    a = m(copy.deepcopy(data)); sa = copy.deepcopy(m.embedding.last_atom_route_stats)
    b = m(perm)
    torch.testing.assert_close(a["node_features"][p], b["node_features"], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(a["edge_features"][ep], b["edge_features"], atol=2e-6, rtol=2e-5)
    for key in ("alpha_mean", "alpha_std", "effective_atoms", "effective_structures"):
        torch.testing.assert_close(torch.tensor(sa["1"][key]), torch.tensor(m.embedding.last_atom_route_stats["1"][key]))


def test_random_rotation_invariants_and_zero_gradient():
    ir = o3.Irreps("3x0e+2x0o+3x1o+2x1e+2x2e")
    x = torch.randn(8, ir.dim, dtype=torch.float64, requires_grad=True)
    for sign in (1, -1):
        d = ir.D_from_matrix(sign * o3.rand_matrix(dtype=torch.float64))
        torch.testing.assert_close(node_invariants(x, ir), node_invariants(x @ d.T, ir), atol=1e-7, rtol=1e-7)
    zero = torch.zeros_like(x, requires_grad=True)
    node_invariants(zero, ir).sum().backward()
    assert zero.grad.isfinite().all()


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
def test_complete_embedding_rotation_and_ao_prior_contract(parameterization):
    m = atom_model(parameterization).eval()
    emb = m.embedding
    data = atom_batch(m)
    c = emb.init_layer._h0_cg_change_of_basis
    coupled = copy.deepcopy(data)
    for key in ("node_h0", "edge_h0"):
        coupled[key] = data[key] @ c.T
    coupled[_keys.H0_COUPLED_RME_KEY] = torch.ones(1, dtype=torch.bool)
    a = emb(copy.deepcopy(data))
    alpha = emb.router.last_topk()[1].detach().clone()
    b = emb(copy.deepcopy(coupled))
    for key in ("node_features", "edge_features"):
        torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(alpha, emb.router.last_topk()[1], atol=1e-6, rtol=1e-5)
    rotation = o3.rand_matrix()
    # LEM spherical coordinates use xyz -> yzx.
    dr = m.idp.get_irreps().D_from_matrix(rotation[[1, 2, 0]][:, [1, 2, 0]])
    rotated = copy.deepcopy(coupled)
    rotated["pos"] = coupled["pos"] @ rotation.T
    for key in ("node_h0", "edge_h0"):
        rotated[key] = coupled[key] @ dr.T
    r = emb(rotated)
    torch.testing.assert_close(alpha, emb.router.last_topk()[1], atol=3e-5, rtol=3e-5)
    for key in ("node_features", "edge_features"):
        torch.testing.assert_close(b[key] @ dr.T, r[key], atol=5e-5, rtol=5e-4)


def test_pooling_symmetry_isolation_and_statistics(caplog):
    data = dict(edge_index=torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]]),
                batch=torch.tensor([0, 0, 0, 1, 1, 2]), pos=torch.zeros(6, 3))
    desc = torch.tensor([[1.], [3.], [4.], [8.], [10.], [20.]])
    active = torch.arange(6)
    got = pool_prior(data, active, desc, torch.ones(6), 6)
    torch.testing.assert_close(got[:, 0], torch.tensor([2., 4., 6., 15., 15., 0.]))
    with pytest.raises(ValueError):
        pool_prior(data, active[:-1], desc[:-1], torch.ones(6), 6)
    a = torch.full((6, 4), .25)
    with caplog.at_level(logging.INFO, logger="dptb.nn.atom_route"):
        s = record_atom_routes(a, data, 1, 7, True, torch.tensor([1, 8, 1, 8, 8, 1]))
    assert s["effective_atoms"] == [6.] * 4 and s["effective_structures"] == [3.] * 4
    assert s["alpha_std"] == [0.] * 4 and set(s["element_mean"]) == {"1", "8"}
    assert '"opt_step": 7' in caplog.text


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
def test_checkpoint_optimizer_continuation(parameterization, tmp_path):
    cfg = atom_config(parameterization)
    m = build_model(**cfg); data = atom_batch(m)
    m.embedding._prior_mean.fill_(.17)
    m.embedding._prior_std.fill_(1.3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    def step(model, optimizer):
        optimizer.zero_grad(); out = model(copy.deepcopy(data))
        loss = out["node_features"].square().mean() + out["edge_features"].square().mean()
        loss.backward(); optimizer.step()
        return loss.detach()
    step(m, opt)
    cfg["model_options"] = copy.deepcopy(m.model_options)
    path = tmp_path / "atom.pth"
    torch.save(dict(config=cfg, model_state_dict=m.state_dict(), optimizer=opt.state_dict()), path)
    other = build_model(checkpoint=str(path))
    other_opt = torch.optim.Adam(other.parameters(), lr=1e-3)
    other_opt.load_state_dict(torch.load(path, weights_only=False)["optimizer"])
    for k, v in m.state_dict().items():
        assert torch.equal(v, other.state_dict()[k]), k
    assert torch.equal(step(m, opt), step(other, other_opt))
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, other.state_dict()[k], atol=1e-7, rtol=1e-6, msg=k)


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
@pytest.mark.parametrize("pair", [False, True])
def test_reference_expert_sum_forward_backward(parameterization, pair, monkeypatch):
    torch.manual_seed(14)
    lin = MOLELinear(7, 5, num_experts=4, num_shared_experts=1, bias=not pair,
                     mole_expert_parameterization=parameterization, mole_expert_rank=3,
                     mole_linear_mode="split_loop").double()
    # Catch accidental use of the per-route weight materialization API.
    monkeypatch.setattr(lin, "_mix_expert_parameters", lambda *a, **k: pytest.fail("per-edge weights"))
    h = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 5, 6], [1, 0, 4, 6, 7]])
    x = torch.randn((5, 2, 7) if pair else (5, 7), dtype=torch.float64, requires_grad=True)
    alpha = h.softmax(-1); route = edge_coefficients(alpha, edges)
    y = lin(x, route)
    ref = F.linear(x, lin.weight_shared[0], None if pair else lin.bias_shared[0])
    for k in range(4):
        w = (lin.basis_left @ lin.core_experts[k] @ lin.basis_right.T
             if parameterization == "shared_core" else lin.weight_experts[k])
        val = (alpha[edges[0], k] + alpha[edges[1], k]) * .5
        ref = ref + F.linear(x, w, None if pair else lin.bias_experts[k]) * val.reshape(5, *([1] * (x.ndim - 1)))
    torch.testing.assert_close(y, ref, atol=1e-11, rtol=1e-11)
    args = [x, h, *lin.parameters()]
    ga = torch.autograd.grad(y.square().sum(), args, retain_graph=True)
    gb = torch.autograd.grad(ref.square().sum(), args)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@requires_so2_cuda
@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
def test_gpu_fused_atom_environment_forward_backward(parameterization, record_property):
    from dptb.nn import so2_activation_routes as routes
    ref = atom_model(parameterization, device="cuda:0", mole_linear_mode="cublas_grouped")
    fused = atom_model(parameterization, device="cuda:0", mole_linear_mode="cublas_grouped",
                       so2_fusion_mode="streamed_m_major_fused_p0")
    fused.load_state_dict(ref.state_dict(), strict=True)
    data = atom_batch(ref)
    a = ref(copy.deepcopy(data))
    before = routes.STATS.calls.get(routes.FUSED_P0, 0)
    routed_calls = []
    def before_routed(module, args):
        routed_calls.append(routes.STATS.calls.get(routes.FUSED_P0, 0))
    def after_routed(module, args, result):
        routed_calls[-1] = routes.STATS.calls.get(routes.FUSED_P0, 0) - routed_calls[-1]
    h1 = fused.embedding.layers[1].register_forward_pre_hook(before_routed)
    h2 = fused.embedding.layers[1].register_forward_hook(after_routed)
    b = fused(copy.deepcopy(data))
    h1.remove(); h2.remove()
    calls = routes.STATS.calls.get(routes.FUSED_P0, 0) - before
    assert calls > 0 and routed_calls and min(routed_calls) > 0
    record_property("fused_p0_calls", calls)
    record_property("routed_layer_fused_p0_calls", sum(routed_calls))
    for key in ("node_features", "edge_features"):
        record_property(key + "_max_error", (a[key] - b[key]).abs().max().item())
        torch.testing.assert_close(a[key], b[key], atol=3e-4, rtol=3e-4)
    for out, model in ((a, ref), (b, fused)):
        (out["node_features"].square().sum() + out["edge_features"].square().sum()).backward()
    max_grad_error = 0.
    for (name, pa), (_, pb) in zip(ref.named_parameters(), fused.named_parameters()):
        if pa.grad is None:
            assert pb.grad is None, name
        else:
            max_grad_error = max(max_grad_error, (pa.grad - pb.grad).abs().max().item())
            torch.testing.assert_close(pa.grad, pb.grad, atol=2e-3, rtol=2e-3, msg=name)
    record_property("max_parameter_grad_error", max_grad_error)
    assert fused.embedding.router.net[0].weight.grad.norm() > 0


@pytest.mark.parametrize("option", [dict(top_k=2), dict(so2_moe_layers=[0]),
    dict(edge_router_bias_speed=.005), dict(edge_router_route_drop_p=.2),
    dict(edge_router_input="onehot"), dict(so2_expert_mixing_mode="post_activation_shared")])
def test_invalid_options_fail(option):
    cfg = atom_config(); cfg["model_options"]["embedding"].update(option)
    with pytest.raises(ValueError):
        build_model(**cfg)

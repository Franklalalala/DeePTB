"""Structure route dropout: numerics, graph correlation, gradients and RNG replay."""
import copy
import logging
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from dptb.data import _keys
from dptb.nn.build import build_model
from dptb.nn.route_drop import apply_structure_routes, sample_structure_routes
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear, SO2_Linear
from dptb.nnops.training_state import capture_rng_state, restore_rng_state
from dptb.plugins.saver import Saver
from dptb.tests._requires import requires_so2_cuda
from dptb.tests.model_helpers import _build


def _model(**extra):
    opts = dict(num_experts=4, top_k=2, num_shared_experts=1, n_layers=3,
                edge_router_prior_activate=True, so2_fusion_mode="staged",
                mole_linear_mode="split_loop", mole_expert_rank=3,
                edge_router_bias_speed=0.0)
    opts.update(extra)
    return _build(False, **opts)


def _batch(model):
    # Three disconnected, unequal complete graphs; interleaved edges ensure that
    # neither sorted graph ids nor contiguous edge runs can stand in for batch.
    g = torch.Generator().manual_seed(101)
    sizes = torch.tensor([3, 2, 4])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
    edges = [(i, j) for a, b in zip(ptr[:-1], ptr[1:]) for i in range(a, b) for j in range(a, b) if i != j]
    edge = torch.tensor(edges).t()[:, torch.randperm(len(edges), generator=g)]
    types = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0]).reshape(-1, 1)
    names = [model.idp.type_to_chemical_symbol[int(t)] for t in types.flatten()]
    bond = torch.tensor([model.idp.bond_to_type[names[i] + "-" + names[j]] for i, j in edge.t()])
    rme = model.idp.reduced_matrix_element
    return {_keys.POSITIONS_KEY: torch.randn(9, 3, generator=g) * 0.3,
            _keys.EDGE_INDEX_KEY: edge, _keys.ATOM_TYPE_KEY: types, _keys.EDGE_TYPE_KEY: bond,
            _keys.BATCH_KEY: torch.repeat_interleave(torch.arange(3), sizes), _keys.BATCH_PTR_KEY: ptr,
            _keys.NODE_P23_KEY: torch.randn(9, rme, generator=g),
            _keys.EDGE_P2_KEY: torch.randn(len(edges), rme, generator=g)}


def _close(a, b, record_property, label, atol=1e-10, rtol=0):
    err = float((a.detach().double() - b.detach().double()).abs().max()) if a.numel() else 0.0
    record_property(label, err)
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def _raw_route(emb, features):
    return emb._make_undropped_edge_moe_globals(features, torch.zeros(len(features), dtype=torch.long, device=features.device))[0]


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("training", [False, True])
def test_p0_bitwise_outputs_gradients_state_and_rng(parameterization, top_k, training):
    torch.manual_seed(810)
    legacy = _model(top_k=top_k, mole_expert_parameterization=parameterization)
    torch.manual_seed(810)
    explicit = _model(top_k=top_k, mole_expert_parameterization=parameterization,
                      edge_router_route_drop_p=0.0)
    legacy.train(training); explicit.train(training)
    data = _batch(legacy)
    before = torch.get_rng_state()
    a = legacy(copy.deepcopy(data))
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    b = explicit(copy.deepcopy(data))
    assert torch.equal(after, torch.get_rng_state()) and torch.equal(before, after)
    for key in ("node_features", "edge_features"):
        assert torch.equal(a[key], b[key])
    for out, model in ((a, legacy), (b, explicit)):
        (out["node_features"].square().sum() + out["edge_features"].square().sum()).backward()
    for (name, pa), (_, pb) in zip(legacy.named_parameters(), explicit.named_parameters()):
        assert (pa.grad is None and pb.grad is None) or torch.equal(pa.grad, pb.grad), name
    assert all(torch.equal(v, explicit.state_dict()[k]) for k, v in legacy.state_dict().items())


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("scale", ["inverted", "none"])
def test_p1_three_layer_shared_only_and_eval_ignores_p(parameterization, top_k, scale, record_property):
    torch.manual_seed(32)
    model = _model(top_k=top_k, mole_expert_parameterization=parameterization,
                   edge_router_route_drop_p=1.0, edge_router_route_drop_scale=scale)
    torch.manual_seed(32)
    ref = _model(top_k=top_k, mole_expert_parameterization=parameterization,
                 edge_router_route_drop_p=0.0, edge_router_route_drop_scale=scale)
    # Independent shared-only execution skips routed matrix products altogether.
    raw = ref.embedding._make_edge_moe_globals
    def shared(*args, **kwargs):
        route, *rest = raw(*args, **kwargs)
        route.branch = "shared"
        route.coefficients_sum_to_one = False
        return (route, *rest)
    ref.embedding._make_edge_moe_globals = shared
    data = _batch(model)
    before = torch.get_rng_state()
    a, b = model(copy.deepcopy(data)), ref(copy.deepcopy(data))
    assert torch.equal(before, torch.get_rng_state())
    for key in ("node_features", "edge_features"):
        _close(a[key], b[key], record_property, "p1_" + key, atol=1e-6, rtol=1e-5)
    assert model.embedding.last_route_drop_mask.all()
    for label in ("structure", "edge", "active_edge"):
        assert a["edge_router_route_drop_" + label + "_fraction"] == 1
    del ref.embedding._make_edge_moe_globals
    model.eval(); ref.eval()
    a, b = model(copy.deepcopy(data)), ref(copy.deepcopy(data))
    for key in ("node_features", "edge_features"):
        assert torch.equal(a[key], b[key])
    # Eval never consumes dropout RNG or overwrites the last training snapshot.
    assert torch.equal(before, torch.get_rng_state())
    assert model.embedding.last_route_drop_stats["structure_fraction"] == 1


@pytest.mark.parametrize("method", ["lem_moe_v3_prior_2b", "lem_moe_v3_edge_prior_2b", "lem_moe_v3_edge_h0"])
def test_one_mask_shared_by_every_module_and_three_layers(method, monkeypatch, caplog):
    from dptb.nn.embedding import lem_moe_v3_edge as edge_module
    opts = dict(method=method, edge_router_route_drop_p=0.5)
    if method == "lem_moe_v3_edge_h0":
        # Plain H0's prior keys differ from the pairwise/GNN model.
        opts.update(h0_node_key="node_p23", h0_edge_key="edge_p2", h0_init_scope="both")
    model = _model(**opts)
    data = _batch(model)
    drawn, seen, layers = [], [], []
    original_sample = edge_module.sample_structure_routes
    def sample(*args):
        result = original_sample(*args)
        drawn.append(result)
        return result
    monkeypatch.setattr(edge_module, "sample_structure_routes", sample)
    def inspect(mod, args):
        route = args[1]
        assert isinstance(route, MOLEGlobals)
        seen.append(route)
    handles = [m.register_forward_pre_hook(inspect) for m in model.modules() if isinstance(m, MOLELinear)]
    handles += [m.register_forward_pre_hook(lambda m, a: layers.append(m)) for m in model.embedding.layers]
    torch.manual_seed(2)  # both keep and drop for three structures
    with caplog.at_level(logging.INFO, logger="dptb.nn.route_drop"):
        result = model(copy.deepcopy(data))
    for handle in handles:
        handle.remove()
    assert len(drawn) == 1 and len(layers) == 3 and len(seen) > 3
    assert all(r is seen[0] for r in seen)
    keep, edge_keep = drawn[0]
    assert keep.any() and not keep.all()
    route = seen[0]
    assert not route.coefficients_sum_to_one
    assert torch.equal(route.coefficients.gather(1, route.topk_indices), route.topk_values)
    assert torch.equal(route.topk_values, model.embedding.router.last_topk()[1])
    torch.testing.assert_close(route.coefficients.sum(1), edge_keep.float() * 2)
    assert float(result["edge_router_route_drop_structure_fraction"]) == pytest.approx(float((~keep).float().mean()))
    assert float(result["edge_router_route_drop_edge_fraction"]) == pytest.approx(float((~edge_keep).float().mean()))
    assert sum("route_drop step=" in r.message for r in caplog.records) == 1


def test_independent_structures_empty_graphs_and_active_edge_mapping():
    # Empty graphs 0, 2 and trailing 4 remain in the sampling/structure denominator.
    data = {"batch": torch.tensor([1, 1, 3, 3, 3]), "ptr": torch.tensor([0, 0, 2, 2, 5, 5]),
            "edge_index": torch.tensor([[2, 0, 4, 1], [4, 1, 3, 0]])}
    torch.manual_seed(402)
    outcomes = torch.stack([sample_structure_routes(data, 0.3)[0] for _ in range(6000)]).float()
    assert (outcomes.mean(0) - 0.7).abs().max() < 0.025
    joint = (outcomes.t() @ outcomes) / len(outcomes)
    offdiag = ~torch.eye(5, dtype=torch.bool)
    assert (joint[offdiag] - 0.49).abs().max() < 0.03
    torch.manual_seed(2)
    keep, edge_keep = sample_structure_routes(data, 0.5)
    assert torch.equal(edge_keep, keep[torch.tensor([3, 1, 3, 1])])
    active = torch.tensor([3, 0, 1])
    assert torch.equal(edge_keep[active], keep[data["batch"][data["edge_index"][0, active]]])
    corrupt = dict(data, edge_index=torch.tensor([[0], [2]]))
    with pytest.raises(ValueError, match="different structures"):
        sample_structure_routes(corrupt, 0.3)


@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("empty_graphs", [0, 3])
def test_empty_routes_and_finite_logging(top_k, empty_graphs):
    model = _model(top_k=top_k, edge_router_route_drop_p=0.5)
    emb = model.embedding
    data = {"batch": torch.empty(0, dtype=torch.long), "ptr": torch.zeros(empty_graphs + 1, dtype=torch.long),
            "edge_index": torch.empty(2, 0, dtype=torch.long)}
    route, _, _, _ = emb._make_edge_moe_globals(torch.empty(0, emb.edge_router_in_features),
            torch.empty(0, dtype=torch.long), data=data, active_edges=torch.empty(0, dtype=torch.long))
    assert route.coefficients.shape == (0, 4) and route.topk_values.shape == (0, top_k)
    assert not route.coefficients_sum_to_one
    assert emb.last_route_drop_mask.shape == (empty_graphs,)
    assert emb.last_route_drop_stats["edge_fraction"] == 0
    assert all(torch.isfinite(v) for v in emb.last_route_drop_stats.values())
    lin = MOLELinear(7, 5, num_experts=4, num_shared_experts=1)
    assert lin(torch.empty(0, 7), route).shape == (0, 5)


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("scale", ["inverted", "none"])
@pytest.mark.parametrize("pair_rows", [False, True])
def test_fixed_mask_scaling_and_exact_linear_expectation(parameterization, top_k, scale, pair_rows, record_property):
    torch.manual_seed(743)
    emb = _model(top_k=top_k).embedding.double().eval()
    h = torch.randn(5, emb.edge_router_in_features, dtype=torch.float64)
    route = _raw_route(emb, h)
    lin = MOLELinear(7, 5, num_experts=4, num_shared_experts=1, mole_expert_parameterization=parameterization,
                     mole_expert_rank=3, mole_linear_mode="split_loop", bias=not pair_rows).double()
    x = torch.randn((5, 2, 7) if pair_rows else (5, 7), dtype=torch.float64)
    shared = F.linear(x, lin.weight_shared.sum(0), None if pair_rows else lin.bias_shared.sum(0))
    eval_out = lin(x, route)
    keep = torch.tensor([True, False, True, False, True])
    p = 0.2
    dropped = apply_structure_routes(route, keep, p, scale, top_k)
    got = lin(x, dropped)
    factor = keep.double() / (1 - p) if scale == "inverted" else keep.double()
    shape = (5,) + (1,) * (x.ndim - 1)
    expected = shared + (eval_out - shared) * factor.reshape(shape)
    _close(got, expected, record_property, "fixed_mask_fp64")
    # Exact Bernoulli expectation, independent of Monte Carlo tolerance.
    on = lin(x, apply_structure_routes(route, torch.ones_like(keep), p, scale, top_k))
    off = lin(x, apply_structure_routes(route, torch.zeros_like(keep), p, scale, top_k))
    expected_mean = eval_out if scale == "inverted" else shared + (1 - p) * (eval_out - shared)
    _close((1 - p) * on + p * off, expected_mean, record_property, "expectation_fp64")


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
@pytest.mark.parametrize("top_k", [2, 4])
def test_closed_structure_has_zero_router_and_expert_gradients(parameterization, top_k):
    torch.manual_seed(899)
    model = _model(top_k=top_k, mole_expert_parameterization=parameterization, edge_router_route_drop_p=0.5)
    data = _batch(model)
    torch.manual_seed(2)
    out = model(copy.deepcopy(data))
    mask = model.embedding.last_route_drop_mask
    node_closed = mask[data["batch"]]
    edge_closed = node_closed[data["edge_index"][0]]
    router = list(model.embedding.router.parameters())
    linears = [m for m in model.modules() if isinstance(m, MOLELinear)]
    routed = [p for m in linears for n, p in m.named_parameters() if n not in ("weight_shared", "bias_shared")]
    shared = [m.weight_shared for m in linears if m.weight_shared is not None]
    params = router + routed
    loss = out["node_features"][node_closed].square().sum() + out["edge_features"][edge_closed].square().sum()
    closed_grads = torch.autograd.grad(loss, params + shared, allow_unused=True, retain_graph=True)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in closed_grads[:len(params)])
    assert any(g is not None and g.abs().sum() > 0 for g in closed_grads[len(params):])
    live_loss = out["node_features"][~node_closed].square().sum() + out["edge_features"][~edge_closed].square().sum()
    live_grads = torch.autograd.grad(live_loss, params, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in live_grads[:len(router)])
    assert any(g is not None and g.abs().sum() > 0 for g in live_grads[len(router):])
    # Even an explicitly enabled router z-loss has no closed-edge contribution.
    emb = model.embedding
    features = torch.randn(20, emb.edge_router_in_features, requires_grad=True)
    torch.manual_seed(2)
    emb._make_edge_moe_globals(features, data["edge_type"], data=data, active_edges=torch.arange(20))
    gh, = torch.autograd.grad(emb.router.last_router_z_loss, (features,))
    assert torch.count_nonzero(gh[edge_closed]) == 0 and gh[~edge_closed].abs().sum() > 0


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
@pytest.mark.parametrize("top_k", [2, 4])
def test_real_saver_rng_restore_and_optimizer_step(parameterization, top_k, tmp_path, record_property):
    model = _model(top_k=top_k, mole_expert_parameterization=parameterization, edge_router_route_drop_p=0.2)
    # Exercise the real stage-1-to-GNN initialization hook before training.
    model.load_state_dict(model.state_dict(), strict=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    data = _batch(model)
    def step(m, o):
        o.zero_grad(set_to_none=True)
        out = m(copy.deepcopy(data))
        (out["node_features"].square().mean() + out["edge_features"].square().mean()).backward()
        o.step()
        return out
    torch.manual_seed(901)
    step(model, opt)
    saver = Saver()
    saver.trainer = SimpleNamespace(model=model, task="train", ep=1, iter=1, stats={})
    obj = saver._assemble_checkpoint_obj("step1", "iteration", model.model_options,
                                         dict(basis={"H": "1s", "O": "1s1p"}, overlap=False, dtype="float32", device="cpu"),
                                         {}, model.state_dict(), [dict(optimizer_state_dict=opt.state_dict())])
    assert "rng_state" in obj["training_state"]
    path = tmp_path / "route_drop.pth"
    torch.save(obj, path)
    a = step(model, opt)
    mask = model.embedding.last_route_drop_mask.clone()
    rng_after = torch.get_rng_state()
    resumed = build_model(checkpoint=str(path))
    resumed_opt = torch.optim.SGD(resumed.parameters(), lr=1e-3, momentum=0.9)
    checkpoint = torch.load(path, weights_only=False)
    resumed_opt.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_rng_state(checkpoint["training_state"]["rng_state"])
    b = step(resumed, resumed_opt)
    assert torch.equal(mask, resumed.embedding.last_route_drop_mask)
    assert torch.equal(rng_after, torch.get_rng_state())
    for key in ("node_features", "edge_features"):
        assert torch.equal(a[key], b[key])
    for k, v in model.state_dict().items():
        _close(v, resumed.state_dict()[k], record_property, "resume_" + k, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("extra", [
    {"edge_router_route_drop_p": -0.1}, {"edge_router_route_drop_p": 1.1},
    {"edge_router_route_drop_p": float("nan")}, {"edge_router_route_drop_p": float("inf")},
    {"edge_router_route_drop_p": True}, {"edge_router_route_drop_scale": "bad"},
    {"so2_expert_mixing_mode": "post_activation"}, {"so2_expert_mixing_mode": "post_activation_slot"},
    {"so2_expert_mixing_mode": "post_activation_shared"}, {"top_k": 1},
    {"edge_router_top1_mode": "switch", "top_k": 1, "num_shared_experts": 0},
    {"num_shared_experts": 0}, {"edge_router_prior_activate": False},
])
def test_invalid_options_rejected(extra):
    opts = dict(edge_router_route_drop_p=0.2)
    opts.update(extra)
    with pytest.raises((ValueError, TypeError)):
        _model(**opts)


@requires_so2_cuda
@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("p", [0.2, 1.0])
def test_gpu_fused_dispatch_gradients_and_rng(parameterization, top_k, p, monkeypatch, record_property):
    from dptb.nn import so2_activation_routes as routes
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "1")
    monkeypatch.delenv("DPTB_SO2_FUSION_MODE", raising=False)
    torch.manual_seed(925)
    emb = _model(top_k=top_k, edge_router_route_drop_p=p).embedding.cuda()
    opts = dict(irreps_in="4x0e+3x1o+2x2e", irreps_out="3x0e+2x1o+2x2e", num_experts=4,
                num_shared_experts=1, mole_expert_parameterization=parameterization,
                mole_expert_rank=3, mole_linear_mode="cublas_grouped")
    ref = SO2_Linear(**opts, so2_fusion_mode="staged").cuda()
    fused = SO2_Linear(**opts, so2_fusion_mode="streamed_m_major_fused_p0").cuda()
    fused.load_state_dict(ref.state_dict())
    n = 31
    data = {"batch": torch.arange(n, device="cuda") // 3,
            "ptr": torch.tensor(list(range(0, 31, 3)) + [31], device="cuda"),
            "edge_index": torch.arange(n, device="cuda").expand(2, -1)}
    h = torch.randn(n, emb.edge_router_in_features, device="cuda", requires_grad=True)
    x = torch.randn(n, ref.irreps_in.dim, device="cuda", requires_grad=True)
    r = torch.randn(n, 3, device="cuda")
    # Condition this kernel fixture on exercising both paths. The sampling law
    # itself is tested without conditioning in the CPU independence test.
    for _ in range(64):
        rng = capture_rng_state()
        route, *_ = emb._make_edge_moe_globals(h, torch.zeros(n, dtype=torch.long, device="cuda"),
                                              data=data, active_edges=torch.arange(n, device="cuda"))
        saved_mask = emb.last_route_drop_mask.clone()
        if p == 1 or (saved_mask.any() and not saved_mask.all()):
            break
    else:
        pytest.fail("could not draw a mixed route-drop fixture")
    restore_rng_state(rng)
    replay, *_ = emb._make_edge_moe_globals(h, torch.zeros(n, dtype=torch.long, device="cuda"),
                                           data=data, active_edges=torch.arange(n, device="cuda"))
    assert torch.equal(saved_mask, emb.last_route_drop_mask)
    assert torch.equal(route.coefficients, replay.coefficients)
    a = ref(x, r, route)[0]
    before = routes.STATS.calls.get(routes.FUSED_P0, 0)
    b = fused(x, r, replay)[0]
    calls = routes.STATS.calls.get(routes.FUSED_P0, 0) - before
    record_property("observed_fused_p0_calls", calls)
    assert calls > 0
    _close(a, b, record_property, "gpu_forward", 3e-4, 3e-4)
    ga = torch.autograd.grad(a.square().sum(), [x, h, *emb.router.parameters(), *ref.parameters()], retain_graph=True)
    gb = torch.autograd.grad(b.square().sum(), [x, h, *emb.router.parameters(), *fused.parameters()])
    for i, (aa, bb) in enumerate(zip(ga, gb)):
        _close(aa, bb, record_property, "gpu_grad_%d" % i, 2e-3, 2e-3)
    if p == 1:
        assert torch.count_nonzero(ga[1]) == 0


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("n_nodes", [0, 2])
def test_complete_empty_batch_forward(parameterization, top_k, n_nodes):
    model = _model(top_k=top_k, mole_expert_parameterization=parameterization, edge_router_route_drop_p=0.2)
    data = _batch(model)
    for key in ("pos", "atom_types", "node_p23"):
        data[key] = data[key][:n_nodes]
    for key in ("edge_type", "edge_p2"):
        data[key] = data[key][:0]
    data["edge_index"] = data["edge_index"][:, :0]
    # No batch metadata: with_batch must represent even a single empty structure.
    data.pop("batch"); data.pop("ptr")
    out = model(data)
    assert out["node_features"].shape[0] == n_nodes and out["edge_features"].shape[0] == 0
    assert torch.isfinite(out["node_features"]).all()
    assert model.embedding.last_route_drop_mask.shape == (1,)
    assert out["edge_router_route_drop_edge_fraction"] == 0
    assert out["router_z_loss"] == 0


def test_active_subset_with_empty_structures_and_no_stale_z_loss():
    emb = _model(edge_router_route_drop_p=0.5).embedding
    data = {"batch": torch.tensor([1, 1, 3, 3, 3]), "ptr": torch.tensor([0, 0, 2, 2, 5, 5]),
            "edge_index": torch.tensor([[2, 0, 4, 1], [4, 1, 3, 0]])}
    active = torch.tensor([3, 0, 1])
    features = torch.randn(3, emb.edge_router_in_features)
    torch.manual_seed(1)
    route, *_ = emb._make_edge_moe_globals(features, torch.zeros(3, dtype=torch.long), data=data, active_edges=active)
    closed = emb.last_route_drop_mask
    mapped = closed[data['batch'][data['edge_index'][0, active]]]
    torch.testing.assert_close(route.coefficients.sum(1), (~mapped).float() * 2)
    assert emb.last_route_drop_stats['structure_count'] == 5
    assert emb.last_route_drop_stats['edge_count'] == 4
    assert emb.last_route_drop_stats['active_edge_count'] == 3
    assert emb.router.last_router_z_loss > 0
    emb._make_edge_moe_globals(features[:0], torch.empty(0, dtype=torch.long), data=data, active_edges=active[:0])
    assert emb.router.last_router_z_loss == 0
    assert emb.router.last_topk()[1].shape == (0, 2)

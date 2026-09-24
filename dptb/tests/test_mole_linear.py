"""MoLE linear backends, routers and edge-MoE route rows against explicit per-row references."""
import types

import pytest
import torch
import torch.nn.functional as F

from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear, MOLERouterV3, _mole_graph_index
from dptb.tests._requires import requires_cuda, requires_module, requires_so2_cuda


@pytest.fixture(autouse=True)
def _no_mode_env(monkeypatch):
    monkeypatch.delenv("DPTB_MOLE_LINEAR_MODE", raising=False)


def _layer(mode, in_features=11, out_features=13, num_experts=8, shared=2, bias=True, dtype=torch.float64,
           device="cpu", seed=20260423):
    torch.manual_seed(seed)
    return MOLELinear(in_features, out_features, num_experts=num_experts, num_shared_experts=shared, bias=bias,
                      mole_linear_mode=mode).to(device=device, dtype=dtype)


def _per_row_reference(layer, x, coefficients, route):
    """Each row applies its route's mix sum_e c_e W_e (+ shared experts), written as one einsum."""
    weight = torch.einsum("ge,eoi->goi", coefficients, layer.weight_experts)
    bias = coefficients @ layer.bias_experts if layer.bias_experts is not None else None
    if layer.num_shared_experts:
        weight = weight + layer.weight_shared.sum(0)
        if bias is not None:
            bias = bias + layer.bias_shared.sum(0)
    out = torch.einsum("noi,n...i->n...o", weight[route], x)
    if bias is not None:
        out = out + bias[route].reshape(route.numel(), *([1] * (x.ndim - 2)), -1)
    return out


def _grads(out, inputs, probe):
    return torch.autograd.grad((out * probe).sum(), inputs, allow_unused=True, retain_graph=True)


def _assert_all_close(got, want, tol):
    for a, b in zip(got, want):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=tol, rtol=tol)


BACKENDS = [
    pytest.param("split_loop", "cpu", torch.float64, 1e-10, id="split_loop"),
    pytest.param("indexed_ref", "cpu", torch.float64, 1e-10, id="indexed_ref"),
    pytest.param("cueq_indexed_linear", "cuda", torch.float32, 2e-4,
                 marks=[requires_cuda, requires_module("cuequivariance_torch")], id="cueq_indexed_linear"),
    pytest.param("cublas_grouped", "cuda", torch.float32, 2e-4, marks=requires_so2_cuda, id="cublas_grouped"),
]
LAYOUTS = [  # (trailing input shape, bias, shared experts, rows per route, routes given by)
    pytest.param((11,), True, 2, (3, 5, 2, 7), "sizes", id="bias-shared-sizes"),
    pytest.param((2, 11), True, 2, (3, 5, 2, 7), "split_sizes", id="pair-axis-bias-shared"),
    pytest.param((2, 11), False, 0, (1, 4, 6), "split_sizes", id="pair-axis-plain"),
    pytest.param((11,), True, 0, (1, 4, 6), "sizes", id="bias-no-shared"),
    pytest.param((2, 11), False, 2, (1, 4, 6), "split_sizes", id="pair-axis-shared-no-bias"),
    pytest.param((2, 11), True, 1, (5, 0, 7, 4), "split_sizes", id="empty-route"),
]


@pytest.mark.parametrize("trailing, bias, shared, rows, given_by", LAYOUTS)
@pytest.mark.parametrize("mode, device, dtype, tol", BACKENDS)
def test_backend_matches_per_row_reference(mode, device, dtype, tol, trailing, bias, shared, rows, given_by):
    """Forward and gradients to inputs, coefficients and every parameter."""
    layer = _layer(mode, shared=shared, bias=bias, dtype=dtype, device=device)
    coefficients = torch.rand(len(rows), 8, device=device, dtype=dtype)
    coefficients = (coefficients / coefficients.sum(-1, keepdim=True)).requires_grad_(True)
    routes = torch.tensor(rows, device=device)
    mole_globals = (MOLEGlobals(coefficients=coefficients, sizes=routes) if given_by == "sizes"
                    else MOLEGlobals(coefficients=coefficients, split_sizes=rows))
    route = torch.repeat_interleave(torch.arange(len(rows), device=device), routes)
    x = torch.randn(route.numel(), *trailing, device=device, dtype=dtype, requires_grad=True)
    inputs = [x, coefficients, *layer.parameters()]
    got = layer(x, mole_globals)
    want = _per_row_reference(layer, x, coefficients, route)
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)
    probe = torch.randn_like(want)
    _assert_all_close(_grads(got, inputs, probe), _grads(want, inputs, probe), tol)


def test_mole_globals_from_sizes_caches_split_sizes():
    sizes = torch.tensor([2, 1], dtype=torch.long)
    mole_globals = MOLEGlobals(coefficients=torch.zeros(2, 3), sizes=sizes)
    assert mole_globals.split_sizes == (2, 1)


@pytest.mark.parametrize("misleading", [dict(sizes=torch.tensor([1, 1, 1])), dict(graph_index=torch.tensor([1, 0, 2, 0, 1, 2]))],
                         ids=["sizes", "graph_index"])
@pytest.mark.parametrize("mode", ["split_loop", "indexed_ref"])
def test_explicit_split_sizes_take_precedence(mode, misleading):
    layer = _layer(mode, in_features=6, out_features=8, num_experts=5, shared=1)
    coefficients = torch.softmax(torch.randn(3, 5, dtype=torch.float64), -1)
    split_sizes = (2, 1, 3)
    mole_globals = MOLEGlobals(coefficients=coefficients, split_sizes=split_sizes, **misleading)
    route = torch.tensor([0, 0, 1, 2, 2, 2])
    torch.testing.assert_close(_mole_graph_index(mole_globals, 6, device=torch.device("cpu")), route)
    x = torch.randn(6, 6, dtype=torch.float64)
    torch.testing.assert_close(layer(x, mole_globals), _per_row_reference(layer, x, coefficients, route))


@pytest.mark.parametrize("mode", ["split_loop", "indexed_ref"])
def test_missing_coefficients_average_the_experts(mode):
    layer = _layer(mode, in_features=5, out_features=3, num_experts=4, shared=1)
    x = torch.randn(6, 2, 5, dtype=torch.float64)
    weight = layer.weight_experts.mean(0) + layer.weight_shared.sum(0)
    bias = layer.bias_experts.mean(0) + layer.bias_shared.sum(0)
    torch.testing.assert_close(layer(x, None), F.linear(x, weight, bias))


def test_topk_metadata_mixes_like_dense_coefficients():
    layer = _layer("indexed_ref", in_features=4, out_features=3, num_experts=5, shared=1)
    topk_indices = torch.tensor([[0, 2], [3, 1], [4, 0]])
    topk_values = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.9, 0.1]], dtype=torch.float64)
    coefficients = torch.zeros(3, 5, dtype=torch.float64).scatter(1, topk_indices, topk_values)
    route = torch.tensor([2, 0, 1, 1, 2, 0, 0])
    x = torch.randn(7, 4, dtype=torch.float64)
    dense = layer(x, MOLEGlobals(coefficients=coefficients, graph_index=route))
    topk = layer(x, MOLEGlobals(coefficients=coefficients, graph_index=route, topk_indices=topk_indices,
                                topk_values=topk_values))
    torch.testing.assert_close(topk, dense, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(topk, _per_row_reference(layer, x, coefficients, route))


@pytest.mark.parametrize("env", ["indexed_ref", "cublas_grouped", "cueq_indexed_linear"])
def test_mode_from_environment(monkeypatch, env):
    monkeypatch.setenv("DPTB_MOLE_LINEAR_MODE", env)
    assert MOLELinear(4, 4).mole_linear_mode == env


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="mole_linear_mode"):
        MOLELinear(4, 4, mole_linear_mode="bad")


def test_cueq_mode_single_route_on_cpu_matches_split_loop():
    """A single route takes the plain linear path, so the cueq mode also runs where cuequivariance cannot."""
    base, cueq = _layer("split_loop", 4, 6, 5, 1), _layer("cueq_indexed_linear", 4, 6, 5, 1)
    coefficients = torch.softmax(torch.randn(1, 5, dtype=torch.float64), -1).requires_grad_(True)
    x = torch.randn(7, 2, 4, dtype=torch.float64, requires_grad=True)
    results = []
    for layer in (base, cueq):
        out = layer(x, MOLEGlobals(coefficients=coefficients, split_sizes=(7,)))
        results.append((out, *_grads(out, [x, coefficients, *layer.parameters()], torch.ones_like(out))))
    _assert_all_close(results[1], results[0], 1e-10)


@requires_cuda
@requires_module("cuequivariance_torch")
def test_cueq_mode_rejects_half_precision():
    layer = _layer("cueq_indexed_linear", 3, 5, 4, 0, bias=False, dtype=torch.float16, device="cuda")
    coefficients = torch.softmax(torch.rand(2, 4, device="cuda"), -1).half()
    with pytest.raises(RuntimeError, match="float32/float64"):
        layer(torch.randn(5, 3, device="cuda", dtype=torch.float16), MOLEGlobals(coefficients=coefficients,
                                                                                  split_sizes=(2, 3)))


@requires_so2_cuda
def test_grouped_gemm_multi_matches_per_group_linear():
    from dptb.nn.cublas_grouped_gemm import grouped_gemm_multi

    torch.manual_seed(20260521)
    ptr = torch.tensor([0, 4, 4, 9])  # group 1 is empty
    xs = [torch.randn(9, 5, device="cuda", requires_grad=True), torch.randn(9, 3, device="cuda", requires_grad=True)]
    ws = [torch.randn(3, 7, 5, device="cuda", requires_grad=True), torch.randn(3, 4, 3, device="cuda", requires_grad=True)]
    got = grouped_gemm_multi(xs, [ptr, ptr], ws)
    want = [torch.cat([F.linear(x[ptr[g]:ptr[g + 1]], w[g]) for g in range(3)]) for x, w in zip(xs, ws)]
    probes = [torch.randn_like(y) for y in want]
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
    grads = [torch.autograd.grad(sum((y * p).sum() for y, p in zip(ys, probes)), [*xs, *ws]) for ys in (got, want)]
    _assert_all_close(grads[0], grads[1], 2e-4)


# ---------------------------------------------------------------------------
# per-row expert routing (activation space, apply_experts, Switch top-1)
# ---------------------------------------------------------------------------
EXPERT_BACKENDS = [
    pytest.param("indexed_ref", "cpu", torch.float64, 1e-10, id="indexed_ref"),
    pytest.param("cublas_grouped", "cuda", torch.float32, 1e-4, marks=requires_so2_cuda, id="cublas_grouped"),
]


def _count_grouped_gemm(monkeypatch):
    """Count cuBLAS grouped GEMM calls (the module needs SO2CUDA's loader, so only import it where used)."""
    import dptb.nn.cublas_grouped_gemm as cublas

    calls = []
    original = cublas.grouped_gemm

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cublas, "grouped_gemm", counted)
    return calls


@pytest.mark.parametrize("sum_to_one", [True, False], ids=["folded-shared", "separate-shared"])
@pytest.mark.parametrize("mode, device, dtype, tol", EXPERT_BACKENDS)
def test_activation_space_matches_per_row_reference(monkeypatch, mode, device, dtype, tol, sum_to_one):
    """sum_j c_j (x W_{e_j} + b_{e_j}): the shared expert joins every routed expert when the coefficients sum
    to one, and is added once otherwise."""
    calls = _count_grouped_gemm(monkeypatch) if mode == "cublas_grouped" else []
    layer = _layer(mode, 5, 4, num_experts=4, shared=1, dtype=dtype, device=device)
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(9, 2, 5, generator=gen, dtype=dtype).to(device).requires_grad_(True)
    val, idx = torch.randn(9, 4, generator=gen, dtype=dtype).to(device).topk(2, dim=-1)
    val = (val.softmax(-1) if sum_to_one else val.sigmoid()).detach().requires_grad_(True)
    got = layer(x, MOLEGlobals(topk_indices=idx, topk_values=val, coefficients=val, activation_space=True,
                               coefficients_sum_to_one=sum_to_one))
    shared_w, shared_b = layer.weight_shared.sum(0), layer.bias_shared.sum(0)
    want = 0 if sum_to_one else F.linear(x, shared_w, shared_b)
    for j in range(2):
        weight, bias = layer.weight_experts[idx[:, j]], layer.bias_experts[idx[:, j], None]
        if sum_to_one:
            weight, bias = weight + shared_w, bias + shared_b
        want = want + val[:, j, None, None] * (torch.einsum("noi,nki->nko", weight, x) + bias)
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)
    inputs = [x, val, *layer.parameters()]
    probe = torch.randn_like(want)
    _assert_all_close(_grads(got, inputs, probe), _grads(want, inputs, probe), tol)
    assert len(calls) == (2 if mode == "cublas_grouped" else 0)


def test_rewritten_topk_indices_invalidate_the_slot_layout():
    layer = _layer("indexed_ref", 3, 2, num_experts=4, shared=0)
    idx = torch.tensor([[0], [3], [1], [3]])
    val = torch.ones(4, 1, dtype=torch.float64)
    mole_globals = MOLEGlobals(topk_indices=idx, topk_values=val, coefficients=val, activation_space=True)
    x = torch.randn(4, 3, dtype=torch.float64)
    layer(x, mole_globals)
    idx[:, 0] = torch.tensor([2, 2, 0, 1])  # in place, as a router reusing its buffer would
    fresh = MOLEGlobals(topk_indices=idx.clone(), topk_values=val, coefficients=val, activation_space=True)
    torch.testing.assert_close(layer(x, mole_globals), layer(x, fresh))


@pytest.mark.parametrize("leading", [(7,), (5, 2)])
@pytest.mark.parametrize("mode, device, dtype, tol", EXPERT_BACKENDS)
def test_apply_experts_matches_per_row_expert(mode, device, dtype, tol, leading):
    layer = _layer(mode, 4, 5, num_experts=3, shared=0, dtype=dtype, device=device)
    x = torch.randn(*leading, 4, device=device, dtype=dtype, requires_grad=True)
    expert = torch.tensor([2, 0, 1, 2, 1, 0, 2][:leading[0]], device=device)
    got = layer.apply_experts(x, expert)
    rows = expert.reshape(-1, *([1] * (x.ndim - 2))).expand(x.shape[:-1])
    want = torch.einsum("...oi,...i->...o", layer.weight_experts[rows], x) + layer.bias_experts[rows]
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)
    inputs = [x, layer.weight_experts, layer.bias_experts]
    probe = torch.randn_like(want)
    _assert_all_close(_grads(got, inputs, probe), _grads(want, inputs, probe), tol)


def test_top1_router_gate_is_the_softmax_probability_with_full_gradient():
    from dptb.nn.top1_prior import Top1PriorRouter

    router = Top1PriorRouter(4, 4)
    router.net = torch.nn.Identity()
    logits = torch.tensor([[.1, 1.5, -.4, .3]], requires_grad=True)
    route, monitor, _ = router(logits)
    assert route.topk_indices.tolist() == [[1]]
    probabilities = logits.detach().softmax(-1)
    gate = probabilities[:, 1:2]
    torch.testing.assert_close(route.topk_values, gate)
    gradient, = torch.autograd.grad(route.topk_values.sum(), logits)
    expected = -gate * probabilities
    expected[:, 1] += gate[:, 0]
    torch.testing.assert_close(gradient, expected)
    assert torch.all(gradient != 0)
    torch.testing.assert_close(monitor, gate.mean())
    router.eval()
    torch.testing.assert_close(router(logits)[0].topk_values, gate)
    torch.testing.assert_close(router(torch.zeros(2, 4))[0].topk_values, torch.full((2, 1), .25))
    empty, value, _ = router(torch.empty(0, 4))
    assert empty.topk_values.shape == (0, 1) and value == 0


@pytest.mark.parametrize("pair_axis", [False, True])
@pytest.mark.parametrize("mode, device, dtype, tol", EXPERT_BACKENDS)
def test_top1_linear_is_the_gated_selected_expert(mode, device, dtype, tol, pair_axis):
    from dptb.nn.top1_prior import COUNTS, Top1Route

    layer = _layer(mode, 3, 5, num_experts=4, shared=0, dtype=dtype, device=device, seed=71)
    x = torch.randn((7, 2, 3) if pair_axis else (7, 3), device=device, dtype=dtype, requires_grad=True)
    logits = torch.randn(7, 4, device=device, dtype=dtype, requires_grad=True)
    ids = torch.tensor([[0], [3], [1], [3], [0], [2], [1]], device=device)
    gates = logits.softmax(-1).gather(1, ids)
    before = COUNTS["grouped_cuda"]
    got = layer(x, Top1Route(ids, gates))
    assert COUNTS["grouped_cuda"] - before == (1 if mode == "cublas_grouped" else 0)
    want = torch.einsum("n...i,noi->n...o", x, layer.weight_experts[ids[:, 0]])
    want = want + layer.bias_experts[ids[:, 0]].reshape(7, *([1] * (x.ndim - 2)), 5)
    want = want * gates.reshape(7, *([1] * (x.ndim - 1)))
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)
    inputs = [x, logits, *layer.parameters()]
    probe = torch.randn_like(want)
    _assert_all_close(_grads(got, inputs, probe), _grads(want, inputs, probe), tol)


# ---------------------------------------------------------------------------
# weight-space router
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("top_k", [6, None, 2], ids=["all-experts", "dense", "true-topk"])
def test_full_expert_fast_path_matches_the_top_k_route(top_k):
    """With top_k >= num_experts the fast path returns the top-k route's coefficients and leaves the load
    statistics alone; with a true top-k the flag changes nothing."""
    torch.manual_seed(20260423)
    kwargs = dict(in_features=9, num_experts=6, top_k=top_k, aux_loss_free=True, bias_update_speed=0.005)
    base = MOLERouterV3(**kwargs, full_expert_fast_path=False).train()
    fast = MOLERouterV3(**kwargs, full_expert_fast_path=True).train()
    fast.load_state_dict(base.state_dict(), strict=True)
    features = torch.randn(5, 9)
    sizes = torch.tensor([2, 4, 3, 5, 7], dtype=torch.float32)
    for got, want in zip(fast(features, sizes=sizes), base(features, sizes=sizes)):
        torch.testing.assert_close(got, want, atol=1e-7, rtol=1e-7)
    if top_k == 6:
        torch.testing.assert_close(fast.expert_bias, torch.zeros(6))
        torch.testing.assert_close(fast.ema_load, torch.ones(6))
        assert not torch.equal(base.ema_load, fast.ema_load)
    else:
        torch.testing.assert_close(fast.expert_bias, base.expert_bias)
        torch.testing.assert_close(fast.ema_load, base.ema_load)


# ---------------------------------------------------------------------------
# edge-MoE route rows (dptb.nn.embedding.lem_moe_v3_edge)
# ---------------------------------------------------------------------------
class _EchoRouter:
    """Routing coefficients = the router input; dense routing, no top-k metadata."""

    def __call__(self, features, sizes=None):
        return features, features.new_zeros(()), features.new_zeros(())

    def last_topk(self):
        return None, None


def _edge_route(coefficients, bond_type, *, unique_types, compact_min_edges):
    from dptb.nn.embedding.lem_moe_v3_edge import LemMoEV3Edge

    owner = types.SimpleNamespace(
        edge_router_in_features=coefficients.shape[1], edge_router_top1_mode="legacy",
        edge_router_prior_activate=False, edge_router_unique_types=unique_types, edge_moe_compact_dispatch=True,
        edge_moe_compact_min_edges=compact_min_edges, num_experts=coefficients.shape[1], router=_EchoRouter())
    return LemMoEV3Edge._make_edge_moe_globals(owner, coefficients, bond_type)[0]


@pytest.mark.parametrize("mode", ["split_loop", "indexed_ref"])
@pytest.mark.parametrize("unique_types, compact_min_edges", [(False, 16384), (True, 16384), (True, 0)],
                         ids=["per-edge", "unique-expanded", "unique-compact"])
def test_edge_routes_mix_each_edge_with_its_own_coefficients(mode, unique_types, compact_min_edges):
    layer = _layer(mode, 3, 2, num_experts=4, shared=1, seed=20260923)
    bond_type = torch.tensor([2, 0, 2, 1, 0])
    per_type = torch.softmax(torch.randn(3, 4, dtype=torch.float64), dim=-1)
    coefficients = per_type.index_select(0, bond_type)  # equal rows within a bond type
    x = torch.randn(5, 3, dtype=torch.float64)
    got = layer(x, _edge_route(coefficients, bond_type, unique_types=unique_types, compact_min_edges=compact_min_edges))
    torch.testing.assert_close(got, _per_row_reference(layer, x, coefficients, torch.arange(5)))


def test_empty_route_batch_keeps_zero_gradients():
    layer = _layer("split_loop", 3, 2, num_experts=4, shared=1)
    x = torch.empty(0, 2, 3, dtype=torch.float64, requires_grad=True)
    coefficients = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    out = layer(x, MOLEGlobals(coefficients=coefficients, graph_index=torch.empty(0, dtype=torch.long)))
    assert out.shape == (0, 2, 2)
    for grad in torch.autograd.grad(out.sum(), (x, coefficients, *layer.parameters())):
        assert torch.count_nonzero(grad) == 0

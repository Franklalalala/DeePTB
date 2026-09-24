"""SO2CUDA kernel routes of the SO2 layers against the reference routes.

Every parity test sets the route's STRICT switch and replaces the reference fallback of the
layer under test with a function that fails, so a test passes only where the CUDA route
itself produced the output and gradients.
"""
import os

import pytest
import torch

from dptb.tests._requires import requires_module, requires_so2_cuda

experimental = pytest.mark.skipif(
    os.environ.get("DPTB_TEST_SO2_EXPERIMENTAL") != "1",
    reason="experimental SO2CUDA route; set DPTB_TEST_SO2_EXPERIMENTAL=1 to run it",
)
requires_cutlass_root = pytest.mark.skipif(
    not any(os.environ.get(name) for name in ("DPTB_CUTLASS_ROOT", "SO2_CUDA_CUTLASS_ROOT",
                                              "DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT",
                                              "DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_ROOT")),
    reason="needs a CUTLASS root (DPTB_CUTLASS_ROOT) for the CUTLASS/CuTe build",
)


def _forbid(monkeypatch, layer, fallback):
    def fell_back(*args, **kwargs):
        raise AssertionError(f"the CUDA route declined and {fallback} ran")

    monkeypatch.setattr(layer, fallback, fell_back)


def _assert_grads_match(ref, got, atol):
    for (name, p_ref), (name_got, p_got) in zip(ref.named_parameters(), got.named_parameters()):
        assert name == name_got
        if p_ref.grad is None:
            assert p_got.grad is None or not p_got.grad.any(), name
        else:
            assert p_got.grad is not None, name
            torch.testing.assert_close(p_got.grad, p_ref.grad, atol=atol, rtol=atol, msg=name)


# ---------------------------------------------------------------------------
# non-MoE SO2_Linear (dptb.nn.tensor_product)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env, explicit, expected", [
    (None, None, "indexed_sandwich_cuda_multi"),
    ("cublas_grouped", None, "indexed_sandwich_multi"),
    ("indexed_sandwich_multi", "standard", "standard"),
], ids=["default", "legacy-alias-from-env", "explicit-beats-env"])
def test_non_moe_mode_selection(monkeypatch, env, explicit, expected):
    from dptb.nn.tensor_product import SO2_Linear

    if env is None:
        monkeypatch.delenv("DPTB_SO2_M_LINEAR_MODE", raising=False)
    else:
        monkeypatch.setenv("DPTB_SO2_M_LINEAR_MODE", env)
    assert SO2_Linear("1x0e + 1x1o", "1x0e + 1x1o", so2_m_linear_mode=explicit).so2_m_linear_mode == expected


def test_non_moe_rejects_unknown_mode():
    from dptb.nn.tensor_product import SO2_Linear

    with pytest.raises(ValueError, match="so2_m_linear_mode"):
        SO2_Linear("1x0e + 1x1o", "1x0e + 1x1o", so2_m_linear_mode="block_direct")


class _CudaShapedInput:
    def __init__(self, rows, dim):
        self.device = torch.device("cuda")
        self.dtype = torch.float32
        self.shape = (rows, dim)


@pytest.mark.parametrize("env, takes_cuda_route", [
    ({"DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES": "999999"}, False),
    ({"SO2_CUDA_MIN_EDGES": "999999"}, False),
    ({"SO2_CUDA_MIN_EDGES": "999999", "DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES": "0"}, True),
], ids=["dptb-gate", "so2-alias-gate", "dptb-overrides-alias"])
def test_non_moe_minimum_edge_gate(monkeypatch, env, takes_cuda_route):
    from dptb.nn.tensor_product import SO2_Linear

    for name in ("DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES", "SO2_CUDA_MIN_EDGES"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    layer = SO2_Linear("1x0e + 1x1o", "1x0e + 1x1o", so2_m_linear_mode="indexed_sandwich_cuda")
    assert layer._use_indexed_sandwich_cuda_path(_CudaShapedInput(4, layer.irreps_in.dim)) is takes_cuda_route


IRREPS = {
    True: ("3x0e + 4x1o + 2x2e", "2x0e + 3x1o + 3x2e"),   # input no wider than output: radial before the m linears
    False: ("5x0e + 4x1o + 3x2e", "2x0e + 3x1o + 3x2e"),  # input wider: radial after the m linears
}
_MULTI = "indexed_sandwich_cuda_multi"
_LAYOUT = "DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT"
_STRICT = {
    "indexed_sandwich_cuda_multi": "DPTB_SO2_INDEXED_SANDWICH_CUDA_STRICT",
    "indexed_sandwich_cuda": "DPTB_SO2_INDEXED_SANDWICH_CUDA_STRICT",
    "indexed_sandwich_multi": None,  # no fallback inside the route
    "indexed_sandwich_scheduled": "DPTB_SO2_SCHEDULED_SANDWICH_STRICT",
    "indexed_sandwich_materialized": "DPTB_SO2_MATERIALIZED_STRICT",
    "indexed_sandwich_materialized_scheduled": "DPTB_SO2_MATERIALIZED_SCHEDULED_STRICT",
}
# environment that selects or gates a non-MoE route; cleared so only the case's own settings apply
_ROUTE_ENV = (
    _LAYOUT, "SO2_CUDA_GEMM_LAYOUT", "SO2_CUDA_GEMM_STRATEGY",
    "DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE", "SO2_CUDA_EPILOGUE_SCHEDULE",
    "DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES", "SO2_CUDA_MIN_EDGES",
    "DPTB_SO2_INDEXED_SANDWICH_CUDA_MAX_EDGES", "SO2_CUDA_MAX_EDGES",
    "DPTB_SO2_SCHEDULED_SANDWICH_MIN_EDGES", "DPTB_SO2_SCHEDULED_SANDWICH_MAX_EDGES",
    "DPTB_SO2_MATERIALIZED_MIN_EDGES", "DPTB_SO2_MATERIALIZED_MAX_EDGES",
    "SO2_CUDA_MATERIALIZED_MIN_EDGES", "SO2_CUDA_MATERIALIZED_MAX_EDGES",
    "DPTB_SO2_MATERIALIZED_SCHEDULED_MIN_EDGES", "DPTB_SO2_MATERIALIZED_SCHEDULED_MAX_EDGES",
)
NON_MOE_ROUTES = [
    pytest.param(_MULTI, {}, True, id="cuda_multi"),  # production default: raw GEMM, output-major epilogue
    pytest.param(_MULTI, {}, False, id="cuda_multi-back"),
    pytest.param(_MULTI, {"DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE": "per_m"}, True,
                 id="cuda_multi-per_m"),
    *[pytest.param(_MULTI, {_LAYOUT: layout}, True, id=f"cuda_multi-{layout}") for layout in (
        "grouped_raw", "raw_cached", "grouped_raw_v2", "raw_pack_v2", "raw_pack_v2_m0_cuda",
        "raw_pack_v2_m0_cuda_grouped_v2", "raw_pack_v2_m0_cuda_fused")],
    pytest.param("indexed_sandwich_cuda", {}, True, id="cuda"),
    pytest.param("indexed_sandwich_cuda", {}, False, id="cuda-back"),
    pytest.param("indexed_sandwich_multi", {}, True, id="cublas_multi"),
    pytest.param("indexed_sandwich_multi", {}, False, id="cublas_multi-back"),
    *[pytest.param("indexed_sandwich_scheduled", {"DPTB_SO2_SCHEDULED_SANDWICH_MAINLOOP": mainloop}, True,
                   id=f"scheduled-{mainloop}") for mainloop in ("warp_collective", "scalar")],
    *[pytest.param("indexed_sandwich_materialized", {"DPTB_SO2_MATERIALIZED_GEMM_STRATEGY": strategy,
                                                     "DPTB_SO2_MATERIALIZED_EPILOGUE_SCHEDULE": epilogue},
                   True, id=f"materialized-{strategy}-{epilogue}")
      for strategy in ("grouped", "block_dense") for epilogue in ("per_m", "output_major")],
    pytest.param("indexed_sandwich_materialized_scheduled",
                 {"DPTB_SO2_MATERIALIZED_SCHEDULED_MAINLOOP": "warp_collective"}, True, id="materialized_scheduled"),
    *[pytest.param("indexed_sandwich_materialized_scheduled",
                   {"DPTB_SO2_MATERIALIZED_SCHEDULED_GEMM_STRATEGY": "block_dense"}, front,
                   id=f"materialized_scheduled-block_dense{'' if front else '-back'}") for front in (True, False)],
]


@requires_so2_cuda
@pytest.mark.parametrize("mode, env, front", NON_MOE_ROUTES)
def test_non_moe_cuda_route_matches_standard(monkeypatch, mode, env, front):
    """Forward and every gradient of a non-MoE SO2CUDA route equal the standard route."""
    from dptb.nn.tensor_product import SO2_Linear

    for name in _ROUTE_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if _STRICT[mode]:
        monkeypatch.setenv(_STRICT[mode], "1")

    torch.manual_seed(20260523)
    irreps_in, irreps_out = IRREPS[front]
    kwargs = dict(irreps_in=irreps_in, irreps_out=irreps_out, radial_emb=True, latent_dim=7, radial_channels=[11])
    ref = SO2_Linear(**kwargs, so2_m_linear_mode="standard").cuda().train()
    layer = SO2_Linear(**kwargs, so2_m_linear_mode=mode).cuda().train()
    layer.load_state_dict(ref.state_dict(), strict=True)
    assert bool(layer.front) is front
    _forbid(monkeypatch, layer, "_forward_standard")

    n = 29
    x = torch.randn(n, ref.irreps_in.dim, device="cuda")
    r = torch.randn(n, 3, device="cuda")
    latents = torch.randn(n, 7, device="cuda")
    probe = None
    results = []
    for module in (ref, layer):
        xi, li = x.clone().requires_grad_(True), latents.clone().requires_grad_(True)
        out, _ = module(xi, r, li)
        probe = torch.randn_like(out) if probe is None else probe
        (out * probe).sum().backward()
        results.append((out.detach(), xi.grad, li.grad))
    for got, want in zip(results[1], results[0]):
        torch.testing.assert_close(got, want, atol=3e-5, rtol=3e-5)
    _assert_grads_match(ref, layer, atol=6e-5)


# ---------------------------------------------------------------------------
# weight-space MoE SO2_Linear (dptb.nn.tensor_product_moe_v3)
# ---------------------------------------------------------------------------
def _moe_layers(fusion_mode, wigner_apply_mode, mole_linear_mode="cublas_grouped"):
    from dptb.nn.tensor_product_moe_v3 import SO2_Linear

    torch.manual_seed(20260526)
    kwargs = dict(irreps_in="2x0e + 2x1o + 1x2e", irreps_out="1x0e + 2x1o + 2x2e", radial_emb=True, latent_dim=5,
                  radial_channels=[7], num_experts=5, num_shared_experts=1, wigner_apply_mode=wigner_apply_mode,
                  mole_linear_mode=mole_linear_mode)
    ref = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_ref").cuda().train()
    layer = SO2_Linear(**kwargs, so2_fusion_mode=fusion_mode).cuda().train()
    layer.load_state_dict(ref.state_dict(), strict=True)
    return ref, layer


def _routing(kind, num_experts=5):
    """Fresh routing for one layer: graph top-k (grad to topk_values) or split sizes (grad to coefficients)."""
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

    gen = torch.Generator().manual_seed(3)
    if kind == "graph_topk":
        topk_indices = torch.tensor([[0, 2], [1, 3], [2, 4]], device="cuda")
        values = torch.rand(3, 2, generator=gen).cuda()
        values = (values / values.sum(-1, keepdim=True)).requires_grad_(True)
        coefficients = torch.zeros(3, num_experts, device="cuda").scatter(1, topk_indices, values.detach())
        graph_index = torch.tensor([0, 1, 2, 0, 1, 2, 0], device="cuda")
        return MOLEGlobals(coefficients=coefficients, graph_index=graph_index, topk_indices=topk_indices,
                           topk_values=values), values
    coefficients = torch.rand(2, num_experts, generator=gen).cuda()
    coefficients = (coefficients / coefficients.sum(-1, keepdim=True)).requires_grad_(True)
    return MOLEGlobals(coefficients=coefficients, split_sizes=(3, 4)), coefficients


def _assert_moe_route_matches(ref, layer, routing):
    n = 7
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(n, ref.irreps_in.dim, generator=gen).cuda()
    latents = torch.randn(n, 5, generator=gen).cuda()
    R = torch.randn(n, 3, generator=gen).cuda()
    target = torch.randn(n, ref.irreps_out.dim, generator=gen).cuda()
    results = []
    for module in (ref, layer):
        mole_globals, routed = _routing(routing)
        xi, li = x.clone().requires_grad_(True), latents.clone().requires_grad_(True)
        out, _ = module(xi, R, mole_globals, li)
        (out * target).sum().backward()
        results.append((out.detach(), xi.grad, li.grad, routed.grad))
    for got, want, atol in zip(results[1], results[0], (8e-4, 2e-3, 3e-3, 3e-3)):
        torch.testing.assert_close(got, want, atol=atol, rtol=atol)
    _assert_grads_match(ref, layer, atol=4e-3)


_FP0 = "DPTB_SO2_MOE_FUSED_P0_"
FUSED_P0_MODES = [
    pytest.param("scalar", {}, id="scalar"),  # the default without DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE
    pytest.param("indexed_sandwich_multi", {}, id="indexed_sandwich_multi"),
    pytest.param("scalar", {_FP0 + "FUSE_M0": "1", _FP0 + "STRICT_M0": "1"}, marks=experimental, id="scalar-fused_m0"),
    pytest.param("scalar", {_FP0 + "BACKWARD_MODE": "cublas_segmented"}, marks=experimental,
                 id="scalar-cublas_segmented"),
    *[pytest.param(mode, {}, marks=experimental, id=mode) for mode in (
        "indexed_sandwich", "indexed_sandwich_multi_grouped", "indexed_sandwich_multi_direct_warp")],
    *[pytest.param(mode, {}, marks=[experimental, requires_cutlass_root], id=mode) for mode in (
        "cutlass_tiled4", "indexed_sandwich_multi_cute_tiled")],
]


@requires_so2_cuda
@pytest.mark.parametrize("routing", ["graph_topk", "split_sizes"])
@pytest.mark.parametrize("wigner_apply_mode", ["compact_blocks", "full_dense"])
@pytest.mark.parametrize("forward_mode, env", FUSED_P0_MODES)
def test_weight_space_fused_p0_matches_streamed_ref(monkeypatch, forward_mode, env, wigner_apply_mode, routing):
    """Weight-space fused-P0 forward and all gradients equal the streamed reference route."""
    for name in (_FP0 + "FUSE_M0", _FP0 + "STRICT_M0", _FP0 + "BACKWARD_MODE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {_FP0 + "FORWARD_MODE": forward_mode, _FP0 + "STRICT_FORWARD_MODE": "1", **env}.items():
        monkeypatch.setenv(name, value)
    ref, layer = _moe_layers("streamed_m_major_fused_p0", wigner_apply_mode)
    _forbid(monkeypatch, layer, "_forward_streamed_m_major_grouped")
    _assert_moe_route_matches(ref, layer, routing)


@requires_so2_cuda
@requires_module("cuequivariance_torch")
@experimental
def test_weight_space_fused_p0_cueq_sandwich_matches_streamed_ref(monkeypatch):
    monkeypatch.setenv("DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE", "cueq_sandwich")
    monkeypatch.setenv("DPTB_SO2_MOE_FUSED_P0_STRICT_FORWARD_MODE", "1")
    ref, layer = _moe_layers("streamed_m_major_fused_p0", "compact_blocks", mole_linear_mode="cueq_indexed_linear")
    _forbid(monkeypatch, layer, "_forward_streamed_m_major_grouped")
    _assert_moe_route_matches(ref, layer, "graph_topk")


@requires_so2_cuda
@experimental
@pytest.mark.parametrize("include_m0, mainloop", [
    ("0", "warp_collective"),
    ("1", "warp_collective"),
    pytest.param("0", "cutlass_native", marks=requires_cutlass_root),
])
def test_persistent_grouped_p1_matches_streamed_ref(monkeypatch, include_m0, mainloop):
    monkeypatch.setenv("DPTB_SO2_MOE_PERSISTENT_P1_STRICT", "1")
    monkeypatch.setenv("DPTB_SO2_MOE_PERSISTENT_P1_INCLUDE_M0", include_m0)
    monkeypatch.setenv("DPTB_SO2_MOE_PERSISTENT_P1_BACKWARD_MODE", "cuda_cublas_segmented")
    monkeypatch.setenv("DPTB_SO2_MOE_PERSISTENT_P1_MAINLOOP", mainloop)
    if mainloop == "cutlass_native":
        monkeypatch.setenv("DPTB_SO2_MOE_PERSISTENT_P1_FORCE_CUTLASS_NATIVE", "1")
    ref, layer = _moe_layers("streamed_m_major_persistent_grouped_p1", "compact_blocks")
    _forbid(monkeypatch, layer, "_forward_streamed_m_major_grouped")
    _assert_moe_route_matches(ref, layer, "graph_topk")


@requires_so2_cuda
@experimental
def test_persistent_grouped_p1_kernel_matches_hand_computed_m0_m1():
    """P1 kernels on one scalar + one l=1 triplet, identity Wigner, two routes."""
    from dptb.nn.so2_moe_persistent_grouped import _load_extension

    torch.manual_seed(1234)
    cuda = dict(device="cuda", dtype=torch.long)
    x = torch.randn(6, 4, device="cuda")
    graph_index = torch.tensor([0, 1, 0, 1, 1, 0], **cuda)
    w_m0 = torch.tensor([[[1.25]], [[-0.75]]], device="cuda")
    w_m1 = torch.tensor([[[0.5], [0.25]], [[-1.0], [0.125]]], device="cuda")
    b_m0 = torch.tensor([[0.10], [-0.20]], device="cuda")
    b_m1 = torch.tensor([[0.30, -0.40], [0.05, 0.15]], device="cuda")
    empty_float = torch.empty(0, device="cuda")
    args = (
        x, empty_float,  # wigner_mode 0: identity
        torch.tensor([0, 2, 5, 1, 3, 4], **cuda), torch.tensor([0, 3, 6], **cuda),  # edge order, route pointers
        torch.tensor([0, 2, 4, 6, 8], **cuda),  # tiles per (route, m) with block 2x1
        torch.cat([w_m0.reshape(-1), w_m1.reshape(-1)]), torch.tensor([0, w_m0.numel()], **cuda),
        torch.cat([b_m0.reshape(-1), b_m1.reshape(-1)]), torch.tensor([0, b_m0.numel()], **cuda),
        torch.tensor([0, 1], **cuda),  # m values
        torch.tensor([0, 1, 2], **cuda), torch.tensor([0, 1], **cuda), torch.tensor([0, 1], **cuda),  # input maps
        torch.tensor([0, 1, 2], **cuda), torch.tensor([0, 1], **cuda), torch.tensor([0, 1], **cuda),  # output maps
        torch.tensor([0, 1], **cuda), torch.empty(0, **cuda),  # Wigner offsets, compact offsets
        empty_float, torch.empty(0, **cuda),  # no radial
        4, 2, False, False, False, 0, 0, 2, 1, 0,
    )
    ref = torch.zeros(6, 4, device="cuda")
    for e in range(6):
        r = int(graph_index[e])
        x0, x1 = x[e, 1], x[e, 3]
        wr, wi = w_m1[r, 0, 0], w_m1[r, 1, 0]
        ref[e, 0] = x[e, 0] * w_m0[r, 0, 0] + b_m0[r, 0]
        ref[e, 1] = x0 * wr - x1 * wi + b_m1[r, 0]
        ref[e, 3] = x1 * wr + x0 * wi + b_m1[r, 1]
    extension = _load_extension()
    torch.testing.assert_close(extension.persistent_grouped_forward_fp32(*args), ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(extension.persistent_grouped_forward_warp_fp32(*args), ref, rtol=1e-5, atol=1e-6)

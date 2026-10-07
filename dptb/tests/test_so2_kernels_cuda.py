"""Retained SO2CUDA routes match independent PyTorch outputs and gradients."""
import pytest
import torch

from dptb.tests._requires import requires_so2_cuda


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


IRREPS = {
    True: ("3x0e + 4x1o + 2x2e", "2x0e + 3x1o + 3x2e"),   # input no wider than output: radial before the m linears
    False: ("5x0e + 4x1o + 3x2e", "2x0e + 3x1o + 3x2e"),  # input wider: radial after the m linears
}


@requires_so2_cuda
@pytest.mark.parametrize("front", [True, False])
@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize("radial", [True, False])
def test_non_moe_cuda_route_matches_standard(monkeypatch, front, rotate, radial):
    """Forward and every gradient of a non-MoE SO2CUDA route equal the standard route."""
    from dptb.nn.tensor_product import SO2LinearCached as SO2_Linear

    torch.manual_seed(20260523)
    monkeypatch.delenv("DPTB_SO2_M_LINEAR_MODE", raising=False)
    irreps_in, irreps_out = IRREPS[front]
    kwargs = dict(irreps_in=irreps_in, irreps_out=irreps_out, radial_emb=radial, latent_dim=7, radial_channels=[11],
                  rotate_in=rotate, rotate_out=rotate)
    ref = SO2_Linear(**kwargs, so2_m_linear_mode="standard").cuda().train()
    layer = SO2_Linear(**kwargs).cuda().train()
    layer.load_state_dict(ref.state_dict(), strict=True)
    assert bool(layer.front) is front
    _forbid(monkeypatch, layer, "_forward_standard")

    n = 29
    x = torch.randn(n, ref.irreps_in.dim, device="cuda")
    r = torch.randn(n, 3, device="cuda") if rotate else None
    latents = torch.randn(n, 7, device="cuda")
    probe = None
    results = []
    for module in (ref, layer):
        xi, li = x.clone().requires_grad_(True), latents.clone().requires_grad_(True)
        out, cache = module(xi, r, li)
        assert (cache is not None) is rotate
        probe = torch.randn_like(out) if probe is None else probe
        (out * probe).sum().backward()
        results.append((out.detach(), xi.grad) + ((li.grad,) if radial else ()))
        if not radial:
            assert li.grad is None
    for got, want in zip(results[1], results[0]):
        assert torch.isfinite(got).all() and torch.isfinite(want).all()
        torch.testing.assert_close(got, want, atol=3e-5, rtol=3e-5)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    _assert_grads_match(ref, layer, atol=6e-5)


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


@requires_so2_cuda
@pytest.mark.parametrize("routing", ["graph_topk", "split_sizes"])
@pytest.mark.parametrize("wigner_apply_mode", ["compact_blocks", "full_dense"])
@pytest.mark.parametrize("forward_mode", ["scalar", "indexed_sandwich_multi"])
def test_weight_space_fused_p0_matches_streamed_ref(monkeypatch, forward_mode, wigner_apply_mode, routing):
    """Weight-space fused-P0 forward and all gradients equal the streamed reference route."""
    monkeypatch.setenv("DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE", forward_mode)
    ref, layer = _moe_layers("streamed_m_major_fused_p0", wigner_apply_mode)
    _forbid(monkeypatch, layer, "_forward_streamed_m_major_grouped")
    _assert_moe_route_matches(ref, layer, routing)

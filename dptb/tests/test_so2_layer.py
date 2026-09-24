"""SO(2) convolution layers on the torch routes: m blocks, Wigner layouts, streamed routes, expert routes."""
import pytest
import torch
from e3nn import o3

from dptb.nn.tensor_product import SO2_m_Linear, complex_pair_output
from dptb.nn.tensor_product_moe_v3 import (
    MOLEGlobals,
    SO2PostActivationExpertMixer,
    SO2_Linear,
    _Jd,
    batch_wigner_D,
    batch_wigner_D_blocks,
)
from dptb.tests._requires import requires_cuda, requires_module


@pytest.fixture(autouse=True)
def _no_route_env(monkeypatch):
    monkeypatch.delenv("DPTB_SO2_FUSION_MODE", raising=False)
    monkeypatch.delenv("DPTB_MOLE_LINEAR_MODE", raising=False)


def test_so2_m_linear_is_the_complex_product_of_its_weight_halves():
    torch.manual_seed(20260923)
    # m = 1 keeps the l >= 1 channels: 5 in (2x1o + 3x2e), 3 out (1x1o + 2x3o)
    layer = SO2_m_Linear(1, o3.Irreps("4x0e + 2x1o + 3x2e"), o3.Irreps("2x0e + 1x1o + 2x3o")).double()
    x = torch.randn(7, 2, 5, dtype=torch.float64)
    weight_r, weight_i = layer.fc.weight.detach().split(3, dim=0)
    expected = (weight_r + 1j * weight_i) @ (x[:, 0] + 1j * x[:, 1]).unsqueeze(-1)
    got = layer(x)
    torch.testing.assert_close(got[:, 0], expected.real.squeeze(-1))
    torch.testing.assert_close(got[:, 1], expected.imag.squeeze(-1))
    t = torch.randn(4, 2, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: complex_pair_output(v, 3), (t,))


def test_compact_wigner_blocks_match_full_dense_slices():
    alpha = torch.tensor([0.1, -0.2, 0.7], dtype=torch.float64)
    beta = torch.tensor([0.3, 0.5, -0.4], dtype=torch.float64)
    gamma = torch.zeros_like(alpha)
    full = batch_wigner_D(4, alpha, beta, gamma, _Jd)
    compact = batch_wigner_D_blocks(4, alpha, beta, gamma, _Jd)
    for l in range(5):
        torch.testing.assert_close(compact.block(l), full[:, l * l:(l + 1) ** 2, l * l:(l + 1) ** 2])


IRREPS = {
    "out_lmax_gt_in": ("1x0e + 2x1o + 1x2e", "1x0e + 1x1o + 1x2e + 1x3o"),  # radial before the m linears
    "back": ("2x0e + 2x1o + 2x2e", "1x0e + 1x1o + 1x2e"),                   # radial after the m linears
}
ROUTE_CASES = [(route, wigner) for route in ("staged", "streamed_m_major_ref", "streamed_m_major_cueq")
               for wigner in ("compact_blocks", "full_dense") if (route, wigner) != ("staged", "full_dense")]


@pytest.mark.parametrize("rotate_in, rotate_out", [(True, True), (False, True), (True, False), (False, False)])
@pytest.mark.parametrize("irreps", IRREPS.values(), ids=IRREPS.keys())
@pytest.mark.parametrize("route, wigner_apply_mode", ROUTE_CASES)
def test_route_and_wigner_layout_match_staged_dense(route, wigner_apply_mode, irreps, rotate_in, rotate_out):
    """Forward and gradients (x, R, latents, parameters) equal the staged route with a dense Wigner."""
    torch.manual_seed(20260423)
    kwargs = dict(irreps_in=irreps[0], irreps_out=irreps[1], radial_emb=True, latent_dim=5, radial_channels=[7],
                  num_experts=3, num_shared_experts=1, rotate_in=rotate_in, rotate_out=rotate_out)
    ref = SO2_Linear(**kwargs, so2_fusion_mode="staged", wigner_apply_mode="full_dense").double()
    layer = SO2_Linear(**kwargs, so2_fusion_mode=route, wigner_apply_mode=wigner_apply_mode).double()
    layer.load_state_dict(ref.state_dict(), strict=True)
    globals_ = MOLEGlobals(coefficients=torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]], dtype=torch.float64),
                           split_sizes=(2, 3))
    x = torch.randn(5, ref.irreps_in.dim, dtype=torch.float64)
    R = torch.randn(5, 3, dtype=torch.float64)
    latents = torch.randn(5, 5, dtype=torch.float64)
    probe = None
    results = []
    for module in (ref, layer):
        inputs = [t.clone().requires_grad_(True) for t in (x, R, latents)]
        out, wigner = module(inputs[0], inputs[1], globals_, inputs[2])
        if rotate_in or rotate_out:
            assert hasattr(wigner, "block") is (module.wigner_apply_mode == "compact_blocks")
        probe = torch.randn_like(out) if probe is None else probe
        (out * probe).sum().backward()
        results.append((out.detach(), *[t.grad for t in inputs]))
    for got, want in zip(results[1], results[0]):
        if want is None:  # R carries no gradient when nothing is rotated
            assert got is None and not (rotate_in or rotate_out)
        else:
            torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-9)
    for (name, p_ref), (_, p_got) in zip(ref.named_parameters(), layer.named_parameters()):
        torch.testing.assert_close(p_got.grad, p_ref.grad, atol=1e-9, rtol=1e-9, msg=name)


@pytest.mark.parametrize("env_mode", ["streamed_m_major_ref", "streamed_m_major_cueq"])
def test_fusion_mode_env_replaces_the_staged_default(monkeypatch, env_mode):
    monkeypatch.setenv("DPTB_SO2_FUSION_MODE", env_mode)
    layer = SO2_Linear("1x0e + 1x1o", "1x0e + 1x1o", num_experts=2, num_shared_experts=0)
    assert layer.so2_fusion_mode == env_mode


@pytest.mark.parametrize("so2_fusion_mode", ["staged", "streamed_m_major_ref", "streamed_m_major_cueq"])
def test_radial_layer_requires_latents(so2_fusion_mode):
    layer = SO2_Linear("1x0e + 1x1o", "1x0e + 1x1o", radial_emb=True, latent_dim=4, radial_channels=[5],
                       num_experts=2, num_shared_experts=0, rotate_in=False, rotate_out=False,
                       so2_fusion_mode=so2_fusion_mode)
    with pytest.raises(ValueError, match="latents"):
        layer(torch.randn(3, layer.irreps_in.dim), torch.randn(3, 3), None, latents=None)


def test_rejects_an_external_wigner_smaller_than_lmax():
    layer = SO2_Linear("1x0e + 1x1o", "1x0e + 1x1o", num_experts=2, num_shared_experts=0,
                       wigner_apply_mode="full_dense")
    with pytest.raises(ValueError):
        layer(torch.randn(3, layer.irreps_in.dim), torch.randn(3, 3), None, wigner_D_all=torch.eye(1).repeat(3, 1, 1))


@requires_cuda
@requires_module("cuequivariance_torch")
def test_streamed_cueq_indexed_linear_matches_staged_on_cuda():
    torch.manual_seed(20260423)
    kwargs = dict(irreps_in="2x0e + 2x1o + 1x2e", irreps_out="1x0e + 2x1o + 2x2e + 1x3o", radial_emb=True,
                  latent_dim=6, radial_channels=[8], num_experts=6, num_shared_experts=0)
    staged = SO2_Linear(**kwargs, so2_fusion_mode="staged", mole_linear_mode="split_loop").cuda()
    cueq = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_cueq", mole_linear_mode="cueq_indexed_linear").cuda()
    cueq.load_state_dict(staged.state_dict(), strict=True)
    coeffs = torch.rand(3, 6, device="cuda")
    globals_ = MOLEGlobals(coefficients=coeffs / coeffs.sum(-1, keepdim=True), split_sizes=(3, 5, 4))
    x, R, latents = torch.randn(12, staged.irreps_in.dim, device="cuda"), torch.randn(12, 3, device="cuda"), \
        torch.randn(12, 6, device="cuda")
    probe = None
    results = []
    for module in (staged, cueq):
        inputs = [t.clone().requires_grad_(True) for t in (x, R, latents)]
        out, _ = module(inputs[0], inputs[1], globals_, inputs[2])
        probe = torch.randn_like(out) if probe is None else probe
        (out * probe).mean().backward()
        results.append((out.detach(), *[t.grad for t in inputs]))
    for got, want in zip(results[1], results[0]):
        torch.testing.assert_close(got, want, atol=4e-4, rtol=4e-4)


def _expert_layer(num_experts):
    torch.manual_seed(20260604)
    return SO2_Linear(irreps_in="2x0e + 1x1o", irreps_out="2x0e + 1x1o", num_experts=num_experts,
                      num_shared_experts=0, mole_linear_mode="indexed_ref",
                      so2_fusion_mode="streamed_m_major_ref").double()


def test_forward_expert_routes_matches_one_hot_coefficients():
    layer = _expert_layer(3)
    x = torch.randn(6, layer.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    R = torch.randn(6, 3, dtype=torch.float64)
    expert_index = torch.tensor([0, 2, 1, 2, 0, 1])
    one_hot = MOLEGlobals(coefficients=torch.nn.functional.one_hot(expert_index, 3).double(),
                          graph_index=torch.arange(6))
    ref, _ = layer(x, R, one_hot)
    out, _ = layer.forward_expert_routes(x, R, expert_index)
    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)
    got = torch.autograd.grad(out.square().sum(), (x, *layer.parameters()), allow_unused=True)
    want = torch.autograd.grad(ref.square().sum(), (x, *layer.parameters()), allow_unused=True)
    for a, b in zip(got, want):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_post_activation_mixer_mixes_activated_expert_outputs():
    """The mixer averages act(expert_e(x)) under a uniform router, which differs from act(mixed-weight layer)."""
    layer = _expert_layer(2)
    activation = torch.nn.SiLU()
    router = torch.nn.Linear(2, 1, bias=False).double()
    torch.nn.init.zeros_(router.weight)
    mixer = SO2PostActivationExpertMixer(tp=layer, activation=activation, router_from_0e=router, scalar_dim=2,
                                         route_chunk_size=2, checkpoint_routes=True)
    x = torch.randn(5, layer.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    R = torch.randn(5, 3, dtype=torch.float64)
    mole_globals = MOLEGlobals(coefficients=torch.full((1, 2), 0.5, dtype=torch.float64), sizes=torch.tensor([5]),
                               topk_indices=torch.tensor([[0, 1]]), topk_values=torch.full((1, 2), 0.5,
                                                                                            dtype=torch.float64))
    out, _ = mixer(x, R, mole_globals)
    experts = [activation(layer.forward_expert_routes(x, R, torch.full((5,), e))[0]) for e in range(2)]
    torch.testing.assert_close(out, torch.stack(experts, 1).mean(1), atol=1e-10, rtol=1e-10)
    assert (out - activation(layer(x, R, mole_globals)[0])).abs().max() > 1e-8
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in [*layer.parameters(), *router.parameters()])

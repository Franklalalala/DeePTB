import pytest
from pathlib import Path


def test_so2_streamed_grouped_source_has_flash_aggregate_and_direct_fallback():
    source = Path(__file__).parents[1] / "nn" / "tensor_product_moe_v3.py"
    text = source.read_text(encoding="utf-8")
    start = text.index("    def _forward_streamed_m_major_grouped(")
    end = text.index("\n\nclass SO2_m_Linear", start)
    body = text[start:end]

    assert "DPTB_SO2_FLASH_AGGREGATE" in text
    assert "so2_flash_aggregate_mode" in text
    assert "rotate_input_once" in body
    assert "aggregate_output_once" in body
    assert "_rotate_input_l_groups_once" in body
    assert "_accumulate_direct_m0_output_" in body


@pytest.mark.parametrize("wigner_apply_mode", ["compact_blocks", "full_dense"])
@pytest.mark.parametrize(
    "so2_fusion_mode",
    ["streamed_m_major_ref", "streamed_m_major_cueq"],
)
@pytest.mark.parametrize(
    "rotate_in, rotate_out",
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_so2_streamed_handles_out_lmax_gt_in_lmax(so2_fusion_mode, wigner_apply_mode, rotate_in, rotate_out):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(20260423)
    dtype = torch.float64
    irreps_in = "1x0e + 2x1o + 1x2e"
    irreps_out = "1x0e + 1x1o + 1x2e + 1x3o"
    kwargs = dict(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        radial_emb=True,
        latent_dim=5,
        radial_channels=[7],
        num_experts=3,
        num_shared_experts=1,
        rotate_in=rotate_in,
        rotate_out=rotate_out,
        wigner_apply_mode=wigner_apply_mode,
    )
    staged = SO2_Linear(**kwargs, so2_fusion_mode="staged").to(dtype=dtype)
    streamed = SO2_Linear(**kwargs, so2_fusion_mode=so2_fusion_mode).to(dtype=dtype)
    streamed.load_state_dict(staged.state_dict(), strict=True)

    x0 = torch.randn(5, staged.irreps_in.dim, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    R0 = torch.randn(5, 3, dtype=dtype, requires_grad=True)
    R1 = R0.detach().clone().requires_grad_(True)
    lat0 = torch.randn(5, 5, dtype=dtype, requires_grad=True)
    lat1 = lat0.detach().clone().requires_grad_(True)
    coeffs = torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]], dtype=dtype)
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=(2, 3))

    out0, _ = staged(x0, R0, globals_, lat0)
    out1, _ = streamed(x1, R1, globals_, lat1)
    torch.testing.assert_close(out1, out0, atol=1e-9, rtol=1e-9)

    probe = torch.randn_like(out0)
    (out0 * probe).sum().backward()
    (out1 * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-8, rtol=1e-8)
    if rotate_in or rotate_out:
        torch.testing.assert_close(R1.grad, R0.grad, atol=1e-8, rtol=1e-8)
    else:
        assert R0.grad is None
        assert R1.grad is None
    torch.testing.assert_close(lat1.grad, lat0.grad, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "env_mode",
    ["streamed_m_major_ref", "streamed_m_major_cueq"],
)
def test_so2_fusion_mode_env_selects_streamed_modes(monkeypatch, env_mode):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_FUSION_MODE", env_mode)
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        num_experts=2,
        num_shared_experts=0,
    )

    assert layer.so2_fusion_mode == env_mode


@pytest.mark.parametrize("so2_fusion_mode", ["staged", "streamed_m_major_ref", "streamed_m_major_cueq"])
def test_so2_radial_requires_latents(so2_fusion_mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import SO2_Linear

    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        radial_emb=True,
        latent_dim=4,
        radial_channels=[5],
        num_experts=2,
        num_shared_experts=0,
        rotate_in=False,
        rotate_out=False,
        so2_fusion_mode=so2_fusion_mode,
    )
    x = torch.randn(3, layer.irreps_in.dim)
    R = torch.randn(3, 3)

    with pytest.raises(ValueError, match="latents"):
        layer(x, R, None, latents=None)


def test_so2_streamed_grouped_direct_fallback_avoids_output_group_buffer(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    monkeypatch.setenv("DPTB_SO2_FLASH_AGGREGATE", "0")
    torch.manual_seed(20260425)
    dtype = torch.float64
    kwargs = dict(
        irreps_in="1x0e + 1x1o + 1x2e",
        irreps_out="1x0e + 1x1o + 1x2e",
        radial_emb=False,
        num_experts=3,
        num_shared_experts=1,
        rotate_in=False,
        rotate_out=False,
        wigner_apply_mode="compact_blocks",
    )
    staged = SO2_Linear(**kwargs, so2_fusion_mode="staged").to(dtype=dtype)
    streamed = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_cueq").to(dtype=dtype)
    streamed.load_state_dict(staged.state_dict(), strict=True)

    def fail_output_group_alloc(*_args, **_kwargs):
        raise AssertionError("grouped streamed route should write directly to final output")

    monkeypatch.setattr(streamed, "_alloc_output_l_groups", fail_output_group_alloc)

    x = torch.randn(6, staged.irreps_in.dim, dtype=dtype, requires_grad=True)
    r = torch.randn(6, 3, dtype=dtype)
    coeffs = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype=dtype)
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=(2, 4))

    out0, _ = staged(x, r, globals_)
    out1, _ = streamed(x, r, globals_)

    torch.testing.assert_close(out1, out0, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("flash_mode", ["input", "output", "1"])
@pytest.mark.parametrize(
    "rotate_in, rotate_out",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_so2_streamed_grouped_flash_aggregate_matches_direct_forward_and_grad(
        monkeypatch,
        flash_mode,
        rotate_in,
        rotate_out,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(20260425)
    dtype = torch.float64
    kwargs = dict(
        irreps_in="2x0e + 2x1o + 1x2e",
        irreps_out="2x0e + 2x1o + 1x2e",
        radial_emb=False,
        num_experts=3,
        num_shared_experts=1,
        rotate_in=rotate_in,
        rotate_out=rotate_out,
        wigner_apply_mode="compact_blocks",
        so2_fusion_mode="streamed_m_major_cueq",
        mole_linear_mode="indexed_ref",
    )
    monkeypatch.setenv("DPTB_SO2_FLASH_AGGREGATE", "0")
    direct = SO2_Linear(**kwargs).to(dtype=dtype)
    monkeypatch.setenv("DPTB_SO2_FLASH_AGGREGATE", flash_mode)
    flash = SO2_Linear(**kwargs).to(dtype=dtype)
    flash.load_state_dict(direct.state_dict(), strict=True)

    n_rows = 9
    split_sizes = (4, 5)
    x0 = torch.randn(n_rows, direct.irreps_in.dim, dtype=dtype)
    r = torch.randn(n_rows, 3, dtype=dtype)
    coeff0 = torch.softmax(torch.randn(len(split_sizes), 3, dtype=dtype), dim=-1)

    x_direct = x0.detach().clone().requires_grad_(True)
    x_flash = x0.detach().clone().requires_grad_(True)
    coeff_direct = coeff0.detach().clone().requires_grad_(True)
    coeff_flash = coeff0.detach().clone().requires_grad_(True)

    out_direct, _ = direct(x_direct, r, MOLEGlobals(coefficients=coeff_direct, split_sizes=split_sizes))
    out_flash, _ = flash(x_flash, r, MOLEGlobals(coefficients=coeff_flash, split_sizes=split_sizes))
    torch.testing.assert_close(out_flash, out_direct, atol=1e-10, rtol=1e-10)

    grad = torch.randn_like(out_direct)
    out_direct.backward(grad)
    out_flash.backward(grad)

    torch.testing.assert_close(x_flash.grad, x_direct.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(coeff_flash.grad, coeff_direct.grad, atol=1e-10, rtol=1e-10)
    for (_, direct_param), (_, flash_param) in zip(direct.named_parameters(), flash.named_parameters()):
        if direct_param.grad is None and flash_param.grad is None:
            continue
        assert direct_param.grad is not None
        assert flash_param.grad is not None
        torch.testing.assert_close(flash_param.grad, direct_param.grad, atol=1e-10, rtol=1e-10)


def test_so2_streamed_grouped_no_rotation_external_wigner_keeps_direct_output(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    monkeypatch.setenv("DPTB_SO2_FLASH_AGGREGATE", "1")
    torch.manual_seed(20260425)
    dtype = torch.float64
    kwargs = dict(
        irreps_in="1x0e + 1x1o + 1x2e",
        irreps_out="1x0e + 1x1o + 1x2e",
        radial_emb=False,
        num_experts=3,
        num_shared_experts=1,
        rotate_in=False,
        rotate_out=False,
        wigner_apply_mode="compact_blocks",
    )
    staged = SO2_Linear(**kwargs, so2_fusion_mode="staged").to(dtype=dtype)
    streamed = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_cueq").to(dtype=dtype)
    streamed.load_state_dict(staged.state_dict(), strict=True)

    fake_blocks = {
        l: torch.eye(2 * l + 1, dtype=dtype).expand(4, -1, -1).clone()
        for l in range(1, streamed.l_max + 1)
    }
    monkeypatch.setattr(streamed, "_make_wigner_block_cache", lambda _wigner: fake_blocks)

    def fail_output_group_alloc(*_args, **_kwargs):
        raise AssertionError("no-rotation route should stay on direct-output even with external Wigner blocks")

    monkeypatch.setattr(streamed, "_alloc_output_l_groups", fail_output_group_alloc)

    x = torch.randn(4, staged.irreps_in.dim, dtype=dtype)
    r = torch.randn(4, 3, dtype=dtype)
    coeffs = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype=dtype)
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=(2, 2))

    out0, _ = staged(x, r, globals_, wigner_D_all=torch.empty(0))
    out1, _ = streamed(x, r, globals_, wigner_D_all=torch.empty(0))

    torch.testing.assert_close(out1, out0, atol=1e-10, rtol=1e-10)


def test_so2_rejects_too_small_external_wigner_dense():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import SO2_Linear

    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        radial_emb=False,
        num_experts=2,
        num_shared_experts=0,
        rotate_in=True,
        rotate_out=True,
        wigner_apply_mode="full_dense",
    )
    x = torch.randn(3, layer.irreps_in.dim)
    R = torch.randn(3, 3)
    too_small = torch.eye(1).repeat(3, 1, 1)

    with pytest.raises(ValueError, match="l_max|block"):
        layer(x, R, None, wigner_D_all=too_small)


def test_so2_streamed_cueq_indexed_linear_matches_staged_if_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    if not torch.cuda.is_available():
        pytest.skip("SO2 aggressive cueq indexed-linear integration requires CUDA")

    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(20260423)
    device = torch.device("cuda")
    dtype = torch.float32
    kwargs = dict(
        irreps_in="2x0e + 2x1o + 1x2e",
        irreps_out="1x0e + 2x1o + 2x2e + 1x3o",
        radial_emb=True,
        latent_dim=6,
        radial_channels=[8],
        num_experts=6,
        num_shared_experts=0,
        rotate_in=True,
        rotate_out=True,
        wigner_apply_mode="compact_blocks",
    )

    staged = SO2_Linear(
        **kwargs,
        so2_fusion_mode="staged",
        mole_linear_mode="split_loop",
    ).to(device=device, dtype=dtype)
    aggressive = SO2_Linear(
        **kwargs,
        so2_fusion_mode="streamed_m_major_cueq",
        mole_linear_mode="cueq_indexed_linear",
    ).to(device=device, dtype=dtype)
    aggressive.load_state_dict(staged.state_dict(), strict=True)

    split_sizes = (3, 5, 4)
    n_edges = sum(split_sizes)
    coeffs = torch.rand(len(split_sizes), kwargs["num_experts"], device=device, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=split_sizes)

    x0 = torch.randn(n_edges, staged.irreps_in.dim, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    R0 = torch.randn(n_edges, 3, device=device, dtype=dtype, requires_grad=True)
    R1 = R0.detach().clone().requires_grad_(True)
    lat0 = torch.randn(n_edges, kwargs["latent_dim"], device=device, dtype=dtype, requires_grad=True)
    lat1 = lat0.detach().clone().requires_grad_(True)

    out0, _ = staged(x0, R0, globals_, lat0)
    out1, _ = aggressive(x1, R1, globals_, lat1)
    torch.testing.assert_close(out1, out0, atol=3e-4, rtol=3e-4)

    probe = torch.randn_like(out0)
    (out0 * probe).mean().backward()
    (out1 * probe).mean().backward()
    torch.testing.assert_close(x1.grad, x0.grad, atol=4e-4, rtol=4e-4)
    torch.testing.assert_close(R1.grad, R0.grad, atol=4e-4, rtol=4e-4)
    torch.testing.assert_close(lat1.grad, lat0.grad, atol=4e-4, rtol=4e-4)

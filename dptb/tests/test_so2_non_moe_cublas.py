import pytest


def test_non_moe_so2_indexed_sandwich_multi_matches_standard_forward_backward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE SO2 indexed_sandwich_multi backend requires CUDA")

    from dptb.nn.tensor_product import SO2_Linear

    torch.manual_seed(20260523)
    kwargs = dict(
        irreps_in="3x0e + 4x1o + 2x2e",
        irreps_out="2x0e + 3x1o + 3x2e",
        radial_emb=True,
        latent_dim=7,
        radial_channels=[11],
        rotate_in=True,
        rotate_out=True,
    )
    ref = SO2_Linear(**kwargs, so2_m_linear_mode="standard").cuda().float().train()
    grouped = SO2_Linear(**kwargs, so2_m_linear_mode="indexed_sandwich_multi").cuda().float().train()
    grouped.load_state_dict(ref.state_dict(), strict=True)

    x_ref = torch.randn(23, ref.irreps_in.dim, device="cuda", requires_grad=True)
    x_grouped = x_ref.detach().clone().requires_grad_(True)
    r = torch.randn(23, 3, device="cuda")
    latents_ref = torch.randn(23, 7, device="cuda", requires_grad=True)
    latents_grouped = latents_ref.detach().clone().requires_grad_(True)

    out_ref, _ = ref(x_ref, r, latents_ref)
    out_grouped, _ = grouped(x_grouped, r, latents_grouped)
    torch.testing.assert_close(out_grouped, out_ref, atol=2e-5, rtol=2e-5)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_grouped.backward(grad)

    torch.testing.assert_close(x_grouped.grad, x_ref.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(latents_grouped.grad, latents_ref.grad, atol=2e-5, rtol=2e-5)
    for (name_ref, param_ref), (name_grouped, param_grouped) in zip(ref.named_parameters(), grouped.named_parameters()):
        assert name_ref == name_grouped
        if param_ref.grad is not None:
            torch.testing.assert_close(param_grouped.grad, param_ref.grad, atol=3e-5, rtol=3e-5)


def test_non_moe_so2_m_linear_mode_env_selects_indexed_sandwich_multi(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_M_LINEAR_MODE", "indexed_sandwich_multi")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
    )

    assert layer.so2_m_linear_mode == "indexed_sandwich_multi"


def test_non_moe_so2_m_linear_mode_accepts_legacy_cublas_grouped_alias(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_M_LINEAR_MODE", "cublas_grouped")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
    )

    assert layer.so2_m_linear_mode == "indexed_sandwich_multi"


def test_non_moe_so2_m_linear_mode_accepts_cuda_pack_scatter_alias(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_M_LINEAR_MODE", "cuda_pack_scatter")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
    )

    assert layer.so2_m_linear_mode == "indexed_sandwich_cuda"


def test_non_moe_so2_m_linear_mode_accepts_cuda_pack_scatter_multi_alias(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_M_LINEAR_MODE", "cuda_pack_scatter_multi")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
    )

    assert layer.so2_m_linear_mode == "indexed_sandwich_cuda_multi"


def test_non_moe_so2_m_linear_mode_accepts_scheduled_sandwich_alias(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_M_LINEAR_MODE", "scheduled_sandwich")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
    )

    assert layer.so2_m_linear_mode == "indexed_sandwich_scheduled"


def test_non_moe_so2_m_linear_mode_accepts_materialized_sandwich_alias(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_M_LINEAR_MODE", "materialized_sandwich")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
    )

    assert layer.so2_m_linear_mode == "indexed_sandwich_materialized"


@pytest.mark.parametrize(
    ("mode", "epilogue_schedule"),
    [
        ("indexed_sandwich_cuda", None),
        ("indexed_sandwich_cuda_multi", "output_major"),
        ("indexed_sandwich_cuda_multi", "per_m"),
    ],
)
def test_non_moe_so2_indexed_sandwich_cuda_matches_standard_forward_backward(monkeypatch, mode, epilogue_schedule):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE SO2 indexed_sandwich_cuda backends require CUDA")

    from dptb.nn.tensor_product import SO2_Linear

    if epilogue_schedule is not None:
        monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE", epilogue_schedule)

    torch.manual_seed(20260523)
    kwargs = dict(
        irreps_in="3x0e + 4x1o + 2x2e",
        irreps_out="2x0e + 3x1o + 3x2e",
        radial_emb=True,
        latent_dim=7,
        radial_channels=[11],
        rotate_in=True,
        rotate_out=True,
    )
    ref = SO2_Linear(**kwargs, so2_m_linear_mode="standard").cuda().float().train()
    cuda_layer = SO2_Linear(**kwargs, so2_m_linear_mode=mode).cuda().float().train()
    cuda_layer.load_state_dict(ref.state_dict(), strict=True)

    x_ref = torch.randn(29, ref.irreps_in.dim, device="cuda", requires_grad=True)
    x_cuda = x_ref.detach().clone().requires_grad_(True)
    r = torch.randn(29, 3, device="cuda")
    latents_ref = torch.randn(29, 7, device="cuda", requires_grad=True)
    latents_cuda = latents_ref.detach().clone().requires_grad_(True)

    out_ref, _ = ref(x_ref, r, latents_ref)
    out_cuda, _ = cuda_layer(x_cuda, r, latents_cuda)
    torch.testing.assert_close(out_cuda, out_ref, atol=2e-5, rtol=2e-5)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_cuda.backward(grad)

    torch.testing.assert_close(x_cuda.grad, x_ref.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(latents_cuda.grad, latents_ref.grad, atol=2e-5, rtol=2e-5)
    for (name_ref, param_ref), (name_cuda, param_cuda) in zip(ref.named_parameters(), cuda_layer.named_parameters()):
        assert name_ref == name_cuda
        if param_ref.grad is not None:
            torch.testing.assert_close(param_cuda.grad, param_ref.grad, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("mainloop", ["warp_collective", "scalar"])
def test_non_moe_so2_indexed_sandwich_scheduled_matches_standard_forward_backward(monkeypatch, mainloop):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE scheduled SO2 sandwich backend requires CUDA")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_SCHEDULED_SANDWICH_MAINLOOP", mainloop)
    monkeypatch.setenv("DPTB_SO2_SCHEDULED_SANDWICH_STRICT", "1")

    torch.manual_seed(20260524)
    kwargs = dict(
        irreps_in="3x0e + 4x1o + 2x2e",
        irreps_out="2x0e + 3x1o + 3x2e",
        radial_emb=True,
        latent_dim=7,
        radial_channels=[11],
        rotate_in=True,
        rotate_out=True,
    )
    ref = SO2_Linear(**kwargs, so2_m_linear_mode="standard").cuda().float().train()
    scheduled = SO2_Linear(**kwargs, so2_m_linear_mode="indexed_sandwich_scheduled").cuda().float().train()
    scheduled.load_state_dict(ref.state_dict(), strict=True)

    x_ref = torch.randn(31, ref.irreps_in.dim, device="cuda", requires_grad=True)
    x_scheduled = x_ref.detach().clone().requires_grad_(True)
    r = torch.randn(31, 3, device="cuda")
    latents_ref = torch.randn(31, 7, device="cuda", requires_grad=True)
    latents_scheduled = latents_ref.detach().clone().requires_grad_(True)

    out_ref, _ = ref(x_ref, r, latents_ref)
    out_scheduled, _ = scheduled(x_scheduled, r, latents_scheduled)
    torch.testing.assert_close(out_scheduled, out_ref, atol=3e-5, rtol=3e-5)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_scheduled.backward(grad)

    torch.testing.assert_close(x_scheduled.grad, x_ref.grad, atol=4e-5, rtol=4e-5)
    torch.testing.assert_close(latents_scheduled.grad, latents_ref.grad, atol=4e-5, rtol=4e-5)
    for (name_ref, param_ref), (name_scheduled, param_scheduled) in zip(ref.named_parameters(), scheduled.named_parameters()):
        assert name_ref == name_scheduled
        if param_ref.grad is not None:
            torch.testing.assert_close(param_scheduled.grad, param_ref.grad, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("strategy", ["grouped", "block_dense"])
def test_non_moe_so2_indexed_sandwich_materialized_matches_standard_forward_backward(monkeypatch, strategy):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE materialized SO2 sandwich backend requires CUDA")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_MATERIALIZED_GEMM_STRATEGY", strategy)
    monkeypatch.setenv("DPTB_SO2_MATERIALIZED_STRICT", "1")

    torch.manual_seed(20260525)
    kwargs = dict(
        irreps_in="3x0e + 4x1o + 2x2e",
        irreps_out="2x0e + 3x1o + 3x2e",
        radial_emb=True,
        latent_dim=7,
        radial_channels=[11],
        rotate_in=True,
        rotate_out=True,
    )
    ref = SO2_Linear(**kwargs, so2_m_linear_mode="standard").cuda().float().train()
    materialized = SO2_Linear(**kwargs, so2_m_linear_mode="indexed_sandwich_materialized").cuda().float().train()
    materialized.load_state_dict(ref.state_dict(), strict=True)

    x_ref = torch.randn(33, ref.irreps_in.dim, device="cuda", requires_grad=True)
    x_materialized = x_ref.detach().clone().requires_grad_(True)
    r = torch.randn(33, 3, device="cuda")
    latents_ref = torch.randn(33, 7, device="cuda", requires_grad=True)
    latents_materialized = latents_ref.detach().clone().requires_grad_(True)

    out_ref, _ = ref(x_ref, r, latents_ref)
    out_materialized, _ = materialized(x_materialized, r, latents_materialized)
    torch.testing.assert_close(out_materialized, out_ref, atol=3e-5, rtol=3e-5)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_materialized.backward(grad)

    torch.testing.assert_close(x_materialized.grad, x_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(latents_materialized.grad, latents_ref.grad, atol=5e-5, rtol=5e-5)
    for (name_ref, param_ref), (name_materialized, param_materialized) in zip(ref.named_parameters(), materialized.named_parameters()):
        assert name_ref == name_materialized
        if param_ref.grad is not None:
            torch.testing.assert_close(param_materialized.grad, param_ref.grad, atol=6e-5, rtol=6e-5)


def test_non_moe_so2_indexed_sandwich_cuda_shape_gate_falls_back(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES", "999999")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda",
    )

    assert layer.so2_m_linear_mode == "indexed_sandwich_cuda"
    assert not layer._use_indexed_sandwich_cuda_path(__import__("torch").zeros(4, layer.irreps_in.dim))

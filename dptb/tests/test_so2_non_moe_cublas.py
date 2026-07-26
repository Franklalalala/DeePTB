import pytest
import shutil


def _require_cuda_extension_toolchain():
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None or shutil.which("ninja") is None:
        pytest.skip("SO2 CUDA extension tests require nvcc and ninja")


def _require_so2_cuda_ops():
    pytest.importorskip(
        "so2_cuda_ops",
        reason="install the optional DeePTB so2 extra for this CUDA backend",
    )
    _require_cuda_extension_toolchain()


class _FakeCudaInput:
    def __init__(self, torch, rows, dim):
        self.device = type("Device", (), {"type": "cuda"})()
        self.dtype = torch.float32
        self.shape = (rows, dim)


def test_non_moe_so2_indexed_sandwich_multi_matches_standard_forward_backward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE SO2 indexed_sandwich_multi backend requires CUDA")
    _require_cuda_extension_toolchain()

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


def test_non_moe_so2_m_linear_mode_accepts_materialized_scheduler_alias(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_M_LINEAR_MODE", "materialized_cuda_scheduler")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
    )

    assert layer.so2_m_linear_mode == "indexed_sandwich_materialized_scheduled"


def test_non_moe_so2_scheduler_public_function_is_neutral():
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.so2_cuda_scheduler import SO2CudaSchedulerFunction

    assert SO2CudaSchedulerFunction.__name__ == "SO2CudaSchedulerFunction"
    assert SO2CudaSchedulerFunction.__module__ == "dptb.nn.so2_cuda_scheduler"


def test_non_moe_so2_scheduler_single_route_layout_without_graph_index():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("SO2 CUDA scheduler layout helper requires CUDA")
    _require_so2_cuda_ops()

    from dptb.nn.so2_cuda_scheduler import prepare_so2_single_route_layout

    out_ptr = torch.tensor([0, 3, 8, 10], device="cuda", dtype=torch.long)

    edge_order, route_ptr, prefix = prepare_so2_single_route_layout(
        num_rows=17,
        n_problems=3,
        block_m=8,
        block_n=4,
        out_ptr=out_ptr,
        raw_pair_tiles=False,
    )
    torch.testing.assert_close(edge_order.cpu(), torch.arange(17, dtype=torch.long))
    torch.testing.assert_close(route_ptr.cpu(), torch.tensor([0, 17], dtype=torch.long))
    torch.testing.assert_close(prefix.cpu(), torch.tensor([0, 3, 9, 12], dtype=torch.long))

    edge_order_again, route_ptr_again, prefix_again = prepare_so2_single_route_layout(
        num_rows=17,
        n_problems=3,
        block_m=8,
        block_n=4,
        out_ptr=out_ptr,
        raw_pair_tiles=False,
    )
    assert edge_order_again.data_ptr() == edge_order.data_ptr()
    assert route_ptr_again.data_ptr() == route_ptr.data_ptr()
    assert prefix_again.data_ptr() == prefix.data_ptr()

    _, _, pair_prefix = prepare_so2_single_route_layout(
        num_rows=17,
        n_problems=3,
        block_m=8,
        block_n=4,
        out_ptr=out_ptr,
        raw_pair_tiles=True,
    )
    torch.testing.assert_close(pair_prefix.cpu(), torch.tensor([0, 6, 15, 18], dtype=torch.long))

    _, _, nosync_prefix = prepare_so2_single_route_layout(
        num_rows=17,
        n_problems=3,
        block_m=8,
        block_n=4,
        out_ptr=out_ptr,
        raw_pair_tiles=False,
        nosync=True,
    )
    torch.testing.assert_close(nosync_prefix.cpu(), torch.tensor([0, 3, 9, 12], dtype=torch.long))


@pytest.mark.parametrize(
    ("mode", "epilogue_schedule", "gemm_layout"),
    [
        ("indexed_sandwich_cuda", None, None),
        ("indexed_sandwich_cuda_multi", None, None),
        ("indexed_sandwich_cuda_multi", "output_major", None),
        ("indexed_sandwich_cuda_multi", "per_m", None),
        ("indexed_sandwich_cuda_multi", "output_major", "grouped_raw"),
        ("indexed_sandwich_cuda_multi", "output_major", "raw_cached"),
        ("indexed_sandwich_cuda_multi", "output_major", "grouped_raw_v2"),
        ("indexed_sandwich_cuda_multi", "output_major", "raw_pack_v2"),
        ("indexed_sandwich_cuda_multi", "output_major", "raw_pack_v2_m0_cuda"),
        ("indexed_sandwich_cuda_multi", "output_major", "raw_pack_v2_m0_cuda_grouped_v2"),
        ("indexed_sandwich_cuda_multi", "output_major", "raw_pack_v2_m0_cuda_fused"),
        ("indexed_sandwich_cuda_multi", "output_major", "block_direct"),
    ],
)
def test_non_moe_so2_indexed_sandwich_cuda_matches_standard_forward_backward(
    monkeypatch,
    mode,
    epilogue_schedule,
    gemm_layout,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE SO2 indexed_sandwich_cuda backends require CUDA")
    _require_so2_cuda_ops()

    from dptb.nn.tensor_product import SO2_Linear

    if epilogue_schedule is not None:
        monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE", epilogue_schedule)
    if gemm_layout is not None:
        monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT", gemm_layout)

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


@pytest.mark.parametrize(
    ("env_value", "layout", "tag"),
    [
        ("raw_pack_v2", "raw_pack_v2", "raw_output_major_v3_pack_v2"),
        ("pack_v2", "raw_pack_v2", "raw_output_major_v3_pack_v2"),
        ("raw_output_major_v3_pack", "raw_pack_v2", "raw_output_major_v3_pack_v2"),
        ("raw_pack_v2_m0_cuda", "raw_pack_v2_m0_cuda", "raw_output_major_v3_pack_v2_m0_cuda"),
        ("pack_v2_m0_cuda", "raw_pack_v2_m0_cuda", "raw_output_major_v3_pack_v2_m0_cuda"),
        ("raw_pack_v2_m0_cuda_grouped_v2", "raw_pack_v2_m0_cuda_grouped_v2", "raw_output_major_v4_pack_v2_m0_cuda_grouped"),
        ("m0_cuda_grouped_v2", "raw_pack_v2_m0_cuda_grouped_v2", "raw_output_major_v4_pack_v2_m0_cuda_grouped"),
        ("raw_pack_v2_m0_cuda_fused", "raw_pack_v2_m0_cuda_fused", "raw_output_major_v5_pack_v2_m0_fused_epilogue"),
        ("m0_cuda_fused", "raw_pack_v2_m0_cuda_fused", "raw_output_major_v5_pack_v2_m0_fused_epilogue"),
    ],
)
def test_non_moe_so2_cuda_multi_accepts_pack_v2_layout_aliases(monkeypatch, env_value, layout, tag):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT", env_value)
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda_multi",
    )

    assert layer._indexed_sandwich_cuda_multi_gemm_layout() == layout
    assert layer._indexed_sandwich_cuda_multi_execution_tag() == tag


@pytest.mark.parametrize(
    ("spec", "layout"),
    [
        ("indexed_sandwich_cuda_multi:output_major:raw_pack_v2", "raw_pack_v2"),
        ("indexed_sandwich_cuda_multi:output_major:pack_v2", "raw_pack_v2"),
        ("indexed_sandwich_cuda_multi:output_major:raw_pack_v2_m0_cuda", "raw_pack_v2_m0_cuda"),
        ("indexed_sandwich_cuda_multi:output_major:raw_pack_v2_m0_cuda_grouped_v2", "raw_pack_v2_m0_cuda_grouped_v2"),
        ("indexed_sandwich_cuda_multi:output_major:raw_pack_v2_m0_cuda_fused", "raw_pack_v2_m0_cuda_fused"),
    ],
)
def test_bench_so2_variant_parser_accepts_pack_v2_layouts(spec, layout):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not hasattr(torch, "cuda"):
        pytest.skip("bench parser imports torch CUDA helpers")

    from tools.bench_so2_non_moe_cublas import _parse_variant

    variant = _parse_variant(spec)

    assert variant.mode == "indexed_sandwich_cuda_multi"
    assert ("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT", layout) in variant.env


def test_non_moe_so2_indexed_sandwich_cuda_multi_block_complex_matches_standard(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE SO2 block-complex CUDA backend requires CUDA")
    _require_so2_cuda_ops()

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT", "block_complex")
    monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE", "output_major")
    monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_STRICT", "1")

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
    block = SO2_Linear(**kwargs, so2_m_linear_mode="indexed_sandwich_cuda_multi").cuda().float().train()
    block.load_state_dict(ref.state_dict(), strict=True)

    x_ref = torch.randn(27, ref.irreps_in.dim, device="cuda", requires_grad=True)
    x_block = x_ref.detach().clone().requires_grad_(True)
    r = torch.randn(27, 3, device="cuda")
    latents_ref = torch.randn(27, 7, device="cuda", requires_grad=True)
    latents_block = latents_ref.detach().clone().requires_grad_(True)

    out_ref, _ = ref(x_ref, r, latents_ref)
    out_block, _ = block(x_block, r, latents_block)
    torch.testing.assert_close(out_block, out_ref, atol=3e-5, rtol=3e-5)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_block.backward(grad)

    torch.testing.assert_close(x_block.grad, x_ref.grad, atol=4e-5, rtol=4e-5)
    torch.testing.assert_close(latents_block.grad, latents_ref.grad, atol=4e-5, rtol=4e-5)
    for (name_ref, param_ref), (name_block, param_block) in zip(ref.named_parameters(), block.named_parameters()):
        assert name_ref == name_block
        if param_ref.grad is not None:
            torch.testing.assert_close(param_block.grad, param_ref.grad, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("mainloop", ["warp_collective", "scalar"])
def test_non_moe_so2_indexed_sandwich_scheduled_matches_standard_forward_backward(monkeypatch, mainloop):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE scheduled SO2 sandwich backend requires CUDA")
    _require_so2_cuda_ops()

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
@pytest.mark.parametrize("epilogue_schedule", ["per_m", "output_major"])
def test_non_moe_so2_indexed_sandwich_materialized_matches_standard_forward_backward(monkeypatch, strategy, epilogue_schedule):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE materialized SO2 sandwich backend requires CUDA")
    _require_so2_cuda_ops()

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_MATERIALIZED_GEMM_STRATEGY", strategy)
    monkeypatch.setenv("DPTB_SO2_MATERIALIZED_EPILOGUE_SCHEDULE", epilogue_schedule)
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


def test_non_moe_so2_indexed_sandwich_materialized_scheduled_matches_standard_forward_backward(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE materialized scheduled SO2 backend requires CUDA")
    _require_so2_cuda_ops()

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.delenv("DPTB_SO2_MATERIALIZED_SCHEDULED_MIN_EDGES", raising=False)
    monkeypatch.setenv("DPTB_SO2_MATERIALIZED_SCHEDULED_STRICT", "1")
    monkeypatch.setenv("DPTB_SO2_MATERIALIZED_SCHEDULED_MAINLOOP", "warp_collective")

    torch.manual_seed(20260526)
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
    scheduled = SO2_Linear(**kwargs, so2_m_linear_mode="indexed_sandwich_materialized_scheduled").cuda().float().train()
    scheduled.load_state_dict(ref.state_dict(), strict=True)

    x_ref = torch.randn(17, ref.irreps_in.dim, device="cuda", requires_grad=True)
    assert scheduled._use_indexed_sandwich_materialized_scheduled_path(x_ref)
    x_scheduled = x_ref.detach().clone().requires_grad_(True)
    r = torch.randn(17, 3, device="cuda")
    latents_ref = torch.randn(17, 7, device="cuda", requires_grad=True)
    latents_scheduled = latents_ref.detach().clone().requires_grad_(True)

    out_ref, _ = ref(x_ref, r, latents_ref)
    out_scheduled, _ = scheduled(x_scheduled, r, latents_scheduled)
    torch.testing.assert_close(out_scheduled, out_ref, atol=3e-5, rtol=3e-5)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_scheduled.backward(grad)

    torch.testing.assert_close(x_scheduled.grad, x_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(latents_scheduled.grad, latents_ref.grad, atol=5e-5, rtol=5e-5)
    for (name_ref, param_ref), (name_scheduled, param_scheduled) in zip(ref.named_parameters(), scheduled.named_parameters()):
        assert name_ref == name_scheduled
        if param_ref.grad is not None:
            torch.testing.assert_close(param_scheduled.grad, param_ref.grad, atol=6e-5, rtol=6e-5)


@pytest.mark.parametrize(
    ("irreps_in", "irreps_out", "expected_front"),
    [
        ("3x0e + 4x1o + 2x2e", "2x0e + 3x1o + 3x2e", True),
        ("5x0e + 4x1o + 3x2e", "2x0e + 3x1o + 3x2e", False),
    ],
)
def test_non_moe_so2_materialized_scheduled_block_dense_strategy_matches_standard(
    monkeypatch,
    irreps_in,
    irreps_out,
    expected_front,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("non-MoE materialized scheduled SO2 backend requires CUDA")
    _require_so2_cuda_ops()

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.delenv("DPTB_SO2_MATERIALIZED_SCHEDULED_MIN_EDGES", raising=False)
    monkeypatch.setenv("DPTB_SO2_MATERIALIZED_SCHEDULED_STRICT", "1")
    monkeypatch.setenv("DPTB_SO2_MATERIALIZED_SCHEDULED_GEMM_STRATEGY", "block_dense")
    monkeypatch.setenv("DPTB_SO2_MATERIALIZED_SCHEDULED_MAINLOOP", "invalid_mainloop_if_strategy_is_ignored")

    torch.manual_seed(20260527)
    kwargs = dict(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        radial_emb=True,
        latent_dim=7,
        radial_channels=[11],
        rotate_in=True,
        rotate_out=True,
    )
    ref = SO2_Linear(**kwargs, so2_m_linear_mode="standard").cuda().float().train()
    scheduled = SO2_Linear(**kwargs, so2_m_linear_mode="indexed_sandwich_materialized_scheduled").cuda().float().train()
    scheduled.load_state_dict(ref.state_dict(), strict=True)
    assert bool(scheduled.front) is expected_front

    x_ref = torch.randn(19, ref.irreps_in.dim, device="cuda", requires_grad=True)
    x_scheduled = x_ref.detach().clone().requires_grad_(True)
    r = torch.randn(19, 3, device="cuda")
    latents_ref = torch.randn(19, 7, device="cuda", requires_grad=True)
    latents_scheduled = latents_ref.detach().clone().requires_grad_(True)

    out_ref, _ = ref(x_ref, r, latents_ref)
    out_scheduled, _ = scheduled(x_scheduled, r, latents_scheduled)
    torch.testing.assert_close(out_scheduled, out_ref, atol=3e-5, rtol=3e-5)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_scheduled.backward(grad)

    torch.testing.assert_close(x_scheduled.grad, x_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(latents_scheduled.grad, latents_ref.grad, atol=5e-5, rtol=5e-5)


def test_non_moe_so2_indexed_sandwich_cuda_shape_gate_falls_back(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES", "999999")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda",
    )

    assert layer.so2_m_linear_mode == "indexed_sandwich_cuda"
    assert not layer._use_indexed_sandwich_cuda_path(_FakeCudaInput(torch, 4, layer.irreps_in.dim))


def test_non_moe_so2_indexed_sandwich_cuda_shape_gate_accepts_so2_env_alias(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.delenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES", raising=False)
    monkeypatch.setenv("SO2_CUDA_MIN_EDGES", "999999")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda",
    )

    assert not layer._use_indexed_sandwich_cuda_path(_FakeCudaInput(torch, 4, layer.irreps_in.dim))


def test_non_moe_so2_indexed_sandwich_cuda_shape_gate_prefers_dptb_env(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("SO2_CUDA_MIN_EDGES", "999999")
    monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES", "0")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda",
    )

    assert layer._use_indexed_sandwich_cuda_path(_FakeCudaInput(torch, 4, layer.irreps_in.dim))


def test_non_moe_so2_indexed_sandwich_cuda_multi_defaults_to_output_major(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.delenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE", raising=False)
    monkeypatch.delenv("SO2_CUDA_EPILOGUE_SCHEDULE", raising=False)
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda_multi",
    )

    assert layer._indexed_sandwich_cuda_multi_epilogue_schedule() == "output_major"


def test_non_moe_so2_indexed_sandwich_cuda_multi_accepts_so2_epilogue_alias(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.delenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE", raising=False)
    monkeypatch.setenv("SO2_CUDA_EPILOGUE_SCHEDULE", "per_m")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda_multi",
    )

    assert layer._indexed_sandwich_cuda_multi_epilogue_schedule() == "per_m"


def test_non_moe_so2_indexed_sandwich_cuda_multi_epilogue_prefers_dptb_env(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("SO2_CUDA_EPILOGUE_SCHEDULE", "output_major")
    monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE", "per_m")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda_multi",
    )

    assert layer._indexed_sandwich_cuda_multi_epilogue_schedule() == "per_m"


def test_non_moe_so2_indexed_sandwich_cuda_multi_accepts_so2_gemm_strategy_alias(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.delenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT", raising=False)
    monkeypatch.delenv("SO2_CUDA_GEMM_LAYOUT", raising=False)
    monkeypatch.setenv("SO2_CUDA_GEMM_STRATEGY", "grouped_raw")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda_multi",
    )

    assert layer._indexed_sandwich_cuda_multi_gemm_layout() == "grouped_raw"


def test_non_moe_so2_indexed_sandwich_cuda_multi_gemm_prefers_dptb_env(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("SO2_CUDA_GEMM_STRATEGY", "grouped_raw")
    monkeypatch.setenv("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT", "raw")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda_multi",
    )

    assert layer._indexed_sandwich_cuda_multi_gemm_layout() == "raw"


def test_non_moe_so2_indexed_sandwich_cuda_multi_accepts_v2_cached_tag(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv(
        "DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT",
        "raw_output_major_v2_cached",
    )
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda_multi",
    )

    assert layer._indexed_sandwich_cuda_multi_gemm_layout() == "raw_cached"
    assert layer._indexed_sandwich_cuda_multi_execution_tag() == "raw_output_major_v2_cached"


def test_non_moe_so2_indexed_sandwich_cuda_multi_accepts_grouped_v2_tag(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.setenv("SO2_CUDA_GEMM_STRATEGY", "raw_output_major_v2_grouped")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_cuda_multi",
    )

    assert layer._indexed_sandwich_cuda_multi_gemm_layout() == "grouped_raw_v2"
    assert layer._indexed_sandwich_cuda_multi_execution_tag() == "raw_output_major_v2_grouped"


def test_bench_so2_non_moe_cublas_parses_v2_tags():
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from tools.bench_so2_non_moe_cublas import _parse_variant

    cached = _parse_variant("indexed_sandwich_cuda_multi:output_major:raw_cached")
    grouped = _parse_variant("indexed_sandwich_cuda_multi:output_major:grouped_raw_v2")

    assert cached.env == (
        ("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE", "output_major"),
        ("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT", "raw_cached"),
    )
    assert grouped.env == (
        ("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE", "output_major"),
        ("DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT", "grouped_raw_v2"),
    )


def test_non_moe_so2_materialized_shape_gate_accepts_so2_env_alias(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product import SO2_Linear

    monkeypatch.delenv("DPTB_SO2_MATERIALIZED_MIN_EDGES", raising=False)
    monkeypatch.setenv("SO2_CUDA_MATERIALIZED_MIN_EDGES", "999999")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        so2_m_linear_mode="indexed_sandwich_materialized",
    )

    assert not layer._use_indexed_sandwich_materialized_path(_FakeCudaInput(torch, 4, layer.irreps_in.dim))

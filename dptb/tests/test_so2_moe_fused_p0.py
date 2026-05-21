import pytest
import os


def test_so2_fused_p0_mode_is_opt_in():
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.tensor_product_moe_v3 import SO2_Linear

    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        num_experts=2,
        num_shared_experts=0,
        so2_fusion_mode="streamed_m_major_fused_p0",
    )

    assert layer.so2_fusion_mode == "streamed_m_major_fused_p0"


def test_so2_fused_p0_pair_segment_layout_is_cached():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    from dptb.nn.so2_moe_fused_p0 import _pair_segment_layout

    graph_index = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    first = _pair_segment_layout(graph_index, 3)
    second = _pair_segment_layout(graph_index, 3)

    assert first[3] is second[3]
    assert first[2].tolist() == [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert first[3].tolist() == [0, 4, 6, 10]


def test_cutlass_grouped_gemm_matches_torch_if_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUTLASS grouped GEMM smoke requires CUDA")
    if not (
        os.environ.get("DPTB_CUTLASS_ROOT")
        or os.environ.get("DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT")
    ):
        pytest.skip("CUTLASS root not configured")

    from dptb.nn.cutlass_grouped_gemm import grouped_gemm, grouped_gemm_backward_weight

    device = torch.device("cuda")
    torch.manual_seed(20260523)
    x = torch.randn(11, 5, device=device)
    weight = torch.randn(3, 7, 5, device=device)
    ptr = torch.tensor([0, 4, 4, 11], dtype=torch.long)

    out = grouped_gemm(x, ptr, weight)
    ref = torch.empty_like(out)
    ref[0:4] = x[0:4].matmul(weight[0].t())
    ref[4:11] = x[4:11].matmul(weight[2].t())
    torch.testing.assert_close(out, ref, atol=5e-4, rtol=5e-4)

    grad_out = torch.randn_like(out)
    grad_w = grouped_gemm_backward_weight(grad_out, x, ptr, 3)
    ref_w = torch.zeros_like(grad_w)
    ref_w[0] = grad_out[0:4].t().matmul(x[0:4])
    ref_w[2] = grad_out[4:11].t().matmul(x[4:11])
    torch.testing.assert_close(grad_w, ref_w, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("wigner_kind", ["dense", "compact"])
def test_so2_fused_p0_cuda_pair_ops_match_torch_helpers_if_available(wigner_kind):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SO2 MoE fused P0 CUDA pair-op smoke requires CUDA")

    from dptb.nn.so2_moe_fused_p0 import (
        _output_pair_grad_cuda,
        _output_pair_grad_torch,
        _pack_pair_cuda,
        _pack_pair_torch,
        _scatter_pair_grad_cuda,
        _scatter_pair_grad_torch,
    )

    torch.manual_seed(20260524)
    device = torch.device("cuda")
    n_edges = 13
    lmax = 3
    dims = [2 * l + 1 for l in range(lmax + 1)]
    offsets_data = [0]
    for dim in dims[:-1]:
        offsets_data.append(offsets_data[-1] + dim)
    offsets = torch.tensor(offsets_data, device=device, dtype=torch.long)
    d_total = sum(dims)

    dense = torch.zeros((n_edges, d_total, d_total), device=device, dtype=torch.float32)
    compact_pieces = []
    compact_offsets_data = []
    cursor = 0
    for l, dim in enumerate(dims):
        block = torch.randn((n_edges, dim, dim), device=device, dtype=torch.float32)
        start = offsets_data[l]
        dense[:, start:start + dim, start:start + dim] = block
        compact_offsets_data.append(cursor)
        compact_pieces.append(block.reshape(n_edges, dim * dim))
        cursor += dim * dim

    if wigner_kind == "dense":
        wigner = dense.contiguous()
        compact_offsets = torch.empty((0,), device=device, dtype=torch.long)
        wigner_mode = 1
        wigner_stride = d_total
    else:
        wigner = torch.cat(compact_pieces, dim=1).contiguous()
        compact_offsets = torch.tensor(compact_offsets_data, device=device, dtype=torch.long)
        wigner_mode = 2
        wigner_stride = cursor

    m = 1
    in_base = torch.tensor([0, 3, 8, 15], device=device, dtype=torch.long)
    in_l = torch.tensor([1, 2, 3, 3], device=device, dtype=torch.long)
    out_base = torch.tensor([0, 3, 8], device=device, dtype=torch.long)
    out_l = torch.tensor([1, 2, 3], device=device, dtype=torch.long)
    in_dim = 22
    out_dim = 15

    x = torch.randn((n_edges, in_dim), device=device, dtype=torch.float32)
    grad_out = torch.randn((n_edges, out_dim), device=device, dtype=torch.float32)
    grad_pair = torch.randn((n_edges, 2, in_base.numel()), device=device, dtype=torch.float32)

    pack_ref = _pack_pair_torch(
        x, wigner, in_base, in_l, offsets, compact_offsets, m, True, wigner_mode
    )
    pack_cuda = _pack_pair_cuda(
        x, wigner, in_base, in_l, offsets, compact_offsets, m, True, wigner_mode, wigner_stride
    )
    torch.testing.assert_close(pack_cuda, pack_ref, atol=1e-6, rtol=1e-6)

    out_grad_ref = _output_pair_grad_torch(
        grad_out, wigner, out_base, out_l, offsets, compact_offsets, m, True, wigner_mode
    )
    out_grad_cuda = _output_pair_grad_cuda(
        grad_out, wigner, out_base, out_l, offsets, compact_offsets, m, True, wigner_mode, wigner_stride
    )
    torch.testing.assert_close(out_grad_cuda, out_grad_ref, atol=1e-6, rtol=1e-6)

    scatter_ref = _scatter_pair_grad_torch(
        grad_pair, wigner, in_base, in_l, offsets, compact_offsets, in_dim, m, True, wigner_mode
    )
    scatter_cuda = _scatter_pair_grad_cuda(
        grad_pair, wigner, in_base, in_l, offsets, compact_offsets,
        in_dim, m, True, wigner_mode, wigner_stride
    )
    torch.testing.assert_close(scatter_cuda, scatter_ref, atol=1e-6, rtol=1e-6)


def test_so2_fused_p0_forward_matches_streamed_ref_if_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("SO2 MoE fused P0 smoke requires CUDA")

    from dptb.nn.so2_moe_fused_p0 import try_forward_so2_moe_fused_p0
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(20260521)
    device = torch.device("cuda")
    dtype = torch.float32
    kwargs = dict(
        irreps_in="2x0e + 2x1o + 1x2e",
        irreps_out="1x0e + 2x1o + 2x2e",
        radial_emb=True,
        latent_dim=5,
        radial_channels=[7],
        num_experts=5,
        num_shared_experts=1,
        rotate_in=True,
        rotate_out=True,
        wigner_apply_mode="full_dense",
        mole_linear_mode="cublas_grouped",
    )
    ref = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_ref").to(device=device, dtype=dtype)
    fused = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_fused_p0").to(device=device, dtype=dtype)
    fused.load_state_dict(ref.state_dict(), strict=True)

    split_sizes = (3, 5)
    n_edges = sum(split_sizes)
    coeffs = torch.rand(len(split_sizes), kwargs["num_experts"], device=device, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=split_sizes)
    x = torch.randn(n_edges, ref.irreps_in.dim, device=device, dtype=dtype)
    R = torch.randn(n_edges, 3, device=device, dtype=dtype)
    latents = torch.randn(n_edges, kwargs["latent_dim"], device=device, dtype=dtype)

    with torch.no_grad():
        ref_out, wigner = ref(x, R, globals_, latents)
        result = try_forward_so2_moe_fused_p0(fused, x, R, globals_, latents, wigner)

    assert result is not None
    fused_out, returned_wigner = result
    assert returned_wigner is wigner
    torch.testing.assert_close(fused_out, ref_out, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("backward_mode", ["cublas_segmented", "cuda_cublas_segmented"])
def test_so2_fused_p0_compact_backward_matches_streamed_ref_if_available(monkeypatch, backward_mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not torch.cuda.is_available():
        pytest.skip("SO2 MoE fused P0 backward smoke requires CUDA")
    monkeypatch.setenv("DPTB_SO2_MOE_FUSED_P0_BACKWARD_MODE", backward_mode)

    from dptb.nn.so2_moe_fused_p0 import try_forward_so2_moe_fused_p0
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(20260522)
    device = torch.device("cuda")
    dtype = torch.float32
    kwargs = dict(
        irreps_in="2x0e + 2x1o + 1x2e",
        irreps_out="1x0e + 2x1o + 2x2e",
        radial_emb=True,
        latent_dim=5,
        radial_channels=[7],
        num_experts=5,
        num_shared_experts=1,
        rotate_in=True,
        rotate_out=True,
        wigner_apply_mode="compact_blocks",
        mole_linear_mode="cublas_grouped",
    )
    ref = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_ref").to(device=device, dtype=dtype)
    fused = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_fused_p0").to(device=device, dtype=dtype)
    fused.load_state_dict(ref.state_dict(), strict=True)
    ref.train()
    fused.train()

    n_edges = 7
    routes = 3
    top_k = 2
    graph_index = torch.tensor([0, 1, 2, 0, 1, 2, 0], device=device, dtype=torch.long)
    topk_indices = torch.tensor([[0, 2], [1, 3], [2, 4]], device=device, dtype=torch.long)
    topk_values_data = torch.rand((routes, top_k), device=device, dtype=dtype)
    topk_values_data = topk_values_data / topk_values_data.sum(dim=-1, keepdim=True)
    topk_values_ref = topk_values_data.detach().clone().requires_grad_(True)
    topk_values_fused = topk_values_data.detach().clone().requires_grad_(True)
    coeffs_ref = torch.zeros((routes, kwargs["num_experts"]), device=device, dtype=dtype)
    coeffs_fused = torch.zeros_like(coeffs_ref)
    coeffs_ref.scatter_(1, topk_indices, topk_values_ref.detach())
    coeffs_fused.scatter_(1, topk_indices, topk_values_fused.detach())
    ref_globals = MOLEGlobals(
        coefficients=coeffs_ref,
        graph_index=graph_index,
        topk_indices=topk_indices,
        topk_values=topk_values_ref,
    )
    fused_globals = MOLEGlobals(
        coefficients=coeffs_fused,
        graph_index=graph_index,
        topk_indices=topk_indices,
        topk_values=topk_values_fused,
    )

    x_data = torch.randn(n_edges, ref.irreps_in.dim, device=device, dtype=dtype)
    latents_data = torch.randn(n_edges, kwargs["latent_dim"], device=device, dtype=dtype)
    R = torch.randn(n_edges, 3, device=device, dtype=dtype)
    x_ref = x_data.detach().clone().requires_grad_(True)
    x_fused = x_data.detach().clone().requires_grad_(True)
    latents_ref = latents_data.detach().clone().requires_grad_(True)
    latents_fused = latents_data.detach().clone().requires_grad_(True)
    target = torch.randn(n_edges, ref.irreps_out.dim, device=device, dtype=dtype)

    ref_out, wigner = ref(x_ref, R, ref_globals, latents_ref)
    fused_result = try_forward_so2_moe_fused_p0(
        fused,
        x_fused,
        R,
        fused_globals,
        latents_fused,
        wigner,
    )

    assert fused_result is not None
    fused_out, returned_wigner = fused_result
    assert returned_wigner is wigner
    torch.testing.assert_close(fused_out, ref_out, atol=8e-4, rtol=8e-4)

    (ref_out * target).sum().backward()
    (fused_out * target).sum().backward()

    torch.testing.assert_close(x_fused.grad, x_ref.grad, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(latents_fused.grad, latents_ref.grad, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(topk_values_fused.grad, topk_values_ref.grad, atol=3e-3, rtol=3e-3)

    for name, fused_param in fused.named_parameters():
        ref_param = dict(ref.named_parameters())[name]
        assert fused_param.grad is not None, name
        torch.testing.assert_close(fused_param.grad, ref_param.grad, atol=4e-3, rtol=4e-3)

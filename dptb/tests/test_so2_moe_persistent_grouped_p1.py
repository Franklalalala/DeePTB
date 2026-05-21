import math

import pytest
import torch

from dptb.nn.so2_moe_persistent_grouped import _load_extension


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for persistent grouped P1")
def test_direct_persistent_grouped_p1_m0_m1_identity_wigner():
    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.float32

    # Feature layout: one l=0 scalar followed by one l=1 triplet.
    # m=0 uses the scalar channel. m=1 uses rows l-m=0 and l+m=2 of the l=1 triplet.
    n_edges = 6
    in_dim = 4
    out_dim = 4
    n_routes = 2
    x = torch.randn(n_edges, in_dim, device=device, dtype=dtype)

    graph_index = torch.tensor([0, 1, 0, 1, 1, 0], device=device, dtype=torch.long)
    edge_order = torch.tensor([0, 2, 5, 1, 3, 4], device=device, dtype=torch.long)
    route_ptr = torch.tensor([0, 3, 6], device=device, dtype=torch.long)

    m_values = torch.tensor([0, 1], device=device, dtype=torch.long)
    in_ptr = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
    in_base = torch.tensor([0, 1], device=device, dtype=torch.long)
    in_l = torch.tensor([0, 1], device=device, dtype=torch.long)
    out_ptr = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
    out_base = torch.tensor([0, 1], device=device, dtype=torch.long)
    out_l = torch.tensor([0, 1], device=device, dtype=torch.long)
    offsets = torch.tensor([0, 1], device=device, dtype=torch.long)
    compact_offsets = torch.empty(0, device=device, dtype=torch.long)

    # block_m=2, block_n=1.  Each of four (route, m) problems has ceil(3/2)*1 = 2 tiles.
    problem_tile_prefix = torch.tensor([0, 2, 4, 6, 8], device=device, dtype=torch.long)

    w_m0 = torch.tensor([[[1.25]], [[-0.75]]], device=device, dtype=dtype)
    w_m1 = torch.tensor([[[0.5], [0.25]], [[-1.0], [0.125]]], device=device, dtype=dtype)
    weight_flat = torch.cat([w_m0.reshape(-1), w_m1.reshape(-1)]).contiguous()
    weight_offsets = torch.tensor([0, w_m0.numel()], device=device, dtype=torch.long)

    b_m0 = torch.tensor([[0.10], [-0.20]], device=device, dtype=dtype)
    b_m1 = torch.tensor([[0.30, -0.40], [0.05, 0.15]], device=device, dtype=dtype)
    bias_flat = torch.cat([b_m0.reshape(-1), b_m1.reshape(-1)]).contiguous()
    bias_offsets = torch.tensor([0, b_m0.numel()], device=device, dtype=torch.long)

    empty_float = torch.empty(0, device=device, dtype=dtype)
    empty_long = torch.empty(0, device=device, dtype=torch.long)

    args = (
        x,
        empty_float,  # wigner_mode=0 identity
        edge_order,
        route_ptr,
        problem_tile_prefix,
        weight_flat,
        weight_offsets,
        bias_flat,
        bias_offsets,
        m_values,
        in_ptr,
        in_base,
        in_l,
        out_ptr,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        empty_float,  # no radial
        empty_long,
        out_dim,
        n_routes,
        False,
        False,
        False,
        0,
        0,
        2,
        1,
        0,
    )
    out = _load_extension().persistent_grouped_forward_fp32(*args)
    out_warp = _load_extension().persistent_grouped_forward_warp_fp32(*args)

    ref = torch.zeros_like(out)
    for e in range(n_edges):
        r = int(graph_index[e])
        ref[e, 0] += x[e, 0] * w_m0[r, 0, 0] + b_m0[r, 0]
        x0 = x[e, 1 + 0]
        x1 = x[e, 1 + 2]
        wr = w_m1[r, 0, 0]
        wi = w_m1[r, 1, 0]
        ref[e, 1 + 0] += x0 * wr - x1 * wi + b_m1[r, 0]
        ref[e, 1 + 2] += x1 * wr + x0 * wi + b_m1[r, 1]

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_warp, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("include_m0", ["0", "1"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for persistent grouped P1")
def test_persistent_grouped_p1_train_matches_streamed_ref(monkeypatch, include_m0):
    torch.manual_seed(20260527)
    monkeypatch.setenv("DPTB_SO2_MOE_PERSISTENT_P1_INCLUDE_M0", include_m0)
    monkeypatch.setenv("DPTB_SO2_MOE_PERSISTENT_P1_BACKWARD_MODE", "cuda_cublas_segmented")
    monkeypatch.setenv("DPTB_SO2_MOE_PERSISTENT_P1_MAINLOOP", "warp_collective")

    from dptb.nn.so2_moe_persistent_grouped import try_forward_so2_moe_persistent_grouped_p1
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

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
    fused = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_persistent_grouped_p1").to(device=device, dtype=dtype)
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
    fused_result = try_forward_so2_moe_persistent_grouped_p1(
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

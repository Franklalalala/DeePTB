import os

import pytest


def test_cutlass_gemm_universal_smoke_matches_torch_if_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUTLASS GemmUniversal smoke requires CUDA")
    if not (
        os.environ.get("DPTB_CUTLASS_ROOT")
        or os.environ.get("DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_ROOT")
    ):
        pytest.skip("CUTLASS root not configured")

    from dptb.nn.cutlass_so2_gemm_universal_smoke import gemm_universal_smoke

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260522)
    device = torch.device("cuda")
    a = torch.randn((73, 37), device=device, dtype=torch.float32)
    b = torch.randn((29, 37), device=device, dtype=torch.float32)

    out = gemm_universal_smoke(a, b)
    ref = a @ b.t()

    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)


def test_cutlass_gemm_universal_pair_epilogue_scatter_matches_torch_if_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUTLASS GemmUniversal epilogue smoke requires CUDA")
    if not (
        os.environ.get("DPTB_CUTLASS_ROOT")
        or os.environ.get("DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_ROOT")
    ):
        pytest.skip("CUTLASS root not configured")

    from dptb.nn.cutlass_so2_gemm_universal_smoke import gemm_universal_pair_epilogue_smoke

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260523)
    device = torch.device("cuda")
    n_edges, cin, cout = 19, 23, 7
    pair = torch.randn((n_edges, 2, cin), device=device, dtype=torch.float32)
    weight = torch.randn((2 * cout, cin), device=device, dtype=torch.float32)
    wigner = torch.randn((n_edges, 3, 3), device=device, dtype=torch.float32)

    out = gemm_universal_pair_epilogue_smoke(pair, weight, wigner)

    raw = pair.reshape(n_edges * 2, cin) @ weight.t()
    raw = raw.reshape(n_edges, 2, 2 * cout)
    ref = torch.zeros((n_edges, 3 * cout), device=device, dtype=torch.float32)
    for c in range(cout):
        rr0 = raw[:, 0, c]
        ii0 = raw[:, 0, cout + c]
        rr1 = raw[:, 1, c]
        ii1 = raw[:, 1, cout + c]
        y0 = rr0 - ii1
        y1 = rr1 + ii0
        ref[:, 3 * c : 3 * c + 3] = y0[:, None] * wigner[:, :, 0] + y1[:, None] * wigner[:, :, 2]

    torch.testing.assert_close(out, ref, atol=4e-5, rtol=4e-5)


def test_cutlass_gemm_universal_raw_a_loader_pair_epilogue_matches_packed_if_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUTLASS GemmUniversal raw-A smoke requires CUDA")
    if not (
        os.environ.get("DPTB_CUTLASS_ROOT")
        or os.environ.get("DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_ROOT")
    ):
        pytest.skip("CUTLASS root not configured")

    from dptb.nn.cutlass_so2_gemm_universal_smoke import (
        gemm_universal_pair_epilogue_smoke,
        gemm_universal_raw_a_loader_pair_epilogue_smoke,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260524)
    device = torch.device("cuda")
    n_edges, cin, cout = 17, 24, 6
    x = torch.randn((n_edges, 3 * cin), device=device, dtype=torch.float32)
    weight = torch.randn((2 * cout, cin), device=device, dtype=torch.float32)
    wigner = torch.randn((n_edges, 3, 3), device=device, dtype=torch.float32)

    pair = torch.empty((n_edges, 2, cin), device=device, dtype=torch.float32)
    x3 = x.reshape(n_edges, cin, 3)
    pair[:, 0, :] = (x3 * wigner[:, None, :, 0]).sum(dim=2)
    pair[:, 1, :] = (x3 * wigner[:, None, :, 2]).sum(dim=2)

    ref = gemm_universal_pair_epilogue_smoke(pair, weight, wigner)
    out = gemm_universal_raw_a_loader_pair_epilogue_smoke(x, weight, wigner)

    torch.testing.assert_close(out, ref, atol=5e-5, rtol=5e-5)

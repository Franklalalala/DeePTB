import math

import pytest


def _require_cueq_cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    if not torch.cuda.is_available():
        pytest.skip("cuEquivariance Wigner tests require CUDA")
    return torch


def _deterministic_angles(torch, *, dtype, device):
    fixed = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.7, 1.1],
            [-1.3, 0.9, -0.4],
            [math.pi / 2.0, math.pi / 3.0, -math.pi / 5.0],
            [-math.pi + 0.03, 1.7, math.pi - 0.11],
        ],
        dtype=dtype,
        device=device,
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260423)
    random_angles = (torch.rand((11, 3), generator=gen, dtype=dtype) * 2.0 - 1.0) * math.pi
    return torch.cat([fixed, random_angles.to(device)], dim=0)


def test_cueq_rotation_matches_deeptb_wigner_blocks_l0_to_l6():
    torch = _require_cueq_cuda()
    from dptb.nn.tensor_product_moe_v3 import (
        _Jd,
        batch_wigner_D,
        batch_wigner_D_cueq_blocks,
    )

    dtype = torch.float64
    device = torch.device("cuda")
    angles = _deterministic_angles(torch, dtype=dtype, device=device)
    alpha, beta, gamma = angles.unbind(dim=1)

    full = batch_wigner_D(6, alpha, beta, gamma, _Jd)
    cueq = batch_wigner_D_cueq_blocks(6, alpha, beta, gamma)

    for l_value in range(7):
        dim = 2 * l_value + 1
        start = l_value * l_value
        torch.testing.assert_close(
            cueq.block(l_value),
            full[:, start : start + dim, start : start + dim],
            atol=1e-8,
            rtol=1e-8,
        )


def test_cueq_optional_dependency_is_lazy(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn import tensor_product_moe_v3 as tp

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in {"cuequivariance", "cuequivariance_torch"}:
            raise ImportError("blocked cueq import for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    tp._CUEQ_ROTATION_CACHE.clear()

    alpha = torch.tensor([0.1], dtype=torch.float64)
    beta = torch.tensor([0.2], dtype=torch.float64)
    gamma = torch.zeros_like(alpha)

    tp.batch_wigner_D_blocks(1, alpha, beta, gamma, tp._Jd)
    with pytest.raises(ImportError, match="so2_wigner_apply_mode='cueq_rotation'"):
        tp.batch_wigner_D_cueq_blocks(1, alpha, beta, gamma)


def test_so2_linear_cueq_mode_matches_full_dense_forward_and_grad():
    torch = _require_cueq_cuda()
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(23)
    irreps = "3x0e + 2x1e + 2x2e + 1x3e + 1x4e + 1x5e + 1x6e"
    dense_layer = SO2_Linear(
        irreps,
        irreps,
        num_experts=4,
        num_shared_experts=1,
        wigner_apply_mode="full_dense",
    ).cuda()
    cueq_layer = SO2_Linear(
        irreps,
        irreps,
        num_experts=4,
        num_shared_experts=1,
        wigner_apply_mode="cueq_rotation",
    ).cuda()
    cueq_layer.load_state_dict(dense_layer.state_dict())

    dtype = torch.float64
    dense_layer = dense_layer.to(dtype=dtype)
    cueq_layer = cueq_layer.to(dtype=dtype)

    x_dense = torch.randn(7, dense_layer.irreps_in.dim, device="cuda", dtype=dtype, requires_grad=True)
    x_cueq = x_dense.detach().clone().requires_grad_(True)
    R_dense = (torch.randn(7, 3, device="cuda", dtype=dtype) + 0.2).requires_grad_(True)
    R_cueq = R_dense.detach().clone().requires_grad_(True)
    coeffs = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ],
        device="cuda",
        dtype=dtype,
    )
    mole_globals = MOLEGlobals(coefficients=coeffs, sizes=torch.tensor([3, 4], device="cuda"))

    dense_out, dense_wigner = dense_layer(x_dense, R_dense, mole_globals)
    cueq_out, cueq_wigner = cueq_layer(x_cueq, R_cueq, mole_globals)

    torch.testing.assert_close(cueq_out, dense_out, atol=1e-8, rtol=1e-8)
    assert not hasattr(dense_wigner, "block")
    assert hasattr(cueq_wigner, "apply")

    dense_out.square().sum().backward()
    cueq_out.square().sum().backward()
    torch.testing.assert_close(x_cueq.grad, x_dense.grad, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(R_cueq.grad, R_dense.grad, atol=1e-8, rtol=1e-8)

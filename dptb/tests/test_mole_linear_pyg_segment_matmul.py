import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _clear_mole_linear_mode_env(monkeypatch):
    monkeypatch.delenv("DPTB_MOLE_LINEAR_MODE", raising=False)


def _install_fake_pyg_segment_matmul(monkeypatch):
    torch = pytest.importorskip("torch")

    def segment_matmul(inputs, ptr, other, bias=None):
        parts = []
        for idx in range(other.shape[0]):
            start = int(ptr[idx])
            end = int(ptr[idx + 1])
            out = inputs[start:end].matmul(other[idx])
            if bias is not None:
                out = out + bias[idx]
            parts.append(out)
        return torch.cat(parts, dim=0) if parts else inputs.new_empty((0, other.shape[-1]))

    fake_pyg_lib = types.SimpleNamespace(
        ops=types.SimpleNamespace(segment_matmul=segment_matmul),
        sampler=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "pyg_lib", fake_pyg_lib)


def _make_globals(torch, *, device, sizes=(3, 11, 2, 7), num_experts=6, dtype=None):
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

    dtype = torch.float32 if dtype is None else dtype
    coeffs = torch.rand(len(sizes), num_experts, device=device, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    return MOLEGlobals(coefficients=coeffs, split_sizes=sizes), sum(sizes)


def _assert_pyg_matches_split_loop(torch, *, shape, bias, num_shared_experts, device, sizes=(3, 11, 2, 7)):
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    globals_, _ = _make_globals(torch, device=device, sizes=sizes)
    base = MOLELinear(
        shape[-1],
        9,
        num_experts=6,
        num_shared_experts=num_shared_experts,
        bias=bias,
        mole_linear_mode="split_loop",
    ).to(device=device, dtype=torch.float32)
    pyg = MOLELinear(
        shape[-1],
        9,
        num_experts=6,
        num_shared_experts=num_shared_experts,
        bias=bias,
        mole_linear_mode="pyg_segment_matmul",
    ).to(device=device, dtype=torch.float32)
    pyg.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0 = base(x0, globals_)
    y1 = pyg(x1, globals_)
    torch.testing.assert_close(y1, y0, atol=2e-5, rtol=2e-5)

    probe = torch.randn_like(y0)
    (y0 * probe).mean().backward()
    (y1 * probe).mean().backward()
    torch.testing.assert_close(x1.grad, x0.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(pyg.weight_experts.grad, base.weight_experts.grad, atol=2e-5, rtol=2e-5)
    if bias:
        torch.testing.assert_close(pyg.bias_experts.grad, base.bias_experts.grad, atol=2e-5, rtol=2e-5)
    if num_shared_experts > 0:
        torch.testing.assert_close(pyg.weight_shared.grad, base.weight_shared.grad, atol=2e-5, rtol=2e-5)
        if bias:
            torch.testing.assert_close(pyg.bias_shared.grad, base.bias_shared.grad, atol=2e-5, rtol=2e-5)


def test_pyg_segment_matmul_matches_split_loop_with_fake_backend(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_fake_pyg_segment_matmul(monkeypatch)

    torch.manual_seed(20260521)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, n_edges = _make_globals(torch, device=device)
    _assert_pyg_matches_split_loop(
        torch,
        shape=(n_edges, 13),
        bias=True,
        num_shared_experts=1,
        device=device,
    )
    _assert_pyg_matches_split_loop(
        torch,
        shape=(n_edges, 2, 13),
        bias=True,
        num_shared_experts=1,
        device=device,
    )
    _assert_pyg_matches_split_loop(
        torch,
        shape=(n_edges, 2, 13),
        bias=False,
        num_shared_experts=0,
        device=device,
    )


def test_pyg_segment_ptr_cache_uses_cpu_ptr_and_inner_size(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_fake_pyg_segment_matmul(monkeypatch)
    from dptb.nn.tensor_product_moe_v3 import _mole_segment_ptr_cached

    globals_, n_edges = _make_globals(torch, device=torch.device("cpu"), sizes=(2, 5, 3))
    split_sizes, ptr0 = _mole_segment_ptr_cached(globals_, n_edges, inner_size=2)
    _, ptr1 = _mole_segment_ptr_cached(globals_, n_edges, inner_size=2)

    assert split_sizes == (2, 5, 3)
    assert ptr0 is ptr1
    assert ptr0.device.type == "cpu"
    assert ptr0.tolist() == [0, 4, 14, 20]


def test_pyg_segment_matmul_requires_matching_graph_count(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_fake_pyg_segment_matmul(monkeypatch)
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    coeffs = torch.ones(2, 4) / 4
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=(2, 3, 4))
    layer = MOLELinear(3, 5, num_experts=4, mole_linear_mode="pyg_segment_matmul")
    x = torch.randn(9, 3)
    with pytest.raises(ValueError, match="split_sizes"):
        layer(x, globals_)


def test_pyg_segment_matmul_rejects_non_float32(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_fake_pyg_segment_matmul(monkeypatch)
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    globals_, n_edges = _make_globals(torch, device=torch.device("cpu"), sizes=(2, 3), dtype=torch.float64)
    layer = MOLELinear(4, 6, num_experts=6, mole_linear_mode="pyg_segment_matmul").to(dtype=torch.float64)
    x = torch.randn(n_edges, 4, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="float32"):
        layer(x, globals_)


def test_pyg_segment_matmul_real_backend_if_available(monkeypatch):
    torch = pytest.importorskip("torch")
    pyg_lib = pytest.importorskip("pyg_lib")
    if not hasattr(getattr(pyg_lib, "ops", None), "segment_matmul"):
        pytest.skip("pyg_lib.ops.segment_matmul is unavailable")

    torch.manual_seed(20260522)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, n_edges = _make_globals(torch, device=device, sizes=(4, 17, 3, 9))
    _assert_pyg_matches_split_loop(
        torch,
        shape=(n_edges, 2, 11),
        bias=True,
        num_shared_experts=1,
        device=device,
        sizes=(4, 17, 3, 9),
    )


def test_mole_linear_env_selects_pyg_segment_matmul(monkeypatch):
    pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    monkeypatch.setenv("DPTB_MOLE_LINEAR_MODE", "pyg_segment_matmul")
    assert MOLELinear(4, 4).mole_linear_mode == "pyg_segment_matmul"

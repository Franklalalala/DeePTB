import pytest


def _assert_forward_and_grad_close(torch, tp_ref, tp_fast, x, y, fast_weight=None):
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    x_fast = x.detach().clone().requires_grad_(True)
    y_fast = y.detach().clone().requires_grad_(True)

    out_ref = tp_ref(x_ref, y_ref)
    out_fast = tp_fast(x_fast, y_fast)

    torch.testing.assert_close(out_fast, out_ref, atol=1e-10, rtol=1e-10)

    loss_ref = out_ref.square().sum()
    loss_fast = out_fast.square().sum()
    loss_ref.backward()
    loss_fast.backward()

    torch.testing.assert_close(x_fast.grad, x_ref.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(y_fast.grad, y_ref.grad, atol=1e-10, rtol=1e-10)
    weight = tp_fast.weight if fast_weight is None else fast_weight
    torch.testing.assert_close(weight.grad, tp_ref.weight.grad, atol=1e-10, rtol=1e-10)


def test_scalar_fast_matches_uvu_tensor_product_forward_and_grad():
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")
    from dptb.nn.embedding.lem_moe_v3 import ScalarOnehotTP

    torch.manual_seed(20260424)
    dtype = torch.float64
    irreps = o3.Irreps("3x0e + 2x1o + 1x2e")
    onehot_irreps = o3.Irreps("7x0e")
    instructions = [(i, 0, i, "uvu", True) for i, _ in enumerate(irreps)]

    tp_ref = o3.TensorProduct(irreps, onehot_irreps, irreps, instructions).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(11, irreps.dim, dtype=dtype)
    y = torch.randn(11, onehot_irreps.dim, dtype=dtype)

    _assert_forward_and_grad_close(torch, tp_ref, tp_fast, x, y)


@pytest.mark.parametrize("leading", [(7,), (0,), (3, 5)])
def test_scalar_fast_packed_uvu_matches_repeated_irreps(leading):
    """The packed-gain uvu path (UpdateNode/UpdateEdge) with repeated irrep types and
    two instructions writing the same output irrep."""
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")
    from dptb.nn.embedding.lem_moe_v3 import ScalarOnehotTP

    torch.manual_seed(20260923)
    dtype = torch.float64
    irreps = o3.Irreps("2x0e + 1x1o + 3x2e + 4x3o + 2x0e + 1x1o")
    onehot_irreps = o3.Irreps("5x0e")
    instructions = [(i, 0, i, "uvu", True) for i, _ in enumerate(irreps)]
    instructions.append((4, 0, 0, "uvu", True))  # a second path into output irrep 0

    tp_ref = o3.TensorProduct(irreps, onehot_irreps, irreps, instructions).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(*leading, irreps.dim, dtype=dtype)
    y = torch.randn(*leading, onehot_irreps.dim, dtype=dtype)

    _assert_forward_and_grad_close(torch, tp_ref, tp_fast, x, y)


def test_scalar_fast_matches_fully_connected_scalar_tp_forward_and_grad():
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")
    from dptb.nn.embedding.lem_moe_v3 import ScalarOnehotTP

    torch.manual_seed(20260424)
    dtype = torch.float64
    irreps_in = o3.Irreps("3x0e + 2x1o + 2x2e")
    irreps_out = o3.Irreps("2x0e + 3x1o + 1x2e")
    onehot_irreps = o3.Irreps("5x0e")

    tp_ref = o3.FullyConnectedTensorProduct(irreps_in, onehot_irreps, irreps_out).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(13, irreps_in.dim, dtype=dtype)
    y = torch.randn(13, onehot_irreps.dim, dtype=dtype)

    _assert_forward_and_grad_close(torch, tp_ref, tp_fast, x, y)


def test_scalar_fast_fully_connected_unsimplified_irreps():
    """Orbital-pair irreps list each irrep type several times, so one output irrep
    receives paths from several input irreps; an irrep type missing from the input
    yields a zero output block; two scalar blocks give two groups per irrep type."""
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")
    from dptb.nn.embedding.lem_moe_v3 import ScalarOnehotTP

    torch.manual_seed(20260923)
    dtype = torch.float64
    irreps_in = o3.Irreps("3x0e + 2x1o + 2x0e + 1x2e + 3x1o + 1x0e + 2x2e")
    irreps_out = o3.Irreps("2x0e + 1x1o + 2x1e + 3x0e + 2x2e + 1x1o")
    onehot_irreps = o3.Irreps("4x0e + 3x0e")

    tp_ref = o3.FullyConnectedTensorProduct(irreps_in, onehot_irreps, irreps_out).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(9, irreps_in.dim, dtype=dtype)
    y = torch.randn(9, onehot_irreps.dim, dtype=dtype)

    _assert_forward_and_grad_close(torch, tp_ref, tp_fast, x, y)


def test_scalar_fast_incomplete_uvw_paths_match_tensor_product():
    """uvw instructions that do not couple every input with every output irrep of
    their type take the per-instruction path."""
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")
    from dptb.nn.embedding.lem_moe_v3 import ScalarOnehotTP

    torch.manual_seed(20260923)
    dtype = torch.float64
    irreps_in = o3.Irreps("2x0e + 3x1o + 1x0e")
    irreps_out = o3.Irreps("2x0e + 1x1o + 3x0e")
    onehot_irreps = o3.Irreps("4x0e")
    instructions = [
        (0, 0, 0, "uvw", True),
        (2, 0, 0, "uvw", True),
        (0, 0, 2, "uvw", True),  # (2 -> 2) missing: the 0e group is incomplete
        (1, 0, 1, "uvw", True),  # complete 1o group
    ]

    tp_ref = o3.TensorProduct(irreps_in, onehot_irreps, irreps_out, instructions).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(6, irreps_in.dim, dtype=dtype)
    y = torch.randn(6, onehot_irreps.dim, dtype=dtype)

    _assert_forward_and_grad_close(torch, tp_ref, tp_fast, x, y)


def test_scalar_fast_on_an_e3nn_module_trains_its_weight():
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")
    from dptb.nn.embedding.lem_moe_v3 import _apply_onehot_tp

    torch.manual_seed(20260923)
    dtype = torch.float64
    irreps_in = o3.Irreps("2x0e + 1x1o + 2x0e")
    irreps_out = o3.Irreps("1x0e + 2x1o + 2x0e")
    onehot_irreps = o3.Irreps("3x0e")

    tp_ref = o3.FullyConnectedTensorProduct(irreps_in, onehot_irreps, irreps_out).to(dtype=dtype)
    tp_e3nn = o3.FullyConnectedTensorProduct(irreps_in, onehot_irreps, irreps_out).to(dtype=dtype)
    tp_e3nn.load_state_dict(tp_ref.state_dict())
    keys = set(tp_e3nn.state_dict())

    class _Apply(torch.nn.Module):
        def forward(self, x, y):
            return _apply_onehot_tp(tp_e3nn, x, y, "scalar_fast")

    x = torch.randn(5, irreps_in.dim, dtype=dtype)
    y = torch.randn(5, onehot_irreps.dim, dtype=dtype)
    _assert_forward_and_grad_close(torch, tp_ref, _Apply(), x, y, fast_weight=tp_e3nn.weight)
    assert set(tp_e3nn.state_dict()) == keys


def test_scalar_fast_state_dict_holds_only_the_e3nn_weight():
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")
    from dptb.nn.embedding.lem_moe_v3 import ScalarOnehotTP

    irreps_in = o3.Irreps("2x0e + 1x1o + 2x0e")
    irreps_out = o3.Irreps("1x0e + 2x1o")
    tp_ref = o3.FullyConnectedTensorProduct(irreps_in, "3x0e", irreps_out)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref)

    assert list(tp_fast.state_dict()) == ["weight"]
    restored = ScalarOnehotTP.from_e3nn(o3.FullyConnectedTensorProduct(irreps_in, "3x0e", irreps_out))
    restored.load_state_dict({"weight": tp_ref.weight.detach()}, strict=True)
    torch.testing.assert_close(restored.weight, tp_ref.weight)

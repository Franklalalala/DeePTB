"""Split-once forwards of the irreps layers against e3nn or an explicit per-channel formula."""
import os

import pytest
import torch
from e3nn import nn as e3nn_nn
from e3nn import o3

from dptb.nn import e3nn_fast
from dptb.nn.embedding.eqv3_grid_helpers import EquivariantMergedRMSNormFlat, _get_grid_mats
from dptb.nn.embedding.lem_moe_v3 import ScalarOnehotTP, _apply_onehot_tp, _scalar_onehot_tp_fast
from dptb.nn.embedding.lem_moe_v3_plugins import build_gate_activation
from dptb.nn.embedding.oeq_tp import OEQTensorProduct, get_feasible_tp
from dptb.nn.rescale import E3ElementLinear
from dptb.nn.tensor_product_moe_v3 import permute_rows
from dptb.tests._requires import requires_cuda, requires_module

opt_in_oeq = pytest.mark.skipif(os.environ.get("DPTB_TEST_OEQ") != "1",
                                 reason="opt-in OpenEquivariance JIT build; set DPTB_TEST_OEQ=1 to run it")


def _grads(out, inputs, probe):
    return torch.autograd.grad((out * probe).sum(), inputs)


@pytest.mark.parametrize("irreps_in, irreps_out, biases", [
    ("16x0e+8x1o+4x2e+2x3o", "12x0e+6x1o+3x2e+2x3o", True),
    ("5x0e+3x0o+4x1o+4x1e+2x2e", "7x0e+2x0o+3x1o+3x1e+2x2e+1x3o", True),
    ("3x0e+2x1o+2x0e+1x1o", "4x0e+3x1o", False),     # repeated irreps: two paths per output
    ("4x1o+2x2e", "3x1o+2x2e+1x3o", False),          # an output without any input
])
def test_linear_matches_e3nn(irreps_in, irreps_out, biases):
    torch.manual_seed(0)
    ref = o3.Linear(irreps_in, irreps_out, biases=biases).double()
    fast = e3nn_fast.Linear(irreps_in, irreps_out, biases=biases).double()
    with torch.no_grad():
        if ref.bias.numel():
            ref.bias.normal_()
    fast.load_state_dict(ref.state_dict())
    assert list(fast.state_dict()) == list(ref.state_dict())
    x = torch.randn(3, 5, ref.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    got, want = fast(x), ref(x)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(want)
    params = [fast.weight] + ([fast.bias] if fast.bias.numel() else [])
    ref_params = [ref.weight] + ([ref.bias] if ref.bias.numel() else [])
    for a, b in zip(_grads(got, [x, *params], probe), _grads(want, [x, *ref_params], probe)):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_linear_external_weights_use_the_e3nn_forward():
    lin = e3nn_fast.Linear("4x0e+2x1o", "3x0e+1x1o", internal_weights=False, shared_weights=False).double()
    x = torch.randn(6, lin.irreps_in.dim, dtype=torch.float64)
    w = torch.randn(6, lin.weight_numel, dtype=torch.float64)
    torch.testing.assert_close(lin(x, w), o3.Linear.forward(lin, x, w))


@pytest.mark.parametrize("irreps", ["32x0e+8x0o+16x1o+8x1e+8x2e+4x3o+2x4e", "8x1o+4x2e", "6x0e+2x0o"])
def test_gate_matches_e3nn(irreps):
    torch.manual_seed(1)
    fast = build_gate_activation(o3.Irreps(irreps))
    assert isinstance(fast, e3nn_fast.Gate) and fast._products is not None
    x = torch.randn(7, fast.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    got = fast(x)
    want = e3nn_nn.Gate.forward(fast, x)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(want)
    torch.testing.assert_close(_grads(got, [x], probe)[0], _grads(want, [x], probe)[0], rtol=1e-12, atol=1e-12)


def _element_linear_reference(irreps, x, weights):
    """x * scale per irrep channel, + shift per 0e channel, written channel by channel."""
    out, offset, k_scale, k_shift = [], 0, 0, sum(mul for mul, _ in irreps)
    for mul, ir in irreps:
        for _ in range(mul):
            block = x[:, offset:offset + ir.dim] * weights[:, k_scale:k_scale + 1]
            if str(ir) == "0e" and weights.shape[1] > k_shift:
                block = block + weights[:, k_shift:k_shift + 1]
                k_shift += 1
            out.append(block)
            offset += ir.dim
            k_scale += 1
    return torch.cat(out, dim=1)


@pytest.mark.parametrize("with_shifts", [True, False])
def test_element_linear_matches_channel_formula(with_shifts):
    torch.manual_seed(2)
    irreps = o3.Irreps("3x0e + 2x1o + 4x2e + 1x3o + 2x0e")
    layer = E3ElementLinear(irreps, dtype=torch.float64)
    x = torch.randn(11, irreps.dim, dtype=torch.float64, requires_grad=True)
    width = layer.weight_numel if with_shifts else layer.num_scales
    weights = torch.randn(11, width, dtype=torch.float64, requires_grad=True)
    got = layer(x, weights)
    want = _element_linear_reference(irreps, x, weights)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(want)
    for a, b in zip(_grads(got, [x, weights], probe), _grads(want, [x, weights], probe)):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    assert layer(x) is x


def test_element_linear_ignores_extra_trailing_weight_columns():
    torch.manual_seed(5)
    irreps = o3.Irreps("3x0e + 2x1o")
    layer = E3ElementLinear(irreps, dtype=torch.float64)
    x = torch.randn(4, irreps.dim, dtype=torch.float64)
    weights = torch.randn(4, layer.weight_numel, dtype=torch.float64)
    extra = torch.randn(4, 2, dtype=torch.float64)
    got = layer(x, torch.cat([weights, extra], dim=1))
    torch.testing.assert_close(got, layer(x, weights), rtol=0.0, atol=0.0)


def test_element_linear_too_few_weight_columns_raises():
    irreps = o3.Irreps("3x0e + 2x1o")
    layer = E3ElementLinear(irreps, dtype=torch.float64)
    x = torch.randn(4, irreps.dim, dtype=torch.float64)
    short = torch.randn(4, layer.num_scales - 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="expects"):
        layer(x, short)


def _norm_reference(norm, x):
    """Centre the scalar channels, divide by the RMS merged over degrees, affine, channel by channel."""
    irreps = norm.irreps
    blocks, offset = [], 0
    for mul, ir in irreps:
        blocks.append((mul, ir, x[:, offset:offset + mul * ir.dim].reshape(-1, mul, ir.dim)))
        offset += mul * ir.dim
    scalar = [(ir.l == 0 and (ir.p == 1 or norm.treat_0o_as_scalar)) for _, ir, _ in blocks]
    if norm.center_0e and any(scalar):
        values = torch.cat([b[2].reshape(x.shape[0], -1) for b, s in zip(blocks, scalar) if s], dim=1)
        mean = values.mean(dim=1)
        blocks = [(mul, ir, v - mean[:, None, None] if s else v) for (mul, ir, v), s in zip(blocks, scalar)]
    per_degree = {}
    for mul, ir, v in blocks:
        ms = v.square().sum(-1) / (ir.dim if norm.normalization == "component" else 1)
        per_degree.setdefault(ir.l, []).append(ms)
    if norm.std_balance_degrees:
        merged = torch.stack([torch.cat(ms, dim=1).mean(dim=1) for ms in per_degree.values()], dim=1).mean(dim=1)
    else:
        merged = torch.cat([m for ms in per_degree.values() for m in ms], dim=1).mean(dim=1)
    scale = torch.rsqrt(merged + norm.eps)
    out, group, k = [], 0, 0
    for (mul, ir, v), s in zip(blocks, scalar):
        w = norm.affine_weight[0, group:group + mul] if norm.affine else torch.ones(mul, dtype=x.dtype)
        block = (v * scale[:, None, None] * w[None, :, None]).reshape(x.shape[0], -1)
        if s and norm.affine and norm.affine_bias is not None:
            block = block + norm.affine_bias[0, k:k + mul]
            k += mul
        out.append(block)
        group += mul
    return torch.cat(out, dim=1)


@pytest.mark.parametrize("irreps, kwargs", [
    ("8x0e+8x1o+8x2e+4x3o+4x4e+2x5o+2x6e", {}),
    ("4x0e+2x0o+3x1o+3x1e+2x2e", dict(treat_0o_as_scalar=True)),
    ("3x1o+2x2e", {}),
    ("5x0e+4x1o", dict(std_balance_degrees=False, normalization="norm")),
    ("4x0e+2x1o", dict(affine=False)),
    ("6x0e+2x1o", dict(center_0e=False)),
])
def test_equivariant_norm_matches_channel_formula(irreps, kwargs):
    torch.manual_seed(3)
    norm = EquivariantMergedRMSNormFlat(irreps, dtype=torch.float64, **kwargs)
    with torch.no_grad():
        for p in norm.parameters():
            p.normal_()
    x = torch.randn(9, norm.dim, dtype=torch.float64, requires_grad=True)
    got = norm(x)
    want = _norm_reference(norm, x)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(want)
    params = list(norm.parameters())
    for a, b in zip(_grads(got, [x, *params], probe), _grads(want, [x, *params], probe)):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# EquivariantMergedRMSNormFlat precision: fp64 must not be silently downcast to fp32
# ---------------------------------------------------------------------------

_PRECISION_IRREPS = o3.Irreps("4x0e+4x1o+4x2e")


def _precision_norm(dtype):
    return EquivariantMergedRMSNormFlat(_PRECISION_IRREPS, eps=1e-12, dtype=dtype, device=torch.device("cpu"))


def test_norm_float64_retains_double_precision():
    torch.manual_seed(0)
    x64 = torch.randn(6, _PRECISION_IRREPS.dim, dtype=torch.float64)
    # Perturbation far below float32 resolution but well above float64's.
    delta = 1e-9
    y_base = _precision_norm(torch.float64)(x64)
    y_pert = _precision_norm(torch.float64)(x64 + delta)
    diff = (y_pert - y_base).abs().max().item()
    assert y_base.dtype == torch.float64
    # A float32 internal cast would round the 1e-9 perturbation away entirely
    # (relative eps ~1.2e-7 on O(1) values); float64 must propagate it.
    assert 1e-11 < diff < 1e-6


def test_norm_half_inputs_still_upcast_to_float32():
    x = torch.randn(4, _PRECISION_IRREPS.dim, dtype=torch.float16)
    out = _precision_norm(torch.float32)(x)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_grid_mat_cache_is_dtype_keyed():
    prior = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        to32, _ = _get_grid_mats(2, 2, "integral", (8, 8))
        torch.set_default_dtype(torch.float64)
        to64, _ = _get_grid_mats(2, 2, "integral", (8, 8))
    finally:
        torch.set_default_dtype(prior)
    assert to32.dtype == torch.float32
    assert to64.dtype == torch.float64


def test_row_permutation_gradients():
    torch.manual_seed(4)
    x = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    order = torch.randperm(9)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(9)
    torch.testing.assert_close(permute_rows(x, order, inverse), x[order], rtol=0, atol=0)
    assert torch.autograd.gradcheck(lambda t: permute_rows(t, order, inverse), (x,))
    assert torch.autograd.gradgradcheck(lambda t: permute_rows(t, order, inverse).square(), (x,))


# ---------------------------------------------------------------------------
# ScalarOnehotTP: matches e3nn uvu/FullyConnected tensor products, in fwd and grad
# ---------------------------------------------------------------------------

def _assert_forward_and_grad_close(tp_ref, tp_fast, x, y, fast_weight=None):
    x_ref, y_ref = x.detach().clone().requires_grad_(True), y.detach().clone().requires_grad_(True)
    x_fast, y_fast = x.detach().clone().requires_grad_(True), y.detach().clone().requires_grad_(True)

    out_ref, out_fast = tp_ref(x_ref, y_ref), tp_fast(x_fast, y_fast)
    torch.testing.assert_close(out_fast, out_ref, atol=1e-10, rtol=1e-10)

    loss_ref, loss_fast = out_ref.square().sum(), out_fast.square().sum()
    loss_ref.backward()
    loss_fast.backward()

    torch.testing.assert_close(x_fast.grad, x_ref.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(y_fast.grad, y_ref.grad, atol=1e-10, rtol=1e-10)
    weight = tp_fast.weight if fast_weight is None else fast_weight
    torch.testing.assert_close(weight.grad, tp_ref.weight.grad, atol=1e-10, rtol=1e-10)


def test_scalar_fast_matches_uvu_tensor_product_forward_and_grad():
    torch.manual_seed(20260424)
    dtype = torch.float64
    irreps = o3.Irreps("3x0e + 2x1o + 1x2e")
    onehot_irreps = o3.Irreps("7x0e")
    instructions = [(i, 0, i, "uvu", True) for i, _ in enumerate(irreps)]

    tp_ref = o3.TensorProduct(irreps, onehot_irreps, irreps, instructions).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(11, irreps.dim, dtype=dtype)
    y = torch.randn(11, onehot_irreps.dim, dtype=dtype)
    _assert_forward_and_grad_close(tp_ref, tp_fast, x, y)


@pytest.mark.parametrize("leading", [(7,), (0,), (3, 5)])
def test_scalar_fast_packed_uvu_matches_repeated_irreps(leading):
    """The packed-gain uvu path (UpdateNode/UpdateEdge) with repeated irrep types and
    two instructions writing the same output irrep."""
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
    _assert_forward_and_grad_close(tp_ref, tp_fast, x, y)


def test_scalar_fast_matches_fully_connected_scalar_tp_forward_and_grad():
    torch.manual_seed(20260424)
    dtype = torch.float64
    irreps_in, irreps_out = o3.Irreps("3x0e + 2x1o + 2x2e"), o3.Irreps("2x0e + 3x1o + 1x2e")
    onehot_irreps = o3.Irreps("5x0e")

    tp_ref = o3.FullyConnectedTensorProduct(irreps_in, onehot_irreps, irreps_out).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(13, irreps_in.dim, dtype=dtype)
    y = torch.randn(13, onehot_irreps.dim, dtype=dtype)
    _assert_forward_and_grad_close(tp_ref, tp_fast, x, y)


def test_scalar_fast_fully_connected_unsimplified_irreps():
    """Orbital-pair irreps list each irrep type several times, so one output irrep
    receives paths from several input irreps; an irrep type missing from the input
    yields a zero output block; two scalar blocks give two groups per irrep type."""
    torch.manual_seed(20260923)
    dtype = torch.float64
    irreps_in = o3.Irreps("3x0e + 2x1o + 2x0e + 1x2e + 3x1o + 1x0e + 2x2e")
    irreps_out = o3.Irreps("2x0e + 1x1o + 2x1e + 3x0e + 2x2e + 1x1o")
    onehot_irreps = o3.Irreps("4x0e + 3x0e")

    tp_ref = o3.FullyConnectedTensorProduct(irreps_in, onehot_irreps, irreps_out).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(9, irreps_in.dim, dtype=dtype)
    y = torch.randn(9, onehot_irreps.dim, dtype=dtype)
    _assert_forward_and_grad_close(tp_ref, tp_fast, x, y)


def test_scalar_fast_incomplete_uvw_paths_match_tensor_product():
    """uvw instructions that do not couple every input with every output irrep of
    their type take the per-instruction path."""
    torch.manual_seed(20260923)
    dtype = torch.float64
    irreps_in, irreps_out = o3.Irreps("2x0e + 3x1o + 1x0e"), o3.Irreps("2x0e + 1x1o + 3x0e")
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
    _assert_forward_and_grad_close(tp_ref, tp_fast, x, y)


def test_scalar_fast_on_an_e3nn_module_trains_its_weight():
    torch.manual_seed(20260923)
    dtype = torch.float64
    irreps_in, irreps_out = o3.Irreps("2x0e + 1x1o + 2x0e"), o3.Irreps("1x0e + 2x1o + 2x0e")
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
    _assert_forward_and_grad_close(tp_ref, _Apply(), x, y, fast_weight=tp_e3nn.weight)
    assert set(tp_e3nn.state_dict()) == keys


def test_scalar_fast_state_dict_holds_only_the_e3nn_weight():
    irreps_in, irreps_out = o3.Irreps("2x0e + 1x1o + 2x0e"), o3.Irreps("1x0e + 2x1o")
    tp_ref = o3.FullyConnectedTensorProduct(irreps_in, "3x0e", irreps_out)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref)

    assert list(tp_fast.state_dict()) == ["weight"]
    restored = ScalarOnehotTP.from_e3nn(o3.FullyConnectedTensorProduct(irreps_in, "3x0e", irreps_out))
    restored.load_state_dict({"weight": tp_ref.weight.detach()}, strict=True)
    torch.testing.assert_close(restored.weight, tp_ref.weight)


def test_e3nn_adapter_after_inference_warmup_and_under_torch_func():
    torch.manual_seed(20260924)
    tp = o3.FullyConnectedTensorProduct("2x0e + 1x1o + 1x0e", "3x0e", "1x0e + 2x1o + 2x0e").double()
    keys = tuple(tp.state_dict())
    x = torch.randn(4, tp.irreps_in1.dim, dtype=torch.float64)
    y = torch.randn(4, tp.irreps_in2.dim, dtype=torch.float64)

    # a first call inside a torch.func transform must not leave interpreter tensors in the cache
    out, tangent = torch.func.jvp(lambda xx: _scalar_onehot_tp_fast(tp, xx, y), (x,), (torch.ones_like(x),))
    torch.testing.assert_close(out, tp(x, y), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(tangent, tp(torch.ones_like(x), y), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(torch.vmap(lambda xx, yy: _scalar_onehot_tp_fast(tp, xx, yy))(x, y), tp(x, y),
                                atol=1e-10, rtol=1e-10)
    # a first call under inference_mode must build constants a later backward can save
    tp.__dict__.pop("_scalar_onehot_layout", None)
    with torch.inference_mode():
        _scalar_onehot_tp_fast(tp, x, y)
    x.requires_grad_(True)
    y.requires_grad_(True)
    got, ref = _scalar_onehot_tp_fast(tp, x, y), tp(x, y)
    for a, b in zip(torch.autograd.grad(got.sum(), (x, y, tp.weight)), torch.autograd.grad(ref.sum(), (x, y, tp.weight))):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    # the cached layout holds no parameter; the module's state_dict is unchanged
    assert not list(tp.__dict__["_scalar_onehot_layout"][1].parameters())
    assert tuple(tp.state_dict()) == keys


@pytest.mark.parametrize("rows", [0, 3])
def test_scalar_fast_empty_output(rows):
    fast = ScalarOnehotTP.from_e3nn(o3.FullyConnectedTensorProduct("2x0e", "3x0e", "").double())
    assert fast(torch.randn(rows, 2, dtype=torch.float64), torch.randn(rows, 3, dtype=torch.float64)).shape == (rows, 0)


# ---------------------------------------------------------------------------
# OEQTensorProduct / get_feasible_tp: instruction-plan shape and the scalar_direct backend
# ---------------------------------------------------------------------------

def test_oeq_get_feasible_tp_unit_paths_match_existing_instruction_shape():
    irreps_mid, instructions = get_feasible_tp(
        o3.Irreps("2x0e + 1x1o"), o3.Irreps("3x0e"), o3.Irreps("4x0e + 1x1o"),
        tp_mode="uvw", trainable=True, path_normalization="unit", sort_irreps=False,
    )
    assert irreps_mid == o3.Irreps("4x0e + 1x1o")
    assert instructions == [(0, 0, 0, "uvw", True, 1.0), (1, 0, 1, "uvw", True, 1.0)]


def test_oeq_get_feasible_tp_path_normalization_sorts_outputs():
    irreps_mid, instructions = get_feasible_tp(
        o3.Irreps("1x1o + 2x0e"), o3.Irreps("1x0e"), o3.Irreps("2x0e + 1x1o"),
        tp_mode="uvw", trainable=True, path_normalization="e3nn", sort_irreps=True,
    )
    assert irreps_mid == o3.Irreps("2x0e + 1x1o")
    assert len(instructions) == 2
    assert instructions[0][:5] == (0, 0, 1, "uvw", True)
    assert instructions[1][:5] == (1, 0, 0, "uvw", True)
    assert all(path_weight > 0 for *_, path_weight in instructions)


def test_scalar_side_direct_tp_matches_e3nn_reference():
    irreps_in1, irreps_in2 = o3.Irreps("2x0e + 1x1o"), o3.Irreps("3x0e")
    irreps_out = o3.Irreps("4x0e + 2x1o")
    irreps_mid, instructions = get_feasible_tp(
        irreps_in1, irreps_in2, irreps_out, tp_mode="uvw", trainable=True, path_normalization="unit", sort_irreps=False,
    )
    direct = OEQTensorProduct(irreps_in1, irreps_in2, irreps_out, internal_weights=False, backend="scalar_direct")
    reference = o3.TensorProduct(irreps_in1, irreps_in2, irreps_mid, instructions,
                                  internal_weights=False, shared_weights=True)

    torch.manual_seed(0)
    x = torch.randn(5, irreps_in1.dim)
    y = torch.randn(5, irreps_in2.dim)
    oeq_weight = torch.randn(direct.weight_numel)
    e3nn_weights, offset = [], 0
    for i_in1, i_in2, i_out, mode, _, _ in instructions:
        assert mode == "uvw"
        mul_1, mul_2, mul_out = irreps_in1[i_in1].mul, irreps_in2[i_in2].mul, irreps_mid[i_out].mul
        block_numel = mul_1 * mul_2 * mul_out
        e3nn_weights.append(oeq_weight[offset:offset + block_numel].reshape(mul_2, mul_1, mul_out).permute(1, 0, 2).reshape(-1))
        offset += block_numel
    e3nn_weight = torch.cat(e3nn_weights)

    assert direct.weight_numel == reference.weight_numel
    torch.testing.assert_close(direct(x, y, oeq_weight), reference(x, y, e3nn_weight), atol=1e-6, rtol=1e-6)


@opt_in_oeq
@requires_cuda
@requires_module("openequivariance")
def test_scalar_side_direct_tp_matches_oeq_when_available():
    # The import can succeed while the JIT build fails, hence the opt-in env on top of the gates.
    irreps_in1, irreps_in2 = o3.Irreps("2x0e + 1x1o"), o3.Irreps("3x0e")
    irreps_out = o3.Irreps("4x0e + 2x1o")
    direct = OEQTensorProduct(irreps_in1, irreps_in2, irreps_out, internal_weights=False, backend="scalar_direct").cuda()
    oeq_tp = OEQTensorProduct(irreps_in1, irreps_in2, irreps_out, internal_weights=False, backend="oeq").cuda()

    torch.manual_seed(1)
    x = torch.randn(7, irreps_in1.dim, device="cuda")
    y = torch.randn(7, irreps_in2.dim, device="cuda")
    weight = torch.randn(direct.weight_numel, device="cuda")

    assert direct.weight_numel == oeq_tp.weight_numel
    torch.testing.assert_close(direct(x, y, weight), oeq_tp(x, y, weight), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("irreps", ["3x0e + 2x1o + 1x2e", "2x1o + 1x2e"])
def test_e3_element_linear_scripts(irreps):
    """E3ElementLinear keeps its TorchScript compile mode, with and without 0e shifts."""
    layer = E3ElementLinear(o3.Irreps(irreps))
    scripted = torch.jit.script(layer)
    x = torch.randn(5, layer.irreps_in.dim)
    for width in (layer.num_scales, layer.weight_numel):
        w = torch.randn(5, width)
        torch.testing.assert_close(scripted(x, w), layer(x, w), rtol=0, atol=0)


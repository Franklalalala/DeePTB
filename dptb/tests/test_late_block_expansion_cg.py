"""Equivalence matrix for the fused ``LateBlockExpansionCGHead`` forward.

The fused path (``_forward_fused``, an opt-in performance forward selected via
``cg_head_impl="fused"``) collapses the legacy per-path Python loop
(``_forward_legacy``, the compatibility default) into ``G`` grouped batched einsums
plus a single ``index_add`` scatter. This is a numerical-association change only:
the batched einsums and the scatter reassociate the same floating-point sums in a
different order, so bitwise identity is not expected (fp addition is
non-associative) but the drift must stay at roundoff.

fp64 is bit-compatible up to ~4e-16 forward / ~6e-14 grad, so it is certified with a
tight absolute bound. fp32 drift is pure fp32 roundoff, which is *relative* (it
scales with the output magnitude), so fp32 parity is certified with rtol plus a
small atol floor -- a flat absolute atol on unit-scale inputs would silently
overclaim and fail under a wider scale sweep. A failure above these tolerances
signals a real semantic divergence, not precision -- do not relax a tolerance to
make such a failure pass.

Covers: forward and gradient parity fused-vs-legacy (both bases, both symmetrize
modes, a fp32 magnitude sweep); an independent SO(3) equivariance oracle (e3nn D
matrices, not the legacy loop); Hermitian symmetrize behaviour; structurally-empty
canvas cells staying bit-zero; old-checkpoint strict-load compatibility; dispatch
precedence (config opt-in / env rollback); and the certified-domain routing guard
that falls fused configs back to legacy under autocast, non-fp32/64 dtypes, or
``use_deterministic_algorithms(True)`` -- on CPU and on CUDA.
"""
from __future__ import annotations

import pytest
import torch
from e3nn import o3

from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.late_block_expansion_cg import (
    LateBlockExpansionCGHead,
    _FUSED_CG_HEAD_ENV,
    _LEGACY_CG_HEAD_ENV,
)
from dptb.tests._requires import requires_cuda

# A realistic ordinary_hidden (full l<=4, both parities), verbatim from
# dptb/tests/test_output_routes.py and the design probes.
ORDINARY_HIDDEN = "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e"

WATER_BASIS = {"H": "2s1p", "O": "3s2p1d"}
CRYSTAL_BASIS = {"C": "2s2p1d", "Si": "3s3p2d"}

# The exact learnable-tensor inventory the state_dict must keep.
STATE_DICT_KEYS = {
    "static_weights", "condition_down.weight", "condition_down.bias", "dynamic_up.weight", "dynamic_up.bias",
}

# Certified reassociation-only tolerances (headroom over measured values).
FWD_ATOL = {torch.float64: 1e-12}          # fp64 absolute (reassociation-only)
FWD_RTOL_FP32 = 5e-6                        # fp32 relative reassociation drift
FWD_ATOL_FLOOR_FP32 = 1e-6                  # floor for near-zero canvas cells
GRAD_ATOL_FP64 = 1e-10
GRAD_RTOL_FP32 = 5e-6
GRAD_ATOL_FLOOR_FP32 = 1e-6
SCALE_SWEEP_REL = 1e-5                      # fp32-roundoff ceiling across a 1e-2..1e2 sweep
EQUIVARIANCE_ATOL = 1e-10


def _full_basis(basis):
    return tuple(OrbitalMapper(basis=basis, method="e3tb").full_basis)


def _build_head(basis, *, symmetrize, dtype=torch.float64, rank=8, init=0.3, seed=0, irreps_in=ORDINARY_HIDDEN,
                randomize=True, cg_head_impl="legacy"):
    """Build a head and (by default) randomize every learnable tensor so the static
    AND dynamic path weights are non-trivial. ``cg_head_impl`` defaults to the
    module's own compatibility default; pass "fused" to opt into the performance path."""
    torch.manual_seed(seed)
    head = LateBlockExpansionCGHead(o3.Irreps(irreps_in), _full_basis(basis), symmetrize=symmetrize, rank=rank,
                                    init=init, dtype=dtype, cg_head_impl=cg_head_impl)
    if randomize:
        with torch.no_grad():
            head.dynamic_up.weight.normal_(0.0, 0.5)
            head.dynamic_up.bias.normal_(0.0, 0.5)
            head.condition_down.weight.normal_(0.0, 0.5)
            head.condition_down.bias.normal_(0.0, 0.5)
            head.static_weights.normal_(0.0, 0.5)
    return head


def _max_abs(a, b):
    return float((a - b).abs().max())


def _max_rel(fused, legacy):
    """Relative reassociation drift in max-norm: max|fused-legacy| / max|legacy|.

    Normalising by the output scale makes this invariant to a global rescaling of
    weights/inputs (unlike the flat absolute drift): the absolute drift grows with
    magnitude, the relative does not."""
    ref = float(legacy.abs().max())
    if ref == 0.0:
        return _max_abs(fused, legacy)
    return _max_abs(fused, legacy) / ref


@pytest.fixture
def fused_opt_in(monkeypatch):
    """Set the (permanently inert) ``DPTB_FUSED_CG_HEAD`` affirmation for a dispatch
    test: the fused path is reached only via the ``cg_head_impl="fused"``
    constructor/config opt-in, never via an environment variable, so this must never
    disturb dispatch either way."""
    monkeypatch.setenv(_FUSED_CG_HEAD_ENV, "1")
    return monkeypatch


# ---------------------------------------------------------------------------
# Forward and gradient parity, fused vs legacy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("symmetrize", [False, True])
@pytest.mark.parametrize("batch_shape", [(7,), (2, 3)])
def test_forward_parity_fused_matches_legacy(basis, dtype, symmetrize, batch_shape):
    """Node (symmetrize=True) and edge (symmetrize=False) shapes, both bases, fp64
    tight-absolute / fp32 relative (rtol + atol floor). batch_shape=(2,3) also
    exercises the multi-dim *batch reshape/scatter path."""
    head = _build_head(basis, symmetrize=symmetrize, dtype=dtype)
    x = torch.randn(*batch_shape, head.irreps_in.dim, dtype=dtype)

    legacy = head._forward_legacy(x)
    fused = head._forward_fused(x)

    assert fused.shape == legacy.shape == (*batch_shape, head.max_norb, head.max_norb)
    if dtype == torch.float64:
        drift = _max_abs(legacy, fused)
        assert drift <= FWD_ATOL[dtype], (
            f"fp64 forward drift {drift:.3e} exceeds the certified reassociation tolerance "
            f"{FWD_ATOL[dtype]:.0e} (symmetrize={symmetrize}) -- semantic divergence, not roundoff."
        )
    else:
        # fp32: certify the RELATIVE reassociation drift. A flat atol would overclaim --
        # the absolute drift scales with |output|.
        assert torch.allclose(fused, legacy, rtol=FWD_RTOL_FP32, atol=FWD_ATOL_FLOOR_FP32), (
            f"fp32 forward parity exceeds rtol={FWD_RTOL_FP32:.0e}/atol={FWD_ATOL_FLOOR_FP32:.0e} "
            f"(abs={_max_abs(legacy, fused):.3e}, rel={_max_rel(fused, legacy):.3e}, symmetrize={symmetrize})."
        )


def test_scale_sweep_relative_drift_stays_fp32_roundoff():
    """The fp32 fused-vs-legacy drift is RELATIVE roundoff, not a flat absolute bound.
    Rescaling weights and inputs by 1e-2 / 1 / 1e2 leaves the relative max-norm drift
    at fp32-roundoff scale at every scale, while the absolute drift grows with the
    output magnitude (so a flat atol=1e-5 would falsely fail at 1e2)."""
    base = _build_head(WATER_BASIS, symmetrize=False, dtype=torch.float32, seed=11)
    torch.manual_seed(11)
    base_x = torch.randn(6, base.irreps_in.dim, dtype=torch.float32)

    abs_by_scale = {}
    for scale in (1e-2, 1.0, 1e2):
        head = _build_head(WATER_BASIS, symmetrize=False, dtype=torch.float32, seed=11)
        with torch.no_grad():
            for param in head.parameters():
                param.mul_(scale)
        x = base_x * scale
        legacy = head._forward_legacy(x)
        fused = head._forward_fused(x)
        rel = _max_rel(fused, legacy)
        abs_by_scale[scale] = _max_abs(legacy, fused)
        assert rel <= SCALE_SWEEP_REL, (
            f"scale={scale:g}: relative drift {rel:.3e} exceeds the fp32-roundoff ceiling "
            f"{SCALE_SWEEP_REL:.0e} (abs={abs_by_scale[scale]:.3e})"
        )

    # The absolute drift does grow with scale: at 1e2 it exceeds a naive flat atol=1e-5,
    # while at 1e-2 it is far below it (relative stayed tiny at every scale above).
    assert abs_by_scale[1e2] > 1e-5 > abs_by_scale[1e-2]


@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("symmetrize", [False, True])
def test_gradient_parity_params_and_scalar_condition_input(basis, dtype, symmetrize):
    """Backprop out.pow(2).sum() through fused vs legacy; compare grads on all five
    learnable tensors AND on the input (the scalar-condition gradient flows
    features -> index_select -> condition_down -> dynamic_up -> mix)."""
    head = _build_head(basis, symmetrize=symmetrize, dtype=dtype)
    x = torch.randn(6, head.irreps_in.dim, dtype=dtype)

    def grads(fn):
        xg = x.clone().requires_grad_(True)
        head.zero_grad(set_to_none=True)
        fn(xg).pow(2).sum().backward()
        return {n: p.grad.clone() for n, p in head.named_parameters()}, xg.grad.clone()

    g_legacy, gx_legacy = grads(head._forward_legacy)
    g_fused, gx_fused = grads(head._forward_fused)

    assert set(g_legacy) == STATE_DICT_KEYS
    if dtype == torch.float64:
        for name in g_legacy:
            drift = _max_abs(g_legacy[name], g_fused[name])
            assert drift <= GRAD_ATOL_FP64, f"grad[{name}] drift {drift:.3e}"
        assert _max_abs(gx_legacy, gx_fused) <= GRAD_ATOL_FP64  # scalar-condition input
    else:
        for name in g_legacy:
            assert torch.allclose(g_fused[name], g_legacy[name], rtol=GRAD_RTOL_FP32, atol=GRAD_ATOL_FLOOR_FP32), (
                f"fp32 grad[{name}] parity exceeds rtol={GRAD_RTOL_FP32:.0e}/atol={GRAD_ATOL_FLOOR_FP32:.0e} "
                f"(abs={_max_abs(g_legacy[name], g_fused[name]):.3e})"
            )
        assert torch.allclose(gx_fused, gx_legacy, rtol=GRAD_RTOL_FP32, atol=GRAD_ATOL_FLOOR_FP32)


def test_fused_backward_gradcheck_small_instance():
    """Independent fp64 gradcheck of the fused backward (small basis/rank)."""
    torch.manual_seed(1)
    head = LateBlockExpansionCGHead(o3.Irreps("2x0e+2x1o+1x2e"), _full_basis({"H": "1s", "C": "1s1p"}),
                                    symmetrize=True, rank=2, init=0.3, dtype=torch.float64)
    with torch.no_grad():
        head.dynamic_up.weight.normal_(0.0, 0.5)
        head.static_weights.normal_(0.0, 0.5)
    x = torch.randn(2, head.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(head._forward_fused, (x,), eps=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# SO(3) equivariance (independent e3nn oracle), Hermiticity, zero regions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
@pytest.mark.parametrize("symmetrize", [False, True])
def test_fused_head_is_so3_wigner_d_equivariant(basis, symmetrize):
    """Rotate the hidden features by D_in and the AO canvas by the shell-block D_ao;
    the fused head must commute: fused(f @ D_in^T) == D_ao fused(f) D_ao^T. The
    oracle is e3nn's Wigner-D of the input irreps and of the AO shells, from the
    SAME random proper rotation -- independent of the legacy loop, so it catches an
    equivariance error the two implementations could share."""
    head = _build_head(basis, symmetrize=symmetrize, dtype=torch.float64, seed=3)

    # e3nn's D_from_matrix routes angle intermediates through the default dtype, so
    # fp64 covariance needs a fp64 default (else it caps out near 1e-7).
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(3)
        rotation = o3.rand_matrix(dtype=torch.float64)  # proper SO(3)
        d_in = head.irreps_in.D_from_matrix(rotation)
        d_ao = head.ao_irreps.D_from_matrix(rotation)

        features = torch.randn(4, head.irreps_in.dim, dtype=torch.float64)
        rotated_input = head._forward_fused(features @ d_in.transpose(-1, -2))
        rotated_output = torch.einsum("ij,njk,lk->nil", d_ao, head._forward_fused(features), d_ao)
        drift = _max_abs(rotated_input, rotated_output)
        assert drift <= EQUIVARIANCE_ATOL, f"equivariance drift {drift:.3e}"
    finally:
        torch.set_default_dtype(previous_default)


@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
def test_symmetrize_makes_output_hermitian_and_edge_head_is_directed(basis):
    node = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=4)
    edge = _build_head(basis, symmetrize=False, dtype=torch.float64, seed=4)
    x = torch.randn(5, node.irreps_in.dim, dtype=torch.float64)

    node_out = node._forward_fused(x)
    assert _max_abs(node_out, node_out.transpose(-1, -2)) <= 1e-12  # Hermitian

    edge_out = edge._forward_fused(x)
    # The directed (edge) head is NOT symmetric in general: guards against a
    # symmetrize leaking into the edge path.
    assert _max_abs(edge_out, edge_out.transpose(-1, -2)) > 1e-6


def test_symmetrize_is_exactly_half_pre_symmetrize_sum():
    """The node output equals 0.5*(C + C^T) of the fused edge (pre-symmetrize) canvas
    built from identical params -- symmetrize is applied on the same canvas."""
    basis = WATER_BASIS
    node = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=5)
    edge = _build_head(basis, symmetrize=False, dtype=torch.float64, seed=5)
    assert torch.equal(node.static_weights, edge.static_weights)  # same seed => identical params
    x = torch.randn(3, node.irreps_in.dim, dtype=torch.float64)
    canvas = edge._forward_fused(x)
    expected = 0.5 * (canvas + canvas.transpose(-1, -2))
    assert _max_abs(node._forward_fused(x), expected) <= 1e-12


def test_canvas_zero_region_stays_bit_zero_and_matches_legacy():
    """A restricted hidden (0e+1o only) leaves some shell-pair blocks with no
    contributing path (e.g. s-d); those canvas cells must be EXACTLY zero in the
    fused output and coincide with the legacy zero mask."""
    head = _build_head(WATER_BASIS, symmetrize=False, dtype=torch.float64, seed=6, irreps_in="4x0e+4x1o")
    n = head.max_norb
    touched = torch.zeros(n * n, dtype=torch.bool)
    touched[head._fused_scatter_index] = True
    untouched = ~touched
    assert int(untouched.sum()) > 0, "expected structurally-empty blocks for 0e+1o"

    x = torch.randn(5, head.irreps_in.dim, dtype=torch.float64)
    fused = head._forward_fused(x).reshape(5, n * n)
    legacy = head._forward_legacy(x).reshape(5, n * n)

    assert torch.count_nonzero(fused[:, untouched]) == 0  # bit-zero in the fused canvas ...
    assert torch.count_nonzero(legacy[:, untouched]) == 0  # ... and the legacy loop agrees
    assert torch.count_nonzero(fused[:, touched]) > 0  # touched cells are actually populated


# ---------------------------------------------------------------------------
# state_dict round-trip / checkpoint compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
def test_state_dict_is_exactly_five_learnable_tensors(basis):
    head = _build_head(basis, symmetrize=True, dtype=torch.float64)
    sd = head.state_dict()
    assert set(sd.keys()) == STATE_DICT_KEYS
    # No non-persistent buffer (per-path Wigners, fused stacks, scalar indices) may
    # leak into the persistent state.
    for key in sd:
        assert "_path_coefficient" not in key
        assert "_fused_" not in key
        assert "_scalar_indices" not in key
    npw = head.num_path_weights
    assert sd["static_weights"].shape == (npw,)
    assert sd["dynamic_up.weight"].shape == (npw, head.rank)
    assert sd["dynamic_up.bias"].shape == (npw,)
    assert sd["condition_down.weight"].shape == (head.rank, len(head._scalar_indices))
    assert sd["condition_down.bias"].shape == (head.rank,)


def test_old_checkpoint_loads_strict_true_and_reproduces_forward():
    """The state_dict is byte-identical to the legacy module, so an OLD checkpoint
    (exactly the 5 learnable tensors, no fused/Wigner buffers) loads strict=True and
    reproduces the forward."""
    basis = CRYSTAL_BASIS
    source = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=7)
    old_ckpt = {k: v.detach().clone() for k, v in source.state_dict().items()}
    assert set(old_ckpt) == STATE_DICT_KEYS

    fresh = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=999)
    missing_unexpected = fresh.load_state_dict(old_ckpt, strict=True)
    assert missing_unexpected.missing_keys == []
    assert missing_unexpected.unexpected_keys == []

    x = torch.randn(4, source.irreps_in.dim, dtype=torch.float64)
    assert torch.equal(fresh._forward_fused(x), source._forward_fused(x))  # bitwise-identical
    # And the fused module round-trips into a legacy-semantics module strict=True.
    other = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=1234)
    other.load_state_dict(fresh.state_dict(), strict=True)
    assert torch.equal(other._forward_fused(x), source._forward_fused(x))


# ---------------------------------------------------------------------------
# Dispatch precedence: config opt-in, env rollback, inert affirmation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("cg_head_impl", "fused_env", "legacy_env", "expect_fused"),
    (
        ("legacy", None, None, False),   # default: no config, no env -> legacy
        ("legacy", "1", None, False),    # DPTB_FUSED_CG_HEAD is inert without the config opt-in
        ("fused", None, None, True),     # cg_head_impl="fused" alone selects the fused path
        ("fused", None, "1", False),     # DPTB_LEGACY_CG_HEAD=1 overrides a fused config
        ("fused", "1", "1", False),      # ... even with the inert affirmation also set
        ("fused", "1", "0", True),       # only the literal "1" activates the override
        ("fused", "1", "true", True),
        ("fused", "1", "yes", True),
        ("fused", "1", "", True),
    ),
    ids=["default_legacy", "inert_fused_affirmation", "config_fused", "legacy_env_overrides_fused",
         "legacy_env_and_affirmation", "legacy_env_non_one_0", "legacy_env_non_one_true",
         "legacy_env_non_one_yes", "legacy_env_non_one_empty"],
)
def test_dispatch_precedence(cg_head_impl, fused_env, legacy_env, expect_fused, monkeypatch):
    monkeypatch.delenv(_FUSED_CG_HEAD_ENV, raising=False)
    monkeypatch.delenv(_LEGACY_CG_HEAD_ENV, raising=False)
    if fused_env is not None:
        monkeypatch.setenv(_FUSED_CG_HEAD_ENV, fused_env)
    if legacy_env is not None:
        monkeypatch.setenv(_LEGACY_CG_HEAD_ENV, legacy_env)

    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float64, seed=21, cg_head_impl=cg_head_impl)
    assert head.cg_head_impl == cg_head_impl
    x = torch.randn(5, head.irreps_in.dim, dtype=torch.float64)
    want = head._forward_fused(x) if expect_fused else head._forward_legacy(x)
    assert torch.equal(head(x), want)


def test_invalid_cg_head_impl_rejected():
    """An unrecognized cg_head_impl value fails fast at construction, not silently
    at forward-dispatch time."""
    with pytest.raises(ValueError):
        _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float64, cg_head_impl="turbo")


# ---------------------------------------------------------------------------
# Certified-domain routing guard: outside eager fp32/fp64, forward() falls back to
# the legacy loop regardless of cg_head_impl (precedence: domain guard > config).
# Every head below is explicitly configured with cg_head_impl="fused", so a routed
# result equal to _forward_legacy can only mean the guard fired (fused would differ
# by reassociation roundoff) -- these are routing proofs, not numeric-parity claims.
# ---------------------------------------------------------------------------

def test_cpu_autocast_bf16_routes_to_legacy(fused_opt_in):
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=15, cg_head_impl="fused")
    x = torch.randn(4, head.irreps_in.dim, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dispatched = head(x)
        reference = head._forward_legacy(x)
    assert torch.equal(dispatched, reference)


def test_deterministic_mode_routes_to_legacy_cpu(fused_opt_in):
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float64, seed=14, cg_head_impl="fused")
    x = torch.randn(5, head.irreps_in.dim, dtype=torch.float64)
    torch.use_deterministic_algorithms(True)
    try:
        dispatched = head(x)
        assert torch.equal(dispatched, head._forward_legacy(x))
    finally:
        torch.use_deterministic_algorithms(False)


def test_half_input_routes_to_legacy(fused_opt_in):
    """features.half() is outside the certified fp32/fp64 domain, so forward() routes
    to the legacy loop even when cg_head_impl="fused".

    (a) With an fp32-param head, the mixed Half/Float Linear makes the legacy loop
        raise -- forward() (routed identically) must raise the same error TYPE.
    (b) With a genuine float16 head the legacy loop succeeds, so forward() must
        route to it and be bit-identical -- a positive proof of the dtype guard."""
    head32 = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=13, cg_head_impl="fused")
    x = torch.randn(4, head32.irreps_in.dim, dtype=torch.float32)
    xh = x.half()

    legacy_exc = None
    try:
        head32._forward_legacy(xh)
    except RuntimeError as exc:
        legacy_exc = exc
    assert legacy_exc is not None, "expected legacy to raise on Half input vs fp32 params"
    with pytest.raises(RuntimeError) as forward_exc:
        head32(xh)
    assert type(forward_exc.value) is type(legacy_exc)  # routed identically into the legacy loop

    head16 = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float16, seed=13, cg_head_impl="fused")
    assert torch.equal(head16(xh), head16._forward_legacy(xh))


@requires_cuda
def test_cuda_fp32_forward_parity_fused_vs_legacy():
    """fp32 fused-vs-legacy parity on CUDA, certified with the same relative
    tolerance (rtol + atol floor) as the CPU fp32 case."""
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=17).cuda()
    x = torch.randn(6, head.irreps_in.dim, dtype=torch.float32, device="cuda")
    legacy = head._forward_legacy(x)
    fused = head._forward_fused(x)
    assert torch.allclose(fused, legacy, rtol=FWD_RTOL_FP32, atol=FWD_ATOL_FLOOR_FP32), (
        f"cuda fp32 parity exceeds rtol={FWD_RTOL_FP32:.0e}/atol={FWD_ATOL_FLOOR_FP32:.0e} "
        f"(abs={_max_abs(legacy, fused):.3e}, rel={_max_rel(fused, legacy):.3e})"
    )


@requires_cuda
def test_cuda_deterministic_mode_routes_to_legacy(fused_opt_in):
    """Both paths share cuBLAS matmuls, so CUBLAS_WORKSPACE_CONFIG (set in
    conftest.py before CUDA starts) is what lets this assertion run at all; it is the
    routing -- proven by torch.equal -- that this certifies."""
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=18, cg_head_impl="fused").cuda()
    x = torch.randn(5, head.irreps_in.dim, dtype=torch.float32, device="cuda")
    torch.use_deterministic_algorithms(True)
    try:
        dispatched = head(x)
        assert torch.equal(dispatched, head._forward_legacy(x))
    finally:
        torch.use_deterministic_algorithms(False)


@requires_cuda
def test_cuda_autocast_fp16_routes_to_legacy(fused_opt_in):
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=19, cg_head_impl="fused").cuda()
    x = torch.randn(4, head.irreps_in.dim, dtype=torch.float32, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        dispatched = head(x)
        reference = head._forward_legacy(x)
    assert torch.equal(dispatched, reference)

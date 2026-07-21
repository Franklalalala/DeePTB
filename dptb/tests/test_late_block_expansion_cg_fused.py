"""Equivalence matrix for the fused ``LateBlockExpansionCGHead`` (design A.5).

The fused path (``_forward_fused``, the PERFORMANCE-DEFAULT forward inside the
certified eager fp32/fp64 domain; ``_forward_legacy`` is the rollback/oracle, and
the inert ``DPTB_FUSED_CG_HEAD`` name is NOT read -- fused is already the default)
collapses the legacy per-path Python loop into ``G``
grouped batched einsums plus a single ``index_add`` scatter.  It is a
numerical-ASSOCIATION change only: the batched einsums and the scatter reassociate
the same floating-point sums in a different order than the reference
``_forward_legacy`` loop.  fp is non-associative, so bitwise identity is NOT expected
-- absent cancellation the drift is pure roundoff (these parity tests exercise that
non-cancellation regime directly via ``_forward_fused`` vs ``_forward_legacy``).

The tolerances below CERTIFY reassociation-only drift (measured fp64 <= ~4e-16 fwd /
~6e-14 grad, fp32 <= ~2e-7 fwd / ~2e-5 grad; design A.4).  **A failure above these
tolerances signals a real semantic divergence, not precision** -- do not "relax" a
tolerance to make such a failure pass.  The fp32 drift is pure fp32 roundoff, which
is RELATIVE (it scales with the output magnitude), so fp32 parity is certified with
rtol + a small atol floor -- a flat absolute atol on unit-scale inputs silently
overclaims and fails under a 1e-2..1e2 scale sweep (design finding a).

Cases (design A.5 / A.8):
  1. forward parity fused-vs-legacy: fp64 tight-absolute (1e-12) / fp32 relative
     (rtol 5e-6), water AND crystal bases, both symmetrize modes, node & edge shapes,
     plus a weights/inputs scale sweep (1e-2/1/1e2) proving the drift stays relative;
  2. gradient parity on the 5 params AND the scalar-condition input (fp64 absolute /
     fp32 relative), plus a fp64 ``gradcheck`` of the fused backward on a small instance;
  3. SO(3) Wigner-D equivariance of the FUSED head via an INDEPENDENT dual-path
     oracle (e3nn D matrices), not the legacy loop;
  4. Hermiticity / symmetrize behaviour;
  5. padding / canvas zero-region invariance (untouched blocks stay bit-zero);
  6. state_dict round-trip: exact 5-tensor inventory + old->new strict=True load;
  7. env-var dispatch: default (no env) => fused (certified domain);
     ``DPTB_LEGACY_CG_HEAD=1`` overrides to legacy; ``DPTB_FUSED_CG_HEAD`` is inert;
  8. group-partition invariant + intra-group Wigner bit-identity;
  9. certified-domain routing guard (design A.8): forward() falls back to the legacy
     loop under autocast (CPU bf16 / CUDA fp16), non-fp32/64 input dtype, or
     ``use_deterministic_algorithms(True)`` -- on CPU AND on the local CUDA GPU.
"""
from __future__ import annotations

import os

# Enable deterministic cuBLAS BEFORE torch initializes its CUDA BLAS handle, so the
# CUDA determinism-routing test can execute under use_deterministic_algorithms(True)
# without tripping the cuBLAS nondeterminism guard.  BOTH the fused and the legacy
# paths use cuBLAS matmuls (condition_down / dynamic_up Linear layers), so this is
# required merely to *run* the routing assertion -- it is the routing, not cuBLAS,
# that this file certifies.  ``setdefault`` respects any value already in the env.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch
from e3nn import o3

from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.late_block_expansion_cg import (
    LateBlockExpansionCGHead,
    _FUSED_CG_HEAD_ENV,
    _LEGACY_CG_HEAD_ENV,
)


# A realistic ordinary_hidden (full l<=4, both parities), verbatim from
# dptb/tests/test_output_head_route_matrix.py and the design probes.
ORDINARY_HIDDEN = "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e"

WATER_BASIS = {"H": "2s1p", "O": "3s2p1d"}
CRYSTAL_BASIS = {"C": "2s2p1d", "Si": "3s3p2d"}

# The exact learnable-tensor inventory the state_dict must keep (design "Audit").
STATE_DICT_KEYS = {
    "static_weights",
    "condition_down.weight",
    "condition_down.bias",
    "dynamic_up.weight",
    "dynamic_up.bias",
}

# Certified reassociation-only tolerances (design A.4, headroom over measured).
#
# fp64 is bit-compatible up to ~4e-16 fwd / ~6e-14 grad, so a tight ABSOLUTE bound is
# meaningful.  fp32 drift is pure fp32 roundoff, which is RELATIVE -- it scales with
# the output magnitude -- so fp32 parity is certified with rtol + a small atol floor,
# not a flat atol (finding a; see test_scale_sweep_relative_drift_stays_fp32_roundoff).
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


def _build_head(
    basis,
    *,
    symmetrize,
    dtype=torch.float64,
    rank=8,
    init=0.3,
    seed=0,
    irreps_in=ORDINARY_HIDDEN,
    randomize=True,
):
    """Build a head and (by default) randomize every learnable tensor so the
    static AND dynamic path weights are non-trivial (design ``probe_equiv``)."""
    torch.manual_seed(seed)
    head = LateBlockExpansionCGHead(
        o3.Irreps(irreps_in),
        _full_basis(basis),
        symmetrize=symmetrize,
        rank=rank,
        init=init,
        dtype=dtype,
    )
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
    """Relative reassociation drift in max-norm: ``max|fused-legacy| / max|legacy|``.

    Normalising by the output scale makes this invariant to a global rescaling of
    weights/inputs (unlike the flat absolute drift), which is the whole point of the
    fp32 scale sweep -- the absolute drift grows with magnitude, the relative does not.
    """
    ref = float(legacy.abs().max())
    if ref == 0.0:
        return _max_abs(fused, legacy)
    return _max_abs(fused, legacy) / ref


@pytest.fixture
def fused_opt_in(monkeypatch):
    """Set the (now inert) ``DPTB_FUSED_CG_HEAD`` affirmation for a dispatch test.

    Fused is the PERFORMANCE-DEFAULT forward in the certified eager fp32/fp64 domain,
    so a routing test does not need any env to make fused the would-run path.  This
    fixture sets ``DPTB_FUSED_CG_HEAD=1`` -- a reserved name ``forward`` does NOT read
    (see the module comment) -- to assert it does NOT disturb the fused default; a
    test that then toggles ``DPTB_LEGACY_CG_HEAD`` or a certified-domain guard proves
    the override/guard actually fired.  Returns ``monkeypatch`` for that toggling.
    """
    monkeypatch.setenv(_FUSED_CG_HEAD_ENV, "1")
    return monkeypatch


# ===========================================================================
# A.5 #1 -- forward parity fused vs legacy
# ===========================================================================
@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("symmetrize", [False, True])
@pytest.mark.parametrize("batch_shape", [(7,), (2, 3)])
def test_forward_parity_fused_matches_legacy(basis, dtype, symmetrize, batch_shape):
    """Node (symmetrize=True) AND edge (symmetrize=False) invocation shapes, both
    bases, fp64 tight-absolute / fp32 relative (rtol + atol floor).
    ``batch_shape=(2,3)`` also exercises the multi-dim ``*batch`` reshape/scatter
    path."""
    head = _build_head(basis, symmetrize=symmetrize, dtype=dtype)
    x = torch.randn(*batch_shape, head.irreps_in.dim, dtype=dtype)

    legacy = head._forward_legacy(x)
    fused = head._forward_fused(x)

    assert fused.shape == legacy.shape == (*batch_shape, head.max_norb, head.max_norb)
    if dtype == torch.float64:
        drift = _max_abs(legacy, fused)
        assert drift <= FWD_ATOL[dtype], (
            f"fp64 forward drift {drift:.3e} exceeds the certified reassociation "
            f"tolerance {FWD_ATOL[dtype]:.0e} (symmetrize={symmetrize}) -- this is a "
            f"semantic divergence, not roundoff."
        )
    else:
        # fp32: certify the RELATIVE reassociation drift.  A flat atol would overclaim
        # -- the absolute drift scales with |output| (finding a).
        assert torch.allclose(
            fused, legacy, rtol=FWD_RTOL_FP32, atol=FWD_ATOL_FLOOR_FP32
        ), (
            f"fp32 forward parity exceeds rtol={FWD_RTOL_FP32:.0e}/"
            f"atol={FWD_ATOL_FLOOR_FP32:.0e} (abs={_max_abs(legacy, fused):.3e}, "
            f"rel={_max_rel(fused, legacy):.3e}, symmetrize={symmetrize}) -- semantic "
            f"divergence, not roundoff."
        )


def test_forward_parity_covers_G_equals_19_invariant():
    """Both documented bases collapse to the load-bearing G=19 groups (design A.1),
    while the raw path count differs (56 water / 122 crystal)."""
    water = _build_head(WATER_BASIS, symmetrize=True)
    crystal = _build_head(CRYSTAL_BASIS, symmetrize=True)
    assert water._fused_num_groups == 19
    assert crystal._fused_num_groups == 19
    assert len(water._paths) == 56
    assert len(crystal._paths) == 122


def test_scale_sweep_relative_drift_stays_fp32_roundoff():
    """Finding (a): the fp32 fused-vs-legacy drift is RELATIVE roundoff, not a flat
    absolute bound.  Rescaling weights AND inputs by 1e-2 / 1 / 1e2 leaves the
    relative max-norm drift at fp32-roundoff scale at every scale, while the ABSOLUTE
    drift grows with the output magnitude (so the old flat atol=1e-5 would falsely
    fail at 1e2).  This is precisely why fp32 parity is certified with rtol, not atol."""
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
            f"scale={scale:g}: relative drift {rel:.3e} exceeds the fp32-roundoff "
            f"ceiling {SCALE_SWEEP_REL:.0e} (abs={abs_by_scale[scale]:.3e})"
        )

    # The ABSOLUTE drift does grow with scale: at 1e2 it blows past the naive flat
    # atol=1e-5 the unit-scale tests used, while at 1e-2 it is far below it -- exactly
    # the overclaim finding (a) flagged.  (Relative stayed tiny at every scale above.)
    assert abs_by_scale[1e2] > 1e-5 > abs_by_scale[1e-2]


# ===========================================================================
# A.5 #2 -- gradient parity (params + scalar-condition input) + gradcheck
# ===========================================================================
@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("symmetrize", [False, True])
def test_gradient_parity_params_and_scalar_condition_input(basis, dtype, symmetrize):
    """Backprop ``out.pow(2).sum()`` through fused vs legacy; compare grads on all
    five learnable tensors AND on the input (the scalar-condition gradient flows
    ``features -> index_select -> condition_down -> dynamic_up -> mix``).  fp64
    tight-absolute; fp32 relative (rtol + atol floor), same reasoning as the forward
    parity."""
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
            assert torch.allclose(
                g_fused[name], g_legacy[name],
                rtol=GRAD_RTOL_FP32, atol=GRAD_ATOL_FLOOR_FP32,
            ), (
                f"fp32 grad[{name}] parity exceeds rtol={GRAD_RTOL_FP32:.0e}/"
                f"atol={GRAD_ATOL_FLOOR_FP32:.0e} "
                f"(abs={_max_abs(g_legacy[name], g_fused[name]):.3e})"
            )
        assert torch.allclose(  # scalar-condition input
            gx_fused, gx_legacy, rtol=GRAD_RTOL_FP32, atol=GRAD_ATOL_FLOOR_FP32
        )


def test_fused_backward_gradcheck_small_instance():
    """Independent fp64 ``gradcheck`` of the fused backward (small basis/rank)."""
    torch.manual_seed(1)
    head = LateBlockExpansionCGHead(
        o3.Irreps("2x0e+2x1o+1x2e"),
        _full_basis({"H": "1s", "C": "1s1p"}),
        symmetrize=True,
        rank=2,
        init=0.3,
        dtype=torch.float64,
    )
    with torch.no_grad():
        head.dynamic_up.weight.normal_(0.0, 0.5)
        head.static_weights.normal_(0.0, 0.5)
    x = torch.randn(2, head.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(head._forward_fused, (x,), eps=1e-6, atol=1e-6)


# ===========================================================================
# A.5 #3 -- SO(3) Wigner-D equivariance (INDEPENDENT dual-path oracle)
# ===========================================================================
@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
@pytest.mark.parametrize("symmetrize", [False, True])
def test_fused_head_is_so3_wigner_d_equivariant(basis, symmetrize):
    """Rotate the hidden features by ``D_in`` and the AO canvas by the shell-block
    ``D_ao``; the fused head must commute:  ``fused(f @ D_in^T) == D_ao fused(f) D_ao^T``.

    The oracle is e3nn's Wigner-D of the input irreps and of the AO shells, both
    from the SAME random proper rotation -- independent of the legacy loop, so it
    catches an equivariance error the two implementations could share.  (For the
    node head the ``0.5(H+H^T)`` symmetrize commutes with ``H -> D H D^T``.)
    """
    head = _build_head(basis, symmetrize=symmetrize, dtype=torch.float64, seed=3)

    # e3nn's D_from_matrix routes angle intermediates through the default dtype,
    # so fp64 covariance needs a fp64 default (else it caps out near 1e-7).
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(3)
        rotation = o3.rand_matrix(dtype=torch.float64)  # proper SO(3)
        d_in = head.irreps_in.D_from_matrix(rotation)   # [dim_in, dim_in]
        d_ao = head.ao_irreps.D_from_matrix(rotation)   # [max_norb, max_norb]

        features = torch.randn(4, head.irreps_in.dim, dtype=torch.float64)
        rotated_input = head._forward_fused(features @ d_in.transpose(-1, -2))
        rotated_output = torch.einsum(
            "ij,njk,lk->nil", d_ao, head._forward_fused(features), d_ao
        )
        drift = _max_abs(rotated_input, rotated_output)
        assert drift <= EQUIVARIANCE_ATOL, f"equivariance drift {drift:.3e}"
    finally:
        torch.set_default_dtype(previous_default)


# ===========================================================================
# A.5 #4 -- Hermiticity / symmetrize behaviour
# ===========================================================================
@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
def test_symmetrize_makes_output_hermitian_and_edge_head_is_directed(basis):
    node = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=4)
    edge = _build_head(basis, symmetrize=False, dtype=torch.float64, seed=4)
    x = torch.randn(5, node.irreps_in.dim, dtype=torch.float64)

    node_out = node._forward_fused(x)
    assert _max_abs(node_out, node_out.transpose(-1, -2)) <= 1e-12  # Hermitian

    edge_out = edge._forward_fused(x)
    # The directed (edge) head is NOT symmetric in general -- guards against a
    # symmetrize leaking into the edge path.
    assert _max_abs(edge_out, edge_out.transpose(-1, -2)) > 1e-6


def test_symmetrize_is_exactly_half_pre_symmetrize_sum():
    """The node output equals ``0.5*(C + C^T)`` of the fused edge (pre-symmetrize)
    canvas built from identical params -- symmetrize is applied on the same canvas."""
    basis = WATER_BASIS
    node = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=5)
    edge = _build_head(basis, symmetrize=False, dtype=torch.float64, seed=5)
    # Same seed => identical params; edge output IS the pre-symmetrize canvas.
    assert torch.equal(node.static_weights, edge.static_weights)
    x = torch.randn(3, node.irreps_in.dim, dtype=torch.float64)
    canvas = edge._forward_fused(x)
    expected = 0.5 * (canvas + canvas.transpose(-1, -2))
    assert _max_abs(node._forward_fused(x), expected) <= 1e-12


# ===========================================================================
# A.5 #5 -- padding / canvas zero-region invariance
# ===========================================================================
def test_canvas_zero_region_stays_bit_zero_and_matches_legacy():
    """A restricted hidden (``0e+1o`` only) leaves some shell-pair blocks with no
    contributing path (e.g. s-d); those canvas cells must be EXACTLY zero in the
    fused output and coincide with the legacy zero mask.  ``_fused_scatter_index``
    must never touch them."""
    head = _build_head(
        WATER_BASIS,
        symmetrize=False,
        dtype=torch.float64,
        seed=6,
        irreps_in="4x0e+4x1o",
    )
    n = head.max_norb
    touched = torch.zeros(n * n, dtype=torch.bool)
    touched[head._fused_scatter_index] = True
    untouched = ~touched
    assert int(untouched.sum()) > 0, "expected structurally-empty blocks for 0e+1o"

    x = torch.randn(5, head.irreps_in.dim, dtype=torch.float64)
    fused = head._forward_fused(x).reshape(5, n * n)
    legacy = head._forward_legacy(x).reshape(5, n * n)

    # untouched cells are exactly bit-zero in the fused canvas ...
    assert torch.count_nonzero(fused[:, untouched]) == 0
    # ... and the legacy loop zeroes exactly the same cells.
    assert torch.count_nonzero(legacy[:, untouched]) == 0
    # touched cells are actually populated (non-vacuous).
    assert torch.count_nonzero(fused[:, touched]) > 0


# ===========================================================================
# A.5 #6 -- state_dict round-trip / checkpoint compatibility
# ===========================================================================
@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
def test_state_dict_is_exactly_five_learnable_tensors(basis):
    head = _build_head(basis, symmetrize=True, dtype=torch.float64)
    sd = head.state_dict()
    assert set(sd.keys()) == STATE_DICT_KEYS
    # No non-persistent buffer (per-path Wigners, fused stacks, scalar indices)
    # may leak into the persistent state (design A.3 / R5).
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
    (exactly the 5 learnable tensors, no fused/Wigner buffers) loads ``strict=True``
    and reproduces the forward.  Since the fused module's own ``state_dict()`` is
    exactly those 5 keys, saving from it and loading strict=True is that test."""
    basis = CRYSTAL_BASIS
    source = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=7)
    # An "old checkpoint": ONLY the five learnable tensors, cloned/detached.
    old_ckpt = {k: v.detach().clone() for k, v in source.state_dict().items()}
    assert set(old_ckpt) == STATE_DICT_KEYS

    fresh = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=999)
    missing_unexpected = fresh.load_state_dict(old_ckpt, strict=True)
    assert missing_unexpected.missing_keys == []
    assert missing_unexpected.unexpected_keys == []

    x = torch.randn(4, source.irreps_in.dim, dtype=torch.float64)
    # Identical params after strict load => bitwise-identical fused forward.
    assert torch.equal(fresh._forward_fused(x), source._forward_fused(x))
    # And the fused module round-trips into a legacy-semantics module strict=True.
    other = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=1234)
    other.load_state_dict(fresh.state_dict(), strict=True)
    assert torch.equal(other._forward_fused(x), source._forward_fused(x))


# ===========================================================================
# A.5 #7 -- env-var rollback
# ===========================================================================
@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
def test_env_var_rollback_selects_legacy_path(basis, monkeypatch):
    """``DPTB_LEGACY_CG_HEAD=1`` overrides the fused default back to the legacy loop;
    removing the legacy override leaves the fused default active (the inert
    ``DPTB_FUSED_CG_HEAD`` affirmation does not change either outcome)."""
    head = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=8)
    x = torch.randn(5, head.irreps_in.dim, dtype=torch.float64)

    # Opt in to the fused path, then prove the legacy override still wins over it.
    monkeypatch.setenv(_FUSED_CG_HEAD_ENV, "1")
    monkeypatch.setenv(_LEGACY_CG_HEAD_ENV, "1")
    dispatched = head(x)
    assert torch.equal(dispatched, head._forward_legacy(x))  # legacy env => legacy exactly

    monkeypatch.delenv(_LEGACY_CG_HEAD_ENV, raising=False)
    fused = head(x)
    assert torch.equal(fused, head._forward_fused(x))  # fused opt-in => fused exactly

    # The two paths still agree within the certified fp64 tolerance.
    assert _max_abs(dispatched, fused) <= FWD_ATOL[torch.float64]


def test_env_var_non_one_values_keep_fused(monkeypatch):
    """Only the literal string "1" activates the legacy override: non-"1" values of
    ``DPTB_LEGACY_CG_HEAD`` leave the fused default path selected."""
    head = _build_head(WATER_BASIS, symmetrize=False, dtype=torch.float64, seed=8)
    x = torch.randn(3, head.irreps_in.dim, dtype=torch.float64)
    monkeypatch.setenv(_FUSED_CG_HEAD_ENV, "1")  # inert affirmation; fused is default
    for value in ("0", "true", "yes", ""):
        monkeypatch.setenv(_LEGACY_CG_HEAD_ENV, value)
        assert torch.equal(head(x), head._forward_fused(x))


def test_default_dispatch_is_fused(monkeypatch):
    """Project decision PERFORMANCE-DEFAULT: with NO env override set, forward()
    routes to the fused grouped-einsum path inside the certified eager fp32/fp64
    domain (precedence (c)); ``DPTB_FUSED_CG_HEAD=1`` is inert (not read)."""
    monkeypatch.delenv(_LEGACY_CG_HEAD_ENV, raising=False)
    monkeypatch.delenv(_FUSED_CG_HEAD_ENV, raising=False)
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float64, seed=21)
    x = torch.randn(5, head.irreps_in.dim, dtype=torch.float64)
    assert torch.equal(head(x), head._forward_fused(x))
    monkeypatch.setenv(_FUSED_CG_HEAD_ENV, "1")
    assert torch.equal(head(x), head._forward_fused(x))


def test_legacy_env_overrides_fused_default(monkeypatch):
    """Precedence (a) > (c): ``DPTB_LEGACY_CG_HEAD=1`` forces the legacy loop even
    when the inert ``DPTB_FUSED_CG_HEAD=1`` affirmation is also set."""
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float64, seed=22)
    x = torch.randn(4, head.irreps_in.dim, dtype=torch.float64)
    monkeypatch.setenv(_FUSED_CG_HEAD_ENV, "1")
    monkeypatch.setenv(_LEGACY_CG_HEAD_ENV, "1")
    assert torch.equal(head(x), head._forward_legacy(x))


# ===========================================================================
# A.5 #8 -- group-partition invariant + intra-group Wigner bit-identity
# ===========================================================================
@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
def test_group_partition_invariant_and_sums(basis):
    """Grouping by ``(input_index,row_l,col_l)`` equals grouping by
    ``(l_in,p_in,row_l,col_l)``; ``sum(P_g)==n_paths`` and
    ``sum(P_g*u_g)==num_path_weights`` (guards a future irreps with repeated (l,p))."""
    head = _build_head(basis, symmetrize=True, dtype=torch.float64)

    by_index = {}
    by_lin = {}
    for path in head._paths:
        input_index, row_index, col_index = path[0], path[1], path[2]
        mul_in, ir_in = head.irreps_in[input_index]
        row_l = head._shell_specs[row_index][2]
        col_l = head._shell_specs[col_index][2]
        by_index.setdefault((input_index, row_l, col_l), []).append(path)
        by_lin.setdefault((ir_in.l, ir_in.p, row_l, col_l), []).append(path)

    # Identical partition (same multiset of member-groups).
    assert len(by_index) == len(by_lin) == head._fused_num_groups
    part_index = sorted(sorted(id(p) for p in g) for g in by_index.values())
    part_lin = sorted(sorted(id(p) for p in g) for g in by_lin.values())
    assert part_index == part_lin

    # Sums recorded in the fused plan (num_paths, mul_in=u_g per group).
    total_paths = sum(grp[4] for grp in head._fused_groups)
    total_weights = sum(grp[4] * grp[2] for grp in head._fused_groups)
    assert total_paths == len(head._paths)
    assert total_weights == head.num_path_weights


@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
def test_intragroup_wigner_tensors_are_bit_identical(basis):
    """The fusion thesis: every per-path Wigner in a group is BIT-identical, so the
    group needs one shared ``[i,j,k]`` tensor (design A.1, ``wigner_intragroup_id=0``)."""
    head = _build_head(basis, symmetrize=True, dtype=torch.float64)
    groups = {}
    for path in head._paths:
        input_index, row_index, col_index = path[0], path[1], path[2]
        row_l = head._shell_specs[row_index][2]
        col_l = head._shell_specs[col_index][2]
        groups.setdefault((input_index, row_l, col_l), []).append(path)
    for members in groups.values():
        ref = getattr(head, members[0][5])
        for path in members[1:]:
            assert torch.equal(getattr(head, path[5]), ref)


def test_fused_scatter_index_matches_block_flatten_order():
    """The head-level scatter index length equals the total flattened block width
    ``sum_g P_g*i_g*j_g`` -- guards the cat/scatter alignment."""
    head = _build_head(CRYSTAL_BASIS, symmetrize=False, dtype=torch.float64)
    expected = sum(grp[4] * grp[5] * grp[6] for grp in head._fused_groups)
    assert head._fused_scatter_index.numel() == expected


# ===========================================================================
# A.8 -- certified-domain routing guard (project decision)
#
# The fused path is the DEFAULT ONLY inside the certified domain: eager fp32/fp64.
# Outside it -- autocast, non-fp32/64 input dtype, or
# ``use_deterministic_algorithms(True)`` -- forward() must transparently route to the
# bit-compatible legacy loop.  Each routing test proves the guard forces legacy via
# ``torch.equal`` to ``_forward_legacy`` in the SAME context: because fused is the
# in-domain default, if the guard did NOT fire forward() would run the fused path,
# which differs from legacy by reassociation roundoff, so bit-equality can only mean
# the guard fired (these are routing proofs, NOT numeric-parity claims -- low
# precision is exactly where fused and legacy are ALLOWED to diverge).  These tests
# are load-bearing on the fused DEFAULT alone; the ``fused_opt_in`` fixture also sets
# the inert ``DPTB_FUSED_CG_HEAD`` to assert that reserved name never disturbs the
# guard.
# ===========================================================================
def test_cpu_autocast_bf16_routes_to_legacy(fused_opt_in):
    """Inside ``torch.autocast('cpu', bfloat16)``, forward() equals _forward_legacy()
    bit-for-bit (both run the legacy loop under the same autocast)."""
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=15)
    x = torch.randn(4, head.irreps_in.dim, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dispatched = head(x)
        reference = head._forward_legacy(x)
    assert torch.equal(dispatched, reference)


def test_deterministic_mode_routes_to_legacy_cpu(fused_opt_in):
    """Under ``use_deterministic_algorithms(True)`` forward() routes to the per-path
    loop (which has no duplicate-destination scatter) and equals it exactly.
    try/finally always restores the global flag."""
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float64, seed=14)
    x = torch.randn(5, head.irreps_in.dim, dtype=torch.float64)
    torch.use_deterministic_algorithms(True)
    try:
        dispatched = head(x)
        assert torch.equal(dispatched, head._forward_legacy(x))
    finally:
        torch.use_deterministic_algorithms(False)


def test_half_input_routes_to_legacy(fused_opt_in):
    """``features.half()`` is outside the certified fp32/fp64 domain, so forward()
    routes to the legacy loop.

    (a) With an fp32-param head, the mixed Half/Float Linear makes the legacy loop
        raise -- forward() (routed identically) must raise the SAME error, encoding the
        actual behavior (not a numeric-parity claim).
    (b) With a genuine float16 head the legacy loop SUCCEEDS, so forward() must route
        to it and be bit-identical -- a positive proof of the dtype guard (the fused
        path would differ by roundoff)."""
    head32 = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=13)
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
    # Routed identically into the legacy loop => identical failure surface.
    assert type(forward_exc.value) is type(legacy_exc)
    assert str(forward_exc.value) == str(legacy_exc)

    head16 = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float16, seed=13)
    assert torch.equal(head16(xh), head16._forward_legacy(xh))


# ===========================================================================
# A.8 (CUDA) -- executed on the local GPU; skipped only when CUDA is unavailable.
# ===========================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fp32_forward_parity_fused_vs_legacy():
    """fp32 fused-vs-legacy parity on CUDA, certified with the same RELATIVE tolerance
    (rtol + atol floor) as the CPU fp32 case."""
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=17).cuda()
    x = torch.randn(6, head.irreps_in.dim, dtype=torch.float32, device="cuda")
    legacy = head._forward_legacy(x)
    fused = head._forward_fused(x)
    assert torch.allclose(
        fused, legacy, rtol=FWD_RTOL_FP32, atol=FWD_ATOL_FLOOR_FP32
    ), (
        f"cuda fp32 parity exceeds rtol={FWD_RTOL_FP32:.0e}/atol={FWD_ATOL_FLOOR_FP32:.0e} "
        f"(abs={_max_abs(legacy, fused):.3e}, rel={_max_rel(fused, legacy):.3e})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_deterministic_mode_routes_to_legacy(fused_opt_in):
    """On CUDA under ``use_deterministic_algorithms(True)`` forward() routes to the
    legacy loop and equals it, with NO exception.  Both paths share cuBLAS matmuls, so
    CUBLAS_WORKSPACE_CONFIG (set at import) is what lets the assertion run at all; it is
    the routing -- proven by ``torch.equal`` -- that this certifies (fused would differ
    by roundoff)."""
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=18).cuda()
    x = torch.randn(5, head.irreps_in.dim, dtype=torch.float32, device="cuda")
    torch.use_deterministic_algorithms(True)
    try:
        dispatched = head(x)
        assert torch.equal(dispatched, head._forward_legacy(x))
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_autocast_fp16_routes_to_legacy(fused_opt_in):
    """Inside ``torch.autocast('cuda', float16)``, forward() equals _forward_legacy()
    bit-for-bit -- proves routing under the CUDA/device-agnostic autocast flag."""
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=19).cuda()
    x = torch.randn(4, head.irreps_in.dim, dtype=torch.float32, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        dispatched = head(x)
        reference = head._forward_legacy(x)
    assert torch.equal(dispatched, reference)


# ===========================================================================
# H12-H14 -- reassociation-cancellation CHARACTERIZATION (round-3 heavy tests).
#
# The reviewer ran a 5000-seed Monte Carlo at unit-ish scale and found 34/5000
# seeds (seed 1135 among them) where the fused-vs-legacy fp32 drift EXCEEDS the
# atol=1e-6+rtol=5e-6 parity envelope in a locally-cancelling canvas cell, while
# BOTH paths stay individually accurate vs the fp64 truth -- i.e. reassociation
# roundoff at a cancellation, NOT a semantic bug.  That 5000-seed sweep is
# intentionally NOT run as a test (runtime); the external measurement is 34/5000.
#
# The exact reviewer harness is not bit-reproducible from the public module API, so
# H12 reproduces the SAME phenomenon deterministically: apply the per-tensor
# log-uniform [0.1,10] scale recipe and amplify the input magnitude (the drift is
# RELATIVE, so it only becomes envelope-visible at magnitude -- finding a), then
# scan a fixed (seed, symmetrize) grid (starting at the reviewer's 1135) for the
# first cancellation case.  These are CHARACTERIZATIONS of reassociation-not-bug,
# NOT parity gates: a failure here is NOT fixed by moving a tolerance.
# ===========================================================================
_H12_INPUT_SCALE = 100.0
_H12_SEED_GRID = [1135] + list(range(0, 48))
_H12_ENVELOPE_ATOL = 1e-6
_H12_ENVELOPE_RTOL = 5e-6


def _loguniform_scaled_head_and_input(seed, symmetrize, *, dtype=torch.float32, device="cpu"):
    """fp32 water head with per-tensor log-uniform [0.1,10] scaled params + an
    amplified input, all drawn deterministically from ``seed``."""
    head = _build_head(WATER_BASIS, symmetrize=symmetrize, dtype=dtype, seed=seed)
    if device != "cpu":
        head = head.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for param in head.parameters():
            exponent = torch.empty(1, device=device).uniform_(-1.0, 1.0, generator=gen).item()
            param.mul_(10.0 ** exponent)  # 10**[-1,1] == log-uniform in [0.1, 10]
    x = torch.randn(6, head.irreps_in.dim, generator=gen, dtype=dtype, device=device)
    x = x * _H12_INPUT_SCALE
    return head, x


def _fp64_truth_of(head, x, symmetrize):
    """The fp64 legacy forward of the SAME (promoted) weights on the SAME inputs."""
    head64 = _build_head(WATER_BASIS, symmetrize=symmetrize, dtype=torch.float64, seed=0)
    head64.load_state_dict(
        {k: v.detach().double().cpu() for k, v in head.state_dict().items()}, strict=True
    )
    return head64._forward_legacy(x.detach().double().cpu())


def _first_cancellation_case(device="cpu"):
    """Scan the fixed (seed, symmetrize) grid for the first fp32 case whose
    fused-vs-legacy drift exceeds the parity envelope with both outputs finite."""
    for seed in _H12_SEED_GRID:
        for symmetrize in (True, False):
            head, x = _loguniform_scaled_head_and_input(seed, symmetrize, device=device)
            legacy = head._forward_legacy(x)
            fused = head._forward_fused(x)
            envelope = _H12_ENVELOPE_ATOL + _H12_ENVELOPE_RTOL * legacy.abs()
            exceeds = bool(((fused - legacy).abs() > envelope).any())
            finite = bool(torch.isfinite(fused).all() and torch.isfinite(legacy).all())
            if exceeds and finite:
                return seed, symmetrize, head, x, legacy, fused
    return None


def test_h12_reassociation_cancellation_exceeds_envelope_but_both_stay_accurate():
    """H12: a cancellation case where fused-vs-legacy EXCEEDS the atol=1e-6+rtol=5e-6
    envelope while both fp32 outputs are finite AND individually within ~1e-3
    relative of the fp64 truth -- documenting reassociation-not-bug.  This is a
    CHARACTERIZATION, not a parity gate."""
    found = _first_cancellation_case()
    assert found is not None, (
        "no reassociation-cancellation case found across the scanned (seed, "
        "symmetrize) grid -- the fused/legacy reassociation regime changed; "
        "investigate before adjusting any tolerance"
    )
    _seed, symmetrize, _head, x, legacy, fused = found

    # (1) the parity envelope is genuinely exceeded (reassociation at a cancellation).
    assert not torch.allclose(fused, legacy, rtol=_H12_ENVELOPE_RTOL, atol=_H12_ENVELOPE_ATOL)
    # (2) both fp32 outputs are finite.
    assert torch.isfinite(fused).all() and torch.isfinite(legacy).all()
    # (3) both fp32 paths are individually accurate vs the fp64 truth of the SAME
    # inputs -- so the mutual drift is roundoff at a cancelling cell, not an error.
    truth = _fp64_truth_of(_head, x, symmetrize)
    ref = float(truth.abs().max())
    assert ref > 0.0
    assert (legacy.double() - truth).abs().max().item() / ref <= 1e-3
    assert (fused.double() - truth).abs().max().item() / ref <= 1e-3


def test_h12b_cancellation_paths_have_no_guaranteed_accuracy_ordering():
    """H12b (FINDING F -- honest accuracy claim): at a cancellation the fused and legacy
    fp32 outputs are NOT 'equally close' to the fp64 truth.  They sit at DIFFERENT
    distances from truth and neither path has a universal elementwise ordering over the
    other; both merely stay within the loose reassociation tolerance.  This pins the
    claim the docs were overstating (they used to say the paths stay 'equally close to
    the fp64 truth').  Reuses the deterministically-found cancellation case."""
    found = _first_cancellation_case()
    assert found is not None, (
        "no reassociation-cancellation case found across the scanned (seed, "
        "symmetrize) grid -- the reassociation regime changed; investigate before "
        "adjusting any tolerance"
    )
    _seed, symmetrize, _head, x, legacy, fused = found
    truth = _fp64_truth_of(_head, x, symmetrize)
    ref = float(truth.abs().max())
    assert ref > 0.0

    legacy_err = (legacy.double() - truth).abs()
    fused_err = (fused.double() - truth).abs()
    # At the cell of largest fused-vs-legacy divergence the two paths sit at DIFFERENT
    # distances from the truth (not "equally close") -- non-degenerate because
    # fused != legacy there and the truth is a single value.
    j = int((fused.double() - legacy.double()).abs().argmax())
    assert legacy_err.flatten()[j].item() != fused_err.flatten()[j].item()
    # Neither path is asserted closer than the other; both merely within the loose
    # (documented, NOT tight) reassociation envelope of the truth.
    assert legacy_err.max().item() / ref <= 1e-3
    assert fused_err.max().item() / ref <= 1e-3


@pytest.mark.parametrize("placement", ["static_weights", "dynamic_up", "input"])
def test_h13_independent_per_tensor_scale_preserves_per_path_accuracy(placement):
    """H13: a x1e2 scale on a SINGLE tensor (static_weights / dynamic_up / input)
    leaves EACH fp32 path individually within 1e-4 relative of the fp64 truth
    (accuracy preserved under scale imbalance).  Mutual fused-vs-legacy agreement is
    deliberately NOT asserted -- reassociation is allowed to drift them apart."""
    head = _build_head(WATER_BASIS, symmetrize=False, dtype=torch.float32, seed=7)
    torch.manual_seed(100)
    x = torch.randn(6, head.irreps_in.dim, dtype=torch.float32)
    with torch.no_grad():
        if placement == "static_weights":
            head.static_weights.mul_(1e2)
        elif placement == "dynamic_up":
            head.dynamic_up.weight.mul_(1e2)
            head.dynamic_up.bias.mul_(1e2)
        else:  # input
            x = x * 1e2

    head64 = _build_head(WATER_BASIS, symmetrize=False, dtype=torch.float64, seed=0)
    head64.load_state_dict(
        {k: v.detach().double() for k, v in head.state_dict().items()}, strict=True
    )
    truth = head64._forward_legacy(x.double())
    ref = float(truth.abs().max())
    assert ref > 0.0

    legacy = head._forward_legacy(x)
    fused = head._forward_fused(x)
    rel_legacy = (legacy.double() - truth).abs().max().item() / ref
    rel_fused = (fused.double() - truth).abs().max().item() / ref
    assert rel_legacy <= 1e-4, f"legacy rel drift {rel_legacy:.3e} (placement={placement})"
    assert rel_fused <= 1e-4, f"fused rel drift {rel_fused:.3e} (placement={placement})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_h14_cuda_fused_parity_and_cancellation_characterization():
    """H14 (runs on the local RTX GPU): fused-vs-legacy fp32 parity on random
    unit-scale CUDA inputs, plus one cancellation-style seed on CUDA asserting both
    outputs finite and within 1e-3 relative of the fp64 truth."""
    # (a) unit-scale fused-vs-legacy fp32 parity on CUDA (RELATIVE tolerance).
    head = _build_head(WATER_BASIS, symmetrize=True, dtype=torch.float32, seed=17).cuda()
    x = torch.randn(6, head.irreps_in.dim, dtype=torch.float32, device="cuda")
    legacy = head._forward_legacy(x)
    fused = head._forward_fused(x)
    assert torch.allclose(fused, legacy, rtol=FWD_RTOL_FP32, atol=FWD_ATOL_FLOOR_FP32), (
        f"cuda fp32 parity exceeds rtol={FWD_RTOL_FP32:.0e}/atol={FWD_ATOL_FLOOR_FP32:.0e} "
        f"(abs={_max_abs(legacy, fused):.3e}, rel={_max_rel(fused, legacy):.3e})"
    )

    # (b) a cancellation-style seed ON CUDA: exceeds the envelope, both finite, both
    # within 1e-3 relative of the fp64 truth (reassociation, not bug).
    found = _first_cancellation_case(device="cuda")
    assert found is not None, "no CUDA cancellation case found across the scanned grid"
    _seed, symmetrize, cuda_head, cx, cl, cf = found
    assert torch.isfinite(cf).all() and torch.isfinite(cl).all()
    truth = _fp64_truth_of(cuda_head, cx, symmetrize)
    ref = float(truth.abs().max())
    assert ref > 0.0
    assert (cl.double().cpu() - truth).abs().max().item() / ref <= 1e-3
    assert (cf.double().cpu() - truth).abs().max().item() / ref <= 1e-3

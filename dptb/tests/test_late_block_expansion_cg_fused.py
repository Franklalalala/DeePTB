"""Equivalence matrix for the fused ``LateBlockExpansionCGHead`` (design A.5).

The fused path (``_forward_fused``, default) collapses the legacy per-path Python
loop into ``G`` grouped batched einsums plus a single ``index_add`` scatter.  It is
a numerical-ASSOCIATION change only: the batched einsums and the scatter reassociate
the same floating-point sums in a different order than the reference
``_forward_legacy`` loop.  fp is non-associative, so bitwise identity is NOT expected
-- the drift is pure roundoff.

The tolerances below CERTIFY reassociation-only drift (measured fp64 <= ~4e-16 fwd /
~6e-14 grad, fp32 <= ~2e-7 fwd / ~2e-5 grad; design A.4).  **A failure above these
tolerances signals a real semantic divergence, not precision** -- do not "relax" a
tolerance to make such a failure pass.

Cases (design A.5):
  1. forward parity fused-vs-legacy: fp64 tight (1e-12) / fp32 loose (1e-5), water
     AND crystal bases, both symmetrize modes, node & edge invocation shapes;
  2. gradient parity on the 5 params AND the scalar-condition input, plus a fp64
     ``gradcheck`` of the fused backward on a small instance;
  3. SO(3) Wigner-D equivariance of the FUSED head via an INDEPENDENT dual-path
     oracle (e3nn D matrices), not the legacy loop;
  4. Hermiticity / symmetrize behaviour;
  5. padding / canvas zero-region invariance (untouched blocks stay bit-zero);
  6. state_dict round-trip: exact 5-tensor inventory + old->new strict=True load;
  7. env-var rollback (``DPTB_LEGACY_CG_HEAD=1``) selects the legacy path;
  8. group-partition invariant + intra-group Wigner bit-identity.
"""
from __future__ import annotations

import os

import pytest
import torch
from e3nn import o3

from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.late_block_expansion_cg import (
    LateBlockExpansionCGHead,
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
FWD_ATOL = {torch.float64: 1e-12, torch.float32: 1e-5}
GRAD_ATOL_FP64 = 1e-10
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


# ===========================================================================
# A.5 #1 -- forward parity fused vs legacy
# ===========================================================================
@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("symmetrize", [False, True])
@pytest.mark.parametrize("batch_shape", [(7,), (2, 3)])
def test_forward_parity_fused_matches_legacy(basis, dtype, symmetrize, batch_shape):
    """Node (symmetrize=True) AND edge (symmetrize=False) invocation shapes, both
    bases, fp64 tight / fp32 loose.  ``batch_shape=(2,3)`` also exercises the
    multi-dim ``*batch`` reshape/scatter path."""
    head = _build_head(basis, symmetrize=symmetrize, dtype=dtype)
    x = torch.randn(*batch_shape, head.irreps_in.dim, dtype=dtype)

    legacy = head._forward_legacy(x)
    fused = head._forward_fused(x)

    assert fused.shape == legacy.shape == (*batch_shape, head.max_norb, head.max_norb)
    drift = _max_abs(legacy, fused)
    assert drift <= FWD_ATOL[dtype], (
        f"forward drift {drift:.3e} exceeds the certified reassociation tolerance "
        f"{FWD_ATOL[dtype]:.0e} ({dtype}, symmetrize={symmetrize}) -- this is a "
        f"semantic divergence, not roundoff."
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


# ===========================================================================
# A.5 #2 -- gradient parity (params + scalar-condition input) + gradcheck
# ===========================================================================
@pytest.mark.parametrize("basis", [WATER_BASIS, CRYSTAL_BASIS])
@pytest.mark.parametrize("symmetrize", [False, True])
def test_gradient_parity_params_and_scalar_condition_input(basis, symmetrize):
    """Backprop ``out.pow(2).sum()`` through fused vs legacy; compare grads on all
    five learnable tensors AND on the input (the scalar-condition gradient flows
    ``features -> index_select -> condition_down -> dynamic_up -> mix``)."""
    head = _build_head(basis, symmetrize=symmetrize, dtype=torch.float64)
    x = torch.randn(6, head.irreps_in.dim, dtype=torch.float64)

    def grads(fn):
        xg = x.clone().requires_grad_(True)
        head.zero_grad(set_to_none=True)
        fn(xg).pow(2).sum().backward()
        return {n: p.grad.clone() for n, p in head.named_parameters()}, xg.grad.clone()

    g_legacy, gx_legacy = grads(head._forward_legacy)
    g_fused, gx_fused = grads(head._forward_fused)

    assert set(g_legacy) == STATE_DICT_KEYS
    for name in g_legacy:
        drift = _max_abs(g_legacy[name], g_fused[name])
        assert drift <= GRAD_ATOL_FP64, f"grad[{name}] drift {drift:.3e}"
    assert _max_abs(gx_legacy, gx_fused) <= GRAD_ATOL_FP64  # scalar-condition input


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
    head = _build_head(basis, symmetrize=True, dtype=torch.float64, seed=8)
    x = torch.randn(5, head.irreps_in.dim, dtype=torch.float64)

    monkeypatch.setenv(_LEGACY_CG_HEAD_ENV, "1")
    dispatched = head(x)
    assert torch.equal(dispatched, head._forward_legacy(x))  # env => legacy exactly

    monkeypatch.delenv(_LEGACY_CG_HEAD_ENV, raising=False)
    default = head(x)
    assert torch.equal(default, head._forward_fused(x))  # default => fused exactly

    # The two paths still agree within the certified fp64 tolerance.
    assert _max_abs(dispatched, default) <= FWD_ATOL[torch.float64]


def test_env_var_non_one_values_keep_fused(monkeypatch):
    head = _build_head(WATER_BASIS, symmetrize=False, dtype=torch.float64, seed=8)
    x = torch.randn(3, head.irreps_in.dim, dtype=torch.float64)
    for value in ("0", "true", "yes", ""):
        monkeypatch.setenv(_LEGACY_CG_HEAD_ENV, value)
        assert torch.equal(head(x), head._forward_fused(x))


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

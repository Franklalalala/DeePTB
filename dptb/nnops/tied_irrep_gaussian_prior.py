from __future__ import annotations

"""Helpers for a rowwise multiplicity-tied irreducible Gaussian prior."""

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from e3nn import o3

from dptb.nnops.tied_irrep_constants import TIED_IRREP_CANONICAL_IRREPS

TIED_IRREP_GAUSSIAN_PRIOR = "tied_irrep_gaussian"
TIED_IRREP_SUPPORTED_MODE = "so3_tied"
# TIED_IRREP_CANONICAL_IRREPS itself now lives in the dependency-free
# dptb.nnops.tied_irrep_constants (PR#31 review finding P2-2/P2-1: this
# literal was independently hardcoded in 3+ places, including a
# regex-string-diff copy in dptb.utils.argcheck that could silently diverge
# from this module's o3.Irreps-equivalence check). Re-exported here under the
# same name so existing ``from dptb.nnops.tied_irrep_gaussian_prior import
# TIED_IRREP_CANONICAL_IRREPS`` call sites (e.g. flow.py) keep working
# unchanged.
TIED_IRREP_LATENT_WIDTH = 9
TIED_IRREP_EFFECTIVE_VARIANCES = {0: 3.0, 1: 2.0, 2: 1.0}


@dataclass(frozen=True)
class TiedIrrepLayout:
    """One independent standard-normal ``(2l+1)``-vector per SO(3) degree.

    ``multiplicities[i]`` is how many whole irrep slices of ``degrees[i]`` the
    TARGET representation contains.  It is the per-degree variance factor, which
    is exactly what the historical hardcoded ``{0: 3.0, 1: 2.0, 2: 1.0}``
    encoded: the canonical ``3x0e + 2x1e + 1x2e`` noise tensor is "a sum of 3
    scalar copies, 2 vector copies and 1 rank-2 copy", so each degree's draw
    carries the variance of that many independent copies.  Deriving the same
    quantity from the target's own ``orbpair_irreps`` generalizes the rule to
    any basis instead of freezing water's.

    Multiplicity is counted GLOBALLY (over the mapper's whole irrep layout),
    not per row: a per-row live-slice count would make the prior's amplitude
    element-dependent, which is a physical bias the fixed-constant version
    never had.
    """

    degrees: Tuple[int, ...]
    multiplicities: Tuple[int, ...]
    offsets: Tuple[int, ...]
    width: int

    def offset_of(self, degree: int) -> Optional[int]:
        try:
            return self.offsets[self.degrees.index(int(degree))]
        except ValueError:
            return None

    def multiplicity_of(self, degree: int) -> int:
        try:
            return self.multiplicities[self.degrees.index(int(degree))]
        except ValueError:
            return 0


def _build_layout(counts: dict) -> TiedIrrepLayout:
    degrees = tuple(sorted(int(d) for d in counts))
    offsets, offset = [], 0
    for degree in degrees:
        offsets.append(offset)
        offset += 2 * degree + 1
    return TiedIrrepLayout(
        degrees=degrees,
        multiplicities=tuple(int(counts[d]) for d in degrees),
        offsets=tuple(offsets),
        width=offset,
    )


# Water's frozen layout, kept as the default so every caller that does not opt
# into a target-derived layout keeps byte-identical behavior (degrees 0/1/2 with
# variance factors 3/2/1, and every degree >= 3 left exactly zero).
CANONICAL_TIED_IRREP_LAYOUT = _build_layout({0: 3, 1: 2, 2: 1})


def tied_irrep_layout_from_slices(
    slices: Iterable[Tuple[int, int, int]]
) -> TiedIrrepLayout:
    """Derive the latent layout from the TARGET's own irrep slices.

    Every SO(3) degree the target representation contains gets its own
    independent ``(2l+1)``-vector, so no degree is silently left at the
    deterministic zero start state.  There is deliberately no configuration
    entry for this: the noise tensor's irrep content is a property of what is
    being transported, not a free hyper-parameter.
    """
    counts: dict = {}
    for start, end, degree in tuple(slices):
        start, end, degree = int(start), int(end), int(degree)
        width = end - start
        if degree < 0 or width <= 0:
            raise ValueError(
                "tied_irrep_gaussian received an invalid irrep slice "
                f"({start}, {end}, {degree})."
            )
        if width != 2 * degree + 1:
            raise ValueError(
                "tied_irrep_gaussian requires whole SO(3) irrep slices; "
                f"degree {degree} expects width {2 * degree + 1}, got {width}."
            )
        counts[degree] = counts.get(degree, 0) + 1
    if not counts:
        raise ValueError(
            "tied_irrep_gaussian cannot derive a latent layout from an empty "
            "irrep slice list."
        )
    return _build_layout(counts)


def normalize_tied_irrep_mode(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def validate_tied_irrep_options(*, mode: object, irreps: object = None) -> None:
    """Validate the sampler mode; ``irreps`` is deprecated and ignored.

    The latent layout is now derived from the target representation
    (:func:`tied_irrep_layout_from_slices`), so ``tied_irrep_irreps`` is no
    longer an input.  Historical configurations that still carry the canonical
    string keep loading; anything else warns rather than hard-failing, because
    the value no longer selects anything.
    """
    mode_name = normalize_tied_irrep_mode(mode)
    if mode_name != TIED_IRREP_SUPPORTED_MODE:
        raise ValueError(
            "tied_irrep_gaussian requires tied_irrep_mode='so3_tied'; "
            f"got {mode!r}."
        )
    if irreps is None or str(irreps).strip() == "":
        return
    try:
        parsed = o3.Irreps(str(irreps))
    except Exception as exc:
        raise ValueError(
            "tied_irrep_gaussian requires a valid tied_irrep_irreps string."
        ) from exc
    if parsed != o3.Irreps(TIED_IRREP_CANONICAL_IRREPS):
        warnings.warn(
            "flow_options.tied_irrep_irreps is deprecated and ignored: the "
            "tied-irrep latent layout is derived from the target's own "
            f"orbpair_irreps. Got {irreps!r}; remove the key.",
            FutureWarning,
            stacklevel=2,
        )


def draw_standard_tied_irrep_latent(
    row_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    layout: TiedIrrepLayout = CANONICAL_TIED_IRREP_LAYOUT,
) -> torch.Tensor:
    """Draw the independent standard-normal total-L variables per row."""
    return torch.randn(
        int(row_count),
        int(layout.width),
        device=device,
        dtype=dtype,
        generator=generator,
    )


def effective_tied_irrep_latent(
    standard: torch.Tensor,
    *,
    layout: TiedIrrepLayout = CANONICAL_TIED_IRREP_LAYOUT,
) -> torch.Tensor:
    """Apply per-degree multiplicity factors before any public sigma scale.

    Layout is the concatenation, in ascending degree order, of one
    ``(2l+1)``-vector per degree present in the target.  Component variance for
    degree ``l`` is that degree's global slice multiplicity, matching a sum of
    that many independent copies.  For the default (water) layout this is the
    historical ``[L0, L1_m0..m2, L2_m0..m4]`` with variances 3, 2 and 1.
    """
    if standard.ndim != 2 or int(standard.shape[-1]) != int(layout.width):
        raise ValueError(
            "tied_irrep_gaussian latent must have shape "
            f"[rows, {int(layout.width)}]."
        )
    out = standard.clone()
    for degree, multiplicity, offset in zip(
        layout.degrees, layout.multiplicities, layout.offsets
    ):
        if multiplicity == 1:
            continue
        out[:, offset:offset + 2 * degree + 1] *= math.sqrt(float(multiplicity))
    return out


def _as_bool_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(mask, device=like.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(like.shape):
        raise ValueError(
            "tied_irrep_gaussian mask shape "
            f"{tuple(mask.shape)} does not match feature shape {tuple(like.shape)}."
        )
    return mask


def fill_tied_irrep_rme(
    like: torch.Tensor,
    slices: Iterable[Tuple[int, int, int]],
    mask: torch.Tensor,
    effective_latent: torch.Tensor,
    *,
    sigma: float,
    layout: TiedIrrepLayout = CANONICAL_TIED_IRREP_LAYOUT,
) -> torch.Tensor:
    """Broadcast one rowwise total-L field into every compatible RME slice.

    All slices with the same SO(3) degree receive the same rowwise component.
    Degrees absent from ``layout`` stay exactly zero before any block
    projection -- with the default (water) layout that is every degree >= 3;
    with a target-derived layout (:func:`tied_irrep_layout_from_slices`) it is
    none of them.
    """
    if like.ndim < 2:
        raise ValueError(
            "tied_irrep_gaussian requires rank-2 DeePTB RME feature rows."
        )
    mask = _as_bool_mask(mask, like)
    latent = torch.as_tensor(
        effective_latent, device=like.device, dtype=like.dtype
    )
    if latent.ndim != 2 or int(latent.shape[0]) != int(like.shape[0]):
        raise ValueError(
            "tied_irrep_gaussian latent row count must match feature rows."
        )
    if int(latent.shape[-1]) != int(layout.width):
        raise ValueError(
            "tied_irrep_gaussian latent width must be "
            f"{int(layout.width)}."
        )
    if not math.isfinite(float(sigma)):
        raise ValueError("tied_irrep_gaussian sigma must be finite.")

    out = torch.zeros_like(like)
    for start, end, degree in tuple(slices):
        start = int(start)
        end = int(end)
        degree = int(degree)
        width = end - start
        expected_width = 2 * degree + 1
        if start < 0 or end > int(like.shape[-1]) or width <= 0:
            raise ValueError(
                "tied_irrep_gaussian received an invalid irrep slice "
                f"({start}, {end}, {degree})."
            )
        if width != expected_width:
            raise ValueError(
                "tied_irrep_gaussian requires whole SO(3) irrep slices; "
                f"degree {degree} expects width {expected_width}, got {width}."
            )
        seg_mask = mask[:, start:end]
        partial = seg_mask.any(dim=-1) & ~seg_mask.all(dim=-1)
        if bool(partial.any().item()):
            raise ValueError(
                "tied_irrep_gaussian requires masks that keep or drop each "
                "whole irrep slice per row; partial component masks are unsupported."
            )
        offset = layout.offset_of(degree)
        if offset is None:
            continue
        source = latent[:, offset:offset + expected_width]
        if int(source.shape[-1]) != width:
            raise ValueError(
                "tied_irrep_gaussian latent degree width does not match slice "
                f"({start}, {end}, {degree})."
            )
        out[:, start:end] = source
    return out.masked_fill(~mask, 0.0) * float(sigma)


def dense_all_one_irrep_expansion(
    z: torch.Tensor,
    *,
    irrep_spec: str = TIED_IRREP_CANONICAL_IRREPS,
) -> torch.Tensor:
    """Reference dense all-one expansion for the canonical 14D irrep vector.

    Matches the real production path (``fill_tied_irrep_rme`` ->
    ``BlockStateCodec.rme_to_blocks``, i.e. ``E3Hamiltonian``) bit-exactly for
    EVERY block of water's real ``3s2p1d`` oxygen basis, mirror pairs and
    cross-degree pairs included -- see
    ``test_dense_all_one_expansion_matches_production_codec_full_water_oxygen_matrix``
    and ``test_dense_all_one_expansion_matches_production_codec_on_homonuclear_edge``
    in ``dptb/tests/test_tied_irrep_gaussian_prior.py`` (the latter confirms
    edge rows, not just node rows, reduce to the identical convention once a
    directed edge and its reverse partner are tied to the same field).

    Shell-index convention: within one irrep chunk (``mul`` copies of the same
    ``ir``, e.g. the two ``1e`` radial copies), copy order follows ascending
    position in the flattened input -- i.e. copy 0 is the "first" shell, copy
    1 the "second", matching ``OrbitalMapper.get_orbpair_maps``'s ascending
    radial-index storage convention exactly.  Across chunks, ascending chunk
    order (the irrep string's own declared order, e.g. s before p before d)
    matches ``OrbitalMapper.get_orbpairtype_maps``'s ascending type order.

    Production (``BlockStateCodec.rme_to_blocks`` ->
    ``feature_tensors_to_block_tensors``) stores each such shell pair exactly
    once, in ascending shell-index order, and reconstructs the descending
    ("mirror") direction of every off-diagonal pair by a literal, unsigned
    transpose of that canonical block -- never an independent recompute (see
    ``dptb/data/interfaces/blockwise_tensor.py``'s
    ``symmetrize_onsite``/``complete_edge_blocks_from_reverse`` paths). A
    single triple-nested loop that computes every ordered ``(j, k)`` chunk
    pair independently -- the earlier version of this function -- disagrees
    with that convention in two ways, both fixed by the post-processing below
    instead of by an independent recompute of the descending direction:

    1. Same-chunk, multiple radial copies (e.g. the two ``1e`` "p" copies):
       the per-``(j, k)`` einsum's ``weights`` tensor is all-ones and does not
       depend on the copy indices ``(u, v)``, so every copy-pair sub-block
       comes out numerically IDENTICAL across ``(u, v)`` -- including the
       ``u > v`` "descending copy" ones -- instead of the ``u <= v`` value's
       transpose.
    2. Cross-chunk, different total degree (e.g. "p" vs "d"): the independent
       computation for chunk pair ``(k, j)`` uses a fresh
       ``wigner_3j(ir_out2.l, ir_out1.l, ir_in.l)`` call.  Swapping a
       Wigner-3j's first two arguments equals the transpose of the original
       only when ``ir_out1.l + ir_out2.l + ir_in.l`` is even; a block that
       sums more than one coupled degree of mixed parity (e.g. p-d's L=1 and
       L=2) then disagrees with the transpose on some channels and agrees on
       others -- neither the correct value nor even internally
       transpose-consistent with its own canonical direction.
    """
    validate_tied_irrep_options(mode=TIED_IRREP_SUPPORTED_MODE, irreps=irrep_spec)
    irreps = o3.Irreps(irrep_spec)
    x = torch.as_tensor(z)
    squeeze = False
    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze = True
    if x.ndim != 2 or int(x.shape[-1]) != int(irreps.dim):
        raise ValueError(
            "dense_all_one_irrep_expansion expects shape "
            f"[rows, {int(irreps.dim)}]."
        )
    batch = int(x.shape[0])
    pieces = [
        x[:, sl].reshape(batch, int(mul), int(ir.dim))
        for sl, (mul, ir) in zip(irreps.slices(), irreps)
    ]
    outputs = {}
    for i, (mul_in, ir_in) in enumerate(irreps):
        for j, (mul_out1, ir_out1) in enumerate(irreps):
            for k, (mul_out2, ir_out2) in enumerate(irreps):
                if ir_in not in ir_out1 * ir_out2:
                    continue
                weights = torch.ones(
                    int(mul_in),
                    int(mul_out1),
                    int(mul_out2),
                    device=x.device,
                    dtype=x.dtype,
                )
                # Standard Wigner-3j -> Clebsch-Gordan normalization: e3nn's
                # raw ``wigner_3j`` is missing the sqrt(2L+1) factor relative
                # to a normalized CG coefficient, where L is the coupled
                # (summed/input) degree ``ir_in.l`` here.  This must match the
                # production codec exactly: ``E3Hamiltonian._initialize_CG_basis``
                # (dptb/nn/hamiltonian.py) builds its ``cgbasis`` as
                # ``wigner_3j(l1, l2, l_ird) * (2 * l_ird + 1) ** 0.5`` for
                # every coupled degree ``l_ird``, which is exactly ``ir_in.l``
                # in this loop's naming.  Dropping this factor was PR#31
                # review finding P1-1: every L>=1 contribution here was
                # under-scaled by 1/sqrt(2L+1) relative to
                # ``BlockStateCodec.rme_to_blocks`` (the real production path).
                w3j = o3.wigner_3j(
                    int(ir_out1.l),
                    int(ir_out2.l),
                    int(ir_in.l),
                    dtype=x.dtype,
                    device=x.device,
                ) * (2 * int(ir_in.l) + 1) ** 0.5
                result = torch.einsum(
                    "wuv,ijk,bwk->buivj",
                    weights,
                    w3j,
                    pieces[i],
                ).reshape(batch, int(mul_out1 * ir_out1.dim), int(mul_out2 * ir_out2.dim))
                key = (j, k)
                outputs[key] = result if key not in outputs else outputs[key] + result

    # Fix 1 (same-chunk copy order): every (u, v) copy-pair sub-block the loop
    # above produced for a diagonal chunk key (j, j) is numerically identical
    # (see docstring point 1).  Keep that shared value for u <= v (the
    # ascending/canonical copy order, already correct -- e.g. p1-p1, p1-p2,
    # p2-p2) and overwrite u > v (e.g. p2-p1) with its transpose.
    for j, (mul_j, ir_j) in enumerate(irreps):
        mul_j = int(mul_j)
        if mul_j <= 1:
            continue
        block = outputs.get((j, j))
        if block is None:
            continue
        dim_j = int(ir_j.dim)
        template = block[:, 0:dim_j, 0:dim_j]
        fixed = block.clone()
        for u in range(mul_j):
            for v in range(mul_j):
                if u <= v:
                    continue
                fixed[:, u * dim_j:(u + 1) * dim_j, v * dim_j:(v + 1) * dim_j] = (
                    template.transpose(-1, -2)
                )
        outputs[(j, j)] = fixed

    # Fix 2 (cross-chunk mirror): discard whatever the independent
    # (k, j)-with-k>j computation above produced (see docstring point 2) and
    # replace it with the literal transpose of the ascending-chunk-order
    # (j, k) canonical block, matching production exactly.
    n_chunks = len(irreps)
    for j in range(n_chunks):
        for k in range(j + 1, n_chunks):
            canonical = outputs.get((j, k))
            if canonical is not None:
                outputs[(k, j)] = canonical.transpose(-1, -2)

    rows = []
    for i, (_mul_i, ir_i) in enumerate(irreps):
        blocks = []
        for j, (_mul_j, ir_j) in enumerate(irreps):
            block = outputs.get((i, j))
            if block is None:
                block = torch.zeros(
                    batch,
                    int(irreps[i].dim),
                    int(irreps[j].dim),
                    device=x.device,
                    dtype=x.dtype,
                )
            blocks.append(block)
        rows.append(torch.cat(blocks, dim=-1))
    out = torch.cat(rows, dim=-2)
    return out.reshape(out.shape[-2], out.shape[-1]) if squeeze else out

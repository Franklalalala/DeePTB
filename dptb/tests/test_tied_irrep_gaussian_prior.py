from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml
from e3nn import o3

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_mask_from_shapes,
    strict_reverse_edge_index,
)
from dptb.nnops.block_flow_codec import project_block_state
from dptb.nnops.tied_irrep_gaussian_prior import (
    TIED_IRREP_EFFECTIVE_VARIANCES,
    dense_all_one_irrep_expansion,
    effective_tied_irrep_latent,
    fill_tied_irrep_rme,
)
from dptb.utils.argcheck import flow_options, validate_block_ode_contract

from test_residual_ao_block_ode import (  # noqa: E402
    FP64_ATOL,
    _EndpointSpy,
    _LinearEchoModel,
    _b_flow,
    _b_record,
    _mapper,
    _rotate_canvas_blocks,
    _shared_canvas_wigner_d,
    _water_graph,
    _water_mapper,
)


_TIED_SEED = 20260721


def _tied_options(seed=_TIED_SEED, **overrides):
    options = {
        "prior": "tied_irrep_gaussian",
        "tied_irrep_mode": "so3_tied",
        "tied_irrep_irreps": "3x0e + 2x1e + 1x2e",
        "node_sigma": 1.0,
        "edge_sigma": 1.0,
        "tied_irrep_sigma": 1.0,
        "tied_irrep_validation_seed": seed,
    }
    options.update(overrides)
    return options


def _b_tied_flow(mapper, *, dtype=torch.float64, seed=_TIED_SEED, **overrides):
    return _b_flow(mapper, dtype=dtype, **_tied_options(seed=seed, **overrides))


# The docs' Section-4 worked-example z vector, water oxygen's canonical
# 3x0e+2x1e+1x2e (3s2p1d) basis.  Shared by the dense-mirror full-matrix
# cross-check (node row) and its edge-row sibling below.
_CANONICAL_O_Z = torch.tensor(
    [
        0.10, -0.20, 0.30,
        0.01, 0.02, 0.03,
        -0.04, 0.05, -0.06,
        0.07, -0.08, 0.09, -0.10, 0.11,
    ],
    dtype=torch.float64,
)

# The 6 individual shells packed into _CANONICAL_O_Z's 14 components, in
# ascending (canonical, per OrbitalMapper.get_orbpair_maps) shell-index
# order: 3 tied s-copies, then 2 tied p-copies, then 1 d-copy.
_CANONICAL_O_SHELLS = (
    ("s1", slice(0, 1)),
    ("s2", slice(1, 2)),
    ("s3", slice(2, 3)),
    ("p1", slice(3, 6)),
    ("p2", slice(6, 9)),
    ("d1", slice(9, 14)),
)


def _canonical_o_effective_latent(z=_CANONICAL_O_Z):
    """The tied effective (g0, g1, g2) fields _CANONICAL_O_Z's copies sum to."""
    g0 = z[0] + z[1] + z[2]
    g1 = z[3:6] + z[6:9]
    g2 = z[9:14]
    return torch.cat([g0.reshape(1), g1, g2])


def _o2_edge_graph(mapper, *, dtype=torch.float64):
    """A synthetic two-oxygen graph with both directions of one O-O bond.

    ``dense_all_one_irrep_expansion`` only ever produces a square block
    matching a single species' full canonical basis (O's 3s2p1d here), so
    exercising the codec's EDGE path against it needs a homonuclear bond --
    water's own O-H/H-O edges are rectangular (14 x 5) and not comparable.
    Both directions (0,1) and (1,0) must be present as a genuine reverse-edge
    pair, not a single one-off edge: non-SOC edge RME storage keeps only the
    ascending-shell-index half of each directed edge's OWN content (mirroring
    onsite's compression -- see
    ``OrbitalMapper.get_orbpairtype_maps``/``get_orbpair_maps``) and relies on
    ``complete_edge_blocks_from_reverse`` (``H_ij(R)=H_ji(-R)^T``) for the
    rest, so a missing reverse partner would leave the descending-shell-index
    entries unresolved (``rme_to_blocks`` runs with
    ``strict_complete_edges=True``).
    """
    t_o = mapper.chemical_symbol_to_type["O"]
    return {
        _keys.ATOM_TYPE_KEY: torch.tensor([[t_o], [t_o]], dtype=torch.long),
        _keys.BATCH_KEY: torch.zeros(2, dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.zeros(2, 3, dtype=dtype),
        _keys.EDGE_TYPE_KEY: torch.tensor(
            [[mapper.bond_to_type["O-O"]], [mapper.bond_to_type["O-O"]]]
        ),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=dtype),
        "cell": torch.eye(3, dtype=dtype) * 10.0,
        "pbc": torch.tensor([False, False, False]),
        _keys.SAMPLE_UID_KEY: torch.tensor([1], dtype=torch.long),
    }


def _node_tied_draw(flow, dim, *, types, batch, uids, seed, device):
    like = torch.zeros(len(batch), dim, dtype=torch.float64)
    payload = {
        _keys.ATOM_TYPE_KEY: torch.tensor([[t] for t in types], dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor(batch, dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0], [0]], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.tensor([[0]], dtype=torch.long),
        _keys.SAMPLE_UID_KEY: torch.tensor(uids, dtype=torch.long),
    }
    return flow._tied_irrep_gaussian_prior_like(
        like,
        flow.node_sigma,
        data=payload,
        label="node",
        num_graphs=len(uids),
        generator=flow._seeded_generator(device, seed),
    )


def test_dense_all_one_expansion_fixed_vector_fixture():
    """Golden fixture for ``dense_all_one_irrep_expansion``, regenerated after
    PR#31 review finding P1-1 (missing sqrt(2L+1) Wigner-3j -> Clebsch-Gordan
    normalization for every L>=1 channel).  Values below were printed by a
    throwaway script calling the FIXED implementation, not hand-derived --
    see ``test_dense_all_one_expansion_matches_production_codec_full_water_oxygen_matrix``
    just below for the independent cross-check against the real production
    codec that these numbers were verified against (bit-exact to fp64).

    Every sub-block asserted here (s-s, p1-p1 raw/symmetrized, s1-p1, s1-d1)
    sits at an ascending shell-index (canonical) or self-pair position, so the
    dense-mirror fix (below, dedicated dense-vs-mirror review lane) leaves all
    of their values bit-identical -- only the matrix-rank assertions change,
    see the comment there.
    """
    z = torch.tensor(
        [
            0.10,
            -0.20,
            0.30,
            0.01,
            0.02,
            0.03,
            -0.04,
            0.05,
            -0.06,
            0.07,
            -0.08,
            0.09,
            -0.10,
            0.11,
        ],
        dtype=torch.float64,
    )
    block = dense_all_one_irrep_expansion(z)

    torch.testing.assert_close(
        block[:3, :3],
        torch.full((3, 3), 0.2, dtype=torch.float64),
        rtol=0.0,
        atol=5e-5,
    )
    torch.testing.assert_close(
        block[3:6, 3:6],
        torch.tensor(
            [
                [0.0009, -0.0778, 0.0000],
                [-0.0354, 0.1890, -0.0919],
                [0.0990, -0.0495, 0.1565],
            ],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=5e-5,
    )
    torch.testing.assert_close(
        0.5 * (block[3:6, 3:6] + block[3:6, 3:6].T),
        torch.tensor(
            [
                [0.0009, -0.0566, 0.0495],
                [-0.0566, 0.1890, -0.0707],
                [0.0495, -0.0707, 0.1565],
            ],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=5e-5,
    )
    torch.testing.assert_close(
        block[0:1, 3:6],
        torch.tensor([[-0.0300, 0.0700, -0.0300]], dtype=torch.float64),
        rtol=0.0,
        atol=5e-5,
    )
    torch.testing.assert_close(
        block[0:1, 9:14],
        torch.tensor(
            [[0.0700, -0.0800, 0.0900, -0.1000, 0.1100]],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=5e-5,
    )
    # Dense-mirror review (F:\claude\0721_pr31_fix dense-mirror lane): before
    # the mirror-direction fix, the three descending-shell-index blocks
    # (p2-p1, d1-p1, d1-p2) were erroneously IDENTICAL to their ascending
    # partners (p1-p2, p1-d1, p2-d1) instead of those partners' transpose --
    # a spurious extra linear dependency that dropped the observed rank of
    # THIS SPECIFIC z's expansion from the fixed value below to 9.  11 is not
    # a general invariant (see docs Section 8, "numerical AO-matrix rank...
    # need not equal the prior support rank") -- it is this fixture's own
    # verified value, confirmed well clear of the 1e-12 cutoff: the 11th
    # singular value is ~2e-2 and the 12th is ~8e-17 (a 15-order-of-magnitude
    # gap), both before and after symmetrization.
    assert int(torch.linalg.matrix_rank(block, tol=1e-12).item()) == 11
    sym = 0.5 * (block + block.T)
    assert int(torch.linalg.matrix_rank(sym, tol=1e-12).item()) == 11


def test_dense_all_one_expansion_matches_production_codec_full_water_oxygen_matrix():
    """Dense-mirror review (fix/0721-dense-mirror lane) fast-vs-dense
    full-matrix cross-check, superseding the narrower canonical-only version
    of this test (formerly
    ``test_dense_all_one_expansion_matches_production_codec_on_water_oxygen_row``,
    which asserted only 4 single-copy/canonical-direction blocks -- s-s,
    s1-p1, s1-d1, and raw p1-p1 -- and explicitly scoped OUT mirror-direction
    blocks).  PR#31 review's original gap: the two "halves" of this prior's
    math -- ``fill_tied_irrep_rme`` (production) and
    ``dense_all_one_irrep_expansion`` (docs'/tests' reference) -- were never
    cross-checked against each other anywhere; the pr31fix lane's follow-up
    doubt was whether the narrow version's carved-out mirror/cross-degree
    scope limit hid a real bug or just an untested corner.

    Investigation result: REAL bug, not a misreading.  Feeding the doc's own
    Section-4 ``z`` vector through both paths on water's real oxygen
    ``3s2p1d`` basis (the canonical ``3x0e+2x1e+1x2e`` shape) and comparing
    EVERY ordered pair of O's 6 individual shells (s1, s2, s3, p1, p2, d1 --
    36 blocks: 6 self-pairs + 15 canonical/ascending pairs + 15 mirror/
    descending pairs) found exactly 3 mismatches before the fix: p2-p1,
    d1-p1, d1-p2 -- all descending-shell-index mirrors of either a
    multi-copy pair (p1-p2, tied to the identical field so dense's old
    ``weights``-are-copy-index-blind einsum made p2-p1 identically equal to
    p1-p2 instead of its transpose) or a cross-degree pair (p-d, whose
    coupled L=1/L=2 channels have mixed parity, so dense's old independent
    ``wigner_3j(ir_out2.l, ir_out1.l, ir_in.l)`` recompute for d-p disagreed
    with production's transpose-of-p-d on the L=2 channel's sign).  All 3
    turned out to be EXACTLY production's transpose of dense's own (already
    correct) canonical block -- confirmed for all 15 unordered pairs, not
    just the 3 that were broken -- matching literally how
    ``BlockStateCodec.rme_to_blocks`` -> ``feature_tensors_to_block_tensors``
    derives every off-diagonal onsite mirror
    (``symmetrize_onsite``'s ``sub_block[:, col, row] = part.transpose(-1,
    -2)``).  Fixed in ``dense_all_one_irrep_expansion`` by deriving every
    mirror block that way instead of an independent recompute; see that
    function's docstring and inline comments for the full mechanism, and
    ``test_dense_all_one_expansion_matches_production_codec_on_homonuclear_edge``
    below for the edge-row half of this cross-check.
    """
    mapper = _water_mapper()
    flow = _b_tied_flow(mapper)
    data = _water_graph(mapper, dtype=torch.float64)
    dim = mapper.orbpair_irreps.dim

    z = _CANONICAL_O_Z
    effective_latent = _canonical_o_effective_latent(z)

    # _water_graph's atomic_numbers=[8,1,1] -> row 0 is the O node; only that
    # row's mask is set so the H rows stay exactly zero (irrelevant here).
    node_like = torch.zeros(3, dim, dtype=torch.float64)
    node_mask = torch.zeros_like(node_like, dtype=torch.bool)
    node_mask[0, :] = True
    node_latent = torch.zeros(3, 9, dtype=torch.float64)
    node_latent[0] = effective_latent

    slices = flow._te_irrep_slices(dim)
    node_rme = fill_tied_irrep_rme(node_like, slices, node_mask, node_latent, sigma=1.0)
    n_edges = int(data[_keys.EDGE_INDEX_KEY].shape[1])
    edge_rme = torch.zeros(n_edges, dim, dtype=torch.float64)

    packed = flow.block_codec.rme_to_blocks(data, node_rme, edge_rme, project=False)
    production = packed.node_blocks[0]
    dense = dense_all_one_irrep_expansion(z)

    # Headline claim: bit-exact over the WHOLE 14x14 matrix, not a subset.
    torch.testing.assert_close(production, dense, rtol=0.0, atol=FP64_ATOL)

    # Per-shell-pair breakdown of all 36 ordered pairs (every canonical AND
    # every mirror direction) so a future regression fails pinpointing a
    # specific shell pair instead of an opaque whole-matrix diff.
    for a_name, a_slice in _CANONICAL_O_SHELLS:
        for b_name, b_slice in _CANONICAL_O_SHELLS:
            torch.testing.assert_close(
                production[a_slice, b_slice],
                dense[a_slice, b_slice],
                rtol=0.0,
                atol=FP64_ATOL,
                msg=f"{a_name}-{b_name} block mismatch (production vs dense)",
            )


def test_dense_all_one_expansion_matches_production_codec_on_homonuclear_edge():
    """Dense-mirror review: edge-row half of the full-matrix cross-check.

    The node/onsite check above alone would leave ``fill_tied_irrep_rme`` ->
    ``rme_to_blocks``'s EDGE path -- a structurally different mirror
    mechanism (``complete_edge_blocks_from_reverse``'s reverse-edge
    transpose ``H_ij(R)=H_ji(-R)^T``, not onsite's same-block
    ``symmetrize_onsite`` transpose) -- unverified against dense, even after
    the fix above.  Feeding the SAME tied ``z`` pattern to both directions of
    a genuine reverse-edge pair (``_o2_edge_graph``, a synthetic homonuclear
    O-O bond so the edge block is square and comparable to dense at all,
    unlike water's own rectangular O-H/H-O edges) makes the two mechanisms
    converge: each edge's own ascending-shell-index content is tied to the
    identical value production's onsite path would use, and the completed
    descending-shell-index content one edge borrows from its reverse partner
    is therefore that partner's own (identically tied) ascending content,
    transposed -- exactly the fixed ``dense_all_one_irrep_expansion``'s
    mirror convention.  Confirmed empirically: both edge directions come out
    bit-identical to ``dense`` (and, as a side effect of this specific
    symmetric setup -- not a general edge-prior property -- to each other).
    """
    mapper = _water_mapper()
    flow = _b_tied_flow(mapper)
    dim = mapper.orbpair_irreps.dim
    data = _o2_edge_graph(mapper)

    z = _CANONICAL_O_Z
    effective_latent = _canonical_o_effective_latent(z)

    slices = flow._te_irrep_slices(dim)
    node_like = torch.zeros(2, dim, dtype=torch.float64)
    node_mask = torch.zeros_like(node_like, dtype=torch.bool)
    node_rme = fill_tied_irrep_rme(
        node_like, slices, node_mask, torch.zeros(2, 9, dtype=torch.float64), sigma=1.0
    )

    edge_like = torch.zeros(2, dim, dtype=torch.float64)
    edge_mask = torch.ones_like(edge_like, dtype=torch.bool)
    edge_latent = effective_latent.reshape(1, 9).expand(2, 9).clone()
    edge_rme = fill_tied_irrep_rme(edge_like, slices, edge_mask, edge_latent, sigma=1.0)

    packed = flow.block_codec.rme_to_blocks(data, node_rme, edge_rme, project=False)
    dense = dense_all_one_irrep_expansion(z)

    torch.testing.assert_close(
        packed.edge_blocks[0], dense, rtol=0.0, atol=FP64_ATOL, msg="O(0)->O(1) edge block"
    )
    torch.testing.assert_close(
        packed.edge_blocks[1], dense, rtol=0.0, atol=FP64_ATOL, msg="O(1)->O(0) edge block"
    )


def test_effective_latent_variances_match_multiplicity_sums():
    """PR#31 review nit-3: assert each L=1/L=2 component individually rather
    than only their mean-over-degree.  The original mean-based check was
    statistically fine (``effective_tied_irrep_latent`` applies a uniform
    per-slice scale, so no single m-component can get a different factor than
    its siblings) but was a strictly weaker regression guard: it could not by
    itself catch a future bug that scaled, say, only ``variances[1]``
    differently from ``variances[2]``/``variances[3]`` as long as their mean
    still landed near 2.0.  Same sampling budget/tolerance as before (8192
    draws, seed 0, rtol=0.08) -- per-component sampling noise at this N still
    sits several standard deviations inside that tolerance.
    """
    generator = torch.Generator().manual_seed(0)
    standard = torch.randn(8192, 9, dtype=torch.float64, generator=generator)
    effective = effective_tied_irrep_latent(standard)
    variances = effective.var(dim=0, unbiased=True)
    torch.testing.assert_close(
        variances[0],
        torch.tensor(TIED_IRREP_EFFECTIVE_VARIANCES[0], dtype=torch.float64),
        rtol=0.08,
        atol=0.0,
    )
    for component in range(1, 4):
        torch.testing.assert_close(
            variances[component],
            torch.tensor(TIED_IRREP_EFFECTIVE_VARIANCES[1], dtype=torch.float64),
            rtol=0.08,
            atol=0.0,
        )
    for component in range(4, 9):
        torch.testing.assert_close(
            variances[component],
            torch.tensor(TIED_IRREP_EFFECTIVE_VARIANCES[2], dtype=torch.float64),
            rtol=0.08,
            atol=0.0,
        )


def test_rme_draw_is_degree_tied_low_rank_and_zeros_high_l():
    mapper = _water_mapper()
    flow = _b_tied_flow(mapper)
    dim = mapper.orbpair_irreps.dim
    t_o = mapper.chemical_symbol_to_type["O"]
    data = {
        _keys.ATOM_TYPE_KEY: torch.full((64, 1), t_o, dtype=torch.long),
        _keys.BATCH_KEY: torch.zeros(64, dtype=torch.long),
        _keys.SAMPLE_UID_KEY: torch.tensor([1], dtype=torch.long),
    }
    generator = torch.Generator().manual_seed(123)
    noise = flow._tied_irrep_gaussian_prior_like(
        torch.zeros(64, dim, dtype=torch.float64),
        flow.node_sigma,
        data=data,
        label="node",
        num_graphs=1,
        generator=generator,
    )
    assert int(torch.linalg.matrix_rank(noise, tol=1e-10).item()) <= 9

    first_by_degree = {}
    for start, end, degree in flow._te_irrep_slices(dim):
        segment = noise[:, start:end]
        if degree >= 3:
            assert torch.count_nonzero(segment) == 0
            continue
        if degree not in first_by_degree:
            first_by_degree[degree] = segment
        else:
            torch.testing.assert_close(
                segment,
                first_by_degree[degree],
                rtol=0.0,
                atol=0.0,
            )


def test_partial_irrep_mask_rejects_instead_of_component_truncating():
    like = torch.zeros(1, 4, dtype=torch.float64)
    slices = ((0, 1, 0), (1, 4, 1))
    mask = torch.ones_like(like, dtype=torch.bool)
    mask[0, 2] = False
    latent = torch.zeros(1, 9, dtype=torch.float64)
    with pytest.raises(ValueError, match="whole irrep"):
        fill_tied_irrep_rme(like, slices, mask, latent, sigma=1.0)


def test_seeded_draw_is_batch_composition_invariant_per_uid():
    mapper = _mapper()
    flow = _b_tied_flow(mapper)
    data_a, h0_a, _ = _b_record(mapper, seed=0)
    node_base, _ = flow.block_codec.blocks_to_rme(copy.deepcopy(data_a), h0_a)
    dim = int(node_base.shape[-1])
    device = node_base.device
    t_h = mapper.chemical_symbol_to_type["H"]
    t_c = mapper.chemical_symbol_to_type["C"]
    uid_a, uid_b, uid_c = 11, 22, 33

    a_alone = _node_tied_draw(
        flow,
        dim,
        types=[t_h, t_c],
        batch=[0, 0],
        uids=[uid_a],
        seed=_TIED_SEED,
        device=device,
    )
    ab = _node_tied_draw(
        flow,
        dim,
        types=[t_h, t_c, t_h, t_c],
        batch=[0, 0, 1, 1],
        uids=[uid_a, uid_b],
        seed=_TIED_SEED,
        device=device,
    )
    ba = _node_tied_draw(
        flow,
        dim,
        types=[t_h, t_c, t_h, t_c],
        batch=[0, 0, 1, 1],
        uids=[uid_b, uid_a],
        seed=_TIED_SEED,
        device=device,
    )
    assert torch.equal(ab[0:2], a_alone)
    assert torch.equal(ba[2:4], a_alone)
    assert torch.count_nonzero(a_alone) > 0

    c_alone = _node_tied_draw(
        flow,
        dim,
        types=[t_h, t_c],
        batch=[0, 0],
        uids=[uid_c],
        seed=_TIED_SEED,
        device=device,
    )
    assert not torch.equal(c_alone, a_alone)

    no_uid = {
        _keys.ATOM_TYPE_KEY: torch.tensor([[t_h], [t_c]], dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor([0, 0], dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0], [0]], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.tensor([[0]], dtype=torch.long),
    }
    with pytest.raises(ValueError, match=_keys.SAMPLE_UID_KEY):
        flow._tied_irrep_gaussian_prior_like(
            torch.zeros(2, dim, dtype=torch.float64),
            flow.node_sigma,
            data=no_uid,
            label="node",
            num_graphs=1,
            generator=flow._seeded_generator(device, _TIED_SEED),
        )


def test_seeded_node_and_edge_tied_draws_are_independent_substreams():
    """PR#31 review nit-5 / section 3a: node and edge draws for the SAME
    graph must come from independent substreams, not accidentally alias each
    other. ``_residual_tied_irrep_gaussian_eps`` calls
    ``_tied_irrep_gaussian_prior_like`` twice with the SAME ``generator``
    object (once ``label="node"``, once ``label="edge"``); the two only stay
    distinct because ``_prior_uid_substream_seed`` XORs in a distinct
    per-component constant before deriving each substream. This is
    pre-existing, ``projected_te``-shared infrastructure PR#31 reuses
    unmodified (review confirmed it sound by reading the source), but
    ``test_seeded_draw_is_batch_composition_invariant_per_uid`` above only
    ever exercises node draws for ``tied_irrep_gaussian`` -- nothing asserted
    node != edge specifically, so a future refactor that accidentally
    aliased the two component streams would pass silently.
    """
    mapper = _mapper()
    flow = _b_tied_flow(mapper)
    data, h0, _d1 = _b_record(mapper)
    node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)

    generator = flow._seeded_generator(node_base.device, _TIED_SEED)
    node_noise = flow._tied_irrep_gaussian_prior_like(
        torch.zeros_like(node_base),
        flow.node_sigma,
        data=data,
        label="node",
        num_graphs=1,
        generator=generator,
    )
    edge_noise = flow._tied_irrep_gaussian_prior_like(
        torch.zeros_like(edge_base),
        flow.edge_sigma,
        data=data,
        label="edge",
        num_graphs=1,
        generator=generator,
    )
    assert torch.count_nonzero(node_noise) > 0
    assert torch.count_nonzero(edge_noise) > 0
    # Same shape here (this graph has 2 nodes and 2 directed edges) is what
    # makes the elementwise inequality below meaningful rather than vacuous.
    assert node_noise.shape == edge_noise.shape
    assert not torch.equal(node_noise, edge_noise)


def test_prepare_batch_and_sampler_share_tied_prior_d0_law():
    mapper = _mapper()
    data, h0, d1 = _b_record(mapper)
    flow = _b_tied_flow(mapper)
    node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)
    eps = flow._residual_tied_irrep_gaussian_eps(
        copy.deepcopy(data),
        node_base,
        edge_base,
        generator=flow._seeded_generator(node_base.device, _TIED_SEED),
        certify_image=True,
    )

    model_data, _ref, ctx = flow.prepare_batch(
        copy.deepcopy(data),
        copy.deepcopy(data),
        t=torch.tensor([0.25], dtype=torch.float64),
        prior_seed=_TIED_SEED,
    )
    expected = project_block_state(
        copy.deepcopy(data),
        mapper,
        BlockTensorResult(
            0.75 * eps.node_blocks + 0.25 * d1.node_blocks,
            0.75 * eps.edge_blocks + 0.25 * d1.edge_blocks,
            d1.node_shapes,
            d1.edge_shapes,
        ),
    )
    torch.testing.assert_close(
        model_data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY],
        expected.node_blocks,
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        model_data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY],
        expected.edge_blocks,
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(ctx.node_prior, eps.node_blocks, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(ctx.edge_prior, eps.edge_blocks, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        torch.as_tensor(model_data[flow.node_h0_key]), node_base, rtol=0.0, atol=FP64_ATOL
    )

    spy = _EndpointSpy(
        [(d1.node_blocks.clone(), d1.edge_blocks.clone())],
        flow.node_h0_key,
        flow.edge_h0_key,
    )
    result = flow.sample(spy, copy.deepcopy(data), num_steps=1, prior_seed=_TIED_SEED)
    torch.testing.assert_close(spy.spatial_inputs[0][0], eps.node_blocks, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(spy.spatial_inputs[0][1], eps.edge_blocks, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        result[_keys.NODE_PRED_HAMIL_BLOCKS_KEY],
        h0.node_blocks + d1.node_blocks,
        rtol=0.0,
        atol=1e-12,
    )


def test_tied_prior_eps_satisfies_physical_projection_and_codec_contract():
    mapper = _mapper()
    data, h0, _ = _b_record(mapper)
    flow = _b_tied_flow(mapper)
    node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)
    eps = flow._residual_tied_irrep_gaussian_eps(
        copy.deepcopy(data),
        node_base,
        edge_base,
        generator=flow._seeded_generator(node_base.device, _TIED_SEED),
        certify_image=True,
    )

    projected = project_block_state(copy.deepcopy(data), mapper, eps)
    torch.testing.assert_close(projected.node_blocks, eps.node_blocks, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(projected.edge_blocks, eps.edge_blocks, rtol=0.0, atol=1e-12)

    torch.testing.assert_close(
        eps.node_blocks,
        eps.node_blocks.transpose(-1, -2),
        rtol=0.0,
        atol=1e-12,
    )
    rev = strict_reverse_edge_index(copy.deepcopy(data), idp=mapper)
    torch.testing.assert_close(
        eps.edge_blocks,
        eps.edge_blocks.index_select(0, rev).transpose(-1, -2),
        rtol=0.0,
        atol=1e-12,
    )

    node_mask = block_mask_from_shapes(eps.node_shapes, tuple(eps.node_blocks.shape[-2:]))
    edge_mask = block_mask_from_shapes(eps.edge_shapes, tuple(eps.edge_blocks.shape[-2:]))
    assert torch.count_nonzero(eps.node_blocks.masked_fill(node_mask, 0.0)) == 0
    assert torch.count_nonzero(eps.edge_blocks.masked_fill(edge_mask, 0.0)) == 0

    node_rme, edge_rme = flow.block_codec.blocks_to_rme(copy.deepcopy(data), eps)
    roundtrip = flow.block_codec.rme_to_blocks(
        copy.deepcopy(data), node_rme, edge_rme, project=True
    )
    torch.testing.assert_close(roundtrip.node_blocks, eps.node_blocks, rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(roundtrip.edge_blocks, eps.edge_blocks, rtol=0.0, atol=FP64_ATOL)


def _certified_tied_latent(flow, data, h0, seed=_TIED_SEED):
    """A valid transformable latent: the seeded tied_irrep_gaussian eps (codec-image).

    Mirrors ``test_residual_ao_block_ode._certified_latent`` with
    ``flow._residual_te_eps`` swapped for ``flow._residual_tied_irrep_gaussian_eps``
    -- the two share an identical signature (PR#31 review section 3a/3d), so
    this is otherwise a verbatim copy.
    """
    node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)
    return flow._residual_tied_irrep_gaussian_eps(
        copy.deepcopy(data),
        node_base,
        edge_base,
        generator=flow._seeded_generator(node_base.device, seed),
        certify_image=True,
    )


def test_h1_tied_irrep_prior_state_is_pathwise_equivariant_while_seeded_is_layout_replay():
    """PR#31 review P1-2: no SO(3) rotation-equivariance test existed for
    ``tied_irrep_gaussian``, despite the ready-to-reuse pathwise-equivariance
    harness ``test_residual_ao_block_ode.py`` built for the sibling
    ``projected_te`` prior.  This is that harness ported near-verbatim (only
    the flow builder and the eps-drawing call change), closing the gap: the
    explicit ``prior_state`` latent IS pathwise equivariant under simultaneous
    input rotation, while the SEEDED per-uid draw is only layout-replay (same
    numeric draw regardless of structure orientation, since it is keyed by
    ``sample_uid`` rather than by geometry) -- exactly the contrast the
    projected_te version documents, now verified for a tied-irrep-sourced
    latent too instead of resting on code-reading alone (review section 2c).
    """
    mapper = _mapper()
    flow = _b_tied_flow(mapper)  # fp64 tied_irrep_gaussian B-mode flow
    data, h0, _d1 = _b_record(mapper)
    eps = _certified_tied_latent(flow, data, h0)

    base = flow.sample(
        _LinearEchoModel(0.7),
        copy.deepcopy(data),
        num_steps=1,
        prior_state=BlockTensorResult(
            eps.node_blocks.clone(), eps.edge_blocks.clone(), eps.node_shapes, eps.edge_shapes
        ),
    )
    base_node = base[_keys.NODE_PRED_HAMIL_BLOCKS_KEY].clone()
    base_edge = base[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY].clone()

    # e3nn's D_from_matrix routes angle intermediates through the default dtype, so
    # fp64 covariance requires a fp64 default (mirrors the section-1 covariance tests).
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        rotation = o3.rand_matrix(dtype=torch.float64)  # proper SO(3)
        d_ao = _shared_canvas_wigner_d(rotation)

        rotated = copy.deepcopy(data)
        rotated["pos"] = data["pos"] @ rotation.transpose(-1, -2)
        rotated[_keys.NODE_H0_BLOCKS_KEY] = _rotate_canvas_blocks(h0.node_blocks, d_ao)
        rotated[_keys.EDGE_H0_BLOCKS_KEY] = _rotate_canvas_blocks(h0.edge_blocks, d_ao)
        rotated_latent = BlockTensorResult(
            _rotate_canvas_blocks(eps.node_blocks, d_ao),
            _rotate_canvas_blocks(eps.edge_blocks, d_ao),
            eps.node_shapes,
            eps.edge_shapes,
        )

        rot = flow.sample(
            _LinearEchoModel(0.7), copy.deepcopy(rotated), num_steps=1, prior_state=rotated_latent
        )

        atol = flow.block_inverse_atol * 10.0
        torch.testing.assert_close(
            rot[_keys.NODE_PRED_HAMIL_BLOCKS_KEY],
            _rotate_canvas_blocks(base_node, d_ao),
            rtol=0.0,
            atol=atol,
        )
        torch.testing.assert_close(
            rot[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY],
            _rotate_canvas_blocks(base_edge, d_ao),
            rtol=0.0,
            atol=atol,
        )

        # CONTRAST: seeded draws are layout-replay only -- the per-uid eps is the
        # SAME block draw for x and R.x (same shapes, same sample_uid), so it is NOT
        # rotated with the input and the seeded output breaks pathwise covariance.
        seed_base = flow.sample(
            _LinearEchoModel(0.7), copy.deepcopy(data), num_steps=1, prior_seed=_TIED_SEED
        )
        seed_rot = flow.sample(
            _LinearEchoModel(0.7), copy.deepcopy(rotated), num_steps=1, prior_seed=_TIED_SEED
        )
        assert not torch.allclose(
            seed_rot[_keys.NODE_PRED_HAMIL_BLOCKS_KEY],
            _rotate_canvas_blocks(seed_base[_keys.NODE_PRED_HAMIL_BLOCKS_KEY], d_ao),
            atol=1e-6,
        )
    finally:
        torch.set_default_dtype(previous_default)


def test_tied_prior_constructor_rejects_unsupported_contracts():
    mapper = _mapper()
    with pytest.raises(ValueError, match="tied_irrep_mode"):
        _b_flow(
            mapper,
            prior="tied_irrep_gaussian",
            tied_irrep_validation_seed=_TIED_SEED,
        )
    with pytest.raises(ValueError, match="tied_irrep_mode"):
        _b_tied_flow(mapper, tied_irrep_mode="split_parity")
    with pytest.raises(ValueError, match="tied_irrep_irreps"):
        _b_tied_flow(mapper, tied_irrep_irreps="1x0e")
    with pytest.raises(ValueError, match="tied_irrep_validation_seed"):
        _b_tied_flow(mapper, tied_irrep_validation_seed=True)
    with pytest.raises(ValueError, match="finite positive scales"):
        _b_tied_flow(mapper, tied_irrep_sigma=0.0)


def _load_tied_config():
    path = Path("configs") / "h_b0_block_ode_water_residual_tied_irrep.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_tied_prior_config_validates_and_schema_strict_loads():
    cfg = _load_tied_config()
    assert validate_block_ode_contract(cfg) is None

    flow = dict(cfg["train_options"]["flow_options"])
    value = flow_options().normalize_value(flow)
    flow_options().check_value(value, strict=True)
    assert value["prior"] == "tied_irrep_gaussian"
    assert value["tied_irrep_mode"] == "so3_tied"
    assert value["tied_irrep_validation_seed"] == _TIED_SEED

    bad = copy.deepcopy(cfg)
    del bad["train_options"]["flow_options"]["tied_irrep_validation_seed"]
    with pytest.raises(ValueError, match="tied_irrep_validation_seed"):
        validate_block_ode_contract(bad)

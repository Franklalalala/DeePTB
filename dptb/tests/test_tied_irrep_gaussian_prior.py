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
    see ``test_dense_all_one_expansion_matches_production_codec_on_water_oxygen_row``
    just below for the independent cross-check against the real production
    codec that these numbers were verified against (bit-exact to fp64).
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
    assert int(torch.linalg.matrix_rank(block, tol=1e-12).item()) == 9
    sym = 0.5 * (block + block.T)
    assert int(torch.linalg.matrix_rank(sym, tol=1e-12).item()) == 9


def test_dense_all_one_expansion_matches_production_codec_on_water_oxygen_row():
    """PR#31 review P1-1 fast-vs-dense cross-check (the gap review found: the
    two "halves" of this prior's math -- ``fill_tied_irrep_rme`` (production)
    and ``dense_all_one_irrep_expansion`` (docs'/tests' reference) -- were
    never cross-checked against each other anywhere).

    Feeds the doc's own Section-4 ``z`` vector through BOTH paths on water's
    real oxygen ``3s2p1d`` basis (literally the canonical ``3x0e+2x1e+1x2e``
    shape): the dense reference directly, and the production path via
    ``fill_tied_irrep_rme`` -> ``flow.block_codec.rme_to_blocks`` (i.e. the
    real ``E3Hamiltonian``).  Asserts ``allclose`` to fp64 precision for every
    block REVIEW_PR31.md's ratio table checked (s-s, s1-p1, s1-d, and the raw
    p1-p1 mix of L=0/1/2): before the sqrt(2L+1) fix these ratios were
    1/sqrt(3)/sqrt(5)/non-scalar; after it, all four are bit-exact.

    Scope note: this intentionally does NOT assert full-matrix equality.
    ``dense_all_one_irrep_expansion`` independently recomputes the "mirror"
    direction of an off-diagonal multi-copy or cross-degree shell pair (e.g.
    the second p-shell against the first, or the d-shell against a p-shell),
    while production instead derives that direction by a literal transpose of
    the canonical (ascending shell-index) block. Swapping a Wigner-3j's first
    two arguments equals the transpose only when ``l_out1+l_out2+l_in`` is
    even (verified directly against ``o3.wigner_3j``); for an odd-parity
    coupled degree (e.g. p-p's L=1, or p-d's L=2) dense's independent
    recompute and production's transpose disagree in sign on that channel.
    This is a real, narrower, separate gap from P1-1 -- out of this task's
    scope (P1-1 is specifically the missing sqrt(2L+1) factor) -- and is
    never exercised by the single-copy-target blocks checked here or by any
    block the docs' worked example shows.
    """
    mapper = _water_mapper()
    flow = _b_tied_flow(mapper)
    data = _water_graph(mapper, dtype=torch.float64)
    dim = mapper.orbpair_irreps.dim

    z = torch.tensor(
        [
            0.10, -0.20, 0.30,
            0.01, 0.02, 0.03,
            -0.04, 0.05, -0.06,
            0.07, -0.08, 0.09, -0.10, 0.11,
        ],
        dtype=torch.float64,
    )
    g0 = z[0] + z[1] + z[2]
    g1 = z[3:6] + z[6:9]
    g2 = z[9:14]

    # _water_graph's atomic_numbers=[8,1,1] -> row 0 is the O node; only that
    # row's mask is set so the H rows stay exactly zero (irrelevant here).
    node_like = torch.zeros(3, dim, dtype=torch.float64)
    node_mask = torch.zeros_like(node_like, dtype=torch.bool)
    node_mask[0, :] = True
    effective_latent = torch.zeros(3, 9, dtype=torch.float64)
    effective_latent[0] = torch.cat([g0.reshape(1), g1, g2])

    slices = flow._te_irrep_slices(dim)
    node_rme = fill_tied_irrep_rme(node_like, slices, node_mask, effective_latent, sigma=1.0)
    n_edges = int(data[_keys.EDGE_INDEX_KEY].shape[1])
    edge_rme = torch.zeros(n_edges, dim, dtype=torch.float64)

    packed = flow.block_codec.rme_to_blocks(data, node_rme, edge_rme, project=False)
    production = packed.node_blocks[0]
    dense = dense_all_one_irrep_expansion(z)

    torch.testing.assert_close(production[:3, :3], dense[:3, :3], rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(production[0:1, 3:6], dense[0:1, 3:6], rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(production[0:1, 9:14], dense[0:1, 9:14], rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(production[3:6, 3:6], dense[3:6, 3:6], rtol=0.0, atol=FP64_ATOL)


def test_effective_latent_variances_match_multiplicity_sums():
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
    torch.testing.assert_close(
        variances[1:4].mean(),
        torch.tensor(TIED_IRREP_EFFECTIVE_VARIANCES[1], dtype=torch.float64),
        rtol=0.08,
        atol=0.0,
    )
    torch.testing.assert_close(
        variances[4:9].mean(),
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

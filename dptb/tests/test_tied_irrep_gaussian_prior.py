from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

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
    _b_flow,
    _b_record,
    _mapper,
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


def test_dense_all_one_expansion_fixed_vector_fixture():
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
                [0.0643, -0.0375, -0.0064],
                [-0.0131, 0.1483, -0.0439],
                [0.0507, -0.0194, 0.1338],
            ],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=5e-5,
    )


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


def test_rme_draw_is_rowwise_degree_tied_and_zeros_high_l():
    mapper = _water_mapper()
    flow = _b_tied_flow(mapper)
    dim = mapper.orbpair_irreps.dim
    t_o = mapper.chemical_symbol_to_type["O"]
    data = {
        _keys.ATOM_TYPE_KEY: torch.full((64, 1), t_o, dtype=torch.long),
        _keys.BATCH_KEY: torch.zeros(64, dtype=torch.long),
    }
    noise = flow._tied_irrep_gaussian_prior_like(
        torch.zeros(64, dim, dtype=torch.float64),
        flow.node_sigma,
        data=data,
        label="node",
        generator=torch.Generator().manual_seed(123),
    )
    assert int(torch.linalg.matrix_rank(noise, tol=1e-10).item()) <= 9
    assert not torch.equal(noise[0], noise[1])

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
    torch.testing.assert_close(ctx.node_prior, eps.node_blocks, rtol=0.0, atol=1e-12)

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


def test_tied_prior_overlay_validates_and_flow_schema_strict_loads():
    path = Path("configs") / "h_b0_block_ode_water_residual_tied_irrep.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate_block_ode_contract(cfg) is None

    flow = dict(cfg["train_options"]["flow_options"])
    value = flow_options().normalize_value(flow)
    flow_options().check_value(value, strict=True)
    assert value["prior"] == "tied_irrep_gaussian"
    assert value["tied_irrep_mode"] == "so3_tied"
    assert value["tied_irrep_validation_seed"] == _TIED_SEED

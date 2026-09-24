"""Residual block-ODE routes: the non-SOC direct residual (``residual_ao_block_ode``) and the compact
uu-real residual (``uureal_block_ode``) -- projectors, prepare_batch bridges, rollouts and priors."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from e3nn import o3

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_mask_from_shapes,
    infer_block_shapes,
    mapper_max_norb,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.lem_moe_v3_h0_helpers import (
    DirectSpatialResidualBlockProjector,
    DirectUuRealBlockProjector,
)
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.nnops.block_flow_codec import project_block_state
from dptb.tests.block_ode_fixtures import (
    FP64_ATOL,
    UUREAL_STATE_KEYS,
    _TE_SEED,
    _EndpointSpy,
    _LinearEchoModel,
    _b_flow,
    _b_record,
    _b_te_flow,
    _fresh,
    _mapper,
    _projected_state,
    _rotate_canvas_blocks,
    _shared_canvas_wigner_d,
    _uureal_flow,
    _uureal_mapper,
    _uureal_record,
    _water_graph,
    _water_mapper,
)

NODE_PRED = _keys.NODE_PRED_HAMIL_BLOCKS_KEY
EDGE_PRED = _keys.EDGE_PRED_HAMIL_BLOCKS_KEY
T05_32 = torch.tensor([0.5], dtype=torch.float32)

_PROJECTOR_CLASS = {"spatial": DirectSpatialResidualBlockProjector, "uureal": DirectUuRealBlockProjector}
_PROJECTOR_KEYS = {
    "spatial": (
        _keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY,
        _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY,
        _keys.NODE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY,
        _keys.EDGE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY,
    ),
    "uureal": (
        _keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY,
        _keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY,
        _keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
        _keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
    ),
}


@contextmanager
def _default_float64():
    """e3nn's D_from_matrix computes its angles in the default dtype; fp64 covariance needs fp64 there."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _rotation():
    """A fixed generic proper rotation."""
    angles = torch.tensor([0.7, 1.1, -0.4], dtype=torch.float64)
    return o3.angles_to_matrix(*angles)


def _projector_mapper(kind, basis):
    if kind == "uureal":
        return _uureal_mapper(basis)
    mapper = OrbitalMapper(basis, method="e3tb")
    mapper.get_irreps()
    return mapper


def _projector(kind, mapper, dtype=torch.float64):
    return _PROJECTOR_CLASS[kind](mapper, mapper.get_irreps().sort()[0].simplify(), dtype=dtype, device="cpu")


def _with_state(data, kind, node, edge, node_shapes, edge_shapes):
    """Attach a residual block state under the projector's input keys."""
    out = dict(data)
    out.update(zip(_PROJECTOR_KEYS[kind], (node, edge, node_shapes, edge_shapes)))
    return out


def _certified_latent(flow, data, h0, seed=_TE_SEED):
    """The seeded projected_te residual draw for this record (a codec-image latent)."""
    node_base, edge_base = flow.block_codec.blocks_to_rme(_fresh(data), h0)
    return flow._residual_te_eps(
        _fresh(data),
        node_base,
        edge_base,
        generator=flow._seeded_generator(node_base.device, seed),
        certify_image=True,
    )


# ---------------------------------------------------------------------------
# Residual block projectors (spatial = non-SOC, uureal = compact uu-real)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["spatial", "uureal"])
def test_projector_contraction_inverts_the_e3hamiltonian_forward_cg(kind):
    """Expanding random coupled RME with E3Hamiltonian's p-p CG basis and contracting the AO block
    recovers the RME (in the projector's sorted-irrep order): a genuine CG decomposition."""
    mapper = _projector_mapper(kind, {"C": "1p"})
    projector = _projector(kind, mapper)
    oracle = E3Hamiltonian(idp=OrbitalMapper({"C": "1p"}, method="e3tb"), decompose=False, dtype=torch.float64)
    cg = oracle.cgbasis["p-p"].to(torch.float64)
    rme = torch.randn(int(mapper.reduced_matrix_element), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    p_slice = mapper.orbital_maps["C"]["1p"]
    block = torch.zeros(1, projector.canvas, projector.canvas, dtype=torch.float64)
    block[0, p_slice, p_slice] = torch.einsum("ijr,r->ij", cg, rme)
    atom_types = torch.tensor([[mapper.chemical_symbol_to_type["C"]]])
    recovered = projector._contract(block, atom_types, projector.node_plan)[0]
    torch.testing.assert_close(recovered, rme.index_select(0, projector.sort_index), rtol=0.0, atol=FP64_ATOL)


@pytest.mark.parametrize("kind", ["spatial", "uureal"])
def test_projector_is_rotation_covariant_on_the_nested_water_canvas(kind):
    """contract(D B D^T) == contract(B) D_irreps^T for onsite and hetero-edge blocks, with cross-n shell
    pairs and H's species-compact frame (p shell at compact slots 2-4) nested in O's 14-wide canvas."""
    mapper = _projector_mapper(kind, {"H": "2s1p", "O": "3s2p1d"})
    irreps_in = mapper.get_irreps().sort()[0].simplify()
    projector = _projector(kind, mapper)
    canvas = projector.canvas
    assert canvas == 14
    t_o, t_h = mapper.chemical_symbol_to_type["O"], mapper.chemical_symbol_to_type["H"]
    atom_types = torch.tensor([[t_o], [t_h]])
    node_shapes = torch.tensor([[14, 14], [5, 5]])
    bond_types = torch.tensor([[mapper.bond_to_type["H-O"]], [mapper.bond_to_type["O-H"]]])
    edge_shapes = torch.tensor([[5, 14], [14, 5]])
    with _default_float64():
        rotation = _rotation()
        d_o = o3.Irreps("3x0e+2x1o+1x2e").D_from_matrix(rotation)
        d_h = torch.eye(canvas, dtype=torch.float64)
        d_h[:5, :5] = o3.Irreps("2x0e+1x1o").D_from_matrix(rotation)
        d_irreps = irreps_in.D_from_matrix(rotation)
    generator = torch.Generator().manual_seed(2)
    raw = torch.randn(2, canvas, canvas, dtype=torch.float64, generator=generator)
    onsite = 0.5 * (raw + raw.transpose(-1, -2)) * block_mask_from_shapes(node_shapes, (canvas, canvas))
    edges = torch.randn(2, canvas, canvas, dtype=torch.float64, generator=generator)
    edges = edges * block_mask_from_shapes(edge_shapes, (canvas, canvas))
    rotated_onsite = torch.stack((d_o @ onsite[0] @ d_o.T, d_h @ onsite[1] @ d_h.T))
    rotated_edges = torch.stack((d_h @ edges[0] @ d_o.T, d_o @ edges[1] @ d_h.T))
    for blocks, rotated, types, plan in (
        (onsite, rotated_onsite, atom_types, projector.node_plan),
        (edges, rotated_edges, bond_types, projector.edge_plan),
    ):
        lhs = projector._contract(rotated, types, plan)
        rhs = projector._contract(blocks, types, plan) @ d_irreps.T
        torch.testing.assert_close(lhs, rhs, rtol=0.0, atol=FP64_ATOL)


@pytest.mark.parametrize("kind", ["spatial", "uureal"])
def test_projector_maps_a_zero_residual_to_exactly_zero_hidden(kind):
    """A zero residual gives bit-exact zero hidden features while the packed residual does not."""
    if kind == "spatial":
        mapper = _mapper()
        data, _, d1 = _b_record(mapper)
        node, edge, shapes, dtype = d1.node_blocks, d1.edge_blocks, (d1.node_shapes, d1.edge_shapes), torch.float64
    else:
        mapper = _uureal_mapper()
        data = _uureal_record(mapper)
        node, edge = data[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY], data[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY]
        shapes = (data[_keys.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY], data[_keys.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY])
        dtype = torch.float32
    projector = _projector(kind, mapper, dtype)
    edges = torch.arange(data["edge_index"].shape[1])
    live = projector(_with_state(data, kind, node, edge, *shapes), data["atom_types"], data["edge_type"], edges)
    assert all(torch.count_nonzero(hidden) > 0 for hidden in live)
    zero_state = _with_state(data, kind, torch.zeros_like(node), torch.zeros_like(edge), *shapes)
    for hidden in projector(zero_state, data["atom_types"], data["edge_type"], edges):
        assert torch.equal(hidden, torch.zeros_like(hidden))


def test_spatial_projector_node_and_edge_paths_are_independent():
    mapper = _mapper()
    data, _, d1 = _b_record(mapper)
    projector = _projector("spatial", mapper)
    shapes = (d1.node_shapes, d1.edge_shapes)
    for zero_node in (True, False):
        node = torch.zeros_like(d1.node_blocks) if zero_node else d1.node_blocks
        edge = d1.edge_blocks if zero_node else torch.zeros_like(d1.edge_blocks)
        state = _with_state(data, "spatial", node, edge, *shapes)
        node_hidden, edge_hidden = projector(state, data["atom_types"], data["edge_type"], torch.arange(2))
        zero_hidden, live_hidden = (node_hidden, edge_hidden) if zero_node else (edge_hidden, node_hidden)
        assert torch.equal(zero_hidden, torch.zeros_like(zero_hidden))
        assert torch.count_nonzero(live_hidden) > 0


def test_spatial_projector_input_gates_on_the_nested_water_canvas():
    """Zero in -> zero out on the nested canvas; a wrong canvas or wrong stored shapes raise ValueError,
    missing state keys raise KeyError."""
    mapper = _water_mapper()
    data = _water_graph(mapper)
    node_shapes, edge_shapes = infer_block_shapes(data, mapper)
    canvas = mapper_max_norb(mapper)
    n, e = int(node_shapes.shape[0]), int(edge_shapes.shape[0])
    projector = _projector("spatial", mapper)

    def run(batch):
        return projector(batch, data["atom_types"], data["edge_type"], torch.arange(e))

    def zeros(width):
        return torch.zeros(n, width, width, dtype=torch.float64), torch.zeros(e, width, width, dtype=torch.float64)

    for hidden in run(_with_state(data, "spatial", *zeros(canvas), node_shapes, edge_shapes)):
        assert torch.equal(hidden, torch.zeros_like(hidden))
    live = _projected_state(mapper, data, canvas=canvas, n=n, e=e, dtype=torch.float64, seed=7)
    assert torch.count_nonzero(run(_with_state(data, "spatial", live.node_blocks, live.edge_blocks, node_shapes, edge_shapes))[0]) > 0
    with pytest.raises(ValueError):
        run(_with_state(data, "spatial", *zeros(canvas + 1), node_shapes, edge_shapes))
    with pytest.raises(ValueError):
        run(_with_state(data, "spatial", live.node_blocks, live.edge_blocks, node_shapes + 1, edge_shapes))
    with pytest.raises(KeyError):
        run(dict(data))


def test_residual_projectors_reject_the_other_mapper_kind():
    with pytest.raises((ValueError, NotImplementedError), match="SOC"):
        _projector("spatial", _uureal_mapper({"C": "1s1p"}))
    with pytest.raises(ValueError, match="SOC"):
        _projector("uureal", _mapper())


# ---------------------------------------------------------------------------
# prepare_batch bridges and record gates
# ---------------------------------------------------------------------------
def test_residual_prepare_is_an_exact_scalar_bridge_with_constant_physical_h0():
    """Zero prior: the attached state is exactly t * D1, the H0 keys carry blocks_to_rme(H0) for every t,
    and the endpoint/H0 block side channels stay out of the model input."""
    mapper = _mapper()
    data, h0, d1 = _b_record(mapper)
    flow = _b_flow(mapper)
    model_data, _, ctx = flow.prepare_batch(_fresh(data), _fresh(data), t=torch.tensor([0.25], dtype=torch.float64))
    torch.testing.assert_close(model_data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY], 0.25 * d1.node_blocks, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(model_data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY], 0.25 * d1.edge_blocks, rtol=0.0, atol=1e-12)
    node_base, edge_base = flow.block_codec.blocks_to_rme(_fresh(data), h0)
    torch.testing.assert_close(torch.as_tensor(model_data[flow.node_h0_key]), node_base, rtol=0.0, atol=FP64_ATOL)
    torch.testing.assert_close(torch.as_tensor(model_data[flow.edge_h0_key]), edge_base, rtol=0.0, atol=FP64_ATOL)
    low, high = (
        flow.prepare_batch(_fresh(data), _fresh(data), t=torch.tensor([t], dtype=torch.float64))[0] for t in (0.1, 0.9)
    )
    for key in (flow.node_h0_key, flow.edge_h0_key):
        assert torch.equal(torch.as_tensor(low[key]), torch.as_tensor(high[key]))
    assert ctx.node_base is not None and ctx.edge_base is not None
    assert ctx.node_target is None and ctx.edge_target is None
    assert ctx.block_target_semantics == "residual_dh"
    for key in (
        flow.node_block_target_key,
        flow.edge_block_target_key,
        flow.node_h0_block_key,
        flow.edge_h0_block_key,
        flow.node_h0_block_shape_key,
        flow.edge_h0_block_shape_key,
    ):
        assert key not in model_data


@pytest.mark.parametrize("key", ["soc_uureal_compact", _keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY])
def test_residual_prepare_rejects_uureal_markers_on_a_raw_record(key):
    mapper = _mapper()
    data, _, d1 = _b_record(mapper)
    data[key] = True if key == "soc_uureal_compact" else d1.node_blocks.clone()
    with pytest.raises(ValueError, match=key):
        _b_flow(mapper).prepare_batch(_fresh(data), _fresh(data), t=torch.tensor([0.5], dtype=torch.float64))


def test_uureal_prepare_is_an_exact_scalar_bridge_without_h0_blocks():
    mapper = _uureal_mapper()
    data = _uureal_record(mapper)
    model_data, ref, ctx = _uureal_flow(mapper).prepare_batch(
        _fresh(data), _fresh(data), t=torch.tensor([0.25], dtype=torch.float32)
    )
    assert torch.equal(model_data[_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY], 0.25 * data[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY])
    assert torch.equal(model_data[_keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY], 0.25 * data[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY])
    assert _keys.NODE_H0_BLOCKS_KEY not in data and _keys.EDGE_H0_BLOCKS_KEY not in data
    assert ctx.block_target_semantics == "residual_dh"
    assert ref["blockwise_target_mode"] == "already-delta"


@pytest.mark.parametrize(("key", "bad"), [
    ("blockwise_spatial_schema", "wrong/v0"),
    ("blockwise_target_mode", "absolute"),
    ("blockwise_source_target_feature_width", 15),
    ("blockwise_source_h0_feature_width", 15),
    ("blockwise_source_h0_feature_width", "wide"),
    ("soc_uureal_compact", False),
    ("soc_uureal_full_rme", 16),
    ("soc_uureal_keep", 15),
])
def test_uureal_prepare_rejects_each_bad_metadata_field(key, bad):
    mapper = _uureal_mapper()
    data = _uureal_record(mapper)
    assert data["soc_uureal_keep"] == 16  # so a width of 15 is below keep
    data[key] = bad
    with pytest.raises(ValueError, match=key):
        _uureal_flow(mapper).prepare_batch(_fresh(data), _fresh(data), t=T05_32)


def test_uureal_prepare_accepts_converter_source_width_and_identical_collated_metadata():
    """A source width above keep (the converter records the full-SOC width) is provenance, not an error;
    collated metadata lists/tensors are accepted when identical and rejected when mixed."""
    mapper = _uureal_mapper()
    flow = _uureal_flow(mapper)
    data = _uureal_record(mapper)
    keep = data["soc_uureal_keep"]
    data["blockwise_source_target_feature_width"] = keep * 8
    data["blockwise_source_h0_feature_width"] = keep * 8
    flow.prepare_batch(_fresh(data), _fresh(data), t=T05_32)

    collated = _uureal_record(mapper)
    collated["blockwise_spatial_schema"] = ["deeptb.blockwise_spatial/v1"]
    collated["blockwise_target_mode"] = ["already-delta"]
    for key in (
        "blockwise_source_target_feature_width",
        "blockwise_source_h0_feature_width",
        "soc_uureal_compact",
        "soc_uureal_full_rme",
        "soc_uureal_keep",
    ):
        collated[key] = torch.as_tensor([collated[key]])
    flow.prepare_batch(_fresh(collated), _fresh(collated), t=T05_32)
    collated["blockwise_spatial_schema"] = ["deeptb.blockwise_spatial/v1", "wrong/v0"]
    with pytest.raises(ValueError):
        flow.prepare_batch(_fresh(collated), _fresh(collated), t=T05_32)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize("prior", ["zero", "projected_te"])
def test_residual_rollout_keeps_h0_constant_and_adds_it_exactly_once(prior, steps):
    """Every step sees the same physical-H0 RME; the zero prior starts at D=0 (projected_te at a nonzero
    draw); a constant endpoint D1 ends at H0 + D1, assembled once outside the ODE."""
    mapper = _mapper()
    data, h0, d1 = _b_record(mapper)
    flow = _b_flow(mapper) if prior == "zero" else _b_te_flow(mapper)
    seed = {} if prior == "zero" else {"prior_seed": _TE_SEED}
    node_base, edge_base = flow.block_codec.blocks_to_rme(_fresh(data), h0)
    spy = _EndpointSpy([(d1.node_blocks, d1.edge_blocks)] * steps, flow.node_h0_key, flow.edge_h0_key)
    result = flow.sample(spy, _fresh(data), num_steps=steps, **seed)

    start_node, start_edge = spy.spatial_inputs[0]
    if prior == "zero":
        assert torch.count_nonzero(start_node) == 0 and torch.count_nonzero(start_edge) == 0
        if steps >= 2:
            torch.testing.assert_close(spy.spatial_inputs[1][0], d1.node_blocks / steps, rtol=0.0, atol=1e-12)
            torch.testing.assert_close(spy.spatial_inputs[1][1], d1.edge_blocks / steps, rtol=0.0, atol=1e-12)
    else:
        assert torch.count_nonzero(start_node) > 0
    for node_h0, edge_h0 in spy.h0_inputs:
        torch.testing.assert_close(node_h0, node_base, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(edge_h0, edge_base, rtol=0.0, atol=1e-12)
    assert spy.times[0].reshape(-1)[0].item() == 0.0
    torch.testing.assert_close(result[NODE_PRED], h0.node_blocks + d1.node_blocks, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result[EDGE_PRED], h0.edge_blocks + d1.edge_blocks, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(torch.as_tensor(result[flow.node_h0_key]), node_base, rtol=0.0, atol=FP64_ATOL)
    assert torch.allclose(result[flow.flow_time_key], torch.ones_like(result[flow.flow_time_key]))


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_uureal_rollout_starts_at_zero_and_ends_on_the_last_residual_endpoint(steps):
    mapper = _uureal_mapper()
    data = _uureal_record(mapper)
    node, edge = data[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY], data[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY]
    endpoints = [(node * (i + 1), edge * (i + 1)) for i in range(steps)]
    spy = _EndpointSpy(endpoints, "node_h0", "edge_h0", state_keys=UUREAL_STATE_KEYS)
    result = _uureal_flow(mapper).sample(spy, _fresh(data), num_steps=steps)
    assert torch.count_nonzero(spy.spatial_inputs[0][0]) == 0
    assert torch.count_nonzero(spy.spatial_inputs[0][1]) == 0
    if steps >= 2:
        torch.testing.assert_close(spy.spatial_inputs[1][0], node / steps, rtol=0.0, atol=1e-8)
        torch.testing.assert_close(spy.spatial_inputs[1][1], edge / steps, rtol=0.0, atol=1e-8)
    assert torch.equal(result[NODE_PRED], endpoints[-1][0])
    assert torch.equal(result[EDGE_PRED], endpoints[-1][1])


# ---------------------------------------------------------------------------
# projected_te stochastic bridge and explicit prior_state latents
# ---------------------------------------------------------------------------
def test_projected_te_bridge_interpolates_between_the_seeded_draw_and_d1():
    """D_t = project((1 - t) eps + t D1) for the seeded draw eps (nonzero), which ctx exposes as the
    prior; the H0 RME conditioning does not depend on the prior."""
    mapper = _mapper()
    data, h0, d1 = _b_record(mapper)
    flow = _b_te_flow(mapper)
    node_base, _ = flow.block_codec.blocks_to_rme(_fresh(data), h0)
    eps = _certified_latent(flow, data, h0)
    assert torch.count_nonzero(eps.node_blocks) > 0 and torch.count_nonzero(eps.edge_blocks) > 0
    model_data, _, ctx = flow.prepare_batch(
        _fresh(data), _fresh(data), t=torch.tensor([0.25], dtype=torch.float64), prior_seed=_TE_SEED
    )
    expected = project_block_state(
        _fresh(data),
        mapper,
        BlockTensorResult(
            0.75 * eps.node_blocks + 0.25 * d1.node_blocks,
            0.75 * eps.edge_blocks + 0.25 * d1.edge_blocks,
            d1.node_shapes,
            d1.edge_shapes,
        ),
    )
    exact = dict(rtol=0.0, atol=1e-12)
    torch.testing.assert_close(model_data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY], expected.node_blocks, **exact)
    torch.testing.assert_close(model_data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY], expected.edge_blocks, **exact)
    torch.testing.assert_close(ctx.node_prior, eps.node_blocks, **exact)
    torch.testing.assert_close(ctx.edge_prior, eps.edge_blocks, **exact)
    assert ctx.block_target_semantics == "residual_dh"
    torch.testing.assert_close(torch.as_tensor(model_data[flow.node_h0_key]), node_base, rtol=0.0, atol=FP64_ATOL)


def test_projected_te_sampling_is_seed_deterministic():
    mapper = _mapper()
    data, _, d1 = _b_record(mapper)
    flow = _b_te_flow(mapper)

    def run(seed):
        spy = _EndpointSpy([(d1.node_blocks, d1.edge_blocks)], flow.node_h0_key, flow.edge_h0_key)
        return spy, flow.sample(spy, _fresh(data), num_steps=1, prior_seed=seed)

    (spy_a, result_a), (spy_b, result_b), (spy_c, _) = run(_TE_SEED), run(_TE_SEED), run(_TE_SEED + 1)
    assert torch.equal(spy_a.spatial_inputs[0][0], spy_b.spatial_inputs[0][0])
    assert torch.equal(spy_a.spatial_inputs[0][1], spy_b.spatial_inputs[0][1])
    assert torch.equal(result_a[NODE_PRED], result_b[NODE_PRED])
    assert not torch.equal(spy_a.spatial_inputs[0][0], spy_c.spatial_inputs[0][0])


def test_prior_state_latent_is_pathwise_equivariant():
    """sample(R x, prior_state=R eps) == R sample(x, prior_state=eps) for an equivariant model."""
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    data, h0, _ = _b_record(mapper)
    eps = _certified_latent(flow, data, h0)
    base = flow.sample(
        _LinearEchoModel(0.7),
        _fresh(data),
        num_steps=1,
        prior_state=BlockTensorResult(eps.node_blocks.clone(), eps.edge_blocks.clone(), eps.node_shapes, eps.edge_shapes),
    )
    with _default_float64():
        rotation = _rotation()
        d_ao = _shared_canvas_wigner_d(rotation)
    rotated = _fresh(data)
    rotated["pos"] = data["pos"] @ rotation.T
    rotated[_keys.NODE_H0_BLOCKS_KEY] = _rotate_canvas_blocks(h0.node_blocks, d_ao)
    rotated[_keys.EDGE_H0_BLOCKS_KEY] = _rotate_canvas_blocks(h0.edge_blocks, d_ao)
    latent = BlockTensorResult(
        _rotate_canvas_blocks(eps.node_blocks, d_ao), _rotate_canvas_blocks(eps.edge_blocks, d_ao),
        eps.node_shapes, eps.edge_shapes,
    )
    rot = flow.sample(_LinearEchoModel(0.7), rotated, num_steps=1, prior_state=latent)
    atol = flow.block_inverse_atol * 10.0
    torch.testing.assert_close(rot[NODE_PRED], _rotate_canvas_blocks(base[NODE_PRED], d_ao), rtol=0.0, atol=atol)
    torch.testing.assert_close(rot[EDGE_PRED], _rotate_canvas_blocks(base[EDGE_PRED], d_ao), rtol=0.0, atol=atol)


def test_any_codec_image_prior_state_is_accepted():
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    data, h0, _ = _b_record(mapper)
    eps = _certified_latent(flow, data, h0)
    other = _certified_latent(flow, data, h0, seed=_TE_SEED + 5)
    assert not torch.equal(eps.node_blocks, other.node_blocks)
    for latent in (eps, (other.node_blocks.clone(), other.edge_blocks.clone())):
        assert NODE_PRED in flow.sample(_LinearEchoModel(0.5), _fresh(data), num_steps=1, prior_state=latent)


@pytest.mark.parametrize("case", [
    "trimmed_canvas", "nan", "off_image", "state_and_seed", "zero_prior_with_state", "zero_prior_with_seed",
])
def test_prior_state_and_prior_seed_are_validated_before_use(case):
    """Latents with the wrong shape, non-finite values or outside the codec image are rejected, as are a
    latent together with a seed and any latent or seed for the exact zero prior."""
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    data, h0, _ = _b_record(mapper)
    eps = _certified_latent(flow, data, h0)
    node, edge = eps.node_blocks.clone(), eps.edge_blocks.clone()
    kwargs, offending = {"prior_state": (node, edge)}, "prior_state"
    if case == "trimmed_canvas":
        kwargs = {"prior_state": BlockTensorResult(node[:, :3, :3], edge, eps.node_shapes, eps.edge_shapes)}
    elif case == "nan":
        node[0, 0, 0] = float("nan")
    elif case == "off_image":
        node[1, 0, 1] += 3.0  # breaks the C onsite symmetry
    elif case == "state_and_seed":
        kwargs = {"prior_state": eps, "prior_seed": 3}
    elif case == "zero_prior_with_state":
        flow = _b_flow(mapper)
    else:
        flow, kwargs, offending = _b_flow(mapper), {"prior_seed": 7}, "prior_seed"
    with pytest.raises(ValueError, match=offending):
        flow.sample(_LinearEchoModel(0.5), _fresh(data), num_steps=1, **kwargs)


def _node_te_draw(flow, dim, *, types, batch, uids, seed):
    """The per-uid seeded node draw for a hand-built collated batch."""
    payload = {
        _keys.ATOM_TYPE_KEY: torch.tensor([[t] for t in types], dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor(batch, dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0], [0]], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.tensor([[0]], dtype=torch.long),
        _keys.SAMPLE_UID_KEY: torch.tensor(uids, dtype=torch.long),
    }
    return flow._te_prior_like(
        torch.zeros(len(batch), dim, dtype=torch.float64),
        flow.node_sigma,
        data=payload,
        label="node",
        reference_scale=False,
        num_graphs=len(uids),
        generator=flow._seeded_generator(torch.device("cpu"), seed),
    )


def test_seeded_prior_rows_depend_only_on_the_graph_uid():
    """A graph's seeded draw is the same alone, first or second in a batch; another uid draws differently;
    a fresh flow with the same seed replays it via its validation base seed; no uid fails closed."""
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    data, h0, _ = _b_record(mapper)
    dim = int(flow.block_codec.blocks_to_rme(_fresh(data), h0)[0].shape[-1])
    pair = [mapper.chemical_symbol_to_type["H"], mapper.chemical_symbol_to_type["C"]]

    def draw(uids, seed=_TE_SEED, owner=flow):
        batch = [graph for graph in range(len(uids)) for _ in pair]
        return _node_te_draw(owner, dim, types=pair * len(uids), batch=batch, uids=uids, seed=seed)

    alone = draw([11])
    assert torch.count_nonzero(alone) > 0
    assert torch.equal(draw([11, 22])[0:2], alone)
    assert torch.equal(draw([22, 11])[2:4], alone)
    assert not torch.equal(draw([33]), alone)

    replay = _b_te_flow(mapper)
    base = flow.validation_prior_base_seed()
    assert replay.validation_prior_base_seed() == base
    assert torch.equal(draw([42], seed=base, owner=replay), draw([42], seed=base))

    unidentified = _fresh(data)
    del unidentified[_keys.SAMPLE_UID_KEY]
    with pytest.raises(ValueError, match=_keys.SAMPLE_UID_KEY):
        flow.sample(_LinearEchoModel(0.5), unidentified, num_steps=1, prior_seed=_TE_SEED)


@pytest.mark.parametrize("draw", ["all_zero", "nan"])
def test_collapsed_or_nonfinite_prior_draws_are_rejected(draw, monkeypatch):
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    data, _, _ = _b_record(mapper)

    def fake_draw(like, *_args, **_kwargs):
        out = torch.zeros_like(like)
        if draw == "nan" and out.numel():
            out.reshape(-1)[0] = float("nan")
        return out

    monkeypatch.setattr(flow, "_te_prior_like", fake_draw)
    with pytest.raises(ValueError):
        flow.sample(_LinearEchoModel(0.5), _fresh(data), num_steps=1, prior_seed=_TE_SEED)

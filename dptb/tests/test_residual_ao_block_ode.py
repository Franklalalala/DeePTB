"""Tests for the non-SOC direct-residual block-ODE mode ``residual_ao_block_ode``.

This mirrors the mode-sibling suite ``test_uureal_block_ode.py`` (compact-uu real)
and the generic ``test_block_ode_flow.py`` full-H block-ODE, but exercises the new
plain non-SOC direct-CG-entry residual mode:

  * projector ``DirectSpatialResidualBlockProjector`` (structural sibling of
    ``DirectUuRealBlockProjector`` reusing the shared CG machinery, guarded by
    ``ensure_non_soc_mapper`` and reading the ``*_spatial_residual_blocks`` keys);
  * flow mode ``output_space='residual_ao_block_ode'`` whose rollout keeps the
    physical H0 RME constant in the h0 keys, blends a pure-D residual state, and
    assembles the final full H exactly once outside the ODE (block_codec present);
  * loader gate ``require_residual_from_full_h_target`` consuming the SAME raw
    ``absolute_full_h`` LMDB records the A arm uses (D1 = raw H - H0 materialized);
  * ``validate_block_ode_contract`` acceptance of the B arm yaml + rejection matrix.

The oracles are independent of the projection under test: E3Hamiltonian's forward
CG basis, e3nn Wigner-D covariance, and hand-written pooled masked MSE reductions.

Design reference: F:\\claude\\0719_hb0_residual\\DESIGN.md (Test plan, items 1-8).
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch
import yaml
from e3nn import o3

from dptb.data import _keys
from dptb.data.build import DatasetBuilder
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_mask_from_shapes,
    infer_block_shapes,
    mapper_max_norb,
    strict_reverse_edge_index,
)
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    H0_RESIDUAL_SEMANTICS,
    RAW_HAMILTONIAN_SAMPLE_SCHEMA,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.lem_moe_v3_h0_helpers import (
    DirectSpatialResidualBlockProjector,
    DirectUuRealBlockProjector,
)
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state
from dptb.nnops.flow import HamiltonianCFM
from dptb.utils.argcheck import validate_block_ode_contract


FP64_ATOL = 1e-10


# ---------------------------------------------------------------------------
# Fixtures: non-SOC mappers, graphs, and B-mode flow builder
# ---------------------------------------------------------------------------
def _mapper():
    """A fast two-species non-SOC mapper (canvas nesting: H 1s vs C 1s1p)."""
    mapper = OrbitalMapper({"H": "1s", "C": "1s1p"}, method="e3tb")
    mapper.get_irreps()
    return mapper


def _water_mapper():
    """Water-basis non-SOC mapper (H 2s1p=5, O 3s2p1d=14; canvas nesting)."""
    mapper = OrbitalMapper({"H": "2s1p", "O": "3s2p1d"}, method="e3tb")
    mapper.get_irreps()
    return mapper


def _single_species_mapper(basis):
    mapper = OrbitalMapper({"C": basis}, method="e3tb")
    mapper.get_irreps()
    return mapper


def _uureal_mapper(basis="1s1p"):
    mapper = OrbitalMapper(
        {"C": basis},
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=True,
        full_soc_prediction=False,
    )
    mapper.get_irreps()
    return mapper


def _nonscalar_norm(irreps: o3.Irreps, coords: torch.Tensor) -> torch.Tensor:
    """L2 norm of every non-``0e`` coordinate of a coupled-RME vector."""
    pieces = []
    offset = 0
    for mul, ir in irreps:
        width = mul * ir.dim
        if ir.l != 0:
            pieces.append(coords[offset:offset + width])
        offset += width
    if not pieces:
        return coords.new_zeros(())
    return torch.cat(pieces).norm()


def _graph(mapper, *, dtype=torch.float64):
    """A two-atom H-C molecule (pbc FFF, reverse edge pair)."""
    t_h = mapper.chemical_symbol_to_type["H"]
    t_c = mapper.chemical_symbol_to_type["C"]
    return {
        "atomic_numbers": torch.tensor([1, 6]),
        "atom_types": torch.tensor([[t_h], [t_c]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.zeros(2, 3, dtype=dtype),
        "edge_type": torch.tensor(
            [[mapper.bond_to_type["H-C"]], [mapper.bond_to_type["C-H"]]]
        ),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype),
        "cell": torch.eye(3, dtype=dtype) * 8.0,
        "pbc": torch.tensor([False, False, False]),
        "batch": torch.zeros(2, dtype=torch.long),
        # Seeded projected_te draws require the stable per-graph record identity.
        _keys.SAMPLE_UID_KEY: torch.tensor([1], dtype=torch.long),
    }


def _water_graph(mapper, *, dtype=torch.float64):
    """A three-atom O-H-H molecule exercising the 14-wide O canvas."""
    t_o = mapper.chemical_symbol_to_type["O"]
    t_h = mapper.chemical_symbol_to_type["H"]
    return {
        "atomic_numbers": torch.tensor([8, 1, 1]),
        "atom_types": torch.tensor([[t_o], [t_h], [t_h]]),
        "edge_index": torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long),
        "edge_cell_shift": torch.zeros(4, 3, dtype=dtype),
        "edge_type": torch.tensor(
            [
                [mapper.bond_to_type["O-H"]],
                [mapper.bond_to_type["H-O"]],
                [mapper.bond_to_type["O-H"]],
                [mapper.bond_to_type["H-O"]],
            ]
        ),
        "pos": torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.3, 0.7, 0.0]], dtype=dtype
        ),
        "cell": torch.eye(3, dtype=dtype) * 10.0,
        "pbc": torch.tensor([False, False, False]),
        "batch": torch.zeros(3, dtype=torch.long),
        # Seeded projected_te draws require the stable per-graph record identity.
        _keys.SAMPLE_UID_KEY: torch.tensor([1], dtype=torch.long),
    }


def _projected_state(mapper, data, *, canvas, n, e, dtype, seed):
    """Draw a projector-invariant (packer-image) block state for the graph."""
    generator = torch.Generator().manual_seed(seed)
    return project_block_state(
        data,
        mapper,
        BlockTensorResult(
            torch.randn(n, canvas, canvas, generator=generator, dtype=dtype),
            torch.randn(e, canvas, canvas, generator=generator, dtype=dtype),
            *infer_block_shapes(data, mapper),
        ),
    )


def _b_record(mapper, *, dtype=torch.float64, seed=0):
    """A B-mode record: physical H0 blocks + residual D1 endpoint blocks.

    Returns ``(data, h0_blocks, d1_blocks)`` where the H0 and D1 blocks are both
    projector-invariant (already inside the canonical packer image) so the loader
    contract and the exact scalar/interpolation bridges hold to fp precision.
    """
    data = _graph(mapper, dtype=dtype)
    node_shapes, edge_shapes = infer_block_shapes(data, mapper)
    canvas = mapper_max_norb(mapper)
    n = int(node_shapes.shape[0])
    e = int(edge_shapes.shape[0])
    h0 = _projected_state(mapper, data, canvas=canvas, n=n, e=e, dtype=dtype, seed=seed)
    d1 = _projected_state(
        mapper, data, canvas=canvas, n=n, e=e, dtype=dtype, seed=seed + 1000
    )
    data[_keys.NODE_H0_BLOCKS_KEY] = h0.node_blocks.clone()
    data[_keys.EDGE_H0_BLOCKS_KEY] = h0.edge_blocks.clone()
    data[_keys.NODE_H0_BLOCK_SHAPE_KEY] = h0.node_shapes.clone()
    data[_keys.EDGE_H0_BLOCK_SHAPE_KEY] = h0.edge_shapes.clone()
    data[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY] = d1.node_blocks.clone()
    data[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY] = d1.edge_blocks.clone()
    data[_keys.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = d1.node_shapes.clone()
    data[_keys.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = d1.edge_shapes.clone()
    return data, h0, d1


def _b_flow(mapper, *, dtype=torch.float64, **overrides):
    """Build a ``residual_ao_block_ode`` flow from the B arm's flow_options."""
    options = {
        "enabled": True,
        "objective": "cfm",
        "mode": "residual",
        "prior": "zero",
        "output_space": "residual_ao_block_ode",
        "block_ode": True,
        "state_space": "residual_ao_block",
        "block_input_adapter": "direct_cg",
        "h0_condition_space": "spatial_h0_rme",
        "block_export_final_full_h": True,
        "target_semantics": "residual_dh",
        "prediction_add_h0": False,
        "time_conditioning_required": True,
        "strict_h0": True,
        "t0_probability": 0.15,
        "block_inverse_mode": "strict",
        "block_inverse_atol": 1e-10 if dtype == torch.float64 else 2e-5,
        "strict_certification": "always",
        "node_block_target_key": "node_delta_hamil_blocks",
        "edge_block_target_key": "edge_delta_hamil_blocks",
        "node_block_shape_key": "node_delta_hamil_block_shape",
        "edge_block_shape_key": "edge_delta_hamil_block_shape",
        "validation_ode_steps": [1],
    }
    options.update(overrides)
    return HamiltonianCFM(options, idp=mapper, dtype=dtype)


_TE_SEED = 20260720


def _te_options(seed=_TE_SEED):
    """Full projected_te option set for the residual stochastic-bridge arm."""
    return {
        "prior": "projected_te",
        "te_prior_mode": "irrep",
        "node_sigma": 1.0,
        "edge_sigma": 1.0,
        "te_prior_sigma": 1.0,
        "te_prior_validation_seed": seed,
    }


def _b_te_flow(mapper, *, dtype=torch.float64, seed=_TE_SEED, **overrides):
    """Build a projected_te ``residual_ao_block_ode`` flow (B-te arm, v2)."""
    options = _te_options(seed)
    options.update(overrides)
    return _b_flow(mapper, dtype=dtype, **options)


# ===========================================================================
# 1. CG oracles on the new projector
# ===========================================================================
def test_projector_inverts_e3hamiltonian_forward_cgbasis():
    """1a: the AO-block contraction inverts E3Hamiltonian's forward p-p CG.

    Independent oracle: E3Hamiltonian owns the production RME->block forward CG
    (``cgbasis[pairtype] = wigner_3j(l1,l2,L)*sqrt(2L+1)``).  Expanding a random
    coupled-RME vector with that oracle and contracting the resulting AO block
    with the non-SOC projector must recover the same RME (in the projector's
    sorted-irrep order) to fp64 precision -- proving a genuine product->coupled CG
    decomposition, not a flatten/gather.
    """
    mapper = _single_species_mapper("1p")
    irreps_in = mapper.get_irreps().sort()[0].simplify()
    projector = DirectSpatialResidualBlockProjector(
        mapper, irreps_in, dtype=torch.float64, device="cpu"
    )
    atom_types = torch.tensor([[mapper.chemical_symbol_to_type["C"]]])
    p_slice = mapper.orbital_maps["C"]["1p"]

    oracle = E3Hamiltonian(
        idp=OrbitalMapper({"C": "1p"}, method="e3tb"),
        decompose=False,
        dtype=torch.float64,
        device="cpu",
    )
    cg = oracle.cgbasis["p-p"].to(torch.float64)  # (3, 3, 9) forward CG
    torch.manual_seed(0)
    rme = torch.randn(int(mapper.reduced_matrix_element), dtype=torch.float64)
    forward_block = torch.zeros(1, projector.canvas, projector.canvas, dtype=torch.float64)
    forward_block[0, p_slice, p_slice] = torch.einsum("ijr,r->ij", cg, rme)

    recovered = projector._contract(forward_block, atom_types, projector.node_plan)[0]
    expected = rme.index_select(0, projector.sort_index)
    assert torch.allclose(recovered, expected, atol=1e-10, rtol=0.0)


def test_projector_pp_onsite_scalar_has_zero_nonscalar_channels():
    """1b: a pure-scalar p-p onsite block (I3) has zero non-scalar RME.

    A ``p-p`` onsite block equal to the 3x3 identity is a pure rotational scalar
    (the ``0e`` trace channel); a correct inverse-CG leaves every ``1e``/``2e``
    coordinate exactly zero while the scalar channel is genuinely populated.  A
    flatten/gather produced a non-scalar norm ~1.414 here.
    """
    mapper = _single_species_mapper("1p")
    irreps_in = mapper.get_irreps().sort()[0].simplify()
    projector = DirectSpatialResidualBlockProjector(
        mapper, irreps_in, dtype=torch.float64, device="cpu"
    )
    atom_types = torch.tensor([[mapper.chemical_symbol_to_type["C"]]])
    p_slice = mapper.orbital_maps["C"]["1p"]

    block = torch.zeros(1, projector.canvas, projector.canvas, dtype=torch.float64)
    block[0, p_slice, p_slice] = torch.eye(3, dtype=torch.float64)
    coupled = projector._contract(block, atom_types, projector.node_plan)[0]

    assert _nonscalar_norm(irreps_in, coupled) <= 1e-10
    assert coupled.norm() > 1.0


def test_projector_is_so3_rotation_covariant():
    """1c: rotating the AO block equals rotating the coupled output by Wigner-D.

    Independent oracle: e3nn Wigner-D of the AO shells (``1x0e+1x1o`` canvas) and
    of the coupled ``irreps_in``.  The non-SOC projector packs the onsite upper
    triangle, so covariance is asserted on a projector-invariant (symmetrized)
    input: for a random proper rotation the two paths
    ``contract(D.B.D^T)`` and ``D_irreps.contract(B)`` agree to fp64 precision.
    """
    mapper = _single_species_mapper("1s1p")
    irreps_in = mapper.get_irreps().sort()[0].simplify()
    projector = DirectSpatialResidualBlockProjector(
        mapper, irreps_in, dtype=torch.float64, device="cpu"
    )
    atom_types = torch.tensor([[mapper.chemical_symbol_to_type["C"]]])

    # e3nn's D_from_matrix routes angle intermediates through the *default* dtype,
    # so fp64 covariance requires a fp64 default (else it caps out near 1e-7).
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        rotation = o3.rand_matrix(dtype=torch.float64)  # proper SO(3)
        d_ao = o3.Irreps("1x0e+1x1o").D_from_matrix(rotation)  # canvas 4x4
        d_irreps = irreps_in.D_from_matrix(rotation)

        raw = torch.randn(projector.canvas, projector.canvas, dtype=torch.float64)
        block = 0.5 * (raw + raw.transpose(-1, -2))  # projector-invariant onsite
        rotated = (d_ao @ block @ d_ao.transpose(-1, -2)).unsqueeze(0)

        lhs = projector._contract(rotated, atom_types, projector.node_plan)[0]
        rhs = d_irreps @ projector._contract(
            block.unsqueeze(0), atom_types, projector.node_plan
        )[0]
        assert torch.allclose(lhs, rhs, atol=1e-10, rtol=0.0)
    finally:
        torch.set_default_dtype(previous_default)


def test_projector_edge_blocks_are_so3_rotation_covariant_across_species():
    """1c-edge: hetero-bond edge blocks transform covariantly under Wigner-D.

    The non-SOC edge plan keeps only union-order shell pairs per directed bond
    (the reverse mate carries the rest), but the canvas rotation is shell-block
    -diagonal, so every planned ``(l1, l2)`` sub-block rotates within itself and
    the dual-path equation must hold for BOTH directed bond types of a hetero
    bond: ``contract(D.B.D^T) == contract(B).D_irreps^T``.
    """
    mapper = _mapper()
    irreps_in = mapper.get_irreps().sort()[0].simplify()
    projector = DirectSpatialResidualBlockProjector(
        mapper, irreps_in, dtype=torch.float64, device="cpu"
    )
    bond_types = torch.tensor(
        [[mapper.bond_to_type["C-H"]], [mapper.bond_to_type["H-C"]]]
    )
    edge_shapes = torch.tensor([[4, 1], [1, 4]], dtype=torch.long)

    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(1)
        rotation = o3.rand_matrix(dtype=torch.float64)
        # C's species-compact frame equals the union canvas (1s+1p); H's single
        # 1s lands on slot 0, whose 0e Wigner block is the scalar 1, so one
        # shared canvas D covers both row and column species here.
        d_ao = o3.Irreps("1x0e+1x1o").D_from_matrix(rotation)
        d_irreps = irreps_in.D_from_matrix(rotation)

        blocks = torch.randn(2, projector.canvas, projector.canvas, dtype=torch.float64)
        blocks = blocks * block_mask_from_shapes(
            edge_shapes, (projector.canvas, projector.canvas)
        ).to(blocks.dtype)
        rotated = d_ao @ blocks @ d_ao.transpose(-1, -2)

        lhs = projector._contract(rotated, bond_types, projector.edge_plan)
        rhs = projector._contract(blocks, bond_types, projector.edge_plan) @ d_irreps.transpose(-1, -2)
        assert torch.allclose(lhs, rhs, atol=FP64_ATOL, rtol=0.0)
    finally:
        torch.set_default_dtype(previous_default)


def test_projector_water_basis_multishell_rotation_covariance():
    """1c-water: cross-n shell pairs + nested species-compact canvas stay covariant.

    The water union canvas (3s+2p+1d = 14) exercises multi-shell multiplicities
    (cross-n ``(ns, n's)`` and ``(2p, 2p')`` pairs) and the species-COMPACT
    block frame: H(2s1p) packs [1s,2s,1p] into the leading 5 slots, so its p
    shell sits at compact slots 2-4, NOT at the union p offset. Each species
    therefore needs its own compact-frame Wigner-D (identity on padding); a
    union-slot D would be wrong for H and this test would catch it.
    """
    mapper = _water_mapper()
    irreps_in = mapper.get_irreps().sort()[0].simplify()
    projector = DirectSpatialResidualBlockProjector(
        mapper, irreps_in, dtype=torch.float64, device="cpu"
    )
    canvas = projector.canvas
    o_type = mapper.chemical_symbol_to_type["O"]
    h_type = mapper.chemical_symbol_to_type["H"]
    atom_types = torch.tensor([[o_type], [h_type]])
    node_shapes = torch.tensor([[14, 14], [5, 5]], dtype=torch.long)
    bond_types = torch.tensor(
        [[mapper.bond_to_type["H-O"]], [mapper.bond_to_type["O-H"]]]
    )
    edge_shapes = torch.tensor([[5, 14], [14, 5]], dtype=torch.long)

    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(2)
        rotation = o3.rand_matrix(dtype=torch.float64)
        d_o = o3.Irreps("3x0e+2x1o+1x2e").D_from_matrix(rotation)
        d_h = torch.eye(canvas, dtype=torch.float64)
        d_h[:5, :5] = o3.Irreps("2x0e+1x1o").D_from_matrix(rotation)
        d_irreps = irreps_in.D_from_matrix(rotation)

        raw = torch.randn(2, canvas, canvas, dtype=torch.float64)
        onsite = 0.5 * (raw + raw.transpose(-1, -2))
        onsite = onsite * block_mask_from_shapes(node_shapes, (canvas, canvas)).to(onsite.dtype)
        rotated_onsite = torch.stack(
            (
                d_o @ onsite[0] @ d_o.transpose(-1, -2),
                d_h @ onsite[1] @ d_h.transpose(-1, -2),
            )
        )
        edges = torch.randn(2, canvas, canvas, dtype=torch.float64)
        edges = edges * block_mask_from_shapes(edge_shapes, (canvas, canvas)).to(edges.dtype)
        rotated_edges = torch.stack(
            (
                d_h @ edges[0] @ d_o.transpose(-1, -2),
                d_o @ edges[1] @ d_h.transpose(-1, -2),
            )
        )

        for blocks, rotated, types, plan in (
            (onsite, rotated_onsite, atom_types, projector.node_plan),
            (edges, rotated_edges, bond_types, projector.edge_plan),
        ):
            lhs = projector._contract(rotated, types, plan)
            rhs = projector._contract(blocks, types, plan) @ d_irreps.transpose(-1, -2)
            assert torch.allclose(lhs, rhs, atol=FP64_ATOL, rtol=0.0)
    finally:
        torch.set_default_dtype(previous_default)


def _attach_spatial_blocks(data, state):
    data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY] = state.node_blocks
    data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY] = state.edge_blocks
    data[_keys.NODE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY] = state.node_shapes
    data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY] = state.edge_shapes


def test_projector_zero_preservation_is_structural_hard_gate():
    """1d: bias-free projector maps a zero residual to a bit-exact zero hidden.

    (1) neither node/edge equivariant linear owns a learnable bias; (2) a zero
    residual yields *exactly* zero hidden (torch.equal); (3) a packed non-zero
    residual moves the hidden, so the zero gate is not vacuously satisfied.
    """
    mapper = _mapper()
    data, _h0, d1 = _b_record(mapper)
    projector = DirectSpatialResidualBlockProjector(
        mapper, mapper.get_irreps().sort()[0].simplify(), dtype=torch.float64, device="cpu"
    )

    # (1) structural: no learnable bias parameter on either equivariant linear.
    assert [n for n, _ in projector.node_linear.named_parameters() if "bias" in n] == []
    assert [n for n, _ in projector.edge_linear.named_parameters() if "bias" in n] == []

    # (3) non-triviality: the packed (non-zero) residual moves the hidden.
    live = copy.deepcopy(data)
    _attach_spatial_blocks(live, d1)
    node_live, edge_live = projector(
        live, live["atom_types"], live["edge_type"], torch.arange(2)
    )
    assert torch.count_nonzero(node_live) > 0
    assert torch.count_nonzero(edge_live) > 0

    # (2) bit-level zero-preservation.
    zero = copy.deepcopy(data)
    _attach_spatial_blocks(
        zero,
        BlockTensorResult(
            torch.zeros_like(d1.node_blocks),
            torch.zeros_like(d1.edge_blocks),
            d1.node_shapes,
            d1.edge_shapes,
        ),
    )
    node_hidden, edge_hidden = projector(
        zero, zero["atom_types"], zero["edge_type"], torch.arange(2)
    )
    assert torch.equal(node_hidden, torch.zeros_like(node_hidden))
    assert torch.equal(edge_hidden, torch.zeros_like(edge_hidden))


def test_projector_node_edge_parameters_are_independent():
    """1e: zero node input -> node hidden bit-zero while edge hidden nonzero (and vice versa)."""
    mapper = _mapper()
    data, _h0, d1 = _b_record(mapper)
    projector = DirectSpatialResidualBlockProjector(
        mapper, mapper.get_irreps().sort()[0].simplify(), dtype=torch.float64, device="cpu"
    )
    zeros_node = torch.zeros_like(d1.node_blocks)
    zeros_edge = torch.zeros_like(d1.edge_blocks)

    node_zero_input = copy.deepcopy(data)
    _attach_spatial_blocks(
        node_zero_input,
        BlockTensorResult(zeros_node, d1.edge_blocks, d1.node_shapes, d1.edge_shapes),
    )
    node_hidden, edge_hidden = projector(
        node_zero_input, node_zero_input["atom_types"], node_zero_input["edge_type"], torch.arange(2)
    )
    assert torch.equal(node_hidden, torch.zeros_like(node_hidden))
    assert torch.count_nonzero(edge_hidden) > 0

    edge_zero_input = copy.deepcopy(data)
    _attach_spatial_blocks(
        edge_zero_input,
        BlockTensorResult(d1.node_blocks, zeros_edge, d1.node_shapes, d1.edge_shapes),
    )
    node_hidden, edge_hidden = projector(
        edge_zero_input, edge_zero_input["atom_types"], edge_zero_input["edge_type"], torch.arange(2)
    )
    assert torch.count_nonzero(node_hidden) > 0
    assert torch.equal(edge_hidden, torch.zeros_like(edge_hidden))


def test_projector_water_canvas_nesting_zero_and_shape_gate():
    """1f (water): canvas nesting (O 14 / H 5) zero-preservation, shape + key gates."""
    mapper = _water_mapper()
    data = _water_graph(mapper)
    node_shapes, edge_shapes = infer_block_shapes(data, mapper)
    canvas = mapper_max_norb(mapper)
    assert canvas == 14  # the O 3s2p1d block nests H's 5-wide block inside it
    n = int(node_shapes.shape[0])
    e = int(edge_shapes.shape[0])
    projector = DirectSpatialResidualBlockProjector(
        mapper, mapper.get_irreps().sort()[0].simplify(), dtype=torch.float64, device="cpu"
    )

    # zero residual -> bit-exact zero hidden on the nested canvas.
    zero = copy.deepcopy(data)
    _attach_spatial_blocks(
        zero,
        BlockTensorResult(
            torch.zeros(n, canvas, canvas, dtype=torch.float64),
            torch.zeros(e, canvas, canvas, dtype=torch.float64),
            node_shapes,
            edge_shapes,
        ),
    )
    node_hidden, edge_hidden = projector(
        zero, zero["atom_types"], zero["edge_type"], torch.arange(e)
    )
    assert torch.equal(node_hidden, torch.zeros_like(node_hidden))
    assert torch.equal(edge_hidden, torch.zeros_like(edge_hidden))

    # a packed non-zero residual moves the hidden (non-vacuity on this canvas).
    live = copy.deepcopy(data)
    live_state = _projected_state(mapper, data, canvas=canvas, n=n, e=e, dtype=torch.float64, seed=7)
    _attach_spatial_blocks(live, live_state)
    node_live, _ = projector(live, live["atom_types"], live["edge_type"], torch.arange(e))
    assert torch.count_nonzero(node_live) > 0

    # wrong canvas -> ValueError; missing keys -> KeyError.
    bad_canvas = copy.deepcopy(data)
    _attach_spatial_blocks(
        bad_canvas,
        BlockTensorResult(
            torch.zeros(n, canvas + 1, canvas + 1, dtype=torch.float64),
            torch.zeros(e, canvas + 1, canvas + 1, dtype=torch.float64),
            node_shapes,
            edge_shapes,
        ),
    )
    with pytest.raises(ValueError):
        projector(bad_canvas, bad_canvas["atom_types"], bad_canvas["edge_type"], torch.arange(e))

    with pytest.raises(KeyError):
        projector(copy.deepcopy(data), data["atom_types"], data["edge_type"], torch.arange(e))


def test_projector_rejects_wrong_block_shapes():
    """1f: stored block shapes disagreeing with the mapper plans -> ValueError."""
    mapper = _mapper()
    data, _h0, d1 = _b_record(mapper)
    projector = DirectSpatialResidualBlockProjector(
        mapper, mapper.get_irreps().sort()[0].simplify(), dtype=torch.float64, device="cpu"
    )
    bad = copy.deepcopy(data)
    _attach_spatial_blocks(bad, d1)
    bad[_keys.NODE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY] = d1.node_shapes + 1
    with pytest.raises(ValueError):
        projector(bad, bad["atom_types"], bad["edge_type"], torch.arange(2))


def test_projector_guards_reject_soc_mappers_both_directions():
    """1g: the spatial projector rejects a uu-real (SOC) mapper; the uu-real
    projector still rejects a plain non-SOC mapper (frozen cross-check)."""
    uureal = _uureal_mapper("1s1p")
    with pytest.raises((ValueError, NotImplementedError), match="non-SOC"):
        DirectSpatialResidualBlockProjector(
            uureal, uureal.get_irreps().sort()[0].simplify(),
            dtype=torch.float64, device="cpu",
        )

    non_soc = _mapper()
    with pytest.raises(ValueError, match="SOC uu_real"):
        DirectUuRealBlockProjector(
            non_soc, non_soc.get_irreps().sort()[0].simplify(),
            dtype=torch.float64, device="cpu",
        )


# ===========================================================================
# 2. prepare_batch bridge + H0-invariance + masquerade rejection
# ===========================================================================
def test_prepare_batch_exact_scalar_bridge_and_constant_physical_h0():
    """2: D_t = t.project(D1) exactly; the h0 keys carry blocks_to_rme(H0),
    constant in t (contract-2 lock); ctx fields; authority fields dropped."""
    mapper = _mapper()
    data, h0, d1 = _b_record(mapper)
    flow = _b_flow(mapper)

    model_data, ref, ctx = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.25], dtype=torch.float64)
    )

    # exact scalar bridge: attached spatial state == 0.25 * D1 (D1 is packer-image).
    torch.testing.assert_close(
        model_data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY],
        0.25 * d1.node_blocks,
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        model_data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY],
        0.25 * d1.edge_blocks,
        rtol=0.0,
        atol=1e-12,
    )

    # h0 keys carry the physical H0 RME (== blocks_to_rme(H0)), independent of t.
    expected_node_base, expected_edge_base = flow.block_codec.blocks_to_rme(
        copy.deepcopy(data), h0
    )
    torch.testing.assert_close(
        torch.as_tensor(model_data[flow.node_h0_key]), expected_node_base, rtol=0.0, atol=1e-10
    )
    torch.testing.assert_close(
        torch.as_tensor(model_data[flow.edge_h0_key]), expected_edge_base, rtol=0.0, atol=1e-10
    )

    low_t, _, _ = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.1], dtype=torch.float64)
    )
    high_t, _, _ = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.9], dtype=torch.float64)
    )
    assert torch.equal(
        torch.as_tensor(low_t[flow.node_h0_key]), torch.as_tensor(high_t[flow.node_h0_key])
    )
    assert torch.equal(
        torch.as_tensor(low_t[flow.edge_h0_key]), torch.as_tensor(high_t[flow.edge_h0_key])
    )

    # ctx: physical base set, no absolute target, residual semantics.
    assert ctx.node_base is not None and ctx.edge_base is not None
    assert ctx.node_target is None and ctx.edge_target is None
    assert ctx.block_target_semantics == "residual_dh"

    # the certified endpoint/H0 block side channels stay outside model input.
    for key in (
        flow.node_block_target_key,
        flow.edge_block_target_key,
        flow.node_h0_block_key,
        flow.edge_h0_block_key,
        flow.node_h0_block_shape_key,
        flow.edge_h0_block_shape_key,
    ):
        assert key not in model_data


def test_prepare_batch_rejects_uureal_metadata_masquerade():
    """2: uu-real compact metadata on a raw non-SOC record fails closed."""
    mapper = _mapper()
    data, _h0, _d1 = _b_record(mapper)
    data["soc_uureal_compact"] = True
    with pytest.raises(ValueError, match="residual_ao_block_ode"):
        _b_flow(mapper).prepare_batch(
            copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.5], dtype=torch.float64)
        )


def test_prepare_batch_rejects_uureal_state_key_masquerade():
    """2: a compact-uu residual state key present on a raw non-SOC record fails closed."""
    mapper = _mapper()
    data, _h0, d1 = _b_record(mapper)
    data[_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY] = d1.node_blocks.clone()
    with pytest.raises(ValueError, match="residual_ao_block_ode"):
        _b_flow(mapper).prepare_batch(
            copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.5], dtype=torch.float64)
        )


# ===========================================================================
# 3. t0 boundary mass
# ===========================================================================
def test_t0_probability_defaults_positive_and_rejects_explicit_zero():
    """3: default 0.15; explicit non-positive t0_probability fails closed."""
    mapper = _mapper()
    assert _b_flow(mapper).t0_probability == pytest.approx(0.15)
    assert _b_flow(mapper, t0_probability=0.25).t0_probability == pytest.approx(0.25)
    with pytest.raises(ValueError, match="t0_probability"):
        _b_flow(mapper, t0_probability=0.0)
    with pytest.raises(ValueError, match="t0_probability"):
        _b_flow(mapper, t0_probability=-0.1)


def test_t0_injection_bypasses_t_min_clamp():
    """3: t0 injection yields exact zeros and every non-zero sample honours t_min."""
    mapper = _mapper()
    # p=1 is now rejected at construction (full boundary collapse is a
    # misconfiguration); force full injection AFTER construction to keep
    # exercising the runtime bypass mechanism itself.
    flow = _b_flow(mapper, t_min=0.5, t0_probability=0.15)
    flow.t0_probability = 1.0
    t = flow._sample_t(num_graphs=64, device=torch.device("cpu"), dtype=torch.float64)
    assert torch.equal(t, torch.zeros_like(t))

    flow = _b_flow(mapper, t_min=0.5, t0_probability=0.5)
    generator = torch.Generator().manual_seed(0)
    t = flow._sample_t(
        num_graphs=512, device=torch.device("cpu"), dtype=torch.float64, generator=generator
    )
    zero = t == 0.0
    assert bool(zero.any()) and bool((~zero).any())
    assert bool((t[~zero] >= 0.5).all())


# ===========================================================================
# 4. Sampler closed loop / exactly-once assembly / label-free / H0-constancy
# ===========================================================================
class _EndpointSpy(torch.nn.Module):
    """Return pre-set residual endpoint blocks per call, recording model inputs.

    Mirrors ``test_uureal_block_ode._EndpointSpy`` but records both the pure-D
    spatial-state keys and the constant physical-H0 RME keys so the tests can
    assert H0-constancy and the pure-D state across steps.
    """

    def __init__(self, endpoints, node_h0_key, edge_h0_key):
        super().__init__()
        self.endpoints = endpoints
        self.spatial_inputs = []
        self.h0_inputs = []
        self.times = []
        self._node_h0_key = node_h0_key
        self._edge_h0_key = edge_h0_key

    def forward(self, data):
        self.spatial_inputs.append(
            (
                data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY].clone(),
                data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY].clone(),
            )
        )
        self.h0_inputs.append(
            (data[self._node_h0_key].clone(), data[self._edge_h0_key].clone())
        )
        self.times.append(data["flow_time"].clone())
        node, edge = self.endpoints[len(self.spatial_inputs) - 1]
        out = data.copy()
        out[_keys.NODE_PRED_HAMIL_BLOCKS_KEY] = node.clone()
        out[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY] = edge.clone()
        return out


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_sampler_spy_closed_loop_and_exactly_once_assembly(steps):
    """4: pure-D blend closure in D-space; final pred == H0 + D_blend once;
    every step sees the SAME physical H0 RME and a pure-D spatial state."""
    mapper = _mapper()
    data, h0, d1 = _b_record(mapper)
    flow = _b_flow(mapper)

    node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)

    # A constant residual endpoint D1: with the endpoint-blend schedule the last
    # alpha is exactly 1, so the final residual state equals D1 for every #steps.
    endpoints = [(d1.node_blocks.clone(), d1.edge_blocks.clone()) for _ in range(steps)]
    spy = _EndpointSpy(endpoints, flow.node_h0_key, flow.edge_h0_key)
    result = flow.sample(spy, copy.deepcopy(data), num_steps=steps)

    # zero-start pure-D state; step 1 (if any) is the first blend alpha_0 = 1/steps.
    assert torch.count_nonzero(spy.spatial_inputs[0][0]) == 0
    assert torch.count_nonzero(spy.spatial_inputs[0][1]) == 0
    if steps >= 2:
        torch.testing.assert_close(
            spy.spatial_inputs[1][0], d1.node_blocks / steps, rtol=0.0, atol=1e-12
        )
        torch.testing.assert_close(
            spy.spatial_inputs[1][1], d1.edge_blocks / steps, rtol=0.0, atol=1e-12
        )

    # H0-constancy across the rollout (contract-2 rollout lock).
    for step in range(steps):
        torch.testing.assert_close(spy.h0_inputs[step][0], node_base, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(spy.h0_inputs[step][1], edge_base, rtol=0.0, atol=1e-12)
    assert spy.times[0].reshape(-1)[0].item() == 0.0

    # exactly-once assembly: predicted full-H blocks == H0 + D_blend (D_blend==D1).
    torch.testing.assert_close(
        result[_keys.NODE_PRED_HAMIL_BLOCKS_KEY],
        h0.node_blocks + d1.node_blocks,
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY],
        h0.edge_blocks + d1.edge_blocks,
        rtol=0.0,
        atol=1e-12,
    )
    # h0 keys keep the physical H0 RME; flow_time lands on 1.
    torch.testing.assert_close(
        torch.as_tensor(result[flow.node_h0_key]), node_base, rtol=0.0, atol=1e-10
    )
    assert torch.allclose(
        result[flow.flow_time_key], torch.ones_like(result[flow.flow_time_key])
    )


def test_sampler_is_label_free():
    """4: inference starts from D_0=0 and never reads the delta endpoint labels."""
    mapper = _mapper()
    data, h0, d1 = _b_record(mapper)
    for key in (
        _keys.NODE_DELTA_HAMIL_BLOCKS_KEY,
        _keys.EDGE_DELTA_HAMIL_BLOCKS_KEY,
        _keys.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
        _keys.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    ):
        del data[key]
    flow = _b_flow(mapper)
    spy = _EndpointSpy(
        [(d1.node_blocks.clone(), d1.edge_blocks.clone())], flow.node_h0_key, flow.edge_h0_key
    )
    result = flow.sample(spy, copy.deepcopy(data), num_steps=1)
    torch.testing.assert_close(
        result[_keys.NODE_PRED_HAMIL_BLOCKS_KEY],
        h0.node_blocks + d1.node_blocks,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "missing_key",
    [_keys.NODE_PRED_HAMIL_BLOCKS_KEY, _keys.EDGE_PRED_HAMIL_BLOCKS_KEY],
)
def test_sampler_requires_fresh_endpoint_outputs(missing_key):
    """4: a model that drops a fresh output key at step 2 fails closed."""
    mapper = _mapper()
    data, _h0, d1 = _b_record(mapper)

    class FirstStepCompleteSecondStepIncomplete(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            out = batch.copy()
            outputs = {
                _keys.NODE_PRED_HAMIL_BLOCKS_KEY: d1.node_blocks,
                _keys.EDGE_PRED_HAMIL_BLOCKS_KEY: d1.edge_blocks,
            }
            for key, value in outputs.items():
                if self.calls == 1 or key != missing_key:
                    out[key] = value
            return out

    with pytest.raises(ValueError, match=rf"step 2.*{missing_key}"):
        _b_flow(mapper).sample(
            FirstStepCompleteSecondStepIncomplete(), copy.deepcopy(data), num_steps=2
        )


# ===========================================================================
# 5. Loss caliber: golden training scalar + H0-cancellation parity lock
# ===========================================================================
def _pooled_masked_mse(ref, idp, pred_state, target_state):
    """Reproduce ``_block_ode_endpoint_loss``'s pooled global_elements MSE."""
    pred = project_block_state(ref, idp, pred_state)
    target = project_block_state(ref, idp, target_state)
    node_diff = pred.node_blocks - target.node_blocks
    edge_diff = pred.edge_blocks - target.edge_blocks

    node_valid = block_mask_from_shapes(pred.node_shapes, tuple(node_diff.shape[-2:]))
    upper = torch.triu(torch.ones(tuple(node_diff.shape[-2:]), dtype=torch.bool))
    node_mask = node_valid & upper.unsqueeze(0)

    edge_valid = block_mask_from_shapes(pred.edge_shapes, tuple(edge_diff.shape[-2:]))
    rev = strict_reverse_edge_index(ref, idp=idp)
    rows = torch.arange(edge_diff.shape[0])
    canonical_rows = rows <= rev
    edge_mask = edge_valid & canonical_rows.view(-1, 1, 1)
    self_reverse = rows == rev
    if bool(self_reverse.any()):
        edge_mask[self_reverse] &= upper.unsqueeze(0)

    node_f = node_mask.to(node_diff.dtype)
    edge_f = edge_mask.to(edge_diff.dtype)
    square_sum = (node_diff.square() * node_f).sum() + (edge_diff.square() * edge_f).sum()
    count = node_f.sum() + edge_f.sum()
    return square_sum / count.clamp_min(1.0)


def test_training_loss_matches_golden_pooled_masked_mse():
    """5a: training loss == manual pooled masked MSE of (Dhat - D1)."""
    mapper = _mapper()
    data, _h0, d1 = _b_record(mapper)
    flow = _b_flow(mapper)
    batch, ref, ctx = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.4], dtype=torch.float64)
    )
    node_shapes = torch.as_tensor(ref[flow.node_block_shape_key])
    edge_shapes = torch.as_tensor(ref[flow.edge_block_shape_key])
    dhat_node = 2.0 * torch.as_tensor(ref[flow.node_block_target_key])
    dhat_edge = 2.0 * torch.as_tensor(ref[flow.edge_block_target_key])

    pred = batch.copy()
    pred[flow.node_output_key] = dhat_node
    pred[flow.edge_output_key] = dhat_edge
    loss, _state = flow.loss(pred, ref, ctx)

    golden = _pooled_masked_mse(
        ref,
        mapper,
        BlockTensorResult(dhat_node, dhat_edge, node_shapes, edge_shapes),
        BlockTensorResult(
            torch.as_tensor(ref[flow.node_block_target_key]),
            torch.as_tensor(ref[flow.edge_block_target_key]),
            node_shapes,
            edge_shapes,
        ),
    )
    assert golden.item() > 0.0
    torch.testing.assert_close(loss, golden, rtol=0.0, atol=1e-12)


def test_h0_cancellation_parity_and_sample_loss_routing():
    """5b/5c: scoring the assembled full-H sample equals the residual training
    scalar (H0 cancels), and loss_on_sample routes through the compatible path."""
    mapper = _mapper()
    data, _h0, d1 = _b_record(mapper)
    flow = _b_flow(mapper)
    batch, ref, ctx = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.4], dtype=torch.float64)
    )
    node_shapes = torch.as_tensor(ref[flow.node_block_shape_key])
    edge_shapes = torch.as_tensor(ref[flow.edge_block_shape_key])
    dhat = BlockTensorResult(
        2.0 * torch.as_tensor(ref[flow.node_block_target_key]),
        2.0 * torch.as_tensor(ref[flow.edge_block_target_key]),
        node_shapes,
        edge_shapes,
    )

    # (a) residual training scalar.
    train_pred = batch.copy()
    train_pred[flow.node_output_key] = dhat.node_blocks
    train_pred[flow.edge_output_key] = dhat.edge_blocks
    train_loss, _ = flow.loss(train_pred, ref, ctx)

    # (b) assemble the physical full-H sample H0 + Dhat with the SAME H0 the
    # scorer reconstructs from ctx.node_base, then score it as a sample.
    h0_from_base = flow.block_codec.rme_to_blocks(
        ref, ctx.node_base, ctx.edge_base, project=True
    )
    full = flow.block_codec.endpoint_to_full(dhat, h0_from_base)
    sample_pred = batch.copy()
    sample_pred[flow.node_output_key] = full.node_blocks
    sample_pred[flow.edge_output_key] = full.edge_blocks

    compatible_loss, _ = flow.compatible_loss_on_sample(sample_pred, ref, ctx)
    torch.testing.assert_close(compatible_loss, train_loss, rtol=0.0, atol=1e-10)

    # (c) loss_on_sample routes non-uureal block_ode through the compatible path.
    routed_loss, _ = flow.loss_on_sample(sample_pred, ref, ctx)
    torch.testing.assert_close(routed_loss, compatible_loss, rtol=0.0, atol=0.0)


# ===========================================================================
# 6. argcheck: B yaml acceptance + rejection matrix
# ===========================================================================
def _load_b_config():
    path = Path("configs") / "h_b0_block_ode_water_residual.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_b_arm_yaml_passes_validate_block_ode_contract():
    """6: the actual B arm config file validates."""
    assert validate_block_ode_contract(_load_b_config()) is None


def _mutate(cfg, path, value):
    cfg = copy.deepcopy(cfg)
    cursor = cfg
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return cfg


REJECTIONS = [
    # (path, value, match)
    (("common_options", "has_soc"), True, "has_soc"),
    (("common_options", "nextham_uureal_mask"), True, "nextham_uureal_mask"),
    (("train_options", "flow_options", "state_space"), "wrong", "state_space"),
    (("train_options", "flow_options", "block_input_adapter"), "wrong", "block_input_adapter"),
    (("train_options", "flow_options", "h0_condition_space"), "wrong", "h0_condition_space"),
    (("train_options", "flow_options", "block_export_final_full_h"), False, "block_export"),
    (("train_options", "flow_options", "t0_probability"), 0.0, "t0_probability"),
    (("train_options", "flow_options", "prediction_add_h0"), True, "prediction_add_h0"),
    (("model_options", "prediction", "add_h0"), True, "add_h0"),
    (
        ("train_options", "flow_options", "node_block_target_key"),
        "node_full_hamil_target_blocks",
        "node_block_target_key",
    ),
    # NOTE (v2): prior='projected_te' is NO LONGER a rejection for the residual
    # mode -- it is an accepted stochastic-bridge arm.  Acceptance and the retained
    # fail-closed (missing te_prior_validation_seed) are asserted by
    # test_te_arm_yaml_* below and test_flow_ctor_accepts_projected_te_prior_*.
    (
        ("model_options", "embedding", "use_spatial_residual_block_input"),
        False,
        "use_spatial_residual_block_input",
    ),
    (
        ("model_options", "embedding", "use_uureal_residual_block_input"),
        True,
        "use_uureal_residual_block_input",
    ),
    (("data_options", "train", "residual_hamiltonian"), False, "residual_hamiltonian"),
    (("data_options", "train", "require_full_h_target"), True, "require_full_h_target"),
    (("data_options", "train", "require_residual_h_target"), True, "require_residual_h_target"),
    (
        ("data_options", "train", "require_residual_from_full_h_target"),
        False,
        "require_residual_from_full_h_target",
    ),
    (("data_options", "train", "require_uureal_block_ode"), True, "require_uureal_block_ode"),
    (("train_options", "flow_options", "output_space"), "spatial_residual_block_ode", "output_space"),
]


@pytest.mark.parametrize("path,value,match", REJECTIONS)
def test_b_arm_rejection_matrix(path, value, match):
    """6: each single-field violation of the B contract fails closed."""
    cfg = _mutate(_load_b_config(), path, value)
    with pytest.raises(ValueError, match=match):
        validate_block_ode_contract(cfg)


def test_b_arm_hyphenated_alias_normalizes_and_validates():
    """6: the hyphenated output_space alias normalizes and still validates."""
    cfg = _mutate(
        _load_b_config(),
        ("train_options", "flow_options", "output_space"),
        "residual-ao-block-ode",
    )
    assert validate_block_ode_contract(cfg) is None


def _load_te_config():
    path = Path("configs") / "h_b0_block_ode_water_residual_te.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_te_arm_yaml_passes_validate_block_ode_contract():
    """6d (v2): the third-arm projected_te config file validates (acceptance)."""
    assert validate_block_ode_contract(_load_te_config()) is None


def test_te_arm_rejects_missing_validation_seed():
    """6d (v2): dropping te_prior_validation_seed fails closed (retained gate)."""
    cfg = copy.deepcopy(_load_te_config())
    del cfg["train_options"]["flow_options"]["te_prior_validation_seed"]
    with pytest.raises(ValueError, match="te_prior_validation_seed"):
        validate_block_ode_contract(cfg)


def test_te_arm_is_frozen_b_arm_except_prior_block():
    """6d (v2): the te arm differs from the frozen B arm ONLY in the prior +
    the projected_te option block; every other field is byte-identical."""
    b = _load_b_config()
    te = _load_te_config()
    te_only_keys = {
        "prior",
        "te_prior_mode",
        "node_sigma",
        "edge_sigma",
        "te_prior_sigma",
        "te_prior_validation_seed",
    }
    b_flow = dict(b["train_options"]["flow_options"])
    te_flow = dict(te["train_options"]["flow_options"])
    b_flow.pop("prior", None)
    for key in te_only_keys:
        te_flow.pop(key, None)
    assert b_flow == te_flow
    # everything outside flow_options is untouched.
    assert b["common_options"] == te["common_options"]
    assert b["model_options"] == te["model_options"]
    assert b["data_options"] == te["data_options"]


# ===========================================================================
# 7. Loader: tmp_path real LMDB gates
# ===========================================================================
def _raw_absolute_full_h_record() -> dict:
    """A raw H2 record declaring absolute_full_h semantics (A/B share this)."""
    h_blocks = {
        "0_0_0_0_0": np.asarray([[10.0]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[12.0]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[3.0]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[3.0]], dtype=np.float32),
    }
    h0_blocks = {
        "0_0_0_0_0": np.asarray([[9.0]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[11.0]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[2.5]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[2.5]], dtype=np.float32),
    }
    return {
        _keys.CELL_KEY: np.eye(3, dtype=np.float32) * 8.0,
        _keys.POSITIONS_KEY: np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1, 1], dtype=np.int64),
        _keys.PBC_KEY: np.asarray([False, False, False]),
        "case_id": "h2",
        _keys.EDGE_INDEX_KEY: np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        _keys.EDGE_CELL_SHIFT_KEY: np.zeros((2, 3), dtype=np.float32),
        "hamiltonian": h_blocks,
        "hamiltonian_0": h0_blocks,
        SAMPLE_SCHEMA_KEY: RAW_HAMILTONIAN_SAMPLE_SCHEMA,
        TARGET_SEMANTICS_KEY: ABSOLUTE_FULL_H_SEMANTICS,
        TARGET_SOURCE_KEY: "raw_hamiltonian",
    }


def _build_dataset(tmp_path, record, *, name, **kwargs):
    lmdb_path = tmp_path / f"{name}.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=True)
    try:
        with env.begin(write=True) as txn:
            txn.put((0).to_bytes(4, "big"), pickle.dumps(record))
    finally:
        env.close()
    return DatasetBuilder()(
        root=str(tmp_path),
        r_max=2.0,
        type="LMDBDataset",
        prefix=name,
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_H0=True,
        **kwargs,
    )


def test_loader_materializes_residual_from_absolute_full_h_record(tmp_path):
    """7: an absolute_full_h record + require_residual_from_full_h_target loads;
    delta blocks == raw - H0 exactly and physical-H0 blocks attach."""
    dataset = _build_dataset(
        tmp_path,
        _raw_absolute_full_h_record(),
        name="b-residual-from-full-h",
        residual_hamiltonian=True,
        require_full_h_target=False,
        require_residual_h_target=False,
        require_residual_from_full_h_target=True,
    )
    sample = dataset.get(0)
    torch.testing.assert_close(
        sample[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([1.0, 1.0])
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([0.5, 0.5])
    )
    torch.testing.assert_close(
        sample[_keys.NODE_H0_BLOCKS_KEY].flatten(), torch.tensor([9.0, 11.0])
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_H0_BLOCKS_KEY].flatten(), torch.tensor([2.5, 2.5])
    )


def test_loader_rejects_h0_residual_semantics_under_new_flag(tmp_path):
    """7: the new gate demands absolute_full_h; an h0_residual record fails closed."""
    record = _raw_absolute_full_h_record()
    record[TARGET_SEMANTICS_KEY] = H0_RESIDUAL_SEMANTICS
    dataset = _build_dataset(
        tmp_path,
        record,
        name="b-wrong-semantics",
        residual_hamiltonian=True,
        require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError, match="absolute_full_h"):
        dataset.get(0)


def test_loader_requires_physical_h0_dictionary(tmp_path):
    """7: a record missing the raw H0 dictionary fails closed under the new gate."""
    record = _raw_absolute_full_h_record()
    record.pop("hamiltonian_0")
    dataset = _build_dataset(
        tmp_path,
        record,
        name="b-missing-h0",
        residual_hamiltonian=True,
        require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError, match="hamiltonian_0"):
        dataset.get(0)


def test_loader_new_flag_mutually_exclusive_with_residual_h_target(tmp_path):
    """7: the new flag is mutually exclusive with require_residual_h_target (ctor)."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_dataset(
            tmp_path,
            _raw_absolute_full_h_record(),
            name="b-excl-residual-h",
            residual_hamiltonian=True,
            require_residual_h_target=True,
            require_residual_from_full_h_target=True,
        )


def test_loader_new_flag_requires_residual_hamiltonian(tmp_path):
    """7: the new flag requires residual_hamiltonian=True (ctor interlock)."""
    with pytest.raises(ValueError, match="residual_hamiltonian"):
        _build_dataset(
            tmp_path,
            _raw_absolute_full_h_record(),
            name="b-needs-residual",
            residual_hamiltonian=False,
            require_residual_from_full_h_target=True,
        )


def test_loader_new_flag_mutually_exclusive_with_uureal_block_ode(tmp_path):
    """7: the new flag conflicts with require_uureal_block_ode (ctor interlock)."""
    with pytest.raises(ValueError, match="mutually exclusive|already-delta|stay false"):
        _build_dataset(
            tmp_path,
            _raw_absolute_full_h_record(),
            name="b-excl-uureal",
            residual_hamiltonian=True,
            require_uureal_block_ode=True,
            require_residual_from_full_h_target=True,
        )


def test_frozen_residual_h_target_gate_still_rejects_absolute_record(tmp_path):
    """7: the FROZEN require_residual_h_target gate still rejects an absolute
    record (h0_residual demanded) -- unchanged by the new mode."""
    dataset = _build_dataset(
        tmp_path,
        _raw_absolute_full_h_record(),
        name="b-frozen-residual-gate",
        residual_hamiltonian=True,
        require_residual_h_target=True,
    )
    with pytest.raises(ValueError, match="h0_residual"):
        dataset.get(0)


# ===========================================================================
# 8. Flow ctor guards
# ===========================================================================
def test_flow_ctor_rejects_soc_mapper():
    """8: residual_ao_block_ode may not be built on a uu-real / SOC mapper."""
    with pytest.raises((ValueError, NotImplementedError), match="non-SOC"):
        _b_flow(_uureal_mapper("1s1p"))


def test_flow_ctor_rejects_absolute_full_h_semantics():
    """8: residual_ao_block_ode requires residual_dh target semantics."""
    with pytest.raises(ValueError, match="target_semantics"):
        _b_flow(_mapper(), target_semantics="absolute_full_h")


def test_flow_ctor_accepts_projected_te_prior_with_full_te_options():
    """8 (v2 lift): residual_ao_block_ode accepts prior='projected_te' with the
    full shared te options, and still fails closed when te_prior_validation_seed
    is missing (the mandatory-seed gate is retained, not the whole-prior ban)."""
    flow = _b_te_flow(_mapper())
    assert flow.prior == "projected_te"
    assert flow.residual_ao_block_ode is True
    assert flow.te_prior_mode == "irrep"
    assert flow.te_prior_validation_seed == _TE_SEED
    # fail-closed: projected_te without an explicit validation seed is rejected.
    with pytest.raises(ValueError, match="te_prior_validation_seed"):
        _b_flow(
            _mapper(),
            prior="projected_te",
            te_prior_mode="irrep",
            node_sigma=1.0,
            edge_sigma=1.0,
            te_prior_sigma=1.0,
        )
    # a generic (non-projected) TE prior is still rejected for the mode.
    with pytest.raises(ValueError, match="projected_te"):
        _b_flow(_mapper(), prior="gaussian")


def test_flow_ctor_requires_block_export_final_full_h():
    """8: final full-H assembly happens once outside the ODE; the flag is required."""
    with pytest.raises(ValueError, match="exactly once outside"):
        _b_flow(_mapper(), block_export_final_full_h=False)


def test_flow_ctor_constructs_block_codec_for_residual_mode():
    """8: unlike uureal_block_ode, B keeps a real exact-RME block codec."""
    flow = _b_flow(_mapper())
    assert flow.block_codec is not None
    assert flow.residual_ao_block_ode is True


# ===========================================================================
# 9. projected_te stochastic bridge (v2 lift)
# ===========================================================================
def test_residual_te_eps_is_reproducible_and_certifies_in_codec_image():
    """9c: the internal eps draw is seed-deterministic and certifies in-image.

    Two draws with the same seed are bit-identical (reproducibility), the epsilon
    is non-trivial, and (certification due) it repacks within block_inverse_atol --
    i.e. it lies in the certified codec image.  An out-of-image epsilon is
    impossible by construction (eps = project(rme_to_blocks(noise, project=False))
    is the linear forward image of an RME vector), so that branch is not mockable
    here; the in-image assertion is the meaningful check.
    """
    mapper = _mapper()
    data, h0, _d1 = _b_record(mapper)
    flow = _b_te_flow(mapper)
    node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)

    eps1 = flow._residual_te_eps(
        copy.deepcopy(data),
        node_base,
        edge_base,
        generator=flow._seeded_generator(node_base.device, _TE_SEED),
        certify_image=True,  # certification is due and must NOT raise
    )
    eps2 = flow._residual_te_eps(
        copy.deepcopy(data),
        node_base,
        edge_base,
        generator=flow._seeded_generator(node_base.device, _TE_SEED),
        certify_image=True,
    )
    assert torch.equal(eps1.node_blocks, eps2.node_blocks)
    assert torch.equal(eps1.edge_blocks, eps2.edge_blocks)
    assert torch.count_nonzero(eps1.node_blocks) > 0
    assert torch.count_nonzero(eps1.edge_blocks) > 0

    # explicit in-image certification: repack roundtrip residual <= atol.
    node_rme, edge_rme = flow.block_codec.blocks_to_rme(copy.deepcopy(data), eps1)
    roundtrip = flow.block_codec.rme_to_blocks(
        copy.deepcopy(data), node_rme, edge_rme, project=True
    )
    residual = max(
        (roundtrip.node_blocks - eps1.node_blocks).abs().max().item(),
        (roundtrip.edge_blocks - eps1.edge_blocks).abs().max().item(),
    )
    assert residual <= flow.block_inverse_atol


def test_prepare_batch_projected_te_bridge_identity():
    """9a: D_t = project((1 - t) * eps + t * D1) with eps reproduced via the
    flow's own seeded draw path; ctx.node_prior carries the same eps."""
    mapper = _mapper()
    data, h0, d1 = _b_record(mapper)
    flow = _b_te_flow(mapper)
    node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)

    # Reproduce eps with the same seed (proven bit-reproducible above).
    eps = flow._residual_te_eps(
        copy.deepcopy(data),
        node_base,
        edge_base,
        generator=flow._seeded_generator(node_base.device, _TE_SEED),
        certify_image=True,
    )

    t = torch.tensor([0.25], dtype=torch.float64)
    model_data, _ref, ctx = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=t, prior_seed=_TE_SEED
    )

    # D1 (the delta endpoint) is packer-image, so endpoint == d1 exactly.
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
    # ctx exposes the drawn epsilon (telemetry) instead of zeros_like.
    torch.testing.assert_close(ctx.node_prior, eps.node_blocks, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(ctx.edge_prior, eps.edge_blocks, rtol=0.0, atol=1e-12)
    assert ctx.block_target_semantics == "residual_dh"
    # the physical H0 RME channel is unchanged by the prior choice.
    torch.testing.assert_close(
        torch.as_tensor(model_data[flow.node_h0_key]), node_base, rtol=0.0, atol=1e-10
    )


def test_prepare_batch_projected_te_bridge_is_not_the_t_shortcut():
    """9a: with a non-zero eps the attached D_t at t=0.25 is NOT 0.25 * D1 (the
    deterministic bridge shortcut the zero prior leaves open)."""
    mapper = _mapper()
    data, _h0, d1 = _b_record(mapper)
    flow = _b_te_flow(mapper)
    model_data, _ref, _ctx = flow.prepare_batch(
        copy.deepcopy(data),
        copy.deepcopy(data),
        t=torch.tensor([0.25], dtype=torch.float64),
        prior_seed=_TE_SEED,
    )
    shortcut = 0.25 * d1.node_blocks
    assert not torch.allclose(
        model_data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY], shortcut, atol=1e-6
    )


def test_sampler_projected_te_seed_determinism():
    """9b: same prior_seed => identical D0/output; a different seed => different
    (nonzero) D0.  With num_steps=1 the endpoint-blend collapses the final state
    onto D1 regardless of D0, so seed-sensitivity is asserted on the injected D0."""
    mapper = _mapper()
    data, _h0, d1 = _b_record(mapper)
    flow = _b_te_flow(mapper)

    def _run(seed):
        spy = _EndpointSpy(
            [(d1.node_blocks.clone(), d1.edge_blocks.clone())],
            flow.node_h0_key,
            flow.edge_h0_key,
        )
        result = flow.sample(spy, copy.deepcopy(data), num_steps=1, prior_seed=seed)
        return spy, result

    spy_a, res_a = _run(_TE_SEED)
    spy_b, res_b = _run(_TE_SEED)
    spy_c, _res_c = _run(_TE_SEED + 1)

    # same seed => bit-identical injected D0 and bit-identical assembled output.
    assert torch.equal(spy_a.spatial_inputs[0][0], spy_b.spatial_inputs[0][0])
    assert torch.equal(spy_a.spatial_inputs[0][1], spy_b.spatial_inputs[0][1])
    assert torch.equal(
        res_a[_keys.NODE_PRED_HAMIL_BLOCKS_KEY], res_b[_keys.NODE_PRED_HAMIL_BLOCKS_KEY]
    )
    # D0 is the injected epsilon (nonzero), NOT the zero-prior boundary.
    assert torch.count_nonzero(spy_a.spatial_inputs[0][0]) > 0
    # different seed => different injected D0.
    assert not torch.equal(spy_a.spatial_inputs[0][0], spy_c.spatial_inputs[0][0])


def test_sampler_zero_prior_byte_identical_and_rejects_prior_seed():
    """9b: the zero-prior sampler is byte-unchanged (D0 == 0) and a prior_seed is
    rejected -- the exact-zero symmetry mirrors _block_initial_state."""
    mapper = _mapper()
    data, _h0, d1 = _b_record(mapper)
    flow = _b_flow(mapper)  # prior='zero'

    spy = _EndpointSpy(
        [(d1.node_blocks.clone(), d1.edge_blocks.clone())],
        flow.node_h0_key,
        flow.edge_h0_key,
    )
    flow.sample(spy, copy.deepcopy(data), num_steps=1)
    assert torch.count_nonzero(spy.spatial_inputs[0][0]) == 0
    assert torch.count_nonzero(spy.spatial_inputs[0][1]) == 0

    with pytest.raises(ValueError, match="prior_seed"):
        flow.sample(
            _EndpointSpy(
                [(d1.node_blocks.clone(), d1.edge_blocks.clone())],
                flow.node_h0_key,
                flow.edge_h0_key,
            ),
            copy.deepcopy(data),
            num_steps=1,
            prior_seed=7,
        )


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_sampler_te_h0_constant_and_exactly_once_assembly(steps):
    """9e: under projected_te the H0-constancy and exactly-once assembly locks
    still hold -- the h0 keys stay the physical H0 RME across the rollout, the
    injected D0 is the nonzero epsilon, and the final blocks equal H0 + D1 once
    (the last blend alpha == 1 collapses the accumulated state onto D1)."""
    mapper = _mapper()
    data, h0, d1 = _b_record(mapper)
    flow = _b_te_flow(mapper)
    node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)

    endpoints = [(d1.node_blocks.clone(), d1.edge_blocks.clone()) for _ in range(steps)]
    spy = _EndpointSpy(endpoints, flow.node_h0_key, flow.edge_h0_key)
    result = flow.sample(spy, copy.deepcopy(data), num_steps=steps, prior_seed=_TE_SEED)

    # projected_te injects a NONZERO D0 (contrast with the zero-prior spy test).
    assert torch.count_nonzero(spy.spatial_inputs[0][0]) > 0

    # H0-constancy across the rollout (contract-2 rollout lock), prior-independent.
    for step in range(steps):
        torch.testing.assert_close(spy.h0_inputs[step][0], node_base, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(spy.h0_inputs[step][1], edge_base, rtol=0.0, atol=1e-12)
    assert spy.times[0].reshape(-1)[0].item() == 0.0

    # exactly-once assembly: predicted full-H blocks == H0 + D1 (D_blend == D1).
    torch.testing.assert_close(
        result[_keys.NODE_PRED_HAMIL_BLOCKS_KEY],
        h0.node_blocks + d1.node_blocks,
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY],
        h0.edge_blocks + d1.edge_blocks,
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        torch.as_tensor(result[flow.node_h0_key]), node_base, rtol=0.0, atol=1e-10
    )
    assert torch.allclose(
        result[flow.flow_time_key], torch.ones_like(result[flow.flow_time_key])
    )


# ===========================================================================
# 10. Round-3 heavy tests (H1-H6): projected_te explicit latent (prior_state),
#     per-uid seeded substreams, the TA-3 draw belt, and the TA-4 zero-prior
#     exception ordering.  These exercise the P1 flow.py fix lane.
# ===========================================================================
class _LinearEchoModel(torch.nn.Module):
    """Trivially equivariant model double: endpoint = alpha * spatial-residual state.

    It reads the pure-D spatial residual block state the sampler attaches every
    step and echoes it (scaled) into the prediction keys.  Because the echo is a
    per-element scalar multiply in canvas-block space, it commutes with the shared
    -canvas Wigner-D conjugation ``B -> D B D^T``, so the WHOLE residual sampler
    pipeline (H0 blocks -> RME conditioning, verbatim D0, endpoint blend, assemble
    H = H0 + D once) is pathwise equivariant iff the sampler's block bookkeeping is.
    Contrast :class:`_EndpointSpy`, which returns a FIXED endpoint and so is not
    itself input-covariant.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, data):
        out = data.copy()
        out[_keys.NODE_PRED_HAMIL_BLOCKS_KEY] = self.alpha * data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY]
        out[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY] = self.alpha * data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY]
        return out


def _shared_canvas_wigner_d(rotation):
    """The shared H:1s / C:1s1p canvas Wigner-D (``1x0e+1x1o``).

    Reuses the compact-frame D-matrix logic of
    ``test_projector_water_basis_multishell_rotation_covariance``: H's single 1s
    lands on scalar slot 0 (its ``0e`` Wigner block is the identity 1), so one
    shared canvas D covers BOTH species and ``D @ B @ D^T`` preserves every block's
    row/col padding structure.
    """
    return o3.Irreps("1x0e+1x1o").D_from_matrix(rotation)


def _rotate_canvas_blocks(blocks, d_ao):
    """Conjugate every canvas block by the shared Wigner-D: ``B -> D B D^T``."""
    return d_ao @ blocks @ d_ao.transpose(-1, -2)


def _certified_latent(flow, data, h0, seed=_TE_SEED):
    """A valid transformable latent: the seeded projected_te eps (codec-image)."""
    node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)
    return flow._residual_te_eps(
        copy.deepcopy(data),
        node_base,
        edge_base,
        generator=flow._seeded_generator(node_base.device, seed),
        certify_image=True,
    )


def test_h1_prior_state_is_pathwise_equivariant_while_seeded_is_layout_replay():
    """H1: the explicit ``prior_state`` latent is PATHWISE equivariant; the seeded
    draw is only layout-replay.

    With the trivially-equivariant echo model, ``sample(R.x, prior_state=R.D0)``
    equals ``R.sample(x, prior_state=D0)`` on the returned prediction blocks (a
    rotated latent stays in the codec image and rides the input rotation), whereas
    ``sample(R.x, prior_seed=s)`` != ``R.sample(x, prior_seed=s)`` because the
    per-uid seeded eps is the SAME block draw for x and R.x (documenting that the
    seeded path is distributional/layout-replay, not pathwise equivariant).
    """
    mapper = _mapper()
    flow = _b_te_flow(mapper)  # fp64 projected_te B-mode flow
    data, h0, _d1 = _b_record(mapper)
    eps = _certified_latent(flow, data, h0)

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
            _LinearEchoModel(0.7), copy.deepcopy(data), num_steps=1, prior_seed=_TE_SEED
        )
        seed_rot = flow.sample(
            _LinearEchoModel(0.7), copy.deepcopy(rotated), num_steps=1, prior_seed=_TE_SEED
        )
        assert not torch.allclose(
            seed_rot[_keys.NODE_PRED_HAMIL_BLOCKS_KEY],
            _rotate_canvas_blocks(seed_base[_keys.NODE_PRED_HAMIL_BLOCKS_KEY], d_ao),
            atol=1e-6,
        )
    finally:
        torch.set_default_dtype(previous_default)


def test_h2_prior_state_validation_gates():
    """H2: prior_state is validated exactly like the seeded draw before use.

    shape mismatch -> raise; NaN -> raise; off-codec-image (broken onsite symmetry
    beyond atol) -> raise; prior_state+prior_seed -> raise (mutually exclusive);
    prior_state under a zero prior -> raise; and any valid codec-image latent
    (including a second independently drawn one) is accepted.
    """
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    data, h0, _d1 = _b_record(mapper)
    eps = _certified_latent(flow, data, h0)

    # valid latent accepted (produces a full-H prediction).
    accepted = flow.sample(_LinearEchoModel(0.5), copy.deepcopy(data), num_steps=1, prior_state=eps)
    assert _keys.NODE_PRED_HAMIL_BLOCKS_KEY in accepted
    # a SECOND, independently-drawn valid codec-image latent is equally accepted.
    eps2 = _certified_latent(flow, data, h0, seed=_TE_SEED + 5)
    assert not torch.equal(eps.node_blocks, eps2.node_blocks)
    flow.sample(
        _LinearEchoModel(0.5),
        copy.deepcopy(data),
        num_steps=1,
        prior_state=(eps2.node_blocks.clone(), eps2.edge_blocks.clone()),
    )

    # shape mismatch (node canvas trimmed) -> raise naming the shape.
    with pytest.raises(ValueError, match="shape"):
        flow.sample(
            _LinearEchoModel(0.5),
            copy.deepcopy(data),
            num_steps=1,
            prior_state=BlockTensorResult(
                eps.node_blocks[:, :3, :3].clone(), eps.edge_blocks.clone(), eps.node_shapes, eps.edge_shapes
            ),
        )

    # NaN in the latent -> raise (finiteness enforced).
    nan_node = eps.node_blocks.clone()
    nan_node[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        flow.sample(
            _LinearEchoModel(0.5),
            copy.deepcopy(data),
            num_steps=1,
            prior_state=(nan_node, eps.edge_blocks.clone()),
        )

    # off-codec-image: break the C onsite block's symmetry beyond atol -> raise.
    asym_node = eps.node_blocks.clone()
    asym_node[1, 0, 1] = asym_node[1, 0, 1] + 3.0
    with pytest.raises(ValueError, match="onsite|codec image"):
        flow.sample(
            _LinearEchoModel(0.5),
            copy.deepcopy(data),
            num_steps=1,
            prior_state=(asym_node, eps.edge_blocks.clone()),
        )

    # prior_state AND prior_seed together -> mutually exclusive.
    with pytest.raises(ValueError, match="mutually exclusive"):
        flow.sample(
            _LinearEchoModel(0.5), copy.deepcopy(data), num_steps=1, prior_state=eps, prior_seed=3
        )

    # prior_state under a ZERO prior -> rejected.
    with pytest.raises(ValueError, match="prior_state"):
        _b_flow(mapper).sample(
            _LinearEchoModel(0.5), copy.deepcopy(data), num_steps=1, prior_state=eps
        )


def _node_te_draw(flow, dim, *, types, batch, uids, seed, device):
    """Draw the per-uid seeded node TE noise for a hand-built (collated) batch.

    Exercises the internal seeded substream path (``_te_prior_like`` ->
    ``_prior_substream_generators`` -> ``_seeded_rows_by_graph``) directly with a
    hand-built batch dict, avoiding the heavy public collation plumbing while still
    driving the exact code the batch-composition invariance lives in.
    """
    like = torch.zeros(len(batch), dim, dtype=torch.float64)
    payload = {
        _keys.ATOM_TYPE_KEY: torch.tensor([[t] for t in types], dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor(batch, dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0], [0]], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.tensor([[0]], dtype=torch.long),
        _keys.SAMPLE_UID_KEY: torch.tensor(uids, dtype=torch.long),
    }
    return flow._te_prior_like(
        like,
        flow.node_sigma,
        data=payload,
        label="node",
        reference_scale=False,
        num_graphs=len(uids),
        generator=flow._seeded_generator(device, seed),
    )


def test_h3_seeded_draw_is_batch_composition_invariant_per_uid():
    """H3: a graph's seeded eps rows are invariant to batch composition/order.

    graph A's node epsilon rows are byte-identical whether A is drawn alone, first
    in ``[A, B]``, or second in ``[B, A]`` (distinct sample_uids); a different uid
    yields different rows; the node substream is independent of the edge substream
    (it never reads the edge structure); and a SEEDED draw missing sample_uid fails
    closed.
    """
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    data_a, h0_a, _ = _b_record(mapper, seed=0)
    node_base, _edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data_a), h0_a)
    dim = int(node_base.shape[-1])
    device = node_base.device
    t_h = mapper.chemical_symbol_to_type["H"]
    t_c = mapper.chemical_symbol_to_type["C"]
    uid_a, uid_b, uid_c = 11, 22, 33

    a_alone = _node_te_draw(flow, dim, types=[t_h, t_c], batch=[0, 0], uids=[uid_a], seed=_TE_SEED, device=device)
    ab = _node_te_draw(
        flow, dim, types=[t_h, t_c, t_h, t_c], batch=[0, 0, 1, 1], uids=[uid_a, uid_b], seed=_TE_SEED, device=device
    )
    ba = _node_te_draw(
        flow, dim, types=[t_h, t_c, t_h, t_c], batch=[0, 0, 1, 1], uids=[uid_b, uid_a], seed=_TE_SEED, device=device
    )
    # A's rows identical whether alone, first in [A,B], or second in [B,A].
    assert torch.equal(ab[0:2], a_alone)
    assert torch.equal(ba[2:4], a_alone)
    assert torch.count_nonzero(a_alone) > 0

    # A different uid draws different rows (same shapes/types).
    c_alone = _node_te_draw(flow, dim, types=[t_h, t_c], batch=[0, 0], uids=[uid_c], seed=_TE_SEED, device=device)
    assert not torch.equal(c_alone, a_alone)

    # Node substream independent of edge structure: the node draw never reads the
    # edge count, so A's node rows are unchanged if the co-batched graph B's edge
    # topology changes (proven here by the node draw ignoring EDGE_INDEX entirely).
    ab_more_edges = _node_te_draw(
        flow, dim, types=[t_h, t_c, t_h, t_c], batch=[0, 0, 1, 1], uids=[uid_a, uid_b], seed=_TE_SEED, device=device
    )
    assert torch.equal(ab_more_edges[0:2], a_alone)

    # A SEEDED draw without the per-graph identity fails closed.
    like = torch.zeros(2, dim, dtype=torch.float64)
    no_uid = {
        _keys.ATOM_TYPE_KEY: torch.tensor([[t_h], [t_c]], dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor([0, 0], dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0], [0]], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.tensor([[0]], dtype=torch.long),
    }
    with pytest.raises(ValueError, match=_keys.SAMPLE_UID_KEY):
        flow._te_prior_like(
            like,
            flow.node_sigma,
            data=no_uid,
            label="node",
            reference_scale=False,
            num_graphs=1,
            generator=flow._seeded_generator(device, 5),
        )


def test_h3b_validation_prior_base_seed_is_batch_index_independent():
    """H3b: the validation caller's prior base seed does NOT depend on batch position.

    H3 proves a uid's epsilon is invariant to batch composition *given a fixed base
    seed*.  This pins the other half of the contract -- the base seed the real
    ``Trainer.validation()`` / ``MultiTrainer.validation()`` loops thread into that
    substream -- to ``validation_prior_base_seed()``, which is a pure function of
    ``te_prior_validation_seed`` and independent of the running batch index.  The two
    together give per-record replay across re-batching/resharding.  Regression for the
    prior caller that used ``validation_seed(num_batches, "prior")`` (batch-indexed).
    """
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    data_a, h0_a, _ = _b_record(mapper, seed=0)
    node_base, _ = flow.block_codec.blocks_to_rme(copy.deepcopy(data_a), h0_a)
    dim = int(node_base.shape[-1])
    device = node_base.device
    t_h = mapper.chemical_symbol_to_type["H"]
    t_c = mapper.chemical_symbol_to_type["C"]

    # The base seed is constant and independent of any batch index ...
    base = flow.validation_prior_base_seed()
    assert base == flow.validation_prior_base_seed()
    # ... whereas validation_seed() itself IS batch-indexed (the old caller's bug).
    assert base != flow.validation_seed(1, "prior")
    assert base != flow.validation_seed(7, "prior")
    assert flow.validation_seed(1, "prior") != flow.validation_seed(7, "prior")

    # Threading the constant base seed through the real substream path reproduces a
    # uid's rows no matter which validation batch position it lands in.
    draw_new_a = _node_te_draw(flow, dim, types=[t_h, t_c], batch=[0, 0], uids=[42], seed=base, device=device)
    draw_new_b = _node_te_draw(flow, dim, types=[t_h, t_c], batch=[0, 0], uids=[42], seed=base, device=device)
    assert torch.equal(draw_new_a, draw_new_b)
    assert torch.count_nonzero(draw_new_a) > 0

    # Contrast: had the caller kept mixing the batch index into the base seed, the
    # SAME record would have been redrawn when it moved between validation batches.
    old_pos0 = _node_te_draw(
        flow, dim, types=[t_h, t_c], batch=[0, 0], uids=[42],
        seed=flow.validation_seed(0, "prior"), device=device,
    )
    old_pos1 = _node_te_draw(
        flow, dim, types=[t_h, t_c], batch=[0, 0], uids=[42],
        seed=flow.validation_seed(1, "prior"), device=device,
    )
    assert not torch.equal(old_pos0, old_pos1)


def test_h4_seeded_multi_graph_determinism_and_global_rng_isolation():
    """H4: two same-seed draws of a 2-graph batch are byte-identical and the global
    torch RNG is untouched (the per-uid substreams use private generators)."""
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    data_a, h0_a, _ = _b_record(mapper, seed=0)
    node_base, _ = flow.block_codec.blocks_to_rme(copy.deepcopy(data_a), h0_a)
    dim = int(node_base.shape[-1])
    device = node_base.device
    t_h = mapper.chemical_symbol_to_type["H"]
    t_c = mapper.chemical_symbol_to_type["C"]

    rng_before = torch.random.get_rng_state()
    first = _node_te_draw(
        flow, dim, types=[t_h, t_c, t_h, t_c], batch=[0, 0, 1, 1], uids=[11, 22], seed=123, device=device
    )
    second = _node_te_draw(
        flow, dim, types=[t_h, t_c, t_h, t_c], batch=[0, 0, 1, 1], uids=[11, 22], seed=123, device=device
    )
    rng_after = torch.random.get_rng_state()

    assert torch.equal(first, second)
    assert torch.equal(rng_before, rng_after)


def test_h5_ta3_draw_belt_rejects_collapsed_and_nonfinite_draws():
    """H5: the TA-3 runtime belt rejects an all-zero draw under a nonzero effective
    scale and a NaN draw, while a zero-scale component passes."""
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    zeros = torch.zeros(2, 4, 4, dtype=torch.float64)
    nan = zeros.clone()
    nan[0, 0, 0] = float("nan")
    live = torch.randn(2, 4, 4, dtype=torch.float64)

    # all-zero draw under a nonzero scale -> ValueError naming scale/dtype/component.
    with pytest.raises(ValueError) as excinfo:
        flow._assert_projected_te_draw_finite_and_scaled((("node", zeros, 1.0),))
    message = str(excinfo.value)
    assert "node" in message and "scale" in message and "float64" in message

    # NaN draw -> ValueError.
    with pytest.raises(ValueError, match="NaN"):
        flow._assert_projected_te_draw_finite_and_scaled((("edge", nan, 1.0),))

    # a genuinely nonzero draw and a zero-scale component both pass.
    flow._assert_projected_te_draw_finite_and_scaled((("node", live, 1.0),))
    flow._assert_projected_te_draw_finite_and_scaled((("node", zeros, 0.0),))


def test_h5_ta3_belt_fires_through_residual_eps_draw_path(monkeypatch):
    """H5: the belt is wired INTO the eps draw path -- a collapsed (all-zero) inner
    draw and a NaN inner draw each surface as a ValueError from _residual_te_eps."""
    mapper = _mapper()
    flow = _b_te_flow(mapper)
    data, h0, _ = _b_record(mapper)
    node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)

    monkeypatch.setattr(flow, "_te_prior_like", lambda like, *a, **k: torch.zeros_like(like))
    with pytest.raises(ValueError, match="zero"):
        flow._residual_te_eps(
            copy.deepcopy(data), node_base, edge_base,
            generator=flow._seeded_generator(node_base.device, _TE_SEED), certify_image=True,
        )

    def _nan_like(like, *a, **k):
        out = torch.zeros_like(like)
        if out.numel():
            out.reshape(-1)[0] = float("nan")
        return out

    monkeypatch.setattr(flow, "_te_prior_like", _nan_like)
    with pytest.raises(ValueError, match="NaN"):
        flow._residual_te_eps(
            copy.deepcopy(data), node_base, edge_base,
            generator=flow._seeded_generator(node_base.device, _TE_SEED), certify_image=True,
        )


def test_h6_ta4_zero_prior_prior_seed_rejection_precedes_contract_check():
    """H6 (reviewer's exact repro): a zero-prior residual flow given data MISSING an
    H0 block key but called with ``prior_seed`` fails with the prior_seed ValueError,
    NOT the contract KeyError -- proving the early zero-prior rejection in sample()
    precedes the data-contract check inside the dispatched sampler."""
    mapper = _mapper()
    flow = _b_flow(mapper)  # prior='zero'
    data, _h0, _d1 = _b_record(mapper)
    del data[_keys.NODE_H0_BLOCKS_KEY]  # break the data contract

    # prior_seed given: the zero-prior rejection fires FIRST (ValueError, not KeyError).
    with pytest.raises(ValueError, match="prior_seed"):
        flow.sample(_LinearEchoModel(0.5), copy.deepcopy(data), num_steps=1, prior_seed=7)

    # Contrast: without prior_seed the SAME missing-H0 data reaches the contract
    # check and raises KeyError -- so the ordering above is load-bearing.
    with pytest.raises(KeyError):
        flow.sample(_LinearEchoModel(0.5), copy.deepcopy(data), num_steps=1)

"""Bit-exact rollout characterization test for the block-ODE sampler refactor.

PR5b (``refactor/block-ode-rollout-engine``) extracts the shared Euler main loop
of the three block-ODE samplers (``_sample_block_ode`` / ``_sample_uureal_block_ode``
/ ``_sample_residual_ao_block_ode``) into a single engine
(``dptb.nnops.block_ode.rollout``) parameterized by three route adapters.  That is
a pure code-motion: no floating-point operation is added, removed, or reordered,
so the samplers must return **bit-identical** ``node_hamil_blocks`` /
``edge_hamil_blocks`` before and after the refactor.

This module freezes that guarantee.  ``_run_route`` builds one fixed fixture batch
per route (a fixed ``prior_seed`` and ``num_steps=3``) and runs ``sample``; the
returned prediction blocks are compared with :func:`torch.equal` (bit-for-bit, not
``allclose``) against a golden captured from the pre-refactor code and committed
alongside this file.

The fixtures are deliberately self-contained (minimal copies of the proven
builders in ``test_block_ode_flow.py`` / ``test_uureal_block_ode.py`` /
``test_residual_ao_block_ode.py``) so this oracle does not depend on the evolution
of those sibling test modules; the three routes together exercise every
route-specific sliver the engine is parameterized by:

    * full-H     -- ``_block_initial_state`` (zero prior), the ``endpoint_to_full``
                    decode unique to this route, per-step ``blocks_to_rme``
                    state-write-in, and LEM-sidecar reinjection/stripping;
    * uureal     -- the compact-uu zero initial state, ``_attach_uureal_residual_state``
                    write-in, and the pure-residual endpoint decode;
    * residual   -- the projected_te STOCHASTIC initial state
                    (``_residual_stochastic_eps`` + ``_seeded_generator``), the
                    per-step physical-H0 re-assert, ``_attach_spatial_residual_state``,
                    and the H = H0 + D exactly-once finalize.

Regenerate the golden ONLY from a known-good baseline::

    python dptb/tests/test_block_ode_rollout_bitexact.py
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_tensors_to_feature_tensors,
    infer_block_shapes,
    mapper_max_norb,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state
from dptb.nnops.flow import HamiltonianCFM


GOLDEN_PATH = Path(__file__).with_name("test_block_ode_rollout_bitexact_golden.pt")
_GLOBAL_SEED = 20260722
_TE_SEED = 20260720
_ROUTES = ("full_h", "uureal", "residual")


# ---------------------------------------------------------------------------
# full-H route (absolute_full_h ao_block_ode, zero prior)
# ---------------------------------------------------------------------------
class _FullHEndpoint(torch.nn.Module):
    """Input-sensitive full-H endpoint: read the updated H0 RME, emit gain*RME blocks."""

    def __init__(self, codec, gain):
        super().__init__()
        self.codec = codec
        self.gain = float(gain)

    def forward(self, data):
        node = data[_keys.NODE_H0_KEY]
        edge = data[_keys.EDGE_H0_KEY]
        endpoint = self.codec.rme_to_blocks(
            data, node * self.gain, edge * self.gain, project=True
        )
        out = data.copy()
        out[_keys.NODE_PRED_HAMIL_BLOCKS_KEY] = endpoint.node_blocks
        out[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY] = endpoint.edge_blocks
        return out


def _full_h_case():
    idp = OrbitalMapper({"H": ["1s"], "C": ["2p"]}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    atom_types = torch.tensor(
        [idp.chemical_symbol_to_type["H"], idp.chemical_symbol_to_type["C"]]
    )
    edge_types = torch.tensor([idp.bond_to_type["H-C"], idp.bond_to_type["C-H"]])
    data = {
        "pos": torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]], dtype=torch.float64),
        "cell": torch.eye(3, dtype=torch.float64) * 5.0,
        "batch": torch.zeros(2, dtype=torch.long),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64
        ),
        _keys.PBC_KEY: torch.tensor([True, False, False]),
        "atom_types": atom_types,
        _keys.EDGE_TYPE_KEY: edge_types,
    }
    codec = BlockStateCodec(idp, dtype=torch.float64)
    raw = BlockTensorResult(
        torch.randn(2, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(11)),
        torch.randn(2, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(12)),
        torch.tensor([[1, 1], [3, 3]]),
        torch.tensor([[1, 3], [3, 1]]),
    )
    h0_blocks = project_block_state(data, idp, raw)
    node_h0, edge_h0 = codec.blocks_to_rme(data, h0_blocks)
    data["node_h0"] = node_h0
    data["edge_h0"] = edge_h0
    data[_keys.NODE_H0_BLOCKS_KEY] = h0_blocks.node_blocks.clone()
    data[_keys.EDGE_H0_BLOCKS_KEY] = h0_blocks.edge_blocks.clone()
    data[_keys.NODE_H0_BLOCK_SHAPE_KEY] = h0_blocks.node_shapes.clone()
    data[_keys.EDGE_H0_BLOCK_SHAPE_KEY] = h0_blocks.edge_shapes.clone()
    data["node_features"] = node_h0.clone()
    data["edge_features"] = edge_h0.clone()
    data[_keys.SAMPLE_UID_KEY] = torch.tensor([1], dtype=torch.long)
    return idp, data, codec


def _full_h_flow(idp):
    options = {
        "enabled": True,
        "objective": "cfm",
        "mode": "residual",
        "prior": "zero",
        "output_space": "ao_block_ode",
        "block_ode": True,
        "target_semantics": "absolute_full_h",
        "prediction_add_h0": False,
        "time_conditioning_required": True,
        "block_inverse_mode": "strict",
        "block_inverse_atol": 1e-10,
        "validation_ode_steps": [1, 3],
        "te_prior_validation_seed": 20260719,
        "node_block_target_key": "node_full_hamil_target_blocks",
        "edge_block_target_key": "edge_full_hamil_target_blocks",
        "node_block_shape_key": "node_full_hamil_target_block_shape",
        "edge_block_shape_key": "edge_full_hamil_target_block_shape",
    }
    return HamiltonianCFM(options, idp=idp, dtype=torch.float64)


def _run_full_h():
    idp, data, codec = _full_h_case()
    metadata = {
        _keys.LEM_ACTIVE_EDGES_KEY: torch.tensor([0, 1], dtype=torch.long),
        _keys.LEM_CUTOFF_COEFFS_KEY: torch.tensor([0.75, 0.5], dtype=torch.float64),
        _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY: torch.tensor([2], dtype=torch.long),
    }
    rollout_input = {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in data.items()
    }
    rollout_input.update(metadata)
    model = _FullHEndpoint(codec, gain=0.7)
    return _full_h_flow(idp).sample(model, rollout_input, num_steps=3)


# ---------------------------------------------------------------------------
# uureal route (uureal_block_ode, zero prior)
# ---------------------------------------------------------------------------
class _UuRealEndpointSpy(torch.nn.Module):
    def __init__(self, endpoints):
        super().__init__()
        self.endpoints = endpoints
        self.inputs = []

    def forward(self, data):
        self.inputs.append(
            (
                data[_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY].clone(),
                data[_keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY].clone(),
            )
        )
        node, edge = self.endpoints[len(self.inputs) - 1]
        out = data.copy()
        out[_keys.NODE_PRED_HAMIL_BLOCKS_KEY] = node.clone()
        out[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY] = edge.clone()
        return out


def _uureal_mapper():
    mapper = OrbitalMapper(
        {"H": "1s", "C": "1s1p"},
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=True,
        full_soc_prediction=False,
    )
    mapper.get_irreps()
    return mapper


def _uureal_record(mapper):
    data = {
        "atomic_numbers": torch.tensor([1, 6]),
        "atom_types": torch.tensor(
            [[mapper.chemical_symbol_to_type["H"]], [mapper.chemical_symbol_to_type["C"]]]
        ),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.zeros(2, 3),
        "edge_type": torch.tensor(
            [[mapper.bond_to_type["H-C"]], [mapper.bond_to_type["C-H"]]]
        ),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "cell": torch.eye(3) * 8.0,
        "pbc": torch.tensor([False, False, False]),
        "batch": torch.zeros(2, dtype=torch.long),
    }
    node = torch.zeros(2, 4, 4)
    node[0, 0, 0] = 0.25
    c = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 100.0
    node[1] = 0.5 * (c + c.T)
    edge = torch.zeros(2, 4, 4)
    edge[0, :1, :4] = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    edge[1, :4, :1] = edge[0, :1, :4].T
    packed = BlockTensorResult(
        node, edge, torch.tensor([[1, 1], [4, 4]]), torch.tensor([[1, 4], [4, 1]])
    )
    node_h0, edge_h0 = block_tensors_to_feature_tensors(
        data, mapper, node_blocks=node * 0.5, edge_blocks=edge * 0.5
    )
    keep = int(mapper.reduced_matrix_element)
    data.update(
        {
            "node_h0": node_h0,
            "edge_h0": edge_h0,
            "node_delta_hamil_blocks": packed.node_blocks,
            "edge_delta_hamil_blocks": packed.edge_blocks,
            "node_delta_hamil_block_shape": packed.node_shapes,
            "edge_delta_hamil_block_shape": packed.edge_shapes,
            "blockwise_spatial_schema": "deeptb.blockwise_spatial/v1",
            "blockwise_target_mode": "already-delta",
            "blockwise_source_target_feature_width": keep,
            "blockwise_source_h0_feature_width": keep,
            "soc_uureal_compact": True,
            "soc_uureal_full_rme": keep * 8,
            "soc_uureal_keep": keep,
        }
    )
    return data


def _uureal_flow(mapper):
    options = {
        "enabled": True,
        "mode": "residual",
        "prior": "zero",
        "output_space": "uureal_block_ode",
        "block_ode": True,
        "state_space": "residual_ao_block",
        "target_semantics": "residual_dh",
        "block_input_adapter": "direct_cg",
        "h0_condition_space": "compact_uureal_rme",
        "block_export_final_full_h": False,
        "prediction_add_h0": False,
        "time_conditioning_required": True,
        "node_block_target_key": "node_delta_hamil_blocks",
        "edge_block_target_key": "edge_delta_hamil_blocks",
        "node_block_shape_key": "node_delta_hamil_block_shape",
        "edge_block_shape_key": "edge_delta_hamil_block_shape",
        "validation_ode_steps": [1, 3],
    }
    return HamiltonianCFM(options, idp=mapper, dtype=torch.float32)


def _run_uureal():
    mapper = _uureal_mapper()
    data = _uureal_record(mapper)
    node = data["node_delta_hamil_blocks"]
    edge = data["edge_delta_hamil_blocks"]
    endpoints = [(node * (i + 1), edge * (i + 1)) for i in range(3)]
    model = _UuRealEndpointSpy(endpoints)
    fresh = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in data.items()}
    return _uureal_flow(mapper).sample(model, fresh, num_steps=3)


# ---------------------------------------------------------------------------
# residual route (residual_ao_block_ode, projected_te stochastic prior)
# ---------------------------------------------------------------------------
class _ResidualEndpointSpy(torch.nn.Module):
    def __init__(self, endpoints, node_h0_key, edge_h0_key):
        super().__init__()
        self.endpoints = endpoints
        self.spatial_inputs = []
        self._node_h0_key = node_h0_key
        self._edge_h0_key = edge_h0_key

    def forward(self, data):
        self.spatial_inputs.append(
            (
                data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY].clone(),
                data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY].clone(),
            )
        )
        node, edge = self.endpoints[len(self.spatial_inputs) - 1]
        out = data.copy()
        out[_keys.NODE_PRED_HAMIL_BLOCKS_KEY] = node.clone()
        out[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY] = edge.clone()
        return out


def _residual_mapper():
    mapper = OrbitalMapper({"H": "1s", "C": "1s1p"}, method="e3tb")
    mapper.get_irreps()
    return mapper


def _residual_graph(mapper, *, dtype=torch.float64):
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
        _keys.SAMPLE_UID_KEY: torch.tensor([1], dtype=torch.long),
    }


def _residual_projected_state(mapper, data, *, canvas, n, e, dtype, seed):
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


def _residual_record(mapper, *, dtype=torch.float64, seed=0):
    data = _residual_graph(mapper, dtype=dtype)
    node_shapes, edge_shapes = infer_block_shapes(data, mapper)
    canvas = mapper_max_norb(mapper)
    n = int(node_shapes.shape[0])
    e = int(edge_shapes.shape[0])
    h0 = _residual_projected_state(mapper, data, canvas=canvas, n=n, e=e, dtype=dtype, seed=seed)
    d1 = _residual_projected_state(
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


def _residual_te_flow(mapper, *, dtype=torch.float64, seed=_TE_SEED):
    options = {
        "enabled": True,
        "objective": "cfm",
        "mode": "residual",
        "prior": "projected_te",
        "te_prior_mode": "irrep",
        "node_sigma": 1.0,
        "edge_sigma": 1.0,
        "te_prior_sigma": 1.0,
        "te_prior_validation_seed": seed,
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
        "block_inverse_atol": 1e-10,
        "strict_certification": "always",
        "node_block_target_key": "node_delta_hamil_blocks",
        "edge_block_target_key": "edge_delta_hamil_blocks",
        "node_block_shape_key": "node_delta_hamil_block_shape",
        "edge_block_shape_key": "edge_delta_hamil_block_shape",
        "validation_ode_steps": [1],
    }
    return HamiltonianCFM(options, idp=mapper, dtype=dtype)


def _run_residual():
    mapper = _residual_mapper()
    data, _h0, d1 = _residual_record(mapper)
    flow = _residual_te_flow(mapper)
    endpoints = [(d1.node_blocks.clone(), d1.edge_blocks.clone()) for _ in range(3)]
    model = _ResidualEndpointSpy(endpoints, flow.node_h0_key, flow.edge_h0_key)
    fresh = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in data.items()}
    return flow.sample(model, fresh, num_steps=3, prior_seed=_TE_SEED)


_RUNNERS = {
    "full_h": _run_full_h,
    "uureal": _run_uureal,
    "residual": _run_residual,
}


def _run_route(route):
    """Deterministically build the route's fixture and return its prediction blocks.

    Used identically by the pytest comparison and the ``__main__`` capture, so the
    golden and the checked value are produced by one code path (only the sampler
    implementation underneath differs pre- vs. post-refactor)."""
    torch.manual_seed(_GLOBAL_SEED)
    result = _RUNNERS[route]()
    return {
        "node": result[_keys.NODE_PRED_HAMIL_BLOCKS_KEY].detach().cpu().clone(),
        "edge": result[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY].detach().cpu().clone(),
    }


@pytest.mark.parametrize("route", _ROUTES)
def test_rollout_is_bit_identical_to_pre_refactor_golden(route):
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"missing rollout golden {GOLDEN_PATH.name}; regenerate with "
            "`python dptb/tests/test_block_ode_rollout_bitexact.py` from a known-good baseline."
        )
    golden = torch.load(GOLDEN_PATH, weights_only=True)
    got = _run_route(route)
    # torch.equal, NOT allclose: a pure code-motion must be bit-for-bit identical;
    # any mismatch means the shared-loop abstraction reordered a float operation.
    assert torch.equal(got["node"], golden[route]["node"]), f"{route}: node blocks diverged"
    assert torch.equal(got["edge"], golden[route]["edge"]), f"{route}: edge blocks diverged"


if __name__ == "__main__":
    golden = {route: _run_route(route) for route in _ROUTES}
    torch.save(golden, GOLDEN_PATH)
    for route in _ROUTES:
        print(
            f"captured {route}: "
            f"node{tuple(golden[route]['node'].shape)} edge{tuple(golden[route]['edge'].shape)}"
        )
    print("wrote", GOLDEN_PATH)

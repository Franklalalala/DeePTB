"""Shared block-ODE test builders: mappers, graphs, records, flows, contract configs, model doubles.

Not collected by pytest; import as ``from dptb.tests.block_ode_fixtures import ...``."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml
from e3nn import o3

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_mask_from_shapes,
    block_tensors_to_feature_tensors,
    canonical_block_tensors_to_feature_tensors,
    infer_block_shapes,
    mapper_max_norb,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.block_flow_codec import BlockStateCodec, project_block_state
from dptb.nnops.flow import HamiltonianCFM

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
B_CONFIG = "h_b0_block_ode_water_residual.yaml"
TE_CONFIG = "h_b0_block_ode_water_residual_te.yaml"

FP64_ATOL = 1e-10

FULL_H_TARGET_FIELDS = {
    "node_block_target_key": "node_full_hamil_target_blocks",
    "edge_block_target_key": "edge_full_hamil_target_blocks",
    "node_block_shape_key": "node_full_hamil_target_block_shape",
    "edge_block_shape_key": "edge_full_hamil_target_block_shape",
}
RESIDUAL_TARGET_FIELDS = {
    "node_block_target_key": "node_delta_hamil_blocks",
    "edge_block_target_key": "edge_delta_hamil_blocks",
    "node_block_shape_key": "node_delta_hamil_block_shape",
    "edge_block_shape_key": "edge_delta_hamil_block_shape",
}
DELTA_LABEL_KEYS = (
    _keys.NODE_DELTA_HAMIL_BLOCKS_KEY,
    _keys.EDGE_DELTA_HAMIL_BLOCKS_KEY,
    _keys.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    _keys.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
)
SPATIAL_STATE_KEYS = (_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY, _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY)
UUREAL_STATE_KEYS = (_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY, _keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY)


def _fresh(data):
    """Clone every tensor of a record so the callee cannot mutate the caller's copy."""
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}


def _load_config(name):
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _load_b_config():
    """The frozen ``residual_ao_block_ode`` B-arm training config (water-residual yaml)."""
    return _load_config(B_CONFIG)


def _load_te_config():
    """The ``residual_ao_block_ode`` projected_te-arm training config (water-residual-te yaml)."""
    return _load_config(TE_CONFIG)


def _mutate(cfg, path, value):
    cfg = copy.deepcopy(cfg)
    cursor = cfg
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return cfg


# ---------------------------------------------------------------------------
# Mappers
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


def _uureal_mapper(basis=None):
    """A compact uu-real SOC mapper (default H 1s / C 1s1p)."""
    mapper = OrbitalMapper(
        basis or {"H": "1s", "C": "1s1p"},
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=True,
        full_soc_prediction=False,
    )
    mapper.get_irreps()
    return mapper


# ---------------------------------------------------------------------------
# Generic full-H block-ODE route (output_space='ao_block_ode')
# ---------------------------------------------------------------------------
def _case():
    """H 1s / C 2p periodic dimer with physical H0 blocks and their coupled RME."""
    idp = OrbitalMapper({"H": ["1s"], "C": ["2p"]}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    data = {
        "pos": torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]], dtype=torch.float64),
        "cell": torch.eye(3, dtype=torch.float64) * 5.0,
        "batch": torch.zeros(2, dtype=torch.long),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64),
        _keys.PBC_KEY: torch.tensor([True, False, False]),
        "atom_types": torch.tensor([idp.chemical_symbol_to_type["H"], idp.chemical_symbol_to_type["C"]]),
        _keys.EDGE_TYPE_KEY: torch.tensor([idp.bond_to_type["H-C"], idp.bond_to_type["C-H"]]),
        _keys.SAMPLE_UID_KEY: torch.tensor([1], dtype=torch.long),
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
    _attach_h0(data, h0_blocks, node_h0, edge_h0)
    return idp, data, codec, h0_blocks


def _attach_h0(data, h0_blocks, node_rme, edge_rme):
    data[_keys.NODE_H0_KEY] = node_rme
    data[_keys.EDGE_H0_KEY] = edge_rme
    data[_keys.NODE_FEATURES_KEY] = node_rme.clone()
    data[_keys.EDGE_FEATURES_KEY] = edge_rme.clone()
    data[_keys.NODE_H0_BLOCKS_KEY] = h0_blocks.node_blocks.clone()
    data[_keys.EDGE_H0_BLOCKS_KEY] = h0_blocks.edge_blocks.clone()
    data[_keys.NODE_H0_BLOCK_SHAPE_KEY] = h0_blocks.node_shapes.clone()
    data[_keys.EDGE_H0_BLOCK_SHAPE_KEY] = h0_blocks.edge_shapes.clone()


def _pd_legacy_product_h0_case():
    """C 2p / Si 3d dimer whose legacy H0 feature keys hold AO-product (not coupled) values.

    Returns ``(idp, data, codec, h0_blocks, canonical_node_h0, canonical_edge_h0)``; the
    authoritative physical H0 lives only in the H0 block keys.
    """
    idp = OrbitalMapper({"C": ["2p"], "Si": ["3d"]}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    data = {
        _keys.POSITIONS_KEY: torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.2, 0.1]], dtype=torch.float64),
        _keys.CELL_KEY: torch.eye(3, dtype=torch.float64).unsqueeze(0) * 6.0,
        _keys.PBC_KEY: torch.tensor([True, False, False]),
        _keys.BATCH_KEY: torch.zeros(2, dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.tensor([[1, 0, 0], [-1, 0, 0]], dtype=torch.long),
        _keys.ATOM_TYPE_KEY: torch.tensor([idp.chemical_symbol_to_type["C"], idp.chemical_symbol_to_type["Si"]]),
        _keys.EDGE_TYPE_KEY: torch.tensor([idp.bond_to_type["C-Si"], idp.bond_to_type["Si-C"]]),
        _keys.SAMPLE_UID_KEY: torch.tensor([1], dtype=torch.long),
    }
    generator = torch.Generator().manual_seed(20260719)
    raw = BlockTensorResult(
        node_blocks=torch.randn(2, 5, 5, dtype=torch.float64, generator=generator),
        edge_blocks=torch.randn(2, 5, 5, dtype=torch.float64, generator=generator),
        node_shapes=torch.tensor([[3, 3], [5, 5]], dtype=torch.long),
        edge_shapes=torch.tensor([[3, 5], [5, 3]], dtype=torch.long),
    )
    h0_blocks = project_block_state(data, idp, raw)
    codec = BlockStateCodec(idp, dtype=torch.float64)
    canonical_node_h0, canonical_edge_h0 = codec.blocks_to_rme(data, h0_blocks)
    gathered = canonical_block_tensors_to_feature_tensors(
        data,
        idp,
        node_blocks=h0_blocks.node_blocks,
        edge_blocks=h0_blocks.edge_blocks,
        node_shapes=h0_blocks.node_shapes,
        edge_shapes=h0_blocks.edge_shapes,
        mode="strict",
        atol=FP64_ATOL,
    )
    # The p/d change of basis is nontrivial, so product coordinates differ from coupled RME.
    assert (gathered.node_features - canonical_node_h0).abs().max().item() > 1.0e-3
    assert (gathered.edge_features - canonical_edge_h0).abs().max().item() > 1.0e-3
    _attach_h0(data, h0_blocks, gathered.node_features.clone(), gathered.edge_features.clone())
    return idp, data, codec, h0_blocks, canonical_node_h0, canonical_edge_h0


def _flow(idp, semantics="absolute_full_h", **updates):
    """A float64 generic block-ODE flow with full-H or residual block targets."""
    options = {
        "enabled": True,
        "objective": "cfm",
        "mode": "residual",
        "prior": "zero",
        "output_space": "ao_block_ode",
        "block_ode": True,
        "target_semantics": semantics,
        "prediction_add_h0": False,
        "time_conditioning_required": True,
        "block_inverse_mode": "strict",
        "block_inverse_atol": FP64_ATOL,
        "validation_ode_steps": [1, 3],
        "te_prior_validation_seed": 20260719,
        **(FULL_H_TARGET_FIELDS if semantics == "absolute_full_h" else RESIDUAL_TARGET_FIELDS),
    }
    options.update(updates)
    return HamiltonianCFM(options, idp=idp, dtype=torch.float64)


def _scaled_endpoint(codec, data, node_rme, edge_rme, scale):
    return codec.rme_to_blocks(data, node_rme * scale, edge_rme * scale, project=True)


def _ref_for(flow, data, endpoint, node_rme, edge_rme):
    """A reference record carrying the endpoint blocks and legacy endpoint features."""
    ref = _fresh(data)
    ref[flow.node_target_key] = node_rme
    ref[flow.edge_target_key] = edge_rme
    ref[flow.node_block_target_key] = endpoint.node_blocks.clone()
    ref[flow.edge_block_target_key] = endpoint.edge_blocks.clone()
    ref[flow.node_block_shape_key] = endpoint.node_shapes.clone()
    ref[flow.edge_block_shape_key] = endpoint.edge_shapes.clone()
    return ref


def _blend(data, idp, current, endpoint, alpha):
    return project_block_state(
        data,
        idp,
        BlockTensorResult(
            (1.0 - alpha) * current.node_blocks + alpha * endpoint.node_blocks,
            (1.0 - alpha) * current.edge_blocks + alpha * endpoint.edge_blocks,
            current.node_shapes,
            current.edge_shapes,
        ),
    )


def _assert_state_invariants(state, atol=FP64_ATOL):
    """Symmetric onsite blocks, transpose-paired reverse edges (edge 0 <-> 1), zero padding."""
    assert (state.node_blocks - state.node_blocks.transpose(-1, -2)).abs().max() <= atol
    assert (state.edge_blocks[0] - state.edge_blocks[1].T).abs().max() <= atol
    node_mask = block_mask_from_shapes(state.node_shapes, tuple(state.node_blocks.shape[-2:]))
    edge_mask = block_mask_from_shapes(state.edge_shapes, tuple(state.edge_blocks.shape[-2:]))
    assert torch.count_nonzero(state.node_blocks[~node_mask]) == 0
    assert torch.count_nonzero(state.edge_blocks[~edge_mask]) == 0


class _EndpointSequence(torch.nn.Module):
    """Return preset endpoint block states (repeating the last) and record the H0 RME input and time."""

    def __init__(self, endpoints):
        super().__init__()
        self.endpoints = list(endpoints)
        self.inputs = []
        self.times = []

    def forward(self, data):
        index = len(self.inputs)
        self.inputs.append((data["node_h0"].clone(), data["edge_h0"].clone()))
        self.times.append(data["flow_time"].clone())
        endpoint = self.endpoints[min(index, len(self.endpoints) - 1)]
        out = data.copy()
        out[_keys.NODE_PRED_HAMIL_BLOCKS_KEY] = endpoint.node_blocks
        out[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY] = endpoint.edge_blocks
        return out


def _valid_contract():
    """A complete generic full-H ``ao_block_ode`` training config that validates."""
    return {
        "train_options": {
            "flow_options": {
                "enabled": True,
                "mode": "residual",
                "prior": "zero",
                "output_space": "ao_block_ode",
                "block_ode": True,
                "target_semantics": "absolute_full_h",
                "time_conditioning_required": True,
                "missing_h0_policy": "error",
                "block_inverse_mode": "strict",
                **FULL_H_TARGET_FIELDS,
                "validation_ode_steps": [1, 3],
            }
        },
        "common_options": {"has_soc": False},
        "model_options": {
            "embedding": {
                "method": "lem_moe_v3_h0",
                "output_route": "h_b0",
                "require_full_block_edge_coverage": True,
                "use_flow_time_embedding": True,
                "flow_time_condition_edges": True,
                "flow_time_allow_missing": False,
                "flow_time_key": "flow_time",
                "h0_merge_mode": "replace",
                "h0_init_scope": "both",
            },
            "prediction": {
                "method": "block_native",
                "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True,
                "reconstruction": "direct",
            },
        },
        "data_options": {
            "train": {
                "type": "LMDBDataset",
                "get_Hamiltonian": True,
                "get_H0": True,
                "residual_hamiltonian": False,
                "require_full_h_target": True,
            }
        },
    }


# ---------------------------------------------------------------------------
# Compact uu-real residual route (output_space='uureal_block_ode'), float32
# ---------------------------------------------------------------------------
def _uureal_record(mapper):
    """An already-delta compact uu-real H-C record with its converter metadata."""
    f32 = torch.float32
    data = {
        "atomic_numbers": torch.tensor([1, 6]),
        "atom_types": torch.tensor([[mapper.chemical_symbol_to_type["H"]], [mapper.chemical_symbol_to_type["C"]]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.zeros(2, 3, dtype=f32),
        "edge_type": torch.tensor([[mapper.bond_to_type["H-C"]], [mapper.bond_to_type["C-H"]]]),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=f32),
        "cell": torch.eye(3, dtype=f32) * 8.0,
        "pbc": torch.tensor([False, False, False]),
        "batch": torch.zeros(2, dtype=torch.long),
    }
    node = torch.zeros(2, 4, 4, dtype=f32)
    node[0, 0, 0] = 0.25
    c = torch.arange(16, dtype=f32).reshape(4, 4) / 100.0
    node[1] = 0.5 * (c + c.T)
    edge = torch.zeros(2, 4, 4, dtype=f32)
    edge[0, :1, :4] = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=f32)
    edge[1, :4, :1] = edge[0, :1, :4].T
    node_h0, edge_h0 = block_tensors_to_feature_tensors(data, mapper, node_blocks=node * 0.5, edge_blocks=edge * 0.5)
    keep = int(mapper.reduced_matrix_element)
    data.update(
        {
            "node_h0": node_h0,
            "edge_h0": edge_h0,
            "node_delta_hamil_blocks": node,
            "edge_delta_hamil_blocks": edge,
            "node_delta_hamil_block_shape": torch.tensor([[1, 1], [4, 4]]),
            "edge_delta_hamil_block_shape": torch.tensor([[1, 4], [4, 1]]),
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


def _uureal_flow(mapper, **overrides):
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
        **RESIDUAL_TARGET_FIELDS,
        "validation_ode_steps": [1, 3],
    }
    options.update(overrides)
    return HamiltonianCFM(options, idp=mapper, dtype=torch.float32)


def _uureal_config(**flow_overrides):
    """A complete ``uureal_block_ode`` training config that validates."""
    config = {
        "common_options": {
            "dtype": "float32",
            "has_soc": True,
            "nextham_uureal_mask": True,
            "full_soc_prediction": False,
        },
        "train_options": {
            "flow_options": {
                "enabled": True,
                "mode": "residual",
                "prior": "zero",
                "output_space": "uureal_block_ode",
                "block_ode": True,
                "state_space": "residual_ao_block",
                "target_semantics": "residual_dh",
                "block_input_adapter": "direct_cg",
                "h0_condition_space": "compact_uureal_rme",
                "prediction_add_h0": False,
                "time_conditioning_required": True,
                **RESIDUAL_TARGET_FIELDS,
                "validation_ode_steps": [1, 3],
            }
        },
        "model_options": {
            "embedding": {
                "method": "lem_moe_v3_h0",
                "output_route": "h_b0",
                "require_full_block_edge_coverage": True,
                "use_uureal_residual_block_input": True,
                "use_flow_time_embedding": True,
                "flow_time_condition_edges": True,
                "flow_time_allow_missing": False,
                "h0_merge_mode": "replace",
                "use_h0_node_init": True,
                "use_h0_edge_init": True,
            },
            "prediction": {
                "method": "block_native",
                "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True,
                "add_h0": False,
            },
        },
        "data_options": {
            "train": {
                "type": "LMDBDataset",
                "get_Hamiltonian": True,
                "get_H0": True,
                "residual_hamiltonian": False,
                "require_full_h_target": False,
                "require_residual_h_target": False,
                "require_uureal_block_ode": True,
            }
        },
    }
    config["train_options"]["flow_options"].update(flow_overrides)
    return config


# ---------------------------------------------------------------------------
# Non-SOC direct-residual route (output_space='residual_ao_block_ode')
# ---------------------------------------------------------------------------
def _graph(mapper, *, dtype=torch.float64):
    """A two-atom H-C molecule (pbc FFF, reverse edge pair)."""
    t_h = mapper.chemical_symbol_to_type["H"]
    t_c = mapper.chemical_symbol_to_type["C"]
    return {
        "atomic_numbers": torch.tensor([1, 6]),
        "atom_types": torch.tensor([[t_h], [t_c]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.zeros(2, 3, dtype=dtype),
        "edge_type": torch.tensor([[mapper.bond_to_type["H-C"]], [mapper.bond_to_type["C-H"]]]),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype),
        "cell": torch.eye(3, dtype=dtype) * 8.0,
        "pbc": torch.tensor([False, False, False]),
        "batch": torch.zeros(2, dtype=torch.long),
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
        "pos": torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.3, 0.7, 0.0]], dtype=dtype),
        "cell": torch.eye(3, dtype=dtype) * 10.0,
        "pbc": torch.tensor([False, False, False]),
        "batch": torch.zeros(3, dtype=torch.long),
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
    """A residual record: physical H0 blocks + residual D1 endpoint blocks.

    Returns ``(data, h0_blocks, d1_blocks)``; both block states are already in the
    canonical packer image, so the scalar and interpolation bridges hold to fp precision.
    """
    data = _graph(mapper, dtype=dtype)
    node_shapes, edge_shapes = infer_block_shapes(data, mapper)
    canvas = mapper_max_norb(mapper)
    n = int(node_shapes.shape[0])
    e = int(edge_shapes.shape[0])
    h0 = _projected_state(mapper, data, canvas=canvas, n=n, e=e, dtype=dtype, seed=seed)
    d1 = _projected_state(mapper, data, canvas=canvas, n=n, e=e, dtype=dtype, seed=seed + 1000)
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
    """Build a ``residual_ao_block_ode`` flow with the water-residual config's flow options."""
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
        **RESIDUAL_TARGET_FIELDS,
        "validation_ode_steps": [1],
    }
    options.update(overrides)
    return HamiltonianCFM(options, idp=mapper, dtype=dtype)


_TE_SEED = 20260720


def _b_te_flow(mapper, *, dtype=torch.float64, seed=_TE_SEED, **overrides):
    """Build a projected_te (stochastic bridge) ``residual_ao_block_ode`` flow."""
    options = {
        "prior": "projected_te",
        "te_prior_mode": "irrep",
        "node_sigma": 1.0,
        "edge_sigma": 1.0,
        "te_prior_sigma": 1.0,
        "te_prior_validation_seed": seed,
    }
    options.update(overrides)
    return _b_flow(mapper, dtype=dtype, **options)


class _EndpointSpy(torch.nn.Module):
    """Return pre-set residual endpoint blocks per call, recording model inputs.

    Records the residual block state (``state_keys``, spatial by default), the H0 RME
    conditioning keys and the flow time of every step.
    """

    def __init__(self, endpoints, node_h0_key, edge_h0_key, *, state_keys=SPATIAL_STATE_KEYS):
        super().__init__()
        self.endpoints = endpoints
        self.spatial_inputs = []
        self.h0_inputs = []
        self.times = []
        self._node_h0_key = node_h0_key
        self._edge_h0_key = edge_h0_key
        self._state_keys = tuple(state_keys)

    def forward(self, data):
        node_key, edge_key = self._state_keys
        self.spatial_inputs.append((data[node_key].clone(), data[edge_key].clone()))
        self.h0_inputs.append((data[self._node_h0_key].clone(), data[self._edge_h0_key].clone()))
        self.times.append(data["flow_time"].clone())
        node, edge = self.endpoints[len(self.spatial_inputs) - 1]
        out = data.copy()
        out[_keys.NODE_PRED_HAMIL_BLOCKS_KEY] = node.clone()
        out[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY] = edge.clone()
        return out


class _LinearEchoModel(torch.nn.Module):
    """Trivially equivariant model double: endpoint = alpha * spatial-residual state.

    The echo is a per-element scalar multiply in canvas-block space, so it commutes with
    the shared-canvas Wigner-D conjugation ``B -> D B D^T``; the residual sampler pipeline
    is then pathwise equivariant iff its block bookkeeping is.
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

    H's single 1s lands on scalar slot 0 (its ``0e`` block is the identity), so one
    shared canvas D covers both species and ``D @ B @ D^T`` keeps every block's padding.
    """
    return o3.Irreps("1x0e+1x1o").D_from_matrix(rotation)


def _rotate_canvas_blocks(blocks, d_ao):
    """Conjugate every canvas block by the shared Wigner-D: ``B -> D B D^T``."""
    return d_ao @ blocks @ d_ao.transpose(-1, -2)

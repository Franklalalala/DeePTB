"""Bit-exact characterization gate for ``HamiltonianCFM.prepare_batch`` routes.

PR5c (``refactor/block-ode-prepare-batch-routes``) moves the three block-ODE
route branches of ``prepare_batch`` (uureal / residual / generic full-H) out of
``HamiltonianCFM`` and into the per-route adapters in
``dptb.nnops.block_ode.route_adapters``.  That is a pure behaviour-preserving
code motion, so for a fixed-seed fixture per route the ``CFMContext`` fields
``node_current`` / ``edge_current`` / ``node_prior`` / ``edge_prior`` must be
**bit-identical** before and after the move.

The ``_GOLDEN`` block below is the "before" snapshot: it was captured by running
these exact fixture builders against the pre-refactor base commit
(``bdaf54a`` = ``refactor/block-ode-route-contracts``, where ``prepare_batch``
still held all four route branches inline).  The values are embedded as literals
rather than a binary fixture to match this repo's in-code test-data convention;
``tensor.tolist()`` round-trips bit-exactly for both fp32 and fp64 (a Python
float carries the full double, and re-rounding to the source dtype is the
identity), so ``torch.equal`` is a legitimate assertion here, not ``allclose``.

The three fixtures are deliberately chosen to exercise the non-trivial paths:

* ``uureal`` -- zero prior, so ``node_prior``/``edge_prior`` are ``zeros_like``
  telemetry while ``node_current``/``edge_current`` carry the ``t``-scaled delta
  endpoint;
* ``residual`` -- ``projected_te`` stochastic bridge, so ``node_prior`` carries
  the drawn epsilon and ``node_current`` is ``project((1 - t) * eps + t * D1)``;
* ``full_h`` -- ``projected_te`` initial-state draw through
  ``_block_initial_state``, exercising the seeded prior on the generic route.

To regenerate the golden (only ever against the pre-refactor base), run these
builders and print ``ctx.<field>.tolist()`` per route; do **not** regenerate from
post-refactor code -- that would defeat the differential oracle.
"""
from __future__ import annotations

import copy

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


# ---------------------------------------------------------------------------
# uureal_block_ode fixture (mirrors test_uureal_block_ode.py::_record/_flow)
# ---------------------------------------------------------------------------
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
        "node_block_target_key": "node_delta_hamil_blocks",
        "edge_block_target_key": "edge_delta_hamil_blocks",
        "node_block_shape_key": "node_delta_hamil_block_shape",
        "edge_block_shape_key": "edge_delta_hamil_block_shape",
        "validation_ode_steps": [1, 3],
    }
    options.update(overrides)
    return HamiltonianCFM(options, idp=mapper, dtype=torch.float32)


def _uureal_case():
    mapper = _uureal_mapper()
    data = _uureal_record(mapper)
    flow = _uureal_flow(mapper)
    return flow, data, copy.deepcopy(data), dict(t=torch.tensor([0.25]))


# ---------------------------------------------------------------------------
# residual_ao_block_ode fixture (mirrors test_residual_ao_block_ode.py)
# ---------------------------------------------------------------------------
_TE_SEED = 20260720


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


def _projected_state(mapper, data, *, canvas, n, e, dtype, seed):
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
    return data


def _residual_flow(mapper, *, dtype=torch.float64, **overrides):
    options = {
        "enabled": True,
        "objective": "cfm",
        "mode": "residual",
        "prior": "projected_te",
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
        "te_prior_mode": "irrep",
        "node_sigma": 1.0,
        "edge_sigma": 1.0,
        "te_prior_sigma": 1.0,
        "te_prior_validation_seed": _TE_SEED,
        "node_block_target_key": "node_delta_hamil_blocks",
        "edge_block_target_key": "edge_delta_hamil_blocks",
        "node_block_shape_key": "node_delta_hamil_block_shape",
        "edge_block_shape_key": "edge_delta_hamil_block_shape",
        "validation_ode_steps": [1],
    }
    options.update(overrides)
    return HamiltonianCFM(options, idp=mapper, dtype=dtype)


def _residual_case():
    mapper = _residual_mapper()
    data = _residual_record(mapper)
    flow = _residual_flow(mapper)
    kwargs = dict(t=torch.tensor([0.25], dtype=torch.float64), prior_seed=_TE_SEED)
    return flow, data, copy.deepcopy(data), kwargs


# ---------------------------------------------------------------------------
# generic full-H block_ode fixture (mirrors test_block_ode_flow.py::_case/_flow)
# ---------------------------------------------------------------------------
_ATOL = 1e-10


def _fresh(data):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}


def _fullh_case_data():
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


def _fullh_flow(idp, semantics="absolute_full_h", **updates):
    target_fields = {
        "node_block_target_key": "node_full_hamil_target_blocks",
        "edge_block_target_key": "edge_full_hamil_target_blocks",
        "node_block_shape_key": "node_full_hamil_target_block_shape",
        "edge_block_shape_key": "edge_full_hamil_target_block_shape",
    }
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
        "block_inverse_atol": _ATOL,
        "validation_ode_steps": [1, 3],
        "te_prior_validation_seed": 20260719,
        **target_fields,
    }
    options.update(updates)
    return HamiltonianCFM(options, idp=idp, dtype=torch.float64)


def _fullh_case():
    idp, data, codec = _fullh_case_data()
    # Deterministic endpoint = 1.25x-scaled H0 RME re-projected to blocks.
    endpoint = codec.rme_to_blocks(
        data, data["node_h0"] * 1.25, data["edge_h0"] * 1.25, project=True
    )
    ref = _fresh(data)
    ref.update(
        {
            _keys.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY: endpoint.node_blocks.clone(),
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY: endpoint.edge_blocks.clone(),
            _keys.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: endpoint.node_shapes.clone(),
            _keys.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY: endpoint.edge_shapes.clone(),
        }
    )
    flow = _fullh_flow(
        idp,
        "absolute_full_h",
        prior="projected_te",
        te_prior_mode="irrep",
        te_prior_sigma=1.0,
        node_sigma=1.0,
        edge_sigma=1.0,
    )
    kwargs = dict(t=torch.tensor([0.3], dtype=torch.float64), prior_seed=20260719)
    return flow, _fresh(data), ref, kwargs


_CASES = {
    "uureal": _uureal_case,
    "residual": _residual_case,
    "full_h": _fullh_case,
}

_FIELDS = ("node_current", "edge_current", "node_prior", "edge_prior")


# ---------------------------------------------------------------------------
# Golden CFMContext fields captured from base bdaf54a (pre-refactor prepare_batch).
# ---------------------------------------------------------------------------
_GOLDEN = {
    'uureal': {
        'node_current': (
            'float32',
            [[[0.0625, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.00624999962747097, 0.01249999925494194, 0.01874999888241291], [0.00624999962747097, 0.012500000186264515, 0.01875000074505806, 0.02499999850988388], [0.01249999925494194, 0.01875000074505806, 0.02500000037252903, 0.03125], [0.01874999888241291, 0.02499999850988388, 0.03125, 0.03750000149011612]]],
        ),
        'edge_current': (
            'float32',
            [[[0.02500000037252903, 0.05000000074505806, 0.07500000298023224, 0.10000000149011612], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.02500000037252903, 0.0, 0.0, 0.0], [0.05000000074505806, 0.0, 0.0, 0.0], [0.07500000298023224, 0.0, 0.0, 0.0], [0.10000000149011612, 0.0, 0.0, 0.0]]],
        ),
        'node_prior': (
            'float32',
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        ),
        'edge_prior': (
            'float32',
            [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        ),
    },
    'residual': {
        'node_current': (
            'float64',
            [[[-0.07788178013652025, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.11732599428490932, 0.31322367921417094, 0.4783135301356264, 1.546967855232313], [0.31322367921417094, -1.3222194562200267, 0.38216255374422714, 0.32723438358051604], [0.4783135301356264, 0.38216255374422714, -0.21318886896599687, 0.5168023526790788], [1.546967855232313, 0.32723438358051604, 0.5168023526790788, 1.3059903822699561]]],
        ),
        'edge_current': (
            'float64',
            [[[-0.04407026619864976, 0.5348684515131238, 0.02012142974766656, 0.3869259534976273], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.04407026619864976, 0.0, 0.0, 0.0], [0.5348684515131238, 0.0, 0.0, 0.0], [0.02012142974766656, 0.0, 0.0, 0.0], [0.3869259534976273, 0.0, 0.0, 0.0]]],
        ),
        'node_prior': (
            'float64',
            [[[0.30130634443711685, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.30130634443711685, 0.2794841786798032, 0.3115777492736572, 1.6743984143409378], [0.2794841786798032, -1.7911772062988134, 0.6325139728318494, 0.503963321832853], [0.3115777492736572, 0.6325139728318494, -1.0785396143857107, 0.7791223067351118], [1.6743984143409378, 0.503963321832853, 0.7791223067351118, 2.008310678900107]]],
        ),
        'edge_prior': (
            'float64',
            [[[0.0, 0.3789189202446655, 0.026025990523981905, 0.41504306944140384], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.3789189202446655, 0.0, 0.0, 0.0], [0.026025990523981905, 0.0, 0.0, 0.0], [0.41504306944140384, 0.0, 0.0, 0.0]]],
        ),
    },
    'full_h': {
        'node_current': (
            'float64',
            [[-0.11308201054611754, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.9602571670656418, 0.0, 0.0, 0.0, -1.762824536925499, -1.3010519035013468, -2.3845803208542216, -0.25157697272353025, -1.6146686427745744]],
        ),
        'edge_current': (
            'float64',
            [[0.0, 1.0632386813218828, 1.8180318847810704, -0.8530269517696033, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ),
        'node_prior': (
            'float64',
            [[-0.5890576356004706, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1444090257532087, 0.0, 0.0, 0.0, 0.28662158818311023, 0.09205269631094626, -0.04609851631967299, -0.29131698932977557, -0.23005340173454725]],
        ),
        'edge_prior': (
            'float64',
            [[0.0, 0.5273310222849898, 0.89156955426988, -1.283273102052775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ),
    },
}


def _golden_tensor(route, field):
    dtype_str, values = _GOLDEN[route][field]
    return torch.tensor(values, dtype=getattr(torch, dtype_str))


@pytest.mark.parametrize("route", list(_CASES))
def test_prepare_batch_route_fields_bit_identical(route):
    """Each block-ODE route's CFMContext state fields match the pre-refactor base.

    Bit-exact (``torch.equal``, not ``allclose``): PR5c is a pure code motion, so
    reordering or dtype drift in any of the four fields is a real regression, not
    numerical noise.
    """
    flow, data, ref, kwargs = _CASES[route]()
    _model_data, _ref, ctx = flow.prepare_batch(data, ref, **kwargs)
    for field in _FIELDS:
        actual = torch.as_tensor(getattr(ctx, field)).detach()
        golden = _golden_tensor(route, field)
        assert actual.dtype == golden.dtype, (
            f"{route}.{field} dtype {actual.dtype} != golden {golden.dtype}"
        )
        assert tuple(actual.shape) == tuple(golden.shape), (
            f"{route}.{field} shape {tuple(actual.shape)} != golden {tuple(golden.shape)}"
        )
        assert torch.equal(actual, golden), (
            f"{route}.{field} is not bit-identical to the pre-refactor base golden"
        )

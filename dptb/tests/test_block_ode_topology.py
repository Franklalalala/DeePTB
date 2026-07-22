"""Differential-oracle round-trip tests for dptb.nnops.block_ode.topology.

PR2 of the block-ode/flow.py extraction moves the route-agnostic
block-topology contract (key sets, snapshot/restore, cross-check, and
output-only-key filtering -- "Group F" of the refactor plan, plus
``_max_abs``) out of ``HamiltonianCFM`` into plain functions in
``dptb.nnops.block_ode.topology``.  ``HamiltonianCFM`` keeps every moved
method name as a 1-3 line delegator that forwards into the new module.

Every test below exercises that delegator/module-function pair on the same
synthetic ``AtomicDataDict`` fixture -- calling the operation once through
the ``HamiltonianCFM`` instance (the exact call sites production code uses)
and once through ``dptb.nnops.block_ode.topology`` directly -- and asserts
the two are identical.  This pins PR2's central invariant: the delegator is
a faithful, lossless forward with no behavior drift.  Snapshot/restore/
compare round-trips are exercised directly against the fixture to prove the
move preserved the underlying semantics, not just the call signature.

Fixtures are reused from ``test_block_ode_flow.py`` (``_case``, ``_flow``,
``_fresh``) rather than re-invented, matching this repo's existing
test-reuse convention (see ``test_block_ode_redteam.py``'s identical import
block).
"""

from __future__ import annotations

import inspect

import pytest
import torch

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import BlockTensorResult
from dptb.nnops.block_ode import topology
from dptb.nnops.flow import _BLOCK_ODE_OUTPUT_ONLY_KEYS, HamiltonianCFM
from dptb.tests.test_block_ode_flow import _case, _flow, _fresh


# ---------------------------------------------------------------------------
# Manifest canary: pins the exact set of functions PR2 was scoped to move.
# ---------------------------------------------------------------------------


def test_topology_module_exposes_exactly_the_migrated_functions():
    expected = {
        "_require_real_finite_tensor",
        "_block_primary_topology_keys",
        "_block_topology_keys",
        "_missing_keys",
        "_metadata_scalar",
        "_clone_sidecar_value",
        "_snapshot_block_topology",
        "_drop_block_authority_fields",
        "_block_ode_output_only_keys",
        "_require_fresh_block_ode_outputs",
        "_require_matching_block_topology",
        "_restore_block_topology",
        "_attach_uureal_residual_state",
        "_attach_spatial_residual_state",
        "_max_abs",
    }
    actual = {
        name
        for name, obj in vars(topology).items()
        if inspect.isfunction(obj) and obj.__module__ == topology.__name__
    }
    assert actual == expected
    # The two route-specific contract checks stay on HamiltonianCFM (they
    # belong to a future route_adapters.py PR); they must NOT have moved.
    assert hasattr(HamiltonianCFM, "_require_uureal_block_contract")
    assert hasattr(HamiltonianCFM, "_require_spatial_residual_block_contract")
    assert not hasattr(topology, "_require_uureal_block_contract")
    assert not hasattr(topology, "_require_spatial_residual_block_contract")


def test_block_ode_topology_module_does_not_import_flow():
    # Import-direction rule (refactor plan §2.1/§4.4): no module under
    # dptb/nnops/block_ode/ may import dptb.nnops.flow.  Parse the actual
    # import statements via ast (not a raw substring search, which would
    # also flag this module's own docstring describing the rule) and check
    # none of them names dptb.nnops.flow.
    import ast

    tree = ast.parse(inspect.getsource(topology))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("dptb.nnops.flow")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            assert not module_name.startswith("dptb.nnops.flow")
    # And no module-level binding named "flow" (e.g. a re-exported alias).
    assert "flow" not in vars(topology)


# ---------------------------------------------------------------------------
# Stateless key-set / helper functions: delegator vs. module function.
# ---------------------------------------------------------------------------


def test_block_primary_topology_keys_delegator_matches_module():
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp)
    assert flow._block_primary_topology_keys() == topology._block_primary_topology_keys()
    keys = topology._block_primary_topology_keys()
    assert len(keys) == len(set(keys))
    for key in ("edge_index", "edge_cell_shift", "atom_types", "pos", "batch", "pbc", "edge_type", "cell"):
        assert key in keys


def test_block_topology_keys_is_superset_of_primary_keys():
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp)
    primary = set(topology._block_primary_topology_keys())
    full = set(topology._block_topology_keys())
    assert primary.issubset(full)
    assert flow._block_topology_keys() == topology._block_topology_keys()
    assert _keys.EDGE_VECTORS_KEY in full
    assert _keys.EDGE_LENGTH_KEY in full


def test_missing_keys_delegator_matches_module():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    keys = ("pos", "not_a_real_key", "batch", "also_missing")
    assert flow._missing_keys(data, keys) == topology._missing_keys(data, keys)
    assert topology._missing_keys(data, keys) == ["not_a_real_key", "also_missing"]
    assert topology._missing_keys(data, ("pos", "batch")) == []


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([3.0, 3.0, 3.0]),
        [2, 2, 2],
        (1.5, 1.5),
    ],
)
def test_metadata_scalar_delegator_matches_module_on_uniform_values(value):
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp)
    assert flow._metadata_scalar(value) == topology._metadata_scalar(value)


def test_metadata_scalar_rejects_non_uniform_values():
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp)
    with pytest.raises(ValueError, match="nonempty and identical"):
        flow._metadata_scalar(torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="nonempty and identical"):
        topology._metadata_scalar(torch.tensor([1.0, 2.0]))


def test_require_real_finite_tensor_delegator_matches_module():
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp)
    value = torch.tensor([1.0, 2.0, 3.0])
    out_flow = flow._require_real_finite_tensor(value, label="x")
    out_topo = topology._require_real_finite_tensor(value, label="x")
    assert torch.equal(out_flow, out_topo)


def test_require_real_finite_tensor_rejects_complex_and_nonfinite():
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp)
    with pytest.raises(ValueError, match="must be real"):
        flow._require_real_finite_tensor(torch.tensor([1.0 + 1j]), label="x")
    with pytest.raises(ValueError, match="NaN or Inf"):
        flow._require_real_finite_tensor(torch.tensor([float("nan")]), label="x")
    with pytest.raises(ValueError, match="must be real"):
        topology._require_real_finite_tensor(torch.tensor([1.0 + 1j]), label="x")
    with pytest.raises(ValueError, match="NaN or Inf"):
        topology._require_real_finite_tensor(torch.tensor([float("nan")]), label="x")


def test_clone_sidecar_value_isolates_tensor_mutation():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    original_tensor = data["pos"]
    cloned_via_flow = flow._clone_sidecar_value(original_tensor)
    cloned_via_topology = topology._clone_sidecar_value(original_tensor)
    assert torch.equal(cloned_via_flow, cloned_via_topology)
    cloned_via_flow += 100.0
    assert not torch.equal(cloned_via_flow, original_tensor)
    assert torch.equal(original_tensor, data["pos"])


def test_clone_sidecar_value_deep_copies_non_tensor_values():
    original = [1, 2, {"a": 1}]
    cloned = topology._clone_sidecar_value(original)
    assert cloned == original
    assert cloned is not original
    cloned[2]["a"] = 999
    assert original[2]["a"] == 1


def test_max_abs_delegator_matches_module():
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp)
    value = torch.tensor([-3.0, 1.0, 2.0])
    assert flow._max_abs(value) == topology._max_abs(value)
    assert flow._max_abs(value) == pytest.approx(3.0)
    empty = torch.zeros(0)
    assert flow._max_abs(empty) == 0.0 == topology._max_abs(empty)


def test_max_abs_rejects_nonfinite_via_require_real_finite_tensor():
    with pytest.raises(ValueError, match="NaN or Inf"):
        topology._max_abs(torch.tensor([float("inf")]))


# ---------------------------------------------------------------------------
# Snapshot / restore / compare round-trip on the synthetic AtomicDataDict
# fixture (differential-oracle: HamiltonianCFM delegator vs. module call).
# ---------------------------------------------------------------------------


def test_snapshot_block_topology_delegator_matches_module():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    snapshot_flow = flow._snapshot_block_topology(data)
    snapshot_topo = topology._snapshot_block_topology(data)
    assert set(snapshot_flow) == set(snapshot_topo)
    for key in snapshot_flow:
        left, right = snapshot_flow[key], snapshot_topo[key]
        if torch.is_tensor(left):
            assert torch.equal(left, right)
            # Each snapshot call clones independently -- no shared storage.
            assert left.data_ptr() != right.data_ptr() or left.numel() == 0
        else:
            assert left == right


def test_snapshot_restore_round_trip_is_lossless():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    original = _fresh(data)
    snapshot = flow._snapshot_block_topology(data)

    mutated = _fresh(data)
    mutated["pos"] = mutated["pos"] + 5.0
    mutated["batch"] = mutated["batch"] + 0
    mutated.pop("cell", None)  # simulate a model step dropping a topology key

    flow._restore_block_topology(mutated, snapshot)
    for key in topology._block_topology_keys():
        if key in original:
            assert key in mutated
            left, right = mutated[key], original[key]
            if torch.is_tensor(left):
                assert torch.equal(left, right)
            else:
                assert left == right
        else:
            assert key not in mutated


def test_restore_block_topology_drops_keys_absent_from_snapshot():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    reduced_snapshot = {
        key: value for key, value in flow._snapshot_block_topology(data).items() if key != "cell"
    }
    target = _fresh(data)
    assert "cell" in target
    topology._restore_block_topology(target, reduced_snapshot)
    assert "cell" not in target


def test_restore_block_topology_clone_values_flag_isolates_snapshot():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    snapshot = flow._snapshot_block_topology(data)

    target = {}
    flow._restore_block_topology(target, snapshot, clone_values=True)
    target["pos"] += 1000.0
    assert not torch.equal(target["pos"], snapshot["pos"])

    target_topo = {}
    topology._restore_block_topology(target_topo, snapshot, clone_values=True)
    target_topo["pos"] += 1000.0
    assert not torch.equal(target_topo["pos"], snapshot["pos"])


def test_restore_block_topology_without_clone_shares_snapshot_tensor():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    snapshot = flow._snapshot_block_topology(data)
    target = {}
    flow._restore_block_topology(target, snapshot, clone_values=False)
    assert target["pos"].data_ptr() == snapshot["pos"].data_ptr()


def test_require_matching_block_topology_accepts_identical_topology():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    left = flow._snapshot_block_topology(data)
    right = flow._snapshot_block_topology(_fresh(data))
    flow._require_matching_block_topology(left, right)
    topology._require_matching_block_topology(left, right)


def test_require_matching_block_topology_rejects_value_mismatch():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    left = flow._snapshot_block_topology(data)
    mutated = _fresh(data)
    mutated["pos"] = mutated["pos"] + 1.0
    right = flow._snapshot_block_topology(mutated)
    with pytest.raises(ValueError, match="cannot be mixed"):
        flow._require_matching_block_topology(left, right)
    with pytest.raises(ValueError, match="cannot be mixed"):
        topology._require_matching_block_topology(left, right)


def test_require_matching_block_topology_rejects_key_presence_mismatch():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    left = flow._snapshot_block_topology(data)
    right = dict(left)
    right.pop("cell", None)
    with pytest.raises(ValueError, match="present in only one dictionary"):
        flow._require_matching_block_topology(left, right)
    with pytest.raises(ValueError, match="present in only one dictionary"):
        topology._require_matching_block_topology(left, right)


# ---------------------------------------------------------------------------
# Owner-reference functions: _drop_block_authority_fields /
# _block_ode_output_only_keys / _require_fresh_block_ode_outputs.  These
# read instance state (self.node_output_key, etc.) so the module-level
# versions take the owning HamiltonianCFM explicitly as their first
# ("owner") argument.
# ---------------------------------------------------------------------------


def test_drop_block_authority_fields_removes_only_authority_keys():
    idp, data, _codec, _h0 = _case()
    flow = _flow(idp)
    payload = _fresh(data)
    authority_keys = {
        flow.node_block_target_key,
        flow.edge_block_target_key,
        flow.node_block_shape_key,
        flow.edge_block_shape_key,
        flow.node_h0_block_key,
        flow.edge_h0_block_key,
        flow.node_h0_block_shape_key,
        flow.edge_h0_block_shape_key,
    }
    for key in authority_keys:
        payload[key] = torch.zeros(1)
    before_keys = set(payload)

    via_flow = _fresh(payload)
    flow._drop_block_authority_fields(via_flow)

    via_topo = _fresh(payload)
    topology._drop_block_authority_fields(flow, via_topo)

    assert set(via_flow) == set(via_topo) == before_keys - authority_keys
    # Non-authority keys (e.g. the topology fixture's own fields) survive.
    assert "pos" in via_flow and "batch" in via_flow


def test_block_ode_output_only_keys_dedupes_and_matches_module():
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp)
    result_via_flow = flow._block_ode_output_only_keys()
    result_via_topology = topology._block_ode_output_only_keys(flow, _BLOCK_ODE_OUTPUT_ONLY_KEYS)
    assert result_via_flow == result_via_topology
    # Default node/edge output keys collide with the first two constant
    # entries; dict.fromkeys must dedupe the 6 candidates to 4 unique keys.
    assert len(result_via_flow) == 4
    assert result_via_flow[0] == flow.node_output_key
    assert result_via_flow[1] == flow.edge_output_key


def test_block_ode_output_only_keys_preserves_non_colliding_extra_key():
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp, node_output_key="custom_node_output")
    result = topology._block_ode_output_only_keys(flow, _BLOCK_ODE_OUTPUT_ONLY_KEYS)
    assert result[0] == "custom_node_output"
    assert set(_BLOCK_ODE_OUTPUT_ONLY_KEYS).issubset(set(result))
    assert len(result) == 5  # no longer collides with the node entry


def test_require_fresh_block_ode_outputs_accepts_fresh_prediction():
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp)
    prediction = {
        flow.node_output_key: torch.zeros(1),
        flow.edge_output_key: torch.zeros(1),
    }
    flow._require_fresh_block_ode_outputs(prediction, step=1)
    topology._require_fresh_block_ode_outputs(flow, prediction, step=1)


def test_require_fresh_block_ode_outputs_rejects_missing_prediction():
    idp, _data, _codec, _h0 = _case()
    flow = _flow(idp)
    prediction = {flow.node_output_key: torch.zeros(1)}
    with pytest.raises(ValueError, match=r"step 2 is missing fresh model output keys"):
        flow._require_fresh_block_ode_outputs(prediction, step=2)
    with pytest.raises(ValueError, match=r"step 2 is missing fresh model output keys"):
        topology._require_fresh_block_ode_outputs(flow, prediction, step=2)


# ---------------------------------------------------------------------------
# Residual-state sidecar attachment.
# ---------------------------------------------------------------------------


def test_attach_uureal_residual_state_writes_expected_keys():
    idp, _data, _codec, h0_blocks = _case()
    state = BlockTensorResult(
        node_blocks=h0_blocks.node_blocks.clone(),
        edge_blocks=h0_blocks.edge_blocks.clone(),
        node_shapes=h0_blocks.node_shapes.clone(),
        edge_shapes=h0_blocks.edge_shapes.clone(),
    )
    flow = _flow(idp)
    target_flow: dict = {}
    flow._attach_uureal_residual_state(target_flow, state)
    target_topo: dict = {}
    topology._attach_uureal_residual_state(target_topo, state)
    expected_keys = {
        _keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY,
        _keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY,
        _keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
        _keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
    }
    assert set(target_flow) == set(target_topo) == expected_keys
    for key in target_flow:
        assert torch.equal(target_flow[key], target_topo[key])


def test_attach_spatial_residual_state_writes_expected_keys():
    idp, _data, _codec, h0_blocks = _case()
    state = BlockTensorResult(
        node_blocks=h0_blocks.node_blocks.clone(),
        edge_blocks=h0_blocks.edge_blocks.clone(),
        node_shapes=h0_blocks.node_shapes.clone(),
        edge_shapes=h0_blocks.edge_shapes.clone(),
    )
    flow = _flow(idp)
    target_flow: dict = {}
    flow._attach_spatial_residual_state(target_flow, state)
    target_topo: dict = {}
    topology._attach_spatial_residual_state(target_topo, state)
    expected_keys = {
        _keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY,
        _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY,
        _keys.NODE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY,
        _keys.EDGE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY,
    }
    assert set(target_flow) == set(target_topo) == expected_keys
    for key in target_flow:
        assert torch.equal(target_flow[key], target_topo[key])

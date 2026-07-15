"""Tests for node/edge-aware ensemble output stitching.

These pin the fix for the silent node-output drop in
:class:`dptb.nn.build.DistanceEnsembleWrapper`: multi-expert inference used to
stitch outputs by *guessing* alignment from ``tensor.shape[0]``, which discarded
every ``node_*`` output (node_hamiltonian / node_overlap / node_h0 / ... /
onsite blocks) produced by experts ``1..N``.  Stitching is now driven by an
explicit :class:`dptb.nn.output_spec.ModelOutputSpec`.
"""

import logging

import pytest
import torch
import torch.nn as nn

import dptb.nn.build as build_mod
from dptb.nn.build import DistanceEnsembleWrapper
from dptb.nn.output_spec import (
    ModelOutputSpec,
    ModelOutputSpecError,
    OutputFieldSpec,
    default_output_spec,
)


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------
class _StubExpert(nn.Module):
    """A stand-in expert that returns a fixed (cloned) output dict.

    Mirrors the real experts, whose ``forward`` returns the plain
    ``AtomicDataDict`` (``Dict[str, Tensor]``).  Also exposes the attributes the
    wrapper's ``__init__`` reads from ``experts[0]``.
    """

    def __init__(self, outputs=None):
        super().__init__()
        self._outputs = outputs or {}
        self.name = "stub_expert"
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model_options = {}

    def forward(self, batch):
        return {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in self._outputs.items()
        }


def _make_wrapper(strict=False, num_experts=2, expert_outputs=None):
    if expert_outputs is None:
        experts = [_StubExpert() for _ in range(num_experts)]
    else:
        experts = [_StubExpert(o) for o in expert_outputs]
    ranges = [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)][:num_experts]
    return DistanceEnsembleWrapper(
        experts=experts, distance_ranges=ranges, strict_output_spec=strict
    )


class _ListHandler(logging.Handler):
    """Captures log records from a specific logger regardless of propagation."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _capture_build_warnings():
    handler = _ListHandler()
    handler.setLevel(logging.WARNING)
    build_mod.log.addHandler(handler)
    prev_level = build_mod.log.level
    build_mod.log.setLevel(logging.WARNING)
    return handler, prev_level


def _release_build_warnings(handler, prev_level):
    build_mod.log.removeHandler(handler)
    build_mod.log.setLevel(prev_level)


# ---------------------------------------------------------------------------
# default_output_spec classification sanity
# ---------------------------------------------------------------------------
def test_default_spec_classifies_node_edge_graph_and_structural():
    spec = default_output_spec()

    assert isinstance(spec, ModelOutputSpec)

    node = spec.get("node_hamiltonian")
    assert node is not None and node.alignment == "node" and node.merge == "masked_replace"

    edge = spec.get("edge_hamiltonian")
    assert edge is not None and edge.alignment == "edge" and edge.merge == "masked_replace"

    # *_block_shape is graph metadata even though it carries a node_/edge_ prefix.
    bshape = spec.get("node_hamil_block_shape")
    assert bshape is not None and bshape.alignment == "graph" and bshape.merge == "keep_first"

    # Structural keys that share the edge_ prefix must NOT be edge-aligned.
    for structural in ("edge_index", "edge_type", "edge_lengths", "atom_types", "pos"):
        s = spec.get(structural)
        assert s is not None and s.alignment == "graph" and s.merge == "keep_first", structural

    # The onsite / node families named in the bug report all become node-merged.
    for node_key in (
        "node_overlap",
        "node_h0",
        "node_p2",
        "node_p23",
        "node_delta_hamil_blocks",
        "node_hamil_blocks",
    ):
        s = spec.get(node_key)
        assert s is not None and s.alignment == "node" and s.merge == "masked_replace", node_key

    # An undeclared key is genuinely absent (drives the strict "undeclared" path).
    assert spec.get("mystery_out") is None
    assert "mystery_out" not in spec


def test_output_field_spec_rejects_bad_vocab():
    with pytest.raises(ValueError):
        OutputFieldSpec("x", "planet", "masked_replace")
    with pytest.raises(ValueError):
        OutputFieldSpec("x", "node", "teleport")


# ---------------------------------------------------------------------------
# (a) node outputs land under node_mask; (b) edge outputs stitch via edge_mask
# ---------------------------------------------------------------------------
def test_node_and_edge_outputs_stitch_under_their_masks():
    w = _make_wrapper(strict=False)

    n_node, n_edge, dim = 3, 5, 4

    res = {
        "node_hamiltonian": torch.zeros(n_node, dim),
        "edge_hamiltonian": torch.zeros(n_edge, 2),
        "node_hamil_block_shape": torch.tensor([2, 2]),
    }
    res_i = {
        "node_hamiltonian": (torch.arange(n_node * dim, dtype=torch.float32).reshape(n_node, dim) + 100.0),
        "edge_hamiltonian": (torch.arange(n_edge * 2, dtype=torch.float32).reshape(n_edge, 2) + 200.0),
        "node_hamil_block_shape": torch.tensor([9, 9]),
    }

    node_mask = torch.tensor([True, False, True])
    edge_mask = torch.tensor([False, True, True, False, True])

    w._stitch_outputs(res, res_i, edge_mask, node_mask)

    # (a) expert-1 node rows land exactly where node_mask is True; others kept.
    assert torch.equal(res["node_hamiltonian"][node_mask], res_i["node_hamiltonian"][node_mask])
    assert torch.equal(res["node_hamiltonian"][~node_mask], torch.zeros(1, dim))

    # (b) expert-1 edge rows land where edge_mask is True; others kept.
    assert torch.equal(res["edge_hamiltonian"][edge_mask], res_i["edge_hamiltonian"][edge_mask])
    assert torch.equal(res["edge_hamiltonian"][~edge_mask], torch.zeros(2, 2))

    # graph-level *_block_shape keeps expert-0's value (keep_first).
    assert torch.equal(res["node_hamil_block_shape"], torch.tensor([2, 2]))


def test_node_stitch_was_previously_dropped_now_merges_even_when_shape0_differs():
    # num_nodes != num_edges is the case where the old shape-guess dropped nodes.
    w = _make_wrapper(strict=False)
    n_node, n_edge = 2, 7

    res = {"node_h0": torch.zeros(n_node, 3)}
    res_i = {"node_h0": torch.ones(n_node, 3) * 5.0}

    node_mask = torch.tensor([True, True])
    edge_mask = torch.zeros(n_edge, dtype=torch.bool)

    w._stitch_outputs(res, res_i, edge_mask, node_mask)

    assert torch.equal(res["node_h0"], torch.ones(n_node, 3) * 5.0)


# ---------------------------------------------------------------------------
# (c) strict mode raises on undeclared keys (only they need shape inference)
# ---------------------------------------------------------------------------
def test_strict_mode_raises_on_undeclared_key():
    w = _make_wrapper(strict=True)
    n_node, n_edge = 3, 5

    res = {
        "node_hamiltonian": torch.zeros(n_node, 4),
        "mystery_out": torch.zeros(n_edge, 4),
    }
    res_i = {
        "node_hamiltonian": torch.ones(n_node, 4),
        "mystery_out": torch.ones(n_edge, 4),
    }
    node_mask = torch.tensor([True, False, True])
    edge_mask = torch.tensor([False, True, True, False, True])

    with pytest.raises(ModelOutputSpecError, match="mystery_out"):
        w._stitch_outputs(res, res_i, edge_mask, node_mask)


def test_strict_mode_stitches_declared_fields_despite_node_edge_ambiguity():
    # num_nodes == num_edges no longer refuses outright in strict mode:
    # DECLARED fields carry a declared alignment, so no shape disambiguation is
    # needed and the merge proceeds under the declared mask.
    w = _make_wrapper(strict=True)
    n = 4  # num_nodes == num_edges

    res = {
        "node_hamiltonian": torch.zeros(n, 4),
        "edge_hamiltonian": torch.zeros(n, 2),
    }
    res_i = {
        "node_hamiltonian": torch.ones(n, 4),
        "edge_hamiltonian": torch.ones(n, 2) * 2.0,
    }
    node_mask = torch.tensor([True, False, True, False])
    edge_mask = torch.tensor([False, True, False, True])

    w._stitch_outputs(res, res_i, edge_mask, node_mask)  # must not raise

    assert torch.equal(res["node_hamiltonian"][node_mask], torch.ones(2, 4))
    assert torch.equal(res["node_hamiltonian"][~node_mask], torch.zeros(2, 4))
    assert torch.equal(res["edge_hamiltonian"][edge_mask], torch.full((2, 2), 2.0))
    assert torch.equal(res["edge_hamiltonian"][~edge_mask], torch.zeros(2, 2))


def test_strict_mode_node_edge_ambiguity_raises_only_for_undeclared_keys():
    # With an UNDECLARED key present, strict mode still refuses -- alignment
    # would have to be guessed from shape and num_nodes == num_edges makes that
    # guess impossible.  The error names the offending key and the ambiguity.
    w = _make_wrapper(strict=True)
    n = 4  # num_nodes == num_edges

    res = {
        "node_hamiltonian": torch.zeros(n, 4),
        "mystery_out": torch.zeros(n, 4),
    }
    res_i = {
        "node_hamiltonian": torch.ones(n, 4),
        "mystery_out": torch.ones(n, 4),
    }
    mask = torch.tensor([True, False, True, False])

    with pytest.raises(ModelOutputSpecError, match="mystery_out"):
        w._stitch_outputs(res, res_i, mask, mask)

    with pytest.raises(ModelOutputSpecError, match="Ambiguous"):
        w._stitch_outputs(res, res_i, mask, mask)


def test_strict_mode_raises_on_alignment_shape_mismatch():
    w = _make_wrapper(strict=True)
    n_node, n_edge = 3, 5

    # node_hamiltonian declared 'node' but leading dim matches neither mask well:
    # give it a wrong leading dim to trip the strict alignment assertion.
    res = {"node_hamiltonian": torch.zeros(n_node, 4)}
    res_i = {"node_hamiltonian": torch.ones(n_node + 1, 4)}
    node_mask = torch.tensor([True, False, True])
    edge_mask = torch.tensor([False, True, True, False, True])

    with pytest.raises(ModelOutputSpecError):
        w._stitch_outputs(res, res_i, edge_mask, node_mask)


# ---------------------------------------------------------------------------
# (d) permissive mode warns + keeps expert 0
# ---------------------------------------------------------------------------
def test_permissive_mode_legacy_stitches_undeclared_edge_key_and_keeps_others():
    # Backward-compat: an UNDECLARED but edge-shaped output (leading dim ==
    # num_edges) must still be stitched via edge_mask, exactly as the old
    # shape-based implementation did -- otherwise experts 1..N silently lose
    # their contribution to Full-H reconstruction fields not listed in _keys.py.
    # An undeclared key that is NOT edge-shaped keeps expert-0's value + warns.
    w = _make_wrapper(strict=False)
    n_node, n_edge = 3, 5

    res = {
        "node_hamiltonian": torch.zeros(n_node, 4),
        "mystery_edge": torch.full((n_edge, 4), 7.0),   # undeclared, edge-shaped
        "mystery_scalar": torch.full((2,), 7.0),        # undeclared, not edge-shaped
    }
    res_i = {
        "node_hamiltonian": torch.ones(n_node, 4),
        "mystery_edge": torch.full((n_edge, 4), 99.0),
        "mystery_scalar": torch.full((2,), 99.0),
    }
    node_mask = torch.tensor([True, False, True])
    edge_mask = torch.tensor([False, True, True, False, True])

    handler, prev_level = _capture_build_warnings()
    try:
        w._stitch_outputs(res, res_i, edge_mask, node_mask)
    finally:
        _release_build_warnings(handler, prev_level)

    messages = [rec.getMessage() for rec in handler.records]
    # undeclared edge-shaped key legacy-stitched via edge_mask
    expected_edge = torch.full((n_edge, 4), 7.0)
    expected_edge[edge_mask] = 99.0
    assert torch.equal(res["mystery_edge"], expected_edge)
    # undeclared non-edge-shaped key kept expert-0's value, with a warning
    assert torch.equal(res["mystery_scalar"], torch.full((2,), 7.0))
    assert any("mystery_scalar" in m for m in messages), messages
    # declared node output still merged as normal
    assert torch.equal(res["node_hamiltonian"][node_mask], torch.ones(2, 4))


def test_permissive_mode_does_not_raise_on_ambiguity():
    # In permissive mode num_nodes == num_edges is tolerated (declared alignment
    # is used, not a shape guess), so no exception.
    w = _make_wrapper(strict=False)
    n = 4
    res = {"node_hamiltonian": torch.zeros(n, 2)}
    res_i = {"node_hamiltonian": torch.ones(n, 2)}
    mask = torch.tensor([True, False, True, False])

    w._stitch_outputs(res, res_i, mask, mask)  # must not raise
    assert torch.equal(res["node_hamiltonian"][mask], torch.ones(2, 2))


# ---------------------------------------------------------------------------
# Core fix: forward() passes BOTH masks to the stitch method
# ---------------------------------------------------------------------------
def test_forward_multiexpert_passes_both_edge_and_node_masks():
    n_node, n_edge = 3, 4

    def out(offset):
        return {
            "node_hamiltonian": torch.zeros(n_node, 2) + float(offset),
            "edge_hamiltonian": torch.zeros(n_edge, 2) + float(offset),
        }

    w = _make_wrapper(strict=False, num_experts=2, expert_outputs=[out(0), out(1)])

    recorded = {"calls": 0}
    orig = w._stitch_outputs

    def spy(res, res_i, edge_mask, node_mask):
        recorded["calls"] += 1
        recorded["edge_len"] = int(edge_mask.shape[0])
        recorded["node_len"] = int(node_mask.shape[0])
        return orig(res, res_i, edge_mask, node_mask)

    w._stitch_outputs = spy
    try:
        batch = {
            "pos": torch.zeros(n_node, 3),
            # two edges in expert-0 range [0,2), two in expert-1 range [2,10)
            "edge_lengths": torch.tensor([1.0, 1.5, 3.0, 4.0]),
        }
        out_res = w.forward(batch)
    finally:
        del w._stitch_outputs  # revert to the class method

    # _stitch_outputs was invoked once (for expert 1) with BOTH masks.
    assert recorded["calls"] == 1
    assert recorded["edge_len"] == n_edge
    assert recorded["node_len"] == n_node

    # Edge outputs stitched under expert-1's edge_mask ([F,F,T,T]).
    expected_edge = torch.zeros(n_edge, 2)
    expected_edge[2:] = 1.0
    assert torch.equal(out_res["edge_hamiltonian"], expected_edge)

    # Node outputs owned by expert 0 (expert-1 node_mask is all-False), so kept.
    assert torch.equal(out_res["node_hamiltonian"], torch.zeros(n_node, 2))


# ---------------------------------------------------------------------------
# e2e: forward() honors NON-DEFAULT node ownership by a non-zero expert
# ---------------------------------------------------------------------------
class _CustomNodeRoutingWrapper(DistanceEnsembleWrapper):
    """Routing override giving expert 1 genuine node ownership.

    Default routing assigns all nodes to the ``d_min == 0`` expert; this
    subclass reroutes nodes ``1..N-1`` to expert 1 (node 0 stays with expert 0)
    so the ensemble ``forward()`` must merge expert 1's node outputs.
    """

    def _build_expert_masks(self, batch, expert_idx):
        edge_mask, node_mask = super()._build_expert_masks(batch, expert_idx)
        custom = torch.zeros_like(node_mask)
        if expert_idx == 0:
            custom[0] = True
        elif expert_idx == 1:
            custom[1:] = True
        return edge_mask, custom


def test_forward_merges_node_outputs_owned_by_nonzero_expert():
    n_node, n_edge = 3, 4

    def out(offset):
        return {
            "node_hamiltonian": torch.zeros(n_node, 2) + float(offset),
            "node_h0": torch.zeros(n_node, 3) + 10.0 * float(offset),
            "edge_hamiltonian": torch.zeros(n_edge, 2) + float(offset),
        }

    experts = [_StubExpert(out(0)), _StubExpert(out(1))]
    w = _CustomNodeRoutingWrapper(
        experts=experts,
        distance_ranges=[(0.0, 3.0), (3.0, 6.0)],
        strict_output_spec=False,
    )

    batch = {
        "pos": torch.zeros(n_node, 3),
        # two edges in expert-0 range [0,3), two in expert-1 range [3,6)
        "edge_lengths": torch.tensor([1.0, 1.5, 3.0, 4.0]),
    }
    res = w.forward(batch)

    # Expert 1 genuinely owns nodes 1..2: its node outputs must land in the
    # merged result (this drives forward(), not just _stitch_outputs).
    expected_node = torch.zeros(n_node, 2)
    expected_node[1:] = 1.0
    assert torch.equal(res["node_hamiltonian"], expected_node)

    expected_h0 = torch.zeros(n_node, 3)
    expected_h0[1:] = 10.0
    assert torch.equal(res["node_h0"], expected_h0)

    # Edge stitching is unchanged: expert 1 owns edges 2..3 by distance.
    expected_edge = torch.zeros(n_edge, 2)
    expected_edge[2:] = 1.0
    assert torch.equal(res["edge_hamiltonian"], expected_edge)

    # The injected routing masks never leak into the merged output.
    assert "expert_edge_mask" not in res
    assert "expert_node_mask" not in res


def test_wrapper_documents_default_node_ownership_policy():
    # 1(c): the ownership policy (all nodes -> d_min == 0 expert) is routing
    # semantics, documented on the wrapper; the spec only fixed stitching.
    doc = DistanceEnsembleWrapper.__doc__ or ""
    assert "d_min == 0" in doc
    assert "ownership" in doc.lower()


def test_input_descriptor_fields_are_structural_not_stitched():
    # Regression (natlan P2 multi-expert smoke): node_attrs/edge_attrs are
    # INPUT descriptors (one-hot species / pair attributes) consumed by
    # embedding forwards whose backward needs the original values. They carry
    # node_/edge_ prefixes but must NOT be masked_replace-stitched — the
    # in-place write mutated a tensor still referenced by autograd and broke
    # loss.backward() with a version-counter error at the first iteration.
    spec = default_output_spec()
    for key in ("node_attrs", "edge_attrs", "edge_cell_shift", "edge_vectors"):
        s = spec.get(key)
        assert s is not None and s.alignment == "graph" and s.merge == "keep_first", key

    # And the stitch itself must leave them untouched.
    w = _make_wrapper(strict=False)
    res = {"node_attrs": torch.zeros(3, 2), "edge_attrs": torch.zeros(5, 4)}
    res_i = {"node_attrs": torch.ones(3, 2), "edge_attrs": torch.ones(5, 4)}
    node_mask = torch.tensor([True, True, True])
    edge_mask = torch.ones(5, dtype=torch.bool)
    w._stitch_outputs(res, res_i, edge_mask, node_mask)
    assert torch.equal(res["node_attrs"], torch.zeros(3, 2))
    assert torch.equal(res["edge_attrs"], torch.zeros(5, 4))

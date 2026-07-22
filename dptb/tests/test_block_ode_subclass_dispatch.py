"""RF1 characterization: subclass topology-key overrides must stay live.

The block-ode refactor (PR2) moved the block-topology contract out of
``HamiltonianCFM`` into plain functions in ``dptb.nnops.block_ode.topology``
and left the ``HamiltonianCFM`` methods as ``@classmethod`` delegators whose
``cls`` is dropped -- the delegators call the *module-global* helpers, which
in turn call each other module-globally.  The pre-refactor bodies used
``cls._block_primary_topology_keys()`` / ``cls._block_topology_keys()`` /
``cls._clone_sidecar_value()`` dynamic dispatch, so a ``HamiltonianCFM``
subclass that widened the block-topology authority set by overriding
``_block_primary_topology_keys`` participated in snapshot / restore / match.
After the refactor that override is silently bypassed (RF1, P1): the refactor
docstring itself promises "subclasses ... working unchanged", and a bypassed
authority key is a *silent* metadata-corruption failure mode (the model may
overwrite the field in one forward step and neither snapshot nor restore
guards it).

These tests define a *real* ``HamiltonianCFM`` subclass that extends the
primary topology keys and assert the override is honoured end to end.  They
FAIL on clean ``c7d097f`` (the override is bypassed) and PASS once the module
helpers dispatch overridable leaves through the owning class again.

The snapshot / restore / match entry points exercised here are ``@classmethod``
on ``HamiltonianCFM`` (they read no instance state), so the subclass is driven
through the class object directly -- no heavyweight ``__init__`` is needed and
the test stays fast and hermetic while still going through the real
production delegators and the real ``dptb.nnops.block_ode.topology`` module.
"""

from __future__ import annotations

import pytest
import torch

from dptb.nnops.block_ode import topology
from dptb.nnops.flow import HamiltonianCFM


_CUSTOM_KEY = "custom_row_identity"


class _TopoExtendedCFM(HamiltonianCFM):
    """A subclass that widens the immutable block-topology authority set.

    This is exactly the supported extension point RF1 concerns: a downstream
    model that adds a row-identity / edge-partition / routing-index field the
    block ODE must treat as authoritative (snapshot on entry, restore after a
    model step, cross-check data-vs-ref).
    """

    @staticmethod
    def _block_primary_topology_keys():
        base = HamiltonianCFM._block_primary_topology_keys()
        # dict.fromkeys-style dedupe happens downstream; just append.
        return (*base, _CUSTOM_KEY)


def test_base_class_does_not_carry_the_subclass_key():
    # Control: the override is genuinely a *subclass* extension; the base
    # class (and the plain module path) must NOT know about the custom key.
    assert _CUSTOM_KEY not in HamiltonianCFM._block_primary_topology_keys()
    assert _CUSTOM_KEY not in HamiltonianCFM._block_topology_keys()
    assert _CUSTOM_KEY not in topology._block_primary_topology_keys()
    assert _CUSTOM_KEY not in topology._block_topology_keys()


def test_subclass_override_reaches_block_topology_keys():
    # The widened primary set must propagate up through the derived-key
    # classmethod (_block_topology_keys calls _block_primary_topology_keys).
    primary = _TopoExtendedCFM._block_primary_topology_keys()
    full = _TopoExtendedCFM._block_topology_keys()
    assert _CUSTOM_KEY in primary
    assert _CUSTOM_KEY in full


def test_subclass_custom_key_is_snapshotted():
    source = {
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        _CUSTOM_KEY: torch.tensor([17]),
    }
    snapshot = _TopoExtendedCFM._snapshot_block_topology(source)
    # RF1: on clean c7d097f the module snapshot uses the module-global key set
    # and never records the subclass-added key.
    assert _CUSTOM_KEY in snapshot
    assert int(snapshot[_CUSTOM_KEY].item()) == 17
    # snapshot clones -- it must not alias the source tensor.
    assert snapshot[_CUSTOM_KEY].data_ptr() != source[_CUSTOM_KEY].data_ptr()


def test_subclass_custom_key_is_restored_after_model_overwrite():
    source = {
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        _CUSTOM_KEY: torch.tensor([17]),
    }
    snapshot = _TopoExtendedCFM._snapshot_block_topology(source)

    # Simulate a model step that silently overwrites the authority field.
    state = {
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        _CUSTOM_KEY: torch.tensor([99]),
    }
    _TopoExtendedCFM._restore_block_topology(state, snapshot)
    # RF1: bypassed dispatch leaves the model-written 99 in place; honoured
    # dispatch restores the authoritative 17.
    assert int(state[_CUSTOM_KEY].item()) == 17


def test_subclass_custom_key_is_dropped_when_absent_from_snapshot():
    # If the custom key was absent on entry, a model-returned value for it must
    # be dropped on restore (recompute from the immutable graph next step),
    # exactly like any other topology key.
    snapshot_without_custom = _TopoExtendedCFM._snapshot_block_topology(
        {"edge_index": torch.tensor([[0, 1], [1, 0]])}
    )
    assert _CUSTOM_KEY not in snapshot_without_custom
    state = {
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        _CUSTOM_KEY: torch.tensor([99]),
    }
    _TopoExtendedCFM._restore_block_topology(state, snapshot_without_custom)
    assert _CUSTOM_KEY not in state


def test_subclass_custom_key_participates_in_topology_match():
    # The cross-check between data and ref topology must also honour the
    # widened authority set: a mismatch on the subclass key must be rejected.
    left = _TopoExtendedCFM._snapshot_block_topology(
        {
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
            _CUSTOM_KEY: torch.tensor([17]),
        }
    )
    right = _TopoExtendedCFM._snapshot_block_topology(
        {
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
            _CUSTOM_KEY: torch.tensor([42]),
        }
    )
    with pytest.raises(ValueError, match="cannot be mixed"):
        _TopoExtendedCFM._require_matching_block_topology(left, right)

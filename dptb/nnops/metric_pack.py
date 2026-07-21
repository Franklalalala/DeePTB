"""Typed metric packs for MultiTrainer's distributed-reduction wire format.

Historically ``MultiTrainer`` packed per-step training metrics into flat 1-D
tensors addressed by *magic integer* indices (``_P_*``, ``_DB_*``) so they
could be shipped through ``torch.distributed.all_reduce`` / ``all_gather``.
The dataclasses in this module replace that positional protocol with named
fields **without changing the wire layout**: the element order produced by
``to_tensor()`` reproduces the original index layout EXACTLY, so the bytes an
``all_reduce`` / ``all_gather`` observes are byte-for-byte identical before and
after the refactor. Only the Python-side access changes (named field instead of
``pack[MultiTrainer._P_SOMETHING]``).

Wire compatibility with the legacy layout is pinned by
``dptb/tests/test_metric_pack.py`` (index-position assertions + round-trip
equality + a legacy-replica byte-equality check against the retained
``MultiTrainer._P_*`` / ``_DB_*`` constants).

Design notes
------------
* ``_ORDER`` (a plain class attribute, NOT a dataclass field) is the single
  source of truth for the field -> slot mapping. ``to_tensor`` / ``from_tensor``
  / ``index`` are all driven by it, so a slot cannot silently drift.
* ``to_tensor`` allocates ``torch.zeros`` and assigns only non-``None`` fields,
  mirroring the legacy builders which initialised a zero vector and left
  conditionally-unset slots (z / cv stats, absent dynamic-batch state) at zero.
* ``from_tensor`` wraps the *views* ``tensor[idx]`` (0-dim tensor views), so
  ``MetricPack.from_tensor(t).loss_opt_sum`` is exactly ``t[<slot>]`` — the same
  object semantics the old positional reads had.
"""

from dataclasses import dataclass, fields as _dataclass_fields
from typing import Any

import torch


class _PackMixin:
    """Shared ``to_tensor`` / ``from_tensor`` machinery driven by ``_ORDER``."""

    # Overridden by every concrete pack. Plain class attribute (no annotation)
    # so the @dataclass decorator never mistakes it for a field.
    _ORDER = ()

    @classmethod
    def index(cls, name: str) -> int:
        """Return the wire slot index for ``name`` (raises if unknown)."""
        return cls._ORDER.index(name)

    @classmethod
    def length(cls) -> int:
        return len(cls._ORDER)

    def to_tensor(self, *, dtype, device) -> torch.Tensor:
        """Serialize to the flat wire tensor.

        Allocates a zero vector of ``len(_ORDER)`` and assigns each non-None
        field into its slot. Assignment order is irrelevant to the produced
        bytes because every slot is distinct; only the field -> slot mapping
        (``_ORDER``) determines the layout.
        """
        order = type(self)._ORDER
        vec = torch.zeros((len(order),), dtype=dtype, device=device)
        for idx, name in enumerate(order):
            value = getattr(self, name)
            if value is not None:
                vec[idx] = value
        return vec

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "_PackMixin":
        """Wrap a flat wire tensor as named fields.

        Fields hold the 0-dim *views* ``tensor[idx]``, preserving the exact
        object/aliasing semantics of the legacy ``pack[_P_*]`` reads.
        """
        order = cls._ORDER
        return cls(**{name: tensor[idx] for idx, name in enumerate(order)})

    def _order_consistent_with_fields(self) -> bool:
        """True iff ``_ORDER`` names line up with the dataclass field order.

        Used by the tests to guarantee the declared field order and the wire
        order cannot diverge.
        """
        field_names = tuple(f.name for f in _dataclass_fields(self))
        return field_names == tuple(type(self)._ORDER)


@dataclass
class MetricPack(_PackMixin):
    """The 17-slot per-step training metric pack (legacy ``_P_*`` layout).

    ``active_nodes_sum`` and ``active_edges_sum`` retain their historical wire
    names and positions, but are the denominators paired with
    ``onsite_weighted_sum`` / ``hopping_weighted_sum``.  For payloads carrying
    cadence-gated metrics they therefore serialize ``onsite_weight`` /
    ``hopping_weight`` (zero when that metric was omitted), while raw active
    node/edge telemetry stays in :class:`ExpertDisplayMetric`.  This fixes the
    averaging semantics without widening or reordering the 17-slot protocol.

    Slot indices (must match ``MultiTrainer._P_*``)::

        0  loss_opt_sum          9  hopping_mse_sum
        1  onsite_weighted_sum  10  hopping_cnt_sum
        2  hopping_weighted_sum 11  z_sum
        3  active_nodes_sum     12  z_cnt
        4  active_edges_sum     13  cv_sum
        5  onsite_l1_sum        14  cv_cnt
        6  onsite_mse_sum       15  grad_norm_sum
        7  onsite_cnt_sum       16  step_count
        8  hopping_l1_sum
    """

    loss_opt_sum: Any = None
    onsite_weighted_sum: Any = None
    hopping_weighted_sum: Any = None
    active_nodes_sum: Any = None
    active_edges_sum: Any = None
    onsite_l1_sum: Any = None
    onsite_mse_sum: Any = None
    onsite_cnt_sum: Any = None
    hopping_l1_sum: Any = None
    hopping_mse_sum: Any = None
    hopping_cnt_sum: Any = None
    z_sum: Any = None
    z_cnt: Any = None
    cv_sum: Any = None
    cv_cnt: Any = None
    grad_norm_sum: Any = None
    step_count: Any = None

    _ORDER = (
        "loss_opt_sum",
        "onsite_weighted_sum",
        "hopping_weighted_sum",
        "active_nodes_sum",
        "active_edges_sum",
        "onsite_l1_sum",
        "onsite_mse_sum",
        "onsite_cnt_sum",
        "hopping_l1_sum",
        "hopping_mse_sum",
        "hopping_cnt_sum",
        "z_sum",
        "z_cnt",
        "cv_sum",
        "cv_cnt",
        "grad_norm_sum",
        "step_count",
    )
    LENGTH = 17


@dataclass
class DynamicBatchStat(_PackMixin):
    """The 7-slot dynamic-batch OOM/cost pack (legacy ``_DB_*`` layout).

    Slot indices (must match ``MultiTrainer._DB_*``)::

        0  num_graphs_sum      4  max_item_cost_sum
        1  cost_sum            5  step_count
        2  num_nodes_sum       6  oom_skipped_count
        3  num_edges_sum
    """

    num_graphs_sum: Any = None
    cost_sum: Any = None
    num_nodes_sum: Any = None
    num_edges_sum: Any = None
    max_item_cost_sum: Any = None
    step_count: Any = None
    oom_skipped_count: Any = None

    _ORDER = (
        "num_graphs_sum",
        "cost_sum",
        "num_nodes_sum",
        "num_edges_sum",
        "max_item_cost_sum",
        "step_count",
        "oom_skipped_count",
    )
    LENGTH = 7


@dataclass
class ExpertDisplayMetric(_PackMixin):
    """The 6-slot per-expert display gather vector (bare legacy layout).

    Slot indices::

        0  expert_onsite      3  lr
        1  expert_hopping     4  active_nodes
        2  grad_norm          5  active_edges

    ``grad_norm`` (slot 2) is written into the gather vector but never read back
    by the display flush; it is retained here so the wire vector stays 6-wide
    and byte-identical to the legacy ``torch.stack`` layout.
    """

    expert_onsite: Any = None
    expert_hopping: Any = None
    grad_norm: Any = None
    lr: Any = None
    active_nodes: Any = None
    active_edges: Any = None

    _ORDER = (
        "expert_onsite",
        "expert_hopping",
        "grad_norm",
        "lr",
        "active_nodes",
        "active_edges",
    )
    LENGTH = 6

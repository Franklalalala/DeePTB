"""Shared fixtures for the flow test files (not collected; import from dptb.tests.flow_helpers).

Kept minimal on purpose: only the handful of fixtures actually needed by more than
one flow test file live here (test_flow_core.py's pure flow-numerics tests and
test_flow_training_metrics.py's Trainer/MultiTrainer plumbing tests both build on
the same two-graph batch and the same stats-vs-forward loss stub).
"""
from __future__ import annotations

import torch

from dptb.data import _keys
from dptb.nnops.flow import HamiltonianCFM


def two_graph_batch():
    return {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        "node_h0": torch.zeros(3, 1),
        "edge_h0": torch.zeros(2, 1),
        "node_features": torch.zeros(3, 1),
        "edge_features": torch.zeros(2, 1),
    }


def two_graph_ref():
    return {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        "node_features": torch.full((3, 1), 2.0),
        "edge_features": torch.full((2, 1), 4.0),
    }


class StatsCompatibleLoss(torch.nn.Module):
    """A criterion whose stats-based fast path must match its own forward().

    ``compatible_loss_from_stats`` reconstructs the same L1+RMSE blend forward()
    would compute, from pre-reduced sums instead of a raw pred/ref recompute; the
    call counters let a test assert the fast path was used instead of forward().
    """

    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.stats_calls = 0
        self.onsite_boost = False
        self.element_average = False
        self.z_loss_coef = 0.0

    def forward(self, pred, ref):
        self.forward_calls += 1
        raise AssertionError("compatible logging must not re-run the full criterion")

    def compatible_loss_from_stats(
        self,
        *,
        onsite_l1_sum,
        onsite_mse_sum,
        onsite_count,
        hopping_l1_sum,
        hopping_mse_sum,
        hopping_count,
        z_loss=None,
        global_step=None,
    ):
        self.stats_calls += 1
        onsite = 0.5 * (
            onsite_l1_sum / onsite_count.clamp_min(1.0)
            + torch.sqrt(onsite_mse_sum / onsite_count.clamp_min(1.0) + 1e-12)
        )
        hopping = 0.5 * (
            hopping_l1_sum / hopping_count.clamp_min(1.0)
            + torch.sqrt(hopping_mse_sum / hopping_count.clamp_min(1.0) + 1e-12)
        )
        return 0.5 * (onsite + hopping), onsite, hopping


class BlockEndpointFallbackLoss(StatsCompatibleLoss):
    """A block-endpoint criterion with both a forward() and a stats fast path.

    Used both by the flow-level ``assert_model_in_loss_endpoint_metric_space``
    guard (metric-space label checks) and by the Trainer/MultiTrainer stats
    plumbing (raw-batch fallback vs stats fast path).
    """

    endpoint_metric_space = "block"

    def forward(self, pred, ref):
        self.forward_calls += 1
        self.last_endpoint_loss = torch.tensor(15.0)
        self.last_endpoint_metric_space = "block"
        self.last_onsite_loss = torch.tensor(10.0)
        self.last_hopping_loss = torch.tensor(20.0)
        self.last_onsite_l1_sum = torch.tensor(30.0)
        self.last_onsite_mse_sum = torch.tensor(300.0)
        self.last_onsite_count = torch.tensor(3.0)
        self.last_hopping_l1_sum = torch.tensor(40.0)
        self.last_hopping_mse_sum = torch.tensor(800.0)
        self.last_hopping_count = torch.tensor(2.0)
        return torch.tensor(100.0)


class FakeIr:
    def __init__(self, degree: int):
        self.l = degree
        self.dim = 2 * degree + 1


class FakeIrreps:
    """Minimal e3nn.Irreps-like object for flow-prior unit tests."""

    def __init__(self, items=None):
        self._items = list(items or [(1, FakeIr(0)), (1, FakeIr(1))])
        self.dim = sum(
            int(mul) * int(getattr(ir, "dim", 2 * getattr(ir, "l", 0) + 1))
            for mul, ir in self._items
        )

    def sort(self):
        return (self, None)

    def simplify(self):
        return self

    def __iter__(self):
        return iter(self._items)


class FakeIDP:
    """Minimal OrbitalMapper-like object: non-SOC, identity uureal<->RME layout."""

    def __init__(self, *, device: torch.device):
        self.orbpair_irreps = FakeIrreps()
        self.mask_to_nrme = torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 0],
            ],
            device=device,
            dtype=torch.bool,
        )
        self.mask_to_erme = torch.tensor(
            [
                [1, 1, 1, 1],
                [0, 1, 1, 0],
            ],
            device=device,
            dtype=torch.bool,
        )


def make_batch(*, device: torch.device, dtype: torch.dtype):
    """A 3-node/4-edge batch with node/edge H0 and a fixed target offset."""
    node_base = torch.arange(12, device=device, dtype=dtype).reshape(3, 4) / 100.0
    edge_base = torch.arange(16, device=device, dtype=dtype).reshape(4, 4) / 100.0
    node_target = node_base + torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 0.0, 0.0, 0.0],
            [1.5, 2.5, 3.5, 4.5],
        ],
        device=device,
        dtype=dtype,
    )
    edge_target = edge_base + torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 2.0, 3.0, 0.0],
            [1.5, 2.5, 3.5, 4.5],
            [0.0, 1.0, 2.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    data = {
        _keys.NODE_H0_KEY: node_base.clone(),
        _keys.EDGE_H0_KEY: edge_base.clone(),
        _keys.EDGE_INDEX_KEY: torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], device=device, dtype=torch.long
        ),
        _keys.BATCH_KEY: torch.tensor([0, 0, 1], device=device, dtype=torch.long),
        _keys.ATOM_TYPE_KEY: torch.tensor([0, 1, 0], device=device, dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long),
        _keys.NODE_FEATURES_KEY: node_base.clone(),
        _keys.EDGE_FEATURES_KEY: edge_base.clone(),
    }
    ref = {
        _keys.NODE_FEATURES_KEY: node_target,
        _keys.EDGE_FEATURES_KEY: edge_target,
    }
    return data, ref


def build_cfm(prior: str, *, device: torch.device, dtype: torch.dtype, **extra) -> HamiltonianCFM:
    opts = {
        "enabled": True,
        "mode": "residual",
        "prior": prior,
        "detach_interpolated_h0": False,
        "te_prior_sigma": 1.0,
        "te_prior_per_graph": False,
    }
    opts.update(extra)
    return HamiltonianCFM(opts, idp=FakeIDP(device=device), device=device, dtype=dtype)

from __future__ import annotations

import pytest
import torch

from dptb.data import AtomicDataDict, _keys
from dptb.nnops.flow import HamiltonianCFM
from dptb.utils.argcheck import flow_options


class _FakeIr:
    def __init__(self, degree: int):
        self.l = degree
        self.dim = 2 * degree + 1


class _FakeIrreps:
    """Minimal e3nn.Irreps-like object for flow-prior unit tests."""

    def __init__(self):
        self._items = [(1, _FakeIr(0)), (1, _FakeIr(1))]
        self.dim = 4

    def sort(self):
        return (self, None)

    def simplify(self):
        return self

    def __iter__(self):
        return iter(self._items)


class _FakeIDP:
    def __init__(self, *, device: torch.device):
        self.orbpair_irreps = _FakeIrreps()
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


def _make_batch(*, device: torch.device, dtype: torch.dtype):
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
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1, 0], device=device, dtype=torch.long),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long),
        _keys.NODE_FEATURES_KEY: node_base.clone(),
        _keys.EDGE_FEATURES_KEY: edge_base.clone(),
    }
    ref = {
        _keys.NODE_FEATURES_KEY: node_target,
        _keys.EDGE_FEATURES_KEY: edge_target,
    }
    return data, ref


def _flow(prior: str, *, device: torch.device, dtype: torch.dtype, **extra) -> HamiltonianCFM:
    opts = {
        "enabled": True,
        "mode": "residual",
        "prior": prior,
        "detach_interpolated_h0": False,
        "te_prior_sigma": 1.0,
        "te_prior_per_graph": False,
    }
    opts.update(extra)
    return HamiltonianCFM(opts, idp=_FakeIDP(device=device), device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_prior_residual_path_is_unchanged(dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("zero", device=device, dtype=dtype)

    t = torch.tensor([0.25, 0.75], device=device, dtype=dtype)
    out, _ref, ctx = flow.prepare_batch(data, ref, t=t)

    node_t = torch.tensor([0.25, 0.25, 0.75], device=device, dtype=dtype).reshape(3, 1)
    edge_t = torch.tensor([0.25, 0.25, 0.25, 0.75], device=device, dtype=dtype).reshape(4, 1)
    expected_node = data[_keys.NODE_H0_KEY] + node_t * (
        ref[_keys.NODE_FEATURES_KEY] - data[_keys.NODE_H0_KEY]
    )
    expected_edge = data[_keys.EDGE_H0_KEY] + edge_t * (
        ref[_keys.EDGE_FEATURES_KEY] - data[_keys.EDGE_H0_KEY]
    )

    assert torch.allclose(ctx.node_prior, torch.zeros_like(ctx.node_prior))
    assert torch.allclose(ctx.edge_prior, torch.zeros_like(ctx.edge_prior))
    assert torch.allclose(out[_keys.NODE_H0_KEY], expected_node)
    assert torch.allclose(out[_keys.EDGE_H0_KEY], expected_edge)


def test_te_prior_produces_nonzero_residual_noise():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode="irrep")

    torch.manual_seed(7)
    out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    assert ctx.node_prior.shape == ref[_keys.NODE_FEATURES_KEY].shape
    assert ctx.edge_prior.shape == ref[_keys.EDGE_FEATURES_KEY].shape
    assert ctx.node_prior.dtype == dtype
    assert ctx.edge_prior.device == ref[_keys.EDGE_FEATURES_KEY].device
    assert torch.count_nonzero(ctx.node_prior).item() > 0
    assert torch.count_nonzero(ctx.edge_prior).item() > 0
    assert torch.allclose(out[_keys.NODE_H0_KEY], data[_keys.NODE_H0_KEY] + ctx.node_prior)
    assert torch.allclose(out[_keys.EDGE_H0_KEY], data[_keys.EDGE_H0_KEY] + ctx.edge_prior)


def test_te_prior_respects_node_edge_masks_and_active_rows():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    data, ref = _make_batch(device=device, dtype=dtype)
    data["expert_node_mask"] = torch.tensor([1, 0, 1], device=device, dtype=torch.bool)
    data["expert_edge_mask"] = torch.tensor([1, 1, 0, 1], device=device, dtype=torch.bool)
    flow = _flow("structured_te", device=device, dtype=dtype, te_prior_mode="irrep")

    torch.manual_seed(11)
    _out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    assert torch.all(ctx.node_prior[1] == 0)
    assert torch.all(ctx.edge_prior[1, [0, 3]] == 0)
    assert torch.count_nonzero(ctx.edge_prior[1, [1, 2]]).item() > 0
    assert torch.all(ctx.edge_prior[2] == 0)


def test_te_prior_is_reproducible_under_deterministic_seed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode="irrep", te_prior_per_graph=True)
    t = torch.tensor([0.0, 0.0], device=device, dtype=dtype)

    torch.manual_seed(1234)
    _out1, _ref1, ctx1 = flow.prepare_batch(data, ref, t=t)
    torch.manual_seed(1234)
    _out2, _ref2, ctx2 = flow.prepare_batch(data, ref, t=t)
    torch.manual_seed(4321)
    _out3, _ref3, ctx3 = flow.prepare_batch(data, ref, t=t)

    assert torch.allclose(ctx1.node_prior, ctx2.node_prior)
    assert torch.allclose(ctx1.edge_prior, ctx2.edge_prior)
    assert not torch.allclose(ctx1.node_prior, ctx3.node_prior)


def test_te_prior_sampling_interface_keeps_existing_keys():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode="block")

    class _EndpointEcho(torch.nn.Module):
        def forward(self, batch):
            out = batch.copy()
            out[_keys.NODE_FEATURES_KEY] = batch[_keys.NODE_H0_KEY]
            out[_keys.EDGE_FEATURES_KEY] = batch[_keys.EDGE_H0_KEY]
            return out

    torch.manual_seed(99)
    sampled = flow.sample(_EndpointEcho(), data, num_steps=1)

    assert sampled[_keys.NODE_H0_KEY].shape == ref[_keys.NODE_FEATURES_KEY].shape
    assert sampled[_keys.EDGE_H0_KEY].shape == ref[_keys.EDGE_FEATURES_KEY].shape
    assert sampled[_keys.NODE_FEATURES_KEY].shape == ref[_keys.NODE_FEATURES_KEY].shape
    assert sampled[_keys.EDGE_FEATURES_KEY].shape == ref[_keys.EDGE_FEATURES_KEY].shape
    assert torch.allclose(sampled[flow.flow_time_key], torch.ones(2, device=device, dtype=dtype))


def test_validation_t0_call_with_te_prior_keeps_prepare_batch_signature():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode="typewise")

    torch.manual_seed(5)
    out, ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    assert out[flow.flow_time_key].shape == (2,)
    assert ref_out[flow.flow_time_key].shape == (2,)
    assert ctx.node_current.shape == ref[_keys.NODE_FEATURES_KEY].shape
    assert ctx.edge_current.shape == ref[_keys.EDGE_FEATURES_KEY].shape


def test_flow_options_argcheck_accepts_te_prior_config_keys():
    schema = flow_options()

    value = schema.normalize_value(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "te",
            "te_prior_sigma": 0.5,
            "te_prior_mode": "typewise",
            "te_prior_per_graph": False,
        }
    )
    schema.check_value(value, strict=True)

    assert value["prior"] == "te"
    assert value["te_prior_sigma"] == pytest.approx(0.5)
    assert value["te_prior_mode"] == "typewise"
    assert value["te_prior_per_graph"] is False

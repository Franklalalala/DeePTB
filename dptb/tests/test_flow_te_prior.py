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

    def __init__(self, items=None):
        self._items = list(items or [(1, _FakeIr(0)), (1, _FakeIr(1))])
        self.dim = 4

    def sort(self):
        return (self, None)

    def simplify(self):
        return self

    def __iter__(self):
        return iter(self._items)


class _UnsortedFakeIrreps(_FakeIrreps):
    def __init__(self):
        super().__init__([(1, _FakeIr(1)), (1, _FakeIr(0))])

    def sort(self):
        return (_FakeIrreps([(1, _FakeIr(0)), (1, _FakeIr(1))]), None)


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


class _UnsortedIrrepIDP(_FakeIDP):
    def __init__(self, *, device: torch.device):
        super().__init__(device=device)
        self.orbpair_irreps = _UnsortedFakeIrreps()


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


def test_te_irrep_slices_preserve_raw_feature_order():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    opts = {
        "enabled": True,
        "mode": "residual",
        "prior": "te",
        "te_prior_mode": "irrep",
    }
    flow = HamiltonianCFM(opts, idp=_UnsortedIrrepIDP(device=device), device=device, dtype=dtype)

    assert flow._te_irrep_slices(4) == ((0, 3, 1), (3, 4, 0))


@pytest.mark.parametrize("mode", ["irrep", "typewise"])
def test_te_prior_irrep_modes_fail_loud_on_layout_mismatch(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode=mode)
    flow.idp.orbpair_irreps = _FakeIrreps([(1, _FakeIr(1))])

    with pytest.raises(ValueError, match="orbpair_irreps raw feature spans"):
        flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))


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


def test_te_prior_mask_alignment_fails_closed_for_present_but_short_masks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    data[AtomicDataDict.ATOM_TYPE_KEY] = data[AtomicDataDict.ATOM_TYPE_KEY][:2]
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode="irrep")
    flow.idp.mask_to_nrme = flow.idp.mask_to_nrme[:, :2]

    torch.manual_seed(17)
    _out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    assert torch.all(ctx.node_prior[2] == 0)
    assert torch.all(ctx.node_prior[:, 2:] == 0)


def test_typewise_te_prior_uses_unsorted_raw_slices_end_to_end():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode="typewise")
    flow.idp = _UnsortedIrrepIDP(device=device)
    flow.idp.mask_to_nrme = torch.tensor(
        [
            [1, 1, 0, 1],
            [0, 0, 0, 0],
        ],
        device=device,
        dtype=torch.bool,
    )
    flow.idp.mask_to_erme = torch.zeros((2, 4), device=device, dtype=torch.bool)

    def unit_radius(row_count, active_dim, graph_index, *, device, dtype):
        return active_dim.to(device=device, dtype=dtype).sqrt()

    flow._te_radius = unit_radius

    torch.manual_seed(23)
    _out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    node_res = ref[_keys.NODE_FEATURES_KEY] - data[_keys.NODE_H0_KEY]
    type0_rows = data[AtomicDataDict.ATOM_TYPE_KEY] == 0
    raw_l1_scale = node_res[type0_rows][:, :3][:, [0, 1]].square().mean().sqrt()
    raw_l0_scale = node_res[type0_rows][:, 3].square().mean().sqrt()

    torch.testing.assert_close(
        torch.linalg.vector_norm(ctx.node_prior[0, :3]),
        raw_l1_scale * torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype)),
    )
    torch.testing.assert_close(torch.linalg.vector_norm(ctx.node_prior[0, 3:4]), raw_l0_scale)
    torch.testing.assert_close(
        torch.linalg.vector_norm(ctx.node_prior[2, :3]),
        raw_l1_scale * torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype)),
    )
    torch.testing.assert_close(torch.linalg.vector_norm(ctx.node_prior[2, 3:4]), raw_l0_scale)
    assert torch.all(ctx.node_prior[type0_rows, 2] == 0)
    assert torch.all(ctx.node_prior[~type0_rows] == 0)
    assert torch.all(ctx.edge_prior == 0)


def test_block_te_alias_defaults_to_block_mode_unless_explicit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    flow = _flow("block-te", device=device, dtype=dtype)
    explicit = _flow("block_te", device=device, dtype=dtype, te_prior_mode="irrep")

    assert flow.prior == "block_te"
    assert flow.te_prior_mode == "block"
    assert explicit.te_prior_mode == "irrep"


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

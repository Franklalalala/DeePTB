from __future__ import annotations

import inspect

import pytest
import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.transforms_upper_triangle import OrbitalMapper
from dptb.nnops.onsite_database import onsite_energy_database
from dptb.nnops import prior_physical
from dptb.nnops.flow import (
    CFMContext,
    HamiltonianCFM,
    HamiltonianPixelMeanFlow,
    assert_flow_h0_keys_reach_model,
    build_hamiltonian_flow,
)
from dptb.nnops.layout import project_uureal_to_like
from dptb.nnops.loss import HamilLossAbs
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.nnops.trainer import Trainer
from dptb.utils.argcheck import common_options, flow_options


class _FakeIr:
    def __init__(self, degree: int):
        self.l = degree
        self.dim = 2 * degree + 1


class _FakeIrreps:
    """Minimal e3nn.Irreps-like object for flow-prior unit tests."""

    def __init__(self, items=None):
        self._items = list(items or [(1, _FakeIr(0)), (1, _FakeIr(1))])
        self.dim = sum(int(mul) * int(getattr(ir, "dim", 2 * getattr(ir, "l", 0) + 1)) for mul, ir in self._items)

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


class _LazyIrrepIDP(_FakeIDP):
    def __init__(self, *, device: torch.device):
        super().__init__(device=device)
        self.orbpair_irreps = None
        self.get_irreps_calls = 0

    def get_irreps(self):
        self.get_irreps_calls += 1
        self.orbpair_irreps = _FakeIrreps()
        return self.orbpair_irreps


class _CompactUuRealIDP:
    nextham_uureal_mask = True
    has_soc = True
    soc_complex_doubling = True

    def __init__(self):
        self.mask_uureal = torch.ones(5, dtype=torch.bool)
        self.orbpair_maps = {
            "1s-1s": slice(0, 2),
            "1s-2p": slice(2, 5),
        }

    def get_orbpair_maps(self):
        return self.orbpair_maps


class _CompactUuRealNoMaskIDP(_CompactUuRealIDP):
    def __init__(self):
        super().__init__()
        del self.mask_uureal


class _FakeCompatibleLoss:
    supports_endpoint_triplet = True
    onsite_boost = False
    z_loss_coef = 0.0

    def __call__(self, pred_data, ref_data):
        self.last_onsite_loss = torch.tensor(1.25)
        self.last_hopping_loss = torch.tensor(0.25)
        return torch.tensor(0.75)


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


def test_split_external_prior_can_mix_node_overlap_with_edge_h0():
    device = torch.device("cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    node_overlap = data[_keys.NODE_H0_KEY] + 0.25
    data["node_overlap"] = node_overlap
    flow = _flow(
        "external",
        device=device,
        dtype=dtype,
        prior_node="external",
        prior_edge="external",
        prior_node_key="node_overlap",
        prior_edge_key=_keys.EDGE_H0_KEY,
        external_prior_strict=True,
    )

    model_data, _ref, ctx = flow.prepare_batch(
        data, ref, t=torch.zeros(2, device=device, dtype=dtype)
    )

    torch.testing.assert_close(ctx.node_current, node_overlap)
    torch.testing.assert_close(ctx.edge_current, data[_keys.EDGE_H0_KEY])
    torch.testing.assert_close(model_data[_keys.NODE_H0_KEY], node_overlap)
    torch.testing.assert_close(model_data[_keys.EDGE_H0_KEY], data[_keys.EDGE_H0_KEY])


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


def test_project_uureal_to_like_handles_full_soc_overlap_for_compact_idp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    idp = _CompactUuRealIDP()
    like = torch.zeros(2, 5, device=device, dtype=dtype)
    raw = torch.zeros(2, 40, device=device, dtype=dtype)
    raw[:, 0:2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
    raw[:, 16:19] = torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], device=device, dtype=dtype)
    raw[:, 2:16] = 100.0
    raw[:, 19:] = 200.0

    projected, mask = project_uureal_to_like(idp, raw, like)

    expected = torch.cat([raw[:, 0:2], raw[:, 16:19]], dim=-1)
    torch.testing.assert_close(projected, expected)
    assert mask is not None
    assert int(mask.sum().item()) == 5


def test_project_uureal_to_like_infers_full_soc_overlap_without_mask_uureal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    idp = _CompactUuRealNoMaskIDP()
    like = torch.zeros(2, 5, device=device, dtype=dtype)
    raw = torch.zeros(2, 40, device=device, dtype=dtype)
    raw[:, 0:2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
    raw[:, 16:19] = torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], device=device, dtype=dtype)

    projected, mask = project_uureal_to_like(idp, raw, like)

    expected = torch.cat([raw[:, 0:2], raw[:, 16:19]], dim=-1)
    torch.testing.assert_close(projected, expected)
    assert mask is not None
    assert int(mask.sum().item()) == 5


@pytest.mark.parametrize("mode", ["irrep", "typewise"])
def test_te_prior_irrep_modes_fail_loud_on_layout_mismatch(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode=mode)
    flow.idp.orbpair_irreps = _FakeIrreps([(1, _FakeIr(1))])

    with pytest.raises(ValueError, match="orbpair_irreps raw feature spans"):
        flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))


def test_typewise_te_prior_lazily_initializes_orbpair_irreps():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode="typewise")
    flow.idp = _LazyIrrepIDP(device=device)

    flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    assert flow.idp.get_irreps_calls == 1
    assert flow.idp.orbpair_irreps is not None


def test_typewise_residual_scale_stays_on_device_and_matches_group_reduction():
    source = inspect.getsource(HamiltonianCFM._apply_typewise_residual_scale)
    assert ".cpu()" not in source
    assert ".item()" not in source

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode="typewise")
    noise = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0],
         [2.0, 3.0, 4.0, 5.0],
         [3.0, 4.0, 5.0, 6.0]],
        device=device,
        dtype=dtype,
    )
    reference = torch.tensor(
        [[2.0, 4.0, 3.0, 5.0],
         [7.0, 8.0, 0.0, 0.0],
         [6.0, 8.0, 7.0, 9.0]],
        device=device,
        dtype=dtype,
    )
    mask = torch.tensor(
        [[1, 1, 1, 1],
         [1, 1, 0, 0],
         [1, 1, 1, 1]],
        device=device,
        dtype=torch.bool,
    )
    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(
            [0, 1, 0], device=device, dtype=torch.long
        )
    }
    slices = ((0, 2, 1), (2, 4, 0))

    out = flow._apply_typewise_residual_scale(
        noise, reference, mask, data, "node", slices, reference_scale=True
    )

    type0_first = torch.tensor([2.0, 4.0, 6.0, 8.0], device=device, dtype=dtype).square().mean().sqrt()
    type1_first = torch.tensor([7.0, 8.0], device=device, dtype=dtype).square().mean().sqrt()
    type0_second = torch.tensor([3.0, 5.0, 7.0, 9.0], device=device, dtype=dtype).square().mean().sqrt()
    expected = noise.clone()
    expected[[0, 2], 0:2] *= type0_first
    expected[1, 0:2] *= type1_first
    expected[[0, 2], 2:4] *= type0_second
    torch.testing.assert_close(out, expected)


def test_typewise_te_prior_projects_uureal_raw_layout_to_compressed_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    data, _ref = _make_batch(device=device, dtype=dtype)
    ref = {
        _keys.NODE_FEATURES_KEY: data[_keys.NODE_H0_KEY]
        + torch.tensor(
            [
                [2.0, 4.0, 3.0, 5.0],
                [0.0, 0.0, 0.0, 0.0],
                [6.0, 8.0, 7.0, 9.0],
            ],
            device=device,
            dtype=dtype,
        ),
        _keys.EDGE_FEATURES_KEY: data[_keys.EDGE_H0_KEY],
    }
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode="typewise")
    flow.idp.orbpair_irreps = _FakeIrreps(
        [
            (1, _FakeIr(1)),
            (1, _FakeIr(0)),
            (1, _FakeIr(1)),
            (1, _FakeIr(0)),
        ]
    )
    flow.idp.mask_uureal = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    flow.idp.mask_to_nrme = torch.tensor(
        [
            [1, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        device=device,
        dtype=torch.bool,
    )
    flow.idp.mask_to_erme = torch.zeros((2, 8), device=device, dtype=torch.bool)

    def unit_radius(row_count, active_dim, graph_index, *, device, dtype):
        return active_dim.to(device=device, dtype=dtype).sqrt()

    flow._te_radius = unit_radius

    torch.manual_seed(29)
    _out, _ref_out, ctx = flow.prepare_batch(
        data, ref, t=torch.zeros(2, device=device, dtype=dtype)
    )

    type0_rows = data[AtomicDataDict.ATOM_TYPE_KEY] == 0
    first_l1_scale = torch.tensor([2.0, 4.0, 6.0, 8.0], device=device, dtype=dtype).square().mean().sqrt()
    l0_scale = torch.tensor([3.0, 7.0], device=device, dtype=dtype).square().mean().sqrt()
    second_l1_scale = torch.tensor([5.0, 9.0], device=device, dtype=dtype).square().mean().sqrt()

    torch.testing.assert_close(
        torch.linalg.vector_norm(ctx.node_prior[0, 0:2]),
        first_l1_scale * torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype)),
    )
    torch.testing.assert_close(torch.linalg.vector_norm(ctx.node_prior[0, 2:3]), l0_scale)
    torch.testing.assert_close(torch.linalg.vector_norm(ctx.node_prior[0, 3:4]), second_l1_scale)
    assert torch.all(ctx.node_prior[type0_rows, 0:2] != 0)
    assert torch.all(ctx.node_prior[type0_rows, 2:4] != 0)
    assert torch.all(ctx.node_prior[~type0_rows] == 0)
    assert torch.all(ctx.edge_prior == 0)


def test_flow_loss_projects_raw_uureal_predictions_to_compressed_targets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    raw_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    target = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ],
        device=device,
        dtype=dtype,
    )
    pred_raw = torch.zeros(3, 8, device=device, dtype=dtype)
    pred_raw[:, raw_mask] = target + 0.5

    flow = _flow("zero", device=device, dtype=dtype)
    flow.idp.mask_uureal = raw_mask
    flow.idp.mask_to_nrme = raw_mask.expand(2, -1).clone()
    flow.idp.mask_to_erme = raw_mask.expand(2, -1).clone()

    pred_data = {
        _keys.NODE_FEATURES_KEY: pred_raw,
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1, 0], device=device, dtype=torch.long),
    }
    ref_data = {_keys.NODE_FEATURES_KEY: target}
    ctx = CFMContext(
        t=torch.zeros(1, device=device, dtype=dtype),
        node_t=torch.zeros(3, device=device, dtype=dtype),
        edge_t=None,
        node_base=None,
        edge_base=None,
        node_target=target,
        edge_target=None,
        node_current=None,
        edge_current=None,
        node_prior=None,
        edge_prior=None,
    )

    loss, state = flow.loss(pred_data, ref_data, ctx)

    expected = torch.full((), 0.25, device=device, dtype=dtype)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(state["train_flow_onsite_loss"], expected)


def test_flow_loss_fails_closed_when_raw_orbital_mask_cannot_match_compressed_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    flow = _flow("zero", device=device, dtype=dtype)
    flow.idp.mask_uureal = torch.tensor([1, 1, 0, 1, 0, 1, 0], device=device, dtype=torch.bool)
    flow.idp.mask_to_nrme = torch.ones((2, 8), device=device, dtype=torch.bool)

    target = torch.zeros(3, 4, device=device, dtype=dtype)
    pred_data = {
        _keys.NODE_FEATURES_KEY: torch.ones_like(target),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1, 0], device=device, dtype=torch.long),
    }
    ref_data = {_keys.NODE_FEATURES_KEY: target}
    ctx = CFMContext(
        t=torch.zeros(1, device=device, dtype=dtype),
        node_t=torch.zeros(3, device=device, dtype=dtype),
        edge_t=None,
        node_base=None,
        edge_base=None,
        node_target=target,
        edge_target=None,
        node_current=None,
        edge_current=None,
        node_prior=None,
        edge_prior=None,
    )

    with pytest.raises(ValueError, match="node idp mask layout"):
        flow.loss(pred_data, ref_data, ctx)


def test_flow_loss_keeps_compressed_mask_table_when_prediction_is_raw_uureal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    raw_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    compressed_mask = torch.tensor([1, 0, 1, 0], device=device, dtype=torch.bool)
    target = torch.zeros(3, 4, device=device, dtype=dtype)
    pred_raw = torch.zeros(3, raw_mask.numel(), device=device, dtype=dtype)
    pred_raw[:, raw_mask] = torch.tensor([1.0, 10.0, 2.0, 10.0], device=device, dtype=dtype)

    flow = _flow("zero", device=device, dtype=dtype)
    flow.idp.mask_uureal = raw_mask
    flow.idp.mask_to_nrme = compressed_mask.expand(2, -1).clone()

    pred_data = {
        _keys.NODE_FEATURES_KEY: pred_raw,
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1, 0], device=device, dtype=torch.long),
    }
    ref_data = {_keys.NODE_FEATURES_KEY: target}
    ctx = CFMContext(
        t=torch.zeros(1, device=device, dtype=dtype),
        node_t=torch.zeros(3, device=device, dtype=dtype),
        edge_t=None,
        node_base=None,
        edge_base=None,
        node_target=target,
        edge_target=None,
        node_current=None,
        edge_current=None,
        node_prior=None,
        edge_prior=None,
    )

    loss, state = flow.loss(pred_data, ref_data, ctx)

    expected = torch.full((), 2.5, device=device, dtype=dtype)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(state["train_flow_onsite_loss"], expected)


def test_hamil_abs_compatible_loss_projects_raw_uureal_predictions_to_compressed_targets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    raw_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    idp = _FakeIDP(device=device)
    idp.mask_uureal = raw_mask
    idp.mask_to_nrme = raw_mask.expand(2, -1).clone()
    idp.mask_to_erme = raw_mask.expand(2, -1).clone()
    lossfunc = HamilLossAbs(idp=idp)

    node_target = torch.arange(12, device=device, dtype=dtype).reshape(3, 4) / 10.0
    edge_target = torch.arange(16, device=device, dtype=dtype).reshape(4, 4) / 10.0
    node_pred_raw = torch.zeros(3, raw_mask.numel(), device=device, dtype=dtype)
    edge_pred_raw = torch.zeros(4, raw_mask.numel(), device=device, dtype=dtype)
    node_pred_raw[:, raw_mask] = node_target + 0.5
    edge_pred_raw[:, raw_mask] = edge_target + 0.25

    pred_data = {
        _keys.NODE_FEATURES_KEY: node_pred_raw,
        _keys.EDGE_FEATURES_KEY: edge_pred_raw,
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1, 0], device=device, dtype=torch.long),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long),
    }
    ref_data = {
        _keys.NODE_FEATURES_KEY: node_target,
        _keys.EDGE_FEATURES_KEY: edge_target,
    }

    state = Trainer._compatible_loss_state(lossfunc, pred_data, ref_data, prefix="train")

    assert state["train_onsite_loss"].item() == pytest.approx(0.5)
    assert state["train_hopping_loss"].item() == pytest.approx(0.25)
    assert state["train_loss"].item() == pytest.approx(0.375)


def test_flow_sample_projects_raw_uureal_endpoint_to_compressed_state():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    raw_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("zero", device=device, dtype=dtype)
    flow.idp.mask_uureal = raw_mask

    class _RawEndpoint(torch.nn.Module):
        def forward(self, batch):
            out = batch.copy()
            node_raw = torch.zeros(
                batch[_keys.NODE_H0_KEY].shape[0],
                raw_mask.numel(),
                device=batch[_keys.NODE_H0_KEY].device,
                dtype=batch[_keys.NODE_H0_KEY].dtype,
            )
            edge_raw = torch.zeros(
                batch[_keys.EDGE_H0_KEY].shape[0],
                raw_mask.numel(),
                device=batch[_keys.EDGE_H0_KEY].device,
                dtype=batch[_keys.EDGE_H0_KEY].dtype,
            )
            node_raw[:, raw_mask] = ref[_keys.NODE_FEATURES_KEY].to(node_raw)
            edge_raw[:, raw_mask] = ref[_keys.EDGE_FEATURES_KEY].to(edge_raw)
            out[_keys.NODE_FEATURES_KEY] = node_raw
            out[_keys.EDGE_FEATURES_KEY] = edge_raw
            return out

    sampled = flow.sample(_RawEndpoint(), data, num_steps=1)

    assert sampled[_keys.NODE_FEATURES_KEY].shape == ref[_keys.NODE_FEATURES_KEY].shape
    assert sampled[_keys.EDGE_FEATURES_KEY].shape == ref[_keys.EDGE_FEATURES_KEY].shape
    torch.testing.assert_close(sampled[_keys.NODE_FEATURES_KEY], ref[_keys.NODE_FEATURES_KEY])
    torch.testing.assert_close(sampled[_keys.EDGE_FEATURES_KEY], ref[_keys.EDGE_FEATURES_KEY])


def test_pixel_meanflow_projects_raw_uureal_endpoint_to_compressed_loss_layout():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    raw_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    data, ref = _make_batch(device=device, dtype=dtype)
    idp = _FakeIDP(device=device)
    idp.mask_uureal = raw_mask
    idp.mask_to_nrme = raw_mask.expand(2, -1).clone()
    idp.mask_to_erme = raw_mask.expand(2, -1).clone()
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
            "meanflow": {
                "aux_endpoint_weight": 0.0,
                "jvp_tangent": "path",
            },
        },
        idp=idp,
        device=device,
        dtype=dtype,
    )

    class _RawEndpoint(torch.nn.Module):
        def forward(self, batch):
            out = batch.copy()
            node_raw = torch.zeros(
                batch[_keys.NODE_H0_KEY].shape[0],
                raw_mask.numel(),
                device=batch[_keys.NODE_H0_KEY].device,
                dtype=batch[_keys.NODE_H0_KEY].dtype,
            )
            edge_raw = torch.zeros(
                batch[_keys.EDGE_H0_KEY].shape[0],
                raw_mask.numel(),
                device=batch[_keys.EDGE_H0_KEY].device,
                dtype=batch[_keys.EDGE_H0_KEY].dtype,
            )
            node_raw[:, raw_mask] = ref[_keys.NODE_FEATURES_KEY].to(node_raw)
            edge_raw[:, raw_mask] = ref[_keys.EDGE_FEATURES_KEY].to(edge_raw)
            out[_keys.NODE_FEATURES_KEY] = node_raw
            out[_keys.EDGE_FEATURES_KEY] = edge_raw
            return out

    loss, state = flow.loss_with_model(
        _RawEndpoint(),
        data,
        ref,
        r=torch.tensor([0.25, 0.25], device=device, dtype=dtype),
        t=torch.tensor([0.50, 0.50], device=device, dtype=dtype),
    )

    assert torch.isfinite(loss)
    assert state["train_flow_onsite_endpoint_loss"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert state["train_flow_hopping_endpoint_loss"].item() == pytest.approx(0.0, abs=1.0e-8)


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


def test_full_mode_external_sampling_uses_absolute_prior_once():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, _ref = _make_batch(device=device, dtype=dtype)
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "full",
            "prior": "external",
            "prior_node_key": _keys.NODE_H0_KEY,
            "prior_edge_key": _keys.EDGE_H0_KEY,
        },
        idp=_FakeIDP(device=device),
        device=device,
        dtype=dtype,
    )

    class _EchoCurrentH0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_node_h0 = None
            self.first_edge_h0 = None

        def forward(self, batch):
            if self.first_node_h0 is None:
                self.first_node_h0 = batch[_keys.NODE_H0_KEY].detach().clone()
                self.first_edge_h0 = batch[_keys.EDGE_H0_KEY].detach().clone()
            out = batch.copy()
            out[_keys.NODE_FEATURES_KEY] = batch[_keys.NODE_H0_KEY]
            out[_keys.EDGE_FEATURES_KEY] = batch[_keys.EDGE_H0_KEY]
            return out

    model = _EchoCurrentH0()

    sampled = flow.sample(model, data, num_steps=1)

    torch.testing.assert_close(model.first_node_h0, data[_keys.NODE_H0_KEY])
    torch.testing.assert_close(model.first_edge_h0, data[_keys.EDGE_H0_KEY])
    torch.testing.assert_close(sampled[_keys.NODE_FEATURES_KEY], data[_keys.NODE_H0_KEY])
    torch.testing.assert_close(sampled[_keys.EDGE_FEATURES_KEY], data[_keys.EDGE_H0_KEY])


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


def test_te_prior_passes_num_graphs_to_radius_once_per_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("te", device=device, dtype=dtype, te_prior_mode="irrep")
    seen_num_graphs = []

    def radius(row_count, active_dim, graph_index, *, device, dtype, num_graphs=None):
        seen_num_graphs.append(num_graphs)
        return active_dim.to(device=device, dtype=dtype).sqrt()

    flow._te_radius = radius

    flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    assert seen_num_graphs
    assert all(value == 2 for value in seen_num_graphs)


def _basis_onsite_case(device: torch.device, dtype: torch.dtype):
    idp = OrbitalMapper(
        {"H": ["1s"], "C": ["2s", "2p"]},
        method="e3tb",
        device=device,
    )
    node = torch.zeros(2, idp.reduced_matrix_element, device=device, dtype=dtype)
    edge = torch.zeros(2, idp.reduced_matrix_element, device=device, dtype=dtype)
    data = {
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.long),
        _keys.BATCH_KEY: torch.zeros(2, device=device, dtype=torch.long),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(
            [idp.chemical_symbol_to_type["H"], idp.chemical_symbol_to_type["C"]],
            device=device,
            dtype=torch.long,
        ),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor(
            [idp.bond_to_type["H-C"], idp.bond_to_type["C-H"]],
            device=device,
            dtype=torch.long,
        ),
        _keys.NODE_FEATURES_KEY: node.clone(),
        _keys.EDGE_FEATURES_KEY: edge.clone(),
    }
    ref = {
        _keys.NODE_FEATURES_KEY: node.clone(),
        _keys.EDGE_FEATURES_KEY: edge.clone(),
    }
    return idp, data, ref


def _onsite_diag_indices(idp: OrbitalMapper, symbol: str, orbital: str, device: torch.device):
    full = idp.basis_to_full_basis[symbol][orbital]
    block = idp.orbpair_maps[f"{full}-{full}"]
    degree = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}[full[-1]]
    width = 2 * degree + 1
    diag = torch.arange(width, device=device, dtype=torch.long)
    return int(block.start) + diag * width + diag


def test_basis_onsite_prior_initializes_without_h0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    idp, data, ref = _basis_onsite_case(device, dtype)
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "basis_onsite",
            "strict_h0": False,
            "warn_missing_h0": False,
            "detach_interpolated_h0": False,
        },
        idp=idp,
        device=device,
        dtype=dtype,
    )

    out, _ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    expected = torch.zeros_like(ref[_keys.NODE_FEATURES_KEY])
    expected[0, _onsite_diag_indices(idp, "H", "1s", device)] = onsite_energy_database["H"]["1s"]
    expected[1, _onsite_diag_indices(idp, "C", "2s", device)] = onsite_energy_database["C"]["2s"]
    expected[1, _onsite_diag_indices(idp, "C", "2p", device)] = onsite_energy_database["C"]["2p"]

    torch.testing.assert_close(out[_keys.NODE_H0_KEY], expected)
    torch.testing.assert_close(ctx.node_prior, expected)
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], torch.zeros_like(ref[_keys.EDGE_FEATURES_KEY]))


def test_basis_onsite_prior_is_residualized_against_existing_h0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    idp, data, ref = _basis_onsite_case(device, dtype)
    node_base = torch.full_like(ref[_keys.NODE_FEATURES_KEY], 0.25)
    edge_base = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.5)
    data[_keys.NODE_H0_KEY] = node_base
    data[_keys.EDGE_H0_KEY] = edge_base
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "basis_onsite",
            "detach_interpolated_h0": False,
        },
        idp=idp,
        device=device,
        dtype=dtype,
    )

    out, _ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    expected_abs = torch.zeros_like(ref[_keys.NODE_FEATURES_KEY])
    expected_abs[0, _onsite_diag_indices(idp, "H", "1s", device)] = onsite_energy_database["H"]["1s"]
    expected_abs[1, _onsite_diag_indices(idp, "C", "2s", device)] = onsite_energy_database["C"]["2s"]
    expected_abs[1, _onsite_diag_indices(idp, "C", "2p", device)] = onsite_energy_database["C"]["2p"]

    torch.testing.assert_close(out[_keys.NODE_H0_KEY], expected_abs)
    torch.testing.assert_close(ctx.node_prior, expected_abs - node_base)
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], torch.zeros_like(edge_base))
    torch.testing.assert_close(ctx.edge_prior, -edge_base)


def test_overlap_huckel_prior_uses_overlap_and_basis_without_h0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    idp, data, ref = _basis_onsite_case(device, dtype)
    edge_overlap = torch.linspace(
        -0.3,
        0.4,
        steps=ref[_keys.EDGE_FEATURES_KEY].numel(),
        device=device,
        dtype=dtype,
    ).reshape_as(ref[_keys.EDGE_FEATURES_KEY])
    data[_keys.EDGE_OVERLAP_KEY] = edge_overlap
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "overlap_huckel",
            "strict_h0": False,
            "warn_missing_h0": False,
            "detach_interpolated_h0": False,
            "huckel_k": 2.0,
        },
        idp=idp,
        device=device,
        dtype=dtype,
    )

    out, _ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))
    out_again, _ref_again, ctx_again = flow.prepare_batch(
        data,
        ref,
        t=torch.zeros(1, device=device, dtype=dtype),
    )

    expected_node = torch.zeros_like(ref[_keys.NODE_FEATURES_KEY])
    expected_node[0, _onsite_diag_indices(idp, "H", "1s", device)] = onsite_energy_database["H"]["1s"]
    expected_node[1, _onsite_diag_indices(idp, "C", "2s", device)] = onsite_energy_database["C"]["2s"]
    expected_node[1, _onsite_diag_indices(idp, "C", "2p", device)] = onsite_energy_database["C"]["2p"]
    h_mean = torch.tensor(onsite_energy_database["H"]["1s"], device=device, dtype=dtype)
    c_mean = torch.tensor(
        (onsite_energy_database["C"]["2s"] + 3.0 * onsite_energy_database["C"]["2p"]) / 4.0,
        device=device,
        dtype=dtype,
    )
    edge_energy = 0.5 * (h_mean + c_mean)
    expected_edge = 2.0 * edge_energy * edge_overlap
    expected_edge = expected_edge * flow._prior_mask(data, expected_edge, "edge").to(dtype=dtype)

    torch.testing.assert_close(out[_keys.NODE_H0_KEY], expected_node)
    torch.testing.assert_close(ctx.node_prior, expected_node)
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], expected_edge)
    torch.testing.assert_close(ctx.edge_prior, expected_edge)
    torch.testing.assert_close(out_again[_keys.NODE_H0_KEY], out[_keys.NODE_H0_KEY])
    torch.testing.assert_close(ctx_again.node_prior, ctx.node_prior)
    torch.testing.assert_close(out_again[_keys.EDGE_H0_KEY], out[_keys.EDGE_H0_KEY])
    torch.testing.assert_close(ctx_again.edge_prior, ctx.edge_prior)


def test_overlap_huckel_prior_is_residualized_against_existing_h0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    idp, data, ref = _basis_onsite_case(device, dtype)
    edge_overlap = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    data[_keys.EDGE_OVERLAP_KEY] = edge_overlap
    node_base = torch.full_like(ref[_keys.NODE_FEATURES_KEY], 0.25)
    edge_base = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], -0.5)
    data[_keys.NODE_H0_KEY] = node_base
    data[_keys.EDGE_H0_KEY] = edge_base
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "overlap_huckel",
            "detach_interpolated_h0": False,
            "huckel_k": 1.5,
        },
        idp=idp,
        device=device,
        dtype=dtype,
    )

    out, _ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    expected_node = torch.zeros_like(ref[_keys.NODE_FEATURES_KEY])
    expected_node[0, _onsite_diag_indices(idp, "H", "1s", device)] = onsite_energy_database["H"]["1s"]
    expected_node[1, _onsite_diag_indices(idp, "C", "2s", device)] = onsite_energy_database["C"]["2s"]
    expected_node[1, _onsite_diag_indices(idp, "C", "2p", device)] = onsite_energy_database["C"]["2p"]
    h_mean = torch.tensor(onsite_energy_database["H"]["1s"], device=device, dtype=dtype)
    c_mean = torch.tensor(
        (onsite_energy_database["C"]["2s"] + 3.0 * onsite_energy_database["C"]["2p"]) / 4.0,
        device=device,
        dtype=dtype,
    )
    expected_edge = 1.5 * 0.5 * (h_mean + c_mean) * edge_overlap
    expected_edge = expected_edge * flow._prior_mask(data, expected_edge, "edge").to(dtype=dtype)

    torch.testing.assert_close(out[_keys.NODE_H0_KEY], expected_node)
    torch.testing.assert_close(ctx.node_prior, expected_node - node_base)
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], expected_edge)
    torch.testing.assert_close(ctx.edge_prior, expected_edge - edge_base)


def test_overlap_huckel_prior_requires_edge_overlap_by_default():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    idp, data, ref = _basis_onsite_case(device, dtype)
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "overlap_huckel",
            "strict_h0": False,
            "warn_missing_h0": False,
            "detach_interpolated_h0": False,
        },
        idp=idp,
        device=device,
        dtype=dtype,
    )

    with pytest.raises(KeyError, match="edge_overlap"):
        flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))


def test_overlap_huckel_prior_rejects_edge_index_overlap_row_mismatch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    idp, data, ref = _basis_onsite_case(device, dtype)
    # edge_index carries 3 columns while the edge overlap/feature tensor has 2 rows.
    # Under the default strict basis mode this must fail loud instead of being
    # silently padded/truncated/clamped into a plausible-but-wrong prior.
    data[_keys.EDGE_INDEX_KEY] = torch.tensor(
        [[0, 1, 0], [1, 0, 1]], device=device, dtype=torch.long
    )
    data[_keys.EDGE_OVERLAP_KEY] = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "overlap_huckel",
            "strict_h0": False,
            "warn_missing_h0": False,
            "detach_interpolated_h0": False,
        },
        idp=idp,
        device=device,
        dtype=dtype,
    )

    with pytest.raises(ValueError, match="edge_index"):
        flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))


def test_prior_physical_basis_onsite_table_matches_cfm_method():
    # The shared dptb.nnops.prior_physical.basis_onsite_table must reproduce, value
    # for value, the table the HamiltonianCFM method builds (the online prior and
    # the offline EMolFlow preprocess tool now share this one implementation).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    idp, _data, ref = _basis_onsite_case(device, dtype)
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "basis_onsite",
            "strict_h0": False,
            "warn_missing_h0": False,
            "basis_onsite_scale": 0.5,
        },
        idp=idp,
        device=device,
        dtype=dtype,
    )
    like = ref[_keys.NODE_FEATURES_KEY]

    method_table = flow._basis_onsite_table(like)
    shared_table = prior_physical.basis_onsite_table(
        idp,
        device=like.device,
        dtype=like.dtype,
        scale=flow.basis_onsite_scale,
        missing=flow.basis_onsite_missing_value,
    )

    assert method_table is not None
    assert shared_table is not None
    # The non-SOC test idp uses an identity uureal->like projection, so the raw
    # shared table equals the projected table returned by the CFM method.
    torch.testing.assert_close(shared_table, method_table)
    # basis_onsite_scale must be threaded identically through both paths.
    h_diag = _onsite_diag_indices(idp, "H", "1s", device)
    assert method_table[idp.chemical_symbol_to_type["H"], h_diag[0]].item() == pytest.approx(
        0.5 * onsite_energy_database["H"]["1s"]
    )

    # The derived per-type mean must also match (same active-entry averaging).
    method_mean = flow._basis_onsite_type_mean(like)
    shared_mean = prior_physical.basis_onsite_type_mean(
        shared_table, fallback=flow.huckel_edge_energy_fallback
    )
    assert method_mean is not None
    torch.testing.assert_close(shared_mean, method_mean)


def test_prior_physical_huckel_edge_energy_reproduces_prior_edge_energies():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    idp, data, ref = _basis_onsite_case(device, dtype)
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "overlap_huckel",
            "strict_h0": False,
            "warn_missing_h0": False,
        },
        idp=idp,
        device=device,
        dtype=dtype,
    )
    edge_like = ref[_keys.EDGE_FEATURES_KEY]
    type_mean = flow._basis_onsite_type_mean(edge_like)
    assert type_mean is not None

    shared_energy = prior_physical.huckel_edge_energy(
        type_mean,
        data[_keys.EDGE_INDEX_KEY],
        data[AtomicDataDict.ATOM_TYPE_KEY],
        edge_like.shape[0],
        fallback=flow.huckel_edge_energy_fallback,
        strict=flow.huckel_strict_basis,
    )
    method_energy = flow._huckel_edge_energy_like(edge_like, data=data)

    # The shared helper reproduces the CFM method's per-edge energy exactly.
    torch.testing.assert_close(shared_energy, method_energy.reshape(-1))

    # edge_index = [[0, 1], [1, 0]] over atom_types [H, C], so both edges take the
    # H<->C endpoint average 0.5*(mean_H + mean_C).
    h_mean = type_mean[idp.chemical_symbol_to_type["H"]]
    c_mean = type_mean[idp.chemical_symbol_to_type["C"]]
    expected = torch.stack([0.5 * (h_mean + c_mean), 0.5 * (c_mean + h_mean)])
    torch.testing.assert_close(shared_energy, expected)


def test_prior_physical_huckel_edge_energy_strict_rejects_column_mismatch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    type_mean = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
    atom_types = torch.tensor([0, 1], device=device, dtype=torch.long)
    # 3 edge_index columns but only 2 edge rows: strict mode must fail loud rather
    # than pad/truncate (the strict-basis behavior guarded by the CFM prior).
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 1]], device=device, dtype=torch.long)

    with pytest.raises(ValueError, match="edge_index"):
        prior_physical.huckel_edge_energy(
            type_mean, edge_index, atom_types, 2, fallback=0.0, strict=True
        )

    # The same mismatch is tolerated (pad/truncate) when strict=False.
    out = prior_physical.huckel_edge_energy(
        type_mean, edge_index, atom_types, 2, fallback=0.0, strict=False
    )
    assert out.shape == (2,)


def test_dftb_prior_uses_external_absolute_hamiltonian_keys():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    dftb_node = data[_keys.NODE_H0_KEY] + 3.0
    dftb_edge = data[_keys.EDGE_H0_KEY] - 2.0
    data["dftb_node_h0"] = dftb_node
    data["dftb_edge_h0"] = dftb_edge
    flow = _flow("dftb", device=device, dtype=dtype)

    out, _ref_out, ctx = flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    torch.testing.assert_close(out[_keys.NODE_H0_KEY], dftb_node)
    torch.testing.assert_close(out[_keys.EDGE_H0_KEY], dftb_edge)
    torch.testing.assert_close(ctx.node_prior, dftb_node - data[_keys.NODE_H0_KEY])
    torch.testing.assert_close(ctx.edge_prior, dftb_edge - data[_keys.EDGE_H0_KEY])


def test_external_prior_rejects_complex_features_by_default():
    device = torch.device("cpu")
    flow = _flow("external", device=device, dtype=torch.float32)
    source = torch.tensor([[1.0 + 2.0j]], dtype=torch.complex64)

    with pytest.raises(TypeError, match="complex"):
        flow._coerce_prior_source(
            source,
            torch.zeros((1, 1), dtype=torch.float32),
            key="node_physical_h0",
            label="node",
        )


def test_external_prior_real_projection_requires_explicit_ablation(caplog):
    device = torch.device("cpu")
    flow = _flow(
        "external",
        device=device,
        dtype=torch.float32,
        allow_complex_prior_real_projection=True,
    )
    source = torch.tensor([[1.0 + 2.0j]], dtype=torch.complex64)

    projected = flow._coerce_prior_source(
        source,
        torch.zeros((1, 1), dtype=torch.float32),
        key="node_physical_h0",
        label="node",
    )

    torch.testing.assert_close(projected, torch.tensor([[1.0]]))
    assert "discarding imaginary channels" in caplog.text


def test_external_candidate_keys_are_the_short_documented_list():
    # Behavior change: the external family consults only the explicit key plus the
    # short documented per-prefix set {prefix}_{label}_h0, {label}_{prefix}_h0,
    # {prefix}_{label}_features -- the ~10 other legacy permutations are dropped.
    device = torch.device("cpu")
    dtype = torch.float32
    flow = _flow("dftb", device=device, dtype=dtype, prior_node_key="my_node_prior")
    fam = flow.prior_family

    expected = ["my_node_prior"]
    for prefix in fam._prefixes():
        expected += [f"{prefix}_node_h0", f"node_{prefix}_h0", f"{prefix}_node_features"]
    assert list(fam.candidate_keys("node")) == expected

    edge_keys = set(fam.candidate_keys("edge"))
    for prefix in fam._prefixes():
        assert f"{prefix}_edge_h0" in edge_keys
        assert f"edge_{prefix}_h0" in edge_keys
        assert f"{prefix}_edge_features" in edge_keys
    # Dropped legacy permutations must no longer appear.
    for dropped in (
        "edge_h0_dftb",
        "dftb_edge_hamiltonian",
        "edge_dftb_hamiltonian",
        "dftb_hopping",
        "hopping_dftb",
    ):
        assert dropped not in edge_keys


def test_dftb_external_prior_fails_closed_without_matching_keys():
    # Named external priors require a matching dataset field.
    device = torch.device("cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    # A dropped legacy permutation is present but is no longer consulted.
    data["node_h0_dftb"] = data[_keys.NODE_H0_KEY] + 1.0
    flow = _flow("dftb", device=device, dtype=dtype)

    with pytest.raises(KeyError, match="Tried keys"):
        flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))

    # external_prior_strict=False still degrades the whole external family to a
    # zero absolute prior (H_t at t=0 equals the zero Hamiltonian).
    flow_lax = _flow("dftb", device=device, dtype=dtype, external_prior_strict=False)
    out, _ref_out, _ctx = flow_lax.prepare_batch(
        data, ref, t=torch.zeros(2, device=device, dtype=dtype)
    )
    torch.testing.assert_close(
        out[_keys.NODE_H0_KEY], torch.zeros_like(out[_keys.NODE_H0_KEY])
    )
    torch.testing.assert_close(
        out[_keys.EDGE_H0_KEY], torch.zeros_like(out[_keys.EDGE_H0_KEY])
    )


def test_named_external_prior_alias_ignores_physical_zero_fallback():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = _flow("dftb", device=device, dtype=dtype, physical_prior_fallback="zero")

    with pytest.raises(KeyError, match="flow_options.prior='dftb'"):
        flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))


def test_dftbsk_prior_alias_is_retired():
    # Rejected during flow-option canonicalization (which runs first inside
    # HamiltonianCFM.__init__) so the message does not still list 'dftbsk'
    # among the supported priors, as flow.py's own message does.
    with pytest.raises(ValueError, match="removed together with the SK model route"):
        _flow("dftbsk", device=torch.device("cpu"), dtype=torch.float32)


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


def test_flow_options_argcheck_accepts_physical_prior_config_keys():
    schema = flow_options()

    value = schema.normalize_value(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "external",
            "prior_node_key": "dftb_node_h0",
            "prior_edge_key": "dftb_edge_h0",
            "prior_key_prefixes": ["dftb", "xtb"],
            "allow_complex_prior_real_projection": True,
            "physical_prior_fallback": "basis_onsite",
            "basis_onsite_scale": 0.5,
            "physical_prior_jitter_sigma": 0.01,
            "physical_prior_jitter_reference_scale": True,
            "physical_prior_jitter_edge_decay": 2.0,
        }
    )
    schema.check_value(value, strict=True)

    assert value["prior"] == "external"
    assert value["prior_node_key"] == "dftb_node_h0"
    assert value["prior_key_prefixes"] == ["dftb", "xtb"]
    assert value["allow_complex_prior_real_projection"] is True
    assert value["basis_onsite_scale"] == pytest.approx(0.5)
    assert value["physical_prior_jitter_edge_decay"] == pytest.approx(2.0)


def test_flow_options_argcheck_accepts_overlap_huckel_config_keys():
    schema = flow_options()

    value = schema.normalize_value(
        {
            "enabled": True,
            "mode": "residual",
            "prior": "overlap_huckel",
            "huckel_k": 1.75,
            "huckel_node_overlap_key": "node_overlap",
            "huckel_edge_overlap_key": "edge_overlap",
            "huckel_strict_overlap": True,
            "huckel_edge_energy_fallback": -2.0,
            "huckel_edge_length_decay": 3.0,
        }
    )
    schema.check_value(value, strict=True)

    assert value["prior"] == "overlap_huckel"
    assert value["huckel_k"] == pytest.approx(1.75)
    assert value["huckel_edge_overlap_key"] == "edge_overlap"
    assert value["huckel_strict_overlap"] is True
    assert value["huckel_edge_length_decay"] == pytest.approx(3.0)


def test_haar_dm_prior_selects_precomputed_candidate():
    device = torch.device("cpu")
    dtype = torch.float64
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "full",
            "prior": "haar_dm",
            "haar_candidate_index": 1,
        },
        idp=_FakeIDP(device=device),
        device=device,
        dtype=dtype,
    )
    residual = torch.zeros(2, 4, dtype=dtype)
    node_candidates = torch.stack(
        [torch.full_like(residual, 3.0), torch.full_like(residual, 7.0)],
        dim=1,
    )
    data = {_keys.HAAR_NODE_FEATURES_KEY: node_candidates}

    prior = flow._prior_like(residual, 1.0, data=data, label="node")

    torch.testing.assert_close(prior, torch.full_like(residual, 7.0))
    assert flow.last_haar_candidate_index == 1


def test_haar_dm_prior_requires_precomputed_fields():
    flow = HamiltonianCFM(
        {"enabled": True, "mode": "full", "prior": "haar_dm"},
        idp=_FakeIDP(device=torch.device("cpu")),
    )

    with pytest.raises(KeyError, match="haar_dm"):
        flow._prior_like(torch.zeros(2, 4), 1.0, data={}, label="node")


def test_haar_dm_prior_uses_same_candidate_for_node_and_edge():
    device = torch.device("cpu")
    dtype = torch.float64
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "full",
            "prior": "haar_dm",
            "detach_interpolated_h0": False,
        },
        idp=_FakeIDP(device=device),
        device=device,
        dtype=dtype,
    )

    node_target = ref[_keys.NODE_FEATURES_KEY]
    edge_target = ref[_keys.EDGE_FEATURES_KEY]
    # K=2 candidate axis on both node and edge. Candidate 0 vs 1 are far apart,
    # and index i of the node tensor is coherent with index i of the edge tensor
    # (both encode the same Haar density matrix in production), so the node and
    # edge priors must always select the same candidate.
    node_candidates = torch.stack(
        [torch.full_like(node_target, 3.0), torch.full_like(node_target, 7.0)],
        dim=1,
    )
    edge_candidates = torch.stack(
        [torch.full_like(edge_target, 30.0), torch.full_like(edge_target, 70.0)],
        dim=1,
    )
    data[_keys.HAAR_NODE_FEATURES_KEY] = node_candidates
    data[_keys.HAAR_EDGE_FEATURES_KEY] = edge_candidates

    t = torch.zeros(2, device=device, dtype=dtype)
    seen = set()
    torch.manual_seed(0)
    for _ in range(40):
        _out, _ref_out, ctx = flow.prepare_batch(data, ref, t=t)
        node_c0 = bool(torch.all(ctx.node_prior == 3.0))
        node_c1 = bool(torch.all(ctx.node_prior == 7.0))
        edge_c0 = bool(torch.all(ctx.edge_prior == 30.0))
        edge_c1 = bool(torch.all(ctx.edge_prior == 70.0))
        # Each prior is exactly one candidate, fully (no cross-candidate mixing).
        assert node_c0 != node_c1
        assert edge_c0 != edge_c1
        node_idx = 0 if node_c0 else 1
        edge_idx = 0 if edge_c0 else 1
        assert node_idx == edge_idx
        seen.add(node_idx)
    # The shared draw is genuinely random, not pinned to a single candidate.
    assert seen == {0, 1}


def test_haar_dm_prior_rejects_mismatched_node_edge_candidate_counts():
    device = torch.device("cpu")
    dtype = torch.float64
    data, ref = _make_batch(device=device, dtype=dtype)
    flow = HamiltonianCFM(
        {"enabled": True, "mode": "full", "prior": "haar_dm"},
        idp=_FakeIDP(device=device),
        device=device,
        dtype=dtype,
    )
    node_target = ref[_keys.NODE_FEATURES_KEY]
    edge_target = ref[_keys.EDGE_FEATURES_KEY]
    # Node exposes K=2 candidates, edge exposes K=3: the coherent-pairing
    # assumption is violated and prepare_batch must refuse to draw an index.
    data[_keys.HAAR_NODE_FEATURES_KEY] = torch.stack(
        [torch.full_like(node_target, float(i)) for i in range(2)], dim=1
    )
    data[_keys.HAAR_EDGE_FEATURES_KEY] = torch.stack(
        [torch.full_like(edge_target, float(i)) for i in range(3)], dim=1
    )

    with pytest.raises(ValueError, match="candidate"):
        flow.prepare_batch(data, ref, t=torch.zeros(2, device=device, dtype=dtype))


def test_flow_options_argcheck_accepts_haar_dm_config_keys():
    schema = flow_options()
    value = schema.normalize_value(
        {
            "enabled": True,
            "prior": "haar_dm",
            "haar_node_key": "haar_node_features",
            "haar_edge_key": "haar_edge_features",
            "haar_candidate_index": 0,
            "haar_dm_strict": True,
        }
    )
    schema.check_value(value, strict=True)

    assert value["prior"] == "haar_dm"
    assert value["haar_node_key"] == "haar_node_features"
    assert value["haar_edge_key"] == "haar_edge_features"
    assert value["haar_candidate_index"] == 0
    assert value["haar_dm_strict"] is True


def test_common_options_argcheck_accepts_nextham_uureal_mask():
    schema = common_options()

    value = schema.normalize_value(
        {
            "basis": {"Si": ["3s", "3p"]},
            "has_soc": True,
            "nextham_uureal_mask": True,
        }
    )
    schema.check_value(value, strict=True)

    assert value["has_soc"] is True
    assert value["nextham_uureal_mask"] is True


def test_cfm_forces_compatible_clean_logging_even_when_config_disables_it():
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "log_compatible_loss": False,
            "log_train_compatible_loss": False,
            "log_validation_compatible_loss": False,
        }
    )

    assert flow.log_train_compatible_loss is True
    assert flow.log_validation_compatible_loss is True
    assert flow.compatible_loss_to_legacy_keys is True


def test_compatible_loss_state_copies_total_loss_to_legacy_prefix():
    state = Trainer._compatible_loss_state(
        _FakeCompatibleLoss(),
        {},
        {},
        prefix="train_compatible",
        legacy_prefix="train",
    )

    assert state["train_compatible_loss"].item() == pytest.approx(0.75)
    assert state["train_loss"].item() == pytest.approx(0.75)
    assert state["train_onsite_loss"].item() == pytest.approx(1.25)
    assert state["train_hopping_loss"].item() == pytest.approx(0.25)


def test_compatible_pack_loss_falls_back_to_weighted_clean_components():
    trainer = object.__new__(MultiTrainer)
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer.endpoint_loss_mode = "reduce"

    pack = torch.zeros(MultiTrainer._PACK_LEN, dtype=torch.float32)
    pack[MultiTrainer._P_ONSITE_WEIGHTED_SUM] = 4.0
    pack[MultiTrainer._P_HOPPING_WEIGHTED_SUM] = 2.0
    pack[MultiTrainer._P_ACTIVE_NODES_SUM] = 2.0
    pack[MultiTrainer._P_ACTIVE_EDGES_SUM] = 4.0

    loss = trainer._compute_compatible_loss_from_pack(pack, _FakeCompatibleLoss())

    assert loss.item() == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# P0 regression: flow prior must not be silently deactivated by a key mismatch
# between train_options.flow_options.{node,edge}_h0_key (what the flow writes)
# and model_options.embedding.h0_{node,edge}_key (what the H0-init reads).
# ---------------------------------------------------------------------------


def _h0_init_consumer_stub(h0_node_key, h0_edge_key):
    """Minimal stand-in for the H0InitLayer consumer.

    The build-time guard keys purely on a submodule that exposes string
    ``h0_node_key`` / ``h0_edge_key`` attributes (which is exactly what
    ``H0InitLayer`` registers), so a bare Module carrying those attributes
    reproduces the model side without building a full embedding.
    """
    layer = torch.nn.Module()
    layer.h0_node_key = h0_node_key
    layer.h0_edge_key = h0_edge_key
    model = torch.nn.Module()
    model.init_layer = layer  # registered as a child -> visible via model.modules()
    return model


def _physical_h0_flow(enabled=True):
    return build_hamiltonian_flow(
        {
            "enabled": enabled,
            "mode": "residual",
            "prior": "zero",
            "node_h0_key": "node_physical_h0",
            "edge_h0_key": "edge_physical_h0",
        }
    )


def test_flow_h0_key_guard_raises_when_embedding_keys_not_repointed():
    # The P0: overlay set flow_options -> node/edge_physical_h0 but left the
    # embedding at the stored-h0 defaults; the interpolated state never reaches
    # the network and every mode collapses to a plain stored-H0 run.
    flow = _physical_h0_flow(enabled=True)
    model = _h0_init_consumer_stub("node_h0", "edge_h0")
    with pytest.raises(ValueError, match="silently deactivated"):
        assert_flow_h0_keys_reach_model(flow, model)


def test_flow_h0_key_guard_message_names_both_offending_sides():
    flow = _physical_h0_flow(enabled=True)
    model = _h0_init_consumer_stub("node_h0", "edge_h0")
    with pytest.raises(ValueError) as excinfo:
        assert_flow_h0_keys_reach_model(flow, model)
    message = str(excinfo.value)
    assert "node_physical_h0" in message and "edge_physical_h0" in message
    assert "h0_node_key" in message and "h0_edge_key" in message


def test_flow_h0_key_guard_passes_when_embedding_keys_are_aligned():
    flow = _physical_h0_flow(enabled=True)
    model = _h0_init_consumer_stub("node_physical_h0", "edge_physical_h0")
    # aligned keys -> the fixed configuration must build without error
    assert assert_flow_h0_keys_reach_model(flow, model) is None


def test_flow_h0_key_guard_is_noop_when_flow_disabled():
    flow = _physical_h0_flow(enabled=False)
    model = _h0_init_consumer_stub("node_h0", "edge_h0")
    assert assert_flow_h0_keys_reach_model(flow, model) is None


def test_flow_h0_key_guard_is_noop_without_h0_init_consumer():
    flow = _physical_h0_flow(enabled=True)
    model = torch.nn.Linear(2, 2)  # no module exposes h0_node_key/h0_edge_key
    assert assert_flow_h0_keys_reach_model(flow, model) is None


def test_flow_h0_key_guard_accepts_default_stored_h0_config():
    # Legacy stored-h0 flow: flow and embedding share the node_h0/edge_h0
    # defaults, so the guard must stay silent (no false positive).
    flow = build_hamiltonian_flow({"enabled": True, "mode": "residual", "prior": "zero"})
    model = _h0_init_consumer_stub("node_h0", "edge_h0")
    assert assert_flow_h0_keys_reach_model(flow, model) is None

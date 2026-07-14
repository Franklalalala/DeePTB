from pathlib import Path
import importlib
from types import SimpleNamespace

import pytest
import torch

from dptb.nnops.flow import (
    HamiltonianCFM,
    HamiltonianPixelMeanFlow,
    assert_model_in_loss_endpoint_metric_space,
    build_hamiltonian_flow,
)
from dptb.data import AtomicDataDict
from dptb.nnops.loss import HamilLossAbs
from dptb.nnops import trainer as trainer_module
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.nnops.trainer import Trainer
from dptb.plugins.monitor import TensorBoardMonitor, Validationer

train_entrypoint = importlib.import_module("dptb.entrypoints.train")


def _two_graph_batch():
    return {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        "node_h0": torch.zeros(3, 1),
        "edge_h0": torch.zeros(2, 1),
        "node_features": torch.zeros(3, 1),
        "edge_features": torch.zeros(2, 1),
    }


def _two_graph_ref():
    return {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        "node_features": torch.full((3, 1), 2.0),
        "edge_features": torch.full((2, 1), 4.0),
    }


def test_prepare_batch_samples_and_expands_time_per_graph():
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "prior": "zero",
            "omit_time_scaling": True,
            "strict_h0": True,
        }
    )
    flow._sample_t = lambda *, num_graphs, device, dtype: torch.tensor(
        [0.0, 0.5], device=device, dtype=dtype
    )

    data, ref, ctx = flow.prepare_batch(_two_graph_batch(), _two_graph_ref())

    assert ctx.t.shape == (2,)
    assert torch.equal(ctx.node_t, torch.tensor([0.0, 0.0, 0.5]))
    assert torch.equal(ctx.edge_t, torch.tensor([0.0, 0.5]))
    assert torch.equal(data["flow_time"], torch.tensor([0.0, 0.5]))
    assert torch.equal(ref["flow_time"], torch.tensor([0.0, 0.5]))
    assert torch.equal(data["node_h0"].flatten(), torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(data["edge_h0"].flatten(), torch.tensor([0.0, 2.0]))


def test_residual_flow_fails_fast_when_h0_is_missing():
    flow = HamiltonianCFM({"enabled": True, "mode": "residual", "strict_h0": True})
    data = _two_graph_batch()
    data.pop("node_h0")

    with pytest.raises(KeyError, match="node_h0"):
        flow.prepare_batch(data, _two_graph_ref())


def test_global_element_reduction_does_not_equal_weight_node_and_edge_components():
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "omit_time_scaling": True,
            "component_reduction": "global_elements",
        }
    )
    data = {
        "batch": torch.tensor([0], dtype=torch.long),
        "edge_index": torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long),
        "node_h0": torch.zeros(1, 1),
        "edge_h0": torch.zeros(3, 1),
        "node_features": torch.zeros(1, 1),
        "edge_features": torch.zeros(3, 1),
    }
    ref = {
        "batch": data["batch"],
        "edge_index": data["edge_index"],
        "node_features": torch.zeros(1, 1),
        "edge_features": torch.zeros(3, 1),
    }
    _, ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1))
    pred = {
        "node_features": torch.ones(1, 1),
        "edge_features": torch.full((3, 1), 3.0),
    }

    loss, state = flow.loss(pred, ref, ctx)

    assert loss.item() == pytest.approx(7.0)
    assert state["train_flow_onsite_loss"].item() == pytest.approx(1.0)
    assert state["train_flow_hopping_loss"].item() == pytest.approx(9.0)
    assert state["train_onsite_loss"].item() == pytest.approx(1.0)
    assert state["train_hopping_loss"].item() == pytest.approx(9.0)


def test_cfm_writes_default_legacy_train_tags_and_router_stats():
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "omit_time_scaling": True,
        }
    )
    data, ref, ctx = flow.prepare_batch(_two_graph_batch(), _two_graph_ref(), t=torch.zeros(2))
    pred = {
        "batch": data["batch"],
        "edge_index": data["edge_index"],
        "node_features": ref["node_features"] + 1.0,
        "edge_features": ref["edge_features"] + 3.0,
        "mean_max_prob": torch.tensor(0.75),
        "expert_load_cv": torch.tensor(0.25),
    }

    _, state = flow.loss(pred, ref, ctx)

    assert state["train_flow_onsite_loss"].item() == pytest.approx(1.0)
    assert state["train_flow_hopping_loss"].item() == pytest.approx(9.0)
    assert state["train_onsite_loss"].item() == pytest.approx(1.0)
    assert state["train_hopping_loss"].item() == pytest.approx(9.0)
    assert state["mean_max_prob"].item() == pytest.approx(0.75)
    assert state["expert_load_cv"].item() == pytest.approx(0.25)


def test_single_trainer_effective_expert_lr_state_uses_global_optimizer_lr():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.0125)
    state = {}

    Trainer._add_effective_expert_lr_state(state, optimizer=optimizer, num_experts=2)

    assert state["expert_0_lr"] == pytest.approx(0.0125)
    assert state["expert_1_lr"] == pytest.approx(0.0125)


def test_single_train_entrypoint_passes_train_options_to_build_model():
    text = Path(train_entrypoint.__file__).read_text(encoding="utf-8")
    build_call = text[text.index("model = build_model("): text.index("trainer = Trainer(")]

    assert 'train_options=jdata["train_options"]' in build_call


class _ComponentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.called_with_grad_enabled = None
        self.last_onsite_loss = torch.tensor(123.0)
        self.last_hopping_loss = torch.tensor(456.0)

    def forward(self, pred, ref):
        self.called_with_grad_enabled = torch.is_grad_enabled()
        onsite = (pred["node_features"] - ref["node_features"]).abs().mean()
        hopping = (pred["edge_features"] - ref["edge_features"]).abs().mean()
        self.last_onsite_loss = onsite.detach()
        self.last_hopping_loss = hopping.detach()
        return 0.5 * (onsite + hopping)


class _StatsCompatibleLoss(torch.nn.Module):
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


class _DistinctEndpointLoss(torch.nn.Module):
    endpoint_metric_space = "block"

    def __init__(self):
        super().__init__()
        self.last_endpoint_loss = torch.tensor(99.0)
        self.last_endpoint_metric_space = "saved"
        self.last_onsite_loss = torch.tensor(88.0)
        self.last_hopping_loss = torch.tensor(77.0)

    def forward(self, pred, ref):
        self.last_endpoint_loss = torch.tensor(2.0)
        self.last_endpoint_metric_space = "block"
        self.last_onsite_loss = torch.tensor(1.0)
        self.last_hopping_loss = torch.tensor(3.0)
        return torch.tensor(20.0)


class _BlockEndpointFallbackLoss(_StatsCompatibleLoss):
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


def test_meanflow_block_endpoint_fails_fast_for_default_rme_target_keys():
    flow = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow"}
    )

    with pytest.raises(
        ValueError,
        match=r"endpoint metric-space mismatch.*node_target_key='node_features'.*will not perform",
    ):
        assert_model_in_loss_endpoint_metric_space(
            flow,
            _BlockEndpointFallbackLoss(),
        )


def test_meanflow_block_endpoint_accepts_explicit_block_target_contract():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "node_target_key": "node_full_hamil_blocks",
            "edge_target_key": "edge_full_hamil_blocks",
        }
    )

    assert_model_in_loss_endpoint_metric_space(
        flow,
        _BlockEndpointFallbackLoss(),
    )


def test_non_model_in_loss_cfm_does_not_use_meanflow_metric_space_guard():
    flow = HamiltonianCFM({"enabled": True})

    assert_model_in_loss_endpoint_metric_space(
        flow,
        _BlockEndpointFallbackLoss(),
    )


def test_flow_endpoint_stats_reject_mixed_node_edge_metric_spaces():
    state = {}
    HamiltonianCFM._merge_compatible_clean_stats(
        state,
        {"onsite_l1_sum": torch.tensor(1.0), "metric_space": "rme"},
    )

    with pytest.raises(ValueError, match="onsite and hopping targets"):
        HamiltonianCFM._merge_compatible_clean_stats(
            state,
            {"hopping_l1_sum": torch.tensor(1.0), "metric_space": "block"},
        )


@pytest.mark.parametrize("trainer_cls", [Trainer, MultiTrainer])
def test_single_and_multi_trainer_apply_meanflow_endpoint_contract(trainer_cls):
    trainer = object.__new__(trainer_cls)
    trainer.flow_cfm = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow"}
    )
    trainer.train_lossfunc = _BlockEndpointFallbackLoss()
    trainer.use_reference = False

    with pytest.raises(ValueError, match="trainer initialization"):
        trainer._assert_model_in_loss_endpoint_contract()


def test_meanflow_endpoint_contract_checks_validation_criterion_at_init():
    trainer = object.__new__(Trainer)
    trainer.flow_cfm = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow"}
    )
    trainer.train_lossfunc = SimpleNamespace(endpoint_metric_space="rme")
    trainer.validation_lossfunc = _BlockEndpointFallbackLoss()
    trainer.use_validation = True
    trainer.use_reference = False

    with pytest.raises(ValueError, match=r"metric-space mismatch.*validation criterion"):
        trainer._assert_model_in_loss_endpoint_contract()


def _compatible_clean_stats():
    return {
        "_compatible_clean_stats": {
            "onsite_l1_sum": torch.tensor(4.0),
            "onsite_mse_sum": torch.tensor(10.0),
            "onsite_count": torch.tensor(2.0),
            "hopping_l1_sum": torch.tensor(3.0),
            "hopping_mse_sum": torch.tensor(9.0),
            "hopping_count": torch.tensor(3.0),
        },
        "mean_max_prob": torch.tensor(0.75),
        "expert_load_cv": torch.tensor(0.25),
    }


class _LossIDP:
    def __init__(self):
        self.mask_to_nrme = torch.tensor(
            [[True, True, False], [True, False, True]]
        )
        self.mask_to_erme = torch.tensor(
            [[True, False, True], [False, True, True]]
        )


def _masked_stats(diff, mask):
    mask_f = mask.to(dtype=diff.dtype)
    return {
        "l1_sum": (diff.abs() * mask_f).sum(),
        "mse_sum": (diff.square() * mask_f).sum(),
        "count": mask_f.sum().to(dtype=diff.dtype),
    }


def _scalar(value):
    if torch.is_tensor(value):
        value = value.detach()
        if value.ndim > 0:
            value = value.mean()
        return float(value.item())
    return float(value)


@pytest.mark.parametrize(
    "loss_kwargs",
    [
        {"element_average": False, "z_loss_coef": 0.2},
        {"element_average": True, "z_loss_coef": 0.2},
        {"onsite_boost": True, "onsite_boost_steps": 100, "onsite_boost_max": 3.0},
    ],
)
def test_hamil_abs_compatible_stats_match_forward_semantics(loss_kwargs):
    idp = _LossIDP()
    lossfunc = HamilLossAbs(idp=idp, dtype=torch.float64, **loss_kwargs)
    pred = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1]),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([0, 1]),
        AtomicDataDict.NODE_FEATURES_KEY: torch.tensor(
            [[1.0, -2.0, 9.0], [3.0, 8.0, -4.0]], dtype=torch.float64
        ),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.tensor(
            [[5.0, 7.0, -6.0], [11.0, -8.0, 2.0]], dtype=torch.float64
        ),
        "mean_max_prob": torch.tensor(0.25, dtype=torch.float64),
        "global_step": 25,
    }
    ref = {
        AtomicDataDict.NODE_FEATURES_KEY: torch.tensor(
            [[0.5, -1.0, 0.0], [1.0, 0.0, -1.0]], dtype=torch.float64
        ),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.tensor(
            [[3.0, 0.0, -2.0], [0.0, -5.0, 3.0]], dtype=torch.float64
        ),
    }

    forward_total = lossfunc(pred, ref)
    forward_onsite = lossfunc.last_onsite_loss.detach().clone()
    forward_hopping = lossfunc.last_hopping_loss.detach().clone()
    node_mask = idp.mask_to_nrme[pred[AtomicDataDict.ATOM_TYPE_KEY].flatten()]
    edge_mask = idp.mask_to_erme[pred[AtomicDataDict.EDGE_TYPE_KEY].flatten()]
    node_stats = _masked_stats(
        pred[AtomicDataDict.NODE_FEATURES_KEY] - ref[AtomicDataDict.NODE_FEATURES_KEY],
        node_mask,
    )
    edge_stats = _masked_stats(
        pred[AtomicDataDict.EDGE_FEATURES_KEY] - ref[AtomicDataDict.EDGE_FEATURES_KEY],
        edge_mask,
    )

    stats_total, stats_onsite, stats_hopping = lossfunc.compatible_loss_from_stats(
        onsite_l1_sum=node_stats["l1_sum"],
        onsite_mse_sum=node_stats["mse_sum"],
        onsite_count=node_stats["count"],
        hopping_l1_sum=edge_stats["l1_sum"],
        hopping_mse_sum=edge_stats["mse_sum"],
        hopping_count=edge_stats["count"],
        z_loss=pred["mean_max_prob"],
        global_step=pred["global_step"],
    )

    torch.testing.assert_close(stats_total, forward_total.detach())
    torch.testing.assert_close(stats_onsite, forward_onsite)
    torch.testing.assert_close(stats_hopping, forward_hopping)


def _pred_ref():
    pred = {
        "node_features": torch.tensor([[1.0], [3.0]], requires_grad=True),
        "edge_features": torch.tensor([[2.0]], requires_grad=True),
    }
    ref = {
        "node_features": torch.zeros(2, 1),
        "edge_features": torch.zeros(1, 1),
    }
    return pred, ref


def test_flow_compatible_loss_state_uses_no_grad_and_restores_side_effects():
    lossfunc = _ComponentLoss()
    pred, ref = _pred_ref()

    state = Trainer._compatible_loss_state(
        lossfunc,
        pred,
        ref,
        prefix="train_compatible",
        legacy_prefix=None,
    )

    assert lossfunc.called_with_grad_enabled is False
    assert state["train_compatible_loss"].requires_grad is False
    assert state["train_compatible_onsite_loss"].item() == pytest.approx(2.0)
    assert state["train_compatible_hopping_loss"].item() == pytest.approx(2.0)
    assert "train_onsite_loss" not in state
    assert "train_hopping_loss" not in state

    assert lossfunc.last_onsite_loss.item() == pytest.approx(123.0)
    assert lossfunc.last_hopping_loss.item() == pytest.approx(456.0)


def test_compatible_forward_uses_endpoint_total_not_optimization_total():
    lossfunc = _DistinctEndpointLoss()

    state = Trainer._compatible_loss_state(
        lossfunc,
        {},
        {},
        prefix="validation_compatible_euler_1",
        legacy_prefix="validation",
    )

    assert state["validation_loss"].item() == pytest.approx(2.0)
    assert state["validation_onsite_loss"].item() == pytest.approx(1.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(3.0)
    assert state["validation_compatible_euler_1_loss_opt"].item() == pytest.approx(20.0)
    assert lossfunc.last_endpoint_loss.item() == pytest.approx(99.0)
    assert lossfunc.last_endpoint_metric_space == "saved"


def test_flow_stats_reject_cross_representation_endpoint_reduction():
    from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss

    lossfunc = HamilBlockwiseNexTHamLoss(basis={"H": "1s"})
    flow_state = _compatible_clean_stats()
    flow_state["_compatible_clean_stats"]["metric_space"] = "rme"

    assert Trainer._compatible_loss_state_from_flow_stats(
        lossfunc,
        flow_state,
        source_prefix="train",
        prefix="train_compatible",
        legacy_prefix="train",
    ) is None

    flow_state["_compatible_clean_stats"]["metric_space"] = "block"
    state = Trainer._compatible_loss_state_from_flow_stats(
        lossfunc,
        flow_state,
        source_prefix="train",
        prefix="train_compatible",
        legacy_prefix="train",
    )
    assert state["train_loss"].item() == pytest.approx(
        0.5 * (state["train_onsite_loss"] + state["train_hopping_loss"]).item()
    )


def test_flow_stats_fast_path_preserves_compatible_and_legacy_semantics():
    lossfunc = _StatsCompatibleLoss()
    state = Trainer._compatible_loss_state_from_flow_stats(
        lossfunc,
        _compatible_clean_stats(),
        source_prefix="train",
        prefix="train_compatible",
        legacy_prefix="train",
        global_step=17,
    )

    onsite = 0.5 * (2.0 + (10.0 / 2.0) ** 0.5)
    hopping = 0.5 * (1.0 + 3.0 ** 0.5)
    total = 0.5 * (onsite + hopping)
    assert lossfunc.forward_calls == 0
    assert lossfunc.stats_calls == 1
    assert state["train_compatible_loss"].item() == pytest.approx(total)
    assert state["train_loss"].item() == pytest.approx(total)
    assert state["train_onsite_loss"].item() == pytest.approx(onsite)
    assert state["train_hopping_loss"].item() == pytest.approx(hopping)


def test_flow_compatible_loss_state_explicit_legacy_mapping():
    lossfunc = _ComponentLoss()
    pred, ref = _pred_ref()

    state = Trainer._compatible_loss_state(
        lossfunc,
        pred,
        ref,
        prefix="train_compatible",
        legacy_prefix="train",
    )

    assert state["train_onsite_loss"].item() == pytest.approx(2.0)
    assert state["train_hopping_loss"].item() == pytest.approx(2.0)
    assert lossfunc.last_onsite_loss.item() == pytest.approx(123.0)
    assert lossfunc.last_hopping_loss.item() == pytest.approx(456.0)


def test_flow_compatible_loss_state_maps_validation_clean_legacy_loss():
    lossfunc = _ComponentLoss()
    pred, ref = _pred_ref()

    state = Trainer._compatible_loss_state(
        lossfunc,
        pred,
        ref,
        prefix="validation_compatible_euler_1",
        legacy_prefix="validation",
    )

    assert state["validation_onsite_loss"].item() == pytest.approx(2.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(2.0)
    assert state["validation_loss"].item() == pytest.approx(2.0)


class _ConstantEndpoint(torch.nn.Module):
    def forward(self, data):
        data = data.copy()
        data["node_features"] = torch.full_like(data["node_h0"], 2.0)
        data["edge_features"] = torch.full_like(data["edge_h0"], 4.0)
        return data


class _NonUniformEndpoint(torch.nn.Module):
    def forward(self, data):
        data = data.copy()
        data["node_features"] = torch.tensor(
            [[1.0], [2.0], [0.0]],
            device=data["node_h0"].device,
            dtype=data["node_h0"].dtype,
        )
        data["edge_features"] = torch.tensor(
            [[1.0], [4.0]],
            device=data["edge_h0"].device,
            dtype=data["edge_h0"].dtype,
        )
        return data


class _GradModeRecordingEndpoint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_modes = []

    def forward(self, data):
        self.grad_modes.append(torch.is_grad_enabled())
        data = data.copy()
        data["node_features"] = data["node_h0"].clone()
        data["edge_features"] = data["edge_h0"].clone()
        return data


@pytest.mark.parametrize("num_steps", [1, 3])
def test_euler_sampler_reaches_constant_predicted_endpoint(num_steps):
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "prior": "zero",
            "omit_time_scaling": True,
            "strict_h0": True,
        }
    )

    sampled = flow.sample(_ConstantEndpoint(), _two_graph_batch(), num_steps=num_steps)

    assert torch.allclose(sampled["node_features"], torch.full((3, 1), 2.0))
    assert torch.allclose(sampled["edge_features"], torch.full((2, 1), 4.0))
    assert torch.equal(sampled["flow_time"], torch.ones(2))


def test_build_hamiltonian_flow_selects_pixel_meanflow_objective():
    flow = build_hamiltonian_flow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
        }
    )

    assert isinstance(flow, HamiltonianPixelMeanFlow)
    assert flow.model_in_loss is True


def test_pixel_meanflow_conservative_defaults_to_paper_boundary_tangent():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"profile": "conservative"},
        }
    )

    assert flow.meanflow_profile == "conservative"
    assert flow.meanflow_jvp_tangent == "boundary"
    assert flow.meanflow_norm_p == pytest.approx(0.0)
    assert flow.meanflow_aux_boundary_v_weight == pytest.approx(0.0)


def test_pixel_meanflow_du_dt_backend_accepts_finite_difference_and_jvp():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"du_dt_backend": "finite_difference"},
        }
    )

    assert flow.meanflow_du_dt_backend == "finite_difference"

    jvp_flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"du_dt_backend": "jvp"},
        }
    )
    assert jvp_flow.meanflow_du_dt_backend == "jvp"

    # default stays the opt-in-free finite difference
    default_flow = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow"}
    )
    assert default_flow.meanflow_du_dt_backend == "finite_difference"

    with pytest.raises(ValueError, match="du_dt_backend"):
        HamiltonianPixelMeanFlow(
            {
                "enabled": True,
                "objective": "pixel_meanflow",
                "meanflow": {"du_dt_backend": "spectral"},
            }
        )


def test_pixel_meanflow_semigroup_objective_is_configurable():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"objective": "semigroup"},
        }
    )

    assert flow.meanflow_objective == "semigroup"
    assert flow.meanflow_semigroup_weight == pytest.approx(1.0)
    assert flow.meanflow_semigroup_endpoint_weight == pytest.approx(1.0)

    hybrid_flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"objective": "hybrid", "semigroup_weight": 0.25},
        }
    )
    assert hybrid_flow.meanflow_objective == "hybrid"
    assert hybrid_flow.meanflow_semigroup_weight == pytest.approx(0.25)

    with pytest.raises(ValueError, match="meanflow.objective"):
        HamiltonianPixelMeanFlow(
            {
                "enabled": True,
                "objective": "pixel_meanflow",
                "meanflow": {"objective": "bad"},
            }
        )


def test_pixel_meanflow_aggressive_profile_sets_opt_in_knobs():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"profile": "aggressive"},
        }
    )

    assert flow.meanflow_profile == "aggressive"
    assert flow.meanflow_jvp_tangent == "boundary"
    assert flow.meanflow_norm_p == pytest.approx(1.0)
    assert flow.meanflow_aux_boundary_v_weight > 0.0


def test_flow_apply_to_reference_defaults_false_and_can_opt_in():
    default_flow = HamiltonianPixelMeanFlow({"enabled": True, "objective": "pixel_meanflow"})
    opt_in_flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "apply_to_reference": True,
        }
    )

    assert default_flow.apply_to_reference is False
    assert opt_in_flow.apply_to_reference is True


class _ModelInLossFlow:
    enabled = True
    model_in_loss = True
    apply_to_reference = False
    log_train_compatible_loss = True
    compatible_loss_to_legacy_keys = True

    def loss_with_model(self, model, batch, batch_for_loss):
        assert model is _UNUSED_MODEL
        assert batch is not batch_for_loss
        return torch.tensor(7.0, requires_grad=True), {"train_flow_loss": torch.tensor(7.0)}


class _ModelInLossFlowWithStats(_ModelInLossFlow):
    def loss_with_model(self, model, batch, batch_for_loss):
        loss, state = super().loss_with_model(model, batch, batch_for_loss)
        state.update(_compatible_clean_stats())
        return loss, state


class _FakeBatch:
    __slices__ = {}
    __cumsum__ = {}
    __cat_dims__ = {}
    __num_nodes_list__ = []
    __data_class__ = object

    def to(self, device):
        return self


class _ValidationIdentityModel:
    def eval(self):
        return None

    def __call__(self, batch):
        out = batch.copy()
        out["node_features"] = batch["node_features"].clone()
        out["edge_features"] = batch["edge_features"].clone()
        return out


class _ValidationEndpointModel:
    def eval(self):
        return None

    def __call__(self, batch):
        out = batch.copy()
        out["node_features"] = torch.ones_like(batch["node_features"])
        out["edge_features"] = torch.full_like(batch["edge_features"], 2.0)
        return out


class _ValidationPreparedFlow:
    enabled = True
    model_in_loss = False
    log_validation_compatible_loss = True
    compatible_loss_to_legacy_keys = False
    validation_ode_steps = (1,)

    def _num_graphs(self, batch):
        return 1

    def prepare_batch(self, batch, ref_batch, t=None):
        return batch.copy(), ref_batch.copy(), object()

    def loss(self, pred, ref, ctx):
        return torch.tensor(3.0), {"validation_compatible_euler_1_loss": torch.tensor(9.0)}

    def sample(self, model, batch, *, num_steps):
        return model(batch)


class _EulerOnlyValidationFlow:
    enabled = True
    model_in_loss = False
    log_validation_compatible_loss = True
    compatible_loss_to_legacy_keys = True
    validation_ode_steps = (1,)
    log_validation_random_t_loss = False
    log_validation_t0_loss = False
    log_validation_flow_euler_loss = False

    def __init__(self):
        self.sample_calls = 0

    def _num_graphs(self, batch):
        return 1

    def prepare_batch(self, *args, **kwargs):
        raise AssertionError("Euler-only validation should not prepare random-t/t0 batches")

    def loss(self, *args, **kwargs):
        raise AssertionError("Euler-only validation should not compute flow validation loss")

    def sample(self, model, batch, *, num_steps):
        assert num_steps == 1
        self.sample_calls += 1
        return model(batch)


class _MultiEulerValidationFlow:
    enabled = True
    model_in_loss = False
    log_validation_compatible_loss = True
    compatible_loss_to_legacy_keys = True
    validation_ode_steps = (1, 3)

    def __init__(self):
        self.sample_calls = []

    def prepare_batch(self, *args, **kwargs):
        raise AssertionError("MultiTrainer compatible validation should not use random-t batches")

    def loss(self, *args, **kwargs):
        raise AssertionError("MultiTrainer compatible validation should not use flow loss")

    def sample(self, model, batch, *, num_steps):
        assert "expert_node_mask" in batch
        assert "expert_edge_mask" in batch
        self.sample_calls.append(int(num_steps))
        out = batch.copy()
        out["node_features"] = torch.full_like(batch["node_features"], float(num_steps))
        out["edge_features"] = torch.full_like(batch["edge_features"], float(2 * num_steps))
        return out


class _StatsForwardLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_onsite_loss = None
        self.last_hopping_loss = None
        self.last_z_loss = None
        self.expert_load_cv = None

    def forward(self, pred, ref):
        node_mask = pred.get(
            "expert_node_mask",
            torch.ones(pred["node_features"].shape[0], dtype=torch.bool),
        ).to(dtype=pred["node_features"].dtype).unsqueeze(-1)
        edge_mask = pred.get(
            "expert_edge_mask",
            torch.ones(pred["edge_features"].shape[0], dtype=torch.bool),
        ).to(dtype=pred["edge_features"].dtype).unsqueeze(-1)

        node_diff = (pred["node_features"] - ref["node_features"]) * node_mask
        edge_diff = (pred["edge_features"] - ref["edge_features"]) * edge_mask
        self.last_onsite_l1_sum = node_diff.abs().sum().detach()
        self.last_onsite_mse_sum = node_diff.square().sum().detach()
        self.last_onsite_count = node_mask.sum().detach()
        self.last_hopping_l1_sum = edge_diff.abs().sum().detach()
        self.last_hopping_mse_sum = edge_diff.square().sum().detach()
        self.last_hopping_count = edge_mask.sum().detach()
        onsite = 0.5 * (
            self.last_onsite_l1_sum / self.last_onsite_count.clamp_min(1.0)
            + torch.sqrt(
                self.last_onsite_mse_sum / self.last_onsite_count.clamp_min(1.0)
                + 1e-12
            )
        )
        hopping = 0.5 * (
            self.last_hopping_l1_sum / self.last_hopping_count.clamp_min(1.0)
            + torch.sqrt(
                self.last_hopping_mse_sum / self.last_hopping_count.clamp_min(1.0)
                + 1e-12
            )
        )
        self.last_onsite_loss = onsite.detach()
        self.last_hopping_loss = hopping.detach()
        return 0.5 * (onsite + hopping)


def test_validation_compatible_loss_forces_legacy_keys(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.model = _ValidationIdentityModel()
    trainer.flow_cfm = _ValidationPreparedFlow()
    trainer.validation_loader = [_FakeBatch()]
    trainer.validation_lossfunc = _ComponentLoss()
    trainer.iter = 3

    monkeypatch.setattr(
        trainer_module.AtomicData,
        "to_AtomicDataDict",
        lambda batch: _two_graph_batch(),
    )

    def fake_compatible_from_stats(
        lossfunc,
        stats,
        *,
        source_prefix,
        prefix,
        legacy_prefix=None,
        global_step=None,
    ):
        assert legacy_prefix == "validation"
        return {
            f"{prefix}_loss": torch.tensor(9.0),
            f"{prefix}_onsite_loss": torch.tensor(4.0),
            f"{prefix}_hopping_loss": torch.tensor(5.0),
            "validation_loss": torch.tensor(9.0),
            "validation_onsite_loss": torch.tensor(4.0),
            "validation_hopping_loss": torch.tensor(5.0),
        }

    monkeypatch.setattr(
        Trainer,
        "_compatible_loss_state_from_flow_stats",
        staticmethod(fake_compatible_from_stats),
    )

    loss = trainer.validation(fast=True)

    assert loss.item() == pytest.approx(9.0)
    assert trainer._last_flow_validation_state["validation_loss"].item() == pytest.approx(9.0)
    assert trainer._last_flow_validation_state[
        "validation_compatible_euler_1_loss"
    ].item() == pytest.approx(9.0)


def test_validation_euler_only_compatible_maps_to_legacy_loss(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.model = _ValidationEndpointModel()
    trainer.flow_cfm = _EulerOnlyValidationFlow()
    trainer.validation_loader = [_FakeBatch()]
    trainer.validation_lossfunc = _ComponentLoss()
    trainer.iter = 4

    monkeypatch.setattr(
        trainer_module.AtomicData,
        "to_AtomicDataDict",
        lambda batch: _two_graph_batch(),
    )

    loss = trainer.validation(fast=True)

    # validation() now returns the compatible legacy loss when it exists, so
    # direct callers see the same aligned scalar Validationer reports.
    assert loss.item() == pytest.approx(1.5)
    assert trainer.flow_cfm.sample_calls == 1
    assert "validation_flow_random_t_loss" not in trainer._last_flow_validation_state
    assert "validation_flow_t0_loss" not in trainer._last_flow_validation_state
    assert "validation_flow_euler_1_loss" not in trainer._last_flow_validation_state
    assert trainer._last_flow_validation_state[
        "validation_compatible_euler_1_loss"
    ].item() == pytest.approx(1.5)
    assert trainer._last_flow_validation_state["validation_loss"].item() == pytest.approx(1.5)
    assert trainer._last_flow_validation_state[
        "validation_onsite_loss"
    ].item() == pytest.approx(1.0)
    assert trainer._last_flow_validation_state[
        "validation_hopping_loss"
    ].item() == pytest.approx(2.0)


def test_flow_sample_preserves_expert_masks_across_euler_steps():
    class _MaskCheckingModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            assert "expert_node_mask" in batch
            assert "expert_edge_mask" in batch
            assert "expert_idx" in batch
            out = batch.copy()
            out["node_features"] = batch["node_h0"] + 1.0
            out["edge_features"] = batch["edge_h0"] + 1.0
            return out

    flow = HamiltonianCFM(
        {
            "enabled": True,
            "prior": "zero",
            "strict_h0": True,
        }
    )
    data = _two_graph_batch()
    data["expert_node_mask"] = torch.ones(3, dtype=torch.bool)
    data["expert_edge_mask"] = torch.ones(2, dtype=torch.bool)
    data["expert_idx"] = 0
    model = _MaskCheckingModel()

    flow.sample(model, data, num_steps=3)

    assert model.calls == 3


def test_multitrainer_validation_uses_euler_sample_for_compatible_legacy_loss():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 11
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.model = _ValidationIdentityModel()
    trainer.flow_cfm = _MultiEulerValidationFlow()
    trainer.validation_loader = [_FakeBatch()]
    trainer.validation_lossfunc = _StatsForwardLoss()
    trainer.distributed_expert = True
    trainer.local_expert_idx = 0
    trainer.distance_ranges = [(0.0, 1.0)]
    trainer.world_size = 1
    trainer.endpoint_loss_mode = "reduce"
    trainer._prepare_batch_bundle = lambda batch, with_lengths=True: (_two_graph_batch(), {})
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_features"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_features"].shape[0], dtype=torch.bool),
    )
    trainer._all_reduce_ = lambda tensor, name=None: tensor

    loss = trainer.validation(fast=True)

    assert trainer.flow_cfm.sample_calls == [1, 3]
    assert loss.item() == pytest.approx(1.5)
    state = trainer._last_flow_validation_state
    assert state["validation_loss"].item() == pytest.approx(1.5)
    assert state["validation_onsite_loss"].item() == pytest.approx(1.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(2.0)
    assert state["validation_compatible_euler_1_loss"].item() == pytest.approx(1.5)
    assert state["validation_compatible_euler_3_loss"].item() == pytest.approx(4.5)


_UNUSED_MODEL = object()


class _NoopTagger:
    def tag(self, name, *, it=None, expert=None, extra=""):
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Ctx()


class _FlowPreparedModel:
    def __init__(self):
        self.seen = None

    def __call__(self, batch):
        self.seen = batch.copy()
        out = batch.copy()
        out["node_features"] = batch["node_h0"].clone()
        out["edge_features"] = batch["edge_h0"].clone()
        return out


class _PreparedFlow:
    enabled = True
    model_in_loss = False
    log_train_compatible_loss = False
    compatible_loss_to_legacy_keys = True

    def __init__(self):
        self.prepare_called = False
        self.loss_called = False

    def prepare_batch(self, batch, ref_batch):
        self.prepare_called = True
        out = batch.copy()
        ref = ref_batch.copy()
        out["node_h0"] = batch["node_h0"] + 10.0
        out["edge_h0"] = batch["edge_h0"] + 20.0
        return out, ref, object()

    def loss(self, pred, ref, ctx):
        self.loss_called = True
        assert torch.equal(pred["node_features"], pred["node_h0"])
        assert torch.equal(pred["edge_features"], pred["edge_h0"])
        loss = pred["node_features"].sum() * 0.0 + torch.tensor(5.0)
        return loss, {
            "train_flow_loss": torch.tensor(5.0),
            "train_onsite_loss": torch.tensor(2.0),
            "train_hopping_loss": torch.tensor(3.0),
        }


class _PreparedFlowWithStats(_PreparedFlow):
    log_train_compatible_loss = True

    def loss(self, pred, ref, ctx):
        self.loss_called = True
        return pred["node_features"].sum() * 0.0 + torch.tensor(5.0), {
            "train_flow_loss": torch.tensor(5.0),
            **_compatible_clean_stats(),
        }


class _PreparedFlowWithRMEMetricStats(_PreparedFlow):
    def loss(self, pred, ref, ctx):
        state = _compatible_clean_stats()
        state["_compatible_clean_stats"]["metric_space"] = "rme"
        return pred["node_features"].sum() * 0.0 + torch.tensor(5.0), {
            "train_flow_loss": torch.tensor(5.0),
            **state,
        }


def test_multitrainer_expert_payload_applies_flow_before_model():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 1
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.flow_cfm = _PreparedFlow()
    trainer.model = _FlowPreparedModel()
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_h0"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_h0"].shape[0], dtype=torch.bool),
    )

    batch = _two_graph_batch()
    result = trainer._run_one_expert_loss(
        batch,
        batch_info={},
        criterion=_ComponentLoss(),
        expert_idx=0,
        range_dis=(0.0, 1.0),
        capture_metrics=True,
    )

    assert trainer.flow_cfm.prepare_called
    assert trainer.flow_cfm.loss_called
    assert torch.equal(trainer.model.seen["node_h0"], batch["node_h0"] + 10.0)
    assert torch.equal(trainer.model.seen["edge_h0"], batch["edge_h0"] + 20.0)
    assert result["loss"].item() == pytest.approx(5.0)
    assert result["onsite"].item() == pytest.approx(10.0)
    assert result["hopping"].item() == pytest.approx(20.0)


def test_multitrainer_non_display_step_does_not_run_full_compatible_loss():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 2
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.flow_cfm = _PreparedFlowWithStats()
    trainer.model = _FlowPreparedModel()
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_h0"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_h0"].shape[0], dtype=torch.bool),
    )
    lossfunc = _StatsCompatibleLoss()

    result = trainer._run_one_expert_loss(
        _two_graph_batch(),
        batch_info={},
        criterion=lossfunc,
        expert_idx=0,
        range_dis=(0.0, 1.0),
        capture_metrics=True,
    )

    assert lossfunc.forward_calls == 0
    assert lossfunc.stats_calls == 1
    assert result["last_onsite_count"].item() == pytest.approx(2.0)
    assert result["last_hopping_count"].item() == pytest.approx(3.0)
    assert result["onsite"].item() > 0.0
    assert result["hopping"].item() > 0.0


def test_multitrainer_flow_fallback_replaces_cross_space_raw_stats():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 2
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.flow_cfm = _PreparedFlowWithRMEMetricStats()
    trainer.model = _FlowPreparedModel()
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_h0"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_h0"].shape[0], dtype=torch.bool),
    )
    lossfunc = _BlockEndpointFallbackLoss()

    result = trainer._run_one_expert_loss(
        _two_graph_batch(),
        batch_info={},
        criterion=lossfunc,
        expert_idx=0,
        range_dis=(0.0, 1.0),
        capture_metrics=True,
    )

    assert lossfunc.stats_calls == 0
    assert lossfunc.forward_calls == 1
    assert result["onsite"].item() == pytest.approx(10.0)
    assert result["hopping"].item() == pytest.approx(20.0)
    assert result["last_onsite_l1_sum"].item() == pytest.approx(30.0)
    assert result["last_hopping_l1_sum"].item() == pytest.approx(40.0)


def test_multitrainer_reference_changes_opt_only_and_obeys_flow_scope():
    trainer = object.__new__(MultiTrainer)
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer.flow_cfm = SimpleNamespace(enabled=True, apply_to_reference=False)
    trainer.reference_lossfunc = object()
    calls = []

    main = {
        "loss": torch.tensor(2.0),
        "active_nodes": torch.tensor(3.0),
        "active_edges": torch.tensor(2.0),
        "onsite": torch.tensor(1.0),
        "hopping": torch.tensor(4.0),
        "last_onsite_l1_sum": torch.tensor(3.0),
        "last_onsite_mse_sum": torch.tensor(3.0),
        "last_onsite_count": torch.tensor(3.0),
        "last_hopping_l1_sum": torch.tensor(8.0),
        "last_hopping_mse_sum": torch.tensor(32.0),
        "last_hopping_count": torch.tensor(2.0),
        "z_loss": None,
        "expert_load_cv": None,
    }
    reference = dict(main)
    reference.update(
        loss=torch.tensor(3.0),
        active_nodes=torch.tensor(30.0),
        active_edges=torch.tensor(20.0),
        onsite=torch.tensor(10.0),
        hopping=torch.tensor(40.0),
    )

    def fake_run(**kwargs):
        calls.append(kwargs)
        return reference if kwargs["batch_dict"]["kind"] == "reference" else main

    trainer._run_one_expert_loss = fake_run
    payload = trainer._build_train_payload(
        {"kind": "main"},
        batch_info={},
        expert_idx=0,
        range_dis=(0.0, 1.0),
        ref_batch_dict={"kind": "reference"},
        ref_batch_info={},
        criterion=object(),
    )

    assert payload["loss"].item() == pytest.approx(5.0)
    assert payload["active_nodes"].item() == pytest.approx(3.0)
    assert payload["onsite_l1_sum"].item() == pytest.approx(3.0)
    assert calls[0].get("use_flow") is None
    assert calls[1]["use_flow"] is False
    assert calls[1]["criterion"] is trainer.reference_lossfunc


def test_multitrainer_full_forward_returns_endpoint_not_opt_loss():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 1
    trainer.model = lambda batch: dict(batch)

    assert trainer._run_full_batch_loss(
        {},
        {},
        _DistinctEndpointLoss(),
    ).item() == pytest.approx(2.0)


def test_pixel_meanflow_train_endpoint_stats_feed_compatible_reducer():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 2
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.endpoint_loss_mode = "reduce"
    trainer.flow_cfm = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
            "meanflow": {
                "aux_endpoint_weight": 0.0,
                "aux_boundary_v_weight": 0.0,
                "fd_eps": 1.0e-4,
            },
        }
    )
    trainer.model = _NonUniformEndpoint()
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_h0"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_h0"].shape[0], dtype=torch.bool),
    )
    lossfunc = _StatsCompatibleLoss()

    payload = trainer._build_train_payload(
        _two_graph_batch(),
        batch_info={},
        criterion=lossfunc,
        expert_idx=0,
        range_dis=(0.0, 1.0),
    )
    pack = trainer._make_step_pack(payload)
    stats_calls_before_reduce = lossfunc.stats_calls
    state = trainer._compute_compatible_state_from_pack(
        pack,
        criterion=lossfunc,
        prefix="train",
    )

    onsite = 0.5 * (1.0 + (5.0 / 3.0) ** 0.5)
    hopping = 0.5 * (2.5 + (17.0 / 2.0) ** 0.5)
    total = 0.5 * (onsite + hopping)
    assert payload["onsite_l1_sum"].item() == pytest.approx(3.0)
    assert payload["onsite_mse_sum"].item() == pytest.approx(5.0)
    assert payload["onsite_cnt"].item() == pytest.approx(3.0)
    assert payload["hopping_l1_sum"].item() == pytest.approx(5.0)
    assert payload["hopping_mse_sum"].item() == pytest.approx(17.0)
    assert payload["hopping_cnt"].item() == pytest.approx(2.0)
    assert state["train_onsite_loss"].item() == pytest.approx(onsite)
    assert state["train_hopping_loss"].item() == pytest.approx(hopping)
    assert state["train_loss"].item() == pytest.approx(total)
    assert lossfunc.forward_calls == 0
    assert stats_calls_before_reduce == 1
    assert lossfunc.stats_calls == stats_calls_before_reduce + 1


def test_multitrainer_model_in_loss_uses_flow_stats_for_compatible_metrics():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 2
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.flow_cfm = _ModelInLossFlowWithStats()
    trainer.model = _UNUSED_MODEL
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_h0"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_h0"].shape[0], dtype=torch.bool),
    )
    lossfunc = _StatsCompatibleLoss()

    result = trainer._run_one_expert_loss(
        _two_graph_batch(),
        batch_info={},
        criterion=lossfunc,
        expert_idx=0,
        range_dis=(0.0, 1.0),
        capture_metrics=True,
    )

    onsite = 0.5 * (2.0 + (10.0 / 2.0) ** 0.5)
    hopping = 0.5 * (1.0 + 3.0 ** 0.5)

    assert result["loss"].item() == pytest.approx(7.0)
    assert lossfunc.forward_calls == 0
    assert lossfunc.stats_calls == 1
    assert result["onsite"].item() == pytest.approx(onsite)
    assert result["hopping"].item() == pytest.approx(hopping)
    assert result["last_onsite_count"].item() == pytest.approx(2.0)
    assert result["last_hopping_count"].item() == pytest.approx(3.0)


def _trainer_for_compatible_pack(lossfunc):
    trainer = object.__new__(MultiTrainer)
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer.train_lossfunc = lossfunc
    trainer.endpoint_loss_mode = "reduce"
    trainer.iter = 2
    return trainer


def _pack_with_conflicting_active_component_means():
    pack = torch.zeros(MultiTrainer._PACK_LEN, dtype=torch.float32)
    pack[MultiTrainer._P_LOSS_OPT_SUM] = 5.0
    pack[MultiTrainer._P_STEP_COUNT] = 1.0
    pack[MultiTrainer._P_ONSITE_WEIGHTED_SUM] = 99.0
    pack[MultiTrainer._P_HOPPING_WEIGHTED_SUM] = 77.0
    pack[MultiTrainer._P_ACTIVE_NODES_SUM] = 1.0
    pack[MultiTrainer._P_ACTIVE_EDGES_SUM] = 1.0
    pack[MultiTrainer._P_ONSITE_L1_SUM] = 4.0
    pack[MultiTrainer._P_ONSITE_MSE_SUM] = 10.0
    pack[MultiTrainer._P_ONSITE_CNT_SUM] = 2.0
    pack[MultiTrainer._P_HOPPING_L1_SUM] = 3.0
    pack[MultiTrainer._P_HOPPING_MSE_SUM] = 9.0
    pack[MultiTrainer._P_HOPPING_CNT_SUM] = 3.0
    return pack


def test_compatible_pack_component_tags_use_same_stats_semantics_as_total():
    trainer = _trainer_for_compatible_pack(_StatsCompatibleLoss())
    state = trainer._pack_component_state(
        _pack_with_conflicting_active_component_means(),
        prefix="validation",
        criterion=trainer.train_lossfunc,
    )

    onsite = 0.5 * (2.0 + (10.0 / 2.0) ** 0.5)
    hopping = 0.5 * (1.0 + 3.0 ** 0.5)
    assert state["validation_onsite_loss"].item() == pytest.approx(onsite)
    assert state["validation_hopping_loss"].item() == pytest.approx(hopping)


def test_compatible_pack_component_tags_are_always_emitted():
    trainer = _trainer_for_compatible_pack(_StatsCompatibleLoss())
    state = trainer._pack_component_state(
        _pack_with_conflicting_active_component_means(),
        prefix="validation",
        criterion=trainer.train_lossfunc,
    )

    onsite = 0.5 * (2.0 + (10.0 / 2.0) ** 0.5)
    hopping = 0.5 * (1.0 + 3.0 ** 0.5)
    assert state["validation_onsite_loss"].item() == pytest.approx(onsite)
    assert state["validation_hopping_loss"].item() == pytest.approx(hopping)


def test_display_window_component_tags_use_compatible_stats_semantics():
    trainer = _trainer_for_compatible_pack(_StatsCompatibleLoss())
    trainer.distributed_expert = False
    trainer.num_experts = 1
    trainer.world_size = 1
    trainer._tagger = _NoopTagger()
    trainer.display_sync_freq = 1
    trainer._display_window_pack_local = _pack_with_conflicting_active_component_means()
    trainer._display_window_dynamic_batch_pack_local = torch.zeros(
        MultiTrainer._DB_PACK_LEN, dtype=torch.float32
    )
    trainer._gather_cuda_memory_metrics = lambda: {}
    trainer._all_reduce_ = lambda tensor, name=None: tensor
    trainer._gather_display_window_expert_metrics = lambda: [
        torch.tensor([99.0, 77.0, 0.0, 0.1, 1.0, 1.0])
    ]
    trainer._rank_to_expert_idx = lambda rank_idx: 0
    trainer._add_optimizer_diagnostics_to_state = lambda state: None
    trainer._add_cuda_memory_state = lambda state, metrics: None
    trainer._reset_display_window_buffers = lambda: None

    state = trainer._flush_display_window(time_idx=2)

    onsite = 0.5 * (2.0 + (10.0 / 2.0) ** 0.5)
    hopping = 0.5 * (1.0 + 3.0 ** 0.5)
    total = 0.5 * (onsite + hopping)
    assert _scalar(state["train_loss"]) == pytest.approx(total)
    assert _scalar(state["train_onsite_loss"]) == pytest.approx(onsite)
    assert _scalar(state["train_hopping_loss"]) == pytest.approx(hopping)


def test_display_window_component_tags_are_always_emitted():
    trainer = _trainer_for_compatible_pack(_StatsCompatibleLoss())
    trainer.distributed_expert = False
    trainer.num_experts = 1
    trainer.world_size = 1
    trainer._tagger = _NoopTagger()
    trainer.display_sync_freq = 1
    trainer._display_window_pack_local = _pack_with_conflicting_active_component_means()
    trainer._display_window_dynamic_batch_pack_local = torch.zeros(
        MultiTrainer._DB_PACK_LEN, dtype=torch.float32
    )
    trainer._gather_cuda_memory_metrics = lambda: {}
    trainer._all_reduce_ = lambda tensor, name=None: tensor
    trainer._gather_display_window_expert_metrics = lambda: [
        torch.tensor([99.0, 77.0, 0.0, 0.1, 1.0, 1.0])
    ]
    trainer._rank_to_expert_idx = lambda rank_idx: 0
    trainer._add_optimizer_diagnostics_to_state = lambda state: None
    trainer._add_cuda_memory_state = lambda state, metrics: None
    trainer._reset_display_window_buffers = lambda: None

    state = trainer._flush_display_window(time_idx=2)

    onsite = 0.5 * (2.0 + (10.0 / 2.0) ** 0.5)
    hopping = 0.5 * (1.0 + 3.0 ** 0.5)
    total = 0.5 * (onsite + hopping)
    assert _scalar(state["train_loss"]) == pytest.approx(total)
    assert _scalar(state["train_onsite_loss"]) == pytest.approx(onsite)
    assert _scalar(state["train_hopping_loss"]) == pytest.approx(hopping)


def test_validationer_preserves_clean_validation_loss_from_flow_state():
    class _TrainerWithFlowValidation:
        def __init__(self):
            self.stats = {}
            self.ep = 3
            self._last_flow_validation_state = {}

        def validation(self, fast=True):
            self._last_flow_validation_state = {
                "validation_loss": torch.tensor(1.25),
                "validation_onsite_loss": torch.tensor(0.5),
                "validation_hopping_loss": torch.tensor(2.0),
            }
            return torch.tensor(99.0)

    trainer = _TrainerWithFlowValidation()
    validationer = Validationer(interval=1)
    validationer.trainer = trainer

    value = validationer._get_value(field="iteration", time=10)
    assert _scalar(value) == pytest.approx(1.25)
    assert trainer.stats["validation_loss"]["last"] == pytest.approx(1.25)

    validationer.epoch(time=3)
    assert trainer.stats["validation_loss"]["epoch_mean"] == pytest.approx(1.25)


def test_non_metric_scheduler_does_not_reduce_compatible_scalar():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer = object.__new__(MultiTrainer)
    trainer.update_lr_per_iter = True
    trainer.distributed_expert = True
    trainer.local_expert_idx = 0
    trainer.lr_schedulers = [scheduler]
    trainer.iter = 2
    trainer._tagger = _NoopTagger()
    trainer._mean_expert_dp_scalar = lambda value: pytest.fail(
        "metric-free scheduler must not all_reduce a scalar"
    )

    trainer._local_scheduler_step(None)


def test_disabled_iter_scheduler_does_not_request_metric():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    trainer = object.__new__(MultiTrainer)
    trainer.update_lr_per_iter = False
    trainer.distributed_expert = True
    trainer.local_expert_idx = 0
    trainer.lr_schedulers = [scheduler]

    assert trainer._local_scheduler_requires_metric() is False


def test_tensorboard_monitor_writes_fresh_validation_iter_tags():
    writes = []

    class _Writer:
        def add_scalar(self, tag, value, step):
            writes.append((tag, float(value), int(step)))

        def flush(self):
            pass

    monitor = object.__new__(TensorBoardMonitor)
    monitor.writer = _Writer()
    monitor.flush_every = 0
    monitor.trainer = SimpleNamespace(
        iter=1000,
        num_experts=0,
        stats={
            "validation_loss": {"last": 0.5, "last_updated": 1000},
            "validation_loss_opt": {"last": 0.7, "last_updated": 1000},
            "validation_onsite_loss": {"last": 0.2, "last_updated": 1000},
            "validation_hopping_loss": {"last": 0.3, "last_updated": 1000},
        },
    )

    monitor.iteration(time=1000)

    assert ("validation_loss_iter/iteration", 0.5, 1000) in writes
    assert ("validation_loss_opt_iter/iteration", 0.7, 1000) in writes
    assert ("validation_onsite_loss_iter/iteration", 0.2, 1000) in writes
    assert ("validation_hopping_loss_iter/iteration", 0.3, 1000) in writes


def test_tensorboard_register_writes_endpoint_metric_space_metadata():
    texts = []
    scalars = []

    class _Writer:
        def add_text(self, tag, value, step):
            texts.append((tag, value, step))

        def add_scalar(self, tag, value, step):
            scalars.append((tag, value, step))

        def flush(self):
            pass

    monitor = object.__new__(TensorBoardMonitor)
    monitor.writer = _Writer()
    trainer = SimpleNamespace(
        endpoint_metric_spaces={"train": "block", "validation": "rme"}
    )

    monitor.register(trainer)

    assert ("metadata/train_endpoint_metric_space", "block", 0) in texts
    assert ("metadata/validation_endpoint_metric_space", "rme", 0) in texts
    assert (
        "metadata/train_endpoint_metric_space_is_rme",
        0.0,
        0,
    ) in scalars
    assert (
        "metadata/validation_endpoint_metric_space_is_rme",
        1.0,
        0,
    ) in scalars


def test_tensorboard_monitor_writes_epoch_validation_on_iteration_axis():
    writes = []

    class _Writer:
        def add_scalar(self, tag, value, step):
            writes.append((tag, float(value), int(step)))

        def flush(self):
            pass

    monitor = object.__new__(TensorBoardMonitor)
    monitor.writer = _Writer()
    monitor.flush_every = 0
    monitor.trainer = SimpleNamespace(
        iter=4321,
        ep=4,
        num_experts=0,
        stats={
            "validation_loss": {
                "epoch_mean": 0.5,
                "last": 0.5,
                "epoch_last_updated": 4,
            },
            "validation_onsite_loss": {
                "epoch_mean": 0.2,
                "last": 0.2,
                "epoch_last_updated": 4,
            },
            "validation_hopping_loss": {
                "epoch_mean": 0.3,
                "last": 0.3,
                "epoch_last_updated": 4,
            },
            "validation_compatible_euler_1_loss": {
                "epoch_mean": 0.5,
                "last": 0.5,
                "epoch_last_updated": 4,
            },
            "validation_compatible_euler_1_onsite_loss": {
                "epoch_mean": 0.2,
                "last": 0.2,
                "epoch_last_updated": 4,
            },
            "validation_compatible_euler_1_hopping_loss": {
                "epoch_mean": 0.3,
                "last": 0.3,
                "epoch_last_updated": 4,
            },
            "validation_flow_one_step_loss": {
                "epoch_mean": 0.9,
                "last": 0.9,
                "epoch_last_updated": 4,
            },
        },
    )

    monitor.epoch(time=4)

    assert ("validation_loss_iter/iteration", 0.5, 4321) in writes
    assert ("validation_onsite_loss_iter/iteration", 0.2, 4321) in writes
    assert ("validation_hopping_loss_iter/iteration", 0.3, 4321) in writes
    assert ("validation_compatible_euler_1_loss_iter/iteration", 0.5, 4321) not in writes
    assert ("validation_compatible_euler_1_onsite_loss_iter/iteration", 0.2, 4321) not in writes
    assert ("validation_compatible_euler_1_hopping_loss_iter/iteration", 0.3, 4321) not in writes
    assert ("validation_flow_one_step_loss_iter/iteration", 0.9, 4321) in writes


def test_model_in_loss_skips_train_compatible_loss_from_raw_batch(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.flow_cfm = _ModelInLossFlow()
    trainer.model = _UNUSED_MODEL

    def fake_to_dict(batch):
        return {"raw_batch": True}

    def fail_compatible(*args, **kwargs):
        raise AssertionError("model-in-loss pMF must not log raw-batch train compatible loss")

    monkeypatch.setattr(trainer_module.AtomicData, "to_AtomicDataDict", fake_to_dict)
    monkeypatch.setattr(Trainer, "_compatible_loss_state", staticmethod(fail_compatible))

    with pytest.raises(RuntimeError, match="could not reconstruct"):
        trainer._loss_on_batch(_FakeBatch(), _ComponentLoss())


def test_model_in_loss_train_loss_aligns_from_endpoint_stats(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.flow_cfm = _ModelInLossFlowWithStats()
    trainer.model = _UNUSED_MODEL
    lossfunc = _StatsCompatibleLoss()

    def fake_to_dict(batch):
        return {"raw_batch": True}

    def fail_compatible(*args, **kwargs):
        raise AssertionError("model-in-loss pMF must use flow stats, not raw-batch criterion")

    monkeypatch.setattr(trainer_module.AtomicData, "to_AtomicDataDict", fake_to_dict)
    monkeypatch.setattr(Trainer, "_compatible_loss_state", staticmethod(fail_compatible))

    loss = trainer._loss_on_batch(_FakeBatch(), lossfunc)
    state = trainer._last_flow_state

    onsite = 0.5 * (2.0 + (10.0 / 2.0) ** 0.5)
    hopping = 0.5 * (1.0 + 3.0 ** 0.5)
    aligned_total = 0.5 * (onsite + hopping)

    assert loss.item() == pytest.approx(7.0)
    assert lossfunc.forward_calls == 0
    assert lossfunc.stats_calls == 1
    assert state["train_loss_opt"].item() == pytest.approx(7.0)
    assert state["train_loss"].item() == pytest.approx(aligned_total)
    assert state["train_onsite_loss"].item() == pytest.approx(onsite)
    assert state["train_hopping_loss"].item() == pytest.approx(hopping)


def test_loss_on_batch_can_skip_flow_for_reference_batch(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.flow_cfm = _ModelInLossFlow()

    class ReferenceModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            pred = batch.copy()
            pred["node_features"] = pred["node_features"] + 2.0
            pred["edge_features"] = pred["edge_features"] + 3.0
            return pred

    def fail_loss_with_model(*args, **kwargs):
        raise AssertionError("reference batches should not enter pMF loss_with_model by default")

    def fake_to_dict(batch):
        return {
            "node_features": torch.tensor([[1.0]]),
            "edge_features": torch.tensor([[2.0]]),
        }

    model = ReferenceModel()
    trainer.model = model
    monkeypatch.setattr(trainer.flow_cfm, "loss_with_model", fail_loss_with_model)
    monkeypatch.setattr(trainer_module.AtomicData, "to_AtomicDataDict", fake_to_dict)

    loss = trainer._loss_on_batch(_FakeBatch(), _ComponentLoss(), use_flow=False)

    assert loss.item() == pytest.approx(2.5)
    assert trainer._last_flow_state == {}
    assert model.calls == 1


def test_iteration_reference_batch_does_not_overwrite_main_flow_state():
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.flow_cfm = SimpleNamespace(enabled=True, apply_to_reference=False)
    trainer.train_lossfunc = object()
    trainer.reference_lossfunc = object()
    trainer.clip_grad_norm = 1.0
    trainer.update_lr_per_iter = False
    trainer.optimizer_diagnostics_freq = 999
    trainer.iter = 2
    trainer.num_experts = 0
    captured = {}

    def fake_loss_on_batch(
        self,
        batch,
        lossfunc,
        *,
        use_flow=True,
        allow_self_consistency=True,
    ):
        parameter = next(self.model.parameters())
        if use_flow:
            self._last_flow_state = {
                "train_loss": torch.tensor(2.0),
                "train_onsite_loss": torch.tensor(1.0),
                "train_hopping_loss": torch.tensor(3.0),
                "train_flow_loss": torch.tensor(7.0),
                "train_loss_opt": torch.tensor(7.0),
            }
            self._last_self_consistency_state = {}
            return parameter.sum() * 0.0 + 7.0
        self._last_flow_state = {}
        return parameter.sum() * 0.0 + 3.0

    trainer._loss_on_batch = fake_loss_on_batch.__get__(trainer, Trainer)
    trainer.call_plugins = lambda **kwargs: captured.update(kwargs)

    trainer.iteration(_FakeBatch(), _FakeBatch())

    assert captured["train_loss"].item() == pytest.approx(2.0)
    assert captured["train_onsite_loss"].item() == pytest.approx(1.0)
    assert captured["train_hopping_loss"].item() == pytest.approx(3.0)
    assert captured["train_flow_loss"].item() == pytest.approx(7.0)
    assert captured["train_loss_opt"].item() == pytest.approx(10.0)


def test_pixel_meanflow_oracle_endpoint_has_zero_velocity_loss():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
            "meanflow": {
                "aux_endpoint_weight": 0.0,
                "jvp_backend": "finite_difference",
                "fd_eps": 1.0e-4,
            },
        }
    )
    r = torch.tensor([0.2, 0.3])
    t = torch.tensor([0.5, 0.7])

    loss, state = flow.loss_with_model(_ConstantEndpoint(), _two_graph_batch(), _two_graph_ref(), r=r, t=t)

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_h"].item() == pytest.approx(float((t - r).mean()), abs=1.0e-6)
    assert state["train_flow_onsite_velocity_mse"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_hopping_velocity_mse"].item() == pytest.approx(0.0, abs=1.0e-6)


def test_pixel_meanflow_semigroup_oracle_endpoint_has_zero_state_loss():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
            "meanflow": {
                "objective": "semigroup",
                "semigroup_endpoint_weight": 0.0,
            },
        }
    )
    r = torch.tensor([0.2, 0.3])
    t = torch.tensor([0.5, 0.7])

    loss, state = flow.loss_with_model(_ConstantEndpoint(), _two_graph_batch(), _two_graph_ref(), r=r, t=t)

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_objective_semigroup"].item() == pytest.approx(1.0)
    assert state["train_flow_onsite_semigroup_mse"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_hopping_semigroup_mse"].item() == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.parametrize(
    ("aux_boundary_v_weight", "expected_grad_modes"),
    [
        (0.0, [True, False, False]),
        (0.2, [True, True, False]),
    ],
)
def test_pixel_meanflow_uses_no_grad_boundary_when_boundary_aux_disabled(
    aux_boundary_v_weight, expected_grad_modes
):
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
            "meanflow": {
                "jvp_tangent": "boundary",
                "aux_boundary_v_weight": aux_boundary_v_weight,
                "fd_eps": 1.0e-4,
            },
        }
    )
    model = _GradModeRecordingEndpoint()

    flow.loss_with_model(
        model,
        _two_graph_batch(),
        _two_graph_ref(),
        r=torch.tensor([0.2, 0.3]),
        t=torch.tensor([0.5, 0.7]),
    )

    assert model.grad_modes == expected_grad_modes


def test_pixel_meanflow_one_step_sampler_reaches_constant_endpoint():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
        }
    )

    sampled = flow.sample(_ConstantEndpoint(), _two_graph_batch(), num_steps=1)

    assert torch.allclose(sampled["node_features"], torch.full((3, 1), 2.0))
    assert torch.allclose(sampled["edge_features"], torch.full((2, 1), 4.0))
    assert torch.equal(sampled["flow_time"], torch.zeros(2))
    assert torch.equal(sampled["flow_time_r"], torch.zeros(2))
    assert torch.equal(sampled["flow_time_h"], torch.zeros(2))


class _ConstantEndpointWithBlocks(torch.nn.Module):
    """Block-native surrogate: emits feature keys plus Hamiltonian block keys
    (as a block-native output head would), so pMF sampling can be checked to
    carry the model's full output surface."""

    def forward(self, data):
        data = data.copy()
        data["node_features"] = torch.full_like(data["node_h0"], 2.0)
        data["edge_features"] = torch.full_like(data["edge_h0"], 4.0)
        data["node_hamil_blocks"] = torch.full((data["node_h0"].shape[0], 2, 2), 2.0)
        data["edge_hamil_blocks"] = torch.full((data["edge_h0"].shape[0], 2, 2), 4.0)
        return data


def test_pixel_meanflow_sample_carries_model_block_outputs():
    # Regression: pMF sample() used to return `state.copy()` of the *input*
    # data -- no model outputs at all -- so block-consuming losses (blockwise
    # compatible validation) KeyError'd on node_hamil_blocks/edge_hamil_blocks.
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "prior": "zero",
            "strict_h0": True,
        }
    )
    sampled = flow.sample(_ConstantEndpointWithBlocks(), _two_graph_batch(), num_steps=1)
    assert "node_hamil_blocks" in sampled and "edge_hamil_blocks" in sampled
    assert torch.allclose(sampled["node_hamil_blocks"], torch.full((3, 2, 2), 2.0))
    # integrated endpoint features still win over the final forward's features
    assert torch.allclose(sampled["node_features"], torch.full((3, 1), 2.0))
    assert torch.allclose(sampled["edge_features"], torch.full((2, 1), 4.0))
    assert torch.equal(sampled["flow_time"], torch.zeros(2))


def test_pixel_meanflow_sample_final_forward_opt_out():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "prior": "zero",
            "strict_h0": True,
            "meanflow": {"sample_final_forward": False},
        }
    )
    sampled = flow.sample(_ConstantEndpointWithBlocks(), _two_graph_batch(), num_steps=1)
    assert "node_hamil_blocks" not in sampled
    assert torch.allclose(sampled["node_features"], torch.full((3, 1), 2.0))


# ---------------------------------------------------------------------------
# Pixel MeanFlow validation-semantics alignment with no-CFM/CFM legacy keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("objective", ["pixel_meanflow", "meanflow"])
def test_meanflow_objectives_default_validation_compatible_alignment(objective):
    flow = build_hamiltonian_flow({"enabled": True, "objective": objective})

    assert isinstance(flow, HamiltonianPixelMeanFlow)
    assert flow.log_validation_compatible_loss is True
    assert flow.compatible_loss_to_legacy_keys is True
    assert 1 in {int(n) for n in flow.validation_ode_steps}


def test_pixel_meanflow_validation_compatible_cannot_opt_out():
    flow = build_hamiltonian_flow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"log_validation_compatible_loss": False},
        }
    )

    assert flow.log_validation_compatible_loss is True
    assert flow.compatible_loss_to_legacy_keys is True


def test_pixel_meanflow_train_compatible_alignment_is_forced_on():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
        }
    )
    opt_out = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"log_train_compatible_loss": False},
        }
    )

    assert flow.log_train_compatible_loss is True
    assert opt_out.log_train_compatible_loss is True
    assert opt_out.compatible_loss_to_legacy_keys is True


class _ValidationConstantModel:
    """Constant x-prediction surrogate for end-to-end pMF validation."""

    def eval(self):
        return None

    def __call__(self, batch):
        out = batch.copy()
        out["node_features"] = torch.ones_like(batch["node_h0"])
        out["edge_features"] = torch.full_like(batch["edge_h0"], 2.0)
        return out


def _pixel_meanflow_validation_trainer(monkeypatch, flow_overrides=None):
    options = {
        "enabled": True,
        "objective": "pixel_meanflow",
        "mode": "residual",
        "prior": "zero",
        "strict_h0": True,
        "meanflow": {"fd_eps": 1.0e-4},
    }
    if flow_overrides:
        options["meanflow"].update(flow_overrides.pop("meanflow", {}))
        options.update(flow_overrides)

    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.model = _ValidationConstantModel()
    trainer.flow_cfm = build_hamiltonian_flow(options)
    trainer.validation_loader = [_FakeBatch()]
    trainer.validation_lossfunc = _ComponentLoss()
    trainer.iter = 5

    monkeypatch.setattr(
        trainer_module.AtomicData,
        "to_AtomicDataDict",
        lambda batch: _two_graph_batch(),
    )
    return trainer


def test_pixel_meanflow_validation_writes_legacy_endpoint_compatible_keys(monkeypatch):
    trainer = _pixel_meanflow_validation_trainer(monkeypatch)

    loss = trainer.validation(fast=True)
    st = trainer._last_flow_validation_state

    # Legacy keys must carry the euler/endpoint blockwise compatible loss so
    # pMF curves line up with no-CFM/CFM validation semantics, and validation()
    # itself must return that aligned scalar rather than the flow objective.
    assert loss.item() == pytest.approx(1.5)
    assert st["validation_loss"].item() == pytest.approx(1.5)
    assert st["validation_onsite_loss"].item() == pytest.approx(1.0)
    assert st["validation_hopping_loss"].item() == pytest.approx(2.0)
    assert st["validation_compatible_euler_1_loss"].item() == pytest.approx(1.5)
    assert st["validation_compatible_euler_3_loss"].item() == pytest.approx(1.5)

    # The meanflow objective stays observable under validation_flow_* keys and
    # must not be what the legacy validation_loss reports.
    assert "validation_flow_random_t_loss" in st
    assert "validation_flow_one_step_loss" in st
    assert st["validation_flow_random_t_loss"].item() != pytest.approx(
        st["validation_loss"].item()
    )
    # one_step flow objective scalars are renamespaced under validation_flow_*
    # so the TensorBoard prefix scan picks them up.
    assert "validation_flow_one_step_onsite_velocity_loss" in st
    assert not any(key.startswith("validation_one_step_flow_") for key in st)


class _ScalarOnlyLoss(torch.nn.Module):
    def forward(self, pred, ref):
        return (pred["node_features"] - ref["node_features"]).abs().mean()


def test_pixel_meanflow_validation_fails_closed_without_endpoint_components(
    monkeypatch,
):
    trainer = _pixel_meanflow_validation_trainer(monkeypatch)
    trainer.validation_lossfunc = _ScalarOnlyLoss()

    with pytest.raises(RuntimeError, match="endpoint triplet"):
        trainer.validation(fast=True)


def test_pixel_meanflow_validation_compatible_sampling_is_forced(monkeypatch):
    trainer = _pixel_meanflow_validation_trainer(
        monkeypatch,
        flow_overrides={"meanflow": {"log_validation_compatible_loss": False}},
    )
    sample_calls = []
    original_sample = trainer.flow_cfm.sample

    def counting_sample(model, batch, *, num_steps):
        sample_calls.append(int(num_steps))
        return original_sample(model, batch, num_steps=num_steps)

    monkeypatch.setattr(trainer.flow_cfm, "sample", counting_sample)

    trainer.validation(fast=True)
    st = trainer._last_flow_validation_state

    assert sample_calls == [1, 3]
    assert "validation_loss" in st
    assert "validation_onsite_loss" in st
    assert "validation_hopping_loss" in st


def test_resolve_flow_log_fields_pixel_meanflow_drops_never_computed_fields():
    from dptb.nnops.flow import resolve_flow_log_fields

    flow = build_hamiltonian_flow({"enabled": True, "objective": "pixel_meanflow"})
    fields, register_legacy = resolve_flow_log_fields(flow)

    # pMF never computes the raw-batch train compatible loss; registering the
    # fields would print misleading constant zeros in the terminal logger.
    assert "train_compatible_loss" not in fields
    assert "train_compatible_onsite_loss" not in fields
    assert "train_compatible_hopping_loss" not in fields
    # pMF's validation branch never emits CFM's euler flow objective or the
    # CFM t0 key; it emits validation_flow_one_step_loss instead.
    assert "validation_flow_t0_loss" not in fields
    assert "validation_flow_euler_1_loss" not in fields
    assert "validation_flow_euler_3_loss" not in fields
    assert "validation_flow_one_step_loss" in fields
    assert "validation_flow_random_t_loss" in fields
    # Euler-1 maps to the common validation triplet; only extra steps are logged.
    assert "validation_compatible_euler_1_loss" not in fields
    assert "validation_compatible_euler_3_hopping_loss" in fields
    # canary scalars so silent jvp fallbacks are visible in production logs
    assert "train_flow_du_dt_backend_jvp" in fields
    assert "train_flow_explicit_model_calls" in fields
    assert register_legacy is True


def test_resolve_flow_log_fields_includes_semigroup_meanflow_fields():
    from dptb.nnops.flow import resolve_flow_log_fields

    flow = build_hamiltonian_flow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"objective": "semigroup"},
        }
    )
    fields, register_legacy = resolve_flow_log_fields(flow)

    assert "train_flow_objective_semigroup" in fields
    assert "train_flow_semigroup_split_t" in fields
    assert "train_flow_onsite_semigroup_loss" in fields
    assert "train_flow_hopping_semigroup_loss" in fields
    assert register_legacy is True


def test_resolve_flow_log_fields_cfm_keeps_existing_fields():
    from dptb.nnops.flow import resolve_flow_log_fields

    flow = build_hamiltonian_flow({"enabled": True, "objective": "cfm"})
    fields, register_legacy = resolve_flow_log_fields(flow)

    assert "train_compatible_loss" not in fields
    assert "validation_flow_t0_loss" in fields
    assert "validation_flow_euler_1_loss" in fields
    assert "validation_compatible_euler_1_loss" not in fields
    assert "validation_flow_one_step_loss" not in fields
    assert "train_flow_du_dt_backend_jvp" not in fields
    assert register_legacy is True


def test_resolve_flow_log_fields_uses_common_triplet_for_euler_one():
    from dptb.nnops.flow import resolve_flow_log_fields

    flow = build_hamiltonian_flow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"log_validation_compatible_loss": False},
        }
    )
    fields, register_legacy = resolve_flow_log_fields(flow)

    assert "validation_compatible_euler_1_loss" not in fields
    assert "validation_compatible_euler_1_onsite_loss" not in fields
    assert "validation_compatible_euler_1_hopping_loss" not in fields
    assert register_legacy is True


def test_validation_ode_steps_always_include_euler_one_endpoint_baseline():
    flow = build_hamiltonian_flow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "validation_ode_steps": [3],
        }
    )

    assert flow.validation_ode_steps == (1, 3)


def test_resolve_flow_log_fields_disabled_flow_keeps_legacy_registration():
    from dptb.nnops.flow import resolve_flow_log_fields

    flow = build_hamiltonian_flow({"enabled": False})
    fields, register_legacy = resolve_flow_log_fields(flow)

    assert fields == []
    assert register_legacy is True


def test_train_entrypoint_registers_flow_fields_from_effective_flags():
    text = Path(train_entrypoint.__file__).read_text(encoding="utf-8")

    assert "resolve_flow_log_fields" in text


# ---------------------------------------------------------------------------
# Pixel MeanFlow JVP du/dt backend
# ---------------------------------------------------------------------------


def _jvp_flow_options(meanflow_overrides=None):
    meanflow = {
        "du_dt_backend": "jvp",
        "aux_endpoint_weight": 0.0,
    }
    if meanflow_overrides:
        meanflow.update(meanflow_overrides)
    return {
        "enabled": True,
        "objective": "pixel_meanflow",
        "mode": "residual",
        "prior": "zero",
        "strict_h0": True,
        "meanflow": meanflow,
    }


class _TimeConditionedModel(torch.nn.Module):
    """x-prediction surrogate that is genuinely nonlinear in state and time.

    du/dt therefore has both a state-transport term (through node_h0/edge_h0)
    and an explicit time-conditioning term (through flow_time_t/flow_time_h),
    which is exactly what the JVP backend must reproduce against finite
    differences.
    """

    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.w_node = torch.nn.Parameter(torch.tensor(0.7, dtype=dtype))
        self.w_edge = torch.nn.Parameter(torch.tensor(-0.4, dtype=dtype))
        self.a_node = torch.nn.Parameter(torch.tensor(0.5, dtype=dtype))
        self.a_edge = torch.nn.Parameter(torch.tensor(0.3, dtype=dtype))
        self.grad_modes = []

    def forward(self, data):
        self.grad_modes.append(torch.is_grad_enabled())
        data = data.copy()
        batch = data["batch"]
        node_t = data["flow_time_t"].index_select(0, batch).unsqueeze(-1)
        node_h = data["flow_time_h"].index_select(0, batch).unsqueeze(-1)
        edge_graph = batch.index_select(0, data["edge_index"][0])
        edge_t = data["flow_time_t"].index_select(0, edge_graph).unsqueeze(-1)
        edge_h = data["flow_time_h"].index_select(0, edge_graph).unsqueeze(-1)

        node_z = data["node_h0"]
        edge_z = data["edge_h0"]
        data["node_features"] = (
            self.w_node * node_z
            + self.a_node * torch.sin(node_z) * (node_t + 0.5 * node_t.square())
            + 0.2 * node_h * node_z.square()
        )
        data["edge_features"] = (
            self.w_edge * edge_z
            + self.a_edge * torch.cos(edge_z) * edge_t
            + 0.1 * edge_h * edge_z
        )
        return data


def _double_batch():
    return {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        "node_h0": torch.tensor([[0.1], [-0.2], [0.3]], dtype=torch.float64),
        "edge_h0": torch.tensor([[0.4], [-0.1]], dtype=torch.float64),
        "node_features": torch.zeros(3, 1, dtype=torch.float64),
        "edge_features": torch.zeros(2, 1, dtype=torch.float64),
    }


def _double_ref():
    return {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        "node_features": torch.tensor([[1.1], [0.6], [-0.7]], dtype=torch.float64),
        "edge_features": torch.tensor([[0.9], [-1.3]], dtype=torch.float64),
    }


@pytest.mark.parametrize("jvp_tangent", ["boundary", "path"])
def test_pixel_meanflow_jvp_matches_finite_difference_numerically(jvp_tangent):
    torch.manual_seed(0)
    r = torch.tensor([0.2, 0.3], dtype=torch.float64)
    t = torch.tensor([0.55, 0.8], dtype=torch.float64)

    losses = {}
    states = {}
    for backend, fd_eps in (("finite_difference", 1.0e-6), ("jvp", 1.0e-6)):
        flow = HamiltonianPixelMeanFlow(
            _jvp_flow_options({"du_dt_backend": backend, "fd_eps": fd_eps,
                               "jvp_tangent": jvp_tangent}),
            dtype=torch.float64,
        )
        model = _TimeConditionedModel()
        loss, state = flow.loss_with_model(
            model, _double_batch(), _double_ref(), r=r.clone(), t=t.clone()
        )
        losses[backend] = float(loss.item())
        states[backend] = state

    assert losses["jvp"] == pytest.approx(losses["finite_difference"], rel=1.0e-4, abs=1.0e-8)
    for key in (
        "train_flow_onsite_velocity_mse",
        "train_flow_hopping_velocity_mse",
    ):
        assert states["jvp"][key].item() == pytest.approx(
            states["finite_difference"][key].item(), rel=1.0e-4, abs=1.0e-8
        )


def test_pixel_meanflow_jvp_oracle_endpoint_has_zero_velocity_loss():
    # _ConstantEndpoint's output has no dependence on the state, so its forward
    # tangent is legitimately None -> opt out of the require-tangents guard
    # (that guard exists to catch real models that accidentally drop the dual).
    flow = HamiltonianPixelMeanFlow(
        _jvp_flow_options({"jvp_require_tangents": False})
    )
    r = torch.tensor([0.2, 0.3])
    t = torch.tensor([0.5, 0.7])

    loss, state = flow.loss_with_model(
        _ConstantEndpoint(), _two_graph_batch(), _two_graph_ref(), r=r, t=t
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_onsite_velocity_mse"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_hopping_velocity_mse"].item() == pytest.approx(0.0, abs=1.0e-6)


def test_pixel_meanflow_jvp_require_tangents_falls_back_not_silent_zero(caplog):
    import logging

    # A dropped dual (None tangent) under the default guard must NOT be silently
    # treated as du/dt=0; it raises inside jvp and the run falls back to fd.
    flow = HamiltonianPixelMeanFlow(_jvp_flow_options({"jvp_require_tangents": True}))
    with caplog.at_level(logging.WARNING, logger="dptb.nnops.flow"):
        loss, state = flow.loss_with_model(
            _ConstantEndpoint(),
            _two_graph_batch(),
            _two_graph_ref(),
            r=torch.tensor([0.2, 0.3]),
            t=torch.tensor([0.5, 0.7]),
        )

    assert torch.isfinite(loss)
    assert state["train_flow_du_dt_backend_jvp"].item() == pytest.approx(0.0)
    assert any("finite_difference" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    ("memory_efficient", "aux_boundary_v_weight", "expected_grad_modes", "expected_calls"),
    [
        # fused: boundary + one grad dual forward (primal+tangent together)
        (False, 0.0, [False, True], 2.0),
        (False, 0.2, [True, True], 2.0),
        # split (default): boundary + grad primal + no_grad tangent forward.
        # The extra no_grad forward keeps peak memory ~1x instead of ~2.2x.
        (True, 0.0, [False, True, False], 3.0),
        (True, 0.2, [True, True, False], 3.0),
    ],
)
def test_pixel_meanflow_jvp_model_call_pattern_with_boundary_tangent(
    memory_efficient, aux_boundary_v_weight, expected_grad_modes, expected_calls
):
    flow = HamiltonianPixelMeanFlow(
        _jvp_flow_options(
            {
                "jvp_tangent": "boundary",
                "aux_boundary_v_weight": aux_boundary_v_weight,
                "jvp_memory_efficient": memory_efficient,
            }
        ),
        dtype=torch.float64,
    )
    model = _TimeConditionedModel()

    loss, state = flow.loss_with_model(
        model,
        _double_batch(),
        _double_ref(),
        r=torch.tensor([0.2, 0.3], dtype=torch.float64),
        t=torch.tensor([0.55, 0.8], dtype=torch.float64),
    )

    assert model.grad_modes == expected_grad_modes
    assert state["train_flow_du_dt_backend_jvp"].item() == pytest.approx(1.0)
    assert state["train_flow_explicit_model_calls"].item() == pytest.approx(expected_calls)

    loss.backward()
    assert model.w_node.grad is not None
    assert torch.isfinite(model.w_node.grad)
    assert model.a_node.grad is not None


@pytest.mark.parametrize(
    ("memory_efficient", "expected_grad_modes", "expected_calls"),
    [
        (False, [True], 1.0),
        (True, [True, False], 2.0),
    ],
)
def test_pixel_meanflow_jvp_path_tangent_call_pattern(
    memory_efficient, expected_grad_modes, expected_calls
):
    flow = HamiltonianPixelMeanFlow(
        _jvp_flow_options(
            {
                "jvp_tangent": "path",
                "aux_boundary_v_weight": 0.0,
                "jvp_memory_efficient": memory_efficient,
            }
        ),
        dtype=torch.float64,
    )
    model = _TimeConditionedModel()

    _, state = flow.loss_with_model(
        model,
        _double_batch(),
        _double_ref(),
        r=torch.tensor([0.2, 0.3], dtype=torch.float64),
        t=torch.tensor([0.55, 0.8], dtype=torch.float64),
    )

    assert model.grad_modes == expected_grad_modes
    assert state["train_flow_explicit_model_calls"].item() == pytest.approx(expected_calls)


@pytest.mark.parametrize(
    ("flow_options", "expected"),
    [
        ({"enabled": True, "objective": "pixel_meanflow", "meanflow": {"du_dt_backend": "jvp"}}, True),
        ({"enabled": True, "objective": "meanflow", "meanflow": {"jvp_backend": "jvp"}}, True),
        ({"enabled": True, "objective": "pixel_meanflow"}, False),
        ({"enabled": True, "objective": "cfm"}, False),
        ({"enabled": False, "objective": "pixel_meanflow", "meanflow": {"du_dt_backend": "jvp"}}, False),
        (None, False),
    ],
)
def test_configure_jvp_friendly_backends_switches_e3nn_only_for_jvp(
    monkeypatch, flow_options, expected
):
    from dptb.nnops.flow import configure_jvp_friendly_backends

    e3nn = pytest.importorskip("e3nn")
    calls = []
    monkeypatch.setattr(
        e3nn, "set_optimization_defaults", lambda **kw: calls.append(kw)
    )

    assert configure_jvp_friendly_backends(flow_options) is expected
    if expected:
        assert calls == [{"jit_mode": "eager"}]
    else:
        assert calls == []


def test_train_entrypoints_prepare_jvp_backends_before_model_build():
    import dptb.entrypoints.multi_train as multi_train_entrypoint

    for module in (train_entrypoint, multi_train_entrypoint):
        text = Path(module.__file__).read_text(encoding="utf-8")
        assert "configure_jvp_friendly_backends" in text


def test_pixel_meanflow_jvp_falls_back_to_finite_difference_with_warning(
    monkeypatch, caplog
):
    import logging

    import torch.autograd.forward_ad as fwAD

    flow = HamiltonianPixelMeanFlow(_jvp_flow_options(), dtype=torch.float64)
    calls = {"jvp": 0}

    def boom(*args, **kwargs):
        calls["jvp"] += 1
        raise RuntimeError("forward AD not implemented for fake op")

    # the jvp backend now uses native torch.autograd.forward_ad, not functorch
    monkeypatch.setattr(fwAD, "make_dual", boom)

    model = _TimeConditionedModel()
    with caplog.at_level(logging.WARNING, logger="dptb.nnops.flow"):
        loss, state = flow.loss_with_model(
            model,
            _double_batch(),
            _double_ref(),
            r=torch.tensor([0.2, 0.3], dtype=torch.float64),
            t=torch.tensor([0.55, 0.8], dtype=torch.float64),
        )

    assert torch.isfinite(loss)
    assert calls["jvp"] == 1
    assert any("finite_difference" in rec.message for rec in caplog.records)
    assert state["train_flow_du_dt_backend_jvp"].item() == pytest.approx(0.0)

    # sticky fallback: the second step must not retry the broken jvp path
    flow.loss_with_model(
        _TimeConditionedModel(),
        _double_batch(),
        _double_ref(),
        r=torch.tensor([0.2, 0.3], dtype=torch.float64),
        t=torch.tensor([0.55, 0.8], dtype=torch.float64),
    )
    assert calls["jvp"] == 1

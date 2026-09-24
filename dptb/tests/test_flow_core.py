"""Hamiltonian-flow numerics: prepare_batch/loss/sample contracts, config accept/reject
tables, endpoint metric-space guards, JVP-vs-finite-difference agreement, raw-uureal
layout projection, per-graph time embedding, and flow geometry diagnostics.

Trainer/MultiTrainer logging plumbing built on top of these flows lives in
test_flow_training_metrics.py instead (no Trainer/MultiTrainer import here).
"""
from __future__ import annotations

import logging

import pytest
import torch

from dptb.data import AtomicDataDict, _keys
from dptb.nn.embedding.flow_time import FlowTimeConditioner, sinusoidal_time_embedding
from dptb.nnops.flow import (
    CFMContext,
    HamiltonianCFM,
    HamiltonianPixelMeanFlow,
    assert_model_in_loss_endpoint_metric_space,
    build_hamiltonian_flow,
    configure_jvp_friendly_backends,
    resolve_flow_log_fields,
)
from dptb.nnops.flow_diagnostics import (
    cfm_chord_cosine_diagnostics,
    cosine_similarity_tensors,
    grad_cosine,
    pixel_meanflow_du_dt_diagnostics,
)
from dptb.nnops.layout import project_uureal_to_like
from dptb.tests.flow_helpers import (
    BlockEndpointFallbackLoss,
    build_cfm,
    make_batch,
    two_graph_batch,
    two_graph_ref,
)


# ---------------------------------------------------------------------------
# prepare_batch / loss basics
# ---------------------------------------------------------------------------


def test_prepare_batch_samples_and_expands_time_per_graph():
    flow = HamiltonianCFM(
        {"enabled": True, "prior": "zero", "omit_time_scaling": True, "strict_h0": True}
    )
    flow._sample_t = lambda *, num_graphs, device, dtype: torch.tensor(
        [0.0, 0.5], device=device, dtype=dtype
    )

    data, ref, ctx = flow.prepare_batch(two_graph_batch(), two_graph_ref())

    assert ctx.t.shape == (2,)
    assert torch.equal(ctx.node_t, torch.tensor([0.0, 0.0, 0.5]))
    assert torch.equal(ctx.edge_t, torch.tensor([0.0, 0.5]))
    assert torch.equal(data["flow_time"], torch.tensor([0.0, 0.5]))
    assert torch.equal(ref["flow_time"], torch.tensor([0.0, 0.5]))
    assert torch.equal(data["node_h0"].flatten(), torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(data["edge_h0"].flatten(), torch.tensor([0.0, 2.0]))


def test_residual_flow_fails_fast_when_h0_is_missing():
    flow = HamiltonianCFM({"enabled": True, "mode": "residual", "strict_h0": True})
    data = two_graph_batch()
    data.pop("node_h0")

    with pytest.raises(KeyError, match="node_h0"):
        flow.prepare_batch(data, two_graph_ref())


def test_global_element_reduction_and_router_stats_are_flow_namespaced():
    """global_elements reduction weights by pooled element count (not a plain
    mean of the node/edge components), and router stats (mean_max_prob,
    expert_load_cv) pass through under train_flow_* only, not legacy keys."""
    flow = HamiltonianCFM(
        {"enabled": True, "omit_time_scaling": True, "component_reduction": "global_elements"}
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
    assert "train_onsite_loss" not in state
    assert "train_hopping_loss" not in state

    flow2 = HamiltonianCFM({"enabled": True, "omit_time_scaling": True})
    data2, ref2, ctx2 = flow2.prepare_batch(
        two_graph_batch(), two_graph_ref(), t=torch.zeros(2)
    )
    pred2 = {
        "batch": data2["batch"],
        "edge_index": data2["edge_index"],
        "node_features": ref2["node_features"] + 1.0,
        "edge_features": ref2["edge_features"] + 3.0,
        "mean_max_prob": torch.tensor(0.75),
        "expert_load_cv": torch.tensor(0.25),
    }
    _, state2 = flow2.loss(pred2, ref2, ctx2)
    assert state2["train_flow_onsite_loss"].item() == pytest.approx(1.0)
    assert state2["train_flow_hopping_loss"].item() == pytest.approx(9.0)
    assert "train_onsite_loss" not in state2
    assert "train_hopping_loss" not in state2
    assert state2["mean_max_prob"].item() == pytest.approx(0.75)
    assert state2["expert_load_cv"].item() == pytest.approx(0.25)


def test_flow_endpoint_stats_reject_mixed_node_edge_metric_spaces():
    state = {}
    HamiltonianCFM._merge_compatible_clean_stats(
        state, {"onsite_l1_sum": torch.tensor(1.0), "metric_space": "rme"}
    )

    with pytest.raises(ValueError, match="onsite and hopping targets"):
        HamiltonianCFM._merge_compatible_clean_stats(
            state, {"hopping_l1_sum": torch.tensor(1.0), "metric_space": "block"}
        )


# ---------------------------------------------------------------------------
# Endpoint metric-space guard (assert_model_in_loss_endpoint_metric_space)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flow,expect_error",
    [
        (
            HamiltonianPixelMeanFlow({"enabled": True, "objective": "pixel_meanflow"}),
            r"endpoint metric-space mismatch.*node_target_key='node_features'.*will not perform",
        ),
        (
            HamiltonianPixelMeanFlow(
                {
                    "enabled": True,
                    "objective": "pixel_meanflow",
                    "node_target_key": "node_full_hamil_blocks",
                    "edge_target_key": "edge_full_hamil_blocks",
                }
            ),
            None,
        ),
        (HamiltonianCFM({"enabled": True}), None),
    ],
    ids=[
        "pixel_meanflow_default_rme_keys_mismatch_the_block_criterion",
        "pixel_meanflow_explicit_block_target_contract_matches",
        "non_model_in_loss_cfm_does_not_use_the_guard",
    ],
)
def test_assert_model_in_loss_endpoint_metric_space(flow, expect_error):
    if expect_error:
        with pytest.raises(ValueError, match=expect_error):
            assert_model_in_loss_endpoint_metric_space(flow, BlockEndpointFallbackLoss())
    else:
        assert_model_in_loss_endpoint_metric_space(flow, BlockEndpointFallbackLoss())


# ---------------------------------------------------------------------------
# Samplers reach the constant/predicted endpoint
# ---------------------------------------------------------------------------


class _ConstantEndpoint(torch.nn.Module):
    def forward(self, data):
        data = data.copy()
        data["node_features"] = torch.full_like(data["node_h0"], 2.0)
        data["edge_features"] = torch.full_like(data["edge_h0"], 4.0)
        return data


class _ConstantEndpointWithBlocks(torch.nn.Module):
    """Block-native surrogate: emits feature keys plus Hamiltonian block keys,
    so pMF sampling can be checked to carry the model's full output surface."""

    def forward(self, data):
        data = data.copy()
        data["node_features"] = torch.full_like(data["node_h0"], 2.0)
        data["edge_features"] = torch.full_like(data["edge_h0"], 4.0)
        data["node_hamil_blocks"] = torch.full((data["node_h0"].shape[0], 2, 2), 2.0)
        data["edge_hamil_blocks"] = torch.full((data["edge_h0"].shape[0], 2, 2), 4.0)
        return data


@pytest.mark.parametrize("num_steps", [1, 3])
def test_cfm_euler_sampler_reaches_constant_predicted_endpoint(num_steps):
    flow = HamiltonianCFM(
        {"enabled": True, "prior": "zero", "omit_time_scaling": True, "strict_h0": True}
    )

    sampled = flow.sample(_ConstantEndpoint(), two_graph_batch(), num_steps=num_steps)

    assert torch.allclose(sampled["node_features"], torch.full((3, 1), 2.0))
    assert torch.allclose(sampled["edge_features"], torch.full((2, 1), 4.0))
    assert torch.equal(sampled["flow_time"], torch.ones(2))


def test_pixel_meanflow_one_step_sampler_reaches_constant_endpoint():
    flow = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow", "mode": "residual", "prior": "zero",
         "strict_h0": True}
    )

    sampled = flow.sample(_ConstantEndpoint(), two_graph_batch(), num_steps=1)

    assert torch.allclose(sampled["node_features"], torch.full((3, 1), 2.0))
    assert torch.allclose(sampled["edge_features"], torch.full((2, 1), 4.0))
    assert torch.equal(sampled["flow_time"], torch.zeros(2))
    assert torch.equal(sampled["flow_time_r"], torch.zeros(2))
    assert torch.equal(sampled["flow_time_h"], torch.zeros(2))


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

    flow = HamiltonianCFM({"enabled": True, "prior": "zero", "strict_h0": True})
    data = two_graph_batch()
    data["expert_node_mask"] = torch.ones(3, dtype=torch.bool)
    data["expert_edge_mask"] = torch.ones(2, dtype=torch.bool)
    data["expert_idx"] = 0
    model = _MaskCheckingModel()

    flow.sample(model, data, num_steps=3)

    assert model.calls == 3


@pytest.mark.parametrize("sample_final_forward", [True, False])
def test_pixel_meanflow_sample_final_forward_controls_block_outputs(sample_final_forward):
    options = {"enabled": True, "objective": "pixel_meanflow", "prior": "zero", "strict_h0": True}
    if not sample_final_forward:
        options["meanflow"] = {"sample_final_forward": False}
    flow = HamiltonianPixelMeanFlow(options)

    sampled = flow.sample(_ConstantEndpointWithBlocks(), two_graph_batch(), num_steps=1)

    # integrated endpoint features always win over the final forward's features
    assert torch.allclose(sampled["node_features"], torch.full((3, 1), 2.0))
    assert torch.allclose(sampled["edge_features"], torch.full((2, 1), 4.0))
    if sample_final_forward:
        # Regression: pMF sample() used to return state.copy() of the *input* data --
        # no model outputs at all -- so block-consuming losses KeyError'd on these.
        assert "node_hamil_blocks" in sampled and "edge_hamil_blocks" in sampled
        assert torch.allclose(sampled["node_hamil_blocks"], torch.full((3, 2, 2), 2.0))
        assert torch.equal(sampled["flow_time"], torch.zeros(2))
    else:
        assert "node_hamil_blocks" not in sampled


# ---------------------------------------------------------------------------
# build_hamiltonian_flow / HamiltonianPixelMeanFlow config accept-reject table
# ---------------------------------------------------------------------------


def _case_selects_pixel_meanflow_objective():
    flow = build_hamiltonian_flow(
        {"enabled": True, "objective": "pixel_meanflow", "mode": "residual", "prior": "zero"}
    )
    assert isinstance(flow, HamiltonianPixelMeanFlow)
    assert flow.model_in_loss is True


def _case_conservative_profile_defaults_to_boundary_tangent():
    flow = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow", "meanflow": {"profile": "conservative"}}
    )
    assert flow.meanflow_profile == "conservative"
    assert flow.meanflow_jvp_tangent == "boundary"
    assert flow.meanflow_norm_p == pytest.approx(0.0)
    assert flow.meanflow_aux_boundary_v_weight == pytest.approx(0.0)


def _case_aggressive_profile_sets_opt_in_knobs():
    flow = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow", "meanflow": {"profile": "aggressive"}}
    )
    assert flow.meanflow_profile == "aggressive"
    assert flow.meanflow_jvp_tangent == "boundary"
    assert flow.meanflow_norm_p == pytest.approx(1.0)
    assert flow.meanflow_aux_boundary_v_weight > 0.0


def _case_du_dt_backend_accepts_finite_difference_and_jvp():
    flow = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow",
         "meanflow": {"du_dt_backend": "finite_difference"}}
    )
    assert flow.meanflow_du_dt_backend == "finite_difference"
    jvp_flow = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow", "meanflow": {"du_dt_backend": "jvp"}}
    )
    assert jvp_flow.meanflow_du_dt_backend == "jvp"
    default_flow = HamiltonianPixelMeanFlow({"enabled": True, "objective": "pixel_meanflow"})
    assert default_flow.meanflow_du_dt_backend == "finite_difference"


def _case_du_dt_backend_rejects_unknown():
    HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow", "meanflow": {"du_dt_backend": "spectral"}}
    )


def _case_semigroup_objective_is_configurable():
    flow = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow", "meanflow": {"objective": "semigroup"}}
    )
    assert flow.meanflow_objective == "semigroup"
    assert flow.meanflow_semigroup_weight == pytest.approx(1.0)
    assert flow.meanflow_semigroup_endpoint_weight == pytest.approx(1.0)
    hybrid_flow = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow",
         "meanflow": {"objective": "hybrid", "semigroup_weight": 0.25}}
    )
    assert hybrid_flow.meanflow_objective == "hybrid"
    assert hybrid_flow.meanflow_semigroup_weight == pytest.approx(0.25)


def _case_semigroup_objective_rejects_unknown():
    HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow", "meanflow": {"objective": "bad"}}
    )


def _case_apply_to_reference_defaults_false_and_can_opt_in():
    default_flow = HamiltonianPixelMeanFlow({"enabled": True, "objective": "pixel_meanflow"})
    opt_in_flow = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow", "apply_to_reference": True}
    )
    assert default_flow.apply_to_reference is False
    assert opt_in_flow.apply_to_reference is True



def _case_validation_compatible_cannot_opt_out():
    flow = build_hamiltonian_flow(
        {"enabled": True, "objective": "pixel_meanflow",
         "meanflow": {"log_validation_compatible_loss": False}}
    )
    assert flow.log_validation_compatible_loss is True
    assert flow.compatible_loss_to_legacy_keys is True


def _case_train_compatible_alignment_is_forced_on():
    flow = HamiltonianPixelMeanFlow({"enabled": True, "objective": "pixel_meanflow"})
    opt_out = HamiltonianPixelMeanFlow(
        {"enabled": True, "objective": "pixel_meanflow",
         "meanflow": {"log_train_compatible_loss": False}}
    )
    assert flow.log_train_compatible_loss is True
    assert opt_out.log_train_compatible_loss is True
    assert opt_out.compatible_loss_to_legacy_keys is True


def _case_validation_ode_steps_always_include_euler_one_endpoint_baseline():
    flow = build_hamiltonian_flow(
        {"enabled": True, "objective": "pixel_meanflow", "validation_ode_steps": [3]}
    )
    assert flow.validation_ode_steps == (1, 3)


def _case_cfm_forces_compatible_clean_logging_even_when_config_disables_it():
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


# (build, expected ValueError match, or None for an accepted config)
_ACCEPT_REJECT_CASES = [
    ("selects_pixel_meanflow_objective", _case_selects_pixel_meanflow_objective, None),
    ("conservative_profile_defaults_to_boundary_tangent",
     _case_conservative_profile_defaults_to_boundary_tangent, None),
    ("aggressive_profile_sets_opt_in_knobs", _case_aggressive_profile_sets_opt_in_knobs, None),
    ("du_dt_backend_accepts_finite_difference_and_jvp",
     _case_du_dt_backend_accepts_finite_difference_and_jvp, None),
    ("du_dt_backend_rejects_unknown", _case_du_dt_backend_rejects_unknown, "du_dt_backend"),
    ("semigroup_objective_is_configurable", _case_semigroup_objective_is_configurable, None),
    ("semigroup_objective_rejects_unknown", _case_semigroup_objective_rejects_unknown,
     "meanflow.objective"),
    ("apply_to_reference_defaults_false_and_can_opt_in",
     _case_apply_to_reference_defaults_false_and_can_opt_in, None),
    ("validation_compatible_cannot_opt_out", _case_validation_compatible_cannot_opt_out, None),
    ("train_compatible_alignment_is_forced_on", _case_train_compatible_alignment_is_forced_on,
     None),
    ("validation_ode_steps_always_include_euler_one_endpoint_baseline",
     _case_validation_ode_steps_always_include_euler_one_endpoint_baseline, None),
    ("cfm_forces_compatible_clean_logging_even_when_config_disables_it",
     _case_cfm_forces_compatible_clean_logging_even_when_config_disables_it, None),
]


@pytest.mark.parametrize(
    "build,expect_error",
    [case[1:] for case in _ACCEPT_REJECT_CASES],
    ids=[case[0] for case in _ACCEPT_REJECT_CASES],
)
def test_build_hamiltonian_flow_accepted_and_rejected_configs(build, expect_error):
    if expect_error:
        with pytest.raises(ValueError, match=expect_error):
            build()
    else:
        build()


@pytest.mark.parametrize("objective", ["pixel_meanflow", "meanflow"])
def test_meanflow_objectives_default_validation_compatible_alignment(objective):
    flow = build_hamiltonian_flow({"enabled": True, "objective": objective})
    assert isinstance(flow, HamiltonianPixelMeanFlow)
    assert flow.log_validation_compatible_loss is True
    assert flow.compatible_loss_to_legacy_keys is True
    assert 1 in {int(n) for n in flow.validation_ode_steps}


# ---------------------------------------------------------------------------
# resolve_flow_log_fields: which scalar keys each objective registers
# ---------------------------------------------------------------------------


def _log_fields_pixel_meanflow_drops_never_computed_fields():
    flow = build_hamiltonian_flow({"enabled": True, "objective": "pixel_meanflow"})
    fields, register_legacy = resolve_flow_log_fields(flow)
    # pMF never computes the raw-batch train compatible loss.
    assert "train_compatible_loss" not in fields
    assert "train_compatible_onsite_loss" not in fields
    assert "train_compatible_hopping_loss" not in fields
    # pMF's validation branch never emits CFM's euler flow objective or t0 key.
    assert "validation_flow_t0_loss" not in fields
    assert "validation_flow_euler_1_loss" not in fields
    assert "validation_flow_euler_3_loss" not in fields
    assert "validation_flow_one_step_loss" in fields
    assert "validation_flow_random_t_loss" in fields
    # Euler-1 maps to the common validation triplet; only extra steps are logged.
    assert "validation_compatible_euler_1_loss" not in fields
    assert "validation_compatible_euler_3_hopping_loss" in fields
    assert "train_flow_du_dt_backend_jvp" in fields
    assert "train_flow_explicit_model_calls" in fields
    assert register_legacy is True


def _log_fields_includes_semigroup_meanflow_fields():
    flow = build_hamiltonian_flow(
        {"enabled": True, "objective": "pixel_meanflow", "meanflow": {"objective": "semigroup"}}
    )
    fields, register_legacy = resolve_flow_log_fields(flow)
    assert "train_flow_objective_semigroup" in fields
    assert "train_flow_semigroup_split_t" in fields
    assert "train_flow_onsite_semigroup_loss" in fields
    assert "train_flow_hopping_semigroup_loss" in fields
    assert register_legacy is True


def _log_fields_cfm_keeps_existing_fields():
    flow = build_hamiltonian_flow({"enabled": True, "objective": "cfm"})
    fields, register_legacy = resolve_flow_log_fields(flow)
    assert "train_compatible_loss" not in fields
    assert "validation_flow_t0_loss" in fields
    assert "validation_flow_euler_1_loss" in fields
    assert "validation_compatible_euler_1_loss" not in fields
    assert "validation_flow_one_step_loss" not in fields
    assert "train_flow_du_dt_backend_jvp" not in fields
    assert register_legacy is True


def _log_fields_uses_common_triplet_for_euler_one():
    flow = build_hamiltonian_flow(
        {"enabled": True, "objective": "pixel_meanflow",
         "meanflow": {"log_validation_compatible_loss": False}}
    )
    fields, register_legacy = resolve_flow_log_fields(flow)
    assert "validation_compatible_euler_1_loss" not in fields
    assert "validation_compatible_euler_1_onsite_loss" not in fields
    assert "validation_compatible_euler_1_hopping_loss" not in fields
    assert register_legacy is True


def _log_fields_disabled_flow_keeps_legacy_registration():
    flow = build_hamiltonian_flow({"enabled": False})
    fields, register_legacy = resolve_flow_log_fields(flow)
    assert fields == []
    assert register_legacy is True


_RESOLVE_LOG_FIELDS_CASES = [
    ("pixel_meanflow_drops_never_computed_fields",
     _log_fields_pixel_meanflow_drops_never_computed_fields),
    ("includes_semigroup_meanflow_fields", _log_fields_includes_semigroup_meanflow_fields),
    ("cfm_keeps_existing_fields", _log_fields_cfm_keeps_existing_fields),
    ("uses_common_triplet_for_euler_one", _log_fields_uses_common_triplet_for_euler_one),
    ("disabled_flow_keeps_legacy_registration",
     _log_fields_disabled_flow_keeps_legacy_registration),
]


@pytest.mark.parametrize(
    "run_case",
    [case for _, case in _RESOLVE_LOG_FIELDS_CASES],
    ids=[name for name, _ in _RESOLVE_LOG_FIELDS_CASES],
)
def test_resolve_flow_log_fields_matches_objective_specific_expectations(run_case):
    run_case()


# ---------------------------------------------------------------------------
# Zero-loss oracle: a model that already sits at the endpoint gives zero loss
# ---------------------------------------------------------------------------


def _oracle_finite_difference():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True, "objective": "pixel_meanflow", "mode": "residual", "prior": "zero",
            "strict_h0": True,
            "meanflow": {"aux_endpoint_weight": 0.0, "jvp_backend": "finite_difference",
                         "fd_eps": 1.0e-4},
        }
    )
    r = torch.tensor([0.2, 0.3])
    t = torch.tensor([0.5, 0.7])
    loss, state = flow.loss_with_model(_ConstantEndpoint(), two_graph_batch(), two_graph_ref(),
                                        r=r, t=t)
    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_h"].item() == pytest.approx(float((t - r).mean()), abs=1.0e-6)
    assert state["train_flow_onsite_velocity_mse"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_hopping_velocity_mse"].item() == pytest.approx(0.0, abs=1.0e-6)


def _oracle_semigroup():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True, "objective": "pixel_meanflow", "mode": "residual", "prior": "zero",
            "strict_h0": True,
            "meanflow": {"objective": "semigroup", "semigroup_endpoint_weight": 0.0},
        }
    )
    r = torch.tensor([0.2, 0.3])
    t = torch.tensor([0.5, 0.7])
    loss, state = flow.loss_with_model(_ConstantEndpoint(), two_graph_batch(), two_graph_ref(),
                                        r=r, t=t)
    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_objective_semigroup"].item() == pytest.approx(1.0)
    assert state["train_flow_onsite_semigroup_mse"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_hopping_semigroup_mse"].item() == pytest.approx(0.0, abs=1.0e-6)


def _oracle_jvp():
    # _ConstantEndpoint's output has no dependence on the state, so its forward
    # tangent is legitimately None -> opt out of the require-tangents guard.
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True, "objective": "pixel_meanflow", "mode": "residual", "prior": "zero",
            "strict_h0": True,
            "meanflow": {"du_dt_backend": "jvp", "aux_endpoint_weight": 0.0,
                         "jvp_require_tangents": False},
        }
    )
    r = torch.tensor([0.2, 0.3])
    t = torch.tensor([0.5, 0.7])
    loss, state = flow.loss_with_model(_ConstantEndpoint(), two_graph_batch(), two_graph_ref(),
                                        r=r, t=t)
    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_onsite_velocity_mse"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_hopping_velocity_mse"].item() == pytest.approx(0.0, abs=1.0e-6)


_ZERO_LOSS_ORACLE_CASES = [
    ("finite_difference_backend", _oracle_finite_difference),
    ("semigroup_objective", _oracle_semigroup),
    ("jvp_backend", _oracle_jvp),
]


@pytest.mark.parametrize(
    "run_case",
    [case for _, case in _ZERO_LOSS_ORACLE_CASES],
    ids=[name for name, _ in _ZERO_LOSS_ORACLE_CASES],
)
def test_pixel_meanflow_oracle_endpoint_has_zero_loss(run_case):
    run_case()


# ---------------------------------------------------------------------------
# Pixel MeanFlow JVP du/dt backend
# ---------------------------------------------------------------------------


def _jvp_flow_options(meanflow_overrides=None):
    meanflow = {"du_dt_backend": "jvp", "aux_endpoint_weight": 0.0}
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
    """x-prediction surrogate genuinely nonlinear in state and time: du/dt has
    both a state-transport term and an explicit time-conditioning term, which
    is exactly what the JVP backend must reproduce against finite differences.
    """

    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.w_node = torch.nn.Parameter(torch.tensor(0.7, dtype=dtype))
        self.w_edge = torch.nn.Parameter(torch.tensor(-0.4, dtype=dtype))
        self.a_node = torch.nn.Parameter(torch.tensor(0.5, dtype=dtype))
        self.a_edge = torch.nn.Parameter(torch.tensor(0.3, dtype=dtype))

    def forward(self, data):
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
            _jvp_flow_options(
                {"du_dt_backend": backend, "fd_eps": fd_eps, "jvp_tangent": jvp_tangent}
            ),
            dtype=torch.float64,
        )
        model = _TimeConditionedModel()
        loss, state = flow.loss_with_model(
            model, _double_batch(), _double_ref(), r=r.clone(), t=t.clone()
        )
        losses[backend] = float(loss.item())
        states[backend] = state

    assert losses["jvp"] == pytest.approx(losses["finite_difference"], rel=1.0e-4, abs=1.0e-8)
    for key in ("train_flow_onsite_velocity_mse", "train_flow_hopping_velocity_mse"):
        assert states["jvp"][key].item() == pytest.approx(
            states["finite_difference"][key].item(), rel=1.0e-4, abs=1.0e-8
        )


@pytest.mark.parametrize("jvp_tangent", ["boundary", "path"])
def test_pixel_meanflow_jvp_memory_efficient_matches_and_gives_finite_gradients(jvp_tangent):
    """Replaces 3 private call-order/grad-mode-sequence tests: jvp_memory_efficient
    is purely a compute/memory-order optimization, so True/False must give the
    same loss and gradients (both finite) for either tangent mode."""
    r = torch.tensor([0.2, 0.3], dtype=torch.float64)
    t = torch.tensor([0.55, 0.8], dtype=torch.float64)
    losses = {}
    for memory_efficient in (False, True):
        flow = HamiltonianPixelMeanFlow(
            _jvp_flow_options(
                {
                    "jvp_tangent": jvp_tangent,
                    "aux_boundary_v_weight": 0.2 if jvp_tangent == "boundary" else 0.0,
                    "jvp_memory_efficient": memory_efficient,
                }
            ),
            dtype=torch.float64,
        )
        model = _TimeConditionedModel()
        loss, state = flow.loss_with_model(
            model, _double_batch(), _double_ref(), r=r.clone(), t=t.clone()
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert model.w_node.grad is not None and torch.isfinite(model.w_node.grad)
        assert model.a_node.grad is not None and torch.isfinite(model.a_node.grad)
        assert state["train_flow_du_dt_backend_jvp"].item() == pytest.approx(1.0)
        losses[memory_efficient] = float(loss.item())
    assert losses[False] == pytest.approx(losses[True], rel=1.0e-6, abs=1.0e-9)


def test_configure_jvp_friendly_backends_switches_e3nn_only_for_jvp(monkeypatch):
    e3nn = pytest.importorskip("e3nn")
    calls = []
    monkeypatch.setattr(e3nn, "set_optimization_defaults", lambda **kw: calls.append(kw))

    cases = [
        ({"enabled": True, "objective": "pixel_meanflow", "meanflow": {"du_dt_backend": "jvp"}}, True),
        ({"enabled": True, "objective": "meanflow", "meanflow": {"jvp_backend": "jvp"}}, True),
        ({"enabled": True, "objective": "pixel_meanflow"}, False),
        ({"enabled": True, "objective": "cfm"}, False),
        ({"enabled": False, "objective": "pixel_meanflow", "meanflow": {"du_dt_backend": "jvp"}}, False),
        (None, False),
    ]
    for flow_options, expected in cases:
        calls.clear()
        assert configure_jvp_friendly_backends(flow_options) is expected
        if expected:
            assert calls == [{"jit_mode": "eager"}]
        else:
            assert calls == []


def test_pixel_meanflow_jvp_require_tangents_falls_back_not_silent_zero(caplog):
    # A dropped dual (None tangent) under the default guard must NOT be silently
    # treated as du/dt=0; it raises inside jvp and the run falls back to fd.
    flow = HamiltonianPixelMeanFlow(_jvp_flow_options({"jvp_require_tangents": True}))
    with caplog.at_level(logging.WARNING, logger="dptb.nnops.flow"):
        loss, state = flow.loss_with_model(
            _ConstantEndpoint(), two_graph_batch(), two_graph_ref(),
            r=torch.tensor([0.2, 0.3]), t=torch.tensor([0.5, 0.7]),
        )
    assert torch.isfinite(loss)
    assert state["train_flow_du_dt_backend_jvp"].item() == pytest.approx(0.0)
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_pixel_meanflow_jvp_falls_back_to_finite_difference_on_backend_failure(monkeypatch, caplog):
    import torch.autograd.forward_ad as fwAD

    flow = HamiltonianPixelMeanFlow(_jvp_flow_options(), dtype=torch.float64)
    calls = {"jvp": 0}

    def boom(*args, **kwargs):
        calls["jvp"] += 1
        raise RuntimeError("forward AD not implemented for fake op")

    # the jvp backend uses native torch.autograd.forward_ad, not functorch
    monkeypatch.setattr(fwAD, "make_dual", boom)

    model = _TimeConditionedModel()
    with caplog.at_level(logging.WARNING, logger="dptb.nnops.flow"):
        loss, state = flow.loss_with_model(
            model, _double_batch(), _double_ref(),
            r=torch.tensor([0.2, 0.3], dtype=torch.float64),
            t=torch.tensor([0.55, 0.8], dtype=torch.float64),
        )

    assert torch.isfinite(loss)
    assert calls["jvp"] == 1
    assert any(record.levelno >= logging.WARNING for record in caplog.records)
    assert state["train_flow_du_dt_backend_jvp"].item() == pytest.approx(0.0)

    # sticky fallback: the second step must not retry the broken jvp path
    flow.loss_with_model(
        _TimeConditionedModel(), _double_batch(), _double_ref(),
        r=torch.tensor([0.2, 0.3], dtype=torch.float64),
        t=torch.tensor([0.55, 0.8], dtype=torch.float64),
    )
    assert calls["jvp"] == 1


# ---------------------------------------------------------------------------
# Raw-uureal layout: node/edge features may arrive as the full raw uureal row
# rather than the compressed RME layout; the flow must project transparently.
# ---------------------------------------------------------------------------


class _CompactUuRealIDP:
    nextham_uureal_mask = True
    has_soc = True
    soc_complex_doubling = True

    def __init__(self):
        self.mask_uureal = torch.ones(5, dtype=torch.bool)
        self.orbpair_maps = {"1s-1s": slice(0, 2), "1s-2p": slice(2, 5)}

    def get_orbpair_maps(self):
        return self.orbpair_maps


class _CompactUuRealNoMaskIDP(_CompactUuRealIDP):
    def __init__(self):
        super().__init__()
        del self.mask_uureal


@pytest.mark.parametrize(
    "idp_cls", [_CompactUuRealIDP, _CompactUuRealNoMaskIDP],
    ids=["with_mask_uureal", "infers_full_soc_overlap_without_mask_uureal"],
)
def test_project_uureal_to_like_handles_full_soc_overlap_for_compact_idp(idp_cls):
    device = torch.device("cpu")
    dtype = torch.float32
    idp = idp_cls()
    like = torch.zeros(2, 5, device=device, dtype=dtype)
    raw = torch.zeros(2, 40, device=device, dtype=dtype)
    raw[:, 0:2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
    raw[:, 16:19] = torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], device=device, dtype=dtype)
    if idp_cls is _CompactUuRealIDP:
        # junk in the non-selected columns must be ignored either way
        raw[:, 2:16] = 100.0
        raw[:, 19:] = 200.0

    projected, mask = project_uureal_to_like(idp, raw, like)

    expected = torch.cat([raw[:, 0:2], raw[:, 16:19]], dim=-1)
    torch.testing.assert_close(projected, expected)
    assert mask is not None
    assert int(mask.sum().item()) == 5


def _cfm_context(node_target, *, t=None, node_t=None):
    dtype, device = node_target.dtype, node_target.device
    if t is None:
        t = torch.zeros(1, device=device, dtype=dtype)
    if node_t is None:
        node_t = torch.zeros(node_target.shape[0], device=device, dtype=dtype)
    return CFMContext(
        t=t, node_t=node_t, edge_t=None, node_base=None, edge_base=None,
        node_target=node_target, edge_target=None, node_current=None, edge_current=None,
        node_prior=None, edge_prior=None,
    )


@pytest.mark.parametrize(
    "mask_setup,expected_loss,expect_error",
    [
        ("uniform_raw_mask_both_sides", 0.25, None),
        ("compressed_mask_table", 2.5, None),
        ("mismatched_mask_width", None, "node idp mask layout"),
    ],
)
def test_flow_loss_projects_raw_uureal_predictions_to_compressed_targets(
    mask_setup, expected_loss, expect_error
):
    device = torch.device("cpu")
    dtype = torch.float64
    raw_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    flow = build_cfm("zero", device=device, dtype=dtype)

    if mask_setup == "uniform_raw_mask_both_sides":
        target = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
            device=device, dtype=dtype,
        )
        pred_raw = torch.zeros(3, raw_mask.numel(), device=device, dtype=dtype)
        pred_raw[:, raw_mask] = target + 0.5
        flow.idp.mask_uureal = raw_mask
        flow.idp.mask_to_nrme = raw_mask.expand(2, -1).clone()
        flow.idp.mask_to_erme = raw_mask.expand(2, -1).clone()
    elif mask_setup == "compressed_mask_table":
        compressed_mask = torch.tensor([1, 0, 1, 0], device=device, dtype=torch.bool)
        target = torch.zeros(3, 4, device=device, dtype=dtype)
        pred_raw = torch.zeros(3, raw_mask.numel(), device=device, dtype=dtype)
        pred_raw[:, raw_mask] = torch.tensor([1.0, 10.0, 2.0, 10.0], device=device, dtype=dtype)
        flow.idp.mask_uureal = raw_mask
        flow.idp.mask_to_nrme = compressed_mask.expand(2, -1).clone()
    else:
        target = torch.zeros(3, 4, device=device, dtype=dtype)
        pred_raw = torch.ones_like(target)
        flow.idp.mask_uureal = torch.tensor(
            [1, 1, 0, 1, 0, 1, 0], device=device, dtype=torch.bool
        )
        flow.idp.mask_to_nrme = torch.ones((2, 8), device=device, dtype=torch.bool)

    pred_data = {
        _keys.NODE_FEATURES_KEY: pred_raw,
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1, 0], device=device, dtype=torch.long),
    }
    ref_data = {_keys.NODE_FEATURES_KEY: target}
    ctx = _cfm_context(target)

    if expect_error:
        with pytest.raises(ValueError, match=expect_error):
            flow.loss(pred_data, ref_data, ctx)
        return
    loss, state = flow.loss(pred_data, ref_data, ctx)
    expected = torch.full((), expected_loss, device=device, dtype=dtype)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(state["train_flow_onsite_loss"], expected)


def _raw_uureal_endpoint(raw_mask, node_target, edge_target):
    class _Endpoint(torch.nn.Module):
        def forward(self, batch):
            out = batch.copy()
            node_raw = torch.zeros(
                batch[_keys.NODE_H0_KEY].shape[0], raw_mask.numel(),
                device=batch[_keys.NODE_H0_KEY].device, dtype=batch[_keys.NODE_H0_KEY].dtype,
            )
            edge_raw = torch.zeros(
                batch[_keys.EDGE_H0_KEY].shape[0], raw_mask.numel(),
                device=batch[_keys.EDGE_H0_KEY].device, dtype=batch[_keys.EDGE_H0_KEY].dtype,
            )
            node_raw[:, raw_mask] = node_target.to(node_raw)
            edge_raw[:, raw_mask] = edge_target.to(edge_raw)
            out[_keys.NODE_FEATURES_KEY] = node_raw
            out[_keys.EDGE_FEATURES_KEY] = edge_raw
            return out

    return _Endpoint()


def test_flow_sample_projects_raw_uureal_endpoint_to_compressed_state():
    device = torch.device("cpu")
    dtype = torch.float64
    raw_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    data, ref = make_batch(device=device, dtype=dtype)
    flow = build_cfm("zero", device=device, dtype=dtype)
    flow.idp.mask_uureal = raw_mask

    sampled = flow.sample(
        _raw_uureal_endpoint(raw_mask, ref[_keys.NODE_FEATURES_KEY], ref[_keys.EDGE_FEATURES_KEY]),
        data,
        num_steps=1,
    )

    assert sampled[_keys.NODE_FEATURES_KEY].shape == ref[_keys.NODE_FEATURES_KEY].shape
    assert sampled[_keys.EDGE_FEATURES_KEY].shape == ref[_keys.EDGE_FEATURES_KEY].shape
    torch.testing.assert_close(sampled[_keys.NODE_FEATURES_KEY], ref[_keys.NODE_FEATURES_KEY])
    torch.testing.assert_close(sampled[_keys.EDGE_FEATURES_KEY], ref[_keys.EDGE_FEATURES_KEY])


def test_pixel_meanflow_projects_raw_uureal_endpoint_to_compressed_loss_layout():
    device = torch.device("cpu")
    dtype = torch.float64
    raw_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    data, ref = make_batch(device=device, dtype=dtype)
    from dptb.tests.flow_helpers import FakeIDP

    idp = FakeIDP(device=device)
    idp.mask_uureal = raw_mask
    idp.mask_to_nrme = raw_mask.expand(2, -1).clone()
    idp.mask_to_erme = raw_mask.expand(2, -1).clone()
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True, "objective": "pixel_meanflow", "mode": "residual", "prior": "zero",
            "strict_h0": True,
            "meanflow": {"aux_endpoint_weight": 0.0, "jvp_tangent": "path"},
        },
        idp=idp, device=device, dtype=dtype,
    )

    loss, state = flow.loss_with_model(
        _raw_uureal_endpoint(raw_mask, ref[_keys.NODE_FEATURES_KEY], ref[_keys.EDGE_FEATURES_KEY]),
        data, ref,
        r=torch.tensor([0.25, 0.25], device=device, dtype=dtype),
        t=torch.tensor([0.50, 0.50], device=device, dtype=dtype),
    )

    assert torch.isfinite(loss)
    assert state["train_flow_onsite_endpoint_loss"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert state["train_flow_hopping_endpoint_loss"].item() == pytest.approx(0.0, abs=1.0e-8)


# ---------------------------------------------------------------------------
# Per-graph time embedding (dptb.nn.embedding.flow_time)
# ---------------------------------------------------------------------------


def test_sinusoidal_time_embedding_matches_qhflow2_shape_and_is_time_sensitive():
    t = torch.tensor([0.0, 0.5])
    emb = sinusoidal_time_embedding(t, embedding_dim=4, max_positions=2000)

    assert emb.shape == (2, 4)
    assert not torch.allclose(emb[0], emb[1])


def test_conditioner_maps_per_graph_time_to_nodes_and_only_changes_scalar_channels():
    conditioner = FlowTimeConditioner(
        scalar_channels=4, flow_time_key="flow_time", max_positions=2000
    )
    node_features = torch.zeros(3, 9)
    data = {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "flow_time": torch.tensor([0.0, 0.5]),
    }

    conditioned = conditioner(node_features, data)

    assert conditioned.shape == node_features.shape
    assert torch.allclose(conditioned[0], conditioned[1])
    assert not torch.allclose(conditioned[0], conditioned[2])
    assert torch.count_nonzero(conditioned[:, 4:]) == 0


def test_conditioner_requires_one_time_per_graph():
    conditioner = FlowTimeConditioner(scalar_channels=4)
    data = {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "flow_time": torch.tensor([0.25]),
    }

    with pytest.raises(ValueError, match="one value per graph"):
        conditioner(torch.zeros(3, 4), data)


def test_conditioner_can_use_two_time_meanflow_channels():
    conditioner = FlowTimeConditioner(
        scalar_channels=4,
        flow_time_key="flow_time",
        flow_time_keys=("flow_time_t", "flow_time_r", "flow_time_h"),
        max_positions=2000,
    )
    node_features = torch.zeros(2, 4)
    base = {
        "batch": torch.tensor([0, 1], dtype=torch.long),
        "flow_time": torch.tensor([0.8, 0.8]),
        "flow_time_t": torch.tensor([0.8, 0.8]),
        "flow_time_r": torch.tensor([0.2, 0.6]),
        "flow_time_h": torch.tensor([0.6, 0.2]),
    }

    conditioned = conditioner(node_features, base)

    assert conditioned.shape == node_features.shape
    assert not torch.allclose(conditioned[0], conditioned[1])


def test_conditioner_maps_explicit_edge_batch_to_edge_rows():
    conditioner = FlowTimeConditioner(
        scalar_channels=4, flow_time_key="flow_time", max_positions=2000
    )
    edge_features = torch.zeros(4, 9)
    data = {"flow_time": torch.tensor([0.0, 0.5])}
    edge_batch = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    conditioned = conditioner(edge_features, data, batch=edge_batch)

    assert conditioned.shape == edge_features.shape
    assert torch.allclose(conditioned[0], conditioned[3])
    assert torch.allclose(conditioned[1], conditioned[2])
    assert not torch.allclose(conditioned[0], conditioned[1])
    assert torch.count_nonzero(conditioned[:, 4:]) == 0


@pytest.mark.parametrize(
    "edge_batch",
    [
        torch.tensor([0, 2], dtype=torch.long),  # the middle graph has no edges
        torch.tensor([0, 1], dtype=torch.long),  # the trailing graph has no edges
    ],
    ids=["middle_graph_has_no_edges", "trailing_graphs_have_no_edges"],
)
def test_conditioner_accepts_edge_batch_with_graphs_that_have_no_edges(edge_batch):
    conditioner = FlowTimeConditioner(
        scalar_channels=4, flow_time_key="flow_time", max_positions=2000
    )
    edge_features = torch.zeros(2, 4)
    data = {"flow_time": torch.tensor([0.0, 0.5, 0.9])}

    conditioned = conditioner(edge_features, data, batch=edge_batch)

    assert conditioned.shape == edge_features.shape
    assert not torch.allclose(conditioned[0], conditioned[1])


# ---------------------------------------------------------------------------
# Flow geometry diagnostics (dptb.nnops.flow_diagnostics)
# ---------------------------------------------------------------------------


def test_cosine_similarity_tensors_flattens_complex_as_real_components():
    a = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])
    b = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])

    assert cosine_similarity_tensors(a, b).item() == pytest.approx(1.0)


def test_pixel_meanflow_du_dt_diagnostics_reports_norm_ratio_and_grad_cosine():
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    flow_loss = (param.square()).sum()
    jvp_loss = -(param.square()).sum()

    state = pixel_meanflow_du_dt_diagnostics(
        target_v=torch.tensor([3.0, 4.0]),
        du_dt=torch.tensor([0.0, 5.0]),
        flow_loss=flow_loss,
        jvp_loss=jvp_loss,
        parameters=[param],
    )

    assert state["du_dt_norm"].item() == pytest.approx(5.0)
    assert state["target_v_norm"].item() == pytest.approx(5.0)
    assert state["du_dt_norm_over_target_v_norm"].item() == pytest.approx(1.0)
    assert state["grad_cos_flow_jvp"].item() == pytest.approx(-1.0)


def test_grad_cosine_returns_nan_for_missing_gradients():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    first = torch.tensor(1.0, requires_grad=True)
    second = torch.tensor(2.0, requires_grad=True)

    assert torch.isnan(grad_cosine(first, second, [param]))


def test_cfm_chord_cosine_diagnostics_is_one_for_exact_endpoint_direction():
    current = torch.zeros(2, 2)
    target = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    endpoint = target.clone()
    t = torch.tensor([0.25, 0.5])

    state = cfm_chord_cosine_diagnostics(
        node_current=current, node_target=target, node_endpoint=endpoint, t=t
    )

    assert state["node_cos_v_theta_chord"].item() == pytest.approx(1.0)
    assert state["cos_v_theta_chord"].item() == pytest.approx(1.0)


def test_cfm_chord_cosine_diagnostics_aggregates_node_and_edge_components():
    node_current = torch.zeros(1, 2)
    node_target = torch.tensor([[1.0, 0.0]])
    node_endpoint = torch.tensor([[0.0, 1.0]])
    edge_current = torch.zeros(1, 2)
    edge_target = torch.tensor([[0.0, 2.0]])
    edge_endpoint = torch.tensor([[0.0, 4.0]])

    state = cfm_chord_cosine_diagnostics(
        node_current=node_current, node_target=node_target, node_endpoint=node_endpoint,
        edge_current=edge_current, edge_target=edge_target, edge_endpoint=edge_endpoint,
        t=torch.tensor([0.25]),
    )

    assert state["node_cos_v_theta_chord"].item() == pytest.approx(0.0)
    assert state["edge_cos_v_theta_chord"].item() == pytest.approx(1.0)
    assert state["cos_v_theta_chord"].item() == pytest.approx(
        8.0 / torch.sqrt(torch.tensor(85.0)).item()
    )

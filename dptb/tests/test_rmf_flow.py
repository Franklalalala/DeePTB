from __future__ import annotations

import torch
import pytest

from dptb.data import _keys
from dptb.nnops.flow import HamiltonianCFM, HamiltonianRiemannianMeanFlow, build_hamiltonian_flow
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.nnops.rmf_manifold import EuclideanManifold, build_rmf_manifold


def _toy_batch():
    node_h0 = torch.tensor([[1.0, -1.0], [0.5, 0.0]], dtype=torch.float32)
    edge_h0 = torch.tensor([[0.1, 0.2], [-0.2, 0.3]], dtype=torch.float32)
    node_target = node_h0 + torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=torch.float32)
    edge_target = edge_h0 + torch.tensor([[0.5, -0.1], [-0.3, 0.2]], dtype=torch.float32)
    data = {
        _keys.BATCH_KEY: torch.zeros(node_h0.shape[0], dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.NODE_H0_KEY: node_h0.clone(),
        _keys.EDGE_H0_KEY: edge_h0.clone(),
        _keys.NODE_FEATURES_KEY: node_h0.clone(),
        _keys.EDGE_FEATURES_KEY: edge_h0.clone(),
    }
    ref = {
        _keys.BATCH_KEY: data[_keys.BATCH_KEY],
        _keys.EDGE_INDEX_KEY: data[_keys.EDGE_INDEX_KEY],
        _keys.NODE_FEATURES_KEY: node_target.clone(),
        _keys.EDGE_FEATURES_KEY: edge_target.clone(),
    }
    return data, ref


def _rmf_options():
    return {
        "enabled": True,
        "type": "rmf",
        "objective": "rmf",
        "mode": "residual",
        "prior": "zero",
        "manifold": "euclidean",
        "meanflow": {
            "time_sampling": "uniform",
            "data_proportion": 0.0,
            "tr_uniform_prob": 0.0,
            "fd_eps": 1.0e-3,
            "aux_endpoint_weight": 0.05,
            "norm_p": 0.0,
        },
        "rmf_options": {"endpoint_eps": 1.0e-3, "manifold": "euclidean"},
    }


def _empty_multitrainer_for_pack():
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.log_single_model_compatible_loss = True
    trainer.log_single_model_compatible_loss_mode = "reduce"
    trainer.train_lossfunc = object()
    return trainer


def test_euclidean_rmf_manifold_ops_shape_and_values():
    manifold = build_rmf_manifold("euclidean")
    assert isinstance(manifold, EuclideanManifold)
    x0 = torch.randn(4, 3, 2)
    x1 = torch.randn(4, 3, 2)
    t = torch.linspace(0.0, 1.0, 4)

    assert manifold.project(x0, x1).shape == x0.shape
    assert torch.allclose(manifold.expmap(x0, x1 - x0), x1, atol=1.0e-6)
    assert torch.allclose(manifold.logmap(x0, x1), x1 - x0)
    interp = manifold.geodesic_interpolate(x0, x1, t)
    expected = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    assert interp.shape == x0.shape
    assert torch.allclose(interp, expected)
    assert torch.allclose(manifold.tangent_velocity(x0, x1, t), x1 - x0)


def test_rmf_prepare_batch_uses_h0_to_geodesic_residual_path():
    data, ref = _toy_batch()
    flow = HamiltonianRiemannianMeanFlow(_rmf_options())
    source_r = torch.tensor([0.25])
    target_t = torch.tensor([0.75])
    batch, _, ctx = flow.prepare_batch(data, ref, r=source_r, t=target_t)

    node_residual = ref[_keys.NODE_FEATURES_KEY] - data[_keys.NODE_H0_KEY]
    edge_residual = ref[_keys.EDGE_FEATURES_KEY] - data[_keys.EDGE_H0_KEY]
    assert torch.allclose(batch[_keys.NODE_H0_KEY], data[_keys.NODE_H0_KEY] + source_r * node_residual)
    assert torch.allclose(batch[_keys.EDGE_H0_KEY], data[_keys.EDGE_H0_KEY] + source_r * edge_residual)
    assert torch.allclose(ctx.node_velocity, node_residual)
    assert torch.allclose(ctx.edge_velocity, edge_residual)
    assert torch.allclose(batch["flow_time_r"], source_r)
    assert torch.allclose(batch["flow_time_t"], target_t)
    assert torch.allclose(batch["flow_time_h"], target_t - source_r)


def test_rmf_flow_matching_samples_remain_equal_after_endpoint_clamp(monkeypatch):
    flow = HamiltonianRiemannianMeanFlow(_rmf_options())
    flow.meanflow_data_proportion = 1.0
    monkeypatch.setattr(
        flow,
        "_sample_time_base",
        lambda num_graphs, device, dtype: torch.ones(num_graphs, device=device, dtype=dtype),
    )

    r, t, fm_mask = flow._sample_st(
        num_graphs=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert fm_mask.all()
    assert torch.allclose(r, t)
    assert torch.allclose(r, torch.full_like(r, 1.0 - flow.rmf_endpoint_eps))


def test_rmf_rejects_unimplemented_semigroup_objective():
    options = _rmf_options()
    options["rmf_options"] = dict(options["rmf_options"], objective="semigroup")

    with pytest.raises(NotImplementedError, match="source_time_fd"):
        HamiltonianRiemannianMeanFlow(options)


def test_rmf_endpoint_loss_ignores_inactive_distance_expert_rows():
    flow = HamiltonianRiemannianMeanFlow(_rmf_options())
    flow.meanflow_aux_endpoint_weight = 0.0
    clean = torch.zeros(4, 1)
    prior = torch.zeros_like(clean)
    state_z = torch.zeros_like(clean)
    target_v = torch.zeros_like(clean)
    pred_x = torch.tensor([[1.0], [3.0], [100.0], [100.0]])
    mask = torch.tensor([[True], [True], [False], [False]])

    _, state = flow._component_rmf_loss(
        diff_prefix="train_flow_hopping",
        pred_x=pred_x,
        clean=clean,
        prior=prior,
        state_z=state_z,
        target_v=target_v,
        comp_r=torch.full((4,), 0.25),
        comp_t=torch.full((4,), 0.25),
        pred_x_eps=pred_x,
        mask=mask,
        weight=1.0,
    )

    assert state["train_flow_hopping_endpoint_loss"].item() == pytest.approx(5.0)
    assert state["train_flow_hopping_endpoint_mse"].item() == pytest.approx(5.0)
    assert state["train_flow_hopping_endpoint_mae"].item() == pytest.approx(2.0)
    assert state["train_flow_hopping_endpoint_l1_sum"].item() == pytest.approx(4.0)
    assert state["train_flow_hopping_endpoint_mse_sum"].item() == pytest.approx(10.0)
    assert state["train_flow_hopping_endpoint_count"].item() == pytest.approx(2.0)


def test_multitrainer_flow_metrics_keep_endpoint_counts_for_validation_pack():
    trainer = _empty_multitrainer_for_pack()
    flow_state = {
        "validation_flow_onsite_endpoint_loss": torch.tensor(5.0),
        "validation_flow_hopping_endpoint_loss": torch.tensor(13.0),
        "validation_flow_onsite_endpoint_l1_sum": torch.tensor(4.0),
        "validation_flow_onsite_endpoint_mse_sum": torch.tensor(10.0),
        "validation_flow_onsite_endpoint_count": torch.tensor(2.0),
        "validation_flow_hopping_endpoint_l1_sum": torch.tensor(8.0),
        "validation_flow_hopping_endpoint_mse_sum": torch.tensor(20.0),
        "validation_flow_hopping_endpoint_count": torch.tensor(4.0),
    }

    metrics = trainer._snapshot_flow_metrics(flow_state, "validation")
    payload = {
        "loss_detached": torch.tensor(99.0),
        "onsite_weighted_sum": torch.tensor(0.0),
        "hopping_weighted_sum": torch.tensor(0.0),
        "active_nodes": torch.tensor(1.0),
        "active_edges": torch.tensor(1.0),
        "grad_norm": torch.tensor(0.0),
        "z_values": [],
        "load_cv_values": [],
        **metrics,
    }
    pack = trainer._make_step_pack(payload)

    assert metrics["last_onsite_count"].item() == pytest.approx(2.0)
    assert metrics["last_hopping_count"].item() == pytest.approx(4.0)
    assert trainer._compute_compatible_loss_from_pack(pack, object()) is not None


def test_multitrainer_validation_component_state_uses_flow_weighted_fallback_without_counts():
    trainer = _empty_multitrainer_for_pack()
    pack = torch.zeros((trainer._PACK_LEN,), dtype=trainer.dtype)
    pack[trainer._P_LOSS_OPT_SUM] = 7.0
    pack[trainer._P_STEP_COUNT] = 1.0
    pack[trainer._P_ONSITE_WEIGHTED_SUM] = 20.0
    pack[trainer._P_ACTIVE_NODES_SUM] = 4.0
    pack[trainer._P_HOPPING_WEIGHTED_SUM] = 18.0
    pack[trainer._P_ACTIVE_EDGES_SUM] = 6.0

    state = trainer._validation_component_state_from_pack(
        pack,
        loss=torch.tensor(7.0),
    )

    assert state["validation_loss"].item() == pytest.approx(7.0)
    assert state["validation_onsite_loss"].item() == pytest.approx(5.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(3.0)


def test_multitrainer_compatible_pack_skips_zero_count_components():
    trainer = _empty_multitrainer_for_pack()
    pack = torch.zeros((trainer._PACK_LEN,), dtype=trainer.dtype)
    pack[trainer._P_ONSITE_L1_SUM] = 999.0
    pack[trainer._P_ONSITE_MSE_SUM] = 999.0
    pack[trainer._P_ONSITE_CNT_SUM] = 0.0
    pack[trainer._P_HOPPING_L1_SUM] = 4.0
    pack[trainer._P_HOPPING_MSE_SUM] = 10.0
    pack[trainer._P_HOPPING_CNT_SUM] = 2.0

    loss = trainer._compute_compatible_loss_from_pack(pack, object())

    expected_hopping = 0.5 * (torch.tensor(2.0) + torch.sqrt(torch.tensor(5.0)))
    assert loss.item() == pytest.approx(expected_hopping.item())


def test_multitrainer_compatible_payload_skips_zero_count_components():
    trainer = _empty_multitrainer_for_pack()
    payload = {
        "onsite_l1_sum": torch.tensor(999.0),
        "onsite_mse_sum": torch.tensor(999.0),
        "onsite_cnt": torch.tensor(0.0),
        "hopping_l1_sum": torch.tensor(4.0),
        "hopping_mse_sum": torch.tensor(10.0),
        "hopping_cnt": torch.tensor(2.0),
        "z_values": [],
    }

    loss = trainer._compute_stitched_loss_by_reduce([payload], object())

    expected_hopping = 0.5 * (torch.tensor(2.0) + torch.sqrt(torch.tensor(5.0)))
    assert loss.item() == pytest.approx(expected_hopping.item())


class _EndpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, data):
        out = data.copy()
        if _keys.NODE_H0_KEY in data:
            out[_keys.NODE_FEATURES_KEY] = data[_keys.NODE_H0_KEY] + self.scale * torch.ones_like(
                data[_keys.NODE_H0_KEY]
            )
        if _keys.EDGE_H0_KEY in data:
            out[_keys.EDGE_FEATURES_KEY] = data[_keys.EDGE_H0_KEY] + self.scale * torch.ones_like(
                data[_keys.EDGE_H0_KEY]
            )
        out["mean_max_prob"] = self.scale.square()
        out["expert_load_cv"] = self.scale.abs()
        return out


class _OracleEndpointModel(torch.nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.node_target = ref[_keys.NODE_FEATURES_KEY]
        self.edge_target = ref[_keys.EDGE_FEATURES_KEY]

    def forward(self, data):
        out = data.copy()
        out[_keys.NODE_FEATURES_KEY] = self.node_target.to(data[_keys.NODE_H0_KEY])
        out[_keys.EDGE_FEATURES_KEY] = self.edge_target.to(data[_keys.EDGE_H0_KEY])
        return out


def test_rmf_forward_loss_backward_and_legacy_train_tags():
    data, ref = _toy_batch()
    flow = HamiltonianRiemannianMeanFlow(_rmf_options())
    model = _EndpointModel()
    loss, state = flow.loss_with_model(
        model,
        data,
        ref,
        r=torch.tensor([0.10]),
        t=torch.tensor([0.40]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)
    for key in (
        "train_flow_onsite_loss",
        "train_flow_hopping_loss",
        "train_onsite_loss",
        "train_hopping_loss",
        "mean_max_prob",
        "expert_load_cv",
    ):
        assert key in state
    assert state["train_onsite_loss"].item() == state["train_flow_onsite_endpoint_loss"].item()
    assert state["train_hopping_loss"].item() == state["train_flow_hopping_endpoint_loss"].item()


def test_rmf_forward_source_time_identity_uses_negative_jvp_term():
    flow = HamiltonianRiemannianMeanFlow(_rmf_options())
    flow.meanflow_aux_endpoint_weight = 0.0
    r = torch.tensor([0.25])
    t = torch.tensor([0.75])
    state = torch.tensor([[0.25]])
    clean = torch.tensor([[1.0]])
    prior = torch.zeros_like(clean)
    target_v = torch.ones_like(clean)
    u = torch.tensor([[2.0]])
    du_dr = torch.tensor([[2.0]])
    pred_x = state + (1.0 - r[:, None]) * u
    r_eps = r + flow.meanflow_fd_eps
    state_eps = r_eps[:, None]
    pred_x_eps = state_eps + (1.0 - r_eps[:, None]) * (
        u + flow.meanflow_fd_eps * du_dr
    )

    loss, _ = flow._component_rmf_loss(
        diff_prefix="train_flow_onsite",
        pred_x=pred_x,
        clean=clean,
        prior=prior,
        state_z=state,
        target_v=target_v,
        comp_r=r,
        comp_t=t,
        pred_x_eps=pred_x_eps,
        mask=torch.ones_like(state, dtype=torch.bool),
        weight=1.0,
    )

    assert loss.item() < 1.0e-8


def test_rmf_oracle_endpoint_has_zero_loss_at_backward_fd_boundary():
    data, ref = _toy_batch()
    flow = HamiltonianRiemannianMeanFlow(_rmf_options())

    loss, state = flow.loss_with_model(
        _OracleEndpointModel(ref),
        data,
        ref,
        r=torch.tensor([1.0 - flow.rmf_endpoint_eps]),
        t=torch.tensor([1.0]),
    )

    assert loss.item() < 1.0e-8
    assert state["train_flow_onsite_velocity_mse"].item() < 1.0e-8
    assert state["train_flow_hopping_velocity_mse"].item() < 1.0e-8


def test_rmf_one_step_sampler_reaches_oracle_endpoint():
    data, ref = _toy_batch()
    flow = HamiltonianRiemannianMeanFlow(_rmf_options())

    sampled = flow.sample(_OracleEndpointModel(ref), data, num_steps=1)

    assert torch.allclose(sampled[_keys.NODE_FEATURES_KEY], ref[_keys.NODE_FEATURES_KEY])
    assert torch.allclose(sampled[_keys.EDGE_FEATURES_KEY], ref[_keys.EDGE_FEATURES_KEY])
    assert torch.equal(sampled["flow_time_r"], torch.ones(1))
    assert torch.equal(sampled["flow_time_t"], torch.ones(1))


def test_rmf_is_opt_in_and_default_cfm_is_unchanged():
    default_flow = build_hamiltonian_flow({"enabled": False})
    assert isinstance(default_flow, HamiltonianCFM)
    assert not isinstance(default_flow, HamiltonianRiemannianMeanFlow)
    rmf_flow = build_hamiltonian_flow({"enabled": True, "type": "rmf", "mode": "residual"})
    assert isinstance(rmf_flow, HamiltonianRiemannianMeanFlow)
    cfm_flow = build_hamiltonian_flow({"enabled": True, "objective": "cfm"})
    assert isinstance(cfm_flow, HamiltonianCFM)
    assert not isinstance(cfm_flow, HamiltonianRiemannianMeanFlow)


def test_multitrainer_distance_expert_path_calls_model_in_loss_flow():
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.iter = 3
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.model = object()
    trainer._tagger = type(
        "Tagger",
        (),
        {"tag": lambda self, *args, **kwargs: __import__("contextlib").nullcontext()},
    )()
    trainer._prepare_expert_masks = lambda batch, distance_range, expert_idx: (
        torch.tensor([True, False]),
        torch.tensor([True, True]),
    )

    class _Flow:
        enabled = True
        model_in_loss = True
        apply_to_reference = False

        def loss_with_model(self, model, data, ref_data, prefix):
            assert model is trainer.model
            assert data["expert_idx"] == 1
            assert torch.equal(data["expert_edge_mask"], torch.tensor([True, False]))
            assert torch.equal(data["expert_node_mask"], torch.tensor([True, True]))
            return torch.tensor(2.0, requires_grad=True), {
                f"{prefix}_flow_onsite_endpoint_loss": torch.tensor(3.0),
                f"{prefix}_flow_hopping_endpoint_loss": torch.tensor(4.0),
                "mean_max_prob": torch.tensor(0.6),
                "expert_load_cv": torch.tensor(0.2),
            }

    trainer.flow_cfm = _Flow()

    result = trainer._run_one_expert_loss(
        batch_dict={},
        batch_info={},
        criterion=lambda *args: (_ for _ in ()).throw(AssertionError("legacy loss called")),
        expert_idx=1,
        range_dis=(1.0, 2.0),
        capture_metrics=True,
    )

    assert result["loss"].item() == 2.0
    assert result["onsite"].item() == 3.0
    assert result["hopping"].item() == 4.0
    assert result["z_loss"].item() == pytest.approx(0.6)
    assert result["expert_load_cv"].item() == pytest.approx(0.2)


def test_multitrainer_reference_batch_keeps_legacy_loss_by_default():
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.iter = 3
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer._tagger = type(
        "Tagger",
        (),
        {"tag": lambda self, *args, **kwargs: __import__("contextlib").nullcontext()},
    )()
    trainer._prepare_expert_masks = lambda batch, distance_range, expert_idx: (
        torch.tensor([True]),
        torch.tensor([True, True]),
    )

    class _Flow:
        enabled = True
        model_in_loss = True
        apply_to_reference = False

        def __init__(self):
            self.calls = 0

        def loss_with_model(self, model, data, ref_data, prefix):
            self.calls += 1
            return torch.tensor(2.0, requires_grad=True), {
                "train_flow_onsite_endpoint_loss": torch.tensor(3.0),
                "train_flow_hopping_endpoint_loss": torch.tensor(4.0),
            }

    class _Model:
        def __init__(self):
            self.calls = 0

        def __call__(self, data):
            self.calls += 1
            return data.copy()

    class _Criterion:
        last_onsite_loss = torch.tensor(5.0)
        last_hopping_loss = torch.tensor(6.0)
        last_z_loss = None
        expert_load_cv = None

        def __call__(self, pred, ref):
            return torch.tensor(7.0, requires_grad=True)

    trainer.flow_cfm = _Flow()
    trainer.model = _Model()

    payload = trainer._build_train_payload(
        batch_dict={},
        batch_info={},
        expert_idx=0,
        range_dis=(0.0, 1.0),
        ref_batch_dict={},
        ref_batch_info={},
        criterion=_Criterion(),
    )

    assert trainer.flow_cfm.calls == 1
    assert trainer.model.calls == 1
    assert payload["loss"].item() == pytest.approx(9.0)
    assert payload["expert_onsite"].item() == pytest.approx((3.0 * 2.0 + 5.0 * 2.0) / 4.0)
    assert payload["expert_hopping"].item() == pytest.approx((4.0 + 6.0) / 2.0)

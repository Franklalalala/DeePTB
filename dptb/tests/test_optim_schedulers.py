"""LR schedulers (WSD / warmup_rop / qhflow_poly) and the HybridMuon optimizer:
values against the closed-form schedule, state_dict restore, param routing by
shape/name, clipping and diagnostics, and old-checkpoint compatibility.
"""
import pytest
import torch

from dptb.utils.argcheck import chk_avg_per_iter, train_options
from dptb.utils.tools import get_lr_scheduler, get_optimizer, lr_scheduler_can_step_without_metric


# ---------------------------------------------------------------------------
# LR scheduler values against the closed-form schedule
# ---------------------------------------------------------------------------

def test_wsd_scheduler_matches_dpa4_warmup_stable_decay_formula():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0e-3)

    scheduler = get_lr_scheduler(type="wsd", optimizer=optimizer, total_steps=10, warmup_steps=2, decay_ratio=0.6,
                                 warmup_lr=1.0e-5, min_lr=1.0e-5)

    assert scheduler.get_lr_at_step(0) == pytest.approx([1.0e-5])
    assert scheduler.get_lr_at_step(1) == pytest.approx([5.05e-4])
    assert scheduler.get_lr_at_step(2) == pytest.approx([1.0e-3])
    assert scheduler.get_lr_at_step(5) == pytest.approx([1.0e-3])
    assert scheduler.get_lr_at_step(6) == pytest.approx([1.0e-3])

    halfway_decay = 1.0e-5 + 0.5 * (1.0e-3 - 1.0e-5)
    assert scheduler.get_lr_at_step(8) == pytest.approx([halfway_decay])
    assert scheduler.get_lr_at_step(10) == pytest.approx([1.0e-5])


def test_warmup_rop_linearly_warms_then_uses_plateau_metric():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0e-3)

    scheduler = get_lr_scheduler(type="warmup_rop", optimizer=optimizer, warmup_steps=4, warmup_lr=1.0e-6,
                                 factor=0.5, patience=0, threshold=0.0, min_lr=1.0e-7)

    assert getattr(scheduler, "requires_metric", False) is True
    assert lr_scheduler_can_step_without_metric(scheduler) is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)

    scheduler.step(10.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5075e-4)
    scheduler.step(10.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.005e-4)
    scheduler.step(10.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(7.5025e-4)
    scheduler.step(10.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert lr_scheduler_can_step_without_metric(scheduler) is False

    scheduler.step(10.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    scheduler.step(10.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-4)


def test_warmup_rop_can_step_without_metric_only_during_warmup():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0e-3)
    scheduler = get_lr_scheduler(type="warmup_rop", optimizer=optimizer, warmup_steps=1, warmup_lr=1.0e-6,
                                 factor=0.5, patience=0)

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert lr_scheduler_can_step_without_metric(scheduler) is False

    with pytest.raises(ValueError, match="requires a metric"):
        scheduler.step()


@pytest.mark.parametrize(
    ("scheduler_kwargs", "steps_before_restore", "restore_optimizer_state"),
    (
        (dict(warmup_steps=3, warmup_lr=1.0e-6, factor=0.5, patience=0), 2, False),
        (dict(warmup_steps=1, warmup_lr=1.0e-6, factor=0.5, patience=0, threshold=0.0), 3, True),
    ),
    ids=["mid_warmup_scheduler_state_only", "post_warmup_plateau_with_optimizer_state"],
)
def test_warmup_rop_state_dict_restores_position_and_plateau_state(scheduler_kwargs, steps_before_restore,
                                                                    restore_optimizer_state):
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0e-3)
    scheduler = get_lr_scheduler(type="warmup_rop", optimizer=optimizer, **scheduler_kwargs)
    for _ in range(steps_before_restore):
        scheduler.step(1.0)

    restored_param = torch.nn.Parameter(torch.tensor([1.0]))
    restored_optimizer = torch.optim.SGD([restored_param], lr=1.0e-3)
    restored_scheduler = get_lr_scheduler(type="warmup_rop", optimizer=restored_optimizer, **scheduler_kwargs)
    if restore_optimizer_state:
        restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())

    scheduler.step(1.0)
    restored_scheduler.step(1.0)

    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])


def test_qhflow_poly_scheduler_warms_then_polynomially_decays():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0e-3)

    scheduler = get_lr_scheduler(type="qhflow_poly", optimizer=optimizer, warmup_step=2, num_training_steps=10,
                                 end_lr=1.0e-5, scheduler_power=1.0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-15)
    param.grad = torch.ones_like(param)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-4)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(8.7625e-4)


# ---------------------------------------------------------------------------
# HybridMuon: routes matrix-shaped params to Muon and the rest to AdamW
# ---------------------------------------------------------------------------

def _matrix_vector_case():
    matrix = torch.nn.Parameter(torch.zeros(3, 4))
    vector = torch.nn.Parameter(torch.zeros(4))
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    kwargs = dict(lr=0.1, weight_decay=0.0, muon_beta=0.0, adam_betas=[0.0, 0.0], adam_eps=1.0e-20)

    def check(optimizer):
        assert torch.isfinite(matrix).all()
        assert torch.isfinite(vector).all()
        assert not torch.allclose(matrix, torch.zeros_like(matrix))
        assert torch.allclose(vector, torch.full_like(vector, -0.1))

    return [matrix, vector], kwargs, {"muon": 1, "adam": 1}, check


def _singleton_matrix_case():
    param = torch.nn.Parameter(torch.zeros(1, 3, 4, 1))
    param.grad = torch.ones_like(param)
    kwargs = dict(lr=0.1, weight_decay=0.0, muon_beta=0.0)

    def check(optimizer):
        assert torch.isfinite(param).all()

    return [param], kwargs, {"muon": 1, "adam": 0}, check


def _factorable_flat_weight_case():
    param = torch.nn.Parameter(torch.zeros(32 * 33))
    param.grad = torch.ones_like(param)
    kwargs = dict(lr=0.1, weight_decay=0.0, muon_beta=0.0, magma_lite=False)

    def check(optimizer):
        assert optimizer.route_summary()["params_1d_muon"] == 1
        assert optimizer._effective_shape_for_param(param, optimizer.param_groups[0]) == [32, 33]
        assert torch.isfinite(param).all()

    return [("layers.0.node_onehot_tp.weight", param)], kwargs, {"muon": 1, "adam": 0}, check


def _norm_and_unfactorable_case():
    norm_weight = torch.nn.Parameter(torch.ones(128))
    prime_weight = torch.nn.Parameter(torch.ones(17))
    norm_weight.grad = torch.ones_like(norm_weight)
    prime_weight.grad = torch.ones_like(prime_weight)
    kwargs = dict(lr=0.1, weight_decay=0.1, muon_beta=0.0, adam_betas=[0.0, 0.0], adam_eps=1.0e-20)

    def check(optimizer):
        assert torch.allclose(norm_weight, torch.full_like(norm_weight, 0.89))

    return ([("layers.0.node_norm.weight", norm_weight), ("layers.0.flat_tp.weight", prime_weight)], kwargs,
            {"muon": 0, "adam": 2}, check)


@pytest.mark.parametrize(
    "case_factory",
    [_matrix_vector_case, _singleton_matrix_case, _factorable_flat_weight_case, _norm_and_unfactorable_case],
    ids=["matrix_and_vector", "singleton_matrix_effective_shape", "factorable_flat_weight_by_name",
         "norm_and_prime_unfactorable"],
)
def test_hybrid_muon_routes_params_by_shape_and_name(case_factory):
    model_param, kwargs, expected_route_counts, check = case_factory()
    optimizer = get_optimizer(type="HybridMuon", model_param=model_param, **kwargs)
    optimizer.step()
    assert optimizer.route_counts == expected_route_counts
    check(optimizer)


def test_hybrid_muon_route_summary_refreshes_after_adding_param_group():
    matrix = torch.nn.Parameter(torch.zeros(4, 4))
    vector = torch.nn.Parameter(torch.zeros(4))
    optimizer = get_optimizer(type="HybridMuon", model_param=[matrix], lr=0.1, weight_decay=0.0)

    assert optimizer.route_counts == {"muon": 1, "adam": 0}
    optimizer.add_param_group({"params": [vector]})
    assert optimizer.route_counts == {"muon": 1, "adam": 1}


# ---------------------------------------------------------------------------
# HybridMuon: clipping, diagnostics, and checkpoint compatibility
# ---------------------------------------------------------------------------

def test_hybrid_muon_fixed_clip_caps_update_and_reports_diagnostics():
    param = torch.nn.Parameter(torch.zeros(4, 4))
    param.grad = torch.ones_like(param)
    optimizer = get_optimizer(type="HybridMuon", model_param=[param], lr=1.0, weight_decay=0.0, muon_beta=0.0,
                              muon_scale=10.0, magma_lite=False, muon_clip_mode="fixed", muon_clip_rms=0.25)

    optimizer.step()

    assert (-param).pow(2).mean().sqrt().item() <= 0.250001
    diagnostics = optimizer.get_diagnostics()
    assert diagnostics["muon_clip_events"] >= 1
    assert diagnostics["hybrid_muon_route_numel_muon"] == pytest.approx(16)


def test_hybrid_muon_step_does_not_materialize_diagnostics(monkeypatch):
    matrix = torch.nn.Parameter(torch.zeros(4, 4))
    vector = torch.nn.Parameter(torch.zeros(4))
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    optimizer = get_optimizer(type="HybridMuon", model_param=[matrix, vector], lr=0.1, weight_decay=0.0,
                              muon_beta=0.0, magma_lite=False)

    def fail_on_item(_tensor):
        raise AssertionError("optimizer.step() must not synchronize diagnostics with Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", fail_on_item)
    optimizer.step()


def test_hybrid_muon_diagnostics_accumulate_until_materialized():
    param = torch.nn.Parameter(torch.zeros(4, 4))
    optimizer = get_optimizer(type="HybridMuon", model_param=[param], lr=1.0, weight_decay=0.0, muon_beta=0.0,
                              muon_scale=10.0, magma_lite=False, muon_clip_mode="fixed", muon_clip_rms=0.25)

    for _ in range(3):
        param.grad = torch.ones_like(param)
        optimizer.step()

    diagnostics = optimizer.get_diagnostics()
    assert diagnostics["muon_blocks"] == pytest.approx(3)
    assert diagnostics["muon_clip_events"] == pytest.approx(3)
    assert optimizer.get_diagnostics() == diagnostics

    param.grad = torch.ones_like(param)
    optimizer.step()
    assert optimizer.get_diagnostics()["muon_blocks"] == pytest.approx(1)


def test_hybrid_muon_fills_new_group_defaults_for_old_checkpoint_state():
    param = torch.nn.Parameter(torch.zeros(3, 4))
    param.grad = torch.ones_like(param)
    optimizer = get_optimizer(type="HybridMuon", model_param=[param], lr=0.1)
    for key in list(optimizer.defaults):
        if key.startswith("muon_1d_") or key.startswith("muon_clip"):
            optimizer.param_groups[0].pop(key, None)

    optimizer.step()

    assert optimizer.route_counts == {"muon": 1, "adam": 0}
    assert "muon_clip_mode" in optimizer.param_groups[0]


def test_hybrid_muon_loads_legacy_tensor_adam_step():
    param = torch.nn.Parameter(torch.zeros(4))
    optimizer = get_optimizer(type="HybridMuon", model_param=[param], lr=0.1)
    param.grad = torch.ones_like(param)
    optimizer.step()
    state_dict = optimizer.state_dict()
    state_dict["state"][0]["step"] = torch.tensor(3.0)

    restored_param = torch.nn.Parameter(torch.zeros(4))
    restored = get_optimizer(type="HybridMuon", model_param=[restored_param], lr=0.1)
    restored.load_state_dict(state_dict)

    assert restored.state[restored_param]["step"] == 3
    restored_param.grad = torch.ones_like(restored_param)
    restored.step()
    assert restored.state[restored_param]["step"] == 4


def test_hybrid_muon_step_keeps_dead_expert_replica_finite():
    """A near-zero gradient block (Newton-Schulz's Gram matrix underflows to exactly
    zero in float32) must not blow up into inf/nan for a live neighbor block or an
    unrelated replica in the same optimizer.step()."""
    stacked = torch.nn.Parameter(torch.ones(2, 8, 8))
    replica = torch.nn.Parameter(torch.ones(8, 8))
    stacked.grad = torch.zeros_like(stacked)
    stacked.grad[0] = torch.ones(8, 8)
    stacked.grad[1] = 1.0e-24
    replica.grad = torch.full_like(replica, 1.0e-24)

    optimizer = get_optimizer(
        type="HybridMuon",
        model_param=[("layers.0.edge_tp.weight_experts", stacked), ("layers.0.experts.1.weight", replica)],
        lr=0.1, weight_decay=0.0, muon_beta=0.0, magma_lite=False, muon_clip=False,
    )

    optimizer.step()

    assert optimizer.route_counts == {"muon": 2, "adam": 0}
    assert torch.isfinite(stacked).all()
    assert torch.isfinite(replica).all()
    assert not torch.allclose(stacked[0], torch.ones(8, 8))
    assert torch.allclose(stacked[1], torch.ones(8, 8), atol=1.0e-6)
    assert torch.allclose(replica, torch.ones(8, 8), atol=1.0e-6)


# ---------------------------------------------------------------------------
# train_options argcheck: optimizer/scheduler configs normalize as expected
# ---------------------------------------------------------------------------

def _hybrid_muon_wsd_case():
    config = {
        "num_epoch": 1,
        "optimizer": {"type": "HybridMuon", "lr": 4.5e-4, "weight_decay": 1.0e-3, "adam_eps": 1.0e-20},
        "lr_scheduler": {"type": "wsd", "total_steps": 2_000_000, "warmup_steps": 5_000, "decay_ratio": 0.65,
                        "min_lr": 1.0e-6},
        "update_lr_per_iter": True,
    }

    def check(normalized):
        assert normalized["optimizer"]["type"] == "HybridMuon"
        assert normalized["optimizer"]["magma_lite"] is True
        assert normalized["optimizer"]["muon_1d_route_mode"] == "auto"
        assert normalized["optimizer"]["muon_1d_allow_degenerate_matrix"] is False
        assert normalized["optimizer"]["muon_clip_mode"] == "auto"
        assert normalized["optimizer"]["muon_clip_rms"] == pytest.approx(0.6)
        assert normalized["optimizer"]["muon_clip_max_ratio"] == pytest.approx(0.25)
        assert normalized["lr_scheduler"]["type"] == "wsd"
        assert normalized["lr_scheduler"]["decay_type"] == "cosine"

    return config, check


def _warmup_rop_case():
    config = {
        "num_epoch": 1,
        "optimizer": {"type": "AdamW", "lr": 1.0e-3},
        "lr_scheduler": {"type": "warmup_rop", "warmup_steps": 5_000, "warmup_lr": 1.0e-6, "factor": 0.95,
                        "patience": 1_500, "min_lr": 1.0e-7},
        "update_lr_per_iter": True,
    }

    def check(normalized):
        assert normalized["lr_scheduler"]["type"] == "warmup_rop"
        assert normalized["lr_scheduler"]["warmup_steps"] == 5_000
        assert normalized["lr_scheduler"]["warmup_lr"] == 1.0e-6
        assert chk_avg_per_iter({"train_options": normalized}) is True

    return config, check


def _qhflow_poly_case():
    config = {
        "num_epoch": 1,
        "optimizer": {"type": "AdamW", "lr": 1.0e-3},
        "lr_scheduler": {"type": "qhflow_poly", "warmup_step": 1_000, "num_training_steps": 200_000,
                        "end_lr": 1.0e-9, "scheduler_power": 1.0},
        "update_lr_per_iter": True,
    }

    def check(normalized):
        assert normalized["lr_scheduler"]["type"] == "qhflow_poly"
        assert normalized["lr_scheduler"]["warmup_step"] == 1_000
        assert normalized["lr_scheduler"]["num_training_steps"] == 200_000
        assert normalized["lr_scheduler"]["end_lr"] == 1.0e-9
        assert chk_avg_per_iter({"train_options": normalized}) is False

    return config, check


@pytest.mark.parametrize("case_factory", [_hybrid_muon_wsd_case, _warmup_rop_case, _qhflow_poly_case],
                         ids=["hybrid_muon_wsd", "warmup_rop", "qhflow_poly"])
def test_train_options_accepts_optimizer_and_scheduler_configs(case_factory):
    config, check = case_factory()
    normalized = train_options().normalize_value(config)
    check(normalized)

import pytest
import torch

from dptb.utils.argcheck import chk_avg_per_iter, train_options
from dptb.utils.tools import get_lr_scheduler, get_optimizer, lr_scheduler_can_step_without_metric


def test_wsd_scheduler_matches_dpa4_warmup_stable_decay_formula():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0e-3)

    scheduler = get_lr_scheduler(
        type="wsd",
        optimizer=optimizer,
        total_steps=10,
        warmup_steps=2,
        decay_ratio=0.6,
        warmup_lr=1.0e-5,
        min_lr=1.0e-5,
    )

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

    scheduler = get_lr_scheduler(
        type="warmup_rop",
        optimizer=optimizer,
        warmup_steps=4,
        warmup_lr=1.0e-6,
        factor=0.5,
        patience=0,
        threshold=0.0,
        min_lr=1.0e-7,
    )

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
    scheduler = get_lr_scheduler(
        type="warmup_rop",
        optimizer=optimizer,
        warmup_steps=1,
        warmup_lr=1.0e-6,
        factor=0.5,
        patience=0,
    )

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert lr_scheduler_can_step_without_metric(scheduler) is False

    with pytest.raises(ValueError, match="requires a metric"):
        scheduler.step()


def test_warmup_rop_state_dict_restores_warmup_position():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0e-3)
    scheduler = get_lr_scheduler(
        type="warmup_rop",
        optimizer=optimizer,
        warmup_steps=3,
        warmup_lr=1.0e-6,
        factor=0.5,
        patience=0,
    )
    scheduler.step(1.0)
    scheduler.step(1.0)

    restored_param = torch.nn.Parameter(torch.tensor([1.0]))
    restored_optimizer = torch.optim.SGD([restored_param], lr=1.0e-3)
    restored_scheduler = get_lr_scheduler(
        type="warmup_rop",
        optimizer=restored_optimizer,
        warmup_steps=3,
        warmup_lr=1.0e-6,
        factor=0.5,
        patience=0,
    )
    restored_scheduler.load_state_dict(scheduler.state_dict())

    scheduler.step(1.0)
    restored_scheduler.step(1.0)

    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])


def test_warmup_rop_state_dict_restores_plateau_state_after_warmup():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0e-3)
    scheduler = get_lr_scheduler(
        type="warmup_rop",
        optimizer=optimizer,
        warmup_steps=1,
        warmup_lr=1.0e-6,
        factor=0.5,
        patience=0,
        threshold=0.0,
    )
    scheduler.step(1.0)
    scheduler.step(1.0)
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-4)

    restored_param = torch.nn.Parameter(torch.tensor([1.0]))
    restored_optimizer = torch.optim.SGD([restored_param], lr=1.0e-3)
    restored_scheduler = get_lr_scheduler(
        type="warmup_rop",
        optimizer=restored_optimizer,
        warmup_steps=1,
        warmup_lr=1.0e-6,
        factor=0.5,
        patience=0,
        threshold=0.0,
    )
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())

    scheduler.step(1.0)
    restored_scheduler.step(1.0)

    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])


def test_hybrid_muon_routes_matrix_params_to_muon_and_vectors_to_adamw():
    matrix = torch.nn.Parameter(torch.zeros(3, 4))
    vector = torch.nn.Parameter(torch.zeros(4))
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)

    optimizer = get_optimizer(
        type="HybridMuon",
        model_param=[matrix, vector],
        lr=0.1,
        weight_decay=0.0,
        muon_beta=0.0,
        adam_betas=[0.0, 0.0],
        adam_eps=1.0e-20,
    )

    optimizer.step()

    assert optimizer.route_counts == {"muon": 1, "adam": 1}
    assert torch.isfinite(matrix).all()
    assert torch.isfinite(vector).all()
    assert not torch.allclose(matrix, torch.zeros_like(matrix))
    assert torch.allclose(vector, torch.full_like(vector, -0.1))
    assert "momentum_buffer" in optimizer.state[matrix]
    assert "magma_ema" in optimizer.state[matrix]
    assert "exp_avg" in optimizer.state[vector]


def test_hybrid_muon_uses_effective_shape_for_singleton_matrix_tensors():
    param = torch.nn.Parameter(torch.zeros(1, 3, 4, 1))
    param.grad = torch.ones_like(param)

    optimizer = get_optimizer(
        type="HybridMuon",
        model_param=[param],
        lr=0.1,
        weight_decay=0.0,
        muon_beta=0.0,
    )

    optimizer.step()

    assert optimizer.route_counts == {"muon": 1, "adam": 0}
    assert torch.isfinite(param).all()
    assert "momentum_buffer" in optimizer.state[param]
    assert "magma_ema" in optimizer.state[param]


def test_train_options_accepts_hybrid_muon_optimizer_and_wsd_scheduler():
    normalized = train_options().normalize_value(
        {
            "num_epoch": 1,
            "optimizer": {
                "type": "HybridMuon",
                "lr": 4.5e-4,
                "weight_decay": 1.0e-3,
                "adam_eps": 1.0e-20,
            },
            "lr_scheduler": {
                "type": "wsd",
                "total_steps": 2_000_000,
                "warmup_steps": 5_000,
                "decay_ratio": 0.65,
                "min_lr": 1.0e-6,
            },
            "update_lr_per_iter": True,
        }
    )
    assert normalized["optimizer"]["type"] == "HybridMuon"
    assert normalized["optimizer"]["magma_lite"] is True
    assert normalized["lr_scheduler"]["type"] == "wsd"
    assert normalized["lr_scheduler"]["decay_type"] == "cosine"


def test_train_options_accepts_warmup_rop_scheduler():
    config = {
        "train_options": {
            "num_epoch": 1,
            "optimizer": {
                "type": "AdamW",
                "lr": 1.0e-3,
            },
            "lr_scheduler": {
                "type": "warmup_rop",
                "warmup_steps": 5_000,
                "warmup_lr": 1.0e-6,
                "factor": 0.95,
                "patience": 1_500,
                "min_lr": 1.0e-7,
            },
            "update_lr_per_iter": True,
        }
    }

    normalized = train_options().normalize_value(config["train_options"])

    assert normalized["lr_scheduler"]["type"] == "warmup_rop"
    assert normalized["lr_scheduler"]["warmup_steps"] == 5_000
    assert normalized["lr_scheduler"]["warmup_lr"] == 1.0e-6
    assert chk_avg_per_iter({"train_options": normalized}) is True

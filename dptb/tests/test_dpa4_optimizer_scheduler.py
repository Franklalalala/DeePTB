import pytest
import torch

from dptb.utils.argcheck import train_options
from dptb.utils.tools import get_lr_scheduler, get_optimizer


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

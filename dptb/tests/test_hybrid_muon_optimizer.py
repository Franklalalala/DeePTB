import pytest

torch = pytest.importorskip("torch")

from dptb.utils.argcheck import train_options
from dptb.utils.tools import get_optimizer


def test_train_options_accepts_hybrid_muon_production_keys():
    normalized = train_options().normalize_value(
        {
            "num_epoch": 1,
            "optimizer": {
                "type": "HybridMuon",
                "lr": 0.01,
                "weight_decay": 0.01,
                "adam_betas": [0.98, 0.999],
                "adam_eps": 1.0e-20,
                "muon_beta": 0.95,
                "muon_scale": 0.2,
                "muon_clip": True,
                "muon_clip_rms": 0.2,
                "matrix_min_dim": 2,
                "magma_lite": False,
            },
            "lr_scheduler": {
                "type": "rop",
                "factor": 0.95,
                "patience": 2000,
                "min_lr": 1.0e-5,
            },
        }
    )

    assert normalized["optimizer"]["type"] == "HybridMuon"
    assert normalized["optimizer"]["muon_scale"] == pytest.approx(0.2)
    assert normalized["optimizer"]["muon_clip"] is True
    assert normalized["optimizer"]["muon_clip_rms"] == pytest.approx(0.2)
    assert normalized["optimizer"]["magma_lite"] is False
    assert normalized["lr_scheduler"]["type"] == "rop"


def test_get_optimizer_builds_hybrid_muon_with_named_parameters():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    vector = torch.nn.Parameter(torch.ones(4))

    optimizer = get_optimizer(
        type="HybridMuon",
        model_param=[("matrix.weight", matrix), ("bias", vector)],
        lr=0.01,
        weight_decay=0.0,
        muon_beta=0.95,
        muon_scale=0.2,
        adam_betas=[0.98, 0.999],
        adam_eps=1.0e-20,
        muon_clip=True,
        muon_clip_rms=0.2,
        matrix_min_dim=2,
        magma_lite=False,
    )

    summary = optimizer.route_summary()
    assert summary["params_muon"] == 1
    assert summary["params_adam"] == 1

    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    optimizer.step()

    assert torch.isfinite(matrix).all()
    assert torch.isfinite(vector).all()

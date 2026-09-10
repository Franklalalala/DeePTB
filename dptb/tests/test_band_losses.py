"""Loss API, physical addback, and reference overlap survive modularization."""

import subprocess, sys, types
import pytest
import torch
from dptb.data import AtomicDataDict as A
from dptb.nnops.loss import Loss, EigHamH0ResLoss
from dptb.nnops.band_losses import BandStage2Loss


def test_direct_band_import_is_cycle_free():
    code = "from dptb.nnops.band_losses import FW10EigLoss; from dptb.nnops.loss import FW10EigLoss as Alias; assert Alias is FW10EigLoss"
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "method",
    ["fw10_eig", "band_stage2", "nextham_kspace", "eig_ham_h0res", "hamil_abs_gauged"],
)
def test_registry_builds_spectral_losses(method):
    obj = Loss(method=method, basis={"H": "1s"}, device="cpu")
    assert isinstance(obj, torch.nn.Module)


def test_h0_addback_keeps_residual_gradient_and_reference_overlap():
    residual = torch.tensor([[2.0]], requires_grad=True)
    ref = {
        A.NODE_FEATURES_KEY: residual,
        A.EDGE_FEATURES_KEY: residual,
        A.NODE_H0_KEY: torch.tensor([[3.0]]),
        A.EDGE_H0_KEY: torch.tensor([[4.0]]),
        A.NODE_OVERLAP_KEY: torch.ones(1, 1),
        A.EDGE_OVERLAP_KEY: torch.ones(1, 1),
        A.KPOINT_KEY: torch.zeros(1, 3),
    }
    loss = EigHamH0ResLoss(basis={"H": "1s"})
    physical = loss._add_h0(ref)
    assert physical[A.NODE_FEATURES_KEY].item() == 5.0
    assert ref[A.NODE_FEATURES_KEY].item() == 2.0
    physical[A.NODE_FEATURES_KEY].sum().backward()
    assert residual.grad.item() == 1.0
    physical[A.EDGE_OVERLAP_KEY] = torch.zeros(1, 128)
    solver = loss._solver_dict(physical, ref)
    assert solver[A.EDGE_OVERLAP_KEY].shape == (1, 1)


def test_band_stage2_rejects_batched_graphs():
    loss = BandStage2Loss(basis={"H": "1s"})
    with pytest.raises(RuntimeError, match="one graph"):
        loss({}, {A.BATCH_PTR_KEY: torch.tensor([0, 1, 2])})


@pytest.mark.parametrize("method", ["fw10_eig", "band_stage2", "nextham_kspace"])
def test_spectral_config_schema(method):
    from dptb.utils.argcheck import loss_options

    schema = loss_options()
    cfg = schema.normalize_value({"train": {"method": method}})
    schema.check_value(cfg, strict=True)
    assert cfg["train"]["method"] == method


class _HamProbe(torch.nn.Module):
    def forward(self, data, ref):
        return data["x"].square().sum()


@pytest.mark.parametrize(
    "method,options",
    [
        ("band_stage2", {"band_weight": 0.0}),
        ("nextham_kspace", {"w_p": 0.0, "w_q": 0.0, "w_pq": 0.0, "gauge": False}),
    ],
)
def test_pure_h_control_never_requires_spectral_data(method, options):
    obj = Loss(method=method, basis={"H": "1s"}, **options)
    obj.ham_loss = _HamProbe()
    p = torch.tensor([2.0], requires_grad=True)
    value = obj({"x": p}, {})
    value.backward()
    assert value.item() == 4.0 and p.grad.item() == 4.0
    assert obj._last_parts["band_computed"] is False


def test_band_solver_accepts_normalized_float32_string():
    obj = BandStage2Loss(basis={"H": "1s"}, solver_float64=False, dtype="float32")
    out = obj._solver_dict(
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        {A.NODE_OVERLAP_KEY: torch.ones(1, 1), A.EDGE_OVERLAP_KEY: torch.ones(1, 1)},
        torch.zeros(1, 3),
    )
    assert out[A.NODE_FEATURES_KEY].dtype == torch.float32

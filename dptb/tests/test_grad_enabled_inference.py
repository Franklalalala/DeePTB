import pytest
import torch

from dptb.utils.inference import (
    GRAD_INFERENCE_ENV,
    grad_enabled_inference,
    requires_grad_enabled_inference,
)


class _BackendModule(torch.nn.Module):
    def __init__(self, **attrs):
        super().__init__()
        for name, value in attrs.items():
            setattr(self, name, value)


def test_grad_enabled_inference_defaults_to_no_grad_for_plain_module():
    model = torch.nn.Linear(2, 2)

    with grad_enabled_inference(model):
        assert torch.is_grad_enabled() is False


def test_grad_enabled_inference_explicit_true_reenables_outer_no_grad():
    model = torch.nn.Linear(2, 2)

    with torch.no_grad():
        assert torch.is_grad_enabled() is False
        with grad_enabled_inference(model, enable_grad=True):
            assert torch.is_grad_enabled() is True
        assert torch.is_grad_enabled() is False


def test_grad_enabled_inference_env_override(monkeypatch):
    model = torch.nn.Linear(2, 2)
    monkeypatch.setenv(GRAD_INFERENCE_ENV, "true")

    with grad_enabled_inference(model):
        assert torch.is_grad_enabled() is True

    monkeypatch.setenv(GRAD_INFERENCE_ENV, "0")
    so2_model = torch.nn.Sequential(
        _BackendModule(so2_fusion_mode="streamed_m_major_fused_p0")
    )
    with grad_enabled_inference(so2_model):
        assert torch.is_grad_enabled() is False


def test_grad_enabled_inference_rejects_bad_env_flag(monkeypatch):
    monkeypatch.setenv(GRAD_INFERENCE_ENV, "maybe")

    with pytest.raises(ValueError, match=GRAD_INFERENCE_ENV):
        with grad_enabled_inference(torch.nn.Linear(2, 2)):
            pass


@pytest.mark.parametrize(
    "attrs",
    [
        {"so2_fusion_mode": "streamed_m_major_ref"},
        {"so2_fusion_mode": "streamed_m_major_cueq"},
        {"so2_fusion_mode": "streamed_m_major_fused_p0"},
        {"so2_fusion_mode": "streamed_m_major_persistent_grouped_p1"},
        {"mole_linear_mode": "cublas_grouped"},
        {"mole_linear_mode": "cueq_indexed_linear"},
    ],
)
def test_requires_grad_enabled_inference_detects_fast_backends(attrs):
    assert requires_grad_enabled_inference(_BackendModule(**attrs)) is True


@pytest.mark.parametrize(
    "attrs",
    [
        {},
        {"so2_fusion_mode": "staged"},
        {"mole_linear_mode": "split_loop"},
    ],
)
def test_requires_grad_enabled_inference_ignores_plain_backends(attrs):
    assert requires_grad_enabled_inference(_BackendModule(**attrs)) is False

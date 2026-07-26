"""Device-migration regression tests for the maintained E3 Hamiltonian."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dptb.nn.hamiltonian import E3Hamiltonian


BASIS = {"Si": ["3s", "3p"]}
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for device migration"
)


def _basis_dict(module):
    basis = getattr(module, "cgbasis", None)
    assert isinstance(basis, dict) and basis, "module exposes no CG basis dict"
    return basis


@needs_cuda
def test_to_cuda_migrates_basis_dict_and_syncs_device():
    model = E3Hamiltonian(basis=BASIS, device="cpu")
    assert all(value.device.type == "cpu" for value in _basis_dict(model).values())
    model = model.to("cuda")
    assert all(value.device.type == "cuda" for value in _basis_dict(model).values())
    assert torch.device(model.device).type == "cuda"


def test_double_syncs_dtype_of_basis_and_module():
    model = E3Hamiltonian(basis=BASIS, device="cpu")
    assert model.dtype == torch.float32
    model = model.double()
    assert model.dtype == torch.float64
    assert all(value.dtype == torch.float64 for value in model.cgbasis.values())


def test_state_dict_keys_unchanged_by_migration_fix():
    model = E3Hamiltonian(basis=BASIS, device="cpu")
    for key in model.state_dict():
        assert "cgbasis" not in key
        assert "soc_base_matrix" not in key

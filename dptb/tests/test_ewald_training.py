"""Ewald direct solves preserve learned and geometry derivatives."""
import pytest
import torch

from dptb.nn.heads.charge_response import PeriodicGammaCache, solve_periodic


@pytest.mark.parametrize("length", [8., 24.])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_direct_solvers_match_values_and_derivatives(length, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA derivative qualification")
    generator = torch.Generator().manual_seed(13)
    pos = (torch.rand(12, 3, generator=generator, dtype=torch.float64).to(device) * length).requires_grad_()
    cell = (torch.eye(3, dtype=torch.float64).to(device) * length).requires_grad_()
    chi = torch.randn(12, 1, generator=generator, dtype=torch.float64).to(device).requires_grad_()
    hardness = torch.full_like(chi, 8., requires_grad=True)
    records = []
    for solver in ("dense", "ewald"):
        q, potential = solve_periodic(chi, hardness, pos, cell, sigma=1.2,
                                     g_cut=4., max_modes=100000, solver=solver)
        loss = q.square().sum() + potential.square().sum()
        gradients = torch.autograd.grad(loss, (chi, hardness, pos, cell))
        assert all(torch.isfinite(g).all() for g in gradients)
        records.append((q, potential, *gradients))
    for expected, actual in zip(*records):
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)


def test_training_rejects_inference_cache():
    pos = torch.zeros(2, 3, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 24
    with pytest.raises(RuntimeError, match="inference_mode"):
        solve_periodic(torch.zeros(2, 1), torch.ones(2, 1), pos, cell,
                       sigma=1.2, g_cut=4., max_modes=100000,
                       solver="ewald", gamma_cache=PeriodicGammaCache())

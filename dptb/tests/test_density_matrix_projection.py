import torch

from dptb.postprocess.density_matrix_projection import (
    density_diagnostics,
    project_closed_shell_density,
    project_occupations_capped_simplex,
)


def test_closed_shell_projection_satisfies_constraints():
    torch.manual_seed(0)
    n = 5
    s = torch.eye(n, dtype=torch.double) + 0.05 * torch.randn(n, n, dtype=torch.double)
    s = 0.5 * (s + s.T) + 0.5 * torch.eye(n)
    d = torch.randn(n, n, dtype=torch.double)
    d = 0.5 * (d + d.T)
    result = project_closed_shell_density(d, s, n_electrons=4)
    assert torch.allclose(result.electron_count, torch.tensor(4.0, dtype=torch.double), atol=1e-6)
    assert result.idempotency_error < 1e-6


def test_capped_simplex_projection():
    occ = torch.tensor([2.5, 1.0, -0.2, 0.1])
    proj = project_occupations_capped_simplex(occ, n_electrons=3.0, max_occ=2.0)
    assert torch.all(proj >= -1e-8)
    assert torch.all(proj <= 2.0 + 1e-8)
    assert torch.allclose(proj.sum(), torch.tensor(3.0), atol=1e-6)

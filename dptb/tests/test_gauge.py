import torch

from dptb.nnops.gauge import gauge_mae, solve_mu


def test_solve_mu_recovers_known_shift_real():
    torch.manual_seed(0)
    target = torch.randn(4, 4)
    s = torch.eye(4) + 0.1 * torch.randn(4, 4)
    s = 0.5 * (s + s.T)
    mu_true = torch.tensor(0.37)
    pred = target + mu_true * s
    mu = solve_mu(pred - target, s)
    assert torch.allclose(mu, mu_true, atol=1e-6)


def test_gauge_mae_is_shift_invariant_complex():
    torch.manual_seed(1)
    target = torch.randn(3, 3, dtype=torch.cdouble)
    target = 0.5 * (target + target.mH)
    s = torch.eye(3, dtype=torch.cdouble)
    pred = target + 2.0 * s
    res = gauge_mae(pred, target, s)
    assert res.mae < 1e-10
    assert torch.allclose(res.mu, torch.tensor(2.0, dtype=res.mu.dtype), atol=1e-10)


def test_gauge_mae_leq_raw_mae():
    torch.manual_seed(2)
    target = torch.randn(5, 5)
    s = torch.eye(5)
    pred = target + 0.5 * s + 0.01 * torch.randn(5, 5)
    raw = (pred - target).abs().mean()
    aligned = gauge_mae(pred, target, s).mae
    assert aligned <= raw

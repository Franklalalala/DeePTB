import torch

from dptb.nnops.kspace_pq_core import dense_kspace_pq_loss, generalized_eigh


def test_generalized_eigh_s_orthonormal():
    h = torch.diag(torch.tensor([-1.0, 0.0, 2.0], dtype=torch.double))
    s = torch.tensor([[1.2, 0.1, 0.0], [0.1, 1.1, 0.0], [0.0, 0.0, 0.9]], dtype=torch.double)
    eps, c = generalized_eigh(h, s)
    assert torch.allclose(c.mH @ s @ c, torch.eye(3, dtype=torch.double), atol=1e-8)
    assert torch.all(torch.diff(eps) >= 0)


def test_pq_loss_zero_for_exact():
    h_ref = torch.diag(torch.tensor([-2.0, -1.0, 1.0, 3.0], dtype=torch.double))
    s = torch.eye(4, dtype=torch.double)
    loss, metrics = dense_kspace_pq_loss(h_ref.clone(), h_ref, s, n_occ=2, e_cut_above_fermi=None)
    assert loss < 1e-12
    assert metrics["loss_pq"] < 1e-12


def test_pq_perturbation_is_detected_and_grad_flows():
    h_ref = torch.diag(torch.tensor([-2.0, -1.0, 1.0, 3.0], dtype=torch.double))
    h_pred = h_ref.clone()
    h_pred[0, 2] = h_pred[2, 0] = 0.2
    h_pred.requires_grad_(True)
    s = torch.eye(4, dtype=torch.double)
    loss, metrics = dense_kspace_pq_loss(
        h_pred,
        h_ref,
        s,
        n_occ=2,
        e_cut_above_fermi=None,
        lambda_p=1.0,
        lambda_q=0.1,
        lambda_pq=1.0,
        gauge_mu=False,
    )
    assert metrics["loss_pq"] > 0.0
    loss.backward()
    assert h_pred.grad is not None
    assert torch.isfinite(h_pred.grad).all()


def test_kspace_gauge_shift_invariance():
    h_ref = torch.diag(torch.tensor([-2.0, -1.0, 1.0, 3.0], dtype=torch.double))
    s = torch.eye(4, dtype=torch.double)
    h_pred = h_ref + 0.7 * s
    loss, _ = dense_kspace_pq_loss(h_pred, h_ref, s, n_occ=2, e_cut_above_fermi=None, gauge_mu=True)
    assert loss < 1e-12

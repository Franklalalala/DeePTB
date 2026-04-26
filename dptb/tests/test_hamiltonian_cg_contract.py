import torch

from dptb.nn.hamiltonian import _contract_cg_rme


def _broadcast_reference(cg_basis, rme2):
    out = torch.sum(
        cg_basis[None, :, :, :, None] * rme2[:, None, None, :, :],
        dim=-2,
    )
    return out.permute(0, 3, 1, 2)


def test_cg_rme_contract_matches_broadcast_reference_and_grad():
    torch.manual_seed(11)
    n_rows = 7
    n_left = 5
    n_right = 5
    n_rme = 25
    n_chunk = 4

    cg0 = torch.randn(n_left, n_right, n_rme, dtype=torch.float64)
    rme0 = torch.randn(n_rows, n_rme, n_chunk, dtype=torch.float64)
    modes = ("gemm", "einsum", "tiled_broadcast", "broadcast")
    for mode in modes:
        cg_ref = cg0.detach().clone().requires_grad_(True)
        rme_ref = rme0.detach().clone().requires_grad_(True)
        cg_new = cg0.detach().clone().requires_grad_(True)
        rme_new = rme0.detach().clone().requires_grad_(True)

        ref = _broadcast_reference(cg_ref, rme_ref)
        new = _contract_cg_rme(cg_new, rme_new, mode=mode, tile_rows=3)

        assert torch.allclose(new, ref, atol=1e-12, rtol=1e-12), mode

        grad = torch.randn_like(ref)
        ref.backward(grad)
        new.backward(grad)

        assert torch.allclose(cg_new.grad, cg_ref.grad, atol=1e-12, rtol=1e-12), mode
        assert torch.allclose(rme_new.grad, rme_ref.grad, atol=1e-12, rtol=1e-12), mode

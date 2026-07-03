import torch

from dptb.postprocess.h_repair_dense import generalized_eigh_safe, repair_pair_blocks


def test_generalized_eigh_safe_handles_small_overlap_eig():
    h = torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.double))
    s = torch.diag(torch.tensor([1e-12, 1.0], dtype=torch.double))
    spec = generalized_eigh_safe(h, s, eig_floor=1e-8)
    assert torch.isfinite(spec.eigvals).all()
    assert spec.min_overlap_eig < 1e-8


def test_repair_pair_blocks_creates_reverse_and_hermitianizes():
    block = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    blocks = {(0, 1, 1, 0, 0): block, (0, 0, 0, 0, 0): torch.tensor([[1.0, 2.0], [0.0, 3.0]])}
    repaired = repair_pair_blocks(blocks)
    fwd = repaired[(0, 1, 1, 0, 0)]
    rev = repaired[(1, 0, -1, 0, 0)]
    assert torch.allclose(rev, fwd.mT)
    onsite = repaired[(0, 0, 0, 0, 0)]
    assert torch.allclose(onsite, onsite.mT)

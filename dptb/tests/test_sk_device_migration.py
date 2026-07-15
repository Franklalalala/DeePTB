"""Device-migration regression tests for the E3/SK Hamiltonian basis dicts.

Multi-GPU expert placement moves each expert with ``expert.to(cuda_i)`` after
CPU construction. The CG/SK basis tensors live in plain dicts
(``cgbasis``/``skbasis``/``soc_base_matrix``) which ``nn.Module.to()`` never
migrates, and ``forward()`` allocates work tensors on the construction-time
``self.device`` — producing "Expected all tensors to be on the same device,
cuda:1 and cpu" at the first MoE forward (observed on natlan, 2x L40S).
The ``_apply`` overrides keep the dicts and ``self.device``/``self.dtype``
in step with ``.to()``/``.cuda()``/``.float()`` without touching
``state_dict()`` (old checkpoints must keep loading).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dptb.nn.hamiltonian import E3Hamiltonian, SKHamiltonian, SKHamiltonian_old

BASIS = {"Si": ["3s", "3p"]}

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for device migration"
)


def _basis_dict(module):
    d = getattr(module, "skbasis", None)
    if not isinstance(d, dict) or not d:
        d = getattr(module, "cgbasis", None)
    assert isinstance(d, dict) and d, "module exposes no basis dict"
    return d


@needs_cuda
@pytest.mark.parametrize(
    "factory",
    [
        lambda: SKHamiltonian(basis=BASIS, device="cpu"),
        lambda: SKHamiltonian_old(basis=BASIS, device="cpu"),
        lambda: E3Hamiltonian(basis=BASIS, device="cpu"),
    ],
    ids=["SKHamiltonian", "SKHamiltonian_old", "E3Hamiltonian"],
)
def test_to_cuda_migrates_basis_dict_and_syncs_device(factory):
    m = factory()
    assert all(v.device.type == "cpu" for v in _basis_dict(m).values())
    m = m.to("cuda")
    assert all(v.device.type == "cuda" for v in _basis_dict(m).values())
    # forward() allocates on self.device — it must follow the module.
    assert torch.device(m.device).type == "cuda"


@needs_cuda
def test_skbasis_contraction_works_on_cuda_after_move():
    # Regression for the exact natlan crash line (hamiltonian.py forward):
    #   H_z = torch.sum(self.skbasis[opairtype][None,:,:,:,None]
    #                   * skparam[:,None,None,:,:], dim=-2)
    m = SKHamiltonian(basis=BASIS, device="cpu").to("cuda")
    pairtype = next(iter(m.skbasis))
    bb = m.skbasis[pairtype]  # (2l1+1, 2l2+1, n_skp)
    n_edge, n_pair = 3, 4
    skparam = torch.randn(n_edge, bb.shape[2], n_pair, device="cuda", dtype=m.dtype)
    h_z = torch.sum(bb[None, :, :, :, None] * skparam[:, None, None, :, :], dim=-2)
    assert h_z.device.type == "cuda"
    assert h_z.shape == (n_edge, bb.shape[0], bb.shape[1], n_pair)


def test_double_syncs_dtype_of_basis_and_module():
    m = SKHamiltonian(basis=BASIS, device="cpu")
    assert m.dtype == torch.float32
    m = m.double()
    assert m.dtype == torch.float64
    assert all(v.dtype == torch.float64 for v in m.skbasis.values())


def test_state_dict_keys_unchanged_by_migration_fix():
    # The fix must NOT register the dicts as buffers/parameters: existing
    # production checkpoints do not contain these keys and must keep loading.
    for m in (
        SKHamiltonian(basis=BASIS, device="cpu"),
        SKHamiltonian_old(basis=BASIS, device="cpu"),
        E3Hamiltonian(basis=BASIS, device="cpu"),
    ):
        for key in m.state_dict():
            assert "skbasis" not in key
            assert "cgbasis" not in key
            assert "soc_base_matrix" not in key

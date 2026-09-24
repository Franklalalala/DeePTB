"""E3Hamiltonian/SKHamiltonian: CG-RME contraction, SOC decompose semantics, device migration.

Multi-GPU expert placement moves each expert with ``expert.to(cuda_i)`` after CPU
construction. The CG/SK basis tensors live in plain dicts (``cgbasis``/``skbasis``/
``soc_base_matrix``), which ``nn.Module.to()`` never migrates on its own, and
``forward()`` allocates work tensors on the construction-time ``self.device`` — this
used to raise "Expected all tensors to be on the same device, cuda:1 and cpu" at the
first MoE forward (observed on a two-GPU host). The ``_apply`` overrides keep the
dicts and ``self.device``/``self.dtype`` in step with ``.to()``/``.cuda()``/``.float()``
without touching ``state_dict()`` (old checkpoints must keep loading).
"""
from __future__ import annotations

import pytest
import torch

from dptb.data import AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.nn.hamiltonian import E3Hamiltonian, SKHamiltonian, SKHamiltonian_old, _contract_cg_rme
from dptb.nnops.block_flow_codec import BlockStateCodec
from dptb.tests._requires import requires_cuda

SK_BASIS = {"Si": ["3s", "3p"]}


# ---------------------------------------------------------------------------
# _contract_cg_rme: matches a broadcast-sum reference, including gradients
# ---------------------------------------------------------------------------

def _broadcast_reference(cg_basis, rme2):
    out = torch.sum(cg_basis[None, :, :, :, None] * rme2[:, None, None, :, :], dim=-2)
    return out.permute(0, 3, 1, 2)


def test_cg_rme_contract_matches_broadcast_reference_and_grad():
    torch.manual_seed(11)
    n_rows, n_left, n_right, n_rme, n_chunk = 7, 5, 5, 25, 4

    cg0 = torch.randn(n_left, n_right, n_rme, dtype=torch.float64)
    rme0 = torch.randn(n_rows, n_rme, n_chunk, dtype=torch.float64)
    cg_ref, rme_ref = cg0.detach().clone().requires_grad_(True), rme0.detach().clone().requires_grad_(True)
    cg_new, rme_new = cg0.detach().clone().requires_grad_(True), rme0.detach().clone().requires_grad_(True)

    ref = _broadcast_reference(cg_ref, rme_ref)
    new = _contract_cg_rme(cg_new, rme_new)
    assert torch.allclose(new, ref, atol=1e-12, rtol=1e-12)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    new.backward(grad)
    assert torch.allclose(cg_new.grad, cg_ref.grad, atol=1e-12, rtol=1e-12)
    assert torch.allclose(rme_new.grad, rme_ref.grad, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# E3Hamiltonian SOC/non-SOC decompose: pass-through vs rejection boundaries
# ---------------------------------------------------------------------------

def _two_row_data(module):
    width = int(module.idp.reduced_matrix_element)
    node_features = torch.arange(2 * width, dtype=torch.float64).reshape(2, width)
    edge_features = torch.arange(2 * width, dtype=torch.float64).reshape(2, width) + 0.25
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        AtomicDataDict.ATOM_TYPE_KEY: torch.zeros(2, dtype=torch.long),
        AtomicDataDict.NODE_FEATURES_KEY: node_features.clone(),
        AtomicDataDict.EDGE_FEATURES_KEY: edge_features.clone(),
    }
    return data, node_features, edge_features


def _soc_kwarg_with_basis():
    # decompose=True, soc=True, basis (no explicit idp): the legacy calling convention.
    return E3Hamiltonian(basis={"H": ["1s", "2p"]}, decompose=True, soc=True, dtype=torch.float64, device="cpu")


def _soc_mapper_soc_kwarg_omitted():
    # A SOC mapper is authoritative even when the legacy caller omits soc=True.
    mapper = OrbitalMapper({"C": ["2p"]}, method="e3tb", device="cpu", has_soc=True)
    mapper.get_orbital_maps()
    mapper.get_irreps(no_parity=False)
    module = E3Hamiltonian(idp=mapper, decompose=True, dtype=torch.float64, device="cpu")
    assert module.soc is False
    assert module.idp.has_soc is True
    return module


def _non_soc_mapper_default():
    mapper = OrbitalMapper({"C": ["2p"]}, method="e3tb", device="cpu", has_soc=False)
    mapper.get_orbital_maps()
    mapper.get_irreps(no_parity=False)
    return E3Hamiltonian(idp=mapper, decompose=True, dtype=torch.float64, device="cpu")


@pytest.mark.parametrize(
    "make_module",
    [_soc_kwarg_with_basis, _soc_mapper_soc_kwarg_omitted, _non_soc_mapper_default],
    ids=["soc_kwarg_with_basis", "soc_mapper_soc_kwarg_omitted", "non_soc_mapper_no_inverse_opt_in"],
)
def test_soc_and_non_soc_decompose_is_legacy_pass_through(make_module):
    """decompose=True must not block LMDBDataset.E3statistics pass-through, SOC or not."""
    module = make_module()
    data, node_features, edge_features = _two_row_data(module)

    # p-p CG change of basis is not the identity: distinguishes true pass-through from
    # accidentally running the non-SOC inverse-CG branch.
    p_basis = module.cgbasis["p-p"].reshape(9, 9)
    assert not torch.equal(p_basis, torch.eye(9, dtype=torch.float64))

    result = module(data)
    assert torch.equal(result[AtomicDataDict.NODE_FEATURES_KEY], node_features)
    assert torch.equal(result[AtomicDataDict.EDGE_FEATURES_KEY], edge_features)


def test_inverse_cg_opt_in_rejects_soc_mapper_even_without_soc_kwarg():
    mapper = OrbitalMapper({"C": ["2p"]}, method="e3tb", device="cpu", has_soc=True)
    with pytest.raises(NotImplementedError, match="non-SOC"):
        E3Hamiltonian(idp=mapper, decompose=True, enable_inverse_cg=True, dtype=torch.float64, device="cpu")


def test_soc_decompose_rejects_a_non_soc_mapper():
    mapper = OrbitalMapper({"H": ["1s", "2p"]}, method="e3tb", device="cpu", has_soc=False)
    mapper.get_orbital_maps()
    mapper.get_irreps(no_parity=False)
    module = E3Hamiltonian(idp=mapper, decompose=True, soc=True, dtype=torch.float64, device="cpu")
    data, _, _ = _two_row_data(module)
    with pytest.raises(NotImplementedError, match="non-SOC OrbitalMapper"):
        module(data)


def test_block_state_codec_remains_the_soc_rejection_boundary():
    mapper = OrbitalMapper({"C": ["2p"]}, method="e3tb", device="cpu", has_soc=True)
    with pytest.raises(NotImplementedError, match="does not support SOC"):
        BlockStateCodec(mapper, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Device migration: skbasis/cgbasis/soc_base_matrix dicts follow .to()/.double()
# ---------------------------------------------------------------------------

def _basis_dict(module):
    d = getattr(module, "skbasis", None)
    if not isinstance(d, dict) or not d:
        d = getattr(module, "cgbasis", None)
    assert isinstance(d, dict) and d, "module exposes no basis dict"
    return d


@requires_cuda
@pytest.mark.parametrize(
    "factory",
    [
        lambda: SKHamiltonian(basis=SK_BASIS, device="cpu"),
        lambda: SKHamiltonian_old(basis=SK_BASIS, device="cpu"),
        lambda: E3Hamiltonian(basis=SK_BASIS, device="cpu"),
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


@requires_cuda
def test_skbasis_contraction_matches_cpu_after_move_to_cuda():
    """Regression for the crash line (SKHamiltonian.forward mixing a cuda
    skparam with a CPU skbasis dict): run the real forward on CPU and on a module
    moved to CUDA and require the two to agree, instead of retyping the crash line."""
    torch.manual_seed(0)
    cpu_module = SKHamiltonian(basis=SK_BASIS, dtype=torch.float64, device="cpu")
    cuda_module = SKHamiltonian(basis=SK_BASIS, dtype=torch.float64, device="cpu").to("cuda")
    cuda_module.load_state_dict(cpu_module.state_dict())

    # Fixed once and moved per device — regenerating randn() per call would draw two
    # different samples and compare unrelated inputs.
    edge_features = torch.randn(2, cpu_module.idp_sk.reduced_matrix_element, dtype=torch.float64)
    edge_vectors = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)

    def data_for(device):
        return {
            AtomicDataDict.EDGE_FEATURES_KEY: edge_features.to(device),
            AtomicDataDict.EDGE_VECTORS_KEY: edge_vectors.to(device),
        }

    cpu_out = cpu_module(data_for("cpu"))[AtomicDataDict.EDGE_FEATURES_KEY]
    cuda_out = cuda_module(data_for("cuda"))[AtomicDataDict.EDGE_FEATURES_KEY]
    assert cuda_out.device.type == "cuda"
    torch.testing.assert_close(cuda_out.cpu(), cpu_out, atol=1e-10, rtol=1e-10)


def test_double_syncs_dtype_of_basis_and_module():
    m = SKHamiltonian(basis=SK_BASIS, device="cpu")
    assert m.dtype == torch.float32
    m = m.double()
    assert m.dtype == torch.float64
    assert all(v.dtype == torch.float64 for v in m.skbasis.values())


def test_state_dict_keys_unchanged_by_migration_fix():
    # The fix must NOT register the dicts as buffers/parameters: existing production
    # checkpoints do not contain these keys and must keep loading.
    for m in (SKHamiltonian(basis=SK_BASIS, device="cpu"), SKHamiltonian_old(basis=SK_BASIS, device="cpu"),
              E3Hamiltonian(basis=SK_BASIS, device="cpu")):
        for key in m.state_dict():
            assert "skbasis" not in key
            assert "cgbasis" not in key
            assert "soc_base_matrix" not in key

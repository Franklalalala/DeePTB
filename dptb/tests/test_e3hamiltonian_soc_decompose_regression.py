from __future__ import annotations

import pytest
import torch

from dptb.data import AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.nnops.block_flow_codec import BlockStateCodec


def _two_row_data(module: E3Hamiltonian) -> tuple[dict, torch.Tensor, torch.Tensor]:
    width = int(module.idp.reduced_matrix_element)
    node_features = torch.arange(2 * width, dtype=torch.float64).reshape(2, width)
    edge_features = (
        torch.arange(2 * width, dtype=torch.float64).reshape(2, width) + 0.25
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
        ),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long
        ),
        AtomicDataDict.ATOM_TYPE_KEY: torch.zeros(2, dtype=torch.long),
        AtomicDataDict.NODE_FEATURES_KEY: node_features.clone(),
        AtomicDataDict.EDGE_FEATURES_KEY: edge_features.clone(),
    }
    return data, node_features, edge_features


def test_nontrivial_soc_decompose_preserves_legacy_statistics_pass_through():
    """The non-SOC inverse must not block LMDBDataset.E3statistics for SOC."""
    module = E3Hamiltonian(
        basis={"H": ["1s", "2p"]},
        decompose=True,
        soc=True,
        dtype=torch.float64,
        device="cpu",
    )
    data, node_features, edge_features = _two_row_data(module)

    # Unlike s-s, the p-p CG change of basis is not the identity.  Therefore
    # this fixture distinguishes a true SOC pass-through from accidentally
    # running the newly added non-SOC inverse-CG branch.
    p_basis = module.cgbasis["p-p"].reshape(9, 9)
    assert not torch.equal(p_basis, torch.eye(9, dtype=torch.float64))

    result = module(data)

    assert torch.equal(result[AtomicDataDict.NODE_FEATURES_KEY], node_features)
    assert torch.equal(result[AtomicDataDict.EDGE_FEATURES_KEY], edge_features)


def test_soc_mapper_decompose_passes_through_when_soc_kwarg_is_omitted():
    """A SOC mapper is authoritative even when the legacy caller omits soc=True."""
    mapper = OrbitalMapper(
        {"C": ["2p"]},
        method="e3tb",
        device="cpu",
        has_soc=True,
    )
    mapper.get_orbital_maps()
    mapper.get_irreps(no_parity=False)
    module = E3Hamiltonian(
        idp=mapper,
        decompose=True,
        dtype=torch.float64,
        device="cpu",
    )
    assert module.soc is False
    assert module.idp.has_soc is True
    data, node_features, edge_features = _two_row_data(module)

    # This would change under the non-SOC inverse-CG, so exact equality proves
    # that the mapper's SOC layout selected the legacy pass-through instead.
    p_basis = module.cgbasis["p-p"].reshape(9, 9)
    assert not torch.equal(p_basis, torch.eye(9, dtype=torch.float64))

    result = module(data)

    assert torch.equal(result[AtomicDataDict.NODE_FEATURES_KEY], node_features)
    assert torch.equal(result[AtomicDataDict.EDGE_FEATURES_KEY], edge_features)


def test_non_soc_decompose_is_legacy_pass_through_without_inverse_opt_in():
    mapper = OrbitalMapper(
        {"C": ["2p"]},
        method="e3tb",
        device="cpu",
        has_soc=False,
    )
    mapper.get_orbital_maps()
    mapper.get_irreps(no_parity=False)
    module = E3Hamiltonian(
        idp=mapper,
        decompose=True,
        dtype=torch.float64,
        device="cpu",
    )
    data, node_features, edge_features = _two_row_data(module)
    p_basis = module.cgbasis["p-p"].reshape(9, 9)
    assert not torch.equal(p_basis, torch.eye(9, dtype=torch.float64))

    result = module(data)

    assert torch.equal(result[AtomicDataDict.NODE_FEATURES_KEY], node_features)
    assert torch.equal(result[AtomicDataDict.EDGE_FEATURES_KEY], edge_features)


def test_inverse_cg_opt_in_rejects_soc_mapper_even_without_soc_kwarg():
    mapper = OrbitalMapper(
        {"C": ["2p"]},
        method="e3tb",
        device="cpu",
        has_soc=True,
    )

    with pytest.raises(NotImplementedError, match="non-SOC"):
        E3Hamiltonian(
            idp=mapper,
            decompose=True,
            enable_inverse_cg=True,
            dtype=torch.float64,
            device="cpu",
        )


def test_soc_decompose_rejects_a_non_soc_mapper():
    mapper = OrbitalMapper(
        {"H": ["1s", "2p"]},
        method="e3tb",
        device="cpu",
        has_soc=False,
    )
    mapper.get_orbital_maps()
    mapper.get_irreps(no_parity=False)
    module = E3Hamiltonian(
        idp=mapper,
        decompose=True,
        soc=True,
        dtype=torch.float64,
        device="cpu",
    )
    data, _, _ = _two_row_data(module)

    with pytest.raises(NotImplementedError, match="non-SOC OrbitalMapper"):
        module(data)


def test_block_state_codec_remains_the_soc_rejection_boundary():
    mapper = OrbitalMapper(
        {"C": ["2p"]},
        method="e3tb",
        device="cpu",
        has_soc=True,
    )

    with pytest.raises(NotImplementedError, match="does not support SOC"):
        BlockStateCodec(mapper, dtype=torch.float64)

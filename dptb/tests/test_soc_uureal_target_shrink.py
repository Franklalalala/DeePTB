import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BASIS_0603_SOC = {"C": "4s2p2d1f"}


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_soc_uureal_contract_is_encoded_in_mapper_and_model_paths():
    transforms = _source("dptb/data/transforms.py")
    deeptb = _source("dptb/nn/deeptb.py")
    lem_moe_v3 = _source("dptb/nn/embedding/lem_moe_v3.py")
    hamiltonian = _source("dptb/nn/hamiltonian.py")
    data_build = _source("dptb/data/build.py")
    argcheck = _source("dptb/utils/argcheck.py")
    lmdb_dataset = _source("dptb/data/dataset/lmdb_dataset.py")
    trainer = _source("dptb/nnops/trainer.py")

    assert "full_soc_prediction" in argcheck
    assert "resolve_nextham_uureal_mask" in transforms
    compact_data_build = "".join(data_build.split())
    assert (
        'full_soc_prediction=kwargs.get("full_soc_prediction",False)'
        in compact_data_build
    )
    assert "def _e3tb_soc_feature_factor" in transforms
    assert (
        "return 1 if self.nextham_uureal_mask else "
        "4 * (2 if self.soc_complex_doubling else 1)"
    ) in transforms
    assert "factor = self._e3tb_soc_feature_factor()" in transforms
    assert "spinful=self.has_soc and not self.nextham_uureal_mask" in transforms
    assert (
        "self.soc_complex_doubling and not self.nextham_uureal_mask"
    ) in transforms

    assert "nextham_uureal_mask=self.nextham_uureal_mask" in deeptb
    assert "nextham_uureal_mask=self.nextham_uureal_mask" in lem_moe_v3
    assert "nextham_uureal_mask=self.nextham_uureal_mask" in hamiltonian
    assert "if self.soc and not self.nextham_uureal_mask" in hamiltonian
    assert "target_rme != full_rme" in lmdb_dataset
    assert "return tensor" in lmdb_dataset
    assert "def _loss_kwargs" in trainer
    assert "kwargs.update(common_options)" in trainer


def test_full_soc_prediction_flag_overrides_compact_mask():
    from dptb.utils.soc_target import resolve_nextham_uureal_mask

    assert resolve_nextham_uureal_mask(
        nextham_uureal_mask=True,
        full_soc_prediction=True,
    ) is False
    assert resolve_nextham_uureal_mask(
        nextham_uureal_mask=True,
        full_soc_prediction=False,
    ) is True


_MISSING_DEPS = [
    name
    for name in ("torch", "e3nn", "ase", "torch_runstats")
    if importlib.util.find_spec(name) is None
]

if not _MISSING_DEPS:
    import torch

    from dptb.data.dataset.lmdb_dataset import _expand_soc_uureal_compact
    from dptb.data.transforms import OrbitalMapper


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_orbital_mapper_soc_uureal_target_uses_single_directed_real_block():
    mapper = OrbitalMapper(
        BASIS_0603_SOC,
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=True,
    )

    assert mapper.full_basis_norb == 27
    assert mapper.reduced_matrix_element == 729
    assert mapper.mask_uureal.numel() == 729
    assert int(mapper.mask_uureal.sum().item()) == 729
    assert mapper.get_irreps(no_parity=False).dim == 729


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_orbital_mapper_full_soc_target_keeps_eight_blocks():
    mapper = OrbitalMapper(
        BASIS_0603_SOC,
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=False,
    )

    assert mapper.full_basis_norb == 27
    assert mapper.reduced_matrix_element == 5832
    assert mapper.get_irreps(no_parity=False).dim == 5832


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_orbital_mapper_full_soc_flag_overrides_uureal_mask():
    mapper = OrbitalMapper(
        BASIS_0603_SOC,
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=True,
        full_soc_prediction=True,
    )

    assert mapper.nextham_uureal_mask is False
    assert mapper.soc_uureal_target is False
    assert mapper.full_basis_norb == 27
    assert mapper.reduced_matrix_element == 5832
    assert mapper.get_irreps(no_parity=False).dim == 5832


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_compact_lmdb_features_stay_compact_for_reduced_uureal_target():
    compact = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    keep_mask_from_reduced_mapper = torch.ones(2, dtype=torch.bool)

    actual = _expand_soc_uureal_compact(
        compact,
        {
            "soc_uureal_compact": True,
            "soc_uureal_keep": 2,
            "soc_uureal_full_rme": 4,
        },
        field_name="edge_features",
        keep_mask=keep_mask_from_reduced_mapper,
    )

    assert actual is compact

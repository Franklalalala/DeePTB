import importlib.util

import pytest

_MISSING_DEPS = [
    name
    for name in ("torch", "e3nn", "torch_scatter")
    if importlib.util.find_spec(name) is None
]

if not _MISSING_DEPS:
    import torch

    from dptb.data import AtomicDataDict
    from dptb.data.interfaces.ham_to_feature import feature_to_block
    from dptb.data.transforms import OrbitalMapper
    from dptb.nnops.loss import Loss


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_feature_to_block_accepts_tensor_atomic_numbers_on_edges():
    idp = OrbitalMapper({"H": "1s"}, method="e3tb")
    data = {
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.tensor([[1], [1]]),
        AtomicDataDict.NODE_FEATURES_KEY: torch.zeros((2, 1)),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.ones((1, 1)),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[0], [1]], dtype=torch.long),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.zeros((1, 3), dtype=torch.long),
    }

    blocks = feature_to_block(data, idp)

    assert "0_1_0_0_0" in blocks


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_hamil_fullh_ao_mae_matches_next_ham_gauge_formula():
    pred = torch.tensor([[2.0, 4.0]])
    target = torch.tensor([[1.0, 1.0]])
    overlap = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0]])
    loss = Loss("hamil_fullh_ao_mae", gauge_shift=True)

    actual = loss(pred, target, overlap=overlap, mask=mask)

    assert torch.allclose(actual, torch.tensor(1.0))


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_hamil_fullh_ao_mae_applies_mask_before_element_average():
    pred = torch.tensor([[1.0, 10.0, 3.0]])
    target = torch.tensor([[0.0, 20.0, 5.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    loss = Loss("hamil_fullh_ao_mae")

    actual = loss(pred, target, mask=mask)

    assert torch.allclose(actual, torch.tensor(1.5))

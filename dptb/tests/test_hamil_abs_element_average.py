import pytest
import importlib.util

_MISSING_DEPS = [
    name
    for name in ("torch", "e3nn", "torch_scatter")
    if importlib.util.find_spec(name) is None
]

if not _MISSING_DEPS:
    import torch
    from dptb.data import AtomicDataDict
    from dptb.nnops.loss import HamilLossAbs


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_hamil_abs_default_total_loss_uses_element_average():
    class FakeIdp:
        mask_to_nrme = torch.tensor([[True, True, False, False]])
        mask_to_erme = torch.tensor([[True, True, True, True]])

    loss = HamilLossAbs(idp=FakeIdp())
    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.NODE_FEATURES_KEY: torch.zeros((1, 4)),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.zeros((1, 4)),
    }
    ref_data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.NODE_FEATURES_KEY: torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.tensor([[3.0, 3.0, 3.0, 3.0]]),
    }

    actual = loss(data, ref_data)
    expected_l1 = torch.tensor(14.0 / 6.0)
    expected_rmse = torch.sqrt(torch.tensor(38.0 / 6.0))
    expected = 0.5 * (expected_l1 + expected_rmse)

    assert torch.allclose(actual, expected)

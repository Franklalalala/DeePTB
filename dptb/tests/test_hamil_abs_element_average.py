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
    from dptb.nnops.loss import HamilLossAbs, Loss


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def _sample_hamil_abs_data():
    class FakeIdp:
        mask_to_nrme = torch.tensor([[True, True, False, False]])
        mask_to_erme = torch.tensor([[True, True, True, True]])

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
    return FakeIdp(), data, ref_data


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_hamil_abs_default_total_loss_keeps_component_average():
    idp, data, ref_data = _sample_hamil_abs_data()
    loss = HamilLossAbs(idp=idp)

    actual = loss(data, ref_data)
    expected_onsite = torch.tensor(1.0)
    expected_hopping = torch.tensor(3.0)
    expected = 0.5 * (expected_onsite + expected_hopping)

    assert torch.allclose(actual, expected)


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_hamil_abs_element_avg_uses_element_average():
    idp, data, ref_data = _sample_hamil_abs_data()
    loss = Loss("hamil_abs_element_avg", idp=idp)

    actual = loss(data, ref_data)
    expected_l1 = torch.tensor(14.0 / 6.0)
    expected_rmse = torch.sqrt(torch.tensor(38.0 / 6.0))
    expected = 0.5 * (expected_l1 + expected_rmse)

    assert torch.allclose(actual, expected)


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_hamil_abs_respects_uureal_only_masks():
    class FakeUuRealIdp:
        # Mimic NextHAM SOC uu.real-only masking: one supervised channel out of
        # an eight-channel spin/complex slice.
        mask_to_nrme = torch.tensor(
            [[True, False, False, False, False, False, False, False]]
        )
        mask_to_erme = mask_to_nrme

    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.NODE_FEATURES_KEY: torch.zeros((1, 8)),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.zeros((1, 8)),
    }
    ref_data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.NODE_FEATURES_KEY: torch.tensor(
            [[2.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]]
        ),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.tensor(
            [[4.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]]
        ),
    }

    loss = HamilLossAbs(idp=FakeUuRealIdp())
    actual = loss(data, ref_data)

    assert torch.allclose(actual, torch.tensor(3.0))

import importlib.util

import pytest


_MISSING_DEPS = [
    name
    for name in ("torch", "e3nn", "torch_scatter", "lmdb")
    if importlib.util.find_spec(name) is None
]

if not _MISSING_DEPS:
    import torch

    from dptb.data.dataset.lmdb_dataset import _expand_soc_uureal_compact


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_expand_soc_uureal_compact_fills_full_soc_channels():
    keep = torch.tensor([True, False, True, False])
    compact = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    actual = _expand_soc_uureal_compact(
        compact,
        {
            "soc_uureal_compact": True,
            "soc_uureal_keep": keep,
            "soc_uureal_full_rme": 4,
        },
        field_name="node_features",
    )

    expected = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [3.0, 0.0, 4.0, 0.0],
        ]
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_expand_soc_uureal_compact_uses_type_mapper_mask_with_keep_count():
    keep = torch.tensor([True, False, True, False])

    actual = _expand_soc_uureal_compact(
        torch.tensor([[5.0, 6.0]]),
        {
            "soc_uureal_compact": True,
            "soc_uureal_keep": 2,
            "soc_uureal_full_rme": 4,
        },
        field_name="edge_h0",
        keep_mask=keep,
    )

    assert torch.equal(actual, torch.tensor([[5.0, 0.0, 6.0, 0.0]]))


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_expand_soc_uureal_compact_accepts_full_width_features():
    full = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    actual = _expand_soc_uureal_compact(
        full,
        {
            "soc_uureal_compact": True,
            "soc_uureal_keep": torch.tensor([True, False, True, False]),
            "soc_uureal_full_rme": 4,
        },
        field_name="edge_features",
    )

    assert actual is full


@pytest.mark.skipif(
    bool(_MISSING_DEPS),
    reason=f"missing runtime dependencies: {', '.join(_MISSING_DEPS)}",
)
def test_expand_soc_uureal_compact_rejects_inconsistent_width():
    with pytest.raises(ValueError, match="node_h0"):
        _expand_soc_uureal_compact(
            torch.zeros((2, 3)),
            {
                "soc_uureal_compact": True,
                "soc_uureal_keep": torch.tensor([True, False, True, False]),
                "soc_uureal_full_rme": 4,
            },
            field_name="node_h0",
        )

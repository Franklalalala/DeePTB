"""Feature decoding accepts NumPy atom identifiers across NumPy versions."""

import numpy as np
import torch

from dptb.data import AtomicDataDict
from dptb.data.interfaces.ham_to_feature import feature_to_block
from dptb.data.transforms import OrbitalMapper


def test_feature_to_block_numpy_atomic_numbers(monkeypatch):
    mapper = OrbitalMapper(basis={"H": ["1s"]}, method="e3tb")
    transform_bond = mapper.transform_bond

    def tensor_bonds(source, target):
        # NumPy versions differ when an ndarray is indexed by a torch tensor.
        assert isinstance(source, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        return transform_bond(source, target)

    monkeypatch.setattr(mapper, "transform_bond", tensor_bonds)
    data = {
        AtomicDataDict.ATOMIC_NUMBERS_KEY: np.array([[1], [1]], dtype=np.int64),
        AtomicDataDict.NODE_FEATURES_KEY: torch.tensor([[1.0], [2.0]]),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.tensor([[3.0], [4.0]]),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]]),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.zeros((2, 3), dtype=torch.long),
    }
    blocks = feature_to_block(data, mapper)
    expected = {"0_0_0_0_0": 1.0, "1_1_0_0_0": 2.0,
                "0_1_0_0_0": 3.0, "1_0_0_0_0": 4.0}
    assert blocks.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(blocks[key], np.array([[value]], dtype=np.float32))

"""``group_rows`` reproduces the ``np.unique`` + per-group mask idiom it replaces in the prepare phases."""
import numpy as np
import torch

from dptb.nacf.topology import device_array, group_rows


def reference(keys):
    unique, inverse = np.unique(keys, return_inverse=True)
    return [(int(k), np.flatnonzero(inverse == number)) for number, k in enumerate(unique)]


def test_group_rows_matches_unique_and_masks():
    rng = np.random.default_rng(11)
    for n, span in ((0, 1), (1, 1), (7, 3), (1000, 5), (20000, 400), (5000, 1)):
        keys = rng.integers(0, span, size=n).astype(np.int64) * 7 + 3
        got, want = group_rows(keys), reference(keys)
        assert [k for k, _ in got] == [k for k, _ in want]
        for (_, a), (_, b) in zip(got, want):
            np.testing.assert_array_equal(a, b)
    keys = np.array([5, 5, 5])
    assert len(group_rows(keys)) == 1 and group_rows(keys)[0][0] == 5 and group_rows(keys)[0][1].tolist() == [0, 1, 2]


def test_device_array_copies_and_keeps_dtype():
    src = np.arange(6).reshape(2, 3)
    t = device_array(src, device=torch.device('cpu'), dtype=torch.long)
    src[0, 0] = 99
    assert t[0, 0].item() == 0 and t.dtype == torch.long and t.shape == (2, 3)
    f = device_array([[1.5, 2.5]], device='cpu', dtype=torch.float64)
    assert f.dtype == torch.float64 and f.tolist() == [[1.5, 2.5]]
    if torch.cuda.is_available():
        g = device_array(np.asfortranarray(src), device=torch.device('cuda'), dtype=torch.long)
        assert g.is_cuda and g.tolist() == src.tolist()

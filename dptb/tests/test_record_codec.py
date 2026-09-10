"""Compressed and legacy LMDB records must share the same load semantics."""

import pickle, zlib
import numpy as np
import pytest
from dptb.data.dataset.lmdb_dataset import _loads_with_numpy2_compat


@pytest.mark.parametrize("kind", ["plain", "zlib", "zstd"])
def test_record_roundtrip(kind):
    sample = {
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "node_features": np.array([[1.0, 2.0]], dtype=np.float32),
    }
    data = pickle.dumps(sample, protocol=4)
    if kind == "zlib":
        data = b"ZL1\0" + zlib.compress(data)
    if kind == "zstd":
        zstd = pytest.importorskip("zstandard")
        data = b"ZST1" + zstd.ZstdCompressor().compress(data)
    got = _loads_with_numpy2_compat(data)
    for key in sample:
        np.testing.assert_array_equal(got[key], sample[key])


def test_corrupt_compressed_record_fails():
    with pytest.raises(Exception):
        _loads_with_numpy2_compat(b"ZL1\0not a stream")


def test_zstd_without_python_package(monkeypatch):
    import sys, base64

    monkeypatch.setitem(sys.modules, "zstandard", None)
    fixture = base64.b64decode(
        "WlNUMSi1L/0gHekAAIAElRIAAAAAAAAAfZSMAXiUXZQoSwFLAksDZXMu"
    )
    assert _loads_with_numpy2_compat(fixture) == {"x": [1, 2, 3]}


def test_metadata_and_training_entrypoints_decode_same_record(tmp_path):
    import lmdb
    from types import SimpleNamespace
    from dptb.data.dataset.lmdb_dataset import _read_lmdb_entry, LMDBDataset

    path = tmp_path / "records.lmdb"
    raw = {"node_features": np.array([[1.0, 2.0]], dtype=np.float32)}
    env = lmdb.open(str(path), map_size=1024 * 1024)
    with env.begin(write=True) as txn:
        txn.put((0).to_bytes(4, "big"), b"ZL1\0" + zlib.compress(pickle.dumps(raw)))
    env.close()
    got = _read_lmdb_entry(str(path), 0)
    np.testing.assert_array_equal(got["node_features"], raw["node_features"])

    env = lmdb.open(str(path), readonly=True, lock=False)
    reader = SimpleNamespace(
        index_map=[0], _lmdb_path_map=[str(path)], _get_lmdb_env=lambda _: env
    )
    try:
        trained = LMDBDataset._load_data_dict(reader, 0)
        np.testing.assert_array_equal(trained["node_features"], got["node_features"])
    finally:
        env.close()

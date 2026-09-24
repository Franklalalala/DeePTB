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


ZST1_FIXTURE = "WlNUMSi1L/0gHekAAIAElRIAAAAAAAAAfZSMAXiUXZQoSwFLAksDZXMu"


def test_zstd_falls_back_to_system_libzstd_resolved_once_per_process(monkeypatch):
    """Without the zstandard python package, decoding falls back to the system
    libzstd via ctypes and still decodes correctly; find_library forks
    ldconfig, so it must resolve once per process, not once per record."""
    import base64
    import ctypes.util
    import sys
    from dptb.data.dataset import record_codec

    real_find_library = ctypes.util.find_library
    if real_find_library("zstd") is None:
        pytest.skip("system libzstd is not available")

    monkeypatch.setitem(sys.modules, "zstandard", None)
    monkeypatch.setattr(record_codec, "_ZSTANDARD_MODULE", None)  # re-probe: package absent
    monkeypatch.setattr(record_codec, "_ZSTD_LIBRARY", None)  # re-resolve the ctypes handle
    calls = []
    monkeypatch.setattr(ctypes.util, "find_library",
                        lambda name: calls.append(name) or real_find_library(name))

    fixture = base64.b64decode(ZST1_FIXTURE)
    for _ in range(50):
        assert record_codec.loads_record(fixture) == {"x": [1, 2, 3]}
    assert calls == ["zstd"]


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

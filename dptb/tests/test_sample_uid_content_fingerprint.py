"""Regression tests for content-anchored LMDB shard identity (P2-2).

``sample_uid`` packs ``(shard_ordinal << 32) | row_id`` into a record's stable
per-graph identity (``LMDBDataset._compute_sample_uid``), and per-graph SEEDED
priors (``projected_te``, the PR#31 ``tied_irrep_gaussian``) fold that uid into
their epsilon substream key (see ``dptb/nnops/flow.py``). Before this fix, the
shard ordinal hashed the shard's on-disk REALPATH
(``_stable_shard_ordinal(realpath)``), so relocating a byte-identical LMDB --
a container/multi-node remount, an archive restore, a different working
directory -- silently changed every record's ``sample_uid``, and hence its
seeded prior epsilon, and hence validation metrics, even though nothing about
the DATA changed. The fix (``_shard_content_fingerprint`` +
``_build_shard_uid_offsets``) hashes shard CONTENT (basename + entry count +
first record) instead of the path, so relocation is a no-op for identity.

Fixtures mirror the tmp_path real-LMDB pattern from
``dptb/tests/test_residual_from_full_h_provenance.py``; copied (not imported)
per that file's convention: ``dptb/tests`` is not a package, so pytest imports
sibling test modules by basename, not as ``dptb.tests.*``.
"""
from __future__ import annotations

import copy
import os
import pickle
import shutil

import lmdb
import numpy as np
import pytest

from dptb.data import _keys
from dptb.data.build import DatasetBuilder
from dptb.data.dataset.lmdb_dataset import (
    _build_shard_uid_offsets,
    _shard_content_fingerprint,
    _stable_shard_ordinal,
)
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    RAW_HAMILTONIAN_SAMPLE_SCHEMA,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
)


# ---------------------------------------------------------------------------
# Low-level LMDB helpers (no AtomicData/schema involvement -- these write
# arbitrary picklable payloads directly, for tests that only exercise
# _shard_content_fingerprint / _build_shard_uid_offsets).
# ---------------------------------------------------------------------------
def _write_raw_lmdb(path, values) -> None:
    env = lmdb.open(str(path), map_size=1 << 20, subdir=True)
    try:
        with env.begin(write=True) as txn:
            for row_id, value in enumerate(values):
                txn.put(int(row_id).to_bytes(4, "big"), pickle.dumps(value))
    finally:
        env.close()


# ---------------------------------------------------------------------------
# End-to-end fixture (AtomicData-valid record), copied from
# test_residual_from_full_h_provenance.py's _raw_absolute_full_h_record /
# _build_dataset for the one test that must go through the real loader.
# ---------------------------------------------------------------------------
def _raw_absolute_full_h_record() -> dict:
    h_blocks = {
        "0_0_0_0_0": np.asarray([[10.0]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[12.0]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[3.0]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[3.0]], dtype=np.float32),
    }
    h0_blocks = {
        "0_0_0_0_0": np.asarray([[9.0]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[11.0]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[2.5]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[2.5]], dtype=np.float32),
    }
    return {
        _keys.CELL_KEY: np.eye(3, dtype=np.float32) * 8.0,
        _keys.POSITIONS_KEY: np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1, 1], dtype=np.int64),
        _keys.PBC_KEY: np.asarray([False, False, False]),
        "case_id": "h2",
        _keys.EDGE_INDEX_KEY: np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        _keys.EDGE_CELL_SHIFT_KEY: np.zeros((2, 3), dtype=np.float32),
        "hamiltonian": h_blocks,
        "hamiltonian_0": h0_blocks,
        SAMPLE_SCHEMA_KEY: RAW_HAMILTONIAN_SAMPLE_SCHEMA,
        TARGET_SEMANTICS_KEY: ABSOLUTE_FULL_H_SEMANTICS,
        TARGET_SOURCE_KEY: "raw_hamiltonian",
    }


def _build_dataset(root, record, *, name, **kwargs):
    lmdb_path = os.path.join(str(root), f"{name}.lmdb")
    env = lmdb.open(lmdb_path, map_size=1 << 20, subdir=True)
    try:
        with env.begin(write=True) as txn:
            txn.put((0).to_bytes(4, "big"), pickle.dumps(record))
    finally:
        env.close()
    return DatasetBuilder()(
        root=str(root),
        r_max=2.0,
        type="LMDBDataset",
        prefix=name,
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_H0=True,
        **kwargs,
    )


# ===========================================================================
# 1. Core regression: a byte-identical shard relocated to a new path (same
#    basename, different/unrelated parent directory tree) reproduces the SAME
#    sample_uid for the same row -- through the real DatasetBuilder/loader.
# ===========================================================================
def test_relocated_byte_identical_shard_reproduces_sample_uid(tmp_path):
    old_root = tmp_path / "old_mount"
    new_root = tmp_path / "new_mount" / "nested" / "elsewhere"
    old_root.mkdir()
    new_root.mkdir(parents=True)

    record = _raw_absolute_full_h_record()
    ds_old = _build_dataset(old_root, record, name="shard")
    uid_old = int(ds_old.get(0)[_keys.SAMPLE_UID_KEY].item())

    # Relocate: copy the whole shard directory verbatim (as a container
    # remount / archive restore / cwd change would), keeping the same
    # basename ("shard.lmdb") but under a wholly different, deeper parent path.
    shutil.copytree(old_root / "shard.lmdb", new_root / "shard.lmdb")
    ds_new = DatasetBuilder()(
        root=str(new_root),
        r_max=2.0,
        type="LMDBDataset",
        prefix="shard",
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_H0=True,
    )
    uid_new = int(ds_new.get(0)[_keys.SAMPLE_UID_KEY].item())

    assert uid_new == uid_old, (
        "relocating a byte-identical LMDB shard must not change sample_uid "
        "(and therefore must not change any SEEDED prior's validation epsilon)"
    )


# ===========================================================================
# 2. Pathological configuration: the SAME shard content, mounted twice under
#    a dataset's roots (two distinct realpaths), fails fast and names both
#    paths rather than silently aliasing their identities.
#
#    _build_shard_uid_offsets is the exact function LMDBDataset.__init__
#    delegates to; exercised directly here so the test is deterministic and
#    does not depend on constructing a full dataset with a multi-root/
#    wildcard `root` (which -- separately from this fix -- goes through
#    LMDBDataset's inherited `download()`/`raw_paths` machinery and is not a
#    safe construction to drive through the public API on every OS).
# ===========================================================================
def test_same_content_mounted_twice_fails_closed(tmp_path):
    root_a = tmp_path / "mount_a"
    root_b = tmp_path / "mount_b"
    root_a.mkdir()
    root_b.mkdir()

    _write_raw_lmdb(root_a / "dup.lmdb", [{"case": "only"}])
    shutil.copytree(root_a / "dup.lmdb", root_b / "dup.lmdb")

    path_a = os.path.realpath(str(root_a / "dup.lmdb"))
    path_b = os.path.realpath(str(root_b / "dup.lmdb"))
    assert path_a != path_b  # distinct realpaths, identical content

    with pytest.raises(ValueError) as excinfo:
        _build_shard_uid_offsets([path_a, path_b])

    message = str(excinfo.value)
    assert "collision" in message
    # The message names both paths via repr() (matching the pre-existing
    # collision-message convention), which backslash-escapes on Windows.
    assert repr(path_a) in message
    assert repr(path_b) in message


def test_distinct_content_shards_do_not_collide(tmp_path):
    """No-regression: distinct shards (different basenames/content) build
    offsets cleanly, one ordinal per shard, no exception."""
    root_a = tmp_path / "mount_a"
    root_b = tmp_path / "mount_b"
    root_a.mkdir()
    root_b.mkdir()
    _write_raw_lmdb(root_a / "alpha.lmdb", [{"case": "a"}])
    _write_raw_lmdb(root_b / "beta.lmdb", [{"case": "b"}])

    path_a = os.path.realpath(str(root_a / "alpha.lmdb"))
    path_b = os.path.realpath(str(root_b / "beta.lmdb"))
    offsets = _build_shard_uid_offsets([path_a, path_b])

    assert set(offsets) == {path_a, path_b}
    assert offsets[path_a] != offsets[path_b]


# ===========================================================================
# 3. Direct fingerprint-function checks: same content (+ same basename) at
#    different parents fingerprints identically; different content, different
#    entry counts, and different (but both-empty) shards all fingerprint
#    differently.
# ===========================================================================
def test_fingerprint_same_content_same_basename_different_parent(tmp_path):
    dir_a = tmp_path / "parent_x"
    dir_b = tmp_path / "parent_y" / "deep" / "path"
    dir_a.mkdir()
    dir_b.mkdir(parents=True)

    _write_raw_lmdb(dir_a / "shard.lmdb", [{"payload": [1, 2, 3]}, {"payload": "two"}])
    shutil.copytree(dir_a / "shard.lmdb", dir_b / "shard.lmdb")

    fp_a = _shard_content_fingerprint(str(dir_a / "shard.lmdb"))
    fp_b = _shard_content_fingerprint(str(dir_b / "shard.lmdb"))
    assert fp_a == fp_b
    # ... and the derived 31-bit ordinal is therefore identical too.
    assert _stable_shard_ordinal(fp_a) == _stable_shard_ordinal(fp_b)


def test_fingerprint_differs_on_first_record_content(tmp_path):
    dir_a = tmp_path / "content_a"
    dir_b = tmp_path / "content_b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_raw_lmdb(dir_a / "shard.lmdb", [{"payload": "same-basename-same-count"}])
    _write_raw_lmdb(dir_b / "shard.lmdb", [{"payload": "DIFFERENT-value-bytes"}])

    fp_a = _shard_content_fingerprint(str(dir_a / "shard.lmdb"))
    fp_b = _shard_content_fingerprint(str(dir_b / "shard.lmdb"))
    assert fp_a != fp_b


def test_fingerprint_differs_on_entry_count(tmp_path):
    dir_a = tmp_path / "one_entry"
    dir_b = tmp_path / "two_entries"
    dir_a.mkdir()
    dir_b.mkdir()
    value = {"payload": "identical-first-record"}
    _write_raw_lmdb(dir_a / "shard.lmdb", [value])
    _write_raw_lmdb(dir_b / "shard.lmdb", [value, copy.deepcopy(value)])

    fp_a = _shard_content_fingerprint(str(dir_a / "shard.lmdb"))
    fp_b = _shard_content_fingerprint(str(dir_b / "shard.lmdb"))
    assert fp_a != fp_b


def test_fingerprint_differs_between_distinct_empty_shards(tmp_path):
    """Two EMPTY shards (0 entries) must not alias: entries and first-record
    components are both trivially empty for ANY empty shard, so the basename
    is the only thing that can (and must) keep two unrelated empty shards --
    e.g. two independent datasets' unpopulated placeholder shards -- apart."""
    dir_a = tmp_path / "empty_a"
    dir_b = tmp_path / "empty_b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_raw_lmdb(dir_a / "alpha.lmdb", [])
    _write_raw_lmdb(dir_b / "beta.lmdb", [])

    fp_a = _shard_content_fingerprint(str(dir_a / "alpha.lmdb"))
    fp_b = _shard_content_fingerprint(str(dir_b / "beta.lmdb"))
    assert fp_a != fp_b


def test_fingerprint_same_for_identically_named_empty_shards(tmp_path):
    """Companion to the above: two empty shards that DO share a basename
    (relocated empty placeholder) fingerprint identically, consistent with
    the relocation invariant applying even in the degenerate 0-entry case."""
    dir_a = tmp_path / "empty_parent_x"
    dir_b = tmp_path / "empty_parent_y"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_raw_lmdb(dir_a / "placeholder.lmdb", [])
    _write_raw_lmdb(dir_b / "placeholder.lmdb", [])

    fp_a = _shard_content_fingerprint(str(dir_a / "placeholder.lmdb"))
    fp_b = _shard_content_fingerprint(str(dir_b / "placeholder.lmdb"))
    assert fp_a == fp_b

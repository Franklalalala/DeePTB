"""Provenance gate for ``require_residual_from_full_h_target`` at the loader.

Regression cover for the reviewer finding: a raw LMDB record can declare a valid
``absolute_full_h`` target AND simultaneously carry converter/compact provenance
markers (e.g. ``blockwise_target_mode="already-delta"`` +
``soc_uureal_compact=True``).  ``residual_ao_block_ode`` materializes its residual
endpoint ``D1 = raw H - H0`` on the fly, so such a masquerade would get H0 double
-subtracted into a wrong delta target.  The flow-side masquerade guard cannot
catch this because the loader only forwards those metadata fields to the flow
when ``require_uureal_block_ode=True``; a ``require_residual_from_full_h_target``
load drops them.  The gate therefore has to live at the loader,
``assert_residual_from_full_h_target_contract``, BEFORE the subtraction.

The record/build fixtures mirror the tmp_path real-LMDB pattern from
``dptb/tests/test_residual_ao_block_ode.py`` (section 7).  They are copied here
(rather than imported) so this file stays self-contained: ``dptb/tests`` is not a
package, so pytest imports sibling test modules by basename, not as
``dptb.tests.*``.
"""
from __future__ import annotations

import copy
import pickle

import lmdb
import numpy as np
import pytest
import torch

from dptb.data import _keys
from dptb.data.build import DatasetBuilder
from dptb.data.dataset.lmdb_dataset import (
    _shard_content_fingerprint,
    _stable_shard_ordinal,
    assert_absolute_full_h_target_contract,
    assert_residual_from_full_h_target_contract,
)
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    DEDICATED_PHYSICAL_H0_SOURCE,
    PHYSICAL_H0_SOURCE_FINGERPRINT_KEY,
    PHYSICAL_H0_SOURCE_KEY,
    RAW_HAMILTONIAN_SAMPLE_SCHEMA,
    RAW_PHYSICAL_H0_SOURCE,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
)


# The seven converter/compact provenance markers the loader gate must reject,
# each paired with a plausible value a genuine converter product would store.
CONVERTER_MARKERS = {
    "blockwise_spatial_schema": "deeptb.blockwise_spatial/v1",
    "blockwise_target_mode": "already-delta",
    "soc_uureal_compact": True,
    "soc_uureal_full_rme": 8,
    "soc_uureal_keep": 1,
    "blockwise_source_target_feature_width": 8,
    "blockwise_source_h0_feature_width": 8,
}

# The reviewer's exact three-marker repro subset.
REVIEWER_MARKERS = {
    "blockwise_target_mode": "already-delta",
    "blockwise_spatial_schema": "deeptb.blockwise_spatial/v1",
    "soc_uureal_compact": True,
}


# ---------------------------------------------------------------------------
# Fixtures (copied minimally from test_residual_ao_block_ode.py, section 7)
# ---------------------------------------------------------------------------
def _raw_absolute_full_h_record() -> dict:
    """A raw H2 record declaring absolute_full_h semantics (A/B share this).

    Onsite H=10/12 over H0=9/11 and offsite H=3 over H0=2.5, so a single
    ``raw H - H0`` subtraction yields node delta [1, 1] and edge delta [.5, .5].
    """
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


def _build_dataset(tmp_path, record, *, name, **kwargs):
    lmdb_path = tmp_path / f"{name}.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=True)
    try:
        with env.begin(write=True) as txn:
            txn.put((0).to_bytes(4, "big"), pickle.dumps(record))
    finally:
        env.close()
    return DatasetBuilder()(
        root=str(tmp_path),
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
# 1. Reviewer's exact repro: absolute_full_h + 3 converter markers -> RAISE
# ===========================================================================
def test_loader_rejects_reviewer_masquerade_naming_the_keys(tmp_path):
    """(1) The reviewer's exact repro: a valid absolute_full_h raw record that
    ALSO carries {blockwise_target_mode, blockwise_spatial_schema,
    soc_uureal_compact} (Hamiltonian onsite 10 over H0 6) must fail closed at the
    loader, before any H0 subtraction, and the error must name the markers."""
    record = _raw_absolute_full_h_record()
    # Honour the reviewer's literal "Hamiltonian=10 / H0=6": if the gate were
    # absent, H0 (6) would be subtracted a second time into a wrong delta.
    record["hamiltonian"]["0_0_0_0_0"] = np.asarray([[10.0]], dtype=np.float32)
    record["hamiltonian_0"]["0_0_0_0_0"] = np.asarray([[6.0]], dtype=np.float32)
    record.update(copy.deepcopy(REVIEWER_MARKERS))

    dataset = _build_dataset(
        tmp_path,
        record,
        name="b-reviewer-masquerade",
        residual_hamiltonian=True,
        require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError) as excinfo:
        dataset.get(0)

    message = str(excinfo.value)
    # The message names every offending key...
    for key in REVIEWER_MARKERS:
        assert key in message, f"expected marker {key!r} named in: {message!r}"
    # ...and conveys the converter-product-masquerade intent for this mode.
    assert "residual_ao_block_ode" in message
    assert "masquerad" in message


# ===========================================================================
# 2. Each single marker alone also rejects (all 7)
# ===========================================================================
@pytest.mark.parametrize("marker,value", sorted(CONVERTER_MARKERS.items()))
def test_loader_rejects_each_single_converter_marker(tmp_path, marker, value):
    """(2) An otherwise-clean absolute_full_h record bearing ONLY one converter
    marker still fails closed at the loader, naming that marker."""
    record = _raw_absolute_full_h_record()
    record[marker] = value

    dataset = _build_dataset(
        tmp_path,
        record,
        name=f"b-single-marker-{marker}",
        residual_hamiltonian=True,
        require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError, match=marker):
        dataset.get(0)


# ===========================================================================
# 3. A clean record still loads: delta == raw - H0 (no regression)
# ===========================================================================
def test_loader_clean_record_still_materializes_residual(tmp_path):
    """(3) No-regression: a marker-free absolute_full_h record loads and its
    materialized delta equals raw - H0 exactly (node [1, 1], edge [.5, .5]),
    with the physical-H0 blocks attached."""
    dataset = _build_dataset(
        tmp_path,
        _raw_absolute_full_h_record(),
        name="b-clean-residual-from-full-h",
        residual_hamiltonian=True,
        require_residual_from_full_h_target=True,
    )
    sample = dataset.get(0)
    torch.testing.assert_close(
        sample[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([1.0, 1.0])
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([0.5, 0.5])
    )
    torch.testing.assert_close(
        sample[_keys.NODE_H0_BLOCKS_KEY].flatten(), torch.tensor([9.0, 11.0])
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_H0_BLOCKS_KEY].flatten(), torch.tensor([2.5, 2.5])
    )


# ===========================================================================
# 4. Direct unit call of the contract fn on a marker-bearing dict raises
# ===========================================================================
def test_contract_fn_direct_call_rejects_marker_bearing_dict():
    """(4) Calling the contract directly on an otherwise-valid record dict that
    carries a converter marker raises ValueError and names the marker."""
    record = _raw_absolute_full_h_record()
    record["soc_uureal_compact"] = True
    with pytest.raises(ValueError) as excinfo:
        assert_residual_from_full_h_target_contract(record)
    assert "soc_uureal_compact" in str(excinfo.value)


@pytest.mark.parametrize("marker,value", sorted(CONVERTER_MARKERS.items()))
def test_contract_fn_direct_call_rejects_each_marker_minimal_dict(marker, value):
    """(4b) The gate fires ahead of the schema/semantics checks, so even a
    minimal marker-only dict is rejected by name (converter products fail fast)."""
    with pytest.raises(ValueError, match=marker):
        assert_residual_from_full_h_target_contract({marker: value})


# ===========================================================================
# H7: residual_shrink_policy H0-quality gate (P3).  The frozen source/shape
# contracts always apply; the magnitude-shrink heuristic is now a configurable
# policy (error / warn / off) with a tunable min_residual_shrink ratio.
# ===========================================================================
import logging

from dptb.data.dataset.lmdb_dataset import build_residual_hamiltonian_target_blocks


def _shrink_inputs(h, h0):
    """A raw (data_dict, blocks) pair with one on-site block: H over H0."""
    data_dict = {"hamiltonian_0": {"0_0_0_0_0": np.asarray([[h0]], dtype=np.float32)}}
    blocks = {"0_0_0_0_0": np.asarray([[h]], dtype=np.float32)}
    return data_dict, blocks


def test_h7_shrink_policy_error_raises_on_non_shrinking_record():
    """error (default): a legit-but-non-shrinking record (H=1, H0=0 -> D=1) fails
    closed, and the message names the H0-quality policy gate."""
    data_dict, blocks = _shrink_inputs(1.0, 0.0)
    with pytest.raises(RuntimeError) as excinfo:
        build_residual_hamiltonian_target_blocks(data_dict, blocks, shrink_policy="error")
    message = str(excinfo.value)
    assert "H0-quality" in message
    assert "residual_shrink_policy" in message


def test_h7_shrink_policy_warn_loads_and_logs_and_keeps_delta(caplog):
    """warn: the same non-shrinking record LOADS (delta == H - H0 == 1) and logs the
    diagnostic instead of raising."""
    data_dict, blocks = _shrink_inputs(1.0, 0.0)
    with caplog.at_level(logging.WARNING, logger="dptb.data.dataset.lmdb_dataset"):
        delta = build_residual_hamiltonian_target_blocks(data_dict, blocks, shrink_policy="warn")
    assert float(delta["0_0_0_0_0"][0, 0]) == pytest.approx(1.0)
    assert any(
        "residual_shrink_policy" in rec.getMessage() and rec.levelno == logging.WARNING
        for rec in caplog.records
    )


def test_h7_shrink_policy_off_loads_silently(caplog):
    """off: the heuristic is skipped entirely -- loads with no warning."""
    data_dict, blocks = _shrink_inputs(1.0, 0.0)
    with caplog.at_level(logging.WARNING, logger="dptb.data.dataset.lmdb_dataset"):
        delta = build_residual_hamiltonian_target_blocks(data_dict, blocks, shrink_policy="off")
    assert float(delta["0_0_0_0_0"][0, 0]) == pytest.approx(1.0)
    assert not any("residual_shrink_policy" in rec.getMessage() for rec in caplog.records)


def test_h7_min_residual_shrink_ratio_is_honored():
    """min_residual_shrink tunes the required ratio: a 1.3x-shrinking record
    (H=1.3, H0=0.3 -> D=1.0) passes at 1.2 but fails at 2.0."""
    data_dict, blocks = _shrink_inputs(1.3, 0.3)
    # passes at the default-ish 1.2 ratio (1.0 * 1.2 < 1.3).
    delta = build_residual_hamiltonian_target_blocks(
        data_dict, blocks, shrink_policy="error", min_shrink=1.2
    )
    assert float(delta["0_0_0_0_0"][0, 0]) == pytest.approx(1.0)
    # fails at a stricter 2.0 ratio (1.0 * 2.0 not < 1.3).
    with pytest.raises(RuntimeError, match="H0-quality"):
        build_residual_hamiltonian_target_blocks(
            data_dict, blocks, shrink_policy="error", min_shrink=2.0
        )


def _raw_non_shrinking_record() -> dict:
    """A raw absolute_full_h record whose H-H0 does NOT shrink (H=1 over H0=0)."""
    record = _raw_absolute_full_h_record()
    for key in record["hamiltonian"]:
        record["hamiltonian"][key] = np.asarray([[1.0]], dtype=np.float32)
        record["hamiltonian_0"][key] = np.asarray([[0.0]], dtype=np.float32)
    return record


def test_h7_dataset_ctor_arg_reaches_the_shrink_gate(tmp_path):
    """Wiring: the LMDBDataset residual_shrink_policy ctor arg reaches the loader
    gate.  A non-shrinking record raises under 'error' and loads under 'off'."""
    error_ds = _build_dataset(
        tmp_path,
        _raw_non_shrinking_record(),
        name="b-nonshrink-error",
        residual_hamiltonian=True,
        require_residual_from_full_h_target=True,
        residual_shrink_policy="error",
    )
    with pytest.raises(RuntimeError, match="H0-quality"):
        error_ds.get(0)

    off_ds = _build_dataset(
        tmp_path,
        _raw_non_shrinking_record(),
        name="b-nonshrink-off",
        residual_hamiltonian=True,
        require_residual_from_full_h_target=True,
        residual_shrink_policy="off",
    )
    sample = off_ds.get(0)
    torch.testing.assert_close(
        sample[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([1.0, 1.0])
    )


# ===========================================================================
# H8: per-record sample_uid stability (P3).  The loader attaches a stable packed
# (shard_id<<32)|row_id identity unconditionally; it is content-independent, so a
# record's uid is stable across re-loads and distinct across records.
# ===========================================================================
def _build_two_record_dataset(tmp_path, records, *, name, **kwargs):
    lmdb_path = tmp_path / f"{name}.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=True)
    try:
        with env.begin(write=True) as txn:
            for row_id, record in enumerate(records):
                txn.put((row_id).to_bytes(4, "big"), pickle.dumps(record))
    finally:
        env.close()
    return DatasetBuilder()(
        root=str(tmp_path),
        r_max=2.0,
        type="LMDBDataset",
        prefix=name,
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_H0=True,
        **kwargs,
    )


def test_h8_sample_uid_is_stable_per_record_and_distinct_across_records(tmp_path):
    rec0 = _raw_absolute_full_h_record()
    rec1 = _raw_absolute_full_h_record()
    rec1["case_id"] = "h2-second"
    dataset = _build_two_record_dataset(tmp_path, [rec0, rec1], name="b-uid-two")

    uid0_first = dataset.get(0)[_keys.SAMPLE_UID_KEY]
    uid0_second = dataset.get(0)[_keys.SAMPLE_UID_KEY]
    uid1 = dataset.get(1)[_keys.SAMPLE_UID_KEY]

    # Same record re-loaded -> byte-identical uid (batch/order independent).
    assert torch.equal(uid0_first, uid0_second)
    # Different records -> different uids ...
    assert not torch.equal(uid0_first, uid1)
    # ... differing only in the low-32 row_id (same shard), consecutive keys.
    v0 = int(uid0_first.item())
    v1 = int(uid1.item())
    assert (v0 >> 32) == (v1 >> 32)  # same packed shard ordinal
    assert (v1 & 0xFFFFFFFF) == (v0 & 0xFFFFFFFF) + 1
    assert v0 >= 0 and v1 >= 0  # positive int64 packing


def test_h8b_sample_uid_is_composition_independent_and_collision_free(tmp_path):
    """H8b: a record's uid is a function of its shard's CONTENT + row ALONE.

    Regression for the dense-ordinal scheme, under which a shard's ordinal was its
    sorted position in whatever set of shards a dataset happened to load, so (a) the
    same physical shard changed uid between single- and multi-shard datasets and
    (b) two independent single-shard datasets both took ordinal 0 and aliased row N
    to the same uid (and hence the same SEEDED prior substream).  The identity is
    now ``(hash(content fingerprint) << 32) | row`` (P2-2: content-anchored, not
    path-anchored, so relocating a byte-identical shard no longer changes uid --
    see ``test_sample_uid_content_fingerprint.py`` for that regression).
    """
    rec = _raw_absolute_full_h_record()
    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"
    dir_a.mkdir()
    dir_b.mkdir()

    # Two DISTINCT single-shard datasets, each with a record at row 0.  Under the
    # old dense ordinal both shards were ordinal 0, so row 0 collided to uid 0.
    # Their basenames ("shard-a.lmdb"/"shard-b.lmdb") differ, so they remain
    # distinct shards under content-anchored fingerprinting too.
    ds_a = _build_two_record_dataset(dir_a, [rec], name="shard-a")
    ds_b = _build_two_record_dataset(dir_b, [rec], name="shard-b")

    # Compute the independently-expected ordinal (and resolve the realpath)
    # BEFORE any .get() call: reading a record caches an open lmdb env handle
    # for the shard's path on the dataset for its lifetime, and
    # _shard_content_fingerprint opens that SAME path again to read the first
    # record -- lmdb refuses to open a path twice in one process, so this must
    # run first.
    realpath_a = ds_a._resolve_shard_realpath(0)
    expected_shard_id_a = _stable_shard_ordinal(_shard_content_fingerprint(realpath_a))

    uid_a0 = int(ds_a.get(0)[_keys.SAMPLE_UID_KEY].item())
    uid_b0 = int(ds_b.get(0)[_keys.SAMPLE_UID_KEY].item())
    assert uid_a0 != uid_b0                   # no cross-dataset collision ...
    assert (uid_a0 >> 32) != (uid_b0 >> 32)   # ... because their shard ids differ

    # The shard id is the content-stable hash of the shard's OWN content
    # fingerprint (basename + entry count + first record) -- not a dense
    # position (which would be 0 for a lone shard) and not a hash of the
    # realpath string itself -- so it is invariant to which other shards a
    # dataset co-loads AND to the shard's on-disk location.
    assert (uid_a0 >> 32) == expected_shard_id_a
    assert (uid_a0 & 0xFFFFFFFF) == 0  # row 0

    # Reloading the SAME physical shard reproduces the SAME uid. ds_a.get(0)
    # above cached an open read env for "A/shard-a.lmdb" on ds_a for its
    # lifetime (LMDBDataset._get_lmdb_env); py-lmdb refuses a second
    # Environment.open() on that same path while it is still open in this
    # process, so release it before _build_two_record_dataset reopens the
    # path for a write (same cleanup __del__ would eventually do).
    for env in ds_a._lmdb_env_cache.values():
        env.close()
    ds_a._lmdb_env_cache.clear()
    ds_a_again = _build_two_record_dataset(dir_a, [rec], name="shard-a")
    assert int(ds_a_again.get(0)[_keys.SAMPLE_UID_KEY].item()) == uid_a0


# ===========================================================================
# H9: physical-H0 authority/fingerprint contract parity (P2-3).  The ordinary
# Full-H route (assert_absolute_full_h_target_contract with require_h0=True,
# wired from record_pipeline.py whenever require_full_h_target=True and
# get_H0=True) and the residual-from-Full-H route
# (assert_residual_from_full_h_target_contract) consume the SAME raw
# absolute_full_h records, so they must agree on whether a record's
# physical_h0_source/physical_h0_source_fingerprint authority declaration is
# self-consistent. Before this fix, only the Full-H route ran
# assert_physical_h0_authority_contract; a record claiming
# physical_h0_source=dedicated_h0_blocks (contradicting the schema-fixed raw
# authority RAW_HAMILTONIAN_SAMPLE_SCHEMA always has) was rejected on the
# Full-H route but silently accepted by the residual route, which subtracted
# raw hamiltonian_0 anyway -- the same record's provenance was
# self-contradictory depending only on which route loaded it.
#
# The "ordinary Full-H route" comparator below calls
# assert_absolute_full_h_target_contract directly (the exact function
# record_pipeline.py invokes for require_full_h_target=True records) rather
# than going through a full require_full_h_target=True dataset: a record
# declaring physical_h0_source=dedicated_h0_blocks ALSO trips
# record_pipeline.py's separate, schema-agnostic
# _assert_dedicated_physical_h0_dataset_fingerprint pre-check (production
# dataset-level dedicated-H0-content auditing, unrelated to this fix) before
# reaching assert_absolute_full_h_target_contract, which would fail the SAME
# record for a DIFFERENT reason (an unconfigured
# expected_physical_h0_source_fingerprint) and obscure exactly which contract
# is under test here.
# ===========================================================================
def _raw_record_with_dedicated_h0_source(*, fingerprint=None) -> dict:
    """A raw absolute_full_h record that ALSO declares dedicated physical-H0
    authority -- contradictory for RAW_HAMILTONIAN_SAMPLE_SCHEMA, whose
    authority is fixed to raw_hamiltonian_0 regardless of what it claims."""
    record = _raw_absolute_full_h_record()
    record[PHYSICAL_H0_SOURCE_KEY] = DEDICATED_PHYSICAL_H0_SOURCE
    if fingerprint is not None:
        record[PHYSICAL_H0_SOURCE_FINGERPRINT_KEY] = fingerprint
    return record


def test_h9_residual_route_matches_full_h_route_and_rejects_too(tmp_path):
    """The bug repro: residual-from-Full-H must fail closed on the SAME
    contradictory record the ordinary Full-H route rejects (fingerprint
    missing), instead of silently subtracting raw hamiltonian_0 under a
    provenance claim the record itself contradicts."""
    dataset = _build_dataset(
        tmp_path,
        _raw_record_with_dedicated_h0_source(),
        name="h9-residual-repro",
        residual_hamiltonian=True,
        require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError, match="authority"):
        dataset.get(0)


def test_h9_residual_route_rejects_dedicated_source_with_wrong_fingerprint(tmp_path):
    """Same repro, but with an explicit (arbitrary/wrong) fingerprint present
    rather than missing -- still contradictory for a raw-schema record (whose
    authority is fixed to raw, not dedicated, regardless of fingerprint), and
    still must fail closed on the residual route."""
    dataset = _build_dataset(
        tmp_path,
        _raw_record_with_dedicated_h0_source(fingerprint="0" * 64),
        name="h9-residual-wrong-fingerprint",
        residual_hamiltonian=True,
        require_residual_from_full_h_target=True,
    )
    with pytest.raises(ValueError, match="authority"):
        dataset.get(0)


def test_h9_contract_fn_direct_call_rejects_dedicated_source():
    """Direct unit call: assert_residual_from_full_h_target_contract itself
    raises on the masquerading authority declaration, naming 'authority'."""
    with pytest.raises(ValueError, match="authority"):
        assert_residual_from_full_h_target_contract(
            _raw_record_with_dedicated_h0_source()
        )
    # ... and the contract it now delegates to raises the identical way.
    with pytest.raises(ValueError, match="authority"):
        assert_absolute_full_h_target_contract(
            _raw_record_with_dedicated_h0_source(), require_h0=True
        )


@pytest.mark.parametrize(
    "physical_h0_source", [None, RAW_PHYSICAL_H0_SOURCE],
    ids=["unset", "explicit-raw"],
)
def test_h9_residual_route_does_not_misfire_on_legitimate_raw_authority(
    tmp_path, physical_h0_source
):
    """No false positive: a genuine raw record either omitting
    physical_h0_source (the common case) or explicitly declaring it as the
    schema-fixed raw_hamiltonian_0 authority still loads and materializes the
    residual target exactly as before (node [1, 1], edge [.5, .5])."""
    record = _raw_absolute_full_h_record()
    if physical_h0_source is not None:
        record[PHYSICAL_H0_SOURCE_KEY] = physical_h0_source

    dataset = _build_dataset(
        tmp_path,
        record,
        name=f"h9-legit-{physical_h0_source or 'unset'}",
        residual_hamiltonian=True,
        require_residual_from_full_h_target=True,
    )
    sample = dataset.get(0)
    torch.testing.assert_close(
        sample[_keys.NODE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([1.0, 1.0])
    )
    torch.testing.assert_close(
        sample[_keys.EDGE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([0.5, 0.5])
    )


def test_h9_contract_fn_direct_call_accepts_legitimate_raw_record():
    """Direct unit call: the clean fixture (no physical_h0_source declared)
    passes the contract, including the new authority check, without raising."""
    assert_residual_from_full_h_target_contract(_raw_absolute_full_h_record())

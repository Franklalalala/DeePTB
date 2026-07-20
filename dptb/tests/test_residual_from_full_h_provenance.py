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
    assert_residual_from_full_h_target_contract,
)
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    RAW_HAMILTONIAN_SAMPLE_SCHEMA,
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

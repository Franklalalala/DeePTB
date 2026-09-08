"""Prior/target slots: one record, two residual keys, no second view."""
from __future__ import annotations

import numpy as np
import pytest

from dptb.data.interfaces.p2_contract import (
    H0_RESIDUAL_RME_SCHEMA,
    NACF_RESIDUAL_RME_SCHEMA,
    NAMED_SLOTS_RME_SCHEMA,
    SAMPLE_SCHEMA_KEY,
    SOC_NAMED_SLOTS_RME_SCHEMA,
    apply_target_slot,
    build_prior_spec,
    build_target_spec,
    residual_rme_schemas_for_prior,
    resolve_target_keys,
    schema_requires_prior_table_provenance,
)
from dptb.data.dataset.record_pipeline import RecordSchemaValidator
from dptb.data.dataset.lmdb_dataset import LMDBDataset
from dptb.nn.embedding.lem_moe_v3_prior_2b import resolve_prior_2b_keys


def _named_record():
    n, e, w = 2, 4, 8
    return {
        SAMPLE_SCHEMA_KEY: NAMED_SLOTS_RME_SCHEMA,
        "node_features": np.ones((n, w), np.float32),  # H-H0
        "edge_features": np.ones((e, w), np.float32),
        "node_h0": np.full((n, w), 2.0, np.float32),
        "edge_h0": np.full((e, w), 2.0, np.float32),
        "node_p23": np.full((n, w), 3.0, np.float32),
        "edge_p2": np.full((e, w), 3.0, np.float32),
        "node_delta_nacf": np.full((n, w), 4.0, np.float32),  # H-P
        "edge_delta_nacf": np.full((e, w), 4.0, np.float32),
    }


def test_prior_keys_are_the_slot():
    assert build_prior_spec("na_cf").rme_fields == ("node_p23", "edge_p2")
    assert build_prior_spec("h0").rme_fields == ("node_h0", "edge_h0")
    assert resolve_prior_2b_keys("na_cf") == ("node_p23", "edge_p2")
    assert resolve_prior_2b_keys("h0") == ("node_h0", "edge_h0")


def test_named_slot_picks_h0res_from_node_features():
    rec = _named_record()
    assert resolve_target_keys(rec, "h0res") == ("node_features", "edge_features")
    apply_target_slot(rec, "h0res")
    assert rec["node_features"][0, 0] == 1.0


def test_named_slot_picks_nacfres_from_delta_keys():
    rec = _named_record()
    assert resolve_target_keys(rec, "nacfres") == (
        "node_delta_nacf",
        "edge_delta_nacf",
    )
    apply_target_slot(rec, "nacfres")
    assert rec["node_features"][0, 0] == 4.0
    assert rec["node_delta_nacf"][0, 0] == 4.0


def test_old_nacfres_view_falls_back_to_node_features():
    rec = {
        SAMPLE_SCHEMA_KEY: NACF_RESIDUAL_RME_SCHEMA,
        "node_features": np.array([[9.0]], np.float32),
        "edge_features": np.array([[9.0]], np.float32),
    }
    assert resolve_target_keys(rec, "nacfres") == ("node_features", "edge_features")


def test_h0res_view_cannot_serve_nacfres_without_named_keys():
    rec = {
        SAMPLE_SCHEMA_KEY: H0_RESIDUAL_RME_SCHEMA,
        "node_features": np.array([[1.0]], np.float32),
        "edge_features": np.array([[1.0]], np.float32),
    }
    with pytest.raises(ValueError, match="target_kind='nacfres'"):
        resolve_target_keys(rec, "nacfres")


def test_na_cf_residual_schemas_include_both_views_and_named_slots():
    allowed = residual_rme_schemas_for_prior("na_cf")
    assert H0_RESIDUAL_RME_SCHEMA in allowed
    assert NACF_RESIDUAL_RME_SCHEMA in allowed
    assert NAMED_SLOTS_RME_SCHEMA in allowed
    assert SOC_NAMED_SLOTS_RME_SCHEMA in allowed


def test_soc_named_slots_skip_table_provenance():
    assert schema_requires_prior_table_provenance(H0_RESIDUAL_RME_SCHEMA)
    assert not schema_requires_prior_table_provenance(SOC_NAMED_SLOTS_RME_SCHEMA)
    assert not build_prior_spec("h0").requires_table_provenance


def _stub_dataset(*, prior_kind, target_kind="", require_residual=True):
    ds = LMDBDataset.__new__(LMDBDataset)
    ds.prior_kind = prior_kind
    ds.prior_spec = build_prior_spec(prior_kind)
    ds.target_kind = target_kind
    ds.target_spec = build_target_spec(target_kind) if target_kind else None
    ds.require_prior_residual_rme_target = require_residual
    ds.require_full_h_target = False
    ds.get_H0 = False
    ds.get_prior = True
    ds.h0_key = "hamiltonian_0"
    return ds


def test_schema_gate_accepts_na_cf_on_h0res_and_nacfres():
    validator = RecordSchemaValidator()
    rec = {SAMPLE_SCHEMA_KEY: H0_RESIDUAL_RME_SCHEMA}
    validator.validate_schema_and_basis(
        _stub_dataset(prior_kind="na_cf", target_kind="h0res"), rec
    )
    rec2 = {SAMPLE_SCHEMA_KEY: NACF_RESIDUAL_RME_SCHEMA}
    validator.validate_schema_and_basis(
        _stub_dataset(prior_kind="na_cf", target_kind="nacfres"), rec2
    )


def test_schema_gate_rejects_p2_prior_on_nacfres_view():
    validator = RecordSchemaValidator()
    rec = {SAMPLE_SCHEMA_KEY: NACF_RESIDUAL_RME_SCHEMA}
    with pytest.raises(ValueError, match="target schema differs"):
        validator.validate_schema_and_basis(
            _stub_dataset(prior_kind="p2", target_kind="p2res"), rec
        )


def test_named_slot_requires_target_kind():
    validator = RecordSchemaValidator()
    rec = {SAMPLE_SCHEMA_KEY: NAMED_SLOTS_RME_SCHEMA}
    with pytest.raises(ValueError, match="target_kind"):
        validator.validate_schema_and_basis(
            _stub_dataset(prior_kind="na_cf", target_kind=""), rec
        )


def test_h0_prior_cannot_select_nacfres_view():
    validator = RecordSchemaValidator()
    rec = {SAMPLE_SCHEMA_KEY: NACF_RESIDUAL_RME_SCHEMA}
    with pytest.raises(ValueError, match="target schema differs"):
        validator.validate_schema_and_basis(
            _stub_dataset(prior_kind="h0", target_kind="nacfres"), rec
        )


def test_apply_target_slot_does_not_run_in_schema_gate():
    rec = _named_record()
    original = rec["node_features"].copy()
    validator = RecordSchemaValidator()
    validator.validate_schema_and_basis(
        _stub_dataset(prior_kind="na_cf", target_kind="nacfres"), rec
    )
    assert np.array_equal(rec["node_features"], original)

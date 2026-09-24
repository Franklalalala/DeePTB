"""LMDB record pipeline: the immutable SampleContext parse, physical-H0/Haar
attach stages, content-anchored shard identity (sample_uid survives
relocation), prior/target residual slots, and the real LMDBDataset.get() H0/
precomputed-feature paths.
"""
from __future__ import annotations

import os
import pickle
import shutil
from types import SimpleNamespace

import lmdb
import numpy as np
import pytest
import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.build import DatasetBuilder
from dptb.data.dataset.lmdb_dataset import (
    LMDBDataset,
    _build_shard_uid_offsets,
    _shard_content_fingerprint,
    _stable_shard_ordinal,
)
from dptb.data.dataset.record_pipeline import RecordSchemaValidator, TargetDecoder, build_sample_context
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    H0_RESIDUAL_RME_SCHEMA,
    NACF_RESIDUAL_RME_SCHEMA,
    NAMED_SLOTS_RME_SCHEMA,
    RAW_HAMILTONIAN_SAMPLE_SCHEMA,
    SAMPLE_SCHEMA_KEY,
    SOC_NAMED_SLOTS_RME_SCHEMA,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
    apply_target_slot,
    build_prior_spec,
    build_target_spec,
    residual_rme_schemas_for_prior,
    resolve_target_keys,
    schema_requires_prior_table_provenance,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.lem_moe_v3_prior_2b import resolve_prior_2b_keys


# ===========================================================================
# record_pipeline stages: physical-H0 / Haar attach (attach-only, read the
# immutable context and write to the accumulator) and the fail-closed
# contract-cache mark gate.
# ===========================================================================
def _bare_dataset(data_dict):
    """A ``__new__`` dataset with every target/prior channel disabled."""
    dataset = LMDBDataset.__new__(LMDBDataset)
    dataset._indices = None
    dataset.get_Hamiltonian = False
    dataset.get_H0 = False
    dataset.get_overlap = False
    dataset.get_DM = False
    dataset.get_eigenvalues = False
    dataset.orthogonal = False
    dataset.h0_key = "hamiltonian_0"
    dataset.prefer_precomputed_h0 = True
    dataset.transform = SimpleNamespace(mask_uureal=None, orbpair_irreps=None)
    dataset.file_map = ["data.0000.lmdb"]
    dataset.info_files = {
        "data.0000.lmdb": {
            "r_max": 1.0, "er_max": None, "oer_max": None, "wave_align": False,
            "train_w_homo_lumo_gap": False, "train_w_eps": False, "train_w_charge": False,
            "train_dip": False, "train_polar": False,
        }
    }
    base = {
        AtomicDataDict.CELL_KEY: torch.eye(3),
        AtomicDataDict.POSITIONS_KEY: torch.zeros((2, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.ones(2, dtype=torch.long),
        AtomicDataDict.PBC_KEY: torch.zeros(3, dtype=torch.bool),
    }
    base.update(data_dict)
    dataset._load_data_dict = lambda idx: base
    return dataset, base


def test_mark_validated_is_gated_on_fingerprinted_graph():
    from dptb.data.dataset.record_pipeline import GraphState

    validator = RecordSchemaValidator()
    graph = GraphState(canonical_stored_edge="edge", canonical_stored_shift="shift")

    cache = {}
    base = dict(record_contract_key=("path", 0), record_contract_already_validated=False,
               validated_record_contracts=cache)

    # Non-fingerprinted records never populate the fail-closed contract cache.
    validator.mark_validated(SimpleNamespace(requires_fingerprinted_graph=False, **base), graph)
    assert cache == {}

    # A fingerprinted record whose checks all passed caches the canonical graph.
    validator.mark_validated(SimpleNamespace(requires_fingerprinted_graph=True, **base), graph)
    assert cache == {("path", 0): ("edge", "shift")}

    # An already-validated read must not re-write the cache.
    cache.clear()
    ctx_seen = SimpleNamespace(requires_fingerprinted_graph=True, record_contract_key=("path", 0),
                              record_contract_already_validated=True, validated_record_contracts=cache)
    validator.mark_validated(ctx_seen, graph)
    assert cache == {}


def test_decode_physical_h0_attaches_and_shape_checks():
    node_ph0 = torch.arange(6.0).reshape(2, 3)
    edge_ph0 = torch.arange(9.0).reshape(3, 3)
    dataset, data_dict = _bare_dataset({
        AtomicDataDict.NODE_PHYSICAL_H0_KEY: node_ph0, AtomicDataDict.EDGE_PHYSICAL_H0_KEY: edge_ph0,
    })
    ctx = build_sample_context(dataset, 0, data_dict, RecordSchemaValidator())
    assert ctx.uses_pre_physical_h0 is True

    atomicdata = {}
    TargetDecoder().decode_physical_h0(ctx, atomicdata, num_nodes=2, num_edges=3)
    assert torch.equal(atomicdata[AtomicDataDict.NODE_PHYSICAL_H0_KEY], node_ph0)
    assert torch.equal(atomicdata[AtomicDataDict.EDGE_PHYSICAL_H0_KEY], edge_ph0)

    # Row count mismatch against the active graph must fail closed.
    with pytest.raises(ValueError, match="offline physical H0 rows do not match"):
        TargetDecoder().decode_physical_h0(ctx, {}, num_nodes=5, num_edges=3)


def test_decode_haar_attaches_u0_and_node_edge_features():
    haar_node = torch.full((2, 4), 1.5)
    haar_edge = torch.full((3, 4), 2.5)
    dataset, data_dict = _bare_dataset({
        AtomicDataDict.HAAR_U0_KEY: torch.eye(2),
        AtomicDataDict.HAAR_NODE_FEATURES_KEY: haar_node,
        AtomicDataDict.HAAR_EDGE_FEATURES_KEY: haar_edge,
    })
    ctx = build_sample_context(dataset, 0, data_dict, RecordSchemaValidator())

    atomicdata = {}
    TargetDecoder().decode_haar(ctx, atomicdata, num_nodes=2, num_edges=3)
    assert AtomicDataDict.HAAR_U0_KEY in atomicdata
    assert torch.equal(atomicdata[AtomicDataDict.HAAR_NODE_FEATURES_KEY], haar_node)
    assert torch.equal(atomicdata[AtomicDataDict.HAAR_EDGE_FEATURES_KEY], haar_edge)

    # Haar edge feature row mismatch must fail closed.
    with pytest.raises(ValueError, match="Haar edge feature rows do not match"):
        TargetDecoder().decode_haar(ctx, {}, num_nodes=2, num_edges=99)


# ===========================================================================
# LMDBDataset.get(): real H0-raw-conversion and stored-edge-graph reuse paths
# ===========================================================================
def test_lmdb_dataset_h0_raw_conversion_can_override_precomputed_h0(monkeypatch):
    dataset = LMDBDataset.__new__(LMDBDataset)
    dataset._indices = None
    dataset.get_Hamiltonian = False
    dataset.get_H0 = True
    dataset.get_overlap = False
    dataset.get_DM = False
    dataset.get_eigenvalues = False
    dataset.orthogonal = False
    dataset.h0_key = "hamiltonian_0"
    dataset.prefer_precomputed_h0 = False
    dataset.transform = OrbitalMapper({"H": "1s"}, method="e3tb", device="cpu")
    dataset.file_map = ["data.0000.lmdb"]
    dataset.info_files = {
        "data.0000.lmdb": {
            "r_max": 1.0, "er_max": None, "oer_max": None, "wave_align": False,
            "train_w_homo_lumo_gap": False, "train_w_eps": False, "train_w_charge": False,
            "train_dip": False, "train_polar": False,
        }
    }
    h0_blocks = {
        "0_0_0_0_0": torch.zeros((1, 1)), "0_1_0_0_0": torch.zeros((1, 1)),
        "1_0_0_0_0": torch.zeros((1, 1)), "1_1_0_0_0": torch.zeros((1, 1)),
    }
    data_dict = {
        AtomicDataDict.CELL_KEY: torch.eye(3),
        AtomicDataDict.POSITIONS_KEY: torch.zeros((2, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.ones(2, dtype=torch.long),
        AtomicDataDict.PBC_KEY: torch.zeros(3, dtype=torch.bool),
        "hamiltonian_0": h0_blocks,
        AtomicDataDict.NODE_H0_KEY: torch.full((2, 3), 9.0),
        AtomicDataDict.EDGE_H0_KEY: torch.full((4, 3), 9.0),
    }
    dataset._load_data_dict = lambda idx: data_dict

    def _fake_from_points(**kwargs):
        from dptb.utils.torch_geometric import Data

        return Data(
            pos=torch.as_tensor(kwargs["pos"]),
            edge_index=torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long),
            edge_cell_shift=torch.zeros((4, 3), dtype=torch.long),
            atomic_numbers=torch.as_tensor(kwargs["atomic_numbers"]),
        )

    calls = []

    def _fake_block_to_feature(atomicdata, type_mapper, blocks, overlap, orthogonal, **kwargs):
        calls.append((blocks, kwargs))
        atomicdata[kwargs["node_field"]] = torch.full((2, 3), 1.0)
        atomicdata[kwargs["edge_field"]] = torch.full((4, 3), 2.0)

    monkeypatch.setattr("dptb.data.dataset.lmdb_dataset.AtomicData.from_points", _fake_from_points)
    monkeypatch.setattr("dptb.data.dataset.lmdb_dataset.block_to_feature", _fake_block_to_feature)

    out = dataset.get(0)

    assert calls == [(h0_blocks, {"node_field": AtomicDataDict.NODE_H0_KEY, "edge_field": AtomicDataDict.EDGE_H0_KEY})]
    assert torch.equal(out[AtomicDataDict.NODE_H0_KEY], torch.full((2, 3), 1.0))
    assert torch.equal(out[AtomicDataDict.EDGE_H0_KEY], torch.full((4, 3), 2.0))


def test_lmdb_dataset_reuses_stored_edge_graph_for_precomputed_features(monkeypatch):
    dataset = LMDBDataset.__new__(LMDBDataset)
    dataset._indices = None
    dataset.get_Hamiltonian = True
    dataset.get_H0 = True
    dataset.get_overlap = False
    dataset.get_DM = False
    dataset.get_eigenvalues = False
    dataset.orthogonal = False
    dataset.h0_key = "hamiltonian_0"
    dataset.prefer_precomputed_h0 = True
    dataset.transform = OrbitalMapper({"H": "1s"}, method="e3tb", device="cpu")
    dataset.file_map = ["data.0000.lmdb"]
    dataset.info_files = {
        "data.0000.lmdb": {
            "r_max": 1.0, "er_max": None, "oer_max": None, "wave_align": False,
            "train_w_homo_lumo_gap": False, "train_w_eps": False, "train_w_charge": False,
            "train_dip": False, "train_polar": False,
        }
    }
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.long)
    edge_shift = torch.zeros((3, 3), dtype=torch.float32)
    data_dict = {
        AtomicDataDict.CELL_KEY: torch.eye(3),
        AtomicDataDict.POSITIONS_KEY: torch.zeros((2, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.ones(2, dtype=torch.long),
        AtomicDataDict.PBC_KEY: torch.ones(3, dtype=torch.bool),
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: edge_shift,
        AtomicDataDict.NODE_FEATURES_KEY: torch.full((2, 4), 1.0),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.full((3, 4), 2.0),
        AtomicDataDict.NODE_H0_KEY: torch.full((2, 4), 3.0),
        AtomicDataDict.EDGE_H0_KEY: torch.full((3, 4), 4.0),
    }
    dataset._load_data_dict = lambda idx: data_dict

    monkeypatch.setattr("dptb.data.dataset.lmdb_dataset.AtomicData.from_points",
                        lambda **kwargs: pytest.fail("stored graph should skip neighbor rebuild"))
    monkeypatch.setattr("dptb.data.dataset.lmdb_dataset.block_to_feature",
                        lambda *args, **kwargs: pytest.fail("precomputed features should skip block conversion"))

    out = dataset.get(0)

    assert torch.equal(out[AtomicDataDict.EDGE_INDEX_KEY], edge_index)
    assert torch.equal(out[AtomicDataDict.EDGE_CELL_SHIFT_KEY], edge_shift)
    assert torch.equal(out[AtomicDataDict.EDGE_FEATURES_KEY], data_dict[AtomicDataDict.EDGE_FEATURES_KEY])
    assert torch.equal(out[AtomicDataDict.EDGE_H0_KEY], data_dict[AtomicDataDict.EDGE_H0_KEY])


# ===========================================================================
# sample_uid: content-anchored LMDB shard identity survives relocation
# ===========================================================================
def _write_raw_lmdb(path, values) -> None:
    env = lmdb.open(str(path), map_size=1 << 20, subdir=True)
    try:
        with env.begin(write=True) as txn:
            for row_id, value in enumerate(values):
                txn.put(int(row_id).to_bytes(4, "big"), pickle.dumps(value))
    finally:
        env.close()


def _raw_absolute_full_h_record() -> dict:
    h_blocks = {
        "0_0_0_0_0": np.asarray([[10.0]], dtype=np.float32), "1_1_0_0_0": np.asarray([[12.0]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[3.0]], dtype=np.float32), "1_0_0_0_0": np.asarray([[3.0]], dtype=np.float32),
    }
    h0_blocks = {
        "0_0_0_0_0": np.asarray([[9.0]], dtype=np.float32), "1_1_0_0_0": np.asarray([[11.0]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[2.5]], dtype=np.float32), "1_0_0_0_0": np.asarray([[2.5]], dtype=np.float32),
    }
    return {
        _keys.CELL_KEY: np.eye(3, dtype=np.float32) * 8.0,
        _keys.POSITIONS_KEY: np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1, 1], dtype=np.int64),
        _keys.PBC_KEY: np.asarray([False, False, False]),
        "case_id": "h2",
        _keys.EDGE_INDEX_KEY: np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        _keys.EDGE_CELL_SHIFT_KEY: np.zeros((2, 3), dtype=np.float32),
        "hamiltonian": h_blocks, "hamiltonian_0": h0_blocks,
        SAMPLE_SCHEMA_KEY: RAW_HAMILTONIAN_SAMPLE_SCHEMA,
        TARGET_SEMANTICS_KEY: ABSOLUTE_FULL_H_SEMANTICS, TARGET_SOURCE_KEY: "raw_hamiltonian",
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
        root=str(root), r_max=2.0, type="LMDBDataset", prefix=name, separator=".", basis={"H": "1s"},
        get_Hamiltonian=True, get_H0=True, **kwargs,
    )


def test_relocated_byte_identical_shard_reproduces_sample_uid(tmp_path):
    """A byte-identical shard relocated to an unrelated parent path (a
    container/multi-node remount, an archive restore, a cwd change) reproduces
    the same sample_uid for the same row through the real DatasetBuilder/loader."""
    old_root = tmp_path / "old_mount"
    new_root = tmp_path / "new_mount" / "nested" / "elsewhere"
    old_root.mkdir()
    new_root.mkdir(parents=True)

    record = _raw_absolute_full_h_record()
    ds_old = _build_dataset(old_root, record, name="shard")
    uid_old = int(ds_old.get(0)[_keys.SAMPLE_UID_KEY].item())

    shutil.copytree(old_root / "shard.lmdb", new_root / "shard.lmdb")
    ds_new = DatasetBuilder()(
        root=str(new_root), r_max=2.0, type="LMDBDataset", prefix="shard", separator=".",
        basis={"H": "1s"}, get_Hamiltonian=True, get_H0=True,
    )
    uid_new = int(ds_new.get(0)[_keys.SAMPLE_UID_KEY].item())

    assert uid_new == uid_old, (
        "relocating a byte-identical LMDB shard must not change sample_uid "
        "(and therefore must not change any SEEDED prior's validation epsilon)"
    )


def test_same_content_mounted_twice_fails_closed(tmp_path):
    """The same shard content mounted twice under two distinct realpaths fails
    fast and names both paths, rather than silently aliasing their identities."""
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
    # Both paths are named via repr() (the collision-message convention),
    # which backslash-escapes on Windows.
    assert repr(path_a) in message
    assert repr(path_b) in message


def test_distinct_content_shards_do_not_collide(tmp_path):
    """No-regression: distinct shards build offsets cleanly, one ordinal each."""
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


@pytest.mark.parametrize(
    ("shard_a", "shard_b", "expect_equal"),
    [
        (("parent_x", "shard.lmdb", [{"payload": [1, 2, 3]}, {"payload": "two"}]),
         ("parent_y/deep/path", "shard.lmdb", [{"payload": [1, 2, 3]}, {"payload": "two"}]), True),
        (("content_a", "shard.lmdb", [{"payload": "same-basename-same-count"}]),
         ("content_b", "shard.lmdb", [{"payload": "DIFFERENT-value-bytes"}]), False),
        (("one_entry", "shard.lmdb", [{"payload": "identical-first-record"}]),
         ("two_entries", "shard.lmdb", [{"payload": "identical-first-record"}, {"payload": "identical-first-record"}]),
         False),
        (("empty_a", "alpha.lmdb", []), ("empty_b", "beta.lmdb", []), False),
        (("empty_parent_x", "placeholder.lmdb", []), ("empty_parent_y", "placeholder.lmdb", []), True),
    ],
    ids=["relocated_same_basename", "differs_on_first_record_content", "differs_on_entry_count",
        "distinct_empty_shards_differ", "identically_named_empty_shards_match"],
)
def test_shard_content_fingerprint_matches_content_not_path(tmp_path, shard_a, shard_b, expect_equal):
    """Two EMPTY shards must not alias (basename is the only signal left for
    them), but two empty shards sharing a basename (a relocated placeholder)
    fingerprint identically, consistent with the relocation invariant."""

    def _make(spec):
        subdir, basename, values = spec
        path = tmp_path / subdir / basename
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_raw_lmdb(path, values)
        return str(path)

    fp_a = _shard_content_fingerprint(_make(shard_a))
    fp_b = _shard_content_fingerprint(_make(shard_b))
    assert (fp_a == fp_b) is expect_equal
    if expect_equal:
        assert _stable_shard_ordinal(fp_a) == _stable_shard_ordinal(fp_b)


# ===========================================================================
# target_slot: prior/target residual slots, one record, no second view
# ===========================================================================
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


@pytest.mark.parametrize(
    ("target_kind", "expected_keys"),
    [("h0res", ("node_features", "edge_features")), ("nacfres", ("node_delta_nacf", "edge_delta_nacf"))],
)
def test_named_slot_resolves_the_target_kind(target_kind, expected_keys):
    assert resolve_target_keys(_named_record(), target_kind) == expected_keys


def test_old_nacfres_view_falls_back_to_node_features():
    rec = {SAMPLE_SCHEMA_KEY: NACF_RESIDUAL_RME_SCHEMA,
          "node_features": np.array([[9.0]], np.float32), "edge_features": np.array([[9.0]], np.float32)}
    assert resolve_target_keys(rec, "nacfres") == ("node_features", "edge_features")


@pytest.mark.parametrize(("target_kind", "expected_value"), [("h0res", 1.0), ("nacfres", 4.0)])
def test_apply_target_slot_writes_the_resolved_view_into_node_features(target_kind, expected_value):
    rec = _named_record()
    apply_target_slot(rec, target_kind)
    assert rec["node_features"][0, 0] == expected_value


def test_apply_target_slot_does_not_run_in_schema_gate():
    rec = _named_record()
    original = rec["node_features"].copy()
    validator = RecordSchemaValidator()
    validator.validate_schema_and_basis(_stub_dataset(prior_kind="na_cf", target_kind="nacfres"), rec)
    assert np.array_equal(rec["node_features"], original)


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


def test_h0res_view_cannot_serve_nacfres_without_named_keys():
    rec = {SAMPLE_SCHEMA_KEY: H0_RESIDUAL_RME_SCHEMA,
          "node_features": np.array([[1.0]], np.float32), "edge_features": np.array([[1.0]], np.float32)}
    with pytest.raises(ValueError, match="target_kind='nacfres'"):
        resolve_target_keys(rec, "nacfres")


@pytest.mark.parametrize(
    ("schema", "prior_kind", "target_kind", "match"),
    [
        (NACF_RESIDUAL_RME_SCHEMA, "p2", "p2res", "target schema differs"),
        (NAMED_SLOTS_RME_SCHEMA, "na_cf", "", "target_kind"),
        (NACF_RESIDUAL_RME_SCHEMA, "h0", "nacfres", "target schema differs"),
    ],
    ids=["p2_prior_on_nacfres_view", "named_slot_requires_target_kind", "h0_prior_cannot_select_nacfres_view"],
)
def test_schema_gate_rejects_mismatched_prior_and_target(schema, prior_kind, target_kind, match):
    validator = RecordSchemaValidator()
    rec = {SAMPLE_SCHEMA_KEY: schema}
    with pytest.raises(ValueError, match=match):
        validator.validate_schema_and_basis(_stub_dataset(prior_kind=prior_kind, target_kind=target_kind), rec)


def test_schema_gate_accepts_na_cf_on_h0res_and_nacfres():
    validator = RecordSchemaValidator()
    validator.validate_schema_and_basis(
        _stub_dataset(prior_kind="na_cf", target_kind="h0res"), {SAMPLE_SCHEMA_KEY: H0_RESIDUAL_RME_SCHEMA}
    )
    validator.validate_schema_and_basis(
        _stub_dataset(prior_kind="na_cf", target_kind="nacfres"), {SAMPLE_SCHEMA_KEY: NACF_RESIDUAL_RME_SCHEMA}
    )

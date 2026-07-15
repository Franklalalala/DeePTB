"""Stage-level unit tests for the extracted LMDB record pipeline.

These exercise individual collaborators of
``dptb.data.dataset.record_pipeline`` in isolation (no on-disk LMDB), pinning
the behaviour that ``LMDBDataset.get()`` used to inline: the immutable
``SampleContext`` parse, the graph short-circuit, the read seam, and the
fail-closed contract-cache mark gate.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from dptb.data import AtomicDataDict
from dptb.data.dataset.lmdb_dataset import LMDBDataset
from dptb.data.dataset.record_pipeline import (
    GraphResolver,
    GraphState,
    RecordSchemaValidator,
    SampleContext,
    StoredRecordReader,
    TargetDecoder,
    build_sample_context,
)


def _minimal_h0_dataset():
    """A ``__new__`` dataset wired for the get_H0 raw-block path (no stored graph)."""
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
    # build_sample_context only reads mask_uureal / orbpair_irreps off the mapper
    # for this (non-fingerprinted) record; a light stub keeps the test hermetic.
    dataset.transform = SimpleNamespace(mask_uureal=None, orbpair_irreps=None)
    dataset.file_map = ["data.0000.lmdb"]
    dataset.info_files = {
        "data.0000.lmdb": {
            "r_max": 1.0,
            "er_max": None,
            "oer_max": None,
            "wave_align": False,
            "train_w_homo_lumo_gap": False,
            "train_w_eps": False,
            "train_w_charge": False,
            "train_dip": False,
            "train_polar": False,
        }
    }
    h0_blocks = {
        "0_0_0_0_0": torch.zeros((1, 1)),
        "1_1_0_0_0": torch.zeros((1, 1)),
    }
    data_dict = {
        AtomicDataDict.CELL_KEY: torch.eye(3),
        AtomicDataDict.POSITIONS_KEY: torch.zeros((2, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.ones(2, dtype=torch.long),
        AtomicDataDict.PBC_KEY: torch.zeros(3, dtype=torch.bool),
        "hamiltonian_0": h0_blocks,
    }
    dataset._load_data_dict = lambda idx: data_dict
    return dataset, data_dict, h0_blocks


def test_stored_record_reader_delegates_to_load_data_dict():
    dataset, data_dict, _ = _minimal_h0_dataset()
    assert StoredRecordReader().read(dataset, 0) is data_dict


def test_build_sample_context_parses_h0_record_and_splits_info():
    dataset, data_dict, h0_blocks = _minimal_h0_dataset()

    ctx = build_sample_context(dataset, 0, data_dict, RecordSchemaValidator())

    assert isinstance(ctx, SampleContext)
    # Configuration snapshot + raw-block routing.
    assert ctx.get_H0 is True
    assert ctx.get_prior is False
    assert ctx.blocks is False  # neither Hamiltonian nor DM requested
    assert ctx.overlap is False
    assert ctx.h0_blocks is h0_blocks
    assert ctx.node_h0 is None and ctx.edge_h0 is None

    # Historical (non-fingerprinted) record: no basis fingerprint resolved.
    assert ctx.basis_fingerprint is None
    assert ctx.stored_basis_fingerprint is None
    assert ctx.record_contract_already_validated is False

    # No stored edge graph -> no fingerprint contract required.
    assert ctx.has_stored_edge_graph is False
    assert ctx.requires_fingerprinted_graph is False
    assert ctx.uses_pre_h0 is False  # node/edge_h0 absent
    assert ctx.load_prior_blocks is False

    # The six training-only flags are split out of the AtomicData constructor
    # kwargs but retained in cache_info.
    assert set(ctx.info) == {"r_max", "er_max", "oer_max"}
    assert set(ctx.cache_info) == {
        "train_polar",
        "train_dip",
        "wave_align",
        "train_w_charge",
        "train_w_eps",
        "train_w_homo_lumo_gap",
    }


def test_sample_context_is_immutable():
    dataset, data_dict, _ = _minimal_h0_dataset()
    ctx = build_sample_context(dataset, 0, data_dict, RecordSchemaValidator())
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.get_H0 = False


def test_graph_resolver_short_circuits_without_stored_graph():
    dataset, data_dict, _ = _minimal_h0_dataset()
    ctx = build_sample_context(dataset, 0, data_dict, RecordSchemaValidator())

    graph = GraphResolver().resolve(ctx)

    assert isinstance(graph, GraphState)
    assert graph.canonical_stored_edge is None
    assert graph.canonical_stored_shift is None


def test_mark_validated_is_gated_on_fingerprinted_graph():
    validator = RecordSchemaValidator()
    graph = GraphState(canonical_stored_edge="edge", canonical_stored_shift="shift")

    cache = {}
    base = dict(
        record_contract_key=("path", 0),
        record_contract_already_validated=False,
        validated_record_contracts=cache,
    )

    # Non-fingerprinted records never populate the fail-closed contract cache.
    ctx_no_fp = SimpleNamespace(requires_fingerprinted_graph=False, **base)
    validator.mark_validated(ctx_no_fp, graph)
    assert cache == {}

    # A fingerprinted record whose checks all passed caches the canonical graph.
    ctx_fp = SimpleNamespace(requires_fingerprinted_graph=True, **base)
    validator.mark_validated(ctx_fp, graph)
    assert cache == {("path", 0): ("edge", "shift")}

    # An already-validated read must not re-write the cache.
    cache.clear()
    ctx_seen = SimpleNamespace(
        requires_fingerprinted_graph=True,
        record_contract_key=("path", 0),
        record_contract_already_validated=True,
        validated_record_contracts=cache,
    )
    validator.mark_validated(ctx_seen, graph)
    assert cache == {}


def _bare_dataset(data_dict):
    """A ``__new__`` dataset with every target/prior channel disabled.

    Used to drive the physical-H0 / Haar decode stages, which are attach-only
    (they read the immutable context and write to the accumulator) and so are
    exercised here with a plain ``dict`` standing in for ``AtomicData``.
    """
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
            "r_max": 1.0,
            "er_max": None,
            "oer_max": None,
            "wave_align": False,
            "train_w_homo_lumo_gap": False,
            "train_w_eps": False,
            "train_w_charge": False,
            "train_dip": False,
            "train_polar": False,
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


def test_decode_physical_h0_attaches_and_shape_checks():
    node_ph0 = torch.arange(6.0).reshape(2, 3)
    edge_ph0 = torch.arange(9.0).reshape(3, 3)
    dataset, data_dict = _bare_dataset(
        {
            AtomicDataDict.NODE_PHYSICAL_H0_KEY: node_ph0,
            AtomicDataDict.EDGE_PHYSICAL_H0_KEY: edge_ph0,
        }
    )
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
    dataset, data_dict = _bare_dataset(
        {
            AtomicDataDict.HAAR_U0_KEY: torch.eye(2),
            AtomicDataDict.HAAR_NODE_FEATURES_KEY: haar_node,
            AtomicDataDict.HAAR_EDGE_FEATURES_KEY: haar_edge,
        }
    )
    ctx = build_sample_context(dataset, 0, data_dict, RecordSchemaValidator())

    atomicdata = {}
    TargetDecoder().decode_haar(ctx, atomicdata, num_nodes=2, num_edges=3)
    assert AtomicDataDict.HAAR_U0_KEY in atomicdata
    assert torch.equal(atomicdata[AtomicDataDict.HAAR_NODE_FEATURES_KEY], haar_node)
    assert torch.equal(atomicdata[AtomicDataDict.HAAR_EDGE_FEATURES_KEY], haar_edge)

    # Haar edge feature row mismatch must fail closed.
    with pytest.raises(ValueError, match="Haar edge feature rows do not match"):
        TargetDecoder().decode_haar(ctx, {}, num_nodes=2, num_edges=99)

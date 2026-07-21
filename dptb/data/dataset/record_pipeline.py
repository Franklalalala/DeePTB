"""Record-decoding pipeline for :class:`LMDBDataset`.

This module decomposes the historical ~900-line ``LMDBDataset.get()`` hot path
into an immutable :class:`SampleContext` plus a set of single-responsibility
collaborators:

* :class:`StoredRecordReader`   -- LMDB read + unpickle (delegates to the
  dataset's worker-local env cache so the on-disk read stays in one place).
* :class:`RecordSchemaValidator` -- schema/basis fingerprint validation and the
  per-record fail-closed contract cache (validate-once-per-worker).
* :class:`GraphResolver`         -- stored-graph selection and canonical
  edge/shift derivation plus graph/row fingerprint checks.
* :class:`AtomicDataFactory`     -- ``AtomicData`` construction (stored graph
  reuse vs. neighbour rebuild) and dynamic-batch metadata.
* :class:`PriorDecoder`          -- P2/P23 physical-prior RME features and AO
  block reconstruction/audit, driven by the dataset ``PriorSpec``.
* :class:`TargetDecoder`         -- H0 / physical-H0 / Haar features and the
  full-H / residual block targets.

``LMDBDataset.get()`` becomes a thin call into :func:`build_atomic_data`, which
threads the immutable context through the collaborators in the exact order of
the original method so that observable behaviour -- the ``AtomicData`` fields,
dtypes and values; the exceptions raised on the same malformed inputs; the
fingerprint fail-closed and worker-local contract-cache semantics -- is
preserved bit for bit.

The patch-sensitive module globals (``block_to_feature``, ``AtomicData``, the
``canonical_edge_graph`` / ``edge_graph_fingerprint`` / ``fingerprint_*`` /
``validate_*`` helpers) are resolved through the ``lmdb_dataset`` module object
(:data:`_host`) at call time.  Tests monkeypatch those names on
``dptb.data.dataset.lmdb_dataset``; routing every call through ``_host``
guarantees the patched implementation is the one that runs, exactly as when the
code lived inside ``get()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dptb.data import AtomicData, AtomicDataDict
from dptb.data.interfaces.p2_contract import (
    BASIS_FINGERPRINT_KEY,
    DEDICATED_PHYSICAL_H0_SOURCE,
    DUAL_PRIOR_SAMPLE_SCHEMA,
    EDGE_GRAPH_FINGERPRINT_KEY,
    FULL_H_TARGET_FINGERPRINT_KEY,
    P2_BUNDLE_FINGERPRINT_KEY,
    P2_SAMPLE_SCHEMA,
    P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
    PHYSICAL_H0_SOURCE_KEY,
    RAW_HAMILTONIAN_SAMPLE_SCHEMA,
    ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
    ROW_ALIGNED_DATA_FINGERPRINT_KEY,
    ROW_ALIGNED_FIELD_CANDIDATES,
    SAMPLE_SCHEMA_KEY,
    TARGET_SOURCE_KEY,
)

# The ``lmdb_dataset`` module hosts the patch-sensitive helpers (``block_to_feature``,
# ``canonical_edge_graph``, ``validate_p2_feature_pair`` ...) as well as the
# private helpers reused here (``_expand_soc_uureal_compact`` ...).  Import the
# module object -- never bind the individual names -- so that monkeypatching
# ``dptb.data.dataset.lmdb_dataset.<name>`` is observed at call time.  This
# import is acyclic: ``lmdb_dataset`` imports this module only lazily inside
# ``get()``.
import dptb.data.dataset.lmdb_dataset as _host


@dataclass(frozen=True)
class SampleContext:
    """Immutable per-record context threaded through the decode pipeline.

    Every field is derived once, up front, from the dataset configuration and
    the unpickled LMDB record.  Collaborators read from it and mutate only the
    freshly built ``AtomicData`` accumulator, never the context.
    """

    # --- record identity / contract state --------------------------------
    idx: int
    data_dict: Dict[str, Any]
    record_contract_key: Any
    record_contract_already_validated: bool
    validated_record_contracts: Dict[Any, Any]
    basis_fingerprint: Optional[str]
    stored_basis_fingerprint: Optional[str]

    # --- configuration snapshot ------------------------------------------
    get_Hamiltonian: bool
    get_H0: bool
    get_prior: bool
    get_DM: bool
    get_overlap: bool
    orthogonal: bool
    h0_key: str
    prefer_precomputed_h0: bool
    prefer_precomputed_prior: bool
    residual_hamiltonian: bool
    require_full_h_target: bool
    require_uureal_block_ode: bool
    residual_shrink_policy: str
    min_residual_shrink: float
    require_prior_blocks: bool
    audit_prior_representations: bool
    expected_prior_source_fingerprint: Optional[str]
    type_mapper: Any
    prior_spec: Any
    prior_raw_key: str

    # --- geometry --------------------------------------------------------
    cell: Any
    pos: Any
    atomic_numbers: Any
    pbc: Any

    # --- main target / overlap -------------------------------------------
    blocks: Any
    overlap: Any

    # --- precomputed features --------------------------------------------
    pre_node_features: Any
    pre_edge_features: Any
    pre_node_overlap: Any
    pre_edge_overlap: Any

    # --- H0 --------------------------------------------------------------
    h0_blocks: Any
    node_h0: Any
    edge_h0: Any

    # --- physical prior (P2/P23) -----------------------------------------
    p2_blocks: Any
    node_p2: Any
    edge_p2: Any
    p2_blockwise_keys: Any
    p2_blockwise_present: List[bool]
    p2_feature_present: Tuple[bool, bool]

    # --- absolute full-H target schema flags -----------------------------
    full_h_target_present: List[bool]
    schema_v2_row_aligned: bool

    # --- offline physical H0 ---------------------------------------------
    node_physical_h0: Any
    edge_physical_h0: Any
    physical_h0_present: Tuple[bool, bool]

    # --- Haar ------------------------------------------------------------
    haar_u0: Any
    haar_node_features: Any
    haar_edge_features: Any

    # --- SOC compact expansion mask --------------------------------------
    soc_uureal_keep_mask: Any

    # --- AtomicData constructor metadata ---------------------------------
    info: Dict[str, Any]
    cache_info: Dict[str, Any]
    mapper_p2_dim: Optional[int]

    # --- derived source-selection flags ----------------------------------
    need_overlap_features: bool
    has_pre_main: bool
    has_pre_overlap: bool
    uses_pre_main: bool
    uses_pre_physical_h0: bool
    uses_pre_h0: bool
    uses_p2: bool
    load_prior_blocks: bool
    dedicated_h0: bool

    # --- derived graph flags ---------------------------------------------
    stored_edge_index: Any
    stored_edge_shift: Any
    has_stored_edge_graph: bool
    requires_stored_p2_graph: bool
    requires_stored_target_graph: bool
    requires_fingerprinted_graph: bool


@dataclass(frozen=True)
class GraphState:
    """Canonical stored-graph edges produced by :class:`GraphResolver`."""

    canonical_stored_edge: Any
    canonical_stored_shift: Any


class StoredRecordReader:
    """Reads and unpickles a stored LMDB record for a dataset index.

    The physical read + worker-local env cache lives on the dataset
    (``_load_data_dict`` / ``_get_lmdb_env``); this collaborator is the pipeline
    seam over it and preserves the ``_last_lmdb_*`` read telemetry set there.
    """

    def read(self, dataset: Any, idx: int) -> Dict[str, Any]:
        return dataset._load_data_dict(idx)


class RecordSchemaValidator:
    """Schema/basis fingerprint validation and the fail-closed contract cache."""

    def resolve_contract_state(self, dataset: Any, idx: int):
        return dataset._record_contract_validation_state(idx)

    def validate_schema_and_basis(self, dataset: Any, data_dict: Dict[str, Any]):
        """Validate the absolute-Full-H contract and the basis fingerprint.

        Returns ``(basis_fingerprint, stored_basis_fingerprint)`` for reuse by
        the downstream graph/prior fingerprint checks.
        """
        sample_schema = data_dict.get(SAMPLE_SCHEMA_KEY)
        if (
            sample_schema == RAW_HAMILTONIAN_SAMPLE_SCHEMA
            and getattr(dataset, "get_prior", False)
        ):
            raise ValueError(
                "Generic raw-H records cannot also supply a P2/P23 dataset "
                "prior. Use the versioned P2/dual-prior sample schema so prior "
                "source and parent provenance remain fail-closed."
            )

        dedicated_h0 = bool(
            getattr(dataset, "require_full_h_target", False)
            and dataset.get_H0
            and data_dict.get(PHYSICAL_H0_SOURCE_KEY)
            == DEDICATED_PHYSICAL_H0_SOURCE
        )
        if dedicated_h0:
            dataset._assert_dedicated_physical_h0_dataset_fingerprint()

        if getattr(dataset, "require_full_h_target", False):
            _host.assert_absolute_full_h_target_contract(
                data_dict,
                h0_key=dataset.h0_key,
                # RAW_HAMILTONIAN_SAMPLE_SCHEMA is the block-ODE raw-H/H0
                # authority. Historical P2 schemas keep their 0715 opt-in
                # behavior unless they explicitly declare a physical-H0 source.
                require_h0=bool(
                    dataset.get_H0
                    and (
                        sample_schema == RAW_HAMILTONIAN_SAMPLE_SCHEMA
                        or PHYSICAL_H0_SOURCE_KEY in data_dict
                    )
                ),
                expected_physical_h0_source_fingerprint=getattr(
                    dataset, "expected_physical_h0_source_fingerprint", None
                ),
            )
        if getattr(dataset, "require_residual_h_target", False):
            _host.assert_residual_h_target_contract(
                data_dict, h0_key=dataset.h0_key
            )
        if getattr(dataset, "require_residual_from_full_h_target", False):
            _host.assert_residual_from_full_h_target_contract(
                data_dict, h0_key=dataset.h0_key
            )
        if getattr(dataset, "require_uureal_block_ode", False):
            self._validate_uureal_block_ode(dataset, data_dict)
        stored_basis_fingerprint = data_dict.get(BASIS_FINGERPRINT_KEY)
        # Historical, non-fingerprinted records need not expose a production
        # OrbitalMapper.  Resolve the mapper fingerprint lazily only when the
        # record actually declares the versioned physical-prior contract.
        basis_fingerprint = None
        if stored_basis_fingerprint is not None:
            if dataset.type_mapper is None:
                raise ValueError(
                    "A fingerprinted Hamiltonian record requires a configured "
                    "OrbitalMapper/basis."
                )
            basis_fingerprint = _host.mapper_basis_fingerprint(dataset.type_mapper)
            stored_basis_fingerprint = _host.require_sha256(
                stored_basis_fingerprint, field=BASIS_FINGERPRINT_KEY
            )
            if stored_basis_fingerprint != basis_fingerprint:
                raise ValueError(
                    "basis_fingerprint mismatch between LMDB record and configured "
                    f"OrbitalMapper: record={stored_basis_fingerprint}, "
                    f"mapper={basis_fingerprint}."
                )
        return basis_fingerprint, stored_basis_fingerprint

    @staticmethod
    def _validate_uureal_block_ode(dataset: Any, data_dict: Dict[str, Any]) -> None:
        """Fail closed on incomplete or mislabeled compact uu_real records."""

        keep = int(dataset.type_mapper.reduced_matrix_element)
        required = (
            "blockwise_spatial_schema",
            "blockwise_target_mode",
            "blockwise_source_target_feature_width",
            "blockwise_source_h0_feature_width",
            "soc_uureal_compact",
            "soc_uureal_full_rme",
            "soc_uureal_keep",
            AtomicDataDict.NODE_H0_KEY,
            AtomicDataDict.EDGE_H0_KEY,
            AtomicDataDict.NODE_DELTA_HAMIL_BLOCKS_KEY,
            AtomicDataDict.EDGE_DELTA_HAMIL_BLOCKS_KEY,
            AtomicDataDict.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
            AtomicDataDict.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
        )
        missing = [key for key in required if key not in data_dict]
        if missing:
            raise KeyError(f"Compact uu_real block record missing keys={missing}.")

        expected = {
            "blockwise_spatial_schema": "deeptb.blockwise_spatial/v1",
            "blockwise_target_mode": "already-delta",
            "soc_uureal_compact": True,
            "soc_uureal_full_rme": keep * 8,
            "soc_uureal_keep": keep,
        }
        for key, wanted in expected.items():
            actual = data_dict[key]
            actual = (
                torch.as_tensor(actual).item()
                if not isinstance(actual, str)
                else actual
            )
            if actual != wanted:
                raise ValueError(
                    "Compact uu_real block record requires "
                    f"{key}={wanted!r}; got {actual!r}."
                )

        # Converter metadata records the original feature widths.  A normal
        # full-SOC -> uu_real conversion therefore has source_width > keep;
        # already-compact input has source_width == keep.
        for key in (
            "blockwise_source_target_feature_width",
            "blockwise_source_h0_feature_width",
        ):
            try:
                actual = int(torch.as_tensor(data_dict[key]).item())
            except (TypeError, ValueError, RuntimeError):
                raise ValueError(
                    "Compact uu_real block record requires integer "
                    f"{key}; got {data_dict[key]!r}."
                )
            if actual < keep:
                raise ValueError(
                    f"Compact uu_real block record requires {key} >= "
                    f"keep={keep}; got {actual}."
                )

    def mark_validated(self, ctx: SampleContext, graph: GraphState) -> None:
        # Mark only after every applicable graph/row/prior/target fingerprint
        # and every downstream structural check has succeeded.  A malformed or
        # tampered first read is never cached as trusted and will fail closed on
        # every retry.
        if ctx.requires_fingerprinted_graph and not ctx.record_contract_already_validated:
            ctx.validated_record_contracts[ctx.record_contract_key] = (
                graph.canonical_stored_edge,
                graph.canonical_stored_shift,
            )


class GraphResolver:
    """Stored-graph selection, canonical edge/shift and graph fingerprint checks."""

    def resolve(self, ctx: SampleContext) -> GraphState:
        canonical_stored_edge = canonical_stored_shift = None
        if ctx.has_stored_edge_graph:
            if ctx.record_contract_already_validated:
                canonical_stored_edge, canonical_stored_shift = (
                    ctx.validated_record_contracts[ctx.record_contract_key]
                )
            else:
                canonical_stored_edge, canonical_stored_shift = _host.canonical_edge_graph(
                    ctx.atomic_numbers, ctx.stored_edge_index, ctx.stored_edge_shift
                )
            if ctx.requires_fingerprinted_graph:
                if ctx.stored_basis_fingerprint is None or ctx.basis_fingerprint is None:
                    raise ValueError(
                        "Row-aligned P2/Full-H tensors require basis_fingerprint metadata."
                    )
                if not ctx.record_contract_already_validated:
                    actual_graph_fingerprint = _host.edge_graph_fingerprint(
                        ctx.atomic_numbers,
                        canonical_stored_edge,
                        canonical_stored_shift,
                        basis_fingerprint=ctx.basis_fingerprint,
                    )
                    _host.assert_record_fingerprint(
                        ctx.data_dict,
                        field=EDGE_GRAPH_FINGERPRINT_KEY,
                        actual=actual_graph_fingerprint,
                    )
                    if ctx.schema_v2_row_aligned:
                        actual_row_fingerprint = _host.fingerprint_present_row_aligned_fields(
                            ctx.data_dict
                        )
                        _host.assert_record_fingerprint(
                            ctx.data_dict,
                            field=ROW_ALIGNED_DATA_FINGERPRINT_KEY,
                            actual=actual_row_fingerprint,
                        )
                        actual_row_bundle = _host.fingerprint_text_fields(
                            {
                                BASIS_FINGERPRINT_KEY: ctx.basis_fingerprint,
                                EDGE_GRAPH_FINGERPRINT_KEY: actual_graph_fingerprint,
                                ROW_ALIGNED_DATA_FINGERPRINT_KEY: actual_row_fingerprint,
                            },
                            (
                                BASIS_FINGERPRINT_KEY,
                                EDGE_GRAPH_FINGERPRINT_KEY,
                                ROW_ALIGNED_DATA_FINGERPRINT_KEY,
                            ),
                        )
                        _host.assert_record_fingerprint(
                            ctx.data_dict,
                            field=ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
                            actual=actual_row_bundle,
                        )
        return GraphState(canonical_stored_edge, canonical_stored_shift)


class AtomicDataFactory:
    """Builds the ``AtomicData`` accumulator and dynamic-batch metadata."""

    def build(self, ctx: SampleContext, graph: GraphState) -> Any:
        info = ctx.info
        data_dict = ctx.data_dict
        needs_missing_env_graph = (
            info.get("er_max", None) is not None
            and data_dict.get(AtomicDataDict.ENV_INDEX_KEY, None) is None
        )
        needs_missing_onsitenv_graph = (
            info.get("oer_max", None) is not None
            and data_dict.get(AtomicDataDict.ONSITENV_INDEX_KEY, None) is None
        )
        use_stored_edge_graph = bool(
            ctx.has_stored_edge_graph
            and (
                ctx.uses_pre_main
                or ctx.uses_pre_h0
                or ctx.uses_pre_physical_h0
                or ctx.uses_p2
                or ctx.requires_stored_target_graph
            )
        )

        if use_stored_edge_graph:
            atomicdata_kwargs = {
                key: value
                for key, value in info.items()
                if key not in _host._ATOMICDATA_CONSTRUCTOR_OPTIONS
            }
            atomicdata_kwargs[AtomicDataDict.EDGE_INDEX_KEY] = _host._lmdb_tensor(
                graph.canonical_stored_edge, torch.long
            )
            atomicdata_kwargs[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = _host._lmdb_tensor(
                graph.canonical_stored_shift, torch.get_default_dtype()
            )
            if data_dict.get(AtomicDataDict.ENV_INDEX_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ENV_INDEX_KEY] = _host._lmdb_tensor(
                    data_dict[AtomicDataDict.ENV_INDEX_KEY], torch.long
                )
            if data_dict.get(AtomicDataDict.ENV_CELL_SHIFT_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ENV_CELL_SHIFT_KEY] = _host._lmdb_tensor(
                    data_dict[AtomicDataDict.ENV_CELL_SHIFT_KEY], torch.get_default_dtype()
                )
            if data_dict.get(AtomicDataDict.ONSITENV_INDEX_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ONSITENV_INDEX_KEY] = _host._lmdb_tensor(
                    data_dict[AtomicDataDict.ONSITENV_INDEX_KEY], torch.long
                )
            if data_dict.get(AtomicDataDict.ONSITENV_CELL_SHIFT_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ONSITENV_CELL_SHIFT_KEY] = _host._lmdb_tensor(
                    data_dict[AtomicDataDict.ONSITENV_CELL_SHIFT_KEY], torch.get_default_dtype()
                )
            if needs_missing_env_graph or needs_missing_onsitenv_graph:
                # Generate only the missing auxiliary graphs, then restore the
                # authoritative stored main graph.  ``from_points`` must never
                # reorder row-aligned P2/target edges as a side effect.
                atomicdata = AtomicData.from_points(
                    pos=ctx.pos.reshape(-1, 3),
                    cell=ctx.cell.reshape(3, 3),
                    atomic_numbers=ctx.atomic_numbers,
                    pbc=ctx.pbc,
                    **info,
                )
                for key, value in atomicdata_kwargs.items():
                    atomicdata[key] = value
            else:
                atomicdata = AtomicData(
                    pos=_host._lmdb_tensor(ctx.pos.reshape(-1, 3), torch.get_default_dtype()),
                    cell=_host._lmdb_tensor(ctx.cell.reshape(3, 3), torch.get_default_dtype()),
                    atomic_numbers=_host._lmdb_tensor(ctx.atomic_numbers, torch.long),
                    pbc=_host._lmdb_tensor(ctx.pbc, torch.bool),
                    **atomicdata_kwargs,
                )
        else:
            atomicdata = AtomicData.from_points(
                pos=ctx.pos.reshape(-1, 3),
                cell=ctx.cell.reshape(3, 3),
                atomic_numbers=ctx.atomic_numbers,
                pbc=ctx.pbc,
                **info,
            )
        return atomicdata


class TargetDecoder:
    """Decodes H0 / physical-H0 / Haar features and full-H / residual targets."""

    def decode_main_features(
        self, ctx: SampleContext, atomicdata: Any, num_nodes: int, num_edges: int
    ) -> None:
        if ctx.has_pre_main and ctx.has_pre_overlap:
            pre_node_features = _host._expand_soc_uureal_compact(
                ctx.pre_node_features,
                ctx.data_dict,
                field_name=AtomicDataDict.NODE_FEATURES_KEY,
                keep_mask=ctx.soc_uureal_keep_mask,
            )
            pre_edge_features = _host._expand_soc_uureal_compact(
                ctx.pre_edge_features,
                ctx.data_dict,
                field_name=AtomicDataDict.EDGE_FEATURES_KEY,
                keep_mask=ctx.soc_uureal_keep_mask,
            )
            if pre_node_features.shape[0] != num_nodes or pre_edge_features.shape[0] != num_edges:
                raise ValueError(
                    "Precomputed LMDB feature rows do not match the active graph: "
                    f"node_features={tuple(pre_node_features.shape)}, "
                    f"edge_features={tuple(pre_edge_features.shape)}, "
                    f"num_nodes={num_nodes}, num_edges={num_edges}."
                )
            atomicdata[AtomicDataDict.NODE_FEATURES_KEY] = pre_node_features
            atomicdata[AtomicDataDict.EDGE_FEATURES_KEY] = pre_edge_features
            if ctx.need_overlap_features:
                pre_edge_overlap = torch.as_tensor(ctx.pre_edge_overlap)
                if pre_edge_overlap.shape[0] != num_edges:
                    raise ValueError(
                        "Precomputed LMDB edge overlap rows do not match the active graph: "
                        f"edge_overlap={tuple(pre_edge_overlap.shape)}, num_edges={num_edges}."
                    )
                if not ctx.orthogonal:
                    pre_node_overlap = torch.as_tensor(ctx.pre_node_overlap)
                    if pre_node_overlap.shape[0] != num_nodes:
                        raise ValueError(
                            "Precomputed LMDB node overlap rows do not match the active graph: "
                            f"node_overlap={tuple(pre_node_overlap.shape)}, num_nodes={num_nodes}."
                        )
                    atomicdata[AtomicDataDict.NODE_OVERLAP_KEY] = pre_node_overlap
                atomicdata[AtomicDataDict.EDGE_OVERLAP_KEY] = pre_edge_overlap
        elif ctx.get_Hamiltonian or ctx.get_DM or ctx.get_overlap:
            _host.block_to_feature(
                atomicdata, ctx.type_mapper, ctx.blocks, ctx.overlap, ctx.orthogonal
            )

    def decode_h0(
        self, ctx: SampleContext, atomicdata: Any, num_nodes: int, num_edges: int
    ) -> None:
        if not ctx.get_H0:
            return
        node_h0 = ctx.node_h0
        edge_h0 = ctx.edge_h0
        h0_blocks = ctx.h0_blocks
        if ctx.prefer_precomputed_h0 and node_h0 is not None and edge_h0 is not None:
            node_h0 = _host._expand_soc_uureal_compact(
                node_h0,
                ctx.data_dict,
                field_name=AtomicDataDict.NODE_H0_KEY,
                keep_mask=ctx.soc_uureal_keep_mask,
            )
            edge_h0 = _host._expand_soc_uureal_compact(
                edge_h0,
                ctx.data_dict,
                field_name=AtomicDataDict.EDGE_H0_KEY,
                keep_mask=ctx.soc_uureal_keep_mask,
            )
            if node_h0.shape[0] != num_nodes or edge_h0.shape[0] != num_edges:
                raise ValueError(
                    "Precomputed LMDB H0 rows do not match the active graph: "
                    f"node_h0={tuple(node_h0.shape)}, edge_h0={tuple(edge_h0.shape)}, "
                    f"num_nodes={num_nodes}, num_edges={num_edges}."
                )
            atomicdata[AtomicDataDict.NODE_H0_KEY] = node_h0
            atomicdata[AtomicDataDict.EDGE_H0_KEY] = edge_h0
        elif h0_blocks is not None:
            _host.block_to_feature(
                atomicdata,
                ctx.type_mapper,
                h0_blocks,
                False,
                ctx.orthogonal,
                node_field=AtomicDataDict.NODE_H0_KEY,
                edge_field=AtomicDataDict.EDGE_H0_KEY,
            )
        elif node_h0 is not None and edge_h0 is not None:
            atomicdata[AtomicDataDict.NODE_H0_KEY] = _host._expand_soc_uureal_compact(
                node_h0,
                ctx.data_dict,
                field_name=AtomicDataDict.NODE_H0_KEY,
                keep_mask=ctx.soc_uureal_keep_mask,
            )
            atomicdata[AtomicDataDict.EDGE_H0_KEY] = _host._expand_soc_uureal_compact(
                edge_h0,
                ctx.data_dict,
                field_name=AtomicDataDict.EDGE_H0_KEY,
                keep_mask=ctx.soc_uureal_keep_mask,
            )

    def decode_physical_h0(
        self, ctx: SampleContext, atomicdata: Any, num_nodes: int, num_edges: int
    ) -> None:
        if not ctx.uses_pre_physical_h0:
            return
        node_physical_h0 = _host._expand_soc_uureal_compact(
            ctx.node_physical_h0,
            ctx.data_dict,
            field_name=AtomicDataDict.NODE_PHYSICAL_H0_KEY,
            keep_mask=ctx.soc_uureal_keep_mask,
        )
        edge_physical_h0 = _host._expand_soc_uureal_compact(
            ctx.edge_physical_h0,
            ctx.data_dict,
            field_name=AtomicDataDict.EDGE_PHYSICAL_H0_KEY,
            keep_mask=ctx.soc_uureal_keep_mask,
        )
        if (
            node_physical_h0.shape[0] != num_nodes
            or edge_physical_h0.shape[0] != num_edges
        ):
            raise ValueError(
                "Precomputed offline physical H0 rows do not match the active graph: "
                f"node_physical_h0={tuple(node_physical_h0.shape)}, "
                f"edge_physical_h0={tuple(edge_physical_h0.shape)}, "
                f"num_nodes={num_nodes}, num_edges={num_edges}."
            )
        if torch.is_complex(node_physical_h0) or torch.is_complex(edge_physical_h0):
            raise TypeError(
                "node_physical_h0/edge_physical_h0 must already be DeePTB real "
                "RME features; raw complex SOC AO blocks are not accepted here."
            )
        if not torch.isfinite(node_physical_h0).all() or not torch.isfinite(edge_physical_h0).all():
            raise ValueError("Offline physical H0 contains NaN or infinity.")
        atomicdata[AtomicDataDict.NODE_PHYSICAL_H0_KEY] = node_physical_h0
        atomicdata[AtomicDataDict.EDGE_PHYSICAL_H0_KEY] = edge_physical_h0

    def decode_haar(
        self, ctx: SampleContext, atomicdata: Any, num_nodes: int, num_edges: int
    ) -> None:
        if ctx.haar_u0 is not None:
            atomicdata[AtomicDataDict.HAAR_U0_KEY] = torch.as_tensor(
                ctx.haar_u0,
                dtype=torch.get_default_dtype(),
            )
        if ctx.haar_node_features is not None:
            haar_node_features = _host._expand_soc_uureal_compact(
                ctx.haar_node_features,
                ctx.data_dict,
                field_name=AtomicDataDict.HAAR_NODE_FEATURES_KEY,
                keep_mask=ctx.soc_uureal_keep_mask,
            )
            if haar_node_features.shape[0] != num_nodes:
                raise ValueError(
                    "Precomputed LMDB Haar node feature rows do not match the active graph: "
                    f"haar_node_features={tuple(haar_node_features.shape)}, "
                    f"num_nodes={num_nodes}."
                )
            atomicdata[AtomicDataDict.HAAR_NODE_FEATURES_KEY] = haar_node_features
        if ctx.haar_edge_features is not None:
            haar_edge_features = _host._expand_soc_uureal_compact(
                ctx.haar_edge_features,
                ctx.data_dict,
                field_name=AtomicDataDict.HAAR_EDGE_FEATURES_KEY,
                keep_mask=ctx.soc_uureal_keep_mask,
            )
            if haar_edge_features.shape[0] != num_edges:
                raise ValueError(
                    "Precomputed LMDB Haar edge feature rows do not match the active graph: "
                    f"haar_edge_features={tuple(haar_edge_features.shape)}, "
                    f"num_edges={num_edges}."
                )
            atomicdata[AtomicDataDict.HAAR_EDGE_FEATURES_KEY] = haar_edge_features

    def assemble_block_tensors(
        self, ctx: SampleContext, atomicdata: Any, num_nodes: int, num_edges: int
    ) -> None:
        # Optional AO-block targets/H0 produced by the blockwise NexTHAM
        # conversion path. Keep this side channel independent of the feature
        # path so existing RME training remains unchanged.
        prior_spec = ctx.prior_spec
        data_dict = ctx.data_dict
        if ctx.get_Hamiltonian and ctx.residual_hamiltonian:
            _host.assert_residual_target_source_is_raw(data_dict)
        if ctx.get_prior and any(ctx.p2_blockwise_present) and not all(ctx.p2_blockwise_present):
            missing = [
                key for key, present in zip(ctx.p2_blockwise_keys, ctx.p2_blockwise_present)
                if not present
            ]
            raise ValueError(
                f"Prepacked {prior_spec.kind.upper()} prior is incomplete; "
                "missing block fields "
                f"{missing}."
            )
        if any(ctx.full_h_target_present) and not all(ctx.full_h_target_present):
            missing = [
                key for key, present in zip(
                    _host._PREPACKED_FULL_H_TARGET_KEYS, ctx.full_h_target_present
                )
                if not present
            ]
            raise ValueError(
                f"Prepacked absolute Full-H target is incomplete; missing {missing}."
            )
        for blockwise_key in (
            AtomicDataDict.NODE_DELTA_HAMIL_BLOCKS_KEY,
            AtomicDataDict.EDGE_DELTA_HAMIL_BLOCKS_KEY,
            AtomicDataDict.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
            AtomicDataDict.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
            AtomicDataDict.NODE_H0_BLOCKS_KEY,
            AtomicDataDict.EDGE_H0_BLOCKS_KEY,
            AtomicDataDict.NODE_H0_BLOCK_SHAPE_KEY,
            AtomicDataDict.EDGE_H0_BLOCK_SHAPE_KEY,
            *_host._PREPACKED_FULL_H_TARGET_KEYS,
        ):
            if blockwise_key in data_dict:
                atomicdata[blockwise_key] = torch.as_tensor(data_dict[blockwise_key])
        if ctx.require_uureal_block_ode:
            for metadata_key in (
                "blockwise_spatial_schema",
                "blockwise_target_mode",
                "blockwise_source_target_feature_width",
                "blockwise_source_h0_feature_width",
                "soc_uureal_compact",
                "soc_uureal_full_rme",
                "soc_uureal_keep",
            ):
                atomicdata[metadata_key] = data_dict[metadata_key]
        if ctx.load_prior_blocks:
            for blockwise_key in ctx.p2_blockwise_keys:
                if blockwise_key in data_dict:
                    atomicdata[blockwise_key] = torch.as_tensor(data_dict[blockwise_key])

        target_node_field = (
            AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY
            if ctx.require_full_h_target
            else AtomicDataDict.NODE_DELTA_HAMIL_BLOCKS_KEY
        )
        if (
            ctx.get_Hamiltonian
            and ctx.blocks is not False
            and ctx.blocks is not None
            and target_node_field not in atomicdata
        ):
            from dptb.data.interfaces.blockwise_tensor import attach_block_tensors, block_dict_to_ordered_tensors

            start_id = 0 if "0_0_0_0_0" in ctx.blocks else 1
            block_target_source = ctx.blocks
            if ctx.residual_hamiltonian:
                block_target_source = _host.build_residual_hamiltonian_target_blocks(
                    data_dict,
                    ctx.blocks,
                    h0_key=ctx.h0_key,
                    shrink_policy=ctx.residual_shrink_policy,
                    min_shrink=ctx.min_residual_shrink,
                )
            target_blocks = block_dict_to_ordered_tensors(
                atomicdata,
                ctx.type_mapper,
                block_target_source,
                start_id=start_id,
                complete_edges=False,
                strict_complete_edges=False,
            )
            attach_block_tensors(
                atomicdata,
                target_blocks,
                prefix=(
                    "full_h_target"
                    if ctx.require_full_h_target
                    else "delta_hamil"
                ),
            )

        if (
            ctx.get_H0
            and ctx.h0_blocks is not None
            and AtomicDataDict.NODE_H0_BLOCKS_KEY not in atomicdata
        ):
            from dptb.data.interfaces.blockwise_tensor import attach_block_tensors, block_dict_to_ordered_tensors

            start_id = 0 if "0_0_0_0_0" in ctx.h0_blocks else 1
            target_h0_blocks = block_dict_to_ordered_tensors(
                atomicdata,
                ctx.type_mapper,
                ctx.h0_blocks,
                start_id=start_id,
                complete_edges=True,
                strict_complete_edges=False,
            )
            attach_block_tensors(atomicdata, target_h0_blocks, prefix="h0")

        if ctx.dedicated_h0 and not ctx.record_contract_already_validated:
            from dptb.data.interfaces.blockwise_tensor import validate_packed_non_soc_blocks

            validate_packed_non_soc_blocks(
                atomicdata,
                ctx.type_mapper,
                atomicdata[AtomicDataDict.NODE_H0_BLOCKS_KEY],
                atomicdata[AtomicDataDict.EDGE_H0_BLOCKS_KEY],
                atomicdata[AtomicDataDict.NODE_H0_BLOCK_SHAPE_KEY],
                atomicdata[AtomicDataDict.EDGE_H0_BLOCK_SHAPE_KEY],
                label="physical H0",
                require_symmetric_edges=True,
            )

        if (
            ctx.load_prior_blocks
            and ctx.p2_blocks is not None
            and prior_spec.node_blocks_key not in atomicdata
        ):
            from dptb.data.interfaces.blockwise_tensor import attach_block_tensors, block_dict_to_ordered_tensors

            start_id = 0 if "0_0_0_0_0" in ctx.p2_blocks else 1
            target_p2_blocks = block_dict_to_ordered_tensors(
                atomicdata,
                ctx.type_mapper,
                ctx.p2_blocks,
                start_id=start_id,
                complete_edges=False,
                strict_complete_edges=False,
            )
            attach_block_tensors(atomicdata, target_p2_blocks, prefix=prior_spec.kind)

    def validate_full_h(self, ctx: SampleContext, atomicdata: Any) -> None:
        if not ctx.require_full_h_target:
            return
        missing = [
            key for key in _host._PREPACKED_FULL_H_TARGET_KEYS if key not in atomicdata
        ]
        if missing:
            raise ValueError(f"Absolute Full-H target fields are missing: {missing}.")
        from dptb.data.interfaces.blockwise_tensor import validate_packed_non_soc_blocks

        if not ctx.record_contract_already_validated:
            validate_packed_non_soc_blocks(
                atomicdata,
                ctx.type_mapper,
                atomicdata[AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY],
                atomicdata[AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY],
                atomicdata[AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY],
                atomicdata[AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY],
                label="absolute Full-H target",
                require_symmetric_edges=True,
            )
        if (
            ctx.data_dict.get(TARGET_SOURCE_KEY) == "dedicated_full_h_blocks"
            and not ctx.record_contract_already_validated
        ):
            actual_target_fingerprint = _host.fingerprint_fields(
                atomicdata, _host._PREPACKED_FULL_H_TARGET_KEYS
            )
            _host.assert_record_fingerprint(
                ctx.data_dict,
                field=FULL_H_TARGET_FINGERPRINT_KEY,
                actual=actual_target_fingerprint,
            )


class PriorDecoder:
    """Decodes P2/P23 physical-prior RME features and AO block reconstruction."""

    def decode_features(
        self, ctx: SampleContext, atomicdata: Any, num_nodes: int, num_edges: int
    ) -> None:
        if not ctx.get_prior:
            return
        prior_spec = ctx.prior_spec
        node_p2 = ctx.node_p2
        edge_p2 = ctx.edge_p2
        if ctx.prefer_precomputed_prior and all(ctx.p2_feature_present):
            node_p2, edge_p2 = _host.validate_p2_feature_pair(
                node_p2,
                edge_p2,
                num_nodes=num_nodes,
                num_edges=num_edges,
                feature_dim=ctx.mapper_p2_dim,
                prior_kind=prior_spec.kind,
                check_finite=not ctx.record_contract_already_validated,
            )
            atomicdata[prior_spec.node_rme_key] = node_p2
            atomicdata[prior_spec.edge_rme_key] = edge_p2
        elif ctx.p2_blocks is not None:
            _host.block_to_feature(
                atomicdata,
                ctx.type_mapper,
                ctx.p2_blocks,
                False,
                ctx.orthogonal,
                node_field=prior_spec.node_rme_key,
                edge_field=prior_spec.edge_rme_key,
                missing_block_policy="error",
            )
            node_p2, edge_p2 = _host.validate_p2_feature_pair(
                atomicdata[prior_spec.node_rme_key]
                if prior_spec.node_rme_key in atomicdata
                else None,
                atomicdata[prior_spec.edge_rme_key]
                if prior_spec.edge_rme_key in atomicdata
                else None,
                num_nodes=num_nodes,
                num_edges=num_edges,
                feature_dim=ctx.mapper_p2_dim,
                prior_kind=prior_spec.kind,
                check_finite=not ctx.record_contract_already_validated,
            )
            atomicdata[prior_spec.node_rme_key] = node_p2
            atomicdata[prior_spec.edge_rme_key] = edge_p2
        elif all(ctx.p2_feature_present):
            node_p2, edge_p2 = _host.validate_p2_feature_pair(
                node_p2,
                edge_p2,
                num_nodes=num_nodes,
                num_edges=num_edges,
                feature_dim=ctx.mapper_p2_dim,
                prior_kind=prior_spec.kind,
                check_finite=not ctx.record_contract_already_validated,
            )
            atomicdata[prior_spec.node_rme_key] = node_p2
            atomicdata[prior_spec.edge_rme_key] = edge_p2
        else:
            raise ValueError(
                f"get_prior=True requires either raw '{ctx.prior_raw_key}' AO "
                f"blocks or precomputed {prior_spec.node_rme_key}/"
                f"{prior_spec.edge_rme_key} features."
            )

        if ctx.requires_stored_p2_graph:
            sample_schema = ctx.data_dict.get(SAMPLE_SCHEMA_KEY)
            if sample_schema not in prior_spec.allowed_sample_schemas:
                raise ValueError(
                    f"Precomputed {prior_spec.kind.upper()} tensors require "
                    "explicit sample schema in "
                    f"{prior_spec.allowed_sample_schemas!r}; got "
                    f"{sample_schema!r}."
                )
            _host._assert_expected_prior_source(
                ctx.data_dict,
                ctx.expected_prior_source_fingerprint,
                prior_spec=prior_spec,
            )
            if prior_spec.kind == "p23":
                parent_p2_bundle = _host.require_sha256(
                    ctx.data_dict.get(P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY),
                    field=P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
                )
                record_p2_bundle = _host.require_sha256(
                    ctx.data_dict.get(P2_BUNDLE_FINGERPRINT_KEY),
                    field=P2_BUNDLE_FINGERPRINT_KEY,
                )
                if parent_p2_bundle != record_p2_bundle:
                    raise ValueError(
                        "P23 parent P2 bundle does not match this dual-prior "
                        f"record: parent={parent_p2_bundle}, "
                        f"record={record_p2_bundle}."
                    )
            if not ctx.record_contract_already_validated:
                actual_rme_fingerprint = _host.fingerprint_fields(
                    atomicdata,
                    prior_spec.rme_fields,
                )
                _host.assert_record_fingerprint(
                    ctx.data_dict,
                    field=prior_spec.rme_fingerprint_key,
                    actual=actual_rme_fingerprint,
                )

    def validate_blocks(
        self, ctx: SampleContext, atomicdata: Any, num_nodes: int, num_edges: int
    ) -> None:
        if not ctx.load_prior_blocks:
            return
        prior_spec = ctx.prior_spec
        data_dict = ctx.data_dict
        required_p2_blocks = (
            prior_spec.node_blocks_key,
            prior_spec.edge_blocks_key,
        )
        available_p2_blocks = [key in atomicdata for key in required_p2_blocks]
        if ctx.require_prior_blocks and not all(
            key in atomicdata for key in ctx.p2_blockwise_keys
        ):
            missing = [key for key in ctx.p2_blockwise_keys if key not in atomicdata]
            raise ValueError(
                "require_prior_blocks=True but the "
                f"{prior_spec.kind.upper()} AO reconstruction contract "
                f"is incomplete; missing {missing}."
            )
        if any(available_p2_blocks) and not all(available_p2_blocks):
            raise ValueError(
                f"{prior_spec.kind.upper()} AO-block reconstruction requires "
                f"both {prior_spec.node_blocks_key} and "
                f"{prior_spec.edge_blocks_key}."
            )
        if all(available_p2_blocks):
            p2_node_blocks = torch.as_tensor(
                atomicdata[prior_spec.node_blocks_key]
            )
            p2_edge_blocks = torch.as_tensor(
                atomicdata[prior_spec.edge_blocks_key]
            )
            _host.validate_non_soc_p2_block_tensors(
                p2_node_blocks,
                p2_edge_blocks,
                atomicdata[prior_spec.node_shape_key],
                atomicdata[prior_spec.edge_shape_key],
                num_nodes=num_nodes,
                num_edges=num_edges,
                data=atomicdata,
                idp=ctx.type_mapper,
                prior_kind=prior_spec.kind,
                expensive_checks=not ctx.record_contract_already_validated,
            )

            if ctx.requires_stored_p2_graph and not ctx.record_contract_already_validated:
                actual_block_fingerprint = _host.fingerprint_fields(
                    atomicdata, ctx.p2_blockwise_keys
                )
                _host.assert_record_fingerprint(
                    data_dict,
                    field=prior_spec.block_fingerprint_key,
                    actual=actual_block_fingerprint,
                )
                actual_bundle = _host.fingerprint_text_fields(
                    {
                        BASIS_FINGERPRINT_KEY: ctx.basis_fingerprint,
                        EDGE_GRAPH_FINGERPRINT_KEY: data_dict[
                            EDGE_GRAPH_FINGERPRINT_KEY
                        ],
                        prior_spec.source_fingerprint_key: data_dict[
                            prior_spec.source_fingerprint_key
                        ],
                        prior_spec.rme_fingerprint_key: data_dict[
                            prior_spec.rme_fingerprint_key
                        ],
                        prior_spec.block_fingerprint_key: data_dict[
                            prior_spec.block_fingerprint_key
                        ],
                        **{
                            field: data_dict[field]
                            for field in prior_spec.bundle_dependency_fields
                        },
                    },
                    (
                        BASIS_FINGERPRINT_KEY,
                        EDGE_GRAPH_FINGERPRINT_KEY,
                        prior_spec.source_fingerprint_key,
                        prior_spec.rme_fingerprint_key,
                        prior_spec.block_fingerprint_key,
                        *prior_spec.bundle_dependency_fields,
                    ),
                )
                _host.assert_record_fingerprint(
                    data_dict,
                    field=prior_spec.bundle_fingerprint_key,
                    actual=actual_bundle,
                )

            if (
                ctx.audit_prior_representations
                and not ctx.record_contract_already_validated
            ):
                from dptb.data.interfaces.blockwise_tensor import (
                    block_mask_from_shapes,
                    feature_tensors_to_block_tensors,
                )

                reconstructed = feature_tensors_to_block_tensors(
                    atomicdata,
                    ctx.type_mapper,
                    node_features=atomicdata[prior_spec.node_rme_key],
                    edge_features=atomicdata[prior_spec.edge_rme_key],
                    node_pad_shape=tuple(p2_node_blocks.shape[-2:]),
                    edge_pad_shape=tuple(p2_edge_blocks.shape[-2:]),
                    complete_edges=True,
                    strict_complete_edges=True,
                )
                node_mask = block_mask_from_shapes(
                    reconstructed.node_shapes, tuple(p2_node_blocks.shape[-2:])
                )
                edge_mask = block_mask_from_shapes(
                    reconstructed.edge_shapes, tuple(p2_edge_blocks.shape[-2:])
                )
                for label, rebuilt, stored, mask in (
                    ("node", reconstructed.node_blocks, p2_node_blocks, node_mask),
                    ("edge", reconstructed.edge_blocks, p2_edge_blocks, edge_mask),
                ):
                    error = float(
                        (rebuilt - stored).abs().masked_select(mask).max().detach().cpu()
                    ) if bool(mask.any()) else 0.0
                    if error > 5.0e-5:
                        raise ValueError(
                            f"{prior_spec.kind.upper()} {label} RME/AO "
                            "representations disagree; "
                            f"max valid-entry error {error:.3e}."
                        )


def build_sample_context(
    dataset: Any,
    idx: int,
    data_dict: Dict[str, Any],
    schema_validator: RecordSchemaValidator,
) -> SampleContext:
    """Parse the record + dataset config into the immutable :class:`SampleContext`.

    This mirrors the head of the original ``get()`` (record read already done):
    contract-cache resolution, schema/basis fingerprint validation, geometry and
    field extraction with fail-closed partial-prior/graph checks, the
    training-flag ``info_files`` side-channel update, and every derived
    source-selection / graph-requirement flag.  Building the pure ``info`` and
    ``mapper_p2_dim`` metadata here (both side-effect-free) is the only
    reordering; the first exception raised is unchanged.
    """
    record_contract_key, validated_record_contracts = (
        schema_validator.resolve_contract_state(dataset, idx)
    )
    record_contract_already_validated = record_contract_key in validated_record_contracts
    prior_spec = getattr(
        dataset,
        "prior_spec",
        _host.resolve_prior_field_spec(getattr(dataset, "prior_kind", "p2")),
    )
    basis_fingerprint, stored_basis_fingerprint = (
        schema_validator.validate_schema_and_basis(dataset, data_dict)
    )

    cell, pos, atomic_numbers = (
        data_dict[AtomicDataDict.CELL_KEY],
        data_dict[AtomicDataDict.POSITIONS_KEY],
        data_dict[AtomicDataDict.ATOMIC_NUMBERS_KEY],
    )
    pbc = data_dict[AtomicDataDict.PBC_KEY]

    get_Hamiltonian = dataset.get_Hamiltonian
    get_H0 = dataset.get_H0
    get_prior = dataset.get_prior
    get_DM = dataset.get_DM
    get_overlap = dataset.get_overlap
    orthogonal = dataset.orthogonal
    h0_key = dataset.h0_key
    prefer_precomputed_h0 = dataset.prefer_precomputed_h0
    prefer_precomputed_prior = dataset.prefer_precomputed_prior
    residual_hamiltonian = getattr(dataset, "residual_hamiltonian", False)
    require_full_h_target = getattr(dataset, "require_full_h_target", False)
    require_uureal_block_ode = getattr(dataset, "require_uureal_block_ode", False)
    residual_shrink_policy = getattr(dataset, "residual_shrink_policy", "error")
    min_residual_shrink = getattr(dataset, "min_residual_shrink", 1.2)
    require_prior_blocks = getattr(dataset, "require_prior_blocks", False)
    audit_prior_representations = getattr(dataset, "audit_prior_representations", False)
    expected_prior_source_fingerprint = getattr(
        dataset, "expected_prior_source_fingerprint", None
    )
    type_mapper = dataset.type_mapper
    prior_raw_key = dataset.prior_raw_key

    if get_Hamiltonian:
        blocks = data_dict.get("hamiltonian", None)

    if get_overlap:
        overlap = data_dict.get("overlap", None)
    else:
        overlap = False

    if get_DM:
        blocks = data_dict.get("density_matrix", None)

    if not (get_Hamiltonian or get_DM):
        blocks = False

    pre_node_features = data_dict.get(AtomicDataDict.NODE_FEATURES_KEY, None)
    pre_edge_features = data_dict.get(AtomicDataDict.EDGE_FEATURES_KEY, None)
    pre_node_overlap = data_dict.get(AtomicDataDict.NODE_OVERLAP_KEY, None)
    pre_edge_overlap = data_dict.get(AtomicDataDict.EDGE_OVERLAP_KEY, None)

    h0_blocks = data_dict.get(h0_key, None) if get_H0 else None
    node_h0 = data_dict.get(AtomicDataDict.NODE_H0_KEY, None) if get_H0 else None
    edge_h0 = data_dict.get(AtomicDataDict.EDGE_H0_KEY, None) if get_H0 else None
    p2_blocks = data_dict.get(prior_raw_key, None) if get_prior else None
    node_p2 = data_dict.get(prior_spec.node_rme_key, None) if get_prior else None
    edge_p2 = data_dict.get(prior_spec.edge_rme_key, None) if get_prior else None
    p2_blockwise_keys = prior_spec.block_fields
    p2_blockwise_present = [key in data_dict for key in p2_blockwise_keys]
    full_h_target_present = [
        key in data_dict for key in _host._PREPACKED_FULL_H_TARGET_KEYS
    ]
    schema_v2_row_aligned = bool(
        data_dict.get(SAMPLE_SCHEMA_KEY)
        in {P2_SAMPLE_SCHEMA, DUAL_PRIOR_SAMPLE_SCHEMA}
        and any(field in data_dict for field in ROW_ALIGNED_FIELD_CANDIDATES)
    )
    p2_feature_present = (node_p2 is not None, edge_p2 is not None)
    if any(p2_feature_present) and not all(p2_feature_present):
        raise ValueError(
            f"{prior_spec.kind.upper()} prior must provide both "
            f"{prior_spec.node_rme_key} and {prior_spec.edge_rme_key}; "
            "refusing a partial prior."
        )
    if p2_blocks is not None and not record_contract_already_validated:
        _host.validate_non_soc_p2_blocks(p2_blocks, key=prior_raw_key)
    node_physical_h0 = data_dict.get(AtomicDataDict.NODE_PHYSICAL_H0_KEY, None)
    edge_physical_h0 = data_dict.get(AtomicDataDict.EDGE_PHYSICAL_H0_KEY, None)
    physical_h0_present = (node_physical_h0 is not None, edge_physical_h0 is not None)
    if any(physical_h0_present) and not all(physical_h0_present):
        raise ValueError(
            "Offline physical H0 must provide both node_physical_h0 and "
            "edge_physical_h0; refusing a partial prior."
        )
    haar_u0 = data_dict.get(AtomicDataDict.HAAR_U0_KEY, None)
    haar_node_features = data_dict.get(AtomicDataDict.HAAR_NODE_FEATURES_KEY, None)
    haar_edge_features = data_dict.get(AtomicDataDict.HAAR_EDGE_FEATURES_KEY, None)
    soc_uureal_keep_mask = getattr(type_mapper, "mask_uureal", None)

    file_entry = dataset.info_files[dataset.file_map[idx]]
    if file_entry['train_dip'] == True:
        file_entry.update({'dip': data_dict['dipole_moment']})

    if file_entry['train_w_charge'] == True:
        file_entry.update({'charge': np.array(data_dict['charge'])})

    if file_entry['train_w_eps'] == True:
        file_entry.update({'dielectric_constant': np.array(data_dict['dielectric_constant'])})
    if file_entry['train_w_homo_lumo_gap'] == True:
        file_entry.update({
            'GAP_eV': np.array(data_dict['GAP_eV']),
            'LUMO_eV': np.array(data_dict['LUMO_eV']),
            'HOMO_eV': np.array(data_dict['HOMO_eV']),
        })

    if file_entry['train_polar'] == True:
        file_entry.update({'polar': data_dict['polarizability']})

    if file_entry['wave_align'] == True:
        orbital_energies = data_dict.get('orbital_energies', 0)
        orbital_coefficients = data_dict.get('orbital_coefficients', 0)
        file_entry.update(
            {'orbital_energies': orbital_energies, 'orbital_coefficients': orbital_coefficients})

    cache_keys = [
        'train_polar',
        'train_dip',
        'wave_align',
        'train_w_charge',
        'train_w_eps',
        'train_w_homo_lumo_gap',
    ]
    cache_info = {
        key: file_entry[key] for key in cache_keys
    }

    # transform blocks to atomicdata features, or use precomputed features directly
    need_overlap_features = bool(get_overlap)
    has_pre_main = pre_node_features is not None and pre_edge_features is not None
    has_pre_overlap = (not need_overlap_features) or (
        pre_edge_overlap is not None and (orthogonal or pre_node_overlap is not None)
    )

    uses_pre_main = bool(has_pre_main and has_pre_overlap)
    uses_pre_physical_h0 = bool(all(physical_h0_present))
    uses_pre_h0 = bool(
        get_H0
        and prefer_precomputed_h0
        and node_h0 is not None
        and edge_h0 is not None
    )
    uses_p2 = bool(
        get_prior
        and (all(p2_feature_present) or p2_blocks is not None)
    )
    stored_edge_index = data_dict.get(AtomicDataDict.EDGE_INDEX_KEY, None)
    stored_edge_shift = data_dict.get(AtomicDataDict.EDGE_CELL_SHIFT_KEY, None)
    graph_present = (stored_edge_index is not None, stored_edge_shift is not None)
    if any(graph_present) and not all(graph_present):
        raise ValueError(
            "Stored edge graph must provide both edge_index and edge_cell_shift; "
            "refusing an XOR/partial graph."
        )
    has_stored_edge_graph = all(graph_present)
    requires_stored_p2_graph = bool(
        get_prior and (all(p2_feature_present) or any(p2_blockwise_present))
    )
    requires_stored_target_graph = bool(
        require_full_h_target
        and any(full_h_target_present)
    )
    requires_fingerprinted_graph = bool(
        requires_stored_p2_graph
        or requires_stored_target_graph
        or schema_v2_row_aligned
    )
    if requires_fingerprinted_graph and not has_stored_edge_graph:
        raise ValueError(
            "Schema-v2 row-aligned tensors require a complete stored edge graph; "
            "graph regeneration could silently change edge row order."
        )

    # Keep dataset metadata immutable across failed reads.  The six training-only
    # flags are excluded from the AtomicData constructor kwargs without deleting
    # them in place, so a fingerprint failure never poisons the dataset.
    info = {
        key: value
        for key, value in file_entry.items()
        if key not in cache_info
    }

    load_prior_blocks = bool(
        get_prior
        and (require_prior_blocks or audit_prior_representations)
    )
    dedicated_h0 = bool(
        require_full_h_target
        and get_H0
        and data_dict.get(PHYSICAL_H0_SOURCE_KEY)
        == DEDICATED_PHYSICAL_H0_SOURCE
    )

    mapper_p2_irreps = getattr(
        getattr(dataset, "type_mapper", None), "orbpair_irreps", None
    )
    mapper_p2_dim = getattr(mapper_p2_irreps, "dim", None)

    return SampleContext(
        idx=idx,
        data_dict=data_dict,
        record_contract_key=record_contract_key,
        record_contract_already_validated=record_contract_already_validated,
        validated_record_contracts=validated_record_contracts,
        basis_fingerprint=basis_fingerprint,
        stored_basis_fingerprint=stored_basis_fingerprint,
        get_Hamiltonian=get_Hamiltonian,
        get_H0=get_H0,
        get_prior=get_prior,
        get_DM=get_DM,
        get_overlap=get_overlap,
        orthogonal=orthogonal,
        h0_key=h0_key,
        prefer_precomputed_h0=prefer_precomputed_h0,
        prefer_precomputed_prior=prefer_precomputed_prior,
        residual_hamiltonian=residual_hamiltonian,
        require_full_h_target=require_full_h_target,
        require_uureal_block_ode=require_uureal_block_ode,
        residual_shrink_policy=residual_shrink_policy,
        min_residual_shrink=min_residual_shrink,
        require_prior_blocks=require_prior_blocks,
        audit_prior_representations=audit_prior_representations,
        expected_prior_source_fingerprint=expected_prior_source_fingerprint,
        type_mapper=type_mapper,
        prior_spec=prior_spec,
        prior_raw_key=prior_raw_key,
        cell=cell,
        pos=pos,
        atomic_numbers=atomic_numbers,
        pbc=pbc,
        blocks=blocks,
        overlap=overlap,
        pre_node_features=pre_node_features,
        pre_edge_features=pre_edge_features,
        pre_node_overlap=pre_node_overlap,
        pre_edge_overlap=pre_edge_overlap,
        h0_blocks=h0_blocks,
        node_h0=node_h0,
        edge_h0=edge_h0,
        p2_blocks=p2_blocks,
        node_p2=node_p2,
        edge_p2=edge_p2,
        p2_blockwise_keys=p2_blockwise_keys,
        p2_blockwise_present=p2_blockwise_present,
        p2_feature_present=p2_feature_present,
        full_h_target_present=full_h_target_present,
        schema_v2_row_aligned=schema_v2_row_aligned,
        node_physical_h0=node_physical_h0,
        edge_physical_h0=edge_physical_h0,
        physical_h0_present=physical_h0_present,
        haar_u0=haar_u0,
        haar_node_features=haar_node_features,
        haar_edge_features=haar_edge_features,
        soc_uureal_keep_mask=soc_uureal_keep_mask,
        info=info,
        cache_info=cache_info,
        mapper_p2_dim=mapper_p2_dim,
        need_overlap_features=need_overlap_features,
        has_pre_main=has_pre_main,
        has_pre_overlap=has_pre_overlap,
        uses_pre_main=uses_pre_main,
        uses_pre_physical_h0=uses_pre_physical_h0,
        uses_pre_h0=uses_pre_h0,
        uses_p2=uses_p2,
        load_prior_blocks=load_prior_blocks,
        dedicated_h0=dedicated_h0,
        stored_edge_index=stored_edge_index,
        stored_edge_shift=stored_edge_shift,
        has_stored_edge_graph=has_stored_edge_graph,
        requires_stored_p2_graph=requires_stored_p2_graph,
        requires_stored_target_graph=requires_stored_target_graph,
        requires_fingerprinted_graph=requires_fingerprinted_graph,
    )


class RecordPipeline:
    """Threads the immutable context through the decode collaborators in order."""

    def __init__(self) -> None:
        self.reader = StoredRecordReader()
        self.schema_validator = RecordSchemaValidator()
        self.graph_resolver = GraphResolver()
        self.atomic_data_factory = AtomicDataFactory()
        self.prior_decoder = PriorDecoder()
        self.target_decoder = TargetDecoder()

    def build(self, dataset: Any, idx: int) -> Any:
        data_dict = self.reader.read(dataset, idx)
        ctx = build_sample_context(dataset, idx, data_dict, self.schema_validator)
        graph = self.graph_resolver.resolve(ctx)
        atomicdata = self.atomic_data_factory.build(ctx, graph)
        # Restore the training-only flags snapshot (no-op re-write kept for
        # exact parity with the historical get()).
        dataset.info_files[dataset.file_map[idx]].update(ctx.cache_info)

        num_edges = atomicdata[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        num_nodes = atomicdata.num_nodes

        self.target_decoder.decode_main_features(ctx, atomicdata, num_nodes, num_edges)
        self.target_decoder.decode_h0(ctx, atomicdata, num_nodes, num_edges)
        self.prior_decoder.decode_features(ctx, atomicdata, num_nodes, num_edges)
        self.target_decoder.decode_physical_h0(ctx, atomicdata, num_nodes, num_edges)
        self.target_decoder.decode_haar(ctx, atomicdata, num_nodes, num_edges)
        self.target_decoder.assemble_block_tensors(ctx, atomicdata, num_nodes, num_edges)
        self.prior_decoder.validate_blocks(ctx, atomicdata, num_nodes, num_edges)
        self.target_decoder.validate_full_h(ctx, atomicdata)
        self.schema_validator.mark_validated(ctx, graph)

        # Attach a stable, composition-independent per-graph record identity for
        # deterministic seeded prior streams.
        atomicdata[AtomicDataDict.SAMPLE_UID_KEY] = torch.tensor(
            [dataset._compute_sample_uid(idx)], dtype=torch.long
        )

        return atomicdata


# Stateless singleton: the collaborators hold no per-dataset state (the dataset
# and immutable context are passed in), so one shared instance serves every
# worker without polluting dataset pickling.
_RECORD_PIPELINE = RecordPipeline()


def build_atomic_data(dataset: Any, idx: int) -> Any:
    """Assemble the :class:`AtomicData` for ``dataset[idx]`` via the pipeline."""
    return _RECORD_PIPELINE.build(dataset, idx)

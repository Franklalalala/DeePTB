#!/usr/bin/env python3
"""Augment a qualified non-SOC P2 Full-H cache with an independent P23 prior.

The input is the compact ``full_h`` view emitted by
``materialize_nonsoc_p2_cache.py``.  It is never modified.  For every stored
canonical graph this tool assembles only the factorized local-three-centre VNA
addition, rotates that addition from the table's ABACUS AO gauge to DeePTB AO
order, and writes one dual-prior record containing both complete representations:

* the original ``node_p2``/``edge_p2`` and packed P2 AO blocks, byte-for-byte;
* independent ``node_p23``/``edge_p23`` and packed P23 AO blocks, where
  ``P23 = P2 + VNA3c_factorized``;
* the original dedicated absolute Full-H target.

Every source record is strict-loaded and all graph, basis, target, P2, P23 and
row-bundle fingerprints are recomputed.  The P23 bundle explicitly depends on
the P2 bundle from the same record.  Resume accepts only an exact identity and
an intact committed prefix; a single LMDB row written ahead of the partial
manifest is treated as an uncommitted tail and removed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import lmdb
import numpy as np
import torch

import dptb.data.AtomicDataDict as AtomicDataDict
from dptb.data.interfaces.abacus import OrbAbacus2DeepTB
from dptb.data.interfaces.blockwise_tensor import (
    block_mask_from_shapes,
    block_tensors_to_feature_tensors,
    feature_tensors_to_block_tensors,
)
from dptb.data.interfaces.h0_lmdb_helper import _build_context_dataset
from dptb.data.interfaces.p2_contract import (
    BASIS_FINGERPRINT_KEY,
    DUAL_PRIOR_SAMPLE_SCHEMA,
    EDGE_GRAPH_FINGERPRINT_KEY,
    FULL_H_TARGET_FINGERPRINT_KEY,
    P2_BLOCK_FINGERPRINT_KEY,
    P2_BUNDLE_FINGERPRINT_KEY,
    P2_RME_FINGERPRINT_KEY,
    P2_SAMPLE_SCHEMA,
    P2_SOURCE_FINGERPRINT_KEY,
    P23_BLOCK_FINGERPRINT_KEY,
    P23_BUNDLE_FINGERPRINT_KEY,
    P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
    P23_RME_FINGERPRINT_KEY,
    P23_SOURCE_FINGERPRINT_KEY,
    ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
    ROW_ALIGNED_DATA_FINGERPRINT_KEY,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
    assert_record_fingerprint,
    edge_graph_fingerprint,
    fingerprint_fields,
    fingerprint_present_row_aligned_fields,
    fingerprint_text_fields,
    mapper_basis_fingerprint,
    require_sha256,
    resolve_prior_field_spec,
)
from dptb.data.interfaces.p23_table import (
    P23_VNA_ASSEMBLY_SCHEMA,
    P23VNAFactorAssembler,
    P23VNAFactorTableStore,
)
from dptb.data.dataset.lmdb_dataset import (
    assert_absolute_full_h_target_contract,
    validate_non_soc_p2_block_tensors,
)
from dptb.utils.constants import Bohr2Ang

from dptb.data.materialization import (
    BoundedEnvCache,
    DatasetAuditor,
    ManifestStore,
    NONSOC_P2_CACHE_SCHEMA as P2_CACHE_SCHEMA,
    ResumeJournal,
    TransactionalLMDBWriter,
    bytes_sha256 as _bytes_sha256,
    drop_uncommitted_tail,
    guard_work_root,
    json_sha256 as _json_sha256,
    load_module as _load_module,
    open_read_env as _open_read,
    sha256_file as _sha256,
)
from dptb.data.materialization.geometry import (
    GEOMETRY_TOLERANCE_ANGSTROM,
    reverse_edge_rows as _reverse_edge_rows,
    structure_geometry_bohr as _structure_geometry_bohr,
)


SCHEMA = "deeptb.nonsoc_dual_prior_cache/v1"
IDENTITY_SCHEMA = "deeptb.nonsoc_dual_prior_cache_identity/v1"
AGGREGATE_P2_CACHE_SCHEMA = "deeptb.raw200_parallel_cache/v1"
# GEOMETRY_TOLERANCE_ANGSTROM is imported from
# dptb.data.materialization.geometry (single source of truth for the
# STRU/compact serialization tolerance; raw200 has legitimate cell round-off
# just above 5e-5 A while 1e-4 stays far below any physical displacement).
# Resume journal bound to this cache's schema; drives heartbeats.
_JOURNAL = ResumeJournal(schema=SCHEMA)
LEGACY_GEOMETRY_IDENTITY_UPGRADES = {
    # 7e67c343: periodic-image aware materializer with the original 5e-5 A
    # serialization tolerance.  The upgrade changes only that tolerance.
    "8ce23a6335aed526f94b6ba73a63cafaa71865c42c2136f7982498b69d3303ed": 5.0e-5,
}
RME_AO_ROUNDTRIP_TOLERANCE_EV = 5.0e-5

_BASE_FIELDS = (
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.ATOMIC_NUMBERS_KEY,
    AtomicDataDict.PBC_KEY,
)
_GRAPH_FIELDS = (
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
)
_MAIN_RME_FIELDS = (
    AtomicDataDict.NODE_FEATURES_KEY,
    AtomicDataDict.EDGE_FEATURES_KEY,
)
_H0_RME_FIELDS = (
    AtomicDataDict.NODE_H0_KEY,
    AtomicDataDict.EDGE_H0_KEY,
)
_H0_BLOCK_FIELDS = (
    AtomicDataDict.NODE_H0_BLOCKS_KEY,
    AtomicDataDict.EDGE_H0_BLOCKS_KEY,
    AtomicDataDict.NODE_H0_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_H0_BLOCK_SHAPE_KEY,
)
_P2_RME_FIELDS = (
    AtomicDataDict.NODE_P2_KEY,
    AtomicDataDict.EDGE_P2_KEY,
)
_P2_BLOCK_FIELDS = (
    AtomicDataDict.NODE_P2_BLOCKS_KEY,
    AtomicDataDict.EDGE_P2_BLOCKS_KEY,
    AtomicDataDict.NODE_P2_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_P2_BLOCK_SHAPE_KEY,
)
_P23_RME_FIELDS = (
    AtomicDataDict.NODE_P23_KEY,
    AtomicDataDict.EDGE_P23_KEY,
)
_P23_BLOCK_FIELDS = (
    AtomicDataDict.NODE_P23_BLOCKS_KEY,
    AtomicDataDict.EDGE_P23_BLOCKS_KEY,
    AtomicDataDict.NODE_P23_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_P23_BLOCK_SHAPE_KEY,
)
_FULL_H_TARGET_FIELDS = (
    AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY,
    AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY,
    AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
)
_P23_PROVENANCE_FIELDS = (
    P23_SOURCE_FINGERPRINT_KEY,
    P23_RME_FINGERPRINT_KEY,
    P23_BLOCK_FINGERPRINT_KEY,
    P23_BUNDLE_FINGERPRINT_KEY,
    P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
    "p23_cache_source",
    "p23_assembly",
)


def _require_fields(record: Mapping[str, Any], fields: Iterable[str], *, label: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise KeyError(f"{label} is missing required fields {missing}.")


def _guard_roots(
    source_root: Path,
    work_root: Path,
    *,
    overwrite: bool,
    resume: bool,
) -> None:
    resolved_work = work_root.resolve()
    guard_work_root(
        source_root,
        work_root,
        overwrite=overwrite,
        resume=resume,
        disjoint_message=(
            "P2 cache root and dual-cache work root must be disjoint; neither "
            "may contain the other."
        ),
        mutually_exclusive_message="--overwrite and --resume are mutually exclusive.",
        missing_work_root_message=f"--resume requires existing work root {resolved_work}.",
    )


def _heartbeat(work_root: Path, **payload: Any) -> None:
    _JOURNAL.heartbeat(work_root, **payload)


def _source_case_map(dataset_root: Path) -> dict[str, Path]:
    ordered_path = dataset_root / "ordered_paths.txt"
    if not ordered_path.is_file():
        raise FileNotFoundError(ordered_path)
    result: dict[str, Path] = {}
    for raw in ordered_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        case = Path(raw.strip())
        if not case.is_absolute():
            case = dataset_root / case
        case = case.resolve()
        if case.name in result:
            raise ValueError(f"ordered_paths.txt repeats case id {case.name!r}.")
        if not (case / "STRU").is_file():
            raise FileNotFoundError(case / "STRU")
        result[case.name] = case
    if not result:
        raise ValueError("ordered_paths.txt selects no source structures.")
    return result


def _split_contracts(
    *,
    p2_cache_root: Path,
    p2_manifest: Mapping[str, Any],
    dataset_root: Path,
    selected_splits: Sequence[str],
    selected_cases: Sequence[str],
) -> dict[str, dict[str, Any]]:
    manifest_splits = p2_manifest.get("splits")
    if not isinstance(manifest_splits, Mapping) or not manifest_splits:
        raise ValueError("P2 cache manifest has no split contract.")
    splits = list(selected_splits) if selected_splits else list(manifest_splits)
    if len(set(splits)) != len(splits):
        raise ValueError("--split values must be unique.")
    case_map = _source_case_map(dataset_root)
    wanted = {str(case) for case in selected_cases}
    found: set[str] = set()
    contracts: dict[str, dict[str, Any]] = {}
    for split in splits:
        if split not in manifest_splits:
            raise KeyError(f"P2 cache manifest has no split {split!r}.")
        source_split = p2_cache_root / split
        if not source_split.is_dir():
            raise NotADirectoryError(source_split)

        def shard_order(path: Path) -> tuple[int, str]:
            stem = path.name[: -len(".lmdb")]
            tail = stem.rsplit(".", 1)[-1]
            return (int(tail), path.name) if tail.isdigit() else (1 << 30, path.name)

        lmdb_paths = sorted(source_split.glob("*.lmdb"), key=shard_order)
        if not lmdb_paths:
            raise FileNotFoundError(f"P2 split {split!r} has no physical LMDB shards.")
        shards: list[dict[str, Any]] = []
        cases: list[str] = []
        source_rows: list[tuple[Path, int, str]] = []
        for lmdb_path in lmdb_paths:
            paths_path = source_split / f"{lmdb_path.name[:-len('.lmdb')]}.paths.txt"
            if not paths_path.is_file():
                raise FileNotFoundError(paths_path)
            shard_case_paths = [
                Path(line.strip())
                for line in paths_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            shard_cases = [path.name for path in shard_case_paths]
            env = _open_read(lmdb_path)
            try:
                with env.begin() as txn:
                    entries = int(txn.stat()["entries"])
            finally:
                env.close()
            if entries != len(shard_cases):
                raise ValueError(
                    f"P2 shard {lmdb_path.name!r} has {entries} rows but "
                    f"{len(shard_cases)} path records."
                )
            offset = len(cases)
            cases.extend(shard_cases)
            source_rows.extend(
                (lmdb_path, local_index, case_id)
                for local_index, case_id in enumerate(shard_cases)
            )
            shards.append(
                {
                    "name": lmdb_path.name,
                    "lmdb": lmdb_path,
                    "paths": paths_path,
                    "paths_sha256": _sha256(paths_path),
                    "entries": entries,
                    "global_offset": offset,
                    "cases": shard_cases,
                }
            )
        split_meta = manifest_splits[split]
        declared_cases = split_meta.get("cases") if isinstance(split_meta, Mapping) else None
        if isinstance(declared_cases, list) and [str(value) for value in declared_cases] != cases:
            raise ValueError(f"P2 split {split!r} physical shard order differs from manifest cases.")
        if isinstance(split_meta, Mapping):
            for field in ("physical_cases", "records"):
                declared = split_meta.get(field)
                if isinstance(declared, int) and int(declared) != len(cases):
                    raise ValueError(
                        f"P2 aggregate split {split!r} declares {field}={declared}, "
                        f"but physical shards contain {len(cases)} rows."
                    )
                if isinstance(declared, list):
                    if len(declared) != len(cases):
                        raise ValueError(
                            f"P2 aggregate split {split!r} declares {len(declared)} "
                            f"{field} rows, but physical shards contain {len(cases)}."
                        )
                    if field == "physical_cases" and [
                        str(value) for value in declared
                    ] != cases:
                        raise ValueError(
                            f"P2 aggregate split {split!r} physical_cases order "
                            "differs from the numeric shard/path order."
                        )
        missing = [case for case in cases if case not in case_map]
        if missing:
            raise KeyError(
                f"Source dataset lacks {split!r} cases declared by P2 cache: {missing[:5]}."
            )
        if len(set(cases)) != len(cases):
            raise ValueError(f"P2 split {split!r} repeats a physical case id.")
        all_cases = list(cases)
        all_source_rows = list(source_rows)
        selected_global_indices = list(range(len(all_cases)))
        if wanted:
            selected_global_indices = [
                index for index, case_id in enumerate(all_cases) if case_id in wanted
            ]
            if not selected_global_indices:
                continue
            found.update(all_cases[index] for index in selected_global_indices)
            cases = [all_cases[index] for index in selected_global_indices]
            source_rows = [all_source_rows[index] for index in selected_global_indices]
        combined_source_identity = _json_sha256(
            {
                "cases": cases,
                "selected_source_global_indices": selected_global_indices,
                "shards": [
                    {
                        "name": shard["name"],
                        "entries": shard["entries"],
                        "paths_sha256": shard["paths_sha256"],
                    }
                    for shard in shards
                ],
            }
        )
        contracts[split] = {
            "source_split": source_split,
            "source_shards": shards,
            "all_source_rows": all_source_rows,
            "source_rows": source_rows,
            "source_paths_sha256": combined_source_identity,
            "cases": cases,
            "case_paths": [case_map[case] for case in cases],
            "entries": len(cases),
            "source_total_entries": len(all_cases),
            "selected_source_global_indices": selected_global_indices,
            "logical_cases": (
                split_meta.get("logical_cases") if isinstance(split_meta, Mapping) else None
            ),
        }
    missing_wanted = sorted(wanted - found)
    if missing_wanted:
        raise KeyError(f"Requested physical cases are absent from selected splits: {missing_wanted}.")
    if not contracts:
        raise ValueError("Case/split selection produced no physical records.")
    return contracts


def _validate_table_lineage(
    p2_manifest: Mapping[str, Any],
    p23_store: P23VNAFactorTableStore,
) -> tuple[str, str]:
    schema = p2_manifest.get("schema")
    if schema not in {P2_CACHE_SCHEMA, AGGREGATE_P2_CACHE_SCHEMA}:
        raise ValueError(
            f"Unsupported P2 cache schema {schema!r}; expected a serial P2 cache "
            "or the qualified raw200 aggregate cache."
        )
    p2_source = p2_manifest.get("p2_source")
    if schema == P2_CACHE_SCHEMA and (
        not isinstance(p2_source, Mapping) or p2_source.get("kind") != "radial_table"
    ):
        raise ValueError("P23 augmentation requires a radial-table P2 cache source.")
    p23_fingerprint = require_sha256(
        p23_store.manifest_sha256, field="p23_source_fingerprint"
    )
    parent = require_sha256(
        p23_store.manifest.get("base_p2_table_manifest_sha256"),
        field="base_p2_table_manifest_sha256",
    )
    declared_p2 = p2_manifest.get("p2_source_fingerprint")
    if declared_p2 not in (None, "") and parent != require_sha256(
        declared_p2, field="p2_source_fingerprint"
    ):
        raise ValueError(
            "P23 table was not built from this cache's P2 table manifest: "
            f"P23 parent={parent}, cache P2={declared_p2}."
        )
    p2_fingerprint = parent
    cache_species_identity = (
        p2_source.get("species_source_identity_sha256")
        if isinstance(p2_source, Mapping)
        else None
    )
    table_species_identity = p23_store.manifest.get(
        "species_source_identity_sha256"
    )
    if cache_species_identity not in (None, ""):
        if require_sha256(
            cache_species_identity, field="P2 species_source_identity_sha256"
        ) != require_sha256(
            table_species_identity, field="P23 species_source_identity_sha256"
        ):
            raise ValueError("P2 cache and P23 table species source identities differ.")

    interface_path = Path(__file__).resolve().parents[1] / "dptb/data/interfaces/p23_table.py"
    expected_interface = p23_store.manifest.get("code_identity", {}).get(
        "p23_interface_sha256"
    )
    if require_sha256(
        expected_interface, field="P23 table p23_interface_sha256"
    ) != _sha256(interface_path):
        raise ValueError(
            "The active P23 assembler code differs from the code frozen in the "
            "factor-table manifest."
        )
    return p2_fingerprint, p23_fingerprint


def _validate_p23_table_audit(
    audit_path: Path,
    *,
    expected_table_manifest_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("P23 table audit must be a JSON mapping.")
    expected_schema = "deeptb.p23_factorized_vna_table_audit/v1"
    schema = str(payload.get("schema", ""))
    if schema != expected_schema:
        raise ValueError(
            f"P23 table audit schema {schema!r} != {expected_schema!r}."
        )
    if payload.get("status") != "pass":
        raise ValueError("P23 table audit status must be exactly 'pass'.")
    declared_manifest = require_sha256(
        payload.get("manifest_sha256"), field="audit manifest_sha256"
    )
    if declared_manifest != require_sha256(
        expected_table_manifest_sha256, field="expected P23 table manifest SHA256"
    ):
        raise ValueError("P23 table audit belongs to a different table manifest.")
    if int(payload.get("verified_shards", -1)) != 950:
        raise ValueError("P23 table audit must verify exactly 950 table shards.")
    for field in (
        "integrity_eligible",
        "experimental_training_eligible",
        "production_eligible",
    ):
        if payload.get(field) is not True:
            raise ValueError(f"P23 table audit must declare {field}=true.")
    physical_oracle = payload.get("physical_oracle")
    if not isinstance(physical_oracle, Mapping) or physical_oracle.get("status") != "pass":
        raise ValueError("P23 table audit physical_oracle.status must be exactly 'pass'.")
    return dict(payload)


def _configure_strict_dataset(dataset: Any, *, kind: str, source: str) -> None:
    spec = resolve_prior_field_spec(kind)
    dataset.get_Hamiltonian = True
    dataset.get_H0 = True
    dataset.get_P2 = True
    dataset.residual_hamiltonian = False
    dataset.prior_spec = spec
    dataset.prior_kind = spec.kind
    dataset.p2_key = spec.raw_key
    dataset.prefer_precomputed_p2 = True
    dataset.require_full_h_target = True
    dataset.expected_p2_source_fingerprint = source
    dataset.audit_p2_representations = True
    dataset.require_p2_blocks = True


def _close_dataset(dataset: Any) -> None:
    for env in getattr(dataset, "_lmdb_env_cache", {}).values():
        env.close()
    dataset._lmdb_env_cache = {}


def _assert_source_p2_contract(
    record: Mapping[str, Any],
    data: Mapping[str, Any],
    idp: Any,
    *,
    expected_p2_source: str,
) -> None:
    if record.get(SAMPLE_SCHEMA_KEY) != P2_SAMPLE_SCHEMA:
        raise ValueError(
            "The augmentation source must be the qualified P2-only compact schema; "
            f"got {record.get(SAMPLE_SCHEMA_KEY)!r}."
        )
    unexpected = [field for field in (*_P23_RME_FIELDS, *_P23_BLOCK_FIELDS, *_P23_PROVENANCE_FIELDS) if field in record]
    if unexpected:
        raise ValueError(
            f"Source P2 record already contains P23 fields {unexpected}; refusing overwrite."
        )
    _require_fields(
        record,
        (
            *_BASE_FIELDS,
            *_GRAPH_FIELDS,
            *_MAIN_RME_FIELDS,
            *_H0_RME_FIELDS,
            *_H0_BLOCK_FIELDS,
            *_P2_RME_FIELDS,
            *_P2_BLOCK_FIELDS,
            *_FULL_H_TARGET_FIELDS,
            BASIS_FINGERPRINT_KEY,
            EDGE_GRAPH_FINGERPRINT_KEY,
            P2_SOURCE_FINGERPRINT_KEY,
            P2_RME_FINGERPRINT_KEY,
            P2_BLOCK_FINGERPRINT_KEY,
            P2_BUNDLE_FINGERPRINT_KEY,
            FULL_H_TARGET_FINGERPRINT_KEY,
            ROW_ALIGNED_DATA_FINGERPRINT_KEY,
            ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
            TARGET_SEMANTICS_KEY,
            TARGET_SOURCE_KEY,
            "p2_cache_source",
        ),
        label="P2 source record",
    )
    assert_absolute_full_h_target_contract(dict(record))
    if record[TARGET_SOURCE_KEY] != "dedicated_full_h_blocks":
        raise ValueError("Dual-prior cache requires dedicated absolute Full-H blocks.")
    basis = mapper_basis_fingerprint(idp)
    assert_record_fingerprint(record, field=BASIS_FINGERPRINT_KEY, actual=basis)
    graph = edge_graph_fingerprint(
        record[AtomicDataDict.ATOMIC_NUMBERS_KEY],
        record[AtomicDataDict.EDGE_INDEX_KEY],
        record[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
        basis_fingerprint=basis,
    )
    assert_record_fingerprint(record, field=EDGE_GRAPH_FINGERPRINT_KEY, actual=graph)
    stored_source = require_sha256(
        record[P2_SOURCE_FINGERPRINT_KEY], field=P2_SOURCE_FINGERPRINT_KEY
    )
    if stored_source != require_sha256(
        expected_p2_source, field="expected_p2_source_fingerprint"
    ):
        raise ValueError("P2 source fingerprint differs from the qualified cache manifest.")
    source_metadata = record["p2_cache_source"]
    if not isinstance(source_metadata, Mapping):
        raise ValueError("P2 source metadata must be a mapping.")
    expected_source_metadata = {
        "kind": "radial_table",
        "energy_unit": "eV",
        "ao_order": "DeePTB",
    }
    for field, expected in expected_source_metadata.items():
        if source_metadata.get(field) != expected:
            raise ValueError(
                f"P2 source metadata {field}={source_metadata.get(field)!r} "
                f"!= {expected!r}."
            )
    if require_sha256(
        source_metadata.get("table_manifest_sha256"),
        field="p2_cache_source.table_manifest_sha256",
    ) != stored_source:
        raise ValueError(
            "P2 source metadata table manifest SHA differs from the record P2 source."
        )
    p2_rme = fingerprint_fields(record, _P2_RME_FIELDS)
    p2_blocks = fingerprint_fields(record, _P2_BLOCK_FIELDS)
    assert_record_fingerprint(record, field=P2_RME_FINGERPRINT_KEY, actual=p2_rme)
    assert_record_fingerprint(record, field=P2_BLOCK_FINGERPRINT_KEY, actual=p2_blocks)
    p2_bundle = fingerprint_text_fields(
        record,
        (
            BASIS_FINGERPRINT_KEY,
            EDGE_GRAPH_FINGERPRINT_KEY,
            P2_SOURCE_FINGERPRINT_KEY,
            P2_RME_FINGERPRINT_KEY,
            P2_BLOCK_FINGERPRINT_KEY,
        ),
    )
    assert_record_fingerprint(record, field=P2_BUNDLE_FINGERPRINT_KEY, actual=p2_bundle)
    assert_record_fingerprint(
        record,
        field=FULL_H_TARGET_FINGERPRINT_KEY,
        actual=fingerprint_fields(record, _FULL_H_TARGET_FIELDS),
    )
    row_data = fingerprint_present_row_aligned_fields(record)
    assert_record_fingerprint(record, field=ROW_ALIGNED_DATA_FINGERPRINT_KEY, actual=row_data)
    assert_record_fingerprint(
        record,
        field=ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
        actual=fingerprint_text_fields(
            record,
            (
                BASIS_FINGERPRINT_KEY,
                EDGE_GRAPH_FINGERPRINT_KEY,
                ROW_ALIGNED_DATA_FINGERPRINT_KEY,
            ),
        ),
    )
    node_shapes = np.asarray(record[AtomicDataDict.NODE_P2_BLOCK_SHAPE_KEY])
    edge_shapes = np.asarray(record[AtomicDataDict.EDGE_P2_BLOCK_SHAPE_KEY])
    for target_shapes in (
        (AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY, node_shapes),
        (AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY, edge_shapes),
        (AtomicDataDict.NODE_H0_BLOCK_SHAPE_KEY, node_shapes),
        (AtomicDataDict.EDGE_H0_BLOCK_SHAPE_KEY, edge_shapes),
    ):
        if not np.array_equal(np.asarray(record[target_shapes[0]]), target_shapes[1]):
            raise ValueError(
                f"P2 source shape field {target_shapes[0]} is not graph/basis aligned."
            )
    validate_non_soc_p2_block_tensors(
        record[AtomicDataDict.NODE_P2_BLOCKS_KEY],
        record[AtomicDataDict.EDGE_P2_BLOCKS_KEY],
        node_shapes,
        edge_shapes,
        num_nodes=int(np.asarray(record[AtomicDataDict.ATOMIC_NUMBERS_KEY]).size),
        num_edges=int(np.asarray(record[AtomicDataDict.EDGE_INDEX_KEY]).shape[1]),
        data=data,
        idp=idp,
        prior_kind="p2",
    )


# _structure_geometry_bohr and _reverse_edge_rows are imported (as module-level
# names) from dptb.data.materialization.geometry; see the import block above.


def transform_vna3c_abacus_to_deeptb(
    *,
    node_addition: Any,
    edge_addition: Any,
    symbols: Sequence[str],
    edge_index: Any,
    edge_cell_shift: Any,
    node_shapes: Any,
    edge_shapes: Any,
    species_contract: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rotate an assembler VNA3c addition into the compact P2 AO gauge.

    Factor tables use the explicit ``deeptb_abacus_real`` harmonic convention,
    i.e. their AO axes are still in ABACUS order.  Compact P2 blocks were
    already transformed by :class:`OrbAbacus2DeepTB`.  Direct addition before
    this step is therefore forbidden.
    """

    symbols = [str(symbol) for symbol in symbols]
    edge = np.asarray(edge_index, dtype=np.int64)
    node_shapes = np.asarray(node_shapes, dtype=np.int64)
    edge_shapes = np.asarray(edge_shapes, dtype=np.int64)
    node_in = np.asarray(node_addition)
    edge_in = np.asarray(edge_addition)
    if node_in.ndim != 3 or edge_in.ndim != 3:
        raise ValueError("VNA3c additions must be packed rank-3 block tensors.")
    if node_in.shape[0] != len(symbols) or edge_in.shape[0] != edge.shape[1]:
        raise ValueError("VNA3c addition rows do not match the canonical graph.")
    if node_shapes.shape != (len(symbols), 2) or edge_shapes.shape != (edge.shape[1], 2):
        raise ValueError("VNA3c shape rows do not match the canonical graph.")
    if np.iscomplexobj(node_in) or np.iscomplexobj(edge_in):
        raise NotImplementedError("P23 gauge transformation is non-SOC only.")
    if not np.isfinite(node_in).all() or not np.isfinite(edge_in).all():
        raise ValueError("VNA3c additions contain NaN or infinity.")

    raw_max_onsite_error = 0.0
    for atom, (rows, cols) in enumerate(node_shapes.tolist()):
        if int(rows) != int(cols):
            raise ValueError(f"assembler node {atom} has a non-square AO extent.")
        block = node_in[atom, : int(rows), : int(cols)]
        error = float(np.max(np.abs(block - block.T), initial=0.0))
        raw_max_onsite_error = max(raw_max_onsite_error, error)
        if not np.array_equal(block, block.T):
            raise ValueError(
                f"assembler node {atom} is not exactly symmetric before gauge transform; "
                f"max error={error:.3e}."
            )
    reverse = _reverse_edge_rows(edge, edge_cell_shift)
    raw_max_reverse_error = 0.0
    for row, reverse_row in enumerate(reverse.tolist()):
        rows, cols = (int(value) for value in edge_shapes[row])
        forward = edge_in[row, :rows, :cols]
        backward_t = edge_in[int(reverse_row), :cols, :rows].T
        error = float(np.max(np.abs(forward - backward_t), initial=0.0))
        raw_max_reverse_error = max(raw_max_reverse_error, error)
        if not np.array_equal(forward, backward_t):
            raise ValueError(
                f"assembler edge {row}/{reverse_row} is not an exact raw reverse "
                f"transpose before gauge transform; max error={error:.3e}."
            )

    converter = OrbAbacus2DeepTB()
    node_out = np.zeros(node_in.shape, dtype=np.float32)
    edge_out = np.zeros(edge_in.shape, dtype=np.float32)
    for atom, symbol in enumerate(symbols):
        shells = [int(value) for value in species_contract[symbol]["orbital_shells"]]
        expected = sum(2 * l + 1 for l in shells)
        rows, cols = (int(value) for value in node_shapes[atom])
        if (rows, cols) != (expected, expected):
            raise ValueError(
                f"node {atom} ({symbol}) AO shape {(rows, cols)} != {(expected, expected)}."
            )
        transformed = converter.transform(
            np.asarray(node_in[atom, :rows, :cols], dtype=np.float64), shells, shells
        )
        transformed = 0.5 * (transformed + transformed.T)
        node_out[atom, :rows, :cols] = transformed.astype(np.float32)

    transformed_pairs = 0
    for row, reverse_row in enumerate(reverse.tolist()):
        if row > int(reverse_row):
            continue
        if row == int(reverse_row):
            raise ValueError("main edge graph must not contain self-reverse onsite rows.")
        i, j = (int(value) for value in edge[:, row])
        sym_i, sym_j = symbols[i], symbols[j]
        shells_i = [int(value) for value in species_contract[sym_i]["orbital_shells"]]
        shells_j = [int(value) for value in species_contract[sym_j]["orbital_shells"]]
        rows, cols = (int(value) for value in edge_shapes[row])
        reverse_rows, reverse_cols = (
            int(value) for value in edge_shapes[int(reverse_row)]
        )
        expected = (
            sum(2 * l + 1 for l in shells_i),
            sum(2 * l + 1 for l in shells_j),
        )
        if (rows, cols) != expected or (reverse_rows, reverse_cols) != expected[::-1]:
            raise ValueError(f"edge {row}/{reverse_row} AO shape contract mismatch.")
        transformed = converter.transform(
            np.asarray(edge_in[row, :rows, :cols], dtype=np.float64),
            shells_i,
            shells_j,
        ).astype(np.float32)
        edge_out[row, :rows, :cols] = transformed
        edge_out[int(reverse_row), :cols, :rows] = transformed.T.copy()
        transformed_pairs += 1

    if not np.isfinite(node_out).all() or not np.isfinite(edge_out).all():
        raise ValueError("DeePTB-gauge VNA3c additions contain NaN or infinity.")
    max_reverse_error = 0.0
    for row, reverse_row in enumerate(reverse.tolist()):
        rows, cols = (int(value) for value in edge_shapes[row])
        reverse_block = edge_out[int(reverse_row), :cols, :rows].T
        max_reverse_error = max(
            max_reverse_error,
            float(np.max(np.abs(edge_out[row, :rows, :cols] - reverse_block), initial=0.0)),
        )
    if max_reverse_error != 0.0:
        raise RuntimeError("DeePTB-gauge VNA3c reverse edges are not exact transposes.")
    return node_out, edge_out, {
        "table_harmonic_convention": "deeptb_abacus_real",
        "assembler_ao_gauge": "ABACUS",
        "stored_ao_gauge": "DeePTB",
        "gauge_transform": "OrbAbacus2DeepTB.transform",
        "onsite_policy": "transform_then_exact_symmetrize",
        "edge_policy": "transform_representative_then_exact_reverse_transpose",
        "raw_max_onsite_symmetry_error_ev": raw_max_onsite_error,
        "raw_max_reverse_error_ev": raw_max_reverse_error,
        "transformed_edge_pairs": transformed_pairs,
        "max_reverse_error_ev": max_reverse_error,
    }


def _valid_max_error(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float((left - right).abs().masked_select(mask).max().detach().cpu())


def augment_p2_record_with_p23(
    *,
    record: Mapping[str, Any],
    data: Mapping[str, Any],
    idp: Any,
    assembler: P23VNAFactorAssembler,
    symbols: Sequence[str],
    positions_bohr: np.ndarray,
    cell_bohr: np.ndarray,
    p23_source_fingerprint: str,
    geometry_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one dual-prior record without mutating the qualified P2 source."""

    p23_source_fingerprint = require_sha256(
        p23_source_fingerprint, field=P23_SOURCE_FINGERPRINT_KEY
    )
    p2_bundle_before = require_sha256(
        record[P2_BUNDLE_FINGERPRINT_KEY], field=P2_BUNDLE_FINGERPRINT_KEY
    )
    node_shapes = np.asarray(record[AtomicDataDict.NODE_P2_BLOCK_SHAPE_KEY])
    edge_shapes = np.asarray(record[AtomicDataDict.EDGE_P2_BLOCK_SHAPE_KEY])
    node_p2 = np.asarray(record[AtomicDataDict.NODE_P2_BLOCKS_KEY])
    edge_p2 = np.asarray(record[AtomicDataDict.EDGE_P2_BLOCKS_KEY])
    node_add_abacus, edge_add_abacus, assembly_stats = assembler.assemble_graph_addition(
        symbols=symbols,
        positions_bohr=np.asarray(positions_bohr, dtype=np.float64),
        cell_bohr=np.asarray(cell_bohr, dtype=np.float64),
        edge_index=np.asarray(record[AtomicDataDict.EDGE_INDEX_KEY]),
        edge_cell_shift=np.asarray(record[AtomicDataDict.EDGE_CELL_SHIFT_KEY]),
        node_shapes=node_shapes,
        edge_shapes=edge_shapes,
        node_pad_shape=tuple(int(value) for value in node_p2.shape[-2:]),
        edge_pad_shape=tuple(int(value) for value in edge_p2.shape[-2:]),
    )
    if assembly_stats.get("schema") != P23_VNA_ASSEMBLY_SCHEMA:
        raise ValueError("P23 assembler did not return its versioned assembly schema.")
    node_add, edge_add, gauge_stats = transform_vna3c_abacus_to_deeptb(
        node_addition=node_add_abacus,
        edge_addition=edge_add_abacus,
        symbols=symbols,
        edge_index=record[AtomicDataDict.EDGE_INDEX_KEY],
        edge_cell_shift=record[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
        node_shapes=node_shapes,
        edge_shapes=edge_shapes,
        species_contract=assembler.store.species,
    )
    if node_add.shape != node_p2.shape or edge_add.shape != edge_p2.shape:
        raise ValueError("P23 addition and P2 packed AO canvases differ in shape.")
    node_p23 = np.add(node_p2, node_add, dtype=np.float32)
    edge_p23 = np.add(edge_p2, edge_add, dtype=np.float32)
    validate_non_soc_p2_block_tensors(
        node_p23,
        edge_p23,
        node_shapes,
        edge_shapes,
        num_nodes=int(node_p23.shape[0]),
        num_edges=int(edge_p23.shape[0]),
        data=data,
        idp=idp,
        prior_kind="p23",
    )
    node_rme, edge_rme = block_tensors_to_feature_tensors(
        data,
        idp,
        node_blocks=torch.as_tensor(node_p23),
        edge_blocks=torch.as_tensor(edge_p23),
    )
    if node_rme is None or edge_rme is None:
        raise RuntimeError("P23 AO-to-RME conversion returned an incomplete feature pair.")
    rebuilt = feature_tensors_to_block_tensors(
        data,
        idp,
        node_features=node_rme,
        edge_features=edge_rme,
        node_pad_shape=tuple(int(value) for value in node_p23.shape[-2:]),
        edge_pad_shape=tuple(int(value) for value in edge_p23.shape[-2:]),
        complete_edges=True,
        strict_complete_edges=True,
    )
    node_mask = block_mask_from_shapes(
        rebuilt.node_shapes, tuple(int(value) for value in node_p23.shape[-2:])
    )
    edge_mask = block_mask_from_shapes(
        rebuilt.edge_shapes, tuple(int(value) for value in edge_p23.shape[-2:])
    )
    node_roundtrip_error = _valid_max_error(
        rebuilt.node_blocks, torch.as_tensor(node_p23), node_mask
    )
    edge_roundtrip_error = _valid_max_error(
        rebuilt.edge_blocks, torch.as_tensor(edge_p23), edge_mask
    )
    max_roundtrip_error = max(node_roundtrip_error, edge_roundtrip_error)
    if max_roundtrip_error > RME_AO_ROUNDTRIP_TOLERANCE_EV:
        raise ValueError(
            "P23 RME/AO roundtrip exceeds tolerance: "
            f"{max_roundtrip_error:.3e} eV."
        )

    dual = dict(record)
    dual.update(
        {
            SAMPLE_SCHEMA_KEY: DUAL_PRIOR_SAMPLE_SCHEMA,
            AtomicDataDict.NODE_P23_KEY: node_rme.detach().cpu().to(torch.float32).numpy(),
            AtomicDataDict.EDGE_P23_KEY: edge_rme.detach().cpu().to(torch.float32).numpy(),
            AtomicDataDict.NODE_P23_BLOCKS_KEY: node_p23,
            AtomicDataDict.EDGE_P23_BLOCKS_KEY: edge_p23,
            AtomicDataDict.NODE_P23_BLOCK_SHAPE_KEY: node_shapes.copy(),
            AtomicDataDict.EDGE_P23_BLOCK_SHAPE_KEY: edge_shapes.copy(),
            P23_SOURCE_FINGERPRINT_KEY: p23_source_fingerprint,
            P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY: p2_bundle_before,
        }
    )
    assembly_provenance = {
        **dict(assembly_stats),
        "geometry": dict(geometry_provenance),
        "ao_gauge_conversion": gauge_stats,
        "p23_semantics": "P2_plus_factorized_local_third_centre_VNA",
        "rme_ao_roundtrip_tolerance_ev": RME_AO_ROUNDTRIP_TOLERANCE_EV,
        "node_rme_ao_roundtrip_error_ev": node_roundtrip_error,
        "edge_rme_ao_roundtrip_error_ev": edge_roundtrip_error,
    }
    dual["p23_assembly"] = assembly_provenance
    dual["p23_cache_source"] = {
        "kind": "factorized_local_three_centre_vna_table",
        "table_manifest_sha256": p23_source_fingerprint,
        "base_p2_bundle_fingerprint": p2_bundle_before,
        "energy_unit": "eV",
        "stored_ao_order": "DeePTB",
        "table_ao_gauge": "ABACUS",
        "no_triton": True,
    }
    dual[P23_RME_FINGERPRINT_KEY] = fingerprint_fields(dual, _P23_RME_FIELDS)
    dual[P23_BLOCK_FINGERPRINT_KEY] = fingerprint_fields(dual, _P23_BLOCK_FIELDS)
    dual[P23_BUNDLE_FINGERPRINT_KEY] = fingerprint_text_fields(
        dual,
        (
            BASIS_FINGERPRINT_KEY,
            EDGE_GRAPH_FINGERPRINT_KEY,
            P23_SOURCE_FINGERPRINT_KEY,
            P23_RME_FINGERPRINT_KEY,
            P23_BLOCK_FINGERPRINT_KEY,
            P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
        ),
    )
    if dual[P2_BUNDLE_FINGERPRINT_KEY] != p2_bundle_before:
        raise RuntimeError("P2 bundle changed while adding P23.")
    dual[ROW_ALIGNED_DATA_FINGERPRINT_KEY] = fingerprint_present_row_aligned_fields(dual)
    dual[ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY] = fingerprint_text_fields(
        dual,
        (
            BASIS_FINGERPRINT_KEY,
            EDGE_GRAPH_FINGERPRINT_KEY,
            ROW_ALIGNED_DATA_FINGERPRINT_KEY,
        ),
    )
    _assert_dual_record_contract(
        dual,
        data,
        idp,
        expected_p23_source=p23_source_fingerprint,
    )
    return dual, assembly_provenance


def _assert_dual_record_contract(
    record: Mapping[str, Any],
    data: Mapping[str, Any],
    idp: Any,
    *,
    expected_p23_source: str,
) -> None:
    if record.get(SAMPLE_SCHEMA_KEY) != DUAL_PRIOR_SAMPLE_SCHEMA:
        raise ValueError("Dual prior record lacks schema v3.")
    _require_fields(
        record,
        (
            *_P2_RME_FIELDS,
            *_P2_BLOCK_FIELDS,
            *_P23_RME_FIELDS,
            *_P23_BLOCK_FIELDS,
            *_FULL_H_TARGET_FIELDS,
            P2_BUNDLE_FINGERPRINT_KEY,
            P23_SOURCE_FINGERPRINT_KEY,
            P23_RME_FINGERPRINT_KEY,
            P23_BLOCK_FINGERPRINT_KEY,
            P23_BUNDLE_FINGERPRINT_KEY,
            P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
            ROW_ALIGNED_DATA_FINGERPRINT_KEY,
            ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
        ),
        label="dual-prior record",
    )
    assert_absolute_full_h_target_contract(dict(record))
    p23_source = require_sha256(
        record[P23_SOURCE_FINGERPRINT_KEY], field=P23_SOURCE_FINGERPRINT_KEY
    )
    if p23_source != require_sha256(
        expected_p23_source, field="expected_p23_source_fingerprint"
    ):
        raise ValueError("P23 record source differs from the selected factor table.")
    parent = require_sha256(
        record[P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY],
        field=P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
    )
    p2_bundle = require_sha256(
        record[P2_BUNDLE_FINGERPRINT_KEY], field=P2_BUNDLE_FINGERPRINT_KEY
    )
    if parent != p2_bundle:
        raise ValueError("P23 parent P2 bundle differs from this record's P2 bundle.")
    assert_record_fingerprint(
        record,
        field=P23_RME_FINGERPRINT_KEY,
        actual=fingerprint_fields(record, _P23_RME_FIELDS),
    )
    assert_record_fingerprint(
        record,
        field=P23_BLOCK_FINGERPRINT_KEY,
        actual=fingerprint_fields(record, _P23_BLOCK_FIELDS),
    )
    assert_record_fingerprint(
        record,
        field=P23_BUNDLE_FINGERPRINT_KEY,
        actual=fingerprint_text_fields(
            record,
            (
                BASIS_FINGERPRINT_KEY,
                EDGE_GRAPH_FINGERPRINT_KEY,
                P23_SOURCE_FINGERPRINT_KEY,
                P23_RME_FINGERPRINT_KEY,
                P23_BLOCK_FINGERPRINT_KEY,
                P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
            ),
        ),
    )
    assert_record_fingerprint(
        record,
        field=FULL_H_TARGET_FINGERPRINT_KEY,
        actual=fingerprint_fields(record, _FULL_H_TARGET_FIELDS),
    )
    row_data = fingerprint_present_row_aligned_fields(record)
    assert_record_fingerprint(record, field=ROW_ALIGNED_DATA_FINGERPRINT_KEY, actual=row_data)
    assert_record_fingerprint(
        record,
        field=ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
        actual=fingerprint_text_fields(
            record,
            (
                BASIS_FINGERPRINT_KEY,
                EDGE_GRAPH_FINGERPRINT_KEY,
                ROW_ALIGNED_DATA_FINGERPRINT_KEY,
            ),
        ),
    )
    if not np.array_equal(
        np.asarray(record[AtomicDataDict.NODE_P23_BLOCK_SHAPE_KEY]),
        np.asarray(record[AtomicDataDict.NODE_P2_BLOCK_SHAPE_KEY]),
    ) or not np.array_equal(
        np.asarray(record[AtomicDataDict.EDGE_P23_BLOCK_SHAPE_KEY]),
        np.asarray(record[AtomicDataDict.EDGE_P2_BLOCK_SHAPE_KEY]),
    ):
        raise ValueError("P2/P23 AO shape contracts differ in one dual record.")
    validate_non_soc_p2_block_tensors(
        record[AtomicDataDict.NODE_P23_BLOCKS_KEY],
        record[AtomicDataDict.EDGE_P23_BLOCKS_KEY],
        record[AtomicDataDict.NODE_P23_BLOCK_SHAPE_KEY],
        record[AtomicDataDict.EDGE_P23_BLOCK_SHAPE_KEY],
        num_nodes=int(np.asarray(record[AtomicDataDict.ATOMIC_NUMBERS_KEY]).size),
        num_edges=int(np.asarray(record[AtomicDataDict.EDGE_INDEX_KEY]).shape[1]),
        data=data,
        idp=idp,
        prior_kind="p23",
    )


def _build_identity(
    *,
    p2_cache_manifest: Path,
    p23_table_manifest: Path,
    p23_table_audit: Path,
    input_json: Path,
    gate1_script: Path,
    dataset_root: Path,
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": IDENTITY_SCHEMA,
        "materializer_script_sha256": _sha256(Path(__file__).resolve()),
        "p23_interface_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "dptb/data/interfaces/p23_table.py"
        ),
        "p2_contract_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "dptb/data/interfaces/p2_contract.py"
        ),
        "p2_cache_manifest_sha256": _sha256(p2_cache_manifest),
        "p23_table_manifest_sha256": _sha256(p23_table_manifest),
        "p23_table_audit_sha256": _sha256(p23_table_audit),
        "input_json_sha256": _sha256(input_json),
        "gate1_script_sha256": _sha256(gate1_script),
        "dataset_ordered_paths_sha256": _sha256(dataset_root / "ordered_paths.txt"),
        "geometry_tolerance_angstrom": GEOMETRY_TOLERANCE_ANGSTROM,
        "rme_ao_roundtrip_tolerance_ev": RME_AO_ROUNDTRIP_TOLERANCE_EV,
        "splits": {
            split: {
                "cases": list(contract["cases"]),
                "source_paths_sha256": contract["source_paths_sha256"],
                "entries": int(contract["entries"]),
                "physical_shards": [
                    {
                        "name": shard["name"],
                        "entries": int(shard["entries"]),
                        "paths_sha256": shard["paths_sha256"],
                    }
                    for shard in contract["source_shards"]
                ],
            }
            for split, contract in contracts.items()
        },
    }
    payload["identity_sha256"] = _json_sha256(payload)
    return payload


def _validate_identity(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    allow_geometry_tolerance_upgrade: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(actual, Mapping) or actual.get("schema") != IDENTITY_SCHEMA:
        raise ValueError("Cannot resume: dual-cache identity is missing or incompatible.")
    declared = require_sha256(
        actual.get("identity_sha256"), field="dual cache identity_sha256"
    )
    without_digest = dict(actual)
    without_digest.pop("identity_sha256", None)
    if declared != _json_sha256(without_digest):
        raise ValueError("Cannot resume: stored dual-cache identity is internally inconsistent.")
    if dict(actual) == dict(expected):
        return None
    if allow_geometry_tolerance_upgrade:
        old_script = actual.get("materializer_script_sha256")
        old_tolerance = actual.get("geometry_tolerance_angstrom")
        legacy_tolerance = LEGACY_GEOMETRY_IDENTITY_UPGRADES.get(old_script)
        actual_comparable = dict(actual)
        expected_comparable = dict(expected)
        actual_comparable.pop("identity_sha256", None)
        expected_comparable.pop("identity_sha256", None)
        actual_comparable["materializer_script_sha256"] = expected_comparable.get(
            "materializer_script_sha256"
        )
        actual_comparable["geometry_tolerance_angstrom"] = expected_comparable.get(
            "geometry_tolerance_angstrom"
        )
        if (
            legacy_tolerance is not None
            and float(old_tolerance) == float(legacy_tolerance)
            and float(expected["geometry_tolerance_angstrom"]) == 1.0e-4
            and actual_comparable == expected_comparable
        ):
            return {
                "kind": "geometry_serialization_tolerance_upgrade",
                "old_materializer_script_sha256": old_script,
                "new_materializer_script_sha256": expected[
                    "materializer_script_sha256"
                ],
                "old_geometry_tolerance_angstrom": float(old_tolerance),
                "new_geometry_tolerance_angstrom": float(
                    expected["geometry_tolerance_angstrom"]
                ),
            }
    changed = [
        key
        for key in sorted(set(actual) | set(expected))
        if actual.get(key) != expected.get(key)
    ]
    raise ValueError(f"Cannot resume: dual-cache identity changed for {changed}.")


def _validated_prefix(
    *,
    env: lmdb.Environment,
    rows: list[dict[str, Any]],
    cases: Sequence[str],
    expected_p23_source: str,
) -> int:
    with env.begin(write=True) as txn:
        drop_uncommitted_tail(txn, committed_rows=len(rows))
    for index, row in enumerate(rows):
        if index >= len(cases) or row.get("case_id") != cases[index]:
            raise ValueError("Cannot resume: committed case order changed.")
        key = index.to_bytes(length=4, byteorder="big")
        with env.begin() as txn:
            payload = txn.get(key)
        if payload is None or _bytes_sha256(payload) != row.get("record_sha256"):
            raise ValueError(f"Cannot resume: committed output row {index} was modified.")
        record = pickle.loads(payload)
        if record.get("case_id") != cases[index]:
            raise ValueError(f"Cannot resume: output row {index} case id changed.")
        if record.get(SAMPLE_SCHEMA_KEY) != DUAL_PRIOR_SAMPLE_SCHEMA:
            raise ValueError(f"Cannot resume: output row {index} schema changed.")
        if require_sha256(
            record.get(P23_SOURCE_FINGERPRINT_KEY), field=P23_SOURCE_FINGERPRINT_KEY
        ) != expected_p23_source:
            raise ValueError(f"Cannot resume: output row {index} P23 source changed.")
    return len(rows)


def _strict_audit_output(
    *,
    input_json: Path,
    full_h_root: Path,
    contracts: Mapping[str, Mapping[str, Any]],
    p2_source: str,
    p23_source: str,
    work_root: Path,
) -> dict[str, Any]:
    # The build/configure/close/heartbeat callables are resolved from this
    # module's globals at call time so tests may monkeypatch them.
    auditor = DatasetAuditor(
        build_context_dataset=_build_context_dataset,
        configure_strict_dataset=_configure_strict_dataset,
        close_dataset=_close_dataset,
        heartbeat=_heartbeat,
    )
    return auditor.audit(
        input_json=input_json,
        split_roots={split: full_h_root / split for split in contracts},
        prior_sources=(("p2", p2_source), ("p23", p23_source)),
        expected_entries={
            split: int(contract["entries"]) for split, contract in contracts.items()
        },
        work_root=work_root,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-cache-root", type=Path, required=True)
    parser.add_argument("--p2-cache-manifest", type=Path, required=True)
    parser.add_argument("--p23-table-root", type=Path, required=True)
    parser.add_argument("--p23-table-audit", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gate1-script", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--require-count", type=int, default=0)
    parser.add_argument("--contraction-chunk-terms", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-geometry-tolerance-upgrade", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    p2_cache_root = args.p2_cache_root.resolve()
    p2_cache_manifest = args.p2_cache_manifest.resolve()
    p23_table_root = args.p23_table_root.resolve()
    p23_table_audit = args.p23_table_audit.resolve()
    dataset_root = args.dataset_root.resolve()
    gate1_script = args.gate1_script.resolve()
    input_json = args.input_json.resolve()
    work_root = args.work_root.resolve()
    for path in (
        p2_cache_root,
        p23_table_root,
        dataset_root,
    ):
        if not path.is_dir():
            raise NotADirectoryError(path)
    for path in (p2_cache_manifest, p23_table_audit, gate1_script, input_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    _guard_roots(
        p2_cache_root,
        work_root,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )

    p2_manifest = json.loads(p2_cache_manifest.read_text(encoding="utf-8"))
    declared_input_sha = p2_manifest.get("input_json_sha256")
    if declared_input_sha not in (None, "") and _sha256(input_json) != str(
        declared_input_sha
    ):
        raise ValueError("Input JSON differs from the one that built the P2 cache.")
    p23_store = P23VNAFactorTableStore(p23_table_root)
    p2_source, p23_source = _validate_table_lineage(p2_manifest, p23_store)
    audit_payload = _validate_p23_table_audit(
        p23_table_audit,
        expected_table_manifest_sha256=p23_source,
    )
    expected_gate1 = p23_store.manifest.get("code_identity", {}).get(
        "gate1_script_sha256"
    )
    if require_sha256(expected_gate1, field="P23 table gate1_script_sha256") != _sha256(
        gate1_script
    ):
        raise ValueError("STRU parser differs from the parser frozen in the P23 table.")
    contracts = _split_contracts(
        p2_cache_root=p2_cache_root,
        p2_manifest=p2_manifest,
        dataset_root=dataset_root,
        selected_splits=args.split,
        selected_cases=args.case,
    )
    total = sum(int(contract["entries"]) for contract in contracts.values())
    if args.require_count and total != int(args.require_count):
        raise ValueError(f"Selected {total} records; required {args.require_count}.")
    identity = _build_identity(
        p2_cache_manifest=p2_cache_manifest,
        p23_table_manifest=p23_store.manifest_path,
        p23_table_audit=p23_table_audit,
        input_json=input_json,
        gate1_script=gate1_script,
        dataset_root=dataset_root,
        contracts=contracts,
    )
    manifest_store = ManifestStore(work_root)
    partial_path = manifest_store.partial_path
    if args.resume:
        if not partial_path.is_file():
            raise FileNotFoundError("Cannot resume without manifest.partial.json.")
        manifest = json.loads(partial_path.read_text(encoding="utf-8"))
        identity_migration = _validate_identity(
            manifest.get("identity"),
            identity,
            allow_geometry_tolerance_upgrade=bool(
                args.allow_geometry_tolerance_upgrade
            ),
        )
        if identity_migration is not None:
            identity_migration.update(
                {
                    "accepted_unix": time.time(),
                    "committed_rows_before_upgrade": {
                        split: len(payload.get("records", []))
                        for split, payload in manifest.get("splits", {}).items()
                    },
                }
            )
            manifest.setdefault("identity_migrations", []).append(identity_migration)
            manifest["identity"] = identity
            manifest_store.write_partial(manifest)
        if manifest.get("complete") is True:
            raise ValueError("Dual cache is already complete; refusing a duplicate run.")
    else:
        manifest = {
            "schema": SCHEMA,
            "complete": False,
            "created_unix": time.time(),
            "identity": identity,
            "p2_cache_root": str(p2_cache_root),
            "p2_cache_manifest": str(p2_cache_manifest),
            "p2_cache_manifest_sha256": _sha256(p2_cache_manifest),
            "p2_source_fingerprint": p2_source,
            "p23_table_root": str(p23_table_root),
            "p23_source_fingerprint": p23_source,
            "p23_table_audit": str(p23_table_audit),
            "p23_table_audit_sha256": _sha256(p23_table_audit),
            "p23_table_audit_schema": audit_payload["schema"],
            "p23_table_verified_shards": int(audit_payload["verified_shards"]),
            "p23_experimental_training_eligible": True,
            "dataset_root": str(dataset_root),
            "input_json": str(input_json),
            "input_json_sha256": _sha256(input_json),
            "target_semantics": "absolute_full_h",
            "p23_semantics": "P2_plus_factorized_local_third_centre_VNA",
            "no_triton": True,
            "splits": {
                split: {"cases": list(contract["cases"]), "records": []}
                for split, contract in contracts.items()
            },
        }
        manifest_store.write_partial(manifest)

    gate1 = _load_module(gate1_script, "deeptb_dual_prior_gate1")
    assembler = P23VNAFactorAssembler(
        p23_store,
        contraction_chunk_terms=int(args.contraction_chunk_terms),
    )
    full_h_root = work_root / "full_h"
    completed_global = 0
    for split, contract in contracts.items():
        source_split = Path(contract["source_split"])
        output_split = full_h_root / split
        output_lmdb = output_split / "data.0000.lmdb"
        dataset, _ = _build_context_dataset(str(input_json), str(source_split))
        _configure_strict_dataset(dataset, kind="p2", source=p2_source)
        if len(dataset) != int(contract["source_total_entries"]):
            _close_dataset(dataset)
            raise ValueError(f"Source dataset {split!r} length changed.")
        all_source_rows = list(contract["all_source_rows"])
        source_rows = list(contract["source_rows"])
        selected_source_global_indices = list(
            contract["selected_source_global_indices"]
        )
        actual_paths = [Path(path).resolve() for path in dataset._lmdb_path_map]
        actual_indices = [int(value) for value in dataset.index_map]
        if len(actual_paths) != len(all_source_rows):
            _close_dataset(dataset)
            raise ValueError(f"Source dataset {split!r} physical row map changed.")
        for global_index, ((expected_path, expected_local, _), actual_path, actual_local) in enumerate(
            zip(all_source_rows, actual_paths, actual_indices)
        ):
            if Path(expected_path).resolve() != actual_path or int(expected_local) != actual_local:
                _close_dataset(dataset)
                raise ValueError(
                    f"Source dataset {split!r} row {global_index} is bound to "
                    "a different physical shard/local index than the paths contract."
                )
        # Source rows are visited in shard order, so a small bounded LRU cache
        # keeps at most a handful of read environments (file descriptors) open
        # at once instead of one per source shard for the whole split.
        source_reader = BoundedEnvCache(opener=_open_read)
        writer = TransactionalLMDBWriter(output_lmdb)
        output_env = writer.env
        split_rows = manifest["splits"][split]["records"]
        start = _validated_prefix(
            env=output_env,
            rows=split_rows,
            cases=contract["cases"],
            expected_p23_source=p23_source,
        )
        completed_global += start
        try:
            for index in range(start, int(contract["entries"])):
                source_path, source_local_index, source_case = source_rows[index]
                source_global_index = int(selected_source_global_indices[index])
                if source_case != contract["cases"][index]:
                    raise RuntimeError("Physical source-row/case contract changed in memory.")
                source_key = int(source_local_index).to_bytes(length=4, byteorder="big")
                key = index.to_bytes(length=4, byteorder="big")
                source_env = source_reader.get(Path(source_path).resolve())
                with source_env.begin() as txn:
                    source_payload = txn.get(source_key)
                if source_payload is None:
                    raise KeyError(f"Missing P2 source row {split}[{index}].")
                source_record = pickle.loads(source_payload)
                case_id = str(contract["cases"][index])
                if source_record.get("case_id") != case_id:
                    raise ValueError(
                        f"P2 source {split}[{index}] case id differs from split order."
                    )
                data = dataset.get(source_global_index)
                _assert_source_p2_contract(
                    source_record,
                    data,
                    dataset.type_mapper,
                    expected_p2_source=p2_source,
                )
                symbols, positions_bohr, cell_bohr, geometry = _structure_geometry_bohr(
                    record=source_record,
                    case_path=Path(contract["case_paths"][index]),
                    gate1=gate1,
                    table_species=p23_store.species,
                )
                dual_record, assembly = augment_p2_record_with_p23(
                    record=source_record,
                    data=data,
                    idp=dataset.type_mapper,
                    assembler=assembler,
                    symbols=symbols,
                    positions_bohr=positions_bohr,
                    cell_bohr=cell_bohr,
                    p23_source_fingerprint=p23_source,
                    geometry_provenance=geometry,
                )
                output_payload = pickle.dumps(
                    dual_record, protocol=pickle.HIGHEST_PROTOCOL
                )
                writer.put_new(
                    key,
                    output_payload,
                    exists_message=(
                        f"Output row {split}[{index}] already exists outside "
                        "the validated committed prefix."
                    ),
                )
                row = {
                    "index": index,
                    "case_id": case_id,
                    "source_record_sha256": _bytes_sha256(source_payload),
                    "record_sha256": _bytes_sha256(output_payload),
                    "record_bytes": len(output_payload),
                    "p2_bundle_fingerprint": dual_record[P2_BUNDLE_FINGERPRINT_KEY],
                    "p23_bundle_fingerprint": dual_record[P23_BUNDLE_FINGERPRINT_KEY],
                    "p23_directed_true_third_centre_terms": int(
                        assembly.get("directed_true_third_centre_terms", 0)
                    ),
                    "p23_unique_factor_queries": int(
                        assembly.get("unique_factor_queries", 0)
                    ),
                    "p23_node_rme_ao_roundtrip_error_ev": float(
                        assembly["node_rme_ao_roundtrip_error_ev"]
                    ),
                    "p23_edge_rme_ao_roundtrip_error_ev": float(
                        assembly["edge_rme_ao_roundtrip_error_ev"]
                    ),
                    "stru_sha256": geometry["stru_sha256"],
                }
                split_rows.append(row)
                completed_global += 1
                manifest["updated_unix"] = time.time()
                manifest_store.write_partial(manifest)
                _heartbeat(
                    work_root,
                    stage="materialize",
                    split=split,
                    completed=index + 1,
                    split_total=int(contract["entries"]),
                    completed_global=completed_global,
                    total=total,
                    case_id=case_id,
                    failed=0,
                )
                print(
                    json.dumps(
                        {
                            "stage": "materialize",
                            "split": split,
                            "completed": index + 1,
                            "total": int(contract["entries"]),
                            "case_id": case_id,
                            "terms": row["p23_directed_true_third_centre_terms"],
                        }
                    ),
                    flush=True,
                )
        finally:
            source_reader.close_all()
            writer.close()
            _close_dataset(dataset)
        output_split.mkdir(parents=True, exist_ok=True)
        (output_split / "data.0000.paths.txt").write_text(
            "".join(f"{path}\n" for path in contract["case_paths"]),
            encoding="utf-8",
        )
        manifest["splits"][split]["entries"] = len(split_rows)
        manifest["splits"][split]["source_paths_sha256"] = contract[
            "source_paths_sha256"
        ]
        manifest_store.write_partial(manifest)

    audit = _strict_audit_output(
        input_json=input_json,
        full_h_root=full_h_root,
        contracts=contracts,
        p2_source=p2_source,
        p23_source=p23_source,
        work_root=work_root,
    )
    manifest["strict_loader_audit"] = audit
    manifest["full_h_root"] = str(full_h_root)
    manifest["completed_unix"] = time.time()
    manifest["updated_unix"] = manifest["completed_unix"]
    manifest["complete"] = True
    manifest_store.write_partial(manifest)
    final_path = manifest_store.final_path
    manifest_store.write_final(manifest)
    _heartbeat(
        work_root,
        stage="complete",
        completed=total,
        total=total,
        failed=0,
        manifest=str(final_path),
        manifest_sha256=_sha256(final_path),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "records": total,
                "full_h_root": str(full_h_root),
                "manifest": str(final_path),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

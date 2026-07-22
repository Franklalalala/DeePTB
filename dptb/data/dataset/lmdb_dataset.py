import numpy as np
from typing import Tuple, Dict, Any, List, Callable, Union, Optional, Mapping, Iterable

import torch
from dptb.utils.tools import download_url, extract_zip

import hashlib
import logging
import os
import os.path as osp
import glob
from dptb.data import (
    AtomicData,
    AtomicDataDict,
    register_fields,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
)
from tqdm import tqdm
from ..transforms import TypeMapper
from ._base_datasets import (
    AtomicDataset,
    _dynamic_batch_parts_from_data,
)
from dptb.nn.hamiltonian import E3Hamiltonian
import lmdb
from dptb.data.interfaces.ham_to_feature import block_to_feature
from dptb.data.interfaces.blockwise_tensor import is_soc_uureal_mapper
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    BASIS_FINGERPRINT_KEY,
    DEDICATED_PHYSICAL_H0_SOURCE,
    DUAL_PRIOR_SAMPLE_SCHEMA,
    EDGE_GRAPH_FINGERPRINT_KEY,
    FULL_H_TARGET_FINGERPRINT_KEY,
    H0_RESIDUAL_SEMANTICS,
    P2_BUNDLE_FINGERPRINT_KEY,
    P2_SAMPLE_SCHEMA,
    PHYSICAL_H0_SOURCE_FINGERPRINT_KEY,
    PHYSICAL_H0_SOURCE_KEY,
    RAW_HAMILTONIAN_SAMPLE_SCHEMA,
    RAW_PHYSICAL_H0_SOURCE,
    P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
    ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
    ROW_ALIGNED_DATA_FINGERPRINT_KEY,
    ROW_ALIGNED_FIELD_CANDIDATES,
    SAMPLE_SCHEMA_KEY,
    TARGET_SEMANTICS_KEY,
    TARGET_SOURCE_KEY,
    assert_record_fingerprint,
    canonical_edge_graph,
    edge_graph_fingerprint,
    fingerprint_fields,
    fingerprint_present_row_aligned_fields,
    fingerprint_text_fields,
    mapper_basis_fingerprint,
    physical_h0_dataset_fingerprint,
    physical_h0_record_fingerprint,
    require_sha256,
    resolve_prior_field_spec,
)
import pickle


log = logging.getLogger(__name__)

# Register the stable per-graph record identity as a graph-level long field so it
# is stacked (never concatenated) during batching and survives collation as one
# int64 value per graph. Registration is idempotent (set-based) and safe to run
# at import time. See ``LMDBDataset._compute_sample_uid`` for the packing scheme.
register_fields(
    graph_fields=[AtomicDataDict.SAMPLE_UID_KEY],
    long_fields=[AtomicDataDict.SAMPLE_UID_KEY],
)


def _shard_content_fingerprint(lmdb_path: str) -> str:
    """Hex-digest fingerprint of an LMDB shard's *content*, independent of its path.

    Feeds ``_stable_shard_ordinal`` for real on-disk shards so a record's
    ``sample_uid`` (and any SEEDED prior epsilon derived from it) is a function of
    what a shard contains, never of where it happens to be mounted.  Opens the env
    read-only and reads exactly one record, so this stays O(1) regardless of shard
    size. The digest is sha1 over three components, each closing a distinct
    collision mode:

    * the shard's basename -- disambiguates two otherwise-different datasets whose
      remaining components coincide, most importantly two *empty* shards (0
      entries): with no first record to read, the entry-count and first-record
      components below are both trivially empty for ANY empty shard, so the
      basename is what keeps two unrelated empty shards from aliasing.
    * the total entry count (``txn.stat()['entries']``) -- a whole-shard content
      summary that changes whenever records are added, removed, or the shard is
      regenerated, obtained without a full scan.
    * the first record's raw key bytes, plus the sha1 digest of its raw (still
      pickled) value bytes -- anchors the fingerprint to actual on-disk content,
      not merely size, while reading only the single first cursor entry.

    Two byte-identical copies of the same shard directory living at different
    filesystem paths (different container mount, node, restored archive, working
    directory, ...) but keeping the same directory name hash to the SAME
    fingerprint here -- that aliasing is intentional, it is what makes
    ``sample_uid`` (and validation's seeded prior epsilon) reproducible across
    where the data happens to live.
    """
    db_env = lmdb.open(
        lmdb_path, readonly=True, lock=False, readahead=False, max_readers=2048
    )
    try:
        with db_env.begin(buffers=True) as txn:
            entries = int(txn.stat()["entries"])
            first_key = b""
            first_value_digest = b""
            cursor = txn.cursor()
            if cursor.first():
                first_key = bytes(cursor.key())
                first_value_digest = hashlib.sha1(bytes(cursor.value())).digest()
    finally:
        db_env.close()

    basename = os.path.basename(os.path.normpath(lmdb_path))
    hasher = hashlib.sha1()
    for component in (
        os.fsencode(basename),
        str(entries).encode("ascii"),
        first_key,
        first_value_digest,
    ):
        # Length-prefix each component so e.g. an entry count and a first key
        # cannot shift bytes across the boundary between fields and coincide
        # with a different (basename, entries, key, value) tuple.
        hasher.update(len(component).to_bytes(8, "big"))
        hasher.update(component)
    return hasher.hexdigest()


def _stable_shard_ordinal(shard_identity: str) -> int:
    """Composition-independent 31-bit shard ordinal from a content-stable hash.

    A record's ``sample_uid`` packs this ordinal with the LMDB row key.  For real
    on-disk shards the caller passes ``_shard_content_fingerprint(realpath)`` --
    NOT the realpath itself -- so the ordinal is a function of the shard's
    content ALONE: not its on-disk path, and not its position in whatever set of
    shards a given dataset happens to load.  The same physical (byte-identical)
    shard therefore maps to the same ordinal across single- vs multi-shard
    datasets, rank-local shard subsets, concatenations, AND relocations (a
    different mount point, container, restored archive, or working directory).
    sha1 keeps it stable across runs and worker processes (unlike the per-
    process-salted builtin ``hash``). The constructor detects the astronomically
    unlikely 31-bit collision between two distinct fingerprints -- including the
    non-astronomically-unlikely case of the SAME content genuinely mounted twice
    under the dataset's shard set -- and fails closed (see ``_shard_uid_offsets``).
    The same hash also serves the lightweight ``__new__``/no-path-map fixtures,
    which pass an opaque logical identity string directly (no real shard to open
    and fingerprint).
    """
    digest = hashlib.sha1(os.fsencode(shard_identity)).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _build_shard_uid_offsets(shard_realpaths: Iterable[str]) -> Dict[str, int]:
    """Map each unique shard realpath to its content-anchored 31-bit ordinal.

    Extracted out of ``LMDBDataset.__init__`` (which calls this with the
    deduplicated realpaths of every shard the dataset loaded) so the
    collision-detection behavior is directly unit-testable without
    constructing a full dataset. Two DISTINCT realpaths whose CONTENT
    fingerprints coincide -- most notably the pathological case of the same
    physical shard mounted twice under a dataset's roots, but also the
    astronomically unlikely 31-bit truncation collision between two
    genuinely different shards -- fail closed rather than silently letting
    two shards share a per-record identity (and SEEDED prior substream).
    """
    offsets: Dict[str, int] = {}
    ordinal_owner: Dict[int, str] = {}
    for realpath in sorted(set(shard_realpaths)):
        fingerprint = _shard_content_fingerprint(realpath)
        ordinal = _stable_shard_ordinal(fingerprint)
        clash = ordinal_owner.get(ordinal)
        if clash is not None and clash != realpath:
            raise ValueError(
                "sample_uid shard-ordinal collision: "
                f"{clash!r} and {realpath!r} have the same content "
                f"fingerprint (basename + entry count + first record), both "
                f"hashing to the 31-bit shard ordinal {ordinal}, so two "
                "shards would share a per-record identity (and SEEDED prior "
                "substream). If this is the same physical shard mounted "
                "twice under this dataset's roots, remove the duplicate "
                "path; if these are genuinely different shards that "
                "coincidentally share a basename, entry count, and first "
                "record, rename one so its basename differs."
            )
        ordinal_owner[ordinal] = realpath
        offsets[realpath] = ordinal
    return offsets


def _parse_lmdb_block_key(key: Any):
    if not isinstance(key, str):
        return None
    parts = key.split("_")
    if len(parts) != 5:
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


def assert_residual_target_shrinks(
    blocks: Dict[Any, Any],
    delta_blocks: Dict[Any, Any],
    *,
    h0_key: str = "hamiltonian_0",
    min_shrink: float = 1.2,
) -> None:
    """Check the H0-quality shrink heuristic; raise when the target does not shrink.

    This is a configurable H0-quality policy gate, not a provenance proof:
    magnitude alone proves neither the provenance nor the correctness of H0.
    Some historical LMDBs store an already-residual dH in the Hamiltonian
    slot (delta-in-H-slot convention, e.g. the 0516 NexTHam crystal sets);
    enabling residual_hamiltonian there would double-subtract and inflate the
    target to H0 scale instead of shrinking it (a genuine full-H set shrinks
    ~16x on the water QHFlow2 data), which is the leading (but not the only)
    reason a target fails to shrink. Magnitudes are compared over all stored
    block entries. The caller
    (:func:`build_residual_hamiltonian_target_blocks`) decides whether a
    non-shrinking target errors, warns, or is ignored via
    ``residual_shrink_policy``; the math here is unchanged.
    """
    def _mean_abs(values) -> float:
        flattened = [
            np.abs(np.asarray(value)).astype(np.float64, copy=False).ravel()
            for value in values
        ]
        if not flattened:
            raise ValueError("Cannot validate an empty Hamiltonian block dictionary.")
        return float(np.mean(np.concatenate(flattened)))

    # Take abs before converting to float so complex/SOC blocks retain their
    # imaginary contribution in the safety check.
    h_mag = _mean_abs(blocks.values())
    d_mag = _mean_abs(delta_blocks.values())
    if not d_mag * min_shrink < h_mag:
        raise RuntimeError(
            "residual_hamiltonian=True, but subtracting H0 does not shrink the "
            f"target magnitude (mean|H|={h_mag:.4g} vs mean|H-H0|={d_mag:.4g}, "
            f"required shrink >= {min_shrink}x). This is a configurable "
            "H0-quality policy gate (residual_shrink_policy), NOT a provenance "
            "proof: the magnitude alone proves neither the provenance nor the "
            "correctness of H0. Hint: the Hamiltonian slot may already store a "
            "residual/delta target (delta-in-H-slot convention), which would "
            f"double-subtract, or '{h0_key}' may not be a valid H0 for it. Set "
            "residual_shrink_policy='warn'/'off' to accept it, or disable "
            "residual_hamiltonian for this dataset."
        )


_PREPACKED_HAMILTONIAN_TARGET_KEYS = (
    AtomicDataDict.NODE_DELTA_HAMIL_BLOCKS_KEY,
    AtomicDataDict.EDGE_DELTA_HAMIL_BLOCKS_KEY,
    AtomicDataDict.NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
)

_PREPACKED_FULL_H_TARGET_KEYS = (
    AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY,
    AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY,
    AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY,
)

_PREPACKED_H0_BLOCK_KEYS = (
    AtomicDataDict.NODE_H0_BLOCKS_KEY,
    AtomicDataDict.EDGE_H0_BLOCKS_KEY,
    AtomicDataDict.NODE_H0_BLOCK_SHAPE_KEY,
    AtomicDataDict.EDGE_H0_BLOCK_SHAPE_KEY,
)

def assert_physical_h0_authority_contract(
    data_dict: Dict[str, Any],
    *,
    schema: str,
    h0_key: str = "hamiltonian_0",
    expected_source_fingerprint: Optional[str] = None,
) -> None:
    """Require one explicit physical-H0 authority for Full-H supervision."""

    source = data_dict.get(PHYSICAL_H0_SOURCE_KEY)
    if schema == RAW_HAMILTONIAN_SAMPLE_SCHEMA:
        if source not in {None, RAW_PHYSICAL_H0_SOURCE}:
            raise ValueError(
                f"{RAW_HAMILTONIAN_SAMPLE_SCHEMA!r} fixes physical-H0 authority "
                f"to {RAW_PHYSICAL_H0_SOURCE!r}; got {source!r}."
            )
        source = RAW_PHYSICAL_H0_SOURCE
    elif source not in {RAW_PHYSICAL_H0_SOURCE, DEDICATED_PHYSICAL_H0_SOURCE}:
        raise ValueError(
            "P2/dual Full-H supervision with physical H0 requires explicit "
            f"{PHYSICAL_H0_SOURCE_KEY}={RAW_PHYSICAL_H0_SOURCE!r} or "
            f"{DEDICATED_PHYSICAL_H0_SOURCE!r}; got {source!r}."
        )

    prepacked_h0 = [key for key in _PREPACKED_H0_BLOCK_KEYS if key in data_dict]
    if source == RAW_PHYSICAL_H0_SOURCE:
        raw_h0 = data_dict.get(h0_key)
        if not isinstance(raw_h0, Mapping) or not raw_h0:
            raise ValueError(
                "Raw physical-H0 authority requires a non-empty physical-H0 "
                f"block dictionary at {h0_key!r}."
            )
        if prepacked_h0:
            raise ValueError(
                "Raw physical-H0 authority must derive H0 blocks from "
                f"{h0_key!r}; prepacked H0 block fields {prepacked_h0} are "
                "not allowed to override that authority."
            )
        return

    if schema not in {P2_SAMPLE_SCHEMA, DUAL_PRIOR_SAMPLE_SCHEMA}:
        raise ValueError(
            "Dedicated physical-H0 blocks are supported only by the P2/dual-prior "
            f"compact schemas; got {schema!r}."
        )
    if h0_key in data_dict:
        raise ValueError(
            "Dedicated physical-H0 authority must not also expose raw "
            f"{h0_key!r}; refusing ambiguous dual authority."
        )
    missing = [key for key in _PREPACKED_H0_BLOCK_KEYS if data_dict.get(key) is None]
    if missing:
        raise ValueError(
            "Dedicated physical-H0 block bundle is incomplete; "
            f"missing {missing}."
        )
    for field in (
        BASIS_FINGERPRINT_KEY,
        EDGE_GRAPH_FINGERPRINT_KEY,
        ROW_ALIGNED_DATA_FINGERPRINT_KEY,
        ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
    ):
        require_sha256(data_dict.get(field), field=field)
    actual_source = require_sha256(
        data_dict.get(PHYSICAL_H0_SOURCE_FINGERPRINT_KEY),
        field=PHYSICAL_H0_SOURCE_FINGERPRINT_KEY,
    )
    if expected_source_fingerprint in {None, ""}:
        raise ValueError(
            "Dedicated physical-H0 authority requires an externally trusted "
            "expected_physical_h0_source_fingerprint."
        )
    expected_source = require_sha256(
        expected_source_fingerprint,
        field="expected_physical_h0_source_fingerprint",
    )
    if actual_source != expected_source:
        raise ValueError(
            "Dedicated physical-H0 source fingerprint does not match the "
            "externally trusted cache/source manifest."
        )


def assert_absolute_full_h_target_contract(
    data_dict: Dict[str, Any],
    *,
    h0_key: str = "hamiltonian_0",
    require_h0: bool = False,
    expected_physical_h0_source_fingerprint: Optional[str] = None,
) -> None:
    """Require an explicit, versioned absolute-H target declaration."""

    schema = data_dict.get(SAMPLE_SCHEMA_KEY)
    allowed_schemas = {
        P2_SAMPLE_SCHEMA,
        DUAL_PRIOR_SAMPLE_SCHEMA,
        RAW_HAMILTONIAN_SAMPLE_SCHEMA,
    }
    if schema not in allowed_schemas:
        raise ValueError(
            "Full-H supervision requires explicit sample schema "
            f"in {sorted(allowed_schemas)!r}; got "
            f"{data_dict.get(SAMPLE_SCHEMA_KEY)!r}."
        )
    if data_dict.get(TARGET_SEMANTICS_KEY) != ABSOLUTE_FULL_H_SEMANTICS:
        raise ValueError(
            "Full-H supervision requires hamiltonian_target_semantics="
            f"{ABSOLUTE_FULL_H_SEMANTICS!r}; refusing to infer semantics from "
            "residual_hamiltonian=false or historical field names."
        )
    source = data_dict.get(TARGET_SOURCE_KEY)
    if source not in {"raw_hamiltonian", "dedicated_full_h_blocks"}:
        raise ValueError(
            "hamiltonian_target_source must be 'raw_hamiltonian' or "
            f"'dedicated_full_h_blocks', got {source!r}."
        )
    present = [key in data_dict for key in _PREPACKED_FULL_H_TARGET_KEYS]
    if any(present) and not all(present):
        missing = [
            key for key, is_present in zip(_PREPACKED_FULL_H_TARGET_KEYS, present)
            if not is_present
        ]
        raise ValueError(f"Dedicated Full-H target is incomplete; missing {missing}.")
    if source == "raw_hamiltonian" and data_dict.get("hamiltonian") is None:
        raise ValueError("raw_hamiltonian Full-H target source is absent from the record.")
    if source == "dedicated_full_h_blocks" and not all(present):
        raise ValueError(
            "dedicated_full_h_blocks target source requires all dedicated target fields."
        )
    if schema == RAW_HAMILTONIAN_SAMPLE_SCHEMA:
        if source != "raw_hamiltonian":
            raise ValueError(
                f"{RAW_HAMILTONIAN_SAMPLE_SCHEMA!r} requires "
                "hamiltonian_target_source='raw_hamiltonian'."
            )
        if any(present):
            raise ValueError(
                "Generic raw-H records must derive the Full-H target from the "
                "authoritative raw 'hamiltonian' dictionary; independently "
                "prepacked Full-H target fields are not allowed."
            )
        raw_hamiltonian = data_dict.get("hamiltonian")
        if not isinstance(raw_hamiltonian, Mapping) or not raw_hamiltonian:
            raise ValueError(
                "Generic raw-H Full-H supervision requires a non-empty raw "
                "'hamiltonian' block dictionary."
            )
    if require_h0:
        assert_physical_h0_authority_contract(
            data_dict,
            schema=schema,
            h0_key=h0_key,
            expected_source_fingerprint=expected_physical_h0_source_fingerprint,
        )
    # Historical target fields are never accepted as evidence of absolute H.
    legacy = [key for key in _PREPACKED_HAMILTONIAN_TARGET_KEYS if key in data_dict]
    if (
        source == "dedicated_full_h_blocks"
        or schema == RAW_HAMILTONIAN_SAMPLE_SCHEMA
    ) and legacy:
        raise ValueError(
            "Absolute Full-H records must not also expose ambiguous historical "
            f"delta-named targets {legacy}."
        )


def _assert_expected_prior_source(
    data_dict: Dict[str, Any],
    expected: Optional[str],
    *,
    prior_spec: Any,
) -> str:
    actual = require_sha256(
        data_dict.get(prior_spec.source_fingerprint_key),
        field=prior_spec.source_fingerprint_key,
    )
    if expected:
        normalized = require_sha256(expected, field="expected_p2_source_fingerprint")
        if normalized != actual:
            raise ValueError(
                f"{prior_spec.kind.upper()} source fingerprint does not match "
                "the configured table/provenance: "
                f"record={actual}, expected={normalized}."
            )
    return actual


def assert_residual_target_source_is_raw(data_dict: Dict[str, Any]) -> None:
    """Reject ambiguous prepacked targets before the loader can bypass raw H-H0."""
    prepacked = [key for key in _PREPACKED_HAMILTONIAN_TARGET_KEYS if key in data_dict]
    if prepacked:
        raise ValueError(
            "residual_hamiltonian=True is ambiguous for an LMDB record that "
            f"already contains prepacked block targets {prepacked}. Their "
            "absolute-H versus delta-H provenance cannot be inferred safely; "
            "disable residual_hamiltonian for an already-residual dataset or "
            "rebuild the block targets from raw Hamiltonian/H0 dictionaries."
        )
    prepacked_h0 = [key for key in _PREPACKED_H0_BLOCK_KEYS if key in data_dict]
    if prepacked_h0:
        raise ValueError(
            "residual_hamiltonian=True must derive physical-H0 blocks from the "
            "raw H0 dictionary used to form H-H0; prepacked H0 block fields "
            f"{prepacked_h0} are not allowed to override that authority."
        )


def assert_residual_h_target_contract(
    data_dict: Dict[str, Any],
    *,
    h0_key: str = "hamiltonian_0",
) -> None:
    """Require a versioned raw-H/raw-H0 declaration for residual supervision."""

    schema = data_dict.get(SAMPLE_SCHEMA_KEY)
    if schema != RAW_HAMILTONIAN_SAMPLE_SCHEMA:
        raise ValueError(
            "Residual H-H0 supervision requires explicit sample schema "
            f"{RAW_HAMILTONIAN_SAMPLE_SCHEMA!r}; got {schema!r}."
        )
    semantics = data_dict.get(TARGET_SEMANTICS_KEY)
    if semantics != H0_RESIDUAL_SEMANTICS:
        raise ValueError(
            "Residual H-H0 supervision requires hamiltonian_target_semantics="
            f"{H0_RESIDUAL_SEMANTICS!r}; got {semantics!r}."
        )
    source = data_dict.get(TARGET_SOURCE_KEY)
    if source != "raw_hamiltonian":
        raise ValueError(
            "Residual H-H0 supervision requires "
            "hamiltonian_target_source='raw_hamiltonian'; "
            f"got {source!r}."
        )

    raw_hamiltonian = data_dict.get("hamiltonian")
    if not isinstance(raw_hamiltonian, Mapping) or not raw_hamiltonian:
        raise ValueError(
            "Residual H-H0 supervision requires a non-empty raw "
            "'hamiltonian' block dictionary."
        )
    raw_h0 = data_dict.get(h0_key)
    if not isinstance(raw_h0, Mapping) or not raw_h0:
        raise ValueError(
            "Residual H-H0 supervision requires a non-empty physical-H0 "
            f"block dictionary at {h0_key!r}."
        )

    assert_residual_target_source_is_raw(data_dict)
    prepacked_full_h = [
        key for key in _PREPACKED_FULL_H_TARGET_KEYS if key in data_dict
    ]
    if prepacked_full_h:
        raise ValueError(
            "Generic raw-H residual supervision must derive H-H0 from the raw "
            "Hamiltonian dictionaries; independently prepacked Full-H target "
            f"fields {prepacked_full_h} are not allowed."
        )


def assert_residual_from_full_h_target_contract(
    data_dict: Dict[str, Any],
    *,
    h0_key: str = "hamiltonian_0",
    expected_physical_h0_source_fingerprint: Optional[str] = None,
) -> None:
    """Require an absolute-Full-H raw record whose residual dH is materialized online.

    ``residual_ao_block_ode`` reuses the SAME absolute_full_h raw LMDB records as
    the ao_block_ode Full-H arm (schema deeptb.raw_hamiltonian_training_sample/v1,
    target semantics absolute_full_h, source raw_hamiltonian), but with
    residual_hamiltonian=True the loader materializes the residual endpoints
    D1 = raw H - H0 on the fly.  This is a distinct contract from
    :func:`assert_residual_h_target_contract`, which demands the h0_residual
    semantics and would therefore reject these absolute_full_h records; the new
    opt-in gate certifies the absolute-Full-H declaration instead while still
    forbidding any prepacked/ambiguous target.

    Also runs :func:`assert_physical_h0_authority_contract` -- the SAME
    physical_h0_source/physical_h0_source_fingerprint authority check the
    ordinary Full-H route (``assert_absolute_full_h_target_contract`` with
    ``require_h0=True``) runs on this schema.  Without it, a record that
    declares ``physical_h0_source=dedicated_h0_blocks`` (with a missing or
    wrong fingerprint) would be rejected on the Full-H route -- raw-schema
    records fix physical-H0 authority to ``raw_hamiltonian_0`` -- but silently
    accepted here, subtracting the raw ``hamiltonian_0`` dict anyway: the same
    record's provenance would be self-contradictory depending only on which
    route loaded it. ``h0_key`` is already validated to be a non-empty raw
    dict above, and ``schema`` to be ``RAW_HAMILTONIAN_SAMPLE_SCHEMA``, by the
    time this runs, so it exercises exactly the ``source in {None,
    RAW_PHYSICAL_H0_SOURCE}`` branch for a genuine raw record.
    """

    # Converter/compact provenance markers must be rejected HERE, at the loader,
    # BEFORE the residual H-H0 subtraction is materialized.  A record can declare
    # a valid absolute_full_h target AND carry these uu_real/compact converter
    # fields at the same time; the flow-side masquerade guard never sees them
    # because the loader only forwards these metadata keys when
    # require_uureal_block_ode=True (a require_residual_from_full_h_target load
    # drops them), so an unguarded masquerade would get H0 double-subtracted into
    # a wrong delta target.  Certifying an already-converted product as a raw
    # record therefore has to fail closed at this loader gate.
    converter_markers = [
        key
        for key in (
            "blockwise_spatial_schema",
            "blockwise_target_mode",
            "soc_uureal_compact",
            "soc_uureal_full_rme",
            "soc_uureal_keep",
            "blockwise_source_target_feature_width",
            "blockwise_source_h0_feature_width",
        )
        if key in data_dict
    ]
    if converter_markers:
        raise ValueError(
            "residual-from-Full-H supervision consumes raw absolute_full_h "
            "records; a converter product masquerading as a raw record cannot "
            "feed residual_ao_block_ode. Forbidden converter/compact provenance "
            f"markers found: {converter_markers}."
        )

    schema = data_dict.get(SAMPLE_SCHEMA_KEY)
    if schema != RAW_HAMILTONIAN_SAMPLE_SCHEMA:
        raise ValueError(
            "residual-from-Full-H supervision requires explicit sample schema "
            f"{RAW_HAMILTONIAN_SAMPLE_SCHEMA!r}; got {schema!r}."
        )
    semantics = data_dict.get(TARGET_SEMANTICS_KEY)
    if semantics != ABSOLUTE_FULL_H_SEMANTICS:
        raise ValueError(
            "residual-from-Full-H supervision requires hamiltonian_target_semantics="
            f"{ABSOLUTE_FULL_H_SEMANTICS!r}; got {semantics!r}."
        )
    source = data_dict.get(TARGET_SOURCE_KEY)
    if source != "raw_hamiltonian":
        raise ValueError(
            "residual-from-Full-H supervision requires "
            "hamiltonian_target_source='raw_hamiltonian'; "
            f"got {source!r}."
        )

    raw_hamiltonian = data_dict.get("hamiltonian")
    if not isinstance(raw_hamiltonian, Mapping) or not raw_hamiltonian:
        raise ValueError(
            "residual-from-Full-H supervision requires a non-empty raw "
            "'hamiltonian' block dictionary."
        )
    raw_h0 = data_dict.get(h0_key)
    if not isinstance(raw_h0, Mapping) or not raw_h0:
        raise ValueError(
            "residual-from-Full-H supervision requires a non-empty physical-H0 "
            f"block dictionary at {h0_key!r}."
        )

    # Same physical-H0 authority/fingerprint contract the ordinary Full-H route
    # enforces (see docstring): fails closed on a raw-schema record that
    # contradicts its fixed raw_hamiltonian_0 authority (e.g. declares
    # physical_h0_source=dedicated_h0_blocks) instead of silently subtracting
    # raw_hamiltonian_0 under a mismatched provenance claim.
    assert_physical_h0_authority_contract(
        data_dict,
        schema=schema,
        h0_key=h0_key,
        expected_source_fingerprint=expected_physical_h0_source_fingerprint,
    )

    assert_residual_target_source_is_raw(data_dict)
    prepacked_full_h = [
        key for key in _PREPACKED_FULL_H_TARGET_KEYS if key in data_dict
    ]
    if prepacked_full_h:
        raise ValueError(
            "residual-from-Full-H supervision must derive H-H0 from the raw "
            "Hamiltonian dictionaries; independently prepacked Full-H target "
            f"fields {prepacked_full_h} are not allowed."
        )


def build_residual_hamiltonian_target_blocks(
    data_dict: Dict[str, Any],
    blocks: Dict[Any, Any],
    *,
    h0_key: str = "hamiltonian_0",
    shrink_policy: str = "error",
    min_shrink: float = 1.2,
) -> Dict[Any, np.ndarray]:
    """Build H-H0 targets with fail-closed source and shape validation.

    ``shrink_policy`` splits the H0-quality shrink heuristic from the frozen
    provenance/shape contracts (which always apply): ``"error"`` (default,
    byte-identical to the historical behaviour) raises when the target fails
    to shrink, ``"warn"`` logs the same diagnostics and proceeds, and
    ``"off"`` skips the heuristic entirely. ``min_shrink`` is the required
    ratio forwarded to :func:`assert_residual_target_shrinks`.
    """
    policy = str(shrink_policy).lower()
    if policy not in {"error", "warn", "off"}:
        raise ValueError(
            "residual_shrink_policy must be 'error', 'warn', or 'off'; "
            f"got {shrink_policy!r}."
        )
    assert_residual_target_source_is_raw(data_dict)

    h0_blocks = data_dict.get(h0_key, None)
    if h0_blocks is None:
        raise ValueError(
            f"residual_hamiltonian=True requires '{h0_key}' blocks in the LMDB record."
        )
    missing = [key for key in blocks if key not in h0_blocks]
    if missing:
        raise ValueError(
            f"residual_hamiltonian=True: '{h0_key}' is missing {len(missing)} "
            f"block key(s) present in the Hamiltonian (first: {missing[:3]}); "
            "cannot form H-H0."
        )

    delta_blocks: Dict[Any, np.ndarray] = {}
    for key, value in blocks.items():
        h_value = np.asarray(value)
        h0_value = np.asarray(h0_blocks[key])
        if h_value.shape != h0_value.shape:
            raise ValueError(
                f"residual_hamiltonian=True: block {key!r} has mismatched "
                f"Hamiltonian/H0 shapes {h_value.shape} vs {h0_value.shape}."
            )
        delta_blocks[key] = h_value - h0_value

    # H0-quality shrink heuristic. Residual datasets are opt-in; the default
    # "error" policy keeps the historical fail-closed behaviour, while "warn"/
    # "off" let a reviewer accept a legitimately non-shrinking H0 without
    # editing this gate. The frozen source/shape contracts above always apply.
    if policy != "off":
        try:
            assert_residual_target_shrinks(
                blocks, delta_blocks, h0_key=h0_key, min_shrink=min_shrink
            )
        except RuntimeError as shrink_error:
            if policy != "warn":
                raise
            log.warning(
                "residual_shrink_policy='warn': proceeding despite the "
                "H0-quality shrink gate. %s",
                shrink_error,
            )
    return delta_blocks


def _count_offsite_lmdb_blocks(blocks: Any) -> int:
    if not isinstance(blocks, dict):
        return 0

    count = 0
    for key in blocks.keys():
        parsed = _parse_lmdb_block_key(key)
        if parsed is None:
            continue
        i, j, rx, ry, rz = parsed
        if i == j and rx == 0 and ry == 0 and rz == 0:
            continue
        count += 1
    return count


def _read_lmdb_entry(path: str, index: int):
    db_env = lmdb.open(
        path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=2048,
    )
    try:
        with db_env.begin(buffers=True) as txn:
            data = txn.get(int(index).to_bytes(length=4, byteorder='big'))
            if data is None:
                raise IndexError(f"LMDB entry {index} not found in {path}")
            return pickle.loads(bytes(data))
    finally:
        db_env.close()


def _lmdb_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _lmdb_scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    array = np.asarray(value)
    if array.shape == ():
        return bool(array.item())
    return bool(value)


def validate_non_soc_p2_blocks(blocks: Any, *, key: str = "hamiltonian_p2") -> None:
    """Fail closed on malformed or SOC physical-prior AO dictionaries."""
    prior_label = "P23" if str(key) == "hamiltonian_p23" else "P2"
    if not isinstance(blocks, dict) or not blocks:
        raise ValueError(f"'{key}' must be a non-empty AO-block dictionary.")
    for block_key, value in blocks.items():
        if _parse_lmdb_block_key(block_key) is None:
            raise ValueError(
                f"'{key}' contains invalid AO-block key {block_key!r}; expected i_j_rx_ry_rz."
            )
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(
                f"'{key}' block {block_key!r} must be rank-2, got shape {array.shape}."
            )
        if np.iscomplexobj(array):
            raise NotImplementedError(
                f"The {prior_label} prior path is non-SOC only; complex block "
                f"{block_key!r} is not accepted."
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(
                f"'{key}' block {block_key!r} must be floating point, got {array.dtype}."
            )
        if 0 in array.shape:
            raise ValueError(f"'{key}' block {block_key!r} must be non-empty.")
        if not np.isfinite(array).all():
            raise ValueError(f"'{key}' block {block_key!r} contains NaN or infinity.")


def validate_p2_feature_pair(
    node_p2: Any,
    edge_p2: Any,
    *,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    feature_dim: Optional[int] = None,
    prior_kind: str = "p2",
    check_finite: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate a first-class real-RME physical-prior feature pair.

    The historical function name remains public for P2 callers.  ``prior_kind``
    selects only the error label; field selection is handled by the loader's
    :class:`PriorFieldSpec`.
    """
    prior_label = str(prior_kind).strip().upper()
    node_field = f"node_{str(prior_kind).strip().lower()}"
    edge_field = f"edge_{str(prior_kind).strip().lower()}"
    present = (node_p2 is not None, edge_p2 is not None)
    if any(present) and not all(present):
        raise ValueError(
            f"{prior_label} prior must provide both {node_field} and {edge_field}; "
            "refusing a partial prior."
        )
    if not all(present):
        raise ValueError(f"{prior_label} prior features are absent.")
    node = torch.as_tensor(node_p2)
    edge = torch.as_tensor(edge_p2)
    if node.ndim != 2 or edge.ndim != 2:
        raise ValueError(
            f"{prior_label} node/edge fields must be rank-2 real-RME tensors; got "
            f"{tuple(node.shape)} and {tuple(edge.shape)}."
        )
    if torch.is_complex(node) or torch.is_complex(edge):
        raise NotImplementedError(
            f"The {prior_label} prior path is non-SOC only; complex RME "
            "features are not accepted."
        )
    if not torch.is_floating_point(node) or not torch.is_floating_point(edge):
        raise TypeError(
            f"{prior_label} node/edge fields must be floating-point real-RME tensors; got "
            f"{node.dtype} and {edge.dtype}."
        )
    if check_finite and (
        not torch.isfinite(node).all() or not torch.isfinite(edge).all()
    ):
        raise ValueError(f"{prior_label} node/edge fields contain NaN or infinity.")
    if num_nodes is not None and node.shape[0] != int(num_nodes):
        raise ValueError(
            f"{prior_label} node rows {node.shape[0]} do not match num_nodes={int(num_nodes)}."
        )
    if num_edges is not None and edge.shape[0] != int(num_edges):
        raise ValueError(
            f"{prior_label} edge rows {edge.shape[0]} do not match num_edges={int(num_edges)}."
        )
    if feature_dim is not None and (
        node.shape[1] != int(feature_dim) or edge.shape[1] != int(feature_dim)
    ):
        raise ValueError(
            f"{prior_label} node/edge widths must match the mapper RME width "
            f"{int(feature_dim)}; got {node.shape[1]} and {edge.shape[1]}."
        )
    return node, edge


def validate_non_soc_p2_block_tensors(
    node_blocks: Any,
    edge_blocks: Any,
    node_shapes: Any,
    edge_shapes: Any,
    *,
    num_nodes: int,
    num_edges: int,
    data: Optional[Any] = None,
    idp: Optional[Any] = None,
    prior_kind: str = "p2",
    expensive_checks: bool = True,
) -> None:
    """Validate packed physical-prior AO canvases and valid extents."""
    prior_label = str(prior_kind).strip().upper()
    entries = (
        ("node", torch.as_tensor(node_blocks), torch.as_tensor(node_shapes), num_nodes),
        ("edge", torch.as_tensor(edge_blocks), torch.as_tensor(edge_shapes), num_edges),
    )
    for label, blocks, shapes, rows in entries:
        if blocks.ndim != 3:
            raise ValueError(
                f"{label}_{prior_kind}_blocks must be rank-3 [rows, ni, nj], "
                f"got {tuple(blocks.shape)}."
            )
        if blocks.shape[0] != int(rows):
            raise ValueError(
                f"{label}_{prior_kind}_blocks rows {blocks.shape[0]} do not match "
                f"{label} count {int(rows)}."
            )
        if torch.is_complex(blocks):
            raise NotImplementedError(
                f"The {prior_label} Full-H reconstruction path is non-SOC only."
            )
        if not torch.is_floating_point(blocks):
            raise TypeError(
                f"{label}_{prior_kind}_blocks must be floating point, got {blocks.dtype}."
            )
        if expensive_checks and not torch.isfinite(blocks).all():
            raise ValueError(f"{label}_{prior_kind}_blocks contain NaN or infinity.")
        if shapes.ndim != 2 or tuple(shapes.shape) != (int(rows), 2):
            raise ValueError(
                f"{label}_{prior_kind}_block_shape must have shape ({int(rows)}, 2), "
                f"got {tuple(shapes.shape)}."
            )
        if torch.is_complex(shapes) or not torch.isfinite(shapes).all():
            raise ValueError(
                f"{label}_{prior_kind}_block_shape must contain finite real extents."
            )
        long_shapes = shapes.to(dtype=torch.long)
        if not torch.equal(shapes, long_shapes.to(dtype=shapes.dtype)):
            raise ValueError(f"{label}_{prior_kind}_block_shape extents must be integers.")
        if (long_shapes < 0).any():
            raise ValueError(
                f"{label}_{prior_kind}_block_shape extents must be non-negative."
            )
        limits = torch.tensor(
            blocks.shape[-2:], dtype=torch.long, device=long_shapes.device
        )
        if (long_shapes > limits).any():
            raise ValueError(
                f"{label}_{prior_kind}_block_shape exceeds packed canvas "
                f"{tuple(blocks.shape[-2:])}."
            )
    if (data is None) != (idp is None):
        raise ValueError(f"Strict {prior_label} validation requires data and idp together.")
    if data is not None and expensive_checks:
        from dptb.data.interfaces.blockwise_tensor import validate_packed_non_soc_blocks

        validate_packed_non_soc_blocks(
            data,
            idp,
            node_blocks,
            edge_blocks,
            node_shapes,
            edge_shapes,
            label=prior_label,
            require_symmetric_edges=True,
        )


def _lmdb_scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    array = np.asarray(value)
    if array.shape == ():
        return int(array.item())
    return int(value)


def _soc_uureal_keep_mask(
    data_dict: Dict[str, Any],
    full_rme: int,
    keep_mask: Optional[Any] = None,
) -> torch.Tensor:
    if keep_mask is not None:
        mask = torch.as_tensor(keep_mask, dtype=torch.bool).flatten()
    else:
        keep = data_dict.get("soc_uureal_keep", None)
        if keep is None:
            raise ValueError(
                "Compact SOC uu_real LMDB entry is missing soc_uureal_keep metadata."
            )
        keep_tensor = torch.as_tensor(keep)
        if keep_tensor.ndim == 0:
            keep_count = int(keep_tensor.item())
            if keep_count == full_rme:
                mask = torch.ones(full_rme, dtype=torch.bool)
            else:
                raise ValueError(
                    "Compact SOC uu_real LMDB entry stores only a keep count. "
                    "Enable nextham_uureal_mask so the dataset can use the "
                    "type_mapper.mask_uureal layout mask for expansion."
                )
        elif keep_tensor.dtype == torch.bool:
            mask = keep_tensor.flatten()
        else:
            keep_flat = keep_tensor.flatten().to(dtype=torch.long)
            if keep_flat.numel() == full_rme and (
                keep_flat.numel() == 0 or int(keep_flat.max().item()) <= 1
            ):
                mask = keep_flat.to(dtype=torch.bool)
            else:
                mask = torch.zeros(full_rme, dtype=torch.bool)
                mask[keep_flat] = True

    if mask.numel() != full_rme:
        raise ValueError(
            "Compact SOC uu_real mask width does not match full RME: "
            f"mask={mask.numel()}, full_rme={full_rme}."
        )
    return mask


def _expand_soc_uureal_compact(
    value: Any,
    data_dict: Dict[str, Any],
    field_name: str,
    keep_mask: Optional[Any] = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not _lmdb_scalar_bool(data_dict.get("soc_uureal_compact", False)):
        return tensor

    full_rme_value = data_dict.get("soc_uureal_full_rme", None)
    if full_rme_value is None:
        raise ValueError(
            f"Compact SOC uu_real LMDB field {field_name} is missing "
            "soc_uureal_full_rme metadata."
        )
    full_rme = _lmdb_scalar_int(full_rme_value)
    if full_rme <= 0:
        raise ValueError(
            f"Compact SOC uu_real LMDB field {field_name} has invalid "
            f"soc_uureal_full_rme={full_rme}."
        )

    if keep_mask is not None:
        target_mask = torch.as_tensor(keep_mask, dtype=torch.bool).flatten()
        target_rme = int(target_mask.numel())
        if target_rme != full_rme and bool(target_mask.all().item()):
            if tensor.shape[-1] == target_rme:
                return tensor
            raise ValueError(
                f"Compact SOC uu_real LMDB field {field_name} has width "
                f"{tensor.shape[-1]}; reduced uu_real target expects compact "
                f"width {target_rme}."
            )

    if tensor.shape[-1] == full_rme:
        return tensor

    mask = _soc_uureal_keep_mask(data_dict, full_rme, keep_mask=keep_mask)
    compact_rme = int(mask.sum().item())
    if tensor.shape[-1] != compact_rme:
        raise ValueError(
            f"Compact SOC uu_real LMDB field {field_name} has incompatible "
            f"width {tensor.shape[-1]}; expected compact width {compact_rme} "
            f"or full width {full_rme}."
        )

    expanded = torch.zeros(
        (*tensor.shape[:-1], full_rme),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    expanded[..., mask.to(device=tensor.device)] = tensor
    return expanded


_ATOMICDATA_CONSTRUCTOR_OPTIONS = {"r_max", "er_max", "oer_max", "self_interaction"}


def _reconcile_prior_alias(new_value, old_value, *, default, new_name, old_name):
    """Merge a renamed prior_* constructor kwarg with its deprecated P2 alias.

    ``None`` means "not supplied".  If both the canonical and the deprecated
    name are supplied they must agree; otherwise whichever was supplied wins,
    falling back to ``default``.
    """

    if new_value is not None and old_value is not None:
        if new_value != old_value:
            raise ValueError(
                f"LMDBDataset received conflicting {new_name}={new_value!r} and "
                f"deprecated {old_name}={old_value!r}; the deprecated alias must "
                "equal the canonical value."
            )
        return new_value
    if new_value is not None:
        return new_value
    if old_value is not None:
        return old_value
    return default


class LMDBDataset(AtomicDataset):
    prefer_loaded_dynamic_batch_cost_parts = True
    # Class defaults keep lightweight ``__new__``-based tooling and historical
    # tests backward compatible; normal construction always sets instances.
    # The public surface was renamed to the prior_*/get_prior family so that a
    # single ``prior_kind`` derives every field name (see PriorSpec).
    get_prior = False
    prior_kind = "p2"
    prior_raw_key = "hamiltonian_p2"
    prefer_precomputed_prior = True
    require_full_h_target = False
    require_residual_h_target = False
    require_uureal_block_ode = False
    require_residual_from_full_h_target = False
    expected_prior_source_fingerprint = None
    expected_physical_h0_source_fingerprint = None
    audit_prior_representations = False
    require_prior_blocks = False
    residual_shrink_policy = "error"
    min_residual_shrink = 1.2

    # --- Deprecated P2-named public attribute aliases -------------------------
    # Read/write properties forward the historical attribute names to the
    # prior_* storage so old callers (and tools that mutate the dataset in
    # place, e.g. cache materializers) keep working after the rename.
    @property
    def get_P2(self):
        return self.get_prior

    @get_P2.setter
    def get_P2(self, value):
        self.get_prior = value

    @property
    def p2_key(self):
        return self.prior_raw_key

    @p2_key.setter
    def p2_key(self, value):
        self.prior_raw_key = value

    @property
    def prefer_precomputed_p2(self):
        return self.prefer_precomputed_prior

    @prefer_precomputed_p2.setter
    def prefer_precomputed_p2(self, value):
        self.prefer_precomputed_prior = value

    @property
    def expected_p2_source_fingerprint(self):
        return self.expected_prior_source_fingerprint

    @expected_p2_source_fingerprint.setter
    def expected_p2_source_fingerprint(self, value):
        self.expected_prior_source_fingerprint = value

    @property
    def audit_p2_representations(self):
        return self.audit_prior_representations

    @audit_p2_representations.setter
    def audit_p2_representations(self, value):
        self.audit_prior_representations = value

    @property
    def require_p2_blocks(self):
        return self.require_prior_blocks

    @require_p2_blocks.setter
    def require_p2_blocks(self, value):
        self.require_prior_blocks = value

    def __init__(
            self,
            root: str,
            info_files: dict,
            url: Optional[str] = None,
            include_frames: Optional[List[int]] = None,
            type_mapper: TypeMapper = None,
            orthogonal: bool = False,
            get_Hamiltonian: bool = False,
            get_H0: bool = False,
            get_prior: Optional[bool] = None,
            residual_hamiltonian: bool = False,
            residual_shrink_policy: str = "error",
            min_residual_shrink: float = 1.2,
            get_overlap: bool = False,
            get_DM: bool = False,
            get_eigenvalues: bool = False,
            h0_key: str = "hamiltonian_0",
            prefer_precomputed_h0: bool = True,
            prior_kind: str = "p2",
            prior_raw_key: Optional[str] = None,
            prefer_precomputed_prior: Optional[bool] = None,
            require_full_h_target: bool = False,
            require_residual_h_target: bool = False,
            require_uureal_block_ode: bool = False,
            require_residual_from_full_h_target: bool = False,
            expected_prior_source_fingerprint: Optional[str] = None,
            expected_physical_h0_source_fingerprint: Optional[str] = None,
            audit_prior_representations: Optional[bool] = None,
            require_prior_blocks: Optional[bool] = None,
            *,
            get_P2: Optional[bool] = None,
            p2_key: Optional[str] = None,
            prefer_precomputed_p2: Optional[bool] = None,
            expected_p2_source_fingerprint: Optional[str] = None,
            audit_p2_representations: Optional[bool] = None,
            require_p2_blocks: Optional[bool] = None,
    ):
        # Deprecated P2-named kwargs are accepted as aliases of the prior_*
        # family; a supplied alias must equal the canonical value.
        get_prior = _reconcile_prior_alias(
            get_prior, get_P2, default=False,
            new_name="get_prior", old_name="get_P2",
        )
        prior_raw_key = _reconcile_prior_alias(
            prior_raw_key, p2_key, default=None,
            new_name="prior_raw_key", old_name="p2_key",
        )
        prefer_precomputed_prior = _reconcile_prior_alias(
            prefer_precomputed_prior, prefer_precomputed_p2, default=True,
            new_name="prefer_precomputed_prior", old_name="prefer_precomputed_p2",
        )
        expected_prior_source_fingerprint = _reconcile_prior_alias(
            expected_prior_source_fingerprint, expected_p2_source_fingerprint,
            default=None,
            new_name="expected_prior_source_fingerprint",
            old_name="expected_p2_source_fingerprint",
        )
        audit_prior_representations = _reconcile_prior_alias(
            audit_prior_representations, audit_p2_representations, default=False,
            new_name="audit_prior_representations",
            old_name="audit_p2_representations",
        )
        require_prior_blocks = _reconcile_prior_alias(
            require_prior_blocks, require_p2_blocks, default=False,
            new_name="require_prior_blocks", old_name="require_p2_blocks",
        )
        # TO DO, this may be simplified
        # See if a subclass defines some inputs
        self.url = getattr(type(self), "URL", url)
        self.include_frames = include_frames
        self.info_files = info_files  # there should be one info file for one LMDB Dataset
        # print(self.info_files)

        self.data = None
        # !!! don't delete this block.
        # otherwise the inherent children class
        # will ignore the download function here
        class_type = type(self)
        if class_type != LMDBDataset:
            if "download" not in self.__class__.__dict__:
                class_type.download = LMDBDataset.download

        # Initialize the InMemoryDataset, which runs download and process
        # See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets
        # Then pre-process the data if disk files are not found
        super().__init__(root=root, type_mapper=type_mapper)  # the type_mapper will be called in getitem in PyG data class
        self.get_Hamiltonian = get_Hamiltonian
        self.get_H0 = get_H0
        self.get_prior = bool(get_prior)
        if self.get_prior and bool(getattr(type_mapper, "has_soc", False)):
            raise NotImplementedError(
                "The first-class P2/P23 physical-prior route is non-SOC only."
            )
        self.residual_hamiltonian = residual_hamiltonian
        if self.residual_hamiltonian and not self.get_Hamiltonian:
            raise ValueError(
                "residual_hamiltonian=True requires get_Hamiltonian=True; "
                "otherwise the target switch would be a silent no-op."
            )
        # H0-quality shrink heuristic policy (see
        # build_residual_hamiltonian_target_blocks). The default "error" keeps the
        # historical fail-closed behaviour byte-for-byte; magnitude alone proves
        # neither provenance nor correctness, so a reviewer may relax it per split.
        self.residual_shrink_policy = str(residual_shrink_policy).lower()
        if self.residual_shrink_policy not in {"error", "warn", "off"}:
            raise ValueError(
                "residual_shrink_policy must be 'error', 'warn', or 'off'; "
                f"got {residual_shrink_policy!r}."
            )
        self.min_residual_shrink = float(min_residual_shrink)
        if not np.isfinite(self.min_residual_shrink) or self.min_residual_shrink <= 0.0:
            raise ValueError(
                "min_residual_shrink must be a finite positive float; "
                f"got {min_residual_shrink!r}."
            )
        self.get_overlap = get_overlap
        self.get_DM = get_DM
        self.get_eigenvalues = get_eigenvalues
        self.orthogonal = orthogonal
        self.h0_key = h0_key
        self.prefer_precomputed_h0 = prefer_precomputed_h0
        self.prior_spec = resolve_prior_field_spec(prior_kind)
        self.prior_kind = self.prior_spec.kind
        # The raw LMDB prior key is DERIVED from prior_kind; an explicit
        # prior_raw_key/p2_key is accepted only as a deprecated echo that must
        # match the derived value, so a single prior_kind selects everything.
        if prior_raw_key in (None, ""):
            self.prior_raw_key = self.prior_spec.raw_key
        else:
            self.prior_raw_key = str(prior_raw_key)
        if self.get_prior and self.prior_raw_key != self.prior_spec.raw_key:
            raise ValueError(
                f"prior_kind={self.prior_kind!r} requires the raw prior key "
                f"{self.prior_spec.raw_key!r}; got {self.prior_raw_key!r}. "
                "Refusing to select RME fields from one prior and raw fallback "
                "blocks from another."
            )
        self.prefer_precomputed_prior = bool(prefer_precomputed_prior)
        self.require_full_h_target = bool(require_full_h_target)
        self.require_residual_h_target = bool(require_residual_h_target)
        self.require_uureal_block_ode = bool(require_uureal_block_ode)
        self.require_residual_from_full_h_target = bool(
            require_residual_from_full_h_target
        )
        self.expected_prior_source_fingerprint = (
            str(expected_prior_source_fingerprint)
            if expected_prior_source_fingerprint not in {None, ""}
            else None
        )
        self.expected_physical_h0_source_fingerprint = (
            str(expected_physical_h0_source_fingerprint)
            if expected_physical_h0_source_fingerprint not in {None, ""}
            else None
        )
        self.audit_prior_representations = bool(audit_prior_representations)
        self.require_prior_blocks = bool(require_prior_blocks)
        if self.require_full_h_target and not self.get_Hamiltonian:
            raise ValueError("require_full_h_target=True requires get_Hamiltonian=True.")
        if self.require_residual_h_target:
            if not self.get_Hamiltonian or not self.get_H0:
                raise ValueError(
                    "require_residual_h_target=True requires get_Hamiltonian=True "
                    "and get_H0=True."
                )
            if not self.residual_hamiltonian:
                raise ValueError(
                    "require_residual_h_target=True requires "
                    "residual_hamiltonian=True."
                )
        if self.require_full_h_target and self.require_residual_h_target:
            raise ValueError(
                "Full-H and residual-H target contracts are mutually exclusive."
            )
        if self.require_uureal_block_ode:
            if not self.get_Hamiltonian or not self.get_H0:
                raise ValueError(
                    "require_uureal_block_ode=True requires get_Hamiltonian/get_H0=true."
                )
            if self.residual_hamiltonian:
                raise ValueError(
                    "Compact uu_real records are already-delta; "
                    "residual_hamiltonian must stay false."
                )
            if not is_soc_uureal_mapper(type_mapper):
                raise ValueError(
                    "require_uureal_block_ode=True requires an explicit SOC "
                    "uu_real mapper."
                )
        if self.require_residual_from_full_h_target:
            if not self.get_Hamiltonian or not self.get_H0:
                raise ValueError(
                    "require_residual_from_full_h_target=True requires "
                    "get_Hamiltonian=True and get_H0=True."
                )
            if not self.residual_hamiltonian:
                raise ValueError(
                    "require_residual_from_full_h_target=True requires "
                    "residual_hamiltonian=True (the loader materializes D1 = H - H0)."
                )
            if (
                self.require_full_h_target
                or self.require_residual_h_target
                or self.require_uureal_block_ode
            ):
                raise ValueError(
                    "require_residual_from_full_h_target is mutually exclusive with "
                    "require_full_h_target, require_residual_h_target, and "
                    "require_uureal_block_ode."
                )
        if self.require_prior_blocks and not self.get_prior:
            raise ValueError("require_prior_blocks=True requires get_prior=True.")
        assert not get_Hamiltonian * get_DM, "Hamiltonian and Density Matrix can only loaded one at a time, for which will occupy the same attribute in the AtomicData."

        self.num_graphs = 0
        self.file_map = []
        self.index_map = []
        self._lmdb_path_map = []
        self._lmdb_env_cache = {}
        self._dynamic_batch_cost_parts_cache = {}
        # LMDB records are immutable for the lifetime of a read-only dataset
        # worker.  Cryptographic graph/row/prior/target validation is therefore
        # required only on the first successful read of each physical record in
        # that worker.  Keep this cache process-local: a forked/spawned worker
        # must establish the fail-closed contract for itself before reusing it.
        self._validated_record_contracts = {}
        self._validated_record_contracts_pid = os.getpid()
        self._validated_physical_h0_dataset = None
        for file in self.info_files.keys():
            lmdb_paths = self.simple_get_lmdb_path(file)
            for lmdb_path in lmdb_paths:
                db_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=2048)
                with db_env.begin(buffers=True) as txn:
                    self.num_graphs += txn.stat()['entries']
                    self.file_map += [file] * txn.stat()['entries']
                    self.index_map += list(range(txn.stat()['entries']))
                    self._lmdb_path_map += [lmdb_path] * txn.stat()['entries']
                db_env.close()

        # Composition-INDEPENDENT AND path-INDEPENDENT shard identities for
        # sample_uid.  The shard ordinal must be a function of the shard's
        # CONTENT alone -- not of which other shards happen to be co-loaded, and
        # not of the shard's on-disk path -- otherwise the same physical record's
        # packed uid (and thus its SEEDED prior epsilon) changes when the dataset
        # composition changes (single-shard debug vs multi-shard production, an
        # added/removed shard, a rank-local shard subset, a ConcatDataset) OR when
        # the byte-identical LMDB is simply relocated (container/multi-node mount,
        # archive restore, a different working directory).  A content-stable
        # fingerprint of the shard (``_shard_content_fingerprint``, hashed down by
        # ``_stable_shard_ordinal``) gives that; hashing the realpath itself does
        # not (it is stable across composition but not across relocation), and the
        # dense 0..K-1 ordinal it originally replaced gave neither (a shard's
        # position in the sorted set moved with the set).  Two shards whose content
        # fingerprints collide -- including the pathological case of the SAME
        # content genuinely mounted twice under this dataset's roots -- fail
        # closed (see ``_build_shard_uid_offsets``) rather than silently
        # correlating two shards' per-record identities and prior noise.
        self._shard_uid_offsets = _build_shard_uid_offsets(
            os.path.realpath(path) for path in self._lmdb_path_map
        )

    def len(self):
        return self.num_graphs

    def simple_get_lmdb_path(self, folder_name: str):
        """
        Finds LMDB directory paths matching the given folder name under root path(s).
        Supports wildcards in root paths and returns all existing matches.

        Args:
            folder_name: Folder name (or path). Only the base name is used for matching.

        Returns:
            list[str]: List of existing LMDB paths. Empty list if none found.

        Notes:
            - Uses only the base name of `folder_name` (e.g., "data" from "/path/to/data")
            - Processes wildcards (*, ?, []) in root paths via `glob`
            - Handles both single root (str) and multiple roots (list)
        """
        folder_name = os.path.split(folder_name)[-1]  # Keep only base name

        # Normalize root paths to list for consistent processing
        root_paths = [self.root] if isinstance(self.root, str) else self.root
        candidate_paths = []

        for root_path in root_paths:
            abs_path = os.path.abspath(root_path)

            # Handle wildcard-containing roots
            if any(char in abs_path for char in ['*', '?', '[']):
                for expanded_path in glob.glob(abs_path):
                    if os.path.isdir(expanded_path):
                        candidate_paths.append(os.path.join(expanded_path, folder_name))
            # Standard path processing
            else:
                candidate_paths.append(os.path.join(abs_path, folder_name))

        # Return all existing paths
        return [path for path in candidate_paths if os.path.exists(path)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lmdb_env_cache"] = {}
        state["_validated_record_contracts"] = {}
        state["_validated_record_contracts_pid"] = None
        state["_validated_physical_h0_dataset"] = None
        # Drop the process-local dynamic-batch cost cache so it is not pickled
        # into DataLoader worker processes (stale/oversized across a fork) (BUG 6).
        state["_dynamic_batch_cost_parts_cache"] = None
        return state

    def _record_contract_validation_state(self, idx: int):
        """Return the worker-local immutable-record cache key and cache.

        The normal constructor builds an exact LMDB path for every dataset
        index.  The tuple also contains the LMDB-internal integer key, so two
        shards with the same local row number cannot alias.  A deterministic
        path tuple is retained only for lightweight historical ``__new__``
        callers that do not expose ``_lmdb_path_map``.
        """

        pid = os.getpid()
        cache_pid = getattr(self, "_validated_record_contracts_pid", None)
        cache = getattr(self, "_validated_record_contracts", None)
        if not isinstance(cache, dict) or cache_pid != pid:
            cache = {}
            self._validated_record_contracts = cache
            self._validated_record_contracts_pid = pid

        raw_idx = int(idx)
        index_map = getattr(self, "index_map", None)
        record_idx = (
            int(index_map[raw_idx])
            if index_map is not None and len(index_map) > raw_idx
            else raw_idx
        )
        lmdb_paths = getattr(self, "_lmdb_path_map", None)
        if (
            lmdb_paths is not None
            and index_map is not None
            and len(lmdb_paths) == len(index_map)
        ):
            path_identity = (os.path.realpath(lmdb_paths[raw_idx]),)
        elif hasattr(self, "root") and hasattr(self, "file_map"):
            path_identity = tuple(
                os.path.realpath(path)
                for path in self.simple_get_lmdb_path(self.file_map[raw_idx])
            )
        else:
            # Compatibility only for lightweight ``__new__`` unit fixtures;
            # normal LMDBDataset instances always take one of the path-backed
            # branches above.
            file_map = getattr(self, "file_map", ())
            logical_file = file_map[raw_idx] if len(file_map) > raw_idx else "<memory>"
            path_identity = (f"logical:{logical_file}:dataset:{id(self)}",)
        return (path_identity, record_idx), cache

    def _resolve_shard_realpath(self, raw_idx: int) -> str:
        """Resolve the shard realpath for ``raw_idx`` the way ``_load_data_dict`` does.

        The path-backed ``_lmdb_path_map`` branch is authoritative for normal
        instances and is independent of any intervening ``_load_data_dict`` call
        (unlike ``_last_lmdb_record_identity``, which the dedicated-H0 audit loop
        overwrites). Lightweight ``__new__`` fixtures fall back to the logical
        file identity.
        """
        index_map = getattr(self, "index_map", None)
        lmdb_paths = getattr(self, "_lmdb_path_map", None)
        if (
            lmdb_paths is not None
            and index_map is not None
            and len(lmdb_paths) == len(index_map)
        ):
            return os.path.realpath(lmdb_paths[raw_idx])
        if hasattr(self, "file_map") and len(getattr(self, "file_map", ())) > raw_idx:
            # Lightweight __new__ fixtures may carry a file_map without the
            # path-resolution attributes (root, ...); fall back to the logical
            # identity rather than assuming the full constructor ran.
            if hasattr(self, "root"):
                try:
                    candidates = self.simple_get_lmdb_path(self.file_map[raw_idx])
                except (AttributeError, TypeError):
                    candidates = None
                if candidates:
                    return os.path.realpath(candidates[0])
            return f"logical:{self.file_map[raw_idx]}"
        return "logical:<memory>"

    def _compute_sample_uid(self, idx: int) -> int:
        """Return the stable per-graph record identity as a packed int64.

        Packing: ``(shard_id << 32) | (row_id & 0xFFFFFFFF)``.

        * ``row_id`` is the LMDB integer record key within its shard
          (``index_map[idx]``), i.e. the 4-byte big-endian key used to fetch the
          record, so ``0 <= row_id < 2**32``.
        * ``shard_id`` is a content-stable 31-bit hash of the shard's CONTENT
          fingerprint (``_shard_content_fingerprint`` + ``_stable_shard_ordinal``,
          cached in ``_shard_uid_offsets`` keyed by realpath for O(1) lookup) -- a
          function of what the shard contains, not of its on-disk path nor of the
          dataset's shard set.

        The result fits in 63 bits (a positive int64). It is independent of batch
        composition/DataLoader order, of which other shards are co-loaded, AND of
        the shard's on-disk location (relocating a byte-identical LMDB does not
        change its records' uids), because the shard ordinal is a hash of the
        shard's own content and the row key is a deterministic function of the
        immutable on-disk record -- neither depends on the enclosing dataset's
        shard set or mount path.
        """
        raw_idx = int(idx)
        index_map = getattr(self, "index_map", None)
        if index_map is not None and len(index_map) > raw_idx:
            row_id = int(index_map[raw_idx])
        else:
            row_id = raw_idx
        shard_realpath = self._resolve_shard_realpath(raw_idx)
        offsets = getattr(self, "_shard_uid_offsets", None) or {}
        if shard_realpath in offsets:
            shard_id = int(offsets[shard_realpath])
        else:
            # No precomputed offset: only reachable from lightweight ``__new__``
            # fixtures with no real ``_shard_uid_offsets`` map (see
            # ``_resolve_shard_realpath``), where ``shard_realpath`` is an opaque
            # logical identity string, not an openable LMDB path -- hash it
            # directly rather than attempting to fingerprint non-existent content.
            shard_id = _stable_shard_ordinal(shard_realpath)
        return ((shard_id & 0x7FFFFFFF) << 32) | (row_id & 0xFFFFFFFF)

    def _assert_dedicated_physical_h0_dataset_fingerprint(self) -> None:
        """Verify the externally pinned, ordered H0 content once per worker."""

        expected = require_sha256(
            getattr(self, "expected_physical_h0_source_fingerprint", None),
            field="expected_physical_h0_source_fingerprint",
        )
        cached = getattr(self, "_validated_physical_h0_dataset", None)
        cache_key = (os.getpid(), expected)
        if cached == cache_key:
            return

        record_fingerprints = []
        for index in range(len(self.index_map)):
            record = self._load_data_dict(index)
            if record.get(PHYSICAL_H0_SOURCE_KEY) != DEDICATED_PHYSICAL_H0_SOURCE:
                raise ValueError(
                    "A dedicated physical-H0 dataset cannot mix H0 authority modes."
                )
            declared = require_sha256(
                record.get(PHYSICAL_H0_SOURCE_FINGERPRINT_KEY),
                field=PHYSICAL_H0_SOURCE_FINGERPRINT_KEY,
            )
            if declared != expected:
                raise ValueError(
                    "Dedicated physical-H0 source fingerprint differs from the "
                    "externally trusted split commitment."
                )
            record_fingerprints.append(physical_h0_record_fingerprint(record))

        actual = physical_h0_dataset_fingerprint(record_fingerprints)
        if actual != expected:
            raise ValueError(
                "Dedicated physical-H0 block content does not match the externally "
                "trusted split commitment."
            )
        self._validated_physical_h0_dataset = cache_key

    def invalidate_dynamic_batch_costs(self) -> None:
        super().invalidate_dynamic_batch_costs()
        self._dynamic_batch_cost_parts_cache = {}

    def __del__(self):
        for env in getattr(self, "_lmdb_env_cache", {}).values():
            try:
                env.close()
            except Exception:
                pass

    def _get_lmdb_env(self, path: str):
        cache = getattr(self, "_lmdb_env_cache", None)
        if cache is None:
            cache = {}
            self._lmdb_env_cache = cache
        env = cache.get(path)
        if env is None:
            env = lmdb.open(
                path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048,
            )
            cache[path] = env
        return env

    def _load_data_dict(self, idx: int):
        lmdb_paths = getattr(self, "_lmdb_path_map", None)
        if lmdb_paths is not None and len(lmdb_paths) == len(self.index_map):
            candidate_paths = [lmdb_paths[idx]]
        else:
            candidate_paths = self.simple_get_lmdb_path(self.file_map[idx])

        key = self.index_map[int(idx)].to_bytes(length=4, byteorder='big')
        for lmdb_path in candidate_paths:
            env = self._get_lmdb_env(lmdb_path)
            with env.begin(buffers=True) as txn:
                data = txn.get(key)
                if data is not None:
                    serialized = bytes(data)
                    # Read-only diagnostics used by the standalone loader
                    # benchmark.  Recording the size of the bytes that are
                    # already copied for pickle does not add another LMDB read
                    # or warm the filesystem cache ahead of the timed get.
                    self._last_lmdb_pickle_bytes = len(serialized)
                    self._last_lmdb_record_identity = (
                        os.path.realpath(lmdb_path),
                        int(self.index_map[int(idx)]),
                    )
                    return pickle.loads(serialized)
        raise IndexError(f"LMDB entry {self.index_map[int(idx)]} not found for dataset index {idx}")

    def get_dynamic_batch_cost_parts(self, idx: int) -> Dict[str, int]:
        raw_idx = self._resolve_dynamic_batch_index(idx)
        cache = getattr(self, "_dynamic_batch_cost_parts_cache", None)
        if cache is None:
            cache = {}
            self._dynamic_batch_cost_parts_cache = cache
        if raw_idx in cache:
            return dict(cache[raw_idx])

        data_dict = self._load_data_dict(raw_idx)
        parts = _dynamic_batch_parts_from_data(data_dict)
        block_keys = [
            "hamiltonian",
            getattr(self, "h0_key", "hamiltonian_0"),
            getattr(self, "prior_raw_key", "hamiltonian_p2"),
            "hamiltonian_0",
            "density_matrix",
            "overlap",
        ]
        for key in dict.fromkeys(block_keys):
            block_count = _count_offsite_lmdb_blocks(data_dict.get(key, None))
            if block_count > 0:
                parts["block"] = block_count
                break
        cache[raw_idx] = dict(parts)
        return parts

    @property
    def raw_file_names(self):
        # TODO: this is not implemented.
        # need to give a valid path so the download would not be triggered
        return ["data.mdb", "lock.mdb"]

    @property
    def raw_dir(self):
        return self.root

    def download(self):
        if (not hasattr(self, "url")) or (self.url is None):
            # Don't download, assume present. Later could have FileNotFound if the files don't actually exist
            pass
        else:
            download_path = download_url(self.url, self.raw_dir)
            if download_path.endswith(".zip"):
                extract_zip(download_path, self.raw_dir)

    def get(self, idx):
        # Thin orchestration: the ~900-line record decode was extracted into
        # dptb.data.dataset.record_pipeline (StoredRecordReader,
        # RecordSchemaValidator, GraphResolver, AtomicDataFactory, PriorDecoder,
        # TargetDecoder) threaded by an immutable SampleContext.  Imported
        # lazily to keep the record_pipeline <-> lmdb_dataset dependency acyclic.
        from dptb.data.dataset.record_pipeline import build_atomic_data

        return build_atomic_data(self, idx)

    def E3statistics(self, model: torch.nn.Module = None):

        if not self.get_Hamiltonian and not self.get_DM:
            return None

        if model is not None:
            if not isinstance(model.node_prediction_h, torch.nn.Module):
                return None

        assert self.transform is not None
        idp = model.embedding.idp
        has_soc = model.embedding.idp.has_soc

        e3h = E3Hamiltonian(basis=idp.basis, decompose=True, soc=has_soc)
        idp.get_irreps()

        # [FIX] Correctly count n_scalar for both SOC (0e+0o) and non-SOC (0e) cases.
        # Original code only counted the first type of scalar in sorted irreps.
        # sorted_irreps = idp.orbpair_irreps.sort()[0].simplify()
        # n_scalar = sorted_irreps[0].mul if sorted_irreps[0].ir.l == 0 else 0
        n_scalar = sum(mul for mul, (l, p) in idp.orbpair_irreps if l == 0)

        # init a count dict of atom species
        count_at = {}
        for at, tp in idp.chemical_symbol_to_type.items():
            count_at[tp] = 0

        count_bt = {}
        for bt, tp in idp.bond_to_type.items():
            count_bt[tp] = 0

        # calculate norm & mean
        node_norm_ave = torch.zeros(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        node_square_ave = torch.zeros(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        node_norm_std = torch.ones(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        node_scalar_ave = torch.zeros(len(idp.chemical_symbol_to_type), n_scalar)
        node_scalar_square_ave = torch.zeros(len(idp.chemical_symbol_to_type), n_scalar)
        node_scalar_std = torch.ones(len(idp.chemical_symbol_to_type), n_scalar)
        edge_norm_ave = torch.zeros(len(idp.bond_types), idp.orbpair_irreps.num_irreps)
        edge_square_ave = torch.zeros(len(idp.bond_types), idp.orbpair_irreps.num_irreps)
        edge_norm_std = torch.ones(len(idp.bond_types), idp.orbpair_irreps.num_irreps)
        edge_scalar_ave = torch.zeros(len(idp.bond_types), n_scalar)
        edge_scalar_square_ave = torch.zeros(len(idp.bond_types), n_scalar)
        edge_scalar_std = torch.ones(len(idp.bond_types), n_scalar)

        for idx in tqdm(range(self.len()), desc="Collecting E3 irreps statistics: "):
            with torch.no_grad():
                atomicdata = idp(self.get(idx=idx)).to_dict()
                if atomicdata[AtomicDataDict.EDGE_FEATURES_KEY].abs().sum() < 1e-7:
                    continue
                atomicdata = e3h(atomicdata)

                subcount_at = {}
                for at, tp in idp.chemical_symbol_to_type.items():
                    subcount_at[tp] = 0

                subcount_bt = {}
                for bt, tp in idp.bond_to_type.items():
                    subcount_bt[tp] = 0

                onsite_mask = idp.mask_to_nrme[atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten()]

                for at, tp in idp.chemical_symbol_to_type.items():
                    count_scalar = 0
                    at_mask = onsite_mask[atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten().eq(tp)]
                    n_at = at_mask.shape[0]

                    if n_at > 0:
                        at_onsite = atomicdata[AtomicDataDict.NODE_FEATURES_KEY][
                            atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten().eq(tp)]
                        for ir, s in enumerate(idp.orbpair_irreps.slices()):
                            sub_tensor = at_onsite[:, s]
                            if sub_tensor.shape[-1] == 1:
                                count_scalar += 1
                            norms = torch.norm(sub_tensor, p=2, dim=1)
                            # we do a running avg and var here
                            node_norm_ave[tp][ir] = (node_norm_ave[tp][ir] * count_at[tp] + norms.sum(dim=0)) / (
                                        count_at[tp] + n_at)
                            node_square_ave[tp][ir] = (node_square_ave[tp][ir] * count_at[tp] + (norms ** 2).sum(
                                dim=0)) / (count_at[tp] + n_at)
                            if count_at[tp] + n_at > 1:
                                node_norm_std[tp][ir] = torch.nan_to_num(torch.sqrt(
                                    (count_at[tp] + n_at) / (count_at[tp] + n_at - 1) * (
                                                node_square_ave[tp][ir] - node_norm_ave[tp][ir] ** 2)), nan=0.0)
                            else:
                                node_norm_std[tp][ir] = 1.0

                            if sub_tensor.shape[-1] == 1:
                                # is scalar
                                node_scalar_ave[tp][count_scalar - 1] = (node_scalar_ave[tp][count_scalar - 1] *
                                                                         count_at[tp] + sub_tensor.sum()) / (
                                                                                    count_at[tp] + n_at)
                                node_scalar_square_ave[tp][count_scalar - 1] = (node_scalar_square_ave[tp][
                                                                                    count_scalar - 1] * count_at[tp] + (
                                                                                            sub_tensor ** 2).sum()) / (
                                                                                           count_at[tp] + n_at)
                                if count_at[tp] + n_at > 1:
                                    node_scalar_std[tp][count_scalar - 1] = torch.nan_to_num(torch.sqrt(
                                        (count_at[tp] + n_at) / (count_at[tp] + n_at - 1) * (
                                                    node_scalar_square_ave[tp][count_scalar - 1] - node_scalar_ave[tp][
                                                count_scalar - 1] ** 2)), nan=0.0)
                                else:
                                    node_scalar_std[tp][count_scalar - 1] = 1.0
                        subcount_at[tp] = n_at
                        count_at[tp] += n_at
                assert sum(subcount_at.values()) == atomicdata[AtomicDataDict.POSITIONS_KEY].shape[0]

                # edge statistics
                hopping_mask = idp.mask_to_erme[atomicdata[AtomicDataDict.EDGE_TYPE_KEY].flatten()]
                for bt, tp in idp.bond_to_type.items():
                    count_scalar = 0
                    bt_mask = hopping_mask[atomicdata[AtomicDataDict.EDGE_TYPE_KEY].flatten().eq(tp)]
                    n_bt = bt_mask.shape[0]

                    if n_bt > 0:
                        bt_hopping = atomicdata[AtomicDataDict.EDGE_FEATURES_KEY][
                            atomicdata[AtomicDataDict.EDGE_TYPE_KEY].flatten().eq(tp)]
                        for ir, s in enumerate(idp.orbpair_irreps.slices()):
                            sub_tensor = bt_hopping[:, s]
                            if sub_tensor.shape[-1] == 1:
                                count_scalar += 1

                            norms = torch.norm(sub_tensor, p=2, dim=1)
                            # we do a running avg and var here
                            edge_norm_ave[tp][ir] = (edge_norm_ave[tp][ir] * count_bt[tp] + norms.sum(dim=0)) / (
                                        count_bt[tp] + n_bt)
                            edge_square_ave[tp][ir] = (edge_square_ave[tp][ir] * count_bt[tp] + (norms ** 2).sum(
                                dim=0)) / (count_bt[tp] + n_bt)
                            if count_bt[tp] + n_bt > 1:
                                edge_norm_std[tp][ir] = torch.nan_to_num(torch.sqrt(
                                    (count_bt[tp] + n_bt) / (count_bt[tp] + n_bt - 1) * (
                                                edge_square_ave[tp][ir] - edge_norm_ave[tp][ir] ** 2)), nan=0.0)
                            else:
                                edge_norm_std[tp][ir] = 1.0
                            if sub_tensor.shape[-1] == 1:
                                # is scalar
                                edge_scalar_ave[tp][count_scalar - 1] = (edge_scalar_ave[tp][count_scalar - 1] *
                                                                         count_bt[tp] + sub_tensor.sum()) / (
                                                                                    count_bt[tp] + n_bt)
                                edge_scalar_square_ave[tp][count_scalar - 1] = (edge_scalar_square_ave[tp][
                                                                                    count_scalar - 1] * count_bt[tp] + (
                                                                                            sub_tensor ** 2).sum()) / (
                                                                                           count_bt[tp] + n_bt)
                                if count_bt[tp] + n_bt > 1:
                                    edge_scalar_std[tp][count_scalar - 1] = torch.nan_to_num(torch.sqrt(
                                        (count_bt[tp] + n_bt) / (count_bt[tp] + n_bt - 1) * (
                                                    edge_scalar_square_ave[tp][count_scalar - 1] - edge_scalar_ave[tp][
                                                count_scalar - 1] ** 2)), nan=0.0)
                                else:
                                    edge_scalar_std[tp][count_scalar - 1] = 1.0

                        subcount_bt[tp] = n_bt
                        count_bt[tp] += n_bt
                assert sum(subcount_bt.values()) == atomicdata[AtomicDataDict.EDGE_INDEX_KEY].shape[1]

        stats = {}
        stats["node"] = {
            "norm_ave": node_norm_ave,
            "norm_std": node_norm_std,
            "scalar_ave": node_scalar_ave,
            "scalar_std": node_scalar_std
        }
        stats["edge"] = {
            "norm_ave": edge_norm_ave,
            "norm_std": edge_norm_std,
            "scalar_ave": edge_scalar_ave,
            "scalar_std": edge_scalar_std,
        }

        if model is not None:
            # initilize the model param with statistics
            scalar_mask = torch.BoolTensor([ir.dim == 1 for ir in model.idp.orbpair_irreps])
            node_shifts = stats["node"]["scalar_ave"]
            node_scales = stats["node"]["norm_ave"]
            node_scales[:, scalar_mask] = stats["node"]["scalar_std"]

            edge_shifts = stats["edge"]["scalar_ave"]
            edge_scales = stats["edge"]["norm_ave"]
            edge_scales[:, scalar_mask] = stats["edge"]["scalar_std"]
            model.node_prediction_h.set_scale_shift(scales=node_scales, shifts=node_shifts)
            model.edge_prediction_h.set_scale_shift(scales=edge_scales, shifts=edge_shifts)

        return stats

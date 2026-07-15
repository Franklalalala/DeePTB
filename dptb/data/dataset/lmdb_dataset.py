import numpy as np
from typing import Tuple, Dict, Any, List, Callable, Union, Optional

import torch
from dptb.utils.tools import download_url, extract_zip

import os
import os.path as osp
import glob
from dptb.data import (
    AtomicData,
    AtomicDataDict,
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
from dptb.data.interfaces.p2_contract import (
    ABSOLUTE_FULL_H_SEMANTICS,
    BASIS_FINGERPRINT_KEY,
    DUAL_PRIOR_SAMPLE_SCHEMA,
    EDGE_GRAPH_FINGERPRINT_KEY,
    FULL_H_TARGET_FINGERPRINT_KEY,
    P2_BUNDLE_FINGERPRINT_KEY,
    P2_SAMPLE_SCHEMA,
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
    require_sha256,
    resolve_prior_field_spec,
)
import pickle


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
    """Refuse residual targets that do not shrink when H0 is subtracted.

    Some historical LMDBs store an already-residual dH in the Hamiltonian
    slot (delta-in-H-slot convention, e.g. the 0516 NexTHam crystal sets).
    Enabling residual_hamiltonian there would double-subtract and inflate
    the target to H0 scale instead of shrinking it (a genuine full-H set
    shrinks ~16x on the water QHFlow2 data). Magnitudes are compared over
    all stored block entries.
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
            f"required shrink >= {min_shrink}x). The Hamiltonian slot most "
            "likely already stores a residual/delta target (delta-in-H-slot "
            f"convention), or '{h0_key}' is not a valid H0 for it. Refusing to "
            "double-subtract; disable residual_hamiltonian for this dataset."
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


def assert_absolute_full_h_target_contract(data_dict: Dict[str, Any]) -> None:
    """Require an explicit, versioned absolute-H target declaration."""

    allowed_schemas = {P2_SAMPLE_SCHEMA, DUAL_PRIOR_SAMPLE_SCHEMA}
    if data_dict.get(SAMPLE_SCHEMA_KEY) not in allowed_schemas:
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
    # Historical target fields are never accepted as evidence of absolute H.
    legacy = [key for key in _PREPACKED_HAMILTONIAN_TARGET_KEYS if key in data_dict]
    if source == "dedicated_full_h_blocks" and legacy:
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


def build_residual_hamiltonian_target_blocks(
    data_dict: Dict[str, Any],
    blocks: Dict[Any, Any],
    *,
    h0_key: str = "hamiltonian_0",
) -> Dict[Any, np.ndarray]:
    """Build H-H0 targets with fail-closed source and shape validation."""
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

    # Validate every accessed record. Residual datasets are opt-in, and the
    # small extra reduction is preferable to silently accepting a mixed shard.
    assert_residual_target_shrinks(blocks, delta_blocks, h0_key=h0_key)
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
    expected_prior_source_fingerprint = None
    audit_prior_representations = False
    require_prior_blocks = False

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
            get_overlap: bool = False,
            get_DM: bool = False,
            get_eigenvalues: bool = False,
            h0_key: str = "hamiltonian_0",
            prefer_precomputed_h0: bool = True,
            prior_kind: str = "p2",
            prior_raw_key: Optional[str] = None,
            prefer_precomputed_prior: Optional[bool] = None,
            require_full_h_target: bool = False,
            expected_prior_source_fingerprint: Optional[str] = None,
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
        self.expected_prior_source_fingerprint = (
            str(expected_prior_source_fingerprint)
            if expected_prior_source_fingerprint not in {None, ""}
            else None
        )
        self.audit_prior_representations = bool(audit_prior_representations)
        self.require_prior_blocks = bool(require_prior_blocks)
        if self.require_full_h_target and not self.get_Hamiltonian:
            raise ValueError("require_full_h_target=True requires get_Hamiltonian=True.")
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
        data_dict = self._load_data_dict(idx)
        record_contract_key, validated_record_contracts = (
            self._record_contract_validation_state(idx)
        )
        record_contract_already_validated = record_contract_key in validated_record_contracts
        prior_spec = getattr(
            self,
            "prior_spec",
            resolve_prior_field_spec(getattr(self, "prior_kind", "p2")),
        )
        if getattr(self, "require_full_h_target", False):
            assert_absolute_full_h_target_contract(data_dict)
        stored_basis_fingerprint = data_dict.get(BASIS_FINGERPRINT_KEY)
        # Historical, non-fingerprinted records need not expose a production
        # OrbitalMapper.  Resolve the mapper fingerprint lazily only when the
        # record actually declares the versioned physical-prior contract.
        basis_fingerprint = None
        if stored_basis_fingerprint is not None:
            if self.type_mapper is None:
                raise ValueError(
                    "A fingerprinted Hamiltonian record requires a configured "
                    "OrbitalMapper/basis."
                )
            basis_fingerprint = mapper_basis_fingerprint(self.type_mapper)
            stored_basis_fingerprint = require_sha256(
                stored_basis_fingerprint, field=BASIS_FINGERPRINT_KEY
            )
            if stored_basis_fingerprint != basis_fingerprint:
                raise ValueError(
                    "basis_fingerprint mismatch between LMDB record and configured "
                    f"OrbitalMapper: record={stored_basis_fingerprint}, "
                    f"mapper={basis_fingerprint}."
                )
        cell, pos, atomic_numbers = \
            data_dict[AtomicDataDict.CELL_KEY], \
                data_dict[AtomicDataDict.POSITIONS_KEY], \
                data_dict[AtomicDataDict.ATOMIC_NUMBERS_KEY]

        pbc = data_dict[AtomicDataDict.PBC_KEY]

        if self.get_Hamiltonian:
            blocks = data_dict.get("hamiltonian", None)
            # kk, vv = blocks.keys(), blocks.values()
            # vv = map(lambda x: np.frombuffer(x, np.float32).reshape, vv)
            # blocks = dict(zip(kk, vv))
            # del kk
            # del vv

        if self.get_overlap:
            overlap = data_dict.get("overlap", None)
            # kk, vv = overlap.keys(), overlap.values()
            # vv = map(lambda x: np.frombuffer(x, np.float32), vv)
            # overlap = dict(zip(kk, vv))
            # del kk
            # del vv
        else:
            overlap = False

        if self.get_DM:
            blocks = data_dict.get("density_matrix", None)
            # kk, vv = blocks.keys(), blocks.values()
            # vv = map(lambda x: np.frombuffer(x, np.float32), vv)
            # blocks = dict(zip(kk, vv))
            # del kk
            # del vv

        if not (self.get_Hamiltonian or self.get_DM):
            blocks = False

        pre_node_features = data_dict.get(AtomicDataDict.NODE_FEATURES_KEY, None)
        pre_edge_features = data_dict.get(AtomicDataDict.EDGE_FEATURES_KEY, None)
        pre_node_overlap = data_dict.get(AtomicDataDict.NODE_OVERLAP_KEY, None)
        pre_edge_overlap = data_dict.get(AtomicDataDict.EDGE_OVERLAP_KEY, None)

        h0_blocks = data_dict.get(self.h0_key, None) if self.get_H0 else None
        node_h0 = data_dict.get(AtomicDataDict.NODE_H0_KEY, None) if self.get_H0 else None
        edge_h0 = data_dict.get(AtomicDataDict.EDGE_H0_KEY, None) if self.get_H0 else None
        p2_blocks = data_dict.get(self.prior_raw_key, None) if self.get_prior else None
        node_p2 = data_dict.get(prior_spec.node_rme_key, None) if self.get_prior else None
        edge_p2 = data_dict.get(prior_spec.edge_rme_key, None) if self.get_prior else None
        p2_blockwise_keys = prior_spec.block_fields
        p2_blockwise_present = [key in data_dict for key in p2_blockwise_keys]
        full_h_target_present = [
            key in data_dict for key in _PREPACKED_FULL_H_TARGET_KEYS
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
            validate_non_soc_p2_blocks(p2_blocks, key=self.prior_raw_key)
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
        soc_uureal_keep_mask = getattr(self.type_mapper, "mask_uureal", None)

        if self.info_files[self.file_map[idx]]['train_dip'] == True:
            self.info_files[self.file_map[idx]].update({'dip': data_dict['dipole_moment']})

        if self.info_files[self.file_map[idx]]['train_w_charge'] == True:
            self.info_files[self.file_map[idx]].update({'charge': np.array(data_dict['charge'])})

        if self.info_files[self.file_map[idx]]['train_w_eps'] == True:
            self.info_files[self.file_map[idx]].update({'dielectric_constant': np.array(data_dict['dielectric_constant'])})
        if self.info_files[self.file_map[idx]]['train_w_homo_lumo_gap'] == True:
            self.info_files[self.file_map[idx]].update({
                'GAP_eV': np.array(data_dict['GAP_eV']),
                'LUMO_eV': np.array(data_dict['LUMO_eV']),
                'HOMO_eV': np.array(data_dict['HOMO_eV']),
            })

        if self.info_files[self.file_map[idx]]['train_polar'] == True:
            self.info_files[self.file_map[idx]].update({'polar': data_dict['polarizability']})

        if self.info_files[self.file_map[idx]]['wave_align'] == True:
            orbital_energies = data_dict.get('orbital_energies', 0)
            orbital_coefficients = data_dict.get('orbital_coefficients', 0)
            self.info_files[self.file_map[idx]].update(
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
            key: self.info_files[self.file_map[idx]][key] for key in cache_keys
        }

        # transform blocks to atomicdata features, or use precomputed features directly
        need_main_features = bool(self.get_Hamiltonian or self.get_DM)
        need_overlap_features = bool(self.get_overlap)
        has_pre_main = pre_node_features is not None and pre_edge_features is not None
        has_pre_overlap = (not need_overlap_features) or (
            pre_edge_overlap is not None and (self.orthogonal or pre_node_overlap is not None)
        )

        uses_pre_main = bool(has_pre_main and has_pre_overlap)
        uses_pre_physical_h0 = bool(all(physical_h0_present))
        uses_pre_h0 = bool(
            self.get_H0
            and self.prefer_precomputed_h0
            and node_h0 is not None
            and edge_h0 is not None
        )
        uses_p2 = bool(
            self.get_prior
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
            self.get_prior and (all(p2_feature_present) or any(p2_blockwise_present))
        )
        requires_stored_target_graph = bool(
            getattr(self, "require_full_h_target", False)
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
        canonical_stored_edge = canonical_stored_shift = None
        if has_stored_edge_graph:
            if record_contract_already_validated:
                canonical_stored_edge, canonical_stored_shift = (
                    validated_record_contracts[record_contract_key]
                )
            else:
                canonical_stored_edge, canonical_stored_shift = canonical_edge_graph(
                    atomic_numbers, stored_edge_index, stored_edge_shift
                )
            if requires_fingerprinted_graph:
                if stored_basis_fingerprint is None or basis_fingerprint is None:
                    raise ValueError(
                        "Row-aligned P2/Full-H tensors require basis_fingerprint metadata."
                    )
                if not record_contract_already_validated:
                    actual_graph_fingerprint = edge_graph_fingerprint(
                        atomic_numbers,
                        canonical_stored_edge,
                        canonical_stored_shift,
                        basis_fingerprint=basis_fingerprint,
                    )
                    assert_record_fingerprint(
                        data_dict,
                        field=EDGE_GRAPH_FINGERPRINT_KEY,
                        actual=actual_graph_fingerprint,
                    )
                    if schema_v2_row_aligned:
                        actual_row_fingerprint = fingerprint_present_row_aligned_fields(
                            data_dict
                        )
                        assert_record_fingerprint(
                            data_dict,
                            field=ROW_ALIGNED_DATA_FINGERPRINT_KEY,
                            actual=actual_row_fingerprint,
                        )
                        actual_row_bundle = fingerprint_text_fields(
                            {
                                BASIS_FINGERPRINT_KEY: basis_fingerprint,
                                EDGE_GRAPH_FINGERPRINT_KEY: actual_graph_fingerprint,
                                ROW_ALIGNED_DATA_FINGERPRINT_KEY: actual_row_fingerprint,
                            },
                            (
                                BASIS_FINGERPRINT_KEY,
                                EDGE_GRAPH_FINGERPRINT_KEY,
                                ROW_ALIGNED_DATA_FINGERPRINT_KEY,
                            ),
                        )
                        assert_record_fingerprint(
                            data_dict,
                            field=ROW_ALIGNED_BUNDLE_FINGERPRINT_KEY,
                            actual=actual_row_bundle,
                        )
        # Keep dataset metadata immutable across failed reads.  Previously the
        # six training-only flags were deleted in-place and restored only after
        # graph construction; a fingerprint failure left the dataset poisoned
        # and the next retry raised KeyError instead of the original contract
        # error.
        info = {
            key: value
            for key, value in self.info_files[self.file_map[idx]].items()
            if key not in cache_info
        }
        needs_missing_env_graph = (
            info.get("er_max", None) is not None
            and data_dict.get(AtomicDataDict.ENV_INDEX_KEY, None) is None
        )
        needs_missing_onsitenv_graph = (
            info.get("oer_max", None) is not None
            and data_dict.get(AtomicDataDict.ONSITENV_INDEX_KEY, None) is None
        )
        use_stored_edge_graph = bool(
            has_stored_edge_graph
            and (
                uses_pre_main
                or uses_pre_h0
                or uses_pre_physical_h0
                or uses_p2
                or requires_stored_target_graph
            )
        )

        if use_stored_edge_graph:
            atomicdata_kwargs = {
                key: value
                for key, value in info.items()
                if key not in _ATOMICDATA_CONSTRUCTOR_OPTIONS
            }
            atomicdata_kwargs[AtomicDataDict.EDGE_INDEX_KEY] = _lmdb_tensor(
                canonical_stored_edge, torch.long
            )
            atomicdata_kwargs[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = _lmdb_tensor(
                canonical_stored_shift, torch.get_default_dtype()
            )
            if data_dict.get(AtomicDataDict.ENV_INDEX_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ENV_INDEX_KEY] = _lmdb_tensor(
                    data_dict[AtomicDataDict.ENV_INDEX_KEY], torch.long
                )
            if data_dict.get(AtomicDataDict.ENV_CELL_SHIFT_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ENV_CELL_SHIFT_KEY] = _lmdb_tensor(
                    data_dict[AtomicDataDict.ENV_CELL_SHIFT_KEY], torch.get_default_dtype()
                )
            if data_dict.get(AtomicDataDict.ONSITENV_INDEX_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ONSITENV_INDEX_KEY] = _lmdb_tensor(
                    data_dict[AtomicDataDict.ONSITENV_INDEX_KEY], torch.long
                )
            if data_dict.get(AtomicDataDict.ONSITENV_CELL_SHIFT_KEY, None) is not None:
                atomicdata_kwargs[AtomicDataDict.ONSITENV_CELL_SHIFT_KEY] = _lmdb_tensor(
                    data_dict[AtomicDataDict.ONSITENV_CELL_SHIFT_KEY], torch.get_default_dtype()
                )
            if needs_missing_env_graph or needs_missing_onsitenv_graph:
                # Generate only the missing auxiliary graphs, then restore the
                # authoritative stored main graph.  ``from_points`` must never
                # reorder row-aligned P2/target edges as a side effect.
                atomicdata = AtomicData.from_points(
                    pos=pos.reshape(-1, 3),
                    cell=cell.reshape(3, 3),
                    atomic_numbers=atomic_numbers,
                    pbc=pbc,
                    **info,
                )
                for key, value in atomicdata_kwargs.items():
                    atomicdata[key] = value
            else:
                atomicdata = AtomicData(
                    pos=_lmdb_tensor(pos.reshape(-1, 3), torch.get_default_dtype()),
                    cell=_lmdb_tensor(cell.reshape(3, 3), torch.get_default_dtype()),
                    atomic_numbers=_lmdb_tensor(atomic_numbers, torch.long),
                    pbc=_lmdb_tensor(pbc, torch.bool),
                    **atomicdata_kwargs,
                )
        else:
            atomicdata = AtomicData.from_points(
                pos=pos.reshape(-1, 3),
                cell=cell.reshape(3, 3),
                atomic_numbers=atomic_numbers,
                pbc=pbc,
                **info
            )
        self.info_files[self.file_map[idx]].update(cache_info)

        num_edges = atomicdata[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        num_nodes = atomicdata.num_nodes
        mapper_p2_irreps = getattr(
            getattr(self, "type_mapper", None), "orbpair_irreps", None
        )
        mapper_p2_dim = getattr(mapper_p2_irreps, "dim", None)

        if has_pre_main and has_pre_overlap:
            pre_node_features = _expand_soc_uureal_compact(
                pre_node_features,
                data_dict,
                field_name=AtomicDataDict.NODE_FEATURES_KEY,
                keep_mask=soc_uureal_keep_mask,
            )
            pre_edge_features = _expand_soc_uureal_compact(
                pre_edge_features,
                data_dict,
                field_name=AtomicDataDict.EDGE_FEATURES_KEY,
                keep_mask=soc_uureal_keep_mask,
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
            if need_overlap_features:
                pre_edge_overlap = torch.as_tensor(pre_edge_overlap)
                if pre_edge_overlap.shape[0] != num_edges:
                    raise ValueError(
                        "Precomputed LMDB edge overlap rows do not match the active graph: "
                        f"edge_overlap={tuple(pre_edge_overlap.shape)}, num_edges={num_edges}."
                    )
                if not self.orthogonal:
                    pre_node_overlap = torch.as_tensor(pre_node_overlap)
                    if pre_node_overlap.shape[0] != num_nodes:
                        raise ValueError(
                            "Precomputed LMDB node overlap rows do not match the active graph: "
                            f"node_overlap={tuple(pre_node_overlap.shape)}, num_nodes={num_nodes}."
                        )
                    atomicdata[AtomicDataDict.NODE_OVERLAP_KEY] = pre_node_overlap
                atomicdata[AtomicDataDict.EDGE_OVERLAP_KEY] = pre_edge_overlap
        elif self.get_Hamiltonian or self.get_DM or self.get_overlap:
            block_to_feature(atomicdata, self.type_mapper, blocks, overlap, self.orthogonal)

        if self.get_H0:
            if self.prefer_precomputed_h0 and node_h0 is not None and edge_h0 is not None:
                node_h0 = _expand_soc_uureal_compact(
                    node_h0,
                    data_dict,
                    field_name=AtomicDataDict.NODE_H0_KEY,
                    keep_mask=soc_uureal_keep_mask,
                )
                edge_h0 = _expand_soc_uureal_compact(
                    edge_h0,
                    data_dict,
                    field_name=AtomicDataDict.EDGE_H0_KEY,
                    keep_mask=soc_uureal_keep_mask,
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
                block_to_feature(
                    atomicdata,
                    self.type_mapper,
                    h0_blocks,
                    False,
                    self.orthogonal,
                    node_field=AtomicDataDict.NODE_H0_KEY,
                    edge_field=AtomicDataDict.EDGE_H0_KEY,
                )
            elif node_h0 is not None and edge_h0 is not None:
                atomicdata[AtomicDataDict.NODE_H0_KEY] = _expand_soc_uureal_compact(
                    node_h0,
                    data_dict,
                    field_name=AtomicDataDict.NODE_H0_KEY,
                    keep_mask=soc_uureal_keep_mask,
                )
                atomicdata[AtomicDataDict.EDGE_H0_KEY] = _expand_soc_uureal_compact(
                    edge_h0,
                    data_dict,
                    field_name=AtomicDataDict.EDGE_H0_KEY,
                    keep_mask=soc_uureal_keep_mask,
                )

        if self.get_prior:
            if self.prefer_precomputed_prior and all(p2_feature_present):
                node_p2, edge_p2 = validate_p2_feature_pair(
                    node_p2,
                    edge_p2,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    feature_dim=mapper_p2_dim,
                    prior_kind=prior_spec.kind,
                    check_finite=not record_contract_already_validated,
                )
                atomicdata[prior_spec.node_rme_key] = node_p2
                atomicdata[prior_spec.edge_rme_key] = edge_p2
            elif p2_blocks is not None:
                block_to_feature(
                    atomicdata,
                    self.type_mapper,
                    p2_blocks,
                    False,
                    self.orthogonal,
                    node_field=prior_spec.node_rme_key,
                    edge_field=prior_spec.edge_rme_key,
                    missing_block_policy="error",
                )
                node_p2, edge_p2 = validate_p2_feature_pair(
                    atomicdata[prior_spec.node_rme_key]
                    if prior_spec.node_rme_key in atomicdata
                    else None,
                    atomicdata[prior_spec.edge_rme_key]
                    if prior_spec.edge_rme_key in atomicdata
                    else None,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    feature_dim=mapper_p2_dim,
                    prior_kind=prior_spec.kind,
                    check_finite=not record_contract_already_validated,
                )
                atomicdata[prior_spec.node_rme_key] = node_p2
                atomicdata[prior_spec.edge_rme_key] = edge_p2
            elif all(p2_feature_present):
                node_p2, edge_p2 = validate_p2_feature_pair(
                    node_p2,
                    edge_p2,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    feature_dim=mapper_p2_dim,
                    prior_kind=prior_spec.kind,
                    check_finite=not record_contract_already_validated,
                )
                atomicdata[prior_spec.node_rme_key] = node_p2
                atomicdata[prior_spec.edge_rme_key] = edge_p2
            else:
                raise ValueError(
                    f"get_prior=True requires either raw '{self.prior_raw_key}' AO "
                    f"blocks or precomputed {prior_spec.node_rme_key}/"
                    f"{prior_spec.edge_rme_key} features."
                )

            if requires_stored_p2_graph:
                sample_schema = data_dict.get(SAMPLE_SCHEMA_KEY)
                if sample_schema not in prior_spec.allowed_sample_schemas:
                    raise ValueError(
                        f"Precomputed {prior_spec.kind.upper()} tensors require "
                        "explicit sample schema in "
                        f"{prior_spec.allowed_sample_schemas!r}; got "
                        f"{sample_schema!r}."
                    )
                _assert_expected_prior_source(
                    data_dict,
                    getattr(self, "expected_prior_source_fingerprint", None),
                    prior_spec=prior_spec,
                )
                if prior_spec.kind == "p23":
                    parent_p2_bundle = require_sha256(
                        data_dict.get(P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY),
                        field=P23_PARENT_P2_BUNDLE_FINGERPRINT_KEY,
                    )
                    record_p2_bundle = require_sha256(
                        data_dict.get(P2_BUNDLE_FINGERPRINT_KEY),
                        field=P2_BUNDLE_FINGERPRINT_KEY,
                    )
                    if parent_p2_bundle != record_p2_bundle:
                        raise ValueError(
                            "P23 parent P2 bundle does not match this dual-prior "
                            f"record: parent={parent_p2_bundle}, "
                            f"record={record_p2_bundle}."
                        )
                if not record_contract_already_validated:
                    actual_rme_fingerprint = fingerprint_fields(
                        atomicdata,
                        prior_spec.rme_fields,
                    )
                    assert_record_fingerprint(
                        data_dict,
                        field=prior_spec.rme_fingerprint_key,
                        actual=actual_rme_fingerprint,
                    )

        if uses_pre_physical_h0:
            node_physical_h0 = _expand_soc_uureal_compact(
                node_physical_h0,
                data_dict,
                field_name=AtomicDataDict.NODE_PHYSICAL_H0_KEY,
                keep_mask=soc_uureal_keep_mask,
            )
            edge_physical_h0 = _expand_soc_uureal_compact(
                edge_physical_h0,
                data_dict,
                field_name=AtomicDataDict.EDGE_PHYSICAL_H0_KEY,
                keep_mask=soc_uureal_keep_mask,
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

        if haar_u0 is not None:
            atomicdata[AtomicDataDict.HAAR_U0_KEY] = torch.as_tensor(
                haar_u0,
                dtype=torch.get_default_dtype(),
            )
        if haar_node_features is not None:
            haar_node_features = _expand_soc_uureal_compact(
                haar_node_features,
                data_dict,
                field_name=AtomicDataDict.HAAR_NODE_FEATURES_KEY,
                keep_mask=soc_uureal_keep_mask,
            )
            if haar_node_features.shape[0] != num_nodes:
                raise ValueError(
                    "Precomputed LMDB Haar node feature rows do not match the active graph: "
                    f"haar_node_features={tuple(haar_node_features.shape)}, "
                    f"num_nodes={num_nodes}."
                )
            atomicdata[AtomicDataDict.HAAR_NODE_FEATURES_KEY] = haar_node_features
        if haar_edge_features is not None:
            haar_edge_features = _expand_soc_uureal_compact(
                haar_edge_features,
                data_dict,
                field_name=AtomicDataDict.HAAR_EDGE_FEATURES_KEY,
                keep_mask=soc_uureal_keep_mask,
            )
            if haar_edge_features.shape[0] != num_edges:
                raise ValueError(
                    "Precomputed LMDB Haar edge feature rows do not match the active graph: "
                    f"haar_edge_features={tuple(haar_edge_features.shape)}, "
                    f"num_edges={num_edges}."
                )
            atomicdata[AtomicDataDict.HAAR_EDGE_FEATURES_KEY] = haar_edge_features

        # Optional AO-block targets/H0 produced by the blockwise NexTHAM
        # conversion path. Keep this side channel independent of the feature
        # path so existing RME training remains unchanged.
        load_prior_blocks = bool(
            self.get_prior
            and (
                getattr(self, "require_prior_blocks", False)
                or getattr(self, "audit_prior_representations", False)
            )
        )
        if self.get_Hamiltonian and getattr(self, "residual_hamiltonian", False):
            assert_residual_target_source_is_raw(data_dict)
        if self.get_prior and any(p2_blockwise_present) and not all(p2_blockwise_present):
            missing = [
                key for key, present in zip(p2_blockwise_keys, p2_blockwise_present)
                if not present
            ]
            raise ValueError(
                f"Prepacked {prior_spec.kind.upper()} prior is incomplete; "
                "missing block fields "
                f"{missing}."
            )
        if any(full_h_target_present) and not all(full_h_target_present):
            missing = [
                key for key, present in zip(
                    _PREPACKED_FULL_H_TARGET_KEYS, full_h_target_present
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
            *_PREPACKED_FULL_H_TARGET_KEYS,
        ):
            if blockwise_key in data_dict:
                atomicdata[blockwise_key] = torch.as_tensor(data_dict[blockwise_key])
        if load_prior_blocks:
            for blockwise_key in p2_blockwise_keys:
                if blockwise_key in data_dict:
                    atomicdata[blockwise_key] = torch.as_tensor(data_dict[blockwise_key])

        target_node_field = (
            AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY
            if getattr(self, "require_full_h_target", False)
            else AtomicDataDict.NODE_DELTA_HAMIL_BLOCKS_KEY
        )
        if (
            self.get_Hamiltonian
            and blocks is not False
            and blocks is not None
            and target_node_field not in atomicdata
        ):
            from dptb.data.interfaces.blockwise_tensor import attach_block_tensors, block_dict_to_ordered_tensors

            start_id = 0 if "0_0_0_0_0" in blocks else 1
            block_target_source = blocks
            if getattr(self, "residual_hamiltonian", False):
                block_target_source = build_residual_hamiltonian_target_blocks(
                    data_dict, blocks, h0_key=self.h0_key
                )
            target_blocks = block_dict_to_ordered_tensors(
                atomicdata,
                self.type_mapper,
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
                    if getattr(self, "require_full_h_target", False)
                    else "delta_hamil"
                ),
            )

        if (
            self.get_H0
            and h0_blocks is not None
            and AtomicDataDict.NODE_H0_BLOCKS_KEY not in atomicdata
        ):
            from dptb.data.interfaces.blockwise_tensor import attach_block_tensors, block_dict_to_ordered_tensors

            start_id = 0 if "0_0_0_0_0" in h0_blocks else 1
            target_h0_blocks = block_dict_to_ordered_tensors(
                atomicdata,
                self.type_mapper,
                h0_blocks,
                start_id=start_id,
                complete_edges=True,
                strict_complete_edges=False,
            )
            attach_block_tensors(atomicdata, target_h0_blocks, prefix="h0")

        if (
            load_prior_blocks
            and p2_blocks is not None
            and prior_spec.node_blocks_key not in atomicdata
        ):
            from dptb.data.interfaces.blockwise_tensor import attach_block_tensors, block_dict_to_ordered_tensors

            start_id = 0 if "0_0_0_0_0" in p2_blocks else 1
            target_p2_blocks = block_dict_to_ordered_tensors(
                atomicdata,
                self.type_mapper,
                p2_blocks,
                start_id=start_id,
                complete_edges=False,
                strict_complete_edges=False,
            )
            attach_block_tensors(atomicdata, target_p2_blocks, prefix=prior_spec.kind)

        if load_prior_blocks:
            required_p2_blocks = (
                prior_spec.node_blocks_key,
                prior_spec.edge_blocks_key,
            )
            available_p2_blocks = [key in atomicdata for key in required_p2_blocks]
            if getattr(self, "require_prior_blocks", False) and not all(
                key in atomicdata for key in p2_blockwise_keys
            ):
                missing = [key for key in p2_blockwise_keys if key not in atomicdata]
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
                validate_non_soc_p2_block_tensors(
                    p2_node_blocks,
                    p2_edge_blocks,
                    atomicdata[prior_spec.node_shape_key],
                    atomicdata[prior_spec.edge_shape_key],
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    data=atomicdata,
                    idp=self.type_mapper,
                    prior_kind=prior_spec.kind,
                    expensive_checks=not record_contract_already_validated,
                )

                if requires_stored_p2_graph and not record_contract_already_validated:
                    actual_block_fingerprint = fingerprint_fields(
                        atomicdata, p2_blockwise_keys
                    )
                    assert_record_fingerprint(
                        data_dict,
                        field=prior_spec.block_fingerprint_key,
                        actual=actual_block_fingerprint,
                    )
                    actual_bundle = fingerprint_text_fields(
                        {
                            BASIS_FINGERPRINT_KEY: basis_fingerprint,
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
                    assert_record_fingerprint(
                        data_dict,
                        field=prior_spec.bundle_fingerprint_key,
                        actual=actual_bundle,
                    )

                if (
                    getattr(self, "audit_prior_representations", False)
                    and not record_contract_already_validated
                ):
                    from dptb.data.interfaces.blockwise_tensor import (
                        block_mask_from_shapes,
                        feature_tensors_to_block_tensors,
                    )

                    reconstructed = feature_tensors_to_block_tensors(
                        atomicdata,
                        self.type_mapper,
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

        if getattr(self, "require_full_h_target", False):
            missing = [
                key for key in _PREPACKED_FULL_H_TARGET_KEYS if key not in atomicdata
            ]
            if missing:
                raise ValueError(f"Absolute Full-H target fields are missing: {missing}.")
            from dptb.data.interfaces.blockwise_tensor import validate_packed_non_soc_blocks

            if not record_contract_already_validated:
                validate_packed_non_soc_blocks(
                    atomicdata,
                    self.type_mapper,
                    atomicdata[AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCKS_KEY],
                    atomicdata[AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCKS_KEY],
                    atomicdata[AtomicDataDict.NODE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY],
                    atomicdata[AtomicDataDict.EDGE_FULL_HAMIL_TARGET_BLOCK_SHAPE_KEY],
                    label="absolute Full-H target",
                    require_symmetric_edges=True,
                )
            if (
                data_dict.get(TARGET_SOURCE_KEY) == "dedicated_full_h_blocks"
                and not record_contract_already_validated
            ):
                actual_target_fingerprint = fingerprint_fields(
                    atomicdata, _PREPACKED_FULL_H_TARGET_KEYS
                )
                assert_record_fingerprint(
                    data_dict,
                    field=FULL_H_TARGET_FINGERPRINT_KEY,
                    actual=actual_target_fingerprint,
                )

        # Mark only after every applicable graph/row/prior/target fingerprint
        # and every downstream structural check has succeeded.  A malformed or
        # tampered first read is never cached as trusted and will fail closed on
        # every retry.
        if requires_fingerprinted_graph and not record_contract_already_validated:
            validated_record_contracts[record_contract_key] = (
                canonical_stored_edge,
                canonical_stored_shift,
            )

        return atomicdata

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

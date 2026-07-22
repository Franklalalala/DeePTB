"""Route-agnostic block-topology contract for the block-space ODE subsystem.

Primary/derived key sets, snapshot/restore, cross-check, and output-only-key
filtering, extracted verbatim from ``dptb.nnops.flow.HamiltonianCFM`` (the
block-ode refactor's PR2).  Every function here is either fully stateless
(most of these were already ``@staticmethod``/``@classmethod`` on
``HamiltonianCFM``) or takes the owning ``HamiltonianCFM`` instance
explicitly as its first ``owner`` argument -- this module never imports
``dptb.nnops.flow`` (see the ``block_ode`` package's import-direction rule).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.interfaces.blockwise_tensor import BlockTensorResult


def _require_real_finite_tensor(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.is_complex():
        raise ValueError(f"{label} must be real for non-SOC block-space ODE.")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{label} contains NaN or Inf.")
    return tensor


def _block_primary_topology_keys() -> Tuple[str, ...]:
    """Primary graph metadata defining AO row identity and reverse pairs."""
    names = (
        ("EDGE_INDEX_KEY", "edge_index"),
        ("EDGE_CELL_SHIFT_KEY", "edge_cell_shift"),
        ("ATOMIC_NUMBERS_KEY", "atomic_numbers"),
        ("ATOM_TYPE_KEY", "atom_types"),
        ("POSITIONS_KEY", "pos"),
        ("BATCH_KEY", "batch"),
        ("PBC_KEY", "pbc"),
        ("EDGE_TYPE_KEY", "edge_type"),
        ("CELL_KEY", "cell"),
    )
    keys = [str(getattr(_keys, name, fallback)) for name, fallback in names]
    # Some legacy/raw dictionaries use the plural spelling even though the
    # canonical AtomicData key is singular.
    keys.append("edge_types")
    return tuple(dict.fromkeys(keys))


def _block_topology_keys() -> Tuple[str, ...]:
    """Graph metadata a model output may never redefine during block ODE."""
    names = (
        # Derived geometry is topology-dependent too.  If it was absent on
        # entry, discard a model-returned value so the next step recomputes
        # it from the immutable primary graph instead of trusting stale data.
        ("EDGE_VECTORS_KEY", "edge_vectors"),
        ("EDGE_LENGTH_KEY", "edge_lengths"),
    )
    keys = list(_block_primary_topology_keys())
    keys.extend(str(getattr(_keys, name, fallback)) for name, fallback in names)
    return tuple(dict.fromkeys(keys))


def _missing_keys(data: AtomicDataDict.Type, keys: Tuple[str, ...]) -> list[str]:
    return [key for key in keys if key not in data]


def _metadata_scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
        if not values or any(item != values[0] for item in values[1:]):
            raise ValueError(
                "uureal_block_ode batched metadata values must be nonempty and identical."
            )
        return values[0]
    if isinstance(value, (list, tuple)):
        values = [_metadata_scalar(item) for item in value]
        if not values or any(item != values[0] for item in values[1:]):
            raise ValueError(
                "uureal_block_ode batched metadata values must be nonempty and identical."
            )
        return values[0]
    try:
        import numpy as np
    except Exception:
        return value
    array = np.asarray(value)
    values = array.reshape(-1).tolist()
    if not values or any(item != values[0] for item in values[1:]):
        raise ValueError(
            "uureal_block_ode batched metadata values must be nonempty and identical."
        )
    return values[0]


def _attach_uureal_residual_state(data: AtomicDataDict.Type, state: BlockTensorResult) -> None:
    data[_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY] = state.node_blocks
    data[_keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY] = state.edge_blocks
    data[_keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY] = state.node_shapes
    data[_keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY] = state.edge_shapes


def _attach_spatial_residual_state(data: AtomicDataDict.Type, state: BlockTensorResult) -> None:
    data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY] = state.node_blocks
    data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY] = state.edge_blocks
    data[_keys.NODE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY] = state.node_shapes
    data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY] = state.edge_shapes


def _clone_sidecar_value(value: Any) -> Any:
    return value.clone() if torch.is_tensor(value) else copy.deepcopy(value)


def _snapshot_block_topology(data: AtomicDataDict.Type) -> Dict[str, Any]:
    return {
        key: _clone_sidecar_value(data[key])
        for key in _block_topology_keys()
        if key in data
    }


def _drop_block_authority_fields(owner, data: AtomicDataDict.Type) -> None:
    """Keep certified endpoint/H0 block side channels outside model I/O."""
    for key in (
        owner.node_block_target_key,
        owner.edge_block_target_key,
        owner.node_block_shape_key,
        owner.edge_block_shape_key,
        owner.node_h0_block_key,
        owner.edge_h0_block_key,
        owner.node_h0_block_shape_key,
        owner.edge_h0_block_shape_key,
    ):
        data.pop(key, None)


def _block_ode_output_only_keys(
    owner, extra_output_only_keys: Tuple[str, ...]
) -> Tuple[str, ...]:
    """Model outputs that must never be recycled as the next step's input."""
    return tuple(
        dict.fromkeys(
            (
                owner.node_output_key,
                owner.edge_output_key,
                *extra_output_only_keys,
            )
        )
    )


def _require_fresh_block_ode_outputs(
    owner,
    prediction: AtomicDataDict.Type,
    *,
    step: int,
) -> None:
    missing = _missing_keys(
        prediction, (owner.node_output_key, owner.edge_output_key)
    )
    if missing:
        raise ValueError(
            f"Block-space ODE step {step} is missing fresh model output keys={missing}."
        )


def _require_matching_block_topology(
    data_topology: Dict[str, Any],
    ref_topology: Dict[str, Any],
) -> None:
    for key in _block_primary_topology_keys():
        in_data = key in data_topology
        in_ref = key in ref_topology
        if in_data != in_ref:
            raise ValueError(
                "Block-space ODE data/ref topology mismatch: "
                f"key {key!r} is present in only one dictionary."
            )
        if not in_data:
            continue
        left = data_topology[key]
        right = ref_topology[key]
        if torch.is_tensor(left) or torch.is_tensor(right):
            try:
                left_t = torch.as_tensor(left).detach().cpu()
                right_t = torch.as_tensor(right).detach().cpu()
            except Exception as exc:
                raise ValueError(
                    f"Block-space ODE cannot compare data/ref topology key {key!r}."
                ) from exc
            equal = left_t.shape == right_t.shape and torch.equal(left_t, right_t)
        else:
            try:
                equal = bool(left == right)
            except Exception:
                equal = False
        if not equal:
            raise ValueError(
                "Block-space ODE data/ref topology mismatch at "
                f"key {key!r}; row-aligned H0/endpoint blocks cannot be mixed."
            )


def _restore_block_topology(
    data: AtomicDataDict.Type,
    snapshot: Dict[str, Any],
    *,
    clone_values: bool = False,
) -> None:
    for key in _block_topology_keys():
        if key in snapshot:
            value = snapshot[key]
            data[key] = _clone_sidecar_value(value) if clone_values else value
        else:
            data.pop(key, None)


def _max_abs(value: torch.Tensor, *, label: str = "residual") -> float:
    value = _require_real_finite_tensor(value, label=label)
    if value.numel() == 0:
        return 0.0
    return float(value.detach().abs().max().item())

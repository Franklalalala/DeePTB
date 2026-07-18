# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exact non-SOC codec and physical projector for block-state ODEs."""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch

from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_mask_from_shapes,
    canonical_block_tensors_to_feature_tensors,
    feature_tensors_to_block_tensors,
    infer_block_shapes,
    strict_reverse_edge_index,
)
from dptb.nn.hamiltonian import E3Hamiltonian


@dataclass(frozen=True)
class _BlockProjectorMetadata:
    """Validated immutable tensors shared by projector calls for one graph."""

    reverse: torch.Tensor
    pair_index: torch.Tensor
    expected_node_shapes: torch.Tensor
    expected_edge_shapes: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor


_PROJECTOR_CACHE_LIMIT = 16
_PROJECTOR_CACHE: "OrderedDict[Tuple[Any, ...], _BlockProjectorMetadata]" = OrderedDict()
_PROJECTOR_CACHE_LOCK = threading.Lock()
# Private capability proving a cadence-skipped inverse received a state from
# HamiltonianCFM's physical projector path, not an arbitrary external tensor.
_FLOW_PROJECTED_STATE_TOKEN = object()
_PROJECTOR_TOPOLOGY_KEYS = (
    "edge_index",
    "edge_cell_shift",
    "atomic_numbers",
    "atom_types",
    "pos",
    "batch",
    "pbc",
    "edge_type",
    "edge_types",
    "cell",
)


def _update_fingerprint(digest: "hashlib._Hash", label: str, value: Any) -> None:
    """Hash raw metadata without normalizing invalid values into valid ones."""

    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        tensor = tensor.cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    else:
        digest.update(type(value).__qualname__.encode("utf-8"))
        digest.update(repr(value).encode("utf-8"))
    digest.update(b"\xff")


def _projector_topology_fingerprint(data: Mapping[str, Any], idp: Any) -> str:
    """Content-address graph and mapper metadata used by projector guards."""

    digest = hashlib.sha256()
    for key in _PROJECTOR_TOPOLOGY_KEYS:
        if key in data:
            _update_fingerprint(digest, key, data[key])
        else:
            digest.update(f"missing:{key}\0".encode("ascii"))
    for label, value in (
        ("basis", getattr(idp, "basis", None)),
        ("chemical_symbol_to_type", getattr(idp, "chemical_symbol_to_type", None)),
        ("bond_to_type", getattr(idp, "bond_to_type", None)),
        ("norbs", getattr(idp, "norbs", None)),
        ("has_soc", getattr(idp, "has_soc", None)),
        ("reduced_matrix_element", getattr(idp, "reduced_matrix_element", None)),
    ):
        _update_fingerprint(digest, f"mapper:{label}", value)
    return digest.hexdigest()


def _clear_projector_metadata_cache() -> None:
    """Test hook for deterministic cache-hit and invalidation assertions."""

    with _PROJECTOR_CACHE_LOCK:
        _PROJECTOR_CACHE.clear()


def _projector_metadata(
    data: Mapping[str, Any],
    idp: Any,
    *,
    node: torch.Tensor,
    edge: torch.Tensor,
) -> _BlockProjectorMetadata:
    fingerprint = _projector_topology_fingerprint(data, idp)
    key = (
        fingerprint,
        str(node.device),
        str(edge.device),
        tuple(node.shape[-2:]),
        tuple(edge.shape[-2:]),
    )
    with _PROJECTOR_CACHE_LOCK:
        cached = _PROJECTOR_CACHE.get(key)
        if cached is not None:
            _PROJECTOR_CACHE.move_to_end(key)
            return cached

    # Cache only after every fail-closed topology/species guard succeeds.
    reverse = strict_reverse_edge_index(data, device=edge.device, idp=idp)
    expected_node, expected_edge = infer_block_shapes(data, idp)
    expected_node = expected_node.to(device=node.device, dtype=torch.long)
    expected_edge = expected_edge.to(device=edge.device, dtype=torch.long)
    rows = torch.arange(reverse.numel(), device=edge.device, dtype=torch.long)
    pair_index = rows[rows <= reverse]
    metadata = _BlockProjectorMetadata(
        reverse=reverse,
        pair_index=pair_index,
        expected_node_shapes=expected_node,
        expected_edge_shapes=expected_edge,
        node_mask=block_mask_from_shapes(expected_node, tuple(node.shape[-2:])),
        edge_mask=block_mask_from_shapes(expected_edge, tuple(edge.shape[-2:])),
    )
    with _PROJECTOR_CACHE_LOCK:
        existing = _PROJECTOR_CACHE.get(key)
        if existing is not None:
            _PROJECTOR_CACHE.move_to_end(key)
            return existing
        _PROJECTOR_CACHE[key] = metadata
        while len(_PROJECTOR_CACHE) > _PROJECTOR_CACHE_LIMIT:
            _PROJECTOR_CACHE.popitem(last=False)
    return metadata


def _require_same_shapes(a: BlockTensorResult, b: BlockTensorResult, *, label: str) -> None:
    for component in ("node", "edge"):
        a_blocks = getattr(a, f"{component}_blocks")
        b_blocks = getattr(b, f"{component}_blocks")
        a_shapes = getattr(a, f"{component}_shapes")
        b_shapes = getattr(b, f"{component}_shapes")
        if a_blocks is None or b_blocks is None or a_shapes is None or b_shapes is None:
            raise ValueError(f"{label} requires both node and edge block components.")
        if a_blocks.shape != b_blocks.shape or not torch.equal(
            a_shapes.to(device=b_shapes.device, dtype=torch.long), b_shapes.to(dtype=torch.long)
        ):
            raise ValueError(f"{label} node/edge block shapes do not match.")


def project_block_state(
    data: Mapping[str, Any],
    idp: Any,
    state: BlockTensorResult,
) -> BlockTensorResult:
    """Project a padded non-SOC state onto Hermitian graph constraints.

    The operation symmetrizes onsite blocks, averages every directed edge with
    its unique ``(j,i,-R)`` transpose partner, and zeros all species padding.
    Duplicate or missing reverse edges are rejected before any value is used.
    """
    node = state.node_blocks
    edge = state.edge_blocks
    if node is None or edge is None or state.node_shapes is None or state.edge_shapes is None:
        raise ValueError("A block ODE state requires node/edge blocks and shape metadata.")
    if node.ndim != 3 or edge.ndim != 3:
        raise ValueError("Block ODE state tensors must have shape [N,H,W].")
    if node.is_complex() or edge.is_complex():
        raise NotImplementedError("Block-state ODE v1 supports non-SOC real tensors only.")
    node_finite = torch.isfinite(node).all()
    edge_finite = torch.isfinite(edge).all()
    finite = (
        node_finite.logical_and(edge_finite.to(device=node.device))
        if node.device != edge.device
        else node_finite.logical_and(edge_finite)
    )
    if not bool(finite.item()):
        raise ValueError("Block ODE state contains NaN or Inf.")

    # Raw graph indices/shifts/types are validated before any state metadata is
    # trusted.  Successful immutable graph metadata is content-addressed so the
    # six projector calls in one block-ODE iteration share reverse pairs and
    # masks without allowing a different (or mutated) graph to hit the cache.
    metadata = _projector_metadata(data, idp, node=node, edge=edge)
    for label, shapes in (
        ("node", state.node_shapes),
        ("edge", state.edge_shapes),
    ):
        if shapes.dtype == torch.bool or shapes.is_complex() or not bool(
            torch.isfinite(shapes).all().item()
        ):
            raise ValueError(f"{label}_shapes must contain finite integers.")
        if shapes.is_floating_point() and shapes.numel() and bool(
            ((shapes - shapes.round()).abs().max() > 0).item()
        ):
            raise ValueError(f"{label}_shapes must contain integers.")
    node_shapes = state.node_shapes.to(device=node.device, dtype=torch.long)
    edge_shapes = state.edge_shapes.to(device=edge.device, dtype=torch.long)
    if not torch.equal(node_shapes, metadata.expected_node_shapes):
        raise ValueError("node_shapes disagrees with mapper/species metadata.")
    if not torch.equal(edge_shapes, metadata.expected_edge_shapes):
        raise ValueError("edge_shapes disagrees with mapper/species metadata.")
    if node.shape[0] != node_shapes.shape[0] or edge.shape[0] != edge_shapes.shape[0]:
        raise ValueError("Block row count disagrees with shape metadata.")
    if node_shapes.numel() and (
        bool((node_shapes <= 0).any().item())
        or bool((node_shapes[:, 0] > node.shape[-2]).any().item())
        or bool((node_shapes[:, 1] > node.shape[-1]).any().item())
    ):
        raise ValueError("node_shapes is non-positive or exceeds the node canvas.")
    if edge_shapes.numel() and (
        bool((edge_shapes <= 0).any().item())
        or bool((edge_shapes[:, 0] > edge.shape[-2]).any().item())
        or bool((edge_shapes[:, 1] > edge.shape[-1]).any().item())
    ):
        raise ValueError("edge_shapes is non-positive or exceeds the edge canvas.")
    if node.shape[-2] != node.shape[-1]:
        raise ValueError("Onsite padded canvas must be square.")
    if edge.shape[-2] != edge.shape[-1]:
        raise ValueError("Reverse-edge projection requires a square padded canvas.")

    node_projected = 0.5 * (node + node.transpose(-1, -2))
    node_projected = node_projected * metadata.node_mask
    # Multiplication is the fast padding path; normalize signed padding zero so
    # the byte representation remains identical to the former torch.where path.
    node_projected = node_projected.masked_fill(~metadata.node_mask, 0.0)

    pair_index = metadata.pair_index
    pair_mates = metadata.reverse.index_select(0, pair_index)
    averaged = 0.5 * (
        edge.index_select(0, pair_index)
        + edge.index_select(0, pair_mates).transpose(-1, -2)
    )
    direct = averaged * metadata.edge_mask.index_select(0, pair_index)
    opposite = averaged.transpose(-1, -2) * metadata.edge_mask.index_select(
        0, pair_mates
    )
    edge_projected = torch.zeros_like(edge)
    edge_projected.index_copy_(0, pair_index, direct)
    edge_projected.index_copy_(0, pair_mates, opposite)
    edge_projected = edge_projected.masked_fill(~metadata.edge_mask, 0.0)

    return BlockTensorResult(
        node_blocks=node_projected,
        edge_blocks=edge_projected,
        node_shapes=node_shapes,
        edge_shapes=edge_shapes,
    )


class BlockStateCodec:
    """Compose canonical block gather with the exact inverse CG transform."""

    def __init__(
        self,
        idp: Any,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device("cpu"),
        inverse_mode: str = "strict",
        atol: Optional[float] = None,
        target_semantics: str = "absolute_full_h",
    ) -> None:
        if bool(getattr(idp, "has_soc", False)):
            raise NotImplementedError("BlockStateCodec v1 does not support SOC mappers.")
        if inverse_mode not in {"strict", "project"}:
            raise ValueError("inverse_mode must be 'strict' or 'project'.")
        if target_semantics not in {"absolute_full_h", "residual_dh"}:
            raise ValueError(
                "target_semantics must be 'absolute_full_h' or 'residual_dh'."
            )
        self.idp = idp
        self.dtype = dtype
        self.device = torch.device(device)
        self.inverse_mode = inverse_mode
        self.atol = float(
            atol if atol is not None else (1e-10 if dtype == torch.float64 else 2e-5)
        )
        if not math.isfinite(self.atol) or self.atol < 0:
            raise ValueError("BlockStateCodec atol must be finite and non-negative.")
        if inverse_mode == "strict":
            if dtype not in {torch.float32, torch.float64}:
                raise TypeError(
                    "Strict BlockStateCodec requires float32 or float64 for a "
                    "certified inverse tolerance."
                )
            maximum_atol = 1.0e-10 if dtype == torch.float64 else 2.0e-5
            if self.atol > maximum_atol:
                raise ValueError(
                    "Strict BlockStateCodec atol exceeds the certified dtype "
                    f"maximum {maximum_atol:.6g}: got {self.atol:.6g}."
                )
        self.target_semantics = target_semantics
        self._expand = E3Hamiltonian(
            idp=idp,
            decompose=False,
            dtype=dtype,
            device=self.device,
            soc=False,
        )
        self._contract = E3Hamiltonian(
            idp=idp,
            decompose=True,
            enable_inverse_cg=True,
            dtype=dtype,
            device=self.device,
            soc=False,
        )

    @staticmethod
    def _working_data(
        data: Mapping[str, Any], node: torch.Tensor, edge: torch.Tensor
    ) -> dict[str, Any]:
        # E3Hamiltonian is TorchScript-typed as Dict[str, Tensor].  Real
        # trainer batches also carry PyG collation metadata such as
        # ``__slices__`` and ``__data_class__``; keep those in the outer
        # state, but do not feed them through the scripted CG transform.
        work = {key: value for key, value in data.items() if torch.is_tensor(value)}
        work["node_features"] = node.clone()
        work["edge_features"] = edge.clone()
        return work

    def rme_to_blocks(
        self,
        data: Mapping[str, Any],
        node_rme: torch.Tensor,
        edge_rme: torch.Tensor,
        *,
        project: bool = False,
    ) -> BlockTensorResult:
        """Expand coupled RME to product features, then pack species blocks."""
        work = self._working_data(data, node_rme, edge_rme)
        work = self._expand(work)
        packed = feature_tensors_to_block_tensors(
            work,
            self.idp,
            node_features=work["node_features"],
            edge_features=work["edge_features"],
            symmetrize_onsite=True,
            complete_edges=True,
            strict_complete_edges=True,
        )
        return project_block_state(data, self.idp, packed) if project else packed

    def blocks_to_rme(
        self,
        data: Mapping[str, Any],
        state: BlockTensorResult,
        *,
        certify_image: bool = True,
        _construction_token: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Canonical-gather product features, then apply the inverse CG."""
        if not certify_image and _construction_token is not _FLOW_PROJECTED_STATE_TOKEN:
            raise RuntimeError(
                "Skipping strict image certification requires the internal "
                "projected-state construction token."
            )
        metadata = None
        if _construction_token is _FLOW_PROJECTED_STATE_TOKEN:
            metadata = _projector_metadata(
                data,
                self.idp,
                node=state.node_blocks,
                edge=state.edge_blocks,
            )
        gathered = canonical_block_tensors_to_feature_tensors(
            data,
            self.idp,
            node_blocks=state.node_blocks,
            edge_blocks=state.edge_blocks,
            node_shapes=state.node_shapes,
            edge_shapes=state.edge_shapes,
            mode=self.inverse_mode,
            atol=self.atol,
            certify_image=certify_image,
            _expected_node_shapes=(
                None if metadata is None else metadata.expected_node_shapes
            ),
            _expected_edge_shapes=(
                None if metadata is None else metadata.expected_edge_shapes
            ),
            _reverse_validated=metadata is not None,
        )
        if gathered.node_features is None or gathered.edge_features is None:
            raise ValueError("BlockStateCodec requires both node and edge components.")
        work = self._working_data(data, gathered.node_features, gathered.edge_features)
        work = self._contract(work)
        return work["node_features"], work["edge_features"]

    def endpoint_to_full(
        self,
        endpoint: BlockTensorResult,
        h0: BlockTensorResult,
    ) -> BlockTensorResult:
        """Adapt a configured endpoint to one full-H block state exactly once."""
        _require_same_shapes(endpoint, h0, label="endpoint adaptation")
        if self.target_semantics == "absolute_full_h":
            full = endpoint
        else:
            full = BlockTensorResult(
                node_blocks=h0.node_blocks + endpoint.node_blocks,
                edge_blocks=h0.edge_blocks + endpoint.edge_blocks,
                node_shapes=endpoint.node_shapes,
                edge_shapes=endpoint.edge_shapes,
            )
        return full

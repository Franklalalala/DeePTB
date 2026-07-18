# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exact non-SOC codec and physical projector for block-state ODEs."""

from __future__ import annotations

import math
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
    if not bool(torch.isfinite(node).all().item()) or not bool(torch.isfinite(edge).all().item()):
        raise ValueError("Block ODE state contains NaN or Inf.")

    # Validate raw graph indices/shifts/types before any helper casts them or
    # uses Python indexing to infer species-dependent shapes.
    rev = strict_reverse_edge_index(data, device=edge.device, idp=idp)
    expected_node, expected_edge = infer_block_shapes(data, idp)
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
    if not torch.equal(node_shapes, expected_node.to(device=node.device)):
        raise ValueError("node_shapes disagrees with mapper/species metadata.")
    if not torch.equal(edge_shapes, expected_edge.to(device=edge.device)):
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

    node_mask = block_mask_from_shapes(node_shapes, tuple(node.shape[-2:]))
    node_projected = 0.5 * (node + node.transpose(-1, -2))
    node_projected = torch.where(node_mask, node_projected, torch.zeros_like(node_projected))

    edge_mask = block_mask_from_shapes(edge_shapes, tuple(edge.shape[-2:]))
    edge_projected = torch.zeros_like(edge)
    visited = torch.zeros((edge.shape[0],), dtype=torch.bool, device=edge.device)
    for e in range(edge.shape[0]):
        if bool(visited[e].item()):
            continue
        mate = int(rev[e].item())
        averaged = 0.5 * (edge[e] + edge[mate].transpose(-1, -2))
        edge_projected[e] = torch.where(
            edge_mask[e], averaged, torch.zeros_like(averaged)
        )
        edge_projected[mate] = torch.where(
            edge_mask[mate], averaged.transpose(-1, -2), torch.zeros_like(averaged)
        )
        visited[e] = True
        visited[mate] = True

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Canonical-gather product features, then apply the inverse CG."""
        gathered = canonical_block_tensors_to_feature_tensors(
            data,
            self.idp,
            node_blocks=state.node_blocks,
            edge_blocks=state.edge_blocks,
            node_shapes=state.node_shapes,
            edge_shapes=state.edge_shapes,
            mode=self.inverse_mode,
            atol=self.atol,
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

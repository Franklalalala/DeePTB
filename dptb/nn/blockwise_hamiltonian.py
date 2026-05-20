# SPDX-License-Identifier: LGPL-3.0-or-later
"""Block-wise Hamiltonian decoder wrapper.

``BlockwiseE3Hamiltonian`` keeps DeePTB's existing ``E3Hamiltonian`` stage for
now: equivariant/RME outputs are decoded into DeePTB Hamiltonian feature tensors,
then materialized into padded AO blocks.  This is correctness-first and enables
block-level loss without changing the upstream equivariant head.  It is not a
block-native head and therefore is not expected to be speed-positive by itself.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from dptb.data import AtomicDataDict
except Exception:  # pragma: no cover
    class AtomicDataDict:
        NODE_FEATURES_KEY = "node_features"
        EDGE_FEATURES_KEY = "edge_features"

from dptb.data.interfaces.blockwise_tensor import (
    EDGE_H0_BLOCKS_KEY,
    EDGE_PRED_HAMIL_BLOCKS_KEY,
    BlockTensorResult,
    NODE_H0_BLOCKS_KEY,
    NODE_PRED_HAMIL_BLOCKS_KEY,
    attach_prediction_block_tensors,
    block_mask_from_shapes,
    complete_edge_blocks_from_reverse,
    edge_feature_slices,
    edge_index_from_data,
    ensure_non_soc_mapper,
    infer_block_shapes,
    mapper_max_norb,
    onsite_feature_slices,
    reverse_edge_index,
    symbols_from_data,
    feature_tensors_to_block_tensors,
)
from dptb.nn.hamiltonian import E3Hamiltonian


def _as_pad_shape(shape: Optional[Tuple[int, int]], idp: Any) -> Tuple[int, int]:
    if shape is None:
        dim = mapper_max_norb(idp)
        return dim, dim
    return int(shape[0]), int(shape[1])


class DirectAOBlockDecoder(nn.Module):
    """Predict padded AO blocks directly from node/edge feature tensors.

    This decoder can be placed after an existing E3/RME stage when the expensive
    part to remove is deterministic feature-to-block materialization rather than
    the E3 prediction stage itself.
    """

    def __init__(
        self,
        *,
        idp: Any,
        node_in_features: int,
        edge_in_features: int,
        node_field: str = getattr(AtomicDataDict, "NODE_FEATURES_KEY", "node_features"),
        edge_field: str = getattr(AtomicDataDict, "EDGE_FEATURES_KEY", "edge_features"),
        output_node_field: str = NODE_PRED_HAMIL_BLOCKS_KEY,
        output_edge_field: str = EDGE_PRED_HAMIL_BLOCKS_KEY,
        output_node_shape_field: str = "node_hamil_block_shape",
        output_edge_shape_field: str = "edge_hamil_block_shape",
        node_pad_shape: Optional[Tuple[int, int]] = None,
        edge_pad_shape: Optional[Tuple[int, int]] = None,
        symmetrize_onsite: bool = True,
        complete_edges: bool = True,
        strict_complete_edges: bool = False,
        add_h0: bool = False,
        full_output_node_field: str = "node_full_hamil_blocks",
        full_output_edge_field: str = "edge_full_hamil_blocks",
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if isinstance(device, str):
            device = torch.device(device)
        self.idp = idp
        self.node_field = node_field
        self.edge_field = edge_field
        self.output_node_field = output_node_field
        self.output_edge_field = output_edge_field
        self.output_node_shape_field = output_node_shape_field
        self.output_edge_shape_field = output_edge_shape_field
        self.node_pad_shape = _as_pad_shape(node_pad_shape, idp)
        self.edge_pad_shape = _as_pad_shape(edge_pad_shape, idp)
        self.symmetrize_onsite = bool(symmetrize_onsite)
        self.complete_edges = bool(complete_edges)
        self.strict_complete_edges = bool(strict_complete_edges)
        self.add_h0 = bool(add_h0)
        self.full_output_node_field = full_output_node_field
        self.full_output_edge_field = full_output_edge_field
        self.node_decoder = nn.Linear(
            int(node_in_features),
            self.node_pad_shape[0] * self.node_pad_shape[1],
            dtype=dtype,
            device=device,
        )
        self.edge_decoder = nn.Linear(
            int(edge_in_features),
            self.edge_pad_shape[0] * self.edge_pad_shape[1],
            dtype=dtype,
            device=device,
        )
        nn.init.normal_(self.node_decoder.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.node_decoder.bias)
        nn.init.normal_(self.edge_decoder.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.edge_decoder.bias)

    @staticmethod
    def _mask_to_valid_shape(
        blocks: Optional[torch.Tensor],
        shapes: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Zero padded AO entries outside each block's valid atom/pair shape."""
        if blocks is None or shapes is None or blocks.numel() == 0:
            return blocks
        mask = block_mask_from_shapes(
            shapes.to(device=blocks.device, dtype=torch.long),
            tuple(blocks.shape[-2:]),
        )
        mask_dtype = blocks.real.dtype if blocks.is_complex() else blocks.dtype
        return blocks * mask.to(dtype=mask_dtype, device=blocks.device)

    def _symmetrize_reverse_edges(self, data: Dict[str, Any], edge_blocks: torch.Tensor) -> torch.Tensor:
        # Keep control-flow checks on CPU and move only index/mask tensors needed for GPU ops.
        rev_cpu = reverse_edge_index(data)
        if rev_cpu.numel() == 0:
            return edge_blocks
        has_rev_cpu = rev_cpu >= 0
        if self.strict_complete_edges and bool((~has_rev_cpu).any().item()):
            missing = torch.nonzero(~has_rev_cpu, as_tuple=False).flatten()
            preview = missing[:8].detach().cpu().tolist()
            raise RuntimeError(
                "Direct AO block edge symmetrization requires reverse directed edges; "
                f"missing={int(missing.numel())}, first indices={preview}."
            )
        if not bool(has_rev_cpu.any().item()):
            return edge_blocks
        safe_rev = rev_cpu.clamp_min(0).to(device=edge_blocks.device)
        has_rev = has_rev_cpu.to(device=edge_blocks.device)
        reverse_blocks = edge_blocks.index_select(0, safe_rev).transpose(-1, -2)
        if reverse_blocks.is_complex():
            reverse_blocks = reverse_blocks.conj()
        averaged = 0.5 * (edge_blocks + reverse_blocks)
        return torch.where(has_rev.view(-1, 1, 1), averaged, edge_blocks)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        node_features = data.get(self.node_field, None)
        edge_features = data.get(self.edge_field, None)
        if node_features is not None and int(node_features.shape[-1]) != int(self.node_decoder.in_features):
            raise RuntimeError(
                f"Direct AO node decoder expected {self.node_decoder.in_features} input features, "
                f"got {int(node_features.shape[-1])}."
            )
        if edge_features is not None and int(edge_features.shape[-1]) != int(self.edge_decoder.in_features):
            raise RuntimeError(
                f"Direct AO edge decoder expected {self.edge_decoder.in_features} input features, "
                f"got {int(edge_features.shape[-1])}."
            )
        node_shapes, edge_shapes = infer_block_shapes(
            data,
            self.idp,
            device=node_features.device if node_features is not None else (
                edge_features.device if edge_features is not None else None
            ),
        )
        node_blocks = None
        edge_blocks = None

        if node_features is not None:
            node_blocks = self.node_decoder(node_features).reshape(
                node_features.shape[0],
                self.node_pad_shape[0],
                self.node_pad_shape[1],
            )
            if self.symmetrize_onsite:
                node_blocks = 0.5 * (node_blocks + node_blocks.transpose(-1, -2))
            node_blocks = self._mask_to_valid_shape(node_blocks, node_shapes)

        if edge_features is not None:
            edge_blocks = self.edge_decoder(edge_features).reshape(
                edge_features.shape[0],
                self.edge_pad_shape[0],
                self.edge_pad_shape[1],
            )
            if self.complete_edges:
                edge_blocks = self._symmetrize_reverse_edges(data, edge_blocks)
            edge_blocks = self._mask_to_valid_shape(edge_blocks, edge_shapes)

        packed = BlockTensorResult(node_blocks, edge_blocks, node_shapes, edge_shapes)

        attach_prediction_block_tensors(
            data,
            packed,
            node_key=self.output_node_field,
            edge_key=self.output_edge_field,
            node_shape_key=self.output_node_shape_field,
            edge_shape_key=self.output_edge_shape_field,
        )
        if self.add_h0:
            if packed.node_blocks is not None and NODE_H0_BLOCKS_KEY in data:
                data[self.full_output_node_field] = data[NODE_H0_BLOCKS_KEY].to(
                    device=packed.node_blocks.device,
                    dtype=packed.node_blocks.dtype,
                ) + packed.node_blocks
            if packed.edge_blocks is not None and EDGE_H0_BLOCKS_KEY in data:
                data[self.full_output_edge_field] = data[EDGE_H0_BLOCKS_KEY].to(
                    device=packed.edge_blocks.device,
                    dtype=packed.edge_blocks.dtype,
                ) + packed.edge_blocks
        return data


class DirectBlockwiseE3Hamiltonian(nn.Module):
    """Keep E3 feature prediction, then decode AO blocks with a learned head."""

    def __init__(
        self,
        basis: Optional[Dict[str, Union[str, list]]] = None,
        idp: Any = None,
        *,
        decompose: bool = False,
        edge_field: str = getattr(AtomicDataDict, "EDGE_FEATURES_KEY", "edge_features"),
        node_field: str = getattr(AtomicDataDict, "NODE_FEATURES_KEY", "node_features"),
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ) -> None:
        super().__init__()
        block_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key in {
                "output_node_field",
                "output_edge_field",
                "output_node_shape_field",
                "output_edge_shape_field",
                "node_pad_shape",
                "edge_pad_shape",
                "symmetrize_onsite",
                "complete_edges",
                "strict_complete_edges",
                "add_h0",
                "full_output_node_field",
                "full_output_edge_field",
            }
        }
        self.e3 = E3Hamiltonian(
            basis=basis,
            idp=idp,
            decompose=decompose,
            edge_field=edge_field,
            node_field=node_field,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        self.idp = self.e3.idp
        self.decoder = DirectAOBlockDecoder(
            idp=self.idp,
            node_in_features=self.idp.reduced_matrix_element,
            edge_in_features=self.idp.reduced_matrix_element,
            node_field=node_field,
            edge_field=edge_field,
            dtype=dtype,
            device=device,
            **block_kwargs,
        )

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = self.e3(data)
        return self.decoder(data)


def _feature_block_indices(
    entries,
    *,
    pad_shape: Tuple[int, int],
    mirror: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic feature-to-AO scatter indices for one atom/pair type."""
    h_pad, w_pad = int(pad_shape[0]), int(pad_shape[1])
    src_parts = []
    dst_parts = []
    for row, col, feat in entries:
        row_ids = torch.arange(int(row.start), int(row.stop), dtype=torch.long)
        col_ids = torch.arange(int(col.start), int(col.stop), dtype=torch.long)
        rr, cc = torch.meshgrid(row_ids, col_ids, indexing="ij")
        dst_parts.append((rr * w_pad + cc).reshape(-1))
        src_parts.append(torch.arange(int(feat.start), int(feat.stop), dtype=torch.long))
        if mirror and (row.start != col.start or row.stop != col.stop):
            dst_parts.append((cc * w_pad + rr).reshape(-1))
            src_parts.append(torch.arange(int(feat.start), int(feat.stop), dtype=torch.long))
    if not src_parts:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.cat(src_parts, dim=0), torch.cat(dst_parts, dim=0)


class NexTHamAOBlockDecoder(nn.Module):
    """Deterministically scatter structured Hamiltonian features into AO blocks.

    This mirrors the final NexTHAM ``net_out -> H`` philosophy: all learnable
    prediction stays upstream in the equivariant/RME head, while the final AO
    block construction is fixed by orbital angular-momentum bookkeeping.
    """

    def __init__(
        self,
        *,
        idp: Any,
        node_field: str = getattr(AtomicDataDict, "NODE_FEATURES_KEY", "node_features"),
        edge_field: str = getattr(AtomicDataDict, "EDGE_FEATURES_KEY", "edge_features"),
        output_node_field: str = NODE_PRED_HAMIL_BLOCKS_KEY,
        output_edge_field: str = EDGE_PRED_HAMIL_BLOCKS_KEY,
        output_node_shape_field: str = "node_hamil_block_shape",
        output_edge_shape_field: str = "edge_hamil_block_shape",
        node_pad_shape: Optional[Tuple[int, int]] = None,
        edge_pad_shape: Optional[Tuple[int, int]] = None,
        symmetrize_onsite: bool = True,
        complete_edges: bool = True,
        strict_complete_edges: bool = False,
        add_h0: bool = False,
        full_output_node_field: str = "node_full_hamil_blocks",
        full_output_edge_field: str = "edge_full_hamil_blocks",
    ) -> None:
        super().__init__()
        self.idp = ensure_non_soc_mapper(idp)
        self.node_field = node_field
        self.edge_field = edge_field
        self.output_node_field = output_node_field
        self.output_edge_field = output_edge_field
        self.output_node_shape_field = output_node_shape_field
        self.output_edge_shape_field = output_edge_shape_field
        self.node_pad_shape = _as_pad_shape(node_pad_shape, self.idp)
        self.edge_pad_shape = _as_pad_shape(edge_pad_shape, self.idp)
        self.symmetrize_onsite = bool(symmetrize_onsite)
        self.complete_edges = bool(complete_edges)
        self.strict_complete_edges = bool(strict_complete_edges)
        self.add_h0 = bool(add_h0)
        self.full_output_node_field = full_output_node_field
        self.full_output_edge_field = full_output_edge_field
        self.feature_width = int(self.idp.reduced_matrix_element)
        self._node_index_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._edge_index_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _node_indices(self, symbol: str) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._node_index_cache.get(symbol)
        if cached is None:
            cached = _feature_block_indices(
                list(onsite_feature_slices(self.idp, symbol)),
                pad_shape=self.node_pad_shape,
                mirror=self.symmetrize_onsite,
            )
            self._node_index_cache[symbol] = cached
        return cached

    def _edge_indices(self, sym_i: str, sym_j: str) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (sym_i, sym_j)
        cached = self._edge_index_cache.get(key)
        if cached is None:
            cached = _feature_block_indices(
                list(edge_feature_slices(self.idp, sym_i, sym_j)),
                pad_shape=self.edge_pad_shape,
                mirror=False,
            )
            self._edge_index_cache[key] = cached
        return cached

    @staticmethod
    def _scatter_features_to_blocks(
        features: torch.Tensor,
        positions: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        pad_shape: Tuple[int, int],
    ) -> torch.Tensor:
        selected = features.index_select(0, positions)
        flat = torch.zeros(
            (positions.numel(), int(pad_shape[0]) * int(pad_shape[1])),
            dtype=features.dtype,
            device=features.device,
        )
        if src_idx.numel() > 0:
            src_idx = src_idx.to(device=features.device)
            dst_idx = dst_idx.to(device=features.device)
            flat[:, dst_idx] = selected.index_select(1, src_idx)
        return flat.reshape(positions.numel(), int(pad_shape[0]), int(pad_shape[1]))

    @staticmethod
    def _check_feature_width(features: Optional[torch.Tensor], expected: int, name: str) -> None:
        if features is not None and int(features.shape[-1]) != int(expected):
            raise RuntimeError(
                f"NexTHAM-style AO block decoder expected {expected} {name} features, "
                f"got {int(features.shape[-1])}."
            )

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        node_features = data.get(self.node_field, None)
        edge_features = data.get(self.edge_field, None)
        self._check_feature_width(node_features, self.feature_width, "node")
        self._check_feature_width(edge_features, self.feature_width, "edge")

        device = None
        if node_features is not None:
            device = node_features.device
        elif edge_features is not None:
            device = edge_features.device
        node_shapes, edge_shapes = infer_block_shapes(data, self.idp, device=device)
        symbols = symbols_from_data(data, self.idp)

        node_blocks = None
        if node_features is not None:
            node_blocks = torch.zeros(
                (len(symbols), self.node_pad_shape[0], self.node_pad_shape[1]),
                dtype=node_features.dtype,
                device=node_features.device,
            )
            by_symbol: Dict[str, list] = {}
            for i, symbol in enumerate(symbols):
                by_symbol.setdefault(symbol, []).append(i)
            for symbol, positions in by_symbol.items():
                pos = torch.as_tensor(positions, dtype=torch.long, device=node_features.device)
                src_idx, dst_idx = self._node_indices(symbol)
                sub = self._scatter_features_to_blocks(
                    node_features,
                    pos,
                    src_idx,
                    dst_idx,
                    self.node_pad_shape,
                )
                node_blocks.index_copy_(0, pos, sub)

        edge_blocks = None
        edge_direct_mask = None
        if edge_features is not None:
            edge_index = edge_index_from_data(data).detach().cpu()
            n_edges = int(edge_index.shape[1])
            edge_blocks = torch.zeros(
                (n_edges, self.edge_pad_shape[0], self.edge_pad_shape[1]),
                dtype=edge_features.dtype,
                device=edge_features.device,
            )
            edge_direct_mask = torch.zeros(
                (n_edges, self.edge_pad_shape[0] * self.edge_pad_shape[1]),
                dtype=torch.bool,
                device=edge_features.device,
            )
            by_pair: Dict[Tuple[str, str], list] = {}
            for e, (u, v) in enumerate(edge_index.T.tolist()):
                pair = (symbols[int(u)], symbols[int(v)])
                by_pair.setdefault(pair, []).append(e)
            for (sym_i, sym_j), positions in by_pair.items():
                pos = torch.as_tensor(positions, dtype=torch.long, device=edge_features.device)
                src_idx, dst_idx = self._edge_indices(sym_i, sym_j)
                sub = self._scatter_features_to_blocks(
                    edge_features,
                    pos,
                    src_idx,
                    dst_idx,
                    self.edge_pad_shape,
                )
                edge_blocks.index_copy_(0, pos, sub)
                if dst_idx.numel() > 0:
                    dst_dev = dst_idx.to(device=edge_features.device)
                    edge_direct_mask[pos[:, None], dst_dev[None, :]] = True
            edge_direct_mask = edge_direct_mask.reshape(
                n_edges,
                self.edge_pad_shape[0],
                self.edge_pad_shape[1],
            )
            if self.complete_edges:
                edge_blocks, _, _ = complete_edge_blocks_from_reverse(
                    data,
                    self.idp,
                    edge_blocks,
                    direct_mask=edge_direct_mask,
                    strict=self.strict_complete_edges,
                )

        packed = BlockTensorResult(node_blocks, edge_blocks, node_shapes, edge_shapes)
        attach_prediction_block_tensors(
            data,
            packed,
            node_key=self.output_node_field,
            edge_key=self.output_edge_field,
            node_shape_key=self.output_node_shape_field,
            edge_shape_key=self.output_edge_shape_field,
        )
        if self.add_h0:
            if packed.node_blocks is not None and NODE_H0_BLOCKS_KEY in data:
                data[self.full_output_node_field] = data[NODE_H0_BLOCKS_KEY].to(
                    device=packed.node_blocks.device,
                    dtype=packed.node_blocks.dtype,
                ) + packed.node_blocks
            if packed.edge_blocks is not None and EDGE_H0_BLOCKS_KEY in data:
                data[self.full_output_edge_field] = data[EDGE_H0_BLOCKS_KEY].to(
                    device=packed.edge_blocks.device,
                    dtype=packed.edge_blocks.dtype,
                ) + packed.edge_blocks
        return data


class NexTHamBlockwiseE3Hamiltonian(nn.Module):
    """Keep E3/RME prediction and use a parameter-free NexTHAM-style block map."""

    def __init__(
        self,
        basis: Optional[Dict[str, Union[str, list]]] = None,
        idp: Any = None,
        *,
        decompose: bool = False,
        edge_field: str = getattr(AtomicDataDict, "EDGE_FEATURES_KEY", "edge_features"),
        node_field: str = getattr(AtomicDataDict, "NODE_FEATURES_KEY", "node_features"),
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ) -> None:
        super().__init__()
        block_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key in {
                "output_node_field",
                "output_edge_field",
                "output_node_shape_field",
                "output_edge_shape_field",
                "node_pad_shape",
                "edge_pad_shape",
                "symmetrize_onsite",
                "complete_edges",
                "strict_complete_edges",
                "add_h0",
                "full_output_node_field",
                "full_output_edge_field",
            }
        }
        self.e3 = E3Hamiltonian(
            basis=basis,
            idp=idp,
            decompose=decompose,
            edge_field=edge_field,
            node_field=node_field,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        self.idp = self.e3.idp
        self.decoder = NexTHamAOBlockDecoder(
            idp=self.idp,
            node_field=node_field,
            edge_field=edge_field,
            **block_kwargs,
        )

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = self.e3(data)
        return self.decoder(data)


class BlockwiseE3Hamiltonian(nn.Module):
    """Decode equivariant Hamiltonian features and expose AO-block predictions.

    Parameters are forwarded to ``E3Hamiltonian`` except for the block-specific
    output controls below.  Gradients from block loss flow through the
    differentiable feature-to-block materialization back to the original
    feature/RME head.
    """

    def __init__(
        self,
        basis: Optional[Dict[str, Union[str, list]]] = None,
        idp: Any = None,
        *,
        decompose: bool = False,
        edge_field: str = getattr(AtomicDataDict, "EDGE_FEATURES_KEY", "edge_features"),
        node_field: str = getattr(AtomicDataDict, "NODE_FEATURES_KEY", "node_features"),
        output_node_field: str = NODE_PRED_HAMIL_BLOCKS_KEY,
        output_edge_field: str = EDGE_PRED_HAMIL_BLOCKS_KEY,
        output_node_shape_field: str = "node_hamil_block_shape",
        output_edge_shape_field: str = "edge_hamil_block_shape",
        node_pad_shape: Optional[Tuple[int, int]] = None,
        edge_pad_shape: Optional[Tuple[int, int]] = None,
        symmetrize_onsite: bool = True,
        complete_edges: bool = True,
        strict_complete_edges: bool = False,
        add_h0: bool = False,
        full_output_node_field: str = "node_full_hamil_blocks",
        full_output_edge_field: str = "edge_full_hamil_blocks",
        **kwargs,
    ) -> None:
        super().__init__()
        self.e3 = E3Hamiltonian(
            basis=basis,
            idp=idp,
            decompose=decompose,
            edge_field=edge_field,
            node_field=node_field,
            **kwargs,
        )
        self.idp = self.e3.idp
        self.node_field = node_field
        self.edge_field = edge_field
        self.output_node_field = output_node_field
        self.output_edge_field = output_edge_field
        self.output_node_shape_field = output_node_shape_field
        self.output_edge_shape_field = output_edge_shape_field
        self.node_pad_shape = node_pad_shape
        self.edge_pad_shape = edge_pad_shape
        self.symmetrize_onsite = bool(symmetrize_onsite)
        self.complete_edges = bool(complete_edges)
        self.strict_complete_edges = bool(strict_complete_edges)
        self.add_h0 = bool(add_h0)
        self.full_output_node_field = full_output_node_field
        self.full_output_edge_field = full_output_edge_field

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = self.e3(data)
        packed = feature_tensors_to_block_tensors(
            data,
            self.idp,
            node_features=data.get(self.node_field, None),
            edge_features=data.get(self.edge_field, None),
            node_pad_shape=self.node_pad_shape,
            edge_pad_shape=self.edge_pad_shape,
            symmetrize_onsite=self.symmetrize_onsite,
            complete_edges=self.complete_edges,
            strict_complete_edges=self.strict_complete_edges,
        )
        attach_prediction_block_tensors(
            data,
            packed,
            node_key=self.output_node_field,
            edge_key=self.output_edge_field,
            node_shape_key=self.output_node_shape_field,
            edge_shape_key=self.output_edge_shape_field,
        )
        if self.add_h0:
            if packed.node_blocks is not None and NODE_H0_BLOCKS_KEY in data:
                data[self.full_output_node_field] = data[NODE_H0_BLOCKS_KEY].to(
                    device=packed.node_blocks.device,
                    dtype=packed.node_blocks.dtype,
                ) + packed.node_blocks
            if packed.edge_blocks is not None and EDGE_H0_BLOCKS_KEY in data:
                data[self.full_output_edge_field] = data[EDGE_H0_BLOCKS_KEY].to(
                    device=packed.edge_blocks.device,
                    dtype=packed.edge_blocks.dtype,
                ) + packed.edge_blocks
        return data

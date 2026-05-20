# SPDX-License-Identifier: LGPL-3.0-or-later
"""Block-wise Hamiltonian decoder wrapper.

``BlockwiseE3Hamiltonian`` intentionally keeps DeePTB's existing
``E3Hamiltonian`` stage.  The old stage decodes the model's equivariant/RME
output into DeePTB Hamiltonian feature tensors.  This wrapper then materializes
those feature tensors into padded AO blocks for block-level loss.

Only the current non-SOC path is supported in this minimal patch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

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
    NODE_H0_BLOCKS_KEY,
    NODE_PRED_HAMIL_BLOCKS_KEY,
    attach_prediction_block_tensors,
    feature_tensors_to_block_tensors,
)
from dptb.nn.hamiltonian import E3Hamiltonian


class BlockwiseE3Hamiltonian(nn.Module):
    """Decode equivariant Hamiltonian features and expose AO-block predictions.

    Parameters are forwarded to ``E3Hamiltonian`` except for the block-specific
    output controls below.  Gradients from block loss flow through the
    feature-to-block materialization back to the original feature/RME head.
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

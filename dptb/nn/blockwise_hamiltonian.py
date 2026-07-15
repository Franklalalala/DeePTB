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
    NODE_H0_BLOCKS_KEY,
    NODE_PRED_HAMIL_BLOCKS_KEY,
    attach_prediction_block_tensors,
    feature_tensors_to_block_tensors,
)
from dptb.data.interfaces.p2_contract import build_prior_spec
from dptb.nn.hamiltonian import E3Hamiltonian


def attach_full_hamiltonian_from_prior(
    data: Dict[str, Any],
    *,
    pred_node_field: str = NODE_PRED_HAMIL_BLOCKS_KEY,
    pred_edge_field: str = EDGE_PRED_HAMIL_BLOCKS_KEY,
    prior_node_field: str,
    prior_edge_field: str,
    full_output_node_field: str = "node_full_hamil_blocks",
    full_output_edge_field: str = "edge_full_hamil_blocks",
    prior_label: str = "physical prior",
    non_soc_only: bool = False,
    validate_prior_blocks: bool = False,
    missing_hint: str = "",
) -> Dict[str, Any]:
    """Expose Full-H blocks while preserving correction predictions.

    The model head is parameterized as a correction, but the physical task can
    remain absolute Full-H supervision by pointing the blockwise loss at the
    explicit ``full_output_*`` fields produced here.
    """
    required = (
        pred_node_field,
        pred_edge_field,
        prior_node_field,
        prior_edge_field,
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(
            f"Full-H reconstruction from {prior_label} requires correction "
            f"predictions and converted node/edge prior blocks; missing {missing}."
            f"{missing_hint}"
        )

    pred_node = data[pred_node_field]
    pred_edge = data[pred_edge_field]
    if not isinstance(pred_node, torch.Tensor) or not isinstance(pred_edge, torch.Tensor):
        raise TypeError("Hamiltonian correction block fields must be torch.Tensor values.")
    raw_prior_node = data[prior_node_field]
    raw_prior_edge = data[prior_edge_field]
    if not isinstance(raw_prior_node, torch.Tensor) or not isinstance(
        raw_prior_edge, torch.Tensor
    ):
        raise TypeError(f"{prior_label} block fields must be torch.Tensor values.")
    if non_soc_only and (
        torch.is_complex(raw_prior_node)
        or torch.is_complex(raw_prior_edge)
        or torch.is_complex(pred_node)
        or torch.is_complex(pred_edge)
    ):
        raise TypeError(
            f"{prior_label} Full-H reconstruction is non-SOC only in this path; "
            "complex prior blocks are not supported."
        )
    prior_node = raw_prior_node.to(
        device=pred_node.device, dtype=pred_node.dtype
    )
    prior_edge = raw_prior_edge.to(
        device=pred_edge.device, dtype=pred_edge.dtype
    )
    if tuple(prior_node.shape) != tuple(pred_node.shape):
        raise ValueError(
            f"Node {prior_label}/prediction block shapes differ: "
            f"prior={tuple(prior_node.shape)}, pred={tuple(pred_node.shape)}."
        )
    if tuple(prior_edge.shape) != tuple(pred_edge.shape):
        raise ValueError(
            f"Edge {prior_label}/prediction block shapes differ: "
            f"prior={tuple(prior_edge.shape)}, pred={tuple(pred_edge.shape)}."
        )
    if validate_prior_blocks and (
        not torch.isfinite(prior_node).all()
        or not torch.isfinite(prior_edge).all()
    ):
        raise ValueError(f"{prior_label} blocks contain NaN or infinity.")

    data[full_output_node_field] = prior_node + pred_node
    data[full_output_edge_field] = prior_edge + pred_edge
    return data


def attach_full_hamiltonian_from_h0(
    data: Dict[str, Any],
    *,
    pred_node_field: str = NODE_PRED_HAMIL_BLOCKS_KEY,
    pred_edge_field: str = EDGE_PRED_HAMIL_BLOCKS_KEY,
    full_output_node_field: str = "node_full_hamil_blocks",
    full_output_edge_field: str = "edge_full_hamil_blocks",
    validate_prior_blocks: bool = False,
) -> Dict[str, Any]:
    """Backward-compatible H0 wrapper around generic prior reconstruction."""
    return attach_full_hamiltonian_from_prior(
        data,
        pred_node_field=pred_node_field,
        pred_edge_field=pred_edge_field,
        prior_node_field=NODE_H0_BLOCKS_KEY,
        prior_edge_field=EDGE_H0_BLOCKS_KEY,
        full_output_node_field=full_output_node_field,
        full_output_edge_field=full_output_edge_field,
        prior_label="H0",
        non_soc_only=False,
        validate_prior_blocks=validate_prior_blocks,
        missing_hint=" Enable get_H0 for every dataset split used with this model.",
    )


class BlockwiseE3Hamiltonian(nn.Module):
    """Decode equivariant Hamiltonian features and expose AO-block predictions.

    Parameters are forwarded to ``E3Hamiltonian`` except for the block-specific
    output controls below.  Gradients from block loss flow through the
    differentiable feature-to-block materialization back to the original
    feature/RME head.  It supports both non-SOC spatial blocks and the reduced
    SOC ``uu_real`` single-block contract; full spinor SOC remains out of scope.
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
        add_prior: bool = False,
        prior_kind: Optional[str] = None,
        prior_node_block_field: Optional[str] = None,
        prior_edge_block_field: Optional[str] = None,
        prior_label: Optional[str] = None,
        validate_prior_blocks: bool = False,
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
        self.add_prior = bool(add_prior)
        if self.add_h0 and self.add_prior:
            raise ValueError("add_h0 and add_prior are mutually exclusive.")
        # The AO-block field names and the human label are DERIVED from the
        # single prior kind.  Explicit fields are honored only as overrides (for
        # back-compat); an empty/omitted value derives from the kind so p2/p23
        # can never disagree between the RME conditioning and the AO block that
        # is added back for Full-H reconstruction.
        prior_spec = build_prior_spec("p2" if prior_kind in (None, "") else prior_kind)
        self.prior_kind = prior_spec.kind
        self.prior_node_block_field = (
            str(prior_node_block_field)
            if prior_node_block_field not in (None, "")
            else prior_spec.node_blocks_key
        )
        self.prior_edge_block_field = (
            str(prior_edge_block_field)
            if prior_edge_block_field not in (None, "")
            else prior_spec.edge_blocks_key
        )
        self.prior_label = (
            str(prior_label)
            if prior_label not in (None, "")
            else prior_spec.label
        )
        self.validate_prior_blocks = bool(validate_prior_blocks)
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
            attach_full_hamiltonian_from_h0(
                data,
                pred_node_field=self.output_node_field,
                pred_edge_field=self.output_edge_field,
                full_output_node_field=self.full_output_node_field,
                full_output_edge_field=self.full_output_edge_field,
                validate_prior_blocks=self.validate_prior_blocks,
            )
        elif self.add_prior:
            attach_full_hamiltonian_from_prior(
                data,
                pred_node_field=self.output_node_field,
                pred_edge_field=self.output_edge_field,
                prior_node_field=self.prior_node_block_field,
                prior_edge_field=self.prior_edge_block_field,
                full_output_node_field=self.full_output_node_field,
                full_output_edge_field=self.full_output_edge_field,
                prior_label=self.prior_label,
                non_soc_only=True,
                validate_prior_blocks=self.validate_prior_blocks,
            )
        return data

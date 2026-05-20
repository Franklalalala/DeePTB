# SPDX-License-Identifier: LGPL-3.0-or-later
"""Block-level Hamiltonian loss with feature-compatible TensorBoard metrics."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from dptb.nnops.loss import Loss
except Exception:  # pragma: no cover - standalone tests
    class _DummyLoss:
        @staticmethod
        def register(_name):
            def deco(cls):
                return cls
            return deco
    Loss = _DummyLoss()

try:
    from dptb.data.transforms import OrbitalMapper
except Exception:  # pragma: no cover
    OrbitalMapper = None

from dptb.data.interfaces.blockwise_tensor import (
    EDGE_DELTA_HAMIL_BLOCKS_KEY,
    EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    EDGE_PRED_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    NODE_PRED_HAMIL_BLOCKS_KEY,
    ComponentSums,
    add_components,
    as_tensor,
    block_components,
    feature_components_from_blocks,
    infer_block_shapes,
    l1_rmse_from_components,
    mae_from_components,
)


def _get(data: Mapping[str, Any], key: str, default=None):
    try:
        return data[key]
    except Exception:
        return default


@Loss.register("hamil_blockwise_nextham")
@Loss.register("hamil_block_abs")
class HamilBlockwiseNexTHamLoss(nn.Module):
    """AO-block optimization loss plus old feature-level onsite/hopping logs.

    Default optimization is global masked AO-block MAE.  The feature-compatible
    metrics are computed from the same AO blocks under ``torch.no_grad()`` and
    stored in ``last_onsite_loss`` / ``last_hopping_loss`` for existing Trainer
    logging hooks.
    """

    def __init__(
        self,
        basis: Optional[dict] = None,
        idp: Any = None,
        has_soc: bool = False,
        device: Union[str, torch.device, None] = None,
        pred_node_block_key: str = NODE_PRED_HAMIL_BLOCKS_KEY,
        pred_edge_block_key: str = EDGE_PRED_HAMIL_BLOCKS_KEY,
        target_node_block_key: str = NODE_DELTA_HAMIL_BLOCKS_KEY,
        target_edge_block_key: str = EDGE_DELTA_HAMIL_BLOCKS_KEY,
        target_node_shape_key: str = NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
        target_edge_shape_key: str = EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
        optimization: str = "block_mae",
        block_reduction: str = "global",
        complex_reduction: str = "modulus",
        log_feature_compatible: bool = True,
        feature_log_no_grad: bool = True,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        super().__init__()
        mapper_device = torch.device("cpu" if device is None else device)
        if idp is None:
            if basis is None:
                raise ValueError("Provide either idp or basis.")
            if OrbitalMapper is None:
                raise ImportError("OrbitalMapper is required when idp is not provided.")
            self.idp = OrbitalMapper(basis, method="e3tb", device=mapper_device, has_soc=has_soc)
        else:
            self.idp = idp
        if getattr(self.idp, "has_soc", False):
            raise NotImplementedError("This loss package is intentionally non-SOC only.")
        if hasattr(self.idp, "get_orbital_maps"):
            self.idp.get_orbital_maps()
        if hasattr(self.idp, "get_orbpair_maps"):
            self.idp.get_orbpair_maps()

        self.pred_node_block_key = pred_node_block_key
        self.pred_edge_block_key = pred_edge_block_key
        self.target_node_block_key = target_node_block_key
        self.target_edge_block_key = target_edge_block_key
        self.target_node_shape_key = target_node_shape_key
        self.target_edge_shape_key = target_edge_shape_key
        self.optimization = optimization
        self.block_reduction = block_reduction
        self.complex_reduction = complex_reduction
        self.log_feature_compatible = bool(log_feature_compatible)
        self.feature_log_no_grad = bool(feature_log_no_grad)
        self.eps = float(eps)

        self.last_opt_loss = None
        self.last_block_loss = None
        self.last_block_element_mae = None
        self.last_block_onsite_loss = None
        self.last_block_hopping_loss = None
        self.last_feature_compat_loss = None
        self.last_onsite_loss = None
        self.last_hopping_loss = None
        self.last_block_count = None
        self.last_feature_count = None

    def _predictions(self, data: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        node = _get(data, self.pred_node_block_key, None)
        edge = _get(data, self.pred_edge_block_key, None)
        if node is None or edge is None:
            raise KeyError(
                f"Model output must contain {self.pred_node_block_key}/{self.pred_edge_block_key}. "
                "Use BlockwiseE3Hamiltonian or a block-native head before this loss."
            )
        return as_tensor(node), as_tensor(edge)

    def _targets(self, ref_data: Mapping[str, Any], pred_node: torch.Tensor, pred_edge: torch.Tensor):
        node = _get(ref_data, self.target_node_block_key, None)
        edge = _get(ref_data, self.target_edge_block_key, None)
        if node is None or edge is None:
            raise KeyError(
                f"Missing converted block target fields {self.target_node_block_key}/{self.target_edge_block_key}. "
                "Run convert_feature_lmdb_to_blockwise.py first."
            )
        node = as_tensor(node, device=pred_node.device, dtype=pred_node.dtype)
        edge = as_tensor(edge, device=pred_edge.device, dtype=pred_edge.dtype)
        node_shape = _get(ref_data, self.target_node_shape_key, None)
        edge_shape = _get(ref_data, self.target_edge_shape_key, None)
        if node_shape is None or edge_shape is None:
            node_shape, edge_shape = infer_block_shapes(ref_data, self.idp, device=pred_node.device)
        else:
            node_shape = as_tensor(node_shape, device=pred_node.device, dtype=torch.long)
            edge_shape = as_tensor(edge_shape, device=pred_edge.device, dtype=torch.long)
        return node, edge, node_shape, edge_shape

    def _block_loss(self, pred_node, pred_edge, target_node, target_edge, node_shape, edge_shape):
        node_comp = block_components(pred_node, target_node, node_shape, complex_reduction=self.complex_reduction)
        edge_comp = block_components(pred_edge, target_edge, edge_shape, complex_reduction=self.complex_reduction)
        total_comp = add_components(node_comp, edge_comp)
        node_mae = mae_from_components(node_comp)
        edge_mae = mae_from_components(edge_comp)
        global_mae = mae_from_components(total_comp)
        global_l1_rmse = l1_rmse_from_components(total_comp, eps=self.eps)

        if self.optimization in {"block_mae", "mae", "nextham", "nextham_mae"}:
            if self.block_reduction in {"equal", "equal_onsite_hopping", "legacy"}:
                loss = 0.5 * (node_mae + edge_mae)
            else:
                loss = global_mae
        elif self.optimization in {"block_l1_rmse", "l1_rmse"}:
            if self.block_reduction in {"equal", "equal_onsite_hopping", "legacy"}:
                loss = 0.5 * (
                    l1_rmse_from_components(node_comp, eps=self.eps)
                    + l1_rmse_from_components(edge_comp, eps=self.eps)
                )
            else:
                loss = global_l1_rmse
        elif self.optimization in {"feature", "feature_compatible", "compat"}:
            loss = None
        else:
            raise ValueError(f"Unknown optimization mode: {self.optimization}")
        return loss, node_mae, edge_mae, global_mae, total_comp

    def _feature_logs(self, ref_data, pred_node, pred_edge, target_node, target_edge):
        node_comp, edge_comp = feature_components_from_blocks(
            ref_data,
            self.idp,
            pred_node,
            target_node,
            pred_edge,
            target_edge,
            complex_reduction=self.complex_reduction,
        )
        onsite = l1_rmse_from_components(node_comp, eps=self.eps)
        hopping = l1_rmse_from_components(edge_comp, eps=self.eps)
        return 0.5 * (onsite + hopping), onsite, hopping, add_components(node_comp, edge_comp)

    def forward(self, data: Mapping[str, Any], ref_data: Optional[Mapping[str, Any]] = None):
        if ref_data is None:
            ref_data = data
        pred_node, pred_edge = self._predictions(data)
        target_node, target_edge, node_shape, edge_shape = self._targets(ref_data, pred_node, pred_edge)

        block_loss, block_onsite, block_hopping, block_global_mae, block_comp = self._block_loss(
            pred_node, pred_edge, target_node, target_edge, node_shape, edge_shape
        )

        need_feature = self.log_feature_compatible or self.optimization in {"feature", "feature_compatible", "compat"}
        if need_feature:
            if self.feature_log_no_grad and self.optimization not in {"feature", "feature_compatible", "compat"}:
                with torch.no_grad():
                    feature_total, feature_onsite, feature_hopping, feature_comp = self._feature_logs(
                        ref_data,
                        pred_node.detach(),
                        pred_edge.detach(),
                        target_node.detach(),
                        target_edge.detach(),
                    )
            else:
                feature_total, feature_onsite, feature_hopping, feature_comp = self._feature_logs(
                    ref_data, pred_node, pred_edge, target_node, target_edge
                )
        else:
            feature_total = feature_onsite = feature_hopping = None
            feature_comp = None

        opt_loss = feature_total if self.optimization in {"feature", "feature_compatible", "compat"} else block_loss
        if opt_loss is None:
            raise RuntimeError("Could not determine optimization loss.")

        self.last_opt_loss = opt_loss.detach()
        self.last_block_loss = block_loss.detach() if block_loss is not None else None
        self.last_block_element_mae = block_global_mae.detach()
        self.last_block_onsite_loss = block_onsite.detach()
        self.last_block_hopping_loss = block_hopping.detach()
        self.last_block_count = block_comp.count.detach()
        if feature_total is not None:
            self.last_feature_compat_loss = feature_total.detach()
            self.last_onsite_loss = feature_onsite.detach()      # historical TB onsite semantics
            self.last_hopping_loss = feature_hopping.detach()    # historical TB hopping semantics
            self.last_feature_count = feature_comp.count.detach() if feature_comp is not None else None
        else:
            self.last_feature_compat_loss = None
            self.last_onsite_loss = None
            self.last_hopping_loss = None
            self.last_feature_count = None
        return opt_loss

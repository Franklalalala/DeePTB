"""Invariant-conditioned two-body SO(3) refinement for mature edge features."""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from e3nn import o3


def _scalar_0e_indices(
    irreps: o3.Irreps,
    *,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    indices = []
    for term_slice, (_, ir) in zip(irreps.slices(), irreps):
        if ir.l == 0 and ir.p == 1:
            indices.extend(range(term_slice.start, term_slice.stop))
    if not indices:
        raise ValueError(f"PairSO3RefineTP requires at least one 0e scalar; got {irreps}.")
    return torch.tensor(indices, dtype=torch.long, device=device)


class PairSO3RefineTP(torch.nn.Module):
    """Refine one edge row with an invariant-conditioned ``x_i x x_j`` CG TP."""

    def __init__(
        self,
        node_irreps: Union[str, o3.Irreps],
        edge_irreps: Union[str, o3.Irreps],
        *,
        rank: int = 16,
        condition: str = "scalar_0e",
        internal_weights: bool = True,
        dynamic_init: float = 0.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.rank = int(rank)
        self.condition = str(condition).strip().lower()
        self.internal_weights = bool(internal_weights)
        self.dynamic_init = float(dynamic_init)

        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if self.dynamic_init < 0.0:
            raise ValueError(
                f"dynamic_init must be non-negative, got {dynamic_init}."
            )
        if self.condition != "scalar_0e":
            raise ValueError(
                "PairSO3RefineTP supports only condition='scalar_0e', "
                f"got {condition!r}."
            )

        self.register_buffer(
            "_node_scalar_indices",
            _scalar_0e_indices(self.node_irreps, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_edge_scalar_indices",
            _scalar_0e_indices(self.edge_irreps, device=device),
            persistent=False,
        )
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.node_irreps,
            self.node_irreps,
            self.edge_irreps,
            shared_weights=False,
            internal_weights=False,
        )
        if dtype is not None or device is not None:
            self.tensor_product = self.tensor_product.to(dtype=dtype, device=device)
        if self.tensor_product.weight_numel <= 0:
            raise ValueError(
                "No compatible node-node-to-edge CG paths were found for "
                f"node_irreps={self.node_irreps}, edge_irreps={self.edge_irreps}."
            )

        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device
        condition_dim = (
            2 * int(self._node_scalar_indices.numel())
            + int(self._edge_scalar_indices.numel())
        )
        self.condition_down = torch.nn.Linear(
            condition_dim, self.rank, bias=True, **factory_kwargs
        )
        self.dynamic_up = torch.nn.Linear(
            self.rank,
            self.tensor_product.weight_numel,
            bias=True,
            **factory_kwargs,
        )
        if self.internal_weights:
            self.static_weights = torch.nn.Parameter(
                torch.empty(self.tensor_product.weight_numel, **factory_kwargs)
            )
        else:
            self.register_parameter("static_weights", None)
        self.reset_parameters()

    @property
    def weight_numel(self) -> int:
        return int(self.tensor_product.weight_numel)

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(
            self.condition_down.weight, a=math.sqrt(5.0)
        )
        if self.condition_down.bias is not None:
            torch.nn.init.zeros_(self.condition_down.bias)
        if self.dynamic_init == 0.0:
            torch.nn.init.zeros_(self.dynamic_up.weight)
        else:
            torch.nn.init.normal_(
                self.dynamic_up.weight, mean=0.0, std=self.dynamic_init
            )
        if self.dynamic_up.bias is not None:
            torch.nn.init.zeros_(self.dynamic_up.bias)
        if self.static_weights is not None:
            torch.nn.init.normal_(
                self.static_weights,
                mean=0.0,
                std=1.0 / math.sqrt(float(self.weight_numel)),
            )

    def _validate_inputs(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.ndim != 2 or node_features.shape[-1] != self.node_irreps.dim:
            raise ValueError(
                "node_features must have shape [num_nodes, "
                f"{self.node_irreps.dim}], got {tuple(node_features.shape)}."
            )
        if edge_features.ndim != 2 or edge_features.shape[-1] != self.edge_irreps.dim:
            raise ValueError(
                "edge_features must have shape [num_edges, "
                f"{self.edge_irreps.dim}], got {tuple(edge_features.shape)}."
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}."
            )
        if edge_index.shape[1] != edge_features.shape[0]:
            raise ValueError(
                "edge_index and edge_features must describe the same ordered edge "
                f"rows; got {edge_index.shape[1]} and {edge_features.shape[0]}."
            )
        if edge_index.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError(f"edge_index must be integral, got {edge_index.dtype}.")
        return edge_index.to(device=node_features.device, dtype=torch.long)

    def attention_weights(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-edge invariant scalar weights used by the CG TP."""
        edge_index = self._validate_inputs(node_features, edge_features, edge_index)
        src, dst = edge_index[0], edge_index[1]
        condition = torch.cat(
            (
                node_features.index_select(0, src).index_select(
                    1, self._node_scalar_indices
                ),
                node_features.index_select(0, dst).index_select(
                    1, self._node_scalar_indices
                ),
                edge_features.index_select(1, self._edge_scalar_indices),
            ),
            dim=-1,
        )
        invariant = torch.nn.functional.silu(self.condition_down(condition))
        weights = self.dynamic_up(invariant)
        if self.static_weights is not None:
            weights = weights + self.static_weights
        return weights

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = self._validate_inputs(node_features, edge_features, edge_index)
        weights = self.attention_weights(node_features, edge_features, edge_index)
        src, dst = edge_index[0], edge_index[1]
        refinement = self.tensor_product(
            node_features.index_select(0, src),
            node_features.index_select(0, dst),
            weights,
        )
        return edge_features + refinement

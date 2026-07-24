"""Invariant-conditioned two-body SO(3) refinement for mature edge features."""

from __future__ import annotations

import logging
import math
from numbers import Integral
from typing import Optional, Union

import torch
from e3nn import o3


log = logging.getLogger(__name__)


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
    """Refine one edge row with an invariant-conditioned ``x_i x x_j`` CG TP.

    ``weight_mode="full"`` preserves the original behavior and materializes
    one external tensor-product weight per edge and multiplicity path.
    ``weight_mode="per_path"`` instead predicts one scalar delta gate per
    tensor-product instruction.  A gate of zero leaves that instruction's
    learned static tensor-product contribution unchanged.
    ``weight_mode="qhflow"`` sandwiches a per-edge ``uuu`` tensor product
    between shared equivariant linears.  Its conditioner predicts the full
    diagonal ``uuu`` weight vector, avoiding the multiplicity-cubed cost of
    the fully connected tensor product while retaining channel-resolved
    dynamic weights.

    ``dynamic_init=0`` alone is not an identity initialization: with the
    default ``internal_weights=True``, learned static weights are still
    randomly initialized.  Set ``identity_init=True`` to initialize both
    dynamic and static contributions to zero so the initial forward is
    bitwise equal to the input edge features.
    """

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
        max_weight_numel: Optional[int] = None,
        weight_mode: str = "full",
        identity_init: bool = False,
    ) -> None:
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.rank = int(rank)
        self.condition = str(condition).strip().lower()
        self.internal_weights = bool(internal_weights)
        self.dynamic_init = float(dynamic_init)
        if max_weight_numel is None:
            self.max_weight_numel = None
        elif isinstance(max_weight_numel, bool) or not isinstance(
            max_weight_numel,
            Integral,
        ):
            raise ValueError(
                "max_weight_numel must be a non-negative integer or None, "
                f"got {max_weight_numel!r}."
            )
        else:
            self.max_weight_numel = int(max_weight_numel)
        self.weight_mode = str(weight_mode).strip().lower()
        self.identity_init = bool(identity_init)

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
        if self.weight_mode not in {"full", "per_path", "qhflow"}:
            raise ValueError(
                "weight_mode must be 'full', 'per_path', or 'qhflow', "
                f"got {weight_mode!r}."
            )
        if self.weight_mode == "per_path" and not self.internal_weights:
            raise ValueError(
                "weight_mode='per_path' requires internal_weights=True "
                "because its scalar gates modulate learned static weights."
            )
        if self.max_weight_numel is not None and self.max_weight_numel < 0:
            raise ValueError(
                "max_weight_numel must be a non-negative integer or None, "
                f"got {max_weight_numel!r}."
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
        if self.weight_mode == "qhflow":
            self.tensor_product = self._build_qhflow_tensor_product(
                dtype=dtype,
                device=device,
            )
        else:
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
        dtype_for_cost = dtype if dtype is not None else torch.get_default_dtype()
        full_weight_numel = (
            self._full_fctp_weight_numel()
            if self.weight_mode == "qhflow"
            else int(self.tensor_product.weight_numel)
        )
        full_dynamic_weight_bytes = (
            full_weight_numel
            * torch.empty((), dtype=dtype_for_cost).element_size()
        )
        dynamic_dof = (
            len(self.tensor_product.instructions)
            if self.weight_mode == "per_path"
            else int(self.tensor_product.weight_numel)
        )
        dynamic_weight_bytes = (
            dynamic_dof * torch.empty((), dtype=dtype_for_cost).element_size()
        )
        log.info(
            "PairSO3RefineTP external weight cost: weight_mode=%s, "
            "weight_numel=%d, dynamic_dof_per_edge=%d, "
            "dynamic_weight_bytes_per_edge=%d, "
            "full_dynamic_weight_bytes_per_edge=%d (dtype=%s).",
            self.weight_mode,
            int(self.tensor_product.weight_numel),
            dynamic_dof,
            dynamic_weight_bytes,
            full_dynamic_weight_bytes,
            dtype_for_cost,
        )
        if (
            self.max_weight_numel is not None
            and self.tensor_product.weight_numel > self.max_weight_numel
        ):
            raise ValueError(
                "PairSO3RefineTP weight_numel exceeds max_weight_numel: "
                f"actual={int(self.tensor_product.weight_numel)}, "
                f"limit={self.max_weight_numel}. Inspect the configuration with "
                "dptb/utils/pair_refine_cost.py before enabling refinement."
            )

        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device
        if self.weight_mode == "qhflow":
            self.linear_pre = o3.Linear(
                self.node_irreps,
                self.node_irreps,
                internal_weights=True,
                shared_weights=True,
                biases=True,
            ).to(dtype=dtype, device=device)
            self.linear_post = o3.Linear(
                self.edge_irreps,
                self.edge_irreps,
                internal_weights=True,
                shared_weights=True,
                biases=True,
            ).to(dtype=dtype, device=device)
        condition_dim = (
            2 * int(self._node_scalar_indices.numel())
            + int(self._edge_scalar_indices.numel())
        )
        self.condition_down = torch.nn.Linear(
            condition_dim, self.rank, bias=True, **factory_kwargs
        )
        dynamic_dim = (
            self.n_paths if self.weight_mode == "per_path" else self.weight_numel
        )
        self.dynamic_up = torch.nn.Linear(
            self.rank,
            dynamic_dim,
            bias=True,
            **factory_kwargs,
        )
        if self.internal_weights and self.weight_mode != "qhflow":
            self.static_weights = torch.nn.Parameter(
                torch.empty(self.tensor_product.weight_numel, **factory_kwargs)
            )
        else:
            self.register_parameter("static_weights", None)
        if self.weight_mode == "per_path":
            (
                self.path_tensor_product,
                path_output_index,
                path_gate_index,
            ) = self._build_path_tensor_product(dtype=dtype, device=device)
            self.register_buffer(
                "_path_output_index",
                path_output_index,
                persistent=False,
            )
            self.register_buffer(
                "_path_gate_index",
                path_gate_index,
                persistent=False,
            )
        self.reset_parameters()

    @property
    def weight_numel(self) -> int:
        return int(self.tensor_product.weight_numel)

    @property
    def n_paths(self) -> int:
        return len(self.tensor_product.instructions)

    def _full_fctp_weight_numel(self) -> int:
        return sum(
            mul_in1 * mul_in2 * mul_out
            for mul_in1, ir_in1 in self.node_irreps
            for mul_in2, ir_in2 in self.node_irreps
            for mul_out, ir_out in self.edge_irreps
            if ir_out in ir_in1 * ir_in2
        )

    def _build_qhflow_tensor_product(
        self,
        *,
        dtype: Optional[torch.dtype],
        device: Optional[Union[str, torch.device]],
    ) -> o3.TensorProduct:
        """Build the multiplicity-diagonal QHFlow tensor product.

        ``uuu`` requires equal multiplicities for both inputs and the output.
        e3nn's default normalization is used deliberately; the explicit
        weight-count assertion below fixes the expected per-edge layout.
        """
        instructions = []
        expected_weight_numel = 0
        for i_in1, (mul_in1, ir_in1) in enumerate(self.node_irreps):
            for i_in2, (mul_in2, ir_in2) in enumerate(self.node_irreps):
                if mul_in1 != mul_in2:
                    continue
                for i_out, (mul_out, ir_out) in enumerate(self.edge_irreps):
                    if mul_out == mul_in1 and ir_out in ir_in1 * ir_in2:
                        instructions.append(
                            (i_in1, i_in2, i_out, "uuu", True)
                        )
                        expected_weight_numel += mul_out
        if not instructions:
            raise ValueError(
                "No compatible multiplicity-diagonal uuu paths were found for "
                f"node_irreps={self.node_irreps}, edge_irreps={self.edge_irreps}."
            )
        tensor_product = o3.TensorProduct(
            self.node_irreps,
            self.node_irreps,
            self.edge_irreps,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        if dtype is not None or device is not None:
            tensor_product = tensor_product.to(dtype=dtype, device=device)
        if tensor_product.weight_numel != expected_weight_numel:
            raise RuntimeError(
                "QHFlow uuu tensor-product weight layout disagrees with the "
                f"instruction count: {tensor_product.weight_numel} != "
                f"{expected_weight_numel}."
            )
        return tensor_product

    def _build_path_tensor_product(
        self,
        *,
        dtype: Optional[torch.dtype],
        device: Optional[Union[str, torch.device]],
    ) -> tuple[o3.TensorProduct, torch.Tensor, torch.Tensor]:
        """Expose each original FCTP instruction in a separate output block.

        e3nn 0.5.5 stores the final normalization coefficient in
        ``Instruction.path_weight``.  A new tensor product with
        ``irrep_normalization="none"`` and ``path_normalization="none"``
        therefore receives its square as the raw instruction path weight.
        This reproduces each original contribution exactly while accepting
        one shared static weight vector instead of ``[num_edges, weight_numel]``.
        """
        path_output_terms = []
        path_instructions = []
        output_index = []
        gate_index = []
        edge_slices = self.edge_irreps.slices()
        for path, instruction in enumerate(self.tensor_product.instructions):
            mul, irrep = self.edge_irreps[instruction.i_out]
            path_output_terms.append((mul, irrep))
            path_instructions.append(
                (
                    instruction.i_in1,
                    instruction.i_in2,
                    path,
                    instruction.connection_mode,
                    True,
                    float(instruction.path_weight) ** 2,
                )
            )
            edge_slice = edge_slices[instruction.i_out]
            output_index.extend(range(edge_slice.start, edge_slice.stop))
            gate_index.extend([path] * (edge_slice.stop - edge_slice.start))

        path_tensor_product = o3.TensorProduct(
            self.node_irreps,
            self.node_irreps,
            o3.Irreps(path_output_terms),
            path_instructions,
            irrep_normalization="none",
            path_normalization="none",
            shared_weights=True,
            internal_weights=False,
        )
        if dtype is not None or device is not None:
            path_tensor_product = path_tensor_product.to(
                dtype=dtype,
                device=device,
            )
        if path_tensor_product.weight_numel != self.weight_numel:
            raise RuntimeError(
                "Per-path tensor-product static layout disagrees with the "
                f"full FCTP: {path_tensor_product.weight_numel} != "
                f"{self.weight_numel}."
            )
        return (
            path_tensor_product,
            torch.tensor(output_index, dtype=torch.long, device=device),
            torch.tensor(gate_index, dtype=torch.long, device=device),
        )

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(
            self.condition_down.weight, a=math.sqrt(5.0)
        )
        if self.condition_down.bias is not None:
            torch.nn.init.zeros_(self.condition_down.bias)
        if self.identity_init or self.dynamic_init == 0.0:
            torch.nn.init.zeros_(self.dynamic_up.weight)
        else:
            torch.nn.init.normal_(
                self.dynamic_up.weight, mean=0.0, std=self.dynamic_init
            )
        if self.dynamic_up.bias is not None:
            torch.nn.init.zeros_(self.dynamic_up.bias)
        if self.static_weights is not None:
            if self.identity_init:
                torch.nn.init.zeros_(self.static_weights)
            else:
                torch.nn.init.normal_(
                    self.static_weights,
                    mean=0.0,
                    std=1.0 / math.sqrt(float(self.weight_numel)),
                )
        if self.weight_mode == "qhflow" and self.identity_init:
            torch.nn.init.zeros_(self.linear_post.weight)
            if self.linear_post.bias is not None:
                torch.nn.init.zeros_(self.linear_post.bias)

    def _condition_weights(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
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
        return self.dynamic_up(invariant)

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
        """Return full weights or per-instruction scalar delta gates."""
        edge_index = self._validate_inputs(node_features, edge_features, edge_index)
        condition_node_features = (
            self.linear_pre(node_features)
            if self.weight_mode == "qhflow"
            else node_features
        )
        weights = self._condition_weights(
            condition_node_features,
            edge_features,
            edge_index,
        )
        if self.weight_mode == "full" and self.static_weights is not None:
            weights = weights + self.static_weights
        return weights

    @staticmethod
    def _validate_edge_scale(
        edge_scale: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        if not isinstance(edge_scale, torch.Tensor):
            raise ValueError(
                "edge_scale must be a Tensor with shape [num_edges] or "
                f"[num_edges, 1], got {type(edge_scale).__name__}."
            )
        if edge_scale.ndim == 1 and edge_scale.shape[0] == num_edges:
            return edge_scale.unsqueeze(-1)
        if (
            edge_scale.ndim == 2
            and edge_scale.shape[0] == num_edges
            and edge_scale.shape[1] == 1
        ):
            return edge_scale
        raise ValueError(
            "edge_scale must have shape [num_edges] or [num_edges, 1]; "
            f"num_edges={num_edges}, got {tuple(edge_scale.shape)}."
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        edge_index = self._validate_inputs(node_features, edge_features, edge_index)
        if edge_scale is not None:
            edge_scale = self._validate_edge_scale(
                edge_scale,
                edge_features.shape[0],
            )
        if self.weight_mode == "qhflow":
            pair_node_features = self.linear_pre(node_features)
            weights = self._condition_weights(
                pair_node_features,
                edge_features,
                edge_index,
            )
        else:
            pair_node_features = node_features
            weights = self.attention_weights(
                node_features,
                edge_features,
                edge_index,
            )
        src, dst = edge_index[0], edge_index[1]
        node_src = pair_node_features.index_select(0, src)
        node_dst = pair_node_features.index_select(0, dst)
        if self.weight_mode == "full":
            refinement = self.tensor_product(node_src, node_dst, weights)
        elif self.weight_mode == "per_path":
            path_features = self.path_tensor_product(
                node_src,
                node_dst,
                self.static_weights,
            )
            component_gates = weights.index_select(1, self._path_gate_index)
            path_features = path_features * (1.0 + component_gates)
            refinement = edge_features.new_zeros(edge_features.shape)
            refinement.scatter_add_(
                1,
                self._path_output_index.expand(edge_features.shape[0], -1),
                path_features,
            )
        else:
            refinement = self.linear_post(
                self.tensor_product(node_src, node_dst, weights)
            )
        if edge_scale is not None:
            refinement = refinement * edge_scale
        return edge_features + refinement

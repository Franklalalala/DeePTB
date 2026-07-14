# SPDX-License-Identifier: LGPL-3.0-or-later
"""LEM-MoE-v3 embedding driven by a first-class physical prior.

The implementation is deliberately narrow: real, non-SOC P2 or P23 features
are required under dedicated node/edge keys.  The historical H0 embedding is
reused as an implementation primitive but is not modified and none of its
target-fallback behavior is exposed through this public API.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from dptb.data import AtomicDataDict, _keys
from dptb.nn.embedding.emb import Embedding

from .lem_moe_v3_h0 import LemMoEV3H0
from .lem_moe_v3_h0_helpers import H0InitLayer
from .soft_edge_memory import EquivariantSoftEdgeMemory


class P2MemoryInitLayer(torch.nn.Module):
    """Validate the P2 contract, project it, then optionally query edge memory."""

    def __init__(
        self,
        prior_init: H0InitLayer,
        *,
        prior_kind: str,
        node_key: str,
        edge_key: str,
        use_node_init: bool,
        use_edge_init: bool,
        use_soft_edge_memory: bool,
        soft_edge_memory_num_slots: int,
        soft_edge_memory_num_heads: int,
        soft_edge_memory_head_dim: int,
        soft_edge_memory_temperature: float,
        soft_edge_memory_dropout: float,
        soft_edge_memory_gate_mode: str,
        soft_edge_memory_gate_bias: float,
        soft_edge_memory_gate_eps: float,
        soft_edge_memory_zero_init_output: bool,
        soft_edge_memory_input_norm: bool,
        validate_inputs: bool,
        soft_edge_memory_diagnostics_mode: str,
        soft_edge_memory_diagnostics_sample_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.prior_init = prior_init
        self.idp = prior_init.idp
        self.irreps_out = prior_init.irreps_out
        self.prior_dim = prior_init.h0_dim
        self.prior_kind = str(prior_kind).strip().lower()
        self.prior_label = self.prior_kind.upper()
        self.node_key = str(node_key)
        self.edge_key = str(edge_key)
        self.use_node_init = bool(use_node_init)
        self.use_edge_init = bool(use_edge_init)
        self.validate_inputs = bool(validate_inputs)
        self.soft_edge_memory = (
            EquivariantSoftEdgeMemory(
                self.irreps_out,
                num_slots=soft_edge_memory_num_slots,
                num_heads=soft_edge_memory_num_heads,
                head_dim=soft_edge_memory_head_dim,
                temperature=soft_edge_memory_temperature,
                attention_dropout=soft_edge_memory_dropout,
                gate_mode=soft_edge_memory_gate_mode,
                gate_bias=soft_edge_memory_gate_bias,
                gate_eps=soft_edge_memory_gate_eps,
                zero_init_output=soft_edge_memory_zero_init_output,
                use_input_norm=soft_edge_memory_input_norm,
                validate_inputs=self.validate_inputs,
                diagnostics_mode=soft_edge_memory_diagnostics_mode,
                diagnostics_sample_size=soft_edge_memory_diagnostics_sample_size,
                dtype=dtype,
                device=device,
            )
            if use_soft_edge_memory
            else None
        )

    def _validate_feature(
        self,
        data: AtomicDataDict.Type,
        *,
        key: str,
        rows: int,
        label: str,
    ) -> None:
        if key not in data:
            raise KeyError(
                f"lem_moe_v3_prior requires {label} {self.prior_label} "
                f"field {key!r}; "
                "target-H fallback is intentionally disabled."
            )
        value = data[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{label} {self.prior_label} field {key!r} must be a torch.Tensor."
            )
        if value.ndim != 2:
            raise ValueError(
                f"{label} {self.prior_label} field {key!r} must have rank 2 "
                "[rows, rme], "
                f"got shape {tuple(value.shape)}."
            )
        expected = (rows, self.prior_dim)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"{label} {self.prior_label} field {key!r} has shape "
                f"{tuple(value.shape)}; "
                f"expected {expected}."
            )
        if torch.is_complex(value):
            raise TypeError(
                f"{label} {self.prior_label} field {key!r} is complex; "
                "the physical-prior route is non-SOC only."
            )
        if not torch.is_floating_point(value):
            raise TypeError(
                f"{label} {self.prior_label} field {key!r} must be floating "
                f"point, got {value.dtype}."
            )
        if self.validate_inputs and not torch.isfinite(value).all():
            raise ValueError(
                f"{label} {self.prior_label} field {key!r} contains NaN or infinity."
            )

    def forward(
        self,
        data: AtomicDataDict.Type,
        edge_index: torch.Tensor,
        atom_type: torch.Tensor,
        bond_type: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_length: torch.Tensor,
        edge_one_hot: torch.Tensor,
        active_edges: Optional[torch.Tensor] = None,
        cutoff_coeffs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_node_init:
            self._validate_feature(
                data,
                key=self.node_key,
                rows=atom_type.numel(),
                label="node",
            )
        if self.use_edge_init:
            self._validate_feature(
                data,
                key=self.edge_key,
                rows=edge_index.shape[1],
                label="edge",
            )

        result = self.prior_init(
            data,
            edge_index,
            atom_type,
            bond_type,
            edge_sh,
            edge_length,
            edge_one_hot,
            active_edges,
            cutoff_coeffs,
        )
        latents, node_features, edge_features, cutoff_coeffs, active_edges = result
        if self.soft_edge_memory is not None:
            edge_features, diagnostics = self.soft_edge_memory(edge_features)
            diagnostic_fields = {
                "attention_entropy": "soft_edge_memory_attention_entropy",
                "attention_max_probability": (
                    "soft_edge_memory_attention_max_probability"
                ),
                "gate_mean": "soft_edge_memory_gate_mean",
            }
            for source, target in diagnostic_fields.items():
                if source in diagnostics:
                    data[target] = diagnostics[source]
        return latents, node_features, edge_features, cutoff_coeffs, active_edges


@Embedding.register("lem_moe_v3_prior")
class LemMoEV3Prior(LemMoEV3H0):
    """Strict non-SOC P2/P23 embedding with optional soft edge memory."""

    def __init__(
        self,
        *,
        prior_kind: str = "p2",
        use_prior_init: bool = True,
        prior_node_key: str = _keys.NODE_P2_KEY,
        prior_edge_key: str = _keys.EDGE_P2_KEY,
        use_prior_node_init: bool = True,
        use_prior_edge_init: bool = True,
        prior_node_mode: str = "direct",
        prior_merge_mode: str = "replace",
        prior_self_edge_tol: float = 1e-8,
        use_soft_edge_memory: bool = True,
        soft_edge_memory_num_slots: int = 64,
        soft_edge_memory_num_heads: int = 4,
        soft_edge_memory_head_dim: int = 16,
        soft_edge_memory_temperature: float = 1.0,
        soft_edge_memory_dropout: float = 0.0,
        soft_edge_memory_gate_mode: str = "deepseek",
        soft_edge_memory_gate_bias: float = 0.0,
        soft_edge_memory_gate_eps: float = 1e-6,
        soft_edge_memory_zero_init_output: bool = True,
        soft_edge_memory_input_norm: bool = True,
        prior_validate_inputs: bool = False,
        soft_edge_memory_diagnostics_mode: str = "off",
        soft_edge_memory_diagnostics_sample_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        prior_kind = str(prior_kind).lower()
        expected_fields = {
            "p2": (_keys.NODE_P2_KEY, _keys.EDGE_P2_KEY),
            "p23": (_keys.NODE_P23_KEY, _keys.EDGE_P23_KEY),
        }
        if prior_kind not in expected_fields:
            raise ValueError(
                "lem_moe_v3_prior supports prior_kind='p2' or 'p23'; "
                f"got {prior_kind!r}."
            )
        expected_node_key, expected_edge_key = expected_fields[prior_kind]
        if str(prior_node_key) != expected_node_key or str(prior_edge_key) != expected_edge_key:
            raise ValueError(
                f"prior_kind={prior_kind!r} requires prior_node_key="
                f"{expected_node_key!r} and prior_edge_key={expected_edge_key!r}; "
                f"got {prior_node_key!r}/{prior_edge_key!r}. Refusing to mix "
                "physical-prior families."
            )
        if use_soft_edge_memory and not use_prior_init:
            raise ValueError(
                "use_soft_edge_memory=true requires use_prior_init=true in the P2 route."
            )

        super().__init__(
            use_h0_init=use_prior_init,
            h0_node_key=prior_node_key,
            h0_edge_key=prior_edge_key,
            use_h0_node_init=use_prior_node_init,
            use_h0_edge_init=use_prior_edge_init,
            h0_node_mode=prior_node_mode,
            fallback_to_hamiltonian=False,
            h0_fallback_to_hamiltonian=False,
            allow_target_fallback_in_training=False,
            h0_merge_mode=prior_merge_mode,
            h0_self_edge_tol=prior_self_edge_tol,
            **kwargs,
        )
        if bool(getattr(self.idp, "has_soc", False)):
            raise NotImplementedError(
                "lem_moe_v3_prior is intentionally non-SOC in the first implementation."
            )

        self.prior_kind = prior_kind
        self.prior_node_key = str(prior_node_key)
        self.prior_edge_key = str(prior_edge_key)
        self.use_prior_init = bool(use_prior_init)
        self.use_soft_edge_memory = bool(use_soft_edge_memory)
        if self.use_prior_init:
            if not isinstance(self.init_layer, H0InitLayer):
                raise TypeError(
                    "Internal P2 projection setup expected H0InitLayer; got "
                    f"{type(self.init_layer).__name__}."
                )
            self.init_layer = P2MemoryInitLayer(
                self.init_layer,
                prior_kind=self.prior_kind,
                node_key=self.prior_node_key,
                edge_key=self.prior_edge_key,
                use_node_init=use_prior_node_init,
                use_edge_init=use_prior_edge_init,
                use_soft_edge_memory=self.use_soft_edge_memory,
                soft_edge_memory_num_slots=soft_edge_memory_num_slots,
                soft_edge_memory_num_heads=soft_edge_memory_num_heads,
                soft_edge_memory_head_dim=soft_edge_memory_head_dim,
                soft_edge_memory_temperature=soft_edge_memory_temperature,
                soft_edge_memory_dropout=soft_edge_memory_dropout,
                soft_edge_memory_gate_mode=soft_edge_memory_gate_mode,
                soft_edge_memory_gate_bias=soft_edge_memory_gate_bias,
                soft_edge_memory_gate_eps=soft_edge_memory_gate_eps,
                soft_edge_memory_zero_init_output=soft_edge_memory_zero_init_output,
                soft_edge_memory_input_norm=soft_edge_memory_input_norm,
                validate_inputs=prior_validate_inputs,
                soft_edge_memory_diagnostics_mode=(
                    soft_edge_memory_diagnostics_mode
                ),
                soft_edge_memory_diagnostics_sample_size=(
                    soft_edge_memory_diagnostics_sample_size
                ),
                dtype=self.dtype,
                device=self.device,
            )


__all__ = ["LemMoEV3Prior", "P2MemoryInitLayer"]

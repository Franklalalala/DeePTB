# SPDX-License-Identifier: LGPL-3.0-or-later
"""LEM-MoE-v3 embedding driven by a first-class physical prior.

The implementation is deliberately narrow: real, non-SOC P2 or P23 features
are required under dedicated node/edge keys.  The historical H0 embedding is
reused as an implementation primitive but is not modified and none of its
target-fallback behavior is exposed through this public API.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from dptb.configuration import resolve_init_scope
from dptb.data import AtomicDataDict
from dptb.data.interfaces.p2_contract import build_prior_spec
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
        prior_init_scope: Optional[str] = None,
        prior_kind: Optional[str] = None,
        use_prior_init: Optional[bool] = None,
        prior_node_key: str = "",
        prior_edge_key: str = "",
        use_prior_node_init: Optional[bool] = None,
        use_prior_edge_init: Optional[bool] = None,
        prior_node_mode: str = "direct",
        prior_merge_mode: str = "replace",
        prior_self_edge_tol: float = 1e-8,
        soft_edge_memory: Optional[Dict[str, Any]] = None,
        use_soft_edge_memory: Optional[bool] = None,
        soft_edge_memory_num_slots: Optional[int] = None,
        soft_edge_memory_num_heads: Optional[int] = None,
        soft_edge_memory_head_dim: Optional[int] = None,
        soft_edge_memory_temperature: Optional[float] = None,
        soft_edge_memory_dropout: Optional[float] = None,
        soft_edge_memory_gate_mode: Optional[str] = None,
        soft_edge_memory_gate_bias: Optional[float] = None,
        soft_edge_memory_gate_eps: Optional[float] = None,
        soft_edge_memory_zero_init_output: Optional[bool] = None,
        soft_edge_memory_input_norm: Optional[bool] = None,
        prior_validate_inputs: bool = False,
        soft_edge_memory_diagnostics_mode: Optional[str] = None,
        soft_edge_memory_diagnostics_sample_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        prior_kind = (
            "p2" if prior_kind is None else str(prior_kind).strip().lower()
        )
        try:
            prior_spec = build_prior_spec(prior_kind)
        except ValueError as exc:
            raise ValueError(
                "lem_moe_v3_prior supports prior_kind='p2' or 'p23'; "
                f"got {prior_kind!r}."
            ) from exc
        # The node/edge RME keys are DERIVED from the single prior kind.  An
        # explicit key is accepted only as a deprecated echo that must match the
        # derived value, so p2/p23 fields can never be mixed by cross-field skew.
        expected_node_key = prior_spec.node_rme_key
        expected_edge_key = prior_spec.edge_rme_key
        prior_node_key = (
            expected_node_key
            if prior_node_key in (None, "")
            else str(prior_node_key)
        )
        prior_edge_key = (
            expected_edge_key
            if prior_edge_key in (None, "")
            else str(prior_edge_key)
        )
        if prior_node_key != expected_node_key or prior_edge_key != expected_edge_key:
            raise ValueError(
                f"prior_kind={prior_kind!r} requires prior_node_key="
                f"{expected_node_key!r} and prior_edge_key={expected_edge_key!r}; "
                f"got {prior_node_key!r}/{prior_edge_key!r}. Refusing to mix "
                "physical-prior families."
            )
        (
            self.prior_init_scope,
            use_prior_init,
            use_prior_node_init,
            use_prior_edge_init,
        ) = resolve_init_scope(
            prior_init_scope,
            enabled=use_prior_init,
            node=use_prior_node_init,
            edge=use_prior_edge_init,
            option_name="prior_init_scope",
        )

        memory = dict(soft_edge_memory or {})
        legacy_memory = {
            "enabled": use_soft_edge_memory,
            "num_slots": soft_edge_memory_num_slots,
            "num_heads": soft_edge_memory_num_heads,
            "head_dim": soft_edge_memory_head_dim,
            "temperature": soft_edge_memory_temperature,
            "dropout": soft_edge_memory_dropout,
            "gate_mode": soft_edge_memory_gate_mode,
            "gate_bias": soft_edge_memory_gate_bias,
            "gate_eps": soft_edge_memory_gate_eps,
            "zero_init_output": soft_edge_memory_zero_init_output,
            "input_norm": soft_edge_memory_input_norm,
            "diagnostics_mode": soft_edge_memory_diagnostics_mode,
            "diagnostics_sample_size": soft_edge_memory_diagnostics_sample_size,
        }
        for key, value in legacy_memory.items():
            if value is None:
                continue
            if key in memory and memory[key] != value:
                raise ValueError(
                    f"soft_edge_memory.{key} conflicts with deprecated flat option."
                )
            memory[key] = value
        memory_enabled = bool(memory.get("enabled", True))
        if memory_enabled and not use_prior_init:
            raise ValueError(
                "soft_edge_memory.enabled=true requires prior_init_scope != 'none'."
            )

        super().__init__(
            h0_init_scope=self.prior_init_scope,
            h0_node_key=prior_node_key,
            h0_edge_key=prior_edge_key,
            h0_node_mode=prior_node_mode,
            fallback_to_hamiltonian=False,
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
        self.use_soft_edge_memory = memory_enabled
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
                soft_edge_memory_num_slots=int(memory.get("num_slots", 64)),
                soft_edge_memory_num_heads=int(memory.get("num_heads", 4)),
                soft_edge_memory_head_dim=int(memory.get("head_dim", 16)),
                soft_edge_memory_temperature=float(memory.get("temperature", 1.0)),
                soft_edge_memory_dropout=float(memory.get("dropout", 0.0)),
                soft_edge_memory_gate_mode=str(memory.get("gate_mode", "deepseek")),
                soft_edge_memory_gate_bias=float(memory.get("gate_bias", 0.0)),
                soft_edge_memory_gate_eps=float(memory.get("gate_eps", 1e-6)),
                soft_edge_memory_zero_init_output=bool(
                    memory.get("zero_init_output", True)
                ),
                soft_edge_memory_input_norm=bool(memory.get("input_norm", True)),
                validate_inputs=prior_validate_inputs,
                soft_edge_memory_diagnostics_mode=(
                    memory.get("diagnostics_mode", "off")
                ),
                soft_edge_memory_diagnostics_sample_size=(
                    int(memory.get("diagnostics_sample_size", 1024))
                ),
                dtype=self.dtype,
                device=self.device,
            )


__all__ = ["LemMoEV3Prior", "P2MemoryInitLayer"]

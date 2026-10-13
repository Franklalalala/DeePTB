# SPDX-License-Identifier: LGPL-3.0-or-later
"""O(3)-safe soft external memory for edge features.

The module is inspired by external key/value attention, but deliberately reads
and writes only even scalar (``0e``) channels.  Attention weights are therefore
rotation/inversion invariant and ordinary dense parameters never mix tensor
components with ``l > 0``.  Subsequent equivariant layers may propagate the
retrieved scalar context into higher-order Hamiltonian channels.

The default gate adapts the public DeepSeek Engram demo's normalized
key/query similarity, signed square-root compression, and sigmoid.  Unlike the
LLM module, this continuous geometric memory performs no tokenization or hash
lookup.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from e3nn import o3


class EquivariantSoftEdgeMemory(nn.Module):
    """Multi-head soft KV memory acting only on ``0e`` edge channels."""

    def __init__(
        self,
        irreps: Union[str, o3.Irreps],
        *,
        num_slots: int = 64,
        num_heads: int = 4,
        head_dim: int = 16,
        temperature: float = 1.0,
        attention_dropout: float = 0.0,
        gate_mode: str = "deepseek",
        gate_bias: float = 0.0,
        gate_eps: float = 1e-6,
        zero_init_output: bool = True,
        use_input_norm: bool = True,
        validate_inputs: bool = False,
        diagnostics_mode: str = "off",
        diagnostics_sample_size: int = 1024,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.irreps = o3.Irreps(irreps)
        self.num_slots = int(num_slots)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.temperature = float(temperature)
        self.gate_mode = str(gate_mode).lower()
        self.gate_bias = float(gate_bias)
        self.gate_eps = float(gate_eps)
        self.validate_inputs = bool(validate_inputs)
        self.diagnostics_mode = str(diagnostics_mode).lower()
        self.diagnostics_sample_size = int(diagnostics_sample_size)
        if self.num_slots < 2:
            raise ValueError("soft edge memory requires num_slots >= 2.")
        if self.num_heads < 1 or self.head_dim < 1:
            raise ValueError("soft edge memory requires positive num_heads/head_dim.")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("soft edge memory temperature must be finite and positive.")
        if not 0.0 <= float(attention_dropout) < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")
        if self.gate_mode not in {"deepseek", "linear"}:
            raise ValueError("gate_mode must be 'deepseek' or 'linear'.")
        if not math.isfinite(self.gate_bias):
            raise ValueError("gate_bias must be finite.")
        if not math.isfinite(self.gate_eps) or self.gate_eps <= 0.0:
            raise ValueError("gate_eps must be finite and positive.")
        if self.diagnostics_mode not in {"off", "sampled", "full"}:
            raise ValueError("diagnostics_mode must be 'off', 'sampled', or 'full'.")
        if self.diagnostics_sample_size < 1:
            raise ValueError("diagnostics_sample_size must be positive.")

        scalar_indices = []
        for (_mul, ir), sl in zip(self.irreps, self.irreps.slices()):
            if ir.l == 0 and ir.p == 1:
                scalar_indices.extend(range(sl.start, sl.stop))
        if not scalar_indices:
            raise ValueError(
                f"soft edge memory requires at least one invariant 0e channel; got {self.irreps}."
            )
        self.register_buffer(
            "scalar_indices",
            torch.as_tensor(scalar_indices, dtype=torch.long, device=device),
        )
        self.scalar_dim = len(scalar_indices)
        memory_dim = self.num_heads * self.head_dim

        # LayerNorm over one scalar is identically zero and would make the
        # memory query independent of the edge.  Keep the valid scalar_dim=1
        # configuration conditional by using the identity in that corner.
        self.input_norm = (
            nn.LayerNorm(self.scalar_dim, dtype=dtype, device=device)
            if use_input_norm and self.scalar_dim > 1
            else nn.Identity()
        )
        self.query = nn.Linear(
            self.scalar_dim, memory_dim, bias=False, dtype=dtype, device=device
        )
        self.keys = nn.Parameter(
            torch.empty(
                self.num_heads,
                self.num_slots,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
        )
        self.values = nn.Parameter(torch.empty_like(self.keys))
        self.output = nn.Linear(
            memory_dim, self.scalar_dim, bias=True, dtype=dtype, device=device
        )
        if self.gate_mode == "deepseek":
            self.gate_query_weight = nn.Parameter(
                torch.ones(memory_dim, dtype=dtype, device=device)
            )
            self.gate_key_weight = nn.Parameter(
                torch.ones(memory_dim, dtype=dtype, device=device)
            )
            self.gate = None
        else:
            self.register_parameter("gate_query_weight", None)
            self.register_parameter("gate_key_weight", None)
            self.gate = nn.Linear(
                self.scalar_dim,
                self.scalar_dim,
                bias=True,
                dtype=dtype,
                device=device,
            )
        self.attention_dropout = nn.Dropout(float(attention_dropout))

        nn.init.normal_(self.keys, std=1.0 / math.sqrt(self.head_dim))
        nn.init.normal_(self.values, std=1.0 / math.sqrt(self.head_dim))
        nn.init.xavier_uniform_(self.query.weight)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, self.gate_bias)
        if zero_init_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            nn.init.xavier_uniform_(self.output.weight, gain=0.05)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        edge_features: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if edge_features.ndim != 2 or edge_features.shape[-1] != self.irreps.dim:
            raise ValueError(
                "soft edge memory expected [n_edge, irreps.dim] features, got "
                f"{tuple(edge_features.shape)} for irreps dim {self.irreps.dim}."
            )
        if torch.is_complex(edge_features):
            raise TypeError("soft edge memory currently supports non-SOC real features only.")
        if self.validate_inputs and not torch.isfinite(edge_features).all():
            raise ValueError("soft edge memory input contains NaN or infinity.")

        if edge_features.shape[0] == 0:
            diagnostics: Dict[str, torch.Tensor] = {}
            if return_attention or self.diagnostics_mode != "off":
                diagnostics.update(
                    {
                        "attention_entropy": edge_features.new_zeros(()),
                        "attention_max_probability": edge_features.new_zeros(()),
                        "gate_mean": edge_features.new_zeros(()),
                    }
                )
            if return_attention:
                diagnostics["attention"] = edge_features.new_empty(
                    (0, self.num_heads, self.num_slots)
                )
            return edge_features, diagnostics

        scalar_indices = self.scalar_indices.to(device=edge_features.device)
        scalars = edge_features.index_select(1, scalar_indices)
        normalized = self.input_norm(scalars)
        query = self.query(normalized).reshape(
            edge_features.shape[0], self.num_heads, self.head_dim
        )
        scores = torch.einsum("ehd,hsd->ehs", query, self.keys)
        scores = scores / (math.sqrt(self.head_dim) * self.temperature)
        attention = torch.softmax(scores, dim=-1)
        dropped_attention = self.attention_dropout(attention)
        retrieved_key = torch.einsum("ehs,hsd->ehd", attention, self.keys).reshape(
            edge_features.shape[0], -1
        )
        retrieved = torch.einsum(
            "ehs,hsd->ehd", dropped_attention, self.values
        ).reshape(edge_features.shape[0], -1)
        update = self.output(retrieved)
        if self.gate_mode == "deepseek":
            flat_query = query.reshape(edge_features.shape[0], -1)
            query_rms = flat_query.pow(2).mean(dim=-1, keepdim=True)
            key_rms = retrieved_key.pow(2).mean(dim=-1, keepdim=True)
            normed_query = flat_query * torch.rsqrt(query_rms + self.gate_eps)
            normed_key = retrieved_key * torch.rsqrt(key_rms + self.gate_eps)
            normed_query = normed_query * self.gate_query_weight
            normed_key = normed_key * self.gate_key_weight
            similarity = (normed_query * normed_key).sum(dim=-1, keepdim=True)
            similarity = similarity / math.sqrt(flat_query.shape[-1])
            compressed = (
                similarity.abs().clamp_min(self.gate_eps).sqrt()
                * similarity.sign()
            )
            gate = torch.sigmoid(compressed + self.gate_bias)
        else:
            gate = torch.sigmoid(self.gate(normalized))
        updated_scalars = scalars + gate * update

        output = edge_features.clone()
        output[:, scalar_indices] = updated_scalars

        diagnostics: Dict[str, torch.Tensor] = {}
        effective_diagnostics_mode = (
            "full" if return_attention else self.diagnostics_mode
        )
        if effective_diagnostics_mode != "off":
            if effective_diagnostics_mode == "sampled":
                stop = min(attention.shape[0], self.diagnostics_sample_size)
                diagnostic_attention = attention[:stop]
                diagnostic_gate = gate[:stop]
            else:
                diagnostic_attention = attention
                diagnostic_gate = gate
            # Keep reductions stable under autocast without forcing the full
            # memory read/write path to float32.
            diagnostic_attention = diagnostic_attention.float()
            diagnostic_gate = diagnostic_gate.float()
            eps = torch.finfo(diagnostic_attention.dtype).eps
            entropy = -(
                diagnostic_attention
                * diagnostic_attention.clamp_min(eps).log()
            ).sum(dim=-1)
            entropy = entropy / math.log(self.num_slots)
            diagnostics.update(
                {
                    "attention_entropy": entropy.mean(),
                    "attention_max_probability": diagnostic_attention.max(
                        dim=-1
                    ).values.mean(),
                    "gate_mean": diagnostic_gate.mean(),
                }
            )
        if return_attention:
            diagnostics["attention"] = attention
        return output, diagnostics


__all__ = ["EquivariantSoftEdgeMemory"]

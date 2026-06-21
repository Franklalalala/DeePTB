"""Late ordinary multiplicity mixing from hidden irreps to DeePTB RME irreps.

The head performs no angular-momentum product. Every output term receives
information only from input multiplicities carrying the identical ``(l, p)``
representation. Static and scalar-conditioned weights are shared across the
magnetic axis, so the exact output remains a valid irreps tensor for the
existing out-onehot tensor product and E3Hamiltonian path.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from e3nn import o3


class LateRMEExpansionNoCGHead(torch.nn.Module):
    """Create the OrbitalMapper RME layout only at the late output boundary."""

    performs_angular_coupling = False
    output_contract = "rme"
    uses_ict = False

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
        *,
        rank: int = 16,
        init: float = 0.0,
        condition: str = "scalar_0e",
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.rank = int(rank)
        self.dynamic_init = float(init)
        self.condition = str(condition).strip().lower()

        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if self.dynamic_init < 0.0:
            raise ValueError(f"init must be non-negative, got {init}.")
        if self.condition != "scalar_0e":
            raise ValueError(
                "late_rme_expansion_nocg supports only condition='scalar_0e', "
                f"got {condition!r}."
            )

        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device

        input_slices = self.irreps_in.slices()
        scalar_indices = []
        for term_slice, (_, ir) in zip(input_slices, self.irreps_in):
            if ir.l == 0 and ir.p == 1:
                scalar_indices.extend(range(term_slice.start, term_slice.stop))
        if not scalar_indices:
            raise ValueError(
                "late_rme_expansion_nocg requires at least one 0e input "
                f"multiplicity; irreps_in={self.irreps_in}."
            )
        self.register_buffer(
            "_scalar_indices",
            torch.tensor(scalar_indices, dtype=torch.long, device=device),
            persistent=False,
        )

        # One independently learned ordinary channel-mixing matrix per output
        # term. Input blocks are concatenated only when their exact irrep
        # matches that output term.
        self._input_term_indices = []
        self._weight_offsets = []
        static_weights = []
        weight_offset = 0
        for mul_out, ir_out in self.irreps_out:
            matching = [
                index
                for index, (_, ir_in) in enumerate(self.irreps_in)
                if ir_in == ir_out
            ]
            if not matching:
                raise ValueError(
                    "Every output irrep needs matching input channels; missing "
                    f"{ir_out} in irreps_in={self.irreps_in}."
                )
            mul_in = sum(self.irreps_in[index].mul for index in matching)
            weight_count = mul_out * mul_in
            self._input_term_indices.append(tuple(matching))
            self._weight_offsets.append(
                (weight_offset, weight_offset + weight_count, mul_out, mul_in)
            )
            weight_offset += weight_count

            weight = torch.empty(mul_out, mul_in, **factory_kwargs)
            torch.nn.init.normal_(weight, mean=0.0, std=1.0 / math.sqrt(mul_in))
            static_weights.append(weight.reshape(-1))

        self.num_path_weights = weight_offset
        self.static_weights = torch.nn.Parameter(torch.cat(static_weights, dim=0))
        self.condition_down = torch.nn.Linear(
            len(scalar_indices), self.rank, bias=True, **factory_kwargs
        )
        self.dynamic_up = torch.nn.Linear(
            self.rank, self.num_path_weights, bias=True, **factory_kwargs
        )
        self.reset_dynamic_parameters()

    def reset_dynamic_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.condition_down.weight, a=math.sqrt(5.0))
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Expected last dimension {self.irreps_in.dim}, got "
                f"{features.shape[-1]}."
            )

        condition = features.index_select(-1, self._scalar_indices)
        invariant = torch.nn.functional.silu(self.condition_down(condition))
        dynamic_weights = self.dynamic_up(invariant)
        input_slices = self.irreps_in.slices()

        outputs = []
        for term_index, ((mul_out, ir_out), offsets) in enumerate(
            zip(self.irreps_out, self._weight_offsets)
        ):
            blocks = []
            for input_index in self._input_term_indices[term_index]:
                mul_in, ir_in = self.irreps_in[input_index]
                block = features[..., input_slices[input_index]].reshape(
                    *features.shape[:-1], mul_in, ir_in.dim
                )
                blocks.append(block)
            input_block = torch.cat(blocks, dim=-2)

            start, stop, _, mul_in_total = offsets
            static = self.static_weights[start:stop].reshape(mul_out, mul_in_total)
            dynamic = dynamic_weights[..., start:stop].reshape(
                *features.shape[:-1], mul_out, mul_in_total
            )
            mixed = torch.einsum(
                "...oi,...im->...om", static + dynamic, input_block
            )
            outputs.append(
                mixed.reshape(*features.shape[:-1], mul_out * ir_out.dim)
            )

        return torch.cat(outputs, dim=-1)

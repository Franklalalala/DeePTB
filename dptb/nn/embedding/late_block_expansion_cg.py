"""QHFlow-style late Expansion decoder that writes padded AO blocks directly.

This is the only production output path in this patch that performs explicit
angular coupling. The coupling maps hidden irreps straight into AO shell-pair
blocks; no OrbitalMapper reduced-matrix-element vector is created or consumed.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Sequence, Union

import torch
from e3nn import o3


_ANGULAR_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


def _shell_l(shell: str) -> int:
    labels = re.findall(r"[A-Za-z]", str(shell))
    if len(labels) != 1 or labels[0].lower() not in _ANGULAR_L:
        raise ValueError(f"Unsupported AO shell label {shell!r}.")
    return _ANGULAR_L[labels[0].lower()]


class LateBlockExpansionCGHead(torch.nn.Module):
    """Decode hidden irreps directly into ``[max_norb, max_norb]`` AO blocks."""

    performs_angular_coupling = True
    output_contract = "ao_block"
    bypasses_rme = True
    bypasses_e3hamiltonian = True
    uses_ict = False

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        full_basis: Sequence[str],
        *,
        symmetrize: bool,
        rank: int = 16,
        init: float = 0.0,
        condition: str = "scalar_0e",
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.full_basis = tuple(str(shell) for shell in full_basis)
        self.symmetrize = bool(symmetrize)
        self.rank = int(rank)
        self.dynamic_init = float(init)
        self.condition = str(condition).strip().lower()

        if not self.full_basis:
            raise ValueError("full_basis must contain at least one AO shell.")
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if self.dynamic_init < 0.0:
            raise ValueError(f"init must be non-negative, got {init}.")
        if self.condition != "scalar_0e":
            raise ValueError(
                "late_block_expansion_cg supports only condition='scalar_0e', "
                f"got {condition!r}."
            )

        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device

        self._shell_specs = []
        offset = 0
        ao_terms = []
        for shell in self.full_basis:
            ell = _shell_l(shell)
            dim = 2 * ell + 1
            self._shell_specs.append((offset, offset + dim, ell))
            ao_terms.append((1, (ell, (-1) ** ell)))
            offset += dim
        self.max_norb = offset
        self.output_shape = (self.max_norb, self.max_norb)
        self.ao_irreps = o3.Irreps(ao_terms)

        scalar_indices = []
        for term_slice, (_, ir) in zip(self.irreps_in.slices(), self.irreps_in):
            if ir.l == 0 and ir.p == 1:
                scalar_indices.extend(range(term_slice.start, term_slice.stop))
        if not scalar_indices:
            raise ValueError(
                "late_block_expansion_cg requires at least one 0e input "
                f"multiplicity; irreps_in={self.irreps_in}."
            )
        self.register_buffer(
            "_scalar_indices",
            torch.tensor(scalar_indices, dtype=torch.long, device=device),
            persistent=False,
        )

        self._paths = []
        weight_offset = 0
        static_weights = []
        for input_index, (mul_in, ir_in) in enumerate(self.irreps_in):
            for row_index, (_, _, row_l) in enumerate(self._shell_specs):
                row_ir = o3.Irrep(row_l, (-1) ** row_l)
                for col_index, (_, _, col_l) in enumerate(self._shell_specs):
                    col_ir = o3.Irrep(col_l, (-1) ** col_l)
                    if ir_in not in row_ir * col_ir:
                        continue

                    buffer_name = f"_path_coefficient_{len(self._paths)}"
                    coefficient = o3.wigner_3j(
                        row_l, col_l, ir_in.l, dtype=dtype, device=device
                    ) * math.sqrt(2 * ir_in.l + 1)
                    self.register_buffer(buffer_name, coefficient, persistent=False)

                    next_offset = weight_offset + mul_in
                    self._paths.append(
                        (
                            input_index,
                            row_index,
                            col_index,
                            weight_offset,
                            next_offset,
                            buffer_name,
                        )
                    )
                    weight = torch.empty(mul_in, **factory_kwargs)
                    torch.nn.init.normal_(
                        weight, mean=0.0, std=1.0 / math.sqrt(mul_in)
                    )
                    static_weights.append(weight)
                    weight_offset = next_offset

        if not self._paths:
            raise ValueError(
                "No compatible hidden-irrep to AO-shell expansion paths were "
                f"found for irreps_in={self.irreps_in}, full_basis={self.full_basis}."
            )

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

        contributions = {
            (row_index, col_index): []
            for row_index in range(len(self._shell_specs))
            for col_index in range(len(self._shell_specs))
        }
        for (
            input_index,
            row_index,
            col_index,
            weight_start,
            weight_stop,
            buffer_name,
        ) in self._paths:
            mul_in, ir_in = self.irreps_in[input_index]
            input_block = features[..., input_slices[input_index]].reshape(
                *features.shape[:-1], mul_in, ir_in.dim
            )
            static = self.static_weights[weight_start:weight_stop]
            dynamic = dynamic_weights[..., weight_start:weight_stop]
            mixed = torch.einsum(
                "...u,...uk->...k", static + dynamic, input_block
            ) / float(mul_in)
            coefficient = getattr(self, buffer_name).to(
                dtype=features.dtype, device=features.device
            )
            block = torch.einsum("ijk,...k->...ij", coefficient, mixed)
            contributions[(row_index, col_index)].append(block)

        rows = []
        for row_index, (row_start, row_stop, _) in enumerate(self._shell_specs):
            row = []
            row_dim = row_stop - row_start
            for col_index, (col_start, col_stop, _) in enumerate(self._shell_specs):
                values = contributions[(row_index, col_index)]
                if values:
                    block = values[0]
                    for value in values[1:]:
                        block = block + value
                else:
                    block = features.new_zeros(
                        *features.shape[:-1], row_dim, col_stop - col_start
                    )
                row.append(block)
            rows.append(torch.cat(row, dim=-1))
        output = torch.cat(rows, dim=-2)

        if self.symmetrize:
            output = 0.5 * (output + output.transpose(-1, -2))
        return output

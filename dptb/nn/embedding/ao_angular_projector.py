"""Equivariant direct-AO decoder built from frozen angular projectors.

The runtime module consumes a complete AO-pair irrep coordinate system and
uses immutable, convention-checked projector buffers.  Projector generation
and JSON interchange live in :mod:`ao_projector_bank`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from e3nn import o3

try:
    from .ao_projector_bank import (
        _PRECOMPUTED_BACKEND,
        _REFERENCE_BACKEND,
        _projector_key,
        build_ao_decoder_irreps,
        export_projector_bank,
        load_projector_bank,
        normalize_projector_backend,
        reference_projector,
        required_projector_keys,
        shell_l,
    )
except ImportError:  # standalone package-level tests
    from ao_projector_bank import (
        _PRECOMPUTED_BACKEND,
        _REFERENCE_BACKEND,
        _projector_key,
        build_ao_decoder_irreps,
        export_projector_bank,
        load_projector_bank,
        normalize_projector_backend,
        reference_projector,
        required_projector_keys,
        shell_l,
    )


class AOProjectorPath(NamedTuple):
    input_index: int
    row_index: int
    col_index: int
    weight_start: int
    weight_stop: int
    projector_name: str


class AOAngularProjectorHead(torch.nn.Module):
    """Decode complete AO-pair irreps directly into padded AO blocks."""

    performs_angular_coupling = True
    output_contract = "ao_block"
    bypasses_rme = True
    bypasses_e3hamiltonian = True

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        full_basis: Sequence[str],
        *,
        symmetrize: bool,
        rank: int = 16,
        init: float = 0.0,
        condition: str = "scalar_0e",
        normalization: str = "e3hamiltonian",
        basis_convention: str = "deeptb_real_ao",
        projector_backend: str = _REFERENCE_BACKEND,
        projector_bank_path: Optional[Union[str, Path]] = None,
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
        self.normalization = str(normalization).strip().lower()
        self.basis_convention = str(basis_convention).strip().lower()
        self.projector_backend = normalize_projector_backend(projector_backend)
        self.uses_ict = self.projector_backend == _PRECOMPUTED_BACKEND
        self.projector_bank_path = (
            None if projector_bank_path in (None, "") else str(projector_bank_path)
        )

        if not self.full_basis:
            raise ValueError("full_basis must contain at least one AO shell.")
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if self.dynamic_init < 0.0:
            raise ValueError(f"init must be non-negative, got {init}.")
        if self.condition != "scalar_0e":
            raise ValueError(
                "AOAngularProjectorHead supports only condition='scalar_0e', "
                f"got {condition!r}."
            )
        if self.normalization != "e3hamiltonian":
            raise ValueError(
                "Only normalization='e3hamiltonian' is implemented; got "
                f"{normalization!r}."
            )
        if self.basis_convention != "deeptb_real_ao":
            raise ValueError(
                "Only basis_convention='deeptb_real_ao' is implemented; got "
                f"{basis_convention!r}."
            )
        if self.projector_backend == _PRECOMPUTED_BACKEND and not self.projector_bank_path:
            raise ValueError(
                "ao_projector_backend='precomputed' requires "
                "ao_projector_bank_path."
            )

        factory_kwargs = {}
        coefficient_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
            coefficient_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device
            coefficient_kwargs["device"] = device

        loaded_bank = None
        if self.projector_backend == _PRECOMPUTED_BACKEND:
            loaded_bank = load_projector_bank(
                self.projector_bank_path, self.full_basis
            )

        self._shell_specs = []
        ao_terms = []
        offset = 0
        for shell in self.full_basis:
            ell = shell_l(shell)
            dim = 2 * ell + 1
            self._shell_specs.append((offset, offset + dim, ell))
            ao_terms.append((1, (ell, (-1) ** ell)))
            offset += dim
        self.max_norb = offset
        self.output_shape = (offset, offset)
        self.ao_irreps = o3.Irreps(ao_terms)

        scalar_indices = []
        for term_slice, (_, ir) in zip(self.irreps_in.slices(), self.irreps_in):
            if ir.l == 0 and ir.p == 1:
                scalar_indices.extend(range(term_slice.start, term_slice.stop))
        if not scalar_indices:
            raise ValueError(
                "AOAngularProjectorHead requires at least one 0e input channel; "
                f"irreps_in={self.irreps_in}."
            )
        self.register_buffer(
            "_scalar_indices",
            torch.tensor(scalar_indices, dtype=torch.long, device=device),
            persistent=False,
        )

        projector_names: Dict[Tuple[int, int, int], str] = {}
        paths = []
        pair_path_counts: Dict[Tuple[int, int], int] = {}
        weight_offset = 0
        for input_index, (mul_in, ir_in) in enumerate(self.irreps_in):
            for row_index, (_, _, row_l) in enumerate(self._shell_specs):
                row_ir = o3.Irrep(row_l, (-1) ** row_l)
                for col_index, (_, _, col_l) in enumerate(self._shell_specs):
                    col_ir = o3.Irrep(col_l, (-1) ** col_l)
                    if ir_in not in row_ir * col_ir:
                        continue
                    key_tuple = (row_l, col_l, ir_in.l)
                    projector_name = projector_names.get(key_tuple)
                    if projector_name is None:
                        projector_name = f"_projector_{row_l}_{col_l}_{ir_in.l}"
                        if loaded_bank is None:
                            coefficient = reference_projector(
                                row_l, col_l, ir_in.l, **coefficient_kwargs
                            )
                        else:
                            coefficient = loaded_bank[
                                _projector_key(row_l, col_l, ir_in.l)
                            ].to(**coefficient_kwargs)
                        self.register_buffer(
                            projector_name, coefficient, persistent=True
                        )
                        projector_names[key_tuple] = projector_name
                    next_offset = weight_offset + int(mul_in)
                    paths.append(
                        AOProjectorPath(
                            input_index,
                            row_index,
                            col_index,
                            weight_offset,
                            next_offset,
                            projector_name,
                        )
                    )
                    pair = (row_index, col_index)
                    pair_path_counts[pair] = pair_path_counts.get(pair, 0) + int(mul_in)
                    weight_offset = next_offset

        self.paths = tuple(paths)
        self._projector_names = projector_names
        self.num_path_weights = weight_offset
        missing_pairs = [
            (i, j)
            for i in range(len(self._shell_specs))
            for j in range(len(self._shell_specs))
            if (i, j) not in pair_path_counts
        ]
        if missing_pairs:
            raise ValueError(
                "AO decoder irreps do not cover shell pairs "
                f"{missing_pairs}; irreps_in={self.irreps_in}."
            )

        static = torch.empty(self.num_path_weights, **factory_kwargs)
        for path in self.paths:
            fan_in = pair_path_counts[(path.row_index, path.col_index)]
            torch.nn.init.normal_(
                static[path.weight_start:path.weight_stop],
                mean=0.0,
                std=1.0 / math.sqrt(fan_in),
            )
        self.static_weights = torch.nn.Parameter(static)

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

    def projector_for_l(self, row_l: int, col_l: int, out_l: int) -> torch.Tensor:
        name = self._projector_names[(int(row_l), int(col_l), int(out_l))]
        return getattr(self, name)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Expected last dimension {self.irreps_in.dim}, got "
                f"{features.shape[-1]}."
            )

        invariant = features.index_select(-1, self._scalar_indices)
        latent = torch.nn.functional.silu(self.condition_down(invariant))
        path_weights = self.static_weights + self.dynamic_up(latent)
        input_slices = self.irreps_in.slices()
        contributions: Dict[Tuple[int, int], list[torch.Tensor]] = {
            (i, j): []
            for i in range(len(self._shell_specs))
            for j in range(len(self._shell_specs))
        }

        for path in self.paths:
            mul_in, ir_in = self.irreps_in[path.input_index]
            block = features[..., input_slices[path.input_index]].reshape(
                *features.shape[:-1], int(mul_in), ir_in.dim
            )
            weights = path_weights[..., path.weight_start:path.weight_stop]
            reduced = torch.einsum("...u,...um->...m", weights, block)
            projector = getattr(self, path.projector_name).to(
                dtype=features.dtype, device=features.device
            )
            contribution = torch.einsum("...m,ijm->...ij", reduced, projector)
            contributions[(path.row_index, path.col_index)].append(contribution)

        rows = []
        for row_index, (row_start, row_stop, _) in enumerate(self._shell_specs):
            row = []
            for col_index, (col_start, col_stop, _) in enumerate(self._shell_specs):
                values = contributions[(row_index, col_index)]
                block = values[0]
                for value in values[1:]:
                    block = block + value
                expected = (row_stop - row_start, col_stop - col_start)
                if block.shape[-2:] != expected:
                    raise RuntimeError(
                        f"Internal AO block shape {block.shape[-2:]} != {expected}."
                    )
                row.append(block)
            rows.append(torch.cat(row, dim=-1))
        output = torch.cat(rows, dim=-2)
        if self.symmetrize:
            output = 0.5 * (output + output.transpose(-1, -2))
        return output

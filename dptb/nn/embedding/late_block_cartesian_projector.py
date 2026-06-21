"""Direct AO decoder using irreducible Cartesian tensor projectors.

This head is the production candidate in the Cartesian A/B experiment.  It
uses fixed Cartesian-3j intertwiners for the *final* angular decoding step and
writes padded AO shell-pair blocks directly.  No OrbitalMapper RME vector is
created and E3Hamiltonian must not be called afterwards.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from e3nn import o3

from .cartesian_projector import (
    CartesianIrrepProduct,
    CartesianShellPairCoupling,
    ao_shell_layout,
)


class _DirectBlockPath(NamedTuple):
    input_index: int
    mul_in: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    pair_key: Tuple[int, int]
    shell_coupling_key: str
    weight_start: int
    weight_stop: int


class _ProductBlockPath(NamedTuple):
    left_index: int
    right_index: int
    mul_in: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    pair_key: Tuple[int, int]
    irrep_coupling_key: str
    shell_coupling_key: str
    weight_start: int
    weight_stop: int


def _even_scalar_indices(irreps: o3.Irreps) -> List[int]:
    indices: List[int] = []
    for term_slice, (_, ir) in zip(irreps.slices(), irreps):
        if ir.l == 0 and ir.p == 1:
            indices.extend(range(term_slice.start, term_slice.stop))
    return indices


class LateBlockCartesianProjectorHead(torch.nn.Module):
    """Decode final hidden irreps directly to ``[max_norb, max_norb]``.

    Direct paths consume matching hidden irreps.  Cartesian hidden x hidden
    product paths supply shell-pair channels absent from the alternating-parity
    LEM hidden layout.  Unlike Route A, these angular products are not followed
    by a second angular decoder: they are part of the final AO readout itself.
    """

    performs_angular_coupling = True
    output_contract = "ao_block"
    bypasses_rme = True
    bypasses_e3hamiltonian = True
    uses_ict = True

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        full_basis: Sequence[str],
        *,
        symmetrize: bool,
        rank: int = 16,
        init: float = 0.0,
        condition: str = "scalar_0e",
        product_scope: str = "all",
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.full_basis = tuple(str(shell) for shell in full_basis)
        self.layout = ao_shell_layout(self.full_basis)
        self.max_norb = self.layout[-1][1] if self.layout else 0
        self.output_shape = (self.max_norb, self.max_norb)
        self.symmetrize = bool(symmetrize)
        self.rank = int(rank)
        self.dynamic_init = float(init)
        self.condition = str(condition).strip().lower()
        self.product_scope = str(product_scope).strip().lower()
        self.uses_rme_bottleneck = False
        self.uses_cartesian_angular_decoder = True

        if not self.full_basis or self.max_norb <= 0:
            raise ValueError("full_basis must contain at least one AO shell.")
        if self.rank <= 0:
            raise ValueError("rme_fusion_rank must be positive, got {}.".format(rank))
        if self.dynamic_init < 0.0:
            raise ValueError(
                "rme_fusion_init must be non-negative, got {}.".format(init)
            )
        if self.condition != "scalar_0e":
            raise ValueError(
                "late_block_cartesian_projector supports only "
                "condition='scalar_0e', got {!r}.".format(condition)
            )
        if self.product_scope not in {"missing_only", "all"}:
            raise ValueError(
                "rme_cartesian_scope must be 'missing_only' or 'all', got {!r}.".format(
                    product_scope
                )
            )

        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device
        projector_dtype = dtype if dtype is not None else torch.get_default_dtype()
        projector_device = device if device is not None else torch.device("cpu")

        self._in_slices = tuple(self.irreps_in.slices())
        scalar_indices = _even_scalar_indices(self.irreps_in)
        if not scalar_indices:
            raise ValueError(
                "late_block_cartesian_projector requires at least one 0e input channel."
            )
        self.register_buffer(
            "_scalar_indices",
            torch.tensor(scalar_indices, dtype=torch.long, device=projector_device),
            persistent=False,
        )

        self.shell_couplings = torch.nn.ModuleDict()
        self.irrep_couplings = torch.nn.ModuleDict()
        self.direct_paths: List[_DirectBlockPath] = []
        self.product_paths: List[_ProductBlockPath] = []
        self.skipped_multiplicity_paths: List[Tuple[int, int, int, int, int]] = []
        pair_fan_in: Dict[Tuple[int, int], int] = {}
        missing_channels: List[str] = []
        weight_offset = 0

        for row_index, (row_start, row_stop, l_row) in enumerate(self.layout):
            for col_index, (col_start, col_stop, l_col) in enumerate(self.layout):
                pair_key = (row_index, col_index)
                pair_parity = (-1) ** (l_row + l_col)
                for l_out in range(abs(l_row - l_col), l_row + l_col + 1):
                    shell_key = "{}_{}_{}".format(l_row, l_col, l_out)
                    if shell_key not in self.shell_couplings:
                        self.shell_couplings[shell_key] = CartesianShellPairCoupling(
                            l_row,
                            l_col,
                            l_out,
                            dtype=projector_dtype,
                            device=projector_device,
                        )

                    direct_indices = [
                        index
                        for index, (_, ir_in) in enumerate(self.irreps_in)
                        if ir_in.l == l_out and ir_in.p == pair_parity
                    ]
                    channel_has_path = False
                    for input_index in direct_indices:
                        mul_in = int(self.irreps_in[input_index].mul)
                        self.direct_paths.append(
                            _DirectBlockPath(
                                input_index,
                                mul_in,
                                row_start,
                                row_stop,
                                col_start,
                                col_stop,
                                pair_key,
                                shell_key,
                                weight_offset,
                                weight_offset + mul_in,
                            )
                        )
                        weight_offset += mul_in
                        pair_fan_in[pair_key] = pair_fan_in.get(pair_key, 0) + mul_in
                        channel_has_path = True

                    add_products = self.product_scope == "all" or not direct_indices
                    if add_products:
                        for left_index, (mul_left_raw, ir_left) in enumerate(
                            self.irreps_in
                        ):
                            for right_index, (mul_right_raw, ir_right) in enumerate(
                                self.irreps_in
                            ):
                                if ir_left.p * ir_right.p != pair_parity:
                                    continue
                                if not (
                                    abs(ir_left.l - ir_right.l)
                                    <= l_out
                                    <= ir_left.l + ir_right.l
                                ):
                                    continue
                                mul_left = int(mul_left_raw)
                                mul_right = int(mul_right_raw)
                                if mul_left != mul_right:
                                    self.skipped_multiplicity_paths.append(
                                        (
                                            row_index,
                                            col_index,
                                            l_out,
                                            left_index,
                                            right_index,
                                        )
                                    )
                                    continue

                                irrep_key = "{}_{}_{}".format(
                                    ir_left.l, ir_right.l, l_out
                                )
                                if irrep_key not in self.irrep_couplings:
                                    self.irrep_couplings[
                                        irrep_key
                                    ] = CartesianIrrepProduct(
                                        ir_left.l,
                                        ir_right.l,
                                        l_out,
                                        dtype=projector_dtype,
                                        device=projector_device,
                                    )
                                self.product_paths.append(
                                    _ProductBlockPath(
                                        left_index,
                                        right_index,
                                        mul_left,
                                        row_start,
                                        row_stop,
                                        col_start,
                                        col_stop,
                                        pair_key,
                                        irrep_key,
                                        shell_key,
                                        weight_offset,
                                        weight_offset + mul_left,
                                    )
                                )
                                weight_offset += mul_left
                                pair_fan_in[pair_key] = (
                                    pair_fan_in.get(pair_key, 0) + mul_left
                                )
                                channel_has_path = True

                    if not channel_has_path:
                        missing_channels.append(
                            "shell[{},{}] {}{}".format(
                                row_index,
                                col_index,
                                l_out,
                                "e" if pair_parity == 1 else "o",
                            )
                        )

        if missing_channels:
            raise ValueError(
                "late_block_cartesian_projector has no equivariant decoder path "
                "for {}. Mismatched uuw multiplicities are never silently "
                "truncated. irreps_in={}.".format(missing_channels, self.irreps_in)
            )
        if weight_offset <= 0:
            raise ValueError("late_block_cartesian_projector constructed no paths.")

        if self.product_paths:
            self.left = o3.Linear(
                self.irreps_in,
                self.irreps_in,
                shared_weights=True,
                internal_weights=True,
                biases=True,
            )
            self.right = o3.Linear(
                self.irreps_in,
                self.irreps_in,
                shared_weights=True,
                internal_weights=True,
                biases=True,
            )
            if dtype is not None or device is not None:
                self.left = self.left.to(dtype=dtype, device=device)
                self.right = self.right.to(dtype=dtype, device=device)

        self.weight_numel = int(weight_offset)
        self._pair_fan_in = pair_fan_in
        self.static_weights = torch.nn.Parameter(
            torch.empty(self.weight_numel, **factory_kwargs)
        )
        self.condition_down = torch.nn.Linear(
            len(scalar_indices), self.rank, bias=True, **factory_kwargs
        )
        self.dynamic_up = torch.nn.Linear(
            self.rank, self.weight_numel, bias=True, **factory_kwargs
        )
        self.reset_parameters()

    @property
    def paths(self) -> Tuple[object, ...]:
        return tuple(self.direct_paths) + tuple(self.product_paths)

    @property
    def coverage_report(self) -> dict:
        return {
            "ao_shells": len(self.layout),
            "max_norb": self.max_norb,
            "direct_paths": len(self.direct_paths),
            "product_paths": len(self.product_paths),
            "skipped_multiplicity_paths": len(self.skipped_multiplicity_paths),
            "product_scope": self.product_scope,
            "uses_rme_bottleneck": False,
        }

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.condition_down.weight, a=math.sqrt(5.0))
        if self.condition_down.bias is not None:
            torch.nn.init.zeros_(self.condition_down.bias)

        with torch.no_grad():
            for path in self.paths:
                fan_in = self._pair_fan_in[path.pair_key]
                torch.nn.init.normal_(
                    self.static_weights[path.weight_start : path.weight_stop],
                    mean=0.0,
                    std=1.0 / math.sqrt(max(1, fan_in)),
                )

        if self.dynamic_init == 0.0:
            torch.nn.init.zeros_(self.dynamic_up.weight)
        else:
            torch.nn.init.normal_(
                self.dynamic_up.weight, mean=0.0, std=self.dynamic_init
            )
        if self.dynamic_up.bias is not None:
            torch.nn.init.zeros_(self.dynamic_up.bias)

    def _weights(self, flat: torch.Tensor) -> torch.Tensor:
        condition = flat.index_select(1, self._scalar_indices.to(flat.device))
        latent = torch.nn.functional.silu(self.condition_down(condition))
        return self.static_weights.unsqueeze(0) + self.dynamic_up(latent)

    @staticmethod
    def _add_block(
        blocks: torch.Tensor,
        path: Union[_DirectBlockPath, _ProductBlockPath],
        contribution: torch.Tensor,
    ) -> None:
        row_slice = slice(path.row_start, path.row_stop)
        col_slice = slice(path.col_start, path.col_stop)
        blocks[:, row_slice, col_slice] += contribution

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                "Expected last dimension {}, got {}.".format(
                    self.irreps_in.dim, features.shape[-1]
                )
            )

        leading = features.shape[:-1]
        flat = features.reshape(-1, self.irreps_in.dim)
        weights = self._weights(flat)
        blocks = flat.new_zeros(flat.shape[0], self.max_norb, self.max_norb)

        for path in self.direct_paths:
            ir = self.irreps_in[path.input_index].ir
            source = flat[:, self._in_slices[path.input_index]].reshape(
                flat.shape[0], path.mul_in, ir.dim
            )
            projected = self.shell_couplings[path.shell_coupling_key](source)
            contribution = torch.einsum(
                "nu,nuij->nij",
                weights[:, path.weight_start : path.weight_stop],
                projected,
            )
            self._add_block(blocks, path, contribution)

        if self.product_paths:
            left = self.left(flat)
            right = self.right(flat)
            for path in self.product_paths:
                ir_left = self.irreps_in[path.left_index].ir
                ir_right = self.irreps_in[path.right_index].ir
                x = left[:, self._in_slices[path.left_index]].reshape(
                    flat.shape[0], path.mul_in, ir_left.dim
                )
                y = right[:, self._in_slices[path.right_index]].reshape(
                    flat.shape[0], path.mul_in, ir_right.dim
                )
                coupled = self.irrep_couplings[path.irrep_coupling_key](x, y)
                projected = self.shell_couplings[path.shell_coupling_key](coupled)
                contribution = torch.einsum(
                    "nu,nuij->nij",
                    weights[:, path.weight_start : path.weight_stop],
                    projected,
                )
                self._add_block(blocks, path, contribution)

        if self.symmetrize:
            blocks = 0.5 * (blocks + blocks.transpose(-1, -2))
        return blocks.reshape(*leading, self.max_norb, self.max_norb)

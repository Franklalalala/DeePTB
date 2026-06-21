"""Late RME head with controlled irreducible-Cartesian completion paths.

The safe part of this head is ordinary same-irrep multiplicity mixing.  When an
OrbitalMapper output irrep is absent from the final LEM hidden representation,
the default ``missing_only`` policy completes that irrep through a hidden x
hidden Cartesian-3j product.  The result is still the exact RME layout consumed
by DeePTB's E3Hamiltonian.

Cartesian-3j is a change-of-basis form of the same SO(3) intertwiner as
Wigner-3j.  Consequently the product branch is intentionally exposed as an
ablation/control: it can overlap semantically with E3Hamiltonian's later RME to
AO angular assembly.  It is not presented as a way to remove that risk.
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Tuple, Union

import torch
from e3nn import o3

from .cartesian_projector import CartesianIrrepProduct


class _DirectPath(NamedTuple):
    input_index: int
    output_index: int
    mul_in: int
    mul_out: int
    weight_start: int
    weight_stop: int


class _ProductPath(NamedTuple):
    left_index: int
    right_index: int
    output_index: int
    mul_in: int
    mul_out: int
    coupling_key: str
    weight_start: int
    weight_stop: int


def _even_scalar_indices(irreps: o3.Irreps) -> List[int]:
    indices: List[int] = []
    for term_slice, (_, ir) in zip(irreps.slices(), irreps):
        if ir.l == 0 and ir.p == 1:
            indices.extend(range(term_slice.start, term_slice.stop))
    return indices


class LateRMECartesianHybridHead(torch.nn.Module):
    """Create the OrbitalMapper RME tensor only at the late output boundary.

    Parameters
    ----------
    irreps_in
        Final hidden irreps from LEM.
    irreps_out
        Exact ``idp.orbpair_irreps`` layout.
    product_scope
        ``"missing_only"`` uses Cartesian products only for output irrep types
        that have no ordinary same-irrep source. ``"all"`` also adds product
        paths to directly covered output types and is the stronger ablation.
    """

    performs_angular_coupling = True
    output_contract = "rme"
    uses_ict = True

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
        *,
        rank: int = 16,
        init: float = 0.0,
        condition: str = "scalar_0e",
        product_scope: str = "missing_only",
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.rank = int(rank)
        self.dynamic_init = float(init)
        self.condition = str(condition).strip().lower()
        self.product_scope = str(product_scope).strip().lower()
        self.output_is_rme = True
        self.uses_cartesian_angular_coupling = True

        if self.rank <= 0:
            raise ValueError("rme_fusion_rank must be positive, got {}.".format(rank))
        if self.dynamic_init < 0.0:
            raise ValueError(
                "rme_fusion_init must be non-negative, got {}.".format(init)
            )
        if self.condition != "scalar_0e":
            raise ValueError(
                "late_rme_cartesian_hybrid supports only condition='scalar_0e', "
                "got {!r}.".format(condition)
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
        self._out_slices = tuple(self.irreps_out.slices())
        scalar_indices = _even_scalar_indices(self.irreps_in)
        if not scalar_indices:
            raise ValueError(
                "late_rme_cartesian_hybrid requires at least one 0e input channel."
            )
        self.register_buffer(
            "_scalar_indices",
            torch.tensor(scalar_indices, dtype=torch.long, device=projector_device),
            persistent=False,
        )

        # Separate equivariant projections keep the two product factors learned
        # and avoid the symmetry collapse x*x can impose on odd coupled channels.
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

        self.couplings = torch.nn.ModuleDict()
        self.direct_paths: List[_DirectPath] = []
        self.product_paths: List[_ProductPath] = []
        self.skipped_multiplicity_paths: List[Tuple[int, int, int]] = []
        output_fan_in = [0 for _ in self.irreps_out]
        weight_offset = 0
        missing_outputs: List[str] = []

        for output_index, (mul_out_raw, ir_out) in enumerate(self.irreps_out):
            mul_out = int(mul_out_raw)
            direct_indices = [
                index
                for index, (_, ir_in) in enumerate(self.irreps_in)
                if ir_in == ir_out
            ]
            for input_index in direct_indices:
                mul_in = int(self.irreps_in[input_index].mul)
                path_numel = mul_in * mul_out
                self.direct_paths.append(
                    _DirectPath(
                        input_index,
                        output_index,
                        mul_in,
                        mul_out,
                        weight_offset,
                        weight_offset + path_numel,
                    )
                )
                weight_offset += path_numel
                output_fan_in[output_index] += mul_in

            add_products = self.product_scope == "all" or not direct_indices
            if add_products:
                for left_index, (mul_left_raw, ir_left) in enumerate(self.irreps_in):
                    for right_index, (mul_right_raw, ir_right) in enumerate(
                        self.irreps_in
                    ):
                        mul_left = int(mul_left_raw)
                        mul_right = int(mul_right_raw)
                        if ir_left.p * ir_right.p != ir_out.p:
                            continue
                        if not (
                            abs(ir_left.l - ir_right.l)
                            <= ir_out.l
                            <= ir_left.l + ir_right.l
                        ):
                            continue
                        if mul_left != mul_right:
                            self.skipped_multiplicity_paths.append(
                                (left_index, right_index, output_index)
                            )
                            continue

                        coupling_key = "{}_{}_{}".format(
                            ir_left.l, ir_right.l, ir_out.l
                        )
                        if coupling_key not in self.couplings:
                            self.couplings[coupling_key] = CartesianIrrepProduct(
                                ir_left.l,
                                ir_right.l,
                                ir_out.l,
                                dtype=projector_dtype,
                                device=projector_device,
                            )
                        path_numel = mul_left * mul_out
                        self.product_paths.append(
                            _ProductPath(
                                left_index,
                                right_index,
                                output_index,
                                mul_left,
                                mul_out,
                                coupling_key,
                                weight_offset,
                                weight_offset + path_numel,
                            )
                        )
                        weight_offset += path_numel
                        output_fan_in[output_index] += mul_left

            if output_fan_in[output_index] == 0:
                missing_outputs.append("{}:{}x{}".format(output_index, mul_out, ir_out))

        if missing_outputs:
            raise ValueError(
                "late_rme_cartesian_hybrid has no equivariant path for output "
                "terms {}. irreps_in={}. Mismatched uuw multiplicities are never "
                "silently truncated.".format(missing_outputs, self.irreps_in)
            )
        if weight_offset <= 0:
            raise ValueError("late_rme_cartesian_hybrid constructed no paths.")

        self.weight_numel = int(weight_offset)
        self._output_fan_in = tuple(output_fan_in)
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
        covered = {
            path.output_index for path in self.direct_paths
        } | {path.output_index for path in self.product_paths}
        return {
            "output_terms": len(self.irreps_out),
            "covered_output_terms": len(covered),
            "direct_paths": len(self.direct_paths),
            "product_paths": len(self.product_paths),
            "skipped_multiplicity_paths": len(self.skipped_multiplicity_paths),
            "product_scope": self.product_scope,
        }

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.condition_down.weight, a=math.sqrt(5.0))
        if self.condition_down.bias is not None:
            torch.nn.init.zeros_(self.condition_down.bias)

        with torch.no_grad():
            for path in self.paths:
                fan_in = self._output_fan_in[path.output_index]
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
        output = flat.new_zeros(flat.shape[0], self.irreps_out.dim)

        for path in self.direct_paths:
            ir = self.irreps_in[path.input_index].ir
            source = flat[:, self._in_slices[path.input_index]].reshape(
                flat.shape[0], path.mul_in, ir.dim
            )
            path_weight = weights[:, path.weight_start : path.weight_stop].reshape(
                flat.shape[0], path.mul_in, path.mul_out
            )
            mixed = torch.einsum("nui,nuw->nwi", source, path_weight)
            output[:, self._out_slices[path.output_index]] += mixed.reshape(
                flat.shape[0], -1
            )

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
                coupled = self.couplings[path.coupling_key](x, y)
                path_weight = weights[
                    :, path.weight_start : path.weight_stop
                ].reshape(flat.shape[0], path.mul_in, path.mul_out)
                mixed = torch.einsum("nuk,nuw->nwk", coupled, path_weight)
                output[:, self._out_slices[path.output_index]] += mixed.reshape(
                    flat.shape[0], -1
                )

        return output.reshape(*leading, self.irreps_out.dim)

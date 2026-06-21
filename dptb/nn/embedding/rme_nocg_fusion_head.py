"""Equivariant late fusion in DeePTB's existing RME output space.

The module keeps the legacy equivariant linear projection as its baseline and
adds scalar-conditioned residual modulation.  Each output-irrep multiplicity
receives one scale shared by all of its magnetic components.  Additive shifts
are restricted to even scalars.  Consequently this head does not perform a
second angular-momentum decomposition or AO-block assembly.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from e3nn import o3


_LEGACY_MODE = "legacy_linear"
_FUSION_MODE = "rme_nocg_fusion"
_LATE_RME_NOCG_MODE = "late_rme_expansion_nocg"
_LATE_RME_ICT_MODE = "late_rme_cartesian_hybrid"
_BLOCK_NATIVE_MODE = "block_native_linear"
_LATE_BLOCK_WIGNER_MODE = "late_block_expansion_cg"
_LATE_BLOCK_ICT_MODE = "late_block_cartesian_projector"
_DIRECT_AO_PROJECTOR_MODE = "direct_ao_projector"


def normalize_rme_head_mode(mode: Optional[str]) -> str:
    """Normalize and validate the output-head mode."""
    normalized = (mode or _LEGACY_MODE).strip().lower()
    aliases = {
        "legacy": _LEGACY_MODE,
        "linear": _LEGACY_MODE,
        "nocg": _FUSION_MODE,
        "rme_fusion": _FUSION_MODE,
        "late_nocg": _LATE_RME_NOCG_MODE,
        "late_rme_nocg": _LATE_RME_NOCG_MODE,
        "late_rme_ict_hybrid": _LATE_RME_ICT_MODE,
        "late_rme_cartesian": _LATE_RME_ICT_MODE,
        "block_native": _BLOCK_NATIVE_MODE,
        "block_linear": _BLOCK_NATIVE_MODE,
        "expansion_cg": _LATE_BLOCK_WIGNER_MODE,
        "late_block_wigner": _LATE_BLOCK_WIGNER_MODE,
        "late_block_ict_projector": _LATE_BLOCK_ICT_MODE,
        "late_block_cartesian": _LATE_BLOCK_ICT_MODE,
        "ao_projector": _DIRECT_AO_PROJECTOR_MODE,
        "direct_ao": _DIRECT_AO_PROJECTOR_MODE,
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        _LEGACY_MODE,
        _FUSION_MODE,
        _LATE_RME_NOCG_MODE,
        _LATE_RME_ICT_MODE,
        _BLOCK_NATIVE_MODE,
        _LATE_BLOCK_WIGNER_MODE,
        _LATE_BLOCK_ICT_MODE,
        _DIRECT_AO_PROJECTOR_MODE,
    }
    if normalized not in allowed:
        raise ValueError(
            f"rme_head_mode must be one of {sorted(allowed)}, got {mode!r}."
        )
    return normalized


class RMENoCGFusionHead(torch.nn.Module):
    """Legacy linear RME projection plus equivariant scalar conditioning.

    Parameters
    ----------
    irreps_in
        Final LEM feature representation.
    irreps_out
        Exact ``OrbitalMapper.orbpair_irreps`` representation.
    rank
        Bottleneck rank used to predict residual scales and scalar shifts.
    init
        Standard deviation for the final residual projections.  ``0.0`` makes
        the initialized module exactly equal to its ``legacy`` linear branch.
    condition
        Currently only ``"scalar_0e"`` is supported.
    """

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
        legacy: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.rank = int(rank)
        self.fusion_init = float(init)
        self.condition = str(condition).strip().lower()

        if self.rank <= 0:
            raise ValueError(f"rme_fusion_rank must be positive, got {rank}.")
        if self.fusion_init < 0.0:
            raise ValueError(
                f"rme_fusion_init must be non-negative, got {init}."
            )
        if self.condition != "scalar_0e":
            raise ValueError(
                "rme_fusion_condition currently supports only 'scalar_0e', "
                f"got {condition!r}."
            )

        self.legacy = legacy if legacy is not None else o3.Linear(
            self.irreps_in,
            self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        legacy_in = o3.Irreps(getattr(self.legacy, "irreps_in", self.irreps_in))
        legacy_out = o3.Irreps(getattr(self.legacy, "irreps_out", self.irreps_out))
        if legacy_in != self.irreps_in or legacy_out != self.irreps_out:
            raise ValueError(
                "Provided legacy projection has incompatible irreps: "
                f"{legacy_in} -> {legacy_out}, expected "
                f"{self.irreps_in} -> {self.irreps_out}."
            )

        scalar_indices = []
        for term_slice, (_, ir) in zip(self.irreps_in.slices(), self.irreps_in):
            if ir.l == 0 and ir.p == 1:
                scalar_indices.extend(range(term_slice.start, term_slice.stop))
        if not scalar_indices:
            raise ValueError(
                "rme_nocg_fusion requires at least one even-scalar (0e) input "
                f"channel; irreps_in={self.irreps_in}."
            )
        self.register_buffer(
            "_scalar_indices",
            torch.tensor(scalar_indices, dtype=torch.long),
            persistent=False,
        )

        self._term_gate_offsets = []
        self._scalar_shift_offsets = []
        gate_offset = 0
        shift_offset = 0
        for mul, ir in self.irreps_out:
            self._term_gate_offsets.append((gate_offset, gate_offset + mul))
            gate_offset += mul
            if ir.l == 0 and ir.p == 1:
                self._scalar_shift_offsets.append(
                    (shift_offset, shift_offset + mul)
                )
                shift_offset += mul
            else:
                self._scalar_shift_offsets.append(None)

        self.num_condition_scalars = len(scalar_indices)
        self.num_output_multiplicities = gate_offset
        self.num_scalar_output_multiplicities = shift_offset

        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device

        self.condition_down = torch.nn.Linear(
            self.num_condition_scalars,
            self.rank,
            bias=True,
            **factory_kwargs,
        )
        self.scale_up = torch.nn.Linear(
            self.rank,
            self.num_output_multiplicities,
            bias=True,
            **factory_kwargs,
        )
        self.shift_up = (
            torch.nn.Linear(
                self.rank,
                self.num_scalar_output_multiplicities,
                bias=True,
                **factory_kwargs,
            )
            if self.num_scalar_output_multiplicities > 0
            else None
        )

        if dtype is not None or device is not None:
            self.legacy = self.legacy.to(device=device, dtype=dtype)
        self.reset_fusion_parameters()
        self._legacy_load_prefix = None
        self.register_load_state_dict_post_hook(self._legacy_load_post_hook)

    def reset_fusion_parameters(self) -> None:
        """Initialize only the residual branch; preserve legacy initialization."""
        torch.nn.init.kaiming_uniform_(self.condition_down.weight, a=5**0.5)
        if self.condition_down.bias is not None:
            torch.nn.init.zeros_(self.condition_down.bias)

        final_layers = [self.scale_up]
        if self.shift_up is not None:
            final_layers.append(self.shift_up)
        for layer in final_layers:
            if self.fusion_init == 0.0:
                torch.nn.init.zeros_(layer.weight)
            else:
                torch.nn.init.normal_(
                    layer.weight, mean=0.0, std=self.fusion_init
                )
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Permit a legacy Linear checkpoint to initialize the baseline branch.
        legacy_format = prefix + "weight" in state_dict
        for name in ("weight", "bias", "output_mask"):
            old_key = prefix + name
            new_key = prefix + "legacy." + name
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if legacy_format:
            self._legacy_load_prefix = prefix

    def _legacy_load_post_hook(self, module, incompatible_keys) -> None:
        prefix = self._legacy_load_prefix
        if prefix is None:
            return
        residual_keys = [
            prefix + key
            for key in self.state_dict().keys()
            if not key.startswith("legacy.")
        ]
        for key in residual_keys:
            while key in incompatible_keys.missing_keys:
                incompatible_keys.missing_keys.remove(key)
        self._legacy_load_prefix = None

    def residual_parameters(self):
        """Yield parameters added by the fusion branch, excluding ``legacy``."""
        yield from self.condition_down.parameters()
        yield from self.scale_up.parameters()
        if self.shift_up is not None:
            yield from self.shift_up.parameters()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Expected last dimension {self.irreps_in.dim}, got "
                f"{features.shape[-1]}."
            )

        baseline = self.legacy(features)
        condition = features.index_select(-1, self._scalar_indices)
        latent = torch.nn.functional.silu(self.condition_down(condition))
        scales = self.scale_up(latent)
        scalar_shifts = self.shift_up(latent) if self.shift_up is not None else None

        blocks = []
        for term_index, ((mul, ir), out_slice) in enumerate(
            zip(self.irreps_out, self.irreps_out.slices())
        ):
            block = baseline[..., out_slice].reshape(
                *baseline.shape[:-1], mul, ir.dim
            )
            gate_start, gate_stop = self._term_gate_offsets[term_index]
            scale = scales[..., gate_start:gate_stop].unsqueeze(-1)
            block = block * (1.0 + scale)

            shift_offsets = self._scalar_shift_offsets[term_index]
            if shift_offsets is not None:
                assert scalar_shifts is not None
                shift_start, shift_stop = shift_offsets
                shift = scalar_shifts[..., shift_start:shift_stop].unsqueeze(-1)
                block = block + shift

            blocks.append(block.reshape(*baseline.shape[:-1], mul * ir.dim))

        return torch.cat(blocks, dim=-1)

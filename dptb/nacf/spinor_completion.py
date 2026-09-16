"""Complete SOC H from a real uu residual and a full complex NACF prior.

The NextHAM nonmagnetic residual contract is delta_uu.real = delta_dd.real.
Spin-flip and imaginary components are supplied by the geometry-only prior.
The trained model need not predict the eight-channel spinor representation.
"""
from __future__ import annotations

import torch
from torch import nn


class SOCUURealCompletion(nn.Module):
    """Map compact uu-real model features to a full SOC feature canvas.

    Both mappers must share basis and orbital-pair order semantics. Mapping is
    compiled by named orbital pairs, rather than assuming a global 8x reshape.
    Full features may use eight real channels or four complex channels per
    spatial orbital pair. No CPU transfers occur in extraction or completion.
    """

    def __init__(self, compact_mapper, full_mapper, *, device=None):
        super().__init__()
        if not (compact_mapper.has_soc and compact_mapper.nextham_uureal_mask):
            raise ValueError('model mapper must be compact SOC uu-real')
        if not full_mapper.has_soc or full_mapper.nextham_uureal_mask:
            raise ValueError('prior mapper must contain the full SOC channels')
        if compact_mapper.basis != full_mapper.basis:
            raise ValueError('compact and full SOC AO bases differ')
        compact_mapper.get_orbpair_maps()
        full_mapper.get_orbpair_maps()
        self.compact_width = int(compact_mapper.reduced_matrix_element)
        self.full_width = int(full_mapper.reduced_matrix_element)
        self.real_channels = bool(full_mapper.soc_complex_doubling)
        factor = 8 if self.real_channels else 4
        uu = torch.full((self.compact_width,), -1, dtype=torch.long)
        dd = torch.full_like(uu, -1)
        if set(compact_mapper.orbpair_maps) != set(full_mapper.orbpair_maps):
            raise ValueError('compact and full SOC orbital pairs differ')
        for pair, small in compact_mapper.orbpair_maps.items():
            large = full_mapper.orbpair_maps[pair]
            count = small.stop - small.start
            if large.stop - large.start != factor * count:
                raise ValueError('SOC orbital-pair channel width mismatch: '+pair)
            uu[small] = torch.arange(large.start, large.start + count)
            dd[small] = torch.arange(large.start + 3*count, large.start + 4*count)
        if self.full_width != factor*self.compact_width or (uu < 0).any() or (dd < 0).any():
            raise ValueError('incomplete SOC feature mapping')
        self.register_buffer('uu_indices', uu.to(device))
        self.register_buffer('dd_indices', dd.to(device))

    def _check_full(self, prior):
        if prior.shape[-1] != self.full_width:
            raise ValueError('full SOC prior feature width differs from mapper')
        if prior.is_complex() == self.real_channels:
            raise ValueError('full SOC dtype differs from real/complex channel layout')
        if prior.device != self.uu_indices.device:
            raise ValueError('SOC mapping and features must share a device')

    def extract_prior(self, full_prior):
        """Return the real uu NACF conditioning features for the model."""
        self._check_full(full_prior)
        return full_prior.index_select(-1, self.uu_indices).real

    def forward(self, full_prior, uu_real_residual):
        """Add delta to uu.real and dd.real once; preserve every other entry."""
        self._check_full(full_prior)
        if uu_real_residual.is_complex():
            raise ValueError('uu-real residual must have a real dtype')
        if uu_real_residual.shape != (*full_prior.shape[:-1], self.compact_width):
            raise ValueError('uu-real residual shape differs from full SOC prior')
        if uu_real_residual.device != full_prior.device:
            raise ValueError('SOC prior and residual must share a device')
        result = full_prior.clone()
        delta = uu_real_residual.to(full_prior.dtype)
        result.index_add_(-1, self.uu_indices, delta)
        result.index_add_(-1, self.dd_indices, delta)
        return result

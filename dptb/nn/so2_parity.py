"""O(3) selection rules at the SO(2) weight-bank boundary.

For a channel (l, p), q = p*(-1)**l is its reflection signature relative
to a natural-parity spherical harmonic. A connects equal q, B opposite q.
At m=0 only A exists and only q=+1 can have a bias. This is equivalent to
p_out = p_in*(-1)**l_filter and the CG even/odd A/B selection rule.
"""
import torch
from e3nn import o3


def normalize_so2_parity(mode):
    if mode not in ("none", "enforce"):
        raise ValueError("so2_parity must be 'none' or 'enforce'")
    return mode


def parity_masks(irreps_in, irreps_out, m):
    def signatures(irreps):
        return torch.tensor([ir.p * (-1) ** ir.l for mul, ir in o3.Irreps(irreps)
                             if ir.l >= m for _ in range(mul)], dtype=torch.int8)
    qi, qo = signatures(irreps_in), signatures(irreps_out)
    a = qo[:, None] == qi[None, :]
    return (a if m == 0 else torch.cat((a, ~a), dim=0)), (qo == 1 if m == 0 else None)


class ParityWeightMixin:
    """Expose differentiably masked tensors, keeping raw checkpoint/optimizer leaves.

    All backends (including direct CUDA bank readers) see the same mask. Buffers
    are derived from irreps and non-persistent; state_dict names/shapes stay intact.
    No tensor operation is inserted when the option is disabled. Never cache the
    masked bank across forwards: optimizer updates and autograd must remain live.
    """

    def set_parity_masks(self, weight, bias):
        parameter = next(self.parameters())
        self.register_buffer("_parity_weight_mask", weight.to(parameter.device), persistent=False)
        self.register_buffer("_parity_bias_mask", None if bias is None else bias.to(parameter.device),
                             persistent=False)

    def _parity_value(self, name, value):
        if value is None:
            return value
        if name in ("weight", "weight_experts", "weight_shared"):
            mask = self.__dict__.get("_buffers", {}).get("_parity_weight_mask")
        elif name in ("bias", "bias_experts", "bias_shared"):
            mask = self.__dict__.get("_buffers", {}).get("_parity_bias_mask")
        else:
            return value
        return value if mask is None else value * mask

    def __getattr__(self, name):
        return self._parity_value(name, super().__getattr__(name))


class ParityLinear(ParityWeightMixin, torch.nn.Linear):
    def reset_parameters(self):
        # nn.Linear.reset_parameters edits self.weight in-place. Temporarily
        # expose the leaves rather than an ephemeral masked product.
        masks = {name: self._buffers.pop(name) for name in
                 ("_parity_weight_mask", "_parity_bias_mask") if name in self._buffers}
        try:
            super().reset_parameters()
        finally:
            self._buffers.update(masks)


def enforce_so2_parity(module):
    """Configure an already initialized SO2 layer, without drawing random numbers."""
    for m, fc in [(0, module.fc_m0)] + [(block.m, block.fc) for block in module.m_linear]:
        if not isinstance(fc, ParityWeightMixin):
            if type(fc) is torch.nn.Linear:
                fc.__class__ = ParityLinear
            else:
                raise ValueError("so2_parity='enforce' requires linear m blocks; "
                                 "use_interpolation_out/use_interpolation must be false")
        fc.set_parity_masks(*parity_masks(module.irreps_in, module.irreps_out, m))
    module.so2_parity = "enforce"

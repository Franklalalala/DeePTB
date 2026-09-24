"""Independent Switch-style top-1, without shared experts.

Global float32 softmax -> argmax -> retain selected probability. Ordinary
autograd trains the probability; the hard index itself is not differentiated.
No bias-adjusted selection, selected-only normalization, STE, or gate floor.
This implements the Switch routing rule without capacity dropping.  The Switch
balancing loss E * sum_e f_e * P_e (f_e: fraction of route tokens whose argmax is
e, no gradient; P_e: mean router probability of e) and the ST-MoE z-loss are
computed every forward and enter the training objective only through
loss_options.train.router_aux_loss_coef / router_z_loss_coef (default 0).
"""
from collections import Counter
import torch
from torch import nn
from .tensor_product_moe_v3 import MOLEGlobals, router_z_loss

COUNTS = Counter()


class Top1Route(MOLEGlobals):
    top1_independent = True

    def __init__(self, indices, gates):
        super().__init__(topk_indices=indices, topk_values=gates,
                         activation_space=True, coefficients_sum_to_one=False)


class Top1PriorRouter(nn.Module):
    def __init__(self, in_features, num_experts, top_k=1,
                 aux_loss_free=False, bias_update_speed=0.0):
        super().__init__()
        if top_k != 1 or num_experts < 2:
            raise ValueError("Learnable Switch top-1 requires top_k=1 and E>=2")
        if aux_loss_free or bias_update_speed != 0:
            raise ValueError("Switch argmax must not use a selection-only balancing bias")
        self.top_k, self.num_experts = 1, num_experts
        self.net = nn.Sequential(nn.Linear(in_features, 128), nn.SiLU(),
                                 nn.Linear(128, num_experts))
        self.register_buffer('ema_load', torch.full((num_experts,), 1 / num_experts))
        self._last_topk_indices = self._last_topk_values = None
        self.last_stats = {}
        self.last_router_aux_loss = self.last_router_z_loss = None

    def forward(self, features, sizes=None):
        logits = self.net(features)
        probs = torch.softmax(logits.float(), dim=-1)
        indices = probs.argmax(-1, keepdim=True)
        gates = probs.gather(1, indices)
        weights = (probs.new_ones(features.shape[0]) if sizes is None
                   else sizes.to(probs).reshape(-1))
        with torch.no_grad():
            load = probs.new_zeros(self.num_experts)
            load.scatter_add_(0, indices[:, 0], weights)
        if features.shape[0]:
            total = weights.sum().clamp_min(1e-12)
            mean_prob = (probs * weights.unsqueeze(-1)).sum(0) / total
            self.last_router_aux_loss = self.num_experts * ((load / total) * mean_prob).sum()
        else:
            self.last_router_aux_loss = probs.new_zeros(())
        self.last_router_z_loss = router_z_loss(logits, sizes)
        with torch.no_grad():
            if self.training:
                self.ema_load.mul_(0.9).add_(load, alpha=0.1)
            cv = self.ema_load.std(unbiased=False) / self.ema_load.mean().clamp_min(1e-8)
            if features.shape[0]:
                self.last_stats = dict(gate_min=gates.min().detach(), gate_max=gates.max().detach(),
                                       gate_mean=gates.mean().detach(),
                                       used_experts=(load > 0).sum(),
                                       largest_load_fraction=load.max()/weights.sum().clamp_min(1))
        self._last_topk_indices, self._last_topk_values = indices, gates
        monitor = gates.detach().mean() if features.shape[0] else gates.new_zeros(())
        return Top1Route(indices, gates), monitor, cv.detach()

    def last_topk(self):
        return self._last_topk_indices, self._last_topk_values


def linear(layer, x, route):
    """One grouped selected-expert linear, multiplied by its global probability."""
    if layer.num_shared_experts != 0:
        raise ValueError("Switch top-1 execution requires zero shared experts")
    ids, gates = route.topk_indices, route.topk_values
    if ids.shape != (x.shape[0], 1) or gates.shape != ids.shape:
        raise ValueError("top-1 routes must contain one id/gate per input row")
    layout = route.expert_slot_layout(0, ids[:, 0], layer.num_experts)
    out = layer._apply_expert_with_layout(x, layout, layer.weight_experts,
                                         layer.bias_experts)
    out = out * gates.to(out.dtype).reshape(x.shape[0], *([1] * (x.ndim - 1)))
    cuda = (layer.mole_linear_mode == 'cublas_grouped' and x.is_cuda
            and x.dtype == torch.float32 and layer.weight_experts.dtype == torch.float32)
    COUNTS['grouped_cuda' if cuda else 'reference'] += 1
    return out

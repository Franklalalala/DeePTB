"""Training-only structured noise on Hamiltonian priors, without a flow model.

The historical ``flow_options`` sampling keys remain accepted. This adapter
changes only prior inputs and zero-time conditioning; supervised targets and
losses are untouched. Evaluation is an identity operation and consumes no RNG.
"""
from __future__ import annotations

import math

import torch

from dptb.configuration import canonicalize_flow_options
from dptb.data import _keys
from dptb.nnops.structured_noise import StructuredNoise


def assert_prior_noise_keys_reach_model(noise, model):
    """Reject prior writes that bypass an existing H0 input consumer."""
    if noise is None:
        return
    mismatches = set()
    for module in model.modules():
        node_key = getattr(module, "h0_node_key", None)
        edge_key = getattr(module, "h0_edge_key", None)
        if not (isinstance(node_key, str) and isinstance(edge_key, str)):
            continue
        for label, source, target in (("node", noise.node_h0_key, node_key),
                                      ("edge", noise.edge_h0_key, edge_key)):
            if source != target:
                mismatches.add((label, source, target))
    if mismatches:
        detail = "; ".join(f"{side}: noise key {source!r}, embedding key {target!r}"
                           for side, source, target in sorted(mismatches))
        raise ValueError(f"Prior noise does not reach the model's H0 inputs: {detail}")


class PriorNoiseAugmentation(StructuredNoise):
    """Add the exact structured TE draw used by the historical zero-time hook."""

    def __init__(self, options=None, *, idp=None, dtype=torch.float32, device="cpu"):
        options = canonicalize_flow_options(options)
        self.options = options
        self.idp = idp
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.device = torch.device(device)
        self.enabled = True
        self.block_ode = False
        self.mode = str(options.get("mode", "residual")).lower()
        if self.mode != "residual" or options.get("block_ode", False) \
                or options.get("output_space", "rme") != "rme":
            raise ValueError("prior noise requires residual RME inputs; flow/block objectives are separate")
        for key in ("t_min", "t_max", "t0_probability"):
            if float(options.get(key, 0.0)) != 0.0:
                raise ValueError(f"prior noise requires flow_options.{key}=0")
        self.prior = str(options.get("prior", "te")).lower().replace("-", "_")
        if self.prior not in {"te", "structured_te", "te_like"}:
            raise ValueError("prior noise supports structured TE sampling only")
        self.te_prior_mode = str(options.get("te_prior_mode", "typewise")).lower().replace("-", "_")
        if self.te_prior_mode == "auto":
            self.te_prior_mode = "irrep"
        if self.te_prior_mode == "type":
            self.te_prior_mode = "typewise"
        if self.te_prior_mode not in {"typewise", "irrep"}:
            raise ValueError("invalid prior-noise te_prior_mode")
        self.te_prior_scale_reference = str(options.get("te_prior_scale_reference", "target")).lower()
        if self.te_prior_scale_reference not in {"target", "residual"}:
            raise ValueError("prior-noise scale reference must be target or residual")
        self.te_prior_per_graph = bool(options.get("te_prior_per_graph", True))
        for key, default in (("te_prior_sigma", 0.5), ("node_sigma", 1.0),
                             ("edge_sigma", 1.0), ("residual_sigma_floor", 1e-6)):
            value = float(options.get(key, default))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"prior-noise {key} must be finite and nonnegative")
            setattr(self, key, value)
        self.missing_h0_policy = str(options.get("missing_h0_policy", "error")).lower()
        if self.missing_h0_policy not in {"error", "zero", "warn_zero"}:
            raise ValueError("invalid prior-noise missing_h0_policy")
        self.strict_h0 = self.missing_h0_policy == "error"
        self.warn_missing_h0 = self.missing_h0_policy == "warn_zero"
        self.detach_interpolated_h0 = bool(options.get("detach_interpolated_h0", True))
        for key, default in (("node_h0_key", _keys.NODE_H0_KEY),
                             ("edge_h0_key", _keys.EDGE_H0_KEY),
                             ("node_target_key", _keys.NODE_FEATURES_KEY),
                             ("edge_target_key", _keys.EDGE_FEATURES_KEY),
                             ("flow_time_key", "flow_time")):
            setattr(self, key, str(options.get(key, default)))
        inputs = {self.node_h0_key, self.edge_h0_key, self.flow_time_key}
        targets = {self.node_target_key, self.edge_target_key}
        if len(inputs) != 3 or inputs & targets:
            raise ValueError("prior-noise input keys must be distinct from supervised target keys")
        self._te_irrep_slices_cache = {}

    def __call__(self, data, *, training):
        """Return a shallow input copy on training calls; labels retain identity."""
        if not training:
            return data
        node_target = data.get(self.node_target_key)
        edge_target = data.get(self.edge_target_key)
        like = node_target if node_target is not None else edge_target
        if like is None:
            raise KeyError("prior noise needs supervised node or edge targets to set the noise scale")
        dtype = like.dtype if torch.is_floating_point(like) else self.dtype
        num_graphs = self._num_graphs(data)
        t0 = torch.zeros(num_graphs, device=like.device, dtype=dtype)
        result = data.copy()
        # Keep node before edge and the original zero-time arithmetic. Even
        # apparently redundant products participate in the bitwise contract.
        for label, target, h0_key, sigma in (
                ("node", node_target, self.node_h0_key, self.node_sigma),
                ("edge", edge_target, self.edge_h0_key, self.edge_sigma)):
            if target is None:
                continue
            target = target.to(device=like.device, dtype=dtype)
            base = self._base_like(result, target, h0_key, label)
            residual = target - base
            reference = target if self.te_prior_scale_reference == "target" else residual
            noise = self._te_prior_like(reference, sigma, data=result, label=label,
                                        num_graphs=num_graphs)
            zero = t0.new_zeros((target.shape[0],) + (1,) * (target.ndim - 1))
            current = base + (1.0 - zero) * noise + zero * residual
            result[h0_key] = current.detach() if self.detach_interpolated_h0 else current
        result[self.flow_time_key] = t0.detach()
        return result

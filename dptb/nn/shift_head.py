"""Atomic potential corrections in the physical compact uu-real AO layout.

S is dimensionless; v and the residual Hamiltonian use the label energy unit.
The response reads hidden scalar features. No label or H0 enters the
potential predictor.
"""
from __future__ import annotations

from numbers import Integral

import torch
from torch import nn
from e3nn import o3

from dptb.data import AtomicDataDict as K
from dptb.nnops.distance_expert_mask import edge_mask_for_distance_expert
from dptb.nnops.layout import uureal_projection_mask


def normalize_shift_options(options):
    cfg = dict(mode="off", hidden=64, layers=2, element_dim=0,
               freeze_backbone=False, init_from=None, response=None, overlap_input="physical",
               input_norm="none", capture="output", detach_input=False, output_scale=1.0)
    if options is None:
        return cfg
    if not isinstance(options, dict):
        raise ValueError("shift_head must be a dictionary or None")
    unknown = set(options) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown shift_head options: {sorted(unknown)}")
    cfg.update(options)
    if cfg["mode"] not in {"off", "atom"}:
        raise ValueError("shift_head.mode must be off or atom")
    for key, minimum in (("hidden", 1), ("layers", 1), ("element_dim", 0)):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], Integral) or cfg[key] < minimum:
            raise ValueError(f"shift_head.{key} must be an integer >= {minimum}")
    if not isinstance(cfg["freeze_backbone"], bool):
        raise ValueError("shift_head.freeze_backbone must be bool")
    if cfg["init_from"] is not None and not isinstance(cfg["init_from"], str):
        raise ValueError("shift_head.init_from must be a path string or None")
    if cfg["overlap_input"] not in {"physical", "standard"}:
        raise ValueError("shift_head.overlap_input must be physical or standard")
    # Inactive legacy defaults remain accepted in saved configurations.
    if cfg["freeze_backbone"] or cfg["init_from"] is not None:
        raise ValueError("Frozen-backbone initialization requires an archived model")
    if not isinstance(cfg["detach_input"], bool) or isinstance(cfg["output_scale"], bool):
        raise ValueError("Invalid legacy shift-head input options")
    if cfg["input_norm"] != "none" or cfg["capture"] != "output" or cfg["detach_input"] or cfg["output_scale"] != 1.0:
        raise ValueError("Additive shift heads require an archived model")
    if cfg["mode"] == "atom" and cfg["response"] is None:
        raise ValueError("Atomic shift heads require a charge response")
    if cfg["response"] is not None:
        if cfg["mode"] != "atom":
            raise ValueError("shift_head.response requires mode=atom; use response=null to disable")
        from dptb.nn.response_shift_head import normalize_response_options
        cfg["response"] = normalize_response_options(cfg["response"])
    return cfg


class _ExactAdd(torch.autograd.Function):
    """Addition with exact identity at zero, including the sign bit of zero."""
    @staticmethod
    def forward(ctx, original, delta):
        return torch.where(delta == 0, original, original + delta)

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


def add_compact_delta(idp, prediction, delta):
    if prediction.shape == delta.shape:
        return _ExactAdd.apply(prediction, delta)
    mask = uureal_projection_mask(idp, raw_width=prediction.shape[-1],
                                  target_width=delta.shape[-1], device=prediction.device)
    if mask is None or prediction.shape[0] != delta.shape[0]:
        raise ValueError("shift_head cannot map compact S into the prediction layout")
    out = prediction.clone()
    out[:, mask] = _ExactAdd.apply(prediction[:, mask], delta)
    return out


class PotentialShiftHead(nn.Module):
    """Shared overlap assembly and checkpoint-compatible initialization draws."""
    def __init__(self, idp, irreps, options, *, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.options = normalize_shift_options(options)
        self.mode = self.options["mode"]
        if not (idp.has_soc and idp.nextham_uureal_mask) or idp.full_soc_prediction:
            raise ValueError("shift_head requires a compact SOC uu-real mapper")
        self.idp = idp
        self.distance_policy = None
        irreps = o3.Irreps(irreps)
        indices = [i for (_, ir), sl in zip(irreps, irreps.slices())
                   if ir.l == 0 and ir.p == 1 for i in range(sl.start, sl.stop)]
        if not indices:
            raise ValueError("shift_head needs l=0 trunk features")
        self.register_buffer("scalar_indices", torch.tensor(indices, dtype=torch.long, device=device), persistent=False)
        element_dim = self.options["element_dim"]
        self.element = (nn.Embedding(len(idp.type_names), element_dim, device=device, dtype=dtype)
                        if element_dim else None)
        width = len(indices) + element_dim
        modules = []
        for _ in range(self.options["layers"] - 1):
            modules.extend([nn.Linear(width, self.options["hidden"], device=device, dtype=dtype), nn.SiLU()])
            width = self.options["hidden"]
        # The response adapter discards this readout, but its initialization
        # draws must precede the response network to preserve fresh-model RNG.
        modules.append(nn.Linear(width, 1, device=device, dtype=dtype))
        self.mlp = nn.Sequential(*modules)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self._compact_width = idp.reduced_matrix_element
        self.last_stats = {}


    def assemble(self, data, v, active_edges=None):
        sn = data[K.PHYS_NODE_OVERLAP_KEY]
        se = data[K.PHYS_EDGE_OVERLAP_KEY]
        edge = data[K.EDGE_INDEX_KEY]
        n = v.shape[0]
        expected = self._compact_width
        if sn.shape != (n, expected) or se.shape != (edge.shape[1], expected):
            raise ValueError("shift_head physical S shape does not match nodes/edges/compact slots")
        dn = v * sn
        de = ((v[edge[0]] + v[edge[1]]) * 0.5) * se
        node_mask = torch.ones(n, dtype=torch.bool, device=v.device)
        edge_mask = torch.ones(edge.shape[1], dtype=torch.bool, device=v.device)
        if active_edges is not None:
            edge_mask.zero_()
            edge_mask[active_edges] = True
        if self.distance_policy is not None:
            lo, hi, last, clip = self.distance_policy
            node_mask &= lo == 0
            edge_mask &= edge_mask_for_distance_expert(data[K.EDGE_LENGTH_KEY].flatten(), lo, hi,
                is_last_expert=last, clip_last_expert_range=clip)
        if "expert_node_mask" in data:
            node_mask &= data["expert_node_mask"].flatten()
        if "expert_edge_mask" in data:
            edge_mask &= data["expert_edge_mask"].flatten()
        return dn * node_mask[:, None], de * edge_mask[:, None]


def optimizer_named_parameters(model):
    """Return the jointly trained model parameters in their original order."""
    return model.named_parameters()

"""Atomic potential corrections in the physical compact uu-real AO layout.

S is dimensionless; v and the residual Hamiltonian use the label energy unit.
The trunk features are captured before the existing output heads. No label or
H0 enters the potential predictor. Shell channels denote individual radial
shells of the mapper's super-basis, never pooled angular-momentum classes.
"""
from __future__ import annotations

import logging
from numbers import Integral

import torch
from torch import nn
from e3nn import o3

from dptb.data import AtomicDataDict as K
from dptb.nnops.distance_expert_mask import edge_mask_for_distance_expert
from dptb.nnops.layout import uureal_projection_mask

log = logging.getLogger(__name__)


def normalize_shift_options(options):
    cfg = dict(mode="off", hidden=64, layers=2, element_dim=0,
               freeze_backbone=False, init_from=None)
    if options is None:
        return cfg
    if not isinstance(options, dict):
        raise ValueError("shift_head must be a dictionary or None")
    unknown = set(options) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown shift_head options: {sorted(unknown)}")
    cfg.update(options)
    if cfg["mode"] not in {"off", "atom", "shell"}:
        raise ValueError("shift_head.mode must be off, atom or shell")
    for key, minimum in (("hidden", 1), ("layers", 1), ("element_dim", 0)):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], Integral) or cfg[key] < minimum:
            raise ValueError(f"shift_head.{key} must be an integer >= {minimum}")
    if not isinstance(cfg["freeze_backbone"], bool):
        raise ValueError("shift_head.freeze_backbone must be bool")
    if cfg["init_from"] is not None and not isinstance(cfg["init_from"], str):
        raise ValueError("shift_head.init_from must be a path string or None")
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
                   if ir.l == 0 for i in range(sl.start, sl.stop)]
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
        outputs = 1 if self.mode == "atom" else len(idp.full_basis)
        modules.append(nn.Linear(width, outputs, device=device, dtype=dtype))
        self.mlp = nn.Sequential(*modules)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        idp.get_orbpair_maps()
        slot_i = torch.full((idp.reduced_matrix_element,), -1, dtype=torch.long, device=device)
        slot_j = slot_i.clone()
        for pair, sl in idp.orbpair_maps.items():
            a, b = pair.split("-")
            slot_i[sl] = idp.full_basis.index(a)
            slot_j[sl] = idp.full_basis.index(b)
        if (slot_i < 0).any() or (slot_j < 0).any():
            raise ValueError("shift_head shell mapping does not cover compact slots")
        self.register_buffer("slot_i", slot_i, persistent=False)
        self.register_buffer("slot_j", slot_j, persistent=False)
        # Mask padding/absent shells when logging potentials across species.
        present = torch.zeros(len(idp.type_names), len(idp.full_basis), dtype=torch.bool, device=device)
        for symbol, atom_type in idp.chemical_symbol_to_type.items():
            for shell in idp.basis_to_full_basis[symbol].values():
                present[atom_type, idp.full_basis.index(shell)] = True
        self.register_buffer("shell_present", present, persistent=False)
        self.last_stats = {}

    def predict(self, features, atom_types):
        x = features.index_select(1, self.scalar_indices)
        if self.element is not None:
            x = torch.cat((x, self.element(atom_types.flatten())), dim=-1)
        return self.mlp(x)

    def assemble(self, data, v, active_edges=None):
        sn = data[K.PHYS_NODE_OVERLAP_KEY]
        se = data[K.PHYS_EDGE_OVERLAP_KEY]
        edge = data[K.EDGE_INDEX_KEY]
        n = v.shape[0]
        expected = self.slot_i.numel()
        if sn.shape != (n, expected) or se.shape != (edge.shape[1], expected):
            raise ValueError("shift_head physical S shape does not match nodes/edges/compact slots")
        if self.mode == "atom":
            dn = v * sn
            de = ((v[edge[0]] + v[edge[1]]) * 0.5) * se
        else:
            dn = ((v[:, self.slot_i] + v[:, self.slot_j]) * 0.5) * sn
            de = ((v[edge[0]][:, self.slot_i] + v[edge[1]][:, self.slot_j]) * 0.5) * se
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

    def forward(self, data):
        features = data.pop("_shift_node_features")
        active_edges = data.pop("_shift_active_edges")
        v = self.predict(features, data[K.ATOM_TYPE_KEY])
        dn, de = self.assemble(data, v, active_edges)
        for prediction, delta_key, delta in ((K.NODE_FEATURES_KEY, K.NODE_SHIFT_DELTA_KEY, dn),
                                               (K.EDGE_FEATURES_KEY, K.EDGE_SHIFT_DELTA_KEY, de)):
            data[prediction] = add_compact_delta(self.idp, data[prediction], delta)
            # Detached compact diagnostics; no retained computation graph.
            data[delta_key] = delta.detach()
        if self.training:
            with torch.no_grad():
                values = v.flatten() if self.mode == "atom" else v[self.shell_present[data[K.ATOM_TYPE_KEY].flatten()]]
                zero = v.new_zeros(())
                self.last_stats = {
                    "v_mean": values.mean() if values.numel() else zero,
                    "v_std": values.std(unbiased=False) if values.numel() else zero,
                    "v_absmax": values.abs().max() if values.numel() else zero,
                    "delta_node_rms": dn.square().mean().sqrt() if dn.numel() else zero,
                    "delta_edge_rms": de.square().mean().sqrt() if de.numel() else zero,
                }
                log.info("shift_head %s", " ".join(f"{k}={float(x):.7g}" for k, x in self.last_stats.items()))
        return data


def load_dense_backbone(model, path):
    """Strict same-topology dense initialization; only new shift keys are absent.

    Called only for a fresh build. Resume uses strict full checkpoint loading and
    does not re-read this provenance path, which may no longer exist.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_options = checkpoint["config"]["model_options"]
    if (source_options.get("shift_head") or {}).get("mode", "off") != "off":
        raise ValueError("shift_head.init_from must name a dense checkpoint without a shift head")
    emb = source_options.get("embedding", {})
    if int(emb.get("num_experts", 1)) != 1:
        raise ValueError("shift_head.init_from must name a dense backbone (num_experts=1)")
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    is_head = lambda k: k.startswith("shift_head.") or ".shift_head." in k
    backbone = {k: v for k, v in target.items() if not is_head(k)}
    if source.keys() != backbone.keys():
        raise ValueError(f"shift_head.init_from keys differ: missing={sorted(backbone.keys()-source.keys())}, "
                         f"extra={sorted(source.keys()-backbone.keys())}")
    for name, tensor in source.items():
        if tensor.shape != backbone[name].shape:
            raise ValueError(f"shift_head.init_from shape mismatch: {name}")
    target.update(source)
    model.load_state_dict(target, strict=True)


def optimizer_named_parameters(model):
    """Preserve legacy optimizer groups unless the new freeze switch is active."""
    options = getattr(model, "model_options", {}).get("shift_head") or {}
    params = model.named_parameters()
    if options.get("mode", "off") != "off" and options.get("freeze_backbone", False):
        return ((name, p) for name, p in params if p.requires_grad)
    return params

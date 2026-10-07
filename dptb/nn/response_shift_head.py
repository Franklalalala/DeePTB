"""Neutral charge response and local-only relative potentials."""
from __future__ import annotations

import torch
from e3nn import o3

from dptb.data import AtomicDataDict as K
from dptb.nn.shift_head import PotentialShiftHead, add_compact_delta
from dptb.nn.charge_response import (
    ResponseNetwork, graph_index, onsite_coordinate, weighted_center,
)
from dptb.nnops.layout import normalize_idp_mask_layout


def normalize_response_options(options):
    cfg = dict(kind="context", hidden=64, element_dim=8, local_only=False,
               canonical_onsite=False, auxiliary_weight=0.0, auxiliary_beta=0.05,
               sigma=1.2, g_cut=4.0, hardness_min=5.0, hardness_max=40.0,
               max_atoms=512, max_modes=30000, detach_features=False,
               context="auto", output_scale=1.0, aux_weighting="graph_equal", qeq_local=False)
    if not isinstance(options, dict):
        raise ValueError("shift_head.response must be a dictionary")
    unknown = set(options) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown shift_head.response options: {sorted(unknown)}")
    cfg.update(options)
    if cfg["kind"] not in {"context", "qeq"}:
        raise ValueError("response.kind must be context or qeq")
    for key in ("canonical_onsite", "local_only", "detach_features", "qeq_local"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"response.{key} must be bool")
    # qeq_local: kind=qeq only; v = local readout + kappa * Gamma q (the frozen-probe form)
    # instead of the delivered v = Gamma q. See charge_response.ResponseNetwork.
    if cfg["qeq_local"] and cfg["kind"] != "qeq":
        raise ValueError("response.qeq_local applies to kind=qeq only")
    for key in ("hidden", "max_atoms", "max_modes", "element_dim"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < (0 if key == "element_dim" else 1):
            raise ValueError(f"invalid response.{key}")
    for key in ("auxiliary_beta", "sigma", "g_cut", "hardness_min", "hardness_max"):
        if not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < float("inf"):
            raise ValueError(f"invalid response.{key}")
    if not isinstance(cfg["auxiliary_weight"], (int, float)) or not 0 <= cfg["auxiliary_weight"] < float("inf"):
        raise ValueError("auxiliary_weight must be finite and nonnegative")
    if cfg["canonical_onsite"] or cfg["auxiliary_weight"] != 0 or cfg["detach_features"]:
        raise ValueError("Canonical onsite, auxiliary supervision and detached features require an archived model")
    if cfg["kind"] == "qeq":
        if cfg["context"] not in {"auto", "none"} or cfg["local_only"]:
            raise ValueError("QEq requires local descriptors and context=none")
    elif not cfg["local_only"] or cfg["context"] not in {"auto", "graph"}:
        raise ValueError("The context response is the local_only control")
    if isinstance(cfg["output_scale"], bool) or not isinstance(cfg["output_scale"], (int, float)) \
            or not 0.0 < float(cfg["output_scale"]) < float("inf"):
        raise ValueError("response.output_scale must be a positive finite number")
    cfg["output_scale"] = float(cfg["output_scale"])
    if cfg["aux_weighting"] not in {"graph_equal", "overlap"}:
        raise ValueError("response.aux_weighting must be graph_equal or overlap")
    return cfg


class ResponseShiftHead(PotentialShiftHead):
    def __init__(self, idp, irreps, options, *, dtype=torch.float32, device="cpu"):
        if options["mode"] != "atom":
            raise ValueError("response branch currently implements atom mode only")
        super().__init__(idp, irreps, options, dtype=dtype, device=device)
        irreps = o3.Irreps(irreps)
        indices = [i for (_, ir), sl in zip(irreps, irreps.slices())
                   if ir.l == 0 and ir.p == 1 for i in range(sl.start, sl.stop)]
        if not indices:
            raise ValueError("response requires hidden 0e scalars (not 0o pseudoscalars)")
        self.scalar_indices = torch.tensor(indices, dtype=torch.long, device=device)
        # These legacy readout parameters must not remain as unused optimizer/DDP params.
        del self.mlp
        del self.element
        self.response_options = normalize_response_options(options["response"])
        net_options = {k: v for k, v in self.response_options.items()
                       if k not in {"canonical_onsite", "auxiliary_weight", "auxiliary_beta", "aux_weighting"}}
        self.response_net = ResponseNetwork(len(indices), len(idp.type_names),
                                           dtype=dtype, device=device, **net_options)
        self.last_potential = None
        self.last_charge = None

    def forward(self, data):
        features = data.pop("_shift_node_features")
        active_edges = data.pop("_shift_active_edges")
        sn = data[K.PHYS_NODE_OVERLAP_KEY]
        types = data[K.ATOM_TYPE_KEY].flatten()
        batch = graph_index(data.get(K.BATCH_KEY), len(types), types.device)
        # Do not infer a nonzero total charge from structure/labels. This is a
        # neutral *delta-charge* layer; non-neutral training needs a defined reference.
        for key in ("total_charge", "net_charge"):
            if key in data and bool((torch.as_tensor(data[key]) != 0).any()):
                raise ValueError("response Q/delta-reference for charged structures is not implemented")
        raw_v, aux = self.response_net(features.index_select(1, self.scalar_indices), types,
                                      batch, pos=data.get(K.POSITIONS_KEY),
                                      cell=data.get(K.CELL_KEY), pbc=data.get(K.PBC_KEY))
        node_mask = self.idp.mask_to_nrme.to(device=sn.device)[types]
        node_mask = normalize_idp_mask_layout(self.idp, node_mask, sn, label="response onsite mask")
        if node_mask.shape != sn.shape:
            raise ValueError("response orbital mask does not match physical S")
        # Gauge weights use all valid orbitals independently of task scope.
        _, weights = onsite_coordinate(sn, sn, node_mask)
        v = weighted_center(raw_v, weights, batch)
        dn, de = self.assemble(data, v, active_edges)
        for key, delta_key, delta in ((K.NODE_FEATURES_KEY, K.NODE_SHIFT_DELTA_KEY, dn),
                                      (K.EDGE_FEATURES_KEY, K.EDGE_SHIFT_DELTA_KEY, de)):
            data[key] = add_compact_delta(self.idp, data[key], delta)
            data[delta_key] = delta.detach()
        self.last_potential = v.detach()
        self.last_charge = aux.get("q")
        if self.last_charge is not None:
            self.last_charge = self.last_charge.detach()
        self.last_stats = {"v_std": v.detach().std(unbiased=False),
                           "delta_node_rms": dn.detach().square().mean().sqrt()}
        return data

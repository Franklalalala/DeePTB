"""Support-shrunk chemical residuals of the onsite same-irrep channel maps.

W is the effective channel matrix (including e3nn path normalization), with
no magnetic-component repetition in its Frobenius norm. Bias stays shared.
Only one [out_channels, in_channels] residual per present element/block is
formed; no atom-indexed weight bank is materialized.
"""
from __future__ import annotations

import logging
import random

import numpy as np
import torch
from torch import nn
from e3nn import o3

from dptb.nn.shift_head import _ExactAdd

log = logging.getLogger(__name__)
RANK = 16
COUNT_SCALE = 100
CAP_FRACTION = 0.25


class ChemicalBlock(nn.Module):
    def __init__(self, num_elements, in_channels, out_channels, **factory):
        super().__init__()
        self.P = nn.Parameter(torch.randn(out_channels, RANK, **factory) / RANK**0.5)
        self.Q = nn.Parameter(torch.randn(in_channels, RANK, **factory) / RANK**0.5)
        self.D = nn.Parameter(torch.zeros(num_elements, RANK, RANK, **factory))
        self.register_buffer("c", torch.zeros((), **factory))

    def delta(self, group):
        r = self.P @ self.D[group] @ self.Q.T
        # hypot avoids squaring c; a zero shared block has a zero cap and gradient.
        denominator = torch.hypot(self.c.clamp_min(torch.finfo(r.dtype).tiny), torch.linalg.vector_norm(r))
        return r * (self.c / denominator.clamp_min(torch.finfo(r.dtype).tiny))


class ChemicalCoreReadout(nn.Module):
    def __init__(self, shared, atomic_numbers):
        super().__init__()
        if not isinstance(shared, o3.Linear) or not shared.internal_weights or not shared.shared_weights:
            raise ValueError("chemical_core requires an internal, shared e3nn Linear out_node")
        if getattr(shared, "f_in", None) is not None or getattr(shared, "f_out", None) is not None:
            raise ValueError("chemical_core does not support extra feature axes")
        self.irreps_in, self.irreps_out = shared.irreps_in, shared.irreps_out
        factory = dict(dtype=shared.weight.dtype, device=shared.weight.device)
        self.register_buffer("atomic_numbers", torch.as_tensor(atomic_numbers, dtype=torch.long,
                                                               device=shared.weight.device).clone())
        self.register_buffer("n_g", torch.zeros(len(atomic_numbers), dtype=torch.long, device=shared.weight.device))
        self.register_buffer("counts_ready", torch.tensor(False, device=shared.weight.device))
        self.blocks = nn.ModuleList()
        self.specs = []
        # Nonadjacent repeated irreps belong to one chemical channel block.
        for ir in sorted(set(ir for _, ir in self.irreps_in) & set(ir for _, ir in self.irreps_out)):
            in_parts = [(i, mul) for i, (mul, rep) in enumerate(self.irreps_in) if rep == ir]
            out_parts = [(i, mul) for i, (mul, rep) in enumerate(self.irreps_out) if rep == ir]
            indices_in = [k for i, _ in in_parts for k in range(self.irreps_in.slices()[i].start,
                                                               self.irreps_in.slices()[i].stop)]
            indices_out = [k for i, _ in out_parts for k in range(self.irreps_out.slices()[i].start,
                                                                 self.irreps_out.slices()[i].stop)]
            index = len(self.blocks)
            self.register_buffer(f"input_indices_{index}", torch.tensor(indices_in, dtype=torch.long,
                                 device=shared.weight.device), persistent=False)
            self.register_buffer(f"output_indices_{index}", torch.tensor(indices_out, dtype=torch.long,
                                 device=shared.weight.device), persistent=False)
            self.specs.append((ir, in_parts, out_parts))
            self.blocks.append(ChemicalBlock(len(atomic_numbers), sum(m for _, m in in_parts),
                                            sum(m for _, m in out_parts), **factory))
        self.capture_caps(shared)

    @torch.no_grad()
    def capture_caps(self, shared):
        """Only at construction / explicit dense initialization, never in forward."""
        weights, offset = {}, 0
        for ins in shared.instructions:
            if ins.i_in == -1:
                continue
            size = ins.path_shape[0] * ins.path_shape[1]
            weights[ins.i_in, ins.i_out] = (shared.weight[offset:offset+size].reshape(ins.path_shape)
                                          * ins.path_weight)
            offset += size
        for block, (_, in_parts, out_parts) in zip(self.blocks, self.specs):
            w = torch.cat([torch.cat([weights.get((i, j), shared.weight.new_zeros(mi, mo))
                                     for j, mo in out_parts], dim=1) for i, mi in in_parts], dim=0)
            block.c.copy_(CAP_FRACTION * torch.linalg.vector_norm(w))

    @property
    def rho(self):
        n = self.n_g.to(dtype=self.blocks[0].c.dtype)
        return n / (n + COUNT_SCALE)

    @torch.no_grad()
    def set_counts(self, counts):
        counts = torch.as_tensor(counts, device=self.n_g.device)
        if counts.shape != self.n_g.shape or counts.dtype == torch.bool or counts.is_floating_point() or (counts < 0).any():
            raise ValueError("chemical_core counts must be nonnegative integer structure counts in mapper order")
        if bool(self.counts_ready):
            raise RuntimeError("chemical_core support counts are immutable once initialized")
        self.n_g.copy_(counts)
        self.counts_ready.fill_(True)

    def forward(self, features, atom_types, shared_output):
        if not bool(self.counts_ready):
            raise RuntimeError("chemical_core training support is unset; initialize_chemical_readouts(model, train_dataset) first")
        if atom_types is None:
            raise ValueError("chemical_core requires explicit atom types")
        types = atom_types.reshape(-1)
        if types.shape[0] != features.shape[0] or types.dtype != torch.long:
            raise ValueError("chemical_core expects one long atom type per node")
        if types.numel() and (int(types.min()) < 0 or int(types.max()) >= len(self.n_g)):
            raise ValueError("chemical_core atom type is outside the model basis")
        # One CPU list transfer per batch; all gathers/GEMMs remain on the device.
        groups = [(g, torch.where(types == g)[0]) for g in torch.unique(types).tolist()]
        rho = self.rho
        correction = torch.zeros_like(shared_output)
        for b, (block, (ir, _, _)) in enumerate(zip(self.blocks, self.specs)):
            x = features.index_select(1, getattr(self, f"input_indices_{b}"))
            x = x.reshape(len(types), block.Q.shape[0], ir.dim)
            y = features.new_zeros(len(types), block.P.shape[0], ir.dim)
            for group, rows in groups:
                # The zero-support case also has identically zero parameter gradients.
                delta = rho[group] * block.delta(group)
                value = torch.einsum("oi,nim->nom", delta, x.index_select(0, rows))
                y = y.index_copy(0, rows, value)
            correction = correction.index_copy(1, getattr(self, f"output_indices_{b}"), y.flatten(1))
        return _ExactAdd.apply(shared_output, correction)


def initialize_chemical_readouts(model, train_dataset):
    """Count element presence once per training structure, before sampler/batching.

    Trainer and MultiTrainer use this after loading the model. A restored model
    returns without touching the dataset. Validation/reference sets never enter.
    """
    heads = [m for m in model.modules() if isinstance(m, ChemicalCoreReadout)]
    pending = [h for h in heads if not bool(h.counts_ready)]
    if not pending:
        return
    counts = torch.zeros(119, dtype=torch.long)
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            for idx in range(len(train_dataset)):
                item = train_dataset[idx]
                if hasattr(item, "to_dict"):
                    item = item.to_dict()
                z = item.get("atomic_numbers")
                if z is None:
                    owner = train_dataset
                    while not hasattr(owner, "transform") and hasattr(owner, "dataset"):
                        owner = owner.dataset
                    mapper = getattr(owner, "transform", None)
                    if not hasattr(mapper, "_index_to_Z"):
                        raise ValueError("chemical_core count scan needs atomic_numbers or the dataset OrbitalMapper")
                    z = mapper._index_to_Z.to(item["atom_types"].device)[item["atom_types"]]
                z = z.detach().cpu().reshape(-1).long()
                if z.numel() and (int(z.min()) < 1 or int(z.max()) > 118):
                    raise ValueError("chemical_core count scan found invalid atomic numbers")
                batch = item.get("batch")
                batch = torch.zeros_like(z) if batch is None else batch.detach().cpu().reshape(-1)
                if batch.shape != z.shape:
                    raise ValueError("chemical_core count scan found invalid structure membership")
                for graph in torch.unique(batch):
                    counts[torch.unique(z[batch == graph])] += 1
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    for head in pending:
        head.set_counts(counts[head.atomic_numbers.cpu()])
        log.info("chemical_core support: Z=%s n_g=%s (rank=16, count_scale=100, cap=0.25)",
                 head.atomic_numbers.tolist(), head.n_g.tolist())


def load_dense_chemical_backbone(model, path):
    """Strict same-topology initialization, with only new chemical keys missing."""
    cp = torch.load(path, map_location="cpu", weights_only=False)
    emb = cp["config"]["model_options"].get("embedding", {})
    if emb.get("node_readout", "shared") != "shared" or int(emb.get("num_experts", 1)) != 1:
        raise ValueError("node_readout_init_from requires a dense shared-readout checkpoint")
    target = model.state_dict()
    backbone = {k: v for k, v in target.items() if ".chemical_core." not in k}
    source = cp["model_state_dict"]
    if source.keys() != backbone.keys():
        raise ValueError(f"chemical dense keys differ: missing={sorted(backbone.keys()-source.keys())}, "
                         f"extra={sorted(source.keys()-backbone.keys())}")
    for name, value in source.items():
        if value.shape != backbone[name].shape:
            raise ValueError(f"chemical dense shape mismatch: {name}")
    target.update(source)
    model.load_state_dict(target, strict=True)
    for module in model.modules():
        head = getattr(module, "chemical_core", None)
        if isinstance(head, ChemicalCoreReadout):
            head.capture_caps(module.out_node)

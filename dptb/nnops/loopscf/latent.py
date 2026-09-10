"""Shared equivariant latent recurrence with one frozen encoder call per graph.

This is an endpoint-supervised corrector, not a physical SCF trajectory. H0
enters through inverse CG; no eigensolver or charge population enters the state.
"""

from __future__ import annotations

import torch
from torch import nn
from e3nn import o3
from dptb.data import AtomicDataDict as A
from .adapters import _iter_embeddings
from .representation import AOPriorToRME


class InvariantGate(nn.Module):
    """A scalar gate per irrep copy, identical for all its m components."""

    def __init__(self, irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        n = self.irreps.num_irreps
        self.net = nn.Sequential(nn.Linear(n, 32), nn.SiLU(), nn.Linear(32, n))

    def forward(self, x):
        blocks = [
            x[:, s].reshape(x.shape[0], mul, ir.dim)
            for s, (mul, ir) in zip(self.irreps.slices(), self.irreps)
        ]
        norms = torch.cat([(b.square().mean(-1) + 1e-8).sqrt() for b in blocks], -1)
        gates = self.net(norms).sigmoid().split([mul for mul, _ in self.irreps], -1)
        return torch.cat(
            [(b * g.unsqueeze(-1)).flatten(1) for b, g in zip(blocks, gates)], -1
        )


class SharedLatentCore(nn.Module):
    """Residual same-irrep message passing with optional identity initialization."""

    def __init__(self, irreps, zero_init=True):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.register_buffer("representation_version", torch.tensor(1))
        for name in (
            "node_self",
            "edge_to_node",
            "edge_self",
            "src_to_edge",
            "dst_to_edge",
            "node_out",
            "edge_out",
        ):
            setattr(self, name, o3.Linear(self.irreps, self.irreps, biases=False))
        self.node_gate = InvariantGate(self.irreps)
        self.edge_gate = InvariantGate(self.irreps)
        if zero_init:
            nn.init.zeros_(self.node_out.weight)
            nn.init.zeros_(self.edge_out.weight)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        version = state_dict.get(prefix + "representation_version")
        if version is None or version.numel() != 1 or int(version) != 1:
            raise RuntimeError("latent corrector requires representation version 1")
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, node, edge, edge_index):
        src, dst = edge_index
        aggregate = node.new_zeros(node.shape).index_add(
            0, dst, self.edge_to_node(edge)
        )
        degree = torch.bincount(dst, minlength=len(node)).clamp_min(1).to(node.dtype)
        n = self.node_self(node) + aggregate / degree[:, None]
        e = (
            self.edge_self(edge)
            + self.src_to_edge(node[src])
            + self.dst_to_edge(node[dst])
        ) / 3
        return (
            node + 0.1 * self.node_out(self.node_gate(n)),
            edge + 0.1 * self.edge_out(self.edge_gate(e)),
        )


class ResidualReadout(nn.Module):
    """Learned RME correction to the cached frozen-base prediction."""

    def __init__(self, hidden_irreps, output_irreps):
        super().__init__()
        self.node = o3.Linear(hidden_irreps, output_irreps, biases=False)
        self.edge = o3.Linear(hidden_irreps, output_irreps, biases=False)
        nn.init.zeros_(self.node.weight)
        nn.init.zeros_(self.edge.weight)

    def forward(self, node, edge):
        return self.node(node), self.edge(edge)


def _wrap_cached_embedding(emb, readout="frozen"):
    original_forward = emb.forward
    original_heads = emb._apply_rme_output_heads

    def capture_init(module, args, out):
        ctx = getattr(emb, "_latent_ctx", None)
        if ctx is not None:
            ctx["active_edges"] = out[4].detach().clone()

    emb.init_layer.register_forward_hook(capture_init)

    def capture_heads(node, edge, node_one_hot, edge_one_hot):
        ctx = getattr(emb, "_latent_ctx", None)
        if ctx is not None:
            ctx["initial"] = (node.detach(), edge.detach())
            ctx["onehot"] = (node_one_hot.detach(), edge_one_hot.detach())
        return original_heads(node, edge, node_one_hot, edge_one_hot)

    emb._apply_rme_output_heads = capture_heads

    def forward(data):
        ctx = getattr(emb, "_latent_ctx", None)
        if ctx is None:
            raise RuntimeError("latent embedding must run inside its loop wrapper")
        if "template" not in ctx:
            # Frozen encoder must not acquire dropout/BN updates when Trainer
            # calls model.train(). Readout is re-applied below with autograd.
            emb.eval()
            with torch.no_grad():
                result = original_forward(data)
            if "initial" not in ctx:
                raise RuntimeError(
                    "latent corrector requires the RME output-head route"
                )
            ctx["template"] = {
                k: v.detach().clone() if torch.is_tensor(v) else v
                for k, v in result.items()
            }
            ctx["encoder_calls"] = ctx.get("encoder_calls", 0) + 1
        initial = ctx["initial"]
        node, edge = (
            initial if ctx["strategy"] == "reset" else ctx.get("hidden", initial)
        )
        if ctx["strategy"] == "detach":
            node, edge = node.detach(), edge.detach()
        active = ctx["active_edges"]
        ei = ctx["edge_index"].index_select(1, active)
        node, edge = emb.latent_core(node, edge, ei)
        ctx["hidden"] = (node, edge)
        if readout == "residual":
            nout, eout = emb.latent_readout(node, edge)
        else:
            nout, eout = original_heads(node, edge, *ctx["onehot"])
        result = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in ctx["template"].items()
        }
        full_edge = eout.new_zeros(
            ctx["edge_index"].shape[1], eout.shape[-1]
        ).index_copy(0, active, eout)
        if readout == "residual":
            result[A.NODE_FEATURES_KEY] = result[A.NODE_FEATURES_KEY] + nout
            result[A.EDGE_FEATURES_KEY] = result[A.EDGE_FEATURES_KEY] + full_edge
        else:
            result[A.NODE_FEATURES_KEY] = nout
            result[A.EDGE_FEATURES_KEY] = full_edge
        return result

    emb.forward = forward


def install_latent_corrector(
    model, idp, K=2, strategy="bptt", n_k_train=5, readout="residual"
):
    """Install a shared core per expert; K counts actual corrector applications.

    Strategies share architecture/init/loss. 'detach' truncates only the hidden
    state between steps; 'reset' reads the original state at every step.
    Backbone and existing readout tensors are frozen in all three strategies.
    The default adds learned R(z_t) to the frozen base. readout='frozen' is the
    narrower pilot that reuses the existing head on z_t.
    """
    if K < 1 or n_k_train < 1 or strategy not in ("bptt", "detach", "reset"):
        raise ValueError("invalid latent depth, k count, or strategy")
    if readout not in ("frozen", "residual"):
        raise ValueError("invalid latent readout")
    if getattr(model, "transform", None) is not True or getattr(idp, "has_soc", False):
        raise ValueError("latent pilot requires non-SOC transform=True")
    if hasattr(model, "_wm_prepare"):
        raise ValueError("install on a pristine base model, not another loop wrapper")
    embs = _iter_embeddings(model)
    for _, emb in embs:
        if getattr(emb.output_route_spec, "output_contract", None) == "ao_block":
            raise ValueError("latent pilot does not support block-native output")
    for p in model.parameters():
        p.requires_grad_(False)
    parameter = next(model.parameters())
    converter = AOPriorToRME(idp, dtype=parameter.dtype, device=parameter.device)
    for _, emb in embs:
        emb.latent_core = SharedLatentCore(
            emb.layers[-1].irreps_out, zero_init=(readout == "frozen")
        ).to(parameter)
        if readout == "residual":
            emb.latent_readout = ResidualReadout(
                emb.layers[-1].irreps_out, emb.idp.orbpair_irreps
            ).to(parameter)
        _wrap_cached_embedding(emb, readout)
    original_forward = model.forward

    def prepare(batch):
        at = batch[A.ATOM_TYPE_KEY].reshape(-1)
        bi = batch.get(A.BATCH_KEY)
        ng = int(bi.max()) + 1 if bi is not None and bi.numel() else 1
        with torch.no_grad():
            h0 = converter(batch)
        return {
            "n_atom": len(at),
            "h0_features": h0,
            "kpts": (
                torch.rand(ng, n_k_train, 3, device=at.device)
                if model.training
                else None
            ),
            "contexts": [
                {"edge_index": batch[A.EDGE_INDEX_KEY], "strategy": strategy}
                for _ in embs
            ],
        }

    def one_k(batch, k, _wn, _we, state):
        b = dict(batch)
        b[A.NODE_H0_KEY], b[A.EDGE_H0_KEY] = state["h0_features"]
        for (_, emb), ctx in zip(embs, state["contexts"]):
            emb._latent_ctx = ctx
        try:
            out = original_forward(b)
        finally:
            for _, emb in embs:
                emb._latent_ctx = None
        out[A.NODE_H0_KEY] = batch[A.NODE_H0_KEY]
        out[A.EDGE_H0_KEY] = batch[A.EDGE_H0_KEY]
        if model.training:
            out["_loop_kpts"] = state["kpts"]
        out["_loop_step"] = k
        out["_loop_K"] = K
        return out

    def forward(batch):
        state = model._wm_prepare(batch)
        predictions = []
        for k in range(1, K + 1):
            out = one_k(batch, k, None, None, state)
            predictions.append((out[A.NODE_FEATURES_KEY], out[A.EDGE_FEATURES_KEY]))
        out["_loop_preds"] = predictions
        return out

    model.forward = forward
    model.register_buffer(
        "_latent_strategy_code",
        torch.tensor(
            {"bptt": 0, "detach": 1, "reset": 2}[strategy], device=parameter.device
        ),
    )
    model._wm_prepare = prepare
    model._wm_one_k = one_k
    model._wm_update = lambda out, state: (None, None)
    model._wm_K = K
    model._loopscf_full_bptt = True  # One ordinary backward of the summed objective.
    model._latent_strategy = strategy
    model._latent_readout = readout
    return model


def validate_latent_checkpoint(state_dict, strategy, readout="frozen"):
    saved_readout = (
        "residual" if any(".latent_readout." in k for k in state_dict) else "frozen"
    )
    if saved_readout != readout:
        raise ValueError("saved latent readout differs from requested readout")
    code = state_dict.get("_latent_strategy_code")
    if (
        code is None
        or code.numel() != 1
        or int(code) != {"bptt": 0, "detach": 1, "reset": 2}[strategy]
    ):
        raise ValueError("saved latent strategy differs from requested strategy")

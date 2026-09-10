"""Pretrained whole-stack recurrence and graph-level adaptive computation.

The pretrained stack has different input/output irreps. A learned equivariant
residual bridge closes that interface; it is not an AO/Hamiltonian feedback.
K=1 exactly follows the corrected-H0 base. Zero bridge initialization also
preserves its initial predictions at deeper K, while allowing bridge gradients.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from e3nn import o3
from dptb.data import AtomicDataDict as A
from .adapters import _iter_embeddings
from .representation import AOPriorToRME


class StackBridge(nn.Module):
    def __init__(self, output_irreps, input_irreps, latent_dim):
        super().__init__()
        self.node = o3.Linear(output_irreps, input_irreps, biases=False)
        self.edge = o3.Linear(output_irreps, input_irreps, biases=False)
        nn.init.zeros_(self.node.weight)
        nn.init.zeros_(self.edge.weight)
        self.latent_mix = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, previous_input, output):
        li, ni, ei = previous_input
        lo, no, eo = output[:3]
        # High-l components absent from the output survive in ni/ei.
        return (
            li + self.latent_mix.tanh() * (lo - li),
            ni + self.node(no),
            ei + self.edge(eo),
        )


class GraphExitGate(nn.Module):
    """Rotation invariant shared exit gate, one decision per whole graph."""

    def __init__(self, irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.net = nn.Sequential(
            nn.Linear(self.irreps.num_irreps, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, node, batch):
        blocks = [
            node[:, s].reshape(len(node), mul, ir.dim)
            for s, (mul, ir) in zip(self.irreps.slices(), self.irreps)
        ]
        norm = torch.cat([(b.square().mean(-1) + 1e-8).sqrt() for b in blocks], -1)
        ng = int(batch.max()) + 1
        pooled = norm.new_zeros(ng, norm.shape[-1]).index_add(0, batch, norm)
        pooled = pooled / torch.bincount(batch, minlength=ng).clamp_min(1)[:, None]
        return self.net(torch.log1p(pooled)).squeeze(-1)


def exit_distribution(logits):
    """Hazard -> normalized first-exit law; final step absorbs all survival."""
    K = logits.shape[-1]
    survival = torch.ones_like(logits[..., 0])
    masses = []
    for t in range(K - 1):
        # Fixed remaining-depth offset makes zero learned logits uniform.
        hazard = (logits[..., t] - math.log(K - t - 1)).sigmoid()
        masses.append(survival * hazard)
        survival = survival * (1 - hazard)
    return torch.stack(masses + [survival], -1)


def adaptive_objective(losses, probabilities, beta):
    """Per-graph task losses and probabilities both have shape [graphs, K]."""
    if losses.shape != probabilities.shape or beta < 0:
        raise ValueError("matching per-graph losses/probabilities and beta>=0 required")
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    return ((probabilities * losses).sum(-1) - beta * entropy).mean(), entropy


def predict_until_exit(model, batch, max_steps=3, quantile=0.5):
    """Actually stop the stack at the cumulative-probability threshold.

    Single-graph inference keeps compute accounting exact. Batched variable
    exits need graph compaction and are deliberately not simulated by masking.
    """
    if model.training or max_steps < 1 or not 0 <= quantile <= 1:
        raise ValueError("eval mode, positive depth and quantile in [0,1] required")
    bi = batch.get(A.BATCH_KEY)
    if bi is not None and int(bi.max()) != 0:
        raise ValueError("early-exit inference currently supports one graph")
    state = model._wm_prepare(batch)
    survival = 1.0
    for k in range(1, max_steps + 1):
        out = model._wm_one_k(batch, k, None, None, state)
        logit = out["_stack_logit"]
        if not bool(torch.isfinite(logit).all()):
            raise RuntimeError("nonfinite exit logit")
        if k == max_steps:
            survival = 0.0
        else:
            survival *= 1 - float((logit - math.log(max_steps - k)).sigmoid())
        if 1 - survival >= quantile or k == max_steps:
            out["_exit_step"] = k
            out["_exit_cdf"] = 1 - survival
            out["_stack_counts"] = [
                (c["encoder_calls"], c["stack_calls"]) for c in state["contexts"]
            ]
            return out


def graph_hamiltonian_losses(node, edge, ref, idp):
    """HamilLossAbs per graph: mean of onsite/hopping (L1+RMSE)/2.

    Equal graph weighting is intentional for a graph-level exit decision.
    Padded invalid orbital entries do not enter either numerator or denominator.
    """
    from dptb.nnops.loss import _nrme_mask, _erme_mask

    batch = ref[A.BATCH_KEY].flatten()
    ng = int(batch.max()) + 1
    at, et = ref[A.ATOM_TYPE_KEY].flatten(), ref[A.EDGE_TYPE_KEY].flatten()
    masks = (
        _nrme_mask(idp, at, result_device=node.device),
        _erme_mask(idp, et, result_device=edge.device),
    )
    groups = (batch, batch[ref[A.EDGE_INDEX_KEY][0]])
    parts = []
    for pred, key, mask, group in zip(
        (node, edge), (A.NODE_FEATURES_KEY, A.EDGE_FEATURES_KEY), masks, groups
    ):
        diff = (pred - ref[key]) * mask
        counts = pred.new_zeros(ng).index_add(0, group, mask.sum(-1).to(pred.dtype))
        l1 = pred.new_zeros(ng).index_add(
            0, group, diff.abs().sum(-1)
        ) / counts.clamp_min(1)
        mse = pred.new_zeros(ng).index_add(
            0, group, diff.square().sum(-1)
        ) / counts.clamp_min(1)
        # Same safe zero convention as the base matrix loss.
        rmse = torch.where(mse > 0, mse.clamp_min(1e-24).sqrt(), torch.zeros_like(mse))
        parts.append(0.5 * (l1 + rmse))
    return 0.5 * (parts[0] + parts[1])


def _wrap_stack(emb):
    original_forward = emb.forward
    heads = emb._apply_rme_output_heads

    def capture_input(module, args):
        ctx = getattr(emb, "_stack_ctx", None)
        if ctx is not None and "args" not in ctx:
            # Ensemble stitching mutates EDGE_OVERLAP in place, which the
            # legacy embedding aliases to initial scalar latents. Preserve
            # independent recurrent tensors without detaching their gradients.
            ctx["args"] = tuple(x.clone() if torch.is_tensor(x) else x for x in args)
            ctx["input"] = ctx["args"][:3]

    def capture_output(module, args, output):
        ctx = getattr(emb, "_stack_ctx", None)
        if ctx is not None:
            ctx["output"] = output

    emb.layers[0].register_forward_pre_hook(capture_input)
    emb.layers[-1].register_forward_hook(capture_output)

    def forward(data):
        ctx = emb._stack_ctx
        if "template" not in ctx:
            result = original_forward(data)
            result = {
                k: v.clone() if torch.is_tensor(v) else v for k, v in result.items()
            }
            ctx["template"] = {
                k: v.clone() if torch.is_tensor(v) else v for k, v in result.items()
            }
            ctx["encoder_calls"] = 1
        else:
            old_input, old_output = ctx["input"], ctx["output"]
            if ctx["strategy"] == "detach":
                old_input = tuple(x.detach() for x in old_input)
                old_output = tuple(
                    x.detach() if torch.is_tensor(x) else x for x in old_output
                )
            state = emb.stack_bridge(old_input, old_output)
            ctx["input"] = state
            args = ctx["args"]
            latents, node, edge = state
            wigner = old_output[3]
            for layer in emb.layers:
                latents, node, edge, wigner = layer(
                    latents, node, edge, *args[3:10], wigner, *args[11:]
                )
            full_count = len(ctx["template"][A.ATOM_TYPE_KEY])
            if len(node) < full_count:
                node = torch.cat(
                    [node, node.new_zeros(full_count - len(node), node.shape[-1])], 0
                )
            nout, eout = heads(node, edge, ctx["template"][A.NODE_ATTRS_KEY], args[9])
            active = args[8]
            result = {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in ctx["template"].items()
            }
            result[A.NODE_FEATURES_KEY] = nout
            result[A.EDGE_FEATURES_KEY] = eout.new_zeros(
                args[4].shape[1], eout.shape[-1]
            ).index_copy(0, active, eout)
        ctx["stack_calls"] = ctx.get("stack_calls", 0) + 1
        gate_node = ctx["output"][1]
        batch = ctx["template"][A.BATCH_KEY].flatten()
        if len(gate_node) < len(batch):
            gate_node = torch.cat(
                [
                    gate_node,
                    gate_node.new_zeros(
                        len(batch) - len(gate_node), gate_node.shape[-1]
                    ),
                ],
                0,
            )
        ctx["logit"] = emb.stack_exit(gate_node, batch)
        return result

    emb.forward = forward


def install_stack_loop(model, idp, K=3, strategy="bptt", n_k_train=5):
    """Encode once, reuse all pretrained layers/head, full BPTT by default.

    This version supports the non-SOC graph-MoE H0 RME route without extra
    pair-refinement/flow blocks. All original parameters remain trainable.
    The Python wrapper is reconstructed before strictly loading a checkpoint.
    """
    if K < 1 or strategy not in ("bptt", "detach") or n_k_train < 1:
        raise ValueError("invalid stack recurrence settings")
    if getattr(model, "transform", None) is not True or getattr(idp, "has_soc", False):
        raise ValueError("stack recurrence requires non-SOC transform=True")
    if hasattr(model, "_wm_prepare"):
        raise ValueError("install on a pristine model")
    embs = _iter_embeddings(model)
    parameter = next(model.parameters())
    for _, emb in embs:
        if (
            emb.__class__.__name__ != "LemMoEV3H0"
            or not emb.use_h0_init
            or getattr(emb, "use_block_native_output", False)
            or getattr(emb, "two_stage_pair_enable", False)
            or getattr(emb, "flow_time_conditioner", None) is not None
        ):
            raise ValueError(
                "unsupported stack architecture; do not silently omit blocks"
            )
        emb.stack_bridge = StackBridge(
            emb.layers[-1].irreps_out, emb.layers[0].irreps_in, emb.latent_dim
        ).to(parameter)
        emb.stack_exit = GraphExitGate(emb.layers[-1].irreps_out).to(parameter)
        _wrap_stack(emb)
    model.register_buffer("_stack_version", torch.tensor(1, device=parameter.device))
    model.register_buffer(
        "_stack_strategy_code",
        torch.tensor({"bptt": 0, "detach": 1}[strategy], device=parameter.device),
    )

    def load_check(module, state_dict, prefix, *unused):
        validate_stack_checkpoint(state_dict, strategy, prefix)

    model.register_load_state_dict_pre_hook(load_check)
    original_forward = model.forward
    converter = AOPriorToRME(idp, dtype=parameter.dtype, device=parameter.device)

    def input_signature(batch):
        keys = [
            A.ATOM_TYPE_KEY,
            A.EDGE_TYPE_KEY,
            A.POSITIONS_KEY,
            A.EDGE_INDEX_KEY,
            A.EDGE_CELL_SHIFT_KEY,
            A.CELL_KEY,
            A.BATCH_KEY,
            A.NODE_H0_KEY,
            A.EDGE_H0_KEY,
        ]
        return {
            key: (batch[key].data_ptr(), tuple(batch[key].shape), batch[key]._version)
            for key in keys
            if key in batch and torch.is_tensor(batch[key])
        }

    def prepare(batch):
        at = batch[A.ATOM_TYPE_KEY].reshape(-1)
        bi = batch.get(A.BATCH_KEY)
        ng = int(bi.max()) + 1 if bi is not None else 1
        with torch.no_grad():
            h0 = converter(batch)
        return {
            "n_atom": len(at),
            "h0_features": h0,
            "input_signature": input_signature(batch),
            "kpts": (
                torch.rand(ng, n_k_train, 3, device=at.device)
                if model.training
                else None
            ),
            "contexts": [{"strategy": strategy} for _ in embs],
        }

    def one_k(batch, k, _wn, _we, state):
        if input_signature(batch) != state["input_signature"]:
            raise ValueError(
                "stack context requires the same unmodified graph and H0 tensors"
            )
        b = dict(batch)
        b[A.NODE_H0_KEY], b[A.EDGE_H0_KEY] = state["h0_features"]
        for (_, emb), ctx in zip(embs, state["contexts"]):
            emb._stack_ctx = ctx
        try:
            out = original_forward(b)
        finally:
            for _, emb in embs:
                emb._stack_ctx = None
        out[A.NODE_H0_KEY], out[A.EDGE_H0_KEY] = (
            batch[A.NODE_H0_KEY],
            batch[A.EDGE_H0_KEY],
        )
        if state["kpts"] is not None:
            out["_loop_kpts"] = state["kpts"]
        out["_stack_logit"] = torch.stack([c["logit"] for c in state["contexts"]]).mean(
            0
        )
        return out

    def forward(batch):
        state = model._wm_prepare(batch)
        preds, logits = [], []
        for k in range(1, K + 1):
            out = one_k(batch, k, None, None, state)
            preds.append((out[A.NODE_FEATURES_KEY], out[A.EDGE_FEATURES_KEY]))
            logits.append(out["_stack_logit"])
        out["_loop_preds"] = preds
        out["_stack_logits"] = torch.stack(logits, -1)
        out["_exit_probabilities"] = exit_distribution(out["_stack_logits"])
        out["_stack_counts"] = [
            (c["encoder_calls"], c["stack_calls"]) for c in state["contexts"]
        ]
        return out

    model.forward = forward
    model._wm_prepare, model._wm_one_k = prepare, one_k
    model._wm_K = K
    model._wm_update = lambda out, state: (None, None)
    model._loopscf_full_bptt = True
    return model


def validate_stack_checkpoint(state_dict, strategy="bptt", prefix=""):
    for name, expected in [
        ("_stack_version", 1),
        ("_stack_strategy_code", {"bptt": 0, "detach": 1}[strategy]),
    ]:
        value = state_dict.get(prefix + name)
        if value is None or value.numel() != 1 or int(value) != expected:
            raise ValueError("incompatible stack checkpoint " + name)

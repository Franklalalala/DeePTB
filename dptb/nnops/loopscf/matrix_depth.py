"""Matrix-only depth recurrence, retaining the base H0 and SOC target contract.

No occupation, density, prior conversion or target reconstruction occurs here.
The ordinary model still performs its original readout/CG/SOC assembly each
round.  ``core`` repeats only the middle of a three-layer pretrained stack;
``unshared`` owns separate, initially identical copies of that middle layer.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from dptb.data import AtomicDataDict as A
from .adapters import _iter_embeddings
from .stack import StackBridge, GraphExitGate, exit_distribution
from .latent import SharedLatentCore, ResidualReadout


def clone_data(data):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}


class ReplayDepth:
    """Independent CPU generator, checkpointable without consuming data RNG."""
    def __init__(self, maximum=3, seed=42):
        if maximum < 1:
            raise ValueError("maximum must be positive")
        self.maximum = int(maximum)
        self.generator = torch.Generator().manual_seed(int(seed))

    def sample(self):
        return int(torch.randint(1, self.maximum + 1, (), generator=self.generator))

    def state_dict(self):
        return {"maximum": self.maximum, "rng": self.generator.get_state()}

    def load_state_dict(self, state):
        if state["maximum"] != self.maximum:
            raise ValueError("random-depth maximum changed")
        self.generator.set_state(state["rng"].cpu())


def _heads(emb, node, edge, node_onehot, edge_onehot):
    # The edge-routed class in 0924-stable has an inline hidden-space onehot
    # head. Preserve that exact implementation rather than substituting the
    # graph-routed output-route conditional.
    if emb.__class__.__name__ == "LemMoEV3EdgeH0":
        from dptb.nn.embedding.lem_moe_v3 import _apply_onehot_tp
        n, e = emb.out_node(node), emb.out_edge(edge)
        if emb.use_out_onehot_tp:
            n = n + _apply_onehot_tp(emb.out_node_ele_tp, node, node_onehot, emb.onehot_tp_mode)
            e = e + _apply_onehot_tp(emb.out_edge_ele_tp, edge, edge_onehot, emb.onehot_tp_mode)
        return n, e
    return emb._apply_rme_output_heads(node, edge, node_onehot, edge_onehot)


def _install_embedding(emb, mode, maximum):
    original = emb.forward
    parameter = next(emb.parameters())
    if mode == "stack":
        emb.depth_bridge = StackBridge(emb.layers[-1].irreps_out,
                                       emb.layers[0].irreps_in, emb.latent_dim).to(parameter)
    if mode in ("core", "unshared"):
        if len(emb.layers) != 3 or emb.layers[1].irreps_in != emb.layers[1].irreps_out:
            raise ValueError("middle-core recurrence requires three layers and equal core irreps")
        if mode == "unshared":
            # Deep copies precede hook installation: no copied hook closures.
            emb.depth_extra_cores = nn.ModuleList([copy.deepcopy(emb.layers[1]) for _ in range(maximum - 1)])
    if mode == "latent":
        emb.depth_latent = SharedLatentCore(emb.layers[-1].irreps_out, zero_init=False).to(parameter)
        emb.depth_readout = ResidualReadout(emb.layers[-1].irreps_out, emb.idp.orbpair_irreps).to(parameter)
    emb.depth_gate = GraphExitGate(emb.layers[-1].irreps_out).to(parameter)

    def capture_input(module, args):
        ctx = getattr(emb, "_depth_context", None)
        if ctx is not None and "args" not in ctx:
            ctx["args"] = tuple(x.clone() if torch.is_tensor(x) else x for x in args)
            ctx["input"] = ctx["args"][:3]

    def capture_middle(module, args, output):
        ctx = getattr(emb, "_depth_context", None)
        if ctx is not None:
            ctx["middle"] = output

    def capture_last(module, args, output):
        ctx = getattr(emb, "_depth_context", None)
        if ctx is not None:
            ctx["output"] = output

    emb.layers[0].register_forward_pre_hook(capture_input)
    if mode in ("core", "unshared"):
        emb.layers[1].register_forward_hook(capture_middle)
    emb.layers[-1].register_forward_hook(capture_last)

    def call(layer, state, args):
        return layer(*state[:3], *args[3:10], state[3], *args[11:])

    def forward(data):
        ctx = getattr(emb,'_depth_context',None)
        if ctx is None:
            raise RuntimeError('matrix-depth embedding requires its installed model wrapper')
        first = "template" not in ctx
        if first:
            if mode == "latent":
                # Frozen modules stay in evaluation mode; adapters remain live.
                for name, child in emb.named_children():
                    if not name.startswith("depth_"):
                        child.eval()
                with torch.no_grad():
                    result = original(data)
            else:
                result = original(data)
            ctx["template"] = clone_data(result)
            ctx["encoder_calls"] = 1
        args = ctx["args"]
        if not first and mode == "stack":
            state = emb.depth_bridge(ctx["input"], ctx["output"])
            ctx["input"] = state
            state = (*state, ctx["output"][3])
            for layer in emb.layers:
                state = call(layer, state, args)
            ctx["output"] = state
        elif not first and mode in ("core", "unshared"):
            core = emb.layers[1] if mode == "core" else emb.depth_extra_cores[ctx["round"] - 2]
            state = call(core, ctx["middle"], args)
            ctx["middle"] = state
            ctx["output"] = call(emb.layers[-1], state, args)
        state = ctx["output"]
        node, edge = state[1:3]
        full_nodes = len(ctx["template"][A.ATOM_TYPE_KEY])
        if len(node) < full_nodes:
            node = torch.cat([node, node.new_zeros(full_nodes-len(node), node.shape[-1])])
        if mode == "latent":
            node, edge = ctx.get("latent", (node.detach(), edge.detach()))
            node, edge = emb.depth_latent(node, edge, args[4].index_select(1, args[8]))
            ctx["latent"] = (node, edge)
            nout, eout = emb.depth_readout(node, edge)
            result = clone_data(ctx["template"])
            result[A.NODE_FEATURES_KEY] = result[A.NODE_FEATURES_KEY] + nout
            result[A.EDGE_FEATURES_KEY] = result[A.EDGE_FEATURES_KEY] + eout.new_zeros(
                args[4].shape[1], eout.shape[-1]).index_copy(0, args[8], eout)
        elif not first:
            nout, eout = _heads(emb, node, edge, ctx["template"][A.NODE_ATTRS_KEY], args[9])
            result = clone_data(ctx["template"])
            result[A.NODE_FEATURES_KEY] = nout
            result[A.EDGE_FEATURES_KEY] = eout.new_zeros(args[4].shape[1], eout.shape[-1]).index_copy(0, args[8], eout)
        ctx["logit"] = emb.depth_gate(node, ctx["template"][A.BATCH_KEY].flatten())
        ctx["round_calls"] = ctx.get("round_calls", 0) + 1
        return clone_data(result)

    emb.forward = forward


def install_matrix_depth(model, mode="stack", maximum=6):
    """Install on a strictly loaded base; K=1 retains its physical input path.

    Compact SOC uu-real uses its original mapper/head. Full-spinor H0 input
    remains unsupported by the base initializer. An unshared model cannot extrapolate beyond its
    declared independent cores; callers must not silently reuse its last core.
    """
    if mode not in ("stack", "core", "unshared", "latent") or maximum < 1:
        raise ValueError("invalid matrix-depth settings")
    if hasattr(model, "_matrix_depth_mode"):
        raise ValueError("install on a pristine model")
    embeddings = _iter_embeddings(model)
    if len(embeddings)!=1 or hasattr(model,'_wm_prepare'):
        raise ValueError('matrix-depth requires a pristine single-embedding model')
    for _, emb in embeddings:
        if getattr(emb.idp, "has_soc", False) and not getattr(emb.idp, "soc_uureal_target", False):
            raise ValueError("matrix-depth SOC is qualified only for compact uu-real targets")
        if emb.__class__.__name__ not in ("LemMoEV3H0", "LemMoEV3EdgeH0") or not emb.use_h0_init:
            raise ValueError("unsupported embedding")
        if any(bool(getattr(emb, flag, False)) for flag in
               ("use_block_native_output", "two_stage_pair_enable", "pair_refine_enable")) or getattr(emb, "flow_time_conditioner", None) is not None:
            raise ValueError("matrix-depth cannot omit auxiliary prediction blocks")
    if mode == "latent":
        for p in model.parameters():
            p.requires_grad_(False)
    for _, emb in embeddings:
        _install_embedding(emb, mode, maximum)
    original = model.forward
    model._matrix_depth_mode = mode
    model._matrix_depth_K = 3
    codes={'stack':1,'core':2,'unshared':3,'latent':4}
    contract={'_matrix_depth_version':2,'_matrix_depth_mode_code':codes[mode],'_matrix_depth_maximum':maximum}
    for key,value in contract.items():
        model.register_buffer(key,torch.tensor(value,device=next(model.parameters()).device))
    def check_contract(module,state_dict,prefix,*_):
        for key,value in contract.items():
            saved=state_dict.get(prefix+key)
            if saved is None or saved.numel()!=1 or int(saved)!=value:
                raise RuntimeError('matrix-depth checkpoint contract mismatch: '+prefix+key)
    model.register_load_state_dict_pre_hook(check_contract)

    def forward(batch):
        depth = int(model._matrix_depth_K)
        quantile = getattr(model, "_matrix_depth_exit_quantile", None)
        if quantile is not None:
            bi = batch.get(A.BATCH_KEY)
            if model.training or not 0 <= quantile <= 1 or (bi is not None and int(bi.max()) != 0):
                raise ValueError("adaptive inference requires eval mode, one graph and quantile in [0,1]")
        survival = 1.0
        observed_masses = []
        if depth < 1 or (mode == "unshared" and depth > maximum):
            raise ValueError("depth outside constructed range")
        contexts = [{} for _ in embeddings]
        preds, logits = [], []
        try:
            for k in range(1, depth+1):
                for (_, emb), ctx in zip(embeddings, contexts):
                    ctx["round"] = k
                    emb._depth_context = ctx
                out = original(clone_data(batch))
                preds.append((out[A.NODE_FEATURES_KEY], out[A.EDGE_FEATURES_KEY]))
                logits.append(torch.stack([c["logit"] for c in contexts]).mean(0))
                if quantile is not None:
                    import math
                    if not bool(torch.isfinite(logits[-1]).all()):
                        raise RuntimeError("nonfinite exit logit")
                    hazard = 1.0 if k == depth else float((logits[-1] - math.log(depth-k)).sigmoid())
                    observed_masses.append(survival * hazard)
                    survival *= 1-hazard
                    if 1-survival >= quantile or k == depth:
                        break
        finally:
            for _, emb in embeddings:
                emb._depth_context = None
        out["_loop_preds"] = preds
        out["_stack_logits"] = torch.stack(logits, -1)
        if quantile is None:
            out["_exit_probabilities"] = exit_distribution(out["_stack_logits"])
        else:
            out["_exit_step"] = len(preds)
            out["_exit_cdf"] = 1-survival
            out["_observed_exit_masses"] = observed_masses
            out["_remaining_survival"] = survival
        out["_stack_counts"] = [(c["encoder_calls"], c["round_calls"]) for c in contexts]
        return out

    model.forward = forward
    return model


def matrix_predict_until_exit(model, batch, max_steps=3, quantile=0.5):
    """Single-graph deployment policy; no unexecuted round is computed."""
    old_depth = model._matrix_depth_K
    old_quantile = getattr(model, "_matrix_depth_exit_quantile", None)
    model._matrix_depth_K = max_steps
    model._matrix_depth_exit_quantile = quantile
    try:
        return model(batch)
    finally:
        model._matrix_depth_K = old_depth
        model._matrix_depth_exit_quantile = old_quantile

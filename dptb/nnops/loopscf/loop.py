"""Iterative Hamiltonian prediction with charge feedback and optional diagnostics."""

from __future__ import annotations

from torch import nn
import torch
from dptb.data import AtomicDataDict
from .constants import K_DEFAULT, CORRECTNESS_VERSION
from .kspace import build_k_plan, bloch_phase, assemble_flat
from .occupations import (
    factor_overlap_robust,
    compute_mulliken_fast,
    _eval_occupation_kpoints,
)
from .adapters import _iter_embeddings, _attach_adapters, _wrap_embedding
from .representation import AOScalarFeedback, AOPriorToRME


def install_working_memory_true_diag(
    model: nn.Module,
    mode: str,
    idp,
    h2k=None,
    s2k=None,
    K: int = K_DEFAULT,
    max_k: int = 5,
    n_k_train: int = 5,
    collect_diagnostics: bool = False,
    feedback: bool = True,
    overlap_cutoff: float = 1e-5,
) -> nn.Module:
    """Wraps model forward with True Diagonalization Working Memory loop.

    Training draws ``n_k_train`` random fractional k-points per graph (same
    draw for occupy and fw10). Evaluation occupations use a fixed uniform-BZ
    sample; the band metric independently keeps the dataset DFT path.
    """
    if mode not in ("head", "moe"):
        raise ValueError("mode must be 'head' or 'moe', got %r" % mode)
    if K < 1 or max_k < 1 or n_k_train < 1:
        raise ValueError("K, max_k and n_k_train must be positive")
    if bool(getattr(idp, "has_soc", False)):
        raise NotImplementedError(
            "LoopSCF occupations require a non-SOC spin-degenerate model"
        )
    if getattr(model, "transform", None) is not True:
        raise ValueError(
            "LoopSCF requires transform=True: model outputs must be AO blocks"
        )
    print(
        "[WM-TrueDiag] correctness=%s occupations=global-zero-T eval_k=Sobol-BZ seed=20260910"
        % CORRECTNESS_VERSION,
        flush=True,
    )

    node_key = AtomicDataDict.NODE_FEATURES_KEY
    edge_key = AtomicDataDict.EDGE_FEATURES_KEY

    embs = _iter_embeddings(model)
    print(
        "[WM-TrueDiag] found %d embedding modules, mode=%s K=%d n_k_train=%d (dptb fast enabled)"
        % (len(embs), mode, K, int(n_k_train)),
        flush=True,
    )
    for name, emb in embs:
        print("[WM-TrueDiag] wrap", name, type(emb).__name__, flush=True)
        _attach_adapters(emb, mode)
        _wrap_embedding(emb, mode)

    ao_scalars = AOScalarFeedback(idp)
    reference_parameter = next(model.parameters())
    prior_to_rme = AOPriorToRME(
        idp, dtype=reference_parameter.dtype, device=reference_parameter.device
    )
    if any(emb._wm_n_out != ao_scalars.n_scalars for _, emb in embs):
        raise ValueError("embedding and LoopSCF orbital bases do not match")
    orig_fwd = model.forward

    def _get_field(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        try:
            if key in obj:
                return obj[key]
        except Exception:
            pass
        return getattr(obj, key, default)

    def _occupy_all(buf_H, state):
        qs = []
        at = state["atom_types"].reshape(-1)
        ptr = state["ptr"]
        plan = state["plan"]
        factors = state["factors"]
        nelec_t = state["nelec_t"]
        for g, H_k in enumerate(plan.blocks(buf_H)):
            S_k64, L_inv, bad_g, proj_g = factors[g]
            sl = at[int(ptr[g].item()) : int(ptr[g + 1].item())]
            qs.append(
                compute_mulliken_fast(
                    H_k, S_k64, L_inv, bad_g, proj_g, float(nelec_t[g]), idp, sl
                )
            )
        populations = torch.cat(qs, 0)
        if collect_diagnostics:
            state.setdefault("q_history", []).append(populations.detach().clone())
        return populations

    def _prepare(batch):
        device = next(model.parameters()).device
        atom_types = batch[AtomicDataDict.ATOM_TYPE_KEY]
        edge_index = batch[AtomicDataDict.EDGE_INDEX_KEY]
        shift = batch[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
        n_atom = int(atom_types.reshape(-1).shape[0])

        batch_idx = _get_field(batch, AtomicDataDict.BATCH_KEY)
        if batch_idx is None:
            batch_idx = _get_field(batch, "batch")
        ptr = _get_field(batch, "ptr")
        if batch_idx is None:
            batch_idx = torch.zeros(n_atom, dtype=torch.long, device=device)
        else:
            batch_idx = batch_idx.reshape(-1).to(device=device, dtype=torch.long)
            if batch_idx.numel() != n_atom:
                raise RuntimeError(
                    "batch_idx %s != n_atom %d" % (tuple(batch_idx.shape), n_atom)
                )
        n_graph = int(batch_idx.max().item()) + 1 if n_atom else 1
        if ptr is None or int(ptr.reshape(-1).numel()) != n_graph + 1:
            counts = torch.bincount(batch_idx, minlength=n_graph)
            ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
        else:
            ptr = ptr.reshape(-1).to(device=device, dtype=torch.long)

        nelec_t = batch["nelec"].reshape(-1).to(device=device, dtype=torch.float32)
        if nelec_t.numel() != n_graph:
            raise ValueError(
                "nelec must contain one explicit electron count per graph; got %d for %d graphs"
                % (nelec_t.numel(), n_graph)
            )

        src, dst = edge_index[0], edge_index[1]
        graph_of_edge = batch_idx.index_select(0, src)
        plan = build_k_plan(idp, atom_types, edge_index, batch_idx, ptr, device)

        if model.training:
            kpts = torch.rand(n_graph, int(n_k_train), 3, device=device)
        else:
            kpts = _eval_occupation_kpoints(n_graph, int(max_k), device)
        phase = bloch_phase(kpts, shift, graph_of_edge)

        with torch.no_grad():
            s_node = batch[AtomicDataDict.NODE_OVERLAP_KEY]
            s_edge = batch[AtomicDataDict.EDGE_OVERLAP_KEY]
            buf_S = assemble_flat(plan, s_node, s_edge, phase, torch.complex128)
            overlap_diagnostics = []
            factors = []
            for S in plan.blocks(buf_S):
                diag = {} if collect_diagnostics else None
                factors.append(
                    factor_overlap_robust(S, overlap_cutoff, diagnostics=diag)
                )
                overlap_diagnostics.append(diag)
            nh0 = batch[AtomicDataDict.NODE_H0_KEY]
            eh0 = batch[AtomicDataDict.EDGE_H0_KEY]
            buf_H0 = assemble_flat(plan, nh0, eh0, phase, torch.complex128)
            state = {
                "atom_types": atom_types,
                "ptr": ptr,
                "plan": plan,
                "factors": factors,
                "nelec_t": nelec_t,
                "overlap_diagnostics": overlap_diagnostics,
                "h0_features": prior_to_rme(batch),
            }
            q0 = _occupy_all(buf_H0, state)
            del buf_S, buf_H0

        if int(getattr(model, "_td_bs_log", 0)) < 8:
            print(
                "[WM-TrueDiag] n_graph=%d n_atom=%d kpts=%s train=%s nelec=%s"
                % (
                    n_graph,
                    n_atom,
                    tuple(kpts.shape),
                    bool(model.training),
                    tuple(nelec_t.shape),
                ),
                flush=True,
            )
            model._td_bs_log = int(getattr(model, "_td_bs_log", 0)) + 1

        state.update(
            {
                "kpts": kpts,
                "phase": phase,
                "q0": q0,
                "nh0": nh0,
                "eh0": eh0,
                "src": src,
                "dst": dst,
                "n_graph": n_graph,
                "n_atom": n_atom,
            }
        )
        return state

    def _one_k(batch, k, wm_node, wm_edge, state):
        # Forward replaces feature/overlap fields; retain pristine inputs for
        # every step, including the first. Tensor data are not copied here.
        b = dict(batch)
        # Physical assembly/losses require AO H0, but H0InitLayer's linear
        # projectors consume RME irreps. Never overload one tensor with both.
        model_h0 = state.get("h0_features")
        if model_h0 is None:
            model_h0 = prior_to_rme(batch)
        b[AtomicDataDict.NODE_H0_KEY], b[AtomicDataDict.EDGE_H0_KEY] = model_h0
        for name, emb in embs:
            if feedback and k > 1 and wm_node is not None and wm_edge is not None:
                emb._wm_ctx = {
                    "wm_n": wm_node.clone(),
                    "wm_e": wm_edge.clone(),
                    "active_edges": None,
                }
            else:
                emb._wm_ctx = None
        try:
            out = orig_fwd(b)
        finally:
            for name, emb in embs:
                emb._wm_ctx = None
        out[AtomicDataDict.NODE_H0_KEY] = batch[AtomicDataDict.NODE_H0_KEY]
        out[AtomicDataDict.EDGE_H0_KEY] = batch[AtomicDataDict.EDGE_H0_KEY]
        if model.training:
            out["_loop_kpts"] = state["kpts"]
        out["_loop_K"] = K
        out["_loop_step"] = k
        return out

    def _update(out, state):
        dH_node = out[node_key]
        dH_edge = out[edge_key]
        H_k_node = state["nh0"] + dH_node.detach()
        H_k_edge = state["eh0"] + dH_edge.detach()
        src, dst = state["src"], state["dst"]
        with torch.no_grad():
            buf_Hk = assemble_flat(
                state["plan"], H_k_node, H_k_edge, state["phase"], torch.complex128
            )
            q_k = _occupy_all(buf_Hk, state)
            dq_k = q_k - state["q0"]
            rn_0e = ao_scalars(dH_node).detach()
            re_0e = ao_scalars(dH_edge).detach()
            wm_node = (
                torch.cat([dq_k.unsqueeze(-1), q_k.unsqueeze(-1), rn_0e], dim=-1)
                .detach()
                .clone()
            )
            dq_edge = (dq_k[dst] - dq_k[src]).unsqueeze(-1)
            q_edge = (q_k[dst] - q_k[src]).unsqueeze(-1)
            wm_edge = torch.cat([dq_edge, q_edge, re_0e], dim=-1).detach().clone()
            del buf_Hk, q_k, dq_k, rn_0e, re_0e, dq_edge, q_edge
        return wm_node, wm_edge

    def looped_forward(batch):
        state = _prepare(batch)
        loop_preds = []
        out = None
        wm_node = None
        wm_edge = None
        for k in range(1, K + 1):
            out = _one_k(batch, k, wm_node, wm_edge, state)
            loop_preds.append((out[node_key], out[edge_key]))
            wm_node, wm_edge = _update(out, state)
        out["_loop_preds"] = loop_preds
        out["_loop_K"] = K
        if model.training:
            out["_loop_kpts"] = state["kpts"]
        if collect_diagnostics:
            out["_loop_q"] = state["q_history"]
            out["_loop_overlap"] = state["overlap_diagnostics"]
        return out

    model.forward = looped_forward
    model._wm_mode = mode
    model._wm_K = K
    model._wm_n_k_train = int(n_k_train)
    model._wm_feedback = bool(feedback)
    model._wm_prepare = _prepare
    model._wm_one_k = _one_k
    model._wm_update = _update
    return model

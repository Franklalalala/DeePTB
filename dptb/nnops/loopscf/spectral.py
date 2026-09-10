"""Per-graph spectral objectives using shared overlap factors."""

from __future__ import annotations

import math
import torch
from dptb.data import AtomicDataDict
from .kspace import build_k_plan, bloch_phase, assemble_flat
from .occupations import factor_overlap_robust


def _field(data, key, default=None):
    if isinstance(data, dict):
        val = data.get(key, default)
        if val is not None:
            return val
        try:
            if key in data:
                return data[key]
        except Exception:
            pass
    return getattr(data, key, default)


def _batch_ptr(data, n_atom, device):
    batch_idx = _field(data, AtomicDataDict.BATCH_KEY)
    if batch_idx is None:
        batch_idx = _field(data, "batch")
    ptr = _field(data, "ptr")
    if batch_idx is None:
        batch_idx = torch.zeros(n_atom, dtype=torch.long, device=device)
        ptr = torch.tensor([0, n_atom], dtype=torch.long, device=device)
        return batch_idx, ptr, 1
    batch_idx = batch_idx.reshape(-1).to(device=device, dtype=torch.long)
    if int(batch_idx.numel()) != n_atom:
        raise RuntimeError(
            "batch_idx %s != n_atom %d" % (tuple(batch_idx.shape), n_atom)
        )
    n_graph = int(batch_idx.max().item()) + 1 if n_atom else 1
    if ptr is None or int(ptr.reshape(-1).numel()) != n_graph + 1:
        counts = torch.bincount(batch_idx, minlength=n_graph)
        ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
    else:
        ptr = ptr.reshape(-1).to(device=device, dtype=torch.long)
    return batch_idx, ptr, n_graph


def eigvals_from_factor(H_k, S_k64, L_inv, bad, proj_cache):
    """Generalized eigenvalues H C = e S C, one graph, every k.

    Healthy k-points use the cached Cholesky factor. Ill-conditioned k-points
    reuse the occupation solver's positive-subspace projection so fw10 does not allocate a second
    full-size eigh of S.
    """
    H_eff = L_inv @ H_k.to(S_k64.dtype) @ L_inv.mH
    H_eff = 0.5 * (H_eff + H_eff.mH)
    ev = torch.linalg.eigvalsh(H_eff)
    if bad is None or len(bad) == 0:
        return ev
    n_orb = int(ev.shape[-1])
    rows = list(ev.unbind(0))
    for ik in bad.tolist():
        if ik >= int(H_k.shape[0]):
            continue
        if proj_cache is not None and ik in proj_cache:
            M_h = proj_cache[ik][0]
            Ht = M_h.mH @ H_k[ik].to(S_k64.dtype) @ M_h
            Ht = 0.5 * (Ht + Ht.mH)
            ev_b = torch.linalg.eigvalsh(Ht)
        else:
            w, V = torch.linalg.eigh(S_k64[ik])
            healthy = w > 1e-5
            if int(healthy.sum()) == 0:
                rows[ik] = ev.new_full((n_orb,), 1e4)
                continue
            Vh = V[:, healthy]
            wh = w[healthy].clamp_min(1e-8)
            M_h = Vh * (1.0 / torch.sqrt(wh.to(S_k64.dtype))).unsqueeze(0)
            Ht = M_h.mH @ H_k[ik].to(S_k64.dtype) @ M_h
            ev_b = torch.linalg.eigvalsh(0.5 * (Ht + Ht.mH))
        if int(ev_b.numel()) < n_orb:
            ev_b = torch.cat([ev_b, ev_b.new_full((n_orb - int(ev_b.numel()),), 1e4)])
        rows[ik] = ev_b[:n_orb]
    return torch.stack(rows, 0)


def _fw10_one_graph(eig_p, eig_r, nelec, band_window):
    eig_r = eig_r.to(device=eig_p.device, dtype=eig_p.dtype)
    nb = min(int(eig_p.shape[1]), int(eig_r.shape[1]))
    if nb < 2:
        raise RuntimeError("fw10 graph has n_band=%d" % nb)
    eig_p = eig_p[:, :nb]
    eig_r = eig_r[:, :nb]
    n_occ = max(1, min(int(math.ceil(float(nelec) / 2.0)), nb - 1))
    rel_p = eig_p - eig_p[:, n_occ - 1].max()
    rel_r = eig_r - eig_r[:, n_occ - 1].max()
    mask = (rel_r >= -band_window) & (rel_r <= band_window)
    n_in = int(mask.sum())
    if n_in == 0:
        loss = (rel_p - rel_r).abs().mean()
    else:
        loss = (rel_p[mask] - rel_r[mask]).abs().mean()
    return loss, n_occ, nb, n_in


def patch_fw10_per_graph():
    """Per-graph fw10. Training uses the 5 random k stashed by looped_forward.

    Dataset DFT meshes (144-244 k) are evaluation-only. Training matches
    band_stage2: same n_k for every graph, so collate padding is unused.
    Reference bands are recomputed from label RMEs at those k (labels are
    full H when residual_hamiltonian is false).
    """
    from dptb.nnops.loss import FW10EigLoss

    if getattr(FW10EigLoss._band_loss, "_loopscf_random_k", False):
        print("[WM-TrueDiag] fw10 random-k band loss already patched", flush=True)
        return FW10EigLoss

    def _band_loss(self, pred_phys, ref_phys):
        atom_types = pred_phys[AtomicDataDict.ATOM_TYPE_KEY].reshape(-1)
        edge_index = pred_phys[AtomicDataDict.EDGE_INDEX_KEY]
        shift = pred_phys[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
        device = atom_types.device
        n_atom = int(atom_types.numel())
        batch_idx, ptr, n_graph = _batch_ptr(ref_phys, n_atom, device)
        if int(batch_idx.numel()) != n_atom:
            batch_idx, ptr, n_graph = _batch_ptr(pred_phys, n_atom, device)

        ne = ref_phys.get("nelec", None)
        if ne is None:
            raise KeyError("fw10_eig needs nelec to locate the VBM")
        ne = ne.reshape(-1).to(device=device, dtype=torch.float32)
        if int(ne.numel()) != n_graph:
            raise ValueError("nelec must contain one explicit electron count per graph")

        kpts = _field(pred_phys, "_loop_kpts")
        use_random = kpts is not None
        if use_random:
            kpts = kpts.to(device=device)
            if kpts.ndim == 2:
                kpts = kpts.unsqueeze(0)
            if int(kpts.shape[0]) != n_graph:
                raise RuntimeError(
                    "_loop_kpts %s first dim != n_graph %d"
                    % (tuple(kpts.shape), n_graph)
                )
        else:
            kpts = _field(ref_phys, AtomicDataDict.KPOINT_KEY)
            if kpts is None:
                kpts = _field(pred_phys, AtomicDataDict.KPOINT_KEY)
            if kpts is None:
                raise KeyError("fw10 per-graph needs kpoint or _loop_kpts")
            if kpts.ndim == 2:
                kpts = kpts.unsqueeze(0)
            kpts = kpts.to(device=device)

        src = edge_index[0]
        graph_of_edge = batch_idx.index_select(0, src)
        plan = build_k_plan(self.idp, atom_types, edge_index, batch_idx, ptr, device)
        phase = bloch_phase(kpts, shift, graph_of_edge)

        s_node = ref_phys[AtomicDataDict.NODE_OVERLAP_KEY]
        s_edge = ref_phys[AtomicDataDict.EDGE_OVERLAP_KEY]
        h_node = pred_phys[AtomicDataDict.NODE_FEATURES_KEY]
        h_edge = pred_phys[AtomicDataDict.EDGE_FEATURES_KEY]
        if h_node.dtype in (torch.float32, torch.complex64):
            ctype = torch.complex64
        else:
            ctype = torch.complex128

        with torch.no_grad():
            buf_S = assemble_flat(plan, s_node, s_edge, phase, ctype)
            factors = [
                factor_overlap_robust(plan.block(buf_S, g), work_dtype=ctype)
                for g in range(n_graph)
            ]

        buf_H = assemble_flat(plan, h_node, h_edge, phase, ctype)

        if use_random:
            h_ref_node = ref_phys[AtomicDataDict.NODE_FEATURES_KEY]
            h_ref_edge = ref_phys[AtomicDataDict.EDGE_FEATURES_KEY]
            with torch.no_grad():
                buf_Href = assemble_flat(plan, h_ref_node, h_ref_edge, phase, ctype)
                eig_refs = [
                    eigvals_from_factor(plan.block(buf_Href, g), *factors[g])
                    for g in range(n_graph)
                ]
        else:
            eig_r_all = ref_phys[AtomicDataDict.ENERGY_EIGENVALUE_KEY]
            if torch.is_tensor(eig_r_all) and getattr(eig_r_all, "is_nested", False):
                eig_r_all = eig_r_all[0]
            if eig_r_all.dim() == 2:
                eig_r_all = eig_r_all.unsqueeze(0)
            eig_refs = None

        losses = []
        n_in_total = 0
        n_occ_last = 1
        nb_last = 1
        norb_note = []
        nk_note = []
        for g in range(n_graph):
            S_k64, L_inv, bad, proj = factors[g]
            H_k = plan.block(buf_H, g)
            eig_p = eigvals_from_factor(H_k, S_k64, L_inv, bad, proj)
            if use_random:
                eig_r = eig_refs[g]
            else:
                nk = int(kpts.shape[1])
                eig_r = eig_r_all[g, :nk]
            lg, n_occ, nb, n_in = _fw10_one_graph(
                eig_p, eig_r, float(ne[g]), self.band_window
            )
            losses.append(lg)
            n_in_total += n_in
            n_occ_last, nb_last = n_occ, nb
            norb_note.append(int(plan.norb[g]))
            nk_note.append(int(eig_p.shape[0]))

        loss = torch.stack(losses).mean()
        self._fw_parts = {
            "n_occ": int(n_occ_last),
            "n_bands": int(nb_last),
            "n_in_window": int(n_in_total),
            "n_graph": int(n_graph),
            "norb": norb_note,
            "n_k": nk_note,
            "random_k": bool(use_random),
            "fw10": float(loss.detach()),
        }
        n_log = int(getattr(self, "_fw_pg_log", 0))
        if n_log < 8:
            print(
                "[FW10-per-graph] n_graph=%d n_atom=%d norb=%s n_k=%s random_k=%s fw10=%.6g"
                % (
                    n_graph,
                    n_atom,
                    norb_note,
                    nk_note,
                    bool(use_random),
                    self._fw_parts["fw10"],
                ),
                flush=True,
            )
            self._fw_pg_log = n_log + 1
        return loss

    _band_loss._loopscf_random_k = True
    _band_loss._loopscf_per_graph = True
    FW10EigLoss._band_loss = _band_loss
    print(
        "[WM-TrueDiag] patched FW10EigLoss._band_loss: per-graph eigvalsh, "
        "training random k from _loop_kpts (no DFT-mesh pad)",
        flush=True,
    )
    return FW10EigLoss

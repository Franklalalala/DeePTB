"""Sparse orbital-block assembly of Bloch Hamiltonians and overlaps."""

from __future__ import annotations

import math
import re
import torch
from dptb.utils.constants import anglrMId

_ORBPAIR_PLAN_CACHE = {}


def fast_k_supported(idp) -> bool:
    return not bool(getattr(idp, "has_soc", False))


class _OrbpairPlan(object):
    __slots__ = ("row", "col", "src", "fac", "local_dim", "n_rme")

    def __init__(self, row, col, src, fac, local_dim, n_rme):
        self.row = row
        self.col = col
        self.src = src
        self.fac = fac
        self.local_dim = local_dim
        self.n_rme = n_rme


def _orbpair_scatter_plan(idp, device) -> _OrbpairPlan:
    key = (id(idp), str(device))
    plan = _ORBPAIR_PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    if not fast_k_supported(idp):
        raise NotImplementedError("fast k assembly covers the non-SOC scalar path only")
    if not hasattr(idp, "orbpair_maps"):
        idp.get_orbpair_maps()

    local_dim = int(idp.full_basis_norb)
    rows, cols, srcs, facs = [], [], [], []
    ist = 0
    for i, iorb in enumerate(idp.full_basis):
        jst = 0
        li = anglrMId[re.findall(r"[a-zA-Z]", iorb)[0]]
        dim_i = 2 * li + 1
        for j, jorb in enumerate(idp.full_basis):
            lj = anglrMId[re.findall(r"[a-zA-Z]", jorb)[0]]
            dim_j = 2 * lj + 1
            pair = iorb + "-" + jorb
            if i <= j and pair in idp.orbpair_maps:
                sli = idp.orbpair_maps[pair]
                start, stop = int(sli.start), int(sli.stop)
                if stop - start != dim_i * dim_j:
                    raise RuntimeError(
                        "orbpair %s has width %d but the block is %dx%d"
                        % (pair, stop - start, dim_i, dim_j)
                    )
                fac = 0.5 if i == j else 1.0
                for a in range(dim_i):
                    for b in range(dim_j):
                        rows.append(ist + a)
                        cols.append(jst + b)
                        srcs.append(start + a * dim_j + b)
                        facs.append(fac)
            jst += dim_j
        ist += dim_i

    plan = _OrbpairPlan(
        row=torch.as_tensor(rows, dtype=torch.long, device=device),
        col=torch.as_tensor(cols, dtype=torch.long, device=device),
        src=torch.as_tensor(srcs, dtype=torch.long, device=device),
        fac=torch.as_tensor(facs, dtype=torch.float32, device=device),
        local_dim=local_dim,
        n_rme=int(idp.reduced_matrix_element),
    )
    _ORBPAIR_PLAN_CACHE[key] = plan
    return plan


class KBlockPlan(object):
    __slots__ = (
        "n_graph",
        "norb",
        "flat_off",
        "total_flat",
        "n_rme",
        "node_dst",
        "node_src",
        "node_fac",
        "edge_dst",
        "edge_src",
        "edge_fac",
        "edge_row",
        "device",
    )

    def block(self, buf, g, nk=None):
        n = self.norb[g]
        o = self.flat_off[g]
        sl = buf if nk is None else buf[: int(nk)]
        m = sl[:, o : o + n * n].reshape(-1, n, n)
        return m + m.transpose(1, 2).conj()

    def blocks(self, buf):
        return [self.block(buf, g) for g in range(self.n_graph)]


def build_k_plan(idp, atom_types, edge_index, batch, ptr, device, edge_chunk=65536):
    op = _orbpair_scatter_plan(idp, device)
    atom_types = atom_types.flatten().to(device)
    n_atom = int(atom_types.numel())
    n_graph = int(ptr.numel()) - 1

    atom_norb = idp.atom_norb.to(device)[atom_types].long()
    mask = idp.mask_to_basis.to(device)[atom_types]

    csum = torch.cumsum(atom_norb, 0) - atom_norb
    base = csum.index_select(0, ptr[:-1].to(device))
    off_atom = csum - base.index_select(0, batch)

    norb_g = torch.zeros(n_graph, dtype=torch.long, device=device).index_add_(
        0, batch, atom_norb
    )
    flat_size = norb_g * norb_g
    flat_off_t = torch.cumsum(flat_size, 0) - flat_size

    rank = torch.cumsum(mask.long(), -1) - 1
    gidx = off_atom.unsqueeze(1) + rank

    norb_l = [int(x) for x in norb_g.tolist()]
    off_l = [int(x) for x in flat_off_t.tolist()]
    total_flat = int(flat_size.sum().item())

    def _scatter_for(row_atom, col_atom, graph_of, n_item, want_row):
        dsts, srcs, facs, rows = [], [], [], []
        for s in range(0, n_item, edge_chunk):
            e = min(s + edge_chunk, n_item)
            ra = row_atom[s:e]
            ca = col_atom[s:e]
            ge = graph_of[s:e]
            ng = norb_g.index_select(0, ge).unsqueeze(1)
            fo = flat_off_t.index_select(0, ge).unsqueeze(1)
            mr = mask.index_select(0, ra).index_select(1, op.row)
            mc = mask.index_select(0, ca).index_select(1, op.col)
            ok = mr & mc
            R = gidx.index_select(0, ra).index_select(1, op.row)
            C = gidx.index_select(0, ca).index_select(1, op.col)
            dsts.append((fo + R * ng + C)[ok])
            item = torch.arange(s, e, device=device, dtype=torch.long).unsqueeze(1)
            srcs.append((item * op.n_rme + op.src.unsqueeze(0))[ok])
            facs.append(op.fac.unsqueeze(0).expand(e - s, -1)[ok])
            if want_row:
                rows.append(item.expand(-1, op.row.numel())[ok])

        def _cat(xs, dtype=torch.long):
            if not xs:
                return torch.empty(0, device=device, dtype=dtype)
            return xs[0] if len(xs) == 1 else torch.cat(xs)

        return (
            _cat(dsts),
            _cat(srcs),
            _cat(facs, torch.float32),
            _cat(rows) if want_row else None,
        )

    idx_atom = torch.arange(n_atom, device=device, dtype=torch.long)
    n_dst, n_src, n_fac, _ = _scatter_for(idx_atom, idx_atom, batch, n_atom, False)

    src_atom = edge_index[0].to(device)
    dst_atom = edge_index[1].to(device)
    n_edge = int(src_atom.numel())
    graph_of_edge = batch.index_select(0, src_atom)
    e_dst, e_src, e_fac, e_row = _scatter_for(
        src_atom, dst_atom, graph_of_edge, n_edge, True
    )

    plan = KBlockPlan()
    plan.n_graph = n_graph
    plan.norb = norb_l
    plan.flat_off = off_l
    plan.total_flat = total_flat
    plan.n_rme = op.n_rme
    plan.node_dst, plan.node_src, plan.node_fac = n_dst, n_src, n_fac
    plan.edge_dst, plan.edge_src, plan.edge_fac, plan.edge_row = (
        e_dst,
        e_src,
        e_fac,
        e_row,
    )
    plan.device = device
    return plan


def bloch_phase(kpts, shift, graph_of_edge):
    kk = kpts.index_select(0, graph_of_edge)
    theta = (kk * shift.unsqueeze(1)).sum(-1)
    return torch.exp(-1j * 2 * math.pi * theta).transpose(0, 1).contiguous()


def assemble_flat(plan, feats_node, feats_edge, phase, ctype):
    """Assemble packed AO blocks; ctype controls accumulation precision."""
    if feats_node.shape[-1] != plan.n_rme or feats_edge.shape[-1] != plan.n_rme:
        raise ValueError(
            "feature width %s/%s != packed AO width %d"
            % (feats_node.shape[-1], feats_edge.shape[-1], plan.n_rme)
        )
    n_k = phase.shape[0]
    nv = feats_node.reshape(-1).index_select(0, plan.node_src).to(ctype)
    nv = (nv * plan.node_fac).to(ctype)
    ev = feats_edge.reshape(-1).index_select(0, plan.edge_src).to(ctype)
    ev = (ev * plan.edge_fac).to(ctype)
    buf = torch.zeros(n_k, plan.total_flat, dtype=ctype, device=plan.device)
    buf = buf.index_add(1, plan.node_dst, nv.unsqueeze(0).expand(n_k, -1))
    buf = buf.index_add(
        1,
        plan.edge_dst,
        ev.unsqueeze(0) * phase.index_select(1, plan.edge_row).to(ctype),
    )
    return buf

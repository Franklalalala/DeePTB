"""Periodic neighbour enumeration on the device.

All searches of the prior and inference paths reduce to one query: the directed image pairs (i, j, S) of a periodic
structure with |x_j + S @ cell - x_i| below a pair radius r_i + r_j + margin, where S is an integer lattice translation.
The query runs in FP64 on the device in row chunks over all candidate translations, so positions never leave the GPU
and no cell list is kept on the host. Results are sorted by the key (i, j, S0, S1, S2), the order of ``np.unique`` on
key rows.
"""
import torch


def key_codes(i, j, shift, n_atoms, span):
    """Monotone int64 code of keys (i, j, S) with |S| < span; lexicographic order of (i, j, S0, S1, S2)."""
    base = 2 * span + 1
    s = shift.to(torch.int64) + span
    return (((i.to(torch.int64) * n_atoms + j) * base + s[:, 0]) * base + s[:, 1]) * base + s[:, 2]


def translations(frac, radius, cell):
    """Integer translations S (rows, int64) that can bring any atom within ``radius`` of any other atom.

    ``frac`` are the fractional coordinates as given (not wrapped), so S applies to the original positions.
    """
    dual = torch.linalg.inv(cell).norm(dim=0)
    span = frac.max(0).values - frac.min(0).values if len(frac) else torch.zeros(3, dtype=cell.dtype, device=cell.device)
    hi = torch.ceil(span + radius * dual).to(torch.int64) + 1
    axes = [torch.arange(-int(h), int(h) + 1, device=cell.device) for h in hi.tolist()]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)


def image_pairs(pos, cell, radius, *, margin=0.0, strict=False, self_interaction=False, centres=None,
                targets_radius=None, limit_dtype=None, max_candidates=1 << 24, with_vectors=False):
    """Directed image pairs within the pair radius, sorted by (i, j, S).

    pos [n, 3], cell [3, 3]: FP64 device tensors (any length unit, the same for radii).
    radius [n]: per-atom radius r_i; the pair radius is r_i + r_j + margin (``targets_radius`` gives r_j separately).
    strict: compare with ``<`` (ASE neighbour lists) instead of ``<=``.
    limit_dtype: dtype in which r_i + r_j is formed before the FP64 comparison (float32 for the model graph, whose
        per-species cutoffs DeePTB keeps in float32).
    self_interaction: keep (i, i, 0).
    centres: optional subset of rows i (int64 tensor); default all atoms.
    Returns i, j (int64), S [m, 3] (int64) and, with ``with_vectors``, the FP64 vectors x_j + S @ cell - x_i.
    """
    dev = pos.device
    n = len(pos)
    rj = radius if targets_radius is None else targets_radius
    rows = torch.arange(n, device=dev) if centres is None else centres.to(dev)
    empty = torch.zeros(0, dtype=torch.int64, device=dev)
    if n == 0 or len(rows) == 0:
        out = (empty, empty, torch.zeros((0, 3), dtype=torch.int64, device=dev))
        return out + (torch.zeros((0, 3), dtype=pos.dtype, device=dev),) if with_vectors else out
    reach = float(radius[rows].max() + rj.max()) + float(margin)
    frac = pos @ torch.linalg.inv(cell)
    T = translations(frac, reach, cell)
    shift = T.to(pos.dtype) @ cell
    step = max(1, max_candidates // max(1, n * len(T)))
    parts = []
    for a in range(0, len(rows), step):
        r = rows[a:a + step]
        d = pos[None, :, None, :] + shift[None, None, :, :] - pos[r][:, None, None, :]     # [b, n, nT, 3]
        dist = torch.sqrt((d * d).sum(-1))
        if limit_dtype is None:
            lim = radius[r][:, None, None] + rj[None, :, None] + margin
        else:
            lim = (radius[r].to(limit_dtype)[:, None, None] + rj.to(limit_dtype)[None, :, None]).to(pos.dtype) + margin
        keep = dist < lim if strict else dist <= lim
        if not self_interaction:
            keep &= ~((r[:, None, None] == torch.arange(n, device=dev)[None, :, None]) & (T == 0).all(1)[None, None, :])
        ii, jj, tt = torch.nonzero(keep, as_tuple=True)
        part = [r[ii], jj, T[tt]]
        if with_vectors:
            part.append(d[ii, jj, tt])
        parts.append(part)
        del d, dist, keep
    i = torch.cat([p[0] for p in parts])
    j = torch.cat([p[1] for p in parts])
    S = torch.cat([p[2] for p in parts])
    span = int(S.abs().max()) + 1 if len(S) else 1
    order = torch.argsort(key_codes(i, j, S, n, span))
    out = (i[order], j[order], S[order])
    if with_vectors:
        out = out + (torch.cat([p[3] for p in parts])[order],)
    return out


def span_of(shift):
    """Smallest span with |S| < span for every row of ``shift``."""
    return int(shift.abs().max()) + 1 if shift.numel() else 1


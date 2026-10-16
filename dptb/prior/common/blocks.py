"""Block keys and padded block tables on the device, shared by the GF and H0 priors and the inference engine.

A block is the matrix element between the orbitals of atom i (home cell) and atom j at the lattice image R, keyed
(i, j, Rx, Ry, Rz). Both priors lay their blocks out the same way (``BlockKeys``): the n onsite blocks (i, i, 0) first,
then the other keys sorted by (i, j, R). The reverse of (i, j, R) is (j, i, -R); a block and its reverse are one
Hermitian pair, M_ji(-R) = M_ij(R)^H. An edge is representative when its row is below its reverse's; nodes are their
own reverse. Quantities that are Hermitian pairs by construction are computed on the nodes and representative edges
only and mirrored (``mirror``); quantities taken from tables of ordered pairs are averaged with their reverse
(``herm``).

A ``BlockTable`` holds the blocks of one quantity: ``keys`` [m, 5] int64, the valid ``rows`` and ``cols`` of every
block and ``values`` [m, R, C] zero-padded to the largest block. Spinful blocks list all up orbitals, then all down
orbitals (DeePTB).
"""
import functools

import numpy as np
import torch
import triton
import triton.language as tl

from dptb.prior.common.neighbors import key_codes, span_of
from dptb.utils.constants import ABACUS2DeePTB


class BlockTable:
    """The blocks of one quantity (see the module docstring): keys [m, 5], valid rows and cols [m], values [m, R, C]."""

    def __init__(self, keys, rows, cols, values):
        self.keys, self.rows, self.cols, self.values = keys, rows, cols, values

    def __len__(self):
        return len(self.keys)

    def to_numpy_blocks(self):
        """{'i_j_Rx_Ry_Rz': block} with each block cut to its valid rows and columns."""
        keys = self.keys.cpu().numpy()
        rows = self.rows.cpu().numpy()
        cols = self.cols.cpu().numpy()
        vals = self.values.cpu().numpy()
        return {'%d_%d_%d_%d_%d' % tuple(k): vals[b, :rows[b], :cols[b]] for b, k in enumerate(keys)}


class KeyIndex:
    """Sorted codes of a key table: row lookup and the row of each reverse key (j, i, -R)."""

    def __init__(self, keys, n_atoms, span=None, checks=None):
        self.n_atoms = n_atoms
        self.span = span_of(keys[:, 2:]) if span is None else span
        codes = self.codes(keys)
        self.sorted, self.order = torch.sort(codes)
        if len(codes) > 1:
            dup = (self.sorted[1:] == self.sorted[:-1]).any()
            if checks is None:
                if bool(dup):
                    raise ValueError('duplicate block keys')
            else:
                checks.fail_if(dup, 'duplicate block keys')

    def codes(self, keys):
        return key_codes(keys[:, 0], keys[:, 1], keys[:, 2:], self.n_atoms, self.span)

    def find(self, keys):
        """Row of each key, or -1 (keys outside the span are absent)."""
        inside = (keys[:, 2:].abs() < self.span).all(1)
        codes = self.codes(torch.where(inside[:, None], keys, torch.zeros_like(keys)))
        pos = torch.searchsorted(self.sorted, codes).clamp(max=max(len(self.sorted) - 1, 0))
        hit = inside & (self.sorted[pos] == codes) if len(self.sorted) else torch.zeros_like(inside)
        return torch.where(hit, self.order[pos], torch.full_like(pos, -1))

    def reverse(self, keys):
        """Row of (j, i, -R) for each key (i, j, R) of this table, or -1."""
        rk = torch.stack([keys[:, 1], keys[:, 0], -keys[:, 2], -keys[:, 3], -keys[:, 4]], 1)
        return self.find(rk)


@triton.jit
def key_code(i, j, r0, r1, r2, n_atoms, span):
    """``key_codes`` on the device: the code of (i, j, R) with |R| < span, monotone in (i, j, R)."""
    base = 2 * span + 1
    return (((i * n_atoms + j) * base + (r0 + span)) * base + (r1 + span)) * base + (r2 + span)


@triton.jit
def search_sorted(values, n, key, iters, mask, RIGHT: tl.constexpr):
    """``torch.searchsorted(values[:n], key, right=RIGHT)`` clamped to n - 1, elementwise over the keys (any shape; the
    keys of ``mask``); ``iters`` at least the bit length of n. The device lookup of every kernel: block rows by key
    code, radial-table knots, three-centre centres."""
    lo = key.to(tl.int64) * 0
    hi = lo + n
    for _ in range(iters):
        mid = (lo + hi) // 2
        v = tl.load(values + tl.minimum(mid, n - 1), mask=mask, other=0)
        if RIGHT:
            go = (lo < hi) & (v <= key)
        else:
            go = (lo < hi) & (v < key)
        stay = (lo < hi) & ~go
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(stay, mid, hi)
    return tl.minimum(lo, n - 1)


def unique_keys(i, j, S, n):
    """Sorted unique (i, j, S) rows as keys [m, 5]."""
    span = span_of(S)
    codes = torch.unique(key_codes(i, j, S, n, span))
    base = 2 * span + 1
    s2 = codes % base
    s1 = codes // base % base
    s0 = codes // base ** 2 % base
    ij = codes // base ** 3
    return torch.stack([ij // n, ij % n, s0 - span, s1 - span, s2 - span], 1)


class BlockKeys:
    """Blocks of a prior: the n nodes (i, i, 0), then ``edges`` (sorted unique keys without (i, i, 0)).

    keys [m, 5]; index (``KeyIndex``); reverse [m] (row of (j, i, -R)); is_node, representative (edges with
    row < reverse), canonical (nodes and representative edges) [m] bool; vec [m, 3] = x_j + R.cell - x_i.
    Closure under (i, j, R) -> (j, i, -R) is checked through ``checks`` (``steps.Checks``) when given, else at once.
    """

    def __init__(self, pos, cell, edges, checks=None):
        n = len(pos)
        dev = pos.device
        nodes = torch.zeros((n, 5), dtype=torch.int64, device=dev)
        nodes[:, 0] = nodes[:, 1] = torch.arange(n, device=dev)
        self.n = n
        self.keys = torch.cat([nodes, edges])
        self.index = KeyIndex(self.keys, n, checks=checks)
        reverse = self.index.reverse(self.keys)
        if checks is None:
            if bool((reverse < 0).any()):
                raise ValueError('block support is not closed under (i, j, R) -> (j, i, -R)')
        else:
            checks.fail_if((reverse < 0).any(), 'block support is not closed under (i, j, R) -> (j, i, -R)')
        rows = torch.arange(len(self.keys), device=dev)
        self.reverse = torch.where(reverse < 0, rows, reverse)      # reads stay in bounds until the check fails
        self.is_node = rows < n
        self.representative = (~self.is_node) & (rows < self.reverse)
        self.canonical = self.is_node | self.representative
        self.vec = pos[self.keys[:, 1]] - pos[self.keys[:, 0]] + self.keys[:, 2:].to(pos.dtype) @ cell

    def __len__(self):
        return len(self.keys)

    def herm(self, blocks):
        """(M + M_reverse^H) / 2 for every block (the transpose for real blocks)."""
        return (blocks + blocks[self.reverse].transpose(-1, -2).conj()) * .5

    def mirror(self, blocks):
        """Real blocks given on the canonical rows, completed on the others as M_ij(R) = M_ji(-R)^T; ``blocks``
        [m, a, b] is changed in place and returned. (Spinor blocks need no copy: ``finish`` reads the canonical
        rows.)"""
        rows = torch.nonzero(~self.canonical, as_tuple=True)[0]
        blocks[rows] = blocks[self.reverse[rows]].transpose(-1, -2)
        return blocks


def abacus_gauge(shells, width):
    """Permutation and sign taking ABACUS orbital order to DeePTB for one species (shell angular momenta).

    Returns (perm, sign) of length ``width``: DeePTB row r is sign[r] * ABACUS row perm[r]; padding rows map to themselves.
    """
    u = [np.asarray(ABACUS2DeePTB[l]) for l in shells]
    n = sum(len(x) for x in u)
    full = np.zeros((n, n))
    o = 0
    for x in u:
        full[o:o + len(x), o:o + len(x)] = x
        o += len(x)
    perm = np.argmax(np.abs(full), axis=1)
    sign = full[np.arange(n), perm]
    if not np.array_equal(np.abs(full[np.arange(n), perm]), np.ones(n)) or np.count_nonzero(full) != n:
        raise ValueError('ABACUS to DeePTB transform is not a signed permutation')
    return np.r_[perm, np.arange(n, width)], np.r_[sign, np.ones(width - n)]


@functools.lru_cache(maxsize=64)
def _gauge(shells, width, device):
    hidx = np.zeros((len(shells), 2 * width), dtype=np.int64)
    hsg = np.zeros((len(shells), 2 * width))
    sidx = np.zeros((len(shells), width), dtype=np.int64)
    ssg = np.zeros((len(shells), width))
    for q, sh in enumerate(shells):
        m = sum(2 * l + 1 for l in sh)
        perm, sign = abacus_gauge(list(sh), width)
        hidx[q, :2 * m] = np.r_[perm[:m], width + perm[:m]]
        hsg[q, :2 * m] = np.r_[sign[:m], sign[:m]]
        sidx[q, :m] = perm[:m]
        ssg[q, :m] = sign[:m]
    return tuple(torch.as_tensor(a, device=device) for a in (hidx, hsg, sidx, ssg))


def gauge_tables(shells, width, device):
    """ABACUS -> DeePTB gather tables of a composition (one upload per composition): ``shells`` the shell angular
    momenta of every output species. Returns hidx, hsg [species, 2w] (spin-major rows: up orbitals, then down) and
    sidx, ssg [species, w]: DeePTB row r of species q is hsg[q, r] * ABACUS row hidx[q, r]."""
    return _gauge(tuple(tuple(int(l) for l in sh) for sh in shells), int(width), torch.device(device))

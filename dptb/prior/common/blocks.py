"""Padded block tables on the device, shared by the GF and H0 priors and the inference engine.

A table holds the blocks of one quantity for the keys (i, j, Rx, Ry, Rz): ``keys`` [m, 5] int64, the valid ``rows`` and
``cols`` of every block and ``values`` [m, R, C] zero-padded to the largest block. Block conventions follow DeePTB:
spinful blocks list all up orbitals, then all down orbitals.
"""
import numpy as np
import torch

from dptb.prior.common.neighbors import key_codes, span_of
from dptb.utils.constants import ABACUS2DeePTB


class BlockTable:
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

    def __init__(self, keys, n_atoms, span=None):
        self.n_atoms = n_atoms
        self.span = span_of(keys[:, 2:]) if span is None else span
        codes = self.codes(keys)
        self.sorted, self.order = torch.sort(codes)
        if len(codes) > 1 and bool((self.sorted[1:] == self.sorted[:-1]).any()):
            raise ValueError('duplicate block keys')

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


def symmetrise(values, rev, conjugate):
    """HR2HK symmetrisation 0.5 * (v + v_rev^H) where the reverse block exists, else v."""
    has = rev >= 0
    vr = values[rev.clamp(min=0)].transpose(-1, -2)
    if conjugate:
        vr = vr.conj()
    return torch.where(has[:, None, None], 0.5 * (values + vr), values)


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

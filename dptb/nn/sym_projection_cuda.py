"""Fused CUDA backend of the crystal Reynolds projection (opt-in; the torch backend stays the reference).

One kernel gathers the rows of every symmetry image, rotates the packed AO-product blocks with the group's Wigner-D
matrices and averages, for all structures of a batch at once.  Per structure it needs the inverse row maps as int32,
the edge-reversal flags as uint8 and the concatenated D_0..D_3 of its group; these are created lazily per part and
stay resident in the projector's structure cache.  Supports angular momenta up to f and feature rows up to 1024.
"""
import os
from pathlib import Path

import numpy as np
import torch

MAX_ANGULAR = 3
MAX_WIDTH = 1024
_INFO_STRIDE = 8
_D_SIZE = 84
_D_OFFSET = (0, 1, 10, 35)
_FALSE = {'', '0', 'false', 'False', 'FALSE', 'off', 'OFF', 'no', 'No'}
_EXT = None


def _load_extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        build_dir = Path(os.environ.get('DPTB_SYM_PROJECTION_BUILD_DIR',
                                        Path.home() / '.cache' / 'dptb_sym_projection_fused'))
        build_dir.mkdir(parents=True, exist_ok=True)
        _EXT = load(name='dptb_sym_projection_fused',
                    sources=[str(Path(__file__).resolve().parent / 'symmetry_csrc' / 'sym_projection_fused.cu')],
                    extra_cuda_cflags=['-O3'], build_directory=str(build_dir), with_cuda=True,
                    verbose=os.environ.get('DPTB_SYM_PROJECTION_VERBOSE', '0') not in _FALSE)
    return _EXT


def check_layout(blocks, width):
    """Reject layouts the kernel does not handle (the torch backend accepts them)."""
    if width > MAX_WIDTH:
        raise ValueError(f"fused symmetry projection supports feature rows up to {MAX_WIDTH}, got {width}")
    if max(max(a, b) for a, b, _ in blocks) > MAX_ANGULAR:
        raise ValueError("fused symmetry projection supports angular momenta up to f")


def position_tables(blocks, transpose, width, device):
    """Per feature position: (la | lb << 2 | row << 4 | col << 7 | block start << 10) and its transposed position."""
    packed = np.zeros(width, dtype=np.int64)
    for la, lb, index in blocks:
        da, db = 2 * la + 1, 2 * lb + 1
        idx = np.asarray(index).reshape(-1, da, db)
        rows, cols, base = np.arange(da)[None, :, None], np.arange(db)[None, None, :], idx[:, :1, :1]
        if not np.array_equal(idx, base + rows * db + cols):
            raise ValueError("fused symmetry projection requires contiguous row-major shell-pair blocks")
        packed[idx.reshape(-1)] = (la | (lb << 2) | (rows << 4) | (cols << 7) | (base << 10)).reshape(-1)
    return (torch.as_tensor(packed.astype(np.int32), device=device),
            torch.as_tensor(np.asarray(transpose).astype(np.int32), device=device))


def _inverse(permutations):
    perm = np.asarray(permutations)
    n_ops, n = perm.shape
    inverse = np.empty((n_ops, n), dtype=np.int32)
    np.put_along_axis(inverse, perm, np.broadcast_to(np.arange(n, dtype=np.int32), (n_ops, n)), axis=1)
    return inverse


class FusedAction:
    """Device tables of one structure for one dtype (the fused counterpart of `_Action`)."""

    def __init__(self, group, blocks, transpose, device, dtype):
        from dptb.nn.sym_projection import _orbital_rotations

        del transpose
        self.group, self.device, self.dtype = group, device, dtype
        self.order = len(group.rotations)
        lmax = max(max(a, b) for a, b, _ in blocks)
        d = np.zeros((self.order, _D_SIZE))
        for l, matrices in _orbital_rotations(group.rotations, list(range(lmax + 1))).items():
            d[:, _D_OFFSET[l]:_D_OFFSET[l] + (2 * l + 1) ** 2] = matrices.reshape(self.order, -1).numpy()
        self.rotations = torch.as_tensor(d, device=device, dtype=dtype)
        self._tables = {}

    def tables(self, part):
        """(inverse rows int32 [G, R], reversal flags uint8 [G, R] or None): destination row r of operation g
        takes source row inverse[g, r]; the flag belongs to that source edge."""
        cached = self._tables.get(part)
        if cached is None:
            if part == 'node':
                inverse, flags = _inverse(self.group.atom_permutations), None
            else:
                inverse = _inverse(self.group.edge_permutations)
                flags = np.take_along_axis(np.asarray(self.group.edge_reversed, dtype=bool), inverse, axis=1)
                flags = torch.as_tensor(flags.astype(np.uint8), device=self.device)
            cached = (torch.as_tensor(inverse, device=self.device), flags)
            self._tables[part] = cached
        return cached


class FusedBatch:
    """All structures of a batch in one launch; the interface of `_BatchAction` (apply is self-adjoint)."""

    def __init__(self, actions, rows, part, position, n_rows):
        ext = _load_extension()
        tile = int(ext.tile_rows())
        counts = np.array([len(r) for r in rows], dtype=np.int64)
        orders = np.array([a.order for a in actions], dtype=np.int64)
        keep = [int(s) for s in np.argsort(-orders, kind='stable') if counts[s] > 0]
        info = np.zeros((max(len(keep), 1), _INFO_STRIDE), dtype=np.int64)
        row_off = tile_off = 0
        for k, s in enumerate(keep):
            inverse, flags = actions[s].tables(part)
            info[k, :7] = (inverse.data_ptr(), 0 if flags is None else flags.data_ptr(),
                           actions[s].rotations.data_ptr(), orders[s], counts[s], row_off, tile_off)
            row_off += counts[s]
            tile_off += -(-counts[s] // tile)
        if row_off != n_rows:
            raise ValueError("every feature row must belong to exactly one structure")
        device = actions[0].device
        row_list = (np.concatenate([np.asarray(rows[s]) for s in keep]).astype(np.int32) if keep
                    else np.zeros(1, dtype=np.int32))
        self.info = torch.as_tensor(info, device=device)
        self.row_list = torch.as_tensor(row_list, device=device)
        self.position, self.transpose = position
        self.n_tiles = int(tile_off)
        self.dtype = actions[0].dtype
        self._keepalive = [actions[s] for s in keep]     # the raw pointers in `info` must outlive backward

    def apply(self, x):
        if x.dtype != self.dtype or not x.is_cuda:
            raise ValueError("fused symmetry projection needs CUDA features of the dtype the actions were built for")
        if x.shape[0] == 0:
            return torch.zeros_like(x)
        return _load_extension().reynolds_apply(x.contiguous(), self.info, self.row_list, self.position,
                                                self.transpose, self.n_tiles)

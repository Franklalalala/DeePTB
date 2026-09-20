"""Native integer-image topology, without a Tonari package dependency.

Build the vendored core explicitly with tools/build_nacf_topology.py. Inference
never compiles or downloads code. This module does not change the label graph.
"""
from __future__ import annotations

import ctypes as C
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


def group_rows(keys):
    """Row indices grouped by integer key, keys ascending and rows ascending within a group.

    One stable sort replaces ``np.unique(keys, return_inverse=True)`` followed by one boolean pass per group
    (``flatnonzero(group == number)``), which costs O(groups x rows) on structures with many species pairs
    and triples. Iteration order and row order are exactly those of the replaced idiom.
    """
    keys = np.asarray(keys)
    if keys.size == 0:
        return []
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    cuts = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    return [(int(sorted_keys[s]), rows) for s, rows in zip(starts, np.split(order, cuts))]


def device_array(array, *, device, dtype):
    """Host array -> device tensor with one copy: the host copy is skipped when the transfer itself copies."""
    if getattr(device, 'type', str(device)) == 'cpu':
        array = np.array(array, copy=True)
    else:
        array = np.ascontiguousarray(array)
    import torch
    return torch.as_tensor(array, dtype=dtype, device=device)


@lru_cache(maxsize=4)
def _library(path):
    lib = C.CDLL(path)
    if lib.nacf_topology_abi() != 1:
        raise RuntimeError('unsupported NACF topology ABI')
    ptr, ip, dp, bp = C.c_void_p, C.POINTER(C.c_int64), C.POINTER(C.c_double), C.POINTER(C.c_uint8)
    lib.nacf_topology_build.argtypes = [C.c_int64, dp, dp, bp, dp, dp, C.c_int64, ip, ip, C.c_int64, C.c_char_p]
    lib.nacf_topology_build.restype = ptr
    lib.nacf_topology_count.argtypes = [ptr, C.c_int]
    lib.nacf_topology_count.restype = C.c_int64
    lib.nacf_topology_data.argtypes = [ptr, C.c_int]
    lib.nacf_topology_data.restype = ip
    lib.nacf_topology_seconds.argtypes = [ptr, C.c_int]
    lib.nacf_topology_seconds.restype = C.c_double
    lib.nacf_topology_free.argtypes = [ptr]
    return lib


def build_edge_topology(positions, cell, pbc, ao_cutoffs, centre_cutoffs,
                        edge_index, edge_cell_shift, *, library=None, max_terms=10_000_000,
                        mode='edge_vna'):
    """Return unique factor queries and half-edge third-centre terms.

    Query rows are (AO, centre, sx, sy, sz), so the queried displacement is
    pos[AO] - pos[centre] + shift @ cell. Zero-shift endpoint centres are
    excluded; periodic self images are retained. Cutoffs use Bohr.
    ``projector`` includes endpoints and returns onsite followed by all directed
    edge block rows. ``onsite_vna`` returns only onsite blocks. Projector centre
    cutoff -1 disables a species with no projectors.
    ``density`` returns (AO, neighbour, NEIGHBOUR image) queries, excludes
    (i,0), and groups queries by directed edge in terms[:,0:2], excluding
    its (j,R) endpoint. It accepts missing reverse edges and zero supports.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    lattice = np.ascontiguousarray(cell, dtype=np.float64)
    periodic_raw = np.asarray(pbc)
    ac = np.ascontiguousarray(ao_cutoffs, dtype=np.float64)
    cc = np.ascontiguousarray(centre_cutoffs, dtype=np.float64)
    n = len(pos)
    if pos.shape != (n, 3) or n == 0 or lattice.shape != (3, 3):
        raise ValueError('invalid geometry shape')
    if periodic_raw.shape != (3,) or not np.isin(periodic_raw, [0, 1]).all():
        raise ValueError('pbc must contain three booleans')
    periodic = np.ascontiguousarray(periodic_raw, dtype=np.uint8)
    modes = {'edge_vna': 0, 'projector': 1, 'onsite_vna': 2, 'density': 3}
    if mode not in modes:
        raise ValueError('unknown topology mode')
    valid_cc = ((cc >= 0) | (cc == -1) if mode == 'projector' else
                cc >= 0 if mode == 'density' else cc > 0)
    if ac.shape != (n,) or cc.shape != (n,) or (ac <= 0).any() or not valid_cc.all():
        raise ValueError('cutoffs must be positive per-atom arrays')
    if not all(np.isfinite(x).all() for x in (pos, lattice, ac, cc)):
        raise ValueError('geometry and cutoffs must be finite')
    raw_edges, raw_shifts = np.asarray(edge_index), np.asarray(edge_cell_shift)
    if raw_edges.ndim != 2 or raw_edges.shape[0] != 2 or raw_shifts.shape != (raw_edges.shape[1], 3):
        raise ValueError('graph requires [2,E] indices and [E,3] shifts')
    for x in (raw_edges, raw_shifts):
        if not np.isfinite(x).all() or (np.abs(x) > 2**31-1).any() or not np.equal(x, np.floor(x)).all():
            raise ValueError('graph indices and shifts must be exact supported integers')
    edges = np.ascontiguousarray(raw_edges.T, dtype=np.int64)
    shifts = np.ascontiguousarray(raw_shifts, dtype=np.int64)
    if not isinstance(max_terms, (int, np.integer)) or not 0 < max_terms <= 2**63-1:
        raise ValueError('max_terms must be a positive integer')
    path = library or os.environ.get('DPTB_NACF_TOPOLOGY_LIBRARY')
    if not path:
        raise RuntimeError('Build tools/build_nacf_topology.py and set DPTB_NACF_TOPOLOGY_LIBRARY')
    path = str(Path(path).resolve(strict=True))
    lib = _library(path)
    dp, ip, bp = C.POINTER(C.c_double), C.POINTER(C.c_int64), C.POINTER(C.c_uint8)
    error = C.create_string_buffer(1024)
    args = (n, pos.ctypes.data_as(dp), lattice.ctypes.data_as(dp),
                                    periodic.ctypes.data_as(bp), ac.ctypes.data_as(dp), cc.ctypes.data_as(dp),
                                    len(edges), edges.ctypes.data_as(ip), shifts.ctypes.data_as(ip), max_terms, error)
    if mode == 'edge_vna':
        handle = lib.nacf_topology_build(*args)
    else:
        if not hasattr(lib, 'nacf_topology_build_mode'):
            raise RuntimeError('rebuild the native library for assembly topology support')
        function = lib.nacf_topology_build_mode
        function.argtypes = lib.nacf_topology_build.argtypes + [C.c_int]
        function.restype = C.c_void_p
        handle = function(*args, modes[mode])
    if not handle:
        raise ValueError(error.value.decode('utf-8', errors='replace'))
    try:
        arrays = []
        for which, width in enumerate((5, 3, 1)):
            count = lib.nacf_topology_count(handle, which)
            array = (np.ctypeslib.as_array(lib.nacf_topology_data(handle, which), (count*width,)).copy()
                     if count else np.empty(0, dtype=np.int64))
            arrays.append(array.reshape(count, width) if width > 1 else array)
        return {'queries': arrays[0], 'terms': arrays[1], 'reverse': arrays[2],
                'broad_pairs': lib.nacf_topology_count(handle, 3),
                'search_s': lib.nacf_topology_seconds(handle, 0),
                'join_s': lib.nacf_topology_seconds(handle, 1)}
    finally:
        lib.nacf_topology_free(handle)

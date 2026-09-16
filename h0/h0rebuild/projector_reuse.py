"""Reuse the two-centre factors of the separable nonlocal operator.

This is an opt-in, per-assembly adapter over a frozen CUDATwoCenter. It changes
neither radial tables nor the native candidate reduction. Keys contain exact
float64 displacements, never rounded coordinates. The cache bound covers retained
Q factors only, not the existing per-block gather/contraction working set.
"""
from collections import OrderedDict
import math
import threading
import time

import numpy as np
import torch


def factor_key(projector_species, orbital_species, displacement):
    value = np.ascontiguousarray(displacement, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Projector displacement must be a finite three-vector')
    return (int(projector_species), int(orbital_species), value.tobytes())


class ProjectorReuseTwoCenter:
    """Private factors shared across edges within one immutable table session."""

    def __init__(self, base, max_mb=256., batch_edges=1024):
        if not math.isfinite(max_mb) or max_mb <= 0:
            raise ValueError('projector reuse budget must be finite and positive')
        if isinstance(batch_edges, bool) or not isinstance(batch_edges, int) or batch_edges <= 0:
            raise ValueError('batch_edges must be a positive integer')
        self._base = base
        self._limit = int(max_mb * 1024**2)
        self._batch_edges = batch_edges
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self.stats = dict(requested_factors=0, evaluated_factors=0, native_q_calls=0,
                          resident_hits=0, within_block_reuses=0, evictions=0,
                          resident_bytes=0, peak_resident_bytes=0, budget_bytes=self._limit)
        self.metadata = dict(base.metadata, projector_reuse=self.stats,
                             projector_reuse_key='species IDs + exact float64 displacement bytes',
                             projector_reuse_scope='private per-assembly factors; unchanged native reduction')
        self.metadata['nonlocal_timing_scope'] = 'reuse wrapper wall time including filtering, Q gather, contraction and CPU result; excludes lock wait'

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _evaluate(self, requests):
        from .cuda_two_center import _C
        b = self._base
        disp = torch.tensor(np.asarray([r[2] for r in requests]), dtype=torch.float64, device=b.device)
        ps = torch.tensor([r[0] for r in requests], dtype=torch.int32, device=b.device)
        os = torch.tensor([r[1] for r in requests], dtype=torch.int32, device=b.device)
        self.stats['native_q_calls'] += 1
        self.stats['evaluated_factors'] += len(requests)
        return _C.eval_projector_overlap_batch(
            disp, ps, os, b.Q_coeffs, b.Q_index_map, b.Q_index_map_strides,
            b.gaunt_table, b.gaunt_dims, b.proj_l, b.proj_zeta, b.proj_m,
            b.species_proj_offsets, b.orb_l, b.orb_zeta, b.orb_m,
            b.species_orb_offsets, b.dr, b.cutoff, b.nr, b.max_nproj, b.max_norb)

    def _factors(self, requests):
        keys = [factor_key(*r) for r in requests]
        self.stats['requested_factors'] += len(keys)
        found, missing = {}, OrderedDict()
        stream = torch.cuda.current_stream(self.device)
        for key, req in zip(keys, requests):
            if key in found or key in missing:
                self.stats['within_block_reuses'] += 1
            elif key in self._cache:
                tensor, ready = self._cache[key]
                stream.wait_event(ready)
                tensor.record_stream(stream)
                found[key] = tensor
                self._cache.move_to_end(key)
                self.stats['resident_hits'] += 1
            else:
                missing[key] = req
        items = list(missing.items())
        for start in range(0, len(items), self._batch_edges):
            chunk = items[start:start + self._batch_edges]
            values = self._evaluate([r for _, r in chunk])
            for pos, (key, _) in enumerate(chunk):
                # A clone prevents one cached row retaining an entire batch allocation.
                value = values[pos].clone()
                found[key] = value
                size = value.numel() * value.element_size()
                if size <= self._limit:
                    while self.stats['resident_bytes'] + size > self._limit:
                        _, (old, _) = self._cache.popitem(last=False)
                        self.stats['resident_bytes'] -= old.numel() * old.element_size()
                        self.stats['evictions'] += 1
                    ready = torch.cuda.Event()
                    ready.record(stream)
                    self._cache[key] = (value, ready)
                    self.stats['resident_bytes'] += size
                    self.stats['peak_resident_bytes'] = max(self.stats['peak_resident_bytes'], self.stats['resident_bytes'])
        return torch.stack([found[k] for k in keys])

    def nonlocal_block(self, structure, symbol_i, symbol_j, ci, cj, candidates):
        # Serializes cache publication across callers; GPU event/record_stream
        # protect reuse and eviction on different consumer streams.
        with self._lock, torch.cuda.device(self.device):
            start = time.perf_counter()
            result = self._nonlocal_block(structure, symbol_i, symbol_j, ci, cj, candidates)
            self._base.timing['kernel_nonlocal_seconds'] += time.perf_counter() - start
            self._base.timing['kernel_nonlocal_calls'] += 1
            return result

    def _nonlocal_block(self, structure, symbol_i, symbol_j, ci, cj, candidates):
        from .cuda_two_center import _C
        b = self._base
        ni, nj = b.norb_per_species[symbol_i], b.norb_per_species[symbol_j]
        ci, cj = np.asarray(ci, dtype=np.float64), np.asarray(cj, dtype=np.float64)
        if ci.shape != (3,) or cj.shape != (3,) or not np.isfinite([ci, cj]).all():
            raise ValueError('AO centres must be finite three-vectors')
        from .projector_candidates import PreparedProjectorCandidates
        if isinstance(candidates,PreparedProjectorCandidates):
            if (candidates.owner is not b or candidates.structure is not structure or
                candidates.center.tobytes()!=ci.tobytes() or candidates.rcut_i!=b.sd[symbol_i].orb.rcut):
                raise ValueError('Prepared projector candidates belong to a different anchor/session')
            active=candidates.select(cj,b.sd[symbol_j].orb.rcut)
        else:
            active = []
            for atom_index, pcenter in candidates:
                p = np.asarray(pcenter, dtype=np.float64)
                if p.shape != (3,) or not np.isfinite(p).all():
                    raise ValueError('Projector centres must be finite three-vectors')
                symbol = structure.atoms[atom_index].species
                cutoff = b.sd[symbol].upf.max_projector_cutoff
                if np.linalg.norm(ci-p) > cutoff + b.sd[symbol_i].orb.rcut:
                    continue
                if np.linalg.norm(cj-p) > cutoff + b.sd[symbol_j].orb.rcut:
                    continue
                active.append((b._check_species(symbol), p))
        if not active:
            mult = 2 if b.nspin == 4 else 1
            return np.zeros((mult*ni, mult*nj), dtype=np.complex128 if b.nspin == 4 else np.float64)
        si, sj = b._check_species(symbol_i), b._check_species(symbol_j)
        requests = [(s, si, ci-p) for s,p in active] + [(s, sj, cj-p) for s,p in active]
        q = self._factors(requests)
        n = len(active)
        qi, qj = q[:n, :, :ni].contiguous(), q[n:, :, :nj].contiguous()
        sp = torch.tensor([s for s,p in active], dtype=torch.int32, device=b.device)
        npj = torch.tensor([b.nproj_per_species[b.species[s]] for s,p in active], dtype=torch.int32, device=b.device)
        mask = torch.ones(n, dtype=torch.bool, device=b.device)
        result = _C.assemble_nonlocal_candidates(qi, qj, b.D_padded_cuda[sp], mask, npj, ni, nj, b.nspin)
        return result.cpu().numpy()

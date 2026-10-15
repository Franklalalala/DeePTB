"""GF prior (Flash route): geometry -> prior blocks on the device.

    support   direct AO pairs, model graph edges and pairs sharing a projector centre (topology.support)
    scalar    node = Ry herm(onsite_base + proj) + sym(V_node) + onsite_xc + c S_node
              edge = Ry herm(base + proj) + edge_vna + pair_xc + c S_edge + envxc
              c = 4 pi sum_a M2_a Ry / (3 Omega)
    spinor    H = [[C + i Im X_uu, X_ud], [X_du, C + i Im X_dd]] with X = Ry herm(sum_k P^T D_spinor P), C the scalar block
    gauge     ABACUS -> DeePTB (per-species signed permutation), spin blocks contiguous (up orbitals, then down)

The onsite XC (FP64-bound density kernel) runs in a worker thread on a low-priority stream while the assembly and
environment-XC kernels (latency- and memory-bound) run on a high-priority stream, so the two overlap on the device.

Returns ``PriorBlocks``: keys [m, 5] (nodes, then edges), H complex128 [m, 2w, 2w] (eV), S float64 [m, w, w],
per-atom orbital counts, species sorted by atomic number with each atom's index, valence electron count.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from ase.data import atomic_numbers as Z_OF, chemical_symbols

from dptb.prior.common.blocks import abacus_gauge
from dptb.prior.gf import assembly as A
from dptb.prior.gf.envxc import envxc
from dptb.prior.gf.onsite import OnsiteXC
from dptb.prior.gf.tables import GFTables
from dptb.prior.gf.topology import BOHR, Topology, support


class PriorBlocks:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _P2Species:
    def __init__(self, tables):
        self.tables = tables

    def __getitem__(self, s):
        return self.tables._find('p2', 'species', s)[1]


class GFPrior:
    """GF prior (Flash route). extension / build_missing (off by default): see ``tables.GFTables``."""

    def __init__(self, root, *, soc, pairxc, moments, pporb, p23_extensions=(), device='cuda', table_bytes=4 << 30,
                 extension=None, build_missing=False, species_files=None):
        self.device = torch.device(device)
        self.tables = GFTables(root, soc=soc, pairxc=pairxc, moments=moments, p23_extensions=p23_extensions,
                               device=self.device, budget_bytes=table_bytes, extension=extension,
                               build_missing=build_missing, pporb=[pporb], species_files=species_files)
        self.onsite = OnsiteXC(pporb, _P2Species(self.tables), device=self.device)
        self.concurrent = self.device.type == 'cuda'
        if self.concurrent:
            lo, hi = torch.cuda.Stream.priority_range()
            self._side = torch.cuda.Stream(self.device, priority=lo)
            self._main = torch.cuda.Stream(self.device, priority=hi)
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gf-onsite')

    def prefetch(self, numbers):
        """Read the tables of a composition in the background (call for the next structure)."""
        self.tables.prefetch([chemical_symbols[int(z)] for z in numbers])

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.current_stream(self.device).synchronize()

    def _onsite_job(self, symbols, pos, cell, w):
        t = time.perf_counter()
        with torch.cuda.stream(self._side):
            site, orders = self.onsite(symbols, pos, cell, w)
            done = torch.cuda.Event()
            done.record(self._side)
        return site, orders, done, time.perf_counter() - t

    def __call__(self, z, pos, cell, ei, es):
        """z [n], pos [n, 3], cell [3, 3] (Angstrom, device tensors); model edges ei [2, E], es [E, 3]."""
        tm = {}
        t0 = time.perf_counter()
        symbols = [chemical_symbols[int(x)] for x in z.tolist()]
        self.tables.load(symbols)
        self._sync(); tm['tables_s'] = time.perf_counter() - t0
        pos = pos.to(torch.float64) / BOHR
        cell = cell.to(torch.float64) / BOHR
        sp = {s: self.tables.species(s) for s in set(symbols)}
        if not self.concurrent:
            return self._blocks(symbols, sp, pos, cell, ei, es, tm, t0, None)
        caller = torch.cuda.current_stream(self.device)
        self._side.wait_stream(caller)
        self._main.wait_stream(caller)
        job = self._pool.submit(self._onsite_job, symbols, pos, cell, max(sp[s].norb for s in sp))
        with torch.cuda.stream(self._main):
            out = self._blocks(symbols, sp, pos, cell, ei, es, tm, t0, job)
        caller.wait_stream(self._main)
        for x in (out.keys, out.H, out.S, out.norb, out.spec):
            x.record_stream(caller)
        return out

    def _blocks(self, symbols, sp, pos, cell, ei, es, tm, t0, job):
        t = time.perf_counter()
        rc = torch.as_tensor([sp[s].rc for s in symbols], dtype=torch.float64, device=self.device)
        beta = torch.as_tensor([sp[s].proj_max for s in symbols], dtype=torch.float64, device=self.device)
        topo = Topology(pos, cell, support(pos, cell, rc, beta, (ei[0], ei[1], es)))
        ctx = A.Context(self.tables, symbols, pos, cell, topo)
        n, w = ctx.n, ctx.w
        self._sync(); tm['topology_s'] = time.perf_counter() - t
        t = time.perf_counter()
        base, ovl = A.two_centre(ctx)
        real, spin = A.projector(ctx)
        p2 = A.herm(base + real, topo) * A.RY
        S = A.herm(ovl, topo)
        del base, ovl, real
        vnode, vedge = A.vna(ctx)
        px = A.pair_xc(ctx)
        self._sync(); tm['assembly_s'] = time.perf_counter() - t
        t = time.perf_counter()
        env = envxc(ctx, S[n:])
        self._sync(); tm['envxc_s'] = time.perf_counter() - t
        t = time.perf_counter()
        if job is None:
            site, orders = self.onsite(symbols, pos, cell, w)
            self._sync(); tm['onsite_s'] = time.perf_counter() - t
        else:
            site, orders, done, tm['onsite_s'] = job.result()
            torch.cuda.current_stream(self.device).wait_event(done)
            site.record_stream(torch.cuda.current_stream(self.device))
            tm['onsite_wait_s'] = time.perf_counter() - t
        t = time.perf_counter()
        c = 4.0 * math.pi * sum(sp[s].m2 for s in symbols) * A.RY / (3.0 * abs(float(torch.linalg.det(cell))))
        C = p2
        C[:n] += vnode + site + c * S[:n]
        C[n:] += vedge + px + c * S[n:] + env
        del vnode, vedge, px, env, site
        X = A.herm(spin.reshape(-1, 2, 2, w, w).transpose(2, 3).reshape(-1, 2 * w, 2 * w), topo) * A.RY
        del spin
        X[:, :w, :w] = C + 1j * X[:, :w, :w].imag
        X[:, w:, w:] = C + 1j * X[:, w:, w:].imag
        del C
        # ABACUS -> DeePTB gauge, contiguous spin blocks
        names = sorted(set(symbols), key=lambda x: Z_OF[x])
        spec = torch.as_tensor([names.index(s) for s in symbols], device=self.device)
        hidx = torch.zeros((len(names), 2 * w), dtype=torch.int64, device=self.device)
        hsg = torch.zeros((len(names), 2 * w), dtype=torch.float64, device=self.device)
        sidx = torch.zeros((len(names), w), dtype=torch.int64, device=self.device)
        ssg = torch.zeros((len(names), w), dtype=torch.float64, device=self.device)
        for q, s in enumerate(names):
            m = sp[s].norb
            perm, sign = abacus_gauge(sp[s].shells, w)
            hidx[q, :2 * m] = torch.as_tensor(np.r_[perm[:m], w + perm[:m]])
            hsg[q, :2 * m] = torch.as_tensor(np.r_[sign[:m], sign[:m]])
            sidx[q, :m] = torch.as_tensor(perm[:m])
            ssg[q, :m] = torch.as_tensor(sign[:m])
        si, sj = spec[topo.keys[:, 0]], spec[topo.keys[:, 1]]
        nb = len(topo.keys)
        H = X.gather(1, hidx[si][:, :, None].expand(nb, 2 * w, 2 * w))
        H = H.gather(2, hidx[sj][:, None, :].expand(nb, 2 * w, 2 * w)) * (hsg[si][:, :, None] * hsg[sj][:, None, :])
        del X
        S = S.gather(1, sidx[si][:, :, None].expand(nb, w, w))
        S = S.gather(2, sidx[sj][:, None, :].expand(nb, w, w)) * (ssg[si][:, :, None] * ssg[sj][:, None, :])
        self._sync(); tm['finish_s'] = time.perf_counter() - t
        tm['total_s'] = time.perf_counter() - t0
        nelec = float(sum(sp[s].z_valence for s in symbols))
        info = dict(n_blocks=nb, width=w, c_ev=c, orders=[list(o) for o in orders],
                    fine_atoms=sum(o != (128, 24, 48) for o in orders))
        return PriorBlocks(keys=topo.keys, H=H, S=S, norb=ctx.norb, species=names, spec=spec, nelec=nelec,
                           timing=tm, info=info)

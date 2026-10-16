"""GF prior (Flash route): geometry -> prior blocks on the device.

    support   direct AO pairs, model graph edges and pairs sharing a projector centre (topology.support)
    scalar    C = herm(C'), node C' = Ry herm(onsite_base + proj) + sym(V_node) + onsite_xc + c S_node
                                edge C' = Ry herm(base + proj) + edge_vna + pair_xc + c S_edge + envxc
              c = 4 pi sum_a M2_a Ry / (3 Omega)
    spinor    H = [[C + i Im X_uu, X_ud], [X_du, C + i Im X_dd]] with X = Ry herm(sum_k P^T D_spinor P), C the scalar block
    gauge     ABACUS -> DeePTB (per-species signed permutation), spin blocks contiguous (up orbitals, then down)

Steps (``steps.Steps`` names; shared steps use the operators of ``dptb.prior.common`` as the H0 prior does):
    tables, neighbours (one pair search, ``blocks.BlockKeys``), two_centre, nonlocal (``threecentre.projector``),
    vna, pair_xc, envxc, onsite (``onsite_wait``: time the main stream waited for it), finish (``finish.finish``).
The onsite XC (density kernel) runs in a worker thread on a low-priority stream while the other steps (latency- and
memory-bound) run on a high-priority stream, so the two overlap on the device.

Returns ``PriorBlocks``: keys [m, 5] (nodes, then edges), H complex128 [m, 2w, 2w] (eV), S float64 [m, w, w],
per-atom orbital counts, species sorted by atomic number with each atom's index, valence electron count.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from ase.data import atomic_numbers as Z_OF, chemical_symbols

from dptb.prior.common.blocks import BlockKeys, gauge_tables
from dptb.prior.common.finish import finish
from dptb.prior.common.neighbors import PairBase
from dptb.prior.common.steps import Checks, Steps
from dptb.prior.gf import assembly as A
from dptb.prior.gf.envxc import envxc
from dptb.prior.gf.onsite import OnsiteXC
from dptb.prior.gf.tables import GFTables
from dptb.prior.gf.topology import BOHR, EPS, support


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
        self._kb = {}

    def prefetch(self, numbers):
        """Read the tables of a composition in the background (call for the next structure)."""
        self.tables.prefetch([chemical_symbols[int(z)] for z in numbers])

    def _onsite_job(self, symbols, pos, cell, w):
        t = time.perf_counter()
        with torch.cuda.stream(self._side):
            site, orders = self.onsite(symbols, pos, cell, w)
            done = torch.cuda.Event()
            done.record(self._side)
        return site, orders, done, time.perf_counter() - t

    def __call__(self, z, pos, cell, ei, es):
        """z [n], pos [n, 3], cell [3, 3] (Angstrom, device tensors); model edges ei [2, E], es [E, 3]."""
        steps = Steps(self.device)
        symbols = [chemical_symbols[int(x)] for x in z.tolist()]
        self.tables.load(symbols)
        steps('tables')
        pos = pos.to(torch.float64) / BOHR
        cell = cell.to(torch.float64) / BOHR
        sp = {s: self.tables.species(s) for s in set(symbols)}
        if not self.concurrent:
            return self._blocks(symbols, sp, pos, cell, ei, es, steps, None)
        caller = torch.cuda.current_stream(self.device)
        self._side.wait_stream(caller)
        self._main.wait_stream(caller)
        job = self._pool.submit(self._onsite_job, symbols, pos, cell, max(sp[s].norb for s in sp))
        with torch.cuda.stream(self._main):
            out = self._blocks(symbols, sp, pos, cell, ei, es, steps, job)
        caller.wait_stream(self._main)
        for x in (out.keys, out.H, out.S, out.norb, out.spec):
            x.record_stream(caller)
        return out

    def _blocks(self, symbols, sp, pos, cell, ei, es, steps, job):
        checks = Checks()
        rc = torch.as_tensor([sp[s].rc for s in symbols], dtype=torch.float64, device=self.device)
        beta = torch.as_tensor([sp[s].proj_max for s in symbols], dtype=torch.float64, device=self.device)
        # one pair search for the support and the projector, VNA and density-centre queries (margins <= 1e-9)
        cut = torch.as_tensor([max(sp[s].vna_cut, sp[s].q_cut) for s in symbols], dtype=torch.float64,
                              device=self.device)
        base = PairBase(pos, cell, torch.maximum(torch.maximum(rc + EPS, beta), cut), rc + EPS, 1e-9)
        bk = BlockKeys(pos, cell, support(pos, cell, rc, beta, (ei[0], ei[1], es), base=base), checks)
        ctx = A.Context(self.tables, symbols, pos, cell, bk, self._kb)
        ctx.base = base
        n, w = ctx.n, ctx.w
        steps('neighbours')
        base, ovl = A.two_centre(ctx)
        S = bk.herm(ovl)
        del ovl
        steps('two_centre')
        real, spin, nstats = A.projector(ctx)
        p2 = bk.herm(base + real) * A.RY
        del base, real
        steps('nonlocal')
        v, vstats = A.vna(ctx)
        steps('vna')
        px = A.pair_xc(ctx)
        steps('pair_xc')
        env, estats = envxc(ctx, S)
        steps('envxc')
        if job is None:
            site, orders = self.onsite(symbols, pos, cell, w)
            steps('onsite')
        else:
            site, orders, done, onsite_s = job.result()
            torch.cuda.current_stream(self.device).wait_event(done)
            site.record_stream(torch.cuda.current_stream(self.device))
            steps('onsite_wait')
            steps.add('onsite', onsite_s)
        c = 4.0 * math.pi * sum(sp[s].m2 for s in symbols) * A.RY / (3.0 * abs(float(torch.linalg.det(cell))))
        C = p2
        C[:n] += v[:n] + site + c * S[:n]
        C[n:] += v[n:] + px[n:] + c * S[n:] + env[n:]
        del v, px, env, site
        C = bk.herm(C)              # pair-XC (both directions) and onsite XC are not exact Hermitian pairs
        names = sorted(set(symbols), key=lambda x: Z_OF[x])
        spec = torch.as_tensor([names.index(s) for s in symbols], device=self.device)
        H, S = finish(bk, spec, gauge_tables([sp[s].shells for s in names], w, self.device), C, spin, S, A.RY, checks)
        del spin, C
        stats = checks.read()
        steps('finish')
        nelec = float(sum(sp[s].z_valence for s in symbols))
        info = dict(n_blocks=len(bk), width=w, c_ev=c, nonlocal_=nstats, vna=vstats, envxc=estats,
                    orders=[list(o) for o in orders],
                    fine_atoms=sum(o != (128, 24, 48) for o in orders), onsite=dict(self.onsite.stats), **stats)
        return PriorBlocks(keys=bk.keys, blocks=bk, H=H, S=S, norb=ctx.norb, species=names, spec=spec, nelec=nelec,
                           timing=steps.done(), info=info)

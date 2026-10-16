"""H0 prior (UniTB route): geometry -> prior blocks on the device.

    support   pairs (i, j, R) with |x_j + R.cell - x_i| <= r_c(i) + r_c(j) + 2 max r_beta (+1e-12), r_c the orbital and
              r_beta the projector cutoffs of the composition (bohr); blocks laid out by ``blocks.BlockKeys``
    S, T      two-centre overlap and kinetic tables of the pair bank (``common.twocentre``)
    V_loc     FFT-grid tile integration of the local potential (``field``, ``local``), ecutrho 480 Ry, PBE
    V_NL      the shared nonlocal step (``common.threecentre.projector``) on the bank's projector overlaps
              (``projector``): D_eff part (real) and spin-orbit channels
    H         C = Ry herm(T + V_loc + V_NL,eff) is the real part of the uu and dd blocks, the spin-orbit channels give
              the rest (``common.finish``); S = herm(S); herm(X) = (X + X_reverse^H) / 2
    gauge     ABACUS -> DeePTB (per-species signed permutation), contiguous spin blocks (up orbitals, then down); eV

Steps (``steps.Steps`` names; shared steps use the operators of ``dptb.prior.common`` as the GF prior does): tables,
grid, field, neighbours, two_centre, local, nonlocal, finish.

Returns ``PriorBlocks``: keys [m, 5] (nodes, then edges), H complex128 [m, 2w, 2w] (eV), S float64 [m, w, w], per-atom
orbital counts, species sorted by atomic number with each atom's index, valence electron count, step timings, info.
"""
import threading
from pathlib import Path

import torch
from ase.data import atomic_numbers as Z_OF, chemical_symbols

from dptb.prior.common.blocks import BlockKeys, gauge_tables
from dptb.prior.common.finish import finish
from dptb.prior.common.neighbors import PairBase
from dptb.prior.common.steps import Checks, Steps
from dptb.prior.common.units import BOHR, RY
from dptb.prior.h0.bank import PairBank
from dptb.prior.h0.field import RadialSpecies, build_field, fft_grid
from dptb.prior.h0.local import AOTables, local_blocks
from dptb.prior.h0.projector import kb_matrices, nonlocal_blocks

ECUTRHO = 480.0


class PriorBlocks:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class H0Prior:
    """H0 prior on a pair bank (``bank``: directory of the schema h0-physical-pair-bank/v1, 0.02 bohr grid).

    extension: extension tables root; its ``h0`` bank is stacked after ``bank``. build_missing (off by default): pairs
    neither bank holds are built into it on the GPU (``dptb.prior.build``) before use; ORB/UPF of species the
    deployed bank lacks come from ``species_files`` {symbol: (orb, upf)} or the ``pporb`` directories.
    """

    def __init__(self, bank, *, device='cuda', host_bytes=16 << 30, phi_bytes=4 << 30, extension=None,
                 build_missing=False, species_files=None, pporb=()):
        self.device = torch.device(device)
        self.extension = Path(extension) if extension else None
        self.build_missing, self.species_files, self.pporb = build_missing, dict(species_files or {}), list(pporb)
        self.bank = PairBank(bank, device=self.device, host_bytes=host_bytes,
                             extensions=[self.extension / 'h0'] if self.extension else ())
        self.phi_bytes = phi_bytes
        self._species, self._radial, self._ao, self._kb = {}, {}, {}, {}
        self._lock = threading.Lock()
        self._prefetch = None

    # ---------------------------------------------------------------------------------------------- species and tables
    def species(self, s):
        with self._lock:
            if s not in self._species:
                self._species[s] = self.bank.species(s)
            return self._species[s]

    def _load(self, names):
        missing = sorted({k for a in names for b in names for k in ['|'.join(sorted({a, b}))] if not self.bank.has(k)})
        if missing and self.build_missing and self.extension is not None:
            self._build(missing)
            missing = [k for k in missing if not self.bank.has(k)]
        if missing:
            raise FileNotFoundError(f'H0 pair bank lacks the pairs {missing}')
        sd = {s: self.species(s) for s in names}
        with self._lock:
            comp = self.bank.composition(names)
        return sd, comp

    def _build(self, keys):
        from dptb.prior.build import build_missing
        files = dict(self.species_files)
        new = [s for s in sorted({s for k in keys for s in k.split('|')})
               if s not in files and not any(s in m.get('species', {}) for _, m, _ in self.bank.layers)]
        if new and self.pporb:
            from dptb.prior.gf.build_p23 import species_files as resolve
            files.update(resolve(new, self.pporb))
        with self._lock:
            build_missing('h0', keys, self.extension, reference=self.bank.root, species_files=files or None,
                          device=self.device, log=lambda *_: None)
            self.bank.add_layer(self.extension / 'h0')

    def prefetch(self, numbers):
        """Read the species and pair records of a composition in a background thread (call for the next structure)."""
        names = sorted({chemical_symbols[int(z)] for z in numbers})

        def work():
            try:
                for s in names:
                    self.species(s)
                for i, a in enumerate(names):
                    for b in names[i:]:
                        with self._lock:
                            self.bank.record(a, b)
            except Exception:  # noqa: BLE001  (the foreground load reports the error)
                pass
        self._prefetch = threading.Thread(target=work, daemon=True)
        self._prefetch.start()

    def _radial_species(self, s, sd):
        if s not in self._radial:
            self._radial[s] = RadialSpecies(sd[s].upf, self.device)
        return self._radial[s]

    def _ao_tables(self, names, sd):
        key = tuple(names)
        if key not in self._ao:
            self._ao[key] = AOTables([sd[s].orb for s in names], self.device)
            while len(self._ao) > 8:
                self._ao.pop(next(iter(self._ao)))
        return self._ao[key]

    def _kb_matrices(self, names, comp):
        key = tuple(names)
        if key not in self._kb:
            self._kb[key] = kb_matrices(comp, names, max(comp.nproj))
            while len(self._kb) > 8:
                self._kb.pop(next(iter(self._kb)))
        return self._kb[key]

    # ------------------------------------------------------------------------------------------------------- prior
    def __call__(self, z, pos, cell, ei=None, es=None):
        """z [n], pos [n, 3], cell [3, 3] (Angstrom, device tensors); the model edges ei, es are not needed."""
        steps, checks = Steps(self.device), Checks()
        dev = self.device
        symbols = [chemical_symbols[int(x)] for x in z.tolist()]
        names = sorted(set(symbols))                         # table order (alphabetical), as the bank composition
        if self._prefetch is not None:
            self._prefetch.join()
            self._prefetch = None
        sd, comp = self._load(names)
        steps('tables')
        cell_a = cell.to(dev, torch.float64)
        frac = pos.to(dev, torch.float64) @ torch.linalg.inv(cell_a)
        cb = cell_a * (1.0 / BOHR)
        P = frac @ cb                                        # bohr, positions as given
        cell_np = cb.cpu().numpy()
        shape = fft_grid(cell_np, ECUTRHO, device=dev)
        steps('grid')
        radial = {s: self._radial_species(s, sd) for s in names}
        field = build_field(radial, symbols, frac, cell_np, shape, ecutrho=ECUTRHO)
        steps('field')
        tab = self._ao_tables(names, sd)
        sp = torch.as_tensor([names.index(s) for s in symbols], device=dev)
        rcut = tab.rcut[sp]
        pcut = torch.as_tensor([sd[s].upf.max_projector_cutoff for s in symbols], dtype=torch.float64, device=dev)
        margin = 2 * max(sd[s].upf.max_projector_cutoff for s in symbols) + 1e-12
        # one pair search for the support and the projector queries
        base = PairBase(P, cb, rcut, rcut, margin)
        i, j, S = base.select(rcut, margin=margin, self_interaction=True)
        edge = (i != j) | (S != 0).any(1)
        bk = BlockKeys(P, cb, torch.cat([i[:, None], j[:, None], S], 1)[edge], checks)
        del i, j, S, edge
        steps('neighbours')
        w = tab.wmax
        ki, kj = bk.keys[:, 0], bk.keys[:, 1]
        disp = comp.S.displacements(sp[ki], sp[kj], bk.vec)
        S2, T2 = comp.S(disp, w, w), comp.T(disp, w, w)
        del disp
        steps('two_centre')
        V, lstats = local_blocks(field, tab, sp, P, frac, cb, bk, checks, phi_bytes=self.phi_bytes)
        steps('local')
        real, spin, nstats = nonlocal_blocks(bk, comp, self._kb_matrices(names, comp), sp, P, cb, pcut, rcut, base, w)
        steps('nonlocal')
        C = bk.herm(T2 + V + real) * RY
        del T2, V, real
        species = sorted(names, key=lambda x: Z_OF[x])
        spec = torch.as_tensor([species.index(s) for s in symbols], device=dev)
        shells = [[c.l for c in sd[s].orb.channels] for s in species]
        H, S2 = finish(bk, spec, gauge_tables(shells, w, dev), C, spin, bk.herm(S2), RY, checks)
        del C, spin
        norb = torch.as_tensor([sd[s].orb.norb for s in species], dtype=torch.int64, device=dev)[spec]
        stats = checks.read()
        steps('finish')
        lstats['blocks_scattered'] = int(stats.pop('local_blocks_scattered'))
        nelec = float(sum(sd[s].upf.z_valence for s in symbols))
        info = dict(n_blocks=len(bk), width=w, fft_shape=list(shape), local=lstats, nonlocal_=nstats,
                    field=field.metadata, table_store=str(self.bank.root), two_center_dr_bohr=self.bank.dr)
        return PriorBlocks(keys=bk.keys, blocks=bk, H=H, S=S2, norb=norb, species=species, spec=spec, nelec=nelec,
                           timing=steps.done(), info=info)

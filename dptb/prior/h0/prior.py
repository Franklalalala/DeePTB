"""H0 prior (UniTB route): geometry -> prior blocks on the device.

    support   pairs (i, j, R) with |x_j + R.cell - x_i| <= r_c(i) + r_c(j) + 2 max r_beta (+1e-12), r_c the orbital and
              r_beta the projector cutoffs of the composition (bohr)
    S, T      two-centre overlap and kinetic tables of the pair bank (``common.twocentre``)
    V_local   FFT-grid tile integration of the local potential (``field``, ``local``), ecutrho 480 Ry, PBE
    V_nl      per-centre Kleinman-Bylander contraction with the spinor KB matrices (``projector``)
    H         [[T + V_loc, 0], [0, T + V_loc]] + V_nl (Ry, ABACUS orbital order, spin-major), Hermitised as
              0.5 (H + H_rev^H) with H_rev the block of (j, i, -R); S likewise
    gauge     ABACUS -> DeePTB (per-species signed permutation), contiguous spin blocks (up orbitals, then down); eV

Returns ``PriorBlocks``: keys [m, 5] sorted by (i, j, R), H complex128 [m, 2w, 2w] (eV), S float64 [m, w, w], per-atom
orbital counts, species sorted by atomic number with each atom's index, valence electron count, segment timings, info.
"""
import threading
import time
from pathlib import Path

import numpy as np
import torch
from ase.data import atomic_numbers as Z_OF, chemical_symbols

from dptb.prior.common.blocks import KeyIndex, abacus_gauge
from dptb.prior.common.neighbors import image_pairs
from dptb.prior.h0.bank import PairBank
from dptb.prior.h0.field import RadialSpecies, RadialTable, build_field, fft_grid
from dptb.prior.h0.local import AOTables, local_blocks
from dptb.prior.h0.projector import nonlocal_blocks, spinor_d

RY = 13.605693122994
BOHR = 0.529177210903          # Angstrom
ECUTRHO = 480.0


class PriorBlocks:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class H0Prior:
    """H0 prior on a pair bank (``bank``: directory of the schema h0-physical-pair-bank/v1, 0.02 bohr grid).

    radial_table: tabulate the local-potential radial transforms per species on a uniform |G| grid (step
    ``radial_step`` bohr^-1) and interpolate them (cubic Hermite) instead of the direct quadrature at every |G|.
    extension: extension tables root; its ``h0`` bank is stacked after ``bank``. build_missing (off by default): pairs
    neither bank holds are built into it on the GPU (``dptb.prior.build``) before use; ORB/UPF of species the
    deployed bank lacks come from ``species_files`` {symbol: (orb, upf)} or the ``pporb`` directories.
    """

    def __init__(self, bank, *, device='cuda', host_bytes=16 << 30, radial_table=False, radial_step=5e-4,
                 phi_bytes=4 << 30, extension=None, build_missing=False, species_files=None, pporb=()):
        self.device = torch.device(device)
        self.extension = Path(extension) if extension else None
        self.build_missing, self.species_files, self.pporb = build_missing, dict(species_files or {}), list(pporb)
        self.bank = PairBank(bank, device=self.device, host_bytes=host_bytes,
                             extensions=[self.extension / 'h0'] if self.extension else ())
        self.radial_table, self.radial_step, self.phi_bytes = radial_table, radial_step, phi_bytes
        self._species, self._radial, self._tables, self._ao = {}, {}, {}, {}
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

    def _radial_tables(self, names):
        out = {}
        for s in names:
            key = (s, self.radial_step)
            if key not in self._tables:
                self._tables[key] = RadialTable(self._radial[s], np.sqrt(ECUTRHO) * (1 + 1e-9), self.radial_step)
            out[s] = self._tables[key]
        return out

    def _ao_tables(self, names, sd):
        key = tuple(names)
        if key not in self._ao:
            self._ao[key] = AOTables([sd[s].orb for s in names], self.device)
            while len(self._ao) > 8:
                self._ao.pop(next(iter(self._ao)))
        return self._ao[key]

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------------------------------------------- prior
    def __call__(self, z, pos, cell, ei=None, es=None):
        """z [n], pos [n, 3], cell [3, 3] (Angstrom, device tensors); the model edges ei, es are not needed."""
        tm = {}
        t0 = time.perf_counter()
        dev = self.device
        symbols = [chemical_symbols[int(x)] for x in z.tolist()]
        names = sorted(set(symbols))                         # table order (alphabetical), as the bank composition
        n = len(symbols)
        if self._prefetch is not None:
            self._prefetch.join()
            self._prefetch = None
        sd, comp = self._load(names)
        self._sync(); tm['tables_s'] = time.perf_counter() - t0
        t = time.perf_counter()
        cell_a = cell.to(dev, torch.float64)
        frac = pos.to(dev, torch.float64) @ torch.linalg.inv(cell_a)
        cb = cell_a * (1.0 / BOHR)
        P = frac @ cb                                        # bohr, positions as given
        cell_np = cb.cpu().numpy()
        shape = fft_grid(cell_np, ECUTRHO, device=dev)
        self._sync(); tm['grid_s'] = time.perf_counter() - t
        t = time.perf_counter()
        radial = {s: self._radial_species(s, sd) for s in names}
        tables = self._radial_tables(names) if self.radial_table else None
        field = build_field(radial, symbols, frac, cell_np, shape, ecutrho=ECUTRHO, tables=tables)
        self._sync(); tm['field_s'] = time.perf_counter() - t
        t = time.perf_counter()
        tab = self._ao_tables(names, sd)
        sp = torch.as_tensor([names.index(s) for s in symbols], device=dev)
        rcut = tab.rcut[sp]
        pcut = torch.as_tensor([sd[s].upf.max_projector_cutoff for s in symbols], dtype=torch.float64, device=dev)
        maxp = max(sd[s].upf.max_projector_cutoff for s in symbols)
        i, j, S = image_pairs(P, cb, rcut, margin=2 * maxp + 1e-12, self_interaction=True)
        keys = torch.cat([i[:, None], j[:, None], S], 1)
        index = KeyIndex(keys, n)
        nk = len(keys)
        self._sync(); tm['pairs_s'] = time.perf_counter() - t
        t = time.perf_counter()
        w = tab.wmax
        d = P[j] + S.to(torch.float64) @ cb - P[i]
        S2 = comp.S(sp[i], sp[j], d, w, w)
        T2 = comp.T(sp[i], sp[j], d, w, w)
        del d
        self._sync(); tm['two_center_s'] = time.perf_counter() - t
        t = time.perf_counter()
        V, lstats = local_blocks(field, tab, sp, P, frac, cb, index, keys, phi_bytes=self.phi_bytes)
        self._sync(); tm['local_s'] = time.perf_counter() - t
        t = time.perf_counter()
        mp = max(comp.nproj)
        D = spinor_d([comp.d[s] for s in names], comp.nproj, mp, dev)
        H, nstats = nonlocal_blocks(comp.Q, D, sp, P, frac, cb, pcut, rcut, index, nk, w)
        self._sync(); tm['nonlocal_s'] = time.perf_counter() - t
        t = time.perf_counter()
        # ABACUS -> DeePTB gauge, contiguous spin blocks; output species sorted by atomic number
        species = sorted(names, key=lambda x: Z_OF[x])
        hidx = np.zeros((len(species), 2 * w), dtype=np.int64)
        hsg = np.zeros((len(species), 2 * w))
        sidx = np.zeros((len(species), w), dtype=np.int64)
        ssg = np.zeros((len(species), w))
        norb_s = []
        for q, s in enumerate(species):
            m = sd[s].orb.norb
            perm, sign = abacus_gauge([c.l for c in sd[s].orb.channels], w)
            hidx[q, :2 * m] = np.r_[perm[:m], w + perm[:m]]
            hsg[q, :2 * m] = np.r_[sign[:m], sign[:m]]
            sidx[q, :m] = perm[:m]
            ssg[q, :m] = sign[:m]
            norb_s.append(m)
        hidx, hsg, sidx, ssg = (torch.as_tensor(a, device=dev) for a in (hidx, hsg, sidx, ssg))
        spec = torch.as_tensor([species.index(s) for s in symbols], device=dev)
        norb = torch.as_tensor(norb_s, dtype=torch.int64, device=dev)[spec]
        TV = (T2 + V).to(torch.complex128)
        H[:, :w, :w] += TV
        H[:, w:, w:] += TV
        del TV, V, T2
        rev = index.reverse(keys)
        Hr = H[rev].transpose(1, 2).conj()
        pre_h = (H - Hr).abs().max()
        H = 0.5 * (H + Hr)
        del Hr
        Sr = S2[rev].transpose(1, 2)
        pre_s = (S2 - Sr).abs().max()
        S2 = 0.5 * (S2 + Sr)
        del Sr
        si, sj = spec[i], spec[j]
        H = H.gather(1, hidx[si][:, :, None].expand(nk, 2 * w, 2 * w))
        H = H.gather(2, hidx[sj][:, None, :].expand(nk, 2 * w, 2 * w))
        H = (H * RY) * (hsg[si][:, :, None] * hsg[sj][:, None, :])
        S2 = S2.gather(1, sidx[si][:, :, None].expand(nk, w, w)).gather(2, sidx[sj][:, None, :].expand(nk, w, w))
        S2 = S2 * (ssg[si][:, :, None] * ssg[sj][:, None, :])
        finite = torch.isfinite(torch.view_as_real(H)).all() & torch.isfinite(S2).all()
        closed, pre_h, pre_s, finite = torch.stack([(rev >= 0).all().double(), pre_h, pre_s, finite.double()]).tolist()
        if not closed:
            raise ValueError('pair support is not closed under (i, j, R) -> (j, i, -R)')
        if not finite:
            raise ValueError('non-finite H0 or S')
        pre = dict(hamiltonian_max_abs=pre_h, overlap_max_abs=pre_s)
        self._sync(); tm['finish_s'] = time.perf_counter() - t
        tm['total_s'] = time.perf_counter() - t0
        nelec = float(sum(sd[s].upf.z_valence for s in symbols))
        info = dict(n_blocks=nk, width=w, fft_shape=list(shape), local=lstats, nonlocal_=nstats, field=field.metadata,
                    pre_hermitization=pre, radial_table=bool(self.radial_table), table_store=str(self.bank.root),
                    two_center_dr_bohr=self.bank.dr)
        return PriorBlocks(keys=keys, H=H, S=S2, norb=norb, species=species, spec=spec, nelec=nelec, timing=tm,
                           info=info)

"""H0 pair bank builder on the device (schema h0-physical-pair-bank/v1), without pyabacus.

A pair record (h0-physical-pair/v1) holds, for the views A|B and B|A of an unordered species pair, the S, T and
projector-overlap Q tables as index maps [l1, z1, l2, z2, L] -> rows of cubic Hermite coefficients of S_L(R)/R^L on the
bank's common grid (``dptb.prior.common.sbt.UniformSBT``: dr = cutoff / (nr - 1), FFT-compliant k grid), the orbital and
projector descriptors and spinor KB matrices of both species, and the real Gaunt table; it is what ABACUS ModuleNAO
tabulated for the deployed bank:
  * radial functions on the common grid r_i = i dr: ORB channels (implicit power p = 0) and the aligned UPF KB
    projectors as r beta (p = 1) zeroed from their cutoff index; a source grid containing the common grid is
    subsampled, any other goes through a not-a-knot cubic spline (zero beyond the source grid);
  * F = forward transforms; rows (l1, z1, l2, z2, L) with |l1 - l2| <= L <= l1 + l2, L = l1 + l2 (mod 2), in the
    ABACUS two_center_loop order, P = F1 F2 (S, Q) or k^2 F1 F2 (T) -> y = S_L(R)/R^L and y' -> Hermite coefficients
    (all rows of a pair in one product per L);
  * spinor KB matrices (``dptb.prior.common.kb.spinor_d`` on the (l, zeta, m) descriptors) and descriptors as the
    deployed generator; real Gaunt table ``dptb.prior.common.gaunt``.
The physical contract (grid, cutoff, nspin, projector alignment applied exactly once) is taken from a reference bank;
known species use the reference's prepared (aligned) species data, new species are prepared from their ORB/UPF files.
Pair files are written uncompressed through ``dptb.prior.common.store`` (host encoding overlaps the next pair).
"""
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from dptb.prior.common.gaunt import gaunt_table
from dptb.prior.common.kb import spinor_d
from dptb.prior.common.sbt import UniformSBT, hermite
from dptb.prior.common.store import TableStore, sha256_file
from dptb.prior.common.pporb import abacus_m_order
from dptb.prior.h0 import records

SCHEMA = 'h0-physical-pair-bank/v1'
PAIR_SCHEMA = 'h0-physical-pair/v1'
KINDS = {'S': ('orb', 'orb', False), 'T': ('orb', 'orb', True), 'Q': ('proj', 'orb', False)}


def to_common(r_src, v_src, r):
    """Values on the common grid r: subsample when the source grid contains it, else not-a-knot cubic spline."""
    r_src = np.asarray(r_src, dtype=np.float64)
    v_src = np.asarray(v_src, dtype=np.float64)
    out = np.zeros(len(r))
    inside = r <= r_src[-1] + 1e-12
    h = r_src[1] - r_src[0]
    step = (r[1] - r[0]) / h
    uniform = abs(r_src[0]) < 1e-14 and np.allclose(np.diff(r_src), h, rtol=0, atol=1e-12)
    if uniform and abs(step - round(step)) < 1e-9:
        idx = np.round(r[inside] / h).astype(np.int64)
        if np.max(np.abs(r_src[idx] - r[inside])) < 1e-10:
            out[inside] = v_src[idx]
            return out
    from scipy.interpolate import CubicSpline
    out[inside] = CubicSpline(r_src, v_src, bc_type='not-a-knot')(r[inside])
    return out


class Species:
    """Radial functions of one species on the common grid, their transforms (device), descriptors and spinor D."""

    def __init__(self, symbol, data, sbt, nspin=4):
        self.symbol = symbol
        orb, upf = data.orb, data.upf
        if any(c.l > 4 for c in orb.channels) or any(p.l > 4 for p in upf.projectors):
            raise NotImplementedError('two-centre tables support l <= 4')
        r = sbt.r.cpu().numpy()
        self.desc_orb = [(c.l, c.zeta, m) for c in orb.channels for m in abacus_m_order(c.l)]
        seen, slots, proj = Counter(), {}, []
        for p in upf.projectors:
            z = seen[p.l]
            seen[p.l] += 1
            slots[p.index] = (p.l, z)
            u = np.asarray(p.radial_u, dtype=np.float64).copy()
            u[p.cutoff_index:] = 0.0
            proj.append((p.l, z, to_common(upf.r, u, r)))
        if not proj:
            raise NotImplementedError('KB layout requires at least one projector per species')
        self.desc_proj = [(l, z, m) for l in sorted(seen) for z in range(seen[l]) for m in abacus_m_order(l)]
        if nspin == 4:
            order = [(p.l, slots[p.index][1], m) for p in upf.projectors for m in abacus_m_order(p.l)]
            perm = np.asarray([order.index(x) for x in self.desc_proj])
            perm = np.r_[perm, len(perm) + perm]
            self.d = spinor_d([p.l for p in upf.projectors], [p.j for p in upf.projectors], upf.dij_ry,
                              has_so=bool(upf.has_so))[np.ix_(perm, perm)]
        else:
            lookup = {x: i for i, x in enumerate(self.desc_proj)}
            self.d = np.zeros((len(self.desc_proj),) * 2)
            for p in upf.projectors:
                for q in upf.projectors:
                    if p.l == q.l:
                        for m in abacus_m_order(p.l):
                            self.d[lookup[(*slots[p.index], m)], lookup[(*slots[q.index], m)]] = upf.dij_ry[p.index, q.index]
        orbs = [(c.l, c.zeta, to_common(orb.r, c.radial, r)) for c in orb.channels]
        self.F = {'orb': self._transform(orbs, 0, sbt), 'proj': self._transform(proj, 1, sbt)}

    @staticmethod
    def _transform(funcs, p, sbt):
        """{(l, zeta): F(k)} on the device."""
        out = {}
        for l in sorted({f[0] for f in funcs}):
            sel = [f for f in funcs if f[0] == l]
            F = sbt.forward(torch.as_tensor(np.stack([f[2] for f in sel])), l, p)
            out.update({(l, f[1]): row for f, row in zip(sel, F)})
        return out


def _rows(Fb, Fk, lb, nb, lk, nk):
    """Rows (l1, z1, l2, z2, L) of one view in ABACUS two_center_loop order."""
    rows = []
    for L in range(lb + lk + 1):
        for l1 in range(lb + 1):
            for z1 in range(nb):
                if (l1, z1) in Fb:
                    rows += [(l1, z1, l2, z2, L) for l2 in range(abs(l1 - L), min(lk, L + l1) + 1, 2)
                             for z2 in range(nk) if (l2, z2) in Fk]
    return rows


def pair_record(sbt, species, physical_contract_sha256):
    """The h0-physical-pair/v1 record of a sorted species tuple (one or two ``Species``), before its checksum."""
    sp = list(species)
    views = [(sp[0], sp[0])] if len(sp) == 1 else [(sp[0], sp[1]), (sp[1], sp[0])]
    slot, Fall = {}, []
    for s in sp:
        for kind in ('orb', 'proj'):
            for key, row in s.F[kind].items():
                slot[(s.symbol, kind, key)] = len(Fall)
                Fall.append(row)
    Fall = torch.stack(Fall)
    plan, i1, i2, Ls, kin = [], [], [], [], []
    for name, (bk, kk, kinetic) in KINDS.items():
        lb = max(l for s in sp for (l, _) in s.F[bk])
        nb = max(z + 1 for s in sp for (_, z) in s.F[bk])
        lk = max(l for s in sp for (l, _) in s.F[kk])
        nk = max(z + 1 for s in sp for (_, z) in s.F[kk])
        for a, b in views:
            rows = _rows(a.F[bk], b.F[kk], lb, nb, lk, nk)
            index = torch.full((lb + 1, nb, lk + 1, nk, lb + lk + 1), -1, dtype=torch.int32)
            if rows:
                index[tuple(torch.as_tensor(rows).T)] = torch.arange(len(rows), dtype=torch.int32)
            plan.append((name, a.symbol + '|' + b.symbol, index, len(i1), len(rows)))
            i1 += [slot[(a.symbol, bk, (r[0], r[1]))] for r in rows]
            i2 += [slot[(b.symbol, kk, (r[2], r[3]))] for r in rows]
            Ls += [r[4] for r in rows]
            kin += [kinetic] * len(rows)
    dev = Fall.device
    i1, i2, Ls = (torch.as_tensor(v, device=dev) for v in (i1, i2, Ls))
    P = Fall[i1] * Fall[i2]
    P = torch.where(torch.as_tensor(kin, device=dev)[:, None], P * sbt.k * sbt.k, P)
    y, dy = torch.empty((len(Ls), sbt.nr), dtype=torch.float64, device=dev), torch.empty((len(Ls), sbt.nr),
                                                                                         dtype=torch.float64, device=dev)
    for L in torch.unique(Ls).tolist():
        sel = torch.nonzero(Ls == L)[:, 0]
        y[sel], dy[sel] = sbt.two_centre(P[sel], L)
    coeffs = hermite(y, dy, sbt.dr).cpu()
    blocks = {k: {} for k in KINDS}
    for name, view, index, start, n in plan:
        blocks[name][view] = dict(index=index, coeffs=coeffs[start:start + n])
    lmax = max(4, max(l for s in sp for kind in ('orb', 'proj') for (l, _) in s.F[kind]))
    atoms = {s.symbol: dict(desc={'orb': s.desc_orb, 'proj': s.desc_proj}, d=s.d) for s in sp}
    return dict(schema=PAIR_SCHEMA, species=[s.symbol for s in sp], physical_contract_sha256=physical_contract_sha256,
                blocks=blocks, atoms=atoms, gaunt_table=gaunt_table(lmax), lmax=lmax)


def encode_record(record):
    """Offline (h0-offline/v2) arrays of a record with its fingerprint checksum."""
    record = dict(record)
    record['checksum'] = records.fingerprint(record)
    arrays = {}
    meta = records.encode(record, arrays)
    arrays['__meta__'] = np.asarray(json.dumps({'schema': records.SCHEMA, 'data': meta}))
    return arrays


def prepare_species(orb_path, upf_path):
    """Aligned species data and its key from ORB/UPF files (as the bank's species store: projector alignment once)."""
    from scipy.interpolate import CubicSpline
    from dptb.prior.common.pporb import SpeciesData, align_soc_projectors, read_abacus_orb, read_upf
    data = SpeciesData(read_abacus_orb(orb_path), read_upf(upf_path))
    for obj in (data.orb, data.upf):
        obj.metadata['offline_source_sha256'] = sha256_file(obj.source)
    data.orb.metadata['offline_spline_coefficients'] = [
        CubicSpline(data.orb.r, c.radial, bc_type='not-a-knot', extrapolate=False).c for c in data.orb.channels]
    raw = records.fingerprint(data)
    aligned = align_soc_projectors({'x': data})['x']
    return aligned, records.fingerprint(aligned), raw


class Builder:
    """Species transforms are prepared once on the device; pairs are tabulated on demand."""

    def __init__(self, reference, *, species_files=None, device='cuda'):
        self.ref = Path(reference)
        self.m = json.loads((self.ref / 'manifest.json').read_text())
        if self.m.get('schema') != SCHEMA:
            raise ValueError(f'{reference}: not an H0 pair bank')
        self.catalog = json.loads((self.ref / 'catalog.json').read_text())
        self.sbt = UniformSBT(self.m['cutoff_bohr'], self.m['nr'], device)
        if abs(self.sbt.dr - float(self.m.get('actual_dr_bohr', self.m['dr_bohr']))) > 1e-12:
            raise ValueError('reference grid is not uniform with its recorded step')
        self.files = dict(species_files or {})
        self.data, self.rows, self.species = {}, {}, {}

    def species_data(self, s):
        if s not in self.data:
            if s in self.files:
                data, key, raw = prepare_species(*self.files[s])
                self.rows[s] = dict(key=key, raw_key=raw, orb_sha256=data.orb.metadata['offline_source_sha256'],
                                    upf_sha256=data.upf.metadata['offline_source_sha256'],
                                    status='prepared from ORB/UPF, projector alignment exactly once')
            elif s in self.catalog['species']:
                key = self.catalog['species'][s]
                data = records.read(self.ref / 'species' / f'{key}.npz')
                row = self.m['species'][s]
                self.rows[s] = {k: row[k] for k in ('key', 'raw_key', 'orb_sha256', 'upf_sha256', 'status') if k in row}
            else:
                raise KeyError(f'no species data for {s}: give its ORB/UPF files')
            self.data[s] = data
        return self.data[s]

    def prepared(self, s):
        if s not in self.species:
            self.species[s] = Species(s, self.species_data(s), self.sbt, int(self.m['nspin']))
        return self.species[s]

    def record(self, pair):
        symbols = sorted(set(pair))
        return pair_record(self.sbt, [self.prepared(s) for s in symbols], self.m['physical_contract_sha256'])


def build(pairs, out_dir, *, reference, species_files=None, device='cuda', compress=False, workers=4, log=print):
    """Tabulate unordered species pairs ('A|B') into the H0 bank at ``out_dir`` (created or extended; files are never
    replaced). ``reference``: the bank whose physical contract and prepared species are used; ``species_files``:
    {symbol: (orb, upf)} for species it lacks. Returns {pair: seconds} (tabulation, device-synchronised)."""
    b = Builder(reference, species_files=species_files, device=device)
    m = b.m
    header = {k: m[k] for k in ('dr_bohr', 'actual_dr_bohr', 'cutoff_bohr', 'nr', 'nspin', 'physical_contract_sha256',
                                'grid_policy') if k in m}
    store = TableStore(out_dir, SCHEMA, **header)
    store.manifest.setdefault('projector_alignment', dict(application_count=1, implementation='dptb.prior.common.pporb'
                                                          '.align_soc_projectors', reference=str(Path(reference).resolve())))
    gid = store.generator('dptb.prior.h0.build', 'dptb.prior.common.sbt', 'dptb.prior.common.gaunt',
                          'dptb.prior.h0.records', 'dptb.prior.common.pporb')
    catalog_path = Path(out_dir) / 'catalog.json'
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else dict(
        schema='h0-physical-pair-bank/catalog/v1', pair_bank=True, dr_bohr=m['dr_bohr'], nspin=m['nspin'],
        compositions={}, species={})
    keys = sorted({'|'.join(sorted(set(p.split('|') if isinstance(p, str) else p))) for p in pairs})
    seconds = {}

    def write(key, record, t_tab):
        t0 = time.perf_counter()
        row = store.write_npz(Path('pairs') / ('__'.join(record['species']) + '.npz'), encode_record(record), compress)
        row['file'] = Path(row.pop('path')).name
        return key, dict(row, pair=key, generator=gid, tabulate_seconds=t_tab, write_seconds=time.perf_counter() - t0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for key in keys:
            if store.has('pairs', key):
                continue
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rec = b.record(key.split('|'))
            torch.cuda.synchronize()
            seconds[key] = time.perf_counter() - t0
            futures.append(pool.submit(write, key, rec, seconds[key]))
        for f in futures:
            key, row = f.result()
            store.add('pairs', key, row)
            log(f'H0_BUILT {key} tab={row["tabulate_seconds"]:.3f}s write={row["write_seconds"]:.3f}s')
    for s in sorted({s for k in keys for s in k.split('|')}):
        key = b.rows[s]['key'] if s in b.rows else None
        if key is None or s in catalog['species']:
            continue
        rel = Path('species') / f'{key}.npz'
        src = b.ref / 'species' / f'{key}.npz'
        if s in b.files or not src.exists():
            arrays = {}
            meta = records.encode(b.data[s], arrays)
            arrays['__meta__'] = np.asarray(json.dumps({'schema': records.SCHEMA, 'data': meta}))
            info = store.write_npz(rel, arrays)
        else:
            info = dict(zip(('sha256', 'bytes'), store.write(rel, src.read_bytes())), path=str(rel))
        store.manifest.setdefault('species', {})[s] = dict(b.rows[s], path=info['path'], sha256=info['sha256'])
        catalog['species'][s] = key
    store.flush()
    catalog_path.write_text(json.dumps(catalog, indent=1, sort_keys=True))
    return seconds


def species_of(reference):
    return sorted(json.loads((Path(reference) / 'manifest.json').read_text())['species'])


def present(out_dir):
    path = Path(out_dir) / 'manifest.json'
    return set(json.loads(path.read_text()).get('pairs', {})) if path.exists() else set()


ORDERED = False

"""GPU builder of the environment-XC tables (McWEDA density moments; LDA-PZ81 frozen atomic densities).

Species K: a finite-rank expansion of the multiplicative density operator in the rho_K metric,

    rho_K ~= sum_h |q_h> eps_h <q_h|,   q_h = rho_K p_h,   eps_h = 1 / <p_h|rho_K|p_h>,

on the grid r = 0 (as 1e-12), dr, ... up to the projector cutoff (first mesh point past the last rho_K > threshold).
Per l <= l_max(PAO) + l_buffer the radial seeds are, in order: the PAO radials of that l (padded with the first one
times powers of rho/rho_max + 1e-13), or for l without PAO channels the highest-l PAO radials times r^(l - l_max);
then ``tail_seeds`` functions r^l rho^(1/2) (r/cut)^n. Seeds are normalised in the plain r^2 metric and
Gram-Schmidt orthogonalised (two passes) in the rho_K metric; near-null seeds are dropped (trapezoid integrals).

Tables (two-centre, bond frame, the bipolar quadrature of ``build_pairxc.Bipolar``):

    rhofac   K|A   <q_{K,h} | phi_{A,mu}>                 envfac   K|A   <q_{K,h} | |R_{A,a}|>  (l = 0 envelopes)
    envnorm  A|B   <|R_a| | |R_b|>                        envpair  A|B   <|R_a| | rho_A + rho_B | |R_b|>
    pairmom  A|B   <phi_mu | rho_A + rho_B | phi_nu>

on the knots 0, step, ... covering the support (one more knot when the last lies within step/4 of it), zero at
d >= support - 1e-12, Gauss-Legendre order ``order`` on every axis. A|B kinds are keyed by the sorted pair; for
A = B the block is symmetrised exactly, M(d) = (M + P M^T P) / 2 with P = (-1)^l per AO.

Files and manifest follow the deployed layout read by ``dptb.prior.gf.tables``: ``species/K.npz`` (r, q_radial, q_l,
q_n, epsilon_radial, epsilon_ao, cutoff_bohr, metadata, identity), ``<kind>/L__R.npz`` (distances, values,
left_shells, right_shells, support_bohr, identity) and ``manifest.json``.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from dptb.prior.common.store import TableStore
from dptb.prior.gf.build_pairxc import Bipolar, Radials, Sources, Splines

SCHEMA = 'deeptb.envxc_table/v1'       # deployed manifests carry the legacy id (tables.LEGACY_SCHEMAS)
CENTRE_KINDS = ('rhofac', 'envfac')
PAIR_KINDS = ('envnorm', 'envpair', 'pairmom')
SETTINGS = dict(radial_rank=2, l_buffer=1, tail_seeds=1, density_threshold=1e-7, projector_cutoff=None,
                grid_step=0.005, distance_step=0.05, order=128, background_nodes=None, import_background_sha256=None)


def _sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
                          .encode('ascii')).hexdigest()


def distance_grid(support, step):
    """Uniform knots from 0 covering ``support`` (one more knot when the last lies within step/4 of it)."""
    end = math.ceil(float(support) / float(step)) * float(step)
    if end < support + 0.25 * step:
        end += float(step)
    return np.linspace(0.0, end, int(round(end / float(step))) + 1, dtype=np.float64)


def _trapz(y, x):
    return float(torch.trapezoid(y, x))


class Projectors:
    """Density projectors of one species (module docstring), built on the device (or read back with ``load``)."""

    @classmethod
    def load(cls, path, device='cuda'):
        self = cls.__new__(cls)
        with np.load(path, allow_pickle=False) as z:
            self.r, self.q_radial = z['r'], z['q_radial']
            self.q_l, self.q_n, self.cut = z['q_l'], z['q_n'], float(z['cutoff_bohr'])
            self.epsilon_radial, self.epsilon_ao = z['epsilon_radial'], z['epsilon_ao']
            self.metadata = json.loads(str(z['metadata']))
        self.symbol = self.metadata.get('symbol', '')
        self.radials = self._radials(device)
        return self

    def arrays(self, identity):
        return dict(r=self.r, q_radial=self.q_radial, q_l=self.q_l, q_n=self.q_n, epsilon_radial=self.epsilon_radial,
                    epsilon_ao=self.epsilon_ao, cutoff_bohr=np.array(self.cut),
                    metadata=np.array(json.dumps(self.metadata, sort_keys=True)),
                    identity=np.array(json.dumps(identity, sort_keys=True)))

    def __init__(self, src, radial_rank=2, l_buffer=1, tail_seeds=1, density_threshold=1e-7, grid_step=0.005,
                 cutoff=None, standard_norm_tol=1e-18, metric_tol=1e-14, passes=2):
        dev = src.device
        r_rho = torch.as_tensor(src.r_rho, device=dev)
        rho_mesh = src.density(r_rho)
        active = torch.nonzero(rho_mesh > density_threshold).flatten()
        if not len(active):
            raise ValueError(f'{src.symbol}: density never exceeds {density_threshold}')
        rho_cut = float(src.r_rho[min(int(active[-1]) + 1, len(src.r_rho) - 1)])
        cut = float(rho_cut if cutoff is None else cutoff)
        r_np = np.arange(0.0, cut + grid_step, grid_step)
        r_np[0] = 1e-12
        r = torch.as_tensor(r_np, device=dev)
        rho = src.density(r)
        seed_scale = rho / max(float(rho.max()), 1e-300) + 1e-13
        lmax = max(src.shells)
        by_l = {}
        for c, l in enumerate(src.shells):
            by_l.setdefault(l, []).append(c)
        pao = src.radial(r)

        def inner(a, b, w=None):
            y = r * r * a * b if w is None else r * r * a * b * w
            return _trapz(y, r)

        q_rows, q_l, q_n, eps, kinds, dropped, offdiag = [], [], [], [], [], [], 0.0
        for l in range(lmax + l_buffer + 1):
            seeds, chans = [], by_l.get(l, [])
            if l <= lmax and chans:
                for i in range(radial_rank):
                    seeds.append((pao[chans[i]].clone(), 'pao') if i < len(chans) else
                                 (pao[chans[0]] * seed_scale ** (i - len(chans) + 1), 'pao0_times_rho_power'))
            else:
                top = by_l[lmax]
                for i in range(radial_rank):
                    base = pao[top[min(i, len(top) - 1)]] * r ** (l - lmax)
                    seeds.append((base * seed_scale ** max(0, i - len(top) + 1), 'highest_l_times_r_power'))
            for n in range(tail_seeds):
                seeds.append((r ** l * torch.sqrt(rho.clamp_min(0.0)) * (r / cut) ** n, 'tail_r_l_sqrt_rho'))
            kept, norms = [], []
            for i, (cand, kind) in enumerate(seeds):
                norm2 = inner(cand, cand)
                if not math.isfinite(norm2) or norm2 <= standard_norm_tol:
                    dropped.append(dict(l=l, seed=i, kind=kind, reason='null standard norm'))
                    continue
                p = cand / math.sqrt(norm2)
                for _ in range(passes):
                    for prev, pn in zip(kept, norms):
                        p = p - (inner(prev, p, rho) / pn) * prev
                vnorm = inner(p, p, rho)
                if not math.isfinite(vnorm) or vnorm <= metric_tol * max(norms, default=1.0):
                    dropped.append(dict(l=l, seed=i, kind=kind, reason='near-null rho metric', metric=vnorm))
                    continue
                kept.append(p)
                norms.append(vnorm)
                q_rows.append(rho * p)
                q_l.append(l)
                q_n.append(len(kept) - 1)
                eps.append(1.0 / vnorm)
                kinds.append(kind)
            if kept:
                metric = np.array([[inner(a, b, rho) for b in kept] for a in kept])
                scale = np.sqrt(np.abs(np.diag(metric)))
                normed = np.abs(metric) / np.maximum(scale[:, None] * scale[None, :], 1e-300)
                normed -= np.diag(np.diag(normed))
                offdiag = max(offdiag, float(normed.max(initial=0.0)))
        if not q_rows:
            raise ValueError(f'{src.symbol}: no density projector survived')
        q = torch.stack(q_rows)
        q[:, r >= cut] = 0.0
        self.symbol, self.r, self.cut = src.symbol, r_np, cut
        self.q_radial = q.cpu().numpy()
        self.q_l, self.q_n = np.asarray(q_l, np.int64), np.asarray(q_n, np.int64)
        self.epsilon_radial = np.asarray(eps)
        self.epsilon_ao = np.concatenate([np.full(2 * int(l) + 1, e) for l, e in zip(q_l, eps)])
        self.metadata = dict(
            radial_rank=int(radial_rank), l_buffer=int(l_buffer), tail_seeds=int(tail_seeds), lmax=int(lmax + l_buffer),
            density_threshold_bohr3=float(density_threshold), density_cutoff_bohr=rho_cut, projector_cutoff_bohr=cut,
            orbital_cutoff_bohr=float(src.cut), grid_step_bohr=float(grid_step), radial_projectors=len(eps),
            angular_projectors=int(sum(2 * l + 1 for l in q_l)), normalized_rho_metric_offdiag_max=offdiag,
            epsilon_min_bohr3=float(np.min(eps)), epsilon_max_bohr3=float(np.max(eps)),
            electrons_beyond_cutoff=self._beyond(r_rho, rho_mesh, cut),
            electrons_beyond_orbital_cutoff=self._beyond(r_rho, rho_mesh, src.cut),
            seed_kinds=kinds, dropped_seeds=dropped, symbol=src.symbol)
        self.radials = self._radials(dev)

    @staticmethod
    def _beyond(r, rho, radius):
        tail = r >= radius
        if int(tail.sum()) < 2:
            return 0.0
        return _trapz((4.0 * math.pi * r * r * rho)[tail], r[tail])

    def _radials(self, dev):
        sp = Splines(self.r, self.q_radial, dev)
        lo, hi, cut = float(self.r[0]), float(self.r[-1]), self.cut
        return Radials(self.q_l, cut, lambda x: torch.where((x >= lo) & (x <= hi) & (x < cut), sp(x), 0.0))


def table_values(kind, left, right, settings=SETTINGS, chunk_points=1 << 22):
    """(distances, values [K, nL, nR], left_shells, right_shells, support) of one table on the device.

    ``left`` is the Projectors of K for centre kinds and the AtomicSource of A for pair kinds; ``right`` an
    AtomicSource (the same object as ``left`` for A = A pair tables, which are symmetrised).
    """
    weight = None
    if kind in CENTRE_KINDS:
        L, R = left.radials, (right.orbitals if kind == 'rhofac' else right.envelopes)
    elif kind == 'envnorm':
        L, R = left.envelopes, right.envelopes
    elif kind in ('envpair', 'pairmom'):
        L, R = (left.envelopes, right.envelopes) if kind == 'envpair' else (left.orbitals, right.orbitals)
        weight = lambda ra, rb: left.density(ra) + right.density(rb)  # noqa: E731
    else:
        raise ValueError(f'unknown environment-XC table kind {kind}')
    support = float(L.cut + R.cut)
    d = distance_grid(support, settings['distance_step'])
    values = Bipolar(L, R, weight, chunk_points, right.device).blocks(d, int(settings['order']))
    values[torch.as_tensor(d >= support - 1e-12, device=values.device)] = 0.0
    if kind in PAIR_KINDS and left is right:
        par = torch.as_tensor(np.repeat([(-1.0) ** l for l in L.shells], [2 * l + 1 for l in L.shells]),
                              device=values.device)
        values = 0.5 * (values + par[None, :, None] * par[None, None, :] * values.transpose(1, 2))
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f'{kind} table produced non-finite values')
    return d, values, L.shells, R.shells, support


def items_for(new, existing):
    """Every (kind, left, right) a set of new species needs next to the existing ones."""
    every = sorted(set(new) | set(existing))
    out = set()
    for n in new:
        for s in every:
            out |= {(k, n, s) for k in CENTRE_KINDS} | {(k, s, n) for k in CENTRE_KINDS}
            out |= {(k, *sorted((n, s))) for k in PAIR_KINDS}
    return sorted(out)


GENERATOR = ('dptb.prior.gf.build_envxc', 'dptb.prior.gf.build_pairxc', 'dptb.prior.common.xc',
             'dptb.prior.common.radial')
CONTRACT = dict(length_unit='bohr', density_unit='bohr^-3', xc_energy_unit='eV', harmonic_convention='deeptb_abacus_real',
                endpoint_policy='exclude_i0_and_jR', interpolation='cubic', background_nodes=None,
                xc_functional='LDA-PZ81 unpolarized; density = normalized neutral valence + unscaled NLCC',
                density_operator='rho_K ~= sum_h |rho_K p_h> eps_h <p_h rho_K|, p_h Gram-Schmidt in the rho_K metric')


def build(tables, out_dir, sources_dir=None, device='cuda', settings=None, chunk_points=1 << 22, species_files=None):
    """Build environment-XC tables ('kind:L|R' strings or (kind, L, R)) into the store ``out_dir``.

    Every species involved gets its projectors (read back when the store holds them); tables the store holds are
    skipped (never rewritten). Inputs: ``sources_dir/<S>.npz`` or ``species_files[S] = (orb, upf)``.
    Returns [(item, path, seconds)], species first.
    """
    settings = dict(SETTINGS, **(settings or {}))
    if settings.get('background_nodes') is not None:
        raise ValueError('background (xcbg) tables are not part of this recipe')
    store = TableStore(out_dir, SCHEMA, settings=settings, numerical_settings=settings, **CONTRACT)
    gid = store.generator(*GENERATOR)
    src = Sources(sources_dir, species_files, device)
    items = []
    for t in tables:
        kind, key = t.split(':') if isinstance(t, str) else (t[0], f'{t[1]}|{t[2]}')
        a, b = key.split('|')
        items.append((kind, *(sorted((a, b)) if kind in PAIR_KINDS else (a, b))))
    projectors, results = {}, []
    for s in sorted({s for _, a, b in items for s in (a, b)}):
        t0 = time.perf_counter()
        ident = dict(schema=SCHEMA, kind='species', key=s, generator=gid, sources={s: src[s].sha256})
        if store.has('species', s):
            projectors[s] = Projectors.load(store.root / store.manifest['species'][s]['array_path'], device)
            continue
        proj = projectors[s] = Projectors(src[s], settings['radial_rank'], settings['l_buffer'], settings['tail_seeds'],
                                          settings['density_threshold'], settings['grid_step'],
                                          settings['projector_cutoff'])
        row = store.write_npz(f'species/{s}.npz', proj.arrays(ident))
        store.add('species', s, dict(array_path=row['path'], array_sha256=row['sha256'], q_shells=[int(l) for l in proj.q_l],
                                     q_norb=int(len(proj.epsilon_ao)), q_cutoff_bohr=proj.cut,
                                     orbital_shells=list(src[s].shells), orbital_norb=src[s].norb,
                                     orbital_cutoff_bohr=src[s].cut, z_valence=src[s].z_valence,
                                     metadata=proj.metadata, identity=ident))
        store.manifest.setdefault('sources', {})[s] = dict(source_sha256=src[s].sha256, **src[s].identity)
        results.append((f'species:{s}', store.root / row['path'], time.perf_counter() - t0))
    rows = store.manifest.setdefault('tables', {})
    for kind, a, b in items:
        key = f'{a}|{b}'
        if key in rows.get(kind, {}):
            continue
        t0 = time.perf_counter()
        right = src[b]
        left = projectors[a] if kind in CENTRE_KINDS else (right if a == b else src[a])
        d, values, ls, rs, support = table_values(kind, left, right, settings, chunk_points)
        ident = dict(schema=SCHEMA, kind=kind, key=key, index=None, generator=gid,
                     sources={s: src[s].sha256 for s in {a, b}})
        row = store.write_npz(f'{kind}/{a}__{b}.npz', dict(
            distances=d, values=values.cpu().numpy(), left_shells=np.asarray(ls, np.int64),
            right_shells=np.asarray(rs, np.int64), support_bohr=np.array(support),
            identity=np.array(json.dumps(ident, sort_keys=True))))
        seconds = time.perf_counter() - t0
        rows.setdefault(kind, {})[key] = dict(row, left_shells=list(ls), right_shells=list(rs), support_bohr=support,
                                              knots=len(d), seconds=seconds)
        results.append((f'{kind}:{key}', store.root / row['path'], seconds))
    store.flush(table_count=sum(len(v) for v in rows.values()), failures=[])
    return results

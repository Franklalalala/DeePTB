"""P23 factorised VNA three-centre tables (schema deeptb.p23_factorized_vna_table/v1) on the device, batched over pairs.

Species V, on its ORB grid r, with V_NA the neutral-atom potential of its UPF (``neutral_atom_potential_radial``, Ry ->
eV with 13.605698) linearly interpolated onto r and zero beyond the VNA cutoff r_c (the UPF radius after the last
|V_NA| > max(1e-14, tol max|V_NA|)): radial projectors p_{l,n} (l <= lmax_AO + l_buffer, n < rank) seeded by the PAO
channels of degree l in file order, then R_{l,0} s^n with s = 0.1 V_NA[Ha] + 1e-13; above lmax_AO by the top-degree
channels times r^(l - lmax_AO), then the last one times s^(n - n_top + 1). Each seed is normalised in int r^2 p^2 dr
and orthogonalised to the earlier ones of its degree in <f|g>_V = int r^2 f g V_NA dr (modified Gram-Schmidt, two
passes; trapezoid rule throughout). epsilon_{l,n} = 1 / <p|p>_V (1/eV); q_{l,n} = V_NA p_{l,n} (eV), cutoff
min(r_c, rc_AO).

Ordered pair V|A: the bond-frame overlap of the q functions of V with the AOs of A (``common.sbt.BondFrameSBT``)

    F_ab(d) = sum_L (2L + 1) Re(i^(l_a - l_b - L)) C^L_ab sum_k (2/pi) w_k k^2 j_L(k d) t_a(k) t_b(k),

kept for m_a = m_b and |C^L_ab| > 1e-13, zero for d >= support - 1e-12 with support = cut(q_V) + rc_A, on
``common.sbt.distance_grid(support, step)``. Transforms t are computed once per species; all pairs of one distance grid
are one GEMM [entries, (L + 1) n_k] x [(L + 1) n_k, n_d] over their m-matched entries (chunked), and a thread pool
writes the files while the next chunk computes:
    factors/V__A.npz   distances, values (float32 eV [n_d, n_q, n_AO]), left_shells, right_shells, support_bohr
    species/S.npz      epsilon_ao (per q function), radial_epsilon, radial_l, radial_n
with the manifest rows the deployed readers check (``common.store.TableStore``; existing pairs are never rebuilt).
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from dptb.prior.common.pporb import neutral_atom_potential_radial, read_abacus_orb, read_upf
from dptb.prior.common.sbt import BondFrameSBT, distance_grid, spherical_jn
from dptb.prior.common.store import TableStore, sha256_file

SCHEMA = 'deeptb.p23_factorized_vna_table/v1'
SUPPORT_SEMANTICS = ('strict distance < cutoff(q=V*p) + cutoff(AO); '
                     'q cutoff is min(physical VNA cutoff, PAO projector cutoff)')
SETTINGS = dict(radial_rank=4, l_buffer=2, distance_step_bohr=0.02, kmax_bohr_inv=80.0, n_k=400, n_mu=18, n_phi=36,
                interpolation='cubic', vna_relative_tolerance=1e-8)
RY_TO_EV, HARTREE_TO_EV = 13.605698, 27.211386245988      # the constants of the deployed tables


def species_files(symbols, pporb, reference=None, files=None):
    """{S: (orb, upf)}: ``files`` first, else '<S>.upf' and the single '<S>_*.orb' of the first pporb directory that has
    them, checked against the orbital_sha256 / upf_sha256 of the ``reference`` species rows when given."""
    dirs = [Path(p) for p in ([pporb] if isinstance(pporb, (str, Path)) else pporb)]
    out = {}
    for s in symbols:
        if files and s in files:
            out[s] = tuple(Path(f) for f in files[s])
        else:
            hit = [(orbs[0], d / f'{s}.upf') for d in dirs if (d / f'{s}.upf').is_file()
                   for orbs in [sorted(d.glob(f'{s}_*.orb'))] if len(orbs) == 1]
            if not hit:
                raise FileNotFoundError(f'{s}: no {s}.upf with a single {s}_*.orb in {dirs}')
            out[s] = hit[0]
        row = (reference or {}).get(s)
        if row and (sha256_file(out[s][0]) != row['orbital_sha256'] or sha256_file(out[s][1]) != row['upf_sha256']):
            raise ValueError(f'{s}: ORB/UPF differ from the reference tables')
    return out


def _layout(ls):
    """Per function of channels with degrees ls (m order 0, +1, -1, ...): channel, slot l^2 + q, signed m."""
    rows = [(c, l * l + q, (q + 1) // 2 * (1 if q % 2 else -1)) for c, l in enumerate(ls) for q in range(2 * l + 1)]
    return tuple(np.asarray(x, dtype=np.int64) for x in zip(*rows))


class Species:
    """ORB/UPF of one species, its VNA projectors and the transforms of its AOs and q functions (device)."""

    def __init__(self, symbol, orb_path, upf_path, sbt, st, device):
        self.symbol, self.orb_path, self.upf_path = symbol, Path(orb_path), Path(upf_path)
        orb, upf = read_abacus_orb(orb_path), read_upf(upf_path)
        r = np.asarray(orb.r, dtype=np.float64)
        radial = [np.asarray(c.radial, dtype=np.float64) for c in orb.channels]
        self.shells, self.rc = tuple(int(c.l) for c in orb.channels), float(orb.rcut)
        rv, v = (np.asarray(x, dtype=np.float64) for x in neutral_atom_potential_radial(upf))
        active = np.flatnonzero(np.abs(v) > max(1e-14, st['vna_relative_tolerance'] * float(np.max(np.abs(v)))))
        if not active.size:
            raise ValueError(f'{symbol}: neutral-atom potential has no active radial samples')
        self.vna_cut = float(rv[min(int(active[-1]) + 1, rv.size - 1)])
        src = v * RY_TO_EV
        pot = np.interp(r, rv, src, left=src[0], right=0.0)
        pot[r > self.vna_cut + 1e-12] = 0.0
        self.q_cut = min(self.vna_cut, self.rc, float(r[-1]))
        q = self._projectors(r, radial, pot, st, torch.device(device))
        self.ao = sbt.transforms(r, [(l, x, self.rc) for l, x in zip(self.shells, radial)]).F
        self.q = sbt.transforms(r, [(l, x, self.q_cut) for l, x in zip(self.q_shells, q.cpu().numpy())]).F
        self.ao_layout, self.q_layout = _layout(self.shells), _layout(self.q_shells)

    def _projectors(self, r, radial, pot, st, dev):
        rank, lmax = int(st['radial_rank']), max(self.shells)
        nl = lmax + int(st['l_buffer']) + 1
        rt, V = (torch.as_tensor(x, device=dev) for x in (r, pot))
        R = torch.as_tensor(np.stack(radial), device=dev)
        seed = 0.1 * (V / HARTREE_TO_EV) + 1e-13
        by_l = {}
        for i, l in enumerate(self.shells):
            by_l.setdefault(l, []).append(i)
        cand, kinds = [], []
        for l in range(nl):
            ch = by_l.get(min(l, lmax))
            if not ch:
                raise ValueError(f'{self.symbol}: orbital has no radial channel for l={l}')
            for n in range(rank):
                if l <= lmax:
                    c, k = (R[ch[n]], 'pao') if n < len(ch) else (R[ch[0]] * seed ** n, 'pao0_times_vna_power')
                elif n < len(ch):
                    c, k = R[ch[n]] * rt ** (l - lmax), 'highest_l_times_r_power'
                else:
                    c, k = R[ch[-1]] * seed ** (n - len(ch) + 1), 'highest_l_last_times_vna_power'
                cand.append(c)
                kinds.append(k)
        C = torch.stack(cand).reshape(nl, rank, -1)
        dr = rt[1:] - rt[:-1]

        def inner(f, g, w=None):                  # trapz(r^2 f g [V_NA], r) over the last axis
            y = rt * rt * f * g if w is None else rt * rt * f * g * w
            return (dr * (y[..., 1:] + y[..., :-1]) / 2.0).sum(-1)

        P, vn, norm2 = torch.empty_like(C), C.new_empty((nl, rank)), C.new_empty((nl, rank))
        for n in range(rank):                     # modified Gram-Schmidt in the V metric, all degrees at once
            norm2[:, n] = inner(C[:, n], C[:, n])
            p = C[:, n] / norm2[:, n, None].sqrt()
            for _ in range(2):
                for j in range(n):
                    p = p - (inner(P[:, j], p, V) / vn[:, j])[:, None] * P[:, j]
            P[:, n], vn[:, n] = p, inner(p, p, V)
        if not (torch.isfinite(norm2).all() and (norm2 > 1e-18).all() and torch.isfinite(vn).all()
                and (vn.abs() > 1e-12).all()):
            raise ValueError(f'{self.symbol}: near-null VNA projector (norms {norm2.tolist()}, V norms {vn.tolist()})')
        metric = inner(P[:, :, None], P[:, None, :], V)                     # [nl, rank, rank]
        scale = metric.diagonal(dim1=1, dim2=2).abs().sqrt()
        normalized = metric.abs() / (scale[:, :, None] * scale[:, None, :]).clamp_min(1e-12)
        offdiag = float((normalized - torch.diag_embed(normalized.diagonal(dim1=1, dim2=2))).max())
        self.radial_epsilon = (1.0 / vn).reshape(-1).cpu().numpy()
        self.radial_l = np.repeat(np.arange(nl, dtype=np.int16), rank)
        self.radial_n = np.tile(np.arange(rank, dtype=np.int16), nl)
        self.q_shells = tuple(int(l) for l in self.radial_l)
        self.epsilon_ao = np.repeat(self.radial_epsilon, 2 * self.radial_l.astype(np.int64) + 1)
        e = np.abs(self.radial_epsilon)
        self.metadata = dict(radial_rank=rank, l_buffer=int(st['l_buffer']), lmax=nl - 1,
                             physical_vna_cutoff_bohr=self.vna_cut, projector_cutoff_bohr=self.q_cut,
                             radial_projectors=len(e), angular_projectors=len(self.epsilon_ao),
                             normalized_v_metric_offdiag_max=offdiag, epsilon_abs_min_1_per_ev=float(e.min()),
                             epsilon_abs_max_1_per_ev=float(e.max()),
                             epsilon_abs_dynamic_range=float(e.max() / max(float(e.min()), np.finfo(np.float64).tiny)),
                             seed_kinds=kinds)
        return (V * P).reshape(nl * rank, -1)

    def row(self, array_path, array_sha256, generator):
        return dict(orbital_shells=list(self.shells), orbital_norb=int(sum(2 * l + 1 for l in self.shells)),
                    orbital_cutoff_bohr=self.rc, orbital_sha256=sha256_file(self.orb_path),
                    upf_sha256=sha256_file(self.upf_path), physical_vna_cutoff_bohr=self.vna_cut,
                    vna_cutoff_bohr=self.q_cut, vna_projector_shells=list(self.q_shells),
                    vna_projector_norb=len(self.epsilon_ao), projector_metadata=self.metadata,
                    array_path=str(array_path), array_sha256=array_sha256, generator=generator)


def _sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)


def build(pairs, out_dir, pporb_dir, device='cuda', *, settings=None, reference=None, species_files_map=None,
          chunk_bytes=2 << 30, workers=8, log=print):
    """Factor tables of the ordered pairs 'V|A' into the P23 root ``out_dir`` (created or extended; files are never
    replaced and pairs already there are skipped), plus the species arrays of every species involved.

    ``pporb_dir``: directory or directories with '<S>.upf' and '<S>_*.orb' (``species_files_map`` {S: (orb, upf)}
    overrides); ``reference``: species rows {S: {orbital_sha256, upf_sha256}} the inputs must match.
    Returns (written files, {'V|A': seconds}) with the shared GEMM time split over pairs by entry count.
    """
    t_start = time.perf_counter()
    st = dict(SETTINGS, **(settings or {}))
    dev = torch.device(device)
    keys = [tuple(p.split('|')) if isinstance(p, str) else tuple(p) for p in pairs]
    store = TableStore(out_dir, SCHEMA, length_unit='bohr', factor_energy_unit='eV', epsilon_unit='1/eV',
                       harmonic_convention='deeptb_abacus_real', endpoint_policy='exclude_i0_and_jR',
                       factor_support_semantics=SUPPORT_SEMANTICS, interpolation=st['interpolation'],
                       build_settings=st)
    gid = store.generator('dptb.prior.gf.build_p23', 'dptb.prior.common.sbt', 'dptb.prior.common.pporb')
    todo = list(dict.fromkeys(k for k in keys if not store.has('factor_tables', f'{k[0]}|{k[1]}')))
    symbols = sorted({s for k in keys for s in k})
    files = species_files(symbols, pporb_dir, reference, species_files_map)
    sbt = BondFrameSBT(st['kmax_bohr_inv'], st['n_k'], st['n_mu'], st['n_phi'], device=dev)
    sp, written, seconds = {}, [], {}
    t0 = time.perf_counter()
    for s in symbols:
        sp[s] = Species(s, *files[s], sbt, st, dev)
        if store.has('species', s):
            old = store.manifest['species'][s]
            if old['orbital_sha256'] != sha256_file(files[s][0]) or old['upf_sha256'] != sha256_file(files[s][1]):
                raise ValueError(f'{s}: ORB/UPF differ from the species already in {out_dir}')
            continue
        x = sp[s]
        rel = Path('species') / f'{s}.npz'
        row = store.write_npz(rel, dict(epsilon_ao=x.epsilon_ao, radial_epsilon=x.radial_epsilon,
                                        radial_l=x.radial_l, radial_n=x.radial_n))
        written.append(store.root / rel)
        store.add('species', s, x.row(rel.as_posix(), row['sha256'], gid))
    _sync(dev)
    log(f'P23_SPECIES {len(symbols)} {time.perf_counter() - t0:.3f}s')

    # global function tables: transforms [functions, n_k], angular slots and signed m, per-species offsets
    names = sorted({v for v, _ in todo})
    aos = sorted({a for _, a in todo})
    offq = dict(zip(names, np.cumsum([0] + [len(sp[s].q_layout[0]) for s in names])))
    offa = dict(zip(aos, np.cumsum([0] + [len(sp[s].ao_layout[0]) for s in aos])))
    Tq = torch.cat([sp[s].q[torch.as_tensor(sp[s].q_layout[0], device=dev)] for s in names]) if todo else None
    Ta = torch.cat([sp[s].ao[torch.as_tensor(sp[s].ao_layout[0], device=dev)] for s in aos]) if todo else None
    lq = max((max(sp[s].q_shells) for s in names), default=0)
    la = max((max(sp[s].shells) for s in aos), default=0)
    cpl = sbt.coupling(lq, la) if todo else None                        # [L, slots_q, slots_a]
    if todo:
        cpl = torch.where(cpl.abs() > 1e-13 * (2 * torch.arange(len(cpl), device=dev, dtype=cpl.dtype) + 1)
                          [:, None, None], cpl, torch.zeros_like(cpl))  # the deployed |C^L_ab| > 1e-13 rule

    entries = {}
    for v, a in todo:
        mq, ma = sp[v].q_layout[2], sp[a].ao_layout[2]
        ia, ib = np.nonzero(mq[:, None] == ma[None, :])
        entries[v, a] = (ia, ib, offq[v] + ia, offa[a] + ib, sp[v].q_layout[1][ia], sp[a].ao_layout[1][ib])
    groups = {}
    for v, a in todo:
        groups.setdefault(sp[v].q_cut + sp[a].rc, []).append((v, a))

    pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    pending = []

    def write(v, a, d, values, support, t_compute):
        tw = time.perf_counter()
        rel = Path('factors') / f'{v}__{a}.npz'
        row = store.write_npz(rel, dict(distances=d, values=values,
                                        left_shells=np.asarray(sp[v].q_shells, dtype=np.int16),
                                        right_shells=np.asarray(sp[a].shells, dtype=np.int16),
                                        support_bohr=np.asarray(support, dtype=np.float64)))
        dt = t_compute + time.perf_counter() - tw
        return f'{v}|{a}', rel, dict(path=row['path'], sha256=row['sha256'], support_bohr=float(support),
                                     distance_knots=len(d), shape=list(values.shape), value_dtype='float32',
                                     generator=gid, seconds=dt)

    nk = len(sbt.k)
    for support, members in sorted(groups.items()):
        d = distance_grid(support, st['distance_step_bohr'])
        dt_ = torch.as_tensor(d, device=dev)
        L = max(max(sp[v].q_shells) + max(sp[a].shells) for v, a in members)
        kern = (spherical_jn(L, torch.outer(sbt.k, dt_)) * sbt.factor[None, :, None]).reshape((L + 1) * nk, len(d))
        cut = dt_ >= support - 1e-12
        per = max(1, int(chunk_bytes) // (16 * (L + 1) * nk))           # entries per GEMM chunk
        chunk, size = [], 0
        for i, pair in enumerate(members):
            chunk.append(pair)
            size += len(entries[pair][0])
            if size < per and i + 1 < len(members):
                continue
            _sync(dev)
            tc = time.perf_counter()
            gq, ga, sq, sa = (torch.as_tensor(np.concatenate([entries[p][k] for p in chunk]), device=dev)
                              for k in (2, 3, 4, 5))
            G = (cpl[:L + 1, sq, sa][:, :, None] * (Tq[gq] * Ta[ga])[None])     # [L + 1, E, n_k]
            F = G.transpose(0, 1).reshape(len(gq), -1) @ kern                  # [E, n_d]
            F[:, cut] = 0.0
            _sync(dev)
            share = (time.perf_counter() - tc) / size
            e0 = 0
            for v, a in chunk:
                tp = time.perf_counter()
                ia, ib = entries[v, a][:2]
                out = F.new_zeros((len(d), len(sp[v].epsilon_ao), len(sp[a].ao_layout[0])))
                out[:, torch.as_tensor(ia, device=dev), torch.as_tensor(ib, device=dev)] = F[e0:e0 + len(ia)].T
                values = out.to(torch.float32).cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f'{v}|{a}: non-finite factor table')
                t_own = share * len(ia) + time.perf_counter() - tp
                pending.append(pool.submit(write, v, a, d, values, support, t_own))
                e0 += len(ia)
            chunk, size = [], 0
    for f in pending:
        key, rel, row = f.result()
        store.add('factor_tables', key, row)
        written.append(store.root / rel)
        seconds[key] = row['seconds']
        log(f'P23_BUILT {key} {row["seconds"]:.3f}s')
    pool.shutdown()
    store.flush(active_factor_pairs=sorted(store.manifest.get('factor_tables', {})),
                build_seconds=time.perf_counter() - t_start)
    written.append(store.path)
    return written, seconds

"""Native geometry preparation for the existing P23-onsite/P2-edge recipe.

Only integer neighbourhood construction changes; radial evaluation, units,
projector cutoffs, SOC matrices and output Hermitian symmetrization stay in
the numerical assembly plan.
"""
import numpy as np

from .topology import build_edge_topology, group_rows


def prepare_native(bank, symbols, positions, cell, edges, shifts, pbc,
                   p23_used, *, library=None, max_terms=10_000_000):
    if not isinstance(max_terms, (int, np.integer)) or not 0 < max_terms <= 2**63-1:
        raise ValueError('max_terms must be a positive integer')
    species, codes = np.unique(symbols, return_inverse=True)
    ns, n = len(species), len(symbols)
    sizes = np.array([bank.p2.species[s]['orbital_norb'] for s in symbols], dtype=np.int64)
    cuts = np.array([bank.p2.species[s]['orbital_cutoff_bohr'] for s in symbols])
    query_lists, base_rows, contractions, stats = {}, {}, {}, {}
    delta = positions[edges[1]] - positions[edges[0]] + shifts @ cell
    selected = np.flatnonzero(np.linalg.norm(delta, axis=1) <= cuts[edges[0]]+cuts[edges[1]]+1e-10)
    pair_codes = codes[edges[0, selected]]*ns+codes[edges[1, selected]]
    for pair in np.unique(pair_codes):
        si, sj = divmod(int(pair), ns)
        rows = selected[pair_codes == pair]
        q = np.column_stack((edges[1, rows], edges[0, rows], shifts[rows]))
        b = np.column_stack((n+rows, np.arange(len(rows)), sizes[edges[0, rows]], sizes[edges[1, rows]]))
        for kind in ('p2_base', 'overlap'):
            key = (kind, str(species[si]), str(species[sj]))
            query_lists[key], base_rows[key] = q, b
    block_atoms = np.vstack((np.column_stack((np.arange(n), np.arange(n))), edges.T))
    total_terms = 0
    for kind in ('projector', 'vna'):
        if kind == 'vna' and not p23_used:
            continue
        if kind == 'projector':
            cc = [float(bank.p2.species[s]['projector_max_cutoff_bohr'])
                  if int(bank.p2.species[s]['projector_norb']) else -1.0 for s in symbols]
            if not any(c >= 0 for c in cc):
                stats[kind] = dict(search_s=0., join_s=0., broad_pairs=0, queries=0, terms=0)
                continue
        else:
            cc = [float(bank.p23.species[s]['vna_cutoff_bohr']) for s in symbols]
        topo = build_edge_topology(positions, cell, pbc, cuts, cc, edges, shifts,
                                  mode='projector' if kind == 'projector' else 'onsite_vna',
                                  library=library, max_terms=max(1, max_terms-total_terms))
        q, t = topo['queries'], topo['terms']
        total_terms += len(t)
        if total_terms > max_terms:
            raise ValueError('assembly term budget exceeded')
        stats[kind] = {k: topo[k] for k in ('search_s', 'join_s', 'broad_pairs')}
        stats[kind].update(queries=len(q), terms=len(t))
        pairs = codes[q[:, 1]]*ns+codes[q[:, 0]]
        local = np.empty(len(q), dtype=np.int64)
        for pair, rows in group_rows(pairs):
            sk, sa = divmod(pair, ns)
            local[rows] = np.arange(len(rows))
            query_lists[(kind, str(species[sk]), str(species[sa]))] = q[rows]
        triples = ((codes[block_atoms[t[:, 0], 0]]*ns+codes[block_atoms[t[:, 0], 1]])*ns
                   + codes[q[t[:, 1], 1]])
        for triple, index in group_rows(triples):
            ij, sk = divmod(triple, ns)
            si, sj = divmod(ij, ns)
            rows = t[index]
            rows[:, 1:3] = local[rows[:, 1:3]]
            contractions[(kind, str(species[si]), str(species[sj]), str(species[sk]))] = rows
    return query_lists, base_rows, contractions, stats

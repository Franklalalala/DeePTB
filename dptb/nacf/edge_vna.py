"""Grid-free incremental third-centre VNA with shared tables and batched queries.

This is an explicit new recipe component, not a change to mixed P23/P2 defaults.
It returns scalar AO blocks in eV; for spinors, lift the scalar term onto the
spin diagonal in the caller. No H, H0, density grid or fitted labels are used.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .topology import build_edge_topology, device_array, group_rows


class NACFEdgeVNAPlan(nn.Module):
    """Geometry-bound, possibly batched edge-only VNA correction.

    Each geometry is a dict containing symbols, positions_bohr, cell_bohr,
    edge_index, edge_cell_shift, and optional pbc. Original edge row order is
    preserved within each graph. edge_ptr splits the padded output by graph.
    ``chunk_bytes`` bounds estimated contraction scratch, excluding resident
    tables, all Q factors, topology and output. Inference does not compile.
    """

    def __init__(self, bank, geometries, *, library=None, max_terms=10_000_000,
                 chunk_bytes=32*1024*1024, prune_unused=False):
        super().__init__()
        if not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ValueError('chunk_bytes must be a positive integer')
        if not isinstance(max_terms, (int, np.integer)) or max_terms <= 0:
            raise ValueError('max_terms must be a positive integer')
        if not geometries:
            raise ValueError('at least one geometry is required')
        self.bank = bank
        self.chunk_bytes = chunk_bytes
        self.topology_stats = []
        self.query_specs, self.contraction_specs = [], []
        device, dtype = bank._anchor.device, bank._anchor.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('edge VNA requires float32 or float64')
        itemsize = torch.empty((), dtype=dtype).element_size()

        def reg(name, array, integer=False):
            self.register_buffer(name, device_array(array, dtype=torch.long if integer else dtype, device=device))

        positions, cells, queries, terms, edges, shifts, reverse = [], [], [], [], [], [], []
        all_symbols = []
        edge_ptr = [0]
        n_atoms = n_queries = n_edges = n_terms = 0
        for graph, g in enumerate(geometries):
            symbols = tuple(g['symbols'])
            used, missing = bank.p23_composition(symbols)
            if not used:
                raise KeyError(f'edge VNA requires complete P23 coverage: {missing[:16]}')
            pos, cell = np.asarray(g['positions_bohr']), np.asarray(g['cell_bohr'])
            if pos.shape != (len(symbols), 3):
                raise ValueError('symbols and geometry disagree')
            topo = build_edge_topology(pos, cell, g.get('pbc', (True,True,True)),
                    [bank.p2.species[s]['orbital_cutoff_bohr'] for s in symbols],
                    [bank.p23.species[s]['vna_cutoff_bohr'] for s in symbols],
                    g['edge_index'], g['edge_cell_shift'], library=library,
                    max_terms=max(1, max_terms-n_terms))
            q, t = topo['queries'], topo['terms']
            if n_terms+len(t) > max_terms:
                raise ValueError('third-centre batch term budget exceeded')
            raw_query_count = len(q)
            if prune_unused or not len(t):
                # Linear marking avoids sorting the repeated term references.
                needed = np.zeros(len(q), dtype=bool)
                needed[t[:, 1:3]] = True
                active = np.flatnonzero(needed)
                remap = np.full(len(q), -1, dtype=np.int64)
                remap[active] = np.arange(len(active))
                t[:, 1:3] = remap[t[:, 1:3]]
                q = q[active].copy()
            self.topology_stats.append({k:topo[k] for k in ('broad_pairs','search_s','join_s')} |
                                       {'queries_before_pruning':raw_query_count,'queries':len(q),'terms':len(t)})
            q[:, :2] += n_atoms
            q = np.column_stack((q, np.full(len(q), graph, dtype=np.int64)))
            t[:, 0] += n_edges
            t[:, 1:3] += n_queries
            positions.append(pos); cells.append(cell); queries.append(q); terms.append(t)
            edges.append(np.asarray(g['edge_index'], dtype=np.int64).T+n_atoms)
            shifts.append(np.asarray(g['edge_cell_shift'], dtype=np.int64))
            reverse.append(topo['reverse']+n_edges)
            n_atoms += len(symbols); n_queries += len(q); n_edges += len(topo['reverse']); n_terms += len(t)
            edge_ptr.append(n_edges); all_symbols.extend(symbols)
        all_symbols = np.asarray(all_symbols)
        self.symbols = tuple(all_symbols.tolist())
        self.spinor_input = False  # VNA is scalar even when the shared bank has SOC projectors.
        species, species_codes = np.unique(all_symbols, return_inverse=True)
        self.width = max(int(bank.p2.species[s]['orbital_norb']) for s in species)
        self.nedges, self.nqueries, self.nterms = n_edges, n_queries, n_terms
        self.edge_slices = tuple(zip(edge_ptr[:-1], edge_ptr[1:]))
        reg('positions', np.concatenate(positions)); reg('cells', np.stack(cells))
        edge_array, q, t = np.concatenate(edges), np.concatenate(queries), np.concatenate(terms)
        reg('edge_index', edge_array.T, True); reg('edge_cell_shift', np.concatenate(shifts), True)
        reg('reverse', np.concatenate(reverse), True); reg('edge_ptr', edge_ptr, True)
        reg('representative', np.arange(n_edges) < np.concatenate(reverse), True)
        nspecies = len(species)
        # Dense species IDs permit collision-free integer keys. Avoid sorting
        # hundreds of thousands of multi-column records in Python/NumPy.
        q_pairs = species_codes[q[:,1]]*nspecies+species_codes[q[:,0]]
        local = np.empty(n_queries, dtype=np.int64)
        pair_ids = {}
        for number, (pair, rows) in enumerate(group_rows(q_pairs)):
            centre, ao = divmod(pair, nspecies)
            local[rows] = np.arange(len(rows))
            key = bank.table('vna', str(species[centre]), str(species[ao]))
            reg(f'query_{number}', q[rows], True)
            self.query_specs.append((number, key))
            pair_ids[(int(centre),int(ao))] = number
        triple_keys = ((species_codes[edge_array[t[:,0],0]]*nspecies+
                       species_codes[edge_array[t[:,0],1]])*nspecies+species_codes[q[t[:,1],1]])
        for number, (triple, index) in enumerate(group_rows(triple_keys)):
            pair, sk = divmod(triple, nspecies)
            si, sj = divmod(pair, nspecies)
            rows = t[index]
            rows[:, 1:3] = local[rows[:, 1:3]]
            epsilon = np.asarray(bank.p23.epsilon(str(species[sk])), dtype=np.float64)
            ni, nj = (int(bank.p2.species[str(species[s])]['orbital_norb']) for s in (si,sj))
            # a, b, weighted a and their product; allow one complete term even
            # when the requested budget is smaller than that indivisible unit.
            bytes_per_term = itemsize*(len(epsilon)*(2*ni+nj)+ni*nj)
            chunk = max(1, min(2048, chunk_bytes//max(1,bytes_per_term)))
            reg(f'terms_{number}', rows, True); reg(f'epsilon_{number}', epsilon)
            self.contraction_specs.append((number,pair_ids[(int(sk),int(si))],
                                            pair_ids[(int(sk),int(sj))],ni,nj,chunk))

    def forward(self):
        values = {}
        fused = hasattr(self, 'fused_radial_specs')
        if fused:
            from .fusion import evaluate_plan
            values = evaluate_plan(self)
        for number, key in ([] if fused else self.query_specs):
            q = getattr(self, f'query_{number}')
            translation = torch.einsum('qi,qij->qj',q[:,2:5].to(self.positions.dtype),self.cells[q[:,5]])
            delta = self.positions[q[:,0]]-self.positions[q[:,1]]+translation
            values[number] = self.bank.tables[key](delta)
        output = self.positions.new_zeros((self.nedges,self.width,self.width))
        for number, left, right, ni, nj, chunk in self.contraction_specs:
            rows = getattr(self,f'terms_{number}')
            epsilon = getattr(self,f'epsilon_{number}')
            if getattr(self, 'fused_contraction', False):
                from .fusion import contract_add
                contract_add(values[left], epsilon, values[right], rows, output)
                continue
            for start in range(0,len(rows),chunk):
                part = rows[start:start+chunk]
                a,b = values[left][part[:,1]],values[right][part[:,2]]
                contribution = (a*epsilon[None,:,None]).transpose(-1,-2) @ b
                output[:,:ni,:nj].index_add_(0,part[:,0],contribution)
        output = torch.where(self.representative.bool()[:,None,None],output,
                             output[self.reverse].transpose(-1,-2))
        return {'edge_vna_ao_ev':output,'edge_ptr':self.edge_ptr,
                'edge_index':self.edge_index,'edge_cell_shift':self.edge_cell_shift}

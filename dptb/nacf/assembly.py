"""NACF numerical assembly on a GPU, with an explicit CPU topology plan.

The plan binds a directed graph and its periodic projector neighbours to a
particular geometry. Rebuild it for a new geometry (no implicit neighbour-list
reuse). Its forward evaluates all radial functions, rotates the AO factors and
contracts P2/P23 on the device. It returns ABACUS-gauge AO blocks; conversion to
DeePTB RME is deliberately a separate operation with a basis contract.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np
import torch
from torch import nn

from dptb.data.interfaces.p2_batch import VectorizedNearbyImageEnumerator
from .radial import TorchRadialBlockTable


class NACFTableBank(nn.Module):
    """Caller-owned GPU table cache shared by multiple structure plans.

    Stores must already have passed their source/manifest gates. Missing overlap
    arrays are an error: overlap cannot be substituted by the identity matrix.
    Tables are loaded/checksummed once during preparation, never in forward.
    """

    def __init__(self, p2_store, p23_store, *, overlap_store=None, device='cuda', dtype=torch.float64, backend='auto'):
        super().__init__()
        self.backend = backend
        self.p2_manifest_sha256 = (hashlib.sha256((p2_store.root / 'manifest.json').read_bytes()).hexdigest()
                                   if hasattr(p2_store, 'root') else None)
        if hasattr(p23_store, 'manifest'):
            expected = p23_store.manifest.get('base_p2_table_manifest_sha256')
            if expected != self.p2_manifest_sha256:
                raise ValueError('P23 manifest does not bind the supplied P2 table')
        self.p2 = p2_store
        self.p23 = p23_store
        self.overlap = p2_store if overlap_store is None else overlap_store
        if overlap_store is not None:
            if overlap_store.manifest['source_p2_manifest_sha256'] != self.p2_manifest_sha256:
                raise ValueError('overlap table does not bind the supplied P2 manifest')
        self.tables = nn.ModuleDict()
        self.register_buffer('_anchor', torch.empty(0, device=device, dtype=dtype))

    def table(self, kind, left, right):
        key = f'{kind}_{left}_{right}'
        if key not in self.tables:
            if kind == 'vna':
                source = self.p23.factor(left, right)
            elif kind == 'projector':
                source = self.p2.projector(left, right)
            elif kind == 'overlap':
                source = self.overlap.base_component(left, right, kind)
            else:
                source = self.p2.base_component(left, right, kind)
            self.tables[key] = TorchRadialBlockTable(source, device=self._anchor.device, dtype=self._anchor.dtype, backend=self.backend)
        return key

    def prepare(self, symbols, positions_bohr, cell_bohr, edge_index, edge_cell_shift, *, pbc=(True, True, True)):
        return NACFAssemblyPlan(self, symbols, positions_bohr, cell_bohr, edge_index, edge_cell_shift, pbc=pbc)


class NACFAssemblyPlan(nn.Module):
    """Geometry-bound plan; use ``forward()`` for repeated, identical geometry.

    No training labels, H0, or cached structure-dependent priors are accepted.
    ``node_p23_ao_ev``, ``edge_p2_ao_ev`` and dimensionless overlap share exactly
    the requested graph row order. P2 remains in Ry internally and is converted
    once; the VNA factor contraction is already in eV.
    """

    def __init__(self, bank, symbols, positions, cell, edges, shifts, *, pbc):
        super().__init__()
        self.bank = bank
        symbols = tuple(symbols)
        positions = np.asarray(positions, dtype=np.float64)
        cell = np.asarray(cell, dtype=np.float64)
        edges_raw, shifts_raw = np.asarray(edges), np.asarray(shifts)
        if positions.shape != (len(symbols), 3) or cell.shape != (3, 3):
            raise ValueError('invalid position or cell shape')
        if not np.isfinite(positions).all() or not np.isfinite(cell).all():
            raise ValueError('geometry must be finite')
        if edges_raw.ndim != 2 or edges_raw.shape[0] != 2 or shifts_raw.shape != (edges_raw.shape[1], 3):
            raise ValueError('graph requires [2,E] indices and [E,3] shifts')
        edges, shifts = edges_raw.astype(np.int64), shifts_raw.astype(np.int64)
        if not np.array_equal(edges_raw, edges) or not np.array_equal(shifts_raw, shifts):
            raise ValueError('graph indices and cell shifts must be exact integers')
        if edges.size and (edges.min() < 0 or edges.max() >= len(symbols)):
            raise ValueError('edge atom index out of bounds')
        pbc = np.asarray(pbc, dtype=bool)
        if pbc.shape != (3,) or np.any(shifts[:, ~pbc]):
            raise ValueError('cell shifts are inconsistent with pbc')
        if np.any(pbc & (np.linalg.norm(cell, axis=1) == 0)):
            raise ValueError('periodic lattice vectors must be nonzero')
        if not symbols:
            raise ValueError('empty structures are not supported')
        keys = [(int(i), int(j), *map(int, s)) for (i, j), s in zip(edges.T, shifts)]
        rows = {key: row for row, key in enumerate(keys)}
        if len(rows) != len(keys) or any(i == j and (rx, ry, rz) == (0, 0, 0) for i, j, rx, ry, rz in keys):
            raise ValueError('duplicate edges or onsite self-edge')
        try:
            reverse = [rows[(j, i, -rx, -ry, -rz)] for i, j, rx, ry, rz in keys]
        except KeyError as exc:
            raise ValueError('every edge must have its reverse') from exc
        self.natoms = len(symbols)
        self.symbols = symbols
        self.nedges = len(keys)
        self.width = max(int(bank.p2.species[s]['orbital_norb']) for s in symbols)
        self.node_sizes = tuple(int(bank.p2.species[s]['orbital_norb']) for s in symbols)
        device, dtype = bank._anchor.device, bank._anchor.dtype

        def reg(name, data, integer=False):
            self.register_buffer(name, torch.as_tensor(np.array(data, copy=True), dtype=torch.long if integer else dtype, device=device))

        reg('positions', positions)
        reg('cell', cell)
        reg('reverse', reverse, True)
        reg('edge_index', edges, True)
        reg('edge_cell_shift', shifts, True)
        all_keys = [(i, i, 0, 0, 0) for i in range(self.natoms)] + keys
        nblocks = len(all_keys)
        base = np.zeros((nblocks, self.width, self.width))
        overlap = np.zeros_like(base)
        for i, s in enumerate(symbols):
            n = self.node_sizes[i]
            base[i, :n, :n] = bank.p2.onsite_component(s, 'p2_base')
            overlap[i, :n, :n] = bank.overlap.onsite_component(s, 'overlap')
        reg('onsite_base', base)
        reg('onsite_overlap', overlap)
        query_lists, query_maps = defaultdict(list), defaultdict(dict)
        base_rows, contractions = defaultdict(list), defaultdict(list)

        def query(kind, ao, centre, shift):
            pair = (kind, symbols[centre], symbols[ao])
            key = (ao, centre, *map(int, shift))
            if key not in query_maps[pair]:
                query_maps[pair][key] = len(query_lists[pair])
                query_lists[pair].append(key)
            return pair, query_maps[pair][key]

        # Keep graph construction explicit. All subsequent queries refer to atom
        # indices and integer translations, rather than rounded vector hashes.
        from ase.cell import Cell
        enumeration_cell = Cell(cell).complete().array
        enumerator = VectorizedNearbyImageEnumerator(enumeration_cell) if np.any(pbc) else None
        for block_id, (i, j, rx, ry, rz) in enumerate(all_keys):
            shift = np.array([rx, ry, rz])
            ci, cj = positions[i], positions[j] + shift @ cell
            si, sj = symbols[i], symbols[j]
            ni, nj = self.node_sizes[i], self.node_sizes[j]
            cutoff_i = float(bank.p2.species[si]['orbital_cutoff_bohr'])
            cutoff_j = float(bank.p2.species[sj]['orbital_cutoff_bohr'])
            distance = np.linalg.norm(cj - ci)
            if block_id >= self.natoms and distance <= cutoff_i + cutoff_j + 1e-10:
                for kind in ('p2_base', 'overlap'):
                    # Base table orientation is left AO=i, right AO=j.
                    pair = (kind, si, sj)
                    row = len(query_lists[pair])
                    query_lists[pair].append((j, i, rx, ry, rz))
                    base_rows[pair].append((block_id, row, ni, nj))
            for k, sk in enumerate(symbols):
                for kind in ('projector', 'vna'):
                    if kind == 'vna' and block_id >= self.natoms:
                        continue  # NACF needs P23 onsite only.
                    meta = bank.p2.species[sk] if kind == 'projector' else bank.p23.species[sk]
                    if kind == 'projector' and int(meta['projector_norb']) == 0:
                        continue
                    cutoff = float(meta['projector_max_cutoff_bohr'] if kind == 'projector' else meta['vna_cutoff_bohr'])
                    if enumerator is None:
                        translations, centres = np.zeros((1, 3), dtype=int), positions[k:k + 1]
                    else:
                        translations, centres = enumerator.query_arrays(positions[k], (ci + cj) * .5, cutoff + max(cutoff_i, cutoff_j) + distance * .5)
                    di, dj = ci - centres, cj - centres
                    if kind == 'projector':
                        active = (np.linalg.norm(di, axis=1) <= cutoff + cutoff_i + 1e-12) & (np.linalg.norm(dj, axis=1) <= cutoff + cutoff_j + 1e-12)
                    else:
                        active = (np.linalg.norm(di, axis=1) < cutoff + cutoff_i - 1e-12) & (np.linalg.norm(dj, axis=1) < cutoff + cutoff_j - 1e-12)
                        active &= ~((k == i) & np.all(translations == 0, axis=1))
                    active &= np.all(translations[:, ~pbc] == 0, axis=1)
                    for t in translations[active]:
                        left_pair, left_row = query(kind, i, k, -t)
                        right_pair, right_row = query(kind, j, k, shift - t)
                        contractions[(kind, si, sj, sk)].append((block_id, left_row, right_row))
        self.query_specs = []
        self.pair_ids = {}
        for number, (pair, raw) in enumerate(query_lists.items()):
            kind, left, right = pair
            key = bank.table(kind, left, right)
            arr = np.asarray(raw, dtype=np.int64)
            reg(f'query_{number}', arr, True)
            self.pair_ids[pair] = number
            self.query_specs.append((number, pair, key))
            if kind == 'projector':
                meta = bank.p2.species[left]
                row_cutoffs = np.repeat(meta['projector_cutoffs_bohr'], [2 * int(l) + 1 for l in meta['projector_shells']])
                reg(f'cutoffs_{number}', row_cutoffs + float(bank.p2.species[right]['orbital_cutoff_bohr']) + 1e-12)
        self.base_specs = []
        for number, (pair, raw) in enumerate(base_rows.items()):
            reg(f'base_rows_{number}', np.asarray(raw)[:, :2], True)
            self.base_specs.append((number, pair, raw[0][2], raw[0][3]))
        self.contraction_specs = []
        for number, ((kind, si, sj, sk), raw) in enumerate(contractions.items()):
            reg(f'terms_{number}', raw, True)
            matrix = bank.p2.d_eff(sk) if kind == 'projector' else np.diag(bank.p23.epsilon(sk))
            reg(f'matrix_{number}', matrix)
            self.contraction_specs.append((number, kind, self.pair_ids[(kind, sk, si)], self.pair_ids[(kind, sk, sj)], int(bank.p2.species[si]['orbital_norb']), int(bank.p2.species[sj]['orbital_norb'])))

    def forward(self):
        values = {}
        for number, (kind, _, _), table_key in self.query_specs:
            query = getattr(self, f'query_{number}')
            if self.cell.ndim == 3:
                translation = torch.einsum('ei,eij->ej', query[:, 2:5].to(self.cell.dtype), self.cell[query[:, 5]])
            else:
                translation = query[:, 2:5].to(self.cell.dtype) @ self.cell
            delta = self.positions[query[:, 0]] - self.positions[query[:, 1]] + translation
            block = self.bank.tables[table_key](delta)
            if kind == 'projector':
                active = torch.linalg.vector_norm(delta, dim=-1)[:, None] <= getattr(self, f'cutoffs_{number}')[None, :]
                block = block * active[:, :, None]
            values[number] = block
        p2, overlap = self.onsite_base.clone(), self.onsite_overlap.clone()
        for number, pair, ni, nj in self.base_specs:
            rows = getattr(self, f'base_rows_{number}')
            target = p2 if pair[0] == 'p2_base' else overlap
            target[rows[:, 0], :ni, :nj] = values[self.pair_ids[pair]][rows[:, 1]]
        vna = torch.zeros_like(p2)
        for number, kind, left, right, ni, nj in self.contraction_specs:
            rows = getattr(self, f'terms_{number}')
            target = p2 if kind == 'projector' else vna
            for start in range(0, rows.shape[0], 2048):
                part = rows[start:start + 2048]
                a, b = values[left][part[:, 1]], values[right][part[:, 2]]
                contribution = a.transpose(-1, -2) @ getattr(self, f'matrix_{number}') @ b
                target[:, :ni, :nj].index_add_(0, part[:, 0], contribution)
        # This is the exact non-SOC projection used by materialization: onsite
        # symmetric, reverse edges averaged before conversion to model gauge.
        def hermitian(blocks):
            node, edge = blocks[:self.natoms], blocks[self.natoms:]
            return (node + node.transpose(-1, -2)) * .5, (edge + edge[self.reverse].transpose(-1, -2)) * .5
        node_p2, edge_p2 = hermitian(p2)
        node_vna = (vna[:self.natoms] + vna[:self.natoms].transpose(-1, -2)) * .5
        node_s, edge_s = hermitian(overlap)
        # Match the historical materializer's constant exactly; changing to the
        # newer CODATA constant here would move the checkpoint's physical prior.
        ry_to_ev = 13.605698
        return {'node_p23_ao_ev': node_p2 * ry_to_ev + node_vna,
                'edge_p2_ao_ev': edge_p2 * ry_to_ev,
                'node_overlap_ao': node_s, 'edge_overlap_ao': edge_s,
                'edge_index': self.edge_index, 'edge_cell_shift': self.edge_cell_shift}


class NACFBatchAssemblyPlan(NACFAssemblyPlan):
    """Merge structure-local topology plans into one set of GPU table queries.

    Atoms, output blocks, radial query rows, and projector terms have distinct
    offset maps. Integer image translations retain their own structure's cell;
    no physical interactions can cross the batch boundary.
    """

    def __init__(self, plans):
        nn.Module.__init__(self)
        if not plans:
            raise ValueError('empty plan batch')
        self.bank = plans[0].bank
        if any(p.bank is not self.bank for p in plans):
            raise ValueError('batch plans must share a table bank')
        self.natoms = sum(p.natoms for p in plans)
        self.nedges = sum(p.nedges for p in plans)
        self.width = max(p.width for p in plans)
        self.symbols = tuple(s for p in plans for s in p.symbols)
        self.node_sizes = tuple(n for p in plans for n in p.node_sizes)
        self.register_buffer('positions', torch.cat([p.positions for p in plans]))
        self.register_buffer('cell', torch.stack([p.cell for p in plans]))
        query_parts, base_parts, term_parts = defaultdict(list), defaultdict(list), defaultdict(list)
        query_offsets, counts = {}, defaultdict(int)
        cutoffs, matrices = {}, {}
        edges, shifts, reverses, bases, overlaps = [], [], [], [], []
        node_offset = edge_offset = 0
        for graph, plan in enumerate(plans):
            def map_blocks(ids):
                return torch.where(ids < plan.natoms, ids + node_offset,
                                   ids - plan.natoms + self.natoms + edge_offset)
            edges.append(plan.edge_index + node_offset)
            shifts.append(plan.edge_cell_shift)
            reverses.append(plan.reverse + edge_offset)
            ids = torch.arange(plan.natoms + plan.nedges, device=plan.positions.device)
            mapped = map_blocks(ids)
            bases.append((mapped, plan.onsite_base))
            overlaps.append((mapped, plan.onsite_overlap))
            pairs = {}
            for number, pair, key in plan.query_specs:
                pairs[number] = pair
                query = getattr(plan, f'query_{number}').clone()
                query[:, :2] += node_offset
                query = torch.cat([query, torch.full_like(query[:, :1], graph)], dim=1)
                query_offsets[(graph, pair)] = counts[pair]
                counts[pair] += query.shape[0]
                query_parts[pair].append((query, key))
                if pair[0] == 'projector':
                    cutoffs[pair] = getattr(plan, f'cutoffs_{number}')
            for number, pair, ni, nj in plan.base_specs:
                rows = getattr(plan, f'base_rows_{number}').clone()
                rows[:, 0] = map_blocks(rows[:, 0])
                rows[:, 1] += query_offsets[(graph, pair)]
                base_parts[(pair, ni, nj)].append(rows)
            for number, kind, left, right, ni, nj in plan.contraction_specs:
                lp, rp = pairs[left], pairs[right]
                rows = getattr(plan, f'terms_{number}').clone()
                rows[:, 0] = map_blocks(rows[:, 0])
                rows[:, 1] += query_offsets[(graph, lp)]
                rows[:, 2] += query_offsets[(graph, rp)]
                key = (kind, lp, rp, ni, nj)
                term_parts[key].append(rows)
                matrices[key] = getattr(plan, f'matrix_{number}')
            node_offset += plan.natoms
            edge_offset += plan.nedges
        self.register_buffer('edge_index', torch.cat(edges, dim=1))
        self.register_buffer('edge_cell_shift', torch.cat(shifts))
        self.register_buffer('reverse', torch.cat(reverses))
        for name, parts in (('onsite_base', bases), ('onsite_overlap', overlaps)):
            array = self.positions.new_zeros((self.natoms + self.nedges, self.width, self.width))
            for rows, block in parts:
                array[rows, :block.shape[1], :block.shape[2]] = block
            self.register_buffer(name, array)
        self.query_specs, self.pair_ids = [], {}
        for number, (pair, parts) in enumerate(query_parts.items()):
            self.register_buffer(f'query_{number}', torch.cat([p[0] for p in parts]))
            self.query_specs.append((number, pair, parts[0][1]))
            self.pair_ids[pair] = number
            if pair in cutoffs:
                self.register_buffer(f'cutoffs_{number}', cutoffs[pair])
        self.base_specs = []
        for number, ((pair, ni, nj), parts) in enumerate(base_parts.items()):
            self.register_buffer(f'base_rows_{number}', torch.cat(parts))
            self.base_specs.append((number, pair, ni, nj))
        self.contraction_specs = []
        for number, (key, parts) in enumerate(term_parts.items()):
            kind, left, right, ni, nj = key
            self.register_buffer(f'terms_{number}', torch.cat(parts))
            self.register_buffer(f'matrix_{number}', matrices[key])
            self.contraction_specs.append((number, kind, self.pair_ids[left], self.pair_ids[right], ni, nj))


class NACFFeaturePlan(nn.Module):
    """Fuse the ABACUS signed permutation and RME packing into GPU gathers.

    Mapping is compiled once from the same ``OrbitalMapper`` used by the
    checkpoint. The hot path has no orbital loops or host synchronizations.
    Outputs are raw, triangular non-SOC RME, as expected by prior_2b input.
    """

    def __init__(self, assembly: NACFAssemblyPlan, idp, *, output_dtype=torch.float32):
        super().__init__()
        from dptb.data.interfaces.blockwise_tensor import ensure_spatial_block_mapper, onsite_feature_slices, edge_feature_slices
        from dptb.utils.constants import ABACUS2DeePTB, anglrMId
        from scipy.linalg import block_diag
        ensure_spatial_block_mapper(idp)
        if getattr(idp, 'has_soc', False):
            raise ValueError('NACF GPU features support non-SOC only')
        idp.get_orbital_maps()
        idp.get_orbpair_maps()
        self.assembly = assembly
        self.output_dtype = output_dtype
        width, nfeatures = assembly.width, int(idp.reduced_matrix_element)
        permutations = {}
        for symbol in set(assembly.symbols):
            shells = tuple(map(int, assembly.bank.p2.species[symbol]['orbital_shells']))
            model_shells = tuple(anglrMId[orb[-1]] for orb in idp.basis[symbol])
            if shells != model_shells:
                raise ValueError(f'table and checkpoint AO shells disagree for {symbol}')
            transform = block_diag(*(ABACUS2DeePTB[l] for l in shells))
            if not np.all(np.count_nonzero(transform, axis=1) == 1) or not np.all(np.count_nonzero(transform, axis=0) == 1):
                raise ValueError('AO gauge is not a signed permutation')
            index = np.argmax(np.abs(transform), axis=1)
            permutations[symbol] = index, transform[np.arange(len(index)), index]
        edges = assembly.edge_index.detach().cpu().numpy()
        specs = [('node', [(s, s) for s in assembly.symbols]),
                 ('edge', [(assembly.symbols[i], assembly.symbols[j]) for i, j in edges.T])]
        for name, pairs in specs:
            indices = np.zeros((len(pairs), nfeatures), dtype=np.int64)
            signs = np.zeros((len(pairs), nfeatures))
            for row_id, (left, right) in enumerate(pairs):
                li, ls = permutations[left]
                ri, rs = permutations[right]
                slices = onsite_feature_slices(idp, left) if name == 'node' else edge_feature_slices(idp, left, right)
                for row, col, feature in slices:
                    indices[row_id, feature] = (li[row, None] * width + ri[None, col]).ravel()
                    signs[row_id, feature] = (ls[row, None] * rs[None, col]).ravel()
            self.register_buffer(f'{name}_indices', torch.as_tensor(indices, device=assembly.positions.device))
            self.register_buffer(f'{name}_signs', torch.as_tensor(signs, device=assembly.positions.device, dtype=assembly.positions.dtype))

    def pack(self, node, edge):
        nrme = node.flatten(1).gather(1, self.node_indices) * self.node_signs
        erme = edge.flatten(1).gather(1, self.edge_indices) * self.edge_signs
        return nrme.to(self.output_dtype), erme.to(self.output_dtype)

    def forward(self):
        blocks = self.assembly()
        node_p, edge_p = self.pack(blocks['node_p23_ao_ev'], blocks['edge_p2_ao_ev'])
        node_s, edge_s = self.pack(blocks['node_overlap_ao'], blocks['edge_overlap_ao'])
        return {'node_p23': node_p, 'edge_p2': edge_p,
                'node_overlap': node_s, 'edge_overlap': edge_s}

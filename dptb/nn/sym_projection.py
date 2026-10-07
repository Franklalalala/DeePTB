"""Differentiable crystal Reynolds averaging in packed real AO-product space.

The provider is replaceable: it returns *source -> destination* atom/edge
permutations, edge transpose flags and Cartesian orbital rotations. Geometry
is discrete metadata (no force/geometry derivatives). The default provider
matches the diagnostic GroupAction's input-cell group and invariant metric;
it does not claim primitive/supercell independence or global O(3) covariance.
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol
import math
import re

import numpy as np
import torch

from dptb.data import _keys


@dataclass
class CrystalGroup:
    rotations: np.ndarray                 # [G,3,3], column-vector convention
    atom_permutations: np.ndarray         # [G,N], source -> destination
    edge_permutations: np.ndarray         # [G,E], source -> destination
    edge_reversed: np.ndarray             # [G,E], transpose AFTER rotation
    mapping_error: float = 0.0
    metric_correction: float = 0.0


class GroupProvider(Protocol):
    def __call__(self, positions, cell, species, edge_index, edge_shift,
                 symprec: float) -> CrystalGroup: ...


class SpglibGroupProvider:
    """Input-cell GroupAction convention, without importing evaluation scripts."""
    def __call__(self, positions, cell, species, edge_index, edge_shift, symprec):
        import spglib

        positions = np.asarray(positions, dtype=np.float64)
        cell = np.asarray(cell, dtype=np.float64)
        frac = positions @ np.linalg.inv(cell)
        ds = spglib.get_symmetry_dataset((cell, frac, species), symprec=symprec)
        if ds is None:
            raise ValueError("spglib could not determine the crystal group")
        w, translations = np.asarray(ds.rotations), np.asarray(ds.translations)
        b = cell.T
        metric = b.T @ b
        invariant = sum(x.T @ metric @ x for x in w) / len(w)

        def power(x, exponent):
            values, vectors = np.linalg.eigh(x)
            return (vectors * values ** exponent) @ vectors.T

        bs = b @ power(metric, -.5) @ power(invariant, .5)
        rotations = np.einsum('ab,gbc,cd->gad', bs, w, np.linalg.inv(bs))
        permutations, shifts, errors = [], [], []
        for rotation, translation in zip(w, translations):
            delta = (frac @ rotation.T + translation)[:, None] - frac[None]
            shift = np.rint(delta)
            distance = np.linalg.norm((delta - shift) @ cell, axis=-1)
            distance[species[:, None] != species[None]] = np.inf
            perm = distance.argmin(axis=1)
            error = distance[np.arange(len(species)), perm]
            if len(set(perm.tolist())) != len(species) or error.max() > max(2 * symprec, 1e-7):
                raise ValueError("invalid symmetry atom permutation")
            permutations.append(perm)
            shifts.append(shift[np.arange(len(species)), perm].astype(np.int64))
            errors.extend(error)
        edge_index = np.asarray(edge_index, dtype=np.int64)
        edge_shift = np.asarray(edge_shift, dtype=np.int64)
        n_edges = edge_index.shape[1]
        keys = _edge_keys(edge_index[0], edge_index[1], edge_shift, len(species))
        order = np.argsort(keys, kind='stable')
        sorted_keys = keys[order]
        if n_edges and np.any(sorted_keys[1:] == sorted_keys[:-1]):
            raise ValueError("duplicate stored directed edge")

        def find(query):
            pos = np.searchsorted(sorted_keys, query)
            pos_c = np.minimum(pos, max(n_edges - 1, 0))
            hit = (pos < n_edges) & (sorted_keys[pos_c] == query) if n_edges else np.zeros(len(query), bool)
            return order[pos_c], hit

        edge_maps, reverse = [], []
        for rotation, perm, shift in zip(w, permutations, shifts):
            mapped_shift = (edge_shift @ rotation.T + shift[edge_index[1]]
                            - shift[edge_index[0]])
            pi, pj = perm[edge_index[0]], perm[edge_index[1]]
            dest, hit = find(_edge_keys(pi, pj, mapped_shift, len(species)))
            flipped = ~hit
            if flipped.any():
                dest_f, hit_f = find(_edge_keys(pj[flipped], pi[flipped], -mapped_shift[flipped], len(species)))
                if not hit_f.all():
                    raise ValueError("stored graph is not symmetry-closed")
                dest[flipped] = dest_f
            if len(np.unique(dest)) != len(dest):
                raise ValueError("symmetry edge map is not bijective")
            edge_maps.append(dest)
            reverse.append(flipped)
        return CrystalGroup(rotations, np.asarray(permutations, dtype=np.int64),
                            np.asarray(edge_maps, dtype=np.int64).reshape(len(w), n_edges),
                            np.asarray(reverse, dtype=bool).reshape(len(w), n_edges),
                            float(max(errors)), float(np.linalg.norm(bs - b) / np.linalg.norm(b)))


def _edge_keys(i, j, shift, n_atoms):
    """Exact int64 key of a directed stored edge (i, j, R): (i*N + j) < 2**28 and three 8-bit cell shifts, < 2**52."""
    shift = np.asarray(shift, dtype=np.int64).reshape(-1, 3)
    if shift.size and (shift.min() < -128 or shift.max() >= 128):
        raise ValueError("edge cell shift outside the packed key range [-128, 128)")
    if n_atoms >= 2 ** 14:
        raise ValueError("too many atoms for the packed edge key")
    s = shift + 128
    return ((((np.asarray(i, dtype=np.int64) * n_atoms + np.asarray(j, dtype=np.int64)) * 256
              + s[:, 0]) * 256 + s[:, 1]) * 256 + s[:, 2])


def _angular(shell):
    return 'spdfgh'.index(re.search('[a-z]', shell).group())


def _orbital_rotations(rotations, angular):
    """Float64 e3nn convention, including float64 generator construction.

    e3nn's D_from_matrix uses default-dtype generators even for double input.
    Construct those constants explicitly here without changing global dtype.
    """
    from e3nn import o3
    from e3nn.o3._wigner import change_basis_real_to_complex

    q = torch.tensor([[0., 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float64)
    r = q @ torch.as_tensor(rotations, dtype=torch.float64) @ q.T
    det = torch.linalg.det(r).sign()
    angles = o3.matrix_to_angles(r * det[:, None, None])
    result = {}
    for l in angular:
        m = torch.arange(-l, l, dtype=torch.float64)
        raising = torch.diag(-torch.sqrt(l * (l + 1) - m * (m + 1)), diagonal=-1)
        lowering = torch.diag(torch.sqrt(l * (l + 1) - m * (m + 1)), diagonal=1)
        generators = torch.stack([
            .5 * (raising + lowering),
            torch.diag(1j * torch.arange(-l, l + 1, dtype=torch.float64)),
            -.5j * (raising - lowering),
        ])
        basis = change_basis_real_to_complex(l, dtype=torch.float64)
        generators = (basis.conj().T @ generators @ basis).real
        a, b, c = [(angle[:, None, None] % (2 * math.pi)) for angle in angles]
        d = (torch.matrix_exp(a * generators[1]) @ torch.matrix_exp(b * generators[0])
             @ torch.matrix_exp(c * generators[1]))
        result[l] = d * det[:, None, None] ** l
    return result


class _Action:
    def __init__(self, group, blocks, transpose, device, dtype):
        self.group = group
        self.node_maps = torch.as_tensor(group.atom_permutations, device=device)
        self.edge_maps = torch.as_tensor(group.edge_permutations, device=device)
        self.reverse = torch.as_tensor(group.edge_reversed, device=device)
        self.transpose = transpose.to(device)
        self.blocks = [(a, b, ix.to(device)) for a, b, ix in blocks]
        ds = _orbital_rotations(group.rotations, sorted({l for a, b, _ in blocks for l in (a, b)}))
        self.rotations = {l: d.to(device=device, dtype=dtype) for l, d in ds.items()}
        # Gather form of act(): destination row d under operation g comes from source row inv_g[d]; for edges the
        # transpose flag belongs to that source edge.  Adjoint: source row s gathers destination row maps_g[s].
        inv_node = np.argsort(np.asarray(group.atom_permutations), axis=1, kind='stable')
        inv_edge = np.argsort(np.asarray(group.edge_permutations), axis=1, kind='stable')
        rev_gather = np.take_along_axis(np.asarray(group.edge_reversed, dtype=bool), inv_edge, axis=1)
        self.inverse = {'node': torch.as_tensor(inv_node, device=device),
                        'edge': torch.as_tensor(inv_edge, device=device)}
        self.forward_maps = {'node': self.node_maps, 'edge': self.edge_maps}
        self.rev_gather = torch.as_tensor(rev_gather, device=device)

    def _rotate_many(self, x, g0, g1, adjoint):
        """x: [c, R, W] rows already gathered for operations g0..g1-1; block-rotate each by its own operation."""
        out = torch.empty_like(x)
        for la, lb, index in self.blocks:
            a, b = self.rotations[la][g0:g1], self.rotations[lb][g0:g1]
            if adjoint:
                a, b = a.transpose(1, 2), b.transpose(1, 2)
            block = x[:, :, index].reshape(x.shape[0], x.shape[1], -1, 2 * la + 1, 2 * lb + 1)
            out[:, :, index] = torch.einsum('gij,grkjl,gml->grkim', a, block, b).reshape(x.shape[0], x.shape[1], -1)
        return out

    def average_vec(self, x, part, adjoint=False, chunk=8):
        """Vectorized equivalent of average(): a few batched kernels per chunk of operations instead of a Python
        loop over every operation and block type."""
        n_ops = len(self.group.rotations)
        out = torch.zeros_like(x)
        if x.shape[0] == 0:
            return out
        index = self.forward_maps[part] if adjoint else self.inverse[part]
        for g0 in range(0, n_ops, chunk):
            g1 = min(n_ops, g0 + chunk)
            y = x[index[g0:g1]]                                   # [c, R, W]
            if adjoint:
                if part == 'edge':
                    y = torch.where(self.reverse[g0:g1, :, None], y[..., self.transpose], y)
                y = self._rotate_many(y, g0, g1, adjoint=True)
            else:
                y = self._rotate_many(y, g0, g1, adjoint=False)
                if part == 'edge':
                    y = torch.where(self.rev_gather[g0:g1, :, None], y[..., self.transpose], y)
            out.add_(y.sum(0))
        return out / n_ops

    def rotate(self, x, g, adjoint=False):
        if x.shape[0] == 0:
            return x.clone()
        out = torch.empty_like(x)
        for la, lb, index in self.blocks:
            a, b = self.rotations[la][g], self.rotations[lb][g]
            if adjoint:
                a, b = a.T, b.T
            block = x[:, index].reshape(x.shape[0], -1, 2 * la + 1, 2 * lb + 1)
            out[:, index] = (a @ block @ b.T).flatten(1)
        return out

    def act(self, x, part, g, adjoint=False):
        maps = self.node_maps if part == 'node' else self.edge_maps
        if adjoint:
            y = x.index_select(0, maps[g])
            if part == 'edge':
                y = torch.where(self.reverse[g, :, None], y[:, self.transpose], y)
            return self.rotate(y, g, adjoint=True)
        y = self.rotate(x, g)
        if part == 'edge':
            y = torch.where(self.reverse[g, :, None], y[:, self.transpose], y)
        return torch.empty_like(y).index_copy_(0, maps[g], y)

    def average(self, x, part, adjoint=False):
        out = torch.zeros_like(x)
        for g in range(len(self.group.rotations)):
            out.add_(self.act(x, part, g, adjoint=adjoint))
        return out / len(self.group.rotations)


class _BatchAction:
    """All structures of a batch in one weighted gather-rotate-sum.

    Rows are laid out by descending group order (structures stable, each structure's rows contiguous).  Row r of
    structure s under padded operation g takes source row index[g, r], is block-rotated by operation g of s,
    transposed for reversed edges and weighted 1/|G_s| (0 for g >= |G_s|).  Under operation g only the leading
    rows (structures with more than g operations) contribute, so the work follows sum_s |G_s| R_s rather than
    max_s |G_s| * R.  The projection is self-adjoint (orthogonal representation of a group), so the backward pass
    applies the same operator to the incoming gradient.  apply() takes as many operations per chunk as fit in
    `budget_bytes` of gathered rows.
    """
    def __init__(self, actions, rows, part, transpose, blocks, n_rows, budget_bytes=2 ** 30):
        device = actions[0].node_maps.device
        dtype = actions[0].rotations[next(iter(actions[0].rotations))].dtype
        orders = np.array([len(a.group.rotations) for a in actions], dtype=np.int64)
        counts = np.array([len(r) for r in rows], dtype=np.int64)
        g_max = int(orders.max())
        order = np.argsort(-orders, kind='stable')
        o_s, c_s = orders[order], counts[order]
        start = np.concatenate([[0], np.cumsum(c_s)[:-1]]).astype(np.int64)
        flat = np.concatenate([[0], np.cumsum(o_s * c_s)[:-1]]).astype(np.int64)
        original = np.concatenate([np.asarray(rows[s], dtype=np.int64) for s in order])   # sorted row -> row
        host = np.stack([np.repeat(np.arange(len(order)), c_s),               # sorted structure of each row
                         np.arange(n_rows) - np.repeat(start, c_s),          # row within its structure
                         np.repeat(flat, c_s), np.repeat(c_s, c_s), np.repeat(o_s, c_s), np.repeat(start, c_s)])
        sid, local, offset, size, row_order, row_start = torch.as_tensor(host, device=device).unbind(0)
        g = torch.arange(g_max, device=device)[:, None]
        # Padded operations repeat the structure's last operation; their weight is zero.
        src = offset + torch.minimum(g, row_order - 1) * size + local          # [G_max, R] into the flat maps
        self.index = torch.cat([actions[k].inverse[part].reshape(-1) for k in order])[src] + row_start
        self.rev = (torch.cat([actions[k].rev_gather.reshape(-1) for k in order])[src]
                    if part == 'edge' else None)
        self.weight = (g < row_order).to(dtype) / row_order.to(dtype)
        self.sid = sid
        o_t = torch.as_tensor(o_s, device=device)
        rot = torch.minimum(g, o_t - 1) + torch.cumsum(o_t, 0) - o_t          # [G_max, S] into the flat stacks
        self.rotations = {l: torch.cat([actions[k].rotations[l] for k in order])[rot]
                          for l in actions[0].rotations}                        # [G_max, S, 2l+1, 2l+1]
        self.active = [int(c_s[o_s > k].sum()) for k in range(g_max)]          # rows contributing to operation k
        in_place = np.array_equal(original, np.arange(n_rows))
        self.perm = None if in_place else torch.as_tensor(original, device=device)
        self.pos = None if in_place else torch.as_tensor(np.argsort(original, kind='stable'), device=device)
        self.part, self.transpose, self.blocks = part, transpose.to(device), blocks
        self.budget = budget_bytes

    def apply(self, x):
        if x.shape[0] == 0:
            return torch.zeros_like(x)
        xp = x if self.perm is None else x[self.perm]
        out = torch.zeros_like(xp)
        g_max, g0 = self.index.shape[0], 0
        row_bytes = x.shape[1] * x.element_size()
        while g0 < g_max and self.active[g0]:
            n = self.active[g0]
            g1 = min(g_max, g0 + max(1, self.budget // max(1, n * row_bytes)))
            y = xp[self.index[g0:g1, :n]]                                     # [c, n, W]
            rot = {l: r[g0:g1][:, self.sid[:n]] for l, r in self.rotations.items()}   # [c, n, 2l+1, 2l+1]
            z = torch.empty_like(y)
            for la, lb, index in self.blocks:
                block = y[:, :, index].reshape(y.shape[0], n, -1, 2 * la + 1, 2 * lb + 1)
                z[:, :, index] = torch.einsum('grij,grkjl,grml->grkim', rot[la], block, rot[lb]).reshape(y.shape[0], n, -1)
            if self.part == 'edge':
                z = torch.where(self.rev[g0:g1, :n, None], z[..., self.transpose], z)
            out[:n].add_((z * self.weight[g0:g1, :n, None]).sum(0))
            g0 = g1
        return out if self.pos is None else out[self.pos]


class _BatchReynolds(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, batch_action):
        ctx.batch_action = batch_action
        return batch_action.apply(values)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.batch_action.apply(gradient.contiguous()), None


class _Reynolds(torch.autograd.Function):
    """Linear adjoint backward; save no per-operation activations."""
    @staticmethod
    def forward(ctx, values, action, part):
        ctx.action, ctx.part = action, part
        return action.average_vec(values, part)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.action.average_vec(gradient.contiguous(), ctx.part, adjoint=True), None, None


class SymmetryProjector:
    """Bounded exact-geometry cache plus device-native differentiable averaging.

    Supported layout is the complete ordered real AO-product shell-pair layout,
    including compact SOC uu-real. Full complex spinors and triangular non-SOC
    packing are rejected, rather than silently treated as the same representation.
    A provider adapter can replace SpglibGroupProvider with no trainer changes.
    """
    budget_bytes = 2 ** 30       # gathered rows per chunk of operations in the batched average

    def __init__(self, idp, symprec=1e-3, provider=None, cache_size=256, open_graph='raise', backend='torch'):
        if not math.isfinite(symprec) or symprec <= 0 or cache_size < 1:
            raise ValueError("symprec and cache_size must be positive")
        if open_graph not in ('raise', 'identity'):
            raise ValueError("open_graph must be raise or identity")
        if backend not in ('torch', 'cuda'):
            raise ValueError("backend must be torch or cuda")
        self.open_graph = open_graph
        self.backend = backend
        self.idp, self.symprec = idp, float(symprec)
        self.provider = provider if provider is not None else SpglibGroupProvider()
        self.cache_size = int(cache_size)
        self.cache = OrderedDict()
        self.fallbacks = []          # reasons of structures left unprojected (identity group)
        idp.get_orbpair_maps()
        self.width = int(idp.reduced_matrix_element)
        self.blocks, transpose = {}, torch.empty(self.width, dtype=torch.long)
        coverage = []
        for pair, sl in idp.orbpair_maps.items():
            a, b = pair.split('-')
            la, lb = _angular(a), _angular(b)
            if sl.stop - sl.start != (2 * la + 1) * (2 * lb + 1):
                raise ValueError("symmetry projection requires compact real AO-product blocks")
            other = idp.orbpair_maps.get(b + '-' + a)
            if other is None:
                raise ValueError("symmetry projection requires all ordered shell pairs (SOC uu-real)")
            index = torch.arange(sl.start, sl.stop)
            self.blocks.setdefault((la, lb), []).append(index)
            coverage.extend(index.tolist())
            transpose[sl] = torch.arange(other.start, other.stop).reshape(2 * lb + 1, 2 * la + 1).T.flatten()
        if sorted(coverage) != list(range(self.width)):
            raise ValueError("orbpair_maps must cover the AO-product feature layout")
        self.blocks = [(a, b, torch.cat(ix)) for (a, b), ix in self.blocks.items()]
        self.transpose = transpose
        self._positions = {}
        if backend == 'cuda':
            from dptb.nn.sym_projection_cuda import check_layout
            check_layout(self.blocks, self.width)

    def clear_cache(self):
        self.cache.clear()

    def _fused_positions(self, device):
        if device not in self._positions:
            from dptb.nn.sym_projection_cuda import position_tables
            self._positions[device] = position_tables(self.blocks, self.transpose, self.width, device)
        return self._positions[device]

    def action(self, positions, cell, species, edge_index, edge_shift, *, device, dtype):
        arrays = [np.ascontiguousarray(v) for v in (positions, cell, species, edge_index, edge_shift)]
        # Full precision, dtype/shape, graph and tolerance are part of the key.
        key = (self.symprec, tuple((v.dtype.str, v.shape, v.tobytes()) for v in arrays))
        cached = self.cache.get(key)
        if cached is None:
            try:
                group = self.provider(*arrays, symprec=self.symprec)
            except ValueError as err:
                # A stored neighbour graph that the detected group does not close (an image bond just outside the
                # cutoff), or no spglib group: this structure is left unprojected (identity) instead of stopping
                # training.  Other provider errors indicate inconsistent inputs and still raise.
                if self.open_graph != 'identity' or not re.search(r'not symmetry-closed|could not determine', str(err)):
                    raise
                n, e = len(arrays[2]), arrays[3].shape[1]
                group = CrystalGroup(np.eye(3)[None], np.arange(n)[None], np.arange(e)[None],
                                     np.zeros((1, e), dtype=bool))
                self.fallbacks.append(str(err))
            cached = (group, {})
            self.cache[key] = cached
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        group, devices = cached
        device_key = (str(device), dtype)
        if device_key not in devices:
            if self.backend == 'cuda':
                from dptb.nn.sym_projection_cuda import FusedAction as make_action
            else:
                make_action = _Action
            devices[device_key] = make_action(group, self.blocks, self.transpose, device, dtype)
        return devices[device_key]

    def __call__(self, data, parts=('node', 'edge')):
        """Project the listed parts ('node', 'edge'); an unlisted part is returned unchanged."""
        parts = tuple(parts)
        if any(p not in ('node', 'edge') for p in parts):
            raise ValueError("parts must be a subset of ('node', 'edge')")
        def cpu(key):
            return data[key].detach().cpu().numpy()

        pos = cpu(_keys.POSITIONS_KEY)
        cell = cpu(_keys.CELL_KEY).reshape(-1, 3, 3)
        species_key = _keys.ATOMIC_NUMBERS_KEY if _keys.ATOMIC_NUMBERS_KEY in data else _keys.ATOM_TYPE_KEY
        species = cpu(species_key).reshape(-1)
        if species_key == _keys.ATOM_TYPE_KEY:
            species = species + 1  # spglib accepts positive species IDs
        batch = cpu(_keys.BATCH_KEY).reshape(-1) if _keys.BATCH_KEY in data else np.zeros(len(pos), dtype=int)
        if len(batch) != len(pos) or len(batch) == 0 or batch.min() < 0 or batch.max() >= len(cell):
            raise ValueError("invalid structure IDs for the supplied cells")
        edges = cpu(_keys.EDGE_INDEX_KEY)
        shifts = cpu(_keys.EDGE_CELL_SHIFT_KEY)
        if not np.allclose(shifts, np.rint(shifts), atol=1e-8, rtol=0):
            raise ValueError("edge cell shifts must be integer")
        if _keys.PBC_KEY in data and not cpu(_keys.PBC_KEY).all():
            raise ValueError("crystal projection requires three-dimensional periodicity")
        if np.any(batch[edges[0]] != batch[edges[1]]):
            raise ValueError("edge connects different structures")
        out = data.copy()
        for part in ('node', 'edge'):
            x = data[part + '_features']
            if x.ndim != 2 or x.shape[1] != self.width or x.is_complex():
                raise ValueError("symmetry projection requires packed real AO-product features")
            expected_rows = len(pos) if part == 'node' else edges.shape[1]
            if x.shape[0] != expected_rows:
                raise ValueError("feature rows do not match the stored geometry/graph")
            if x.dtype != data['node_features'].dtype or x.device != data['node_features'].device:
                raise ValueError("node and edge predictions must share dtype and device")
        actions, rows = [], {'node': [], 'edge': []}
        edge_owner = batch[edges[0]]
        for g in range(len(cell)):
            ni = np.flatnonzero(batch == g)
            ei = np.flatnonzero(edge_owner == g)
            if len(ni) == 0:
                raise ValueError("empty or non-contiguous structure IDs")
            inverse = np.full(len(pos), -1, dtype=np.int64)
            inverse[ni] = np.arange(len(ni))
            like = data['node_features']
            actions.append(self.action(pos[ni], cell[g], species[ni], inverse[edges[:, ei]],
                                       np.rint(shifts[ei]).astype(np.int64), device=like.device, dtype=like.dtype))
            rows['node'].append(ni)
            rows['edge'].append(ei)
        for part in ('node', 'edge'):
            x = data[part + '_features']
            if sum(len(r) for r in rows[part]) != x.shape[0]:
                raise ValueError("every feature row must belong to exactly one structure")
            if part not in parts:
                continue
            if self.backend == 'cuda':
                from dptb.nn.sym_projection_cuda import FusedBatch
                batch_action = FusedBatch(actions, rows[part], part, self._fused_positions(x.device), x.shape[0])
            else:
                batch_action = _BatchAction(actions, rows[part], part, actions[0].transpose, actions[0].blocks,
                                            x.shape[0], budget_bytes=self.budget_bytes)
            out[part + '_features'] = _BatchReynolds.apply(x, batch_action)
        return out

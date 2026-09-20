"""Batched CUDA onsite XC on the accepted atom-centred local quadrature.

Exact engineering acceleration of the accepted per-atom loop (``AtomicDensity`` and
``OnsiteQuadrature.evaluate``): the same quadrature nodes and orders, the same fixed-radius
neighbourhoods, the same density splines (normalized neutral valence plus unscaled NLCC,
per-channel positivity, end clamping, support clipping) and the same LDA-PZ81 potential rule
built from the caller-supplied ``v_and_dv``. Only the evaluation order changes: fused CUDA
launches per (species, order) atom group (bounded by a density byte budget) form every neighbour
density sum without distance or spline intermediates, the potential is applied once on
``[atoms, points]`` and the Gram contraction ``basis^T diag(v) basis`` of the atoms runs as one
folded GEMM per launch. Pairs that are exactly zero under the accepted rule (beyond a species' last
knot) are skipped without evaluation, first per neighbour on the host (``onsite_candidates``) and
then per pair in the kernel by the triangle inequality, while the accepted 16-neighbour chunk
partial sums are reproduced bitwise. Remaining differences from the Torch reference are FP64
rounding from summation order. The reference path is retained on identical inputs.

Inference only, CUDA FP64 only. Nothing here chooses physics: densities, potential, radius,
quadrature objects and neighbour lists are supplied by the caller.
"""
import itertools
from types import SimpleNamespace
import numpy as np
import torch

DENSITY_FLOOR = 1e-20      # accepted rule: v = v_and_dv(max(rho, floor))[0], zero where rho <= floor
LEGACY_RADIUS_BOHR = 27.   # accepted onsite neighbourhood truncation of the fixed-cohort runs
PRUNE_MARGIN_BOHR = 1e-9   # minimum padding; coordinate-scaled FP64 rounding bounds are added below and in CUDA
MAX_ATOMS_PER_LAUNCH = 65535


def pz81_potential(v_and_dv):
    """Accepted onsite potential rule from an accepted ``v_and_dv(rho) -> (v, dv)`` callable."""
    def potential(rho):
        v = v_and_dv(rho.clamp_min(DENSITY_FLOOR))[0]
        return torch.where(rho > DENSITY_FLOOR, v, 0.)
    return potential


class SplineDensity:
    """Clamped piecewise-cubic density with the ``AtomicDensity`` call contract.

    ``knots`` [K] strictly increasing; ``coeff`` is a list of [4, K-1] tensors (valence, then
    optional NLCC) in the SciPy ``CubicSpline.c`` power order. Each channel is clamped to
    nonnegative values, the radius is clamped to the knot range, and the result is zero beyond
    the last knot. This is the Torch reference the fused kernel reproduces.
    """

    def __init__(self, knots, coeff):
        self.knots = knots
        self.coeff = list(coeff)

    def __call__(self, r):
        rr = r.clamp(self.knots[0], self.knots[-1])
        idx = torch.searchsorted(self.knots, rr.contiguous(), right=True) - 1
        idx = idx.clamp(0, len(self.knots) - 2)
        d = rr - self.knots[idx]
        out = torch.zeros_like(r)
        for c in self.coeff:
            out += (((c[0, idx] * d + c[1, idx]) * d + c[2, idx]) * d + c[3, idx]).clamp_min(0)
        return torch.where(r > self.knots[-1], 0., out)


# ----------------------------------------------------------------------------- neighbourhoods
def onsite_neighbor_lists(g, radius=LEGACY_RADIUS_BOHR, *, atoms=None):
    """Fixed-radius onsite neighbourhoods for every atom, in the accepted order.

    Returns one ``{species: displacements[n, 3]}`` per requested atom. Set, grouping (species
    in first-appearance order of ``g['symbols']``) and order (source atom index, then
    lexicographic image offsets around the rounded fractional guess) follow the accepted
    per-atom enumerator built on ``VectorizedNearbyImageEnumerator``; the origin atom itself
    is included with zero displacement. Strict ``|d| < radius``. Non-periodic axes of
    ``g['pbc']`` (default all periodic, as in the accepted runs) use the zero offset only.
    """
    pos = np.asarray(g['positions_bohr'], dtype=np.float64)
    cell = np.asarray(g['cell_bohr'], dtype=np.float64)
    symbols = list(g['symbols'])
    if pos.ndim != 2 or pos.shape[1] != 3 or cell.shape != (3, 3) or len(symbols) != len(pos):
        raise ValueError('geometry needs positions_bohr [n,3], cell_bohr [3,3] and n symbols')
    radius = float(radius)
    if not radius > 0:
        raise ValueError('onsite radius must be positive')
    pbc = tuple(bool(x) for x in g.get('pbc', (True, True, True)))
    inverse = np.linalg.inv(cell)
    bounds = np.ceil(radius * np.linalg.norm(inverse, axis=0)).astype(int) + 1
    ranges = [range(-int(b), int(b) + 1) if periodic else range(0, 1) for b, periodic in zip(bounds, pbc)]
    offsets = np.asarray(list(itertools.product(*ranges)), dtype=np.int64)
    targets = list(range(len(pos))) if atoms is None else [int(i) for i in atoms]
    species_order = list(dict.fromkeys(symbols))
    collected = {i: {s: [] for s in species_order} for i in targets}
    origins = pos[targets]
    for j, (species, base) in enumerate(zip(symbols, pos)):
        guess = np.rint((origins - base) @ inverse).astype(np.int64)
        if not all(pbc):
            guess[:, [k for k, periodic in enumerate(pbc) if not periodic]] = 0
        translations = (offsets[None, :, :] + guess[:, None, :]).reshape(-1, 3)
        centers = base[None, :] + translations @ cell
        delta = centers.reshape(len(targets), -1, 3) - origins[:, None, :]
        keep = np.linalg.norm(delta, axis=-1) < radius
        for t, i in enumerate(targets):
            if keep[t].any():
                collected[i][species].append(delta[t][keep[t]])
    result = []
    for i in targets:
        grouped = {}
        for s in species_order:
            parts = collected[i][s]
            if parts:
                grouped[s] = np.concatenate(parts, axis=0)
        result.append(grouped)
    return result


def onsite_neighbor_lists_native(g, radius=LEGACY_RADIUS_BOHR, *, library=None, atoms=None):
    """The lists of :func:`onsite_neighbor_lists` from one native (Tonari cell-list) search of the whole cell.

    Same set, species grouping, order and displacement values as the accepted NumPy enumerator: the native
    search returns every candidate within a padded radius sorted in the accepted order, and the strict
    ``|d| < radius`` test is applied here in the reference expression ``(pos[j] + t @ cell) - pos[i]``.
    The cost is O(neighbours) instead of O(atoms^2 x images).
    """
    from .topology import onsite_neighbour_rows
    pos = np.asarray(g['positions_bohr'], dtype=np.float64)
    cell = np.asarray(g['cell_bohr'], dtype=np.float64)
    symbols = list(g['symbols'])
    if pos.ndim != 2 or pos.shape[1] != 3 or cell.shape != (3, 3) or len(symbols) != len(pos):
        raise ValueError('geometry needs positions_bohr [n,3], cell_bohr [3,3] and n symbols')
    radius = float(radius)
    if not radius > 0:
        raise ValueError('onsite radius must be positive')
    pbc = tuple(bool(x) for x in g.get('pbc', (True, True, True)))
    species_order = list(dict.fromkeys(symbols))
    rank = np.array([species_order.index(s) for s in symbols], dtype=np.int64)
    rows, _ = onsite_neighbour_rows(pos, cell, pbc, radius, rank, library=library)
    translations = np.ascontiguousarray(rows[:, 2:])
    delta = (pos[rows[:, 1]] + translations @ cell) - pos[rows[:, 0]]
    keep = np.linalg.norm(delta, axis=1) < radius
    rows, delta = rows[keep], delta[keep]
    atom_ptr = np.searchsorted(rows[:, 0], np.arange(len(pos) + 1))
    targets = list(range(len(pos))) if atoms is None else [int(i) for i in atoms]
    result = []
    for i in targets:
        a, b = atom_ptr[i], atom_ptr[i + 1]
        ranks = rank[rows[a:b, 1]]
        bounds = np.searchsorted(ranks, np.arange(len(species_order) + 1))
        grouped = {}
        for k, s in enumerate(species_order):
            if bounds[k + 1] > bounds[k]:
                grouped[s] = delta[a + bounds[k]:a + bounds[k + 1]]
        result.append(grouped)
    return result


# ----------------------------------------------------------------------------- density bank packing
def _tensor_identity(value):
    """Storage identity of one spline array without reading its contents.

    Tensors: data pointer, shape, dtype, device and the in-place version counter (``None`` for inference
    tensors, which do not track one). Other array-likes: object id, buffer address and shape.
    """
    if isinstance(value, torch.Tensor):
        try:
            version = value._version
        except RuntimeError:            # inference tensors do not track a version counter
            version = None
        return (value.data_ptr(), tuple(value.shape), str(value.dtype), str(value.device), version)
    array = np.asarray(value)
    return (id(value), array.__array_interface__['data'][0], array.shape, str(array.dtype), None)


def density_identity(density_bank):
    """Cheap, synchronization-free content identity of a density bank.

    Species order, each density object, and the storage identity of its ``knots`` and every ``coeff``
    channel (see :func:`_tensor_identity`). Replacing a species density, or editing a knot or coefficient
    tensor in place (version counter), changes the identity; no array is hashed. In-place edits of
    inference-mode tensors are not tracked by Torch and are invisible here: invalidate explicitly.
    """
    return tuple((s, id(density), _tensor_identity(density.knots), tuple(_tensor_identity(c) for c in density.coeff))
                 for s, density in density_bank.items())


class PackedDensityBank:
    """Concatenated species splines (knots, [channels,4,K-1] coefficients) for the fused kernel.

    The bank records the :func:`density_identity` of its source at construction; ``matches`` tells whether
    a density bank still has exactly that content identity.
    """

    def __init__(self, density_bank, device):
        self.identity = density_identity(density_bank)
        self.species = list(density_bank)
        self.index = {s: k for k, s in enumerate(self.species)}
        knots, coeffs, knot_ptr, coeff_ptr, channels = [], [], [0], [0], []
        for s in self.species:
            density = density_bank[s]
            k = torch.as_tensor(density.knots, device=device, dtype=torch.float64).reshape(-1)
            if k.numel() < 2 or not bool((k[1:] > k[:-1]).all()):
                raise ValueError(f'density knots of {s!r} must hold at least two strictly increasing values')
            c = torch.stack([torch.as_tensor(x, device=device, dtype=torch.float64) for x in density.coeff])
            if c.ndim != 3 or c.shape[1] != 4 or c.shape[2] != k.numel() - 1 or c.shape[0] < 1:
                raise ValueError(f'density coefficients of {s!r} must be a list of [4, knots-1] tensors')
            knots.append(k); coeffs.append(c.reshape(-1))
            knot_ptr.append(knot_ptr[-1] + k.numel()); coeff_ptr.append(coeff_ptr[-1] + c.numel()); channels.append(c.shape[0])
        self.knots = torch.cat(knots).contiguous()
        self.knots_f32 = self.knots.to(torch.float32)   # interval search only; the FP64 fix-up makes the interval exact
        self.coeff = torch.cat(coeffs).contiguous()
        self.knot_ptr = torch.tensor(knot_ptr, device=device, dtype=torch.int64)
        self.coeff_ptr = torch.tensor(coeff_ptr, device=device, dtype=torch.int64)
        self.channels = torch.tensor(channels, device=device, dtype=torch.int64)
        self.device = self.knots.device
        # host copy of every species' last knot: the density is exactly zero beyond it (accepted rule), which is what
        # the exact candidate selection and the in-kernel pair skipping rely on
        self.last_knot_host = np.array([float(k[-1]) for k in knots], dtype=np.float64)

    def matches(self, density_bank):
        return density_identity(density_bank) == self.identity


def _segment_lists(neighbors, bank):
    """Flatten per-atom species dicts into CSR segments; reject empty atoms and unknown species."""
    seg_ptr, segments, parts, total = [0], [], [], 0
    for a, grouped in enumerate(neighbors):
        count = 0
        for s, p in grouped.items():
            if s not in bank.index:
                raise ValueError(f'neighbour species {s!r} has no density in the bank')
            p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
            if p.ndim != 2 or p.shape[1] != 3:
                raise ValueError('neighbour displacements must be [n, 3]')
            if not len(p):
                continue
            segments.append((bank.index[s], total, total + len(p))); parts.append(p)
            total += len(p); count += len(p)
        if not count:
            raise ValueError(f'atom {a} has no onsite neighbours; the origin atom itself must be present')
        seg_ptr.append(len(segments))
    return np.asarray(seg_ptr, dtype=np.int64), np.asarray(segments, dtype=np.int64).reshape(-1, 3), np.concatenate(parts, axis=0)


def onsite_candidates(neighbors, bank, grid_radius, *, prune=True, margin=PRUNE_MARGIN_BOHR):
    """Exact candidate selection for the fused kernel: drop neighbours that cannot reach the grid.

    A neighbour at centre distance ``|d|`` contributes to a grid point at radius ``r_p <= grid_radius`` only if
    ``|p - d| <= k_last`` (the density is exactly zero beyond the species' last knot), and ``|p - d| >= |d| - r_p``,
    so every neighbour with ``|d| > grid_radius + k_last + margin`` contributes exactly zero to every point and is
    removed here; the margin dominates the FP64 rounding of both norms. The survivors keep their position
    (``local``) in the accepted per-species list, which the kernel uses to reproduce the accepted 16-neighbour
    chunk partial sums, so the result is bitwise the unpruned walk. ``prune=False`` keeps every neighbour.

    Returns ``(seg_ptr, segments, candidates[c, 4] = (x, y, z, |d|), local[c], stats)``.
    """
    seg_ptr, segments, positions = _segment_lists(neighbors, bank)
    distance = np.linalg.norm(positions, axis=1)
    counts = segments[:, 2] - segments[:, 1]
    segment_of_row = np.repeat(np.arange(len(segments)), counts)
    local = np.arange(len(positions), dtype=np.int64) - segments[segment_of_row, 1]
    if prune:
        support = bank.last_knot_host[segments[segment_of_row, 0]]
        rounding = 32 * np.finfo(np.float64).eps * (distance + abs(float(grid_radius)) + support + 1)
        reach = float(grid_radius) + support + float(margin) + rounding
        keep = distance <= reach
    else:
        keep = np.ones(len(positions), dtype=bool)
    kept = np.bincount(segment_of_row[keep], minlength=len(segments)).astype(np.int64)
    end = np.cumsum(kept)
    compact = np.column_stack((segments[:, 0], end - kept, end)).astype(np.int64)
    candidates = np.ascontiguousarray(np.column_stack((positions[keep], distance[keep])), dtype=np.float64)
    stats = {'neighbours': int(len(positions)), 'candidates': int(keep.sum())}
    return seg_ptr, compact, candidates, np.ascontiguousarray(local[keep]), stats


def grid_radius(quadrature):
    """Cache the largest point radius by tensor object/storage/version.

    Inference tensors and edits through ``.data``/NumPy views have no usable version
    counter. Call :func:`invalidate_grid_radius` after those edits. Replacement and
    normal PyTorch in-place edits refresh automatically. Retaining the old tensor
    also prevents allocator address reuse from producing a false cache hit.
    """
    xyz = quadrature.xyz
    identity = (_tensor_identity(xyz), tuple(xyz.stride()))
    cached = getattr(quadrature, '_nacf_grid_radius', None)
    if isinstance(cached, tuple) and cached[0] is xyz and cached[1] == identity:
        return cached[2]
    value = float(torch.linalg.vector_norm(xyz, dim=-1).max()) if len(xyz) else 0.0
    try:
        quadrature._nacf_grid_radius = (xyz, identity, value)
    except (AttributeError, TypeError):
        pass
    return value


def invalidate_grid_radius(quadrature):
    """Explicit invalidation after untracked edits of a cached quadrature grid."""
    if hasattr(quadrature, '_nacf_grid_radius'):
        delattr(quadrature, '_nacf_grid_radius')


def _launch_density(native, xyz, candidates, local, seg_ptr, segments, bank, margin):
    device = xyz.device
    return native.onsite_density(
        xyz.contiguous(), torch.as_tensor(candidates, device=device), torch.as_tensor(local, device=device),
        torch.as_tensor(seg_ptr, device=device), torch.as_tensor(segments, device=device),
        bank.knots, bank.knots_f32, bank.knot_ptr, bank.coeff, bank.coeff_ptr, bank.channels, float(margin))


def _check_fused_inputs(xyz, bank):
    from ._cuda import extension, check_device
    if not isinstance(xyz, torch.Tensor) or not xyz.is_cuda:
        raise ValueError('fused onsite density requires a CUDA quadrature grid; use the reference path on CPU')
    if xyz.dtype != torch.float64:
        raise ValueError('fused onsite density is an FP64 contract')
    if xyz.requires_grad:
        raise ValueError('fused onsite density is inference-only')
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('quadrature grid must be [points, 3]')
    if bank.device != xyz.device:
        raise ValueError('density bank and quadrature grid must share a device')
    check_device(xyz.device)
    native = extension()
    if not hasattr(native, 'onsite_density'):
        raise RuntimeError('fused onsite density is absent from this binary; rebuild with python -m dptb.nacf.precompile')
    return native


def _atom_chunks(natoms, points, rho_bytes):
    step = max(1, min(MAX_ATOMS_PER_LAUNCH, int(rho_bytes) // max(1, points * 8)))
    return [(start, min(natoms, start + step)) for start in range(0, natoms, step)]


def _chunk_segments(seg_ptr, segments, start, stop):
    """Segment table and candidate range of atoms [start, stop) of a compact candidate layout."""
    sub = segments[seg_ptr[start]:seg_ptr[stop]].copy()
    if len(sub):
        base, end = int(sub[0, 1]), int(sub[-1, 2])
        sub[:, 1:] -= base
    else:
        base = end = 0
    return seg_ptr[start:stop + 1] - seg_ptr[start], sub, base, end


def fused_onsite_density(xyz, neighbors, bank, *, prune=True, rho_bytes=1 << 30):
    """rho[atoms, points]: fused neighbour density sums on one CUDA FP64 quadrature grid.

    ``xyz`` [P,3] CUDA float64; ``neighbors`` is a list (one entry per atom) of
    ``{species: displacements[n,3]}`` relative to that atom; ``bank`` is a PackedDensityBank
    on the same device. Launches are bounded to ``rho_bytes`` of density per launch (and 65535
    atoms). ``prune=False`` disables the exact candidate/pair skipping (identical result, for checks).
    Inference only; fails on CPU or non-FP64 input instead of falling back.
    """
    native = _check_fused_inputs(xyz, bank)
    if not len(neighbors):
        return xyz.new_zeros((0, xyz.shape[0]))
    radius = grid_radius(SimpleNamespace(xyz=xyz))
    seg_ptr, segments, candidates, local, _ = onsite_candidates(neighbors, bank, radius, prune=prune)
    margin = PRUNE_MARGIN_BOHR if prune else float('inf')
    parts = []
    for start, stop in _atom_chunks(len(neighbors), xyz.shape[0], rho_bytes):
        ptr, sub, base, end = _chunk_segments(seg_ptr, segments, start, stop)
        parts.append(_launch_density(native, xyz, candidates[base:end], local[base:end], ptr, sub, bank, margin))
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def gram_blocks(basis, v, *, chunk_bytes=1 << 30):
    """``basis^T diag(v_a) basis`` for every row of ``v`` [atoms, points].

    The atoms of a chunk are folded into the columns of one GEMM ``[norb, P] @ [P, atoms*norb]``;
    a batched matmul with a broadcast operand costs a flat 30-600 ms in cuBLAS FP64 for these
    tiny-M shapes, the folded GEMM costs milliseconds. The rule ``basis^T (v[:, None] * basis)``
    per atom is unchanged.
    """
    atoms, norb = v.shape[0], basis.shape[1]
    points = basis.shape[0]
    out = basis.new_empty((atoms, norb, norb))
    step = max(1, int(chunk_bytes // max(1, basis.numel() * basis.element_size())))
    for start in range(0, atoms, step):
        chunk = v[start:start + step].T.contiguous()                    # [P, a]
        weighted = chunk[:, :, None] * basis[:, None, :]                # [P, a, norb] contiguous
        gram = basis.T @ weighted.reshape(points, -1)                   # [norb, a*norb]
        out[start:start + step] = gram.reshape(norb, chunk.shape[1], norb).permute(1, 0, 2)
    return out


def fused_onsite_blocks(quadrature, neighbors, bank, potential, *, chunk_bytes=1 << 30, rho_bytes=1 << 30, return_density=False, prune=True, stats=None):
    """Onsite XC AO blocks [atoms, norb, norb] for atoms sharing one quadrature object.

    ``quadrature`` exposes ``xyz`` [P,3] and weight-scaled ``basis`` [P,norb] on CUDA FP64. Atoms are
    processed in chunks bounded by ``rho_bytes`` of FP64 density (and 65535 atoms per launch): density,
    potential and Gram of a chunk complete before the next chunk allocates. ``stats`` (dict) receives the
    neighbour/candidate counts. ``rho_bytes`` bounds temporary density work (with a
    one-atom minimum), not a requested full-density output or Gram intermediates.
    """
    xyz, basis = quadrature.xyz, quadrature.basis
    native = _check_fused_inputs(xyz, bank)
    natoms, norb = len(neighbors), basis.shape[1]
    if not natoms:
        empty = basis.new_zeros((0, norb, norb))
        return (empty, xyz.new_zeros((0, len(xyz)))) if return_density else empty
    seg_ptr, segments, candidates, local, counts = onsite_candidates(neighbors, bank, grid_radius(quadrature), prune=prune)
    if stats is not None:
        stats.update(counts)
    margin = PRUNE_MARGIN_BOHR if prune else float('inf')
    out = basis.new_empty((natoms, norb, norb))
    densities = []
    for start, stop in _atom_chunks(natoms, xyz.shape[0], rho_bytes):
        ptr, sub, base, end = _chunk_segments(seg_ptr, segments, start, stop)
        rho = _launch_density(native, xyz, candidates[base:end], local[base:end], ptr, sub, bank, margin)
        out[start:stop] = gram_blocks(basis, potential(rho), chunk_bytes=chunk_bytes)
        if return_density:
            densities.append(rho)
        del rho
    if return_density:
        return out, (densities[0] if len(densities) == 1 else torch.cat(densities))
    return out


def reference_onsite_blocks(quadrature, neighbors, density_bank, potential, *, chunk=16, return_density=False):
    """Accepted per-atom evaluation (``OnsiteQuadrature.evaluate``) on the same inputs; any device.

    ``density_bank[s]`` must be callable on a distance tensor (``AtomicDensity``/``SplineDensity``).
    """
    xyz, basis = quadrature.xyz, quadrature.basis
    blocks, densities = [], []
    for grouped in neighbors:
        rho = torch.zeros(len(xyz), device=xyz.device, dtype=xyz.dtype)
        for s, positions in grouped.items():
            p = torch.as_tensor(np.asarray(positions, dtype=np.float64), device=xyz.device, dtype=xyz.dtype)
            for start in range(0, len(p), chunk):
                distances = torch.linalg.vector_norm(xyz[:, None, :] - p[None, start:start + chunk, :], dim=-1)
                rho += density_bank[s](distances).sum(dim=1)
        v = potential(rho)
        blocks.append(basis.T @ (v[:, None] * basis)); densities.append(rho)
    if not blocks:
        empty = basis.new_zeros((0, basis.shape[1], basis.shape[1]))
        return (empty, xyz.new_zeros((0, len(xyz)))) if return_density else empty
    blocks = torch.stack(blocks)
    return (blocks, torch.stack(densities)) if return_density else blocks


# ----------------------------------------------------------------------------- site evaluation
class OnsiteXCEvaluator:
    """Drop-in for the accepted ``site_eval(g, width, orders)`` with a fused or reference engine.

    ``qgrid(symbol, order)`` returns the species quadrature (``xyz``, ``basis``); ``density_bank``
    maps species to ``AtomicDensity``-like objects (``knots``, ``coeff``, callable); the potential
    is ``pz81_potential(v_and_dv)`` from the accepted ``v_and_dv`` (default: the in-repo
    ``lda_pz81_v_dv_torch``, the same formula). Atoms are grouped by (species, order) and each
    group is evaluated in one fused launch; ``engine='reference'`` runs the accepted per-atom
    loop on exactly the same neighbours and quadrature objects.

    The packed density bank of the fused engine is bound to the content identity of
    ``density_bank`` (:func:`density_identity`, checked on every use without reading arrays).
    ``density_policy='rebuild'`` repacks when a species density was replaced or edited in place
    (counted in ``bank_rebuilds``); ``'fail'`` raises instead, for banks meant to be immutable.
    ``invalidate()`` drops the packed bank explicitly.
    """

    def __init__(self, qgrid, density_bank, *, v_and_dv=None, potential=None, radius=LEGACY_RADIUS_BOHR,
                 engine='fused', device='cuda', chunk_bytes=1 << 30, rho_bytes=1 << 30, prune=True, density_policy='rebuild',
                 topology_library=None):
        if engine not in ('fused', 'reference'):
            raise ValueError("engine must be 'fused' or 'reference'")
        if density_policy not in ('rebuild', 'fail'):
            raise ValueError("density_policy must be 'rebuild' or 'fail'")
        if potential is None:
            if v_and_dv is None:
                from .envxc import lda_pz81_v_dv_torch as v_and_dv
            potential = pz81_potential(v_and_dv)
        self.qgrid, self.density_bank, self.potential = qgrid, density_bank, potential
        self.radius, self.engine, self.device, self.chunk_bytes = float(radius), engine, torch.device(device), chunk_bytes
        self.rho_bytes, self.prune = int(rho_bytes), bool(prune)   # fused engine: density bytes per launch; exact pair skipping
        self.topology_library = topology_library                   # native accepted-order enumeration when set
        self.density_policy = density_policy
        self._bank = None
        self.bank_rebuilds = 0
        self.last_stats = None

    def invalidate(self):
        """Drop the packed density bank; the next use packs the current ``density_bank`` content."""
        self._bank = None

    def bank(self):
        if self._bank is None:
            self._bank = PackedDensityBank(self.density_bank, self.device)
        elif not self._bank.matches(self.density_bank):
            if self.density_policy == 'fail':
                raise RuntimeError('the density bank changed after the packed bank was built (a species density was '
                                   "replaced or edited in place); call invalidate() or use density_policy='rebuild'")
            self._bank = PackedDensityBank(self.density_bank, self.device)
            self.bank_rebuilds += 1
        return self._bank

    def neighbors(self, g):
        if self.topology_library is not None:
            return onsite_neighbor_lists_native(g, self.radius, library=self.topology_library)
        return onsite_neighbor_lists(g, self.radius)

    def blocks(self, quadrature, neighbors, *, return_density=False, stats=None):
        if self.engine == 'fused':
            return fused_onsite_blocks(quadrature, neighbors, self.bank(), self.potential, chunk_bytes=self.chunk_bytes, rho_bytes=self.rho_bytes,
                                       return_density=return_density, prune=self.prune, stats=stats)
        return reference_onsite_blocks(quadrature, neighbors, self.density_bank, self.potential, return_density=return_density)

    def __call__(self, g, width, orders, *, neighbors=None):
        symbols = list(g['symbols'])
        if len(orders) != len(symbols):
            raise ValueError('one quadrature order per atom is required')
        if neighbors is None:
            neighbors = self.neighbors(g)
        if len(neighbors) != len(symbols):
            raise ValueError('one neighbour dict per atom is required')
        out = torch.zeros((len(symbols), width, width), device=self.device, dtype=torch.float64)
        groups = {}
        for i, (s, order) in enumerate(zip(symbols, orders)):
            groups.setdefault((s, tuple(int(x) for x in order)), []).append(i)
        stats = {'atoms': len(symbols), 'groups': len(groups), 'points': 0, 'neighbours': 0, 'pairs': 0, 'candidates': 0, 'candidate_pairs': 0}
        for (s, order), ids in groups.items():
            quadrature = self.qgrid(s, order)
            group = [neighbors[i] for i in ids]
            group_stats = {}
            blocks = self.blocks(quadrature, group, stats=group_stats)
            n = blocks.shape[1]
            if n > width:
                raise ValueError(f'onsite block of {s!r} ({n}) exceeds the AO width {width}')
            out[torch.as_tensor(ids, device=self.device), :n, :n] = blocks
            counts = [sum(len(p) for p in grouped.values()) for grouped in group]
            candidates = group_stats.get('candidates', sum(counts))
            stats['points'] += len(quadrature.xyz) * len(ids); stats['neighbours'] += sum(counts); stats['candidates'] += candidates
            stats['pairs'] += len(quadrature.xyz) * sum(counts); stats['candidate_pairs'] += len(quadrature.xyz) * candidates
        self.last_stats = stats
        return out

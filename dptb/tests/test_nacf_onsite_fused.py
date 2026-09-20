"""Fused batched onsite XC: neighbourhoods, spline boundary rules, grouping, and CUDA parity."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.interpolate import CubicSpline
from dptb.nacf.onsite import (PRUNE_MARGIN_BOHR, OnsiteXCEvaluator, PackedDensityBank, SplineDensity, fused_onsite_blocks,
                              fused_onsite_density, onsite_candidates, onsite_neighbor_lists, pz81_potential, reference_onsite_blocks)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


def reference_neighbors(g, i, radius):
    """The accepted per-atom enumerator (onsite_neighbors_shared.neighbors_for)."""
    from dptb.data.interfaces.p2_batch import VectorizedNearbyImageEnumerator
    pos = np.asarray(g['positions_bohr']); origin = pos[i]
    enum = VectorizedNearbyImageEnumerator(np.asarray(g['cell_bohr'])); grouped = {}
    for s, p in zip(g['symbols'], pos):
        _, centers = enum.query_arrays(p, origin, radius); delta = centers - origin
        delta = delta[np.linalg.norm(delta, axis=1) < radius]
        if len(delta): grouped.setdefault(s, []).append(delta)
    return {s: np.concatenate(parts, axis=0) for s, parts in grouped.items()}


def synthetic_density(device, nlcc=True, seed=0):
    """Strictly increasing knots with a nonuniform tail; valence and optional NLCC channels."""
    rng = np.random.default_rng(seed)
    knots = np.concatenate(([0.0], np.sort(rng.uniform(0.05, 4.0, 30)), [4.5]))
    valence = np.exp(-knots) * (1 + 0.3 * np.sin(3 * knots))
    channels = [CubicSpline(knots, valence).c]
    if nlcc:
        channels.append(CubicSpline(knots, 0.2 * np.exp(-2 * knots) - 0.05).c)   # crosses zero: exercises the clamp
    knots_t = torch.tensor(knots, device=device, dtype=torch.float64)
    return SplineDensity(knots_t, [torch.tensor(c, device=device, dtype=torch.float64) for c in channels])


def quadrature(device, points=257, norb=5, seed=1):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(points, 3)) * 1.3
    basis = rng.normal(size=(points, norb))
    return SimpleNamespace(xyz=torch.tensor(xyz, device=device, dtype=torch.float64), basis=torch.tensor(basis, device=device, dtype=torch.float64))


@pytest.mark.parametrize('cell', [np.eye(3) * 2.5, np.array([[2.6, 0, 0], [.4, 2.9, 0], [-.3, .5, 3.3]])])
def test_neighbor_lists_match_accepted_enumerator_order_and_include_origin(cell):
    pos = np.array([[.1, .2, .3], [1., .2, .5], [.5, .7, .8], [2.4, 1.1, .2]])
    g = dict(symbols=['X', 'Y', 'X', 'Z'], positions_bohr=pos, cell_bohr=cell, pbc=[True] * 3)
    got = onsite_neighbor_lists(g, 4.7)
    for i in range(len(pos)):
        want = reference_neighbors(g, i, 4.7)
        assert list(got[i]) == list(want)
        for s in want:
            np.testing.assert_allclose(got[i][s], want[s], rtol=0, atol=1e-12)
        own = got[i][g['symbols'][i]]
        assert (np.linalg.norm(own, axis=1) < 1e-12).sum() == 1
    subset = onsite_neighbor_lists(g, 4.7, atoms=[2])
    assert len(subset) == 1 and all(np.array_equal(subset[0][s], got[2][s]) for s in got[2])


def test_neighbor_lists_respect_open_boundaries_and_reject_bad_geometry():
    g = dict(symbols=['X', 'X'], positions_bohr=[[0., 0, 0], [1., 0, 0]], cell_bohr=np.eye(3) * 3., pbc=[False, False, False])
    got = onsite_neighbor_lists(g, 10.)
    assert got[0]['X'].shape == (2, 3) and got[1]['X'].shape == (2, 3)
    with pytest.raises(ValueError):
        onsite_neighbor_lists(dict(symbols=['X'], positions_bohr=[[0., 0, 0]], cell_bohr=np.eye(3)), 0.)
    with pytest.raises(ValueError):
        onsite_neighbor_lists(dict(symbols=['X', 'Y'], positions_bohr=[[0., 0, 0]], cell_bohr=np.eye(3)), 1.)


def test_spline_density_boundary_rules_match_scipy():
    density = synthetic_density('cpu')
    knots = density.knots.numpy()
    valence = CubicSpline(knots, np.exp(-knots) * (1 + 0.3 * np.sin(3 * knots)))
    core = CubicSpline(knots, 0.2 * np.exp(-2 * knots) - 0.05)
    r = torch.tensor([knots[0], knots[1], 0.5 * (knots[3] + knots[4]), knots[-2], knots[-1], knots[-1] + 1e-12, 9.0, -1.0, 1e-300], dtype=torch.float64)
    got = density(r).numpy()
    rr = np.clip(r.numpy(), knots[0], knots[-1])
    want = np.where(r.numpy() > knots[-1], 0., np.maximum(valence(rr), 0) + np.maximum(core(rr), 0))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-13)
    assert got[5] == 0 and got[6] == 0 and got[7] == got[0] == got[8]


def test_packed_bank_layout_and_validation():
    a, b = synthetic_density('cpu', nlcc=True, seed=3), synthetic_density('cpu', nlcc=False, seed=4)
    bank = PackedDensityBank({'A': a, 'B': b}, 'cpu')
    assert bank.species == ['A', 'B'] and bank.channels.tolist() == [2, 1]
    assert bank.knots_f32.dtype == torch.float32 and bank.knots_f32.numel() == bank.knots.numel()
    assert bank.knot_ptr.tolist() == [0, len(a.knots), len(a.knots) + len(b.knots)]
    assert bank.coeff_ptr.tolist() == [0, 2 * 4 * (len(a.knots) - 1), 2 * 4 * (len(a.knots) - 1) + 4 * (len(b.knots) - 1)]
    torch.testing.assert_close(bank.coeff[: 4 * (len(a.knots) - 1)].reshape(4, -1), a.coeff[0], atol=0, rtol=0)
    bad = SplineDensity(torch.tensor([0., 1., 1.]), [torch.zeros(4, 2, dtype=torch.float64)])
    with pytest.raises(ValueError, match='strictly increasing'):
        PackedDensityBank({'bad': bad}, 'cpu')
    with pytest.raises(ValueError, match='coefficients'):
        PackedDensityBank({'bad': SplineDensity(torch.tensor([0., 1.]), [torch.zeros(3, 1, dtype=torch.float64)])}, 'cpu')


def test_reference_engine_groups_atoms_and_fills_width():
    device = 'cpu'
    density = {'X': synthetic_density(device), 'Y': synthetic_density(device, nlcc=False, seed=7)}
    quads = {('X', (2, 2, 2)): quadrature(device, 101, 4, 1), ('Y', (2, 2, 2)): quadrature(device, 77, 3, 2), ('X', (3, 3, 3)): quadrature(device, 130, 4, 5)}
    g = dict(symbols=['X', 'Y', 'X'], positions_bohr=[[0., 0, 0], [1.1, .2, .1], [.3, 1.4, .8]], cell_bohr=np.eye(3) * 3.1)
    orders = [(2, 2, 2), (2, 2, 2), (3, 3, 3)]
    potential = pz81_potential(lambda n: (torch.log1p(n), None))
    ev = OnsiteXCEvaluator(lambda s, o: quads[(s, o)], density, potential=potential, radius=3.0, engine='reference', device=device)
    got = ev(g, 5, orders)
    neighbors = onsite_neighbor_lists(g, 3.0)
    for i, (s, order) in enumerate(zip(g['symbols'], orders)):
        q = quads[(s, order)]
        want = reference_onsite_blocks(q, [neighbors[i]], density, potential)[0]
        n = want.shape[0]
        torch.testing.assert_close(got[i, :n, :n], want, atol=0, rtol=0)
        assert torch.count_nonzero(got[i, n:]) == 0 and torch.count_nonzero(got[i, :, n:]) == 0
    assert ev.last_stats['groups'] == 3 and ev.last_stats['atoms'] == 3
    with pytest.raises(ValueError):
        ev(g, 5, orders[:2])
    with pytest.raises(ValueError):
        ev(g, 3, orders)


def test_potential_rule_zero_below_floor_and_floor_clamp():
    calls = []
    def v_and_dv(n):
        calls.append(n.clone()); return (n * 2, n)
    potential = pz81_potential(v_and_dv)
    rho = torch.tensor([0., 1e-21, 1e-20, 2e-20, 1.], dtype=torch.float64)
    torch.testing.assert_close(potential(rho), torch.tensor([0., 0., 0., 4e-20, 2.], dtype=torch.float64), atol=0, rtol=0)
    assert calls[0].min().item() == 1e-20


def test_fused_density_rejects_cpu_and_empty_atoms():
    density = {'X': synthetic_density('cpu')}
    bank = PackedDensityBank(density, 'cpu')
    q = quadrature('cpu', 11, 2)
    with pytest.raises(ValueError, match='CUDA'):
        fused_onsite_density(q.xyz, [{'X': np.zeros((1, 3))}], bank)
    if torch.cuda.is_available():
        gbank = PackedDensityBank({'X': synthetic_density('cuda')}, 'cuda')
        gq = quadrature('cuda', 11, 2)
        with pytest.raises(ValueError, match='no onsite neighbours'):
            fused_onsite_density(gq.xyz, [{'X': np.zeros((0, 3))}], gbank)
        with pytest.raises(ValueError, match='no density'):
            fused_onsite_density(gq.xyz, [{'Q': np.zeros((1, 3))}], gbank)
        with pytest.raises(ValueError, match='FP64'):
            fused_onsite_density(gq.xyz.float(), [{'X': np.zeros((1, 3))}], gbank)


@cuda
def test_fused_matches_reference_on_boundary_cases_multi_species_and_chunks():
    """Knot ends, outside support, NLCC clamping, mixed species order, >16 neighbours, several atoms."""
    device = 'cuda'
    density = {'A': synthetic_density(device, nlcc=True, seed=11), 'B': synthetic_density(device, nlcc=False, seed=12)}
    bank = PackedDensityBank(density, device)
    q = quadrature(device, 1023, 6, 21)
    rng = np.random.default_rng(5)
    ka = density['A'].knots.cpu().numpy(); kb = density['B'].knots.cpu().numpy()
    x0 = q.xyz[0].cpu().numpy()
    # Neighbours placed so that point 0 sits exactly on the first/last knot, beyond support, and at a random distance.
    special = np.stack([x0 - np.array([ka[0], 0, 0]), x0 - np.array([ka[-1], 0, 0]), x0 - np.array([ka[-1] + 1e-9, 0, 0]), x0 - np.array([0, 0, 0.7])])
    atoms = [
        {'A': np.concatenate([special, rng.normal(size=(37, 3)) * 2.0]), 'B': rng.normal(size=(19, 3)) * 1.5},
        {'B': rng.normal(size=(5, 3)) * 1.5, 'A': rng.normal(size=(16, 3)) * 2.0},
        {'A': np.zeros((1, 3))},
        {'B': np.concatenate([np.zeros((1, 3)), rng.normal(size=(60, 3)) * 30.0])},   # mostly outside support
    ]
    potential = pz81_potential(lambda n: ((3.0 / (4 * math.pi * n)) ** (1.0 / 3.0), None))
    got, rho = fused_onsite_blocks(q, atoms, bank, potential, chunk_bytes=q.basis.numel() * 8 * 2, return_density=True)
    want, rho_ref = reference_onsite_blocks(q, atoms, density, potential, return_density=True)
    assert torch.isfinite(got).all() and torch.isfinite(rho).all()
    torch.testing.assert_close(rho, rho_ref, atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(got, got.transpose(-1, -2), atol=1e-11, rtol=0)
    # Point 0 of atom 0: knot-end and beyond-support contributions equal the direct spline rule.
    d = torch.linalg.vector_norm(q.xyz[0][None] - torch.tensor(special, device=device), dim=-1)
    direct = density['A'](d)
    assert direct[2].item() == 0.0 and direct[0].item() >= 0 and direct[1].item() >= 0
    # Atom 2 has exactly one neighbour at the origin: rho is the single-neighbour density at |xyz|
    # (vector_norm and the kernel's explicit sqrt(dx*dx+dy*dy+dz*dz) round differently by one ulp).
    torch.testing.assert_close(rho[2], density['A'](torch.linalg.vector_norm(q.xyz, dim=-1)), atol=1e-15, rtol=1e-14)
    assert fused_onsite_density(q.xyz, [], bank).shape == (0, 1023)


@cuda
def test_evaluator_fused_equals_reference_engine_on_real_shaped_grouping():
    device = 'cuda'
    density = {'X': synthetic_density(device, seed=31), 'Y': synthetic_density(device, nlcc=False, seed=32)}
    quads = {}
    def qgrid(s, order):
        key = (s, tuple(order))
        if key not in quads:
            quads[key] = quadrature(device, 500 + 100 * len(quads), 4 if s == 'X' else 3, len(quads) + 40)
        return quads[key]
    g = dict(symbols=['X', 'Y', 'X', 'Y', 'X'], positions_bohr=np.random.default_rng(9).uniform(0, 3, (5, 3)), cell_bohr=np.eye(3) * 3.3)
    orders = [(2, 2, 2), (2, 2, 2), (3, 3, 3), (2, 2, 2), (2, 2, 2)]
    fused = OnsiteXCEvaluator(qgrid, density, radius=4.0, engine='fused', device=device)
    reference = OnsiteXCEvaluator(qgrid, density, radius=4.0, engine='reference', device=device)
    neighbors = fused.neighbors(g)
    a = fused(g, 4, orders, neighbors=neighbors); b = reference(g, 4, orders, neighbors=neighbors)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    assert fused.last_stats['groups'] == 3 and fused.last_stats['pairs'] == reference.last_stats['pairs']


def test_packed_bank_is_bound_to_density_content_not_species_names():
    """Reviewer reproduction (F2): replacing a species density under the same key, or editing its spline in
    place, must not leave the fused engine on a stale packed bank while the reference engine reads the new
    density. Identity is checked without reading arrays."""
    from dptb.nacf.onsite import density_identity
    knots = torch.tensor([0., 1., 2.], dtype=torch.float64)

    def density(scale):
        return SplineDensity(knots, [torch.tensor(CubicSpline(knots.numpy(), scale * np.array([1., .5, .1])).c)])

    ev = OnsiteXCEvaluator(lambda s, o: None, {'X': density(1.)}, potential=lambda n: n, device='cpu', engine='reference')
    b1 = ev.bank(); c1 = b1.coeff.clone()
    assert ev.bank() is b1 and ev.bank_rebuilds == 0
    ev.density_bank['X'] = density(2.)                       # same key, new content
    b2 = ev.bank()
    assert b2 is not b1 and ev.bank_rebuilds == 1
    assert not torch.equal(c1, b2.coeff)
    torch.testing.assert_close(b2.coeff.reshape(4, -1), ev.density_bank['X'].coeff[0], atol=0, rtol=0)
    assert float(ev.density_bank['X'](torch.tensor([0.], dtype=torch.float64))[0]) == 2.0
    # in-place edit of a coefficient channel (version counter), then of the knots
    ev.density_bank['X'].coeff[0].mul_(0.5)
    b3 = ev.bank(); assert b3 is not b2 and torch.equal(b3.coeff, 0.5 * b2.coeff)
    ev.density_bank['X'].knots[2] = 2.5
    assert ev.bank() is not b3 and ev.bank_rebuilds == 3
    # identity is stable under untouched banks and a fresh dict with the same objects
    ident = density_identity(ev.density_bank)
    assert ident == density_identity(dict(ev.density_bank))
    assert ev.bank() is ev.bank()
    # fail-closed policy for banks meant to be immutable; invalidate() is the explicit way out
    frozen = OnsiteXCEvaluator(lambda s, o: None, {'X': density(1.)}, potential=lambda n: n, device='cpu', engine='reference', density_policy='fail')
    frozen.bank()
    frozen.density_bank['X'] = density(3.)
    with pytest.raises(RuntimeError, match='density bank changed'):
        frozen.bank()
    frozen.invalidate()
    torch.testing.assert_close(frozen.bank().coeff.reshape(4, -1), frozen.density_bank['X'].coeff[0], atol=0, rtol=0)
    with pytest.raises(ValueError):
        OnsiteXCEvaluator(lambda s, o: None, {'X': density(1.)}, potential=lambda n: n, device='cpu', density_policy='ignore')


def test_candidate_selection_is_exact_and_keeps_accepted_positions():
    """Only neighbours with |d| > r_grid + k_last + margin are dropped; survivors keep their per-species list position."""
    a, b = synthetic_density('cpu', nlcc=True, seed=13), synthetic_density('cpu', nlcc=False, seed=14)
    bank = PackedDensityBank({'A': a, 'B': b}, 'cpu')
    np.testing.assert_allclose(bank.last_knot_host, [a.knots[-1].item(), b.knots[-1].item()])
    rng = np.random.default_rng(2)
    atoms = [{'A': rng.normal(size=(37, 3)) * 6.0, 'B': rng.normal(size=(21, 3)) * 6.0}, {'B': rng.normal(size=(5, 3)) * 20.0}, {'A': np.zeros((1, 3))}]
    radius = 1.7
    seg_ptr, segments, cand, local, stats = onsite_candidates(atoms, bank, radius)
    assert stats['neighbours'] == 64 and stats['candidates'] == len(cand) == len(local) and seg_ptr.tolist() == [0, 2, 3, 4]
    row = 0
    for atom, (begin, end) in zip(atoms, zip(seg_ptr[:-1], seg_ptr[1:])):
        for (species, cb, ce), (name, positions) in zip(segments[begin:end], atom.items()):
            assert bank.species[species] == name
            d = np.linalg.norm(positions, axis=1)
            keep = np.flatnonzero(d <= radius + bank.last_knot_host[species] + PRUNE_MARGIN_BOHR)
            assert ce - cb == len(keep)
            np.testing.assert_array_equal(local[cb:ce], keep)
            np.testing.assert_allclose(cand[cb:ce, :3], positions[keep], rtol=0, atol=0)
            np.testing.assert_allclose(cand[cb:ce, 3], d[keep], rtol=0, atol=1e-15)
            row += len(positions)
    assert 0 < len(cand) < 64                              # the 20-Bohr cloud is almost entirely out of reach
    full = onsite_candidates(atoms, bank, radius, prune=False)
    assert full[4]['candidates'] == 64 and np.array_equal(full[3], np.concatenate([np.arange(len(p)) for atom in atoms for p in atom.values()]))


@cuda
def test_pruned_kernel_is_bitwise_the_unpruned_walk_and_chunked_launches_agree():
    device = 'cuda'
    density = {'A': synthetic_density(device, nlcc=True, seed=21), 'B': synthetic_density(device, nlcc=False, seed=22)}
    bank = PackedDensityBank(density, device)
    q = quadrature(device, 2049, 5, 23)
    rng = np.random.default_rng(6)
    # clouds at several scales so that atom-level pruning, in-kernel pair skipping and full evaluation all occur,
    # plus exact boundary placements at k_last +- tiny for point 0
    x0 = q.xyz[0].cpu().numpy(); ka = density['A'].knots.cpu().numpy()
    special = np.stack([x0 - np.array([ka[-1], 0, 0]), x0 - np.array([ka[-1] + 1e-9, 0, 0]), x0 - np.array([ka[-1] - 1e-9, 0, 0])])
    atoms = [{'A': np.concatenate([special, rng.normal(size=(50, 3)) * 2.0, rng.normal(size=(40, 3)) * 8.0]), 'B': rng.normal(size=(30, 3)) * 5.0},
             {'B': np.concatenate([np.zeros((1, 3)), rng.normal(size=(70, 3)) * 12.0])},
             {'A': rng.normal(size=(33, 3)) * 3.0}, {'A': np.zeros((1, 3)), 'B': rng.normal(size=(17, 3)) * 30.0}]
    pruned = fused_onsite_density(q.xyz, atoms, bank)
    plain = fused_onsite_density(q.xyz, atoms, bank, prune=False)
    assert torch.equal(pruned, plain)
    chunked = fused_onsite_density(q.xyz, atoms, bank, rho_bytes=q.xyz.shape[0] * 8)      # one atom per launch
    assert torch.equal(pruned, chunked)
    _, _, _, _, stats = onsite_candidates(atoms, bank, float(torch.linalg.vector_norm(q.xyz, dim=-1).max()))
    assert 0 < stats['candidates'] < stats['neighbours']
    potential = pz81_potential(lambda n: ((3.0 / (4 * math.pi * n)) ** (1.0 / 3.0), None))
    counts = {}
    blocks, rho = fused_onsite_blocks(q, atoms, bank, potential, rho_bytes=q.xyz.shape[0] * 16, return_density=True, stats=counts)
    ref_blocks, ref_rho = reference_onsite_blocks(q, atoms, density, potential, return_density=True)
    assert torch.equal(rho, pruned) and counts == stats
    torch.testing.assert_close(rho, ref_rho, atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(blocks, ref_blocks, atol=1e-12, rtol=1e-12)
    ev = OnsiteXCEvaluator(lambda s, o: q, density, potential=potential, radius=4.0, engine='fused', device=device, rho_bytes=q.xyz.shape[0] * 8)
    g = dict(symbols=['A', 'B', 'A'], positions_bohr=rng.uniform(0, 3, (3, 3)), cell_bohr=np.eye(3) * 3.4)
    got = ev(g, 5, [(1, 1, 1)] * 3)
    assert ev.last_stats['candidates'] <= ev.last_stats['neighbours'] and ev.last_stats['candidate_pairs'] <= ev.last_stats['pairs']
    torch.testing.assert_close(got, OnsiteXCEvaluator(lambda s, o: q, density, potential=potential, radius=4.0, engine='reference', device=device)(g, 5, [(1, 1, 1)] * 3), atol=1e-11, rtol=1e-11)


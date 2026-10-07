"""The vectorized crystal projection must reproduce the reference loop implementation exactly."""
import numpy as np
import pytest
import torch
from ase.build import bulk

pytest.importorskip('spglib')

from dptb.nn.sym_projection import SymmetryProjector, SpglibGroupProvider, _Reynolds
from dptb.tests.sym_helpers import almg3, p1_structure
from dptb.tests.test_sym_projection import fixture, action


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def reference_edge_maps(edge_index, edge_shift, permutations, shifts, w):
    """The pre-vectorization dictionary lookup, kept verbatim as the oracle."""
    lookup = {tuple([i, j, *s]): e for e, ((i, j), s) in enumerate(zip(edge_index.T, edge_shift))}
    edge_maps, reverse = [], []
    for rotation, perm, shift in zip(w, permutations, shifts):
        mapped_shift = edge_shift @ rotation.T + shift[edge_index[1]] - shift[edge_index[0]]
        dest, rev = [], []
        for (i, j), s in zip(edge_index.T, mapped_shift):
            key = (perm[i], perm[j], *s)
            flipped = key not in lookup
            if flipped:
                key = (perm[j], perm[i], *(-s))
            dest.append(lookup[key])
            rev.append(flipped)
        edge_maps.append(dest)
        reverse.append(rev)
    return np.asarray(edge_maps), np.asarray(reverse, dtype=bool)


STRUCTURES = [None, p1_structure(), bulk('Al', 'fcc', a=4.05), bulk('Al', 'bcc', a=3.3, cubic=True)]


@pytest.mark.parametrize('half', [False, True])
@pytest.mark.parametrize('k', range(len(STRUCTURES)))
def test_vectorized_average_and_adjoint_match_loop(k, half):
    idp, data = fixture(STRUCTURES[k], half=half)
    act = action(SymmetryProjector(idp), data)
    gen = torch.Generator().manual_seed(3 + k)
    for part in ('node', 'edge'):
        x = torch.randn(data[part + '_features'].shape, generator=gen)
        for adjoint in (False, True):
            for chunk in (1, 3, 64):
                torch.testing.assert_close(act.average_vec(x, part, adjoint=adjoint, chunk=chunk),
                                           act.average(x, part, adjoint=adjoint), atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize('half', [False, True])
@pytest.mark.parametrize('k', range(len(STRUCTURES)))
def test_vectorized_provider_matches_dictionary_oracle(k, half):
    idp, data = fixture(STRUCTURES[k], half=half)
    import spglib
    pos, cell = data['pos'].numpy(), data['cell'].reshape(3, 3).numpy()
    species = data['atom_types'].flatten().numpy() + 1
    edges, shift = data['edge_index'].numpy(), data['edge_cell_shift'].numpy().astype(np.int64)
    group = SpglibGroupProvider()(pos, cell, species, edges, shift, symprec=1e-3)
    frac = pos @ np.linalg.inv(cell)
    ds = spglib.get_symmetry_dataset((cell, frac, species), symprec=1e-3)
    w, t = np.asarray(ds.rotations), np.asarray(ds.translations)
    shifts = []
    for rotation, translation, perm in zip(w, t, group.atom_permutations):
        delta = (frac @ rotation.T + translation)[:, None] - frac[None]
        s = np.rint(delta)
        shifts.append(s[np.arange(len(species)), perm].astype(np.int64))
    maps, rev = reference_edge_maps(edges, shift, group.atom_permutations, shifts, w)
    assert np.array_equal(group.edge_permutations, maps.reshape(group.edge_permutations.shape))
    assert np.array_equal(group.edge_reversed, rev.reshape(group.edge_reversed.shape))


def test_backward_is_the_exact_loop_adjoint():
    idp, data = fixture()
    act = action(SymmetryProjector(idp), data)
    gen = torch.Generator().manual_seed(11)
    for part in ('node', 'edge'):
        x = torch.randn(data[part + '_features'].shape, generator=gen, requires_grad=True)
        upstream = torch.randn(x.shape, generator=gen)
        (_Reynolds.apply(x, act, part) * upstream).sum().backward()
        torch.testing.assert_close(x.grad, act.average(upstream, part, adjoint=True), atol=1e-13, rtol=1e-12)


def _batch(parts):
    first = parts[0]
    batch = dict(first)
    for k in ('pos', 'atom_types', 'atomic_numbers', 'node_features', 'edge_features', 'edge_type', 'edge_cell_shift'):
        if k in first:
            batch[k] = torch.cat([p[k] for p in parts], dim=0)
    batch['cell'] = torch.stack([p['cell'].reshape(3, 3) for p in parts])
    offsets = np.cumsum([0] + [len(p['pos']) for p in parts[:-1]])
    batch['edge_index'] = torch.cat([p['edge_index'] + int(o) for p, o in zip(parts, offsets)], dim=1)
    batch['batch'] = torch.cat([torch.full((len(p['pos']),), s, dtype=torch.long) for s, p in enumerate(parts)])
    return batch


def test_batched_projection_and_gradient_match_per_structure_loop():
    idp, a = fixture()
    parts = [a, fixture(p1_structure())[1], fixture(bulk('Al', 'bcc', a=3.3, cubic=True))[1], fixture(bulk('Al', 'fcc', a=4.05))[1]]
    projector = SymmetryProjector(idp)
    batch = _batch(parts)
    for part in ('node', 'edge'):
        batch[part + '_features'] = batch[part + '_features'].clone().requires_grad_()
    out = projector(batch)
    gen = torch.Generator().manual_seed(5)
    upstream = {p: torch.randn(out[p + '_features'].shape, generator=gen) for p in ('node', 'edge')}
    sum((out[p + '_features'] * upstream[p]).sum() for p in ('node', 'edge')).backward()
    n_off, e_off = 0, 0
    for data in parts:
        act = action(projector, data)
        n, e = len(data['pos']), data['edge_index'].shape[1]
        for p, lo, hi in (('node', n_off, n_off + n), ('edge', e_off, e_off + e)):
            torch.testing.assert_close(out[p + '_features'][lo:hi].detach(), act.average(data[p + '_features'], p),
                                       atol=1e-13, rtol=1e-12)
            torch.testing.assert_close(batch[p + '_features'].grad[lo:hi], act.average(upstream[p][lo:hi], p, adjoint=True),
                                       atol=1e-13, rtol=1e-12)
        n_off, e_off = n_off + n, e_off + e


def test_open_graph_identity_fallback_is_opt_in():
    idp, data = fixture()
    keep = torch.ones(data['edge_index'].shape[1], dtype=torch.bool)
    keep[0] = False
    open_graph = dict(data, edge_index=data['edge_index'][:, keep], edge_cell_shift=data['edge_cell_shift'][keep],
                      edge_features=data['edge_features'][keep])
    if 'edge_type' in data:
        open_graph['edge_type'] = data['edge_type'][keep]
    with pytest.raises(ValueError, match='symmetry-closed'):
        SymmetryProjector(idp)(open_graph)
    projector = SymmetryProjector(idp, open_graph='identity')
    out = projector(open_graph)
    for part in ('node', 'edge'):
        torch.testing.assert_close(out[part + '_features'], open_graph[part + '_features'], atol=0, rtol=0)
    assert len(projector.fallbacks) == 1 and 'symmetry-closed' in projector.fallbacks[0]
    with pytest.raises(ValueError):
        SymmetryProjector(idp, open_graph='bogus')


def test_parts_projects_only_the_listed_part():
    idp, data = fixture()
    projector = SymmetryProjector(idp)
    full = projector(data)
    for parts in (('node',), ('edge',), ()):
        out = projector(data, parts=parts)
        for part in ('node', 'edge'):
            want = full[part + '_features'] if part in parts else data[part + '_features']
            torch.testing.assert_close(out[part + '_features'], want, atol=0, rtol=0)
    with pytest.raises(ValueError):
        projector(data, parts=('bogus',))


def test_interleaved_rows_and_one_operation_chunks_match_contiguous_batch():
    """Rows of different structures interleaved and group orders not descending: the sorted layout, the shrinking
    active prefix (one operation per chunk) and the gradient must reproduce the contiguous batch."""
    idp, a = fixture()
    parts = [fixture(p1_structure())[1], a, fixture(bulk('Al', 'fcc', a=4.05))[1],
             fixture(bulk('Al', 'bcc', a=3.3, cubic=True))[1]]
    batch = _batch(parts)
    gen = torch.Generator().manual_seed(7)
    n, e = len(batch['pos']), batch['edge_index'].shape[1]
    node_perm, edge_perm = torch.randperm(n, generator=gen), torch.randperm(e, generator=gen)
    new_of_old = torch.empty_like(node_perm)
    new_of_old[node_perm] = torch.arange(n)
    shuffled = dict(batch)
    for k in ('pos', 'atom_types', 'atomic_numbers', 'node_features', 'batch'):
        if k in batch:
            shuffled[k] = batch[k][node_perm]
    for k in ('edge_features', 'edge_type', 'edge_cell_shift'):
        if k in batch:
            shuffled[k] = batch[k][edge_perm]
    shuffled['edge_index'] = new_of_old[batch['edge_index'][:, edge_perm]]
    perm = {'node': node_perm, 'edge': edge_perm}
    reference = SymmetryProjector(idp)
    want = reference(batch)
    projector = SymmetryProjector(idp)
    projector.budget_bytes = 1
    for p in ('node', 'edge'):
        shuffled[p + '_features'] = shuffled[p + '_features'].clone().requires_grad_()
    out = projector(shuffled)
    upstream = {p: torch.randn(out[p + '_features'].shape, generator=gen) for p in ('node', 'edge')}
    sum((out[p + '_features'] * upstream[p]).sum() for p in ('node', 'edge')).backward()
    contiguous_upstream = dict(batch)
    for p in ('node', 'edge'):
        u = torch.empty_like(upstream[p])
        u[perm[p]] = upstream[p]
        contiguous_upstream[p + '_features'] = u
    want_grad = reference(contiguous_upstream)
    for p in ('node', 'edge'):
        torch.testing.assert_close(out[p + '_features'].detach(), want[p + '_features'][perm[p]], atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(shuffled[p + '_features'].grad, want_grad[p + '_features'][perm[p]],
                                   atol=1e-13, rtol=1e-12)

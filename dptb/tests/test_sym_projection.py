"""Numerical tests for real AO crystal Reynolds averaging."""
import copy
import importlib
import os
import sys

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from e3nn import o3

pytest.importorskip('spglib')

from dptb.data import AtomicData
from dptb.data.transforms import OrbitalMapper
from dptb.nn.sym_projection import SymmetryProjector, SpglibGroupProvider
from dptb.tests.sym_helpers import almg3, p1_structure


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def fixture(atoms=None, half=True):
    idp = OrbitalMapper({'Al': '1s1p1d', 'Mg': '1s1p'}, method='e3tb',
                        has_soc=True, nextham_uureal_mask=True, device='cpu')
    atoms = almg3() if atoms is None else atoms.copy()
    # Generic orientation exercises dense orbital rotations, not only axes.
    rotation = o3.angles_to_matrix(torch.tensor(.37), torch.tensor(.81), torch.tensor(.23)).numpy()
    atoms.positions = atoms.positions @ rotation.T
    atoms.cell = atoms.cell.array @ rotation.T
    atoms.positions += np.array([.19, -.31, .12])
    data = AtomicData.to_AtomicDataDict(AtomicData.from_ase(atoms, r_max=4.5))
    if half:
        edges, shift = data['edge_index'].numpy(), data['edge_cell_shift'].numpy().astype(int)
        keep = torch.tensor([bool(tuple([int(i), int(j), *s]) <= tuple([int(j), int(i), *(-s)]))
                             for (i, j), s in zip(edges.T, shift)], dtype=torch.bool)
        data['edge_index'] = data['edge_index'][:, keep]
        data['edge_cell_shift'] = data['edge_cell_shift'][keep]
    if data['edge_index'].shape[1] == 0:
        data['edge_type'] = torch.empty((0, 1), dtype=torch.long)
    data = idp(data)
    generator = torch.Generator().manual_seed(17)
    for part, n in [('node', len(atoms)), ('edge', data['edge_index'].shape[1])]:
        data[part + '_features'] = torch.randn(n, idp.reduced_matrix_element, generator=generator)
    return idp, data


def action(projector, data):
    return projector.action(data['pos'].numpy(), data['cell'].reshape(3, 3).numpy(),
                            data['atom_types'].flatten().numpy() + 1, data['edge_index'].numpy(),
                            data['edge_cell_shift'].numpy().astype(int), device='cpu', dtype=torch.float64)


@pytest.mark.parametrize('half', [False, True])
def test_idempotence_invariance_and_adjoint_finite_difference(half):
    idp, data = fixture(half=half)
    projector = SymmetryProjector(idp, symprec=1e-7)
    projected = projector(data)
    twice = projector(projected)
    representation = action(projector, data)
    assert len(representation.group.rotations) == 48
    assert bool(representation.reverse.any()) == half
    generator = torch.Generator().manual_seed(19)
    for part in ('node', 'edge'):
        key = part + '_features'
        torch.testing.assert_close(projected[key], twice[key], atol=2e-11, rtol=2e-11)
        for g in range(48):
            torch.testing.assert_close(representation.act(projected[key], part, g), projected[key],
                                       atol=2e-11, rtol=2e-11)
        x = data[key].clone().requires_grad_()
        pred = projector(dict(data, **{key: x}))[key]
        target = torch.randn(pred.shape, generator=generator)
        target_projected = projector(dict(data, **{key: target}))[key]
        torch.testing.assert_close((pred * target).sum(), (x * target_projected).sum(), atol=2e-9, rtol=2e-11)
        loss = (pred - target).square().mean()
        grad, = torch.autograd.grad(loss, x)
        direction = torch.randn(x.shape, generator=generator)
        epsilon = 1e-5
        def evaluate(value):
            return (projector(dict(data, **{key: value}))[key] - target).square().mean()
        finite = (evaluate(x.detach() + epsilon * direction) - evaluate(x.detach() - epsilon * direction)) / (2 * epsilon)
        torch.testing.assert_close((grad * direction).sum(), finite, atol=2e-10, rtol=2e-7)


@pytest.mark.parametrize('atoms', [None, bulk('Mg', 'hcp', a=3.2, c=5.2)])
def test_numpy_groupaction_reference(atoms):
    root = os.environ.get('DPTB_GROUPACTION_REF')
    if not root:
        pytest.skip('set DPTB_GROUPACTION_REF to the read-only diagnostic tools directory')
    sys.path.insert(0, root)
    try:
        reference = importlib.import_module('symmetry_ops')
        idp, data = fixture(atoms)
        layout = reference.Layout(idp)
        group = reference.GroupAction(layout, data['pos'].numpy(), data['cell'].reshape(3,3).numpy(),
                                       data['atom_types'].flatten().numpy() + 1, data['edge_index'].numpy(),
                                       data['edge_cell_shift'].numpy(), prec=1e-7)
        got = SymmetryProjector(idp, symprec=1e-7)(data)
        for part in ('node', 'edge'):
            expected = layout.pack(group.average(layout.unpack(data[part + '_features'].numpy()), part))
            np.testing.assert_allclose(got[part + '_features'].numpy(), expected, atol=1e-9, rtol=1e-9)
    finally:
        sys.path.remove(root)


def test_batch_cache_and_sub_micro_geometry():
    idp, a = fixture()
    _, b = fixture(p1_structure())
    calls = []
    def provider(*args, **kwargs):
        calls.append(1)
        return SpglibGroupProvider()(*args, **kwargs)
    projector = SymmetryProjector(idp, provider=provider, cache_size=3)
    batch = dict(a)
    for k in ('pos', 'atom_types', 'atomic_numbers', 'node_features', 'edge_features', 'edge_type', 'edge_cell_shift'):
        if k in a:
            batch[k] = torch.cat([a[k], b[k]], dim=0)
    batch['cell'] = torch.stack([a['cell'].reshape(3, 3), b['cell'].reshape(3, 3)])
    batch['edge_index'] = torch.cat([a['edge_index'], b['edge_index'] + len(a['pos'])], dim=1)
    batch['batch'] = torch.cat([torch.zeros(len(a['pos']), dtype=torch.long), torch.ones(len(b['pos']), dtype=torch.long)])
    got = projector(batch)
    aa, bb = projector(a), projector(b)
    for part in ('node', 'edge'):
        key = part + '_features'
        # The batched operator pads shorter groups with zero-weight identities, so the summation order (not the
        # operator) differs from a single-structure call: agreement is to rounding (measured 1e-16), not bitwise.
        torch.testing.assert_close(got[key], torch.cat([aa[key], bb[key]]), atol=1e-13, rtol=1e-12)
    assert len(calls) == 2
    changed = dict(a, pos=a['pos'].clone())
    changed['pos'][0, 0] += 1e-9
    projector(changed)
    assert len(calls) == 3
    projector.symprec = 1e-4
    projector(a)
    assert len(calls) == 4 and len(projector.cache) == 3


def test_fail_closed_on_duplicate_or_open_graph():
    idp, data = fixture()
    projector = SymmetryProjector(idp)
    broken = dict(data, edge_index=data['edge_index'][:, 1:], edge_cell_shift=data['edge_cell_shift'][1:],
                  edge_features=data['edge_features'][1:])
    with pytest.raises(ValueError, match='symmetry-closed'):
        projector(broken)
    duplicated = dict(data, edge_index=torch.cat([data['edge_index'], data['edge_index'][:, :1]], dim=1),
                      edge_cell_shift=torch.cat([data['edge_cell_shift'], data['edge_cell_shift'][:1]]),
                      edge_features=torch.cat([data['edge_features'], data['edge_features'][:1]]))
    with pytest.raises(ValueError, match='duplicate'):
        projector(duplicated)






def test_float64_rotations_do_not_depend_on_default_dtype():
    idp, data = fixture()
    expected = SymmetryProjector(idp)(data)
    torch.set_default_dtype(torch.float32)
    try:
        got = SymmetryProjector(idp)(data)
        for part in ('node', 'edge'):
            torch.testing.assert_close(got[part + '_features'], expected[part + '_features'], atol=0, rtol=0)
    finally:
        torch.set_default_dtype(torch.float64)


def test_projection_preserves_empty_edges():
    idp, data = fixture(Atoms('Mg', positions=[[0, 0, 0]], cell=np.eye(3)*10, pbc=True))
    assert data['edge_features'].shape[0] == 0
    projected = SymmetryProjector(idp)(data)
    assert projected['edge_features'].shape == data['edge_features'].shape
    assert torch.isfinite(projected['node_features']).all()

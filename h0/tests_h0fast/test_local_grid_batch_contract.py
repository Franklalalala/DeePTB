"""Public batch wrapper parity with independent CPU periodic contractions."""
from dataclasses import replace

import numpy as np
import pytest

from h0rebuild.models import BlockKey, OrbitalBasis, OrbitalChannel
try:
    from h0rebuild.periodic_collocation import PeriodicFFTGridAOCache
    from h0rebuild.radial import OrbitalEvaluator
except ImportError as _error:
    pytest.skip(f"this scipy predates sph_harm_y, which h0rebuild.harmonics needs: {_error}", allow_module_level=True)
from h0rebuild.reciprocal import PeriodicField, reciprocal_grid


@pytest.mark.h0_extension('_cuda_local_grid')
@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_batch_matches_cpu_with_optional_spin_z(spin, empty):
    from h0rebuild.cuda_local_grid import CudaPeriodicFFTGridAOCache

    cell = np.array([[4., 0., 0.], [.8, 4., 0.], [.2, .5, 4.]])
    shape = (12, 12, 12)
    g, _ = reciprocal_grid(cell, shape)
    xyz = np.indices(shape)
    values = np.cos(xyz[0] * .3) + np.sin(xyz[1] * .6)
    zero = np.zeros(shape)
    field = PeriodicField(
        cell, values, zero, zero, zero, zero, zero.astype(complex), g, {}
    )
    zfield = replace(field, values_ry=values * .37 + 1.) if spin else None
    mesh = np.arange(31) * .1
    radial = np.exp(-mesh) * (1 - mesh / 3) ** 2
    basis = OrbitalBasis(
        "X", 100., mesh, .1,
        [OrbitalChannel(0, 0, radial), OrbitalChannel(1, 0, mesh * radial)],
    )
    evaluator = OrbitalEvaluator(basis)
    positions = np.array([[-1e-7, .5, .3], [2., 1., .5]])
    indices = [] if empty else [
        (0, 0, (0, 0, 0)), (0, 1, (0, 0, 0)),
        (1, 0, (1, 0, 0)), (0, 1, (-1, 0, 0)), (0, 1, (9, 0, 0)),
    ]
    pairs = [
        (i, j, image, positions[i], positions[j] + np.array(image) @ cell)
        for i, j, image in indices
    ]
    cache = CudaPeriodicFFTGridAOCache(field, spin_z_field=zfield)
    blocks, zblocks = cache.contract_pairs_batch(
        pairs, ["X", "X"], {"X": evaluator}, positions
    )
    assert set(blocks) == {BlockKey(i, j, image) for i, j, image in indices}
    if spin:
        assert set(zblocks) == set(blocks)
    else:
        assert zblocks is None
    cpu = PeriodicFFTGridAOCache(field)
    zcpu = PeriodicFFTGridAOCache(zfield) if spin else None
    for i, j, image in indices:
        key = BlockKey(i, j, image)
        reference = cpu.contract_pair(evaluator, evaluator, positions[i], positions[j], image)
        np.testing.assert_allclose(blocks[key], reference, atol=1e-10, rtol=1e-10)
        if spin:
            zreference = zcpu.contract_pair(evaluator, evaluator, positions[i], positions[j], image)
            np.testing.assert_allclose(zblocks[key], zreference, atol=1e-10, rtol=1e-10)
    if not empty:
        assert np.any(blocks[BlockKey(0, 0, (0, 0, 0))] != 0)
        if spin:
            assert np.any(zblocks[BlockKey(0, 0, (0, 0, 0))] != 0)

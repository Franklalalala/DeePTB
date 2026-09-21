import unittest

import numpy as np

from h0rebuild.models import OrbitalBasis, Projector, SpeciesData, UPFData
from h0rebuild.soc_reference import align_soc_projectors


def species(samples):
    r = np.arange(len(samples), dtype=float)
    upf = UPFData(
        element='X', z_valence=1., r=r, rab=np.ones_like(r),
        vloc_ry=-np.ones_like(r), rhoatom_q=np.ones_like(r),
        dij_ry=np.array([[2.]]),
        projectors=[Projector(0, 0, .5, np.array(samples, dtype=float), len(r), 6.)],
        has_so=True,
    )
    return {'X': SpeciesData(OrbitalBasis('X', 100., r, 1., []), upf)}


class SOCReferenceTests(unittest.TestCase):
    def test_abacus_sample_count_and_threshold(self):
        # Independent expected vectors exercise odd/even final indices and
        # the strict threshold, including ABACUS's all-zero fallback.
        cases = [
            ([0., 1., 2., 0., 0.], 3, [0., 1., 2., 0., 0.]),
            ([0., 1., 2., 3., 0.], 3, [0., 1., 2., 0., 0.]),
            ([0., 1., 2., 1e-10, -1e-10], 3, [0., 1., 2., 0., 0.]),
            ([0., 1., 2., 0., -2e-10], 5, [0., 1., 2., 0., -2e-10]),
            ([0., 0., 0., 0.], 5, [0., 0., 0., 0.]),
        ]
        for samples, count, expected in cases:
            with self.subTest(samples=samples):
                actual = align_soc_projectors(species(samples))['X'].upf.projectors[0]
                self.assertEqual(actual.cutoff_index, count)
                np.testing.assert_array_equal(actual.radial_u, expected)

    def test_preserves_original_species_and_physics(self):
        original = species([0., 1., 2., 3., 0.])
        aligned = align_soc_projectors(original)
        source, result = original['X'], aligned['X']
        np.testing.assert_array_equal(source.upf.projectors[0].radial_u, [0., 1., 2., 3., 0.])
        self.assertEqual(source.upf.projectors[0].cutoff_index, 5)
        self.assertEqual(result.upf.projectors[0].cutoff_radius, 6.)
        self.assertEqual(result.upf.projectors[0].j, .5)
        self.assertIs(result.orb, source.orb)
        for name in ('r', 'rab', 'vloc_ry', 'rhoatom_q', 'dij_ry'):
            np.testing.assert_array_equal(getattr(result.upf, name), getattr(source.upf, name))
        result.upf.projectors[0].radial_u[1] = 9.
        self.assertEqual(source.upf.projectors[0].radial_u[1], 1.)


if __name__ == '__main__':
    unittest.main()

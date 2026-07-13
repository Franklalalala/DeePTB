import json
from pathlib import Path

import numpy as np

from dptb.data.interfaces.p2_table import (
    P2_TABLE_SCHEMA,
    P2TableAssembler,
    P2TableStore,
    RadialBlockTable,
    RealHarmonicRotator,
    rotation_z_to,
)


def test_real_harmonic_rotations_are_orthogonal_through_f_shell():
    rotator = RealHarmonicRotator(3)
    q = rotation_z_to(np.asarray([0.31, -0.42, 0.85]))
    for l in range(4):
        matrix = rotator.matrix(l, q)
        np.testing.assert_allclose(
            matrix @ matrix.T,
            np.eye(2 * l + 1),
            atol=2.0e-12,
            rtol=2.0e-12,
        )
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=2.0e-12)


def test_radial_table_rotates_s_to_p_axis_channel():
    distances = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
    values = np.zeros((len(distances), 1, 3), dtype=np.float64)
    # ABACUS p order is (m=0,+1,-1), i.e. z, x-like, y-like with its signs.
    values[:, 0, 0] = 2.0
    table = RadialBlockTable(
        distances=distances,
        values=values,
        left_shells=(0,),
        right_shells=(1,),
        support_bohr=2.0,
    )
    along_z = table.evaluate(np.asarray([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(along_z, [[2.0, 0.0, 0.0]], atol=2.0e-12)
    along_x = table.evaluate(np.asarray([1.0, 0.0, 0.0]))
    assert np.isclose(np.linalg.norm(along_x), 2.0, atol=2.0e-12)
    assert abs(along_x[0, 1]) > 1.999999999
    assert abs(along_x[0, 0]) < 2.0e-12
    assert abs(along_x[0, 2]) < 2.0e-12


def _write_synthetic_table(root: Path) -> None:
    (root / "base").mkdir(parents=True)
    (root / "projector").mkdir()
    (root / "species").mkdir()
    distances_base = np.linspace(0.0, 4.0, 9)
    base = np.ones((len(distances_base), 1, 1), dtype=np.float32)
    base[-1] = 0.0
    np.savez(
        root / "base" / "X__X.npz",
        distances=distances_base,
        values=base,
        left_shells=np.asarray([0], dtype=np.int16),
        right_shells=np.asarray([0], dtype=np.int16),
        support_bohr=np.asarray(4.0),
        onsite_base=np.asarray([[3.0]]),
    )
    distances_projector = np.linspace(0.0, 3.0, 7)
    projector = np.full((len(distances_projector), 1, 1), 0.5, dtype=np.float32)
    projector[-1] = 0.0
    np.savez(
        root / "projector" / "X__X.npz",
        distances=distances_projector,
        values=projector,
        left_shells=np.asarray([0], dtype=np.int16),
        right_shells=np.asarray([0], dtype=np.int16),
        support_bohr=np.asarray(3.0),
    )
    np.savez(
        root / "species" / "X.npz",
        d_eff=np.asarray([[2.0]]),
        onsite_base=np.asarray([[3.0]]),
    )
    manifest = {
        "schema": P2_TABLE_SCHEMA,
        "length_unit": "bohr",
        "energy_unit": "Ry",
        "complete": True,
        "species": {
            "X": {
                "orbital_shells": [0],
                "orbital_norb": 1,
                "orbital_cutoff_bohr": 2.0,
                "projector_shells": [0],
                "projector_norb": 1,
                "projector_cutoffs_bohr": [1.0],
                "projector_max_cutoff_bohr": 1.0,
                "array_path": "species/X.npz",
            }
        },
        "base_tables": {
            "X|X": {
                "path": "base/X__X.npz",
                "interpolation": "linear",
            }
        },
        "projector_tables": {
            "X|X": {
                "path": "projector/X__X.npz",
                "interpolation": "linear",
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_p2_table_assembler_counts_onsite_vna_once_and_all_projector_centres(tmp_path):
    _write_synthetic_table(tmp_path)
    assembler = P2TableAssembler(P2TableStore(tmp_path, verify_checksums=False))
    cell = np.eye(3) * 20.0

    onsite = assembler.assemble_block(
        symbols=["X"],
        positions_bohr=np.asarray([[0.0, 0.0, 0.0]]),
        cell_bohr=cell,
        i=0,
        j=0,
    )
    # Explicit onsite base 3.0 plus 0.5 * D(2.0) * 0.5 from the home projector.
    np.testing.assert_allclose(onsite, [[3.5]], atol=1.0e-12)

    hopping = assembler.assemble_block(
        symbols=["X", "X"],
        positions_bohr=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell_bohr=cell,
        i=0,
        j=1,
    )
    # Pair base 1.0 and one 0.5 contribution from each of the two K centres.
    np.testing.assert_allclose(hopping, [[2.0]], atol=1.0e-12)

    outside = assembler.assemble_block(
        symbols=["X"],
        positions_bohr=np.asarray([[0.0, 0.0, 0.0]]),
        cell_bohr=cell,
        i=0,
        j=0,
        translation=(1, 0, 0),
    )
    np.testing.assert_allclose(outside, [[0.0]], atol=0.0)

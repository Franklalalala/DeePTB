import json
from pathlib import Path

import numpy as np

from dptb.data.interfaces.p2_table import (
    P2_TABLE_SCHEMA,
    P2TableAssembler,
    P2TableStore,
    RadialBlockTable,
    RealHarmonicRotator,
    p2_base_component_contract,
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


def _write_component_table(root: Path) -> None:
    (root / "base").mkdir(parents=True)
    (root / "projector").mkdir()
    (root / "species").mkdir()
    distances_base = np.linspace(0.0, 4.0, 9)
    shape = (len(distances_base), 1, 1)

    def radial_constant(value):
        array = np.full(shape, value, dtype=np.float32)
        array[-1] = 0.0
        return array

    np.savez(
        root / "base" / "X__X.npz",
        distances=distances_base,
        values=radial_constant(6.0),
        overlap_values=radial_constant(0.7),
        kinetic_values=radial_constant(1.0),
        vna_left_values=radial_constant(2.0),
        vna_right_values=radial_constant(3.0),
        left_shells=np.asarray([0], dtype=np.int16),
        right_shells=np.asarray([0], dtype=np.int16),
        support_bohr=np.asarray(4.0),
        onsite_base=np.asarray([[3.0]]),
        onsite_overlap=np.asarray([[1.2]]),
        onsite_kinetic=np.asarray([[1.0]]),
        onsite_vna_endpoint=np.asarray([[2.0]]),
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
        onsite_overlap=np.asarray([[1.2]]),
        onsite_kinetic=np.asarray([[1.0]]),
        onsite_vna_endpoint=np.asarray([[2.0]]),
    )
    contract = p2_base_component_contract("all")
    base_arrays = {
        name: spec["array"]
        for name, spec in contract["base_shard"].items()
        if spec["available"]
    }
    onsite_arrays = {
        name: spec["array"]
        for name, spec in contract["species_shard"].items()
        if spec["available"]
    }
    manifest = {
        "schema": P2_TABLE_SCHEMA,
        "length_unit": "bohr",
        "energy_unit": "Ry",
        "complete": True,
        "base_component_contract": contract,
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
                "onsite_component_arrays": onsite_arrays,
            }
        },
        "base_tables": {
            "X|X": {
                "path": "base/X__X.npz",
                "interpolation": "linear",
                "component_arrays": base_arrays,
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


def test_legacy_table_normalizes_to_none_and_missing_components_fail_closed(tmp_path):
    _write_synthetic_table(tmp_path)
    store = P2TableStore(tmp_path, verify_checksums=False)
    assert store.base_component_mode == "none"
    np.testing.assert_allclose(
        store.base_component("X", "X", "p2_base").evaluate([0.0, 0.0, 1.0]),
        [[1.0]],
    )
    with np.testing.assert_raises(KeyError):
        store.base_component("X", "X", "overlap")
    with np.testing.assert_raises(KeyError):
        store.onsite_component("X", "overlap")


def test_component_value_array_cache_and_onsite_contract_do_not_alias(tmp_path):
    _write_component_table(tmp_path)
    store = P2TableStore(tmp_path, verify_checksums=False)
    p2_base = store.base("X", "X")
    overlap = store.base("X", "X", value_array="overlap_values")
    kinetic = store.base_component("X", "X", "kinetic")
    np.testing.assert_allclose(p2_base.evaluate([0.0, 0.0, 1.0]), [[6.0]])
    np.testing.assert_allclose(overlap.evaluate([0.0, 0.0, 1.0]), [[0.7]])
    np.testing.assert_allclose(kinetic.evaluate([0.0, 0.0, 1.0]), [[1.0]])
    assert p2_base is not overlap
    np.testing.assert_allclose(store.onsite_component("X", "overlap"), [[1.2]])
    np.testing.assert_allclose(store.onsite_component("X", "kinetic"), [[1.0]])
    np.testing.assert_allclose(
        store.onsite_component("X", "vna_endpoint"), [[2.0]]
    )
    with np.testing.assert_raises(KeyError):
        store.base("X", "X", value_array="undeclared_hidden_array")


def test_component_assembler_reconstructs_p2_and_counts_onsite_vna_once(tmp_path):
    _write_component_table(tmp_path)
    assembler = P2TableAssembler(P2TableStore(tmp_path, verify_checksums=False))
    cell = np.eye(3) * 20.0

    onsite_kwargs = dict(
        symbols=["X"],
        positions_bohr=np.asarray([[0.0, 0.0, 0.0]]),
        cell_bohr=cell,
        i=0,
        j=0,
    )
    onsite = assembler.assemble_components_block(**onsite_kwargs)
    np.testing.assert_allclose(onsite["overlap"], [[1.2]])
    np.testing.assert_allclose(onsite["kinetic"], [[1.0]])
    np.testing.assert_allclose(onsite["vna_endpoint_pair"], [[2.0]])
    np.testing.assert_allclose(onsite["vnl"], [[0.5]])
    np.testing.assert_allclose(onsite["p2"], [[3.5]])
    np.testing.assert_allclose(
        onsite["kinetic"] + onsite["vna_endpoint_pair"] + onsite["vnl"],
        onsite["p2"],
    )
    np.testing.assert_allclose(
        assembler.assemble_block(**onsite_kwargs), onsite["p2"]
    )

    hopping_kwargs = dict(
        symbols=["X", "X"],
        positions_bohr=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell_bohr=cell,
        i=0,
        j=1,
    )
    hopping = assembler.assemble_components_block(**hopping_kwargs)
    np.testing.assert_allclose(hopping["overlap"], [[0.7]])
    np.testing.assert_allclose(hopping["kinetic"], [[1.0]])
    np.testing.assert_allclose(hopping["vna_left"], [[2.0]])
    np.testing.assert_allclose(hopping["vna_right"], [[3.0]])
    np.testing.assert_allclose(hopping["vna_endpoint_pair"], [[5.0]])
    np.testing.assert_allclose(hopping["vnl"], [[1.0]])
    np.testing.assert_allclose(hopping["p2"], [[7.0]])
    np.testing.assert_allclose(
        hopping["kinetic"]
        + hopping["vna_left"]
        + hopping["vna_right"]
        + hopping["vnl"],
        hopping["p2"],
    )
    np.testing.assert_allclose(
        assembler.assemble_overlap_block(**hopping_kwargs), [[0.7]]
    )
    np.testing.assert_allclose(
        assembler.assemble_block(**hopping_kwargs), hopping["p2"]
    )

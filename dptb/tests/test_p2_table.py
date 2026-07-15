from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from dptb.data.interfaces.p2_batch import (
    VectorizedNearbyImageEnumerator,
    active_table_working_set,
    interpolate_many,
)
from dptb.data.interfaces.p2_table import (
    P2_TABLE_SCHEMA,
    P2TableAssembler,
    P2TableStore,
    RadialBlockTable,
    RealHarmonicRotator,
    nearby_atom_images,
    p2_base_component_contract,
    rotation_z_to,
)
from dptb.data.interfaces.p23_table import (
    P23_FACTOR_SUPPORT_SEMANTICS,
    P23_VNA_TABLE_SCHEMA,
    P23VNAFactorAssembler,
    P23VNAFactorTableStore,
)

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_nonsoc_p2_tables import (  # noqa: E402
    Channel,
    OrbitalLike,
    _build_values,
    _sbt_context,
)
from build_nonsoc_p23_tables import (  # noqa: E402
    _angular_coefficients,
    _build_values_fast,
    _expanded_transform,
    build_vna_projectors,
)


# ---------------------------------------------------------------------------
# Shared synthetic-table builders
# ---------------------------------------------------------------------------


def _smooth_values(
    distances: np.ndarray,
    n_left: int,
    n_right: int,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    amplitude = rng.normal(size=(n_left, n_right))
    decay = rng.uniform(0.12, 0.45, size=(n_left, n_right))
    radial = distances[:, None, None]
    values = amplitude[None] * np.exp(-decay[None] * radial)
    values[-1] = 0.0
    return np.asarray(values, dtype=np.float32)


def _radial_table(interpolation: str) -> RadialBlockTable:
    distances = np.linspace(0.0, 4.0, 81)
    shells = (0, 1, 2)
    norb = sum(2 * l + 1 for l in shells)
    return RadialBlockTable(
        distances=distances,
        values=_smooth_values(distances, norb, norb, seed=20260714),
        left_shells=shells,
        right_shells=shells,
        support_bohr=4.0,
        interpolation=interpolation,
    )


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


def _write_single_species_table(root: Path) -> None:
    (root / "base").mkdir(parents=True)
    (root / "projector").mkdir()
    (root / "species").mkdir()
    distances_base = np.linspace(0.0, 6.0, 61)
    distances_projector = np.linspace(0.0, 5.4, 55)
    shells = np.asarray([0, 1], dtype=np.int16)
    norb = 4
    np.savez(
        root / "base" / "X__X.npz",
        distances=distances_base,
        values=_smooth_values(distances_base, norb, norb, seed=1),
        left_shells=shells,
        right_shells=shells,
        support_bohr=np.asarray(6.0),
    )
    np.savez(
        root / "projector" / "X__X.npz",
        distances=distances_projector,
        values=_smooth_values(distances_projector, norb, norb, seed=2),
        left_shells=shells,
        right_shells=shells,
        support_bohr=np.asarray(5.4),
    )
    onsite = _smooth_values(np.asarray([0.0, 1.0]), norb, norb, seed=3)[0]
    onsite = 0.5 * (onsite + onsite.T)
    np.savez(
        root / "species" / "X.npz",
        d_eff=np.diag(np.linspace(0.7, 1.3, norb)),
        onsite_base=onsite,
    )
    manifest = {
        "schema": P2_TABLE_SCHEMA,
        "length_unit": "bohr",
        "energy_unit": "Ry",
        "complete": True,
        "species": {
            "X": {
                "orbital_shells": [0, 1],
                "orbital_norb": norb,
                "orbital_cutoff_bohr": 3.0,
                "projector_shells": [0, 1],
                "projector_norb": norb,
                "projector_cutoffs_bohr": [2.4, 2.2],
                "projector_max_cutoff_bohr": 2.4,
                "array_path": "species/X.npz",
            }
        },
        "base_tables": {
            "X|X": {
                "path": "base/X__X.npz",
                "interpolation": "cubic",
            }
        },
        "projector_tables": {
            "X|X": {
                "path": "projector/X__X.npz",
                "interpolation": "cubic",
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_bridge_projector_table(root: Path) -> None:
    (root / "base").mkdir(parents=True)
    (root / "projector").mkdir()
    (root / "species").mkdir()
    shells = np.asarray([0], dtype=np.int16)
    np.savez(
        root / "base" / "X__X.npz",
        distances=np.asarray([0.0, 4.0]),
        values=np.zeros((2, 1, 1), dtype=np.float32),
        left_shells=shells,
        right_shells=shells,
        support_bohr=np.asarray(4.0),
    )
    np.savez(
        root / "projector" / "X__X.npz",
        distances=np.asarray([0.0, 3.0]),
        values=np.full((2, 1, 1), 2.0, dtype=np.float32),
        left_shells=shells,
        right_shells=shells,
        support_bohr=np.asarray(3.0),
    )
    np.savez(
        root / "species" / "X.npz",
        d_eff=np.asarray([[1.0]], dtype=np.float64),
        onsite_base=np.asarray([[0.0]], dtype=np.float64),
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


def _manual_dense(
    assembler: P2TableAssembler, arguments: dict[str, object]
) -> np.ndarray:
    symbols = tuple(arguments["symbols"])  # type: ignore[arg-type]
    r_keys = np.asarray(arguments["r_keys"])
    norb = [int(assembler.store.species[s]["orbital_norb"]) for s in symbols]
    offsets = np.concatenate(([0], np.cumsum(norb))).astype(np.int64)
    output = np.zeros((len(r_keys), int(offsets[-1]), int(offsets[-1])))
    for r_index, r_key in enumerate(r_keys):
        for i in range(len(symbols)):
            rows = slice(int(offsets[i]), int(offsets[i + 1]))
            for j in range(len(symbols)):
                cols = slice(int(offsets[j]), int(offsets[j + 1]))
                output[r_index, rows, cols] = assembler.assemble_block(
                    symbols=symbols,
                    positions_bohr=arguments["positions_bohr"],  # type: ignore[arg-type]
                    cell_bohr=arguments["cell_bohr"],  # type: ignore[arg-type]
                    i=i,
                    j=j,
                    translation=r_key,
                )
    return output


# ---------------------------------------------------------------------------
# Real-harmonic rotation and s->p axis channel
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Vectorized radial batch lookup (evaluate_many) vs scalar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_radial_table_evaluate_many_matches_scalar_edge_cases(interpolation):
    rng = np.random.default_rng(714)
    directions = rng.normal(size=(8, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = rng.uniform(0.1, 3.8, size=(8, 1))
    unique = directions * distances
    query = np.vstack(
        [
            unique,
            unique,
            -unique[:3],
            np.zeros((1, 3)),
            np.asarray([[0.0, 0.0, 4.0], [0.0, 0.0, 4.5]]),
        ]
    )
    table = _radial_table(interpolation)

    expected = np.stack([table.evaluate(vector) for vector in query], axis=0)
    table.clear_rotation_cache()
    actual = table.evaluate_many(query)

    np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)
    np.testing.assert_array_equal(actual[-2:], np.zeros_like(actual[-2:]))
    stats = table.rotation_cache_snapshot()
    assert stats["misses"] > 0
    assert stats["hits"] > 0


def test_radial_table_evaluate_many_preserves_empty_batch_shape():
    table = _radial_table("linear")

    actual = table.evaluate_many(np.empty((0, 3), dtype=np.float64))

    assert actual.shape == (0,) + table.values.shape[1:]
    assert actual.dtype == np.float64


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_radial_batch_lookup_rejects_nonfinite_inputs(bad_value):
    table = _radial_table("linear")

    with pytest.raises(ValueError, match="finite"):
        interpolate_many(table, np.asarray([bad_value], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        table.evaluate_many(
            np.asarray([[bad_value, 0.0, 0.0]], dtype=np.float64)
        )


# ---------------------------------------------------------------------------
# Vectorized nearby-image enumerator
# ---------------------------------------------------------------------------


def test_vectorized_nearby_image_enumerator_matches_scalar_order():
    cell = np.asarray(
        [[5.1, 0.2, -0.1], [0.3, 4.7, 0.4], [-0.2, 0.1, 5.4]],
        dtype=np.float64,
    )
    enumerator = VectorizedNearbyImageEnumerator(cell)
    rng = np.random.default_rng(9001)
    for _ in range(20):
        base = rng.uniform(-1.0, 1.0, size=3)
        target = rng.uniform(-8.0, 8.0, size=3)
        radius = float(rng.uniform(0.2, 7.0))
        expected = list(nearby_atom_images(cell, base, target, radius))
        actual = list(enumerator.iter_images(cell, base, target, radius))
        assert [translation for translation, _ in actual] == [
            translation for translation, _ in expected
        ]
        if expected:
            np.testing.assert_allclose(
                np.stack([center for _, center in actual]),
                np.stack([center for _, center in expected]),
                atol=2.0e-15,
                rtol=2.0e-15,
            )
    stats = enumerator.snapshot()
    assert stats["cell_inverse_computations"] == 1
    assert stats["hits"] > 0
    assert stats["misses"] > 0


def test_vectorized_image_enumerator_chunks_oversized_offset_grid():
    cell = np.eye(3, dtype=np.float64)
    enumerator = VectorizedNearbyImageEnumerator(cell)
    enumerator.max_offset_rows = 8
    enumerator.max_offset_bytes = 8 * 3 * np.dtype(np.int64).itemsize
    base = np.asarray([0.0, 0.0, 0.0])
    target = np.asarray([0.2, -0.1, 0.3])
    radius = 3.5

    expected = list(nearby_atom_images(cell, base, target, radius))
    actual = list(enumerator.iter_images(cell, base, target, radius))

    assert [translation for translation, _ in actual] == [
        translation for translation, _ in expected
    ]
    np.testing.assert_allclose(
        np.stack([center for _, center in actual]),
        np.stack([center for _, center in expected]),
        atol=2.0e-15,
        rtol=2.0e-15,
    )
    stats = enumerator.snapshot()
    assert stats["skipped"] > 0
    assert stats["cached_bound_grids"] == 0


# ---------------------------------------------------------------------------
# Scalar assembler: onsite VNA count, legacy fail-closed, component contract
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Dense == sparse == scalar batch assembly (plus batch stats provenance)
# ---------------------------------------------------------------------------


def test_dense_and_sparse_assembly_agree_with_scalar_and_call_evaluate_many(
    tmp_path, monkeypatch
):
    _write_single_species_table(tmp_path)
    assembler = P2TableAssembler(P2TableStore(tmp_path, verify_checksums=False))
    positions = np.asarray([[0.0, 0.0, 0.0], [1.4, 0.2, 0.1], [0.2, 1.6, 0.3]])
    cell = np.eye(3) * 20.0
    symbols = ("X", "X", "X")
    dense_arguments = {
        "symbols": symbols,
        "positions_bohr": positions,
        "cell_bohr": cell,
        "r_keys": np.zeros((1, 3), dtype=np.int64),
    }

    # Scalar reference assembled block by block.
    scalar_dense = _manual_dense(assembler, dense_arguments)

    # Dense batch path == scalar, using structure-local projector cache.
    dense = assembler.assemble_dense_rkeys(**dense_arguments)
    np.testing.assert_allclose(dense, scalar_dense, atol=2.0e-12, rtol=2.0e-12)
    dense_stats = assembler.batch_stats_snapshot()
    assert dense_stats is not None
    assert dense_stats["assembly"]["mode"] == "dense_rkeys"
    assert dense_stats["assembly"]["status"] == "success"
    assert dense_stats["assembly"]["uses_radial_evaluate_many"] is True
    assert dense_stats["assembly"]["base_evaluate_many_calls"] > 0
    assert dense_stats["assembly"]["projector_evaluate_many_calls"] > 0
    assert dense_stats["assembly"]["projector_overlap_unique"] > 0
    assert dense_stats["assembly"]["projector_overlap_execution"] == "evaluate_many"
    assert dense_stats["assembly"]["projector_overlap_budget_exceeded"] is False
    assert (
        dense_stats["assembly"]["projector_overlap_estimated_bytes"]
        <= dense_stats["assembly"]["projector_overlap_batch_budget_bytes"]
    )
    assert dense_stats["projector_overlap_cache"]["requests"] == 0
    assert dense_stats["assembly"]["vnl_contractions"] > 0
    assert dense_stats["lifecycle"] == "one assemble_dense_rkeys call"
    assert dense_stats["nearby_image_enumerator"]["cell_inverse_computations"] == 1

    # Sparse batch path == scalar (transitively == dense), and it must route
    # through RadialBlockTable.evaluate_many.
    block_keys = [
        (0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 2, 0, 0, 0),
    ]
    scalar_blocks = {
        key: assembler.assemble_block(
            symbols=symbols,
            positions_bohr=positions,
            cell_bohr=cell,
            i=key[0],
            j=key[1],
            translation=key[2:],
        )
        for key in block_keys
    }
    calls: list[int] = []
    original = RadialBlockTable.evaluate_many

    def counted(self, displacements_bohr, *, rotation_cache=None):
        calls.append(int(np.asarray(displacements_bohr).shape[0]))
        return original(self, displacements_bohr, rotation_cache=rotation_cache)

    monkeypatch.setattr(RadialBlockTable, "evaluate_many", counted)

    sparse = assembler.assemble_sparse_blocks(
        symbols=symbols,
        positions_bohr=positions,
        cell_bohr=cell,
        block_keys=block_keys,
    )

    assert calls
    norb = int(assembler.store.species["X"]["orbital_norb"])
    for key, block in sparse.items():
        np.testing.assert_allclose(
            block, scalar_blocks[key], atol=2.0e-12, rtol=2.0e-12
        )
        i, j = key[0], key[1]
        np.testing.assert_allclose(
            block,
            dense[0, i * norb : (i + 1) * norb, j * norb : (j + 1) * norb],
            atol=2.0e-12,
            rtol=2.0e-12,
        )
    sparse_stats = assembler.batch_stats_snapshot()
    assert sparse_stats is not None
    assert sparse_stats["assembly"]["mode"] == "sparse_required_blocks"
    assert sparse_stats["assembly"]["uses_radial_evaluate_many"] is True
    assert sparse_stats["assembly"]["base_evaluate_many_queries"] > 0
    assert sparse_stats["assembly"]["projector_evaluate_many_queries"] > 0
    assert sparse_stats["assembly"]["projector_overlap_execution"] == "evaluate_many"
    assert sparse_stats["assembly"]["projector_overlap_budget_exceeded"] is False
    assert (
        sparse_stats["assembly"]["projector_overlap_estimated_bytes"]
        <= sparse_stats["assembly"]["projector_overlap_batch_budget_bytes"]
    )


def test_sparse_assembly_falls_back_to_byte_bounded_scalar_cache(
    tmp_path, monkeypatch
):
    _write_single_species_table(tmp_path)
    assembler = P2TableAssembler(
        P2TableStore(tmp_path, verify_checksums=False),
        projector_overlap_cache_max_bytes=1,
    )
    arguments = {
        "symbols": ("X", "X", "X"),
        "positions_bohr": np.asarray(
            [[0.0, 0.0, 0.0], [1.4, 0.2, 0.1], [0.2, 1.6, 0.3]]
        ),
        "cell_bohr": np.eye(3) * 20.0,
        "block_keys": [
            (0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0),
            (1, 0, 0, 0, 0),
            (1, 2, 0, 0, 0),
        ],
    }
    expected = {
        key: assembler.assemble_block(
            symbols=arguments["symbols"],
            positions_bohr=arguments["positions_bohr"],
            cell_bohr=arguments["cell_bohr"],
            i=key[0],
            j=key[1],
            translation=key[2:],
        )
        for key in arguments["block_keys"]
    }

    def forbidden_projector_batch(*args, **kwargs):
        raise AssertionError("projector evaluate_many must not run over budget")

    monkeypatch.setattr(
        assembler, "_projector_overlaps_many", forbidden_projector_batch
    )

    actual = assembler.assemble_sparse_blocks(**arguments)

    for key, block in actual.items():
        np.testing.assert_allclose(block, expected[key], atol=2.0e-12, rtol=2.0e-12)
    stats = assembler.batch_stats_snapshot()
    assert stats is not None
    assembly = stats["assembly"]
    assert assembly["status"] == "success"
    assert assembly["projector_overlap_execution"] == "scalar_lru_fallback"
    assert assembly["projector_overlap_budget_exceeded"] is True
    assert assembly["projector_overlap_estimated_bytes"] > 1
    assert assembly["projector_overlap_batch_budget_bytes"] == 1
    assert assembly["projector_overlap_fallback_reason"] == (
        "estimated_projector_overlap_bytes_exceed_batch_budget"
    )
    assert assembly["scalar_fallback_blocks"] == len(arguments["block_keys"])
    assert assembly["uses_radial_evaluate_many"] is False
    assert assembly["base_evaluate_many_calls"] == 0
    assert assembly["projector_evaluate_many_calls"] == 0
    overlap_cache = stats["projector_overlap_cache"]
    assert overlap_cache["requests"] > 0
    assert overlap_cache["skipped"] > 0
    assert overlap_cache["current_bytes"] <= overlap_cache["capacity_bytes"] == 1
    json.dumps(stats)


def test_base_outside_support_still_keeps_bridge_projector_vnl(tmp_path):
    _write_bridge_projector_table(tmp_path)
    assembler = P2TableAssembler(P2TableStore(tmp_path, verify_checksums=False))
    arguments = {
        "symbols": ("X", "X", "X"),
        "positions_bohr": np.asarray(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
        ),
        "cell_bohr": np.eye(3) * 30.0,
    }

    scalar = assembler.assemble_block(**arguments, i=0, j=1, translation=(0, 0, 0))
    sparse = assembler.assemble_sparse_blocks(
        **arguments,
        block_keys=[(0, 1, 0, 0, 0)],
    )[(0, 1, 0, 0, 0)]

    np.testing.assert_allclose(sparse, scalar, atol=2.0e-12, rtol=2.0e-12)
    assert float(sparse[0, 0]) > 1.0
    stats = assembler.batch_stats_snapshot()
    assert stats is not None
    assert stats["assembly"]["outside_support_blocks"] == 1
    assert stats["assembly"]["base_evaluate_many_queries"] == 0
    assert stats["assembly"]["projector_evaluate_many_queries"] > 0


def test_real_si_component_table_dense_batch_smoke():
    configured = os.environ.get("DPTB_P2_REAL_TABLE_ROOT")
    if not configured:
        pytest.skip("DPTB_P2_REAL_TABLE_ROOT is not configured")
    table_root = Path(configured)
    if not table_root.is_dir():
        pytest.skip(f"real Si P2 smoke table is unavailable: {table_root}")
    assembler = P2TableAssembler(P2TableStore(table_root, verify_checksums=True))
    arguments = {
        "symbols": ("Si", "Si"),
        "positions_bohr": np.asarray([[0.0, 0.0, 0.0], [4.2, 0.1, -0.2]]),
        "cell_bohr": np.eye(3) * 30.0,
        "r_keys": np.zeros((1, 3), dtype=np.int64),
    }

    expected = _manual_dense(assembler, arguments)
    actual = assembler.assemble_dense_rkeys(**arguments)

    np.testing.assert_allclose(actual, expected, atol=2.0e-10, rtol=2.0e-10)
    assert np.isfinite(actual).all()
    stats = assembler.batch_stats_snapshot()
    assert stats is not None
    assert stats["assembly"]["uses_radial_evaluate_many"] is True
    assert stats["assembly"]["projector_evaluate_many_queries"] > 0
    assert stats["nearby_image_enumerator"]["cell_inverse_computations"] == 1


def test_active_working_set_counts_base_component_arrays():
    metadata = {
        "X": {"projector_norb": 4},
        "Y": {"projector_norb": 0},
    }
    report = active_table_working_set(
        ["X", "Y"],
        metadata,
        configured_capacity=64,
        base_component_arrays={
            "p2_base": "values",
            "overlap": "overlap_values",
            "kinetic": "kinetic_values",
        },
    )

    assert report["base_component_array_count"] == 3
    assert report["base_table_count"] == 12
    assert report["projector_table_count"] == 2
    assert report["minimum_capacity"] == 14


# ---------------------------------------------------------------------------
# Axial SBT (P23 VNA factor) builder and factor assembler
# ---------------------------------------------------------------------------


def _orbital(symbol: str, shells: tuple[int, ...], shift: float) -> OrbitalLike:
    radial_grid = np.linspace(0.0, 5.0, 201)
    channels = []
    for index, l_value in enumerate(shells):
        radial = (
            np.power(radial_grid, l_value)
            * (1.0 + 0.05 * index * radial_grid)
            * np.exp(-(0.8 + 0.1 * index + shift) * radial_grid)
        )
        radial[radial_grid >= 4.8] = 0.0
        channels.append(
            Channel(
                l=l_value,
                radial=radial,
                cutoff_bohr=4.8,
                source_index=index,
            )
        )
    return OrbitalLike(
        symbol=symbol,
        r=radial_grid,
        channels=tuple(channels),
        source=Path("synthetic.orb"),
    )


def test_axial_sparse_builder_matches_complete_sbt() -> None:
    left = _orbital("L", (0, 1), 0.0)
    right = _orbital("R", (0, 0, 1), 0.11)
    context = _sbt_context(
        left,
        right,
        kmax=24.0,
        n_k=96,
        n_mu=10,
        n_phi=20,
    )
    distances = np.linspace(0.0, 8.0, 17)
    expected = _build_values(context, distances)
    actual = _build_values_fast(
        left=left,
        right=right,
        left_transform=_expanded_transform(left, context.k, left=True),
        right_transform=_expanded_transform(right, context.k, left=False),
        coefficients=_angular_coefficients(left, right, n_mu=10, n_phi=20),
        k=context.k,
        factor=context.factor,
        distances=distances,
        support_bohr=9.6,
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_projector_support_is_limited_by_the_pao_grid() -> None:
    orbital = _orbital("X", (0, 1), 0.0)
    source_r = np.linspace(0.0, 10.0, 401)
    source_vna_ry = -np.exp(-0.2 * source_r)
    projectors = build_vna_projectors(
        orbital,
        source_r,
        source_vna_ry,
        radial_rank=1,
        l_buffer=0,
        vna_cutoff_bohr=9.5,
    )
    assert projectors.orbital.rcut == orbital.rcut
    assert projectors.metadata["physical_vna_cutoff_bohr"] == 9.5
    assert projectors.metadata["projector_cutoff_bohr"] == orbital.rcut


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_scalar_table(root: Path) -> None:
    (root / "species").mkdir(parents=True)
    (root / "factors").mkdir()
    species_path = root / "species" / "X.npz"
    np.savez(
        species_path,
        epsilon_ao=np.asarray([2.0], dtype=np.float64),
        radial_epsilon=np.asarray([2.0], dtype=np.float64),
        radial_l=np.asarray([0], dtype=np.int16),
        radial_n=np.asarray([0], dtype=np.int16),
    )
    distances = np.linspace(0.0, 6.0, 61)
    factor_path = root / "factors" / "X__X.npz"
    np.savez(
        factor_path,
        distances=distances,
        values=(1.0 - distances[:, None, None] / 6.0).astype(np.float32),
        left_shells=np.asarray([0], dtype=np.int16),
        right_shells=np.asarray([0], dtype=np.int16),
        support_bohr=np.asarray(6.0, dtype=np.float64),
    )
    manifest = {
        "schema": P23_VNA_TABLE_SCHEMA,
        "complete": True,
        "length_unit": "bohr",
        "factor_energy_unit": "eV",
        "epsilon_unit": "1/eV",
        "harmonic_convention": "deeptb_abacus_real",
        "endpoint_policy": "exclude_i0_and_jR",
        "factor_support_semantics": P23_FACTOR_SUPPORT_SEMANTICS,
        "interpolation": "linear",
        "species": {
            "X": {
                "orbital_shells": [0],
                "orbital_norb": 1,
                "orbital_cutoff_bohr": 3.0,
                "vna_cutoff_bohr": 3.0,
                "vna_projector_shells": [0],
                "vna_projector_norb": 1,
                "array_path": "species/X.npz",
                "array_sha256": _sha256(species_path),
            }
        },
        "factor_tables": {
            "X|X": {
                "path": "factors/X__X.npz",
                "sha256": _sha256(factor_path),
            }
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def test_factor_assembler_excludes_endpoints_and_sets_exact_reverse(
    tmp_path: Path,
) -> None:
    _write_scalar_table(tmp_path)
    store = P23VNAFactorTableStore(tmp_path)
    assembler = P23VNAFactorAssembler(store, contraction_chunk_terms=2)
    positions = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edge_index = np.asarray(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=np.int64
    )
    node, edge, stats = assembler.assemble_graph_addition(
        symbols=["X", "X", "X"],
        positions_bohr=positions,
        cell_bohr=np.eye(3) * 20.0,
        edge_index=edge_index,
        edge_cell_shift=np.zeros((6, 3), dtype=np.int64),
        node_shapes=np.ones((3, 2), dtype=np.int64),
        edge_shapes=np.ones((6, 2), dtype=np.int64),
        node_pad_shape=(1, 1),
        edge_pad_shape=(1, 1),
    )

    assert stats["onsite_terms"] == 6
    assert stats["hopping_representative_terms"] == 3
    assert stats["directed_true_third_centre_terms"] == 12
    np.testing.assert_allclose(node[0, 0, 0], 82.0 / 36.0, atol=2.0e-7)
    np.testing.assert_allclose(edge[0, 0, 0], 50.0 / 36.0, atol=2.0e-7)
    for row, reverse in ((0, 1), (2, 3), (4, 5)):
        np.testing.assert_array_equal(edge[reverse], edge[row].T)

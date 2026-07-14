from __future__ import annotations

import json
import os
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
    nearby_atom_images,
)


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


def _manual_dense(assembler: P2TableAssembler, arguments: dict[str, object]) -> np.ndarray:
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


def test_dense_assembly_uses_structure_local_projector_cache(tmp_path):
    _write_single_species_table(tmp_path)
    assembler = P2TableAssembler(P2TableStore(tmp_path, verify_checksums=False))
    arguments = {
        "symbols": ("X", "X", "X"),
        "positions_bohr": np.asarray(
            [[0.0, 0.0, 0.0], [1.4, 0.2, 0.1], [0.2, 1.6, 0.3]]
        ),
        "cell_bohr": np.eye(3) * 20.0,
        "r_keys": np.zeros((1, 3), dtype=np.int64),
    }

    expected = _manual_dense(assembler, arguments)
    actual = assembler.assemble_dense_rkeys(**arguments)

    np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)
    stats = assembler.batch_stats_snapshot()
    assert stats is not None
    assert stats["assembly"]["mode"] == "dense_rkeys"
    assert stats["assembly"]["status"] == "success"
    assert stats["assembly"]["uses_radial_evaluate_many"] is True
    assert stats["assembly"]["base_evaluate_many_calls"] > 0
    assert stats["assembly"]["projector_evaluate_many_calls"] > 0
    assert stats["assembly"]["projector_overlap_unique"] > 0
    assert stats["assembly"]["projector_overlap_execution"] == "evaluate_many"
    assert stats["assembly"]["projector_overlap_budget_exceeded"] is False
    assert (
        stats["assembly"]["projector_overlap_estimated_bytes"]
        <= stats["assembly"]["projector_overlap_batch_budget_bytes"]
    )
    assert stats["projector_overlap_cache"]["requests"] == 0
    assert stats["assembly"]["vnl_contractions"] > 0
    assert stats["lifecycle"] == "one assemble_dense_rkeys call"
    images = stats["nearby_image_enumerator"]
    assert images["cell_inverse_computations"] == 1


def test_sparse_assembly_matches_scalar_and_calls_evaluate_many(tmp_path, monkeypatch):
    _write_single_species_table(tmp_path)
    assembler = P2TableAssembler(P2TableStore(tmp_path, verify_checksums=False))
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
    calls: list[int] = []
    original = RadialBlockTable.evaluate_many

    def counted(self, displacements_bohr, *, rotation_cache=None):
        calls.append(int(np.asarray(displacements_bohr).shape[0]))
        return original(self, displacements_bohr, rotation_cache=rotation_cache)

    monkeypatch.setattr(RadialBlockTable, "evaluate_many", counted)

    actual = assembler.assemble_sparse_blocks(**arguments)

    assert calls
    for key, block in actual.items():
        np.testing.assert_allclose(block, expected[key], atol=2.0e-12, rtol=2.0e-12)
    stats = assembler.batch_stats_snapshot()
    assert stats is not None
    assert stats["assembly"]["mode"] == "sparse_required_blocks"
    assert stats["assembly"]["uses_radial_evaluate_many"] is True
    assert stats["assembly"]["base_evaluate_many_queries"] > 0
    assert stats["assembly"]["projector_evaluate_many_queries"] > 0
    assert stats["assembly"]["projector_overlap_execution"] == "evaluate_many"
    assert stats["assembly"]["projector_overlap_budget_exceeded"] is False
    assert (
        stats["assembly"]["projector_overlap_estimated_bytes"]
        <= stats["assembly"]["projector_overlap_batch_budget_bytes"]
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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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

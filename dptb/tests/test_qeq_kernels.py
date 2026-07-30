import math
import pickle

import numpy as np
import pytest

from dptb.nnops import (
    COULOMB_EV_ANGSTROM,
    DEFAULT_QEQ_PARAMETERS,
    QEqGeometryError,
    QEqGeometryResult,
    QEqKernelConditionError,
    QEqParameterError,
    bare_qeq_kernel,
    build_qeq_kernel,
    gaussian_qeq_kernel,
    ohno_qeq_kernel,
    solve_qeq_from_geometry,
    validate_qeq_result,
)


WATER_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],
        [-0.2399872, 0.92662721, 0.0],
    ]
)
WATER_SYMBOLS = ("O", "H", "H")


@pytest.mark.parametrize("kernel", ["bare", "ohno", "gaussian"])
def test_two_atom_geometry_solution_matches_analytic_formula(kernel):
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    result = solve_qeq_from_geometry(
        positions, ["H", "F"], 0.0, kernel=kernel
    )

    chi_a, chi_b = result.electronegativity
    j_aa = result.hardness_kernel[0, 0]
    j_bb = result.hardness_kernel[1, 1]
    j_ab = result.hardness_kernel[0, 1]
    expected_q_a = (chi_b - chi_a) / (j_aa + j_bb - 2.0 * j_ab)

    np.testing.assert_allclose(
        result.charges, [expected_q_a, -expected_q_a], rtol=0.0, atol=1e-12
    )
    validate_qeq_result(result, atol=1e-12)


@pytest.mark.parametrize("kernel", ["ohno", "gaussian"])
def test_electronegativity_ordering_for_hf_and_lih(kernel):
    hf = solve_qeq_from_geometry(
        [[0.0, 0.0, 0.0], [0.917, 0.0, 0.0]],
        ["H", "F"],
        0.0,
        kernel=kernel,
    )
    lih = solve_qeq_from_geometry(
        [[0.0, 0.0, 0.0], [1.595, 0.0, 0.0]],
        ["Li", "H"],
        0.0,
        kernel=kernel,
    )

    assert hf.charges[1] < 0.0
    assert hf.charges[0] > 0.0
    assert lih.charges[0] > 0.0
    assert lih.charges[1] < 0.0


def test_three_pair_kernels_share_the_long_range_coulomb_limit():
    distance = 1.0e6
    positions = [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]
    expected = COULOMB_EV_ANGSTROM / distance
    pair_values = [
        constructor(positions, ["H", "F"])[0, 1]
        for constructor in (
            bare_qeq_kernel,
            ohno_qeq_kernel,
            gaussian_qeq_kernel,
        )
    ]

    np.testing.assert_allclose(
        pair_values, np.full(3, expected), rtol=1e-11, atol=0.0
    )


def test_short_range_bare_diverges_while_regularized_limits_are_finite():
    small_distance = 1.0e-9
    positions = [[0.0, 0.0, 0.0], [small_distance, 0.0, 0.0]]
    bare = bare_qeq_kernel(positions, ["H", "F"])
    ohno = ohno_qeq_kernel(positions, ["H", "F"])
    gaussian = gaussian_qeq_kernel(positions, ["H", "F"])

    assert bare[0, 1] > 1.0e9
    hardness_h = DEFAULT_QEQ_PARAMETERS["H"].hardness
    hardness_f = DEFAULT_QEQ_PARAMETERS["F"].hardness
    np.testing.assert_allclose(
        ohno[0, 1],
        0.5 * (hardness_h + hardness_f),
        rtol=0.0,
        atol=1e-12,
    )
    sigma_h = DEFAULT_QEQ_PARAMETERS["H"].gaussian_width
    sigma_f = DEFAULT_QEQ_PARAMETERS["F"].gaussian_width
    sigma_hf = math.sqrt(sigma_h**2 + sigma_f**2)
    gaussian_limit = (
        COULOMB_EV_ANGSTROM
        * math.sqrt(2.0 / math.pi)
        / sigma_hf
    )
    np.testing.assert_allclose(
        gaussian[0, 1], gaussian_limit, rtol=1e-15, atol=0.0
    )

    # All public cluster constructors reject an exact overlap, including the
    # regularized forms: finite mathematical limits do not make a duplicated
    # atom geometry valid.
    for constructor in (
        bare_qeq_kernel,
        ohno_qeq_kernel,
        gaussian_qeq_kernel,
    ):
        with pytest.raises(QEqGeometryError, match="overlapping atoms"):
            constructor([[0.0, 0.0, 0.0]] * 2, ["H", "F"])


@pytest.mark.parametrize("kernel", ["bare", "ohno", "gaussian"])
def test_kernel_is_translation_and_rotation_invariant(kernel):
    positions = np.array(
        [[0.1, -0.2, 0.3], [1.4, 0.5, -0.7], [-0.6, 1.2, 0.8]]
    )
    angle = 0.731
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translated_rotated = positions @ rotation.T + [12.3, -4.5, 0.7]

    baseline = build_qeq_kernel(
        positions, ["O", "H", "F"], kernel=kernel
    )
    transformed = build_qeq_kernel(
        translated_rotated, ["O", "H", "F"], kernel=kernel
    )

    np.testing.assert_allclose(transformed, baseline, rtol=1e-15, atol=1e-14)


@pytest.mark.parametrize("kernel", ["ohno", "gaussian"])
def test_atom_permutation_is_equivariant_for_kernel_and_charges(kernel):
    permutation = np.array([2, 0, 1])
    baseline = solve_qeq_from_geometry(
        WATER_POSITIONS, WATER_SYMBOLS, 0.0, kernel=kernel
    )
    permuted = solve_qeq_from_geometry(
        WATER_POSITIONS[permutation],
        [WATER_SYMBOLS[index] for index in permutation],
        0.0,
        kernel=kernel,
    )

    np.testing.assert_allclose(
        permuted.hardness_kernel,
        baseline.hardness_kernel[np.ix_(permutation, permutation)],
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        permuted.charges, baseline.charges[permutation], rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize("kernel", ["bare", "ohno", "gaussian"])
def test_kernel_symmetry_is_exact_and_output_is_bytes_backed(kernel):
    matrix = build_qeq_kernel(
        WATER_POSITIONS, WATER_SYMBOLS, kernel=kernel
    )

    assert np.array_equal(matrix, matrix.T)
    assert not matrix.flags.writeable
    with pytest.raises(ValueError):
        matrix.setflags(write=True)


def test_geometry_and_unknown_elements_fail_closed():
    with pytest.raises(QEqGeometryError, match="finite"):
        build_qeq_kernel([[0.0, np.nan, 0.0]], ["H"])
    with pytest.raises(QEqGeometryError, match=r"shape \(natom, 3\)"):
        build_qeq_kernel([[0.0, 0.0]], ["H"])
    with pytest.raises(QEqGeometryError, match="length"):
        build_qeq_kernel([[0.0, 0.0, 0.0]], ["H", "F"])
    with pytest.raises(QEqParameterError, match="not present"):
        build_qeq_kernel([[0.0, 0.0, 0.0]], ["Xe"])
    with pytest.raises(QEqGeometryError, match="unknown QEq geometry kernel"):
        build_qeq_kernel([[0.0, 0.0, 0.0]], ["H"], kernel="pme")


def test_user_parameters_can_override_and_extend_the_default_table():
    overrides = {
        "H": {
            "electronegativity": 5.5,
            "hardness": 8.0,
            "gaussian_width": 0.4,
        },
        "N": {
            "electronegativity": 7.0,
            "hardness": 9.0,
            "gaussian_width": 0.7,
        },
    }
    result = solve_qeq_from_geometry(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ["H", "N"],
        0.0,
        kernel="ohno",
        params=overrides,
    )

    assert result.parameter_table["H"].electronegativity == 5.5
    assert result.parameter_table["N"].hardness == 9.0
    assert result.parameter_table["F"] == DEFAULT_QEQ_PARAMETERS["F"]
    assert "caller-supplied" in result.parameter_table.provenance.source


def test_hu_default_table_values_and_provenance_are_complete():
    expected = {
        "Li": (-3.0000, 10.0241, 1.28),
        "C": (5.8678, 7.0000, 0.76),
        "H": (5.3200, 7.4366, 0.31),
        "O": (8.5000, 8.9989, 0.66),
        "P": (1.8000, 7.0946, 1.07),
        "F": (9.0000, 8.0000, 0.57),
    }
    for symbol, values in expected.items():
        actual = DEFAULT_QEQ_PARAMETERS[symbol]
        assert (
            actual.electronegativity,
            actual.hardness,
            actual.gaussian_width,
        ) == values

    provenance = DEFAULT_QEQ_PARAMETERS.provenance
    assert provenance.doi == "10.1038/s41467-025-62824-5"
    assert provenance.electronegativity_unit == "eV/e"
    assert provenance.hardness_unit == "eV/e^2"
    assert provenance.gaussian_width_unit == "Angstrom"
    assert "PME" in provenance.applicability_note
    assert "dipole correction" in provenance.applicability_note
    assert "not a reproduction" in provenance.applicability_note
    assert len(DEFAULT_QEQ_PARAMETERS.sha256) == 64


def test_geometry_result_carries_stable_hash_provenance_and_pickle_freezing():
    result = solve_qeq_from_geometry(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        ["Li", "H"],
        0.0,
        kernel="gaussian",
    )
    restored = pickle.loads(pickle.dumps(result))

    assert isinstance(result, QEqGeometryResult)
    assert isinstance(result, type(restored))
    assert restored.provenance == result.provenance
    assert restored.parameter_table.sha256 == result.parameter_table.sha256
    assert not restored.positions.flags.writeable
    with pytest.raises(ValueError):
        restored.positions.setflags(write=True)
    validate_qeq_result(restored)


def test_water_three_kernel_comparison_and_bare_fail_closed():
    # With the requested bare diagonal (Hu element hardness only), molecular
    # O-H point interactions exceed the diagonal terms.  The constrained
    # kernel is non-convex and the existing QEq solver correctly rejects it.
    with pytest.raises(QEqKernelConditionError, match="not positive definite"):
        solve_qeq_from_geometry(
            WATER_POSITIONS, WATER_SYMBOLS, 0.0, kernel="bare"
        )

    ohno = solve_qeq_from_geometry(
        WATER_POSITIONS, WATER_SYMBOLS, 0.0, kernel="ohno"
    )
    gaussian = solve_qeq_from_geometry(
        WATER_POSITIONS, WATER_SYMBOLS, 0.0, kernel="gaussian"
    )
    np.testing.assert_allclose(
        ohno.charges,
        [-2.60071304655607, 1.30035652512363, 1.30035652143244],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        gaussian.charges,
        [-0.171976132353201, 0.0859880662032662, 0.0859880661499344],
        rtol=0.0,
        atol=1e-12,
    )

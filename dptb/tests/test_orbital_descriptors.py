import pickle

import numpy as np
import pytest

from dptb.nnops.fixed_mu_operator import fixed_mu_scan, generalized_bands
from dptb.nnops.orbital_descriptors import (
    OrbitalDescriptorError,
    bond_energy_partition,
    fixed_mu_scan_fragment_populations,
    fragment_charge_response,
    fragment_pdos,
    mulliken_lowdin_populations,
    orbital_descriptors,
)


def _h2_system():
    overlap = np.array([[1.0, 0.2], [0.2, 1.0]])
    hamiltonian = np.array([[0.0, -1.0], [-1.0, 0.0]])
    bands = generalized_bands(hamiltonian, overlap)
    return hamiltonian, overlap, bands.eigvals, bands.eigvecs


def _density(eigvecs, occupations):
    return np.einsum(
        "...ni,...i,...mi->...nm",
        eigvecs,
        occupations,
        eigvecs.conj(),
    )


def test_nonorthogonal_h2_populations_pdos_and_bonding_signs():
    hamiltonian, overlap, eigvals, eigvecs = _h2_system()
    occupations = np.array([2.0, 0.0])
    result = orbital_descriptors(
        eigvals,
        eigvecs,
        overlap,
        hamiltonian,
        np.array([0, 1]),
        occupations=occupations,
    )

    np.testing.assert_allclose(result.populations.mulliken, [1.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(result.populations.lowdin, [1.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(result.pdos.weights.sum(axis=-1), 1.0, atol=1e-14)
    expected_center = np.mean(eigvals)
    np.testing.assert_allclose(
        result.pdos.band_centers,
        [expected_center, expected_center],
        atol=1e-14,
    )
    occupied = fragment_pdos(
        eigvals,
        eigvecs,
        overlap,
        [0, 1],
        occupations=occupations,
        window="occupied",
    )
    np.testing.assert_allclose(occupied.band_centers, eigvals[0], atol=1e-14)
    assert occupied.window_policy == "occupied"

    bonding_density = _density(eigvecs, [2.0, 0.0])
    antibonding_density = _density(eigvecs, [0.0, 2.0])
    bonding = bond_energy_partition(bonding_density, hamiltonian, [0, 1])
    antibonding = bond_energy_partition(antibonding_density, hamiltonian, [0, 1])
    assert bonding.pair_energy[0, 1] < 0.0
    assert antibonding.pair_energy[0, 1] > 0.0


def test_asymmetric_dimer_charge_transfer_points_to_lower_onsite_atom():
    overlap = np.array([[1.0, 0.15], [0.15, 1.0]])
    hamiltonian = np.array([[-0.7, -0.4], [-0.4, 0.6]])
    bands = generalized_bands(hamiltonian, overlap)
    density = _density(bands.eigvecs, [2.0, 0.0])
    populations = mulliken_lowdin_populations(density, overlap, [0, 1])

    assert populations.mulliken[0] > populations.mulliken[1]
    assert populations.lowdin[0] > populations.lowdin[1]
    np.testing.assert_allclose(populations.mulliken.sum(), 2.0, atol=1e-14)
    np.testing.assert_allclose(populations.lowdin.sum(), 2.0, atol=1e-14)


def test_weight_normalization_and_band_energy_closure_to_1e12():
    overlap = np.array(
        [
            [1.0, 0.08 + 0.03j, -0.04],
            [0.08 - 0.03j, 1.1, 0.02j],
            [-0.04, -0.02j, 0.9],
        ]
    )
    hamiltonian = np.array(
        [
            [-0.8, -0.25 + 0.07j, 0.1],
            [-0.25 - 0.07j, 0.2, -0.18j],
            [0.1, 0.18j, 0.9],
        ]
    )
    bands = generalized_bands(hamiltonian, overlap)
    occupations = np.array([1.7, 0.6, 0.1])
    density = _density(bands.eigvecs, occupations)
    pdos = fragment_pdos(
        bands.eigvals,
        bands.eigvecs,
        overlap,
        [0, 1, 1],
        occupations=occupations,
        ao_l_index=[0, 0, 1],
    )
    partition = bond_energy_partition(density, hamiltonian, [0, 1, 1])

    np.testing.assert_allclose(pdos.weights.sum(axis=-1), 1.0, atol=1e-12)
    partition_sum = partition.onsite_energy.sum() + partition.pair_energy.sum()
    np.testing.assert_allclose(partition_sum, np.trace(density @ hamiltonian).real, atol=1e-12)
    np.testing.assert_allclose(partition.closure_residual, 0.0, atol=1e-12)
    assert pdos.resolved_labels.tolist() == [[0, 0], [1, 0], [1, 1]]


def test_ao_permutation_equivariance():
    overlap = np.array(
        [
            [1.0, 0.1, -0.03],
            [0.1, 1.2, 0.04],
            [-0.03, 0.04, 0.95],
        ]
    )
    hamiltonian = np.array(
        [
            [-0.5, -0.3, 0.07],
            [-0.3, 0.15, -0.2],
            [0.07, -0.2, 0.8],
        ]
    )
    atom_map = np.array([4, 2, 4])
    l_map = np.array([0, 1, 1])
    bands = generalized_bands(hamiltonian, overlap)
    occupations = np.array([1.8, 0.7, 0.2])
    baseline = orbital_descriptors(
        bands.eigvals,
        bands.eigvecs,
        overlap,
        hamiltonian,
        atom_map,
        occupations=occupations,
        ao_l_index=l_map,
    )

    permutation = np.array([2, 0, 1])
    permuted = orbital_descriptors(
        bands.eigvals,
        bands.eigvecs[permutation, :],
        overlap[np.ix_(permutation, permutation)],
        hamiltonian[np.ix_(permutation, permutation)],
        atom_map[permutation],
        occupations=occupations,
        ao_l_index=l_map[permutation],
    )

    np.testing.assert_array_equal(permuted.populations.atom_labels, baseline.populations.atom_labels)
    np.testing.assert_allclose(permuted.populations.mulliken, baseline.populations.mulliken, atol=1e-13)
    np.testing.assert_allclose(permuted.populations.lowdin, baseline.populations.lowdin, atol=1e-13)
    np.testing.assert_allclose(permuted.pdos.weights, baseline.pdos.weights, atol=1e-13)
    np.testing.assert_allclose(
        permuted.bond_energy.directed_energy,
        baseline.bond_energy.directed_energy,
        atol=1e-13,
    )


def test_energy_zero_gauge_covariance_and_icohp_gauge_dependence():
    hamiltonian, overlap, eigvals, eigvecs = _h2_system()
    occupations = np.array([2.0, 0.0])
    density = _density(eigvecs, occupations)
    shift = 0.37
    shifted_hamiltonian = hamiltonian + shift * overlap
    shifted_bands = generalized_bands(shifted_hamiltonian, overlap)

    original_pdos = fragment_pdos(
        eigvals,
        eigvecs,
        overlap,
        [0, 1],
        occupations=occupations,
    )
    shifted_pdos = fragment_pdos(
        shifted_bands.eigvals,
        shifted_bands.eigvecs,
        overlap,
        [0, 1],
        occupations=occupations,
    )
    np.testing.assert_allclose(
        shifted_pdos.band_centers,
        original_pdos.band_centers + shift,
        atol=1e-13,
    )

    original = bond_energy_partition(density, hamiltonian, [0, 1])
    shifted = bond_energy_partition(density, shifted_hamiltonian, [0, 1])
    metric_partition = bond_energy_partition(density, overlap, [0, 1])
    np.testing.assert_allclose(
        shifted.directed_energy,
        original.directed_energy + shift * metric_partition.directed_energy,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        shifted.trace_energy,
        original.trace_energy + shift * np.trace(density @ overlap).real,
        atol=1e-13,
    )
    assert not np.isclose(
        shifted.pair_energy[0, 1],
        original.pair_energy[0, 1],
        atol=1e-14,
    )


def test_density_response_projection_and_scan_bridge():
    hamiltonian, overlap, eigvals, eigvecs = _h2_system()
    mu_grid = np.linspace(-1.5, 1.7, 17)
    scan = fixed_mu_scan(
        hamiltonian,
        overlap,
        mu_grid,
        kT=0.15,
        spin_degeneracy=2.0,
    )
    curves = fixed_mu_scan_fragment_populations(
        scan,
        eigvecs,
        overlap,
        [0, 1],
    )

    np.testing.assert_allclose(curves.populations.sum(axis=-1), scan.electron_count, atol=1e-13)
    np.testing.assert_allclose(curves.responses.sum(axis=-1), scan.dos_like_response, atol=1e-13)
    np.testing.assert_allclose(curves.populations[:, 0], curves.populations[:, 1], atol=1e-13)
    assert np.all(np.diff(curves.populations[:, 0]) > 0.0)

    response_density = _density(eigvecs, scan.occupation_response[8])
    projected = fragment_charge_response(response_density, overlap, [0, 1])
    np.testing.assert_allclose(projected.mulliken, curves.responses[8], atol=1e-13)
    np.testing.assert_allclose(projected.total_response, scan.dos_like_response[8], atol=1e-13)


def test_k_weighted_scan_bridge_matches_scan_ledgers():
    h0, s0, _, _ = _h2_system()
    hamiltonian = np.stack((h0, h0 + 0.2 * np.diag([1.0, -1.0])))
    overlap = np.stack((s0, s0))
    bands = generalized_bands(hamiltonian, overlap)
    scan = fixed_mu_scan(
        hamiltonian,
        overlap,
        [-0.5, 0.0, 0.5],
        kT=0.2,
        spin_degeneracy=2.0,
        k_axis=0,
        k_weights=[1.0, 3.0],
    )
    curves = fixed_mu_scan_fragment_populations(
        scan,
        bands.eigvecs,
        overlap,
        [0, 1],
    )

    assert curves.populations.shape == (3, 2)
    np.testing.assert_allclose(curves.populations.sum(axis=-1), scan.electron_count, atol=1e-13)
    np.testing.assert_allclose(curves.responses.sum(axis=-1), scan.dos_like_response, atol=1e-13)


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda values: {**values, "ao_atom_index": [0]}, "length 2"),
        (lambda values: {**values, "ao_atom_index": [0, -1]}, "nonnegative"),
        (
            lambda values: {
                **values,
                "overlap": np.array([[1.0, 0.2], [0.3, 1.0]]),
            },
            "Hermitian",
        ),
        (
            lambda values: {
                **values,
                "hamiltonian": np.array([[0.0, -1.0], [-0.8, 0.0]]),
            },
            "Hermitian",
        ),
        (
            lambda values: {
                **values,
                "eigvals": np.array([np.nan, 1.0]),
            },
            "finite",
        ),
        (
            lambda values: {
                **values,
                "eigvecs": np.array([[1.0, 0.0], [0.0, 1.0]]),
            },
            "S-orthonormal",
        ),
    ],
)
def test_fail_closed_bad_mapping_hermiticity_nan_and_eigenvectors(mutator, match):
    hamiltonian, overlap, eigvals, eigvecs = _h2_system()
    values = {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "overlap": overlap,
        "hamiltonian": hamiltonian,
        "ao_atom_index": [0, 1],
        "occupations": [2.0, 0.0],
    }
    with pytest.raises(OrbitalDescriptorError, match=match):
        orbital_descriptors(**mutator(values))


def test_fail_closed_occupation_routes_windows_and_inconsistent_density():
    hamiltonian, overlap, eigvals, eigvecs = _h2_system()
    base = (eigvals, eigvecs, overlap, hamiltonian, [0, 1])
    with pytest.raises(OrbitalDescriptorError, match="complete"):
        orbital_descriptors(*base)
    with pytest.raises(OrbitalDescriptorError, match="not both"):
        orbital_descriptors(
            *base,
            occupations=[2.0, 0.0],
            mu=0.0,
            kT=0.1,
            spin_degeneracy=2.0,
        )
    with pytest.raises(OrbitalDescriptorError, match="requires energy_window"):
        orbital_descriptors(*base, occupations=[2.0, 0.0], window="energy")
    with pytest.raises(OrbitalDescriptorError, match="zero window weight"):
        orbital_descriptors(
            *base,
            occupations=[2.0, 0.0],
            window="energy",
            energy_window=[10.0, 11.0],
        )
    with pytest.raises(OrbitalDescriptorError, match="inconsistent"):
        orbital_descriptors(
            *base,
            occupations=[2.0, 0.0],
            density=np.eye(2),
        )


def test_results_are_bytes_backed_readonly_and_refrozen_after_pickle():
    hamiltonian, overlap, eigvals, eigvecs = _h2_system()
    result = orbital_descriptors(
        eigvals,
        eigvecs,
        overlap,
        hamiltonian,
        [0, 1],
        occupations=[2.0, 0.0],
    )
    restored = pickle.loads(pickle.dumps(result))

    for array in (
        restored.occupations,
        restored.density,
        restored.populations.mulliken,
        restored.pdos.weights,
        restored.bond_energy.pair_energy,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)

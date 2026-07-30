import numpy as np
import pytest

from dptb.nnops import FixedMuScanResult, fixed_mu_scan
import dptb.nnops.fixed_mu_operator as fixed_mu


def _random_hermitian_problem(rng, leading_shape, n):
    shape = leading_shape + (n, n)
    h_raw = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    s_raw = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    h = 0.5 * (h_raw + np.swapaxes(h_raw.conj(), -1, -2))
    s = np.swapaxes(s_raw.conj(), -1, -2) @ s_raw + 0.7 * np.eye(n)
    return h, s


@pytest.mark.parametrize("kT", [0.0, 0.03, 0.2])
def test_fixed_mu_scan_matches_scalar_observables_for_complex_nonorthogonal_inputs(kT):
    rng = np.random.default_rng(20260730)
    h, s = _random_hermitian_problem(rng, (2, 3), 4)
    mu_grid = np.array([-0.7, -0.05, 0.3, 1.1])
    k_weights = np.array([0.2, 0.3, 0.5])

    scan = fixed_mu_scan(
        h,
        s,
        mu_grid,
        kT=kT,
        spin_degeneracy=1.0,
        k_weights=k_weights,
        k_axis=1,
    )

    assert isinstance(scan, FixedMuScanResult)
    assert scan.occupations.shape == (mu_grid.size, 2, 3, 4)
    assert scan.electron_count.shape == (mu_grid.size, 2)
    assert not hasattr(scan, "density")
    assert not hasattr(scan, "density_k")
    for index, mu in enumerate(mu_grid):
        scalar = fixed_mu.fixed_mu_observables(
            h,
            s,
            mu=float(mu),
            kT=kT,
            spin_degeneracy=1.0,
            k_weights=k_weights,
            k_axis=1,
        )
        np.testing.assert_allclose(
            scan.occupations[index],
            scalar.occupations,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            scan.occupation_response[index],
            scalar.occupation_response,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            scan.electron_count[index],
            scalar.electron_count,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            scan.dos_like_response[index],
            scalar.dos_like_response,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        for field_name in (
            "band_energy",
            "minus_t_s",
            "band_free_energy",
            "band_grand_energy",
        ):
            np.testing.assert_allclose(
                getattr(scan.energies, field_name)[index],
                getattr(scalar.energies, field_name),
                rtol=1.0e-12,
                atol=1.0e-13,
            )


def test_fixed_mu_scan_diagonalizes_once_per_input_structure(monkeypatch):
    rng = np.random.default_rng(17)
    h, s = _random_hermitian_problem(rng, (3,), 4)
    calls = 0
    original = fixed_mu._generalized_eigh_single

    def counted(h_one, s_one):
        nonlocal calls
        calls += 1
        return original(h_one, s_one)

    monkeypatch.setattr(fixed_mu, "_generalized_eigh_single", counted)
    fixed_mu_scan(h, s, np.linspace(-1.0, 1.0, 19), kT=0.1)

    assert calls == 3


@pytest.mark.parametrize(
    ("mu_grid", "message"),
    [
        ([], "at least one"),
        ([0.0, np.nan], "finite"),
        ([0.0, np.inf], "finite"),
        ([[0.0, 1.0]], "one-dimensional"),
    ],
)
def test_fixed_mu_scan_rejects_invalid_mu_grids(mu_grid, message):
    with pytest.raises(fixed_mu.FixedMuOperatorError, match=message):
        fixed_mu_scan(np.eye(2), np.eye(2), mu_grid)


def test_fixed_mu_scan_arrays_are_bytes_backed_read_only():
    result = fixed_mu_scan(
        np.diag([-1.0, 1.0]),
        np.array([[1.0, 0.1], [0.1, 1.0]]),
        [-0.2, 0.4],
        kT=0.1,
    )

    for value in (
        result.mu_grid,
        result.eigvals,
        result.occupations,
        result.occupation_response,
        result.electron_count,
        result.dos_like_response,
        result.k_weights,
        result.min_overlap_eig,
        result.overlap_condition,
        result.energies.band_energy,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)

import pickle
from dataclasses import replace

import numpy as np
import pytest

import dptb.nnops.fixed_mu_operator as fixed_mu

#: cond(S) = 1: a sample whose certificate must never be relaxed by its batch.
WELL_CONDITIONED = (np.diag([-1.0, 1.0]), np.eye(2))
#: cond(S) = 1e12, right at the shipped ceiling.
ILL_CONDITIONED = (np.diag([-1.0e-12, 1.0]), np.diag([1.0e-12, 1.0]))


def _forge_scaled_eigvector(result, h, s, item):
    """Rebuild every derived array from a tampered eigenvector of one item."""

    eigvecs = np.asarray(result.eigvecs).copy()
    eigvecs[item, :, 0] *= 1.0001
    density_k = fixed_mu._density_from_vectors(eigvecs, np.asarray(result.occupations))
    density_response_k = fixed_mu._density_from_vectors(
        eigvecs, np.asarray(result.occupation_response)
    )
    ne = fixed_mu._trace_last2(density_k @ s)
    band = fixed_mu._trace_last2(density_k @ h)
    conservation = fixed_mu.ConservationLedger(
        electron_count_from_density=ne,
        band_energy_from_density=band,
        electron_count_residual=ne - np.asarray(result.electron_count),
        band_energy_residual=band - np.asarray(result.energies.band_energy),
        density_hermiticity_error=fixed_mu._density_hermiticity_error(density_k),
        input_h_hermiticity_error=np.zeros(h.shape[0]),
        input_s_hermiticity_error=np.zeros(h.shape[0]),
    )
    return replace(
        result,
        validation_context=result._validation_context,
        eigvecs=eigvecs,
        density_k=density_k,
        density_response_k=density_response_k,
        density=density_k,
        density_response=density_response_k,
        conservation=conservation,
    )


def _verdict(items, tampered):
    """Return the validation verdict for a batch, tampering one member.

    ``None`` means accepted; otherwise the error message.
    """

    h = np.stack([item[0] for item in items])
    s = np.stack([item[1] for item in items])
    result = fixed_mu.fixed_mu_observables(
        h, s, mu=0.0, eig_floor=1.0e-13, max_condition=1.0e12
    )
    if tampered is not None:
        result = _forge_scaled_eigvector(result, h, s, tampered)
    try:
        fixed_mu.validate_conservation(result)
    except fixed_mu.FixedMuOperatorError as exc:
        return str(exc)
    return None


def test_expected_raw_weights_are_normalized_like_the_request():
    h = np.stack([np.diag([-1.0, 0.5]), np.diag([-0.8, 0.6])])
    s = np.stack([np.eye(2), np.eye(2)])
    result = fixed_mu.fixed_mu_observables(
        h, s, mu=0.0, kT=0.1, k_weights=np.array([2.0, 1.0]), k_axis=0
    )
    fixed_mu.validate_conservation(result, expected_k_weights=np.array([2.0, 1.0]))


def test_mixed_batch_condition_does_not_relax_good_item():
    h = np.stack([np.diag([-1.0, 1.0]), np.diag([-1.0e-12, 1.0])])
    s = np.stack([np.eye(2), np.diag([1.0e-12, 1.0])])
    result = fixed_mu.fixed_mu_observables(
        h, s, mu=0.0, eig_floor=1.0e-13, max_condition=1.0e12
    )
    eigvecs = np.asarray(result.eigvecs).copy()
    eigvecs[0, :, 0] *= 1.0001
    density_k = fixed_mu._density_from_vectors(eigvecs, np.asarray(result.occupations))
    density_response_k = fixed_mu._density_from_vectors(
        eigvecs, np.asarray(result.occupation_response)
    )
    ne = fixed_mu._trace_last2(density_k @ s)
    band = fixed_mu._trace_last2(density_k @ h)
    conservation = fixed_mu.ConservationLedger(
        electron_count_from_density=ne,
        band_energy_from_density=band,
        electron_count_residual=ne - np.asarray(result.electron_count),
        band_energy_residual=band - np.asarray(result.energies.band_energy),
        density_hermiticity_error=fixed_mu._density_hermiticity_error(density_k),
        input_h_hermiticity_error=np.zeros(2),
        input_s_hermiticity_error=np.zeros(2),
    )
    forged = replace(
        result,
        validation_context=result._validation_context,
        eigvecs=eigvecs,
        density_k=density_k,
        density_response_k=density_response_k,
        density=density_k,
        density_response=density_response_k,
        conservation=conservation,
    )
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="S-orthonormality"):
        fixed_mu.validate_conservation(forged)


@pytest.mark.parametrize(
    "items, tampered",
    [
        ([WELL_CONDITIONED], 0),
        ([WELL_CONDITIONED, WELL_CONDITIONED], 0),
        ([WELL_CONDITIONED, ILL_CONDITIONED], 0),
        ([ILL_CONDITIONED, WELL_CONDITIONED], 1),
        ([ILL_CONDITIONED, WELL_CONDITIONED, ILL_CONDITIONED], 1),
        ([ILL_CONDITIONED, ILL_CONDITIONED, WELL_CONDITIONED], 2),
    ],
)
def test_batch_order_and_composition_cannot_change_a_good_sample_verdict(items, tampered):
    """Position and company must not decide whether a forged sample is caught."""

    solo = _verdict([WELL_CONDITIONED], 0)
    assert solo is not None and "S-orthonormality" in solo
    assert _verdict(items, tampered) == solo


@pytest.mark.parametrize(
    "items",
    [
        [WELL_CONDITIONED, ILL_CONDITIONED],
        [ILL_CONDITIONED, WELL_CONDITIONED],
        [ILL_CONDITIONED, ILL_CONDITIONED, WELL_CONDITIONED],
    ],
)
def test_untampered_mixed_batches_still_validate(items):
    """The per-item budget must not turn into a false rejection either: the
    ill-conditioned member keeps the loose budget its own cond(S) earns."""

    assert _verdict(items, None) is None


def test_expected_weights_bind_when_the_result_has_no_k_axis():
    """A ``k_axis=None`` result still carries real stored weights, so the caller
    must be able to state them."""

    single = fixed_mu.fixed_mu_observables(*WELL_CONDITIONED, mu=0.0)
    fixed_mu.validate_conservation(single, expected_k_weights=1.0)
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="expected k_weights"):
        fixed_mu.validate_conservation(single, expected_k_weights=0.5)

    h = np.stack([WELL_CONDITIONED[0], WELL_CONDITIONED[0]])
    s = np.stack([WELL_CONDITIONED[1], WELL_CONDITIONED[1]])
    batched = fixed_mu.fixed_mu_observables(h, s, mu=0.0)
    fixed_mu.validate_conservation(batched, expected_k_weights=np.ones(2))
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="expected k_weights"):
        fixed_mu.validate_conservation(batched, expected_k_weights=np.array([1.0, 2.0]))


def test_request_policy_can_bind_axis_normalization_and_safety():
    h = np.zeros((2, 2, 1, 1))
    h[..., 0, 0] = np.array([[-1.0, 1.0], [-1.0, -1.0]])
    s = np.ones_like(h)
    result = fixed_mu.fixed_mu_observables(
        h, s, mu=0.0, k_axis=1, normalize_k_weights=True
    )
    fixed_mu.validate_conservation(
        result,
        expected_k_axis=1,
        expected_normalize_k_weights=True,
        expected_eig_floor=1.0e-10,
        expected_max_condition=1.0e12,
        expected_hermitian_tol=1.0e-8,
    )
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="k_axis"):
        fixed_mu.validate_conservation(result, expected_k_axis=0)
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="normalize_k_weights"):
        fixed_mu.validate_conservation(result, expected_normalize_k_weights=False)
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="max_condition"):
        fixed_mu.validate_conservation(result, expected_max_condition=1.0e6)


@pytest.mark.parametrize("value", ["false", "true", "", 1, 0, 1.0, None])
def test_normalize_k_weights_rejects_truthy_non_boolean_values(value):
    h = np.diag([-1.0, 1.0])
    s = np.eye(2)
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="must be a boolean"):
        fixed_mu.fixed_mu_observables(h, s, mu=0.0, normalize_k_weights=value)


def test_deserialized_result_binds_the_full_request_and_safety_policy():
    """A pickled result loses its private H/S context, so every scalar it carries
    is self-declared data until the caller binds it back to the request."""

    h = np.zeros((2, 2, 1, 1))
    h[..., 0, 0] = np.array([[-1.0, 1.0], [-1.0, -1.0]])
    s = np.ones_like(h)
    result = fixed_mu.fixed_mu_observables(
        h,
        s,
        mu=0.25,
        kT=0.1,
        spin_degeneracy=2.0,
        k_weights=np.array([2.0, 1.0]),
        k_axis=1,
        normalize_k_weights=True,
    )
    restored = pickle.loads(pickle.dumps(result))
    with pytest.raises(fixed_mu.FixedMuOperatorError, match="missing H/S context"):
        fixed_mu.validate_conservation(restored)

    binding = dict(
        h=h,
        s=s,
        expected_mu=0.25,
        expected_kT=0.1,
        expected_spin_degeneracy=2.0,
        expected_k_weights=np.array([2.0, 1.0]),
        expected_k_axis=1,
        expected_normalize_k_weights=True,
        expected_eig_floor=1.0e-10,
        expected_max_condition=1.0e12,
        expected_hermitian_tol=1.0e-8,
    )
    fixed_mu.validate_conservation(restored, **binding)

    for key, wrong, message in [
        ("expected_mu", 0.5, "computed at mu="),
        ("expected_kT", 0.2, "computed at kT="),
        ("expected_spin_degeneracy", 1.0, "spin_degeneracy"),
        ("expected_k_weights", np.array([1.0, 2.0]), "expected k_weights"),
        ("expected_k_axis", 0, "k_axis"),
        ("expected_normalize_k_weights", False, "normalize_k_weights"),
        ("expected_eig_floor", 1.0e-12, "eig_floor"),
        ("expected_max_condition", 1.0e6, "max_condition"),
        ("expected_hermitian_tol", 1.0e-10, "hermitian_tol"),
    ]:
        with pytest.raises(fixed_mu.FixedMuOperatorError, match=message):
            fixed_mu.validate_conservation(restored, **{**binding, key: wrong})


def test_result_arrays_stay_frozen_across_a_pickle_round_trip():
    result = fixed_mu.fixed_mu_observables(*WELL_CONDITIONED, mu=0.0)
    restored = pickle.loads(pickle.dumps(result))
    for field_name in (
        "eigvals",
        "eigvecs",
        "occupations",
        "occupation_response",
        "density_k",
        "density_response_k",
        "electron_count",
        "dos_like_response",
        "density",
        "density_response",
        "k_weights",
        "min_overlap_eig",
        "overlap_condition",
    ):
        with pytest.raises(ValueError):
            getattr(restored, field_name).setflags(write=True)
    with pytest.raises(ValueError):
        restored.energies.band_energy.setflags(write=True)
    with pytest.raises(ValueError):
        restored.conservation.electron_count_from_density.setflags(write=True)

"""Spectral metadata is preserved separately from graph-aligned tensor decoding."""

from types import SimpleNamespace
import pytest
import torch
from dptb.data.dataset.spectral_targets import attach_spectral_targets
from dptb.data import AtomicData, AtomicDataDict as A


def test_explicit_fractional_electrons_survive_and_absence_is_not_guessed():
    out = {}
    attach_spectral_targets(SimpleNamespace(), {"nelec": 2.5}, out)
    assert out["nelec"].item() == 2.5
    out = {}
    attach_spectral_targets(SimpleNamespace(), {"atomic_numbers": [1, 1]}, out)
    assert "nelec" not in out


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), [1.0, 2.0]])
def test_invalid_electron_metadata_rejected(value):
    with pytest.raises(ValueError, match="electron count"):
        attach_spectral_targets(SimpleNamespace(), {"nelec": value}, {})


def test_band_labels_are_decoded_without_overwriting_overlap():
    out = {A.NODE_OVERLAP_KEY: torch.ones(2, 4), A.EDGE_OVERLAP_KEY: torch.ones(2, 4)}
    original = out[A.EDGE_OVERLAP_KEY]
    record = {
        "kpoint": [[0.0, 0.0, 0.0]],
        "eigenvalue": [[-1.0, 1.0]],
        "edge_overlap": [[999.0]],
    }
    attach_spectral_targets(
        SimpleNamespace(get_overlap=True, get_eigenvalues=True), record, out
    )
    assert out[A.EDGE_OVERLAP_KEY] is original
    assert out[A.ENERGY_EIGENVALUE_KEY].shape == (1, 1, 2)


def test_requested_labels_missing_fail_closed():
    with pytest.raises(KeyError, match="get_eigenvalues"):
        attach_spectral_targets(SimpleNamespace(get_eigenvalues=True), {}, {})
    with pytest.raises(KeyError, match="get_overlap"):
        attach_spectral_targets(SimpleNamespace(get_overlap=True), {}, {})


def test_band_k_count_mismatch_rejected():
    record = {"kpoint": [[0.0, 0.0, 0.0]], "eigenvalue": [[-1.0, 1.0], [-2.0, 2.0]]}
    with pytest.raises(ValueError, match="matching kpoint"):
        attach_spectral_targets(SimpleNamespace(get_eigenvalues=True), record, {})


def test_atomicdata_container_preserves_fields():
    out = AtomicData()
    out[A.NODE_OVERLAP_KEY] = torch.ones(1, 1)
    out[A.EDGE_OVERLAP_KEY] = torch.ones(1, 1)
    attach_spectral_targets(SimpleNamespace(get_overlap=True), {"nelec": 1.5}, out)
    assert out["nelec"].item() == 1.5


def test_electron_count_has_declared_graph_alignment_in_strict_ensemble():
    from dptb.nn.output_spec import default_output_spec
    from dptb.tests.test_distance_ensemble_stitch import _make_wrapper

    field = default_output_spec(strict=True).get("nelec")
    assert (
        field is not None and field.alignment == "graph" and field.merge == "keep_first"
    )
    wrapper = _make_wrapper(strict=True)
    base = {"nelec": torch.tensor([88.0, 45.0])}
    other = {"nelec": torch.tensor([7.0, 8.0])}
    mask = torch.tensor([True, False])
    wrapper._stitch_outputs(base, other, mask, mask)
    torch.testing.assert_close(base["nelec"], torch.tensor([88.0, 45.0]))

import json
from pathlib import Path

import pytest
import torch

from dptb.nn.embedding.ao_angular_projector import AOAngularProjectorHead
from dptb.nn.embedding.ao_projector_bank import (
    build_ao_decoder_irreps,
    export_projector_bank,
    load_projector_bank_with_provenance,
    projector_bank_provenance,
    reference_projector,
)
from dptb.nn.embedding.cartesian_ict_bank import (
    cartesian_ict_projector,
    export_cartesian_ict_projector_bank,
)


def test_cartesian_ict_export_has_auditable_provenance(tmp_path):
    full_basis = ("s", "p")
    path = export_cartesian_ict_projector_bank(
        tmp_path / "ict_projectors.json", full_basis
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == "deeptb.ao_angular_projector/v2"
    assert payload["source"] == "cartesian_ict"
    assert payload["shell_order"] == list(full_basis)
    assert payload["generator"]["id"] == "deeptb.cartesian_stf_3j/v1"
    assert payload["generator"]["kind"] == "irreducible_cartesian_tensor"
    assert payload["generator"]["reference_projector_used_as_output"] is False
    assert payload["validation"]["passed"] is True
    assert payload["validation"]["max_abs_error"] <= payload["validation"]["atol"]
    assert max(payload["validation"]["compiled_vs_explicit_max_abs_error"].values()) < 1.0e-12

    bank, provenance = load_projector_bank_with_provenance(path, full_basis)
    assert provenance.uses_ict is True
    assert provenance.source == "cartesian_ict"
    assert provenance.generator_id == "deeptb.cartesian_stf_3j/v1"
    torch.testing.assert_close(bank["1,1,2"], reference_projector(1, 1, 2))


def test_cartesian_projector_is_generated_before_reference_validation():
    actual, explicit_error = cartesian_ict_projector(1, 2, 3)
    expected = reference_projector(1, 2, 3)
    assert explicit_error < 1.0e-12
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-10)


def test_source_string_alone_cannot_enable_ict(tmp_path):
    path = export_projector_bank(tmp_path / "reference.json", ("s", "p"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"] = "cartesian_ict"
    payload["generator"] = {
        "id": "deeptb.cartesian_stf_3j/v1",
        "kind": "irreducible_cartesian_tensor",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    provenance = projector_bank_provenance(path, ("s", "p"))
    assert provenance.schema.endswith("/v1")
    assert provenance.uses_ict is False


def test_untrusted_generator_does_not_enable_ict(tmp_path):
    path = export_cartesian_ict_projector_bank(
        tmp_path / "ict_projectors.json", ("s", "p")
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generator"]["id"] = "unknown.cartesian_generator/v9"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, provenance = load_projector_bank_with_provenance(path, ("s", "p"))
    assert provenance.uses_ict is False


def test_ao_head_uses_ict_only_for_validated_cartesian_bank(tmp_path):
    full_basis = ("s", "p")
    irreps = build_ao_decoder_irreps(full_basis, channels=0)
    ict_path = export_cartesian_ict_projector_bank(
        tmp_path / "ict_projectors.json", full_basis
    )
    reference_path = export_projector_bank(
        tmp_path / "reference_projectors.json", full_basis
    )

    ict = AOAngularProjectorHead(
        irreps,
        full_basis,
        symmetrize=False,
        projector_backend="precomputed",
        projector_bank_path=ict_path,
        rank=4,
        dtype=torch.float64,
    )
    reference = AOAngularProjectorHead(
        irreps,
        full_basis,
        symmetrize=False,
        projector_backend="precomputed",
        projector_bank_path=reference_path,
        rank=4,
        dtype=torch.float64,
    )

    assert ict.uses_precomputed_projector is True
    assert ict.uses_ict is True
    assert ict.projector_source == "cartesian_ict"
    assert ict.projector_provenance.generator_id == "deeptb.cartesian_stf_3j/v1"
    assert reference.uses_precomputed_projector is True
    assert reference.uses_ict is False
    assert reference.projector_source == "reference_wigner"

    reference.load_state_dict(ict.state_dict(), strict=True)
    features = torch.randn(3, irreps.dim, dtype=torch.float64)
    torch.testing.assert_close(ict(features), reference(features), rtol=0.0, atol=2.0e-10)


def test_v2_shell_order_must_match_orbital_mapper_order(tmp_path):
    path = export_cartesian_ict_projector_bank(
        tmp_path / "ict_projectors.json", ("s", "p")
    )
    with pytest.raises(ValueError, match="shell_order"):
        load_projector_bank_with_provenance(path, ("p", "s"))

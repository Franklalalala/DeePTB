from pathlib import Path

import pytest
import torch
from e3nn import o3

from dptb.nn.embedding.ao_projector_bank import (
    build_ao_decoder_irreps,
    export_projector_bank,
)
from dptb.nn.embedding.cartesian_ict_bank import (
    export_cartesian_ict_projector_bank,
)
from dptb.nn.embedding.output_routes import (
    OFFICIAL_OUTPUT_ROUTES,
    OutputHeadContext,
    build_output_heads,
    effective_product_scope,
    get_output_route_spec,
    normalize_legacy_head_mode,
    resolve_output_route,
    select_final_irreps,
    validate_prediction_route,
)


def test_official_route_registry_has_exact_six_route_matrix():
    assert OFFICIAL_OUTPUT_ROUTES == (
        "h_a0",
        "h_a1",
        "h_b0",
        "h_b1",
        "p_b0",
        "p_b1_ict",
    )
    expected = {
        "h_a0": ("ordinary_hidden", "rme", "e3tb", None, True, False),
        "h_a1": ("ordinary_hidden", "rme", "e3tb", None, True, True),
        "h_b0": ("ordinary_hidden", "ao_block", "block_native", "expansion_cg", False, False),
        "h_b1": ("ordinary_hidden", "ao_block", "block_native", "cartesian_projector", False, True),
        "p_b0": ("ao_pair", "ao_block", "block_native", "ao_projector", False, False),
        "p_b1_ict": ("ao_pair", "ao_block", "block_native", "ao_projector", False, True),
    }
    for name, values in expected.items():
        spec = get_output_route_spec(name)
        assert (
            spec.final_irreps_kind,
            spec.output_contract,
            spec.prediction_method,
            spec.block_decoder,
            spec.uses_e3hamiltonian,
            spec.uses_ict,
        ) == values


def test_legacy_aliases_resolve_without_changing_semantics():
    aliases = {
        "late_rme_expansion_nocg": "h_a0",
        "late_rme_cartesian_hybrid": "h_a1",
        "late_block_expansion_cg": "h_b0",
        "late_block_cartesian_projector": "h_b1",
        "rme_fusion": "rme_fusion",
        "block_native_linear": "debug_block_linear",
    }
    for alias, canonical in aliases.items():
        with pytest.warns(DeprecationWarning):
            spec = resolve_output_route(legacy_mode=alias)
        assert spec.canonical_name == canonical
        assert normalize_legacy_head_mode(alias) == spec.legacy_mode


def test_alias_in_canonical_field_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="output_route='h_a0'"):
        spec = resolve_output_route(output_route="late_nocg")
    assert spec.canonical_name == "h_a0"


def test_legacy_direct_ao_alias_resolves_from_backend_and_provenance(tmp_path):
    reference = export_projector_bank(tmp_path / "reference.json", ("s", "p"))
    ict = export_cartesian_ict_projector_bank(tmp_path / "ict.json", ("s", "p"))

    with pytest.warns(DeprecationWarning):
        assert resolve_output_route(
            legacy_mode="direct_ao_projector",
            projector_backend="reference_wigner",
        ).canonical_name == "p_b0"
    with pytest.warns(DeprecationWarning):
        assert resolve_output_route(
            legacy_mode="direct_ao_projector",
            projector_backend="precomputed",
            projector_bank_path=reference,
        ).canonical_name == "p_b1_reference"
    with pytest.warns(DeprecationWarning):
        assert resolve_output_route(
            legacy_mode="direct_ao_projector",
            projector_backend="precomputed",
            projector_bank_path=ict,
        ).canonical_name == "p_b1_ict"


def test_canonical_p_routes_reject_wrong_provenance(tmp_path):
    reference = export_projector_bank(tmp_path / "reference.json", ("s", "p"))
    ict = export_cartesian_ict_projector_bank(tmp_path / "ict.json", ("s", "p"))
    with pytest.raises(ValueError, match="validated Cartesian/ICT"):
        resolve_output_route(
            output_route="p_b1_ict",
            projector_backend="precomputed",
            projector_bank_path=reference,
        )
    with pytest.raises(ValueError, match="non-ICT reference/control"):
        resolve_output_route(
            output_route="p_b1_reference",
            projector_backend="precomputed",
            projector_bank_path=ict,
        )


def test_official_product_policies_are_fixed():
    assert effective_product_scope(get_output_route_spec("h_a1"), "all") == "all"
    assert (
        effective_product_scope(get_output_route_spec("h_b1"), "missing_only")
        == "missing_only"
    )
    with pytest.raises(ValueError, match="fixes product_scope"):
        effective_product_scope(get_output_route_spec("h_a1"), "missing_only")


def test_prediction_validation_is_registry_driven():
    validate_prediction_route(
        get_output_route_spec("h_a1"), {"method": "e3tb", "scale_type": "no_scale"}
    )
    validate_prediction_route(
        get_output_route_spec("h_b1"),
        {"method": "block_native", "block_decoder": "cartesian_projector"},
    )
    with pytest.raises(ValueError, match="prediction.method"):
        validate_prediction_route(
            get_output_route_spec("h_a1"), {"method": "block_native"}
        )
    with pytest.raises(ValueError, match="block_decoder"):
        validate_prediction_route(
            get_output_route_spec("h_b1"),
            {"method": "block_native", "block_decoder": "expansion_cg"},
        )


def test_final_irreps_selection_uses_route_spec():
    hidden = o3.Irreps("2x0e+2x1o")
    orbpair = o3.Irreps("1x1o+2x0e")
    ao_pair = o3.Irreps("2x0e+1x1o")
    assert select_final_irreps(
        get_output_route_spec("h_a0"),
        ordinary_hidden=hidden,
        orbpair_irreps=orbpair,
        ao_pair_irreps=ao_pair,
    ) == hidden
    assert select_final_irreps(
        get_output_route_spec("legacy_rme"),
        ordinary_hidden=hidden,
        orbpair_irreps=orbpair,
        ao_pair_irreps=ao_pair,
    ) == orbpair.sort()[0].simplify()
    assert select_final_irreps(
        get_output_route_spec("p_b0"),
        ordinary_hidden=hidden,
        orbpair_irreps=orbpair,
        ao_pair_irreps=ao_pair,
    ) == ao_pair


def _p_context(full_basis, irreps, *, backend, bank_path=None):
    return OutputHeadContext(
        final_irreps=irreps,
        orbpair_irreps=irreps,
        full_basis=tuple(full_basis),
        max_norb=4,
        rank=4,
        init=0.0,
        condition="scalar_0e",
        product_scope="missing_only",
        ao_projector_normalization="e3hamiltonian",
        ao_projector_basis_convention="deeptb_real_ao",
        ao_projector_backend=backend,
        ao_projector_bank_path=None if bank_path is None else str(bank_path),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )


def test_registry_factory_enforces_p_route_runtime_metadata(tmp_path):
    full_basis = ("s", "p")
    irreps = build_ao_decoder_irreps(full_basis)
    ict_path = export_cartesian_ict_projector_bank(tmp_path / "ict.json", full_basis)

    spec = resolve_output_route(
        output_route="p_b1_ict",
        projector_backend="precomputed",
        projector_bank_path=ict_path,
    )
    edge, node = build_output_heads(
        spec,
        _p_context(
            full_basis, irreps, backend="precomputed", bank_path=ict_path
        ),
    )
    assert type(edge).__name__ == spec.head_class_name
    assert type(node).__name__ == spec.head_class_name
    assert edge.uses_ict is True
    assert edge.uses_precomputed_projector is True

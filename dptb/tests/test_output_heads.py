"""Output head classes in isolation: equivariance, provenance/coverage flags, and
the RME/AO-pair/Cartesian-ICT projector backends they can be built with."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from e3nn import o3

from dptb.nn.embedding.ao_angular_projector import AOAngularProjectorHead
from dptb.nn.embedding.ao_projector_bank import (
    build_ao_decoder_irreps,
    export_projector_bank,
    load_projector_bank_with_provenance,
    projector_bank_provenance,
    reference_projector,
    shell_l,
)
from dptb.nn.embedding.block_native_head import apply_ao_basis_mask
from dptb.nn.embedding.cartesian_ict_bank import cartesian_ict_projector, export_cartesian_ict_projector_bank
from dptb.nn.embedding.output_routes import OutputHeadContext, build_output_heads, resolve_output_route
from dptb.nn.embedding.rme_nocg_fusion_head import RMENoCGFusionHead, normalize_rme_head_mode


# ---------------------------------------------------------------------------
# Official route heads: equivariance, cartesian paths, provenance, masks, grads
# ---------------------------------------------------------------------------

FULL_BASIS = ("s", "p")
HIDDEN = o3.Irreps("4x0e+4x1o+4x1e+4x2e")
AO_PAIR = build_ao_decoder_irreps(FULL_BASIS)
AO_IRREPS = o3.Irreps([(1, (shell_l(shell), (-1) ** shell_l(shell))) for shell in FULL_BASIS])


def _context(route: str, tmp_path: Path) -> tuple:
    backend = "reference_wigner"
    bank = None
    final_irreps = HIDDEN
    product_scope = "missing_only"
    if route == "h_a1":
        product_scope = "all"
    if route in {"p_b0", "p_b1_ict"}:
        final_irreps = AO_PAIR
    if route == "p_b1_ict":
        backend = "precomputed"
        bank = export_cartesian_ict_projector_bank(tmp_path / "sp_ict_projectors.json", FULL_BASIS)

    spec = resolve_output_route(output_route=route, projector_backend=backend, projector_bank_path=bank)
    ctx = OutputHeadContext(
        final_irreps=final_irreps, orbpair_irreps=AO_PAIR, full_basis=FULL_BASIS, max_norb=AO_IRREPS.dim,
        rank=4, init=0.0, condition="scalar_0e", product_scope=product_scope,
        ao_projector_normalization="e3hamiltonian", ao_projector_basis_convention="deeptb_real_ao",
        ao_projector_backend=backend, ao_projector_bank_path=None if bank is None else str(bank),
        dtype=torch.float64, device=torch.device("cpu"),
    )
    edge, _ = build_output_heads(spec, ctx)
    return spec, edge


def _assert_equivariant(spec, head, transform: torch.Tensor) -> None:
    torch.manual_seed(20260621)
    x = torch.randn(3, head.irreps_in.dim, dtype=torch.float64)
    d_in = head.irreps_in.D_from_matrix(transform)
    actual = head(x @ d_in.T)
    expected_raw = head(x)
    if spec.output_contract == "rme":
        d_out = head.irreps_out.D_from_matrix(transform)
        expected = expected_raw @ d_out.T
    else:
        d_ao = AO_IRREPS.D_from_matrix(transform)
        expected = d_ao @ expected_raw @ d_ao.T
    torch.testing.assert_close(actual, expected, rtol=3.0e-6, atol=3.0e-6)


@pytest.mark.parametrize("route", ("h_a0", "h_a1", "h_b0", "h_b1", "p_b0", "p_b1_ict"))
def test_official_heads_are_equivariant_under_proper_rotation(route, tmp_path):
    spec, head = _context(route, tmp_path)
    _assert_equivariant(spec, head, o3.rand_matrix(dtype=torch.float64))


@pytest.mark.parametrize("route", ("h_a0", "h_a1", "h_b0", "h_b1", "p_b0", "p_b1_ict"))
def test_official_heads_are_equivariant_under_inversion(route, tmp_path):
    spec, head = _context(route, tmp_path)
    _assert_equivariant(spec, head, -torch.eye(3, dtype=torch.float64))


def test_h_a1_and_h_b1_execute_the_intended_cartesian_paths(tmp_path):
    _, h_a1 = _context("h_a1", tmp_path)
    _, h_b1 = _context("h_b1", tmp_path)
    assert h_a1.coverage_report["product_paths"] > 0
    assert h_a1.uses_ict is True
    assert hasattr(h_a1, "left") and hasattr(h_a1, "right")
    assert h_b1.coverage_report["direct_paths"] > 0
    assert h_b1.coverage_report["product_paths"] == 0
    assert not hasattr(h_b1, "left")
    assert not hasattr(h_b1, "right")


def test_true_p_b1_provenance_controls_runtime_flags(tmp_path):
    spec, head = _context("p_b1_ict", tmp_path)
    assert spec.uses_ict is True
    assert spec.uses_precomputed_projector is True
    assert head.uses_ict is True
    assert head.uses_precomputed_projector is True
    assert head.projector_source == "cartesian_ict"
    assert head.projector_provenance.generator_id == "deeptb.cartesian_stf_3j/v1"
    assert head.projector_provenance.validation_passed is True


def test_ao_shell_slices_and_atom_bond_masks_align(tmp_path):
    _, head = _context("p_b1_ict", tmp_path)
    x = torch.randn(2, AO_PAIR.dim, dtype=torch.float64)
    blocks = head(x)
    assert blocks.shape == (2, 4, 4)

    shell_slices = (slice(0, 1), slice(1, 4))
    for row_slice, row_l in zip(shell_slices, (0, 1)):
        for col_slice, col_l in zip(shell_slices, (0, 1)):
            assert blocks[:, row_slice, col_slice].shape[-2:] == (2 * row_l + 1, 2 * col_l + 1)

    # Atom 0 has s only; atom 1 has s+p. Directed bond masks use the source
    # mask on rows and destination mask on columns.
    atom_masks = torch.tensor([[True, False, False, False], [True, True, True, True]])
    node_masked = apply_ao_basis_mask(blocks, atom_masks)
    assert torch.count_nonzero(node_masked[0, 1:, :]) == 0
    assert torch.count_nonzero(node_masked[0, :, 1:]) == 0

    edge_masked = apply_ao_basis_mask(blocks, atom_masks[[0, 1]], atom_masks[[1, 0]])
    expected = atom_masks[[0, 1]].unsqueeze(-1) & atom_masks[[1, 0]].unsqueeze(-2)
    assert torch.equal(edge_masked.ne(0), blocks.ne(0) & expected)


def test_all_trainable_parameters_participate_in_backward(tmp_path):
    for route in ("h_a0", "h_a1", "h_b0", "h_b1", "p_b0", "p_b1_ict"):
        _, head = _context(route, tmp_path)
        head.zero_grad(set_to_none=True)
        x = torch.randn(2, head.irreps_in.dim, dtype=torch.float64, requires_grad=True)
        head(x).square().mean().backward()
        unused = [name for name, p in head.named_parameters() if p.requires_grad and p.grad is None]
        assert unused == [], f"{route} has unused trainable parameters: {unused}"


# ---------------------------------------------------------------------------
# RME NoCG fusion head: legacy-equivalent zero-init, training, equivariance
# ---------------------------------------------------------------------------

def _rme_nocg_head(init=0.0, dtype=torch.float64):
    return RMENoCGFusionHead("4x0e + 3x1o + 2x2e", "3x0e + 2x1o + 2x2e", rank=5, init=init, dtype=dtype)


def test_rme_nocg_mode_normalization_and_validation():
    assert normalize_rme_head_mode(None) == "legacy_linear"
    assert normalize_rme_head_mode("nocg") == "rme_nocg_fusion"
    with pytest.raises(ValueError):
        normalize_rme_head_mode("expansion_uuw")


def test_rme_nocg_zero_init_is_exact_legacy_projection():
    torch.manual_seed(7)
    head = _rme_nocg_head(init=0.0)
    x = torch.randn(11, head.irreps_in.dim, dtype=torch.float64)
    assert torch.equal(head(x), head.legacy(x))


def test_rme_nocg_residual_update_is_trainable_and_shape_preserving():
    torch.manual_seed(11)
    head = _rme_nocg_head(init=1.0e-2)
    x = torch.randn(9, head.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    y = head(x)
    assert y.shape == (9, head.irreps_out.dim)
    y.square().mean().backward()
    assert head.scale_up.weight.grad is not None
    assert torch.isfinite(head.scale_up.weight.grad).all()


def test_rme_nocg_rotation_equivariance():
    torch.manual_seed(19)
    head = _rme_nocg_head(init=1.0e-2)
    x = torch.randn(6, head.irreps_in.dim, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    d_in, d_out = head.irreps_in.D_from_matrix(rotation), head.irreps_out.D_from_matrix(rotation)
    torch.testing.assert_close(head(x @ d_in.T), head(x) @ d_out.T, rtol=2.0e-9, atol=2.0e-9)


def test_rme_nocg_strict_load_accepts_legacy_linear_state_dict():
    irreps_in, irreps_out = o3.Irreps("4x0e + 3x1o + 2x2e"), o3.Irreps("3x0e + 2x1o + 2x2e")
    torch.manual_seed(23)
    legacy = o3.Linear(irreps_in, irreps_out, biases=True).to(dtype=torch.float64)
    head = RMENoCGFusionHead(irreps_in, irreps_out, rank=5, init=0.0, dtype=torch.float64)
    head.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randn(4, irreps_in.dim, dtype=torch.float64)
    assert torch.equal(head(x), legacy(x))


# ---------------------------------------------------------------------------
# Cartesian-ICT projector bank: matches the reference, forgery is rejected
# ---------------------------------------------------------------------------

def test_cartesian_ict_export_has_auditable_provenance(tmp_path):
    full_basis = ("s", "p")
    path = export_cartesian_ict_projector_bank(tmp_path / "ict_projectors.json", full_basis)
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
    payload["generator"] = {"id": "deeptb.cartesian_stf_3j/v1", "kind": "irreducible_cartesian_tensor"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    provenance = projector_bank_provenance(path, ("s", "p"))
    assert provenance.schema.endswith("/v1")
    assert provenance.uses_ict is False


def test_untrusted_generator_does_not_enable_ict(tmp_path):
    path = export_cartesian_ict_projector_bank(tmp_path / "ict_projectors.json", ("s", "p"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generator"]["id"] = "unknown.cartesian_generator/v9"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, provenance = load_projector_bank_with_provenance(path, ("s", "p"))
    assert provenance.uses_ict is False


def test_ao_head_uses_ict_only_for_validated_cartesian_bank(tmp_path):
    full_basis = ("s", "p")
    irreps = build_ao_decoder_irreps(full_basis, channels=0)
    ict_path = export_cartesian_ict_projector_bank(tmp_path / "ict_projectors.json", full_basis)
    reference_path = export_projector_bank(tmp_path / "reference_projectors.json", full_basis)

    ict = AOAngularProjectorHead(irreps, full_basis, symmetrize=False, projector_backend="precomputed",
                                  projector_bank_path=ict_path, rank=4, dtype=torch.float64)
    reference = AOAngularProjectorHead(irreps, full_basis, symmetrize=False, projector_backend="precomputed",
                                       projector_bank_path=reference_path, rank=4, dtype=torch.float64)

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
    path = export_cartesian_ict_projector_bank(tmp_path / "ict_projectors.json", ("s", "p"))
    with pytest.raises(ValueError, match="shell_order"):
        load_projector_bank_with_provenance(path, ("p", "s"))


# ---------------------------------------------------------------------------
# Route-matrix head-level contracts: normalization, RME vs AO-block heads,
# reference vs precomputed AO-pair backends
# ---------------------------------------------------------------------------

def test_route_matrix_modes_are_normalized():
    expected = {
        "late_rme_expansion_nocg": "late_rme_expansion_nocg",
        "late_nocg": "late_rme_expansion_nocg",
        "late_rme_cartesian_hybrid": "late_rme_cartesian_hybrid",
        "late_rme_ict_hybrid": "late_rme_cartesian_hybrid",
        "late_block_expansion_cg": "late_block_expansion_cg",
        "late_block_cartesian_projector": "late_block_cartesian_projector",
        "late_block_ict_projector": "late_block_cartesian_projector",
        "direct_ao_projector": "direct_ao_projector",
        "ao_projector": "direct_ao_projector",
    }
    for value, normalized in expected.items():
        assert normalize_rme_head_mode(value) == normalized


def test_ordinary_hidden_rme_heads_keep_rme_contract():
    from dptb.nn.embedding.late_rme_cartesian_hybrid import LateRMECartesianHybridHead
    from dptb.nn.embedding.late_rme_expansion_nocg import LateRMEExpansionNoCGHead

    irreps_in = o3.Irreps("4x0e+3x1o+3x1e+2x2e")
    irreps_out = o3.Irreps("2x0e+1x1o+1x1e+1x2e")
    x = torch.randn(5, irreps_in.dim, dtype=torch.float64)

    nocg = LateRMEExpansionNoCGHead(irreps_in, irreps_out, rank=4, init=0.0, dtype=torch.float64)
    ict = LateRMECartesianHybridHead(irreps_in, irreps_out, rank=4, init=0.0, product_scope="all", dtype=torch.float64)

    assert nocg.output_contract == "rme"
    assert nocg.performs_angular_coupling is False
    assert ict.output_contract == "rme"
    assert ict.performs_angular_coupling is True
    assert ict.uses_ict is True
    assert ict.coverage_report["product_paths"] > 0
    assert nocg(x).shape == (5, irreps_out.dim)
    assert ict(x).shape == (5, irreps_out.dim)


def test_ordinary_hidden_block_heads_bypass_rme_and_e3hamiltonian():
    from dptb.nn.embedding.late_block_cartesian_projector import LateBlockCartesianProjectorHead
    from dptb.nn.embedding.late_block_expansion_cg import LateBlockExpansionCGHead

    irreps_in = o3.Irreps("4x0e+4x1o+4x1e+4x2e")
    full_basis = ("s", "p")
    x = torch.randn(6, irreps_in.dim, dtype=torch.float64)

    wigner = LateBlockExpansionCGHead(irreps_in, full_basis, symmetrize=True, rank=4, dtype=torch.float64)
    ict = LateBlockCartesianProjectorHead(irreps_in, full_basis, symmetrize=False, rank=4,
                                          product_scope="missing_only", dtype=torch.float64)

    assert wigner.output_contract == "ao_block"
    assert wigner.bypasses_rme
    assert wigner.uses_ict is False
    assert ict.output_contract == "ao_block"
    assert ict.bypasses_rme
    assert ict.uses_ict is True
    assert ict.coverage_report["direct_paths"] > 0
    assert ict.coverage_report["product_paths"] == 0
    assert not hasattr(ict, "left")
    assert not hasattr(ict, "right")
    assert wigner(x).shape == (6, 4, 4)
    assert ict(x).shape == (6, 4, 4)


def test_ao_pair_recontract_reference_and_precomputed_backends_match_output(tmp_path):
    full_basis = ("s", "p")
    irreps = build_ao_decoder_irreps(full_basis, channels=0)
    assert irreps.dim == 16
    x = torch.randn(3, irreps.dim, dtype=torch.float64)

    reference = AOAngularProjectorHead(irreps, full_basis, symmetrize=False, projector_backend="reference_wigner",
                                       rank=4, dtype=torch.float64)
    bank_path = export_projector_bank(tmp_path / "reference_projectors.json", full_basis)
    payload = json.loads(Path(bank_path).read_text(encoding="utf-8"))
    assert payload["source"] == "reference_wigner"

    precomputed = AOAngularProjectorHead(irreps, full_basis, symmetrize=False, projector_backend="precomputed",
                                        projector_bank_path=bank_path, rank=4, dtype=torch.float64)

    assert reference.output_contract == "ao_block"
    assert reference.uses_ict is False
    assert precomputed.output_contract == "ao_block"
    assert precomputed.uses_precomputed_projector is True
    assert precomputed.projector_source == "reference_wigner"
    assert precomputed.uses_ict is False

    # Same underlying projector values (the bank was exported from reference_wigner):
    # loading one's weights into the other must reproduce identical output, not just shape.
    precomputed.load_state_dict(reference.state_dict(), strict=True)
    torch.testing.assert_close(reference(x), precomputed(x), rtol=0.0, atol=2.0e-10)


def test_reference_projector_exporter_does_not_forge_ict_source(tmp_path):
    with pytest.raises(ValueError, match="reference_wigner"):
        export_projector_bank(tmp_path / "bad_projectors.json", ("s", "p"), source="ict_cartesian")


def test_ao_pair_decoder_requires_complete_irreps():
    with pytest.raises(ValueError, match="do not cover shell pairs"):
        AOAngularProjectorHead("1x0e", ("s", "p"), symmetrize=False)

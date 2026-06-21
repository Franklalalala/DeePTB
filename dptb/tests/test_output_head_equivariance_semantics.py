from __future__ import annotations

from pathlib import Path

import pytest
import torch
from e3nn import o3

from dptb.nn.embedding.ao_projector_bank import (
    build_ao_decoder_irreps,
    shell_l,
)
from dptb.nn.embedding.block_native_head import apply_ao_basis_mask
from dptb.nn.embedding.cartesian_ict_bank import (
    export_cartesian_ict_projector_bank,
)
from dptb.nn.embedding.output_routes import (
    OutputHeadContext,
    build_output_heads,
    resolve_output_route,
)


FULL_BASIS = ("s", "p")
HIDDEN = o3.Irreps("4x0e+4x1o+4x1e+4x2e")
AO_PAIR = build_ao_decoder_irreps(FULL_BASIS)
AO_IRREPS = o3.Irreps(
    [(1, (shell_l(shell), (-1) ** shell_l(shell))) for shell in FULL_BASIS]
)


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
        bank = export_cartesian_ict_projector_bank(
            tmp_path / "sp_ict_projectors.json", FULL_BASIS
        )

    spec = resolve_output_route(
        output_route=route,
        projector_backend=backend,
        projector_bank_path=bank,
    )
    ctx = OutputHeadContext(
        final_irreps=final_irreps,
        orbpair_irreps=AO_PAIR,
        full_basis=FULL_BASIS,
        max_norb=AO_IRREPS.dim,
        rank=4,
        init=0.0,
        condition="scalar_0e",
        product_scope=product_scope,
        ao_projector_normalization="e3hamiltonian",
        ao_projector_basis_convention="deeptb_real_ao",
        ao_projector_backend=backend,
        ao_projector_bank_path=None if bank is None else str(bank),
        dtype=torch.float64,
        device=torch.device("cpu"),
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


@pytest.mark.parametrize(
    "route", ("h_a0", "h_a1", "h_b0", "h_b1", "p_b0", "p_b1_ict")
)
def test_official_heads_are_equivariant_under_proper_rotation(route, tmp_path):
    spec, head = _context(route, tmp_path)
    _assert_equivariant(spec, head, o3.rand_matrix(dtype=torch.float64))


@pytest.mark.parametrize(
    "route", ("h_a0", "h_a1", "h_b0", "h_b1", "p_b0", "p_b1_ict")
)
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
            block = blocks[:, row_slice, col_slice]
            assert block.shape[-2:] == (2 * row_l + 1, 2 * col_l + 1)

    # Atom 0 has s only; atom 1 has s+p.  Directed bond masks must use the
    # source mask on rows and destination mask on columns.
    atom_masks = torch.tensor(
        [[True, False, False, False], [True, True, True, True]]
    )
    node_masked = apply_ao_basis_mask(blocks, atom_masks)
    assert torch.count_nonzero(node_masked[0, 1:, :]) == 0
    assert torch.count_nonzero(node_masked[0, :, 1:]) == 0

    edge_masked = apply_ao_basis_mask(
        blocks,
        atom_masks[[0, 1]],
        atom_masks[[1, 0]],
    )
    expected = atom_masks[[0, 1]].unsqueeze(-1) & atom_masks[[1, 0]].unsqueeze(-2)
    assert torch.equal(edge_masked.ne(0), blocks.ne(0) & expected)


def test_all_trainable_parameters_participate_in_backward(tmp_path):
    for route in ("h_a0", "h_a1", "h_b0", "h_b1", "p_b0", "p_b1_ict"):
        _, head = _context(route, tmp_path)
        head.zero_grad(set_to_none=True)
        x = torch.randn(
            2, head.irreps_in.dim, dtype=torch.float64, requires_grad=True
        )
        head(x).square().mean().backward()
        unused = [name for name, p in head.named_parameters() if p.requires_grad and p.grad is None]
        assert unused == [], f"{route} has unused trainable parameters: {unused}"

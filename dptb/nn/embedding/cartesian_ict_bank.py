"""Generate projector banks through DeePTB's explicit Cartesian/STF route.

The generator does not relabel reference tensors.  It constructs an
irreducible Cartesian basis, forms Cartesian-3j couplings, compiles them back to
DeePTB's real-AO gauge, and only then compares the result with E3Hamiltonian's
Wigner projector as a convention check.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import e3nn
import torch

from .ao_projector_bank import (
    _PROJECTOR_SCHEMA_V2,
    _projector_key,
    load_projector_bank_with_provenance,
    reference_projector,
    required_projector_keys,
    shell_l,
)
from .cartesian_projector import CartesianShellPairCoupling, cartesian_irrep_basis


_CARTESIAN_GENERATOR_ID = "deeptb.cartesian_stf_3j/v1"
_CARTESIAN_M_ORDER = "e3nn_real_spherical_harmonics_component"


def _basis_diagnostics(ell: int) -> Dict[str, float]:
    to_cartesian, from_cartesian = cartesian_irrep_basis(
        ell, dtype=torch.float64, device=torch.device("cpu")
    )
    identity = torch.eye(2 * ell + 1, dtype=torch.float64)
    return {
        "roundtrip_max_abs_error": float(
            (from_cartesian.matmul(to_cartesian) - identity).abs().max().item()
        ),
        "deterministic_seed": int(0xC0A7E51 + ell),
    }


def cartesian_ict_projector(
    row_l: int, col_l: int, out_l: int
) -> Tuple[torch.Tensor, float]:
    """Return a projector produced through the explicit Cartesian-3j path."""
    coupling = CartesianShellPairCoupling(
        row_l,
        col_l,
        out_l,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    projector = coupling.compiled_projector.detach().cpu().clone()

    basis = torch.eye(coupling.in_dim, dtype=torch.float64)
    compiled = coupling(basis).permute(1, 2, 0).contiguous()
    explicit = coupling.forward_explicit(basis).permute(1, 2, 0).contiguous()
    explicit_error = float((compiled - explicit).abs().max().item())
    if not torch.allclose(projector, explicit, atol=2.0e-12, rtol=0.0):
        raise RuntimeError(
            "Compiled Cartesian projector disagrees with the explicit ICT path "
            f"for ({row_l}, {col_l}) -> {out_l}."
        )
    return projector, explicit_error


def export_cartesian_ict_projector_bank(
    path: Union[str, Path],
    full_basis: Sequence[str],
    *,
    validation_atol: float = 2.0e-10,
) -> Path:
    """Export a v2 bank with auditable Cartesian/ICT provenance."""
    if not full_basis:
        raise ValueError("full_basis must contain at least one shell.")
    validation_atol = float(validation_atol)
    if not math.isfinite(validation_atol) or validation_atol <= 0.0:
        raise ValueError("validation_atol must be finite and positive.")

    projectors = {}
    per_projector_error = {}
    compiled_vs_explicit = {}
    max_error = 0.0
    for key in required_projector_keys(full_basis):
        row_l, col_l, out_l = (int(value) for value in key.split(","))
        projector, explicit_error = cartesian_ict_projector(row_l, col_l, out_l)
        reference = reference_projector(row_l, col_l, out_l)
        error = float((projector - reference).abs().max().item())
        projectors[key] = projector.tolist()
        per_projector_error[key] = error
        compiled_vs_explicit[key] = explicit_error
        max_error = max(max_error, error)

    unique_l = sorted({shell_l(shell) for shell in full_basis})
    payload = {
        "schema": _PROJECTOR_SCHEMA_V2,
        "source": "cartesian_ict",
        "basis_convention": "deeptb_real_ao",
        "normalization": "e3hamiltonian",
        "shell_order": [str(shell) for shell in full_basis],
        "m_order": _CARTESIAN_M_ORDER,
        "generator": {
            "id": _CARTESIAN_GENERATOR_ID,
            "kind": "irreducible_cartesian_tensor",
            "algorithm": "symmetric_monomial_stf_cartesian_3j",
            "implementation": (
                "dptb.nn.embedding.cartesian_projector."
                "CartesianShellPairCoupling"
            ),
            "coupling_seed": "e3nn.o3.wigner_3j",
            "reference_projector_used_as_output": False,
            "torch_version": torch.__version__,
            "e3nn_version": e3nn.__version__,
        },
        "validation": {
            "reference": "deeptb_e3hamiltonian_wigner",
            "atol": validation_atol,
            "rtol": 0.0,
            "max_abs_error": max_error,
            "passed": max_error <= validation_atol,
            "basis": {str(ell): _basis_diagnostics(ell) for ell in unique_l},
            "compiled_vs_explicit_max_abs_error": compiled_vs_explicit,
            "per_projector_max_abs_error": per_projector_error,
        },
        "projectors": projectors,
    }
    if not payload["validation"]["passed"]:
        raise RuntimeError(
            "Cartesian ICT bank failed DeePTB convention validation: "
            f"max error {max_error:.3e} > {validation_atol:.3e}."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Re-load through the production validator.  Generation and validation are
    # separate code paths by design.
    _, provenance = load_projector_bank_with_provenance(
        path, full_basis, atol=validation_atol
    )
    if not provenance.uses_ict:
        raise RuntimeError("Generated bank did not receive validated ICT provenance.")
    return path

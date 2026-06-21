"""Convention-checked angular-projector banks for direct AO decoding.

This module is deliberately independent of the runtime head.  Cartesian-3j,
ICT, Gaunt, or other generators may emit this JSON schema after conversion to
DeePTB's real-AO gauge, parity, normalization, shell order, and m-order.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from e3nn import o3


_ANGULAR_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
_PROJECTOR_SCHEMA = "deeptb.ao_angular_projector/v1"
_REFERENCE_BACKEND = "reference_wigner"
_PRECOMPUTED_BACKEND = "precomputed"
_REFERENCE_SOURCES = {
    "reference_wigner",
    "reference_fixture_replaceable_by_cartesian_ict",
}


def shell_l(shell: str) -> int:
    labels = re.findall(r"[A-Za-z]", str(shell))
    if len(labels) != 1 or labels[0].lower() not in _ANGULAR_L:
        raise ValueError(f"Unsupported AO shell label {shell!r}.")
    return _ANGULAR_L[labels[0].lower()]


def build_ao_decoder_irreps(
    full_basis: Sequence[str], channels: Optional[int] = 0
) -> o3.Irreps:
    """Return the complete ordered AO-pair representation or a rank ablation.

    ``channels <= 0`` is the main route: each ``(L, parity)`` multiplicity is
    exactly the number of ordered shell-pair paths carrying that irrep.  The
    resulting representation has dimension ``max_norb**2`` and is therefore a
    full equivariant coordinate system for a directed AO block.  A positive
    value replaces every multiplicity by that fixed width and is an explicitly
    compressed decoder ablation.

    AO shell parity is ``(-1)^l``.  All product irreps are retained, including
    channels such as ``1e`` from ``p x p`` that are absent from a
    natural-parity-only hidden layout.
    """
    if not full_basis:
        raise ValueError("full_basis must contain at least one shell.")
    requested = 0 if channels is None else int(channels)

    path_counts: Dict[Tuple[int, int], int] = {}
    ao_dimension = 0
    for row_shell in full_basis:
        row_l = shell_l(row_shell)
        row_ir = o3.Irrep(row_l, (-1) ** row_l)
        ao_dimension += 2 * row_l + 1
        for col_shell in full_basis:
            col_l = shell_l(col_shell)
            col_ir = o3.Irrep(col_l, (-1) ** col_l)
            for ir in row_ir * col_ir:
                key = (ir.l, ir.p)
                path_counts[key] = path_counts.get(key, 0) + 1

    ordered = sorted(path_counts, key=lambda item: (item[0], -item[1]))
    irreps = o3.Irreps(
        [
            (requested if requested > 0 else path_counts[(l, p)], o3.Irrep(l, p))
            for l, p in ordered
        ]
    )
    if requested <= 0 and irreps.dim != ao_dimension * ao_dimension:
        raise RuntimeError(
            "Complete AO decoder dimension mismatch: "
            f"{irreps.dim} != {ao_dimension ** 2}."
        )
    return irreps


def _projector_key(row_l: int, col_l: int, out_l: int) -> str:
    return f"{int(row_l)},{int(col_l)},{int(out_l)}"


def required_projector_keys(full_basis: Sequence[str]) -> Tuple[str, ...]:
    unique_l = sorted({shell_l(shell) for shell in full_basis})
    keys = []
    for row_l in unique_l:
        for col_l in unique_l:
            for out_l in range(abs(row_l - col_l), row_l + col_l + 1):
                keys.append(_projector_key(row_l, col_l, out_l))
    return tuple(keys)


def reference_projector(
    row_l: int,
    col_l: int,
    out_l: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Projector used by DeePTB's E3Hamiltonian.

    Its shape is ``(2*l_row+1, 2*l_col+1, 2*L+1)`` and its normalization is
    ``wigner_3j(l_row, l_col, L) * sqrt(2*L+1)``.
    """
    return o3.wigner_3j(
        int(row_l), int(col_l), int(out_l), dtype=dtype, device=device
    ) * math.sqrt(2 * int(out_l) + 1)


def export_projector_bank(
    path: Union[str, Path],
    full_basis: Sequence[str],
    *,
    source: str = "reference_wigner",
) -> Path:
    """Write a safe JSON projector bank in the Route-B interchange schema.

    A Cartesian/ICT generator should emit the same schema after converting its
    tensors to ``deeptb_real_ao`` ordering and ``e3hamiltonian`` normalization.
    """
    source = str(source).strip().lower()
    if source != "reference_wigner":
        raise ValueError(
            "export_projector_bank writes reference Wigner projectors only; "
            "source must be 'reference_wigner'. Use an external ICT/Cartesian "
            "generator for non-reference provenance."
        )

    path = Path(path)
    projectors = {}
    for key in required_projector_keys(full_basis):
        row_l, col_l, out_l = (int(value) for value in key.split(","))
        projectors[key] = reference_projector(row_l, col_l, out_l).tolist()
    payload = {
        "schema": _PROJECTOR_SCHEMA,
        "source": str(source),
        "normalization": "e3hamiltonian",
        "basis_convention": "deeptb_real_ao",
        "projectors": projectors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validated_projector_payload(path: Union[str, Path]) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != _PROJECTOR_SCHEMA:
        raise ValueError(
            f"Unsupported projector schema {payload.get('schema')!r}; "
            f"expected {_PROJECTOR_SCHEMA!r}."
        )
    if payload.get("normalization") != "e3hamiltonian":
        raise ValueError("Projector bank normalization must be 'e3hamiltonian'.")
    if payload.get("basis_convention") != "deeptb_real_ao":
        raise ValueError("Projector bank basis must be 'deeptb_real_ao'.")
    return payload


def projector_bank_source(path: Union[str, Path]) -> str:
    return str(_validated_projector_payload(path).get("source", ""))


def source_uses_ict(source: Optional[str]) -> bool:
    normalized = str(source or "").strip().lower()
    if not normalized or normalized in _REFERENCE_SOURCES:
        return False
    if normalized.startswith("reference_"):
        return False
    return "ict" in normalized or "cartesian" in normalized


def load_projector_bank(
    path: Union[str, Path],
    full_basis: Sequence[str],
    *,
    atol: float = 2.0e-10,
) -> Dict[str, torch.Tensor]:
    """Load and convention-check a precomputed angular-projector bank.

    The numerical reference check is intentional: a Cartesian source may use a
    different gauge, Cartesian component ordering, parity convention, or
    normalization.  Such a bank is accepted only after conversion into the
    exact latent/AO convention used by this DeePTB checkout.
    """
    payload = _validated_projector_payload(path)
    raw = payload.get("projectors")
    if not isinstance(raw, Mapping):
        raise ValueError("Projector bank must contain a 'projectors' mapping.")

    bank: Dict[str, torch.Tensor] = {}
    for key in required_projector_keys(full_basis):
        if key not in raw:
            raise ValueError(f"Projector bank is missing required key {key!r}.")
        row_l, col_l, out_l = (int(value) for value in key.split(","))
        tensor = torch.tensor(raw[key], dtype=torch.float64)
        expected_shape = (2 * row_l + 1, 2 * col_l + 1, 2 * out_l + 1)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Projector {key} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}."
            )
        reference = reference_projector(row_l, col_l, out_l)
        if not torch.allclose(tensor, reference, atol=float(atol), rtol=0.0):
            error = float((tensor - reference).abs().max().item())
            raise ValueError(
                f"Projector {key} violates DeePTB convention; max error={error:.3e}."
            )
        bank[key] = tensor
    return bank


def normalize_projector_backend(value: Optional[str]) -> str:
    backend = (value or _REFERENCE_BACKEND).strip().lower()
    aliases = {
        "wigner": _REFERENCE_BACKEND,
        "wigner_reference": _REFERENCE_BACKEND,
        "e3nn": _REFERENCE_BACKEND,
        "bank": _PRECOMPUTED_BACKEND,
        "projector_bank": _PRECOMPUTED_BACKEND,
    }
    backend = aliases.get(backend, backend)
    if backend not in {_REFERENCE_BACKEND, _PRECOMPUTED_BACKEND}:
        raise ValueError(
            "ao_projector_backend must be 'reference_wigner' or 'precomputed', "
            f"got {value!r}."
        )
    return backend

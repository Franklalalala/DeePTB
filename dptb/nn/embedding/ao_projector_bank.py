"""Convention-checked angular-projector banks for direct AO decoding.

Reference-Wigner and Cartesian/ICT generators share the same runtime tensor
contract only after they have been converted to DeePTB's real-AO gauge, shell
order, m-order, parity convention, and E3Hamiltonian normalization.  The v2
schema records that provenance explicitly; numerical agreement with the
runtime reference is validation, not evidence of how a bank was generated.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from e3nn import o3


_ANGULAR_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
_PROJECTOR_SCHEMA_V1 = "deeptb.ao_angular_projector/v1"
_PROJECTOR_SCHEMA_V2 = "deeptb.ao_angular_projector/v2"
_PROJECTOR_SCHEMA = _PROJECTOR_SCHEMA_V1  # compatibility for older imports
_REFERENCE_BACKEND = "reference_wigner"
_PRECOMPUTED_BACKEND = "precomputed"
_REFERENCE_SOURCES = {
    "reference_wigner",
    "reference_fixture_replaceable_by_cartesian_ict",
}
_TRUSTED_ICT_GENERATORS = {
    "deeptb.cartesian_stf_3j/v1",
}
_REQUIRED_M_ORDER = "e3nn_real_spherical_harmonics_component"


@dataclass(frozen=True)
class ProjectorBankProvenance:
    """Validated provenance attached to a precomputed projector bank."""

    schema: str
    source: str
    generator_id: Optional[str]
    generator_kind: Optional[str]
    normalization: str
    basis_convention: str
    shell_order: Optional[Tuple[str, ...]]
    m_order: Optional[str]
    validation_atol: float
    validation_max_abs_error: float
    validation_passed: bool
    uses_ict: bool

    @property
    def uses_precomputed_projector(self) -> bool:
        return True


def shell_l(shell: str) -> int:
    labels = re.findall(r"[A-Za-z]", str(shell))
    if len(labels) != 1 or labels[0].lower() not in _ANGULAR_L:
        raise ValueError(f"Unsupported AO shell label {shell!r}.")
    return _ANGULAR_L[labels[0].lower()]


def build_ao_decoder_irreps(
    full_basis: Sequence[str], channels: Optional[int] = 0
) -> o3.Irreps:
    """Return the complete ordered AO-pair representation or a rank ablation."""
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
    """Projector used by DeePTB's E3Hamiltonian."""
    return o3.wigner_3j(
        int(row_l), int(col_l), int(out_l), dtype=dtype, device=device
    ) * math.sqrt(2 * int(out_l) + 1)


def export_projector_bank(
    path: Union[str, Path],
    full_basis: Sequence[str],
    *,
    source: str = "reference_wigner",
) -> Path:
    """Write a v1 reference-Wigner fixture.

    This exporter deliberately cannot produce ICT provenance.  Use
    :func:`cartesian_ict_bank.export_cartesian_ict_projector_bank` for the
    explicit STF/Cartesian-3j generation route.
    """
    source = str(source).strip().lower()
    if source != "reference_wigner":
        raise ValueError(
            "export_projector_bank writes reference Wigner projectors only; "
            "source must be 'reference_wigner'. Use the Cartesian ICT exporter "
            "for non-reference provenance."
        )

    path = Path(path)
    projectors = {}
    for key in required_projector_keys(full_basis):
        row_l, col_l, out_l = (int(value) for value in key.split(","))
        projectors[key] = reference_projector(row_l, col_l, out_l).tolist()
    payload = {
        "schema": _PROJECTOR_SCHEMA_V1,
        "source": source,
        "normalization": "e3hamiltonian",
        "basis_convention": "deeptb_real_ao",
        "projectors": projectors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _mapping_field(
    payload: Mapping[str, object], name: str, *, required: bool = True
) -> Mapping[str, object]:
    value = payload.get(name)
    if isinstance(value, Mapping):
        return value
    if required:
        raise ValueError(f"Projector bank must contain a {name!r} mapping.")
    return {}


def _validated_projector_payload(
    path: Union[str, Path], full_basis: Optional[Sequence[str]] = None
) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {_PROJECTOR_SCHEMA_V1, _PROJECTOR_SCHEMA_V2}:
        raise ValueError(
            f"Unsupported projector schema {schema!r}; expected "
            f"{_PROJECTOR_SCHEMA_V1!r} or {_PROJECTOR_SCHEMA_V2!r}."
        )
    if payload.get("normalization") != "e3hamiltonian":
        raise ValueError("Projector bank normalization must be 'e3hamiltonian'.")
    if payload.get("basis_convention") != "deeptb_real_ao":
        raise ValueError("Projector bank basis must be 'deeptb_real_ao'.")

    if schema == _PROJECTOR_SCHEMA_V2:
        shell_order = payload.get("shell_order")
        if not isinstance(shell_order, list) or not all(
            isinstance(shell, str) for shell in shell_order
        ):
            raise ValueError("v2 projector bank requires string-list 'shell_order'.")
        if full_basis is not None and tuple(shell_order) != tuple(str(x) for x in full_basis):
            raise ValueError(
                "Projector bank shell_order does not match OrbitalMapper full_basis: "
                f"{tuple(shell_order)!r} != {tuple(str(x) for x in full_basis)!r}."
            )
        if payload.get("m_order") != _REQUIRED_M_ORDER:
            raise ValueError(
                f"Projector bank m_order must be {_REQUIRED_M_ORDER!r}."
            )
        _mapping_field(payload, "generator")
        _mapping_field(payload, "validation")
    return payload


def projector_bank_provenance(
    path: Union[str, Path], full_basis: Optional[Sequence[str]] = None
) -> ProjectorBankProvenance:
    """Return structured provenance after schema-level validation.

    ``uses_ict`` is intentionally strict: a source string alone is never
    sufficient.  The bank must use the v2 schema, name a trusted ICT generator,
    identify itself as an irreducible-Cartesian generator, and carry a passing
    convention-validation record within its declared tolerance.
    """
    payload = _validated_projector_payload(path, full_basis)
    schema = str(payload["schema"])
    source = str(payload.get("source", "")).strip().lower()
    generator = _mapping_field(payload, "generator", required=False)
    validation = _mapping_field(payload, "validation", required=False)
    generator_id = generator.get("id")
    generator_kind = generator.get("kind")
    validation_atol = float(validation.get("atol", 0.0) or 0.0)
    validation_error = float(validation.get("max_abs_error", math.inf))
    validation_passed = bool(validation.get("passed", False))

    uses_ict = bool(
        schema == _PROJECTOR_SCHEMA_V2
        and source == "cartesian_ict"
        and generator_id in _TRUSTED_ICT_GENERATORS
        and generator_kind == "irreducible_cartesian_tensor"
        and validation_atol > 0.0
        and validation_passed
        and math.isfinite(validation_error)
        and validation_error <= validation_atol
    )
    shell_order_raw = payload.get("shell_order")
    shell_order = (
        tuple(str(x) for x in shell_order_raw)
        if isinstance(shell_order_raw, list)
        else None
    )
    return ProjectorBankProvenance(
        schema=schema,
        source=source,
        generator_id=None if generator_id is None else str(generator_id),
        generator_kind=None if generator_kind is None else str(generator_kind),
        normalization=str(payload["normalization"]),
        basis_convention=str(payload["basis_convention"]),
        shell_order=shell_order,
        m_order=None if payload.get("m_order") is None else str(payload["m_order"]),
        validation_atol=validation_atol,
        validation_max_abs_error=validation_error,
        validation_passed=validation_passed,
        uses_ict=uses_ict,
    )


def projector_bank_source(path: Union[str, Path]) -> str:
    return projector_bank_provenance(path).source


def source_uses_ict(source: Optional[str]) -> bool:
    """Compatibility helper; runtime code must use structured provenance.

    A bare source label cannot prove an ICT generation path, so this helper is
    conservative and returns ``False``.  It remains only to avoid import breaks
    in downstream code written against the v1 API.
    """
    del source
    return False


def load_projector_bank_with_provenance(
    path: Union[str, Path],
    full_basis: Sequence[str],
    *,
    atol: float = 2.0e-10,
) -> Tuple[Dict[str, torch.Tensor], ProjectorBankProvenance]:
    payload = _validated_projector_payload(path, full_basis)
    provenance = projector_bank_provenance(path, full_basis)
    raw = payload.get("projectors")
    if not isinstance(raw, Mapping):
        raise ValueError("Projector bank must contain a 'projectors' mapping.")

    declared_atol = provenance.validation_atol
    if payload.get("schema") == _PROJECTOR_SCHEMA_V2:
        if declared_atol <= 0.0 or declared_atol > float(atol):
            raise ValueError(
                "ICT projector bank validation tolerance must be positive and no "
                f"looser than loader atol={float(atol):.3e}; got {declared_atol:.3e}."
            )

    bank: Dict[str, torch.Tensor] = {}
    max_error = 0.0
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
        error = float((tensor - reference).abs().max().item())
        max_error = max(max_error, error)
        if not torch.allclose(tensor, reference, atol=float(atol), rtol=0.0):
            raise ValueError(
                f"Projector {key} violates DeePTB convention; max error={error:.3e}."
            )
        bank[key] = tensor

    if payload.get("schema") == _PROJECTOR_SCHEMA_V2:
        recorded = provenance.validation_max_abs_error
        record_slack = max(1.0e-15, float(atol) * 1.0e-4)
        if abs(recorded - max_error) > record_slack:
            raise ValueError(
                "Projector bank validation.max_abs_error does not match the "
                f"recomputed value: {recorded:.3e} != {max_error:.3e}."
            )
        if not provenance.validation_passed or max_error > declared_atol:
            raise ValueError(
                "Projector bank carries a failed or inconsistent validation record."
            )
    return bank, provenance


def load_projector_bank(
    path: Union[str, Path],
    full_basis: Sequence[str],
    *,
    atol: float = 2.0e-10,
) -> Dict[str, torch.Tensor]:
    """Load a bank after strict gauge/order/normalization validation."""
    bank, _ = load_projector_bank_with_provenance(path, full_basis, atol=atol)
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

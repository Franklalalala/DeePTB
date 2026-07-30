"""Element parameters for finite-cluster QEq geometry kernels.

The default electronegativities and on-site hardnesses are the constants in
Table S1 of Hu et al., Nature Communications 16, 7379 (2025),
https://doi.org/10.1038/s41467-025-62824-5.  Gaussian widths are the covalent
radii used as ``sigma`` by the authors' accompanying DP-QEq implementation.

The original Hu method contains Gaussian charges plus PME and a dipole
correction.  Using this table with the bare finite-cluster kernel is **not** a
reproduction of that method.  The table is provided as an auditable source of
element constants for the cluster-only reference kernels in
``dptb.nnops.qeq_kernels``.

Units are eV/e for electronegativity, eV/e^2 for hardness, and Angstrom for
Gaussian width.  User mappings passed to :func:`coerce_qeq_parameter_table`
override or extend this default table; an element absent after that resolution
is rejected rather than guessed.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Optional, Tuple

import numpy as np

from dptb.nnops.qeq_operator import QEqOperatorError


class QEqParameterError(QEqOperatorError):
    """Raised when a QEq element parameter table is incomplete or invalid."""


def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QEqParameterError(f"{name} must be a scalar float") from exc
    if not np.isfinite(result):
        raise QEqParameterError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise QEqParameterError(f"{name} must be positive")
    return result


def _canonical_symbol(symbol: Any) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        raise QEqParameterError("element symbols must be non-empty strings")
    stripped = symbol.strip()
    canonical = stripped[0].upper() + stripped[1:].lower()
    if not canonical.isalpha() or len(canonical) > 3:
        raise QEqParameterError(f"invalid element symbol {symbol!r}")
    return canonical


@dataclass(frozen=True)
class QEqElementParameters:
    """Element-level QEq constants in the canonical DeePTB QEq units."""

    electronegativity: float
    hardness: float
    gaussian_width: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "electronegativity",
            _finite_float("electronegativity", self.electronegativity),
        )
        object.__setattr__(
            self,
            "hardness",
            _finite_float("hardness", self.hardness, positive=True),
        )
        object.__setattr__(
            self,
            "gaussian_width",
            _finite_float("gaussian_width", self.gaussian_width, positive=True),
        )


@dataclass(frozen=True)
class QEqParameterProvenance:
    """Source, units, and applicability statement for a parameter table."""

    source: str
    doi: str
    electronegativity_unit: str
    hardness_unit: str
    gaussian_width_unit: str
    applicability_note: str

    def __post_init__(self) -> None:
        for field_name in (
            "source",
            "doi",
            "electronegativity_unit",
            "hardness_unit",
            "gaussian_width_unit",
            "applicability_note",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise QEqParameterError(
                    f"parameter provenance {field_name} must be a non-empty string"
                )


def _coerce_element_parameters(
    symbol: str, value: Any
) -> QEqElementParameters:
    if isinstance(value, QEqElementParameters):
        return value
    if isinstance(value, Mapping):
        required = {"electronegativity", "hardness", "gaussian_width"}
        missing = required.difference(value)
        extra = set(value).difference(required)
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing {sorted(missing)}")
            if extra:
                detail.append(f"unexpected {sorted(extra)}")
            raise QEqParameterError(
                f"parameters for {symbol} must have exactly "
                "electronegativity, hardness, gaussian_width ("
                + "; ".join(detail)
                + ")"
            )
        return QEqElementParameters(
            electronegativity=value["electronegativity"],
            hardness=value["hardness"],
            gaussian_width=value["gaussian_width"],
        )
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return QEqElementParameters(*value)
    raise QEqParameterError(
        f"parameters for {symbol} must be QEqElementParameters, a three-field "
        "mapping, or a (electronegativity, hardness, gaussian_width) triple"
    )


@dataclass(frozen=True)
class QEqParameterTable(Mapping[str, QEqElementParameters]):
    """Immutable, hashable-by-content collection of element parameters.

    ``entries`` accepts a mapping or an iterable of ``(symbol, parameters)``
    pairs.  It is normalized to a sorted tuple so the table remains pickleable
    and its SHA-256 digest is independent of insertion order.
    """

    entries: Any
    provenance: QEqParameterProvenance
    name: str = "qeq-parameter-table"

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, QEqParameterProvenance):
            raise QEqParameterError(
                "provenance must be a QEqParameterProvenance instance"
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise QEqParameterError("parameter table name must be a non-empty string")

        raw_items = (
            self.entries.items()
            if isinstance(self.entries, Mapping)
            else self.entries
        )
        try:
            items = list(raw_items)
        except TypeError as exc:
            raise QEqParameterError(
                "parameter table entries must be a mapping or iterable of pairs"
            ) from exc
        if not items:
            raise QEqParameterError("parameter table must contain at least one element")

        normalized = {}
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise QEqParameterError(
                    "parameter table entries must be (symbol, parameters) pairs"
                )
            symbol = _canonical_symbol(item[0])
            if symbol in normalized:
                raise QEqParameterError(f"duplicate parameters for element {symbol}")
            normalized[symbol] = _coerce_element_parameters(symbol, item[1])
        object.__setattr__(self, "entries", tuple(sorted(normalized.items())))

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[str]:
        return (symbol for symbol, _ in self.entries)

    def __getitem__(self, symbol: str) -> QEqElementParameters:
        canonical = _canonical_symbol(symbol)
        for candidate, value in self.entries:
            if candidate == canonical:
                return value
        raise QEqParameterError(
            f"element {canonical!r} is not present in parameter table {self.name!r}"
        )

    @property
    def sha256(self) -> str:
        """Deterministic digest of numeric entries and provenance."""

        payload = {
            "name": self.name,
            "entries": {
                symbol: {
                    "electronegativity": value.electronegativity,
                    "hardness": value.hardness,
                    "gaussian_width": value.gaussian_width,
                }
                for symbol, value in self.entries
            },
            "provenance": {
                field_name: getattr(self.provenance, field_name)
                for field_name in (
                    "source",
                    "doi",
                    "electronegativity_unit",
                    "hardness_unit",
                    "gaussian_width_unit",
                    "applicability_note",
                )
            },
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def with_overrides(
        self,
        updates: Mapping[str, Any],
        *,
        provenance: Optional[QEqParameterProvenance] = None,
        name: Optional[str] = None,
    ) -> "QEqParameterTable":
        """Return a table with caller values replacing or extending this table."""

        if not isinstance(updates, Mapping):
            raise QEqParameterError("parameter overrides must be a mapping")
        merged = dict(self.items())
        for symbol, value in updates.items():
            canonical = _canonical_symbol(symbol)
            merged[canonical] = _coerce_element_parameters(canonical, value)
        if provenance is None:
            provenance = QEqParameterProvenance(
                source=self.provenance.source + "; caller-supplied overrides/extensions",
                doi=self.provenance.doi,
                electronegativity_unit=self.provenance.electronegativity_unit,
                hardness_unit=self.provenance.hardness_unit,
                gaussian_width_unit=self.provenance.gaussian_width_unit,
                applicability_note=(
                    self.provenance.applicability_note
                    + " Numeric origin of overridden or added entries is supplied "
                    "by the caller."
                ),
            )
        return QEqParameterTable(
            merged,
            provenance=provenance,
            name=name or (self.name + "+user"),
        )


HU_2025_PROVENANCE = QEqParameterProvenance(
    source=(
        "Hu et al., Nature Communications 16, 7379 (2025), Table S1; "
        "Gaussian widths from the accompanying sxu39/DP-QEq code's "
        "R_Covalence table"
    ),
    doi="10.1038/s41467-025-62824-5",
    electronegativity_unit="eV/e",
    hardness_unit="eV/e^2",
    gaussian_width_unit="Angstrom",
    applicability_note=(
        "The original method uses Gaussian charges with periodic PME and a "
        "dipole correction. This finite-cluster table, and especially its use "
        "with the bare cluster kernel, is not a reproduction of that method."
    ),
)


HU_2025_QEQ_PARAMETERS = QEqParameterTable(
    {
        "Li": QEqElementParameters(-3.0000, 10.0241, 1.28),
        "C": QEqElementParameters(5.8678, 7.0000, 0.76),
        "H": QEqElementParameters(5.3200, 7.4366, 0.31),
        "O": QEqElementParameters(8.5000, 8.9989, 0.66),
        "P": QEqElementParameters(1.8000, 7.0946, 1.07),
        "F": QEqElementParameters(9.0000, 8.0000, 0.57),
    },
    provenance=HU_2025_PROVENANCE,
    name="Hu-2025-Table-S1",
)

# Canonical default used by the geometry convenience API.
DEFAULT_QEQ_PARAMETERS = HU_2025_QEQ_PARAMETERS


def coerce_qeq_parameter_table(
    params: Optional[Any],
) -> QEqParameterTable:
    """Resolve ``None``, a full table, or overrides on the Hu default table."""

    if params is None:
        return DEFAULT_QEQ_PARAMETERS
    if isinstance(params, QEqParameterTable):
        return params
    if isinstance(params, Mapping):
        return DEFAULT_QEQ_PARAMETERS.with_overrides(params)
    raise QEqParameterError(
        "params must be None, a QEqParameterTable, or an element override mapping"
    )


__all__ = [
    "DEFAULT_QEQ_PARAMETERS",
    "HU_2025_PROVENANCE",
    "HU_2025_QEQ_PARAMETERS",
    "QEqElementParameters",
    "QEqParameterError",
    "QEqParameterProvenance",
    "QEqParameterTable",
    "coerce_qeq_parameter_table",
]

"""Recipe and quadrature-order policy of the full NACF candidate prior: declarations only, no numerics.

This module owns the canonical vocabulary of the implemented conventions (XC functional, density definition,
cS zero point), the versioned :class:`CandidateRecipe`, the immutable :class:`FusionSettings` and the
geometry-only quadrature :class:`OrderPolicy` family. :mod:`dptb.nacf.candidate` binds these declarations to
injected providers and runs the numerical prepare/forward; :mod:`dptb.nacf.candidate_checks` holds the shared
input/provider checks. Nothing here imports Torch at module level.

Recipes and the built-in policies are frozen dataclasses, so a bound recipe cannot drift away from the identity
the plan hashed at construction. Custom :class:`OrderPolicy` subclasses are not frozen by construction; the plan
snapshots ``identity()`` when it binds a recipe and refuses to prepare a structure once that snapshot differs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:                     # pragma: no cover - annotations only
    from .onsite import OnsiteXCEvaluator

RECIPE_SCHEMA = "nacf-candidate-prior/v1"

# Canonical XC declaration of the implemented recipe: one machine key and one human-readable label. The label is
# the recipe identity field of the 0920 receipts and stays unchanged; the key is what providers declare.
XC_KEY = "lda_pz81_unpolarized"
XC_FUNCTIONAL = "LDA exchange + PZ81 correlation, unpolarized"
DENSITY_DEFINITION = "normalized neutral valence (r=0 repair) + unscaled NLCC"
ZERO_POINT = "cS: c = 4 pi sum(M2_valence) Ry_to_eV / (3 Omega), fully periodic"

# Free-text implementation labels (``onsite_identity['potential']``) are provenance, not physics. They are only
# read for two things: a token that recognizably names the implemented functional, and recognizable
# contradictions (another functional, or no potential at all). Anything else needs the explicit key.
XC_LABEL_TOKENS = frozenset({"pz81"})
XC_CONTRADICTION_TOKENS = frozenset({
    "pbe", "pbesol", "revpbe", "rpbe", "pbe0", "pw86", "pw91", "pw92", "b88", "lyp", "blyp", "b3lyp", "hse", "hse06",
    "scan", "rscan", "r2scan", "tpss", "m06", "m06l", "wb97x", "am05", "vwn", "vwn5", "gga", "mgga", "metagga",
    "hybrid", "exx", "hf", "hartree", "fock", "polarized", "lsda", "spin",
})
_NO_POTENTIAL = re.compile(r"\b(?:zero|null|no|none|without|disabled?)[\s_-]*(?:xc[\s_-]*)?(?:potential|vxc|v_xc|functional)\b"
                           r"|^\s*(?:none|null|zero)\s*$")
_TOKEN = re.compile(r"[a-z0-9]+")


def xc_label_tokens(label: Any) -> set[str]:
    """Lower-case alphanumeric tokens of a free-text implementation label."""
    return set(_TOKEN.findall(str(label).lower()))


def label_denies_potential(label: Any) -> bool:
    """True when a label recognizably declares no XC potential at all (``'zero potential'``, ``'none'``)."""
    return _NO_POTENTIAL.search(str(label).lower()) is not None


class CandidateIdentityError(ValueError):
    """Missing, unsupported or contradictory recipe/table identity."""


# --------------------------------------------------------------------------- order policies
def order_check(atom: int, symbol: str, order, *, checked: bool, converged=None, **measured) -> dict[str, Any]:
    """One per-atom order diagnostic. ``convergence_checked`` says whether any comparison was run; ``converged`` is
    the measured verdict (``None`` when nothing was measured). A fixed order is a choice, not a convergence proof."""
    row = {"atom": int(atom), "symbol": str(symbol), "selected_order": tuple(int(x) for x in order),
           "convergence_checked": bool(checked), "converged": None if converged is None else bool(converged)}
    row.update(measured)
    return row


def _three_positive(order, name):
    order = tuple(int(x) for x in order)
    if len(order) != 3 or min(order) <= 0:
        raise ValueError(f"{name} must be three positive integers (radial, mu, phi)")
    return order


class OrderPolicy:
    """Geometry-only per-atom quadrature order selection. ``select`` returns ``(orders, checks)``.

    ``identity()`` must describe every setting that changes ``select``; the plan hashes it and re-checks it
    before each prepare. Subclasses should be immutable (frozen dataclasses like the built-in policies).
    """

    def identity(self) -> dict[str, Any]:
        raise NotImplementedError

    def select(self, evaluator: "OnsiteXCEvaluator", g: Mapping[str, Any], width: int):
        raise NotImplementedError


@dataclass(frozen=True)
class FixedOrderPolicy(OrderPolicy):
    """One quadrature order for every atom. Its checks record the choice and ``convergence_checked=False``."""
    order: tuple = (128, 24, 48)

    def __post_init__(self):
        object.__setattr__(self, "order", _three_positive(self.order, "order"))

    def identity(self):
        return {"policy": "fixed", "order": list(self.order)}

    def select(self, evaluator, g, width):
        symbols = list(g["symbols"])
        return [self.order] * len(symbols), [order_check(i, s, self.order, checked=False) for i, s in enumerate(symbols)]


@dataclass(frozen=True)
class ConvergenceOrderPolicy(OrderPolicy):
    """The accepted fixed100 rule re-run on the structure at hand (no history lookup).

    Every atom is evaluated at ``medium`` and ``fine``; where max|medium - fine| >= ``tolerance_eV`` the atom uses
    ``fine`` and is re-checked against ``finer``. Optionally the medium block is compared with the block of the
    ``tail_radius_bohr`` neighbourhood (accepted 23 vs 27 Bohr check). ``strict=True`` raises when an atom did not
    converge; otherwise the verdict is only recorded (``converged=False``), which is not a release acceptance.
    Costs two (or three) onsite evaluations per structure.
    """
    medium: tuple = (128, 24, 48)
    fine: tuple = (256, 40, 80)
    finer: tuple = (384, 56, 112)
    tolerance_eV: float = 1e-3
    tail_radius_bohr: float | None = None
    strict: bool = False

    def __post_init__(self):
        for name in ("medium", "fine", "finer"):
            object.__setattr__(self, name, _three_positive(getattr(self, name), name))
        object.__setattr__(self, "tolerance_eV", float(self.tolerance_eV))
        object.__setattr__(self, "tail_radius_bohr", None if self.tail_radius_bohr is None else float(self.tail_radius_bohr))
        object.__setattr__(self, "strict", bool(self.strict))
        if not (math.isfinite(self.tolerance_eV) and self.tolerance_eV > 0):
            raise ValueError("tolerance_eV must be positive")
        if self.tail_radius_bohr is not None and not self.tail_radius_bohr > 0:
            raise ValueError("tail_radius_bohr must be positive")

    def identity(self):
        return {"policy": "convergence", "medium": list(self.medium), "fine": list(self.fine), "finer": list(self.finer),
                "tolerance_eV": self.tolerance_eV, "tail_radius_bohr": self.tail_radius_bohr, "strict": self.strict}

    def select(self, evaluator, g, width):
        from .onsite import onsite_neighbor_lists
        symbols = list(g["symbols"])
        n = len(symbols)
        neighbors = evaluator.neighbors(g)
        medium = evaluator(g, width, [self.medium] * n, neighbors=neighbors)
        fine = evaluator(g, width, [self.fine] * n, neighbors=neighbors)
        error = (medium - fine).abs().flatten(1).amax(1).tolist()
        tail = None
        if self.tail_radius_bohr is not None:
            inner = onsite_neighbor_lists(g, self.tail_radius_bohr)
            tail = (medium - evaluator(g, width, [self.medium] * n, neighbors=inner)).abs().flatten(1).amax(1).tolist()
        orders, checks = [], []
        for i, s in enumerate(symbols):
            measured = {"tolerance_eV": self.tolerance_eV, "medium_fine_eV": error[i], "fine_finer_eV": None}
            if error[i] < self.tolerance_eV:
                order, converged = self.medium, True
            else:
                finer = evaluator.blocks(evaluator.qgrid(s, self.finer), [neighbors[i]])[0]
                m = finer.shape[0]
                measured["fine_finer_eV"] = float((fine[i, :m, :m] - finer).abs().max())
                order, converged = self.fine, measured["fine_finer_eV"] < self.tolerance_eV
            if tail is not None:
                measured["environment_tail_eV"] = tail[i]
                converged = converged and tail[i] < self.tolerance_eV
            orders.append(order)
            checks.append(order_check(i, s, order, checked=True, converged=converged, **measured))
        if self.strict and not all(c["converged"] for c in checks):
            raise RuntimeError("onsite quadrature did not converge for atoms " + str([c["atom"] for c in checks if not c["converged"]]))
        return orders, checks


# --------------------------------------------------------------------------- recipe
@dataclass(frozen=True)
class FusionSettings:
    """Explicit CUDA fusion switches of a recipe. ``contraction`` is never accepted for the candidate prior."""
    radial: bool = False
    contraction: bool = False

    @classmethod
    def coerce(cls, value) -> "FusionSettings":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise CandidateIdentityError("fusion must be a FusionSettings or a mapping with 'radial'/'contraction'")
        unknown = set(value) - {"radial", "contraction"}
        if unknown:
            raise CandidateIdentityError(f"unknown fusion settings {sorted(unknown)}")
        return cls(radial=bool(value.get("radial", False)), contraction=bool(value.get("contraction", False)))

    def as_dict(self) -> dict[str, bool]:
        return {"radial": self.radial, "contraction": self.contraction}


@dataclass(frozen=True)
class CandidateRecipe:
    """Versioned recipe identity; ``envxc_arm`` and ``stabilization`` must be stated explicitly.

    ``xc_functional`` accepts the canonical key :data:`XC_KEY` or the label :data:`XC_FUNCTIONAL`; the identity
    always records the label, so recipes of the 0920 receipts keep their hash. ``fusion`` accepts a mapping and
    is stored as an immutable :class:`FusionSettings`.
    """
    envxc_arm: str
    stabilization: str
    order_policy: OrderPolicy
    onsite_radius_bohr: float = 27.0
    xc_functional: str = XC_FUNCTIONAL
    density_definition: str = DENSITY_DEFINITION
    zero_point: str = ZERO_POINT
    overlap_floor: float = 1e-8
    moment_density_floor: float = 0.0
    fusion: FusionSettings = FusionSettings()

    def __post_init__(self):
        from .envxc import ARMS, STABILIZATIONS
        if self.xc_functional == XC_KEY:
            object.__setattr__(self, "xc_functional", XC_FUNCTIONAL)
        object.__setattr__(self, "fusion", FusionSettings.coerce(self.fusion))
        for name, expected in (("xc_functional", XC_FUNCTIONAL), ("density_definition", DENSITY_DEFINITION), ("zero_point", ZERO_POINT)):
            if getattr(self, name) != expected:
                raise CandidateIdentityError(f"unsupported {name}: {getattr(self, name)!r}; implemented convention is {expected!r}")
        if self.envxc_arm not in ARMS:
            raise CandidateIdentityError(f"unsupported environment arm {self.envxc_arm!r}; expected one of {ARMS}")
        if self.stabilization not in STABILIZATIONS:
            raise CandidateIdentityError(f"unsupported stabilization {self.stabilization!r}; expected one of {STABILIZATIONS}")
        if not isinstance(self.order_policy, OrderPolicy):
            raise CandidateIdentityError("order_policy must be an OrderPolicy instance")
        if not (math.isfinite(self.onsite_radius_bohr) and self.onsite_radius_bohr > 0):
            raise CandidateIdentityError("onsite_radius_bohr must be positive")
        if not (math.isfinite(self.overlap_floor) and self.overlap_floor > 0
                and math.isfinite(self.moment_density_floor) and self.moment_density_floor >= 0):
            raise CandidateIdentityError("overlap floor must be finite positive and moment density floor finite nonnegative")
        if self.fusion.contraction:
            # rejected under the strict FP32 packing gate (hopping fusion report); never a silent default
            raise CandidateIdentityError("fused contraction is not accepted for the candidate prior; evaluate it explicitly outside the recipe")

    @property
    def xc_key(self) -> str:
        """Canonical key of the implemented functional (the only one this recipe can declare)."""
        return XC_KEY

    def identity(self):
        return {"schema": RECIPE_SCHEMA, "envxc_arm": self.envxc_arm, "stabilization": self.stabilization,
                "order_policy": self.order_policy.identity(), "onsite_radius_bohr": self.onsite_radius_bohr,
                "xc_functional": self.xc_functional, "density_definition": self.density_definition, "zero_point": self.zero_point,
                "overlap_floor": self.overlap_floor, "moment_density_floor": self.moment_density_floor, "fusion": self.fusion.as_dict()}


__all__ = ["RECIPE_SCHEMA", "XC_KEY", "XC_FUNCTIONAL", "DENSITY_DEFINITION", "ZERO_POINT", "XC_LABEL_TOKENS", "XC_CONTRADICTION_TOKENS",
           "CandidateIdentityError", "OrderPolicy", "FixedOrderPolicy", "ConvergenceOrderPolicy", "order_check",
           "FusionSettings", "CandidateRecipe", "xc_label_tokens", "label_denies_potential"]

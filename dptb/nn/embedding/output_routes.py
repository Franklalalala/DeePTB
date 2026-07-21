"""Single source of truth for DeePTB output-head routes.

The registry separates the representation produced by the final LEM layer from
the downstream output contract.  Canonical experiment names are H-A0/H-A1,
H-B0/H-B1, P-B0, and P-B1-ICT.  Historical ``rme_head_mode`` values remain
accepted as deprecated aliases so existing checkpoints/configs keep loading.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from e3nn import o3

from .ao_projector_bank import (
    _PRECOMPUTED_BACKEND,
    _REFERENCE_BACKEND,
    normalize_projector_backend,
    projector_bank_provenance,
)


@dataclass(frozen=True)
class OutputRouteSpec:
    canonical_name: str
    legacy_mode: str
    final_irreps_kind: str
    output_contract: str
    head_factory: str
    head_class_name: str
    prediction_method: str
    block_decoder: Optional[str]
    uses_e3hamiltonian: bool
    uses_ict: bool
    uses_precomputed_projector: bool
    projector_source: Optional[str] = None
    product_scope: Optional[str] = None
    onehot_after_head: bool = False
    debug_only: bool = False
    description: str = ""

    @property
    def is_block_native(self) -> bool:
        return self.output_contract == "ao_block"

    @property
    def is_official_equivariant_route(self) -> bool:
        return self.canonical_name in OFFICIAL_OUTPUT_ROUTES

    def metadata(self) -> Dict[str, object]:
        return {
            "canonical_name": self.canonical_name,
            "legacy_mode": self.legacy_mode,
            "final_irreps_kind": self.final_irreps_kind,
            "output_contract": self.output_contract,
            "head_class_name": self.head_class_name,
            "prediction_method": self.prediction_method,
            "block_decoder": self.block_decoder,
            "uses_e3hamiltonian": self.uses_e3hamiltonian,
            "uses_ict": self.uses_ict,
            "uses_precomputed_projector": self.uses_precomputed_projector,
            "projector_source": self.projector_source,
            "product_scope": self.product_scope,
            "debug_only": self.debug_only,
        }


@dataclass(frozen=True)
class OutputHeadContext:
    final_irreps: o3.Irreps
    orbpair_irreps: o3.Irreps
    full_basis: Tuple[str, ...]
    max_norb: int
    rank: int
    init: float
    condition: str
    product_scope: str
    ao_projector_normalization: str
    ao_projector_basis_convention: str
    ao_projector_backend: str
    ao_projector_bank_path: Optional[str]
    dtype: torch.dtype
    device: Union[str, torch.device]
    # P1-3: reduction-path opt-in for the h_b0 late-CG head
    # (``late_block_expansion_cg.LateBlockExpansionCGHead``).  Defaulted so
    # existing direct ``OutputHeadContext(...)`` construction (other routes,
    # tests) keeps working unchanged; only ``_factory_h_b0`` reads it.
    cg_head_impl: str = "legacy"


_ROUTE_SPECS: Dict[str, OutputRouteSpec] = {
    "legacy_rme": OutputRouteSpec(
        canonical_name="legacy_rme",
        legacy_mode="legacy_linear",
        final_irreps_kind="orbpair",
        output_contract="rme",
        head_factory="legacy_linear",
        head_class_name="Linear",
        prediction_method="e3tb",
        block_decoder=None,
        uses_e3hamiltonian=True,
        uses_ict=False,
        uses_precomputed_projector=False,
        description="Legacy orbpair-irrep output followed by E3Hamiltonian.",
    ),
    "rme_fusion": OutputRouteSpec(
        canonical_name="rme_fusion",
        legacy_mode="rme_nocg_fusion",
        final_irreps_kind="orbpair",
        output_contract="rme",
        head_factory="rme_nocg_fusion",
        head_class_name="RMENoCGFusionHead",
        prediction_method="e3tb",
        block_decoder=None,
        uses_e3hamiltonian=True,
        uses_ict=False,
        uses_precomputed_projector=False,
        description="Legacy RME projection with invariant-conditioned fusion.",
    ),
    "debug_block_linear": OutputRouteSpec(
        canonical_name="debug_block_linear",
        legacy_mode="block_native_linear",
        final_irreps_kind="ordinary_hidden",
        output_contract="ao_block",
        head_factory="block_native_linear",
        head_class_name="BlockNativeLinearHead",
        prediction_method="block_native",
        block_decoder="linear",
        uses_e3hamiltonian=False,
        uses_ict=False,
        uses_precomputed_projector=False,
        debug_only=True,
        description="Non-equivariant dense debug/control baseline.",
    ),
    "h_a0": OutputRouteSpec(
        canonical_name="h_a0",
        legacy_mode="late_rme_expansion_nocg",
        final_irreps_kind="ordinary_hidden",
        output_contract="rme",
        head_factory="late_rme_expansion_nocg",
        head_class_name="LateRMEExpansionNoCGHead",
        prediction_method="e3tb",
        block_decoder=None,
        uses_e3hamiltonian=True,
        uses_ict=False,
        uses_precomputed_projector=False,
        onehot_after_head=True,
        description="Ordinary hidden -> same-irrep RME -> E3Hamiltonian.",
    ),
    "h_a1": OutputRouteSpec(
        canonical_name="h_a1",
        legacy_mode="late_rme_cartesian_hybrid",
        final_irreps_kind="ordinary_hidden",
        output_contract="rme",
        head_factory="late_rme_cartesian_hybrid",
        head_class_name="LateRMECartesianHybridHead",
        prediction_method="e3tb",
        block_decoder=None,
        uses_e3hamiltonian=True,
        uses_ict=True,
        uses_precomputed_projector=False,
        product_scope="all",
        onehot_after_head=True,
        description="Ordinary hidden -> Cartesian/ICT hybrid RME -> E3Hamiltonian.",
    ),
    "h_b0": OutputRouteSpec(
        canonical_name="h_b0",
        legacy_mode="late_block_expansion_cg",
        final_irreps_kind="ordinary_hidden",
        output_contract="ao_block",
        head_factory="late_block_expansion_cg",
        head_class_name="LateBlockExpansionCGHead",
        prediction_method="block_native",
        block_decoder="expansion_cg",
        uses_e3hamiltonian=False,
        uses_ict=False,
        uses_precomputed_projector=False,
        description="Ordinary hidden -> fixed Wigner shell expansion -> AO block.",
    ),
    "h_b1": OutputRouteSpec(
        canonical_name="h_b1",
        legacy_mode="late_block_cartesian_projector",
        final_irreps_kind="ordinary_hidden",
        output_contract="ao_block",
        head_factory="late_block_cartesian_projector",
        head_class_name="LateBlockCartesianProjectorHead",
        prediction_method="block_native",
        block_decoder="cartesian_projector",
        uses_e3hamiltonian=False,
        uses_ict=True,
        uses_precomputed_projector=False,
        product_scope="missing_only",
        description="Ordinary hidden -> fixed Cartesian shell projector -> AO block.",
    ),
    "p_b0": OutputRouteSpec(
        canonical_name="p_b0",
        legacy_mode="direct_ao_projector",
        final_irreps_kind="ao_pair",
        output_contract="ao_block",
        head_factory="ao_projector",
        head_class_name="AOAngularProjectorHead",
        prediction_method="block_native",
        block_decoder="ao_projector",
        uses_e3hamiltonian=False,
        uses_ict=False,
        uses_precomputed_projector=False,
        projector_source="reference_wigner",
        description="AO-pair final irreps -> runtime reference-Wigner projector.",
    ),
    "p_b1_reference": OutputRouteSpec(
        canonical_name="p_b1_reference",
        legacy_mode="direct_ao_projector",
        final_irreps_kind="ao_pair",
        output_contract="ao_block",
        head_factory="ao_projector",
        head_class_name="AOAngularProjectorHead",
        prediction_method="block_native",
        block_decoder="ao_projector",
        uses_e3hamiltonian=False,
        uses_ict=False,
        uses_precomputed_projector=True,
        projector_source="reference_wigner",
        description="AO-pair final irreps -> precomputed reference control bank.",
    ),
    "p_b1_ict": OutputRouteSpec(
        canonical_name="p_b1_ict",
        legacy_mode="direct_ao_projector",
        final_irreps_kind="ao_pair",
        output_contract="ao_block",
        head_factory="ao_projector",
        head_class_name="AOAngularProjectorHead",
        prediction_method="block_native",
        block_decoder="ao_projector",
        uses_e3hamiltonian=False,
        uses_ict=True,
        uses_precomputed_projector=True,
        projector_source="cartesian_ict",
        description="AO-pair final irreps -> validated Cartesian/ICT bank.",
    ),
}

OFFICIAL_OUTPUT_ROUTES: Tuple[str, ...] = (
    "h_a0",
    "h_a1",
    "h_b0",
    "h_b1",
    "p_b0",
    "p_b1_ict",
)

_STATIC_ALIASES: Dict[str, str] = {
    "legacy_linear": "legacy_rme",
    "legacy": "legacy_rme",
    "linear": "legacy_rme",
    "rme_nocg_fusion": "rme_fusion",
    "rme_fusion": "rme_fusion",
    "nocg": "rme_fusion",
    "block_native_linear": "debug_block_linear",
    "block_native": "debug_block_linear",
    "block_linear": "debug_block_linear",
    "late_rme_expansion_nocg": "h_a0",
    "late_nocg": "h_a0",
    "late_rme_nocg": "h_a0",
    "late_rme_cartesian_hybrid": "h_a1",
    "late_rme_ict_hybrid": "h_a1",
    "late_rme_cartesian": "h_a1",
    "late_block_expansion_cg": "h_b0",
    "expansion_cg": "h_b0",
    "late_block_wigner": "h_b0",
    "late_block_cartesian_projector": "h_b1",
    "late_block_ict_projector": "h_b1",
    "late_block_cartesian": "h_b1",
}
_DYNAMIC_DIRECT_AO_ALIASES = {
    "direct_ao_projector",
    "ao_projector",
    "direct_ao",
}


def get_output_route_spec(canonical_name: str) -> OutputRouteSpec:
    try:
        return _ROUTE_SPECS[str(canonical_name).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown canonical output route {canonical_name!r}; expected one of "
            f"{sorted(_ROUTE_SPECS)}."
        ) from exc


def _warn_alias(alias: str, canonical: str) -> None:
    warnings.warn(
        f"Output route alias {alias!r} is deprecated; use output_route={canonical!r}.",
        DeprecationWarning,
        stacklevel=3,
    )


def _direct_ao_variant(
    projector_backend: Optional[str],
    projector_bank_path: Optional[Union[str, Path]],
) -> str:
    backend = normalize_projector_backend(projector_backend)
    if backend == _REFERENCE_BACKEND:
        return "p_b0"
    if not projector_bank_path:
        raise ValueError(
            "A precomputed direct-AO route requires ao_projector_bank_path."
        )
    provenance = projector_bank_provenance(projector_bank_path)
    return "p_b1_ict" if provenance.uses_ict else "p_b1_reference"


def _canonicalize_name(
    name: str,
    *,
    projector_backend: Optional[str],
    projector_bank_path: Optional[Union[str, Path]],
    warn_alias: bool,
) -> str:
    normalized = str(name).strip().lower()
    if normalized in _ROUTE_SPECS:
        return normalized
    if normalized in _DYNAMIC_DIRECT_AO_ALIASES:
        canonical = _direct_ao_variant(projector_backend, projector_bank_path)
        if warn_alias:
            _warn_alias(normalized, canonical)
        return canonical
    canonical = _STATIC_ALIASES.get(normalized)
    if canonical is None:
        allowed = sorted(set(_ROUTE_SPECS) | set(_STATIC_ALIASES) | _DYNAMIC_DIRECT_AO_ALIASES)
        raise ValueError(f"Unknown output route {name!r}; expected one of {allowed}.")
    if warn_alias:
        _warn_alias(normalized, canonical)
    return canonical


def _validate_projector_variant(
    spec: OutputRouteSpec,
    projector_backend: Optional[str],
    projector_bank_path: Optional[Union[str, Path]],
) -> None:
    if spec.head_factory != "ao_projector":
        return
    backend = normalize_projector_backend(projector_backend)
    if spec.canonical_name == "p_b0":
        if backend != _REFERENCE_BACKEND:
            raise ValueError("output_route='p_b0' requires reference_wigner backend.")
        return
    if backend != _PRECOMPUTED_BACKEND or not projector_bank_path:
        raise ValueError(
            f"output_route={spec.canonical_name!r} requires a precomputed projector bank."
        )
    provenance = projector_bank_provenance(projector_bank_path)
    if provenance.uses_ict != spec.uses_ict:
        expected = "validated Cartesian/ICT" if spec.uses_ict else "non-ICT reference/control"
        raise ValueError(
            f"output_route={spec.canonical_name!r} requires {expected} provenance; "
            f"bank source={provenance.source!r}, generator={provenance.generator_id!r}."
        )
    if spec.projector_source is not None and provenance.source != spec.projector_source:
        raise ValueError(
            f"output_route={spec.canonical_name!r} requires projector source "
            f"{spec.projector_source!r}, got {provenance.source!r}."
        )


def resolve_output_route(
    *,
    output_route: Optional[str] = None,
    legacy_mode: Optional[str] = None,
    projector_backend: Optional[str] = _REFERENCE_BACKEND,
    projector_bank_path: Optional[Union[str, Path]] = None,
) -> OutputRouteSpec:
    """Resolve canonical route semantics from new or legacy configuration."""
    if output_route is None and legacy_mode is None:
        canonical = "legacy_rme"
    elif output_route is not None:
        canonical = _canonicalize_name(
            output_route,
            projector_backend=projector_backend,
            projector_bank_path=projector_bank_path,
            warn_alias=True,
        )
        if legacy_mode is not None:
            legacy_canonical = _canonicalize_name(
                legacy_mode,
                projector_backend=projector_backend,
                projector_bank_path=projector_bank_path,
                warn_alias=False,
            )
            _warn_alias(str(legacy_mode), legacy_canonical)
            if legacy_canonical != canonical:
                raise ValueError(
                    f"output_route={canonical!r} conflicts with rme_head_mode={legacy_mode!r}."
                )
    else:
        canonical = _canonicalize_name(
            legacy_mode,
            projector_backend=projector_backend,
            projector_bank_path=projector_bank_path,
            warn_alias=False,
        )
        _warn_alias(str(legacy_mode), canonical)

    spec = get_output_route_spec(canonical)
    _validate_projector_variant(spec, projector_backend, projector_bank_path)
    return spec


def normalize_legacy_head_mode(mode: Optional[str]) -> str:
    """Compatibility normalization preserving the historical return values."""
    if mode is None:
        return "legacy_linear"
    normalized = str(mode).strip().lower()
    if normalized in _DYNAMIC_DIRECT_AO_ALIASES:
        return "direct_ao_projector"
    canonical = _canonicalize_name(
        normalized,
        projector_backend=_REFERENCE_BACKEND,
        projector_bank_path=None,
        warn_alias=False,
    )
    return get_output_route_spec(canonical).legacy_mode


def select_final_irreps(
    spec: OutputRouteSpec,
    *,
    ordinary_hidden: o3.Irreps,
    orbpair_irreps: o3.Irreps,
    ao_pair_irreps: Optional[o3.Irreps],
) -> o3.Irreps:
    if spec.final_irreps_kind == "ordinary_hidden":
        return o3.Irreps(ordinary_hidden)
    if spec.final_irreps_kind == "orbpair":
        return o3.Irreps(orbpair_irreps).sort()[0].simplify()
    if spec.final_irreps_kind == "ao_pair":
        if ao_pair_irreps is None:
            raise ValueError("AO-pair output route requires ao_pair_irreps.")
        return o3.Irreps(ao_pair_irreps)
    raise RuntimeError(f"Unsupported final_irreps_kind={spec.final_irreps_kind!r}.")



def effective_product_scope(spec: OutputRouteSpec, configured: Optional[str]) -> str:
    """Return the route-fixed product policy and reject semantic drift."""
    if configured is None:
        return spec.product_scope or "missing_only"
    requested = str(configured).strip().lower()
    if requested not in {"missing_only", "all"}:
        raise ValueError(
            "output product scope must be 'missing_only' or 'all', "
            f"got {configured!r}."
        )
    if spec.product_scope is not None and requested != spec.product_scope:
        raise ValueError(
            f"output_route={spec.canonical_name!r} fixes product_scope="
            f"{spec.product_scope!r}, got {requested!r}."
        )
    return spec.product_scope or requested

def validate_prediction_route(
    spec: OutputRouteSpec, prediction: Mapping[str, object]
) -> None:
    method = str(prediction.get("method", "e3tb")).strip().lower()
    if method != spec.prediction_method:
        raise ValueError(
            f"output_route={spec.canonical_name!r} requires "
            f"prediction.method={spec.prediction_method!r}, got {method!r}."
        )
    if spec.block_decoder is not None:
        decoder = str(prediction.get("block_decoder", spec.block_decoder)).strip().lower()
        if decoder != spec.block_decoder:
            raise ValueError(
                f"output_route={spec.canonical_name!r} requires "
                f"prediction.block_decoder={spec.block_decoder!r}, got {decoder!r}."
            )


def _linear_pair(ctx: OutputHeadContext):
    from e3nn.o3 import Linear

    edge = Linear(
        ctx.final_irreps,
        ctx.orbpair_irreps,
        shared_weights=True,
        internal_weights=True,
        biases=True,
    )
    node = Linear(
        ctx.final_irreps,
        ctx.orbpair_irreps,
        shared_weights=True,
        internal_weights=True,
        biases=True,
    )
    return edge, node


def _common_kwargs(ctx: OutputHeadContext) -> Dict[str, object]:
    return {
        "rank": ctx.rank,
        "init": ctx.init,
        "condition": ctx.condition,
        "dtype": ctx.dtype,
        "device": ctx.device,
    }


def _factory_legacy(ctx: OutputHeadContext):
    return _linear_pair(ctx)


def _factory_rme_fusion(ctx: OutputHeadContext):
    from .rme_nocg_fusion_head import RMENoCGFusionHead

    legacy_edge, legacy_node = _linear_pair(ctx)
    kwargs = _common_kwargs(ctx)
    return (
        RMENoCGFusionHead(
            ctx.final_irreps, ctx.orbpair_irreps, legacy=legacy_edge, **kwargs
        ),
        RMENoCGFusionHead(
            ctx.final_irreps, ctx.orbpair_irreps, legacy=legacy_node, **kwargs
        ),
    )


def _factory_h_a0(ctx: OutputHeadContext):
    from .late_rme_expansion_nocg import LateRMEExpansionNoCGHead

    kwargs = _common_kwargs(ctx)
    return (
        LateRMEExpansionNoCGHead(ctx.final_irreps, ctx.orbpair_irreps, **kwargs),
        LateRMEExpansionNoCGHead(ctx.final_irreps, ctx.orbpair_irreps, **kwargs),
    )


def _factory_h_a1(ctx: OutputHeadContext):
    from .late_rme_cartesian_hybrid import LateRMECartesianHybridHead

    kwargs = _common_kwargs(ctx)
    kwargs["product_scope"] = ctx.product_scope
    return (
        LateRMECartesianHybridHead(ctx.final_irreps, ctx.orbpair_irreps, **kwargs),
        LateRMECartesianHybridHead(ctx.final_irreps, ctx.orbpair_irreps, **kwargs),
    )


def _factory_debug_block(ctx: OutputHeadContext):
    from .block_native_head import BlockNativeLinearHead

    common = {
        "init": ctx.init,
        "dtype": ctx.dtype,
        "device": ctx.device,
    }
    return (
        BlockNativeLinearHead(ctx.final_irreps, ctx.max_norb, symmetrize=False, **common),
        BlockNativeLinearHead(ctx.final_irreps, ctx.max_norb, symmetrize=True, **common),
    )


def _factory_h_b0(ctx: OutputHeadContext):
    from .late_block_expansion_cg import LateBlockExpansionCGHead

    kwargs = _common_kwargs(ctx)
    # P1-3: only the h_b0 head has a legacy/fused reduction-path choice; keep this
    # local to this factory so no other route's head constructor sees the kwarg.
    kwargs["cg_head_impl"] = ctx.cg_head_impl
    return (
        LateBlockExpansionCGHead(ctx.final_irreps, ctx.full_basis, symmetrize=False, **kwargs),
        LateBlockExpansionCGHead(ctx.final_irreps, ctx.full_basis, symmetrize=True, **kwargs),
    )


def _factory_h_b1(ctx: OutputHeadContext):
    from .late_block_cartesian_projector import LateBlockCartesianProjectorHead

    kwargs = _common_kwargs(ctx)
    kwargs["product_scope"] = ctx.product_scope
    return (
        LateBlockCartesianProjectorHead(ctx.final_irreps, ctx.full_basis, symmetrize=False, **kwargs),
        LateBlockCartesianProjectorHead(ctx.final_irreps, ctx.full_basis, symmetrize=True, **kwargs),
    )


def _factory_p_b(ctx: OutputHeadContext):
    from .ao_angular_projector import AOAngularProjectorHead

    kwargs = _common_kwargs(ctx)
    kwargs.update(
        {
            "normalization": ctx.ao_projector_normalization,
            "basis_convention": ctx.ao_projector_basis_convention,
            "projector_backend": ctx.ao_projector_backend,
            "projector_bank_path": ctx.ao_projector_bank_path,
        }
    )
    return (
        AOAngularProjectorHead(ctx.final_irreps, ctx.full_basis, symmetrize=False, **kwargs),
        AOAngularProjectorHead(ctx.final_irreps, ctx.full_basis, symmetrize=True, **kwargs),
    )


_HEAD_FACTORIES: Dict[str, Callable[[OutputHeadContext], Tuple[torch.nn.Module, torch.nn.Module]]] = {
    "legacy_linear": _factory_legacy,
    "rme_nocg_fusion": _factory_rme_fusion,
    "late_rme_expansion_nocg": _factory_h_a0,
    "late_rme_cartesian_hybrid": _factory_h_a1,
    "block_native_linear": _factory_debug_block,
    "late_block_expansion_cg": _factory_h_b0,
    "late_block_cartesian_projector": _factory_h_b1,
    "ao_projector": _factory_p_b,
}


def build_output_heads(
    spec: OutputRouteSpec, ctx: OutputHeadContext
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Instantiate edge/node heads and verify their declared contract."""
    try:
        factory = _HEAD_FACTORIES[spec.head_factory]
    except KeyError as exc:
        raise RuntimeError(f"No head factory for {spec.head_factory!r}.") from exc
    edge, node = factory(ctx)
    for head in (edge, node):
        contract = getattr(head, "output_contract", "rme" if spec.output_contract == "rme" else None)
        if contract != spec.output_contract:
            raise RuntimeError(
                f"{type(head).__name__} declares output_contract={contract!r}, "
                f"expected {spec.output_contract!r}."
            )
        if type(head).__name__ != spec.head_class_name:
            raise RuntimeError(
                f"output_route={spec.canonical_name!r} expected head class "
                f"{spec.head_class_name!r}, got {type(head).__name__!r}."
            )
        actual_ict = bool(getattr(head, "uses_ict", False))
        if actual_ict != spec.uses_ict:
            raise RuntimeError(
                f"output_route={spec.canonical_name!r} expected uses_ict={spec.uses_ict}, "
                f"but {type(head).__name__} reports {actual_ict}."
            )
        actual_precomputed = bool(
            getattr(head, "uses_precomputed_projector", False)
        )
        if actual_precomputed != spec.uses_precomputed_projector:
            raise RuntimeError(
                f"output_route={spec.canonical_name!r} expected "
                f"uses_precomputed_projector={spec.uses_precomputed_projector}, "
                f"but {type(head).__name__} reports {actual_precomputed}."
            )
    return edge, node

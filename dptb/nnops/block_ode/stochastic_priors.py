from __future__ import annotations

"""Stochastic residual/start-state prior draws for the block-space ODE subsystem.

Unifies the draw -> fill -> project -> certify template shared by three
near-identical ``HamiltonianCFM`` bodies: the ``projected_te`` residual
epsilon draw, the ``tied_irrep_gaussian`` residual epsilon draw, and the
H0+noise start-state draw inside ``_block_initial_state`` (extracted from
``dptb.nnops.flow.HamiltonianCFM``, the block-ode refactor's PR4).
``HamiltonianCFM`` keeps every one of the seven functions below under its
original name as a short delegator that forwards to this module, passing
itself as the ``owner`` back-reference -- the same "behaviour object with an
owner back-reference" idiom already used by
``dptb.nnops.block_ode.topology``/``endpoint_stats`` and, before those,
``dptb.nnops.dynamic_batch_controller.DynamicBatchController``/
``dptb.nnops.flow_priors.PriorContext``.

Monkeypatch-through-instance (see the refactor plan's section 4.3): every
call this module makes into another ``HamiltonianCFM`` private method --
the TE/tied-irrep draw engines (``_te_prior_like``/
``_tied_irrep_gaussian_prior_like``, both of which stay in ``flow.py``
proper since they are also used by the legacy non-block-ode prior path --
see the plan's section 2.3), ``_num_graphs``, ``_max_abs``,
``_seeded_generator``, ``_clone_block_state``, and the two assertion belts
below -- is looked up dynamically as ``owner.<name>(...)`` rather than
imported/bound once. ``test_h5_ta3_belt_fires_through_residual_eps_draw_path``
in ``test_residual_ao_block_ode.py`` monkeypatches a live ``HamiltonianCFM``
instance's ``_te_prior_like`` and depends on ``_residual_te_eps`` resolving
the patched version at call time; the one shared draw engine
(``_draw_residual_component``) below is now the single place that lookup
happens for all three prior families, so it must keep the same
``owner.<name>(...)`` dynamic-resolution shape for every one of them, not
just the one path the existing test happens to exercise. Note that
``_draw_residual_component`` itself is a brand-new internal helper that
never existed as a ``HamiltonianCFM`` method before this refactor -- no test
could have been monkeypatching it -- so the three call sites below invoke it
as a plain sibling function rather than through ``owner``.

Deliberate structural differences the shared template must reproduce
exactly (see ``_ResidualDrawConfig``), not paper over:
  * ``projected_te``/``tied_irrep_gaussian`` residual draws are pure epsilon
    (no H0 folded in); the ``_block_initial_state`` start-state draw folds
    ``h0_blocks`` in before projecting.
  * The two residual draws assert finite/nonzero AFTER
    block-conversion+projection (on the codec-image ``eps``); the
    start-state draw asserts on the RAW drawn noise BEFORE it is folded
    into H0 and projected -- these are genuinely different tensors
    (RME-space raw noise vs. block-space projected residual), not just a
    reordering of the same check.
  * The two residual draws only run their roundtrip-certify check when
    ``certify_image`` is due; the start-state draw always converts back to
    RME (it needs ``node_start``/``edge_start`` for its own return value
    regardless of certification) and, as a result, always re-certifies the
    roundtrip too, unconditionally -- this is pre-existing behavior,
    preserved verbatim, not a bug this refactor introduces or fixes.

What is deliberately **not** here: the TE/tied-irrep draw engines
themselves (``_te_prior_like``/``_tied_irrep_gaussian_prior_like``) and the
seeded-RNG substream machinery (``_seeded_generator`` and friends) stay in
``flow.py`` -- both are demonstrably shared with the legacy (pre-block-ode)
CFM prior path too (grep-confirmed against every call site), so moving them
here would force that legacy path to import from ``block_ode/``, inverting
this package's one-way ``block_ode/ -> flow.py`` import direction. This
module imports nothing from ``dptb.nnops.flow``.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.interfaces.blockwise_tensor import BlockTensorResult
from dptb.nnops.block_flow_codec import _FLOW_PROJECTED_STATE_TOKEN, project_block_state


def _assert_stochastic_prior_draw_finite_and_scaled(prior_name: str, components) -> None:
    """Runtime belt for any stochastic block prior draw before it is used.

    ``components`` is an iterable of ``(label, tensor, effective_scale)``.  A
    drawn component must be all-finite, and -- when its configured effective
    scale is nonzero and it has any elements -- must not be entirely zero (a
    silent all-masked / subnormal-underflow collapse of the stochastic
    bridge).  The error names the effective scale, dtype, and component label.
    """
    for label, tensor, effective_scale in components:
        if tensor is None:
            continue
        if tensor.numel() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(
                f"{prior_name} {label} draw contains NaN/Inf (effective scale="
                f"{float(effective_scale):.6g}, dtype={tensor.dtype}, "
                f"component={label})."
            )
        if (
            float(effective_scale) != 0.0
            and tensor.numel() > 0
            and not bool(tensor.detach().any().item())
        ):
            raise ValueError(
                f"{prior_name} {label} draw is entirely zero despite a nonzero "
                f"configured effective scale={float(effective_scale):.6g} "
                f"(dtype={tensor.dtype}, component={label})."
            )


def _assert_projected_te_draw_finite_and_scaled(owner: Any, components) -> None:
    """TA-3 runtime belt for any projected_te draw, before it is used."""
    owner._assert_stochastic_prior_draw_finite_and_scaled("projected_te", components)


@dataclass(frozen=True)
class _ResidualDrawConfig:
    """Per-prior-family configuration for :func:`_draw_residual_component`.

    Replaces 3 near-identical ``HamiltonianCFM`` method bodies with 1
    template + 3 instances of this config (module-level constants below).
    """

    # Owner attribute name of the draw engine (``_te_prior_like`` or
    # ``_tied_irrep_gaussian_prior_like``), resolved via ``getattr`` at call
    # time so an instance-level monkeypatch is still picked up.
    draw_engine_name: str
    # Extra static kwargs forwarded to the draw engine (TE passes
    # ``reference_scale=False``; tied_irrep's engine accepts no such kwarg).
    draw_kwargs: Mapping[str, Any] = field(default_factory=dict)
    # Owner attribute name holding the prior-strength multiplier used to
    # compute each component's ``effective_scale`` for the assertion belt.
    sigma_scale_attr: str = "te_prior_sigma"
    # True: assert via ``owner._assert_projected_te_draw_finite_and_scaled``
    # (fixed "projected_te" label). False: assert via
    # ``owner._assert_stochastic_prior_draw_finite_and_scaled(owner.prior, ...)``
    # (dynamic label). Mirrors the two distinct call chains the original
    # bodies used verbatim.
    assert_via_projected_te: bool = True
    # Label prefix for the roundtrip-certify ValueError text, e.g.
    # "Projected TE residual epsilon" / "Projected TE block start" /
    # "tied_irrep_gaussian residual epsilon".
    certify_error_label: str = ""
    # True only for the H0+noise start-state draw: fold ``h0_blocks`` in
    # before projecting, instead of projecting the pure noise directly.
    add_h0: bool = False
    # True only for the start-state draw: assert on the raw drawn
    # node/edge noise BEFORE it is assembled+projected. False (the two
    # residual draws): assert on the assembled+projected state instead.
    assert_before_assemble: bool = False
    # True only for the start-state draw: always reconvert to RME and
    # re-run the roundtrip-certify check regardless of ``certify_image``
    # (pre-existing behavior -- the conversion is needed for the return
    # value either way, so the check rides along unconditionally).
    certify_unconditional: bool = False
    # True only for the start-state draw: return
    # ``(state, node_conv - node_h0, edge_conv - edge_h0)`` instead of a
    # bare state.
    return_residual_tuple: bool = False


_TE_RESIDUAL_CONFIG = _ResidualDrawConfig(
    draw_engine_name="_te_prior_like",
    draw_kwargs={"reference_scale": False},
    sigma_scale_attr="te_prior_sigma",
    assert_via_projected_te=True,
    certify_error_label="Projected TE residual epsilon",
    add_h0=False,
    assert_before_assemble=False,
    certify_unconditional=False,
    return_residual_tuple=False,
)

_TIED_IRREP_RESIDUAL_CONFIG = _ResidualDrawConfig(
    draw_engine_name="_tied_irrep_gaussian_prior_like",
    draw_kwargs={},
    sigma_scale_attr="tied_irrep_sigma",
    assert_via_projected_te=False,
    certify_error_label="tied_irrep_gaussian residual epsilon",
    add_h0=False,
    assert_before_assemble=False,
    certify_unconditional=False,
    return_residual_tuple=False,
)

_TE_START_STATE_CONFIG = _ResidualDrawConfig(
    draw_engine_name="_te_prior_like",
    draw_kwargs={"reference_scale": False},
    sigma_scale_attr="te_prior_sigma",
    assert_via_projected_te=True,
    certify_error_label="Projected TE block start",
    add_h0=True,
    assert_before_assemble=True,
    certify_unconditional=True,
    return_residual_tuple=True,
)


def _draw_residual_component(
    owner: Any,
    data: AtomicDataDict.Type,
    node_like: torch.Tensor,
    edge_like: torch.Tensor,
    *,
    generator: Optional[torch.Generator],
    certify_image: bool,
    config: _ResidualDrawConfig,
    h0_blocks: Optional[BlockTensorResult] = None,
    node_h0: Optional[torch.Tensor] = None,
    edge_h0: Optional[torch.Tensor] = None,
) -> Any:
    """Shared draw -> fill -> project -> certify template.

    ``config`` selects the draw engine, the assertion dispatch/scale, the
    certify error label, and the structural switches that distinguish the
    pure-residual-epsilon shape (``add_h0=False``, gated certify, bare
    return) from the H0+noise start-state shape (``add_h0=True``,
    unconditional certify, 3-tuple return). See the module docstring for
    why these switches exist and must not be unified away.
    """
    assert owner.block_codec is not None
    prior_data = {
        key: data[key]
        for key in (
            AtomicDataDict.ATOM_TYPE_KEY,
            AtomicDataDict.EDGE_TYPE_KEY,
            _keys.BATCH_KEY,
            _keys.EDGE_INDEX_KEY,
            # TA-2: the stable per-graph id must ride the restricted prior
            # substream data so a SEEDED draw can key per-uid generators.
            _keys.SAMPLE_UID_KEY,
        )
        if key in data
    }
    num_graphs = owner._num_graphs(prior_data)
    draw_fn = getattr(owner, config.draw_engine_name)
    node_noise = draw_fn(
        torch.zeros_like(node_like),
        owner.node_sigma,
        data=prior_data,
        label="node",
        num_graphs=num_graphs,
        generator=generator,
        **config.draw_kwargs,
    )
    edge_noise = draw_fn(
        torch.zeros_like(edge_like),
        owner.edge_sigma,
        data=prior_data,
        label="edge",
        num_graphs=num_graphs,
        generator=generator,
        **config.draw_kwargs,
    )

    sigma_scale = getattr(owner, config.sigma_scale_attr)

    def run_assert(node_component: torch.Tensor, edge_component: torch.Tensor) -> None:
        components = (
            ("node", node_component, owner.node_sigma * sigma_scale),
            ("edge", edge_component, owner.edge_sigma * sigma_scale),
        )
        if config.assert_via_projected_te:
            owner._assert_projected_te_draw_finite_and_scaled(components)
        else:
            owner._assert_stochastic_prior_draw_finite_and_scaled(owner.prior, components)

    # TA-3: runtime belt on the draw before it is folded into/used as the
    # returned state -- all-finite, and not silently all-zero for a
    # component with a nonzero configured effective scale.
    if config.assert_before_assemble:
        run_assert(node_noise, edge_noise)

    noise_blocks = owner.block_codec.rme_to_blocks(
        data, node_noise, edge_noise, project=False
    )
    if config.add_h0:
        assembled = BlockTensorResult(
            h0_blocks.node_blocks + noise_blocks.node_blocks,
            h0_blocks.edge_blocks + noise_blocks.edge_blocks,
            h0_blocks.node_shapes,
            h0_blocks.edge_shapes,
        )
    else:
        assembled = noise_blocks
    state = project_block_state(data, owner.idp, assembled)

    if not config.assert_before_assemble:
        run_assert(state.node_blocks, state.edge_blocks)

    # The start-state draw needs node_conv/edge_conv for its own return
    # value regardless of certification; the two residual draws only need
    # them when a roundtrip-certify is actually due.
    need_conversion = (
        config.certify_unconditional or certify_image or config.return_residual_tuple
    )
    node_conv = edge_conv = None
    if need_conversion:
        node_conv, edge_conv = owner.block_codec.blocks_to_rme(
            data,
            state,
            certify_image=certify_image,
            _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
        )
    if config.certify_unconditional or certify_image:
        roundtrip = owner.block_codec.rme_to_blocks(
            data, node_conv, edge_conv, project=True
        )
        residual = max(
            owner._max_abs(roundtrip.node_blocks - state.node_blocks),
            owner._max_abs(roundtrip.edge_blocks - state.edge_blocks),
        )
        if residual > owner.block_inverse_atol:
            raise ValueError(
                f"{config.certify_error_label} is outside the certified codec "
                f"image: max residual={residual:.6g}, "
                f"atol={owner.block_inverse_atol:.6g}."
            )

    if config.return_residual_tuple:
        return state, node_conv - node_h0, edge_conv - edge_h0
    return state


def _residual_te_eps(
    owner: Any,
    data: AtomicDataDict.Type,
    node_like: torch.Tensor,
    edge_like: torch.Tensor,
    *,
    generator: Optional[torch.Generator],
    certify_image: bool,
) -> BlockTensorResult:
    """Draw the projected_te residual epsilon in block space; NO H0 added.

    Mirrors the TE-noise draw of :func:`_block_initial_state` exactly (same
    node/edge sigma semantics and the same seeded-generator path) but forms
    the PURE noise residual ``eps = project(rme_to_blocks(noise, project=False))``
    instead of A's ``H0 + noise`` start state: the direct-residual mode tracks
    ``D = H - H0``, so the t=0 boundary of D is the prior noise alone.

    ``eps`` is in the certified codec image by construction (the forward CG map
    is linear, ``rme_to_blocks(project=False)`` lands on that image, and
    :func:`project_block_state` is an image-preserving projection).  When
    certification is due the epsilon is certified the same way A certifies its
    noisy start -- a repack roundtrip residual bounded by ``block_inverse_atol``.

    Seeded-replay semantics (TA-1/TA-2): a fixed ``generator`` reproduces the
    SAME epsilon bitwise only for an identical tensor layout, and otherwise
    only DISTRIBUTIONAL equivariance -- sample(R.x, seed) is drawn from the
    rotation of sample(x, seed)'s law, not equal to it.  The draw is keyed
    per graph by ``sample_uid`` (batch-composition invariant) but is NOT
    pathwise equivariant under an input rotation/permutation; for a
    transformable latent, pass an explicit ``prior_state`` to the sampler and
    rotate/permute it alongside the input.
    """
    return _draw_residual_component(
        owner,
        data,
        node_like,
        edge_like,
        generator=generator,
        certify_image=certify_image,
        config=_TE_RESIDUAL_CONFIG,
    )


def _residual_tied_irrep_gaussian_eps(
    owner: Any,
    data: AtomicDataDict.Type,
    node_like: torch.Tensor,
    edge_like: torch.Tensor,
    *,
    generator: Optional[torch.Generator],
    certify_image: bool,
) -> BlockTensorResult:
    """Draw the multiplicity-tied total-L residual epsilon; NO H0 added."""
    return _draw_residual_component(
        owner,
        data,
        node_like,
        edge_like,
        generator=generator,
        certify_image=certify_image,
        config=_TIED_IRREP_RESIDUAL_CONFIG,
    )


def _residual_stochastic_eps(
    owner: Any,
    data: AtomicDataDict.Type,
    node_like: torch.Tensor,
    edge_like: torch.Tensor,
    *,
    generator: Optional[torch.Generator],
    certify_image: bool,
) -> BlockTensorResult:
    if owner.prior in owner._projected_te_prior_names:
        return owner._residual_te_eps(
            data,
            node_like,
            edge_like,
            generator=generator,
            certify_image=certify_image,
        )
    if owner.prior in owner._tied_irrep_gaussian_prior_names:
        return owner._residual_tied_irrep_gaussian_eps(
            data,
            node_like,
            edge_like,
            generator=generator,
            certify_image=certify_image,
        )
    raise RuntimeError(f"Unsupported residual stochastic prior {owner.prior!r}.")


def _block_initial_state(
    owner: Any,
    data: AtomicDataDict.Type,
    h0_blocks: BlockTensorResult,
    *,
    prior_seed: Optional[int] = None,
    certify_image: bool = True,
) -> Tuple[BlockTensorResult, torch.Tensor, torch.Tensor]:
    """Draw one endpoint-independent start state and certify its codec image.

    Seeded-replay semantics (TA-1/TA-2): a fixed ``prior_seed`` reproduces the
    SAME start state bitwise only for an identical tensor layout, plus
    DISTRIBUTIONAL equivariance; the draw is keyed per graph by ``sample_uid``
    (batch-composition invariant) but is NOT pathwise equivariant under an
    input rotation/permutation.

    A-mode asymmetry (TA-1a): unlike ``residual_ao_block_ode``, this A-mode
    start state is the FULL state ``B0 = H0 + eps`` (not a pure residual D0),
    so an explicit transformable ``prior_state`` latent does NOT drop in
    trivially here (a caller-supplied full B0 would need its own H0-relative
    residual bookkeeping and a distinct verbatim-vs-projected contract).  The
    explicit-latent path is therefore offered only by the residual sampler;
    A-mode is left seeded-only by design.
    """
    assert owner.block_codec is not None
    node_h0, edge_h0 = owner.block_codec.blocks_to_rme(
        data,
        h0_blocks,
        certify_image=certify_image,
        _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
    )
    if owner.prior == "zero":
        if prior_seed is not None:
            raise ValueError("block-space prior_seed requires prior='projected_te'.")
        return (
            owner._clone_block_state(h0_blocks),
            torch.zeros_like(node_h0),
            torch.zeros_like(edge_h0),
        )

    if owner.prior not in owner._projected_te_prior_names:
        raise RuntimeError(f"Unsupported block-space prior {owner.prior!r}.")

    def draw(
        generator: Optional[torch.Generator],
    ) -> Tuple[BlockTensorResult, torch.Tensor, torch.Tensor]:
        return _draw_residual_component(
            owner,
            data,
            node_h0,
            edge_h0,
            generator=generator,
            certify_image=certify_image,
            config=_TE_START_STATE_CONFIG,
            h0_blocks=h0_blocks,
            node_h0=node_h0,
            edge_h0=edge_h0,
        )

    if prior_seed is None:
        return draw(None)
    generator = owner._seeded_generator(h0_blocks.node_blocks.device, prior_seed)
    return draw(generator)


def _strict_image_certification_due(owner: Any) -> bool:
    """Schedule only the pure repack/residual image-space self-check."""
    batch = owner._strict_certification_batches
    if owner._strict_certification_mode == "always":
        return True
    if owner._strict_certification_mode == "first_batch":
        return batch == 0
    return batch % owner._strict_certification_period == 0

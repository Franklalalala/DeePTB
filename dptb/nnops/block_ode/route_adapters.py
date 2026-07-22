from __future__ import annotations

"""Per-route adapter objects for the block-space ODE subsystem.

The three block-ODE routes -- generic full-H (``block_ode`` with
``output_space='ao_block_ode'``), ``uureal_block_ode`` (reduced-SOC compact
uu-real residual), and ``residual_ao_block_ode`` (non-SOC direct AO residual)
-- share one Euler rollout engine and one ``prepare_batch`` preamble but differ
in a handful of route-specific decisions. This module homes those decisions in
one adapter object per route:

    * :class:`FullHRouteAdapter`   -- generic full-H route
    * :class:`UuRealRouteAdapter`  -- ``uureal_block_ode`` route
    * :class:`ResidualRouteAdapter`-- ``residual_ao_block_ode`` route

Each adapter is a *behaviour object with an owner back-reference*, matching the
idiom this repository already uses for ``dptb.nnops.flow_priors.PriorContext``
and ``dptb.nnops.dynamic_batch_controller.DynamicBatchController``: it stores the
owning :class:`dptb.nnops.flow.HamiltonianCFM` as ``self._owner`` and reads its
state and helper methods live through that reference. Crucially, every call an
adapter makes back into owner state is spelled ``self._owner._method(...)`` --
resolved dynamically off the live instance at call time, never a reference
captured at import -- so an instance-level monkeypatch of an owner helper on a
real ``HamiltonianCFM`` (the fault-injection mechanism several tests rely on;
see ``test_h5_ta3_belt_fires_through_residual_eps_draw_path`` in
``test_residual_ao_block_ode.py`` for the canonical example) continues to
resolve through the patched instance. ``HamiltonianCFM`` keeps every original
method name as a 1-line delegator forwarding into the corresponding adapter.

Import direction (block_ode package rule): this module must never import
``dptb.nnops.flow`` -- ``flow.py`` imports downward into ``block_ode/*``, never
the reverse.

Responsibility map (this file grows route-by-route across the PR5 sub-lanes;
each adapter class below is deliberately left open for these additions and its
docstring names them):

    PR5a (this PR): route-specific data-contract check
        -- ``UuRealRouteAdapter.require_block_contract``
           (from ``HamiltonianCFM._require_uureal_block_contract``)
        -- ``ResidualRouteAdapter.require_block_contract``
           (from ``HamiltonianCFM._require_spatial_residual_block_contract``)
        -- ``FullHRouteAdapter`` has no analogous check: generic ``block_ode``
           has none today, so none is invented.
    PR5b (rollout engine): the route-specific slivers of each sampler --
        per-step state-write-in, endpoint decode (full-H additionally runs
        ``block_codec.endpoint_to_full`` before blending; the other two already
        track a pure residual), and the route's initial state.
    PR5c (prepare_batch routes): each route's ``prepare_batch`` branch (the
        training-time one-step state assembly / blend / endpoint projection).
"""

from typing import Any, Optional

import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.interfaces.blockwise_tensor import BlockTensorResult
from dptb.nnops.block_flow_codec import _FLOW_PROJECTED_STATE_TOKEN, project_block_state
from dptb.nnops.flow_context import CFMContext


class FullHRouteAdapter:
    """Adapter for the generic full-H block-ODE route (``output_space='ao_block_ode'``).

    Has no route-specific data-contract check today (generic ``block_ode`` never
    grew one), so this class carries only the owner back-reference for now.

    PR5b/PR5c will add: the full-H per-step state-write-in (RME carried under
    ``node_h0_key``/``edge_h0_key``), the endpoint decode that runs
    ``block_codec.endpoint_to_full(...)`` before blending (unique to this route),
    the full-H initial state, and this route's ``prepare_batch`` branch.
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    # -- training-time one-step assembly (PR5c) --------------------------
    def prepare_batch(
        self,
        data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        *,
        t: torch.Tensor,
        node_t: torch.Tensor,
        edge_t: torch.Tensor,
        device: Any,
        dtype: Any,
        prior_seed: Optional[int],
    ) -> tuple:
        """Generic full-H ``prepare_batch`` branch (extracted verbatim from
        ``HamiltonianCFM.prepare_batch``; ``self.`` -> ``owner.``).

        The full-H route's H0 keys carry the *interpolated* state RME (unlike the
        residual route, whose H0 keys carry the constant physical H0 and whose
        residual state travels in the spatial residual keys).
        """
        owner = self._owner
        assert owner.block_codec is not None
        certify_image = owner._strict_image_certification_due()
        h0_blocks, h0_projection_residual = owner._physical_h0_blocks(
            data, device=device, dtype=dtype
        )
        endpoint, endpoint_projection_residual = owner._block_state_from_fields(
            ref_data,
            ref_data,
            node_key=owner.node_block_target_key,
            edge_key=owner.edge_block_target_key,
            node_shape_key=owner.node_block_shape_key,
            edge_shape_key=owner.edge_block_shape_key,
            label=f"{owner.target_semantics} endpoint",
            device=device,
            dtype=dtype,
        )
        # Coupled RME is generated only from certified blocks.  Legacy main
        # and H0 feature side channels are intentionally ignored because
        # older LMDBs store AO-product gathers under the same key names.
        node_base, edge_base = owner.block_codec.blocks_to_rme(
            data,
            h0_blocks,
            certify_image=certify_image,
            _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
        )
        node_target, edge_target = owner.block_codec.blocks_to_rme(
            data,
            endpoint,
            certify_image=certify_image,
            _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
        )
        block_start, node_prior, edge_prior = owner._block_initial_state(
            data,
            h0_blocks,
            prior_seed=prior_seed,
            certify_image=certify_image,
        )

        full_endpoint = owner.block_codec.endpoint_to_full(endpoint, h0_blocks)
        full_endpoint = project_block_state(data, owner.idp, full_endpoint)
        node_alpha = node_t.reshape((-1,) + (1,) * (full_endpoint.node_blocks.ndim - 1))
        edge_alpha = edge_t.reshape((-1,) + (1,) * (full_endpoint.edge_blocks.ndim - 1))
        block_current = project_block_state(
            data,
            owner.idp,
            BlockTensorResult(
                node_blocks=(1.0 - node_alpha) * block_start.node_blocks
                + node_alpha * full_endpoint.node_blocks,
                edge_blocks=(1.0 - edge_alpha) * block_start.edge_blocks
                + edge_alpha * full_endpoint.edge_blocks,
                node_shapes=h0_blocks.node_shapes,
                edge_shapes=h0_blocks.edge_shapes,
            ),
        )
        node_current, edge_current = owner.block_codec.blocks_to_rme(
            data,
            block_current,
            certify_image=certify_image,
            _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
        )
        if owner.detach_interpolated_h0:
            node_current = node_current.detach()
            edge_current = edge_current.detach()
        data[owner.node_h0_key] = node_current
        data[owner.edge_h0_key] = edge_current
        if owner.overwrite_feature_keys:
            data[owner.node_target_key] = node_current
            data[owner.edge_target_key] = edge_current
        data[owner.flow_time_key] = t.detach()
        ref_data[owner.flow_time_key] = t.detach()
        ref_data["_block_ode_target_projection_residual"] = torch.as_tensor(
            endpoint_projection_residual, device=device, dtype=dtype
        )
        ref_data["_block_ode_h0_projection_residual"] = torch.as_tensor(
            h0_projection_residual, device=device, dtype=dtype
        )
        # Endpoint/H0 block side channels are flow authority, not model
        # inputs.  Removing them also breaks the shallow aliases created by
        # Trainer's input/reference copies, so an in-place model cannot
        # rewrite the labels used below by ``loss``.
        owner._drop_block_authority_fields(data)
        # Advance only after all unconditional contract gates and any due
        # certification pass; a failed batch remains due on retry.
        owner._strict_certification_batches += 1
        return data, ref_data, CFMContext(
            t=t,
            node_t=node_t,
            edge_t=edge_t,
            node_base=node_base,
            edge_base=edge_base,
            node_target=node_target,
            edge_target=edge_target,
            node_current=node_current,
            edge_current=edge_current,
            node_prior=node_prior,
            edge_prior=edge_prior,
            block_target_semantics=owner.target_semantics,
        )


class UuRealRouteAdapter:
    """Adapter for the ``uureal_block_ode`` route (reduced-SOC compact uu-real residual).

    PR5b/PR5c will add: the uu-real per-step state-write-in
    (``_attach_uureal_residual_state`` block sidecars), the residual endpoint
    decode (already a pure delta, no ``endpoint_to_full``), the uu-real initial
    state, and this route's ``prepare_batch`` branch.
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    # -- route-specific data contract (PR5a) -----------------------------
    def require_block_contract(
        self,
        data: AtomicDataDict.Type,
        *,
        require_endpoint_labels: bool = True,
    ) -> None:
        keep = int(self._owner.idp.reduced_matrix_element)
        required = (
            "blockwise_spatial_schema",
            "blockwise_target_mode",
            "blockwise_source_target_feature_width",
            "blockwise_source_h0_feature_width",
            "soc_uureal_compact",
            "soc_uureal_full_rme",
            "soc_uureal_keep",
            self._owner.node_h0_key,
            self._owner.edge_h0_key,
        )
        if require_endpoint_labels:
            # Training/scoring needs the residual endpoint labels; label-free
            # inference (sampling) starts from D_0=0 with mapper-derived shapes
            # and never reads them, so it must not demand them.
            required = required + (
                self._owner.node_block_target_key,
                self._owner.edge_block_target_key,
                self._owner.node_block_shape_key,
                self._owner.edge_block_shape_key,
            )
        missing = self._owner._missing_keys(data, required)
        if missing:
            raise KeyError(f"uureal_block_ode data contract missing keys={missing}.")
        expected = {
            "blockwise_spatial_schema": "deeptb.blockwise_spatial/v1",
            "blockwise_target_mode": "already-delta",
            "soc_uureal_compact": True,
            "soc_uureal_full_rme": keep * 8,
            "soc_uureal_keep": keep,
        }
        for key, wanted in expected.items():
            actual = self._owner._metadata_scalar(data[key])
            if actual != wanted:
                raise ValueError(
                    f"uureal_block_ode requires {key}={wanted!r}; got {actual!r}."
                )
        # A normal full-SOC->uu_real conversion records the ORIGINAL source width
        # (e.g. 5832 for a keep=729 compact target).  The contract is that the
        # *stored* tensors are keep-wide and keep matches the mapper; the recorded
        # source width must only be a valid integer >= keep (keep==source when the
        # source was already compact).  Forcing source_width==keep rejected every
        # genuine converter product.
        for key in (
            "blockwise_source_target_feature_width",
            "blockwise_source_h0_feature_width",
        ):
            raw_actual = self._owner._metadata_scalar(data[key])
            try:
                actual = int(raw_actual)
            except (TypeError, ValueError):
                raise ValueError(
                    f"uureal_block_ode requires integer {key}; got {raw_actual!r}."
                )
            if actual < keep:
                raise ValueError(
                    f"uureal_block_ode requires {key} >= keep={keep}; got {actual}."
                )
        for key in (self._owner.node_h0_key, self._owner.edge_h0_key):
            value = torch.as_tensor(data[key])
            if value.ndim != 2 or value.shape[-1] != keep:
                raise ValueError(
                    f"uureal_block_ode requires {key} width {keep}; got {tuple(value.shape)}."
                )
            if value.is_complex() or not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"uureal_block_ode requires finite real {key}.")

    # -- training-time one-step assembly (PR5c) --------------------------
    def prepare_batch(
        self,
        data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        *,
        t: torch.Tensor,
        node_t: torch.Tensor,
        edge_t: torch.Tensor,
        device: Any,
        dtype: Any,
        prior_seed: Optional[int],
    ) -> tuple:
        """``uureal_block_ode`` ``prepare_batch`` branch (extracted verbatim from
        ``HamiltonianCFM.prepare_batch``; ``self.`` -> ``owner.``).

        The compact uu-real residual travels as a pure ``t``-scaled delta in the
        uu-real residual block keys; ``prior_seed`` is unused (zero prior only).
        """
        owner = self._owner
        endpoint, endpoint_projection_residual = owner._block_state_from_fields(
            ref_data,
            ref_data,
            node_key=owner.node_block_target_key,
            edge_key=owner.edge_block_target_key,
            node_shape_key=owner.node_block_shape_key,
            edge_shape_key=owner.edge_block_shape_key,
            label="residual_dh endpoint",
            device=device,
            dtype=dtype,
        )
        node_alpha = node_t.reshape(
            (-1,) + (1,) * (endpoint.node_blocks.ndim - 1)
        )
        edge_alpha = edge_t.reshape(
            (-1,) + (1,) * (endpoint.edge_blocks.ndim - 1)
        )
        current = project_block_state(
            data,
            owner.idp,
            BlockTensorResult(
                node_alpha * endpoint.node_blocks,
                edge_alpha * endpoint.edge_blocks,
                endpoint.node_shapes,
                endpoint.edge_shapes,
            ),
        )
        if owner.detach_interpolated_h0:
            current = BlockTensorResult(
                current.node_blocks.detach(),
                current.edge_blocks.detach(),
                current.node_shapes,
                current.edge_shapes,
            )
        owner._attach_uureal_residual_state(data, current)
        data[owner.flow_time_key] = t.detach()
        ref_data[owner.flow_time_key] = t.detach()
        ref_data["_block_ode_target_projection_residual"] = torch.as_tensor(
            endpoint_projection_residual, device=device, dtype=dtype
        )
        owner._drop_block_authority_fields(data)
        return data, ref_data, CFMContext(
            t=t,
            node_t=node_t,
            edge_t=edge_t,
            node_base=torch.as_tensor(data[owner.node_h0_key]),
            edge_base=torch.as_tensor(data[owner.edge_h0_key]),
            node_target=None,
            edge_target=None,
            node_current=current.node_blocks,
            edge_current=current.edge_blocks,
            node_prior=torch.zeros_like(current.node_blocks),
            edge_prior=torch.zeros_like(current.edge_blocks),
            block_target_semantics="residual_dh",
        )


class ResidualRouteAdapter:
    """Adapter for the ``residual_ao_block_ode`` route (non-SOC direct AO residual).

    PR5b/PR5c will add: the spatial-residual per-step state-write-in
    (``_attach_spatial_residual_state`` block sidecars, re-asserting the constant
    physical-H0 channel each step), the residual endpoint decode (already a pure
    delta, no ``endpoint_to_full``), the D_0=0 initial state
    (``_prior_state_as_residual_D0``), and this route's ``prepare_batch`` branch.
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    # -- route-specific data contract (PR5a) -----------------------------
    def require_block_contract(
        self,
        data: AtomicDataDict.Type,
        *,
        require_endpoint_labels: bool = True,
    ) -> None:
        """Fail-closed non-SOC direct-residual block contract.

        ``residual_ao_block_ode`` consumes plain non-SOC raw records: the physical
        H0 AO blocks (+ shapes) are always required, while the delta endpoint
        labels are required only for training/scoring.  Label-free sampling starts
        from D_0=0 with mapper-derived shapes and never reads them, so it must not
        demand them.  Mapper non-SOC-ness is a ctor check, not a per-batch one.
        """
        required = (
            self._owner.node_h0_block_key,
            self._owner.edge_h0_block_key,
            self._owner.node_h0_block_shape_key,
            self._owner.edge_h0_block_shape_key,
        )
        if require_endpoint_labels:
            required = required + (
                self._owner.node_block_target_key,
                self._owner.edge_block_target_key,
                self._owner.node_block_shape_key,
                self._owner.edge_block_shape_key,
            )
        missing = self._owner._missing_keys(data, required)
        if missing:
            raise KeyError(f"residual_ao_block_ode data contract missing keys={missing}.")
        for key in (self._owner.node_h0_block_key, self._owner.edge_h0_block_key):
            self._owner._require_real_finite_tensor(
                data[key], label=f"residual_ao_block_ode {key}"
            )
        if require_endpoint_labels:
            for key in (self._owner.node_block_target_key, self._owner.edge_block_target_key):
                self._owner._require_real_finite_tensor(
                    data[key], label=f"residual_ao_block_ode {key}"
                )
        # A raw non-SOC record must never carry compact uu_real metadata: those
        # markers belong to uureal_block_ode and would signal a converter product
        # masquerading as a plain spatial record.
        forbidden_metadata = [
            key
            for key in (
                "soc_uureal_compact",
                "soc_uureal_full_rme",
                "soc_uureal_keep",
                "blockwise_spatial_schema",
                "blockwise_target_mode",
            )
            if key in data
        ]
        if forbidden_metadata:
            raise ValueError(
                "residual_ao_block_ode consumes raw non-SOC records; uu-real "
                f"compact metadata is forbidden (found {forbidden_metadata})."
            )
        # Nor the reduced-SOC delta state keys, which would silently redirect the
        # conditioning channel away from the spatial residual projector.
        forbidden_state = [
            key
            for key in (
                _keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY,
                _keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY,
                _keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
                _keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
            )
            if key in data
        ]
        if forbidden_state:
            raise ValueError(
                "residual_ao_block_ode consumes raw non-SOC records; uu-real "
                f"residual state keys are forbidden (found {forbidden_state})."
            )

    # -- training-time one-step assembly (PR5c) --------------------------
    def prepare_batch(
        self,
        data: AtomicDataDict.Type,
        ref_data: AtomicDataDict.Type,
        *,
        t: torch.Tensor,
        node_t: torch.Tensor,
        edge_t: torch.Tensor,
        device: Any,
        dtype: Any,
        prior_seed: Optional[int],
    ) -> tuple:
        """``residual_ao_block_ode`` ``prepare_batch`` branch (extracted verbatim
        from ``HamiltonianCFM.prepare_batch``; ``self.`` -> ``owner.``).

        The physical H0 RME is the constant conditioning channel written once to
        the H0 keys; the pure residual state D travels in the spatial residual
        block keys. ``owner._require_spatial_residual_block_contract`` is called
        through the owner so an instance-level monkeypatch still resolves.
        """
        owner = self._owner
        assert owner.block_codec is not None
        owner._require_spatial_residual_block_contract(data)
        owner._require_spatial_residual_block_contract(ref_data)
        certify_image = owner._strict_image_certification_due()
        h0_blocks, h0_projection_residual = owner._physical_h0_blocks(
            data, device=device, dtype=dtype
        )
        endpoint, endpoint_projection_residual = owner._block_state_from_fields(
            ref_data,
            ref_data,
            node_key=owner.node_block_target_key,
            edge_key=owner.edge_block_target_key,
            node_shape_key=owner.node_block_shape_key,
            edge_shape_key=owner.edge_block_shape_key,
            label="residual_dh endpoint",
            device=device,
            dtype=dtype,
        )
        # Physical H0 RME is the CONSTANT conditioning channel: written to the
        # H0 keys exactly once and never interpolated.  The pure residual state
        # D travels in the spatial residual block keys instead.  This is the
        # deliberate divergence from the generic ao_block_ode branch below,
        # whose H0 keys carry the interpolated state RME.
        node_base, edge_base = owner.block_codec.blocks_to_rme(
            data,
            h0_blocks,
            certify_image=certify_image,
            _construction_token=_FLOW_PROJECTED_STATE_TOKEN,
        )
        data[owner.node_h0_key] = node_base
        data[owner.edge_h0_key] = edge_base
        node_alpha = node_t.reshape((-1,) + (1,) * (endpoint.node_blocks.ndim - 1))
        edge_alpha = edge_t.reshape((-1,) + (1,) * (endpoint.edge_blocks.ndim - 1))
        if owner.prior == "zero":
            # Zero prior: D_t = t * D1 exactly (no (1 - t) * prior term).  Legacy
            # feature-key overwrites are intentionally skipped for B: the H0 keys
            # already expose the physical conditioning channel honestly.
            D_t = project_block_state(
                data,
                owner.idp,
                BlockTensorResult(
                    node_alpha * endpoint.node_blocks,
                    edge_alpha * endpoint.edge_blocks,
                    endpoint.node_shapes,
                    endpoint.edge_shapes,
                ),
            )
            node_prior_blocks = None
            edge_prior_blocks = None
        else:
            # Stochastic bridge: draw eps in the certified codec image
            # (NO H0 added), then D_t = project((1 - t) * eps + t * D1).
            # The (1 - t) * eps term removes the deterministic
            # D_t = t * D1 shortcut left by the zero prior.  prior_seed
            # selects the deterministic validation stream; training keeps
            # fresh rowwise noise.
            eps = owner._residual_stochastic_eps(
                data,
                node_base,
                edge_base,
                generator=owner._seeded_generator(device, prior_seed),
                certify_image=certify_image,
            )
            D_t = project_block_state(
                data,
                owner.idp,
                BlockTensorResult(
                    (1.0 - node_alpha) * eps.node_blocks
                    + node_alpha * endpoint.node_blocks,
                    (1.0 - edge_alpha) * eps.edge_blocks
                    + edge_alpha * endpoint.edge_blocks,
                    endpoint.node_shapes,
                    endpoint.edge_shapes,
                ),
            )
            node_prior_blocks = eps.node_blocks
            edge_prior_blocks = eps.edge_blocks
        if owner.detach_interpolated_h0:
            D_t = BlockTensorResult(
                D_t.node_blocks.detach(),
                D_t.edge_blocks.detach(),
                D_t.node_shapes,
                D_t.edge_shapes,
            )
        owner._attach_spatial_residual_state(data, D_t)
        data[owner.flow_time_key] = t.detach()
        ref_data[owner.flow_time_key] = t.detach()
        ref_data["_block_ode_target_projection_residual"] = torch.as_tensor(
            endpoint_projection_residual, device=device, dtype=dtype
        )
        ref_data["_block_ode_h0_projection_residual"] = torch.as_tensor(
            h0_projection_residual, device=device, dtype=dtype
        )
        owner._drop_block_authority_fields(data)
        owner._strict_certification_batches += 1
        return data, ref_data, CFMContext(
            t=t,
            node_t=node_t,
            edge_t=edge_t,
            node_base=node_base,
            edge_base=edge_base,
            node_target=None,
            edge_target=None,
            node_current=D_t.node_blocks,
            edge_current=D_t.edge_blocks,
            # Zero prior keeps the exact zeros_like telemetry; projected_te
            # exposes the drawn epsilon.  CFMContext.node_prior has no runtime
            # consumer for this mode (the block-ODE loss reads node_current /
            # node_target only), so this is telemetry-only either way.
            node_prior=(
                torch.zeros_like(D_t.node_blocks)
                if node_prior_blocks is None
                else node_prior_blocks
            ),
            edge_prior=(
                torch.zeros_like(D_t.edge_blocks)
                if edge_prior_blocks is None
                else edge_prior_blocks
            ),
            block_target_semantics="residual_dh",
        )


class RouteAdapters:
    """Route-keyed registry of the three block-ODE route adapters.

    Constructed once per ``HamiltonianCFM`` (see ``HamiltonianCFM.__init__``) and
    holds one adapter instance per block-ODE route, each carrying a
    back-reference to the same owning ``HamiltonianCFM``. The three adapters are
    reachable by name (``.full_h``/``.uureal``/``.residual``) and by route via
    :meth:`select`.

    ``select(owner)`` returns the adapter for the owner's active block-ODE route.
    The three routes are mutually exclusive and, once ``block_ode`` is set,
    exhaustive; callers gate this on a block-ODE route being active (the legacy
    non-block-ode ``rme``/``ao_block`` path is out of block-ODE scope and is not
    routed here). PR5b uses ``.full_h``/``.uureal``/``.residual`` to parameterize
    the shared rollout engine; PR5c uses ``select`` to route ``prepare_batch``.
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self.full_h = FullHRouteAdapter(owner)
        self.uureal = UuRealRouteAdapter(owner)
        self.residual = ResidualRouteAdapter(owner)

    def select(self, owner: Any):
        """Return the route adapter for ``owner``'s active block-ODE route.

        Mirrors ``prepare_batch``'s branch order (uureal, then residual, then
        generic full-H); assumes a block-ODE route is active.
        """
        if owner.uureal_block_ode:
            return self.uureal
        if owner.residual_ao_block_ode:
            return self.residual
        return self.full_h

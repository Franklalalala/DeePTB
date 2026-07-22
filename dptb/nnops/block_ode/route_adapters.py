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

from typing import Any

import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    attach_prediction_block_tensors,
    infer_block_shapes,
    mapper_max_norb,
)
from dptb.nnops.block_flow_codec import project_block_state
from dptb.nnops.block_ode.rollout import RolloutContext


class FullHRouteAdapter:
    """Adapter for the generic full-H block-ODE route (``output_space='ao_block_ode'``).

    Has no route-specific data-contract check today (generic ``block_ode`` never
    grew one).

    PR5b (this PR) added the full-H rollout slivers: the initial state
    (``_block_initial_state``), the per-step state-write-in (RME carried under
    ``node_h0_key``/``edge_h0_key``), the endpoint decode that runs
    ``block_codec.endpoint_to_full(...)`` before blending (unique to this route),
    and the final-state assembly. PR5c will add this route's ``prepare_batch``
    branch.
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    # -- rollout slivers (PR5b) ------------------------------------------
    def initial_state(self, owner: Any, state: AtomicDataDict.Type, *, prior_seed, prior_state):
        """Draw the full A-mode start state B0 = H0 + eps and freeze the context.

        ``prior_state`` is not accepted by this route (``sample()`` rejects a
        non-None ``prior_state`` before dispatch); it is threaded through for a
        uniform adapter signature and ignored here.
        """
        if owner.block_codec is None:
            raise RuntimeError("Block-space ODE codec was not constructed.")
        topology_sidecar = owner._snapshot_block_topology(state)
        owner._restore_block_topology(state, topology_sidecar)
        h0_blocks, _ = owner._physical_h0_blocks(state)
        accumulator_dtype = owner.dtype
        h0_blocks = BlockTensorResult(
            h0_blocks.node_blocks.to(dtype=accumulator_dtype),
            h0_blocks.edge_blocks.to(dtype=accumulator_dtype),
            h0_blocks.node_shapes,
            h0_blocks.edge_shapes,
        )
        block_current, _, _ = owner._block_initial_state(
            state, h0_blocks, prior_seed=prior_seed
        )
        owner._drop_block_authority_fields(state)
        ctx = RolloutContext(
            topology_sidecar=topology_sidecar,
            device=block_current.node_blocks.device,
            dtype=accumulator_dtype,
            h0_blocks=h0_blocks,
        )
        return block_current, ctx

    def check_step(self, owner: Any, *, cur_t, next_t, alpha) -> None:
        if not bool((cur_t < 1.0).item()):
            raise RuntimeError("Block-space ODE must never evaluate the model at t=1.")
        if not bool(((alpha > 0.0) & (alpha <= 1.0)).item()):
            raise RuntimeError(
                f"Invalid block-space ODE blend alpha={float(alpha.item())}."
            )

    def write_state_in(self, owner: Any, state: AtomicDataDict.Type, current, ctx) -> None:
        node_rme, edge_rme = owner.block_codec.blocks_to_rme(state, current)
        state[owner.node_h0_key] = node_rme.clone()
        state[owner.edge_h0_key] = edge_rme.clone()
        if owner.overwrite_feature_keys:
            state[owner.node_target_key] = node_rme.clone()
            state[owner.edge_target_key] = edge_rme.clone()

    def decode_endpoint(self, owner: Any, prediction, merged, current, ctx):
        raw_node_endpoint = owner._require_real_finite_tensor(
            prediction[owner.node_output_key],
            label="block-space ODE node endpoint prediction",
        )
        raw_edge_endpoint = owner._require_real_finite_tensor(
            prediction[owner.edge_output_key],
            label="block-space ODE edge endpoint prediction",
        )
        endpoint = BlockTensorResult(
            raw_node_endpoint.to(
                device=current.node_blocks.device, dtype=ctx.dtype
            ),
            raw_edge_endpoint.to(
                device=current.edge_blocks.device, dtype=ctx.dtype
            ),
            current.node_shapes,
            current.edge_shapes,
        )
        full_endpoint = owner.block_codec.endpoint_to_full(endpoint, ctx.h0_blocks)
        full_endpoint = project_block_state(merged, owner.idp, full_endpoint)
        return full_endpoint

    def finalize(self, owner: Any, state: AtomicDataDict.Type, current, ctx, *, num_graphs):
        node_final_rme, edge_final_rme = owner.block_codec.blocks_to_rme(state, current)
        state[owner.node_h0_key] = node_final_rme
        state[owner.edge_h0_key] = edge_final_rme
        if owner.overwrite_feature_keys:
            state[owner.node_target_key] = node_final_rme
            state[owner.edge_target_key] = edge_final_rme
        attach_prediction_block_tensors(
            state,
            current,
            node_key=owner.node_output_key,
            edge_key=owner.edge_output_key,
            node_shape_key=_keys.NODE_PRED_HAMIL_BLOCK_SHAPE_KEY,
            edge_shape_key=_keys.EDGE_PRED_HAMIL_BLOCK_SHAPE_KEY,
        )
        state[owner.flow_time_key] = torch.ones(
            num_graphs, device=current.node_blocks.device, dtype=ctx.dtype
        )
        return state


class UuRealRouteAdapter:
    """Adapter for the ``uureal_block_ode`` route (reduced-SOC compact uu-real residual).

    PR5b (this PR) added the uu-real rollout slivers: the compact-uu zero initial
    state, the per-step state-write-in (``_attach_uureal_residual_state`` block
    sidecars), the pure-residual endpoint decode (already a pure delta, no
    ``endpoint_to_full``), and the final-state assembly. PR5c will add this
    route's ``prepare_batch`` branch.
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

    # -- rollout slivers (PR5b) ------------------------------------------
    def initial_state(self, owner: Any, state: AtomicDataDict.Type, *, prior_seed, prior_state):
        """Roll out the compact-uu residual state D from D0 = 0 (no H0 materialization)."""
        owner._require_uureal_block_contract(state, require_endpoint_labels=False)
        topology_sidecar = owner._snapshot_block_topology(state)
        owner._restore_block_topology(state, topology_sidecar)
        node_shapes, edge_shapes = infer_block_shapes(state, owner.idp, device=owner.device)
        canvas = mapper_max_norb(owner.idp)
        node_count = int(node_shapes.shape[0])
        edge_count = int(edge_shapes.shape[0])
        current = BlockTensorResult(
            torch.zeros((node_count, canvas, canvas), dtype=owner.dtype, device=owner.device),
            torch.zeros((edge_count, canvas, canvas), dtype=owner.dtype, device=owner.device),
            node_shapes,
            edge_shapes,
        )
        owner._drop_block_authority_fields(state)
        ctx = RolloutContext(
            topology_sidecar=topology_sidecar,
            device=owner.device,
            dtype=owner.dtype,
        )
        return current, ctx

    def check_step(self, owner: Any, *, cur_t, next_t, alpha) -> None:
        return None

    def write_state_in(self, owner: Any, state: AtomicDataDict.Type, current, ctx) -> None:
        owner._attach_uureal_residual_state(state, current)

    def decode_endpoint(self, owner: Any, prediction, merged, current, ctx):
        endpoint = BlockTensorResult(
            owner._require_real_finite_tensor(
                prediction[owner.node_output_key],
                label="uureal residual node endpoint",
            ).to(device=owner.device, dtype=owner.dtype),
            owner._require_real_finite_tensor(
                prediction[owner.edge_output_key],
                label="uureal residual edge endpoint",
            ).to(device=owner.device, dtype=owner.dtype),
            current.node_shapes,
            current.edge_shapes,
        )
        endpoint = project_block_state(merged, owner.idp, endpoint)
        return endpoint

    def finalize(self, owner: Any, state: AtomicDataDict.Type, current, ctx, *, num_graphs):
        owner._attach_uureal_residual_state(state, current)
        attach_prediction_block_tensors(
            state,
            current,
            node_key=owner.node_output_key,
            edge_key=owner.edge_output_key,
            node_shape_key=_keys.NODE_PRED_HAMIL_BLOCK_SHAPE_KEY,
            edge_shape_key=_keys.EDGE_PRED_HAMIL_BLOCK_SHAPE_KEY,
        )
        state[owner.flow_time_key] = torch.ones(
            num_graphs, device=owner.device, dtype=owner.dtype
        )
        return state


class ResidualRouteAdapter:
    """Adapter for the ``residual_ao_block_ode`` route (non-SOC direct AO residual).

    PR5b (this PR) added the spatial-residual rollout slivers: the initial state
    (zero / explicit-latent ``_prior_state_as_residual_D0`` / stochastic
    ``_residual_stochastic_eps`` D0, plus the constant physical-H0 RME channel),
    the per-step state-write-in (``_attach_spatial_residual_state`` block sidecars,
    re-asserting the constant physical-H0 channel each step), the pure-residual
    endpoint decode (already a pure delta, no ``endpoint_to_full``), and the
    ``H = H0 + D`` exactly-once finalize. PR5c will add this route's
    ``prepare_batch`` branch.
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

    # -- rollout slivers (PR5b) ------------------------------------------
    def initial_state(self, owner: Any, state: AtomicDataDict.Type, *, prior_seed, prior_state):
        """Build the residual D0 and derive the constant physical-H0 RME channel."""
        if owner.block_codec is None:
            raise RuntimeError("Block-space ODE codec was not constructed.")
        owner._require_spatial_residual_block_contract(
            state, require_endpoint_labels=False
        )
        topology_sidecar = owner._snapshot_block_topology(state)
        owner._restore_block_topology(state, topology_sidecar)
        h0_blocks, _ = owner._physical_h0_blocks(state)
        h0_blocks = BlockTensorResult(
            h0_blocks.node_blocks.to(dtype=owner.dtype),
            h0_blocks.edge_blocks.to(dtype=owner.dtype),
            h0_blocks.node_shapes,
            h0_blocks.edge_shapes,
        )
        # Physical H0 RME is derived ONCE from the certified H0 blocks and written
        # to the H0 keys as the constant conditioning channel (contract-(2) rollout
        # lock): the model sees the SAME physical H0 RME at every step, while only
        # the residual state D advances (carried in the spatial residual keys).
        node_base, edge_base = owner.block_codec.blocks_to_rme(
            state, h0_blocks, certify_image=True
        )
        if owner.prior == "zero":
            # Zero prior: D0 = 0 exactly (byte-identical to v1).  A prior_seed and
            # an explicit prior_state are both meaningless here and rejected,
            # mirroring _block_initial_state's zero-prior symmetry (the top-level
            # sample() guard rejects them before dispatch; this is defense in depth
            # for direct callers).
            if prior_seed is not None:
                raise ValueError(
                    "residual_ao_block_ode with prior='zero' uses an exact zero "
                    "prior and rejects prior_seed."
                )
            if prior_state is not None:
                raise ValueError(
                    "residual_ao_block_ode with prior='zero' uses an exact zero "
                    "prior and rejects prior_state."
                )
            current = BlockTensorResult(
                torch.zeros_like(h0_blocks.node_blocks),
                torch.zeros_like(h0_blocks.edge_blocks),
                h0_blocks.node_shapes,
                h0_blocks.edge_shapes,
            )
        elif prior_state is not None:
            # TA-1 explicit latent: D0 = prior_state verbatim (validated/certified).
            # Mutually exclusive with prior_seed -- the caller supplies the state
            # rather than seeding a draw of it.
            if prior_seed is not None:
                raise ValueError(
                    "residual_ao_block_ode prior_state and prior_seed are mutually "
                    "exclusive: supply an explicit latent OR seed a draw, not both."
                )
            current = owner._prior_state_as_residual_D0(state, prior_state, h0_blocks)
        else:
            # Stochastic residual prior: D0 = eps, drawn deterministically for a
            # given seed and certified in the codec image, matching the training
            # boundary state D_0 = project(eps) exactly.
            current = owner._residual_stochastic_eps(
                state,
                node_base,
                edge_base,
                generator=owner._seeded_generator(node_base.device, prior_seed),
                certify_image=True,
            )
        owner._drop_block_authority_fields(state)
        state[owner.node_h0_key] = node_base
        state[owner.edge_h0_key] = edge_base
        ctx = RolloutContext(
            topology_sidecar=topology_sidecar,
            device=owner.device,
            dtype=owner.dtype,
            h0_blocks=h0_blocks,
            node_base=node_base,
            edge_base=edge_base,
        )
        return current, ctx

    def check_step(self, owner: Any, *, cur_t, next_t, alpha) -> None:
        return None

    def write_state_in(self, owner: Any, state: AtomicDataDict.Type, current, ctx) -> None:
        owner._attach_spatial_residual_state(state, current)
        # Re-assert the constant channel every step: a model that echoed a
        # mutated H0 key back through ``merged`` must not drift contract (2).
        state[owner.node_h0_key] = ctx.node_base
        state[owner.edge_h0_key] = ctx.edge_base

    def decode_endpoint(self, owner: Any, prediction, merged, current, ctx):
        endpoint = BlockTensorResult(
            owner._require_real_finite_tensor(
                prediction[owner.node_output_key],
                label="spatial residual node endpoint",
            ).to(device=owner.device, dtype=owner.dtype),
            owner._require_real_finite_tensor(
                prediction[owner.edge_output_key],
                label="spatial residual edge endpoint",
            ).to(device=owner.device, dtype=owner.dtype),
            current.node_shapes,
            current.edge_shapes,
        )
        endpoint = project_block_state(merged, owner.idp, endpoint)
        return endpoint

    def finalize(self, owner: Any, state: AtomicDataDict.Type, current, ctx, *, num_graphs):
        # Assemble the full Hamiltonian H = H0 + D exactly ONCE, outside the ODE.
        owner._attach_spatial_residual_state(state, current)
        full = project_block_state(
            state,
            owner.idp,
            BlockTensorResult(
                ctx.h0_blocks.node_blocks + current.node_blocks,
                ctx.h0_blocks.edge_blocks + current.edge_blocks,
                ctx.h0_blocks.node_shapes,
                ctx.h0_blocks.edge_shapes,
            ),
        )
        attach_prediction_block_tensors(
            state,
            full,
            node_key=owner.node_output_key,
            edge_key=owner.edge_output_key,
            node_shape_key=_keys.NODE_PRED_HAMIL_BLOCK_SHAPE_KEY,
            edge_shape_key=_keys.EDGE_PRED_HAMIL_BLOCK_SHAPE_KEY,
        )
        # H0 keys stay PHYSICAL H0 RME (deliberate divergence from the generic
        # ao_block_ode sampler, which overwrites them with the final state RME).
        state[owner.node_h0_key] = ctx.node_base
        state[owner.edge_h0_key] = ctx.edge_base
        state[owner.flow_time_key] = torch.ones(
            num_graphs, device=owner.device, dtype=owner.dtype
        )
        return state


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

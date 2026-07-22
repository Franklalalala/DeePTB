"""The single Euler-integration engine shared by the three block-ODE samplers.

The generic full-H (``_sample_block_ode``), compact-uu (``_sample_uureal_block_ode``)
and non-SOC direct-residual (``_sample_residual_ao_block_ode``) samplers ran three
near-identical Euler main loops.  This module homes that loop once, in
:func:`run`, and parameterizes it by a route adapter (see
``dptb.nnops.block_ode.route_adapters``) for the handful of route-specific
decisions:

    * ``initial_state``   -- build the route's start state ``current`` and the
                             immutable :class:`RolloutContext` (topology sidecar,
                             device/dtype, physical-H0 blocks / RME);
    * ``check_step``      -- per-step guards (only the full-H route asserts
                             ``t < 1`` and the blend-``alpha`` bound);
    * ``write_state_in``  -- write ``current`` into the model-input state under
                             the route's keys, just before the model call;
    * ``decode_endpoint`` -- decode the raw model output into the projected block
                             endpoint that the shared blend consumes (the full-H
                             route additionally runs ``block_codec.endpoint_to_full``);
    * ``finalize``        -- assemble the route's returned state after the loop.

The shared middle -- model-input assembly (clone state minus the output-only
keys, restore the frozen topology sidecar, reinject the LEM sidecar), the
``model(...)`` call + freshness check, the merge/restore, and the
``alpha = (next_t - cur_t) / (1 - cur_t)`` endpoint blend -- lives here verbatim.
The extraction is a pure code-motion: no float operation is added, removed, or
reordered, so all three samplers stay bit-for-bit identical (pinned by
``test_block_ode_rollout_bitexact.py``).

Every call back into owner state is spelled ``owner._method(...)`` -- resolved
dynamically off the live ``HamiltonianCFM`` instance at call time, never captured
at import -- so an instance-level monkeypatch of an owner helper continues to
resolve through the patched instance (see the ``block_ode`` package's
owner-back-reference idiom).  Import direction: this module never imports
``dptb.nnops.flow`` (``flow.py`` imports downward into ``block_ode/*``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import BlockTensorResult
from dptb.nnops.block_flow_codec import project_block_state


# Mirror of ``dptb.nnops.flow._LEM_INPUT_SIDECAR_KEYS``, kept local so this module
# honors the block_ode -> flow import-direction rule (it must not import flow.py).
# LemMoEV3H0 consumes and deletes these input-side accelerators, so they are held
# outside the mutable rollout state and an unchanged copy is reinjected at every
# model call (a precomputed active set cannot drift on step 2+).
_LEM_INPUT_SIDECAR_KEYS = (
    _keys.LEM_ACTIVE_EDGES_KEY,
    _keys.LEM_CUTOFF_COEFFS_KEY,
    _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY,
)


@dataclass
class RolloutContext:
    """Immutable per-rollout context handed from ``initial_state`` to the engine.

    Carries only the route-specific values the shared loop and ``finalize`` need:
    the frozen topology ``snapshot`` restored around every model call, the
    ``device``/``dtype`` the route integrates in (used for ``times`` and the
    per-step ``flow_time`` write; ``dtype`` also serves as the full-H accumulator
    dtype), the physical-H0 blocks (``h0_blocks``; full-H uses them for
    ``endpoint_to_full``, residual for the ``H = H0 + D`` finalize), and the
    constant physical-H0 RME channel (``node_base``/``edge_base``; residual only).
    """

    topology_sidecar: Dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    h0_blocks: Optional[BlockTensorResult] = None
    node_base: Optional[torch.Tensor] = None
    edge_base: Optional[torch.Tensor] = None


def run(
    owner: Any,
    adapter: Any,
    model: torch.nn.Module,
    state: Dict[str, Any],
    *,
    num_steps: int,
    num_graphs: int,
    prior_seed: Optional[int] = None,
    prior_state: Any = None,
) -> Dict[str, Any]:
    """Euler-integrate one block-ODE route to the final state and return it.

    ``adapter`` is the route adapter (``owner._route_adapters.full_h`` /
    ``.uureal`` / ``.residual``) supplying the five route-specific slivers named
    in the module docstring; everything else is shared verbatim across routes.
    ``prior_seed``/``prior_state`` are forwarded to ``initial_state`` and consumed
    only by the routes that accept them (the full-H route uses ``prior_seed``; the
    residual route uses both; the uu-real route uses neither).
    """
    current, ctx = adapter.initial_state(
        owner, state, prior_seed=prior_seed, prior_state=prior_state
    )
    output_only_keys = owner._block_ode_output_only_keys()
    lem_sidecar = {
        key: state.pop(key)
        for key in _LEM_INPUT_SIDECAR_KEYS
        if key in state
    }
    times = torch.linspace(
        0.0, 1.0, num_steps + 1, device=ctx.device, dtype=ctx.dtype
    )

    for step in range(num_steps):
        cur_t = times[step]
        next_t = times[step + 1]
        alpha = (next_t - cur_t) / (1.0 - cur_t)
        adapter.check_step(owner, cur_t=cur_t, next_t=next_t, alpha=alpha)

        adapter.write_state_in(owner, state, current, ctx)
        state[owner.flow_time_key] = torch.full(
            (num_graphs,), float(cur_t.item()), device=ctx.device, dtype=ctx.dtype
        )
        # Supported DeePTB modules mutate the dictionary and may use in-place
        # tensor operations.  Give each ODE step owned tensor storage so those
        # writes cannot alias the persistent rollout state.
        model_input = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in state.items()
            if key not in output_only_keys
        }
        owner._restore_block_topology(
            model_input, ctx.topology_sidecar, clone_values=True
        )
        model_input.update(
            {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in lem_sidecar.items()
            }
        )
        prediction = model(model_input)
        owner._require_fresh_block_ode_outputs(prediction, step=step + 1)
        merged = state.copy()
        merged.update(prediction)
        owner._restore_block_topology(merged, ctx.topology_sidecar)
        for key in _LEM_INPUT_SIDECAR_KEYS:
            merged.pop(key, None)

        endpoint = adapter.decode_endpoint(owner, prediction, merged, current, ctx)
        current = project_block_state(
            merged,
            owner.idp,
            BlockTensorResult(
                (1.0 - alpha) * current.node_blocks + alpha * endpoint.node_blocks,
                (1.0 - alpha) * current.edge_blocks + alpha * endpoint.edge_blocks,
                current.node_shapes,
                current.edge_shapes,
            ),
        )
        state = merged

    return adapter.finalize(owner, state, current, ctx, num_graphs=num_graphs)

from __future__ import annotations

"""Block-space ODE endpoint loss for :class:`dptb.nnops.flow.HamiltonianCFM`.

This is a pure code-motion extraction of ``HamiltonianCFM._block_ode_endpoint_loss``
(the fail-closed endpoint loss scored on independent physical block freedoms for
the three block-ODE routes -- generic full-H, ``uureal_block_ode``, and
``residual_ao_block_ode``). ``HamiltonianCFM`` keeps the method under its
original name as a 1-line delegator that forwards to :func:`block_ode_endpoint_loss`
here, passing itself as the ``owner`` back-reference -- the same "behaviour
object with an owner back-reference" idiom already used by
``dptb.nnops.dynamic_batch_controller.DynamicBatchController`` and
``dptb.nnops.flow_priors.PriorContext``. Every helper this function calls
(``_metric_stats``, ``_reduce_component_stats``, ``_compatible_clean_stats``,
``_finalize_loss``, ``_time_weight``, ``_max_abs``) is looked up dynamically as
``owner._method(...)`` rather than imported and bound once, so an instance-level
monkeypatch of any of those helpers on a live ``HamiltonianCFM`` (the mechanism
several existing tests rely on -- see ``test_h5_ta3_belt_fires_through_residual_eps_draw_path``
in ``test_residual_ao_block_ode.py`` for the canonical example on a sibling
method) continues to resolve through the patched instance after this move.

Dual population note (P1-1, already landed in this lane's base commit): the
canonical reduction under ``train_flow_onsite_loss``/``train_flow_hopping_loss``
counts each independent physical freedom once (onsite upper triangle, one side
of each Hermitian edge pair). The ``_compatible_clean_stats`` payload and, for
``uureal_block_ode``, the ``train_compatible_directed_*`` keys instead count
every stored directed AO entry -- the same population
``HamilBlockwiseNexTHamLoss.compatible_loss_from_stats`` uses -- so a label the
criterion treats as directed-full always carries the directed-full population.
See the inline comment ahead of the ``state`` dict below for the full
derivation; it is carried over verbatim from the pre-move code.

What is deliberately **not** here (see the block-ode refactor plan, section
2.2/2.3): ``_block_endpoint_loss`` (the older, unrelated ``output_space='ao_block'``
one-step cross-space adapter) stays in ``flow.py`` -- it is not part of the
block-ODE subsystem. The generic ``_metric_stats``/``_reduce_component_stats``/
``_finalize_loss``/``_compatible_clean_stats``/``_time_weight``/``_max_abs``
cluster also stays in ``flow.py``: it is shared verbatim with the legacy
(pre-block-ode) CFM ``loss()`` path, so moving it here would force that legacy
path to import from ``block_ode/``, inverting this package's one-way
``block_ode/ -> flow.py`` import direction.
"""

from typing import Any, Dict, Tuple

import torch

from dptb.data import AtomicDataDict
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_mask_from_shapes,
    strict_reverse_edge_index,
)
from dptb.nnops.block_flow_codec import project_block_state
from dptb.nnops.flow_context import CFMContext


def block_ode_endpoint_loss(
    owner: Any,
    pred_data: AtomicDataDict.Type,
    ref_data: AtomicDataDict.Type,
    ctx: CFMContext,
    *,
    prediction_is_full_h: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Fail-closed endpoint loss on independent physical block freedoms."""
    if ctx.block_target_semantics != owner.target_semantics:
        raise ValueError(
            "Block-space ODE target semantics changed between prepare_batch and loss."
        )
    missing = [
        key
        for key in (
            owner.node_output_key,
            owner.edge_output_key,
            owner.node_block_target_key,
            owner.edge_block_target_key,
            owner.node_block_shape_key,
            owner.edge_block_shape_key,
        )
        if key not in (pred_data if key in {owner.node_output_key, owner.edge_output_key} else ref_data)
    ]
    if missing:
        raise KeyError(f"Block-space ODE endpoint loss is missing required keys={missing}.")

    node_pred = pred_data[owner.node_output_key]
    edge_pred = pred_data[owner.edge_output_key]
    node_target = torch.as_tensor(
        ref_data[owner.node_block_target_key], device=node_pred.device, dtype=node_pred.dtype
    )
    edge_target = torch.as_tensor(
        ref_data[owner.edge_block_target_key], device=edge_pred.device, dtype=edge_pred.dtype
    )
    if node_pred.shape != node_target.shape or edge_pred.shape != edge_target.shape:
        raise ValueError(
            "Block-space ODE prediction/target canvas mismatch: "
            f"node {tuple(node_pred.shape)} vs {tuple(node_target.shape)}, "
            f"edge {tuple(edge_pred.shape)} vs {tuple(edge_target.shape)}."
        )
    node_shapes = torch.as_tensor(ref_data[owner.node_block_shape_key], device=node_pred.device)
    edge_shapes = torch.as_tensor(ref_data[owner.edge_block_shape_key], device=edge_pred.device)
    pred_state = BlockTensorResult(node_pred, edge_pred, node_shapes, edge_shapes)
    target_state = BlockTensorResult(node_target, edge_target, node_shapes, edge_shapes)

    # Model outputs are predictions, not graph authority.  Pairing and
    # species shapes must come from the independently preserved reference
    # topology or a returned/stale PBC shift can redefine Hermitian mates.
    target_projected = project_block_state(ref_data, owner.idp, target_state)
    target_residual = max(
        owner._max_abs(target_projected.node_blocks - node_target),
        owner._max_abs(target_projected.edge_blocks - edge_target),
    )
    if target_residual > owner.block_inverse_atol:
        raise ValueError(
            "Block-space ODE target violates onsite/reverse/padding constraints: "
            f"max residual={target_residual:.6g}, atol={owner.block_inverse_atol}."
        )

    if prediction_is_full_h and owner.target_semantics == "residual_dh":
        if ctx.node_base is None or ctx.edge_base is None:
            raise ValueError("Residual block sample scoring requires both H0 RME components.")
        h0 = owner.block_codec.rme_to_blocks(
            ref_data, ctx.node_base, ctx.edge_base, project=True
        )
        target_state = owner.block_codec.endpoint_to_full(target_projected, h0)
        target_projected = project_block_state(ref_data, owner.idp, target_state)

    pred_projected = project_block_state(ref_data, owner.idp, pred_state)
    node_diff = pred_projected.node_blocks - target_projected.node_blocks
    edge_diff = pred_projected.edge_blocks - target_projected.edge_blocks

    node_valid = block_mask_from_shapes(
        pred_projected.node_shapes, tuple(node_diff.shape[-2:])
    )
    upper = torch.triu(
        torch.ones(tuple(node_diff.shape[-2:]), dtype=torch.bool, device=node_diff.device)
    )
    node_mask = node_valid & upper.unsqueeze(0)

    edge_valid = block_mask_from_shapes(
        pred_projected.edge_shapes, tuple(edge_diff.shape[-2:])
    )
    rev = strict_reverse_edge_index(
        ref_data, device=edge_diff.device, idp=owner.idp
    )
    rows = torch.arange(edge_diff.shape[0], device=edge_diff.device)
    canonical_rows = rows <= rev
    edge_mask = edge_valid & canonical_rows.view(-1, 1, 1)
    self_reverse = rows == rev
    if bool(self_reverse.any().item()):
        edge_mask[self_reverse] &= upper.unsqueeze(0)

    node_stats = owner._metric_stats(
        node_diff, node_mask, owner.loss_type, owner._time_weight(ctx.node_t)
    )
    edge_stats = owner._metric_stats(
        edge_diff, edge_mask, owner.loss_type, owner._time_weight(ctx.edge_t)
    )
    node_component = node_stats[0]
    edge_component = edge_stats[0]
    total = owner._reduce_component_stats(
        ((node_stats, owner.node_weight), (edge_stats, owner.edge_weight))
    )

    # NOTE: the flow objective publishes its endpoint node/edge metric ONLY under
    # the flow namespace (train_flow_onsite_loss/train_flow_hopping_loss).  It does
    # NOT alias it into the bare train_onsite_loss/train_hopping_loss: those legacy
    # tags carry feature-compatible semantics and are written solely by the
    # compatible pass (Trainer._compatible_loss_state*, legacy_prefix), so the two
    # namespaces never mix under one tag.  (Previously the flow value was aliased
    # here; the compatible pass then overwrote it every step, but if that pass were
    # ever throttled/absent the flow value would leak under the compatible tag.)
    #
    # P1-1 (block endpoint population parity): `_compatible_clean_stats` below is
    # handed to Trainer._compatible_loss_state_from_flow_stats, which reduces it
    # through the *criterion's own* `compatible_loss_from_stats`
    # (HamilBlockwiseNexTHamLoss).  That criterion's `block_components` counts
    # EVERY shape-active directed AO entry -- no onsite lower-triangle drop, no
    # reverse-edge dedup (see blockwise_tensor.block_components /
    # block_mask_from_shapes).  `node_mask`/`edge_mask` above intentionally keep
    # only ONE independent physical freedom per Hermitian pair (onsite upper
    # triangle, one side of each reverse-edge pair) for the *training* reduction
    # -- a DIFFERENT population from the criterion's directed-full count.
    # Publishing the canonical sums under a "block" label the criterion treats as
    # directed-full silently understated L1/RMSE (population off by ~2x on
    # non-symmetric error).  Feed `_compatible_clean_stats` the directed-full
    # `node_valid`/`edge_valid` masks instead -- the same population
    # `train_compatible_directed_*` below already uses -- so the label and the
    # population it carries finally agree with the criterion's own definition.
    # This is an exact recount, not an approximation: Hermitian pairing is
    # bit-exact post-projection (project_block_state symmetrizes onsite as
    # 0.5*(X+X.T) and derives every reverse edge as the literal transpose of a
    # shared `averaged` tensor via index_copy_), so summing node_diff/edge_diff
    # over the full directed mask reproduces exactly what an analytic
    # canonical-sum doubling would give (diagonal/self-reverse counted once,
    # off-diagonal/reverse-pair counted twice) -- see
    # test_block_ode_compatible_stats_match_criterion_directed_population for the
    # locked numerical identity.
    state: Dict[str, torch.Tensor] = {
        "train_flow_t": ctx.t.detach().mean(),
        "train_flow_weight": owner._time_weight(ctx.t).detach().mean(),
        "train_flow_onsite_loss": node_component.detach(),
        "train_flow_hopping_loss": edge_component.detach(),
        "block_ode_target_projection_residual": torch.as_tensor(
            target_residual, device=total.device, dtype=total.dtype
        ),
        "_compatible_clean_stats": {
            **owner._compatible_clean_stats(node_diff, node_valid, "onsite"),
            **owner._compatible_clean_stats(edge_diff, edge_valid, "hopping"),
        },
    }
    if owner.uureal_block_ode:
        # Historical-comparison metric (review P1-5).  Two deliberately
        # coexisting reductions:
        #   * canonical (`train_flow_*_loss` above): each independent physical
        #     freedom counted ONCE -- onsite upper triangle plus one
        #     canonical edge per Hermitian (i,j,R)/(j,i,-R) pair;
        #   * compatible_directed (`train_compatible_directed_*`): every
        #     stored directed coordinate counted, i.e. all mapper-valid
        #     onsite entries and ALL directed edges -- the same population
        #     the historical SOC uu-real RME losses (H-B0/H-A1 runs)
        #     averaged over, and (as of P1-1) the same population now backing
        #     the `_compatible_clean_stats` payload above.  Counts differ
        #     (~2x from canonical) and values differ whenever the error is not
        #     Hermitian-symmetric, so curves plotted against historical runs
        #     must use the directed keys.
        directed_node_stats = owner._metric_stats(
            node_diff, node_valid, owner.loss_type, owner._time_weight(ctx.node_t)
        )
        directed_edge_stats = owner._metric_stats(
            edge_diff, edge_valid, owner.loss_type, owner._time_weight(ctx.edge_t)
        )
        directed_total = owner._reduce_component_stats(
            (
                (directed_node_stats, owner.node_weight),
                (directed_edge_stats, owner.edge_weight),
            )
        )
        state["train_compatible_directed_onsite_loss"] = directed_node_stats[0].detach()
        state["train_compatible_directed_hopping_loss"] = directed_edge_stats[0].detach()
        state["train_compatible_directed_loss"] = directed_total.detach()
    return owner._finalize_loss(total, state, pred_data)

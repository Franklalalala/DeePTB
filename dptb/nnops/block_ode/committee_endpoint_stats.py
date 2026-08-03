# SPDX-License-Identifier: LGPL-3.0-or-later
"""Optional committee (multi-head) auxiliary loss for block-ODE endpoint scoring.

Flag-gated extension called from
:func:`dptb.nnops.block_ode.endpoint_stats.block_ode_endpoint_loss`. The
primary head's loss (``total`` in the caller) is computed exactly as before
this feature existed; this module only ADDS an auxiliary term for the extra
committee heads (index 1..K-1 of a ``committee_num_heads > 1`` embedding, see
``dptb.nn.embedding.lem_moe_v3.LemMoEV3``), so the extra heads receive
gradient and diverge from the primary head instead of sitting frozen at their
random initialization forever.

No-op guarantee: when the embedding's committee flag is off (the default,
``committee_num_heads=1``), the model never writes the
``committee_extra_*_hamiltonian_blocks`` keys this module reads, so
``committee_auxiliary_loss`` returns immediately with ``total``/``state``
untouched -- same object, no new keys, no extra compute.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from dptb.data.interfaces.blockwise_tensor import BlockTensorResult
from dptb.nnops.block_flow_codec import project_block_state

# Mirrors dptb.nn.embedding.lem_moe_v3.{COMMITTEE_EXTRA_NODE_HAMILTONIAN_KEY,
# COMMITTEE_EXTRA_EDGE_HAMILTONIAN_KEY}. Duplicated as plain string literals
# (rather than imported) to avoid a dptb.nnops -> dptb.nn.embedding import
# edge from this loss-side module; both call sites are covered by
# test_committee_heads.py's shared-constant check.
COMMITTEE_EXTRA_NODE_HAMILTONIAN_KEY = "committee_extra_node_hamiltonian_blocks"
COMMITTEE_EXTRA_EDGE_HAMILTONIAN_KEY = "committee_extra_edge_hamiltonian_blocks"


def committee_auxiliary_loss(
    owner: Any,
    pred_data: Dict[str, Any],
    ref_data: Dict[str, Any],
    ctx: Any,
    *,
    total: torch.Tensor,
    state: Dict[str, torch.Tensor],
    target_projected: BlockTensorResult,
    node_shapes: torch.Tensor,
    edge_shapes: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Add the committee extra-heads auxiliary loss to ``total`` when active.

    Active only when BOTH hold:
      (a) the model wrote committee extra-head prediction keys into
          ``pred_data`` (i.e. the embedding's ``committee_num_heads > 1``), and
      (b) ``owner.committee_aux_loss_weight > 0`` (``HamiltonianCFM``'s
          ``flow_options.committee_aux_loss_weight``, default 0.0).

    Either condition being false is a strict no-op: returns ``(total, state)``
    untouched (same ``total`` object, no new ``state`` keys).

    Each extra head's blocks are scored against the SAME projected target and
    the SAME onsite/canonical-edge mask the primary head uses (topology-only,
    prediction-independent, so safe to reuse), via the owner's own
    ``_metric_stats``/``_reduce_component_stats`` -- the identical reduction
    the primary ``block_ode_endpoint_loss`` uses -- so a head-k loss is exactly
    what that head's total would be if it were the primary head. The K-1
    per-head losses are averaged (not summed) so
    ``committee_aux_loss_weight=1.0`` is a natural "one more head's worth" of
    loss regardless of how many extra heads there are.
    """
    weight = float(getattr(owner, "committee_aux_loss_weight", 0.0))
    extra_node_pred = pred_data.get(COMMITTEE_EXTRA_NODE_HAMILTONIAN_KEY, None)
    extra_edge_pred = pred_data.get(COMMITTEE_EXTRA_EDGE_HAMILTONIAN_KEY, None)
    if weight <= 0.0 or extra_node_pred is None or extra_edge_pred is None:
        return total, state

    extra_node_pred = torch.as_tensor(extra_node_pred)
    extra_edge_pred = torch.as_tensor(extra_edge_pred)
    num_extra_heads = int(extra_node_pred.shape[0])
    if num_extra_heads == 0:
        return total, state
    if int(extra_edge_pred.shape[0]) != num_extra_heads:
        raise ValueError(
            "Committee extra-head node/edge prediction counts disagree: "
            f"{num_extra_heads} vs {int(extra_edge_pred.shape[0])}."
        )

    per_head_losses = []
    for head_idx in range(num_extra_heads):
        head_state = BlockTensorResult(
            extra_node_pred[head_idx],
            extra_edge_pred[head_idx],
            node_shapes,
            edge_shapes,
        )
        head_projected = project_block_state(ref_data, owner.idp, head_state)
        node_diff_k = head_projected.node_blocks - target_projected.node_blocks
        edge_diff_k = head_projected.edge_blocks - target_projected.edge_blocks
        node_stats_k = owner._metric_stats(
            node_diff_k, node_mask, owner.loss_type, owner._time_weight(ctx.node_t)
        )
        edge_stats_k = owner._metric_stats(
            edge_diff_k, edge_mask, owner.loss_type, owner._time_weight(ctx.edge_t)
        )
        total_k = owner._reduce_component_stats(
            ((node_stats_k, owner.node_weight), (edge_stats_k, owner.edge_weight))
        )
        per_head_losses.append(total_k)

    committee_loss = torch.stack(per_head_losses).mean()
    state["train_committee_aux_loss"] = committee_loss.detach()
    state["train_committee_num_extra_heads"] = torch.as_tensor(
        num_extra_heads, device=committee_loss.device
    )
    total = total + weight * committee_loss
    return total, state

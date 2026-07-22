"""Per-expert training objectives for :class:`MultiTrainer`.

``MultiTrainer._run_one_expert_loss`` runs one expert's forward + loss and can
take two very different routes: a **standard** supervised route (model forward
then criterion) and a **flow / CFM** route (conditional flow-matching, with an
endpoint-triplet reconstruction in the criterion's metric space). This module
splits those two routes into ``Objective`` (standard) and ``FlowObjective``
(flow), plus the ``Objective.build_train_payload`` aggregation that stitches a
main batch with an optional reference batch.

Both are *behaviour objects* holding a trainer back-reference (``self._t``) and
reaching into the trainer for the tagger, model, ``flow_cfm``, ``iter``, dtype,
and the flow-state helpers. State lives on the trainer; nothing is duplicated.
They are lazily attached (``MultiTrainer._objective`` / ``._flow_objective``) so
trainers built via ``object.__new__`` in unit tests get them without ``__init__``.

Contract preserved verbatim
---------------------------
* ``MultiTrainer._run_one_expert_loss`` keeps its signature and does the common
  prep (masks, ``batch_copy``, active-node/edge counts, ``flow_enabled``
  decision) then dispatches to one of these objectives. It stays monkeypatchable
  — several tests replace ``trainer._run_one_expert_loss`` wholesale.
* ``Objective.build_train_payload`` calls ``self._t._run_one_expert_loss(...)``
  (never a private objective method) so that a monkeypatched
  ``_run_one_expert_loss`` is still what the payload builder invokes, and the
  reference call keeps passing ``use_flow=flow_cfm.apply_to_reference``.

The tagger context names, ``cuda_cache_memory_context`` args, error text, and
metric keys are moved unchanged; pinned by ``test_hamiltonian_flow.py``.
"""

from typing import Any, Dict

import torch

from dptb.utils.cuda_cache_memory import cuda_cache_memory_context
from dptb.nnops.trainer import Trainer


class Objective:
    """Standard supervised per-expert objective (model forward -> criterion)."""

    def __init__(self, trainer):
        self._t = trainer

    def gated_metric_weighted_sum(self, value, active_count):
        """Return a metric numerator and its jointly-gated denominator.

        ``last_onsite_loss`` / ``last_hopping_loss`` are intentionally ``None``
        on cadence-throttled batches.  Such a batch must contribute neither a
        zero-valued numerator nor its active-count denominator; otherwise the
        reported epoch metric is diluted.  Raw active counts remain available
        separately for telemetry.
        """
        t = self._t
        if torch.is_tensor(active_count):
            active_f = active_count.to(device=t.device, dtype=t.dtype)
        else:
            active_f = torch.tensor(
                float(active_count), device=t.device, dtype=t.dtype
            )
        if value is None:
            zero = torch.zeros((), device=t.device, dtype=t.dtype)
            return zero, zero
        return t._as_scalar_tensor(value, default=0.0) * active_f, active_f

    def run(
        self,
        *,
        batch_copy,
        batch_info,
        criterion,
        expert_idx,
        active_nodes,
        active_edges,
        capture_metrics,
    ) -> Dict[str, Any]:
        t = self._t
        with t._tagger.tag("expert/model_forward", it=t.iter, expert=expert_idx):
            with cuda_cache_memory_context(
                iteration=t.iter,
                stage="expert/model_forward",
                expert=expert_idx,
            ):
                pred_batch = t.model(batch_copy)

        pred_batch["global_step"] = int(t.iter)

        with t._tagger.tag("expert/attach_batch_info", it=t.iter, expert=expert_idx):
            pred_batch.update(batch_info)
            batch_for_loss = batch_copy.copy()
            batch_for_loss.update(batch_info)

        with t._tagger.tag("expert/loss_forward", it=t.iter, expert=expert_idx):
            loss = criterion(pred_batch, batch_for_loss)

        out = {
            "loss": loss,
            "active_nodes": active_nodes,
            "active_edges": active_edges,
        }
        if capture_metrics:
            out.update(t._snapshot_loss_metrics(criterion))
        return out

    def build_train_payload(
        self,
        batch_dict,
        batch_info,
        expert_idx,
        range_dis,
        ref_batch_dict=None,
        ref_batch_info=None,
        criterion=None,
        flow_prefix="train",
    ) -> Dict[str, Any]:
        t = self._t
        if criterion is None:
            criterion = t.train_lossfunc

        main = t._run_one_expert_loss(
            batch_dict=batch_dict,
            batch_info=batch_info,
            criterion=criterion,
            expert_idx=expert_idx,
            range_dis=range_dis,
            capture_metrics=True,
            flow_prefix=flow_prefix,
        )

        total_loss = main["loss"]
        active_nodes = main["active_nodes"]
        active_edges = main["active_edges"]

        onsite_weighted_sum, onsite_weight = self.gated_metric_weighted_sum(
            main["onsite"], active_nodes
        )
        hopping_weighted_sum, hopping_weight = self.gated_metric_weighted_sum(
            main["hopping"], active_edges
        )

        onsite_l1_sum = main["last_onsite_l1_sum"]
        onsite_mse_sum = main["last_onsite_mse_sum"]
        onsite_cnt = main["last_onsite_count"]
        hopping_l1_sum = main["last_hopping_l1_sum"]
        hopping_mse_sum = main["last_hopping_mse_sum"]
        hopping_cnt = main["last_hopping_count"]

        z_values = []
        load_cv_values = []
        if main["z_loss"] is not None:
            z_values.append(main["z_loss"])
        if main["expert_load_cv"] is not None:
            load_cv_values.append(main["expert_load_cv"])

        if ref_batch_dict is not None:
            reference_criterion = getattr(t, "reference_lossfunc", criterion)
            ref_res = t._run_one_expert_loss(
                batch_dict=ref_batch_dict,
                batch_info=ref_batch_info,
                criterion=reference_criterion,
                expert_idx=expert_idx,
                range_dis=range_dis,
                capture_metrics=True,
                flow_prefix=flow_prefix,
                use_flow=bool(
                    getattr(getattr(t, "flow_cfm", None), "apply_to_reference", False)
                ),
            )

            total_loss = total_loss + ref_res["loss"]
            # Match the single Trainer contract: reference supervision affects
            # the backward objective but never contaminates the main-batch
            # endpoint triplet. Reference metrics can be reported separately
            # by a future dedicated namespace.

        expert_onsite = onsite_weighted_sum / onsite_weight.clamp_min(1.0)
        expert_hopping = hopping_weighted_sum / hopping_weight.clamp_min(1.0)

        return {
            "loss": total_loss,
            "expert_onsite": expert_onsite.detach(),
            "expert_hopping": expert_hopping.detach(),
            "onsite_weighted_sum": onsite_weighted_sum.detach(),
            "hopping_weighted_sum": hopping_weighted_sum.detach(),
            # These are metric denominators, not raw telemetry.  They equal the
            # active counts on a contributing batch and zero on a throttled one.
            "onsite_weight": onsite_weight.detach(),
            "hopping_weight": hopping_weight.detach(),
            "active_nodes": active_nodes.detach(),
            "active_edges": active_edges.detach(),
            "onsite_l1_sum": onsite_l1_sum.detach() if torch.is_tensor(onsite_l1_sum) else None,
            "onsite_mse_sum": onsite_mse_sum.detach() if torch.is_tensor(onsite_mse_sum) else None,
            "onsite_cnt": onsite_cnt.detach() if torch.is_tensor(onsite_cnt) else None,
            "hopping_l1_sum": hopping_l1_sum.detach() if torch.is_tensor(hopping_l1_sum) else None,
            "hopping_mse_sum": hopping_mse_sum.detach() if torch.is_tensor(hopping_mse_sum) else None,
            "hopping_cnt": hopping_cnt.detach() if torch.is_tensor(hopping_cnt) else None,
            "z_values": [z.detach() for z in z_values],
            "load_cv_values": [cv.detach() for cv in load_cv_values],
        }


class FlowObjective:
    """Conditional-flow-matching per-expert objective (the ``flow_enabled`` route)."""

    def __init__(self, trainer):
        self._t = trainer

    def run(
        self,
        *,
        batch_copy,
        batch_info,
        criterion,
        expert_idx,
        expert_edge_mask,
        expert_node_mask,
        active_nodes,
        active_edges,
        flow_prefix,
    ) -> Dict[str, Any]:
        t = self._t
        batch_for_loss = batch_copy.copy()
        if getattr(t.flow_cfm, "model_in_loss", False):
            with t._tagger.tag("expert/flow_loss_with_model", it=t.iter, expert=expert_idx):
                loss, flow_state = t.flow_cfm.loss_with_model(
                    t.model,
                    batch_copy,
                    batch_for_loss,
                )
            flow_state = t._flow_state_with_prefix(flow_state, flow_prefix)
            flow_state.setdefault(f"{flow_prefix}_loss_opt", loss.detach())
            compatible_prefix = f"{flow_prefix}_compatible"
            compatible_state = Trainer._compatible_loss_state_from_flow_stats(
                criterion,
                flow_state,
                source_prefix=flow_prefix,
                prefix=compatible_prefix,
                legacy_prefix=flow_prefix,
                global_step=t.iter,
                # model_in_loss branch: unlike the `else` branch below, there is
                # no raw-batch criterion-recompute fallback here, so a
                # metric_space mismatch must fail fast with a diagnostic instead
                # of silently falling through to the generic RuntimeError at the
                # bottom of this method (P1-1).
                fail_on_metric_space_mismatch=True,
            )
            if compatible_state is not None:
                flow_state.update(compatible_state)
        else:
            with t._tagger.tag("expert/flow_prepare_batch", it=t.iter, expert=expert_idx):
                flow_batch, flow_ref, flow_ctx = t.flow_cfm.prepare_batch(
                    batch_copy,
                    batch_for_loss,
                )

            with t._tagger.tag("expert/model_forward", it=t.iter, expert=expert_idx):
                with cuda_cache_memory_context(
                    iteration=t.iter,
                    stage="expert/model_forward",
                    expert=expert_idx,
                ):
                    pred_batch = t.model(flow_batch)

            pred_batch["global_step"] = int(t.iter)
            pred_batch.setdefault("expert_edge_mask", expert_edge_mask)
            pred_batch.setdefault("expert_node_mask", expert_node_mask)
            pred_batch.setdefault("expert_idx", int(expert_idx))
            pred_batch.update(batch_info)
            flow_ref.update(batch_info)

            with t._tagger.tag("expert/flow_loss", it=t.iter, expert=expert_idx):
                loss, flow_state = t.flow_cfm.loss(pred_batch, flow_ref, flow_ctx)

            flow_state = t._flow_state_with_prefix(flow_state, flow_prefix)
            flow_state.setdefault(f"{flow_prefix}_loss_opt", loss.detach())
            compatible_prefix = f"{flow_prefix}_compatible"
            compatible_state = Trainer._compatible_loss_state_from_flow_stats(
                criterion,
                flow_state,
                source_prefix=flow_prefix,
                prefix=compatible_prefix,
                legacy_prefix=flow_prefix,
                global_step=t.iter,
            )
            if compatible_state is None:
                compatible_state = Trainer._compatible_loss_state(
                    criterion,
                    pred_batch,
                    flow_ref,
                    prefix=compatible_prefix,
                    legacy_prefix=flow_prefix,
                    include_raw_stats=True,
                )
                fallback_stats = compatible_state.pop("_endpoint_stats", None)
                if fallback_stats is not None:
                    flow_state["_compatible_clean_stats"] = fallback_stats
            if compatible_state is not None:
                flow_state.update(compatible_state)

        if compatible_state is None:
            raise RuntimeError(
                "Enabled flow could not reconstruct an endpoint triplet in "
                "the criterion's metric space. Check that flow target keys "
                "and the configured Hamiltonian loss use the same block/RME "
                "representation."
            )
        Trainer._require_endpoint_triplet(
            flow_state,
            prefix=flow_prefix,
            route="MultiTrainer flow training",
        )

        out = {
            "loss": loss,
            "active_nodes": active_nodes,
            "active_edges": active_edges,
        }
        out.update(t._payload_metrics_from_flow_state(flow_state, prefix=flow_prefix))
        flow_state.pop("_compatible_clean_stats", None)
        return out

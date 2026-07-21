"""Pure metric-reduction math over :mod:`dptb.nnops.metric_pack` typed packs.

``MultiTrainer`` reduces per-step training metrics into human-facing loss
numbers and display-window state dicts. Historically that arithmetic lived
inline in ``MultiTrainer`` as a set of methods reaching into ``self`` for
``dtype`` / ``device`` / the resolved loss module. This module lifts the
*numerical core* out into stateless static functions so it can be unit-tested
in isolation and reasoned about without the trainer's distributed machinery.

Design contract
---------------
* **Stateless.** Every function is a ``@staticmethod`` taking ``dtype`` /
  ``device`` (and any loss-module-derived inputs) explicitly. Nothing here
  reads ``self``. This is deliberate: ``MultiTrainer`` unit tests build trainers
  via ``object.__new__`` (no ``__init__``), so a reducer *instance* stored on the
  trainer would not exist in those paths. Static functions sidestep that.
* **No collectives.** These functions run *after* ``all_reduce`` / ``all_gather``
  have already produced the reduced/gathered tensors. They never touch
  ``torch.distributed``. Collective ordering stays entirely in ``MultiTrainer``.
* **Byte-for-byte behaviour.** The arithmetic is a verbatim move of the former
  ``MultiTrainer`` method bodies; results are identical to the pre-refactor code.
  Pinned by ``dptb/tests/test_metric_reducer.py`` plus the existing
  ``test_hamiltonian_flow`` compatible-pack / display-window regressions that
  drive these paths through the trainer delegators.

The trainer keeps thin delegating methods (``_compute_compatible_state_from_pack``
etc.) that resolve the loss module + endpoint-triplet capability from a criterion
and forward to the functions here, preserving their public signatures (several
are monkeypatched or called directly by tests).
"""

from typing import Any, Callable, Dict, List, Optional

import torch

from dptb.nnops.metric_pack import (
    MetricPack,
    DynamicBatchStat,
    ExpertDisplayMetric,
)


class MetricReducer:
    """Stateless namespace of pure pack-reduction functions."""

    @staticmethod
    def maybe_call_or_value(x, default: float = 1.0) -> float:
        """Coerce ``x`` (value, callable, or None) to a float, tolerating errors."""
        if x is None:
            return float(default)
        if callable(x):
            try:
                return float(x())
            except Exception:
                return float(default)
        try:
            return float(x)
        except Exception:
            return float(default)

    @staticmethod
    def safe_mean(sum_t, cnt_t, *, dtype, device):
        """``sum_t / max(cnt_t, 1)`` with a zero fallback when either is ``None``."""
        if sum_t is None or cnt_t is None:
            return torch.zeros((), dtype=dtype, device=device)
        return sum_t / cnt_t.to(dtype=dtype).clamp_min(1.0)

    @staticmethod
    def compatible_state_from_pack(
        pack: torch.Tensor,
        *,
        loss_module,
        supports_triplet: bool,
        dtype,
        device,
        prefix: str = "train",
        global_step=None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Reduce a :class:`MetricPack` tensor into a compatible loss triplet.

        Returns ``{prefix_loss, prefix_onsite_loss, prefix_hopping_loss}`` or
        ``None`` when the criterion cannot express an endpoint triplet / the pack
        carries no active nodes or edges. ``loss_module`` is the already-resolved
        innermost loss ``nn.Module``; ``supports_triplet`` mirrors
        ``Trainer._supports_endpoint_triplet(criterion)``.
        """
        if not supports_triplet:
            return None

        mp = MetricPack.from_tensor(pack)
        onsite_cnt = mp.onsite_cnt_sum
        hopping_cnt = mp.hopping_cnt_sum
        reduce_from_stats = getattr(loss_module, "compatible_loss_from_stats", None)

        def _safe_mean(sum_t, cnt_t):
            return sum_t / cnt_t.to(dtype=dtype).clamp_min(1.0)

        if float(onsite_cnt.item()) > 0.0 or float(hopping_cnt.item()) > 0.0:
            if callable(reduce_from_stats):
                z_loss = None
                if float(mp.z_cnt.item()) > 0.0:
                    z_loss = mp.z_sum / mp.z_cnt.clamp_min(1.0)
                total, _onsite_loss, _hopping_loss = reduce_from_stats(
                    onsite_l1_sum=mp.onsite_l1_sum,
                    onsite_mse_sum=mp.onsite_mse_sum,
                    onsite_count=onsite_cnt,
                    hopping_l1_sum=mp.hopping_l1_sum,
                    hopping_mse_sum=mp.hopping_mse_sum,
                    hopping_count=hopping_cnt,
                    z_loss=z_loss,
                    global_step=global_step,
                )
                return {
                    f"{prefix}_loss": total.detach(),
                    f"{prefix}_onsite_loss": _onsite_loss.detach(),
                    f"{prefix}_hopping_loss": _hopping_loss.detach(),
                }

            onsite_l1_mean = _safe_mean(mp.onsite_l1_sum, onsite_cnt)
            onsite_mse_mean = _safe_mean(mp.onsite_mse_sum, onsite_cnt)
            hopping_l1_mean = _safe_mean(mp.hopping_l1_sum, hopping_cnt)
            hopping_mse_mean = _safe_mean(mp.hopping_mse_sum, hopping_cnt)

            onsite_loss = 0.5 * (onsite_l1_mean + torch.sqrt(onsite_mse_mean))
            hopping_loss = 0.5 * (hopping_l1_mean + torch.sqrt(hopping_mse_mean))
        else:
            active_nodes = mp.active_nodes_sum
            active_edges = mp.active_edges_sum
            if float(active_nodes.item()) <= 0.0 and float(active_edges.item()) <= 0.0:
                return None
            onsite_loss = _safe_mean(mp.onsite_weighted_sum, active_nodes)
            hopping_loss = _safe_mean(mp.hopping_weighted_sum, active_edges)

        onsite_boost = bool(getattr(loss_module, "onsite_boost", False))
        onsite_boost_w = MetricReducer.maybe_call_or_value(
            getattr(loss_module, "_current_onsite_weight", None), default=1.0
        )
        z_coef = float(getattr(loss_module, "z_loss_coef", 0.0))

        if onsite_boost:
            total = onsite_boost_w * onsite_loss + hopping_loss
        else:
            total = 0.5 * (onsite_loss + hopping_loss)

        if z_coef > 0.0 and float(mp.z_cnt.item()) > 0.0:
            total = total + z_coef * (mp.z_sum / mp.z_cnt.clamp_min(1.0))

        return {
            f"{prefix}_loss": total.detach(),
            f"{prefix}_onsite_loss": onsite_loss.detach(),
            f"{prefix}_hopping_loss": hopping_loss.detach(),
        }

    @staticmethod
    def component_state_from_pack(
        pack: torch.Tensor,
        *,
        loss_module,
        supports_triplet: bool,
        dtype,
        device,
        prefix: str,
        global_step=None,
    ) -> Dict[str, torch.Tensor]:
        """Per-component (onsite/hopping) display tags for a reduced pack.

        Uses the compatible-stats semantics when available, otherwise falls back
        to the active-node/edge weighted means carried in the pack. Returns an
        empty dict when the criterion cannot express an endpoint triplet.
        """
        if not supports_triplet:
            return {}
        compatible_state = MetricReducer.compatible_state_from_pack(
            pack,
            loss_module=loss_module,
            supports_triplet=supports_triplet,
            dtype=dtype,
            device=device,
            prefix=prefix,
            global_step=global_step,
        )
        if compatible_state is not None:
            return {
                f"{prefix}_onsite_loss": compatible_state[f"{prefix}_onsite_loss"],
                f"{prefix}_hopping_loss": compatible_state[f"{prefix}_hopping_loss"],
            }

        mp = MetricPack.from_tensor(pack)
        if (
            float(mp.active_nodes_sum.item()) <= 0.0
            and float(mp.active_edges_sum.item()) <= 0.0
            and float(mp.onsite_weighted_sum.item()) == 0.0
            and float(mp.hopping_weighted_sum.item()) == 0.0
        ):
            # A cadence-skipped pack carries 0/0 for both components. Preserve
            # absence instead of publishing a valid-looking zero metric.
            return {}
        onsite_cnt = mp.active_nodes_sum.to(dtype=dtype).clamp_min(1.0)
        hopping_cnt = mp.active_edges_sum.to(dtype=dtype).clamp_min(1.0)
        return {
            f"{prefix}_onsite_loss": (mp.onsite_weighted_sum / onsite_cnt).detach(),
            f"{prefix}_hopping_loss": (mp.hopping_weighted_sum / hopping_cnt).detach(),
        }

    @staticmethod
    def stitched_loss_reduce(
        payloads: List[Dict[str, Any]],
        *,
        loss_module,
        dtype,
        device,
        global_step,
    ) -> Optional[torch.Tensor]:
        """Stitch per-expert payload stats into one scalar objective.

        Aggregates the l1/mse/count sums across ``payloads`` and reduces them via
        the loss module's ``compatible_loss_from_stats`` when available, else via
        the RMS-of-means fallback. Returns ``None`` when no payload carried stats.
        The caller (``MultiTrainer._compute_stitched_loss_by_reduce``) owns the
        ``endpoint_loss_mode == "reduce"`` gate.
        """
        onsite_l1_sum = None
        onsite_mse_sum = None
        onsite_cnt = None
        hopping_l1_sum = None
        hopping_mse_sum = None
        hopping_cnt = None
        z_vals = []

        for p in payloads:
            if p is None:
                continue

            if p.get("onsite_l1_sum") is not None:
                onsite_l1_sum = p["onsite_l1_sum"] if onsite_l1_sum is None else (onsite_l1_sum + p["onsite_l1_sum"])
                onsite_mse_sum = p["onsite_mse_sum"] if onsite_mse_sum is None else (onsite_mse_sum + p["onsite_mse_sum"])
                onsite_cnt = p["onsite_cnt"] if onsite_cnt is None else (onsite_cnt + p["onsite_cnt"])

            if p.get("hopping_l1_sum") is not None:
                hopping_l1_sum = p["hopping_l1_sum"] if hopping_l1_sum is None else (hopping_l1_sum + p["hopping_l1_sum"])
                hopping_mse_sum = p["hopping_mse_sum"] if hopping_mse_sum is None else (hopping_mse_sum + p["hopping_mse_sum"])
                hopping_cnt = p["hopping_cnt"] if hopping_cnt is None else (hopping_cnt + p["hopping_cnt"])

            for z in p.get("z_values", []):
                if z is not None:
                    z_vals.append(z)

        if onsite_cnt is None and hopping_cnt is None:
            return None

        reduce_from_stats = getattr(loss_module, "compatible_loss_from_stats", None)
        if callable(reduce_from_stats) and all(
            value is not None
            for value in (
                onsite_l1_sum,
                onsite_mse_sum,
                onsite_cnt,
                hopping_l1_sum,
                hopping_mse_sum,
                hopping_cnt,
            )
        ):
            z_loss = None
            if z_vals:
                z_loss = torch.stack([z.to(device, dtype=dtype) for z in z_vals]).mean()
            total, _onsite_loss, _hopping_loss = reduce_from_stats(
                onsite_l1_sum=onsite_l1_sum,
                onsite_mse_sum=onsite_mse_sum,
                onsite_count=onsite_cnt,
                hopping_l1_sum=hopping_l1_sum,
                hopping_mse_sum=hopping_mse_sum,
                hopping_count=hopping_cnt,
                z_loss=z_loss,
                global_step=global_step,
            )
            return total.detach()

        def _safe_mean(sum_t, cnt_t):
            if sum_t is None or cnt_t is None:
                return torch.zeros((), dtype=dtype, device=device)
            return sum_t / cnt_t.to(dtype=dtype).clamp_min(1.0)

        onsite_l1_mean = _safe_mean(onsite_l1_sum, onsite_cnt)
        onsite_mse_mean = _safe_mean(onsite_mse_sum, onsite_cnt)
        hopping_l1_mean = _safe_mean(hopping_l1_sum, hopping_cnt)
        hopping_mse_mean = _safe_mean(hopping_mse_sum, hopping_cnt)

        onsite_loss = 0.5 * (onsite_l1_mean + torch.sqrt(onsite_mse_mean))
        hopping_loss = 0.5 * (hopping_l1_mean + torch.sqrt(hopping_mse_mean))

        onsite_boost = bool(getattr(loss_module, "onsite_boost", False))
        onsite_boost_w = MetricReducer.maybe_call_or_value(
            getattr(loss_module, "_current_onsite_weight", None), default=1.0
        )
        z_coef = float(getattr(loss_module, "z_loss_coef", 0.0))

        if onsite_boost:
            total = onsite_boost_w * onsite_loss + hopping_loss
        else:
            total = 0.5 * (onsite_loss + hopping_loss)

        if z_coef > 0.0 and len(z_vals) > 0:
            z_mean = torch.stack([z.to(device, dtype=dtype) for z in z_vals]).mean()
            total = total + z_coef * z_mean

        return total.detach()

    @staticmethod
    def display_state_from_packs(
        reduced_pack: torch.Tensor,
        reduced_db_pack: torch.Tensor,
        gathered: List[torch.Tensor],
        *,
        total_steps: float,
        num_experts: int,
        rank_to_expert_idx: Callable[[int], int],
        train_loss_module,
        supports_triplet: bool,
        dtype,
        device,
        time_idx: int,
        gathered_fired_counts: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Assemble the display-window ``state`` dict from reduced/gathered packs.

        Purely arithmetic: consumes the *already reduced* metric/dynamic-batch
        packs and the *already gathered* per-rank expert display vectors. The
        caller owns the collectives, the zero-successful-steps guard, the buffer
        reset, and appending optimizer/CUDA diagnostics.
        """
        reduced_mp = MetricPack.from_tensor(reduced_pack)
        reduced_db = DynamicBatchStat.from_tensor(reduced_db_pack)
        dynamic_batch_oom_skips = int(round(float(reduced_db.oom_skipped_count.item())))

        train_loss_opt_mean = reduced_mp.loss_opt_sum / total_steps
        compatible_train_state = MetricReducer.compatible_state_from_pack(
            reduced_pack,
            loss_module=train_loss_module,
            supports_triplet=supports_triplet,
            dtype=dtype,
            device=device,
            prefix="train",
            global_step=time_idx,
        )
        train_loss_show = (
            compatible_train_state["train_loss"]
            if compatible_train_state is not None
            else train_loss_opt_mean
        )

        total_grad_norm_mean = reduced_mp.grad_norm_sum / reduced_mp.step_count.clamp_min(1.0)

        avg_lr = sum(
            float(ExpertDisplayMetric.from_tensor(vec).lr.item()) for vec in gathered
        ) / max(len(gathered), 1)

        state = {
            'field': 'iteration',
            'window_steps': int(round(total_steps)),
            "train_loss": train_loss_show.detach() if torch.is_tensor(train_loss_show) else torch.tensor(float(train_loss_show), device=device, dtype=dtype),
            "train_loss_opt": train_loss_opt_mean.detach() if torch.is_tensor(train_loss_opt_mean) else torch.tensor(float(train_loss_opt_mean), device=device, dtype=dtype),
            "lr": float(avg_lr),
            "total_grad_norm": float(total_grad_norm_mean.item()),
        }
        if compatible_train_state is not None:
            state.update({
                "train_onsite_loss": float(
                    compatible_train_state["train_onsite_loss"].item()
                ),
                "train_hopping_loss": float(
                    compatible_train_state["train_hopping_loss"].item()
                ),
            })

        expert_metrics: Dict[int, List[torch.Tensor]] = {}
        expert_fired_counts: Dict[int, List[torch.Tensor]] = {}
        for rank_idx, vec in enumerate(gathered):
            expert_idx = rank_to_expert_idx(rank_idx)
            expert_metrics.setdefault(expert_idx, []).append(vec)
            if gathered_fired_counts is not None:
                if rank_idx >= len(gathered_fired_counts):
                    raise ValueError(
                        "Expert fired-count sidecar must have one entry per gathered rank."
                    )
                counts = gathered_fired_counts[rank_idx]
                if counts.numel() != 2:
                    raise ValueError(
                        "Expert fired-count sidecar entries must contain exactly "
                        "[onsite_count, hopping_count]."
                    )
                expert_fired_counts.setdefault(expert_idx, []).append(counts.reshape(2))

        if (
            gathered_fired_counts is not None
            and len(gathered_fired_counts) != len(gathered)
        ):
            raise ValueError(
                "Expert fired-count sidecar must have one entry per gathered rank."
            )

        for expert_idx in range(num_experts):
            vecs = expert_metrics.get(expert_idx, [])
            if not vecs:
                continue
            stacked = torch.stack(vecs)
            mean_metric = ExpertDisplayMetric.from_tensor(stacked.mean(dim=0))
            if supports_triplet:
                onsite_value = mean_metric.expert_onsite
                hopping_value = mean_metric.expert_hopping
                count_vecs = expert_fired_counts.get(expert_idx, [])
                if count_vecs:
                    counts = torch.stack(count_vecs).to(
                        dtype=stacked.dtype, device=stacked.device
                    )
                    onsite_count = counts[:, 0].sum()
                    hopping_count = counts[:, 1].sum()
                    onsite_value = torch.where(
                        onsite_count > 0,
                        (stacked[:, 0] * counts[:, 0]).sum()
                        / onsite_count.clamp_min(1.0),
                        stacked[:, 0].new_zeros(()),
                    )
                    hopping_value = torch.where(
                        hopping_count > 0,
                        (stacked[:, 1] * counts[:, 1]).sum()
                        / hopping_count.clamp_min(1.0),
                        stacked[:, 1].new_zeros(()),
                    )
                state[f"expert_{expert_idx}_onsite"] = float(onsite_value.item())
                state[f"expert_{expert_idx}_hopping"] = float(hopping_value.item())
            state[f"expert_{expert_idx}_lr"] = float(mean_metric.lr.item())
            state[f"expert_{expert_idx}_active_nodes"] = float(mean_metric.active_nodes.item())
            state[f"expert_{expert_idx}_active_edges"] = float(mean_metric.active_edges.item())

        if float(reduced_mp.cv_cnt.item()) > 0.0:
            state["expert_load_cv"] = float((reduced_mp.cv_sum / reduced_mp.cv_cnt).item())
        if float(reduced_mp.z_cnt.item()) > 0.0:
            state["mean_max_prob"] = float((reduced_mp.z_sum / reduced_mp.z_cnt).item())

        dynamic_batch_steps = float(reduced_db.step_count.item())
        if dynamic_batch_steps > 0.0:
            denom = reduced_db.step_count.clamp_min(1.0)
            state["batch_num_graphs"] = float((reduced_db.num_graphs_sum / denom).item())
            state["batch_cost"] = float((reduced_db.cost_sum / denom).item())
            state["batch_num_nodes"] = float((reduced_db.num_nodes_sum / denom).item())
            state["batch_num_edges"] = float((reduced_db.num_edges_sum / denom).item())
            state["batch_max_item_cost"] = float((reduced_db.max_item_cost_sum / denom).item())
        if dynamic_batch_oom_skips > 0:
            state["dynamic_batch_oom_skipped_iters"] = dynamic_batch_oom_skips

        return state

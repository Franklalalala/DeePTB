from dptb.nn.shift_head import optimizer_named_parameters
import torch
import logging
import os
import csv
import math
import copy
import json
import torch.nn as nn
from dptb.configuration import migrate_legacy_checkpoint_train_options
from dptb.utils.tools import (
    get_lr_scheduler,
    get_optimizer,
    lr_scheduler_can_step_without_metric,
    lr_scheduler_requires_metric,
)
from dptb.nnops.base_trainer import BaseTrainer
from dptb.nnops.ddp_utils import merge_restart_train_options
from dptb.plugins.monitor import Plugin
from typing import Union, Optional
from dptb.data import AtomicDataset, DataLoader, AtomicData
from dptb.nn import build_model
from dptb.nn.activation_recompute import configure_activation_recompute
from dptb.nnops.loss import Loss
from dptb.nnops.prior_noise import PriorNoiseAugmentation, assert_prior_noise_keys_reach_model
from dptb.nnops.self_consistency import (
    SelfConsistencyScheduler,
    SelfConsistencySchedulerConfig,
    compute_self_consistency_payload_loss,
)
from dptb.nnops.training_state import (
    CHECKPOINT_KIND_EPOCH,
    CHECKPOINT_KIND_ITERATION,
    preflight_restart_checkpoint,
    read_resume_metadata,
    resolve_rank_rng_state,
    restore_rng_state,
    validate_checkpoint_invariants,
    validate_checkpoint_world_size,
)

log = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    # SingleTrainer historically scheduled non-flow runs on the optimization
    # objective.  Public endpoint metrics must not silently change that signal.
    scheduler_metric_prefers_objective = True
    object_keys = ["lr_scheduler", "optimizer"]

    def __init__(
            self,
            train_options: dict,
            common_options: dict,
            model: torch.nn.Module,
            train_datasets: AtomicDataset,
            reference_datasets: Union[AtomicDataset, None] = None,
            validation_datasets: Union[AtomicDataset, None] = None,
    ) -> None:
        self._configure_self_consistency(train_options.get("self_consistency", {}) or {})

        super(Trainer, self).__init__(dtype=common_options["dtype"], device=common_options["device"])

        # init the object
        self.model = model.to(self.device)
        self.num_experts = int(getattr(self.model, "num_experts", 0) or 0)
        self.activation_recompute_state = configure_activation_recompute(
            self.model,
            train_options.get("activation_recompute", None),
        )
        self.optimizer = get_optimizer(model_param=optimizer_named_parameters(self.model), **train_options["optimizer"])
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, **train_options["lr_scheduler"])
        self.update_lr_per_iter = train_options["update_lr_per_iter"]
        self.common_options = common_options
        self.train_options = train_options
        self.optimizer_diagnostics_freq = max(int(train_options.get("display_freq", 1)), 1)

        # ============================================================
        # [修改 1] 初始化 Clip 阈值
        # 如果 options 里没写，默认为 inf (只计算 norm，不截断)
        # ============================================================
        self.clip_grad_norm = train_options.get("clip_grad", float('inf'))

        if self.clip_grad_norm == float('inf'):
            log.info("ℹ️ Gradient Clipping is OFF (Monitoring mode: threshold set to inf)")
        else:
            log.info(f"✂️ Gradient Clipping is ON (Threshold: {self.clip_grad_norm})")

        self.train_datasets = train_datasets
        # ... (原有 task 判断逻辑保持不变) ...
        self.task = None
        if self.train_datasets.get_Hamiltonian:
            self.task = "hamiltonians"
        elif self.train_datasets.get_DM:
            self.task = "DM"
        else:
            self.task = "eigenvalues"

        self.use_reference = False
        if reference_datasets is not None:
            self.reference_datesets = reference_datasets
            self.use_reference = True

        if validation_datasets is not None:
            self.validation_datasets = validation_datasets
            self.use_validation = True
        else:
            self.use_validation = False

        self.train_loader = DataLoader(
            dataset=self.train_datasets,
            batch_size=train_options["batch_size"],
            shuffle=True,
            dynamic_batch=train_options.get("dynamic_batch", None),
            **self._train_loader_worker_kwargs(train_options),
        )

        if self.use_reference:
            self.reference_loader = DataLoader(dataset=self.reference_datesets,
                                               batch_size=train_options["ref_batch_size"], shuffle=True)

        if self.use_validation:
            self.validation_loader_seed = int(common_options.get("seed", 0)) & (
                (1 << 64) - 1
            )
            self.validation_loader_generator = torch.Generator().manual_seed(
                self.validation_loader_seed
            )
            self.validation_loader = DataLoader(dataset=self.validation_datasets,
                                                batch_size=train_options["val_batch_size"],
                                                shuffle=True,
                                                generator=self.validation_loader_generator)

        loss_idp = self._model_loss_idp()

        # loss function
        self.train_lossfunc = Loss(
            **self._loss_kwargs(train_options["loss_options"]["train"], common_options),
            idp=loss_idp,
        )
        if self.use_validation:
            self.validation_lossfunc = Loss(
                **self._loss_kwargs(train_options["loss_options"]["validation"], common_options),
                idp=loss_idp,
            )
        if self.use_reference:
            self.reference_lossfunc = Loss(
                **self._loss_kwargs(train_options["loss_options"]["reference"], common_options),
                idp=loss_idp,
            )

        self.endpoint_metric_spaces = {}
        if train_options.get("symmetry_projection", {}).get("enabled", False):
            raise ValueError("Symmetry-projected training requires an archived model")

        criteria = {"train": self.train_lossfunc}
        if self.use_validation:
            criteria["validation"] = self.validation_lossfunc
        if self.use_reference:
            criteria["reference"] = self.reference_lossfunc
        for name, criterion in criteria.items():
            if not self._supports_endpoint_triplet(criterion):
                continue
            loss_obj = self._loss_component_source(criterion)
            metric_space = str(
                getattr(loss_obj, "endpoint_metric_space", "rme")
            ).lower()
            self.endpoint_metric_spaces[name] = metric_space
            log.info("%s endpoint metric space: %s", name, metric_space)
        if (
            "train" in self.endpoint_metric_spaces
            and "validation" in self.endpoint_metric_spaces
            and self.endpoint_metric_spaces["train"]
            != self.endpoint_metric_spaces["validation"]
        ):
            log.warning(
                "Train and validation endpoint metric spaces differ: %s != %s. "
                "Their common loss tags are not numerically comparable.",
                self.endpoint_metric_spaces["train"],
                self.endpoint_metric_spaces["validation"],
            )

        noise_options = dict(train_options.get("flow_options", None) or {})
        if noise_options.get("enabled", False):
            raise ValueError("Flow training requires an archived model")
        # External trainer extensions may inspect this optional legacy hook.
        self.flow_cfm = None
        self.prior_noise_augmentation = None
        if bool(train_options.get("prior_noise_augmentation", False)):
            self.prior_noise_augmentation = PriorNoiseAugmentation(
                noise_options, idp=getattr(self.train_lossfunc, "idp", loss_idp),
                dtype=self.dtype, device=self.device)
            assert_prior_noise_keys_reach_model(self.prior_noise_augmentation, self.model)
            log.info("prior_noise_augmentation: supervised objective, t = 0 prior draw prior=%s mode=%s ref=%s "
                     "sigma=%s (training batches only)", self.prior_noise_augmentation.prior,
                     self.prior_noise_augmentation.te_prior_mode,
                     self.prior_noise_augmentation.te_prior_scale_reference,
                     self.prior_noise_augmentation.te_prior_sigma)
        self._last_flow_state = {}
        self._last_flow_validation_state = {}
        self._last_self_consistency_state = {}

        if train_options["loss_options"]["train"]["method"] == "skints":
            assert self.model.name == 'nnsk', "The model should be nnsk for the skints loss function."
            assert self.model.onsite_fn.functype in ['none',
                                                     'uniform'], "The onsite function should be none or uniform for the skints loss function."
            log.info("The skints loss function is used for training, the model.transform is then set to False.")
            self.model.transform = False


    def _model_loss_idp(self):
        model = self.model
        hamiltonian = getattr(model, "hamiltonian", None)
        if hamiltonian is not None and getattr(hamiltonian, "idp", None) is not None:
            return hamiltonian.idp
        embedding = getattr(model, "embedding", None)
        if embedding is not None and getattr(embedding, "idp", None) is not None:
            return embedding.idp
        experts = getattr(model, "experts", None)
        if experts:
            first = experts[0]
            hamiltonian = getattr(first, "hamiltonian", None)
            if hamiltonian is not None and getattr(hamiltonian, "idp", None) is not None:
                return hamiltonian.idp
            embedding = getattr(first, "embedding", None)
            if embedding is not None and getattr(embedding, "idp", None) is not None:
                return embedding.idp
        raise AttributeError("Could not resolve OrbitalMapper idp from model hamiltonian or embedding.")

    @staticmethod
    def _loss_component_source(lossfunc):
        """Return the inner loss object that owns component side-effect fields."""
        loss_obj = lossfunc
        for attr in ("lossfunc", "loss_fn", "criterion", "method", "loss"):
            inner = getattr(loss_obj, attr, None)
            if isinstance(inner, nn.Module):
                loss_obj = inner
                break
        return loss_obj

    @staticmethod
    def _supports_endpoint_triplet(lossfunc) -> bool:
        """Whether a criterion participates in the Hamiltonian endpoint API."""

        loss_obj = Trainer._loss_component_source(lossfunc)
        declared = getattr(loss_obj, "supports_endpoint_triplet", None)
        if declared is not None:
            return bool(declared)
        if callable(getattr(loss_obj, "compatible_loss_from_stats", None)):
            return True
        # Backward compatibility for existing Hamiltonian criteria that
        # exposed component side effects before the capability marker existed.
        return all(
            hasattr(loss_obj, name)
            for name in ("last_onsite_loss", "last_hopping_loss")
        )


    @staticmethod
    def _loss_kwargs(loss_options, common_options):
        kwargs = dict(loss_options)
        kwargs.update(common_options)
        return kwargs

    @staticmethod
    def _dynamic_batch_state_from_batch(batch):
        state = {}
        for attr, key in (
            ("__dptb_batch_cost__", "batch_cost"),
            ("__dptb_batch_num_graphs__", "batch_num_graphs"),
            ("__dptb_batch_num_nodes__", "batch_num_nodes"),
            ("__dptb_batch_num_edges__", "batch_num_edges"),
            ("__dptb_batch_max_item_cost__", "batch_max_item_cost"),
        ):
            if hasattr(batch, attr):
                state[key] = getattr(batch, attr)
        return state

    def _configure_self_consistency(self, options):
        self.self_consistency_options = dict(options or {})
        self.self_consistency_enabled = bool(self.self_consistency_options.get("enabled", False))
        self.self_consistency_scheduler = None
        self.self_consistency_weight = float(self.self_consistency_options.get("weight", 0.1))
        self.self_consistency_tensor_keys = tuple(
            self.self_consistency_options.get("tensor_keys", ("node_features", "edge_features"))
        )
        self.self_consistency_sample_mode = str(
            self.self_consistency_options.get("sample_mode", "feature_tensors")
        )
        self.self_consistency_consume_timeout = float(
            self.self_consistency_options.get("consume_timeout", 0.0)
        )
        self._last_self_consistency_state = {}
        if not self.self_consistency_enabled:
            return

        repair_fn = self.self_consistency_options.get("repair_fn")
        if not callable(repair_fn):
            raise NotImplementedError(
                "train_options.self_consistency.enabled=true requires an explicit repair_fn "
                "until the ABACUS hrebuild block serializer is wired into Trainer. This avoids "
                "silently training without a real self-consistency target."
            )
        config = SelfConsistencySchedulerConfig(
            every_n_steps=int(self.self_consistency_options.get("every_n_steps", 100)),
            sample_frac=float(self.self_consistency_options.get("sample_frac", 0.1)),
            staleness_steps=int(self.self_consistency_options.get("staleness_steps", 1)),
            warmup_epochs=int(self.self_consistency_options.get("warmup_epochs", 0)),
            max_workers=int(self.self_consistency_options.get("max_workers", 2)),
            retry_unfinished=bool(self.self_consistency_options.get("retry_unfinished", True)),
        )
        self.self_consistency_scheduler = SelfConsistencyScheduler(repair_fn, config)

    def _self_consistency_current_samples(self, pred_data):
        if not getattr(self, "self_consistency_enabled", False) or not isinstance(pred_data, dict):
            return {}
        sample_mode = getattr(self, "self_consistency_sample_mode", "feature_tensors")
        if sample_mode in {"payload", "atomic_data", "batch"}:
            return {"batch": pred_data}
        samples = {}
        for key in self.self_consistency_tensor_keys:
            value = pred_data.get(key)
            if torch.is_tensor(value):
                samples[str(key)] = value
        return samples

    def _apply_self_consistency_loss(self, loss, pred_data):
        self._last_self_consistency_state = {}
        if (
            not getattr(self, "self_consistency_enabled", False)
            or getattr(self, "self_consistency_scheduler", None) is None
        ):
            return loss

        current_samples = self._self_consistency_current_samples(pred_data)
        like = loss if torch.is_tensor(loss) else torch.as_tensor(loss, device=self.device)
        raw_loss = like.new_zeros(())
        weighted_loss = like.new_zeros(())
        pairs = []
        submitted = False

        if current_samples:
            pairs = self.self_consistency_scheduler.maybe_consume(
                int(getattr(self, "iter", 0)),
                current_samples,
                timeout=self.self_consistency_consume_timeout,
            )
            if pairs:
                raw_loss = torch.stack(
                    [
                        compute_self_consistency_payload_loss(
                            h_pred_now,
                            h_repaired,
                            tensor_keys=self.self_consistency_tensor_keys,
                        )
                        for h_pred_now, h_repaired in pairs
                    ]
                ).mean()
                weighted_loss = raw_loss * self.self_consistency_weight
                loss = loss + weighted_loss
            submitted = self.self_consistency_scheduler.maybe_submit(
                int(getattr(self, "iter", 0)),
                int(getattr(self, "ep", 0)),
                list(current_samples.items()),
            )

        self._last_self_consistency_state = {
            "train_self_consistency_loss": raw_loss.detach(),
            "train_self_consistency_weighted_loss": weighted_loss.detach(),
            "train_self_consistency_pairs": like.new_tensor(float(len(pairs))),
            "train_self_consistency_submitted": like.new_tensor(1.0 if submitted else 0.0),
        }
        return loss

    @staticmethod
    def _add_effective_expert_lr_state(state, *, optimizer, num_experts):
        num_experts = int(num_experts or 0)
        if num_experts <= 0 or not optimizer.param_groups:
            return
        lr_for_expert_tags = float(optimizer.param_groups[0]["lr"])
        for i in range(num_experts):
            state[f"expert_{i}_lr"] = lr_for_expert_tags

    @staticmethod
    def _batch_info(batch):
        return {
            "__slices__": batch.__slices__,
            "__cumsum__": batch.__cumsum__,
            "__cat_dims__": batch.__cat_dims__,
            "__num_nodes_list__": batch.__num_nodes_list__,
            "__data_class__": batch.__data_class__,
        }

    @staticmethod
    def _optimizer_diagnostics(optimizer):
        if not hasattr(optimizer, "get_diagnostics"):
            return {}
        try:
            return optimizer.get_diagnostics()
        except Exception as exc:
            log.debug("optimizer diagnostics collection failed: %s", exc)
            return {}

    def _optimizer_diagnostics_due(self):
        frequency = max(int(getattr(self, "optimizer_diagnostics_freq", 1)), 1)
        return self.iter == 1 or self.iter % frequency == 0

    def _loss_on_batch(self, batch, lossfunc, *, use_flow=True, allow_self_consistency=True):
        batch = batch.to(self.device)
        batch_info = self._batch_info(batch)
        batch = AtomicData.to_AtomicDataDict(batch)
        batch_for_loss = batch.copy()
        self._last_flow_state = {}
        batch = self.model(batch)
        batch.update(batch_info)
        batch_for_loss.update(batch_info)
        loss = lossfunc(batch, batch_for_loss)
        if allow_self_consistency:
            loss = self._apply_self_consistency_loss(loss, batch)
        return loss

    @staticmethod
    def _loss_component_state(lossfunc, *, prefix="train"):
        loss_obj = Trainer._loss_component_source(lossfunc)
        state = {}
        onsite_comp = getattr(loss_obj, "last_onsite_loss", None)
        hopping_comp = getattr(loss_obj, "last_hopping_loss", None)
        z_loss_comp = getattr(loss_obj, "last_z_loss", None)
        expert_load_cv = getattr(loss_obj, "expert_load_cv", None)

        if onsite_comp is not None:
            state[f"{prefix}_onsite_loss"] = onsite_comp
        if hopping_comp is not None:
            state[f"{prefix}_hopping_loss"] = hopping_comp
        if expert_load_cv is not None:
            state["expert_load_cv" if prefix == "train" else f"{prefix}_expert_load_cv"] = expert_load_cv
        if z_loss_comp is not None:
            state["mean_max_prob" if prefix == "train" else f"{prefix}_mean_max_prob"] = z_loss_comp
        return state

    @staticmethod
    def _endpoint_loss_state(lossfunc, optimization_loss, *, prefix):
        """Build the common all/onsite/hopping endpoint metric contract."""

        loss_obj = Trainer._loss_component_source(lossfunc)
        endpoint_loss = getattr(loss_obj, "last_endpoint_loss", None)
        if endpoint_loss is None:
            endpoint_loss = getattr(loss_obj, "last_feature_compat_loss", None)
        if endpoint_loss is None and bool(
            getattr(loss_obj, "sparse_endpoint_metrics", False)
        ):
            if not torch.is_tensor(optimization_loss):
                optimization_loss = torch.as_tensor(optimization_loss)
            # Keep the triplet structurally explicit so the fail-closed route
            # check still distinguishes a cadence omission from a criterion
            # that never implemented endpoint metrics. Accumulators skip None
            # per key, while the optimization loss remains independently named.
            return {
                f"{prefix}_loss": None,
                f"{prefix}_onsite_loss": None,
                f"{prefix}_hopping_loss": None,
                f"{prefix}_loss_opt": optimization_loss.detach(),
            }
        has_distinct_objective = endpoint_loss is not None
        if endpoint_loss is None:
            endpoint_loss = optimization_loss
        if not torch.is_tensor(endpoint_loss):
            endpoint_loss = torch.as_tensor(endpoint_loss)
        if not torch.is_tensor(optimization_loss):
            optimization_loss = endpoint_loss.new_tensor(float(optimization_loss))

        state = {f"{prefix}_loss": endpoint_loss.detach()}
        state.update(Trainer._loss_component_state(lossfunc, prefix=prefix))
        if has_distinct_objective:
            state[f"{prefix}_loss_opt"] = optimization_loss.detach()
        return state

    @staticmethod
    def _require_endpoint_triplet(state, *, prefix, route):
        required = (
            f"{prefix}_loss",
            f"{prefix}_onsite_loss",
            f"{prefix}_hopping_loss",
        )
        missing = list(required) if state is None else [key for key in required if key not in state]
        if missing:
            raise RuntimeError(
                f"{route} must expose the non-CFM-compatible endpoint triplet; "
                f"missing {missing}. Use a Hamiltonian criterion that provides "
                "onsite/hopping metrics and compatible_loss_from_stats."
            )


    @staticmethod
    def _accumulate_metric_state(metric_sums, state, counts=None):
        for key, value in state.items():
            # A None value marks an omitted/invalid metric for this batch (e.g. a
            # feature-compatible onsite/hopping metric throttled by
            # log_feature_compatible_interval): it must contribute neither a
            # numerator nor a denominator, so skip it entirely rather than
            # coercing it to 0.0.  Existing callers never pass None, so this is a
            # no-op for them.
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach()
            metric_sums[key] = metric_sums.get(key, 0.0) + value
            # Per-key valid-batch count: how many batches actually contributed
            # THIS key.  ``validation()`` divides each metric sum by its own count
            # so a batch that omitted the key (throttled onsite/hopping metric,
            # None above) dilutes neither the numerator nor the denominator.  Each
            # metric key is produced at most once per batch across all
            # ``validation()`` accumulation sites, so one bump per add == one bump
            # per contributing batch.  ``counts is None`` for any caller that does
            # not opt in, leaving them byte-identical.
            if counts is not None:
                counts[key] = counts.get(key, 0) + 1

    def _backward_loss(self, loss):
        """Allow external objectives to report an explicitly completed backward.

        An objective that performs backward itself must return a detached scalar
        marked with ``_dptb_backward_done``. Unmarked objectives use the ordinary
        backward path, including its error for unexpectedly detached losses.
        """
        if getattr(loss, "_dptb_backward_done", False):
            if loss.requires_grad:
                raise RuntimeError("completed backward loss must be detached")
            return
        loss.backward()

    def _skip_nonfinite_batch(self, values, batch, ref_batch=None, *, distributed=False):
        """Discard a nonfinite batch before ANY optimizer/scheduler/plugin update.

        Called after backwards (including DDP reductions), so skipping does not
        leave a reducer waiting for hooks on the next forward. Gradient norms
        are the pre-clipping norms. No optimizer or model-state rollback is done.
        """
        if not getattr(self, "train_options", {}).get("skip_nonfinite_batch", True):
            return False
        flags = torch.stack([
            ~torch.isfinite(value.detach()).all() for value in values.values()
        ])
        bad = flags.any().to(dtype=torch.int32)
        if distributed:
            # WORLD includes both expert parallel and expert data-parallel ranks.
            torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if not bool(bad.item()):
            return False

        optimizers = getattr(self, "optimizers", None)
        if optimizers is None:
            optimizers = [self.optimizer]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        # This is a consumed-loader cursor, NOT the successful optimizer clock.
        self._batch_in_epoch = getattr(self, "_batch_in_epoch", 0) + 1
        self._nonfinite_batches_skipped = getattr(self, "_nonfinite_batches_skipped", 0) + 1

        def batch_record(item):
            if item is None:
                return None
            def read(key):
                value = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
                if torch.is_tensor(value):
                    return value.detach().cpu().tolist()
                return value
            return {key: read(key) for key in (
                "__dptb_sample_indices__", "__dptb_batch_cost__",
                "__dptb_batch_num_graphs__", "graph_id", "frame_id",
            ) if read(key) is not None}

        detail = {
            "rank": torch.distributed.get_rank() if distributed else 0,
            "expert": getattr(self, "local_expert_idx", None),
            "nonfinite": [key for key, flag in zip(values, flags.cpu().tolist()) if flag],
            "batch": batch_record(batch),
            "reference_batch": batch_record(ref_batch),
        }
        details = [detail]
        if distributed:
            details = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(details, detail)
        if not distributed or torch.distributed.get_rank() == 0:
            log.warning("NONFINITE_BATCH_SKIPPED %s", json.dumps({
                "epoch": int(self.ep), "batch_in_epoch": self._batch_in_epoch,
                "next_optimizer_step": int(self.iter),
                "skipped_since_start": self._nonfinite_batches_skipped,
                "ranks": details,
            }, ensure_ascii=True))
        return True

    def iteration(self, batch, ref_batch=None):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        dynamic_batch_state = self._dynamic_batch_state_from_batch(batch)

        loss = self._loss_on_batch(batch, self.train_lossfunc, use_flow=True)
        main_self_consistency_state = dict(
            getattr(self, "_last_self_consistency_state", {})
        )
        main_endpoint_state = {}
        main_endpoint_state = self._endpoint_loss_state(
            self.train_lossfunc,
            loss,
            prefix="train",
        )
        if self._supports_endpoint_triplet(self.train_lossfunc):
            self._require_endpoint_triplet(
                main_endpoint_state,
                prefix="train",
                route="Non-CFM training",
            )
        loss_for_log = main_endpoint_state["train_loss"].detach()
        loss_opt_for_log = loss.detach()
        finite_checks = {"main_loss": loss.detach()}
        self._backward_loss(loss)
        del loss

        ref_component_state = {}
        if ref_batch is not None:
            reference_lossfunc = getattr(self, "reference_lossfunc", self.train_lossfunc)
            ref_loss = self._loss_on_batch(
                ref_batch,
                reference_lossfunc,
                use_flow=False,
                allow_self_consistency=False,
            )
            loss_opt_for_log = loss_opt_for_log + ref_loss.detach()
            finite_checks["reference_loss"] = ref_loss.detach()
            self._backward_loss(ref_loss)
            ref_component_state = self._endpoint_loss_state(
                reference_lossfunc,
                ref_loss,
                prefix="ref",
            )
            del ref_loss

        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.clip_grad_norm
        )

        finite_checks.update(objective=loss_opt_for_log, grad_norm=total_norm)
        if self._skip_nonfinite_batch(finite_checks, batch, ref_batch):
            return None

        self.optimizer.step()

        if self.update_lr_per_iter:
            if lr_scheduler_requires_metric(self.lr_scheduler):
                if self.iter > 1:
                    scheduler_stat = "train_loss_opt"
                    self.lr_scheduler.step(
                        self.stats[scheduler_stat]['latest_avg_iter_loss']
                    )
                elif lr_scheduler_can_step_without_metric(self.lr_scheduler):
                    self.lr_scheduler.step()
            else:
                self.lr_scheduler.step()

        # The optimizer step for this batch is now baked into the model; count
        # it as committed *before* call_plugins so a checkpoint fired by
        # Saver.iteration persists the correct mid-epoch fast-forward cursor.
        # getattr-guarded: some unit tests drive iteration() on trainers built
        # without BaseTrainer.__init__.
        self._batch_in_epoch = getattr(self, "_batch_in_epoch", 0) + 1

        state = {
            'field': 'iteration',
            'window_steps': 1,
            "train_loss": loss_for_log,
            "train_loss_opt": loss_opt_for_log,
            "lr": self.optimizer.state_dict()["param_groups"][0]['lr'],
            "total_grad_norm": total_norm.item()
        }
        self._add_effective_expert_lr_state(
            state,
            optimizer=self.optimizer,
            num_experts=getattr(self, "num_experts", getattr(self.model, "num_experts", 0)),
        )
        state.update(dynamic_batch_state)
        state.update(main_endpoint_state)
        state.update(ref_component_state)
        # The backward objective can include a reference batch, whereas the
        # canonical endpoint triplet remains scoped to the main batch.
        state["train_loss_opt"] = loss_opt_for_log
        state.update(main_self_consistency_state)
        if self._optimizer_diagnostics_due():
            state.update(self._optimizer_diagnostics(self.optimizer))

        self.call_plugins(queue_name='iteration', time=self.iter, **state)
        self.iter += 1

        # Match MultiTrainer: iteration() returns the objective that was
        # differentiated, while the comparable endpoint remains in
        # state["train_loss"] for monitors/TensorBoard.
        return loss_opt_for_log

    @classmethod
    def restart(cls, checkpoint, train_datasets, train_options=None, common_options=None, reference_datasets=None,
                validation_datasets=None):
        train_options = {} if train_options is None else train_options
        common_options = {} if common_options is None else common_options
        ckpt = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        validate_checkpoint_invariants(ckpt)
        validate_checkpoint_world_size(ckpt)
        preflight_restart_checkpoint(checkpoint, ckpt, trainer_kind="trainer")
        ckpt_train_options = migrate_legacy_checkpoint_train_options(
            ckpt["config"]["train_options"]
        )
        merged_train_options = merge_restart_train_options(
            train_options,
            ckpt_train_options,
            logger=log,
        )
        merged_common_options = copy.deepcopy(ckpt["config"]["common_options"])
        merged_common_options.update(common_options)
        model_build_train_options = merged_train_options
        model_state = ckpt.get("model_state_dict", {})
        has_flat_distance_state = (
            bool(ckpt_train_options.get("distance_ranges"))
            and isinstance(model_state, dict)
            and not any(k.startswith("experts.") for k in model_state.keys())
        )
        if has_flat_distance_state:
            model_build_train_options = copy.deepcopy(model_build_train_options)
            model_build_train_options.pop("distance_ranges", None)
            log.warning(
                "Detected flat single-model checkpoint with distance_ranges; "
                "building the restart model without DistanceEnsembleWrapper so "
                "optimizer parameter groups match the saved checkpoint."
            )
        model = build_model(
            checkpoint,
            ckpt["config"]["model_options"],
            merged_common_options,
            train_options=model_build_train_options,
        )
        train_options = merged_train_options
        trainer = cls(model=model, train_datasets=train_datasets, reference_datasets=reference_datasets,
                      validation_datasets=validation_datasets, train_options=train_options,
                      common_options=merged_common_options)
        trainer.stats = ckpt["stats"]

        # ---- resume state machine (legacy-tolerant) -----------------------
        resume = read_resume_metadata(ckpt)
        trainer.iter = int(ckpt["iteration"]) + 1
        # Stash restored per-plugin state so plugins restore themselves at
        # register time (plugins are registered by the entrypoint *after*
        # restart()); e.g. Saver's best_loss / retention queues (BUG 2).
        trainer._restored_plugin_state = dict(resume.plugin_state or {})

        # Load optimizer/scheduler *before* any epoch-end LR replay.
        for key in Trainer.object_keys:
            item = getattr(trainer, key, None)
            if item is not None:
                item.load_state_dict(ckpt[key + "_state_dict"])

        # Prefer this rank's own RNG snapshot over the main-rank blob (P0-3).
        own_rng = resolve_rank_rng_state(ckpt, resume)

        if resume.checkpoint_kind == CHECKPOINT_KIND_ITERATION:
            # Mid-epoch checkpoint: re-enter the SAME epoch (do NOT skip it) and
            # fast-forward over the batches already committed into the saved
            # model, so each remaining batch runs exactly once (BUG 1).
            trainer.ep = int(resume.epoch)
            trainer._resume_plan = {
                "target_epoch": int(resume.epoch),
                "skip_batches": int(resume.batch_in_epoch),
                "rng_state": own_rng,
            }
        else:
            # Epoch-committed checkpoint: advance to the next epoch. The per-epoch
            # LR step for the committed epoch fired *after* Saver.epoch persisted
            # the scheduler, so replay exactly one step to avoid an off-by-one
            # LR transition on resume (BUG 1, scheduler tail).
            trainer.ep = int(resume.epoch) + 1
            if own_rng is not None:
                restore_rng_state(own_rng)
                trainer._prepare_epoch_restart_worker_rng()
            if resume.epoch_scheduler_step_pending and not trainer.update_lr_per_iter:
                try:
                    trainer._lr_step_on_epoch_end()
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning(
                        "Failed to replay epoch-end LR step on restart: %s", exc
                    )
        return trainer

    def _prepare_epoch_restart_worker_rng(self):
        """Keep replacement persistent workers off the restored training RNG.

        An uninterrupted loader reuses its workers after the first epoch. A
        restarted loader instead draws a new worker base seed when its first
        iterator is created. Give that extra draw a private copy of the CPU
        RNG, leaving the sampler and model on the checkpoint's original stream.
        Assigning the loader generator after construction deliberately leaves
        a RandomSampler's generator unchanged. Non-persistent loaders still
        consume their usual base-seed draw on every epoch in both paths.

        This does not restore random per-item transforms in worker processes.
        """
        loaders = [self.train_loader]
        if self.use_reference:
            loaders.append(self.reference_loader)
        for loader in loaders:
            if (getattr(loader, "num_workers", 0) > 0
                    and getattr(loader, "persistent_workers", False)
                    and getattr(loader, "generator", None) is None):
                loader.generator = torch.Generator().set_state(torch.get_rng_state())

    def epoch(self) -> None:
        # Reset the per-epoch committed-batch cursor; consume any one-shot
        # resume plan left by restart() so the *first* epoch after a mid-epoch
        # restart fast-forwards over already-committed batches and restores the
        # RNG trajectory before running the remainder exactly once.
        self._batch_in_epoch = 0
        skip_batches, resume_rng = self._consume_resume_plan()
        replay_failure = self._exact_fast_forward_failure()
        if skip_batches and replay_failure is not None:
            # The loader's batch order is not restart-reproducible (e.g. plain
            # shuffle=True without a per-epoch-seeded sampler), so an exact
            # fast-forward is impossible. Re-running the epoch from its start is
            # NOT a safe fallback either: the checkpointed model/optimizer
            # already contain the updates of the committed prefix, so a re-run
            # applies those optimizer steps a second time (momentum/Adam moments
            # and the LR trajectory diverge from an uninterrupted run). Fail
            # closed and tell the user to resume from the last EPOCH checkpoint,
            # unless they explicitly opt into the inexact re-run.
            if os.environ.get("DPTB_ALLOW_INEXACT_RESUME", "").strip() in ("1", "true", "True"):
                log.warning(
                    "DPTB_ALLOW_INEXACT_RESUME=1: re-running epoch %s from its "
                    "start; optimizer updates for the %s already-committed "
                    "batches WILL be applied a second time (training trajectory "
                    "diverges from an uninterrupted run).",
                    int(self.ep), skip_batches,
                )
                skip_batches = 0
                resume_rng = None
                # The restored trainer.stats hold the PARTIAL per-epoch
                # accumulators of the committed prefix; re-running every batch
                # would double-count them into epoch_mean (plateau LR + best
                # gate). Reset so the re-run rebuilds them cleanly. (The exact
                # fast-forward path SKIPS committed batches, so it must NOT
                # reset — the restored partial is still correct there.)
                self._reset_epoch_metric_accumulators()
            else:
                loader_name, loader, failure_reason = replay_failure
                raise RuntimeError(
                    f"Cannot resume mid-epoch (iteration checkpoint, "
                    f"{skip_batches} committed batches) because {loader_name} "
                    f"is not restart-deterministic "
                    f"({type(loader).__name__}: {failure_reason}). Resume from the "
                    f"last epoch checkpoint instead, or set "
                    f"DPTB_ALLOW_INEXACT_RESUME=1 to re-run this epoch from "
                    f"its start (re-applies the committed prefix's optimizer "
                    f"updates)."
                )

        # Seed the committed-batch cursor to the fast-forward offset so a SECOND
        # mid-epoch preemption during this resumed epoch persists an ABSOLUTE
        # batch position. Otherwise the cursor would restart from 0 and a later
        # restart would recompute (and double-step the optimizer over) the
        # already-fast-forwarded batches. In the safe-fallback path
        # skip_batches == 0, so this is a no-op.
        self._batch_in_epoch = skip_batches

        batch_sampler = getattr(self.train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(int(self.ep))
        if self.use_reference:
            ref_batch_sampler = getattr(self.reference_loader, "batch_sampler", None)
            if hasattr(ref_batch_sampler, "set_epoch"):
                ref_batch_sampler.set_epoch(int(self.ep))
        reference_iter = iter(self.reference_loader) if self.use_reference else None
        rng_restored = False
        for batch_index, ibatch in enumerate(self.train_loader):
            if batch_index < skip_batches:
                # Fast-forward: this batch was already committed pre-restart.
                # Keep the reference stream aligned but do not re-optimize.
                if self.use_reference:
                    try:
                        next(reference_iter)
                    except StopIteration:
                        reference_iter = iter(self.reference_loader)
                        next(reference_iter)
                continue
            if not rng_restored:
                # Restore exactly once, right before the first re-executed batch,
                # so it sees the RNG state an uninterrupted run would have had
                # (the fast-forward above must not perturb it).
                if resume_rng is not None:
                    restore_rng_state(resume_rng)
                rng_restored = True
            if self.use_reference:
                try:
                    ref_batch = next(reference_iter)
                except StopIteration:
                    reference_iter = iter(self.reference_loader)
                    ref_batch = next(reference_iter)
                self.iteration(ibatch, ref_batch)
            else:
                self.iteration(ibatch)

    def _reset_epoch_metric_accumulators(self):
        """Zero the per-epoch (sum, count) accumulators in ``trainer.stats``.

        Mirrors what Monitor.epoch does at an epoch boundary. Used only when a
        re-entered epoch is re-run from its start after a restart, so the
        restored partial accumulators are not double-counted into epoch_mean.
        """
        for entry in self.stats.values():
            if isinstance(entry, dict) and 'epoch_stats' in entry:
                entry['epoch_stats'] = (0, 0)

    def _consume_resume_plan(self):
        """Pop the one-shot restart plan for the current epoch.

        Returns ``(skip_batches, rng_state)``.  The plan only applies to the
        first epoch re-entered after restart (matched by target epoch); any
        later epoch runs from a clean cursor.
        """
        plan = getattr(self, "_resume_plan", None)
        if not plan:
            return 0, None
        if int(plan.get("target_epoch", self.ep)) != int(self.ep):
            return 0, None
        # Consume so subsequent epochs are unaffected.
        self._resume_plan = None
        return int(plan.get("skip_batches", 0)), plan.get("rng_state")

    @staticmethod
    def _train_loader_worker_kwargs(train_options):
        """Worker options for the train loader, read like MultiTrainer reads them.

        Without ``train_num_workers`` the loader keeps its in-process defaults.
        Workers only decode items in parallel; the batch sampler stays in the
        main process, so batch order and content are unchanged.
        """
        workers = int(train_options.get("train_num_workers", 0) or 0)
        if workers <= 0:
            return {}
        return {
            "num_workers": workers,
            "pin_memory": bool(train_options.get("data_pin_memory", torch.cuda.is_available())),
            "persistent_workers": bool(train_options.get("data_persistent_workers", True)),
            "prefetch_factor": int(train_options.get("data_prefetch_factor", 2)),
        }

    @staticmethod
    def _loader_exact_replay_status(loader):
        """Return ``(is_exact, reason)`` for one loader's replay boundary.

        This deliberately covers only deterministic batch *order*. Worker RNG
        state and random dataset transforms are not captured, so a
        ``num_workers > 0`` loader is exact only when its dataset declares
        ``deterministic_items`` (item decoding uses no randomness).
        """
        if int(getattr(loader, "num_workers", 0) or 0) > 0:
            dataset = getattr(loader, "dataset", None)
            dataset = getattr(dataset, "dataset", dataset)
            if not getattr(dataset, "deterministic_items", False):
                return False, "num_workers>0 worker RNG/random transforms are not replayed"
        batch_sampler = getattr(loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            return True, "per-epoch-seeded batch sampler"
        try:
            import torch.utils.data as _tud
            if not isinstance(loader, _tud.DataLoader):
                return True, "fixed-order non-DataLoader sequence"
        except Exception:  # pragma: no cover - defensive
            return True, "non-DataLoader sequence"
        # A torch DataLoader with no per-epoch-seeded sampler: safe only if it
        # is not shuffling (deterministic sequential order).
        sampler = getattr(loader, "sampler", None)
        sampler_name = type(sampler).__name__ if sampler is not None else ""
        if "Random" in sampler_name:
            return False, f"random sampler {sampler_name} has no epoch replay token"
        return True, f"deterministic sampler {sampler_name or '<none>'}"

    def _exact_fast_forward_failure(self):
        loaders = [("train_loader", self.train_loader)]
        if getattr(self, "use_reference", False):
            loaders.append(("reference_loader", self.reference_loader))
        for loader_name, loader in loaders:
            exact, reason = self._loader_exact_replay_status(loader)
            if not exact:
                return loader_name, loader, reason
        return None

    def _loaders_support_exact_fast_forward(self):
        """Whether every consumed loader can replay the committed prefix.

        Reference batches advance in lockstep with train batches during resume,
        so ``use_reference=True`` requires both loader streams to be exact.
        """
        return self._exact_fast_forward_failure() is None

    def _loader_supports_exact_fast_forward(self):
        """Backward-compatible alias for the all-loader exactness decision."""
        return self._loaders_support_exact_fast_forward()

    def update(self, **kwargs):
        pass

    def validation(self, fast=True):
        with torch.no_grad():
            loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            validation_metric_sums = {}
            # Count only batches that contribute each endpoint metric.
            validation_metric_counts = {}
            num_batches = 0
            self.model.eval()
            generator = getattr(self, "validation_loader_generator", None)
            if generator is not None:
                generator.manual_seed(self.validation_loader_seed)
            for batch in self.validation_loader:
                batch = batch.to(self.device)
                batch_info = {"__slices__": batch.__slices__, "__cumsum__": batch.__cumsum__,
                              "__cat_dims__": batch.__cat_dims__, "__num_nodes_list__": batch.__num_nodes_list__,
                              "__data_class__": batch.__data_class__}
                batch = AtomicData.to_AtomicDataDict(batch)
                batch_for_loss = batch.copy()
                batch = self.model(batch)
                batch.update(batch_info)
                batch_for_loss.update(batch_info)
                batch_loss = self.validation_lossfunc(batch, batch_for_loss)
                endpoint_state = self._endpoint_loss_state(
                    self.validation_lossfunc,
                    batch_loss,
                    prefix="validation",
                )
                if self._supports_endpoint_triplet(self.validation_lossfunc):
                    self._require_endpoint_triplet(
                        endpoint_state,
                        prefix="validation",
                        route="Non-CFM validation",
                    )
                endpoint_loss = endpoint_state["validation_loss"]
                loss += (
                    endpoint_loss
                    if endpoint_loss is not None
                    else batch_loss.detach()
                )
                self._accumulate_metric_state(
                    validation_metric_sums,
                    endpoint_state,
                    validation_metric_counts,
                )
                num_batches += 1
                if fast: break
        divisor = max(num_batches, 1)
        if not fast:
            loss = loss / divisor
        self._last_flow_validation_state = {
            key: value / validation_metric_counts.get(key, divisor)
            for key, value in validation_metric_sums.items()
        }
        return self._resolve_validation_return(loss)

    def _resolve_validation_return(self, accumulated_loss):
        """Return the public endpoint metric when the criterion provides it."""
        state = getattr(self, "_last_flow_validation_state", None) or {}
        return state.get("validation_loss", accumulated_loss)

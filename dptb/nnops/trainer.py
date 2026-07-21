import torch
import logging
import os
import csv
import math
import copy
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
from dptb.nnops.flow import (
    assert_flow_h0_keys_reach_model,
    assert_model_in_loss_endpoint_metric_space,
    build_hamiltonian_flow,
)
from dptb.nnops.loss import Loss
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
        self.optimizer = get_optimizer(model_param=self.model.named_parameters(), **train_options["optimizer"])
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

        flow_idp = getattr(self.train_lossfunc, "idp", loss_idp)
        self.flow_cfm = build_hamiltonian_flow(
            train_options.get("flow_options", None),
            idp=flow_idp,
            dtype=self.dtype,
            device=self.device,
        )
        # Fail closed when an enabled flow writes its interpolated H0 state to
        # keys the model's H0-init embedding never reads (silent prior
        # deactivation P0: flow.node_h0_key != embedding.h0_node_key).
        assert_flow_h0_keys_reach_model(self.flow_cfm, self.model)
        # Model-in-loss MeanFlow exposes endpoint sums only in its configured
        # target representation.  Reject a block/RME mismatch here instead of
        # failing after the first expensive MeanFlow model calls, and never
        # hide it behind an implicit online representation conversion.
        self._assert_model_in_loss_endpoint_contract()
        self._last_flow_state = {}
        self._last_flow_validation_state = {}
        self._last_self_consistency_state = {}

        if train_options["loss_options"]["train"]["method"] == "skints":
            assert self.model.name == 'nnsk', "The model should be nnsk for the skints loss function."
            assert self.model.onsite_fn.functype in ['none',
                                                     'uniform'], "The onsite function should be none or uniform for the skints loss function."
            log.info("The skints loss function is used for training, the model.transform is then set to False.")
            self.model.transform = False

    def _assert_model_in_loss_endpoint_contract(self) -> None:
        """Validate criteria used directly with a model-in-loss flow."""
        assert_model_in_loss_endpoint_metric_space(
            self.flow_cfm,
            self.train_lossfunc,
            criterion_name="train",
        )
        if getattr(self, "use_validation", False):
            assert_model_in_loss_endpoint_metric_space(
                self.flow_cfm,
                self.validation_lossfunc,
                criterion_name="validation",
            )
        if self.use_reference and getattr(self.flow_cfm, "apply_to_reference", False):
            assert_model_in_loss_endpoint_metric_space(
                self.flow_cfm,
                self.reference_lossfunc,
                criterion_name="reference",
            )

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

    _COMPATIBLE_LOSS_SIDE_EFFECT_ATTRS = (
        "last_endpoint_loss",
        "last_endpoint_metric_space",
        "last_feature_compat_loss",
        "last_opt_loss",
        "last_block_loss",
        "last_block_element_mae",
        "last_block_onsite_loss",
        "last_block_hopping_loss",
        "last_onsite_loss",
        "last_hopping_loss",
        "last_z_loss",
        "expert_load_cv",
        "last_onsite_l1_sum",
        "last_onsite_mse_sum",
        "last_onsite_count",
        "last_hopping_l1_sum",
        "last_hopping_mse_sum",
        "last_hopping_count",
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
        if use_flow and self.flow_cfm.enabled:
            model_in_loss = getattr(self.flow_cfm, "model_in_loss", False)
            if getattr(self.flow_cfm, "model_in_loss", False):
                loss, flow_state = self.flow_cfm.loss_with_model(self.model, batch, batch_for_loss)
                self._last_self_consistency_state = {}
            else:
                batch, batch_for_loss, flow_ctx = self.flow_cfm.prepare_batch(batch, batch_for_loss)
                batch = self.model(batch)
                batch.update(batch_info)
                batch_for_loss.update(batch_info)
                loss, flow_state = self.flow_cfm.loss(batch, batch_for_loss, flow_ctx)
                if allow_self_consistency:
                    loss = self._apply_self_consistency_loss(loss, batch)
            flow_state.setdefault("train_loss_opt", loss.detach())
            compatible_state = self._compatible_loss_state_from_flow_stats(
                lossfunc,
                flow_state,
                source_prefix="train",
                prefix="train_compatible",
                legacy_prefix="train",
                global_step=getattr(self, "iter", None),
            )
            if compatible_state is None and not model_in_loss:
                compatible_state = self._compatible_loss_state(
                    lossfunc,
                    batch,
                    batch_for_loss,
                    prefix="train_compatible",
                    legacy_prefix="train",
                )
            if compatible_state is None:
                raise RuntimeError(
                    "Enabled flow could not reconstruct the non-CFM-compatible "
                    "train endpoint loss from the configured criterion."
                )
            self._require_endpoint_triplet(
                compatible_state,
                prefix="train",
                route="Enabled flow training",
            )
            flow_state.update(compatible_state)
            flow_state.pop("_compatible_clean_stats", None)
            self._last_flow_state = flow_state
            return loss
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
    def _compatible_loss_state(
        lossfunc,
        pred_data,
        ref_data,
        *,
        prefix,
        legacy_prefix=None,
        include_raw_stats=False,
    ):
        loss_obj = Trainer._loss_component_source(lossfunc)
        sentinel = object()
        saved_side_effects = {
            attr: getattr(loss_obj, attr, sentinel)
            for attr in Trainer._COMPATIBLE_LOSS_SIDE_EFFECT_ATTRS
        }

        try:
            with torch.no_grad():
                compatible_loss = lossfunc(pred_data, ref_data)

            state = Trainer._endpoint_loss_state(
                lossfunc,
                compatible_loss,
                prefix=prefix,
            )
            if include_raw_stats:
                raw_stats = {}
                for component in ("onsite", "hopping"):
                    for source_suffix, target_suffix in (
                        ("l1_sum", "l1_sum"),
                        ("mse_sum", "mse_sum"),
                        ("count", "count"),
                    ):
                        value = getattr(
                            loss_obj,
                            f"last_{component}_{source_suffix}",
                            None,
                        )
                        if value is not None:
                            raw_stats[f"{component}_{target_suffix}"] = (
                                value.detach() if torch.is_tensor(value) else value
                            )
                if len(raw_stats) == 6:
                    raw_stats["metric_space"] = getattr(
                        loss_obj,
                        "endpoint_metric_space",
                        "rme",
                    )
                    state["_endpoint_stats"] = raw_stats
        finally:
            for attr, value in saved_side_effects.items():
                if value is sentinel:
                    try:
                        delattr(loss_obj, attr)
                    except AttributeError:
                        pass
                else:
                    setattr(loss_obj, attr, value)

        if legacy_prefix is not None:
            loss_key = f"{prefix}_loss"
            onsite_key = f"{prefix}_onsite_loss"
            hopping_key = f"{prefix}_hopping_loss"
            if loss_key in state:
                state[f"{legacy_prefix}_loss"] = state[loss_key]
            if onsite_key in state:
                state[f"{legacy_prefix}_onsite_loss"] = state[onsite_key]
            if hopping_key in state:
                state[f"{legacy_prefix}_hopping_loss"] = state[hopping_key]
        return state

    @staticmethod
    def _compatible_loss_state_from_flow_stats(
        lossfunc,
        flow_state,
        *,
        source_prefix,
        prefix,
        legacy_prefix=None,
        global_step=None,
    ):
        stats = flow_state.get("_compatible_clean_stats", None)
        if not isinstance(stats, dict):
            return None

        required = (
            "onsite_l1_sum",
            "onsite_mse_sum",
            "onsite_count",
            "hopping_l1_sum",
            "hopping_mse_sum",
            "hopping_count",
        )
        if any(stats.get(key, None) is None for key in required):
            return None

        loss_obj = Trainer._loss_component_source(lossfunc)
        stats_metric_space = stats.get("metric_space", None)
        criterion_metric_space = getattr(
            loss_obj,
            "endpoint_metric_space",
            "rme",
        )
        if (
            stats_metric_space is not None
            and str(stats_metric_space) != str(criterion_metric_space)
        ):
            return None
        reduce_from_stats = getattr(loss_obj, "compatible_loss_from_stats", None)
        if not callable(reduce_from_stats):
            return None

        z_loss = flow_state.get(
            "mean_max_prob",
            flow_state.get(f"{source_prefix}_mean_max_prob", None),
        )
        compatible_loss, onsite_loss, hopping_loss = reduce_from_stats(
            onsite_l1_sum=stats["onsite_l1_sum"],
            onsite_mse_sum=stats["onsite_mse_sum"],
            onsite_count=stats["onsite_count"],
            hopping_l1_sum=stats["hopping_l1_sum"],
            hopping_mse_sum=stats["hopping_mse_sum"],
            hopping_count=stats["hopping_count"],
            z_loss=z_loss,
            global_step=global_step,
        )

        state = {
            f"{prefix}_loss": compatible_loss.detach(),
            f"{prefix}_onsite_loss": onsite_loss.detach(),
            f"{prefix}_hopping_loss": hopping_loss.detach(),
        }

        def _detached_scalar(value, like=compatible_loss, default=0.0):
            if value is None:
                value = default
            if torch.is_tensor(value):
                return value.detach()
            return like.new_tensor(float(value))

        state[f"{prefix}_mean_max_prob"] = _detached_scalar(z_loss)
        state[f"{prefix}_expert_load_cv"] = _detached_scalar(
            flow_state.get(
                "expert_load_cv",
                flow_state.get(f"{source_prefix}_expert_load_cv", None),
            )
        )

        if legacy_prefix is not None:
            state[f"{legacy_prefix}_loss"] = state[f"{prefix}_loss"]
            state[f"{legacy_prefix}_onsite_loss"] = state[f"{prefix}_onsite_loss"]
            state[f"{legacy_prefix}_hopping_loss"] = state[f"{prefix}_hopping_loss"]
        return state

    @staticmethod
    def _renamespace_flow_objective_state(flow_state, *, old_prefix, new_prefix):
        """Re-key model-in-loss flow objective scalars under a flow namespace.

        loss_with_model(prefix="validation_one_step") emits
        validation_one_step_flow_* keys, which match neither the TensorBoard
        prefix scan (validation_flow_*/validation_compatible_*) nor the legacy
        namespace -- they would be accumulated but never plotted. Rename them
        to validation_flow_one_step_* so the flow objective stays observable.
        """
        old = f"{old_prefix}_flow_"
        out = {}
        for key, value in flow_state.items():
            if key.startswith(old):
                out[f"{new_prefix}_{key[len(old):]}"] = value
        return out

    @staticmethod
    def _validation_flow_metric_enabled(flow, name):
        """Read the canonical validation metric set, with mock compatibility.

        Production flows are configured through ``validation_flow_metrics``.
        The legacy boolean attributes remain only as a fallback for older test
        doubles and wrappers that have not adopted the canonical set yet.
        """
        canonical = getattr(flow, "validation_flow_metrics", None)
        if canonical is not None:
            normalized = {
                str(value).lower().replace("-", "_") for value in canonical
            }
            return str(name).lower().replace("-", "_") in normalized
        legacy_attr = {
            "random_t": "log_validation_random_t_loss",
            "one_step": "log_validation_t0_loss",
            "trajectory": "log_validation_flow_euler_loss",
        }[name]
        return bool(getattr(flow, legacy_attr, True))

    @staticmethod
    def _validation_prior_seed(flow):
        """Return the batch-independent projected-TE validation seed, if used."""
        if getattr(flow, "prior", "zero") != "projected_te":
            return None
        base_seed = getattr(flow, "validation_prior_base_seed", None)
        if callable(base_seed):
            return base_seed()
        # Compatibility for wrappers built against the pre-fc1099e API.  The
        # fixed batch index is essential: per-sample UID substreams provide the
        # actual decorrelation and must not inherit loader position.
        validation_seed = getattr(flow, "validation_seed", None)
        if callable(validation_seed):
            return validation_seed(0, "prior")
        return None

    def _score_block_ode_validation_sample(
        self,
        sampled,
        original_batch,
        batch_for_loss,
        batch_info,
        *,
        prior_seed=None,
        trajectory=False,
    ):
        """Score a block-ODE rollout in the flow-owned physical target space.

        Residual block-ODE sampling may return physical Full-H while the ordinary
        criterion still targets dH.  Rebuild the t=0 flow context and delegate the
        conversion/scoring decision to the flow; directly calling the criterion
        here would count H0 as prediction error.
        """
        flow = self.flow_cfm
        num_graphs = flow._num_graphs(original_batch)
        sample_node = sampled.get(
            getattr(flow, "node_output_key", "node_hamil_blocks")
        )
        zero_device = (
            sample_node.device if torch.is_tensor(sample_node) else self.device
        )
        zero_t = torch.zeros(num_graphs, device=zero_device, dtype=self.dtype)
        prepare_kwargs = {"t": zero_t}
        if prior_seed is not None:
            prepare_kwargs["prior_seed"] = prior_seed
        _flow_batch, flow_ref, flow_ctx = flow.prepare_batch(
            original_batch,
            batch_for_loss,
            **prepare_kwargs,
        )
        flow_ref.update(batch_info)
        scorer = flow.loss_on_sample if trajectory else flow.compatible_loss_on_sample
        return scorer(sampled, flow_ref, flow_ctx)

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

    def iteration(self, batch, ref_batch=None):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        dynamic_batch_state = self._dynamic_batch_state_from_batch(batch)

        loss = self._loss_on_batch(batch, self.train_lossfunc, use_flow=True)
        flow_enabled = bool(self.flow_cfm.enabled)
        main_flow_state = dict(getattr(self, "_last_flow_state", {}))
        main_self_consistency_state = dict(
            getattr(self, "_last_self_consistency_state", {})
        )
        main_endpoint_state = {}
        if flow_enabled:
            loss_for_log = main_flow_state["train_loss"].detach()
        else:
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
        loss.backward()
        del loss

        ref_component_state = {}
        if ref_batch is not None:
            reference_lossfunc = getattr(self, "reference_lossfunc", self.train_lossfunc)
            apply_flow_to_reference = bool(
                getattr(self.flow_cfm, "apply_to_reference", False)
            )
            ref_loss = self._loss_on_batch(
                ref_batch,
                reference_lossfunc,
                use_flow=apply_flow_to_reference,
                allow_self_consistency=False,
            )
            loss_opt_for_log = loss_opt_for_log + ref_loss.detach()
            ref_loss.backward()
            if apply_flow_to_reference:
                ref_flow_state = dict(getattr(self, "_last_flow_state", {}))
                for suffix in ("loss", "onsite_loss", "hopping_loss"):
                    source = f"train_{suffix}"
                    if source in ref_flow_state:
                        ref_component_state[f"ref_{suffix}"] = ref_flow_state[source]
            else:
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

        self.optimizer.step()

        if self.update_lr_per_iter:
            if lr_scheduler_requires_metric(self.lr_scheduler):
                if self.iter > 1:
                    scheduler_stat = "train_loss" if flow_enabled else "train_loss_opt"
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
        state.update(main_flow_state)
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
            if resume.epoch_scheduler_step_pending and not trainer.update_lr_per_iter:
                try:
                    trainer._lr_step_on_epoch_end()
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning(
                        "Failed to replay epoch-end LR step on restart: %s", exc
                    )
        return trainer

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
    def _loader_exact_replay_status(loader):
        """Return ``(is_exact, reason)`` for one loader's replay boundary.

        This deliberately covers only deterministic batch *order*. It does not
        capture worker RNG state or arbitrary random dataset transforms, so any
        ``num_workers > 0`` loader is conservatively inexact.
        """
        if int(getattr(loader, "num_workers", 0) or 0) > 0:
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
            flow_metric_sums = {}
            # Per-key valid-batch counts, filled in lock-step with flow_metric_sums
            # by _accumulate_metric_state.  Keys accumulated only through that helper
            # (the throttleable feature-compatible onsite/hopping metrics and every
            # compatible-loss key) are divided by their own count; keys written
            # directly below (validation_flow_random_t/t0/euler_* losses) are gated
            # by per-run-constant config flags, so they are present on every batch
            # or none -- they are absent from this dict and fall back to num_batches
            # (== their true count), keeping the interval=1 result byte-identical.
            flow_metric_counts = {}
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
                if self.flow_cfm.enabled:
                    original_batch = batch.copy()
                    validation_prior_seed = self._validation_prior_seed(
                        self.flow_cfm
                    )
                    log_random_t = self._validation_flow_metric_enabled(
                        self.flow_cfm, "random_t"
                    )
                    log_t0 = self._validation_flow_metric_enabled(
                        self.flow_cfm, "one_step"
                    )
                    log_flow_euler = self._validation_flow_metric_enabled(
                        self.flow_cfm, "trajectory"
                    )
                    if getattr(self.flow_cfm, "model_in_loss", False):
                        batch_for_loss.update(batch_info)
                        if log_random_t:
                            random_t_loss, random_t_state = self.flow_cfm.loss_with_model(
                                self.model, original_batch, batch_for_loss, prefix="validation"
                            )
                            loss += random_t_loss
                            flow_metric_sums["validation_flow_random_t_loss"] = (
                                flow_metric_sums.get("validation_flow_random_t_loss", 0.0)
                                + random_t_loss.detach()
                            )
                            self._accumulate_metric_state(flow_metric_sums, random_t_state, flow_metric_counts)
                        num_graphs = self.flow_cfm._num_graphs(original_batch)
                        zero_t = torch.zeros(num_graphs, device=self.device, dtype=self.dtype)
                        one_t = torch.ones(num_graphs, device=self.device, dtype=self.dtype)
                        if log_t0:
                            one_step_loss, one_step_state = self.flow_cfm.loss_with_model(
                                self.model,
                                original_batch,
                                batch_for_loss,
                                prefix="validation_one_step",
                                r=zero_t,
                                t=one_t,
                            )
                            self._accumulate_metric_state(
                                flow_metric_sums,
                                self._renamespace_flow_objective_state(
                                    one_step_state,
                                    old_prefix="validation_one_step",
                                    new_prefix="validation_flow_one_step",
                                ),
                                flow_metric_counts,
                            )
                        # Endpoint-compatible validation: euler-sample to t=0
                        # and score the blockwise criterion so pMF's legacy
                        # validation_* keys stay comparable with no-CFM/CFM.
                        for num_steps in self.flow_cfm.validation_ode_steps:
                            sample_kwargs = {"num_steps": num_steps}
                            if validation_prior_seed is not None:
                                sample_kwargs["prior_seed"] = validation_prior_seed
                            sampled = self.flow_cfm.sample(
                                self.model, original_batch, **sample_kwargs
                            )
                            sampled.update(batch_info)
                            legacy_prefix = "validation" if int(num_steps) == 1 else None
                            if bool(getattr(self.flow_cfm, "block_ode", False)):
                                sample_loss, sample_state = (
                                    self._score_block_ode_validation_sample(
                                        sampled,
                                        original_batch,
                                        batch_for_loss,
                                        batch_info,
                                        prior_seed=validation_prior_seed,
                                        trajectory=log_flow_euler,
                                    )
                                )
                                if log_flow_euler:
                                    self._accumulate_metric_state(
                                        flow_metric_sums,
                                        {
                                            f"validation_flow_euler_{num_steps}_loss": sample_loss
                                        },
                                        flow_metric_counts,
                                    )
                                compatible_state = (
                                    self._compatible_loss_state_from_flow_stats(
                                        self.validation_lossfunc,
                                        sample_state,
                                        source_prefix="train",
                                        prefix=f"validation_compatible_euler_{num_steps}",
                                        legacy_prefix=legacy_prefix,
                                        global_step=getattr(self, "iter", None),
                                    )
                                )
                            else:
                                compatible_state = self._compatible_loss_state(
                                    self.validation_lossfunc,
                                    sampled,
                                    batch_for_loss,
                                    prefix=f"validation_compatible_euler_{num_steps}",
                                    legacy_prefix=legacy_prefix,
                                )
                            if int(num_steps) == 1:
                                self._require_endpoint_triplet(
                                    compatible_state,
                                    prefix="validation",
                                    route="MeanFlow validation",
                                )
                            self._accumulate_metric_state(
                                flow_metric_sums,
                                compatible_state,
                                flow_metric_counts,
                            )
                        num_batches += 1
                        continue
                    if log_random_t:
                        prepare_kwargs = {}
                        if validation_prior_seed is not None:
                            prepare_kwargs["prior_seed"] = validation_prior_seed
                        validation_seed = getattr(self.flow_cfm, "validation_seed", None)
                        if callable(validation_seed):
                            prepare_kwargs["time_seed"] = validation_seed(
                                num_batches, "time"
                            )
                        flow_batch, flow_ref, flow_ctx = self.flow_cfm.prepare_batch(
                            original_batch,
                            batch_for_loss,
                            **prepare_kwargs,
                        )
                        flow_pred = self.model(flow_batch)
                        random_t_loss, _ = self.flow_cfm.loss(flow_pred, flow_ref, flow_ctx)
                        loss += random_t_loss
                        flow_metric_sums["validation_flow_random_t_loss"] = (
                            flow_metric_sums.get("validation_flow_random_t_loss", 0.0)
                            + random_t_loss.detach()
                        )

                    num_graphs = self.flow_cfm._num_graphs(original_batch)
                    zero_t = torch.zeros(num_graphs, device=self.device, dtype=self.dtype)
                    t0_ref = None
                    t0_ctx = None
                    if log_t0 or log_flow_euler:
                        prepare_kwargs = {"t": zero_t}
                        if validation_prior_seed is not None:
                            prepare_kwargs["prior_seed"] = validation_prior_seed
                        t0_batch, t0_ref, t0_ctx = self.flow_cfm.prepare_batch(
                            original_batch,
                            batch_for_loss,
                            **prepare_kwargs,
                        )
                        if log_t0:
                            t0_pred = self.model(t0_batch)
                            t0_pred.update(batch_info)
                            t0_ref.update(batch_info)
                            t0_loss, _ = self.flow_cfm.loss(t0_pred, t0_ref, t0_ctx)
                            flow_metric_sums["validation_flow_t0_loss"] = (
                                flow_metric_sums.get("validation_flow_t0_loss", 0.0)
                                + t0_loss.detach()
                            )
                    for num_steps in self.flow_cfm.validation_ode_steps:
                        sample_kwargs = {"num_steps": num_steps}
                        if validation_prior_seed is not None:
                            sample_kwargs["prior_seed"] = validation_prior_seed
                        sampled = self.flow_cfm.sample(
                            self.model, original_batch, **sample_kwargs
                        )
                        sampled.update(batch_info)
                        sample_state = None
                        block_ode = bool(getattr(self.flow_cfm, "block_ode", False))
                        if block_ode:
                            sample_loss, sample_state = (
                                self._score_block_ode_validation_sample(
                                    sampled,
                                    original_batch,
                                    batch_for_loss,
                                    batch_info,
                                    prior_seed=validation_prior_seed,
                                    trajectory=log_flow_euler,
                                )
                            )
                            if log_flow_euler:
                                self._accumulate_metric_state(
                                    flow_metric_sums,
                                    {
                                        f"validation_flow_euler_{num_steps}_loss": sample_loss
                                    },
                                    flow_metric_counts,
                                )
                        elif log_flow_euler:
                            if t0_ref is None or t0_ctx is None:
                                prepare_kwargs = {"t": zero_t}
                                if validation_prior_seed is not None:
                                    prepare_kwargs["prior_seed"] = validation_prior_seed
                                _t0_batch, t0_ref, t0_ctx = self.flow_cfm.prepare_batch(
                                    original_batch,
                                    batch_for_loss,
                                    **prepare_kwargs,
                                )
                                t0_ref.update(batch_info)
                            sample_scorer = getattr(
                                self.flow_cfm,
                                "loss_on_sample",
                                self.flow_cfm.loss,
                            )
                            sample_loss, sample_state = sample_scorer(
                                sampled, t0_ref, t0_ctx
                            )
                            self._accumulate_metric_state(
                                flow_metric_sums,
                                {
                                    f"validation_flow_euler_{num_steps}_loss": sample_loss
                                },
                                flow_metric_counts,
                            )
                        legacy_prefix = "validation" if int(num_steps) == 1 else None
                        compatible_state = None
                        if sample_state is not None:
                            compatible_state = self._compatible_loss_state_from_flow_stats(
                                self.validation_lossfunc,
                                sample_state,
                                source_prefix="train",
                                prefix=f"validation_compatible_euler_{num_steps}",
                                legacy_prefix=legacy_prefix,
                                global_step=getattr(self, "iter", None),
                            )
                        if compatible_state is None and not block_ode:
                            compatible_ref = t0_ref if t0_ref is not None else batch_for_loss.copy()
                            compatible_ref.update(batch_info)
                            compatible_state = self._compatible_loss_state(
                                self.validation_lossfunc,
                                sampled,
                                compatible_ref,
                                prefix=f"validation_compatible_euler_{num_steps}",
                                legacy_prefix=legacy_prefix,
                            )
                        if int(num_steps) == 1:
                            self._require_endpoint_triplet(
                                compatible_state,
                                prefix="validation",
                                route="CFM validation",
                            )
                        self._accumulate_metric_state(
                            flow_metric_sums,
                            compatible_state,
                            flow_metric_counts,
                        )
                else:
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
                        flow_metric_sums,
                        endpoint_state,
                        flow_metric_counts,
                    )
                num_batches += 1
                if fast: break
        divisor = max(num_batches, 1)
        if not fast:
            loss = loss / divisor
        self._last_flow_validation_state = {
            # Per-key valid-batch count (filled in lock-step with flow_metric_sums
            # by _accumulate_metric_state): a throttleable feature-compatible metric
            # omitted by _loss_component_state on a non-firing batch accumulated a
            # smaller sum, so dividing it by the uniform num_batches would dilute it
            # (2.0 over one firing batch of two -> 1.0).  Divide each accumulated key
            # by ITS OWN contributing-batch count instead.  Keys written directly to
            # flow_metric_sums (validation_flow_random_t/t0/euler_* losses) never
            # enter flow_metric_counts and fall back to ``divisor``; keys present on
            # every batch have count == num_batches == divisor, so the fallback and
            # the per-key path give the identical divisor and interval=1 stays
            # byte-identical.
            key: value / flow_metric_counts.get(key, divisor)
            for key, value in flow_metric_sums.items()
        }
        # When the endpoint-compatible pass produced a legacy validation_loss,
        # return it: direct callers and scheduler metrics then see the same
        # no-CFM/CFM-comparable scalar that Validationer reports, matching
        # MultiTrainer.validation semantics. The flow objective stays under
        # validation_flow_* keys.  When the legacy key is absent (a legal config
        # such as validation_ode_steps=[3] with the three log_validation_* flags
        # disabled writes only validation_compatible_euler_{n}_loss), fail closed
        # to the smallest-n endpoint-compatible loss instead of the accumulated
        # ``loss`` (which stays 0.0 when no optimization-loss branch ran and would
        # otherwise be read by schedulers/best-checkpoint as a perfect score).
        return self._resolve_validation_return(loss)

    def _resolve_validation_return(self, accumulated_loss):
        """Fail-closed selection of the scalar ``validation()`` returns.

        Preference order (operates on ``self._last_flow_validation_state``, which
        ``validation()`` has just rebuilt from this run's metric sums):

        1. the legacy ``validation_loss`` key -- byte-identical to the historical
           behavior whenever the ``num_steps == 1`` endpoint-compatible pass ran;
        2. otherwise the endpoint-compatible euler loss with the SMALLEST number
           of ODE steps (``validation_compatible_euler_{n}_loss``);
        3. otherwise the accumulated ``loss`` (no compatible metric available).
        """
        state = getattr(self, "_last_flow_validation_state", None) or {}
        if "validation_loss" in state:
            return state["validation_loss"]
        prefix = "validation_compatible_euler_"
        suffix = "_loss"
        best_n = None
        best_key = None
        for key in state:
            if key.startswith(prefix) and key.endswith(suffix):
                middle = key[len(prefix):-len(suffix)]
                if middle.isdigit():
                    n = int(middle)
                    if best_n is None or n < best_n:
                        best_n = n
                        best_key = key
        if best_key is not None:
            return state[best_key]
        return accumulated_loss


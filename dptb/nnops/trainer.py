import torch
import logging
import os
import csv
import math
import copy
import torch.nn as nn
from dptb.utils.tools import (
    get_lr_scheduler,
    get_optimizer,
    lr_scheduler_can_step_without_metric,
    lr_scheduler_requires_metric,
)
from dptb.nnops.base_trainer import BaseTrainer
from dptb.plugins.monitor import Plugin
from typing import Union, Optional
from dptb.data import AtomicDataset, DataLoader, AtomicData
from dptb.nn import build_model
from dptb.nn.activation_recompute import configure_activation_recompute
from dptb.nnops.flow import assert_flow_h0_keys_reach_model, build_hamiltonian_flow
from dptb.nnops.loss import Loss
from dptb.nnops.self_consistency import (
    SelfConsistencyScheduler,
    SelfConsistencySchedulerConfig,
    compute_self_consistency_payload_loss,
)

log = logging.getLogger(__name__)


class Trainer(BaseTrainer):
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

    _COMPATIBLE_LOSS_SIDE_EFFECT_ATTRS = (
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
            if compatible_state is not None:
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
    def _compatible_loss_state(lossfunc, pred_data, ref_data, *, prefix, legacy_prefix=None):
        loss_obj = Trainer._loss_component_source(lossfunc)
        sentinel = object()
        saved_side_effects = {
            attr: getattr(loss_obj, attr, sentinel)
            for attr in Trainer._COMPATIBLE_LOSS_SIDE_EFFECT_ATTRS
        }

        try:
            with torch.no_grad():
                compatible_loss = lossfunc(pred_data, ref_data)

            state = {
                f"{prefix}_loss": compatible_loss.detach()
                if torch.is_tensor(compatible_loss)
                else torch.as_tensor(compatible_loss)
            }
            state.update(Trainer._loss_component_state(lossfunc, prefix=prefix))
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
        loss_for_log = loss.detach()
        loss.backward()
        del loss

        ref_component_state = {}
        if ref_batch is not None:
            reference_lossfunc = getattr(self, "reference_lossfunc", self.train_lossfunc)
            ref_loss = self._loss_on_batch(
                ref_batch,
                reference_lossfunc,
                use_flow=self.flow_cfm.apply_to_reference,
                allow_self_consistency=False,
            )
            loss_for_log = loss_for_log + ref_loss.detach()
            ref_loss.backward()
            ref_component_state = self._loss_component_state(reference_lossfunc, prefix="ref")
            del ref_loss

        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.clip_grad_norm
        )

        self.optimizer.step()

        if self.update_lr_per_iter:
            if lr_scheduler_requires_metric(self.lr_scheduler):
                if self.iter > 1:
                    self.lr_scheduler.step(self.stats["train_loss"]['latest_avg_iter_loss'])
                elif lr_scheduler_can_step_without_metric(self.lr_scheduler):
                    self.lr_scheduler.step()
            else:
                self.lr_scheduler.step()

        state = {
            'field': 'iteration',
            "train_loss": loss_for_log,
            "lr": self.optimizer.state_dict()["param_groups"][0]['lr'],
            "total_grad_norm": total_norm.item()
        }
        self._add_effective_expert_lr_state(
            state,
            optimizer=self.optimizer,
            num_experts=getattr(self, "num_experts", getattr(self.model, "num_experts", 0)),
        )
        state.update(dynamic_batch_state)
        if not self.flow_cfm.enabled:
            state.update(self._loss_component_state(self.train_lossfunc))
        state.update(ref_component_state)
        state.update(getattr(self, "_last_flow_state", {}))
        state.update(getattr(self, "_last_self_consistency_state", {}))
        if self._optimizer_diagnostics_due():
            state.update(self._optimizer_diagnostics(self.optimizer))

        self.call_plugins(queue_name='iteration', time=self.iter, **state)
        self.iter += 1

        return loss_for_log

    @classmethod
    def restart(cls, checkpoint, train_datasets, train_options={}, common_options={}, reference_datasets=None,
                validation_datasets=None):
        ckpt = torch.load(checkpoint, map_location=common_options["device"], weights_only=False)
        ckpt_train_options = ckpt["config"]["train_options"]
        model_build_train_options = train_options if len(train_options) != 0 else ckpt_train_options
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
            ckpt["config"]["common_options"],
            train_options=model_build_train_options,
        )
        if len(train_options) == 0: train_options = ckpt["config"]["train_options"]
        if len(common_options) == 0: common_options = ckpt["config"]["common_options"]
        trainer = cls(model=model, train_datasets=train_datasets, reference_datasets=reference_datasets,
                      validation_datasets=validation_datasets, train_options=train_options,
                      common_options=common_options)
        trainer.ep = ckpt["epoch"] + 1
        trainer.iter = ckpt["iteration"] + 1
        trainer.stats = ckpt["stats"]
        queues_name = list(trainer.plugin_queues.keys())
        for unit in queues_name:
            for plugin in trainer.plugin_queues[unit]:
                plugin = (getattr(trainer, unit) + plugin[0], plugin[1], plugin[2])
        for key in Trainer.object_keys:
            item = getattr(trainer, key, None)
            if item is not None: item.load_state_dict(ckpt[key + "_state_dict"])
        return trainer

    def epoch(self) -> None:
        batch_sampler = getattr(self.train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(int(self.ep))
        if self.use_reference:
            ref_batch_sampler = getattr(self.reference_loader, "batch_sampler", None)
            if hasattr(ref_batch_sampler, "set_epoch"):
                ref_batch_sampler.set_epoch(int(self.ep))
        reference_iter = iter(self.reference_loader) if self.use_reference else None
        for ibatch in self.train_loader:
            if self.use_reference:
                try:
                    ref_batch = next(reference_iter)
                except StopIteration:
                    reference_iter = iter(self.reference_loader)
                    ref_batch = next(reference_iter)
                self.iteration(ibatch, ref_batch)
            else:
                self.iteration(ibatch)

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
                    validation_prior_seed = (
                        self.flow_cfm.validation_seed(num_batches, "prior")
                        if getattr(self.flow_cfm, "prior", "zero") == "projected_te"
                        else None
                    )
                    log_random_t = getattr(self.flow_cfm, "log_validation_random_t_loss", True)
                    log_t0 = getattr(self.flow_cfm, "log_validation_t0_loss", True)
                    log_flow_euler = getattr(
                        self.flow_cfm, "log_validation_flow_euler_loss", True
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
                            self._accumulate_metric_state(
                                flow_metric_sums,
                                self._compatible_loss_state(
                                    self.validation_lossfunc,
                                    sampled,
                                    batch_for_loss,
                                    prefix=f"validation_compatible_euler_{num_steps}",
                                    legacy_prefix=legacy_prefix,
                                ),
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
                        if log_flow_euler:
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
                            sample_loss, sample_state = self.flow_cfm.loss_on_sample(
                                sampled, t0_ref, t0_ctx
                            )
                            key = f"validation_flow_euler_{num_steps}_loss"
                            flow_metric_sums[key] = (
                                flow_metric_sums.get(key, 0.0) + sample_loss.detach()
                            )
                        legacy_prefix = "validation" if int(num_steps) == 1 else None
                        compatible_state = None
                        if sample_state is not None:
                            compatible_state = self._compatible_loss_state_from_flow_stats(
                                self.validation_lossfunc,
                                sample_state,
                                source_prefix=f"validation_compatible_euler_{num_steps}",
                                prefix=f"validation_compatible_euler_{num_steps}",
                                legacy_prefix=legacy_prefix,
                                global_step=getattr(self, "iter", None),
                            )
                        if compatible_state is None:
                            compatible_ref = t0_ref if t0_ref is not None else batch_for_loss.copy()
                            compatible_ref.update(batch_info)
                            compatible_state = self._compatible_loss_state(
                                self.validation_lossfunc,
                                sampled,
                                compatible_ref,
                                prefix=f"validation_compatible_euler_{num_steps}",
                                legacy_prefix=legacy_prefix,
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
                    loss += batch_loss
                    self._accumulate_metric_state(
                        flow_metric_sums,
                        self._loss_component_state(
                            self.validation_lossfunc,
                            prefix="validation",
                        ),
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


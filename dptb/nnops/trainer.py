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
from dptb.nnops.flow import build_hamiltonian_flow
from dptb.nnops.loss import Loss

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
        super(Trainer, self).__init__(dtype=common_options["dtype"], device=common_options["device"])

        # init the object
        self.model = model.to(self.device)
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
            self.validation_loader = DataLoader(dataset=self.validation_datasets,
                                                batch_size=train_options["val_batch_size"], shuffle=True)

        # loss function
        self.train_lossfunc = Loss(
            **self._loss_kwargs(train_options["loss_options"]["train"], common_options),
            idp=self.model.hamiltonian.idp,
        )
        if self.use_validation:
            self.validation_lossfunc = Loss(
                **self._loss_kwargs(train_options["loss_options"]["validation"], common_options),
                idp=self.model.hamiltonian.idp,
            )
        if self.use_reference:
            self.reference_lossfunc = Loss(
                **self._loss_kwargs(train_options["loss_options"]["reference"], common_options),
                idp=self.model.hamiltonian.idp,
            )

        self.flow_cfm = build_hamiltonian_flow(
            train_options.get("flow_options", None),
            idp=self.model.hamiltonian.idp,
            dtype=self.dtype,
            device=self.device,
        )
        self._last_flow_state = {}
        self._last_flow_validation_state = {}

        if train_options["loss_options"]["train"]["method"] == "skints":
            assert self.model.name == 'nnsk', "The model should be nnsk for the skints loss function."
            assert self.model.onsite_fn.functype in ['none',
                                                     'uniform'], "The onsite function should be none or uniform for the skints loss function."
            log.info("The skints loss function is used for training, the model.transform is then set to False.")
            self.model.transform = False

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

    def _loss_on_batch(self, batch, lossfunc, *, use_flow=True):
        batch = batch.to(self.device)
        batch_info = self._batch_info(batch)
        batch = AtomicData.to_AtomicDataDict(batch)
        batch_for_loss = batch.copy()
        if use_flow and self.flow_cfm.enabled:
            if getattr(self.flow_cfm, "model_in_loss", False):
                loss, flow_state = self.flow_cfm.loss_with_model(self.model, batch, batch_for_loss)
            else:
                batch, batch_for_loss, flow_ctx = self.flow_cfm.prepare_batch(batch, batch_for_loss)
                batch = self.model(batch)
                batch.update(batch_info)
                batch_for_loss.update(batch_info)
                loss, flow_state = self.flow_cfm.loss(batch, batch_for_loss, flow_ctx)
            if self.flow_cfm.log_train_compatible_loss and not getattr(
                self.flow_cfm, "model_in_loss", False
            ):
                flow_state.update(
                    self._compatible_loss_state(
                        lossfunc,
                        batch,
                        batch_for_loss,
                        prefix="train_compatible",
                        legacy_prefix=(
                            "train" if self.flow_cfm.compatible_loss_to_legacy_keys else None
                        ),
                    )
                )
            self._last_flow_state = flow_state
            return loss
        self._last_flow_state = {}
        batch = self.model(batch)
        batch.update(batch_info)
        batch_for_loss.update(batch_info)
        return lossfunc(batch, batch_for_loss)

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
            onsite_key = f"{prefix}_onsite_loss"
            hopping_key = f"{prefix}_hopping_loss"
            if onsite_key in state:
                state[f"{legacy_prefix}_onsite_loss"] = state[onsite_key]
            if hopping_key in state:
                state[f"{legacy_prefix}_hopping_loss"] = state[hopping_key]
        return state

    @staticmethod
    def _accumulate_metric_state(metric_sums, state):
        for key, value in state.items():
            if torch.is_tensor(value):
                value = value.detach()
            metric_sums[key] = metric_sums.get(key, 0.0) + value

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
        state.update(dynamic_batch_state)
        if not self.flow_cfm.enabled:
            state.update(self._loss_component_state(self.train_lossfunc))
        state.update(ref_component_state)
        state.update(getattr(self, "_last_flow_state", {}))
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
            num_batches = 0
            self.model.eval()
            for batch in self.validation_loader:
                batch = batch.to(self.device)
                batch_info = {"__slices__": batch.__slices__, "__cumsum__": batch.__cumsum__,
                              "__cat_dims__": batch.__cat_dims__, "__num_nodes_list__": batch.__num_nodes_list__,
                              "__data_class__": batch.__data_class__}
                batch = AtomicData.to_AtomicDataDict(batch)
                batch_for_loss = batch.copy()
                if self.flow_cfm.enabled:
                    original_batch = batch.copy()
                    if getattr(self.flow_cfm, "model_in_loss", False):
                        batch_for_loss.update(batch_info)
                        random_t_loss, _ = self.flow_cfm.loss_with_model(
                            self.model, original_batch, batch_for_loss, prefix="validation"
                        )
                        loss += random_t_loss
                        flow_metric_sums["validation_flow_random_t_loss"] = (
                            flow_metric_sums.get("validation_flow_random_t_loss", 0.0)
                            + random_t_loss.detach()
                        )
                        num_graphs = self.flow_cfm._num_graphs(original_batch)
                        zero_t = torch.zeros(num_graphs, device=self.device, dtype=self.dtype)
                        one_t = torch.ones(num_graphs, device=self.device, dtype=self.dtype)
                        one_step_loss, _ = self.flow_cfm.loss_with_model(
                            self.model,
                            original_batch,
                            batch_for_loss,
                            prefix="validation_one_step",
                            r=zero_t,
                            t=one_t,
                        )
                        flow_metric_sums["validation_flow_one_step_loss"] = (
                            flow_metric_sums.get("validation_flow_one_step_loss", 0.0)
                            + one_step_loss.detach()
                        )
                        for num_steps in self.flow_cfm.validation_ode_steps:
                            sampled = self.flow_cfm.sample(
                                self.model, original_batch, num_steps=num_steps
                            )
                            sampled.update(batch_info)
                            if self.flow_cfm.log_validation_compatible_loss:
                                self._accumulate_metric_state(
                                    flow_metric_sums,
                                    self._compatible_loss_state(
                                        self.validation_lossfunc,
                                        sampled,
                                        batch_for_loss,
                                        prefix=f"validation_compatible_euler_{num_steps}",
                                        legacy_prefix="validation" if int(num_steps) == 1 else None,
                                    ),
                                )
                        num_batches += 1
                        continue
                    flow_batch, flow_ref, flow_ctx = self.flow_cfm.prepare_batch(
                        original_batch, batch_for_loss
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
                    t0_batch, t0_ref, t0_ctx = self.flow_cfm.prepare_batch(
                        original_batch, batch_for_loss, t=zero_t
                    )
                    t0_pred = self.model(t0_batch)
                    t0_pred.update(batch_info)
                    t0_ref.update(batch_info)
                    t0_loss, _ = self.flow_cfm.loss(t0_pred, t0_ref, t0_ctx)
                    flow_metric_sums["validation_flow_t0_loss"] = (
                        flow_metric_sums.get("validation_flow_t0_loss", 0.0) + t0_loss.detach()
                    )
                    for num_steps in self.flow_cfm.validation_ode_steps:
                        sampled = self.flow_cfm.sample(
                            self.model, original_batch, num_steps=num_steps
                        )
                        sampled.update(batch_info)
                        sample_loss, _ = self.flow_cfm.loss(sampled, t0_ref, t0_ctx)
                        key = f"validation_flow_euler_{num_steps}_loss"
                        flow_metric_sums[key] = (
                            flow_metric_sums.get(key, 0.0) + sample_loss.detach()
                        )
                        if self.flow_cfm.log_validation_compatible_loss:
                            self._accumulate_metric_state(
                                flow_metric_sums,
                                self._compatible_loss_state(
                                    self.validation_lossfunc,
                                    sampled,
                                    t0_ref,
                                    prefix=f"validation_compatible_euler_{num_steps}",
                                    legacy_prefix="validation" if int(num_steps) == 1 else None,
                                ),
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
                    )
                num_batches += 1
                if fast: break
        divisor = max(num_batches, 1)
        if not fast:
            loss = loss / divisor
        self._last_flow_validation_state = {
            key: value / divisor for key, value in flow_metric_sums.items()
        }
        return loss


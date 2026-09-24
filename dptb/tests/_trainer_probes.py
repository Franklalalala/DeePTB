"""Shared trainer doubles for the training-loop, restart, checkpoint and OOM tests.

Not collected by pytest; import as ``from dptb.tests._trainer_probes import ...``.

* ``make_fake_trainer`` -- a single-process ``Trainer`` whose real ``iteration()``
  runs on a one-parameter model and records every plugin call.
* ``ProbeTrainer`` / ``make_probe_trainer`` -- the real ``Trainer`` epoch loop,
  fast-forward cursor, ``restart()`` and ``Saver`` path with a stubbed batch step.
* ``DistPathProbeTrainer`` / ``make_dist_probe`` -- the real ``MultiTrainer``
  distributed-expert iteration path in one process (no process group, so every
  collective is skipped) with a stubbed loss payload.
* ``blockwise_payload`` / ``blockwise_criterion`` -- an H-O block payload for the real
  ``HamilBlockwiseNexTHamLoss``.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from dptb.nnops import trainer as trainer_mod
from dptb.nnops.multi_trainer import MultiTrainer, _StageTagger
from dptb.nnops.trainer import Trainer
from dptb.plugins.base_plugin import Plugin, PluginUser
from dptb.plugins.monitor import LearningRateMonitor, TrainLossMonitor, Validationer
from dptb.plugins.saver import Saver


class CountingSGD(torch.optim.SGD):
    """SGD that counts committed optimizer steps."""

    def __init__(self, params, lr=0.1):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


class _NaNGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("nan"))


def corrupt(loss, kind):
    """Make ``loss`` non-finite (``"nan"``/``"inf"``) or give it a NaN gradient (``"gradient"``)."""
    return _NaNGradient.apply(loss) if kind == "gradient" else loss * float(kind)


# --------------------------------------------------------------------------
# single-process Trainer with the real iteration()
# --------------------------------------------------------------------------
class FakeBatch:
    """Batch accepted by ``Trainer._loss_on_batch``; ``batch_cost`` adds dynamic-batch metadata."""

    def __init__(self, name, x, batch_cost=None):
        self.name = name
        self.x = torch.tensor(float(x))
        self.__slices__ = {}
        self.__cumsum__ = {}
        self.__cat_dims__ = {}
        self.__num_nodes_list__ = []
        self.__data_class__ = FakeBatch
        if batch_cost is not None:
            self.__dptb_batch_cost__ = batch_cost
            self.__dptb_batch_num_graphs__ = 1
            self.__dptb_batch_num_nodes__ = 2
            self.__dptb_batch_num_edges__ = 3
            self.__dptb_batch_max_item_cost__ = batch_cost

    def to(self, device):
        return self


def batch_to_dict(batch):
    return {"name": batch.name, "x": batch.x}


class ScaleModel(nn.Module):
    """``pred = weight * x`` with ``weight`` initialised to 1."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        batch = dict(batch)
        batch["pred"] = self.weight * batch["x"]
        return batch


class RecordingLoss:
    """Hamiltonian-style criterion; its component side effects encode the batch value x.

    onsite = 10x + 1, hopping = 10x + 2, z = 10x + 3, load_cv = 10x + 4; the loss is ``pred``.
    """

    def __init__(self):
        self.calls = []
        self.last_onsite_loss = None
        self.last_hopping_loss = None
        self.last_z_loss = None
        self.expert_load_cv = None

    def __call__(self, batch, batch_for_loss):
        self.calls.append(batch["name"])
        base = 10.0 * float(batch["x"])
        self.last_onsite_loss = torch.tensor(base + 1.0)
        self.last_hopping_loss = torch.tensor(base + 2.0)
        self.last_z_loss = torch.tensor(base + 3.0)
        self.expert_load_cv = torch.tensor(base + 4.0)
        return batch["pred"].sum()


class ScalarLoss:
    """A non-Hamiltonian criterion with no endpoint component side effects."""

    def __init__(self):
        self.calls = []

    def __call__(self, batch, batch_for_loss):
        self.calls.append(batch["name"])
        return batch["pred"].sum()


class DistinctEndpointLoss:
    """A criterion whose optimized objective and public endpoint loss differ."""

    supports_endpoint_triplet = True

    def __init__(self, endpoint_loss):
        self.endpoint_loss = float(endpoint_loss)

    def __call__(self, batch, batch_for_loss):
        objective = batch["pred"].sum()
        self.last_endpoint_loss = objective.detach().new_tensor(self.endpoint_loss)
        self.last_onsite_loss = objective.detach().new_tensor(self.endpoint_loss + 1.0)
        self.last_hopping_loss = objective.detach().new_tensor(self.endpoint_loss + 2.0)
        return objective


class RecordingMetricScheduler:
    """An LR scheduler that records every metric passed to ``step``."""

    requires_metric = True

    def __init__(self):
        self.metrics = []

    def step(self, metric):
        self.metrics.append(float(metric))


def make_validation_trainer(monkeypatch, lossfunc):
    """A ``Trainer`` wired only for the non-flow ``validation()`` single-batch path."""
    monkeypatch.setattr(trainer_mod.AtomicData, "to_AtomicDataDict", batch_to_dict)
    trainer = Trainer.__new__(Trainer)
    trainer.model = ScaleModel()
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.validation_loader = [FakeBatch("validation", 3.0)]
    trainer.validation_lossfunc = lossfunc
    trainer.flow_cfm = SimpleNamespace(enabled=False)
    return trainer


def make_fake_trainer(monkeypatch):
    """Return ``(trainer, plugin_calls)``; ``plugin_calls`` collects ``(queue, time, state)``.

    The trainer starts at optimizer step 5 with separate ``RecordingLoss`` criteria for
    train and reference batches.
    """
    monkeypatch.setattr(trainer_mod.AtomicData, "to_AtomicDataDict", batch_to_dict)
    model = ScaleModel()
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.optimizer = CountingSGD(model.parameters())
    trainer.device = torch.device("cpu")
    trainer.clip_grad_norm = 1000.0
    trainer.update_lr_per_iter = False
    trainer.iter = 5
    trainer.stats = {"train_loss": {"latest_avg_iter_loss": torch.tensor(0.0)}}
    trainer.flow_cfm = SimpleNamespace(enabled=False)
    trainer.train_lossfunc = RecordingLoss()
    trainer.reference_lossfunc = RecordingLoss()
    plugin_calls = []
    trainer.call_plugins = lambda queue_name, time, **state: plugin_calls.append((queue_name, time, state))
    return trainer, plugin_calls


# --------------------------------------------------------------------------
# restart / checkpoint probe on the real Trainer
# --------------------------------------------------------------------------
class ProbeModel(nn.Module):
    name = "probe"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.model_options = {"embedding": {}, "prediction": {}}


class ProbeTrainer(Trainer):
    """Real ``Trainer.epoch/run/restart`` with a stubbed one-batch step.

    ``processed`` holds ``(epoch, batch)`` for every optimized batch, ``ref_seen`` the paired
    reference batches and ``rng_trace`` one torch random draw per batch.  The epoch loss
    reported to plugins is ``_epoch_losses[epoch]`` (default ``100 - epoch``).
    """

    def __init__(self, *, model, train_datasets, reference_datasets=None,
                 validation_datasets=None, train_options, common_options):
        PluginUser.__init__(self)
        self.iter = 1
        self.ep = 1
        self._batch_in_epoch = 0
        self._resume_plan = None
        self.update_lr_per_iter = bool(train_options.get("update_lr_per_iter", False))
        self.dtype = torch.float32
        self.device = "cpu"
        self.model = model
        self.common_options = common_options
        self.train_options = train_options
        self.task = "hamiltonians"
        # a finite list is a fixed-order loader, so a mid-epoch fast-forward is exact
        self.train_datasets = train_datasets
        self.train_loader = list(train_datasets)
        self.use_reference = reference_datasets is not None
        if self.use_reference:
            self.reference_loader = list(reference_datasets)
        self.use_validation = False
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.processed = []
        self.ref_seen = []
        self.rng_trace = []
        self._epoch_losses = train_options.get("_epoch_losses", {})

    def iteration(self, ibatch, ref_batch=None):
        self.processed.append((int(self.ep), ibatch))
        if ref_batch is not None:
            self.ref_seen.append(ref_batch)
        self.rng_trace.append(round(float(torch.rand(1).item()), 6))
        self.optimizer.step()
        self._batch_in_epoch = getattr(self, "_batch_in_epoch", 0) + 1
        loss = self._epoch_losses.get(int(self.ep), 100.0 - int(self.ep))
        self.stats.setdefault("train_loss", {})["epoch_mean"] = float(loss)
        self.call_plugins(queue_name="iteration", time=self.iter)
        self.iter += 1
        return torch.tensor(float(loss))

    def validation(self, fast=True):
        return torch.tensor(0.0)


def make_probe_trainer(n_batches=4, *, epoch_losses=None, reference=None, seed=0):
    torch.manual_seed(seed)
    return ProbeTrainer(
        model=ProbeModel(),
        train_datasets=list(range(n_batches)),
        reference_datasets=reference,
        train_options={"max_ckpt": 50, "update_lr_per_iter": False,
                       "_epoch_losses": epoch_losses or {}},
        common_options={"device": "cpu", "dtype": "float32"},
    )


def restart_probe(checkpoint, monkeypatch, n_batches, **kwargs):
    """``ProbeTrainer.restart`` from a real checkpoint file; only ``build_model`` is stubbed."""
    monkeypatch.setattr(trainer_mod, "build_model", lambda *args, **kw: ProbeModel())
    return ProbeTrainer.restart(
        str(checkpoint),
        train_datasets=list(range(n_batches)),
        train_options={"max_ckpt": 50, "update_lr_per_iter": False},
        common_options={"device": "cpu", "dtype": "float32"},
        **kwargs,
    )


# --------------------------------------------------------------------------
# MultiTrainer distributed-expert path in one process
# --------------------------------------------------------------------------
class _ProbeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class ExpertsProbeModel(nn.Module):
    name = "probe"

    def __init__(self, num_experts=1):
        super().__init__()
        self.experts = nn.ModuleList([_ProbeExpert() for _ in range(num_experts)])
        self.model_options = {"embedding": {}, "prediction": {}}


class StubBatch:
    """Carries the dynamic-batch attributes the trainer forwards to plugins."""

    def __init__(self, idx):
        self.__dptb_batch_cost__ = 10.0 + idx
        self.__dptb_batch_num_graphs__ = 2


class DistPathProbeTrainer(MultiTrainer):
    """Real ``MultiTrainer`` iteration with a stubbed batch bundle and loss payload.

    ``distributed_expert=True`` routes ``iteration()`` through the distributed-expert path;
    with no process group initialised every collective helper is a no-op.  Set
    ``distributed_expert = False`` for the single-process path.  Each expert's loss is
    ``weight ** 2``; ``train_losses`` records every value and ``validation_calls`` the steps
    at which ``validation()`` ran.
    """

    def __init__(self, *, save_freq=3, display_freq=100, num_experts=1):
        PluginUser.__init__(self)
        self.iter = 1
        self.ep = 1
        self._batch_in_epoch = 0
        self._resume_plan = None
        self.update_lr_per_iter = False
        self.dtype = torch.float32
        self.device = "cpu"

        self.model = ExpertsProbeModel(num_experts)
        self.common_options = {"device": "cpu", "dtype": "float32"}
        self.train_options = {"max_ckpt": 50, "save_freq": save_freq, "display_freq": display_freq}
        self.task = "hamiltonians"

        # distributed-expert layout of rank 0 in a world of one
        self.distributed_expert = True
        self.distributed_rank0_prepare_batch = False
        self.rank = 0
        self.world_size = 1
        self.is_main_process = True
        self.expert_data_parallel_size = 1
        self.local_expert_idx = 0
        self.expert_dp_rank = 0
        self.expert_group_ranks = [0]
        self.expert_group_src_rank = 0
        self.expert_dp_process_group = None
        self.expert_dp_backend = "manual"
        self.num_experts = num_experts
        self.distance_ranges = [(0.0, 10.0)] * num_experts

        self.display_sync_freq = max(int(display_freq), 1)
        self.clip_grad_norm = 1e9
        self.monitor_cuda_memory = False
        self.debug_tags = False
        self.debug_tag_freq = 1
        self.debug_profile = False
        self._t_last_iter_end = None

        self.optimizers = [CountingSGD(e.parameters()) for e in self.model.experts]
        # real (epoch-cadence) schedulers so Saver can serialize their state
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=1.0)
            for opt in self.optimizers
        ]
        self.train_lossfunc = None
        self.use_reference = False
        self.use_validation = False

        self._tagger = _StageTagger(self, enabled=False, freq=1, cuda_mem=False,
                                    cuda_sync=False, oom_dump=False)
        self._reset_display_window_buffers()

        self.validation_calls = []
        self.train_losses = []

    def _prepare_batch_bundle(self, batch, with_lengths=True):
        return {}, {}

    def _build_train_payload(self, *, batch_dict, batch_info, expert_idx, range_dis,
                             ref_batch_dict=None, ref_batch_info=None, criterion=None):
        expert = self.model.experts[expert_idx]
        loss = (expert.weight ** 2).sum()
        self.train_losses.append(float(loss.detach().item()))
        return {
            "loss": loss,
            "expert_onsite": 0.1,
            "expert_hopping": 0.2,
            "active_nodes": 1.0,
            "active_edges": 1.0,
            "onsite_weighted_sum": 0.1,
            "hopping_weighted_sum": 0.2,
            "onsite_l1_sum": 0.0,
            "onsite_mse_sum": 0.0,
            "onsite_cnt": 0.0,
            "hopping_l1_sum": 0.0,
            "hopping_mse_sum": 0.0,
            "hopping_cnt": 0.0,
            "z_values": [],
            "load_cv_values": [],
        }

    def validation(self, fast=True, **kwargs):
        self.validation_calls.append(int(self.iter))
        return torch.tensor(0.5)


class StateRecordingPlugin(Plugin):
    """``(1, 'iteration')`` plugin recording ``(time, state)`` for every tick."""

    def __init__(self):
        super().__init__([(1, "iteration")])
        self.ticks = []

    def register(self, trainer):
        self.trainer = trainer

    def iteration(self, **kwargs):
        self.ticks.append((kwargs.get("time"), dict(kwargs)))


def make_dist_probe(tmp_path, *, save_freq=3, display_freq=100, with_saver=True,
                    validation_freq=None, monitors=True):
    """Return ``(trainer, recorder)``; a Saver writes to ``tmp_path / "ck"`` when requested."""
    trainer = DistPathProbeTrainer(save_freq=save_freq, display_freq=display_freq)
    recorder = StateRecordingPlugin()
    trainer.register_plugin(recorder)
    if monitors:
        trainer.register_plugin(TrainLossMonitor())
        trainer.register_plugin(LearningRateMonitor())
    if validation_freq:
        trainer.register_plugin(Validationer(interval=[(validation_freq, "iteration")], fast_mode=True))
    if with_saver:
        ckpt_dir = tmp_path / "ck"
        ckpt_dir.mkdir(exist_ok=True)
        trainer.register_plugin(Saver(interval=[(save_freq, "iteration")]), checkpoint_path=str(ckpt_dir))
    trainer.rebase_plugin_cadence()
    return trainer, recorder


# --------------------------------------------------------------------------
# blockwise Hamiltonian loss payload
# --------------------------------------------------------------------------
BLOCKWISE_BASIS = {"H": "1s", "O": "1s1p"}


def blockwise_payload() -> dict:
    """Fresh H-O payload: zero predictions against nonzero onsite/hopping target blocks.

    Active entries: onsite 1 (H) + 16 (O), hopping 4 (H->O) + 4 (O->H).
    """
    max_norb = 4  # 1s1p union
    pred_node = torch.zeros(2, max_norb, max_norb)
    target_node = torch.zeros(2, max_norb, max_norb)
    target_node[0, 0, 0] = 0.5  # H onsite 1x1
    target_node[1, :4, :4] = 0.25  # O onsite 4x4
    pred_edge = torch.zeros(2, max_norb, max_norb)
    target_edge = torch.zeros(2, max_norb, max_norb)
    target_edge[0, :1, :4] = 1.0  # H->O 1x4
    target_edge[1, :4, :1] = -1.0  # O->H 4x1
    return {
        "node_hamil_blocks": pred_node,
        "edge_hamil_blocks": pred_edge,
        "atom_types": torch.tensor([[0], [1]]),
        "atomic_numbers": torch.tensor([1, 8]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "node_delta_hamil_blocks": target_node,
        "edge_delta_hamil_blocks": target_edge,
        "node_delta_hamil_block_shape": torch.tensor([[1, 1], [4, 4]]),
        "edge_delta_hamil_block_shape": torch.tensor([[1, 4], [4, 1]]),
    }


def blockwise_criterion(**kwargs):
    from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss

    options = dict(basis=BLOCKWISE_BASIS, optimization="block_mae", block_reduction="global")
    options.update(kwargs)
    return HamilBlockwiseNexTHamLoss(**options)

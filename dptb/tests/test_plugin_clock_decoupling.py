"""P0-5 plugin-clock decoupling tests (distributed expert path, single process).

Before the fix, ``MultiTrainer``'s distributed-expert iteration path invoked
``call_plugins('iteration', ...)`` ONLY inside the display-window flush
(``_should_flush_display_window_now``), so cadence plugins (Saver ``save_freq``,
Validationer ``validation_freq``) were quantized to — and drifted by —
``display_freq``. After the fix, the dispatcher ticks on EVERY committed
optimizer step: non-flush ticks carry a cheap, locally-available state (no
collectives), the display-flush ticks carry the full gathered state exactly as
before.

The probe subclasses the real ``MultiTrainer`` (mirroring
``test_restart_resume.ProbeTrainer``) with ``distributed_expert=True`` in a
single process: every ``_dist_ready()`` guard is False, so the *production*
distributed code path runs with all collectives skipped.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from dptb.nnops.multi_trainer import MultiTrainer, _StageTagger
from dptb.plugins.base_plugin import Plugin, PluginUser
from dptb.plugins.monitor import LearningRateMonitor, TrainLossMonitor, Validationer
from dptb.plugins.saver import Saver


# ---------------------------------------------------------------------------
# Minimal harness
# ---------------------------------------------------------------------------
class _TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _TinyMultiExpertModel(nn.Module):
    name = "probe"

    def __init__(self, num_experts=1):
        super().__init__()
        self.experts = nn.ModuleList([_TinyExpert() for _ in range(num_experts)])
        self.model_options = {"embedding": {}, "prediction": {}}


class _StubBatch:
    """Carries the dynamic-batch dunder attrs the trainer forwards to plugins."""

    def __init__(self, idx):
        self.__dptb_batch_cost__ = 10.0 + idx
        self.__dptb_batch_num_graphs__ = 2


class _DistPathProbeTrainer(MultiTrainer):
    """Real MultiTrainer distributed-expert iteration path with stubbed loss.

    ``distributed_expert=True`` routes ``iteration()`` through
    ``_iteration_distributed_expert_prepared`` (the P0-5 site); with no process
    group initialized, ``_dist_ready()`` is False and every collective helper
    no-ops, so the plugin-dispatch logic under test is exactly production code.
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

        self.model = _TinyMultiExpertModel(num_experts)
        self.common_options = {"device": "cpu", "dtype": "float32"}
        self.train_options = {
            "max_ckpt": 50,
            "save_freq": save_freq,
            "display_freq": display_freq,
        }
        self.task = "hamiltonians"

        # Distributed-expert layout, single process (rank 0 of world 1).
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

        self.optimizers = [
            torch.optim.SGD(e.parameters(), lr=0.1) for e in self.model.experts
        ]
        # Real (epoch-cadence) schedulers so Saver can serialize their state;
        # they never step here because update_lr_per_iter is False.
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=1.0)
            for opt in self.optimizers
        ]
        self.train_lossfunc = None
        self.use_reference = False
        self.use_validation = False

        self._tagger = _StageTagger(
            self, enabled=False, freq=1, cuda_mem=False, cuda_sync=False,
            oom_dump=False,
        )
        self._reset_display_window_buffers()

        # Observability for assertions.
        self.validation_calls = []
        self.reference_train_losses = []

    # -- stubs -------------------------------------------------------------
    def _prepare_batch_bundle(self, batch, with_lengths=True):
        return {}, {}

    def _build_train_payload(self, *, batch_dict, batch_info, expert_idx,
                             range_dis, ref_batch_dict=None, ref_batch_info=None,
                             criterion=None):
        expert = self.model.experts[expert_idx]
        loss = (expert.weight ** 2).sum()
        self.reference_train_losses.append(float(loss.detach().item()))
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


class _StateRecordingPlugin(Plugin):
    """(1, 'iteration') plugin recording the (time, state-keys) of every tick."""

    def __init__(self):
        super().__init__([(1, "iteration")])
        self.ticks = []  # list of (time, dict-of-kwargs)

    def register(self, trainer):
        self.trainer = trainer

    def iteration(self, **kwargs):
        self.ticks.append((kwargs.get("time"), dict(kwargs)))


def _make_probe(tmp_path, *, save_freq=3, display_freq=100,
                with_saver=True, with_validationer=None, monitors=True):
    trainer = _DistPathProbeTrainer(save_freq=save_freq, display_freq=display_freq)
    recorder = _StateRecordingPlugin()
    trainer.register_plugin(recorder)
    if monitors:
        trainer.register_plugin(TrainLossMonitor())
        trainer.register_plugin(LearningRateMonitor())
    if with_validationer:
        trainer.register_plugin(
            Validationer(interval=[(with_validationer, "iteration")], fast_mode=True)
        )
    saver = None
    if with_saver:
        ckpt_dir = tmp_path / "ck"
        ckpt_dir.mkdir(exist_ok=True)
        saver = Saver(interval=[(save_freq, "iteration")])
        trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    trainer.rebase_plugin_cadence()
    return trainer, recorder, saver


# ---------------------------------------------------------------------------
# Test 1 — Saver fires on its exact save_freq grid, not display-quantized
# ---------------------------------------------------------------------------
def test_saver_fires_on_save_freq_grid_despite_large_display_freq(tmp_path):
    trainer, recorder, _saver = _make_probe(
        tmp_path, save_freq=3, display_freq=100
    )
    for i in range(12):
        trainer.iteration(_StubBatch(i))

    ckpt_dir = tmp_path / "ck"
    for it in (3, 6, 9, 12):
        f = ckpt_dir / f"probe.iter{it}.pth"
        assert f.exists(), f"missing checkpoint for step {it}"
        saved = torch.load(str(f), map_location="cpu", weights_only=False)
        assert saved["iteration"] == it
    # No off-grid checkpoints (pre-fix behavior would have produced none at all
    # in 12 steps, or display-quantized ones at flush boundaries only).
    saved_iters = sorted(
        int(p.name[len("probe.iter"):-len(".pth")])
        for p in ckpt_dir.glob("probe.iter*.pth")
    )
    assert saved_iters == [3, 6, 9, 12]
    assert (ckpt_dir / "probe.latest.pth").exists()


# ---------------------------------------------------------------------------
# Test 2 — dispatcher ticks every committed step; cheap vs full state contents
# ---------------------------------------------------------------------------
def test_plugins_tick_every_committed_step_with_cheap_state(tmp_path):
    trainer, recorder, _ = _make_probe(
        tmp_path, save_freq=3, display_freq=4, with_saver=False
    )
    for i in range(8):
        trainer.iteration(_StubBatch(i))

    times = [t for (t, _s) in recorder.ticks]
    assert times == list(range(1, 9))  # one tick per committed step

    flush_steps = {1, 4, 8}  # it == 1 or it % display_sync_freq == 0
    for t, state in recorder.ticks:
        assert state.get("field") == "iteration"
        if t in flush_steps:
            # Full display state: gathered/window metrics present.
            assert "train_loss" in state
            assert "total_grad_norm" in state
            assert state["window_steps"] >= 1
        else:
            # Cheap per-step state: no gathered metrics, only local fields.
            assert "train_loss" not in state
            assert "total_grad_norm" not in state
            assert state["window_steps"] == 0
            assert state["lr"] == pytest.approx(0.1)
            assert "loss_detached" in state
            # dynamic-batch fields ride along on cheap ticks
            assert state["batch_num_graphs"] == 2


def test_display_freq_larger_than_epoch_flushes_all_steps(tmp_path):
    trainer, _recorder, _ = _make_probe(
        tmp_path, save_freq=3, display_freq=100, with_saver=False
    )
    trainer.train_loader = [_StubBatch(i) for i in range(12)]
    trainer.epoch()  # no crash on cheap ticks; epoch tail forces a full flush

    train_stats = trainer.stats["train_loss"]
    weighted_sum, weighted_count = train_stats["epoch_stats"]
    assert weighted_count == 12
    expected = sum(trainer.reference_train_losses) / 12
    assert weighted_sum / weighted_count == pytest.approx(expected)
    assert train_stats["last_updated"] == 12

    trainer.call_plugins(queue_name="epoch", time=trainer.ep)
    assert train_stats["epoch_mean"] == pytest.approx(expected)
    # lr is locally available -> updated on every committed step.
    assert trainer.stats["lr"]["last_updated"] == 12
    assert trainer.stats["lr"]["last"] == pytest.approx(0.1)


def test_non_divisible_display_window_consumes_two_step_tail(tmp_path):
    trainer, _recorder, _ = _make_probe(
        tmp_path, save_freq=3, display_freq=5, with_saver=False
    )
    trainer.train_loader = [_StubBatch(i) for i in range(12)]
    trainer.epoch()

    weighted_sum, weighted_count = trainer.stats["train_loss"]["epoch_stats"]
    assert weighted_count == 12
    assert trainer.stats["train_loss"]["last_updated"] == 12
    assert weighted_sum / weighted_count == pytest.approx(
        sum(trainer.reference_train_losses) / 12
    )


def test_epoch_mean_weights_windows_by_step_count():
    user = PluginUser()
    user.register_plugin(TrainLossMonitor())

    user.call_plugins(
        queue_name="iteration", time=1, event_clock="display_window",
        field="iteration", train_loss=1.0, window_steps=5,
    )
    user.call_plugins(
        queue_name="iteration", time=2, event_clock="display_window",
        field="iteration", train_loss=7.0, window_steps=1,
    )

    assert user.stats["train_loss"]["epoch_stats"] == pytest.approx((12.0, 6))
    user.call_plugins(queue_name="epoch", time=1)
    assert user.stats["train_loss"]["epoch_mean"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 3 — Validationer runs on its exact validation_freq grid
# ---------------------------------------------------------------------------
def test_validationer_runs_on_validation_freq_grid(tmp_path):
    trainer, _recorder, _ = _make_probe(
        tmp_path, save_freq=3, display_freq=100, with_saver=False,
        with_validationer=5,
    )
    for i in range(12):
        trainer.iteration(_StubBatch(i))

    # Pre-fix: with display_freq=100 no flush fires between steps 2..12, so
    # validation would never run here. Post-fix it hits the exact 5-grid.
    assert trainer.validation_calls == [5, 10]
    assert trainer.stats["validation_loss"]["last"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 4 — explicit plugin_id (Item 3)
# ---------------------------------------------------------------------------
class _StatefulRecorder(Plugin):
    def __init__(self):
        super().__init__([(1, "iteration")])
        self.value = 0

    def register(self, trainer):
        self.trainer = trainer

    def iteration(self, **kwargs):
        pass

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        if state:
            self.value = state.get("value", self.value)


def test_register_plugin_explicit_plugin_id_overrides_default():
    user = PluginUser()
    p_default = _StatefulRecorder()
    p_custom = _StatefulRecorder()
    user.register_plugin(p_default)
    user.register_plugin(p_custom, plugin_id="saver.aux")

    assert p_default._plugin_id == "_StatefulRecorder#0"
    assert p_custom._plugin_id == "saver.aux"

    # Harvested state is keyed by the explicit id.
    p_default.value = 1
    p_custom.value = 42
    harvested = user.harvest_plugin_states()
    assert harvested["_StatefulRecorder#0"] == {"value": 1}
    assert harvested["saver.aux"] == {"value": 42}

    # Restore round-trips through the explicit id on a fresh user.
    user2 = PluginUser()
    user2._restored_plugin_state = harvested
    q_default = _StatefulRecorder()
    q_custom = _StatefulRecorder()
    user2.register_plugin(q_default)
    user2.register_plugin(q_custom, plugin_id="saver.aux")
    assert q_default.value == 1
    assert q_custom.value == 42


def test_register_plugin_default_id_assignment_unchanged():
    user = PluginUser()
    a = _StatefulRecorder()
    b = _StatefulRecorder()
    user.register_plugin(a)
    user.register_plugin(b)
    assert a._plugin_id == "_StatefulRecorder#0"
    assert b._plugin_id == "_StatefulRecorder#1"

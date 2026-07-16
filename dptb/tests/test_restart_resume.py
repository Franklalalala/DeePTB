"""Restart / checkpoint state-machine regression tests (PR-D).

These exercise the resume-correctness fixes end to end on the single-GPU
``Trainer`` using tiny in-memory fixtures (no external LMDB, no real model). A
``ProbeTrainer`` subclasses the real ``Trainer`` so the *production* epoch loop,
fast-forward cursor, restart() decision logic, plugin cadence, and Saver save
path are what actually run; only ``__init__``/``iteration``/``validation`` are
stubbed to a minimal harness.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
from torch import nn

import dptb.nnops.trainer as trainer_mod
from dptb.nnops.base_trainer import BaseTrainer
from dptb.nnops.trainer import Trainer
from dptb.plugins.base_plugin import Plugin, PluginUser
from dptb.plugins.saver import Saver
from dptb.nnops.training_state import (
    CHECKPOINT_KIND_EPOCH,
    CHECKPOINT_KIND_ITERATION,
    CHECKPOINT_SCHEMA_VERSION,
    TrainingState,
    read_resume_metadata,
)


# ---------------------------------------------------------------------------
# Minimal harness
# ---------------------------------------------------------------------------
class TinyModel(nn.Module):
    name = "probe"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.model_options = {"embedding": {}, "prediction": {}}


class ProbeTrainer(Trainer):
    """Real Trainer.epoch/run/restart/rebase with a stubbed iteration."""

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

        # A deterministic finite list is a valid DataLoader stand-in and gives a
        # fixed batch order, so a mid-epoch fast-forward is well defined.
        self.train_datasets = train_datasets
        self.train_loader = list(train_datasets)
        self.use_reference = reference_datasets is not None
        if self.use_reference:
            self.reference_loader = list(reference_datasets)
        self.use_validation = False

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.5
        )

        # Observability for assertions.
        self.processed = []          # (epoch, batch) actually optimized
        self.rng_trace = []          # RNG-derived value seen by each run batch
        self._epoch_losses = train_options.get("_epoch_losses", {})

    # -- stubbed one-batch step ------------------------------------------
    def iteration(self, ibatch, ref_batch=None):
        self.processed.append((int(self.ep), ibatch))
        # RNG draw makes the resume RNG-restore observable/deterministic.
        self.rng_trace.append(round(float(torch.rand(1).item()), 6))
        self.optimizer.step()
        self._batch_in_epoch = getattr(self, "_batch_in_epoch", 0) + 1
        # Feed a per-epoch loss so Saver.epoch has a metric.
        loss = self._epoch_losses.get(int(self.ep), 100.0 - int(self.ep))
        self.stats.setdefault("train_loss", {})["epoch_mean"] = float(loss)
        self.call_plugins(queue_name="iteration", time=self.iter)
        self.iter += 1
        return torch.tensor(float(loss))

    def validation(self, fast=True):
        return torch.tensor(0.0)


def _make_trainer(tmp_path, n_batches=4, update_lr_per_iter=False, epoch_losses=None,
                  seed=0):
    torch.manual_seed(seed)
    train_options = {
        "max_ckpt": 50,
        "update_lr_per_iter": update_lr_per_iter,
        "_epoch_losses": epoch_losses or {},
    }
    return ProbeTrainer(
        model=TinyModel(),
        train_datasets=list(range(n_batches)),
        train_options=train_options,
        common_options={"device": "cpu", "dtype": "float32"},
    )


class _CountingSaverProbe(Saver):
    """Saver that records how many times it wrote a full checkpoint file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_saves = 0

    def _save(self, *args, **kwargs):
        self.full_saves += 1
        return super()._save(*args, **kwargs)


# ---------------------------------------------------------------------------
# Test 1 — mid-epoch resume runs each remaining batch exactly once
# ---------------------------------------------------------------------------
def test_mid_epoch_resume_runs_each_remaining_batch_exactly_once(tmp_path):
    # Reference: an uninterrupted single epoch over 5 batches.
    ref = _make_trainer(tmp_path, n_batches=5)
    ref.ep = 1
    ref.epoch()
    ref_batches = [b for (_e, b) in ref.processed]
    assert ref_batches == [0, 1, 2, 3, 4]

    # Resume: a mid-epoch checkpoint after committing batches 0,1,2.
    resumed = _make_trainer(tmp_path, n_batches=5)
    resumed.ep = 1  # iteration kind re-enters the SAME epoch
    resumed.iter = 4  # 3 committed steps -> next global step is 4
    resumed._resume_plan = {
        "target_epoch": 1,
        "skip_batches": 3,
        "rng_state": None,
    }
    resumed.epoch()

    resumed_batches = [b for (_e, b) in resumed.processed]
    # Exactly the remaining batches, each once, in order.
    assert resumed_batches == [3, 4]
    # Committed-before + resumed-after == the full uninterrupted epoch, once each.
    assert sorted([0, 1, 2] + resumed_batches) == ref_batches
    # The one-shot plan is consumed; a subsequent epoch runs the whole loader.
    assert resumed._resume_plan is None
    resumed.processed.clear()
    resumed.ep = 2
    resumed.epoch()
    assert [b for (_e, b) in resumed.processed] == [0, 1, 2, 3, 4]


def test_plain_shuffle_loader_mid_epoch_resume_fails_closed(tmp_path, monkeypatch):
    # A torch DataLoader(shuffle=True) has no restart-reproducible order, so an
    # exact fast-forward is impossible — and re-running the epoch would apply
    # the committed prefix's optimizer updates a SECOND time. Default: refuse
    # with a clear remedy instead of silently diverging the trajectory.
    import torch.utils.data as tud

    monkeypatch.delenv("DPTB_ALLOW_INEXACT_RESUME", raising=False)
    resumed = _make_trainer(tmp_path, n_batches=5)
    resumed.train_loader = tud.DataLoader(list(range(5)), batch_size=1, shuffle=True)
    assert resumed._loader_supports_exact_fast_forward() is False

    resumed.ep = 1
    resumed._resume_plan = {"target_epoch": 1, "skip_batches": 3, "rng_state": None}
    with pytest.raises(RuntimeError, match="DPTB_ALLOW_INEXACT_RESUME"):
        resumed.epoch()
    assert resumed.processed == []  # nothing ran


def test_plain_shuffle_loader_opt_in_reruns_full_epoch(tmp_path, monkeypatch):
    # Explicit opt-in preserves the previous re-run-from-start behavior (with a
    # loud warning): all batches run, no fast-forward.
    import torch.utils.data as tud

    monkeypatch.setenv("DPTB_ALLOW_INEXACT_RESUME", "1")
    resumed = _make_trainer(tmp_path, n_batches=5)
    resumed.train_loader = tud.DataLoader(list(range(5)), batch_size=1, shuffle=True)
    resumed.ep = 1
    resumed._resume_plan = {"target_epoch": 1, "skip_batches": 3, "rng_state": None}
    resumed.epoch()
    assert len(resumed.processed) == 5


def test_mid_epoch_resume_restores_rng_before_remaining_batches(tmp_path):
    # Uninterrupted RNG trajectory for the whole epoch.
    ref = _make_trainer(tmp_path, n_batches=5, seed=1234)
    ref.epoch()
    full_trace = list(ref.rng_trace)

    # Capture the RNG state exactly after 3 committed batches.
    torch.manual_seed(1234)
    for _ in range(3):
        torch.rand(1)
    rng_after_three = {"torch": torch.get_rng_state()}

    resumed = _make_trainer(tmp_path, n_batches=5, seed=999)  # different seed
    resumed.ep = 1
    resumed.iter = 4
    resumed._resume_plan = {
        "target_epoch": 1,
        "skip_batches": 3,
        "rng_state": rng_after_three,
    }
    resumed.epoch()
    # The two re-executed batches see the same RNG values an uninterrupted run
    # would have produced for batches 3 and 4.
    assert resumed.rng_trace == full_trace[3:]


def test_second_mid_epoch_preemption_persists_absolute_cursor(tmp_path):
    # After a mid-epoch resume that fast-forwards over 3 committed batches and
    # re-executes the remaining 2, the committed-batch cursor must reflect the
    # ABSOLUTE position (5), not just the 2 re-executed batches. Otherwise a
    # SECOND mid-epoch preemption during this resumed epoch would persist
    # skip_batches=2 and a later restart would recompute (and double-step the
    # optimizer over) batches 3 and 4.
    resumed = _make_trainer(tmp_path, n_batches=5)
    resumed.ep = 1
    resumed.iter = 4
    resumed._resume_plan = {"target_epoch": 1, "skip_batches": 3, "rng_state": None}
    resumed.epoch()
    assert [b for (_e, b) in resumed.processed] == [3, 4]
    # 3 fast-forwarded + 2 re-executed == 5 committed batches in this epoch.
    assert resumed._batch_in_epoch == 5


# ---------------------------------------------------------------------------
# Test 2 — committed optimizer/scheduler/global_step/next-epoch LR match
# ---------------------------------------------------------------------------
def test_epoch_resume_matches_uninterrupted_scheduler_and_steps(tmp_path):
    n_batches = 3
    total_epochs = 3

    # Uninterrupted run of 3 epochs.
    ref = _make_trainer(tmp_path, n_batches=n_batches)
    ref.run(epochs=total_epochs)
    ref_last_epoch = ref.lr_scheduler.last_epoch
    ref_lr = ref.optimizer.param_groups[0]["lr"]
    ref_steps = len(ref.processed)

    # Interrupted: save a real EPOCH checkpoint after epoch 1, then restart.
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    original = _make_trainer(tmp_path, n_batches=n_batches)
    saver = Saver(interval=[(1, "epoch")])
    original.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    original.rebase_plugin_cadence()
    original.run(epochs=1)  # commits epoch 1, saves .ep1 + .best/.latest

    ckpt_file = str(ckpt_dir / "probe.latest.pth")
    assert os.path.exists(ckpt_file)

    # restart() with a real on-disk checkpoint; only build_model is stubbed.
    def fake_build_model(checkpoint_path, model_options, common_options, **kwargs):
        return TinyModel()

    orig_build = trainer_mod.build_model
    trainer_mod.build_model = fake_build_model
    try:
        resumed = ProbeTrainer.restart(
            ckpt_file,
            train_datasets=list(range(n_batches)),
            train_options={"max_ckpt": 50, "update_lr_per_iter": False},
            common_options={"device": "cpu", "dtype": "float32"},
        )
    finally:
        trainer_mod.build_model = orig_build

    # Epoch checkpoint -> advance to the next epoch, replay the one pending step.
    assert resumed.ep == 2
    # After restore(last_epoch=0)+replay -> last_epoch == 1 (matches uninterrupted
    # state at the epoch-1 boundary).
    assert resumed.lr_scheduler.last_epoch == 1

    resumed.run(epochs=total_epochs)
    assert resumed.lr_scheduler.last_epoch == ref_last_epoch
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(ref_lr)
    # committed-before (epoch 1) + resumed (epochs 2,3) == uninterrupted total.
    assert n_batches + len(resumed.processed) == ref_steps


# ---------------------------------------------------------------------------
# Test 3 — plugin schedule boundary 999/1000/1001 on the absolute grid
# ---------------------------------------------------------------------------
class _FiringPlugin(Plugin):
    def __init__(self, interval):
        super().__init__(interval)
        self.fired_at = []

    def register(self, trainer):
        self.trainer = trainer

    def iteration(self, **kwargs):
        self.fired_at.append(kwargs.get("time"))

    def epoch(self, **kwargs):
        self.fired_at.append(("epoch", kwargs.get("time")))


@pytest.mark.parametrize(
    "resume_iter, expected_first_fire",
    [(999, 1000), (1000, 1000), (1001, 2000)],
)
def test_plugin_cadence_rebase_absolute_grid(resume_iter, expected_first_fire):
    user = PluginUser()
    user.iter = resume_iter
    user.ep = 1
    plugin = _FiringPlugin(interval=[(1000, "iteration")])
    user.register_plugin(plugin)
    user.rebase_plugin_cadence()

    due = user.plugin_queues["iteration"][0][0]
    assert due == expected_first_fire

    # Drive the counters forward; the plugin must fire first at the grid point
    # (no redundant immediate fire) and then stay on the 1000-grid.
    for t in range(resume_iter, expected_first_fire + 1001):
        user.call_plugins(queue_name="iteration", time=t)
    assert plugin.fired_at[0] == expected_first_fire
    assert expected_first_fire + 1000 in plugin.fired_at
    # No fire strictly between resume and the first grid point.
    assert all(f >= expected_first_fire for f in plugin.fired_at)


# ---------------------------------------------------------------------------
# Test 4 — plugin state round-trips (Saver + monitors + ParamDynamics)
# ---------------------------------------------------------------------------
def test_saver_state_dict_round_trip():
    s = Saver()
    s.best_loss = 0.123
    s.best_quene = ["probe.ep3"]
    s.latest_quene = ["probe.iter90", "probe.iter100"]
    s.epoch_quene = ["probe.ep2", "probe.ep3"]
    blob = s.state_dict()

    s2 = Saver()
    assert s2.best_loss == 1e7
    s2.load_state_dict(blob)
    assert s2.best_loss == pytest.approx(0.123)
    assert s2.best_quene == ["probe.ep3"]
    assert s2.latest_quene == ["probe.iter90", "probe.iter100"]
    assert s2.epoch_quene == ["probe.ep2", "probe.ep3"]
    assert s2._state_restored is True


def test_monitor_running_avg_and_window_survive_reregister():
    from dptb.plugins.monitor import Monitor

    class _T:
        def __init__(self):
            self.stats = {}
            self.iter = 1

    class M(Monitor):
        stat_name = "train_loss"

        def _get_value(self, **kwargs):
            return kwargs.get("val")

    t = _T()
    m = Monitor.__new__(Monitor)
    # Simulate restored stats (BUG 4): a re-register must preserve them.
    t.stats["train_loss"] = {
        "last": 5.0,
        "running_avg": 4.2,
        "epoch_mean": 3.9,
        "epoch_stats": (11.0, 3),
    }
    mon = M(running_average=True, epoch_average=True)
    mon.register(t)
    assert t.stats["train_loss"]["running_avg"] == pytest.approx(4.2)
    assert t.stats["train_loss"]["epoch_mean"] == pytest.approx(3.9)
    assert t.stats["train_loss"]["epoch_stats"] == (11.0, 3)
    assert t.stats["train_loss"]["last"] == pytest.approx(5.0)


def test_cuda_memory_monitor_and_param_dynamics_state_round_trip():
    from dptb.plugins.monitor import CUDAMemoryMonitor, ParamDynamicsMonitor

    cm = CUDAMemoryMonitor()
    cm._epoch_max = {"peak_alloc_mb": 321.0}
    blob = cm.state_dict()
    cm2 = CUDAMemoryMonitor()
    cm2.load_state_dict(blob)
    assert cm2._epoch_max.get("peak_alloc_mb") == pytest.approx(321.0)

    pd = ParamDynamicsMonitor.__new__(ParamDynamicsMonitor)
    pd._dead_streak = {"group_a": 4}
    blob = pd.state_dict()
    pd2 = ParamDynamicsMonitor.__new__(ParamDynamicsMonitor)
    pd2._dead_streak = {}
    pd2.load_state_dict(blob)
    assert pd2._dead_streak.get("group_a") == 4


def test_harvest_and_restore_plugin_states_by_id(tmp_path):
    user = PluginUser()
    saver = Saver(interval=[(1, "epoch")])
    # Give the Saver enough to run register without a real trainer/model.
    class _M:
        name = "probe"
    saver.trainer = type("T", (), {"model": _M(), "stats": {}})()
    saver.checkpoint_path = str(tmp_path)
    user._assign_plugin_id(saver)
    user._registered_plugins.append(saver)
    saver.best_loss = 0.5
    saver.best_quene = ["probe.ep2"]

    harvested = user.harvest_plugin_states()
    assert saver._plugin_id in harvested
    assert harvested[saver._plugin_id]["best_loss"] == pytest.approx(0.5)

    # A fresh user re-registers the same class in the same order -> same id ->
    # state re-injected on register.
    user2 = PluginUser()
    user2._restored_plugin_state = harvested
    saver2 = Saver(interval=[(1, "epoch")])
    saver2.register = lambda *a, **k: None  # skip disk seeding
    user2.register_plugin(saver2)
    assert saver2.best_loss == pytest.approx(0.5)
    assert saver2.best_quene == ["probe.ep2"]


# ---------------------------------------------------------------------------
# Test 5 — monitor CSV continuity across restart
# ---------------------------------------------------------------------------
def test_csv_monitor_preserves_rows_and_single_header(tmp_path):
    from dptb.plugins.monitor import _open_csv_appending

    csv_path = str(tmp_path / "diag.csv")
    header = ["iter", "value"]

    _open_csv_appending(csv_path, header)
    with open(csv_path, "a", newline="") as f:
        import csv as _csv
        _csv.writer(f).writerow([1, "a"])
        _csv.writer(f).writerow([2, "b"])

    # Simulated restart: constructing the monitor again must not wipe rows.
    _open_csv_appending(csv_path, header)
    with open(csv_path, "a", newline="") as f:
        import csv as _csv
        _csv.writer(f).writerow([3, "c"])

    with open(csv_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines[0] == "iter,value"
    assert lines.count("iter,value") == 1  # header written exactly once
    assert lines[1:] == ["1,a", "2,b", "3,c"]  # pre-restart rows survive


def test_pretp_monitor_construction_is_non_destructive(tmp_path):
    from dptb.plugins.monitor import PreTPBlockMonitor

    log_dir = str(tmp_path / "mon")
    m1 = PreTPBlockMonitor(log_dir=log_dir)
    with open(m1.csv_path, "a", newline="") as f:
        import csv as _csv
        _csv.writer(f).writerow([7, "blk", "comp", 0.1, 0.2, 0.5])
    # Re-construct (as a restart would) -> prior row preserved, header once.
    PreTPBlockMonitor(log_dir=log_dir)
    with open(m1.csv_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines.count(",".join(m1.header)) == 1
    assert lines[-1].startswith("7,blk")


# ---------------------------------------------------------------------------
# Test 6 — best.pth is not clobbered by a worse resumed epoch
# ---------------------------------------------------------------------------
def test_best_pth_not_clobbered_by_worse_resumed_epoch(tmp_path):
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()

    # Original run: epoch 1 loss 1.0 is the best; save it.
    original = _make_trainer(tmp_path, n_batches=2, epoch_losses={1: 1.0})
    saver = Saver(interval=[(1, "epoch")])
    original.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    original.run(epochs=1)
    best_link = ckpt_dir / "probe.best.pth"
    assert os.path.exists(best_link)
    best_bytes_before = open(best_link, "rb").read()
    assert saver.best_loss == pytest.approx(1.0)

    # The persisted Saver state carries best_loss=1.0.
    harvested = original.harvest_plugin_states()
    assert harvested[saver._plugin_id]["best_loss"] == pytest.approx(1.0)

    # Resume: a WORSE epoch (loss 5.0). The restored best_loss must prevent the
    # worse epoch from overwriting .best.pth.
    resumed = _make_trainer(tmp_path, n_batches=2, epoch_losses={2: 5.0})
    resumed.ep = 2
    resumed._restored_plugin_state = harvested
    saver2 = Saver(interval=[(1, "epoch")])
    resumed.register_plugin(saver2, checkpoint_path=str(ckpt_dir))
    assert saver2.best_loss == pytest.approx(1.0)  # restored, not 1e7

    resumed.run(epochs=2)  # runs epoch 2 with the worse loss
    assert saver2.best_loss == pytest.approx(1.0)  # unchanged
    best_bytes_after = open(best_link, "rb").read()
    assert best_bytes_after == best_bytes_before  # .best.pth untouched


def test_manifest_written_atomically_and_seeds_best_loss(tmp_path):
    import json

    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    original = _make_trainer(tmp_path, n_batches=2, epoch_losses={1: 0.75})
    saver = Saver(interval=[(1, "epoch")])
    original.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    original.run(epochs=1)

    manifest_path = ckpt_dir / "checkpoint_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert manifest["best"] == "probe.best.pth"
    assert manifest["latest"] == "probe.latest.pth"
    assert manifest["latest_resumable"] == "probe.latest_resumable.pth"
    assert manifest["best_loss"] == pytest.approx(0.75)
    assert manifest["global_step"] >= 1
    assert manifest["latest_resumable_target"] == "probe.ep1.pth"
    entry = next(
        item for item in manifest["entries"]
        if item["filename"] == "probe.ep1.pth"
    )
    assert entry == {
        "filename": "probe.ep1.pth",
        "checkpoint_kind": CHECKPOINT_KIND_EPOCH,
        "resume_capability": "exact",
        "last_completed_step": 2,
        "committed_epoch": 1,
        "world_size": 1,
    }

    # A legacy-style resume (no restored plugin_state) seeds best_loss from the
    # manifest cheaply, so a worse epoch cannot clobber best.
    fresh = _make_trainer(tmp_path, n_batches=2)
    saver2 = Saver(interval=[(1, "epoch")])
    fresh.register_plugin(saver2, checkpoint_path=str(ckpt_dir))
    assert saver2.best_loss == pytest.approx(0.75)
    assert saver2._latest_resumable_checkpoint_name == "probe.ep1"


def test_epoch_best_checkpoint_is_epoch_kind_not_reused_iteration(tmp_path):
    # Regression (audit P1): the epoch-best checkpoint must embed
    # checkpoint_kind=EPOCH. A former optimization byte-copied the last
    # ITERATION checkpoint into best.pth/latest.pth, so those files carried
    # checkpoint_kind='iteration' and a restart mislabeled the committed epoch
    # boundary as mid-epoch (re-running the epoch, skipping its LR step, and
    # restoring a stale best_loss that could clobber the true best model).
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = _make_trainer(tmp_path, n_batches=2, epoch_losses={1: 0.5})
    # Both iteration and epoch saves fire; the iteration save lands on the last
    # committed batch of the epoch -- the former reuse trigger.
    saver = Saver(interval=[(1, "iteration"), (1, "epoch")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    trainer.run(epochs=1)

    best = torch.load(
        str(ckpt_dir / "probe.best.pth"), map_location="cpu", weights_only=False
    )
    assert read_resume_metadata(best).checkpoint_kind == CHECKPOINT_KIND_EPOCH


def test_every_epoch_is_committed_and_epoch_retention_rotates(tmp_path):
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = _make_trainer(
        tmp_path,
        n_batches=1,
        epoch_losses={1: 3.0, 2: 4.0, 3: 2.0, 4: 5.0},
    )
    trainer.train_options.update({"max_ckpt": 1, "max_epoch_ckpt": 2})
    saver = Saver(interval=[(1, "epoch")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    trainer.run(epochs=4)

    # Epochs 2 and 4 are non-best but still produced committed checkpoints.
    assert saver.epoch_quene == ["probe.ep3", "probe.ep4"]
    assert not (ckpt_dir / "probe.ep1.pth").exists()
    assert not (ckpt_dir / "probe.ep2.pth").exists()
    assert (ckpt_dir / "probe.ep3.pth").exists()
    assert (ckpt_dir / "probe.ep4.pth").exists()

    resumable = torch.load(
        str(ckpt_dir / "probe.latest_resumable.pth"),
        map_location="cpu", weights_only=False,
    )
    assert resumable["epoch"] == 4
    assert resumable["checkpoint_kind"] == CHECKPOINT_KIND_EPOCH
    # Existing latest semantics remain best/iteration oriented: non-best epoch
    # 4 does not replace the latest pointer published by best epoch 3.
    latest = torch.load(
        str(ckpt_dir / "probe.latest.pth"), map_location="cpu", weights_only=False
    )
    assert latest["epoch"] == 3


def test_restart_preflight_reports_resumable_path_and_rollback_before_build(
    tmp_path, monkeypatch
):
    import torch.utils.data as tud

    monkeypatch.delenv("DPTB_ALLOW_INEXACT_RESUME", raising=False)
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = _make_trainer(tmp_path, n_batches=1, epoch_losses={1: 1.0})
    trainer.train_loader = tud.DataLoader(
        list(range(1)), batch_size=1, shuffle=True
    )
    saver = Saver(interval=[(1, "iteration"), (1, "epoch")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    trainer.run(epochs=1)

    # Commit one step of epoch 2 without completing the epoch. latest now points
    # at an inexact iteration checkpoint; latest_resumable remains epoch 1.
    trainer._batch_in_epoch = 0
    trainer.iteration(torch.tensor([0]))
    latest_path = ckpt_dir / "probe.latest.pth"
    latest = torch.load(str(latest_path), map_location="cpu", weights_only=False)
    assert latest["resume_capability"] == "inexact_opt_in"

    build_called = {"value": False}

    def must_not_build(*args, **kwargs):
        build_called["value"] = True
        raise AssertionError("build_model must not run before restart preflight")

    monkeypatch.setattr(trainer_mod, "build_model", must_not_build)
    with pytest.raises(RuntimeError) as exc_info:
        ProbeTrainer.restart(
            str(latest_path),
            train_datasets=[0],
            train_options={"max_ckpt": 1},
            common_options={"device": "cpu", "dtype": "float32"},
        )

    message = str(exc_info.value)
    alternative = os.path.abspath(str(ckpt_dir / "probe.latest_resumable.pth"))
    assert alternative in message
    assert "rolls back 1 optimizer step(s)" in message
    assert f'--restart "{alternative}"' in message
    assert build_called["value"] is False


def test_fallback_resume_resets_partial_epoch_stats(tmp_path, monkeypatch):
    # Regression (audit P2): the opt-in inexact re-run (non-deterministic
    # loader + DPTB_ALLOW_INEXACT_RESUME=1) re-runs the whole epoch, so the
    # restored PARTIAL per-epoch accumulators must be zeroed to avoid
    # double-counting them into epoch_mean (plateau LR + best-loss gate).
    import torch.utils.data as tud

    monkeypatch.setenv("DPTB_ALLOW_INEXACT_RESUME", "1")
    resumed = _make_trainer(tmp_path, n_batches=5)
    resumed.train_loader = tud.DataLoader(list(range(5)), batch_size=1, shuffle=True)
    assert resumed._loader_supports_exact_fast_forward() is False
    resumed.ep = 1
    resumed._resume_plan = {"target_epoch": 1, "skip_batches": 3, "rng_state": None}
    resumed.stats["train_loss"] = {"epoch_stats": (12.0, 3), "epoch_mean": 4.0}
    resumed.epoch()
    assert resumed.stats["train_loss"]["epoch_stats"] == (0, 0)


def test_fast_forward_resume_preserves_partial_epoch_stats(tmp_path):
    # The exact-fast-forward path SKIPS the committed batches, so their restored
    # partial-accumulator contribution must be KEPT (not reset) to reconstruct
    # the full epoch_mean.
    resumed = _make_trainer(tmp_path, n_batches=5)  # list loader -> exact fast-forward
    assert resumed._loader_supports_exact_fast_forward() is True
    resumed.ep = 1
    resumed._resume_plan = {"target_epoch": 1, "skip_batches": 3, "rng_state": None}
    resumed.stats["train_loss"] = {"epoch_stats": (12.0, 3), "epoch_mean": 4.0}
    resumed.epoch()
    assert resumed.stats["train_loss"]["epoch_stats"] == (12.0, 3)


def test_atomic_write_leaves_no_partial_file_on_failure(tmp_path):
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = _make_trainer(tmp_path, n_batches=1)
    saver = Saver(interval=[(1, "iteration")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))

    f_path = str(ckpt_dir / "probe.iter1.pth")

    # Simulate a crash mid torch.save: the final file must not exist and no
    # stray tmp file may be left behind.
    real_save = torch.save

    def boom(obj, f):
        # Write a partial temp file then fail, mimicking an interrupted write.
        with open(f, "wb") as fh:
            fh.write(b"partial")
        raise RuntimeError("disk full")

    torch.save = boom
    try:
        with pytest.raises(RuntimeError):
            saver._write_checkpoint_atomic({"x": 1}, f_path)
    finally:
        torch.save = real_save

    assert not os.path.exists(f_path)  # os.replace never ran -> no final file
    leftovers = [p for p in os.listdir(ckpt_dir) if p.startswith("probe.iter1.pth")]
    assert leftovers == []  # tmp cleaned up


# ---------------------------------------------------------------------------
# Test 7 — distributed save failure raises on all ranks (no deadlock)
# ---------------------------------------------------------------------------
def test_distributed_rank0_save_failure_propagates_before_barrier(tmp_path, monkeypatch):
    import dptb.plugins.saver as saver_mod

    trainer = _make_trainer(tmp_path, n_batches=1)
    saver = Saver(interval=[(1, "iteration")])
    saver.trainer = trainer
    saver.checkpoint_path = str(tmp_path)

    # Force the distributed-expert path and a rank0 assembly failure.
    monkeypatch.setattr(saver, "_is_dist_expert", lambda: True)
    monkeypatch.setattr(saver, "_is_main", lambda: True)
    monkeypatch.setattr(
        saver, "_prepare_local_dist_states",
        lambda: {"local_idx": 0, "replica_fingerprint": None},
    )
    monkeypatch.setattr(saver, "_validate_prepared_expert_dp_replicas", lambda statuses: None)
    monkeypatch.setattr(saver, "_gather_dist_states", lambda prepared: ([{}], [None], [None]))
    # Not under test here; the real impl issues an all_gather collective that
    # needs an actual process group (covered by the gloo harness).
    monkeypatch.setattr(saver, "_collect_rank_states", lambda: {"0": {"rng_state": None}})
    def boom(expert_states):
        raise RuntimeError("rank0 assemble failed")
    monkeypatch.setattr(saver, "_assemble_full_model_state", boom)

    barrier_calls = {"n": 0}
    monkeypatch.setattr(saver_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(saver_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(saver_mod.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        saver_mod.dist, "all_gather_object",
        lambda gathered, value: gathered.__setitem__(0, value),
    )
    monkeypatch.setattr(saver_mod.dist, "broadcast_object_list",
                        lambda lst, src=0: None)  # keep rank0's False flag

    def counting_barrier(*a, **k):
        barrier_calls["n"] += 1

    monkeypatch.setattr(saver_mod.dist, "barrier", counting_barrier)

    with pytest.raises(RuntimeError, match="rank0 assemble failed"):
        saver._save(
            name="probe.iter1",
            model=trainer.model,
            model_options=trainer.model.model_options,
            common_options=trainer.common_options,
            train_options=trainer.train_options,
            kind=CHECKPOINT_KIND_ITERATION,
        )
    # The raise happens BEFORE the post-save barrier, so non-rank0 ranks (which
    # would receive the False flag and raise too) never block on it.
    assert barrier_calls["n"] == 0


# ---------------------------------------------------------------------------
# Test 8 — old (flat) checkpoint shape still loads through restart()
# ---------------------------------------------------------------------------
def test_legacy_flat_checkpoint_loads_through_restart(tmp_path, monkeypatch):
    # Historical flat shape: no training_state, flat optimizer/scheduler keys.
    legacy = {
        "config": {
            "train_options": {},
            "model_options": {"embedding": {}, "prediction": {}},
            "common_options": {"device": "cpu", "dtype": "float32",
                               "basis": {"H": "1s"}, "overlap": False, "seed": 1},
        },
        "model_state_dict": {},
        "epoch": 4,
        "iteration": 77,
        "stats": {"train_loss": {"epoch_mean": 2.0}},
        "optimizer_state_dict": {},
        "lr_scheduler_state_dict": {},
    }

    # The legacy reader classifies a shapeless checkpoint as epoch-committed.
    meta = read_resume_metadata(legacy)
    assert meta.checkpoint_kind == CHECKPOINT_KIND_EPOCH
    assert meta.epoch == 4 and meta.global_step == 77
    assert meta.schema_version == 0

    monkeypatch.setattr(trainer_mod.torch, "load", lambda *a, **k: legacy)
    monkeypatch.setattr(trainer_mod, "build_model",
                        lambda *a, **k: TinyModel())

    captured = {}

    class _RestartProbe(Trainer):
        def __init__(self, *, common_options, **kwargs):
            PluginUser.__init__(self)
            self.iter = 1
            self.ep = 1
            self._batch_in_epoch = 0
            self._resume_plan = None
            self.update_lr_per_iter = False
            self.stats = {}
            captured["common_options"] = common_options
            # object_keys optimizer/lr_scheduler intentionally absent -> skipped.

    resumed = _RestartProbe.restart(
        "legacy.pth",
        train_datasets=object(),
        train_options={},
        common_options={"device": "cpu", "dtype": "float32"},
    )
    # Legacy epoch-committed behavior preserved: ep = epoch + 1, iter = iter + 1.
    assert resumed.ep == 5
    assert resumed.iter == 78
    assert resumed._resume_plan is None  # no mid-epoch fast-forward for legacy
    assert resumed._restored_plugin_state == {}


def test_new_checkpoint_carries_schema_and_training_state(tmp_path):
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = _make_trainer(tmp_path, n_batches=2)
    saver = Saver(interval=[(1, "iteration")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    trainer.ep = 1
    trainer.epoch()  # fires Saver.iteration each committed batch

    saved = torch.load(str(ckpt_dir / "probe.latest.pth"), weights_only=False)
    assert saved["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert saved["checkpoint_kind"] == CHECKPOINT_KIND_ITERATION
    ts = saved["training_state"]
    assert ts["checkpoint_kind"] == CHECKPOINT_KIND_ITERATION
    assert ts["batch_in_epoch"] == 2  # both batches committed
    assert "rng_state" in ts
    assert saver._plugin_id in saved["plugin_state"]


def test_epoch_checkpoint_stores_last_completed_step_and_resume_parity(tmp_path, monkeypatch):
    # Schema v3 semantics: BOTH checkpoint kinds store the LAST COMPLETED
    # optimizer step, so restart's uniform `stored + 1` resumes at exactly the
    # next step. Previously the epoch path stored the post-increment counter
    # (next step) and restart's +1 skipped one global-step value, shifting
    # everything keyed off trainer.iter (warmup, cadence, logging).
    n_batches = 3
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()

    # Uninterrupted reference: iter after epoch 1 (3 batches, started at 1).
    ref = _make_trainer(tmp_path, n_batches=n_batches)
    ref.run(epochs=1)
    ref_next_step = ref.iter  # == n_batches + 1

    original = _make_trainer(tmp_path, n_batches=n_batches)
    saver = Saver(interval=[(1, "epoch")])
    original.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    original.run(epochs=1)

    ckpt = torch.load(str(ckpt_dir / "probe.latest.pth"),
                      map_location="cpu", weights_only=False)
    # Stored value = last completed step (3), NOT the post-increment counter (4).
    assert ckpt["iteration"] == n_batches
    assert ckpt["training_state"]["global_step"] == n_batches
    assert ckpt["checkpoint_schema_version"] >= 3

    def fake_build_model(checkpoint_path, model_options, common_options, **kwargs):
        return TinyModel()

    monkeypatch.setattr(trainer_mod, "build_model", fake_build_model)
    resumed = ProbeTrainer.restart(
        str(ckpt_dir / "probe.latest.pth"),
        train_datasets=list(range(n_batches)),
        train_options={"max_ckpt": 50, "update_lr_per_iter": False},
        common_options={"device": "cpu", "dtype": "float32"},
    )
    # Resumed next global step identical to the uninterrupted run: no skip.
    assert resumed.iter == ref_next_step


def test_checkpoint_carries_rank_states_and_restore_prefers_own_rank(tmp_path, monkeypatch):
    # P0-3: checkpoints persist per-rank RNG snapshots under rank_states; the
    # restore path prefers this process's own entry over the main-rank blob
    # inside training_state (which would correlate rank RNG streams).
    from dptb.nnops.training_state import resolve_rank_rng_state

    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = _make_trainer(tmp_path, n_batches=2)
    saver = Saver(interval=[(1, "epoch")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    trainer.run(epochs=1)

    ckpt = torch.load(str(ckpt_dir / "probe.latest.pth"),
                      map_location="cpu", weights_only=False)
    assert "rank_states" in ckpt
    assert "0" in ckpt["rank_states"]
    assert ckpt["rank_states"]["0"]["rng_state"] is not None

    # resolve prefers the own-rank entry...
    resume = read_resume_metadata(ckpt)
    own = resolve_rank_rng_state(ckpt, resume)
    assert own == ckpt["rank_states"]["0"]["rng_state"]
    # ...and falls back to the legacy blob when rank_states is absent.
    legacy = dict(ckpt)
    legacy.pop("rank_states")
    assert resolve_rank_rng_state(legacy, resume) == resume.rng_state


def test_publish_failure_raises_instead_of_silent_continue(tmp_path, monkeypatch):
    # P0-4 (single-process visible part): a failure in the retention/publish/
    # manifest phase must surface as an exception via the agreed publish phase
    # (in distributed runs the same helper broadcasts the failure to all ranks
    # before the barrier so no rank wedges at the next collective).
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = _make_trainer(tmp_path, n_batches=1)
    saver = Saver(interval=[(1, "iteration")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))

    def boom(strict=False):
        raise OSError("manifest disk full")

    monkeypatch.setattr(saver, "_write_manifest", boom)
    with pytest.raises(OSError, match="manifest disk full"):
        trainer.run(epochs=1)


def test_real_manifest_replace_failure_is_strict(tmp_path, monkeypatch):
    import dptb.plugins.saver as saver_mod

    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = _make_trainer(tmp_path, n_batches=1)
    saver = Saver(interval=[(1, "iteration")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))

    manifest_path = os.path.normcase(str(ckpt_dir / Saver.MANIFEST_NAME))
    real_replace = saver_mod.os.replace

    def fail_manifest_replace(src, dst):
        if os.path.normcase(os.fspath(dst)) == manifest_path:
            raise OSError("manifest replace disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(saver_mod.os, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="manifest replace disk full"):
        trainer.run(epochs=1)


def test_iteration_publish_failure_keeps_previous_latest_target(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = _make_trainer(tmp_path, n_batches=2)
    trainer.train_options["max_ckpt"] = 1
    saver = Saver(interval=[(1, "iteration")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))

    real_publish = saver._atomic_publish

    def fail_second_publish(src, dst):
        if str(src).endswith("probe.iter2.pth"):
            raise OSError("latest publish failed")
        return real_publish(src, dst)

    monkeypatch.setattr(saver, "_atomic_publish", fail_second_publish)
    with pytest.raises(OSError, match="latest publish failed"):
        trainer.run(epochs=1)

    old_checkpoint = ckpt_dir / "probe.iter1.pth"
    assert old_checkpoint.exists(), "retention GC ran before publish committed"
    latest = torch.load(
        str(ckpt_dir / "probe.latest.pth"), map_location="cpu", weights_only=False
    )
    assert latest["iteration"] == 1


def test_epoch_publish_failure_keeps_previous_best_target(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = _make_trainer(
        tmp_path, n_batches=1, epoch_losses={1: 2.0, 2: 1.0}
    )
    trainer.train_options["max_ckpt"] = 1
    saver = Saver(interval=[(1, "epoch")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    trainer.run(epochs=1)

    real_publish = saver._atomic_publish

    def fail_second_best(src, dst):
        if str(src).endswith("probe.ep2.pth") and str(dst).endswith("probe.best.pth"):
            raise OSError("best publish failed")
        return real_publish(src, dst)

    monkeypatch.setattr(saver, "_atomic_publish", fail_second_best)
    with pytest.raises(OSError, match="best publish failed"):
        trainer.run(epochs=2)

    old_checkpoint = ckpt_dir / "probe.ep1.pth"
    assert old_checkpoint.exists(), "best retention GC ran before publish committed"
    best = torch.load(
        str(ckpt_dir / "probe.best.pth"), map_location="cpu", weights_only=False
    )
    assert best["epoch"] == 1


def test_overlap_head_inference_from_state_dict():
    # Restart fail-safe: checkpoints written while the entrypoint mutated
    # common_options["overlap"]=False after model construction carry overlap
    # weights but a contradicting config; the weights are the ground truth.
    from dptb.nnops.multi_trainer import _state_dict_has_overlap_head

    assert _state_dict_has_overlap_head(
        {"experts.0.overlaponsite_param": torch.zeros(1)}
    )
    assert _state_dict_has_overlap_head(
        {"experts.1.edge_prediction_s.out_layer.linear.weight": torch.zeros(1)}
    )
    assert not _state_dict_has_overlap_head(
        {"experts.0.node_prediction_h.weight": torch.zeros(1)}
    )
    assert not _state_dict_has_overlap_head(None)

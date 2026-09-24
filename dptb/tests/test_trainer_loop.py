"""Training loop: one optimizer step per batch, reference batches, nonfinite skips, plugin clocks."""
from __future__ import annotations

import copy
import json

import pytest
import torch

from dptb.nnops.trainer import Trainer
from dptb.nnops.training_state import read_resume_metadata
from dptb.plugins.base_plugin import PluginUser
from dptb.plugins.monitor import TrainLossMonitor
from dptb.tests._trainer_probes import (
    DistPathProbeTrainer,
    FakeBatch,
    ScalarLoss,
    StubBatch,
    corrupt,
    make_dist_probe,
    make_fake_trainer,
)


# --------------------------------------------------------------------------
# single-process Trainer
# --------------------------------------------------------------------------
@pytest.mark.parametrize("with_reference", [False, True])
def test_iteration_takes_one_step_and_routes_train_and_reference_metrics(monkeypatch, with_reference):
    trainer, plugin_calls = make_fake_trainer(monkeypatch)
    # one criterion for both batches: the main-batch endpoint must survive the reference call
    trainer.reference_lossfunc = trainer.train_lossfunc
    reference = FakeBatch("reference", 3.0) if with_reference else None

    loss = trainer.iteration(FakeBatch("train", 2.0, batch_cost=7), reference)

    assert trainer.train_lossfunc.calls == (["train", "reference"] if with_reference else ["train"])
    assert trainer.optimizer.step_calls == 1
    # the returned objective includes the reference supervision ...
    assert loss.item() == pytest.approx(5.0 if with_reference else 2.0)
    [(queue, time, state)] = plugin_calls
    assert (queue, time) == ("iteration", 5)
    assert trainer.iter == 6
    # ... while the endpoint metrics stay scoped to the main batch
    assert state["train_loss"].item() == pytest.approx(2.0)
    assert state["train_loss_opt"].item() == pytest.approx(loss.item())
    assert state["train_onsite_loss"].item() == pytest.approx(21.0)
    assert state["train_hopping_loss"].item() == pytest.approx(22.0)
    assert state["batch_cost"] == 7
    assert state["batch_num_nodes"] == 2
    if with_reference:
        assert state["ref_onsite_loss"].item() == pytest.approx(31.0)
        assert state["ref_hopping_loss"].item() == pytest.approx(32.0)
        assert state["ref_mean_max_prob"].item() == pytest.approx(33.0)
        assert state["ref_expert_load_cv"].item() == pytest.approx(34.0)
    else:
        assert not any(key.startswith("ref_") for key in state)


class _CountingIterable:
    def __init__(self, items):
        self.items = list(items)
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(self.items)


@pytest.mark.parametrize("with_reference", [False, True])
def test_epoch_cycles_one_persistent_reference_iterator(with_reference):
    trainer = Trainer.__new__(Trainer)
    trainer.ep = 3
    trainer.use_reference = with_reference
    train = [f"train{i}" for i in range(5)]
    trainer.train_loader = _CountingIterable(train)
    trainer.reference_loader = _CountingIterable(["ref0", "ref1"])
    seen = []
    trainer.iteration = lambda batch, ref=None: seen.append((batch, ref))

    trainer.epoch()

    refs = ["ref0", "ref1", "ref0", "ref1", "ref0"] if with_reference else [None] * 5
    assert seen == list(zip(train, refs))
    # the reference loader restarts only when exhausted, not once per train batch
    assert trainer.reference_loader.iter_calls == (3 if with_reference else 0)


# --------------------------------------------------------------------------
# nonfinite batches never partially commit an update
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["nan", "inf", "gradient"])
@pytest.mark.parametrize("reference", [False, True])
def test_nonfinite_batch_is_discarded_then_training_resumes(monkeypatch, caplog, kind, reference):
    trainer, plugin_calls = make_fake_trainer(monkeypatch)
    trainer.train_lossfunc = ScalarLoss()
    trainer.reference_lossfunc = ScalarLoss()
    trainer.update_lr_per_iter = True
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 10)
    original_loss = trainer._loss_on_batch

    def loss(batch, *args, **kwargs):
        value = original_loss(batch, *args, **kwargs)
        return corrupt(value, kind) if batch.name == "bad" else value

    trainer._loss_on_batch = loss
    bad = FakeBatch("bad", 2)
    bad.__dptb_sample_indices__ = [17, 29]
    before = trainer.model.weight.detach().clone()
    scheduler_before = copy.deepcopy(trainer.lr_scheduler.state_dict())

    assert trainer.iteration(FakeBatch("good", 2) if reference else bad, bad if reference else None) is None
    assert torch.equal(trainer.model.weight, before)
    assert trainer.optimizer.step_calls == 0
    assert trainer.model.weight.grad is None
    assert trainer.lr_scheduler.state_dict() == scheduler_before
    # the skipped batch advances the loader cursor but not the optimizer clock
    assert trainer.iter == 5 and trainer.training_state.batch_in_epoch == 1
    assert not plugin_calls
    # the documented NONFINITE_BATCH_SKIPPED log record names the offending samples
    record = json.loads(next(r.message.split("NONFINITE_BATCH_SKIPPED ", 1)[1]
                             for r in caplog.records if "NONFINITE_BATCH_SKIPPED " in r.message))
    batch_key = "reference_batch" if reference else "batch"
    assert record["ranks"][0][batch_key]["__dptb_sample_indices__"] == [17, 29]

    assert trainer.iteration(FakeBatch("good", 2)).isfinite()
    assert trainer.optimizer.step_calls == 1
    assert trainer.iter == 6 and trainer.training_state.batch_in_epoch == 2
    assert not torch.equal(trainer.model.weight, before)
    assert len(plugin_calls) == 1


@pytest.mark.parametrize("kind", ["nan", "gradient"])
def test_second_expert_failure_does_not_update_the_first(kind):
    trainer = DistPathProbeTrainer(num_experts=2)
    trainer.distributed_expert = False
    build = trainer._build_train_payload

    def payload(**kwargs):
        result = build(**kwargs)
        if kwargs["expert_idx"] == 1:
            result["loss"] = corrupt(result["loss"], kind)
        return result

    trainer._build_train_payload = payload
    before = [p.detach().clone() for p in trainer.model.parameters()]
    assert trainer.iteration(StubBatch(0)) is None
    assert trainer.iter == 1 and trainer.training_state.batch_in_epoch == 1
    assert all(torch.equal(p, old) and p.grad is None
               for p, old in zip(trainer.model.parameters(), before))
    assert all(not opt.state and opt.step_calls == 0 for opt in trainer.optimizers)

    trainer._build_train_payload = build
    assert trainer.iteration(StubBatch(1)).isfinite()
    assert trainer.iter == 2 and trainer.training_state.batch_in_epoch == 2
    assert all(not torch.equal(p, old) for p, old in zip(trainer.model.parameters(), before))


def test_checkpoint_cursor_counts_the_skipped_batch(tmp_path):
    trainer, recorder = make_dist_probe(tmp_path, save_freq=1)
    build = trainer._build_train_payload

    def bad(**kwargs):
        result = build(**kwargs)
        result["loss"] = corrupt(result["loss"], "nan")
        return result

    trainer._build_train_payload = bad
    trainer.iteration(StubBatch(0))
    assert not list((tmp_path / "ck").glob("*.pth"))

    trainer._build_train_payload = build
    trainer.iteration(StubBatch(1))
    ckpt = torch.load(tmp_path / "ck" / "probe.iter1.pth", weights_only=False)
    assert read_resume_metadata(ckpt).batch_in_epoch == 2
    assert ckpt["iteration"] == 1
    assert len(recorder.ticks) == 1


def test_unrelated_exception_propagates(monkeypatch):
    trainer, _ = make_fake_trainer(monkeypatch)

    def broken(*args, **kwargs):
        raise ValueError("corrupt dataset")

    trainer._loss_on_batch = broken
    with pytest.raises(ValueError, match="corrupt dataset"):
        trainer.iteration(FakeBatch("good", 2))
    assert trainer.optimizer.step_calls == 0


# --------------------------------------------------------------------------
# plugin clocks on the MultiTrainer distributed-expert path
# --------------------------------------------------------------------------
def test_saver_fires_on_save_freq_grid_despite_large_display_freq(tmp_path):
    trainer, _ = make_dist_probe(tmp_path, save_freq=3, display_freq=100)
    for i in range(12):
        trainer.iteration(StubBatch(i))

    ckpt_dir = tmp_path / "ck"
    saved_iters = sorted(
        int(p.name[len("probe.iter"):-len(".pth")]) for p in ckpt_dir.glob("probe.iter*.pth")
    )
    assert saved_iters == [3, 6, 9, 12]
    for step in saved_iters:
        saved = torch.load(ckpt_dir / f"probe.iter{step}.pth", map_location="cpu", weights_only=False)
        assert saved["iteration"] == step
    assert (ckpt_dir / "probe.latest.pth").exists()


def test_plugins_tick_every_committed_step_with_cheap_state_between_flushes(tmp_path):
    trainer, recorder = make_dist_probe(tmp_path, save_freq=3, display_freq=4, with_saver=False)
    for i in range(8):
        trainer.iteration(StubBatch(i))

    assert [t for t, _state in recorder.ticks] == list(range(1, 9))
    flush_steps = {1, 4, 8}  # step 1 and every display_freq-th step
    for t, state in recorder.ticks:
        assert state.get("field") == "iteration"
        if t in flush_steps:
            # full display state with gathered window metrics
            assert "train_loss" in state
            assert "total_grad_norm" in state
            assert state["window_steps"] >= 1
        else:
            # cheap per-step state: only locally available fields
            assert "train_loss" not in state
            assert "total_grad_norm" not in state
            assert state["window_steps"] == 0
            assert state["lr"] == pytest.approx(0.1)
            assert "loss_detached" in state
            assert state["batch_num_graphs"] == 2


@pytest.mark.parametrize("display_freq", [100, 5], ids=["window_longer_than_epoch", "two_step_tail"])
def test_epoch_statistics_count_every_step_whatever_the_display_window(tmp_path, display_freq):
    trainer, _ = make_dist_probe(tmp_path, save_freq=3, display_freq=display_freq, with_saver=False)
    trainer.train_loader = [StubBatch(i) for i in range(12)]
    trainer.epoch()  # the epoch tail flushes the open display window

    train_stats = trainer.stats["train_loss"]
    weighted_sum, weighted_count = train_stats["epoch_stats"]
    expected = sum(trainer.train_losses) / 12
    assert weighted_count == 12
    assert weighted_sum / weighted_count == pytest.approx(expected)
    assert train_stats["last_updated"] == 12

    trainer.call_plugins(queue_name="epoch", time=trainer.ep)
    assert train_stats["epoch_mean"] == pytest.approx(expected)
    # lr is locally available, so it updates on every committed step
    assert trainer.stats["lr"]["last_updated"] == 12
    assert trainer.stats["lr"]["last"] == pytest.approx(0.1)


def test_epoch_mean_weights_display_windows_by_step_count():
    user = PluginUser()
    user.register_plugin(TrainLossMonitor())

    user.call_plugins(queue_name="iteration", time=1, event_clock="display_window",
                      field="iteration", train_loss=1.0, window_steps=5)
    user.call_plugins(queue_name="iteration", time=2, event_clock="display_window",
                      field="iteration", train_loss=7.0, window_steps=1)

    assert user.stats["train_loss"]["epoch_stats"] == pytest.approx((12.0, 6))
    user.call_plugins(queue_name="epoch", time=1)
    assert user.stats["train_loss"]["epoch_mean"] == pytest.approx(2.0)


def test_validationer_runs_on_validation_freq_grid(tmp_path):
    trainer, _ = make_dist_probe(tmp_path, save_freq=3, display_freq=100, with_saver=False,
                                 validation_freq=5)
    for i in range(12):
        trainer.iteration(StubBatch(i))

    assert trainer.validation_calls == [5, 10]
    assert trainer.stats["validation_loss"]["last"] == pytest.approx(0.5)

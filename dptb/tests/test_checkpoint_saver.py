"""Checkpoint Saver: best/latest/resumable pointers, manifest, retention and atomic writes."""
from __future__ import annotations

import json
import os

import pytest
import torch

import dptb.plugins.saver as saver_mod
from dptb.nnops.training_state import (
    CHECKPOINT_KIND_EPOCH,
    CHECKPOINT_KIND_ITERATION,
    CHECKPOINT_SCHEMA_VERSION,
    read_resume_metadata,
)
from dptb.plugins.saver import Saver, checkpoint_intervals
from dptb.tests._trainer_probes import make_probe_trainer
from dptb.utils.argcheck import train_options


def _ckpt_dir(tmp_path):
    path = tmp_path / "ck"
    path.mkdir()
    return path


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_worse_resumed_epoch_does_not_overwrite_the_best_checkpoint(tmp_path):
    ckpt_dir = _ckpt_dir(tmp_path)
    original = make_probe_trainer(2, epoch_losses={1: 1.0})
    saver = Saver(interval=[(1, "epoch")])
    original.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    original.run(epochs=1)
    best_before = (ckpt_dir / "probe.best.pth").read_bytes()
    harvested = original.harvest_plugin_states()

    resumed = make_probe_trainer(2, epoch_losses={2: 5.0})
    resumed.ep = 2
    resumed._restored_plugin_state = harvested  # what restart() installs
    restored = Saver(interval=[(1, "epoch")])
    resumed.register_plugin(restored, checkpoint_path=str(ckpt_dir))
    assert restored.best_loss == pytest.approx(1.0)

    resumed.run(epochs=2)
    assert restored.best_loss == pytest.approx(1.0)
    assert (ckpt_dir / "probe.best.pth").read_bytes() == best_before


def test_manifest_records_committed_pointers_and_seeds_best_loss(tmp_path):
    ckpt_dir = _ckpt_dir(tmp_path)
    original = make_probe_trainer(2, epoch_losses={1: 0.75})
    original.register_plugin(Saver(interval=[(1, "epoch")]), checkpoint_path=str(ckpt_dir))
    original.run(epochs=1)

    manifest = json.loads((ckpt_dir / "checkpoint_manifest.json").read_text())
    assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert manifest["best"] == "probe.best.pth"
    assert manifest["latest"] == "probe.latest.pth"
    assert manifest["latest_resumable"] == "probe.latest_resumable.pth"
    assert manifest["latest_resumable_target"] == "probe.ep1.pth"
    assert manifest["best_loss"] == pytest.approx(0.75)
    assert manifest["step_semantics"] == "last_completed"
    assert manifest["global_step"] == 2
    entry = next(item for item in manifest["entries"] if item["filename"] == "probe.ep1.pth")
    assert entry == {
        "filename": "probe.ep1.pth",
        "checkpoint_kind": CHECKPOINT_KIND_EPOCH,
        "resume_capability": "exact",
        "last_completed_step": 2,
        "committed_epoch": 1,
        "world_size": 1,
    }

    # without restored plugin state, a new Saver seeds best_loss from the manifest
    fresh = make_probe_trainer(2)
    seeded = Saver(interval=[(1, "epoch")])
    fresh.register_plugin(seeded, checkpoint_path=str(ckpt_dir))
    assert seeded.best_loss == pytest.approx(0.75)


def test_epoch_best_checkpoint_is_an_epoch_kind_checkpoint(tmp_path):
    """An iteration save on the epoch's last batch must not be reused as the epoch best."""
    ckpt_dir = _ckpt_dir(tmp_path)
    trainer = make_probe_trainer(2, epoch_losses={1: 0.5})
    trainer.register_plugin(Saver(interval=[(1, "iteration"), (1, "epoch")]),
                            checkpoint_path=str(ckpt_dir))
    trainer.run(epochs=1)

    best = _load(ckpt_dir / "probe.best.pth")
    assert read_resume_metadata(best).checkpoint_kind == CHECKPOINT_KIND_EPOCH


def test_every_epoch_is_committed_and_epoch_retention_rotates(tmp_path):
    ckpt_dir = _ckpt_dir(tmp_path)
    trainer = make_probe_trainer(1, epoch_losses={1: 3.0, 2: 4.0, 3: 2.0, 4: 5.0})
    trainer.train_options.update({"max_ckpt": 1, "max_epoch_ckpt": 2})
    saver = Saver(interval=[(1, "epoch")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    trainer.run(epochs=4)

    # non-best epochs 2 and 4 still produce committed checkpoints; only two are retained
    assert saver.epoch_quene == ["probe.ep3", "probe.ep4"]
    assert sorted(p.name for p in ckpt_dir.glob("probe.ep*.pth")) == ["probe.ep3.pth", "probe.ep4.pth"]
    resumable = _load(ckpt_dir / "probe.latest_resumable.pth")
    assert resumable["epoch"] == 4
    assert resumable["checkpoint_kind"] == CHECKPOINT_KIND_EPOCH
    # latest follows the best epoch (3), not the non-best epoch 4
    assert _load(ckpt_dir / "probe.latest.pth")["epoch"] == 3


def test_new_checkpoint_carries_schema_training_state_and_rank_states(tmp_path):
    ckpt_dir = _ckpt_dir(tmp_path)
    trainer = make_probe_trainer(2)
    saver = Saver(interval=[(1, "iteration")])
    trainer.register_plugin(saver, checkpoint_path=str(ckpt_dir))
    trainer.epoch()

    saved = _load(ckpt_dir / "probe.latest.pth")
    assert saved["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert saved["checkpoint_kind"] == CHECKPOINT_KIND_ITERATION
    state = saved["training_state"]
    assert state["checkpoint_kind"] == CHECKPOINT_KIND_ITERATION
    assert state["batch_in_epoch"] == 2
    assert state["rng_state"] is not None
    assert saved["rank_states"]["0"]["rng_state"] is not None
    assert saver._plugin_id in saved["plugin_state"]


def test_failed_torch_save_leaves_no_partial_checkpoint(tmp_path, monkeypatch):
    ckpt_dir = _ckpt_dir(tmp_path)
    trainer = make_probe_trainer(1)
    trainer.register_plugin(Saver(interval=[(1, "iteration")]), checkpoint_path=str(ckpt_dir))

    def interrupted_save(obj, f, *args, **kwargs):
        with open(f, "wb") as handle:
            handle.write(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="disk full"):
        trainer.run(epochs=1)
    assert not [p for p in os.listdir(ckpt_dir) if p.startswith("probe.iter1.pth")]


def _previous_pointer_target(ckpt_dir, pointer, field):
    if pointer.endswith(".json"):
        return json.loads((ckpt_dir / pointer).read_text())[field]
    return _load(ckpt_dir / pointer)[field]


@pytest.mark.parametrize(
    ("kind", "pointer", "field", "previous", "old_checkpoint"),
    [
        ("iteration", "probe.latest.pth", "iteration", 1, "probe.iter1.pth"),
        ("epoch", "probe.best.pth", "epoch", 1, "probe.ep1.pth"),
        ("iteration", "checkpoint_manifest.json", "latest_target", "probe.iter1.pth", "probe.iter1.pth"),
    ],
    ids=["latest_link", "best_link", "manifest"],
)
def test_failed_publish_raises_and_keeps_the_previous_commit(tmp_path, monkeypatch, kind, pointer, field,
                                                             previous, old_checkpoint):
    """The second publish fails on its os.replace; retention must not delete the old target."""
    ckpt_dir = _ckpt_dir(tmp_path)
    if kind == "iteration":
        trainer = make_probe_trainer(2)
    else:
        trainer = make_probe_trainer(1, epoch_losses={1: 2.0, 2: 1.0})
    trainer.train_options["max_ckpt"] = 1
    trainer.register_plugin(Saver(interval=[(1, kind)]), checkpoint_path=str(ckpt_dir))

    target = os.path.normcase(str(ckpt_dir / pointer))
    real_replace = saver_mod.os.replace
    publishes = []

    def failing_second_publish(src, dst):
        if os.path.normcase(os.fspath(dst)) == target:
            publishes.append(dst)
            if len(publishes) == 2:
                raise OSError("publish failed")
        return real_replace(src, dst)

    monkeypatch.setattr(saver_mod.os, "replace", failing_second_publish)
    with pytest.raises(OSError, match="publish failed"):
        trainer.run(epochs=2)

    assert (ckpt_dir / old_checkpoint).exists(), "retention ran before the publish committed"
    assert _previous_pointer_target(ckpt_dir, pointer, field) == previous


# --------------------------------------------------------------------------
# save triggers and expert-parallel payloads
# --------------------------------------------------------------------------
def _normalized(**overrides):
    cfg = {
        "num_epoch": 1,
        "batch_size": 1,
        "save_freq": 1000,
        "optimizer": {"type": "AdamW", "lr": 1e-3},
        "lr_scheduler": {"type": "rop"},
        "loss_options": {"train": {"method": "hamil_blockwise_nextham"}},
        **overrides,
    }
    normalized = train_options().normalize_value(cfg)
    train_options().check_value(normalized, strict=True)
    return normalized


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ({}, [(1000, "iteration"), (1, "epoch")]),
        ({"epoch_checkpoint": True}, [(1000, "iteration"), (1, "epoch")]),
        ({"epoch_checkpoint": False}, [(1000, "iteration")]),
    ],
)
def test_epoch_checkpoint_option_selects_saver_triggers(option, expected):
    assert checkpoint_intervals(_normalized(**option)) == expected


def test_zero_save_freq_registers_no_trigger():
    assert checkpoint_intervals({"save_freq": 0, "epoch_checkpoint": True}) is None


class _CudaTrainer:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.rank = 3
        self.is_main_process = True
        self.model = type("M", (), {"name": "nnenv", "model_options": {}})()


@pytest.mark.parametrize("enabled", [False, True])
def test_cuda_cache_is_cleared_after_iteration_saves_only_when_enabled(monkeypatch, enabled):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 2 * 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 5 * 1024 ** 2)
    if enabled:
        monkeypatch.setenv("DPTB_SAVER_CLEAR_CUDA_CACHE_AFTER_ITER_SAVE", "1")
    else:
        monkeypatch.delenv("DPTB_SAVER_CLEAR_CUDA_CACHE_AFTER_ITER_SAVE", raising=False)
    saver = Saver()
    saver.trainer = _CudaTrainer()

    saver._clear_cuda_cache_after_iteration_save("nnenv.iter1000")

    assert calls == (["empty_cache"] if enabled else [])


class _MustNotSerialize:
    def state_dict(self):
        raise AssertionError("a non-canonical expert-DP replica must not materialize state")


def test_noncanonical_expert_dp_replica_sends_an_empty_payload(monkeypatch):
    trainer = type("T", (), {})()
    trainer.rank = 1
    trainer.world_size = 4
    trainer.is_main_process = False
    trainer.distributed_expert = True
    trainer.local_expert_idx = 0
    trainer.expert_dp_rank = 1
    trainer.expert_data_parallel_size = 2
    trainer.num_experts = 2
    trainer.model = type("M", (), {"experts": [_MustNotSerialize(), _MustNotSerialize()]})()
    trainer.optimizers = [_MustNotSerialize(), _MustNotSerialize()]
    trainer.lr_schedulers = [_MustNotSerialize(), _MustNotSerialize()]
    trainer._unwrap_expert_module = lambda module: module
    saver = Saver()
    saver.trainer = trainer
    saver.checkpoint_path = "."

    gathered = []
    monkeypatch.setattr(saver_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(saver_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(saver_mod.dist, "gather_object",
                        lambda obj, object_gather_list=None, dst=0: gathered.append(obj))

    prepared = saver._prepare_local_dist_states()
    assert prepared["is_canonical"] is False
    assert saver._gather_dist_states(prepared) == (None, None, None)
    assert gathered == [None, None, None]

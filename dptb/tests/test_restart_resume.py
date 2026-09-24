"""Restart and mid-epoch resume.

The ProbeTrainer runs the production ``Trainer`` epoch loop, fast-forward cursor,
``restart()`` decisions and Saver path with a stubbed one-batch step.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.utils.data as tud

import dptb.nnops.trainer as trainer_mod
from dptb.nnops.training_state import (
    CHECKPOINT_KIND_EPOCH,
    CHECKPOINT_KIND_ITERATION,
    CHECKPOINT_SCHEMA_VERSION,
    TrainingState,
    read_resume_metadata,
    resolve_rank_rng_state,
    validate_checkpoint_invariants,
    validate_checkpoint_world_size,
)
from dptb.plugins.base_plugin import Plugin, PluginUser
from dptb.plugins.saver import Saver
from dptb.tests._trainer_probes import (
    ProbeModel,
    ProbeTrainer,
    make_probe_trainer,
    restart_probe,
)


def _mid_epoch_plan(skip):
    """The one-shot plan restart() leaves for an iteration checkpoint of epoch 1."""
    return {"target_epoch": 1, "skip_batches": skip, "rng_state": None}


# --------------------------------------------------------------------------
# mid-epoch resume
# --------------------------------------------------------------------------
@pytest.mark.parametrize("with_reference", [False, True])
def test_iteration_checkpoint_resumes_each_remaining_batch_once_with_restored_rng(
    tmp_path, monkeypatch, with_reference
):
    reference = [10, 11, 12, 13, 14] if with_reference else None
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    uninterrupted = make_probe_trainer(5, reference=reference, seed=1234)
    uninterrupted.register_plugin(Saver(interval=[(1, "iteration")]), checkpoint_path=str(ckpt_dir))
    uninterrupted.epoch()

    torch.manual_seed(999)  # the resumed process starts from an unrelated RNG state
    resumed = restart_probe(ckpt_dir / "probe.iter3.pth", monkeypatch, 5, reference_datasets=reference)
    assert (resumed.ep, resumed.iter) == (1, 4)
    resumed.epoch()

    assert [batch for _epoch, batch in resumed.processed] == [3, 4]
    if with_reference:
        # the reference stream advanced in lockstep during the fast-forward
        assert resumed.ref_seen == [13, 14]
    # the re-executed batches see the RNG values of the uninterrupted run
    assert resumed.rng_trace == uninterrupted.rng_trace[3:]
    # the cursor is absolute, so a second preemption would persist 5, not 2
    assert resumed.training_state.batch_in_epoch == 5

    # the one-shot plan is consumed: the next epoch runs the whole loader
    resumed.processed.clear()
    resumed.ep = 2
    resumed.epoch()
    assert [batch for _epoch, batch in resumed.processed] == [0, 1, 2, 3, 4]


def _shuffled_loader():
    return tud.DataLoader(list(range(5)), batch_size=1, shuffle=True)


@pytest.mark.parametrize(
    ("loader_attr", "make_loader"),
    [
        ("train_loader", _shuffled_loader),
        ("reference_loader", _shuffled_loader),
        ("train_loader", lambda: tud.DataLoader(list(range(5)), batch_size=1, num_workers=1)),
    ],
    ids=["train_shuffle", "reference_shuffle", "train_workers"],
)
def test_mid_epoch_resume_refuses_non_replayable_loaders(monkeypatch, loader_attr, make_loader):
    """Re-running the epoch would apply the committed prefix's updates twice."""
    monkeypatch.delenv("DPTB_ALLOW_INEXACT_RESUME", raising=False)
    resumed = make_probe_trainer(5, reference=list(range(5)) if loader_attr == "reference_loader" else None)
    setattr(resumed, loader_attr, make_loader())
    resumed._resume_plan = _mid_epoch_plan(skip=3)

    with pytest.raises(RuntimeError, match="DPTB_ALLOW_INEXACT_RESUME"):
        resumed.epoch()
    assert resumed.processed == []


@pytest.mark.parametrize(
    ("shuffled", "batches_run", "epoch_stats"),
    [(False, 2, (12.0, 3)), (True, 5, (0, 0))],
    ids=["exact_fast_forward_keeps_prefix_stats", "opt_in_rerun_resets_stats"],
)
def test_resumed_epoch_statistics_match_the_batches_that_rerun(monkeypatch, shuffled, batches_run,
                                                                epoch_stats):
    monkeypatch.setenv("DPTB_ALLOW_INEXACT_RESUME", "1")
    resumed = make_probe_trainer(5)
    if shuffled:
        resumed.train_loader = _shuffled_loader()
    resumed._resume_plan = _mid_epoch_plan(skip=3)
    # restored partial accumulators of the three committed batches
    resumed.stats["train_loss"] = {"epoch_stats": (12.0, 3), "epoch_mean": 4.0}

    resumed.epoch()

    assert len(resumed.processed) == batches_run
    assert resumed.stats["train_loss"]["epoch_stats"] == epoch_stats


# --------------------------------------------------------------------------
# epoch-boundary restart
# --------------------------------------------------------------------------
def test_epoch_checkpoint_restart_matches_uninterrupted_run(tmp_path, monkeypatch):
    n_batches, total_epochs = 3, 3
    uninterrupted = make_probe_trainer(n_batches)
    uninterrupted.run(epochs=total_epochs)

    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    original = make_probe_trainer(n_batches)
    original.register_plugin(Saver(interval=[(1, "epoch")]), checkpoint_path=str(ckpt_dir))
    original.run(epochs=1)
    ckpt = torch.load(ckpt_dir / "probe.latest.pth", map_location="cpu", weights_only=False)
    # both mirrors store the last completed optimizer step
    assert ckpt["iteration"] == ckpt["training_state"]["global_step"] == n_batches

    resumed = restart_probe(ckpt_dir / "probe.latest.pth", monkeypatch, n_batches)
    assert resumed.ep == 2
    assert resumed.iter == n_batches + 1
    # the epoch-end LR step pending at save time was replayed exactly once
    assert resumed.lr_scheduler.last_epoch == 1

    resumed.run(epochs=total_epochs)
    assert resumed.iter == uninterrupted.iter
    assert resumed.lr_scheduler.last_epoch == uninterrupted.lr_scheduler.last_epoch
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(
        uninterrupted.optimizer.param_groups[0]["lr"]
    )
    assert n_batches + len(resumed.processed) == len(uninterrupted.processed)


def test_legacy_flat_checkpoint_restarts_at_next_epoch_with_runtime_options(monkeypatch):
    donor = make_probe_trainer(2)
    legacy = {
        "config": {
            "train_options": {},
            "model_options": {"embedding": {}, "prediction": {}},
            "common_options": {"device": "cuda:7", "dtype": "float32",
                               "basis": {"H": "1s"}, "overlap": False, "seed": 17},
        },
        "model_state_dict": {},
        "epoch": 4,
        "iteration": 77,
        "stats": {"train_loss": {"epoch_mean": 2.0}},
        "optimizer_state_dict": donor.optimizer.state_dict(),
        "lr_scheduler_state_dict": donor.lr_scheduler.state_dict(),
    }
    # a shapeless checkpoint reads as epoch-committed
    meta = read_resume_metadata(legacy)
    assert (meta.checkpoint_kind, meta.epoch, meta.global_step, meta.schema_version) == (
        CHECKPOINT_KIND_EPOCH, 4, 77, 0,
    )

    built = {}

    def fake_build_model(checkpoint, model_options, common_options, **kwargs):
        built["common_options"] = common_options
        return ProbeModel()

    monkeypatch.setattr(trainer_mod.torch, "load", lambda *args, **kwargs: legacy)
    monkeypatch.setattr(trainer_mod, "build_model", fake_build_model)
    resumed = ProbeTrainer.restart(
        "legacy.pth",
        train_datasets=[0, 1],
        train_options={},
        common_options={"device": "cpu", "dtype": "float64"},
    )

    # runtime device/dtype override the checkpoint; the rest of common_options is kept
    expected = {"device": "cpu", "dtype": "float64", "basis": {"H": "1s"}, "overlap": False, "seed": 17}
    assert built["common_options"] == expected
    assert resumed.common_options == expected
    assert (resumed.ep, resumed.iter) == (5, 78)
    resumed.epoch()  # no mid-epoch fast-forward for a legacy checkpoint
    assert [batch for _epoch, batch in resumed.processed] == [0, 1]


def test_restart_preflight_names_the_resumable_checkpoint_before_building_the_model(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("DPTB_ALLOW_INEXACT_RESUME", raising=False)
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    trainer = make_probe_trainer(1, epoch_losses={1: 1.0})
    trainer.train_loader = tud.DataLoader(list(range(1)), batch_size=1, shuffle=True)
    trainer.register_plugin(Saver(interval=[(1, "iteration"), (1, "epoch")]),
                            checkpoint_path=str(ckpt_dir))
    trainer.run(epochs=1)
    # one committed step of epoch 2: latest is now an inexact iteration checkpoint
    trainer._batch_in_epoch = 0
    trainer.iteration(torch.tensor([0]))
    latest_path = ckpt_dir / "probe.latest.pth"
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert latest["resume_capability"] == "inexact_opt_in"

    def must_not_build(*args, **kwargs):
        raise AssertionError("build_model must not run before restart preflight")

    monkeypatch.setattr(trainer_mod, "build_model", must_not_build)
    with pytest.raises(RuntimeError) as exc_info:
        ProbeTrainer.restart(
            str(latest_path),
            train_datasets=[0],
            train_options={"max_ckpt": 1},
            common_options={"device": "cpu", "dtype": "float32"},
        )
    assert os.path.abspath(ckpt_dir / "probe.latest_resumable.pth") in str(exc_info.value)


def test_nnenv_from_reference_preserves_partial_model_and_runtime_overrides(monkeypatch):
    import dptb.nn.deeptb as deeptb_mod

    checkpoint = {
        "config": {
            "model_options": {
                "embedding": {"method": "checkpoint_embedding"},
                "prediction": {"method": "e3tb"},
            },
            "common_options": {"device": "cuda:3", "dtype": "float32",
                               "basis": {"C": "2s2p"}, "overlap": True, "has_soc": False},
        },
        "model_state_dict": {},
    }
    monkeypatch.setattr(deeptb_mod.torch, "load", lambda *args, **kwargs: checkpoint)

    class ProbeNNENV(deeptb_mod.NNENV):
        def __init__(self, **kwargs):
            torch.nn.Module.__init__(self)
            self.received = kwargs

    model = ProbeNNENV.from_reference(
        "dummy.pth",
        embedding={"method": "explicit_embedding"},
        prediction={},
        device="cpu",
        dtype="float64",
        basis={"H": "1s"},
        overlap=False,
        has_soc=True,
    )

    assert model.received["embedding"] == {"method": "explicit_embedding"}
    # an empty prediction section keeps the checkpoint's prediction settings
    assert model.received["prediction"] == {"method": "e3tb"}
    assert model.received["device"] == "cpu"
    assert model.received["dtype"] == "float64"
    assert model.received["basis"] == {"H": "1s"}
    assert model.received["overlap"] is False
    assert model.received["has_soc"] is True


def test_overlap_head_is_inferred_from_the_state_dict():
    """Restart trusts saved overlap weights over a contradicting config flag."""
    from dptb.nnops.multi_trainer import _state_dict_has_overlap_head

    assert _state_dict_has_overlap_head({"experts.0.overlaponsite_param": torch.zeros(1)})
    assert _state_dict_has_overlap_head(
        {"experts.1.edge_prediction_s.out_layer.linear.weight": torch.zeros(1)}
    )
    assert not _state_dict_has_overlap_head({"experts.0.node_prediction_h.weight": torch.zeros(1)})
    assert not _state_dict_has_overlap_head(None)


# --------------------------------------------------------------------------
# plugin cadence and plugin state across a restart
# --------------------------------------------------------------------------
class _FiringPlugin(Plugin):
    def __init__(self, interval):
        super().__init__(interval)
        self.fired_at = []

    def register(self, trainer):
        self.trainer = trainer

    def iteration(self, **kwargs):
        self.fired_at.append(kwargs.get("time"))


@pytest.mark.parametrize(("resume_iter", "expected_first_fire"), [(999, 1000), (1000, 1000), (1001, 2000)])
def test_plugin_cadence_rebases_onto_the_absolute_grid(resume_iter, expected_first_fire):
    user = PluginUser()
    user.iter = resume_iter
    user.ep = 1
    plugin = _FiringPlugin(interval=[(1000, "iteration")])
    user.register_plugin(plugin)
    user.rebase_plugin_cadence()

    for t in range(resume_iter, expected_first_fire + 1001):
        user.call_plugins(queue_name="iteration", time=t)
    # no redundant immediate fire; afterwards the plugin stays on the 1000-grid
    assert plugin.fired_at == [expected_first_fire, expected_first_fire + 1000]


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


def test_plugin_state_is_restored_by_default_and_explicit_ids(tmp_path):
    original = make_probe_trainer()
    recorders = [_StatefulRecorder(), _StatefulRecorder(), _StatefulRecorder()]
    saver = Saver(interval=[(1, "epoch")])
    original.register_plugin(recorders[0])
    original.register_plugin(recorders[1])
    original.register_plugin(recorders[2], plugin_id="saver.aux")
    original.register_plugin(saver, checkpoint_path=str(tmp_path))
    for recorder, value in zip(recorders, (1, 2, 42)):
        recorder.value = value
    saver.best_loss = 0.5
    saver.best_quene = ["probe.ep2"]
    saver.latest_quene = ["probe.iter90", "probe.iter100"]
    saver.epoch_quene = ["probe.ep1", "probe.ep2"]

    harvested = original.harvest_plugin_states()
    # persisted keys: ClassName#index by default, the explicit id when given
    assert harvested["_StatefulRecorder#0"] == {"value": 1}
    assert harvested["_StatefulRecorder#1"] == {"value": 2}
    assert harvested["saver.aux"] == {"value": 42}

    restarted = make_probe_trainer()
    restarted._restored_plugin_state = harvested  # what restart() installs
    clones = [_StatefulRecorder(), _StatefulRecorder(), _StatefulRecorder()]
    restored_saver = Saver(interval=[(1, "epoch")])
    restarted.register_plugin(clones[0])
    restarted.register_plugin(clones[1])
    restarted.register_plugin(clones[2], plugin_id="saver.aux")
    restarted.register_plugin(restored_saver, checkpoint_path=str(tmp_path))

    assert [clone.value for clone in clones] == [1, 2, 42]
    assert restored_saver.state_dict() == saver.state_dict()


def test_saver_and_monitor_state_dicts_round_trip_through_public_api(tmp_path):
    """Persisted plugin state serializes and restores through state_dict/load_state_dict alone."""
    from dptb.plugins.monitor import CUDAMemoryMonitor, ParamDynamicsMonitor

    saver = Saver()
    saver.best_loss = 0.123
    saver.best_quene = ["probe.ep3"]
    saver.latest_quene = ["probe.iter90", "probe.iter100"]
    saver.epoch_quene = ["probe.ep2", "probe.ep3"]
    restored_saver = Saver()
    assert restored_saver.best_loss == 1e7
    restored_saver.load_state_dict(saver.state_dict())
    assert restored_saver.best_loss == pytest.approx(0.123)
    assert restored_saver.best_quene == ["probe.ep3"]
    assert restored_saver.latest_quene == ["probe.iter90", "probe.iter100"]
    assert restored_saver.epoch_quene == ["probe.ep2", "probe.ep3"]
    assert restored_saver._state_restored is True

    cuda_monitor = CUDAMemoryMonitor()
    cuda_monitor._epoch_max = {"peak_alloc_mb": 321.0}
    restored_cuda = CUDAMemoryMonitor()
    restored_cuda.load_state_dict(cuda_monitor.state_dict())
    assert restored_cuda._epoch_max.get("peak_alloc_mb") == pytest.approx(321.0)

    # dead_streak survives the round trip itself; register() re-baselines weight
    # tracking on any restart and (separately) zeroes the counter at that first
    # baseline row -- see the training_monitors module note.
    param_monitor = ParamDynamicsMonitor(str(tmp_path), interval=[(1, "iteration")])
    param_monitor._dead_streak = {"group_a": 4}
    restored_param = ParamDynamicsMonitor(str(tmp_path), interval=[(1, "iteration")])
    restored_param.load_state_dict(param_monitor.state_dict())
    assert restored_param._dead_streak.get("group_a") == 4


# --------------------------------------------------------------------------
# checkpoint validation on load
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("top_field", "nested_field", "bad_value"),
    [
        ("iteration", "global_step", 8),
        ("epoch", "epoch", 3),
        ("checkpoint_kind", "checkpoint_kind", CHECKPOINT_KIND_ITERATION),
        ("checkpoint_schema_version", "schema_version", 2),
    ],
)
def test_schema_v3_mirror_mismatch_is_rejected(top_field, nested_field, bad_value):
    ckpt = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "iteration": 7,
        "epoch": 2,
        "checkpoint_kind": CHECKPOINT_KIND_EPOCH,
        "training_state": TrainingState(
            global_step=7, epoch=2, checkpoint_kind=CHECKPOINT_KIND_EPOCH
        ).state_dict(),
    }
    if top_field == "checkpoint_schema_version":
        ckpt["training_state"][nested_field] = bad_value
    else:
        ckpt[top_field] = bad_value
    with pytest.raises(RuntimeError, match=top_field):
        validate_checkpoint_invariants(ckpt)


_OWN_RNG = {"torch": torch.tensor([1], dtype=torch.uint8)}
_OTHER_RNG = {"torch": torch.tensor([2], dtype=torch.uint8)}
_MAIN_RNG = {"torch": torch.tensor([3], dtype=torch.uint8)}


@pytest.mark.parametrize(
    ("schema", "rank_states", "expected"),
    [
        (3, {"0": {"rng_state": _OWN_RNG}, "1": {"rng_state": _OTHER_RNG}}, _OWN_RNG),
        (3, None, _MAIN_RNG),
        (3, {"1": {"rng_state": _OTHER_RNG}}, RuntimeError),
        (2, {"1": {"rng_state": _OTHER_RNG}}, _MAIN_RNG),
    ],
    ids=["own_rank_entry", "no_rank_states", "v3_missing_own_rank", "v2_missing_own_rank"],
)
def test_restore_prefers_this_ranks_rng_snapshot(schema, rank_states, expected):
    resume = TrainingState(rng_state=_MAIN_RNG, schema_version=schema)
    ckpt = {"checkpoint_schema_version": schema, "training_state": resume.state_dict()}
    if rank_states is not None:
        ckpt["rank_states"] = rank_states
    if expected is RuntimeError:
        with pytest.raises(RuntimeError, match="rank 0"):
            resolve_rank_rng_state(ckpt, resume)
    else:
        assert resolve_rank_rng_state(ckpt, resume) is expected


def test_checkpoint_world_size_must_match_the_running_topology(monkeypatch, caplog):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    with pytest.raises(RuntimeError, match="world_size"):
        validate_checkpoint_world_size({"world_size": 2})

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="world_size"):
        validate_checkpoint_world_size({"world_size": 2}, current_world_size=3)

    # inspecting a multi-rank checkpoint in one process only warns
    caplog.set_level("WARNING")
    validate_checkpoint_world_size({"world_size": 2})
    assert any(record.levelname == "WARNING" and "world_size" in record.getMessage()
               for record in caplog.records)

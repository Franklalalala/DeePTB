"""Nonfinite batches must never partially commit an optimizer update."""
import copy
import json
import subprocess
import sys
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")
from dptb.tests.test_trainer_reference_batches import _fake_trainer, FakeBatch, ScalarLoss
from dptb.tests.test_plugin_clock_decoupling import _DistPathProbeTrainer, _StubBatch, _make_probe


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("nan"))


def corrupt(loss, kind):
    return BadGradient.apply(loss) if kind == "gradient" else loss * float(kind)


@pytest.mark.parametrize("kind", ["nan", "inf", "gradient"])
@pytest.mark.parametrize("reference", [False, True])
def test_single_discards_bad_batch_then_resumes(monkeypatch, caplog, kind, reference):
    trainer, _, states = _fake_trainer(monkeypatch)
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
    assert trainer.iteration(FakeBatch("good", 2) if reference else bad,
                             bad if reference else None) is None
    assert torch.equal(trainer.model.weight, before)
    assert trainer.optimizer.step_calls == 0
    assert trainer.model.weight.grad is None
    assert trainer.lr_scheduler.state_dict() == scheduler_before
    assert trainer.iter == 5 and trainer._batch_in_epoch == 1
    assert not states
    record = json.loads(next(r.message.split("NONFINITE_BATCH_SKIPPED ", 1)[1]
                             for r in caplog.records if "NONFINITE_BATCH_SKIPPED " in r.message))
    batch_key = "reference_batch" if reference else "batch"
    assert record["ranks"][0][batch_key]["__dptb_sample_indices__"] == [17, 29]
    assert trainer.iteration(FakeBatch("good", 2)).isfinite()
    assert trainer.optimizer.step_calls == 1
    assert trainer.iter == 6 and trainer._batch_in_epoch == 2
    assert not torch.equal(trainer.model.weight, before)
    assert len(states) == 1


@pytest.mark.parametrize("kind", ["nan", "gradient"])
def test_second_expert_failure_does_not_update_first(kind):
    trainer = _DistPathProbeTrainer(num_experts=2)
    trainer.distributed_expert = False
    trainer._compute_compatible_state_from_pack = lambda *a, **k: None
    trainer._local_scheduler_step = lambda *a: None
    trainer.call_plugins = lambda **kwargs: None
    build = trainer._build_train_payload

    def payload(**kwargs):
        result = build(**kwargs)
        if kwargs["expert_idx"] == 1:
            result["loss"] = corrupt(result["loss"], kind)
        return result

    trainer._build_train_payload = payload
    before = [p.detach().clone() for p in trainer.model.parameters()]
    assert trainer.iteration(_StubBatch(0)) is None
    assert trainer.iter == 1 and trainer._batch_in_epoch == 1
    assert all(torch.equal(p, old) and p.grad is None
               for p, old in zip(trainer.model.parameters(), before))
    assert all(not opt.state for opt in trainer.optimizers)
    trainer._build_train_payload = build
    assert trainer.iteration(_StubBatch(1)).isfinite()
    assert trainer.iter == 2 and trainer._batch_in_epoch == 2
    assert all(not torch.equal(p, old) for p, old in zip(trainer.model.parameters(), before))


def test_checkpoint_cursor_includes_skipped_batch(tmp_path):
    trainer, recorder, _ = _make_probe(tmp_path, save_freq=1)
    build = trainer._build_train_payload

    def bad(**kwargs):
        result = build(**kwargs)
        result["loss"] = corrupt(result["loss"], "nan")
        return result

    trainer._build_train_payload = bad
    trainer.iteration(_StubBatch(0))
    assert not list((tmp_path / "ck").glob("*.pth"))
    trainer._build_train_payload = build
    trainer.iteration(_StubBatch(1))
    from dptb.nnops.training_state import read_resume_metadata
    ckpt = torch.load(tmp_path / "ck" / "probe.iter1.pth", weights_only=False)
    assert read_resume_metadata(ckpt).batch_in_epoch == 2
    assert ckpt["iteration"] == 1
    assert len(recorder.ticks) == 1


def test_unrelated_exception_propagates(monkeypatch):
    trainer, _, _ = _fake_trainer(monkeypatch)
    def broken(*args, **kwargs):
        raise ValueError("corrupt dataset")
    trainer._loss_on_batch = broken
    with pytest.raises(ValueError, match="corrupt dataset"):
        trainer.iteration(FakeBatch("good", 2))
    assert trainer.optimizer.step_calls == 0


def _distributed_worker(rank, rendezvous, mode):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        trainer = _DistPathProbeTrainer(num_experts=2 if mode == "experts" else 1)
        trainer.rank, trainer.world_size = rank, 2
        trainer.local_expert_idx = rank if mode == "experts" else 0
        idx = trainer.local_expert_idx
        if mode == "ddp":
            trainer.model.experts[0] = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
            trainer.expert_dp_backend = "ddp"
        trainer.optimizers[idx] = torch.optim.Adam(trainer.model.experts[idx].parameters(), lr=0.01)
        trainer.lr_schedulers[idx] = torch.optim.lr_scheduler.StepLR(trainer.optimizers[idx], 10)
        trainer.update_lr_per_iter = True
        trainer._should_flush_display_window_now = lambda _: False
        trainer.call_plugins = lambda **kwargs: None
        build = trainer._build_train_payload
        kind = None

        def payload(**kwargs):
            # Keep the normal metric payload but exercise real DDP.forward hooks.
            if mode == "ddp":
                value = trainer.model.experts[0](torch.ones(1, 1)).square().sum()
                result = {"expert_onsite": 0.1, "expert_hopping": 0.2,
                          "active_nodes": 1., "active_edges": 1.,
                          "onsite_weighted_sum": 0.1, "hopping_weighted_sum": 0.2,
                          "z_values": [], "load_cv_values": []}
            else:
                result = build(**kwargs)
                value = result["loss"]
            result["loss"] = corrupt(value, kind) if rank == 1 and kind else value
            return result

        trainer._build_train_payload = payload
        # Warm optimizer state, then ensure both faults leave it and weights intact.
        trainer.iteration(_StubBatch(0))
        params = list(trainer.model.experts[idx].parameters())
        before = [p.detach().clone() for p in params]
        states = copy.deepcopy(trainer.optimizers[idx].state_dict())
        scheduler = copy.deepcopy(trainer.lr_schedulers[idx].state_dict())
        for kind in ("nan", "gradient"):
            assert trainer.iteration(_StubBatch(1)) is None
            assert trainer.iter == 2
            assert all(torch.equal(p, b) and p.grad is None for p, b in zip(params, before))
            now = trainer.optimizers[idx].state_dict()
            for key, values in states["state"].items():
                for name, value in values.items():
                    assert torch.equal(now["state"][key][name], value)
            assert trainer.lr_schedulers[idx].state_dict() == scheduler
        kind = None
        assert trainer.iteration(_StubBatch(2)).isfinite()
        assert trainer.iter == 3 and trainer._batch_in_epoch == 4
        assert all(not torch.equal(p, b) for p, b in zip(params, before))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["experts", "ddp"])
def test_two_rank_skip_consensus_and_next_good_batch(tmp_path, mode):
    rendezvous = (tmp_path / "rdzv").as_uri()
    workers = [subprocess.Popen([sys.executable, "-m", "dptb.tests.test_nonfinite_batch_skip", str(rank), rendezvous, mode],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
               for rank in range(2)]
    try:
        results = [worker.communicate(timeout=90)[0] for worker in workers]
        assert all(worker.returncode == 0 for worker in workers), "\n".join(results)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait()


if __name__ == "__main__":
    _distributed_worker(int(sys.argv[1]), sys.argv[2], sys.argv[3])

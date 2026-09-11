from types import SimpleNamespace
import pytest
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.plugins.saver import Saver


class FinalSaver(Saver):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.saved = []

    def iteration(self, **kwargs):
        self.saved.append((self.trainer.iter, self.trainer._batch_in_epoch))
        self._last_iteration_checkpoint_iter = self.trainer.iter


def harness(limit, start=1, reference=False):
    t = SimpleNamespace(train_options={'max_steps': limit}, iter=start, ep=1,
                        train_loader=[1, 2, 3, 4], reference_loader=[9],
                        use_reference=reference, distributed_expert=False,
                        update_lr_per_iter=True, plugin_queues={}, epochs=[], steps=[])
    t._set_expert_dp_sampler_epoch = lambda ep: None
    t.call_plugins = lambda **kwargs: t.epochs.append(kwargs)
    t.update = lambda: None
    for name in ['epoch', '_max_steps_reached', '_finish_max_steps']:
        setattr(t, name, getattr(MultiTrainer, name).__get__(t))
    def iteration(*args):
        t.steps.append(t.iter)
        t._batch_in_epoch += 1
        t.iter += 1
    t.iteration = iteration
    t._registered_plugins = [FinalSaver(t)]
    return t


@pytest.mark.parametrize('reference', [False, True])
def test_100k_finishes_normally_with_final_step_and_no_false_epoch(reference):
    t = harness(100000, start=99999, reference=reference)
    MultiTrainer.run(t, epochs=1000)
    assert t.steps == [99999, 100000]
    assert t._registered_plugins[0].saved == [(100000, 2)]
    assert t.iter == 100001
    assert t.epochs == []


def test_completed_restart_does_not_take_an_extra_step():
    t = harness(100000, start=100001)
    MultiTrainer.run(t, epochs=1000)
    assert t.steps == []


def test_unlimited_keeps_epoch_behavior():
    t = harness(None)
    MultiTrainer.run(t, epochs=2)
    assert len(t.steps) == 8
    assert len(t.epochs) == 2


@pytest.mark.parametrize('limit', [0, -1, True, 2.5])
def test_invalid_limits_fail_before_training(limit):
    t = harness(limit)
    with pytest.raises(ValueError, match='max_steps'):
        MultiTrainer.run(t, epochs=2)
    assert t.steps == []

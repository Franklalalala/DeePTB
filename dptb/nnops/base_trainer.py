import torch
import heapq
import logging
from dptb.utils.tools import get_lr_scheduler, j_must_have, get_optimizer, lr_scheduler_requires_metric
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from dptb.utils.constants import dtype_dict
from dptb.plugins.base_plugin import PluginUser
from dptb.nnops.training_state import TrainingState


log = logging.getLogger(__name__)


def _apply_epoch_lr_step(trainer):
    """Apply the single-trainer per-epoch learning-rate transition.

    A free function (taking ``trainer`` explicitly) so it works both when
    ``BaseTrainer.run`` is invoked bound on a real trainer and when it is invoked
    unbound on a lightweight stand-in, and so a restart from an *epoch*
    checkpoint can replay exactly one step: ``Saver.epoch`` persists the
    scheduler *before* this transition runs, so without a single replay a
    resumed run would be one LR step behind for the committed epoch.
    """
    if lr_scheduler_requires_metric(trainer.lr_scheduler):
        flow_enabled = bool(
            getattr(getattr(trainer, "flow_cfm", None), "enabled", False)
        )
        prefer_optimization_metric = bool(
            getattr(trainer, "scheduler_metric_prefers_objective", False)
        ) and not flow_enabled
        validation_name = (
            "validation_loss"
            if not prefer_optimization_metric
            or "validation_loss_opt" not in trainer.stats
            else "validation_loss_opt"
        )
        train_name = (
            "train_loss"
            if not prefer_optimization_metric
            or "train_loss_opt" not in trainer.stats
            else "train_loss_opt"
        )
        if validation_name in trainer.stats and 'epoch_mean' in trainer.stats[validation_name]:
            trainer.lr_scheduler.step(trainer.stats[validation_name]['epoch_mean'])
        else:
            trainer.lr_scheduler.step(trainer.stats[train_name]["epoch_mean"])
    else:
        trainer.lr_scheduler.step()  # modify the lr at each epoch (should we add it to pluggins so we could record the lr scheduler process? update 0927, this has been done in tensorboard monitor.)


class BaseTrainer(PluginUser, metaclass=ABCMeta):

    # ``iter``/``ep`` are backed by a single TrainingState so the live counters
    # and the checkpointed resume state are one source of truth; existing code
    # that reads/writes ``trainer.iter``/``trainer.ep`` is unchanged. The backing
    # store is created lazily so trainers built without BaseTrainer.__init__
    # (several unit-test harnesses) still work.
    def _ts(self):
        ts = self.__dict__.get("_training_state")
        if ts is None:
            ts = TrainingState()
            self.__dict__["_training_state"] = ts
        return ts

    @property
    def training_state(self):
        # Public snapshot: sync the separately-tracked epoch cursor so readers
        # see a consistent view (the hot iter/ep setters keep _ts() minimal).
        ts = self._ts()
        ts.batch_in_epoch = int(getattr(self, "_batch_in_epoch", 0))
        return ts

    @property
    def iter(self):
        return self._ts().global_step

    @iter.setter
    def iter(self, value):
        self._ts().global_step = int(value)

    @property
    def ep(self):
        return self._ts().epoch

    @ep.setter
    def ep(self, value):
        self._ts().epoch = int(value)

    def __init__(
            self,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            ) -> None:
        super(BaseTrainer, self).__init__()

        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]
        self.dtype = dtype
        self.device = device

        ''' Here is for plugins.
                    plugins:
                        - iteration: events  after every batch training iteration.  
                        - update: the updates of model paras including networks and optimiser, such as leaning rate, etc. after the batch training. 
                        - batch: events before batch training. 
                        - epoch: events after epoch batch training 
                    The difference b/w iteration and update the parameters, iteration takes in the batch output, loss etc., while  update takes in model itself.
                '''
        self.iter = 1
        self.ep = 1
        self.update_lr_per_iter = False

        # Restart/checkpoint state machine bookkeeping.
        #   _batch_in_epoch : number of committed (optimizer-stepped) batches in
        #                     the current epoch; read by Saver to persist a
        #                     fast-forward cursor for mid-epoch resume.
        #   _resume_plan    : one-shot plan consumed by the first epoch() call
        #                     after a restart (skip cursor + RNG snapshot).
        self._batch_in_epoch = 0
        self._resume_plan = None

    @abstractmethod
    def restart(self, checkpoint):
        """init trainer from disk
        """
        pass

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            '''对四个事件调用序列进行最小堆排序。'''
            heapq.heapify(q)

        for i in range(self.ep, epochs + 1):
            self.epoch()
            # run plugins of epoch events.
            self.call_plugins(queue_name='epoch', time=i)

            if not self.update_lr_per_iter:
                _apply_epoch_lr_step(self)

            self.update()
            self.ep += 1

    def _lr_step_on_epoch_end(self):
        """Instance entry point for the epoch-end LR step (restart replay)."""
        _apply_epoch_lr_step(self)


    @abstractmethod
    def iteration(self, **data):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        pass

    @abstractmethod
    def epoch(self) -> None:
        """define a training iteration process
        """
        pass

    @abstractmethod
    def validation(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass



if __name__ == '__main__':
    a = [1, 2, 3]

    print(list(enumerate(a, 2)))

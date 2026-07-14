import torch
import heapq
import logging
from dptb.utils.tools import get_lr_scheduler, j_must_have, get_optimizer, lr_scheduler_requires_metric
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from future.utils import with_metaclass
from dptb.utils.constants import dtype_dict
from dptb.plugins.base_plugin import PluginUser


log = logging.getLogger(__name__)

class BaseTrainer(with_metaclass(ABCMeta, PluginUser)):

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
                if lr_scheduler_requires_metric(self.lr_scheduler):
                    flow_enabled = bool(
                        getattr(getattr(self, "flow_cfm", None), "enabled", False)
                    )
                    prefer_optimization_metric = bool(
                        getattr(self, "scheduler_metric_prefers_objective", False)
                    ) and not flow_enabled
                    validation_name = (
                        "validation_loss"
                        if not prefer_optimization_metric
                        or "validation_loss_opt" not in self.stats
                        else "validation_loss_opt"
                    )
                    train_name = (
                        "train_loss"
                        if not prefer_optimization_metric
                        or "train_loss_opt" not in self.stats
                        else "train_loss_opt"
                    )
                    if validation_name in self.stats and 'epoch_mean' in self.stats[validation_name]:
                        self.lr_scheduler.step(self.stats[validation_name]['epoch_mean'])
                    else:
                        self.lr_scheduler.step(self.stats[train_name]["epoch_mean"])
                else:
                    self.lr_scheduler.step()  # modify the lr at each epoch (should we add it to pluggins so we could record the lr scheduler process? update 0927, this has been done in tensorboard monitor.)

            self.update()
            self.ep += 1


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

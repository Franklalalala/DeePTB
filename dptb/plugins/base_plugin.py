import heapq
import logging
from dptb.utils.tools import get_lr_scheduler, j_must_have, get_optimizer
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from future.utils import with_metaclass
from dptb.utils.constants import dtype_dict
from dptb.plugins.checkpoint_state import (
    harvest_plugin_states,
    restore_plugin_state,
)

log = logging.getLogger(__name__)


class Plugin(object):
    def __init__(self, interval=None):
        if interval is None:
            interval = []
        self.trigger_interval = interval

    def register(self, *args):
        raise NotImplementedError

class PluginUser(object):
    def __init__(self) -> None:
        ''' Here is for plugins.
                    plugins:
                        - iteration: events  after every batch training iteration.
                        - update: the updates of model paras including networks and optimiser, such as leaning rate, etc. after the batch training.
                        - batch: events before batch training.
                        - epoch: events after epoch batch training
                    The difference b/w iteration and update the parameters, iteration takes in the batch output, loss etc., while  update takes in model itself.
                '''
        self.stats = {}  # the status of Trainer.
        self.plugin_queues = {'disposable': [], 'iteration': [], 'epoch': [], 'batch': [], 'update': []}
        # EventScheduler / StatefulPlugin bookkeeping.
        self._registered_plugins = []          # de-duplicated registration order
        self._plugin_id_counts = {}            # per-class instance counter -> stable id
        # Populated by <Trainer>.restart() before plugins re-register so each
        # StatefulPlugin can re-inject its checkpointed blob at register time.
        self._restored_plugin_state = {}

    def _assign_plugin_id(self, plugin):
        """Give the plugin a stable, restart-reproducible id.

        Prefers an explicitly configured ``plugin_id`` attribute; otherwise
        ``f"{ClassName}#{n}"`` where ``n`` counts prior instances of that class.
        Registration order is deterministic (driven by the entrypoint), so the
        original run and its restart assign identical ids -> checkpointed plugin
        state round-trips by key.
        """
        configured = getattr(plugin, "plugin_id", None)
        if configured:
            plugin._plugin_id = str(configured)
            return plugin._plugin_id
        cls_name = type(plugin).__name__
        n = self._plugin_id_counts.get(cls_name, 0)
        self._plugin_id_counts[cls_name] = n + 1
        plugin._plugin_id = f"{cls_name}#{n}"
        return plugin._plugin_id

    def register_plugin(self, plugin, **kwargs):
        plugin.register(self, **kwargs)

        # Stable identity + de-duplicated registration bookkeeping, then restore
        # any checkpointed state for this plugin (e.g. Saver best_loss/queues).
        self._assign_plugin_id(plugin)
        if plugin not in self._registered_plugins:
            self._registered_plugins.append(plugin)
        restore_plugin_state(plugin, getattr(self, "_restored_plugin_state", None))

        # the trigger interval of plugin, with the form like: [(1, 'iteration'), (1, 'epoch')]
        intervals = plugin.trigger_interval

        if not isinstance(intervals, list):
            intervals = [intervals]

        for duration, unit in intervals:
            # unit the plugin type.
            queue = self.plugin_queues[unit]
            # Add the plugin events. duration is the trigger interval. len(queue) is the priority levels for the same duration,
            # the smaller the higher and is determined by the order of registration.
            queue.append((duration, len(queue), plugin))

    def harvest_plugin_states(self):
        """Collect ``{plugin_id: state_dict()}`` for all stateful plugins."""
        return harvest_plugin_states(self._registered_plugins)

    def rebase_plugin_cadence(self, counters=None):
        """Realign every scheduled ``iteration``/``epoch`` entry to the absolute
        grid for the *current* counter, then re-heapify.

        Fixes plugin-cadence drift on restart (BUG 3): ``register_plugin`` seeds
        the first due at ``interval`` regardless of where training resumes, so a
        job restarted at iter 1001 with ``save_freq=1000`` would otherwise fire
        immediately then drift to 2001, 3001, ...  After rebasing, the next due
        is the smallest multiple of ``interval`` that is ``>=`` the current
        counter (== current counter when already on the grid), so there is no
        redundant immediate fire and the schedule stays on the 1000-grid.

        Idempotent on a fresh run (counters == 1) since the seeded due already
        equals the grid value.
        """
        if counters is None:
            counters = {
                "iteration": int(getattr(self, "iter", 1)),
                "epoch": int(getattr(self, "ep", 1)),
            }
        for unit, current in counters.items():
            queue = self.plugin_queues.get(unit)
            if not queue:
                continue
            current = int(current)
            rebased = []
            for entry in queue:
                due, priority, plugin = entry
                interval = self._plugin_interval_for(plugin, unit)
                if interval is None or interval <= 0:
                    # One-shot / non-periodic entry: leave the due untouched.
                    rebased.append(entry)
                    continue
                # smallest multiple of interval that is >= current counter
                next_due = ((current + interval - 1) // interval) * interval
                if next_due < current:
                    next_due = current
                rebased.append((next_due, priority, plugin))
            self.plugin_queues[unit] = rebased
            heapq.heapify(self.plugin_queues[unit])

    @staticmethod
    def _plugin_interval_for(plugin, unit):
        intervals = getattr(plugin, "trigger_interval", None)
        if not intervals:
            return None
        if not isinstance(intervals, list):
            intervals = [intervals]
        for trigger in intervals:
            try:
                duration, trigger_unit = trigger
            except (TypeError, ValueError):
                continue
            if trigger_unit == unit:
                return duration
        return None

    def call_plugins(self, queue_name, time, **kwargs):
        # args should contain: [input, target, output, loss]
        # TODO: why we need a time update here?
        # kwargs.update({"time": time})
        # time can be iteration or epoch ...
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            # the plugin must have at-least one of the iteration、batch、epoch and update events.
            getattr(plugin, queue_name)(time=time, **kwargs)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            # 根据插件的事件触发间隔，来更新事件队列里的事件 duration
            if queue[0][0] > 0:
                new_item = (time + interval, queue[0][1], plugin)
                heapq.heappushpop(queue, new_item)
                '''加入新的事件并弹出最小堆的堆头。最小堆重新排序。'''
            else:
                heapq.heappop(queue)
                if len(queue) == 0:
                    return

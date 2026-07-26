import heapq
import logging
from dptb.utils.tools import get_lr_scheduler, j_must_have, get_optimizer
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from dptb.utils.constants import dtype_dict
from dptb.plugins.checkpoint_state import (
    harvest_plugin_states,
    restore_plugin_state,
)

log = logging.getLogger(__name__)


class Plugin(object):
    # Iteration plugins run on one of two independent event clocks. Cadence and
    # control plugins default to the committed optimizer-step clock; consumers
    # of reduced display-window data opt into ``display_window``.
    event_clock = "committed_step"

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
        self.display_window_plugin_queues = {
            'disposable': [], 'iteration': [], 'epoch': [], 'batch': [], 'update': []
        }
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

    def register_plugin(self, plugin, plugin_id=None, event_clock=None, **kwargs):
        """Register ``plugin``; extra kwargs are forwarded to ``plugin.register``.

        ``plugin_id`` (optional) overrides the default ``ClassName#index``
        identity. The explicit id is what persisted/harvested plugin state is
        keyed by, so two instances of the same class can round-trip their
        checkpoint state unambiguously as long as the restart registers them
        with the same ids. When omitted, behavior is identical to before.

        ``event_clock`` selects an independent scheduler for iteration data:
        ``committed_step`` (default) or ``display_window``.  The latter is only
        advanced by calls carrying a complete reduced display window, so cheap
        per-step ticks cannot consume its due events.
        """
        plugin.register(self, **kwargs)

        clock = event_clock or getattr(plugin, "event_clock", "committed_step")
        if clock not in {"committed_step", "display_window"}:
            raise ValueError(
                f"unsupported plugin event_clock={clock!r}; expected "
                "'committed_step' or 'display_window'"
            )
        plugin._event_clock = clock

        # Stable identity + de-duplicated registration bookkeeping, then restore
        # any checkpointed state for this plugin (e.g. Saver best_loss/queues).
        if plugin_id is not None:
            # _assign_plugin_id prefers a configured ``plugin_id`` attribute,
            # so the explicit kwarg wins over the ClassName#index default.
            plugin.plugin_id = str(plugin_id)
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
            queues = (
                self.display_window_plugin_queues
                if clock == "display_window"
                else self.plugin_queues
            )
            queue = queues[unit]
            # Add the plugin events. duration is the trigger interval. len(queue) is the priority levels for the same duration,
            # the smaller the higher and is determined by the order of registration.
            heapq.heappush(queue, (duration, len(queue), plugin))

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
        for queues in (self.plugin_queues, self.display_window_plugin_queues):
            for unit, current in counters.items():
                queue = queues.get(unit)
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
                queues[unit] = rebased
                heapq.heapify(queues[unit])

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

    def call_plugins(self, queue_name, time, event_clock=None, **kwargs):
        # args should contain: [input, target, output, loss]
        # TODO: why we need a time update here?
        # kwargs.update({"time": time})
        # time can be iteration or epoch ...
        if event_clock is None:
            queue_sets = (
                self.plugin_queues,
                self.display_window_plugin_queues,
            )
        elif event_clock == "committed_step":
            queue_sets = (self.plugin_queues,)
        elif event_clock == "display_window":
            queue_sets = (self.display_window_plugin_queues,)
        else:
            raise ValueError(
                f"unsupported plugin event_clock={event_clock!r}; expected "
                "'committed_step' or 'display_window'"
            )

        # A full-state call drives both clocks. Preserve cross-clock plugin
        # registration order so splitting the queues does not reorder existing
        # single-Trainer behavior (for example Validationer -> monitors ->
        # TensorBoard/Logger -> Saver).
        registration_order = {
            id(plugin): index
            for index, plugin in enumerate(self._registered_plugins)
        }
        due_entries = []
        for queues in queue_sets:
            queue = queues[queue_name]
            for entry in queue:
                if entry[0] <= time:
                    due_entries.append(
                        (registration_order.get(id(entry[2]), len(registration_order)), queue, entry)
                    )
        due_entries.sort(key=lambda item: (item[0], item[2][1]))

        for _, queue, entry in due_entries:
            due, priority, plugin = entry
            # the plugin must have at-least one of the iteration、batch、epoch and update events.
            getattr(plugin, queue_name)(time=time, **kwargs)
            interval = self._plugin_interval_for(plugin, queue_name)
            # Mutate the scheduler only after a successful callback. If a
            # plugin raises, its due entry remains retryable as before.
            queue.remove(entry)
            heapq.heapify(queue)
            # 根据插件的事件触发间隔，来更新事件队列里的事件 duration
            if due > 0 and interval is not None:
                heapq.heappush(queue, (time + interval, priority, plugin))

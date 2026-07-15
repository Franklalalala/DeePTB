"""Stateful-plugin protocol and (de)serialisation helpers for restart.

A plugin opts into checkpoint round-tripping simply by defining
``state_dict()`` and ``load_state_dict(state)``.  The trainer harvests those
blobs keyed by a *stable* ``plugin_id`` (see :mod:`dptb.plugins.base_plugin`)
into the checkpoint's ``plugin_state`` map, and re-injects them when the plugin
re-registers after a restart.  Plugins that do not implement the protocol keep
their historical behavior untouched.

This module intentionally has no imports from :mod:`dptb.plugins.base_plugin`
to keep the dependency edge one-directional (base_plugin -> checkpoint_state).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class StatefulPlugin:
    """Optional mixin documenting the restart state-dict protocol.

    Subclassing is *not* required — duck typing on ``state_dict`` /
    ``load_state_dict`` is what :func:`harvest_plugin_states` and
    :func:`restore_plugin_state` check.  The mixin only provides no-op defaults
    so a subclass can call ``super()`` safely.
    """

    def state_dict(self) -> Dict[str, Any]:  # pragma: no cover - overridden
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:  # pragma: no cover
        pass


def plugin_supports_state(plugin: Any) -> bool:
    return callable(getattr(plugin, "state_dict", None)) and callable(
        getattr(plugin, "load_state_dict", None)
    )


def get_plugin_id(plugin: Any) -> Optional[str]:
    """Return the stable id assigned by ``PluginUser.register_plugin``."""
    return getattr(plugin, "_plugin_id", None)


def harvest_plugin_states(registered_plugins: List[Any]) -> Dict[str, Any]:
    """Collect ``{plugin_id: state_dict()}`` for all stateful plugins.

    Failures in an individual plugin's ``state_dict`` are logged and skipped so
    a single misbehaving monitor can never abort a checkpoint save.
    """
    out: Dict[str, Any] = {}
    for plugin in registered_plugins:
        if not plugin_supports_state(plugin):
            continue
        pid = get_plugin_id(plugin)
        if pid is None:
            continue
        try:
            blob = plugin.state_dict()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Plugin %s state_dict() failed during save: %s", pid, exc)
            continue
        if blob is not None:
            out[pid] = blob
    return out


def restore_plugin_state(plugin: Any, restored: Optional[Dict[str, Any]]) -> bool:
    """Re-inject state into a single plugin if a matching blob exists.

    Returns True iff a blob was applied.  Safe to call for every plugin; it is a
    no-op for plugins without the protocol or without a stored blob.
    """
    if not restored or not plugin_supports_state(plugin):
        return False
    pid = get_plugin_id(plugin)
    if pid is None or pid not in restored:
        return False
    try:
        plugin.load_state_dict(restored[pid])
        return True
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Plugin %s load_state_dict() failed on restart: %s", pid, exc)
        return False

"""Programmatic command entry points, loaded when requested."""

from importlib import import_module
import sys
from types import ModuleType

_EXPORTS = {
    "train": (".train", "train"),
    "config": (".config", "config"),
    "test": (".test", "_test"),
    "bond": (".bond", "bond"),
}
__all__ = list(_EXPORTS)


class _EntryPointModule(ModuleType):
    """Preserve callable exports when Python binds same-named submodules."""

    def __setattr__(self, name, value):
        if name in _EXPORTS and isinstance(value, ModuleType):
            module, attribute = _EXPORTS[name]
            if value.__name__ == __name__ + module:
                value = getattr(value, attribute)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _EntryPointModule


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value

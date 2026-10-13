"""Programmatic command entry points, loaded when requested."""

from importlib import import_module

_EXPORTS = {
    "train": (".train", "train"),
    "config": (".config", "config"),
    "test": (".test", "_test"),
    "bond": (".bond", "bond"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value

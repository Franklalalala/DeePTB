"""Lazy aliases for explicitly retired Python module paths.

Only registered full names are redirected. Both paths resolve to one module,
including its globals, classes and canonical import metadata.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from collections.abc import Mapping


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str):
        self.target = target
        self.metadata = {}

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.metadata = {
            key: module.__dict__.get(key)
            for key in ("__name__", "__package__", "__spec__", "__loader__")
        }
        return module

    def exec_module(self, module):
        # module_from_spec assigns the alias spec even when create_module
        # returned an existing module. Retain the canonical module metadata.
        module.__dict__.update(self.metadata)


class _AliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.aliases = {}

    def find_spec(self, fullname, path=None, target=None):
        canonical = self.aliases.get(fullname)
        if canonical is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(canonical))


_FINDER = _AliasFinder()


def install_module_aliases(mapping: Mapping[str, str]) -> None:
    """Install exact-name redirects without importing any target module."""
    for old, new in mapping.items():
        if old == new:
            raise ValueError(f"A module cannot alias itself: {old}")
        existing = _FINDER.aliases.get(old)
        if existing is not None and existing != new:
            raise ValueError(f"Conflicting module alias for {old}: {existing} / {new}")
    _FINDER.aliases.update(mapping)
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)


register_module_aliases = install_module_aliases

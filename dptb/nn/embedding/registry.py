"""Discover and construct embeddings without eagerly importing model families."""
from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType


@dataclass(frozen=True)
class EmbeddingSpec:
    """An import target and its public compatibility status."""

    module: str
    constructor: str
    legacy: bool = False


EMBEDDING_CATALOGUE = MappingProxyType({
    "none": EmbeddingSpec("dptb.nn.embedding.identity", "Identity"),
    "se2": EmbeddingSpec("dptb.nn.embedding.se2", "SE2Descriptor"),
    "baseline": EmbeddingSpec("dptb.nn.embedding.baseline", "BASELINE"),
    "mpnn": EmbeddingSpec("dptb.nn.embedding.mpnn", "MPNN"),
    "deeph-e3": EmbeddingSpec("dptb.nn.embedding.deephe3", "E3DeePH"),
    "e3baseline_6": EmbeddingSpec("dptb.nn.embedding.e3baseline_local6", "E3BaseLineModel6"),
    "e3baseline_nonlocal": EmbeddingSpec("dptb.nn.embedding.e3baseline_nonlocal", "E3BaseLineModelNonLocal"),
    "lem": EmbeddingSpec("dptb.nn.embedding.lem", "Lem"),
    "slem": EmbeddingSpec("dptb.nn.embedding.slem", "Slem"),
    "lem_prior": EmbeddingSpec("dptb.nn.embedding.lem_prior", "LemPrior"),
    "slem_prior": EmbeddingSpec("dptb.nn.embedding.slem_prior", "SlemPrior"),
    "trinity": EmbeddingSpec("dptb.nn.embedding.trinity", "Trinity"),
    "unitb": EmbeddingSpec("dptb.nn.embedding.unitb.model", "UniTB"),
    "lem_moe_v3": EmbeddingSpec("dptb.nn.embedding.legacy.graph", "LemMoEV3", True),
    "lem_moe_v3_edge": EmbeddingSpec("dptb.nn.embedding.legacy.edge", "LemMoEV3Edge", True),
    "lem_moe_v3_edge_h0": EmbeddingSpec("dptb.nn.embedding.legacy.edge", "LemMoEV3EdgeH0", True),
    "lem_moe_v3_h0": EmbeddingSpec("dptb.nn.embedding.legacy.h0", "LemMoEV3H0", True),
    "lem_moe_v3_prior": EmbeddingSpec("dptb.nn.embedding.legacy.prior", "LemMoEV3Prior", True),
    "lem_moe_v3_prior_2b": EmbeddingSpec("dptb.nn.embedding.legacy.prior_2b", "build_prior_2b", True),
    "lem_moe_v3_edge_prior_2b": EmbeddingSpec("dptb.nn.embedding.legacy.prior_2b", "LemMoEV3EdgePrior2b", True),
})


class _LazyRegistry(MutableMapping):
    """The historical mapping interface with lazy built-in constructors."""

    def __init__(self):
        self._loaded = {}
        self._removed = set()
        self._overrides = set()

    def __getitem__(self, method):
        if method in self._loaded:
            return self._loaded[method]
        if method in self._removed:
            raise KeyError(method)
        spec = EMBEDDING_CATALOGUE[method]
        module = import_module(spec.module)
        # Import-time decorators normally populate this entry. Resolve it too
        # when an extension deliberately suppresses registration.
        if method not in self._loaded:
            self._loaded[method] = getattr(module, spec.constructor)
        return self._loaded[method]

    def __setitem__(self, method, constructor):
        if not callable(constructor):
            raise TypeError("An embedding constructor must be callable.")
        self._removed.discard(method)
        self._loaded[method] = constructor
        self._overrides.add(method)

    def __delitem__(self, method):
        if method not in self:
            raise KeyError(method)
        self._loaded.pop(method, None)
        self._removed.add(method)
        self._overrides.discard(method)

    def __iter__(self):
        return iter(dict.fromkeys(
            name for name in (*EMBEDDING_CATALOGUE, *self._loaded)
            if name not in self._removed
        ))

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, method):
        return method not in self._removed and (
            method in self._loaded or method in EMBEDDING_CATALOGUE
        )

    def register(self, target):
        def decorate(constructor):
            if not callable(constructor):
                raise TypeError("An embedding constructor must be callable.")
            method = target if isinstance(target, str) else constructor.__name__
            spec = EMBEDDING_CATALOGUE.get(method)
            builtin = spec is not None and (
                getattr(constructor, "__module__", None) == spec.module
                and getattr(constructor, "__name__", None) == spec.constructor
            )
            # A delayed built-in import must not overwrite an extension that
            # replaced its method before another family imported that module.
            if builtin:
                if method not in self._overrides:
                    self._removed.discard(method)
                    self._loaded[method] = constructor
            else:
                self[method] = constructor
            return constructor
        return decorate(target) if callable(target) else decorate


class Embedding:
    """Compatibility factory and extension registration interface."""

    _register = _LazyRegistry()

    @staticmethod
    def register(target):
        return Embedding._register.register(target)

    def __new__(cls, method: str, **options):
        return build_embedding(method, **options)


def list_embeddings(include_legacy: bool = False) -> tuple[str, ...]:
    """Return available method names without importing their implementations."""
    return tuple(sorted(
        name for name in Embedding._register.keys()
        if include_legacy or name not in EMBEDDING_CATALOGUE
        or not EMBEDDING_CATALOGUE[name].legacy
    ))


def build_embedding(method: str, **options):
    """Build a registered embedding, retaining historical option dispatch."""
    if method in {"lem", "slem"}:
        from dptb.nn.conditioning.inputs import resolve_legacy_prior_method
        method = resolve_legacy_prior_method(method, options)
    if method not in Embedding._register.keys():
        raise ValueError(f"Descriptor mode: {method} is not registered!")
    return Embedding._register[method](**options)


__all__ = ["Embedding", "EmbeddingSpec", "EMBEDDING_CATALOGUE", "build_embedding", "list_embeddings"]

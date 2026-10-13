"""Public embedding registry and package exports."""

import importlib
import pickle
import subprocess
import sys

import pytest

import dptb.nn.embedding as embedding


def test_identity_registry_constructs_and_returns_the_input():
    model = embedding.Embedding(method="none", dtype="float64", device="cpu")
    data = {"sentinel": object()}

    assert isinstance(model, embedding.Identity)
    assert model(data) is data
    assert list(model.parameters()) == []


def test_package_exports_resolve_and_upstream_methods_remain_registered():
    for name in embedding.__all__:
        assert hasattr(embedding, name), name

    expected = {
        "baseline", "deeph-e3", "e3baseline_6", "e3baseline_nonlocal",
        "none", "mpnn", "se2", "trinity", "lem", "slem",
    }
    assert expected <= set(embedding.Embedding._register.keys())


def test_discovery_and_identity_build_do_not_import_other_model_families():
    code = """
import sys
import dptb.nn.embedding as embedding
methods = embedding.list_embeddings()
assert 'unitb' in methods and 'lem' in methods
assert 'lem_moe_v3' not in methods
assert 'lem_moe_v3' in embedding.list_embeddings(include_legacy=True)
assert 'torch' not in sys.modules and 'e3nn' not in sys.modules
embedding.build_embedding('none')
assert 'dptb.nn.embedding.identity' in sys.modules
assert 'dptb.nn.embedding.unitb.model' not in sys.modules
assert not any(name.startswith('dptb.nn.embedding.legacy.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_extension_registration_is_discoverable_and_used_by_both_factories(monkeypatch):
    monkeypatch.setattr(embedding.Embedding, "_register", type(embedding.Embedding._register)())

    @embedding.Embedding.register("example_extension")
    class Extension:
        def __init__(self, value):
            self.value = value

    assert "example_extension" in embedding.list_embeddings()
    assert embedding.build_embedding("example_extension", value=17).value == 17
    assert isinstance(embedding.Embedding(method="example_extension", value=3), Extension)


def test_delayed_builtin_import_preserves_an_explicit_extension_override():
    code = """
from importlib import import_module
from dptb.nn.embedding import Embedding, build_embedding

@Embedding.register('unitb')
class Extension:
    pass

import_module('dptb.nn.embedding.legacy.edge')
assert isinstance(build_embedding('unitb'), Extension)

@Embedding.register('unitb')
class Replacement:
    pass

assert isinstance(build_embedding('unitb'), Replacement)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("legacy_first", [False, True])
def test_retired_module_aliases_preserve_identity_and_canonical_metadata(legacy_first):
    order = ["dptb.nn.embedding.unitb_options", "dptb.nn.embedding.unitb.options"]
    if not legacy_first:
        order.reverse()
    code = f"""
from importlib import import_module
first = import_module({order[0]!r})
second = import_module({order[1]!r})
assert first is second
assert first.__name__ == 'dptb.nn.embedding.unitb.options'
assert first.__spec__.name == first.__name__
first.alias_probe = object()
assert second.alias_probe is first.alias_probe
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_legacy_checkpoint_class_globals_resolve_to_the_canonical_class():
    canonical = importlib.import_module("dptb.nn.conditioning.projection")
    legacy = importlib.import_module("dptb.nn.embedding.prior_common")
    assert canonical is legacy
    # Protocol 0 GLOBAL reproduces the qualified class path saved by old
    # full-module checkpoints without creating a second implementation.
    restored = pickle.loads(b"cdptb.nn.embedding.prior_common\nH0InitLayer\n.")
    assert restored is canonical.H0InitLayer

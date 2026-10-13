"""Public discovery and CLI boundaries, independent of optional backends."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _isolated(code):
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-c", code], cwd=repo, env=env,
        text=True, capture_output=True, check=True,
    )


_NO_MODEL_DEPENDENCIES = '''
import importlib.abc
import sys
class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'e3nn', 'ase', 'pyabacus', 'so2_cuda_ops'}:
            raise ModuleNotFoundError('Optional model stack unavailable: ' + fullname)
sys.meta_path.insert(0, Unavailable())
'''


def test_cli_help_does_not_load_model_or_postprocessing_dependencies():
    result = _isolated(_NO_MODEL_DEPENDENCIES + '''
from dptb.__main__ import main
main(['--help'])
''')
    assert "models" in result.stdout and "h0" in result.stdout
    assert "multi-train" in result.stdout


def test_model_discovery_works_without_the_model_stack():
    result = _isolated(_NO_MODEL_DEPENDENCIES + '''
from dptb.__main__ import main
main(['models', '--json'])
''')
    methods = json.loads(result.stdout)
    assert {"unitb", "lem", "slem", "lem_prior", "slem_prior"} <= set(methods)
    assert not any(name.startswith("lem_moe_v3") for name in methods)
    all_methods = json.loads(_isolated(_NO_MODEL_DEPENDENCIES + '''
from dptb.__main__ import main
main(['models', '--all', '--json'])
''').stdout)
    assert set(methods) < set(all_methods)
    assert "lem_moe_v3_prior_2b" in all_methods


def test_uniform_embedding_factory_preserves_extension_registration():
    from dptb.nn import build_embedding, list_embeddings
    from dptb.nn.embedding import Embedding

    @Embedding.register("test_public_api_extension")
    class Extension:
        def __init__(self, value):
            self.value = value

    try:
        assert "test_public_api_extension" in list_embeddings()
        assert build_embedding("test_public_api_extension", value=7).value == 7
        assert Embedding("test_public_api_extension", value=11).value == 11
    finally:
        # Preserve the registry state for callers outside this test.
        registry = Embedding._register
        if hasattr(registry, "_dict"):
            registry._dict.pop("test_public_api_extension", None)
        else:
            del registry["test_public_api_extension"]


def test_public_model_builder_and_historical_builder_are_identical():
    from dptb.nn import build_model
    from dptb.nn.build import build_model as historical
    from dptb.nn.build import DistanceEnsembleWrapper as legacy_ensemble
    from dptb.nn.ensemble import DistanceEnsembleWrapper

    assert build_model is historical
    assert legacy_ensemble is DistanceEnsembleWrapper


@pytest.mark.parametrize("argv", [["h0", "--help"], ["h0", "reconstruct", "--help"]])
def test_h0_help_uses_the_same_top_level_cli(argv):
    result = _isolated(f"from dptb.__main__ import main; main({argv!r})")
    assert "usage:" in result.stdout.lower()


def test_h0_exit_status_is_propagated(monkeypatch):
    from dptb.__main__ import main
    import dptb.priors.h0.cli as h0_cli

    monkeypatch.setattr(h0_cli, "main", lambda argv: 7)
    assert main(["h0", "reconstruct", "example.json"]) == 7


def test_legacy_command_payload_is_not_a_process_exit_status(tmp_path):
    structure = tmp_path / "hydrogen.xyz"
    structure.write_text("2\nHydrogen molecule\nH 0 0 0\nH 0 0 0.74\n")
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "dptb", "bond", str(structure)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE="1"),
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Bond Type" in result.stdout and "0.74" in result.stdout


@pytest.mark.parametrize("order", ["submodule_first", "export_first"])
def test_command_exports_remain_functions_after_submodule_imports(order):
    _isolated(f'''
import importlib
import dptb.entrypoints as entrypoints

exports = {{'train': 'train', 'test': '_test', 'config': 'config', 'bond': 'bond'}}
for name, attribute in exports.items():
    if {order!r} == 'export_first':
        assert callable(getattr(entrypoints, name))
    module = importlib.import_module('dptb.entrypoints.' + name)
    exported = getattr(entrypoints, name)
    assert callable(exported), (name, exported)
    assert exported is getattr(module, attribute)

namespace = {{}}
exec('from dptb.entrypoints import *', namespace)
for name, attribute in exports.items():
    module = importlib.import_module('dptb.entrypoints.' + name)
    assert namespace[name] is getattr(module, attribute)
''')

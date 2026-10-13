"""Installed-package boundaries and legacy H0 import behavior."""
import importlib
from importlib import resources
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("first", ["h0rebuild", "dptb.priors.h0"])
def test_legacy_submodule_identity_and_block_keys(first, tmp_path):
    """Old and new imports must share dataclasses, including isinstance checks."""
    source_root = Path(__file__).resolve().parents[3]
    code = f"""
import importlib
first = importlib.import_module({first!r} + '.models')
old = importlib.import_module('h0rebuild.models')
new = importlib.import_module('dptb.priors.h0.models')
assert old is new is first
assert old.BlockKey is new.BlockKey
assert new.__spec__.name == 'dptb.priors.h0.models'
assert new.__package__ == 'dptb.priors.h0'
from dptb.priors.h0.deeptb import deeptb_block_key
assert deeptb_block_key(old.BlockKey(0, 1, (2, -3, 4))) == '0_1_2_-3_4'
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                   env={**os.environ, "PYTHONPATH": str(source_root)}, check=True,
                   capture_output=True, text=True)


def test_command_help_leaves_backends_unloaded(tmp_path):
    source_root = Path(__file__).resolve().parents[3]
    code = """
import sys
import h0rebuild
from dptb.priors.h0.cli import main
assert main(['--help']) == 0
for command in ('reconstruct', 'precompile', 'prepare-tables', 'compact-tables', 'accept'):
    try:
        main([command, '--help'])
    except SystemExit as error:
        assert error.code == 0
assert all(name not in sys.modules for name in ('torch', 'numpy', 'scipy', 'pyabacus'))
assert not any(name.startswith('dptb.nn') for name in sys.modules)
assert 'dptb.priors.h0.cuda_two_center' not in sys.modules
assert 'dptb.priors.h0.cuda_local_grid' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                   env={**os.environ, "PYTHONPATH": str(source_root)}, check=True,
                   capture_output=True, text=True)


def test_legacy_module_command_help(tmp_path):
    source_root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, "-m", "h0rebuild.cli", "--help"],
                            cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(source_root)},
                            check=True, capture_output=True, text=True)
    assert "Offline reconstruction" in result.stdout


def test_artifact_bridge_does_not_load_numerical_backends(tmp_path):
    source_root = Path(__file__).resolve().parents[3]
    code = """
import sys
from dptb.priors.h0 import deeptb, io
assert all(name not in sys.modules for name in ('torch', 'scipy', 'pyabacus'))
assert 'dptb.priors.h0.assemble' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                   env={**os.environ, "PYTHONPATH": str(source_root)}, check=True,
                   capture_output=True, text=True)


def test_native_sources_resolve_as_package_resources():
    root = resources.files("dptb.priors.h0")
    for resource in ("csrc/two_center/bindings.cpp", "csrc/two_center/two_center_cuda.cu",
                     "csrc/local_grid/binding.cpp", "csrc/local_grid/support_kernel.cu"):
        assert root.joinpath(resource).read_bytes()


def test_missing_native_manifest_rejects_without_compiling(tmp_path, monkeypatch):
    precompiled = importlib.import_module("dptb.priors.h0.precompiled")
    monkeypatch.setattr(precompiled, "ROOT", tmp_path)
    before = "torch.utils.cpp_extension" in sys.modules
    with pytest.raises(RuntimeError, match="Missing precompiled manifest"):
        precompiled.load("_cuda_two_center")
    assert ("torch.utils.cpp_extension" in sys.modules) == before
    assert not list(tmp_path.iterdir())


def test_mutable_work_directory_is_explicit(tmp_path, monkeypatch):
    from dptb.priors.h0.paths import table_root, work_root
    monkeypatch.delenv("H0_WORK_ROOT", raising=False)
    monkeypatch.delenv("H0_OFFLINE_TABLE_DIR", raising=False)
    with pytest.raises(ValueError, match="H0_WORK_ROOT"):
        work_root()
    monkeypatch.setenv("H0_WORK_ROOT", str(tmp_path))
    assert work_root() == tmp_path.resolve()
    assert table_root() == tmp_path / "offline_tables"
    other = tmp_path / "prepared"
    monkeypatch.setenv("H0_OFFLINE_TABLE_DIR", str(other))
    assert table_root() == other


def test_artifact_schema_survives_package_migration():
    from dptb.priors.h0.provenance import ARTIFACT_SCHEMA
    assert ARTIFACT_SCHEMA == "h0rebuild.artifact/v1"


def test_complex_artifact_roundtrip_retains_units_and_spin_channels(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from dptb.priors.h0.constants import energy_from_ry
    from dptb.priors.h0.deeptb import load_deeptb_artifact
    from dptb.priors.h0.io import save_result
    from h0rebuild.models import BlockKey

    block = np.array([[1.0, 0.5j], [-0.5j, 2.0]], dtype=np.complex128)
    result = SimpleNamespace(
        h_blocks_ry={BlockKey(0, 0, (0, 0, 0)): block},
        s_blocks={BlockKey(0, 0, (0, 0, 0)): np.eye(2)},
        orbital_counts=[1],
        metadata={"nspin": 4, "spin_order": "spin-block-major-per-atom:[up_spatial,down_spatial]"},
    )
    path, _ = save_result(result, tmp_path / "physical-h0.npz", energy_unit="Ry")
    loaded = load_deeptb_artifact(path, target_energy_unit="eV")
    np.testing.assert_array_equal(loaded.blocks["0_0_0_0_0"], block * energy_from_ry(1.0, "eV"))
    assert loaded.source_energy_unit == "Ry" and loaded.target_energy_unit == "eV"
    assert loaded.orbital_counts == (1,)


def test_explicit_build_publishes_package_binary_and_verifiable_manifest(tmp_path, monkeypatch):
    """A mocked native build exercises publication and resource-relative paths."""
    from types import SimpleNamespace
    import torch.utils.cpp_extension as extension
    from dptb.priors.h0 import precompile, precompiled

    package = tmp_path / "installed-h0"
    lane = package / "csrc" / "local_grid"
    lane.mkdir(parents=True)
    for name in ("binding.cpp", "support_kernel.cu", "pair_kernel.cu", "validation.h"):
        lane.joinpath(name).write_text("// mocked compilation unit\n")
    package.joinpath("precompile.py").write_text("# explicit builder\n")
    work = tmp_path / "work"
    monkeypatch.setenv("H0_WORK_ROOT", str(work))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "")
    monkeypatch.setenv("MAX_JOBS", "2")
    monkeypatch.setattr(precompiled, "ROOT", package)

    def mocked_load(*, build_directory, **kwargs):
        assert Path(build_directory).is_relative_to(work)
        assert all(Path(path).is_relative_to(package) for path in kwargs["sources"])
        binary = Path(build_directory) / "_cuda_local_grid.so"
        binary.write_bytes(b"mock-native-binary")
        return SimpleNamespace(__file__=str(binary))

    monkeypatch.setattr(extension, "load", mocked_load)
    precompile.build("_cuda_local_grid")
    manifest = precompiled.verify("_cuda_local_grid")
    assert manifest["binary"] == "_cuda_local_grid.so"
    assert (package / manifest["binary"]).read_bytes() == b"mock-native-binary"
    assert all((package / path).is_file() for path in manifest["sources"])


def test_reference_worker_receives_explicit_attempt_directory(tmp_path, monkeypatch):
    import json
    from dptb.priors.h0.cli import acceptance, worker as worker_module

    monkeypatch.setenv("H0_WORK_ROOT", str(tmp_path))
    monkeypatch.delenv("H0_OFFLINE_TABLE_DIR", raising=False)
    store = tmp_path / "offline_tables"
    store.mkdir()
    store.joinpath("catalog.json").write_text("{}")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    contract = {"case": "probe"}
    lease = {"identity": acceptance.digest(contract), "attempt_id": "test-attempt"}
    attempt.joinpath("attempt.json").write_text(json.dumps(lease))
    monkeypatch.setattr(acceptance, "register_worker", lambda *args: None)
    monkeypatch.setattr(acceptance, "shared_contract", lambda: {})
    monkeypatch.setattr(acceptance, "case_contract", lambda *args: contract)

    def fake_worker(case, run_dir):
        assert case == "probe" and run_dir == attempt
        acceptance.write(run_dir / "probe.json", {"id": case, "status": "PASS"})

    monkeypatch.setattr(worker_module, "worker", fake_worker)
    assert acceptance.run_worker("probe", attempt, 1, 2) == 0
    row = json.loads(attempt.joinpath("probe.json").read_text())
    assert row["identity"] == lease["identity"] and row["contract_verified_after"]

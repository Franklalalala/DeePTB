"""``precompile --check --arch`` must fail (nonzero process exit) when the prebuilt targets differ."""
import subprocess
import sys

import pytest

from dptb.nacf import precompile


def test_check_arch_reports_mismatch_and_accepts_exact_targets(monkeypatch):
    manifest = {'architectures': ['8.9'], 'ptx_architectures': ['8.9'], 'sha256': 'abc'}
    monkeypatch.setattr(precompile, 'verify', lambda device=None: manifest)
    assert precompile.main(['--check', '--arch', '8.9+PTX']) is manifest
    assert precompile.main(['--check']) is manifest                     # no target requested: manifest validity only
    for requested in (['9.0'], ['8.9'], ['8.9+PTX', '9.0'], ['9.0+PTX']):
        with pytest.raises(RuntimeError, match='do not match'):
            precompile.main(['--check', *sum((['--arch', r] for r in requested), [])])


def test_check_propagates_missing_or_corrupt_prebuilt(monkeypatch):
    def broken(device=None):
        raise RuntimeError('NACF prebuilt binary checksum mismatch')
    monkeypatch.setattr(precompile, 'verify', broken)
    with pytest.raises(RuntimeError, match='checksum'):
        precompile.main(['--check', '--arch', '8.9+PTX'])
    with pytest.raises(RuntimeError, match='checksum'):
        precompile.main(['--check'])


def test_check_never_builds_on_mismatch(monkeypatch):
    manifest = {'architectures': ['8.9'], 'ptx_architectures': [], 'sha256': 'abc'}
    monkeypatch.setattr(precompile, 'verify', lambda device=None: manifest)
    import torch.utils.cpp_extension as ext
    monkeypatch.setattr(ext, 'load', lambda *a, **k: pytest.fail('--check must not compile'))
    with pytest.raises(RuntimeError):
        precompile.main(['--check', '--arch', '9.0'])


def test_process_exit_code_is_nonzero_for_unavailable_target():
    # Whatever prebuilt exists on this machine, a 9.9 PTX target is never recorded; and a missing prebuilt is an error too.
    result = subprocess.run([sys.executable, '-m', 'dptb.nacf.precompile', '--check', '--arch', '9.9+PTX'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'do not match' in result.stderr or 'prebuilt' in result.stderr

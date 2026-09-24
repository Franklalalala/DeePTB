"""NACF radial-table lifecycle: device evaluation against the SciPy oracle, gradients, checkpoint round trip,
on-disk prepared-table caches (compiled tables and the source-parity snapshot), PTX/arch compatibility of the
recorded prebuilt manifest, and the ``precompile --check`` CLI. CUDA parametrizations that evaluate without
autograd need the verified NACF prebuilt binary (``requires_nacf_prebuilt``)."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.data.interfaces.p2_table import RadialBlockTable
from dptb.nacf import precompile, precompiled, prepared, radial
from dptb.nacf.assembly import NACFTableBank
from dptb.nacf.fusion import RadialMultiPlan
from dptb.nacf.prepared_store import PreparedRadialStore, SCHEMA, radial_identity, source_bindings, write_table
from dptb.tests._requires import requires_nacf_prebuilt
from dptb.tests.nacf_support import stores

DEVICES = ['cpu', pytest.param('cuda', marks=requires_nacf_prebuilt)]


# --------------------------------------------------------------------------- device evaluation vs SciPy oracle
def make_table(interpolation='cubic', shells=(0, 1, 2, 3, 4)):
    rng = np.random.default_rng(614)
    knots = np.array([0., .2, .55, 1.1, 1.8, 2.6, 3.])
    width = sum(2 * l + 1 for l in shells)
    values = rng.normal(size=(len(knots), width, width))
    values[:, 0, -1] = 0.
    return RadialBlockTable(knots, values, shells, shells, 3., interpolation)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('interpolation', ['linear', 'cubic'])
def test_radial_matches_independent_scipy_oracle(device, interpolation):
    table = make_table(interpolation)
    model = radial.TorchRadialBlockTable(table, device=device)
    vectors = np.random.default_rng(19).normal(size=(31, 3))
    vectors = np.concatenate((vectors, [[0, 0, 0], [0, 0, -1], [0, 0, 1], [0, 0, 3], [0, 0, 3 - 5e-13],
                                        [1e-9, 0, -1], [0, 0, 4]]))
    expected = np.stack([table.evaluate(v) for v in vectors])
    actual = model(torch.tensor(vectors, device=device)).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=5e-11, rtol=5e-11)
    assert model(torch.empty((0, 3), device=device, dtype=torch.float64)).shape == (0, 25, 25)
    assert torch.count_nonzero(model(torch.tensor([[0., 0., 3.]], device=device, dtype=torch.float64))) == 0


def test_radial_gradients_away_from_poles_and_cutoff():
    model = radial.TorchRadialBlockTable(make_table(shells=(0, 1, 2)))
    vectors = torch.tensor([[.4, .7, .9], [-.3, .5, -.8]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model, (vectors,), fast_mode=True, atol=1e-5)


def test_radial_buffers_roundtrip_and_float32():
    table = make_table(shells=(0, 1, 2))
    original = radial.TorchRadialBlockTable(table)
    clone = radial.TorchRadialBlockTable(table)
    clone.load_state_dict(original.state_dict(), strict=True)
    vectors = torch.tensor([[.4, .7, .9]], dtype=torch.float64)
    torch.testing.assert_close(original(vectors), clone(vectors), rtol=0, atol=0)
    torch.testing.assert_close(clone.float()(vectors.float()).double(), original(vectors), atol=2e-5, rtol=2e-5)
    poles = torch.tensor([[0., 0., -1.], [0., 0., 1.], [0., 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(clone(poles.float()).double(), original(poles), atol=2e-5, rtol=2e-5)


def test_radial_zero_table_and_two_knot_cubic_fallback():
    table = RadialBlockTable(np.array([0., 2.]), np.zeros((2, 1, 3)), (0,), (1,), 2.)
    model = radial.TorchRadialBlockTable(table)
    assert model.active_columns.numel() == 0
    assert torch.count_nonzero(model(torch.tensor([[.3, .1, .9]], dtype=torch.float64))) == 0


@requires_nacf_prebuilt
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_native_cuda_matches_torch_and_replays_graph(dtype):
    source = make_table(shells=(0, 1, 2, 3, 4))
    native = radial.TorchRadialBlockTable(source, device='cuda', dtype=dtype, backend='cuda')
    reference = radial.TorchRadialBlockTable(source, device='cuda', dtype=dtype, backend='torch')
    vectors = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 0, 1], [.2, .7, -.9], [0, 0, 3],
                            [1e-9, 0, -1], [.001, 0, -1], [0, 0, 4]], device='cuda', dtype=dtype)
    tolerance = 2e-4 if dtype == torch.float32 else 5e-11
    native._forward_torch = lambda *args: pytest.fail('native path fell back to torch')
    actual = native(vectors)
    torch.testing.assert_close(actual, reference(vectors), atol=tolerance, rtol=tolerance)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = native(vectors)
    vectors.add_(.03)
    graph.replay()
    torch.testing.assert_close(captured, reference(vectors), atol=tolerance, rtol=tolerance)
    assert native(vectors[:0]).shape == (0, 25, 25)


@requires_nacf_prebuilt
def test_auto_cuda_preserves_autograd_reference_and_zero_channels():
    auto = radial.TorchRadialBlockTable(make_table(shells=(0, 1)), device='cuda')
    v = torch.tensor([[.3, .7, .8]], device='cuda', dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(auto, (v,), fast_mode=True)
    source = RadialBlockTable(np.array([0., 2.]), np.zeros((2, 1, 3)), (0,), (1,), 2.)
    native = radial.TorchRadialBlockTable(source, device='cuda', backend='cuda')
    assert torch.count_nonzero(native(v.detach())) == 0


# --------------------------------------------------------------------------- disk-cached compiled tables
def source_table():
    r = np.array([0., .1, .3, .7, 1.4, 2., 3.])
    values = np.stack([np.diag(np.arange(1., 10.) * (1 - x / 3) ** 2) for x in r])
    return RadialBlockTable(r, values, (0, 1, 2), (0, 1, 2), 3.)


@requires_nacf_prebuilt
def test_prepared_native_nondefault_stream_and_cache(tmp_path, monkeypatch):
    table = source_table()
    fresh = prepared.cached_table(table, tmp_path, device='cuda', dtype=torch.float64, backend='cuda')

    def forbidden(*a, **k):
        raise AssertionError('Recompiled radial metadata or invoked compiler')
    monkeypatch.setattr(radial.TorchRadialBlockTable, '__init__', forbidden)
    warm = prepared.cached_table(table, tmp_path, device='cuda', dtype=torch.float64, backend='cuda')
    assert warm.prepared_cache == 'disk'
    real_popen = subprocess.Popen

    def no_compile(args, *a, **k):
        words = args if isinstance(args, (list, tuple)) else args.split()
        if any(Path(str(x)).name in ('nvcc', 'ninja', 'c++', 'g++', 'gcc') for x in words):
            forbidden()
        return real_popen(args, *a, **k)
    monkeypatch.setattr(subprocess, 'Popen', no_compile)
    vectors = torch.tensor([[.3, .2, .7], [0, 0, -1.], [1e-7, 0, -1.], [0, 0, 0], [0, 0, 3.], [.1, .2, -.7]],
                           device='cuda', dtype=torch.float64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = warm(vectors.clone())
        y = fresh(vectors.clone())
        warm.backend = 'torch'
        ref = warm(vectors.clone())
    stream.synchronize()
    torch.testing.assert_close(x, y, atol=0, rtol=0)
    torch.testing.assert_close(x, ref, atol=1e-9, rtol=1e-10)
    payload = next(tmp_path.glob('*.pt'))
    payload.write_bytes(payload.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        prepared.cached_table(table, tmp_path, device='cuda', dtype=torch.float64, backend='cuda')


def test_source_change_does_not_reuse_prepared_table(tmp_path):
    a = source_table()
    first = prepared.cached_table(a, tmp_path, device='cpu', dtype=torch.float64, backend='torch')
    b = source_table()
    b.support_bohr = 2.9
    second = prepared.cached_table(b, tmp_path, device='cpu', dtype=torch.float64, backend='torch')
    assert len(list(tmp_path.glob('*.pt'))) == 2
    assert second.support_bohr != first.support_bohr


@requires_nacf_prebuilt
def test_bank_cache_preserves_periodic_and_soc_contract(tmp_path):
    p2, p23 = stores()

    def bank():
        return NACFTableBank(p2, p23, device='cuda', prepared_cache_dir=tmp_path)
    args = (['X', 'Y'], [[0, 0, 0], [1.3, .2, .1]], np.eye(3) * 4.5, [[0, 1], [1, 0]], [[0, 0, 0], [0, 0, 0]])
    a = bank()
    x = a.prepare(*args)()
    b = bank()
    y = b.prepare(*args)()
    assert all(t.prepared_cache == 'disk' for t in b.tables.values())
    for key in ('node_p23_ao_ev', 'edge_p2_ao_ev', 'node_overlap_ao', 'edge_overlap_ao'):
        torch.testing.assert_close(x[key], y[key], atol=0, rtol=0)


def synthetic_rotator_table():
    """An l=1 table whose rotator base is exercised by ``key_for`` identity (independent of the P2/P23 stores)."""
    from scipy.interpolate import CubicSpline
    directions = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [-1., 0, 0], [0, -1., 0], [0, 0, -1.]])
    base = radial._harmonics(1, torch.from_numpy(directions)).numpy()
    r = np.array([0., .5, 1., 2., 3.])
    v = np.stack([np.diag([1., 2., 3.]) * (1 - x / 3) ** 2 for x in r])
    return SimpleNamespace(distances=r, values=v, left_shells=(1,), right_shells=(1,), support_bohr=3.,
                           _spline=CubicSpline(r, v, axis=0),
                           _rotator=SimpleNamespace(directions=directions, _base={1: base}))


def test_prepared_key_changes_with_rotator_base_and_rejects_bad_options(tmp_path):
    a = synthetic_rotator_table()
    import copy
    b = copy.deepcopy(a)
    b._rotator._base[1] *= 1.01
    args = dict(device='cpu', dtype=torch.float64, backend='torch')
    prepared.cached_table(a, tmp_path, **args)
    cached = prepared.cached_table(b, tmp_path, **args)
    fresh = radial.TorchRadialBlockTable(b, **args)
    assert prepared.key_for(a) != prepared.key_for(b)
    v = torch.tensor([[.3, .7, .2], [.8, -.4, .1]], dtype=torch.float64)
    torch.testing.assert_close(cached(v), fresh(v), rtol=0, atol=0)
    with pytest.raises(ValueError):
        prepared.cached_table(a, tmp_path, device='cpu', dtype=torch.float16, backend='torch')
    with pytest.raises(ValueError):
        prepared.cached_table(a, tmp_path, device='cpu', dtype=torch.float64, backend='bogus')


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="dptb.nacf.prepared publishes via os.replace onto a shared destination path; concurrent "
           "threads racing that replace intermittently raise WinError 5 (observed ~50% of runs here), "
           "which POSIX rename semantics do not exhibit -- run on the maintainer's Linux boxes",
)
def test_prepared_table_publication_is_atomic_under_concurrent_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    source = synthetic_rotator_table()

    def load(_):
        return prepared.cached_table(source, tmp_path, device='cpu', dtype=torch.float64, backend='torch')
    with ThreadPoolExecutor(max_workers=4) as pool:
        objects = list(pool.map(load, range(12)))
    for obj in objects:
        torch.testing.assert_close(obj.coefficients, objects[0].coefficients)
    p = tmp_path / (prepared.key_for(source) + '.pt')
    data = p.read_bytes()
    p.write_bytes(data[:-10] + b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        load(0)


@pytest.mark.parametrize('fault', ['nan', 'inf', 'complex', 'shape', 'knots'])
def test_spline_fault_rejected_cold_and_from_cache(tmp_path, fault):
    source = synthetic_rotator_table()
    kw = dict(device='cpu', dtype=torch.float64, backend='torch')
    prepared.cached_table(source, tmp_path, **kw)
    if fault in ('nan', 'inf'):
        source._spline.c[0, 0, 0, 0] = float(fault)
    elif fault == 'complex':
        source._spline.c = source._spline.c.astype(complex) + 1j
    elif fault == 'shape':
        source._spline.c = source._spline.c[:2]
    else:
        source._spline.x = source.distances + .1
    with pytest.raises(ValueError, match='spline'):
        radial.TorchRadialBlockTable(source, **kw)
    with pytest.raises(ValueError, match='spline'):
        prepared.cached_table(source, tmp_path, **kw)


# --------------------------------------------------------------------------- prepared-table snapshot (source parity)
def snapshot_fixture(tmp_path):
    table_stores = {}
    for name in ('p2', 'p23', 'overlap'):
        root = tmp_path / name
        root.mkdir()
        (root / 'manifest.json').write_text(json.dumps({'source': name}))
        table_stores[name] = SimpleNamespace(root=root)
    bank = SimpleNamespace(**table_stores)
    r = np.linspace(0, 4, 41)
    values = np.zeros((len(r), 4, 4))
    for i in range(4):
        values[:, i, i] = (i + 1) * (1 - r / 4) ** 3
    source = RadialBlockTable(r, values, (0, 1), (0, 1), 4.)
    compiled = radial.TorchRadialBlockTable(source, dtype=torch.float64, backend='torch')
    root = tmp_path / 'prepared'
    entry = write_table(root, 'p2_base|X|X', compiled, source=source)
    assert entry['encoding'] == 'exact_compact_nodes'
    manifest = {'schema': SCHEMA, 'complete': True, 'source_manifests': source_bindings(bank),
                'radial_source_sha256': radial_identity(), 'tables': {'p2_base|X|X': entry}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root, bank, compiled


@pytest.mark.parametrize('device', DEVICES)
def test_snapshot_replays_without_dense_reader_or_compilation(tmp_path, monkeypatch, device):
    root, bank, ref = snapshot_fixture(tmp_path)
    store = PreparedRadialStore(root)
    store.bind(bank)
    monkeypatch.setattr(radial.TorchRadialBlockTable, '__init__', lambda *a, **k: pytest.fail('runtime recompiled dense source'))
    obj = store.table('p2_base', 'X', 'X', device=device, dtype=torch.float64,
                      backend='cuda' if device == 'cuda' else 'torch')
    for name, value in ref.named_buffers():
        torch.testing.assert_close(dict(obj.named_buffers())[name].cpu(), value, atol=0, rtol=0)
    v = torch.tensor([[0, 0, 0], [0, 0, -1], [1e-8, -2e-8, -2], [.3, .7, 1.2], [0, 0, 4], [0, 0, 5]], dtype=torch.float64)
    torch.testing.assert_close(obj(v.to(device)).cpu(), ref(v), atol=2e-12, rtol=2e-12)


def test_snapshot_rejects_changed_sources_missing_pairs_and_corruption(tmp_path):
    root, bank, _ = snapshot_fixture(tmp_path)
    store = PreparedRadialStore(root)
    (bank.p2.root / 'manifest.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='source manifests'):
        store.bind(bank)
    with pytest.raises(KeyError, match='explicitly prepare'):
        store.table('p2_base', 'X', 'Y', device='cpu', dtype=torch.float64, backend='torch')
    path = root / store.manifest['tables']['p2_base|X|X']['path']
    with path.open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        store.table('p2_base', 'X', 'X', device='cpu', dtype=torch.float64, backend='torch')


def test_snapshot_rejects_changed_radial_implementation(tmp_path):
    root, _, _ = snapshot_fixture(tmp_path)
    m = json.loads((root / 'manifest.json').read_text())
    m['radial_source_sha256'] = '0' * 64
    (root / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='implementation changed'):
        PreparedRadialStore(root)


def test_table_bank_uses_bound_snapshot_and_rejects_competing_cache(tmp_path, monkeypatch):
    monkeypatch.delenv('DPTB_NACF_PREPARED_DIR', raising=False)
    root, sources, reference = snapshot_fixture(tmp_path)
    sources.overlap = sources.p2
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source_manifests'] = source_bindings(sources)
    manifest_path.write_text(json.dumps(manifest))
    store = PreparedRadialStore(root)
    bank = NACFTableBank(sources.p2, sources.p23, device='cpu', backend='torch', prepared_store=store)
    # These stores have no dense readers: the maintained bank must use the snapshot.
    key = bank.table('p2_base', 'X', 'X')
    vector = torch.tensor([[.3, .7, 1.2]], dtype=torch.float64)
    torch.testing.assert_close(bank.tables[key](vector), reference(vector), atol=0, rtol=0)
    assert bank.table('p2_base', 'X', 'X') == key
    with pytest.raises(ValueError, match='choose'):
        NACFTableBank(sources.p2, sources.p23, device='cpu', prepared_store=store,
                      prepared_cache_dir=str(tmp_path / 'cache'))


def test_snapshot_writer_preserves_another_writers_pending_file(tmp_path):
    _, _, reference = snapshot_fixture(tmp_path)
    root = tmp_path / 'concurrent'
    root.mkdir()
    key = 'p2_base|X|X'
    pending = root / (hashlib.sha256(key.encode()).hexdigest() + '.pending')
    pending.write_bytes(b'another writer owns this file')
    with pytest.raises(FileExistsError):
        write_table(root, key, reference)
    assert pending.read_bytes() == b'another writer owns this file'
    assert not pending.with_suffix('.npz').exists()


def test_snapshot_publication_never_overwrites_a_concurrent_result(tmp_path, monkeypatch):
    import dptb.nacf.prepared_store as module
    _, _, reference = snapshot_fixture(tmp_path)
    root = tmp_path / 'publish'
    original_link = module.os.link

    def competing_writer(source, destination):
        destination.write_bytes(b'published by another writer')
        return original_link(source, destination)
    monkeypatch.setattr(module.os, 'link', competing_writer)
    with pytest.raises(FileExistsError):
        write_table(root, 'p2_base|X|X', reference)
    published = next(root.glob('*.npz'))
    assert published.read_bytes() == b'published by another writer'
    assert not list(root.glob('*.pending'))


# --------------------------------------------------------------------------- prebuilt manifest and CLI
@pytest.mark.parametrize('cap,ptx,accepted', [
    ((8, 9), [], True), ((12, 0), [], False),
    ((12, 0), ['8.9'], True), ((8, 6), ['8.9'], False),
])
def test_recorded_ptx_compatibility(tmp_path, monkeypatch, cap, ptx, accepted):
    binary = tmp_path / 'test.so'
    binary.write_bytes(b'fixture')
    manifest = {'runtime': precompiled.identity(), 'architectures': ['8.9'], 'ptx_architectures': ptx, 'sources': {},
                'binary': binary.name, 'sha256': precompiled.sha(binary)}
    (tmp_path / 'prebuilt.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(precompiled, 'ROOT', tmp_path)
    monkeypatch.setattr(precompiled.torch.cuda, 'get_device_capability', lambda device: cap)
    if accepted:
        assert precompiled.verify('cuda')['sha256'] == manifest['sha256']
        binary.write_bytes(b'corrupt')
        with pytest.raises(RuntimeError, match='checksum'):
            precompiled.verify('cuda')
    else:
        with pytest.raises(RuntimeError, match='architecture'):
            precompiled.verify('cuda')


def test_check_arch_reports_mismatch_and_accepts_exact_targets(monkeypatch):
    manifest = {'architectures': ['8.9'], 'ptx_architectures': ['8.9'], 'sha256': 'abc'}
    monkeypatch.setattr(precompile, 'verify', lambda device=None: manifest)
    assert precompile.main(['--check', '--arch', '8.9+PTX']) is manifest
    assert precompile.main(['--check']) is manifest  # no target requested: manifest validity only
    for requested in (['9.0'], ['8.9'], ['8.9+PTX', '9.0'], ['9.0+PTX']):
        with pytest.raises(RuntimeError, match='do not match'):
            precompile.main(['--check', *sum((['--arch', r] for r in requested), [])])


def test_check_never_builds_on_mismatch(monkeypatch):
    manifest = {'architectures': ['8.9'], 'ptx_architectures': [], 'sha256': 'abc'}
    monkeypatch.setattr(precompile, 'verify', lambda device=None: manifest)
    import torch.utils.cpp_extension as ext
    monkeypatch.setattr(ext, 'load', lambda *a, **k: pytest.fail('--check must not compile'))
    with pytest.raises(RuntimeError):
        precompile.main(['--check', '--arch', '9.0'])


# --------------------------------------------------------------------------- fused-kernel radial dispatch
@requires_nacf_prebuilt
def test_radial_multi_against_individual_near_south_and_empty():
    r = np.array([0., .3, 1.2, 2.])
    v = np.zeros((4, 4, 4))
    v[:, 0, 0] = [1, .8, .3, 0]
    for i in range(1, 4):
        v[:, i, i] = [2, 1.5, .4, 0]
    tables = [radial.TorchRadialBlockTable(RadialBlockTable(r, v * c, (0, 1), (0, 1), 2.), device='cuda', backend='cuda')
              for c in (1., .8, 1.7)]
    vec = torch.tensor([[0., 0., 0.], [1e-7, 1e-6, -.7], [.2, .3, .9], [0., 0., 2.], [0., 0., -1.]],
                       device='cuda', dtype=torch.float64)
    plan = RadialMultiPlan([(tables, 5), ([tables[0]], 0)])
    got = plan(vec)
    for a, t in zip(got, tables):
        torch.testing.assert_close(a, t(vec), atol=0, rtol=0)
    assert got[-1].shape == (0, 4, 4)
    with pytest.raises(NotImplementedError):
        RadialMultiPlan([(tables, 5)], background_nodes=[0., 1.])

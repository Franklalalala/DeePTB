import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.data.interfaces.p2_table import RadialBlockTable
from dptb.nacf.radial import TorchRadialBlockTable
from dptb.nacf.assembly import NACFTableBank
from dptb.nacf.prepared_store import (
    PreparedRadialStore, SCHEMA, radial_identity, source_bindings, write_table,
)


def fixture(tmp_path):
    stores = {}
    for name in ('p2', 'p23', 'overlap'):
        root = tmp_path / name
        root.mkdir()
        (root / 'manifest.json').write_text(json.dumps({'source': name}))
        stores[name] = SimpleNamespace(root=root)
    bank = SimpleNamespace(**stores)
    r = np.linspace(0, 4, 41)
    values = np.zeros((len(r), 4, 4))
    for i in range(4):
        values[:, i, i] = (i + 1) * (1 - r / 4) ** 3
    source = RadialBlockTable(r, values, (0, 1), (0, 1), 4.)
    compiled = TorchRadialBlockTable(source, dtype=torch.float64, backend='torch')
    root = tmp_path / 'prepared'
    entry = write_table(root, 'p2_base|X|X', compiled, source=source)
    assert entry['encoding'] == 'exact_compact_nodes'
    manifest = {'schema': SCHEMA, 'complete': True, 'source_manifests': source_bindings(bank),
                'radial_source_sha256': radial_identity(), 'tables': {'p2_base|X|X': entry}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root, bank, compiled


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_snapshot_replays_without_dense_reader_or_compilation(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    root, bank, ref = fixture(tmp_path)
    store = PreparedRadialStore(root)
    store.bind(bank)
    monkeypatch.setattr(TorchRadialBlockTable, '__init__', lambda *a, **k: pytest.fail('runtime recompiled dense source'))
    obj = store.table('p2_base', 'X', 'X', device=device, dtype=torch.float64,
                      backend='cuda' if device == 'cuda' else 'torch')
    for name, value in ref.named_buffers():
        torch.testing.assert_close(dict(obj.named_buffers())[name].cpu(), value, atol=0, rtol=0)
    v = torch.tensor([[0, 0, 0], [0, 0, -1], [1e-8, -2e-8, -2], [.3, .7, 1.2], [0, 0, 4], [0, 0, 5]], dtype=torch.float64)
    torch.testing.assert_close(obj(v.to(device)).cpu(), ref(v), atol=2e-12, rtol=2e-12)


def test_snapshot_rejects_changed_sources_missing_pairs_and_corruption(tmp_path):
    root, bank, _ = fixture(tmp_path)
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
    root, _, _ = fixture(tmp_path)
    m = json.loads((root / 'manifest.json').read_text())
    m['radial_source_sha256'] = '0' * 64
    (root / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='implementation changed'):
        PreparedRadialStore(root)


def test_table_bank_uses_bound_snapshot_and_rejects_competing_cache(tmp_path, monkeypatch):
    monkeypatch.delenv('DPTB_NACF_PREPARED_DIR', raising=False)
    root, sources, reference = fixture(tmp_path)
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
    _, _, reference = fixture(tmp_path)
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
    _, _, reference = fixture(tmp_path)
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

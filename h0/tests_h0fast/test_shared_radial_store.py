import json
import sqlite3
import numpy as np
import pytest
import torch
from h0rebuild import offline, shared_radial_store as shared, radial_codec as codec


@pytest.fixture
def stores(tmp_path, monkeypatch):
    from h0rebuild import table_contract
    monkeypatch.setattr(table_contract, 'verify_contract', lambda p: None)
    source = tmp_path/'original'; source.mkdir()
    (source/'species').mkdir()
    offline.save(source/'species'/'s.npz', {'spline':np.arange(12.).reshape(4,3)})
    (source/'source_contract.json').write_text('{}')
    x=np.arange(17,dtype=float)/8
    curve=np.stack([3+2*x[:-1]+x[:-1]**3, 2+3*x[:-1]**2,3*x[:-1],np.ones(16)],axis=-1)
    curve[0,2]=-0.0
    original={}
    for i, dr in enumerate((.125,.125,.25)):
        coeff=torch.from_numpy(np.stack([curve,curve,curve*2]))
        state={'S_coeffs':coeff.clone(), 'dr':dr,'cutoff':dr*16,'nr':17,
            'D':torch.tensor([[1,2j],[-2j,-3]],dtype=torch.complex128),
            'S_index_map':torch.tensor([2,0,1,-1]),'metadata':{'note':'unchanged'},
            **{k:coeff.clone() for k in shared.COEFFS}}
        key=str(i)*64
        record={'key':key,'sources':{},'checksum':offline.fingerprint(state),'state':state}
        offline.save(source/'two_center'/f'{key}.npz',record)
        original[key]=record
    destination=tmp_path/'shared'
    manifest=shared.convert_store(source,destination)
    return source,destination,manifest,original


def test_complete_roundtrip_shares_only_identical_grids(stores):
    source,dest,m,original=stores
    assert m['unique_curves']==4  # two curves, two distinct grids
    assert m['reloaded_tables']==3
    assert (source/'source_contract.json').read_bytes()==(dest/'source_contract.json').read_bytes()
    for key,record in original.items():
        got=shared.read_record(dest,key)
        assert offline.fingerprint(got)==offline.fingerprint(record)
    assert shared.is_shared_store(dest)


def test_corrupt_curve_is_rejected_after_reconstruction(stores):
    _,dest,_,original=stores
    with sqlite3.connect(dest/'curves.sqlite') as db:
        db.execute('UPDATE curves SET nodes=zeroblob(length(nodes))')
    with pytest.raises(ValueError,match='checksum'):
        shared.read_record(dest,next(iter(original)))


def test_index_swap_is_rejected(stores):
    _,dest,_,original=stores
    keys=list(original)
    (dest/'two_center'/f'{keys[0]}.npz').write_bytes((dest/'two_center'/f'{keys[1]}.npz').read_bytes())
    with pytest.raises(ValueError,match='index checksum'):
        shared.read_record(dest,keys[0])


def test_codec_incompatible_arithmetic_is_rejected(monkeypatch):
    c=np.ones((1,3,4));r=codec.pack(c,.01)
    derive=codec._derive
    def changed(*args):
        a=derive(*args);a[0,0,0]=np.nextafter(a[0,0,0],np.inf);return a
    monkeypatch.setattr(codec,'_derive',changed)
    with pytest.raises(ValueError,match='checksum'):codec.unpack(r)


def test_codec_overflow_policy_does_not_prevent_bit_correction():
    c=np.full((1,2,4),1e308);c[:,1]=-1e308
    with np.errstate(all='raise'):
        assert np.array_equal(codec.unpack(codec.pack(c,.01)).view(np.uint64),c.view(np.uint64))

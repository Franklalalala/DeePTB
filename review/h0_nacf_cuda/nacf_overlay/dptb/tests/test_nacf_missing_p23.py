"""Geometry-only inference must reproduce the pinned training composition rule."""
import numpy as np
import pytest
import torch
from dptb.tests.test_nacf_gpu import stores
from dptb.nacf.assembly import NACFTableBank, NACFBatchAssemblyPlan
from dptb.data.interfaces.p2_table import P2TableAssembler

PIN='a'*64


def bank_with_missing_pair(device='cpu', **kwargs):
    p2,p23=stores()
    p23.manifest_sha256=PIN
    p23.has_factor=lambda a,b:(a,b)!=('X','Y')
    bank=NACFTableBank(p2,p23,device=device,backend='torch',**kwargs)
    return bank


def geometry(symbols):
    n=len(symbols)
    edges=np.array([[0,1],[1,0]]) if n==2 else np.empty((2,0),dtype=int)
    return (symbols,np.array([[i*1.3,.2*i,.1*i] for i in range(n)]),np.eye(3)*4.5,
            edges,np.zeros((edges.shape[1],3),dtype=int))


def test_missing_pair_requires_explicit_pinned_policy():
    with pytest.raises(ValueError,match='trusted P23'):
        bank_with_missing_pair(p23_missing_policy='p2_if_missing_pairs')
    with pytest.raises(ValueError,match='fingerprint'):
        bank_with_missing_pair(p23_missing_policy='p2_if_missing_pairs',expected_p23_sha256='b'*64)
    with pytest.raises(KeyError,match='X\\|Y'):
        bank_with_missing_pair().prepare(*geometry(['X','Y']))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_pinned_composition_fallback_matches_p2_and_mixed_batch_keeps_p23(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    bank=bank_with_missing_pair(device,p23_missing_policy='p2_if_missing_pairs',expected_p23_sha256=PIN)
    missing=bank.prepare(*geometry(['X','Y']))
    complete=bank.prepare(*geometry(['X','X']))
    assert not missing.p23_used and missing.p23_missing==('X|Y',)
    assert complete.p23_used
    assert all(pair[0]!='vna' for _,pair,_ in missing.query_specs)
    assert any(pair[0]=='vna' for _,pair,_ in complete.query_specs)
    actual=missing()
    symbols,pos,cell,_,_=geometry(['X','Y'])
    cpu=P2TableAssembler(bank.p2)
    expected=np.stack([cpu.assemble_block(symbols=symbols,positions_bohr=pos,cell_bohr=cell,
                 i=i,j=i,translation=(0,0,0)) for i in range(2)])*bank.ry_to_ev
    np.testing.assert_allclose(actual['node_p23_ao_ev'].cpu(),expected,atol=2e-8)
    merged=NACFBatchAssemblyPlan([missing,complete])
    assert merged.p23_used==(False,True)
    combined=merged()
    complete_values=complete()
    for key in actual:
        expected=(torch.cat([actual[key],complete_values[key]+missing.natoms],dim=1)
                  if key=='edge_index' else torch.cat([actual[key],complete_values[key]]))
        torch.testing.assert_close(combined[key],expected,atol=2e-8,rtol=1e-8)


def test_listed_but_missing_payload_does_not_enable_fallback():
    bank=bank_with_missing_pair(p23_missing_policy='p2_if_missing_pairs',expected_p23_sha256=PIN)
    def missing_file(*args):raise FileNotFoundError('listed P23 payload missing')
    bank.p23.factor=missing_file
    with pytest.raises(FileNotFoundError,match='payload'):
        bank.prepare(*geometry(['X','X']))

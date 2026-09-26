"""Real-record join validation, row alignment, shared LMDB cache and workers."""
import copy
import json
import os
import pickle
from pathlib import Path
import shutil

import lmdb
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dptb.data.dataloader import Collater
from dptb.data.dataset.record_codec import loads_record
from dptb.tests.shift_head_helpers import mini_dataset,close_dataset


@pytest.fixture
def fixture_root(tmp_path):
    source=os.environ.get('R12B_MINI_ROOT')
    if not source:pytest.skip('R12B_MINI_ROOT must point to the rekeyed real-record fixture')
    root=tmp_path/'mini';shutil.copytree(source,root)
    return root


def raw_records(root, part):
    env=lmdb.open(str(root/part/'data.test.s000.lmdb'),readonly=True,lock=False)
    with env.begin() as txn:records=[loads_record(v) for k,v in txn.cursor()]
    env.close();return records


@pytest.mark.parametrize('cutoff',[0.5,8.0])
def test_real_join_collate_row_order_and_disabled_identity(fixture_root,cutoff):
    raw=raw_records(fixture_root,'test');side=raw_records(fixture_root,'ovl_test')
    ds=mini_dataset(fixture_root,r_max=cutoff)
    samples=[ds[i] for i in range(4)]
    assert len(ds._lmdb_env_cache)==2
    close_dataset(ds)
    plain=mini_dataset(fixture_root,False,r_max=cutoff)
    for i,sample in enumerate(samples):
        old=plain[i]
        assert 'phys_node_overlap' not in old and 'phys_edge_overlap' not in old
        for key in old.keys:
            if torch.is_tensor(old[key]):assert torch.equal(sample[key],old[key]),key
        for key in ['node_features','edge_features','edge_index','edge_cell_shift']:
            assert torch.equal(sample[key],torch.as_tensor(raw[i][key]).to(sample[key]))
        for part in ('node','edge'):
            assert torch.equal(sample['phys_'+part+'_overlap'],torch.from_numpy(side[i][part+'_overlap']))
    b=Collater()([samples[2],samples[0],samples[3],samples[1]])
    for part in ('node','edge'):
        assert torch.equal(b['phys_'+part+'_overlap'],torch.cat([samples[i]['phys_'+part+'_overlap'] for i in [2,0,3,1]]))
        assert torch.equal(b[part+'_features'],torch.cat([samples[i][part+'_features'] for i in [2,0,3,1]]))
    # Directed edges must receive the same offsets as their S rows.
    offset=0;edges=[]
    for i in [2,0,3,1]:edges.append(samples[i]['edge_index']+offset);offset+=samples[i].num_nodes
    assert torch.equal(b['edge_index'],torch.cat(edges,dim=1))
    close_dataset(plain)


@pytest.mark.parametrize('corrupt',['idx','main_idx','edge_graph_fingerprint','record_uid','node_shape','edge_shape','missing_field','missing_key','missing_shard','nan','dtype'])
def test_sidecar_fails_closed(fixture_root,corrupt):
    part='test' if corrupt=='main_idx' else 'ovl_test'
    path=fixture_root/part/'data.test.s000.lmdb'
    if corrupt=='missing_shard':shutil.rmtree(path)
    else:
        env=lmdb.open(str(path),map_size=16*1024**2)
        with env.begin(write=True) as txn:
            key=(0).to_bytes(4,'big');d=loads_record(txn.get(key))
            if corrupt=='missing_key':txn.delete(key)
            else:
                if corrupt in ('idx','main_idx'):d['idx']=99
                elif corrupt=='edge_graph_fingerprint':d[corrupt]='0'*64
                elif corrupt=='record_uid':d[corrupt]='wrong-record'
                elif corrupt=='node_shape':d['node_overlap']=d['node_overlap'][:,:-1]
                elif corrupt=='edge_shape':d['edge_overlap']=d['edge_overlap'][:-1]
                elif corrupt=='missing_field':d.pop('node_overlap')
                elif corrupt=='nan':d['edge_overlap'][0,0]=np.nan
                elif corrupt=='dtype':d['node_overlap']=d['node_overlap'].astype(np.float64)
                txn.put(key,pickle.dumps(d))
        env.close()
    ds=mini_dataset(fixture_root)
    with pytest.raises((ValueError,FileNotFoundError),match='Overlap sidecar'):ds[0]
    close_dataset(ds)


def pickle_collate(samples):
    # Pipe-only result transport for sandboxes that prohibit AF_UNIX FD sharing.
    # The real Collater still executes inside the DataLoader worker.
    return pickle.dumps(Collater()(samples), protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize('context',['spawn','fork'])
def test_cached_dataset_pickle_and_multiprocess(fixture_root,context):
    ds=mini_dataset(fixture_root)
    expected=Collater()([ds[i] for i in range(4)])
    assert len(ds._lmdb_env_cache)==2
    state=ds.__getstate__();assert state['_lmdb_env_cache']=={}
    restored=pickle.loads(pickle.dumps(ds));assert restored._lmdb_env_cache=={}
    # Spawn must reopen both shards; fork may inherit read-only cache safely.
    pipe_only=os.environ.get('R12B_PICKLE_WORKER_RESULTS')=='1'
    loader=DataLoader(ds,batch_size=4,num_workers=2,collate_fn=pickle_collate if pipe_only else Collater(),
                      multiprocessing_context=context,timeout=45)
    got=list(loader);assert len(got)==1
    if pipe_only:got=[pickle.loads(x) for x in got]
    for key in ('phys_node_overlap','phys_edge_overlap','node_features','edge_features','edge_index'):
        assert torch.equal(got[0][key],expected[key]),key
    close_dataset(ds)
    assert torch.equal(restored[0]['phys_node_overlap'],expected['phys_node_overlap'][:2])
    close_dataset(restored)


def test_main_and_sidecar_use_same_pread_hook_and_cache(fixture_root,monkeypatch):
    ds=mini_dataset(fixture_root)
    real_open=lmdb.open;opens=[]
    def pread_hook(path,**kwargs):
        # Sitecustomize replaces lmdb.open at this same boundary on Hopper.
        opens.append((path,kwargs.copy()))
        return real_open(path,**kwargs)
    monkeypatch.setattr(lmdb,'open',pread_hook)
    ds[0];ds[1]
    assert len(opens)==2
    assert opens[0][1]==opens[1][1]
    assert opens[1][1]==dict(readonly=True,lock=False,readahead=False,max_readers=2048)
    assert Path(opens[0][0]).name==Path(opens[1][0]).name
    close_dataset(ds)

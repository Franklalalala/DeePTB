"""Opt-in small real-record training, with measured step times and CUDA memory."""
import copy
import json
import os
from pathlib import Path
import time

import pytest
import torch

from dptb.data.dataloader import Collater
from dptb.nn.build import build_model
from dptb.nnops.loss import HamilLossAbs
from dptb.tests.shift_head_helpers import config,mini_dataset,close_dataset


@pytest.mark.parametrize('mode,freeze,scope',[('atom',True,'onsite'),('shell',False,'onsite'),('atom',False,'hopping')])
def test_mini_real_training(mode,freeze,scope,record_property):
    root=os.environ.get('R12B_MINI_ROOT')
    if not root:pytest.skip('Set R12B_MINI_ROOT to the real-record fixture')
    device=os.environ.get('R12B_TEST_DEVICE','cpu')
    if device.startswith('cuda'):assert torch.cuda.is_available(), 'GPU test must execute, never silently skip'
    ds=mini_dataset(root)
    data=Collater()([ds[i] for i in range(4)]).to_dict();close_dataset(ds)
    cfg=config(mode=mode,device=device,scope=scope)
    cfg['common_options']['basis']=json.loads((Path(root)/'basis.json').read_text())
    cfg['model_options']['shift_head']['freeze_backbone']=freeze
    cfg['model_options']['embedding'].update(n_layers=1,irreps_hidden='2x0e+2x1o+2x2e',
        latent_dim=4,latent_channels=[4],edge_one_hot_dim=4,env_embed_multiplicity=1,r_max=8.0,
        so2_fusion_mode=os.environ.get('R12B_SO2_MODE','staged'),
        mole_linear_mode=os.environ.get('R12B_MOLE_MODE','split_loop'))
    if scope=='hopping':cfg['train_options']['distance_ranges']=[[1e-6,8.0]]
    model=build_model(**cfg)
    data={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in data.items()}
    loss_fn=HamilLossAbs(idp=model.idp,device=device)
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=0.001)
    rows=[]
    for step in range(8):
        if device.startswith('cuda'):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();opt.zero_grad(set_to_none=True)
        out=model(copy.deepcopy(data));em,nm=model._build_expert_masks(out,0)
        out.update(expert_edge_mask=em,expert_node_mask=nm)
        loss=loss_fn(out,data)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        if device.startswith('cuda'):torch.cuda.synchronize()
        rows.append(dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-start,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
            shift_stats={k:float(v) for k,v in loss_fn.last_shift_stats.items()}))
    assert rows[-1]['loss']<rows[0]['loss'],rows
    record_property('initial_loss',rows[0]['loss']);record_property('final_loss',rows[-1]['loss'])
    record_property('mean_seconds',sum(r['seconds'] for r in rows)/len(rows))
    dest=os.environ.get('R12B_METRICS_DIR')
    if dest:
        Path(dest).mkdir(parents=True,exist_ok=True)
        Path(dest,f'train_{mode}_{scope}_freeze{freeze}_{device.replace(":","_")}.json').write_text(json.dumps(dict(
            device=device,config=cfg,rows=rows,model_parameters=sum(p.numel() for p in model.parameters()),
            head_parameters=sum(p.numel() for p in model.shift_head.parameters()),
            nodes=data['pos'].shape[0],edges=data['edge_index'].shape[1],
            overlap_bytes=sum(data[k].numel()*data[k].element_size() for k in ('phys_node_overlap','phys_edge_overlap'))),indent=2))

"""Opt-in real LMDB training benchmark: synchronized forward/backward/update."""
import copy
import json
import os
from pathlib import Path
import time

import pytest
import torch
from dptb.data.dataloader import Collater
from dptb.nn.build import build_model
from dptb.nn.chemical_readout import initialize_chemical_readouts
from dptb.nnops.loss import HamilLossAbs
from dptb.tests.chemical_readout_helpers import chemical_config, bitwise
from dptb.tests.shift_head_helpers import mini_dataset, close_dataset, save_model


@pytest.mark.parametrize('mode', ['shared', 'chemical_core'])
def test_real_training_and_timing(mode, tmp_path):
    root = os.environ.get('R14B_MINI_ROOT')
    if not root: pytest.skip('Set R14B_MINI_ROOT to the supplied four-record LMDB fixture')
    device = os.environ.get('R14B_TEST_DEVICE', 'cpu')
    if device.startswith('cuda'): assert torch.cuda.is_available(), 'GPU tests must execute'
    dataset = mini_dataset(root, sidecar=False)
    cfg = chemical_config(mode, device=device, scope='onsite')
    cfg['common_options']['basis'] = json.loads((Path(root)/'basis.json').read_text())
    cfg['model_options']['embedding'].update(n_layers=1, irreps_hidden='2x0e+2x1o+2x2e',
        latent_dim=4, latent_channels=[4], edge_one_hot_dim=4, env_embed_multiplicity=1, r_max=8.0,
        so2_fusion_mode=os.environ.get('R14B_SO2_MODE','staged'),
        mole_linear_mode=os.environ.get('R14B_MOLE_MODE','split_loop'))
    model = build_model(**cfg)
    initialize_chemical_readouts(model, dataset)
    data = Collater()([dataset[i] for i in range(len(dataset))]).to_dict()
    # Independent structure-presence oracle, not atom frequency.
    head = model.experts[0].embedding.chemical_core
    if head is not None:
        expected=torch.zeros_like(head.n_g)
        for i in range(len(dataset)): expected[dataset[i]['atom_types'].unique().to(expected.device)]+=1
        assert torch.equal(head.n_g,expected)
    close_dataset(dataset)
    data={k:v.to(device) if torch.is_tensor(v) else v for k,v in data.items()}
    loss_fn=HamilLossAbs(idp=model.idp,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    rows=[]
    for step in range(8):
        if device.startswith('cuda'):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();opt.zero_grad(set_to_none=True)
        out=model(copy.deepcopy(data));em,nm=model._build_expert_masks(out,0)
        assert not em.any() and nm.all()
        out.update(expert_edge_mask=em,expert_node_mask=nm)
        out['edge_features'].retain_grad()
        loss=loss_fn(out,data);assert torch.isfinite(loss)
        loss.backward()
        assert out['edge_features'].grad is None or out['edge_features'].grad.count_nonzero()==0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if head is not None:
            assert sum(b.D.grad.abs().sum().item() for b in head.blocks)>0
            if step>0: assert sum(b.P.grad.abs().sum().item()+b.Q.grad.abs().sum().item() for b in head.blocks)>0
        opt.step()
        if device.startswith('cuda'): torch.cuda.synchronize()
        rows.append(dict(step=step, loss=float(loss.detach()), seconds=time.perf_counter()-start,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                         peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.startswith('cuda') else None))
    assert rows[-1]['loss']<rows[0]['loss']
    saved=tmp_path/'real_checkpoint.pth';save_model(model,cfg,saved)
    restored=build_model(checkpoint=str(saved),device=device)
    bitwise(model(copy.deepcopy(data))['node_features'],restored(copy.deepcopy(data))['node_features'])
    result=dict(mode=mode,device=device,torch_version=torch.__version__,config=cfg,rows=rows,
                mean_seconds_after_2_warmup=sum(r['seconds'] for r in rows[2:])/6,
                nodes=data['pos'].shape[0],edges=data['edge_index'].shape[1],structures=4,
                model_parameters=sum(p.numel() for p in model.parameters()),
                readout_parameters=sum(p.numel() for p in model.experts[0].embedding.out_node.parameters()),
                chemical_parameters=sum(p.numel() for p in head.parameters()) if head else 0,
                counts=head.n_g.tolist() if head else None,atomic_numbers=head.atomic_numbers.tolist() if head else None,
                blocks=[dict(irrep=str(ir),in_channels=b.Q.shape[0],out_channels=b.P.shape[0],cap=float(b.c))
                        for b,(ir,_,_) in zip(head.blocks,head.specs)] if head else None,
                optimizer='Adam(lr=0.001), smoke only; production configs inherit HybridMuon')
    if device.startswith('cuda'):result['gpu']=torch.cuda.get_device_name()
    dest=os.environ.get('R14B_METRICS_DIR')
    if dest:
        Path(dest).mkdir(parents=True,exist_ok=True)
        Path(dest,f'train_{mode}_{device.replace(":","_")}.json').write_text(json.dumps(result,indent=2))

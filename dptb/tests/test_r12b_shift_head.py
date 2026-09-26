"""Potential formula, exact initialization, gradient scope, resume and loss contract."""
import copy
import os
from pathlib import Path

import pytest
import torch

from dptb.data.transforms import OrbitalMapper
from dptb.nn.build import build_model
from dptb.nn.shift_head import PotentialShiftHead, add_compact_delta, normalize_shift_options, optimizer_named_parameters
from dptb.nnops.layout import project_uureal_to_like
from dptb.nnops.loss import HamilLossAbs
from dptb.tests.shift_head_helpers import config, make_model, batch, save_model

DEVICES = [os.environ.get("R12B_TEST_DEVICE", "cpu")]


def bitwise(a,b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("moe", [False,True])
@pytest.mark.parametrize("mode", ["off", "atom", "shell"])
def test_off_and_zero_initialization_outputs_gradients(mode,moe,device):
    base = make_model(moe=moe,device=device)
    shifted = make_model(mode=mode,moe=moe,device=device)
    data = batch(base)
    a,b = base(copy.deepcopy(data)),shifted(copy.deepcopy(data))
    for k in ("node_features","edge_features"):
        bitwise(a[k],b[k])
    for out in (a,b):
        (out['node_features'].square().sum()+out['edge_features'].square().sum()).backward()
    bp=dict(base.named_parameters())
    for name,p in shifted.named_parameters():
        if name.startswith('shift_head.'):
            continue
        if bp[name].grad is None:
            assert p.grad is None
        else:
            bitwise(bp[name].grad,p.grad)


@pytest.mark.parametrize("mode", ["atom","shell"])
@pytest.mark.parametrize("scope", ["both","onsite","hopping"])
def test_formula_reverse_edges_and_distance_mask(mode,scope):
    model = make_model(mode=mode,scope=scope)
    data=batch(model)
    data = __import__('dptb.data.AtomicDataDict',fromlist=['with_edge_vectors']).with_edge_vectors(data,with_lengths=True)
    h=model.shift_head
    # Independent AO-level oracle, including separate 1s / 2s channels.
    shell_ao=torch.tensor([0,1,2,2,2])
    v=torch.arange(1,1+3*(1 if mode=='atom' else 3),dtype=torch.float32).reshape(3,-1)/8
    nmat=torch.arange(75,dtype=torch.float32).reshape(3,5,5)/16
    emat=torch.arange(150,dtype=torch.float32).reshape(6,5,5)/16
    for a,b in [(0,1),(2,3),(4,5)]: emat[b]=emat[a].T
    # Encode physical matrices independently with the public AO block slices.
    from dptb.data.interfaces.blockwise_tensor import edge_feature_slices
    from dptb.data.interfaces.blockwise_tensor import onsite_feature_slices
    def encode(mats, node=False):
        result=torch.zeros(len(mats),25)
        iterator = onsite_feature_slices(model.idp,'H') if node else edge_feature_slices(model.idp,'H','H')
        for si,sj,sl in iterator:
            result[:,sl]=mats[:,si,sj].reshape(len(mats),-1)
        return result
    data['phys_node_overlap']=encode(nmat,True); data['phys_edge_overlap']=encode(emat)
    vi=v.expand(-1,5) if mode=='atom' else v[:,shell_ao]
    en=((vi[:,:,None]+vi[:,None,:])*0.5)*nmat
    i,j=data['edge_index']; ee=((vi[i,:,None]+vi[j,None,:])*0.5)*emat
    if scope=='onsite': ee.zero_()
    if scope=='hopping':
        en.zero_();ee[data['edge_lengths'].flatten()>=2.0]=0
    dn,de=h.assemble(data,v)
    bitwise(dn,encode(en,True));bitwise(de,encode(ee))
    # Check transpose in AO coordinates; encoded layout is not a naive 5x5 flatten.
    recovered=torch.zeros_like(ee)
    for si,sj,sl in edge_feature_slices(model.idp,'H','H'):
        recovered[:,si,sj]=de[:,sl].reshape(len(de),si.stop-si.start,sj.stop-sj.start)
    for a,b in [(0,1),(2,3),(4,5)]:bitwise(recovered[b],recovered[a].transpose(0,1))
    _,active=h.assemble(data,v,torch.tensor([0,1]))
    assert torch.count_nonzero(active[2:])==0


@pytest.mark.parametrize('scope',['onsite','hopping'])
@pytest.mark.parametrize('device',DEVICES)
def test_nonzero_shift_loss_and_eval_projection_agree(scope,device):
    model=make_model(mode='atom',scope=scope,device=device)
    data=batch(model)
    before=model(copy.deepcopy(data))
    with torch.no_grad():model.shift_head.mlp[-1].bias.fill_(.25)
    out=model(copy.deepcopy(data))
    # diag_eval_v1f uses this exact helper on model(bc), then the target mask.
    for part in ('node','edge'):
        p,_=project_uureal_to_like(model.idp,out[part+'_features'],data[part+'_features'])
        old,_=project_uureal_to_like(model.idp,before[part+'_features'],data[part+'_features'])
        bitwise(p,old+out[part+'_shift_delta'])
    edge_mask,node_mask=model._build_expert_masks(out,0)
    out['expert_edge_mask']=edge_mask;out['expert_node_mask']=node_mask
    loss=HamilLossAbs(idp=model.idp,device=device)
    value=loss(out,data)
    a=project_uureal_to_like(model.idp,out['node_features'],data['node_features'])[0]
    b=project_uureal_to_like(model.idp,out['edge_features'],data['edge_features'])[0]
    torch.testing.assert_close(loss.last_onsite_l1_sum,((a-data['node_features'])*node_mask[:,None]).abs().sum())
    torch.testing.assert_close(loss.last_hopping_l1_sum,((b-data['edge_features'])*edge_mask[:,None]).abs().sum())
    assert torch.isfinite(value) and 'task_loss_change' in loss.last_shift_stats
    before.update(expert_edge_mask=edge_mask,expert_node_mask=node_mask)
    base_loss=HamilLossAbs(idp=model.idp,device=device)(before,data)
    torch.testing.assert_close(loss.last_shift_stats['task_loss_change'],value.detach()-base_loss.detach(),atol=2e-7,rtol=2e-6)
    for name,expected in [('v_mean',.25),('v_std',0.),('v_absmax',.25)]:
        assert float(model.shift_head.last_stats[name])==expected
    value.backward()
    assert model.shift_head.mlp[-1].bias.grad.abs().sum()>0


@pytest.mark.parametrize('device',DEVICES)
def test_freeze_optimizer_and_checkpoint_restore(tmp_path,device):
    cfg=config(mode='atom',scope='onsite',device=device)
    cfg['model_options']['shift_head']['freeze_backbone']=True
    model=build_model(**cfg)
    original={k:v.clone() for k,v in model.named_parameters() if '.shift_head.' not in k}
    from dptb.utils.tools import get_optimizer
    opt=get_optimizer('Adam',optimizer_named_parameters(model),lr=.001)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.shift_head.parameters()}
    loss=HamilLossAbs(idp=model.idp,device=device)
    data=batch(model);out=model(copy.deepcopy(data));em,nm=model._build_expert_masks(out,0)
    out.update(expert_edge_mask=em,expert_node_mask=nm)
    loss(out,data).backward();opt.step()
    for name,p in model.named_parameters():
        if '.shift_head.' not in name:
            assert p.grad is None and not p.requires_grad;bitwise(p,original[name])
    from dptb.plugins.saver import Saver
    from types import SimpleNamespace
    saver=Saver();saver.trainer=SimpleNamespace(model=model,task="train",ep=1,iter=1,stats={})
    obj=saver._assemble_checkpoint_obj("frozen","iteration",model.model_options,cfg['common_options'],
        cfg['train_options'],model.state_dict(),[dict(optimizer_state_dict=opt.state_dict())])
    path=tmp_path/'frozen.pth';torch.save(obj,path)
    restored=build_model(checkpoint=str(path),device=device)
    for name,p in restored.named_parameters():assert p.requires_grad==('.shift_head.' in name)
    opt2=get_optimizer('Adam',optimizer_named_parameters(restored),lr=.001)
    opt2.load_state_dict(obj['optimizer_state_dict'])
    assert len(opt2.param_groups[0]['params'])==len(list(restored.shift_head.parameters()))
    bitwise(model(copy.deepcopy(data))['node_features'],restored(copy.deepcopy(data))['node_features'])


@pytest.mark.parametrize('scope',['both','onsite','hopping'])
def test_dense_init_from_strict_and_resume_without_source(tmp_path,scope):
    cfg=config(scope=scope);dense=build_model(**cfg)
    with torch.no_grad():
        next(dense.parameters()).add_(.125)
    path=tmp_path/'dense.pth';save_model(dense,cfg,path)
    newcfg=config(mode='atom',scope=scope)
    newcfg['model_options']['shift_head']['init_from']=str(path)
    new=build_model(**newcfg);data=batch(dense)
    for k in ('node_features','edge_features'):
        bitwise(dense(copy.deepcopy(data))[k],new(copy.deepcopy(data))[k])
    resumed=tmp_path/'resume.pth';save_model(new,newcfg,resumed)
    path.unlink();restored=build_model(checkpoint=str(resumed))
    bitwise(new(copy.deepcopy(data))['node_features'],restored(copy.deepcopy(data))['node_features'])


@pytest.mark.parametrize('corrupt',['missing','extra','shape'])
def test_dense_init_rejects_corrupt_state(tmp_path,corrupt):
    cfg=config();dense=build_model(**cfg);path=tmp_path/'dense.pth';save_model(dense,cfg,path)
    cp=torch.load(path,weights_only=False);state=cp['model_state_dict'];key=next(iter(state))
    if corrupt=='missing':state.pop(key)
    elif corrupt=='extra':state['bogus']=torch.zeros(1)
    else:state[key]=torch.zeros(991)
    torch.save(cp,path);cfg['model_options']['shift_head']={'mode':'atom','init_from':str(path)}
    with pytest.raises(ValueError,match='keys differ|shape mismatch'):build_model(**cfg)


@pytest.mark.parametrize('options',[{'mode':'bad'},{'mode':'atom','experts':4},{'hidden':0},{'layers':True},{'freeze_backbone':1}])
def test_invalid_options(options):
    with pytest.raises(ValueError):normalize_shift_options(options)


def test_signed_zero_identity_and_live_gradient():
    idp=OrbitalMapper({'H':'1s'},method='e3tb',has_soc=True,nextham_uureal_mask=True)
    original=torch.tensor([[-0.]],requires_grad=True);delta=torch.zeros_like(original,requires_grad=True)
    out=add_compact_delta(idp,original,delta);bitwise(out,original)
    out.sum().backward();assert delta.grad.item()==1 and original.grad.item()==1


def test_shell_maps_distinct_elements_and_all_angular_blocks():
    from dptb.data.interfaces.blockwise_tensor import edge_feature_slices
    idp=OrbitalMapper({'H':'1s','C':'2s1p','Au':'4s2p2d1f'},method='e3tb',has_soc=True,nextham_uureal_mask=True)
    idp.get_orbital_maps()
    head=PotentialShiftHead(idp,'4x0e+2x1o',{'mode':'shell'})
    names=['H','C','Au'];v=torch.arange(27,dtype=torch.float32).reshape(3,9)/16
    data={'edge_index':torch.tensor([[0,1,2],[1,2,0]]),
          'phys_node_overlap':torch.ones(3,729),'phys_edge_overlap':torch.ones(3,729)}
    _,delta=head.assemble(data,v)
    for row,(a,b) in enumerate(zip(names,names[1:]+names[:1])):
        # Consumer AO slices resolve element-local radial shells independently.
        for si,sj,sl in edge_feature_slices(idp,a,b):
            source_shell=next(k for k,x in idp.orbital_maps[a].items() if x==si)
            dest_shell=next(k for k,x in idp.orbital_maps[b].items() if x==sj)
            u=idp.full_basis.index(idp.basis_to_full_basis[a][source_shell])
            w=idp.full_basis.index(idp.basis_to_full_basis[b][dest_shell])
            expected=(v[row,u]+v[(row+1)%3,w])*.5
            assert torch.equal(delta[row,sl],expected.expand(sl.stop-sl.start))


def test_only_scalar_features_and_optional_element_embedding():
    idp=OrbitalMapper({'H':'1s','He':'1s'},method='e3tb',has_soc=True,nextham_uureal_mask=True)
    h=PotentialShiftHead(idp,'2x1o+3x0e+2x2e',{'mode':'atom','element_dim':4})
    with torch.no_grad():h.mlp[-1].weight.fill_(.125)
    x=torch.randn(2,19);y=x.clone();y[:,:6]+=9;y[:,9:]+=11
    types=torch.tensor([0,1]);bitwise(h.predict(x,types),h.predict(y,types))
    assert not torch.equal(h.predict(x,types),h.predict(x,types.flip(0)))


def test_off_optimizer_groups_remain_legacy():
    model=make_model()
    names=list(model.named_parameters());names[0][1].requires_grad_(False)
    actual=list(optimizer_named_parameters(model))
    assert [n for n,p in actual]==[n for n,p in names]
    assert [id(p) for n,p in actual]==[id(p) for n,p in names]


def test_distance_policy_applies_to_direct_training_expert():
    model=make_model(mode='atom',scope='hopping')
    with torch.no_grad():model.shift_head.mlp[-1].bias.fill_(.5)
    data=batch(model);out=model.experts[0](copy.deepcopy(data))
    assert torch.count_nonzero(out['node_shift_delta'])==0
    outside=out['edge_lengths'].flatten()>=2
    assert torch.count_nonzero(out['edge_shift_delta'][outside])==0
    assert torch.count_nonzero(out['edge_shift_delta'][~outside])>0


def test_5832_native_lift_matches_729_loss_and_eval():
    # Independent full-SOC mapper defines the physical uu-real part of each block.
    # The tested branch currently emits compact outputs; this covers the legacy
    # 5832-wide boundary explicitly rather than assuming that smoke exercised it.
    compact=OrbitalMapper({'Au':'4s2p2d1f'},method='e3tb',has_soc=True,nextham_uureal_mask=True)
    full=OrbitalMapper({'Au':'4s2p2d1f'},method='e3tb',has_soc=True,nextham_uureal_mask=False,full_soc_prediction=True)
    full.get_orbpair_maps()
    select=torch.zeros(5832,dtype=torch.bool)
    for sl in full.orbpair_maps.values():select[sl.start:sl.start+(sl.stop-sl.start)//8]=True
    assert select.sum()==729
    native=torch.randn(2,5832,requires_grad=True)
    delta=torch.randn(2,729,requires_grad=True)*.01
    out=add_compact_delta(compact,native,delta)
    bitwise(out[:,~select],native[:,~select])
    bitwise(out[:,select],native[:,select]+delta)
    like=torch.randn(2,729)
    projected,_=project_uureal_to_like(compact,out,like)
    bitwise(projected,out[:,select])
    bitwise(add_compact_delta(compact,native,torch.zeros_like(delta)),native)
    ids=compact({'atomic_numbers':torch.tensor([79,79]),'edge_index':torch.tensor([[0,1],[1,0]])})
    pred={**ids,'node_features':out,'edge_features':out}
    target={'node_features':like,'edge_features':like}
    fn=HamilLossAbs(idp=compact)
    value=fn(pred,target)
    expected=(projected-like)
    torch.testing.assert_close(fn.last_onsite_l1_sum,expected.abs().sum())
    torch.testing.assert_close(fn.last_hopping_mse_sum,expected.square().sum())
    value.backward();assert torch.isfinite(native.grad).all()

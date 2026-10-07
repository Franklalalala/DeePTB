"""Potential formula, exact initialization, gradient scope, resume and loss contract."""
import copy
import os

import pytest
import torch

from dptb.data.transforms import OrbitalMapper
from dptb.nn.shift_head import add_compact_delta, normalize_shift_options, optimizer_named_parameters
from dptb.nnops.layout import project_uureal_to_like
from dptb.nnops.loss import HamilLossAbs
from dptb.tests.shift_head_helpers import make_model, batch

DEVICES = [os.environ.get("R12B_TEST_DEVICE", "cpu")]


def bitwise(a,b):
    assert a.shape == b.shape and a.dtype == b.dtype
    # reshape(-1): torch>=2.8 refuses to view a 0-dim float tensor as bytes.
    assert torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8), b.detach().contiguous().reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("moe", [False,True])
@pytest.mark.parametrize("mode", ["off", "atom"])
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


@pytest.mark.parametrize("mode", ["atom"])
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


@pytest.mark.parametrize('options',[{'mode':'bad'},{'mode':'atom','experts':4},{'hidden':0},{'layers':True},{'freeze_backbone':1}])
def test_invalid_options(options):
    with pytest.raises(ValueError):normalize_shift_options(options)


def test_signed_zero_identity_and_live_gradient():
    idp=OrbitalMapper({'H':'1s'},method='e3tb',has_soc=True,nextham_uureal_mask=True)
    original=torch.tensor([[-0.]],requires_grad=True);delta=torch.zeros_like(original,requires_grad=True)
    out=add_compact_delta(idp,original,delta);bitwise(out,original)
    out.sum().backward();assert delta.grad.item()==1 and original.grad.item()==1


def test_off_optimizer_groups_remain_legacy():
    model=make_model()
    names=list(model.named_parameters());names[0][1].requires_grad_(False)
    actual=list(optimizer_named_parameters(model))
    assert [n for n,p in actual]==[n for n,p in names]
    assert [id(p) for n,p in actual]==[id(p) for n,p in names]


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

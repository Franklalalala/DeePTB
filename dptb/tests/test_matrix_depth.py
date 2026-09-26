"""Matrix recurrence: base identity, independent depth and replay contracts."""
import copy
import pytest
import torch
from dptb.data import AtomicDataDict as A
from dptb.nn.build import build_model
from dptb.tests.model_helpers import _embedding, _data
from dptb.nnops.loopscf.matrix_depth import install_matrix_depth, ReplayDepth, clone_data
from dptb.nnops.loopscf.stack import exit_distribution


def tiny(compact=True):
    options = _embedding(False, method="lem_moe_v3_edge_h0", n_layers=3,
                         num_shared_experts=0, h0_ao_cg=True, use_h0_init=True)
    options.pop("only2b"); options.pop("prior_kind"); options.pop("prior_merge_mode"); options.pop("prior_init_scope")
    model = build_model(common_options={"basis":{"H":"1s","O":"1s1p"},"dtype":"float32",
                         "device":"cpu","overlap":False,"has_soc":True,
                         "nextham_uureal_mask":compact,"full_soc_prediction":not compact},
                        model_options={"embedding":options,"prediction":{"method":"e3tb","scale_type":"no_scale"}},
                        train_options={}).eval()
    data = _data(model)
    data[A.NODE_H0_KEY] = data.pop(A.NODE_P23_KEY)
    data[A.EDGE_H0_KEY] = data.pop(A.EDGE_P2_KEY)
    return model, data


@pytest.mark.parametrize("mode",["stack","core","unshared","latent"])
def test_k1_identity_and_multiround_gradients(mode):
    torch.manual_seed(42)
    model,data = tiny(True)
    with torch.no_grad(): expected=model(clone_data(data))
    model=install_matrix_depth(model,mode,maximum=3)
    model._matrix_depth_K=1
    actual=model(clone_data(data))
    for key in (A.NODE_FEATURES_KEY,A.EDGE_FEATURES_KEY):
        torch.testing.assert_close(actual[key],expected[key],atol=1e-6,rtol=1e-6)
    model._matrix_depth_K=3
    actual=model(clone_data(data))
    loss=sum(n.square().mean()+e.square().mean() for n,e in actual['_loop_preds'])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.testing.assert_close(actual['_exit_probabilities'].sum(-1),torch.ones(1))
    assert actual['_stack_counts']==[(1,3)]
    if mode=='unshared':
        emb=model.embedding
        cores=[emb.layers[1],*emb.depth_extra_cores]
        addresses=[{p.data_ptr() for p in c.parameters()} for c in cores]
        assert all(not addresses[i]&addresses[j] for i in range(3) for j in range(i))
        before=next(cores[1].parameters()).detach().clone()
        with torch.no_grad():next(cores[0].parameters()).add_(1)
        assert torch.equal(next(cores[1].parameters()),before)
    if mode=='latent':
        assert all(not p.requires_grad for n,p in model.named_parameters() if '.depth_' not in n)


def test_random_depth_checkpoint_replay_and_exit_law():
    sampler=ReplayDepth(3,42)
    for _ in range(7):sampler.sample()
    state=copy.deepcopy(sampler.state_dict());expected=[sampler.sample() for _ in range(31)]
    other=ReplayDepth(3,0);other.load_state_dict(state)
    assert [other.sample() for _ in range(31)]==expected
    for depth in range(1,7):
        p=exit_distribution(torch.randn(13,depth)*20)
        torch.testing.assert_close(p.sum(-1),torch.ones(13))
        assert bool((p>=0).all())


@pytest.mark.parametrize("head",["onsite","hopping"])
@pytest.mark.parametrize("scale",[1.,1e-7,0.])
def test_graph_allocation_preserves_native_loss_and_excluded_gradients(head,scale):
    from dptb.nnops.loss import HamilLossAbs
    from dptb.nnops.loopscf.matrix_objective import native_graph_contributions, attach_adaptive_matrix_loss
    model,_=tiny(True); idp=model.idp; width=idp.reduced_matrix_element
    data={A.BATCH_KEY:torch.tensor([0,0,1]),A.ATOM_TYPE_KEY:torch.tensor([[0],[1],[1]]),
          A.EDGE_INDEX_KEY:torch.tensor([[0,1,2],[1,0,2]]),A.EDGE_TYPE_KEY:torch.tensor([1,2,3]),
          A.NODE_FEATURES_KEY:torch.randn(3,width,requires_grad=True),
          A.EDGE_FEATURES_KEY:torch.randn(3,width,requires_grad=True),
          'expert_node_mask':torch.full((3,),head=='onsite'),
          'expert_edge_mask':torch.full((3,),head=='hopping')}
    ref={**data,A.NODE_FEATURES_KEY:data[A.NODE_FEATURES_KEY].detach()+scale*torch.randn(3,width),
         A.EDGE_FEATURES_KEY:data[A.EDGE_FEATURES_KEY].detach()+scale*torch.randn(3,width)}
    criterion=HamilLossAbs(idp=idp,z_loss_coef=0)
    expected=criterion(data,ref)
    parts=native_graph_contributions(data,ref,idp)
    torch.testing.assert_close(parts.sum(),expected)
    before=torch.autograd.grad(expected,(data[A.NODE_FEATURES_KEY],data[A.EDGE_FEATURES_KEY]),retain_graph=True)
    data['_loop_preds']=[(data[A.NODE_FEATURES_KEY],data[A.EDGE_FEATURES_KEY])]
    data['_exit_probabilities']=torch.ones(2,1)
    attach_adaptive_matrix_loss(criterion)
    actual=criterion(data,ref)
    torch.testing.assert_close(actual,expected)
    after=torch.autograd.grad(actual,(data[A.NODE_FEATURES_KEY],data[A.EDGE_FEATURES_KEY]))
    for a,b in zip(before,after):torch.testing.assert_close(a,b)
    assert torch.count_nonzero(after[1 if head=='onsite' else 0])==0


def test_full_spinor_blocks_roundtrip_and_reverse_conjugation():
    import numpy as np
    from dptb.data.transforms import OrbitalMapper
    from dptb.data.interfaces.ham_to_feature import block_to_feature,feature_to_block
    idp=OrbitalMapper({'C':'1s1p'},method='e3tb',has_soc=True,full_soc_prediction=True)
    rng=np.random.default_rng(13)
    rand=lambda: (rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))).astype(np.complex64)
    h1=rand();h1=h1+h1.conj().T
    h2=rand();h2=h2+h2.conj().T
    hop=rand()
    blocks={'0_0_0_0_0':h1,'1_1_0_0_0':h2,'0_1_0_0_0':hop}
    data={A.ATOMIC_NUMBERS_KEY:torch.tensor([6,6]),A.EDGE_INDEX_KEY:torch.tensor([[0,1],[1,0]]),
          A.EDGE_CELL_SHIFT_KEY:torch.zeros(2,3,dtype=torch.long)}
    block_to_feature(data,idp,blocks=blocks)
    restored=feature_to_block(data,idp)
    for key,value in blocks.items():np.testing.assert_allclose(restored[key],value,atol=2e-6)
    if '1_0_0_0_0' in restored:
        np.testing.assert_allclose(restored['1_0_0_0_0'],hop.conj().T,atol=2e-6)
    # Repack both directed edges; the reverse was resolved through conjugate
    # transpose, including exchange of ud/du and the imaginary sign.
    explicit={**blocks,'1_0_0_0_0':hop.conj().T}
    other={k:v for k,v in data.items() if k not in (A.NODE_FEATURES_KEY,A.EDGE_FEATURES_KEY)}
    block_to_feature(other,idp,blocks=explicit)
    torch.testing.assert_close(data[A.EDGE_FEATURES_KEY],other[A.EDGE_FEATURES_KEY])


def test_shared_independent_initialization_and_actual_early_exit():
    from dptb.nnops.loopscf.matrix_depth import matrix_predict_until_exit
    base,data=tiny(True)
    # Reconstruct the full model: its CG ScriptFunctions are not picklable.
    other,_=tiny(True)
    other.load_state_dict(base.state_dict())
    shared=install_matrix_depth(base,'core',3)
    independent=install_matrix_depth(other,'unshared',3)
    with torch.no_grad():
        a=shared(clone_data(data));b=independent(clone_data(data))
        for x,y in zip(a['_loop_preds'],b['_loop_preds']):
            for u,v in zip(x,y):torch.testing.assert_close(u,v)
        stopped=matrix_predict_until_exit(shared,clone_data(data),3,0.5)
    assert stopped['_exit_step']==2
    assert stopped['_stack_counts']==[(1,2)]
    assert sum(stopped['_observed_exit_masses'])+stopped['_remaining_survival']==pytest.approx(1.)
    assert shared._matrix_depth_K==3 and shared._matrix_depth_exit_quantile is None

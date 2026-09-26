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
@pytest.mark.parametrize("compact",[True,False])
def test_k1_identity_and_multiround_gradients(mode,compact):
    torch.manual_seed(42)
    model,data = tiny(compact)
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

"""Chemical grouping, bounded amplitude, equivariance and checkpoint contracts."""
import copy
import os
import pytest
import torch
from e3nn import o3

from dptb.nn.chemical_readout import ChemicalCoreReadout, initialize_chemical_readouts
from dptb.nn.e3nn_fast import Linear
from dptb.nn.build import build_model
from dptb.tests.chemical_readout_helpers import chemical_config, bitwise, support
from dptb.tests.shift_head_helpers import batch, save_model

DEVICE = os.environ.get('R14B_TEST_DEVICE', 'cpu')


def small_head(zero=False):
    torch.manual_seed(21)
    linear = Linear('2x0e+3x1o+2x2e+1x0e+1x0o',
                    '2x2e+2x0e+2x1o+1x0e+1x0o', biases=True).to(device=DEVICE, dtype=torch.float64)
    if zero:
        with torch.no_grad(): linear.weight.zero_()
    h = ChemicalCoreReadout(linear, [1, 8, 26]).to(DEVICE)
    h.set_counts(torch.tensor([1, 100, 0], device=DEVICE))
    return linear, h


@pytest.mark.parametrize('scale', [0., .2, 1e6])
def test_formula_groups_cap_and_unseen(scale):
    linear, h = small_head()
    types = torch.tensor([1, 0, 2, 1, 0], device=DEVICE)
    x = torch.randn(5, linear.irreps_in.dim, device=DEVICE, dtype=torch.float64)
    with torch.no_grad():
        for b in h.blocks: b.D.normal_().mul_(scale)
    y = h(x, types, linear(x))
    # Independent per-atom oracle; gather repeated irreps in channel order.
    reference = linear(x).clone()
    for b, (ir, ins, outs) in zip(h.blocks, h.specs):
        ii = [k for i, _ in ins for k in range(linear.irreps_in.slices()[i].start, linear.irreps_in.slices()[i].stop)]
        oo = [k for i, _ in outs for k in range(linear.irreps_out.slices()[i].start, linear.irreps_out.slices()[i].stop)]
        for row, group in enumerate(types.tolist()):
            r = b.P @ b.D[group] @ b.Q.T
            delta = b.c*r/torch.sqrt(b.c*b.c+(r*r).sum())
            assert torch.linalg.vector_norm(delta) <= b.c*(1+1e-14)
            reference[row, oo] += (h.n_g[group]/(h.n_g[group]+100) * delta @ x[row, ii].reshape(-1, ir.dim)).flatten()
        # c really measures the shared channel map: probe one magnetic component per input channel.
        probe = x.new_zeros(len(ii)//ir.dim, x.shape[1])
        probe[torch.arange(len(probe), device=DEVICE), torch.tensor(ii[::ir.dim], device=DEVICE)] = 1
        w = (linear(probe)-linear(torch.zeros_like(probe)))[:, oo[::ir.dim]]
        torch.testing.assert_close(b.c, .25*w.norm(), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(y, reference, atol=1e-9, rtol=1e-7)
    bitwise(y[2], linear(x)[2])
    torch.testing.assert_close(h.rho, torch.tensor([1/101, .5, 0], device=DEVICE, dtype=torch.float64))
    if scale == 0:
        bitwise(y, linear(x))
    y.square().sum().backward()
    for b in h.blocks:
        assert torch.isfinite(b.D.grad).all()
        assert torch.count_nonzero(b.D.grad[2]) == 0
        if scale == 0: assert b.D.grad[:2].abs().sum() > 0


def test_zero_cap_signed_zero_empty_and_gradient():
    linear, h = small_head(zero=True)
    x = torch.randn(3, linear.irreps_in.dim, device=DEVICE, dtype=torch.float64, requires_grad=True)
    original = torch.full((3, linear.irreps_out.dim), -0., device=DEVICE, dtype=torch.float64)
    y = h(x, torch.tensor([0, 1, 2], device=DEVICE), original)
    bitwise(y, original)
    y.sum().backward()
    for b in h.blocks:
        for p in b.parameters(): assert torch.isfinite(p.grad).all() and p.grad.count_nonzero()==0
    assert h(x[:0], torch.zeros(0, dtype=torch.long, device=DEVICE), original[:0]).shape == original[:0].shape


@pytest.mark.parametrize('reflection', [False, True])
def test_random_rotation_and_permutation(reflection):
    linear, h = small_head()
    with torch.no_grad():
        for b in h.blocks: b.D.normal_()
    x = torch.randn(7, linear.irreps_in.dim, device=DEVICE, dtype=torch.float64)
    types = torch.tensor([0, 1, 2, 1, 0, 1, 0], device=DEVICE)
    rotation = o3.rand_matrix(dtype=torch.float64, device=DEVICE)
    if reflection: rotation = -rotation
    din = linear.irreps_in.D_from_matrix(rotation)
    dout = linear.irreps_out.D_from_matrix(rotation)
    y = h(x, types, linear(x))
    xr = x @ din.T
    torch.testing.assert_close(h(xr, types, linear(xr)), y @ dout.T, atol=2e-9, rtol=2e-9)
    p = torch.tensor([5, 0, 2, 6, 1, 4, 3], device=DEVICE)
    bitwise(h(x[p], types[p], linear(x[p])), y[p])


@pytest.mark.parametrize('moe', [False, True])
@pytest.mark.parametrize('mode', ['shared', 'chemical_core'])
def test_model_exact_initialization_and_off_state(mode, moe):
    cfg = chemical_config('shared', moe=moe)
    cfg['model_options']['embedding'].pop('node_readout')
    base = build_model(**cfg)
    changed = build_model(**chemical_config(mode, moe=moe)); support(changed)
    data = batch(base)
    a, b = base(copy.deepcopy(data)), changed(copy.deepcopy(data))
    for key in ['node_features', 'edge_features']: bitwise(a[key], b[key])
    for out in [a, b]: (out['node_features'].square().sum()+out['edge_features'].square().sum()).backward()
    source = dict(base.named_parameters())
    for name, p in changed.named_parameters():
        if '.chemical_core.' in name: continue
        if source[name].grad is None: assert p.grad is None
        else: bitwise(source[name].grad, p.grad)
    for name, value in base.state_dict().items(): bitwise(value, changed.state_dict()[name])
    if mode == 'shared': assert base.state_dict().keys() == changed.state_dict().keys()


def test_support_structure_presence_subset_rng_and_no_rescan():
    linear, h = small_head()
    fresh = ChemicalCoreReadout(linear, [1, 8, 26]).to(DEVICE)
    data = [dict(atomic_numbers=torch.tensor([1, 1, 8])),
            dict(atomic_numbers=torch.tensor([8, 8, 1, 8]), batch=torch.tensor([0, 0, 1, 1]))]
    rng = torch.get_rng_state().clone()
    initialize_chemical_readouts(fresh, data)
    bitwise(rng, torch.get_rng_state())
    assert fresh.n_g.tolist() == [2, 3, 0]
    class NoRead:
        def __len__(self): raise AssertionError('rescanned training data')
    initialize_chemical_readouts(fresh, NoRead())
    with pytest.raises(RuntimeError): fresh.set_counts([9, 9, 9])
    other = ChemicalCoreReadout(linear, [1, 8, 26]).to(DEVICE)
    from torch.utils.data import Subset
    initialize_chemical_readouts(other, Subset(data, [0]))
    assert other.n_g.tolist() == [1, 1, 0]


def test_checkpoint_restore_and_dense_loading(tmp_path):
    cfg = chemical_config('shared', scope='onsite'); dense = build_model(**cfg)
    # Dense checkpoint has moved weights: the new cap must follow THESE weights.
    with torch.no_grad(): dense.experts[0].embedding.out_node.weight.mul_(2.5)
    dense_path = tmp_path/'dense.pth'; save_model(dense, cfg, dense_path)
    newcfg = chemical_config(scope='onsite')
    newcfg['model_options']['embedding']['node_readout_init_from'] = str(dense_path)
    chem = build_model(**newcfg); support(chem)
    data = batch(dense)
    for key in ['node_features', 'edge_features']:
        bitwise(dense(copy.deepcopy(data))[key], chem(copy.deepcopy(data))[key])
    head = chem.experts[0].embedding.chemical_core
    oracle = ChemicalCoreReadout(dense.experts[0].embedding.out_node, [1,8])
    for a,b in zip(head.blocks, oracle.blocks): bitwise(a.c,b.c)
    opt = torch.optim.Adam(chem.parameters(), lr=.001)
    for _ in range(2):
        opt.zero_grad(); chem(copy.deepcopy(data))['node_features'].square().mean().backward(); opt.step()
    saved = tmp_path/'chemical.pth'
    from types import SimpleNamespace
    from dptb.plugins.saver import Saver
    saver=Saver();saver.trainer=SimpleNamespace(model=chem,task='train',ep=1,iter=2,stats={})
    cp=saver._assemble_checkpoint_obj('chemical','iteration',chem.model_options,newcfg['common_options'],
         newcfg['train_options'],chem.state_dict(),[dict(optimizer_state_dict=opt.state_dict())])
    torch.save(cp,saved)
    optimizer_state = copy.deepcopy(opt.state_dict())
    dense_path.unlink()
    restored = build_model(checkpoint=str(saved), device=DEVICE)
    class NoRead:
        def __len__(self): raise AssertionError('restore scanned dataset')
    initialize_chemical_readouts(restored, NoRead())
    for k,v in chem.state_dict().items(): bitwise(v,restored.state_dict()[k])
    bitwise(chem(copy.deepcopy(data))['node_features'], restored(copy.deepcopy(data))['node_features'])
    opt2 = torch.optim.Adam(restored.parameters(), lr=.001); opt2.load_state_dict(optimizer_state)
    for m,o in [(chem,opt),(restored,opt2)]:
        o.zero_grad(); m(copy.deepcopy(data))['node_features'].square().mean().backward(); o.step()
    for k,v in chem.state_dict().items(): bitwise(v,restored.state_dict()[k])
    # A damaged chemical checkpoint must not silently replace counts or c.
    cp = torch.load(saved, weights_only=False)
    cp['model_state_dict'].pop('experts.0.embedding.chemical_core.n_g')
    torch.save(cp,saved)
    with pytest.raises(RuntimeError): build_model(checkpoint=str(saved), device=DEVICE)


@pytest.mark.parametrize('corrupt', ['missing', 'extra', 'shape'])
def test_dense_corruption_rejected(tmp_path, corrupt):
    cfg = chemical_config('shared'); dense=build_model(**cfg)
    p=tmp_path/'bad.pth'; save_model(dense,cfg,p)
    cp=torch.load(p,weights_only=False);s=cp['model_state_dict'];key=next(iter(s))
    if corrupt=='missing':s.pop(key)
    elif corrupt=='extra':s['bogus']=torch.zeros(1)
    else:s[key]=torch.zeros(919)
    torch.save(cp,p);cfg=chemical_config();cfg['model_options']['embedding']['node_readout_init_from']=str(p)
    with pytest.raises(ValueError):build_model(**cfg)


def test_full_embedding_rotation_and_atom_permutation():
    cfg=chemical_config(); cfg['model_options']['embedding']['method']='lem_moe_v3_edge'
    model=build_model(**cfg);support(model)
    with torch.no_grad():
        for b in model.embedding.chemical_core.blocks:b.D.normal_()
    d=batch(model);a=model.embedding(copy.deepcopy(d))
    rot=o3.rand_matrix(device=DEVICE)
    dr=copy.deepcopy(d);dr['pos']=d['pos']@rot.T
    b=model.embedding(dr)
    for key in ['node_features','edge_features']:
        expected=a[key]@model.idp.orbpair_irreps.D_from_matrix(rot[[1,2,0]][:,[1,2,0]]).T
        torch.testing.assert_close(b[key],expected,atol=2e-5,rtol=2e-5)
    perm=torch.tensor([2,0,1],device=DEVICE);inv=torch.argsort(perm)
    dp=copy.deepcopy(d)
    for k,v in dp.items():
        if torch.is_tensor(v) and v.ndim and v.shape[0]==3:dp[k]=v[perm]
    dp['edge_index']=inv[d['edge_index']]
    c=model.embedding(dp)
    torch.testing.assert_close(c['node_features'],a['node_features'][perm],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(c['edge_features'],a['edge_features'],atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('mode',['shared','chemical_core','invalid'])
def test_configuration_mode_and_fixed_constants(mode):
    from dptb.utils.argcheck import model_options
    from dargs.dargs import ArgumentError
    schema=model_options();cfg=chemical_config(mode)['model_options']
    if mode=='invalid':
        with pytest.raises(ArgumentError):schema.check_value(schema.normalize_value(cfg),strict=True)
    else:
        parsed=schema.normalize_value(cfg);schema.check_value(parsed,strict=True)
        assert parsed['embedding']['node_readout']==mode
        parsed['embedding']['node_readout_rank']=8
        with pytest.raises(ArgumentError):schema.check_value(parsed,strict=True)


def test_hybrid_muon_updates_core_without_special_groups():
    from dptb.utils.tools import get_optimizer
    model=build_model(**chemical_config());support(model)
    opt=get_optimizer('HybridMuon',model.named_parameters(),lr=.001,weight_decay=.01,
                      adam_betas=(.98,.999),adam_eps=1e-20,magma_lite=False)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters()}
    data=batch(model);model(copy.deepcopy(data))['node_features'].square().mean().backward();opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert sum(b.D.abs().sum().item() for b in model.embedding.chemical_core.blocks)>0

import copy

import pytest
import torch
from e3nn import o3

from dptb.nn.build import build_model
from dptb.nn.structure_mole import raw_statistics, coefficient_metrics, svd_split_state, make_route
from dptb.nn.tensor_product_moe_v3 import MOLELinear, SO2_Linear
from dptb.tests.shift_head_helpers import config, batch, save_model
from dptb.tests.structure_mole_helpers import calibrated_model, model_config, structures, join, route


def assert_outputs(a, b, **tol):
    for k in ("node_features", "edge_features"):
        torch.testing.assert_close(a[k], b[k], **tol)


def test_weighted_statistics_against_two_pair_oracle_and_isolated_atom():
    from dptb.nn.structure_mole import StructureStats
    stats = StructureStats(2, 5, 10.).double()
    data = dict(atom_types=torch.tensor([0,1,1,0]), batch=torch.tensor([0,0,0,1]),
                pos=torch.tensor([[0.,0.,0.],[1.,0.,0.],[3.,0.,0.],[50.,0.,0.]],dtype=torch.float64),
                edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
                edge_vectors=torch.tensor([[1.,0.,0.],[-1.,0.,0.],[2.,0.,0.],[-2.,0.,0.]],dtype=torch.float64))
    gram = torch.arange(20,dtype=torch.float64).reshape(4,5) / 10.
    cutoff = torch.tensor([.2,.4,.8,.6],dtype=torch.float64)
    raw,_ = stats.raw(data,gram,cutoff)
    torch.testing.assert_close(raw[0,:5],torch.tensor([1/3,2/3,0.,.3,.7],dtype=torch.float64))
    torch.testing.assert_close(raw[1,:2],torch.tensor([1.,0.],dtype=torch.float64))
    assert torch.count_nonzero(raw[1,2:]) == 0
    a=((gram[0]+gram[1])/2)@stats.projection
    b=((gram[2]+gram[3])/2)@stats.projection
    mean=.3*a+.7*b
    std=(.3*(a-mean)**2+.7*(b-mean)**2).sqrt()
    torch.testing.assert_close(raw[0,-64:-32],mean,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(raw[0,-32:],std,atol=1e-14,rtol=1e-14)
    centers=torch.linspace(0.,10.,16,dtype=torch.float64)
    rbf=.3*torch.exp(-.5*((1-centers)/(10/15))**2)+.7*torch.exp(-.5*((2-centers)/(10/15))**2)
    torch.testing.assert_close(raw[0,5:21],rbf,atol=1e-14,rtol=1e-14)


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_block_reference_merged_all_gradients_no_bank(pair, empty, monkeypatch):
    torch.manual_seed(14)
    layer = MOLELinear(7, 10, num_experts=4, num_shared_experts=1, bias=not pair,
                       mole_expert_parameterization="shared_core", mole_expert_rank=3).double()
    n = 0 if empty else 17
    graph = torch.arange(n) % 3
    x = torch.randn((n, 2, 7) if pair else (n, 7), dtype=torch.float64, requires_grad=True)
    logits = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    alpha = logits.softmax(-1)
    ref = layer(x, route(alpha, graph, "reference"))
    def forbidden(*a, **kw):
        raise AssertionError("merged execution expanded an expert bank")
    monkeypatch.setattr(layer, "_expert_weight_bank", forbidden)
    actual = layer(x, route(alpha, graph, "merged_core"))
    torch.testing.assert_close(ref, actual, atol=1e-12, rtol=1e-12)
    cotangent = torch.randn_like(ref)
    args = [x, logits, *layer.parameters()]
    a = torch.autograd.grad((ref * cotangent).sum(), args, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad((actual * cotangent).sum(), args, allow_unused=True)
    for param, ga, gb in zip(args, a, b):
        torch.testing.assert_close(torch.zeros_like(param) if ga is None else ga,
                                   torch.zeros_like(param) if gb is None else gb, atol=2e-12, rtol=2e-12)


def test_model_reference_merged_and_shared_route_all_layers():
    model, data = calibrated_model(n_layers=3, tp_radial_emb=True, use_interpolation_out=True)
    other = build_model(**model_config(n_layers=3, tp_radial_emb=True, use_interpolation_out=True))
    other.load_state_dict(model.state_dict(), strict=True)
    other.embedding.structure_mole_options["execution"] = "reference"
    actual = model(copy.deepcopy(data))
    ref = other(copy.deepcopy(data))
    assert_outputs(actual, ref, atol=2e-5, rtol=2e-5)
    for m, out in ((model, actual), (other, ref)):
        (out["node_features"].square().sum() + out["edge_features"].square().sum()).backward()
    for (ka, a), (kb, b) in zip(model.named_parameters(), other.named_parameters()):
        assert ka == kb and (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-4, rtol=1e-4, msg=ka)
    seen = []
    handles = [layer.register_forward_pre_hook(lambda mod, args: seen.append(args[-1]))
               for layer in model.embedding.layers]
    model(copy.deepcopy(data))
    for h in handles:
        h.remove()
    assert len(seen) > 1 and all(x is seen[0] for x in seen)


def test_calibration_frozen_checkpoint_and_no_labels(tmp_path):
    model, data = calibrated_model()
    emb = model.embedding
    before = copy.deepcopy(emb.structure_stats.state_dict())
    raw = raw_statistics(emb, data)[0]
    changed = copy.deepcopy(data)
    changed["node_features"].fill_(1e8)
    changed["edge_features"].fill_(-1e8)
    changed["structure_id"] = "arbitrary non-tensor identifier, never read by the descriptor"
    changed["fitted_potential_shift"] = torch.randn(9)
    assert torch.equal(raw, raw_statistics(emb, changed)[0])
    with pytest.raises(ValueError, match="train"):
        emb.structure_stats.fit([raw], split="validation")
    with pytest.raises(ValueError, match="frozen"):
        emb.structure_stats.fit([raw], split="train")
    model(copy.deepcopy(data)); model.eval()(copy.deepcopy(data))
    assert all(torch.equal(before[k], v) for k, v in emb.structure_stats.state_dict().items())
    path = tmp_path / "model.pth"
    save_model(model, model_config(), path)
    loaded = build_model(checkpoint=str(path), common_options=model_config()["common_options"], train_options={})
    assert_outputs(model(copy.deepcopy(data)), loaded.eval()(copy.deepcopy(data)), atol=0, rtol=0)
    fresh = build_model(**model_config())
    with pytest.raises(RuntimeError, match="uncalibrated"):
        fresh(copy.deepcopy(data))


def test_batch_isolation_permutation_reverse_and_supercell():
    model, data = calibrated_model()
    model.eval()
    emb = model.embedding
    items = structures(model)
    raw, _ = raw_statistics(emb, data)
    single = torch.cat([raw_statistics(emb, d)[0] for d in items])
    torch.testing.assert_close(raw, single, atol=2e-6, rtol=2e-6)
    batched = model(copy.deepcopy(data))
    separate = [model(copy.deepcopy(d)) for d in items]
    for k in ("node_features", "edge_features"):
        torch.testing.assert_close(batched[k], torch.cat([d[k] for d in separate]), atol=3e-5, rtol=3e-5)
    # Reorder atoms and edges independently, including interleaved structures.
    p = torch.randperm(9); ip = p.argsort(); e = torch.randperm(18)
    permuted = copy.deepcopy(data)
    for k in ("pos", "atom_types", "node_h0", "node_features", "batch"):
        permuted[k] = permuted[k][p]
    for k in ("edge_h0", "edge_features", "edge_type"):
        permuted[k] = permuted[k][e]
    permuted["edge_index"] = ip[data["edge_index"][:, e]]
    torch.testing.assert_close(raw_statistics(emb, permuted)[0], raw, atol=2e-6, rtol=2e-6)
    out = model(copy.deepcopy(permuted))
    torch.testing.assert_close(out["node_features"], batched["node_features"][p], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(out["edge_features"], batched["edge_features"][e], atol=3e-5, rtol=3e-5)
    # Swap the prior descriptors of every reverse pair before symmetrization.
    swapped = copy.deepcopy(data)
    rev = torch.arange(18).reshape(-1, 2).flip(1).flatten()
    swapped["edge_h0"] = swapped["edge_h0"][rev]
    torch.testing.assert_close(raw_statistics(emb, swapped)[0], raw, atol=2e-6, rtol=2e-6)
    r, *_ = make_route(emb, copy.deepcopy(data), torch.arange(18))
    assert torch.equal(r.coefficients[r.graph_index], r.coefficients[r.graph_index[rev]])
    doubled = join([items[0], items[0]], separate=False)
    torch.testing.assert_close(raw_statistics(emb, doubled)[0], single[:1], atol=2e-6, rtol=2e-6)
    # A true periodic supercell: a two-atom chain with four directed edges.
    primitive = copy.deepcopy(items[0])
    for k in ("pos", "atom_types", "node_h0", "node_features"):
        primitive[k] = primitive[k][:2]
    primitive["pos"] = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    primitive["edge_index"] = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    primitive["edge_cell_shift"] = torch.tensor([[0.,0.,0.], [0.,0.,0.], [-1.,0.,0.], [1.,0.,0.]])
    primitive["cell"] = torch.diag(torch.tensor([2., 10., 10.])).unsqueeze(0)
    primitive["pbc"] = torch.tensor([[True, False, False]])
    for k in ("edge_h0", "edge_type", "edge_features"):
        primitive[k] = primitive[k][:2].repeat((2, 1) if primitive[k].ndim == 2 else (2,))
    supercell = join([primitive, primitive], separate=False)
    supercell["pos"] = torch.cat((primitive["pos"], primitive["pos"] + torch.tensor([2.,0.,0.])))
    supercell["edge_index"] = torch.tensor([[0,1,0,3,2,3,2,1],[1,0,3,0,3,2,1,2]])
    supercell["edge_cell_shift"] = torch.tensor([[0.,0.,0.],[0.,0.,0.],[-1.,0.,0.],[1.,0.,0.],
                                                [0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    supercell["cell"] = torch.diag(torch.tensor([4.,10.,10.])).unsqueeze(0)
    supercell["pbc"] = primitive["pbc"]
    torch.testing.assert_close(raw_statistics(emb, primitive)[0], raw_statistics(emb, supercell)[0], atol=2e-6, rtol=2e-6)
    bad = copy.deepcopy(data); bad["edge_index"][1, 0] = 5
    with pytest.raises(ValueError, match="same batch"):
        raw_statistics(emb, bad)


@pytest.mark.parametrize("coupled", [False, True], ids=["AO_product", "coupled_RME"])
@pytest.mark.parametrize("seed", [221, 887])
def test_random_rotation_statistics_and_so2(coupled, seed):
    model, data = calibrated_model()
    emb = model.embedding
    torch.manual_seed(seed)
    q = o3.rand_matrix()
    # RME convention is the repository's cyclic xyz -> yzx convention.
    yzx = torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    rotation = yzx @ q @ yzx.T
    from dptb.nn.embedding.lem_moe_v3_h0_helpers import _sorted_irrep_coordinate_index
    irreps, idx = _sorted_irrep_coordinate_index(model.idp)
    data["_h0_coupled_rme"] = torch.tensor([coupled] * 3)
    transformed = copy.deepcopy(data)
    transformed["pos"] = data["pos"] @ q.T
    change = emb.init_layer._h0_cg_change_of_basis.double()
    d = irreps.D_from_matrix(rotation).double()
    for key in ("node_h0", "edge_h0"):
        source = data[key].double() if coupled else data[key].double() @ change.T
        rotated = (source[:, idx] @ d.T)[:, idx.argsort()]
        transformed[key] = (rotated if coupled else torch.linalg.solve(change, rotated.T).T).float()
    torch.testing.assert_close(raw_statistics(emb, transformed)[0], raw_statistics(emb, data)[0], atol=4e-5, rtol=4e-5)
    # Complete backbone/output readout in coupled RME coordinates. The unchanged
    # Hamiltonian CG decoder is omitted to make the rotation law explicit.
    model.transform = False
    a, b = model(copy.deepcopy(data)), model(copy.deepcopy(transformed))
    dout = model.idp.orbpair_irreps.D_from_matrix(rotation)
    for key in ("node_features", "edge_features"):
        torch.testing.assert_close(b[key], a[key] @ dout.T, atol=2e-4, rtol=2e-4)
    layer = SO2_Linear("4x0e+3x1o+2x2e", "3x0e+2x1o+2x2e", num_experts=4,
                       num_shared_experts=1, mole_expert_parameterization="shared_core", mole_expert_rank=3)
    x, v = torch.randn(18, layer.irreps_in.dim), torch.randn(18, 3)
    g = route(torch.randn(3,4).softmax(-1), torch.arange(18) % 3, "merged_core")
    a = layer(x, v, g)[0]
    b = layer(x @ layer.irreps_in.D_from_matrix(rotation).T, v @ q.T, g)[0]
    torch.testing.assert_close(b, a @ layer.irreps_out.D_from_matrix(rotation).T, atol=2e-4, rtol=2e-4)


def test_coupled_gram_matches_existing_edge_router_and_ao_adapter():
    model, data = calibrated_model()
    emb = model.embedding
    from dptb.data.AtomicDataDict import with_edge_vectors
    data = with_edge_vectors(data, with_lengths=True)
    change = emb.init_layer._h0_cg_change_of_basis
    coupled = copy.deepcopy(data)
    coupled["edge_h0"] = data["edge_h0"] @ change.T
    coupled["_h0_coupled_rme"] = True
    torch.testing.assert_close(raw_statistics(emb,data)[0], raw_statistics(emb,coupled)[0])
    edges = torch.arange(18)
    gram = emb._gram_descriptor(emb._raw_prior_source(coupled,coupled["edge_type"].flatten(),edges))
    cutoff = emb.init_layer.base_init.cutoff_coefficients(data["edge_lengths"],data["edge_type"].flatten())
    old_gram_stats = emb.structure_stats.raw(coupled,gram,cutoff)[0]
    torch.testing.assert_close(raw_statistics(emb,coupled)[0],old_gram_stats)
    bad=copy.deepcopy(data); bad.pop("edge_h0")
    with pytest.raises(KeyError):
        raw_statistics(emb,bad)


def test_prior_ablation_constant_and_far_fragments():
    model, data = calibrated_model()
    emb = model.embedding
    route_a, *_ = make_route(emb, copy.deepcopy(data), torch.arange(18))
    z = emb.last_structure_z.clone()
    emb.structure_mole_options["prior_stats"] = False
    make_route(emb, copy.deepcopy(data), torch.arange(18))
    assert torch.equal(z[:, :-64], emb.last_structure_z[:, :-64])
    assert torch.count_nonzero(emb.last_structure_z[:, -64:]) == 0
    assert emb.router[0].in_features == z.shape[1]
    items = structures(model)
    emb.structure_mole_options["prior_stats"] = True
    combined = join(items[:2], separate=False)
    make_route(emb, copy.deepcopy(combined), torch.arange(12))
    assert not torch.allclose(emb.last_structure_alpha, route_a.coefficients[:1])
    moved = copy.deepcopy(combined); moved["pos"][3:] += 100.
    torch.testing.assert_close(raw_statistics(emb, combined)[0], raw_statistics(emb, moved)[0], atol=2e-5, rtol=2e-5)
    constant, _ = calibrated_model(scope="constant")
    assert constant.embedding.router is None and constant.embedding.structure_stats is None
    assert not any("router" in k for k in constant.state_dict())
    out = constant(copy.deepcopy(data))
    (out["node_features"].square().sum() + out["edge_features"].square().sum()).backward()
    assert all(p.grad is not None for k, p in constant.named_parameters() if "core_experts" in k)


@pytest.mark.parametrize("scope", ["structure", "constant"])
@pytest.mark.parametrize("interpolation", [False, True])
def test_svd_dense_step_zero_m_positive_and_symmetry_break(scope, interpolation, record_property):
    torch.manual_seed(314)
    cfg = config()
    cfg["model_options"]["embedding"].update(use_interpolation_out=interpolation, tp_radial_emb=True)
    dense = build_model(**cfg)
    model, data = calibrated_model(scope=scope, use_interpolation_out=interpolation, tp_radial_emb=True)
    svd_split_state(model, dense.state_dict())
    a = dense(copy.deepcopy(data)); b = model(copy.deepcopy(data))
    assert_outputs(a, b, atol=2e-5, rtol=2e-5)
    record_property("dense_output_max_error", max(float((a[k]-b[k]).detach().abs().max()) for k in ("node_features", "edge_features")))
    for name, layer in model.named_modules():
        if isinstance(layer, MOLELinear):
            w = layer.weight_shared[0] + layer.basis_left @ layer.core_experts[0] @ layer.basis_right.T
            torch.testing.assert_close(w, dense.state_dict()[name+".weight_experts"][0], atol=1e-7, rtol=1e-6)
            if ".m_linear." in name:
                assert layer.bias_experts is None and layer.out_features % 2 == 0
    (b["node_features"].square().sum() + b["edge_features"].square().sum()).backward()
    if scope == "structure":
        alpha = model.embedding.last_structure_alpha
        assert alpha.std(0).max() > 1e-3
        core_grad = next(p.grad for k,p in model.named_parameters() if "core_experts" in k)
        assert not torch.allclose(core_grad[0], core_grad[1])
        # Router gradient is mathematically zero until identical cores diverge.
        assert max(float(p.grad.abs().max()) for p in model.embedding.router.parameters()) < 2e-5


def test_metrics_independent_equations():
    a = torch.tensor([[.1,.2,.3,.4],[.4,.3,.2,.1],[.25,.25,.25,.25]])
    m = coefficient_metrics(a)
    torch.testing.assert_close(m["n_eff"], a.double().sum(0)**2/(a.double()**2).sum(0))
    torch.testing.assert_close(m["covariance_effective_rank"], torch.tensor(1., dtype=torch.float64), atol=1e-5, rtol=1e-5)
    assert coefficient_metrics(torch.ones(5,1))["covariance_effective_rank"] == 0
    assert coefficient_metrics(torch.ones(5,1))["n_eff"] == 5


def test_svd_direct_paired_input_gradient():
    torch.manual_seed(135)
    dense = MOLELinear(11, 14, num_experts=1, num_shared_experts=0, bias=False).double()
    target = MOLELinear(11, 14, num_experts=4, num_shared_experts=1, bias=False,
                        mole_expert_parameterization="shared_core", mole_expert_rank=4).double()
    svd_split_state(target, dense.state_dict())
    x = torch.randn(21, 2, 11, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    alpha = logits.softmax(-1)
    a = torch.nn.functional.linear(x, dense.weight_experts[0])
    b = target(x, route(alpha, torch.arange(21) % 3, "merged_core"))
    # Explicit real/imaginary pair assembly, not independent real channels.
    def assemble(v):
        real, imag = v.split(7, -1)
        return torch.stack((real[:,0] - imag[:,1], real[:,1] + imag[:,0]), 1)
    a, b = assemble(a), assemble(b)
    torch.testing.assert_close(a,b,atol=2e-14,rtol=2e-14)
    ga = torch.autograd.grad(a.square().sum(), x)[0]
    gb, gr = torch.autograd.grad(b.square().sum(), (x,logits))
    torch.testing.assert_close(ga,gb,atol=3e-14,rtol=3e-14)
    assert gr.abs().max() < 2e-14


def test_fresh_checkpoint_entry_stats_and_resume_without_files(tmp_path, record_property):
    from dptb.nn.structure_mole import fit_structure_stats
    dense_cfg = config(scope="hopping")
    dense = build_model(**dense_cfg)
    dense_path = tmp_path / "dense.pth"
    save_model(dense, dense_cfg, dense_path)
    cfg = model_config()
    cfg["train_options"] = dense_cfg["train_options"]
    target = build_model(**cfg)
    stats = fit_structure_stats(target, structures(target), split="train")
    stats_path = tmp_path / "stats.pt"
    torch.save(stats, stats_path)
    cfg["model_options"]["embedding"]["structure_mole"].update(init_from=str(dense_path), stats_path=str(stats_path))
    model = build_model(**cfg)
    data = join(structures(model))
    assert_outputs(dense(copy.deepcopy(data)), model(copy.deepcopy(data)), atol=2e-5, rtol=2e-5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    def step(m, o):
        o.zero_grad(set_to_none=True)
        out = m(copy.deepcopy(data))
        (out["node_features"].square().mean()+out["edge_features"].square().mean()).backward()
        o.step()
        return out
    step(model, opt)
    path = tmp_path / "upgraded.pth"
    save_model(model,cfg,path)
    dense_path.unlink(); stats_path.unlink()
    loaded = build_model(checkpoint=str(path), common_options=cfg["common_options"], train_options=cfg["train_options"])
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, loaded.state_dict()[k], atol=0, rtol=0)
    new_opt = torch.optim.Adam(loaded.parameters(),lr=1e-4)
    new_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert_outputs(step(model,opt),step(loaded,new_opt),atol=0,rtol=0)
    record_property("resumed_step_state_max_error", max(float((v.double()-loaded.state_dict()[k].double()).abs().max())
                    for k,v in model.state_dict().items() if v.numel()))
    for k, v in model.state_dict().items():
        # FP32 backward accumulation can differ after reconstruction of the
        # autograd graph; buffers and the pre-update checkpoint above are exact.
        torch.testing.assert_close(v,loaded.state_dict()[k],atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize("kw", [dict(top_k=2), dict(edge_router_route_drop_p=.2),
                              dict(edge_router_prior_activate=True), dict(so2_expert_mixing_mode="post_activation")])
def test_reject_incompatible_contract(kw):
    with pytest.raises(ValueError):
        build_model(**model_config(**kw))

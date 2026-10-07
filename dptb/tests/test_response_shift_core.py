"""Dependency-light CPU tests. Load the shipped cores without importing dptb.

These are real torch solvers/networks, not mocked e3nn integration tests.
Full v9 integration tests are in test_response_shift_integration.py.
"""
from pathlib import Path
import importlib.util

import pytest
import torch


def load_core(name):
    path = Path(__file__).resolve().parents[1] / "nn" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load_core("charge_response")
S = load_core("shift_overlap")


@pytest.fixture(autouse=True)
def deterministic_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


def rotation():
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def structure():
    return torch.tensor([[.1, .2, .3], [1.4, .1, .2], [2.2, 1.2, .6]], dtype=torch.float64)


def test_charge_constraint_kkt_gauge_and_gradients():
    gamma = C.isolated_gamma(structure())
    chi = torch.tensor([1., -2., .7], dtype=torch.float64, requires_grad=True)
    j = torch.tensor([5., 8., 12.], dtype=torch.float64, requires_grad=True)
    q = C.constrained_charge(chi, j, gamma, .25)
    torch.testing.assert_close(q.sum(), torch.tensor(.25, dtype=q.dtype), atol=1e-14, rtol=0)
    mu = chi + j * q + gamma @ q
    torch.testing.assert_close(mu, mu.mean().expand_as(mu), atol=1e-13, rtol=0)
    torch.testing.assert_close(q, C.constrained_charge(chi + 100., j, gamma, .25), atol=5e-15, rtol=0)
    assert torch.autograd.gradcheck(lambda x, y: C.constrained_charge(x, y, gamma), (chi, j))


@pytest.mark.parametrize("periodic", [False, True])
def test_kernel_rotation_translation_permutation_psd(periodic):
    p = structure()
    cell = torch.tensor([[6., .3, .1], [0., 7., .5], [.2, 0., 8.]], dtype=p.dtype)
    q = rotation()
    kernel = lambda x, a: C.periodic_gamma(x, a) if periodic else C.isolated_gamma(x)
    g = kernel(p, cell)
    g2 = kernel(p @ q + torch.tensor([5., -2., 1.]), cell @ q)
    torch.testing.assert_close(g, g2, atol=2e-13, rtol=2e-13)
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(kernel(p[order], cell), g[order][:, order], atol=2e-13, rtol=2e-13)
    assert float(torch.linalg.eigvalsh(g).min()) > -1e-12


def test_periodic_images_cutoff_and_supercell_replication():
    p = structure()
    cell = torch.diag(torch.tensor([6., 7., 8.], dtype=p.dtype))
    gamma = C.periodic_gamma(p, cell, sigma=1.2, g_cut=4.)
    wrapped = p.clone(); wrapped[1] += cell[0]; wrapped[2] -= cell[2]
    torch.testing.assert_close(C.periodic_gamma(wrapped, cell), gamma, atol=3e-13, rtol=0)
    converged = C.periodic_gamma(p, cell, sigma=1.2, g_cut=6.)
    torch.testing.assert_close(gamma, converged, atol=3e-10, rtol=0)
    chi, j = torch.tensor([1., -2., .7], dtype=p.dtype), torch.full((3,), 10., dtype=p.dtype)
    q = C.constrained_charge(chi, j, gamma)
    bigp = torch.cat((p, p + cell[0]))
    bigcell = cell.clone(); bigcell[0] *= 2
    gbig = C.periodic_gamma(bigp, bigcell)
    qbig = C.constrained_charge(chi.repeat(2), j.repeat(2), gbig)
    torch.testing.assert_close(qbig, q.repeat(2), atol=2e-13, rtol=2e-13)
    v, vbig = gamma @ q, gbig @ qbig
    torch.testing.assert_close(vbig - vbig.mean(), (v - v.mean()).repeat(2), atol=3e-13, rtol=0)


def test_far_environment_changes_local_charge():
    p = torch.tensor([[0., 0., 0.], [1., 0., 0.], [12., 0., 0.]], dtype=torch.float64)
    gamma = C.isolated_gamma(p)
    j = torch.full((3,), 10., dtype=p.dtype)
    a = C.constrained_charge(torch.tensor([0., 1., 2.], dtype=p.dtype), j, gamma)
    b = C.constrained_charge(torch.tensor([0., 1., -2.], dtype=p.dtype), j, gamma)
    # First two local features/geometries unchanged, remote electronegativity changed.
    assert abs(float(a[0] - b[0])) > .05


@pytest.mark.parametrize("kind", ["context", "qeq"])
def test_network_batch_independence_and_replication(kind):
    net = C.ResponseNetwork(3, 2, kind=kind, local_only=kind == "context", hidden=12, dtype=torch.float64)
    with torch.no_grad():
        net.readout.weight.normal_(0, .2)
    x = torch.randn(6, 3, dtype=torch.float64)
    types = torch.tensor([0, 1, 0, 1, 0, 1])
    batch = torch.tensor([0, 1, 0, 1, 0, 1])  # deliberately interleaved
    pos = torch.cat((structure(), structure() + 15))
    pbc = torch.zeros((2, 3), dtype=torch.bool)
    v, extra = net(x, types, batch, pos=pos, pbc=pbc)
    for g in (0, 1):
        rows = torch.where(batch == g)[0]
        vg, _ = net(x[rows], types[rows], pos=pos[rows])
        torch.testing.assert_close(v[rows], vg, atol=1e-14, rtol=1e-12)
    if kind == "context":
        vr, _ = net(x.repeat(2, 1), types.repeat(2))
        vs, _ = net(x, types)
        torch.testing.assert_close(vr, vs.repeat(2, 1), atol=1e-14, rtol=1e-12)
    else:
        assert abs(float(extra["q"][batch == 0].detach().sum())) < 1e-14
        assert abs(float(extra["q"][batch == 1].detach().sum())) < 1e-14


def test_reject_partial_pbc_invalid_hardness_missing_pbc_and_budget():
    net = C.ResponseNetwork(3, 2, kind="qeq", dtype=torch.float64)
    x, types = torch.randn(3, 3, dtype=torch.float64), torch.tensor([0, 1, 0])
    with pytest.raises(ValueError, match="partial PBC"):
        net(x, types, pos=structure(), pbc=torch.tensor([True, True, False]), cell=torch.eye(3))
    with pytest.raises(ValueError, match="explicit pbc"):
        net(x, types, pos=structure(), cell=torch.eye(3))
    with pytest.raises(ValueError, match="hardness"):
        C.constrained_charge(torch.ones(3), -torch.ones(3), torch.eye(3))
    with pytest.raises(ValueError, match="budget"):
        C.periodic_gamma(structure(), torch.eye(3, dtype=torch.float64) * 10, max_modes=1)


@pytest.mark.parametrize("kind", ["qeq"])
def test_synthetic_trainability_and_held_out_geometry(kind):
    # Physics-generated target: local descriptor chi_i, structure-specific gamma.
    # No teacher network weights or label-derived features are fed to the learner.
    ngraph, n = 18, 3
    p = structure().repeat(ngraph, 1)
    scale = torch.linspace(.8, 1.8, ngraph, dtype=torch.float64)
    p = p * scale.repeat_interleave(n).unsqueeze(1)
    types = torch.tensor([0, 1, 0]).repeat(ngraph)
    x = torch.randn(ngraph * n, 3, dtype=torch.float64)
    batch = torch.arange(ngraph).repeat_interleave(n)
    x.zero_(); x[:, 0] = torch.where(types == 0, 1., -1.)
    target_list = []
    for g in range(ngraph):
        rows = slice(g*n, (g+1)*n)
        gamma = C.isolated_gamma(p[rows])
        chi = 1.1 * x[rows, 0]
        q = C.constrained_charge(chi, torch.full((n,), 22.5, dtype=x.dtype), gamma)
        target_list.append((gamma @ q).unsqueeze(1))
    target = torch.cat(target_list)
    w = torch.ones((len(x), 1), dtype=x.dtype)
    target = C.weighted_center(target, w, batch)
    net = C.ResponseNetwork(3, 2, kind=kind, hidden=24, dtype=torch.float64)
    for par in net.hardness.parameters():
        par.requires_grad_(False)
    train = batch < 14
    opt = torch.optim.Adam([par for par in net.parameters() if par.requires_grad], lr=.01)
    first_grad = None
    for step in range(180):
        opt.zero_grad()
        v, _ = net(x[train], types[train], batch[train], pos=p[train],
                   pbc=torch.zeros((14,3), dtype=torch.bool))
        v = C.weighted_center(v, w[train], batch[train])
        loss = (v - target[train]).square().mean()
        loss.backward()
        if step == 0:
            first_grad = net.readout.weight.grad.norm().item()
        opt.step()
    assert first_grad > 1e-8  # zero last layer is not a dead branch
    with torch.no_grad():
        predicted, _ = net(x, types, batch, pos=p, pbc=torch.zeros((ngraph,3), dtype=torch.bool))
        predicted = C.weighted_center(predicted, w, batch)
        train_rms = (predicted[train] - target[train]).square().mean().sqrt()
        dev_rms = (predicted[~train] - target[~train]).square().mean().sqrt()
    assert float(train_rms) < 2e-4
    assert float(dev_rms) < 2e-4


def packet_fixture():
    p = structure().float()
    edge = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]])
    shift = torch.tensor([[0,0,0],[0,0,0],[1,0,0],[-1,0,0]], dtype=torch.float32)
    data = dict(pos=p, atomic_numbers=torch.tensor([1,8,1]), cell=torch.eye(3)*5,
                pbc=torch.tensor([True,True,True]), edge_index=edge, edge_cell_shift=shift)
    packet = {k: v.clone() for k,v in data.items()}
    packet.update(basis_token="unit-basis", layout="compact_uureal_physical_ao",
                  phys_node_overlap=torch.randn(3,4), phys_edge_overlap=torch.randn(4,4))
    return data, packet


def test_overlap_packet_reorders_periodic_edges_and_live_gradients():
    data, packet = packet_fixture()
    expected = packet["phys_edge_overlap"].clone()
    order = torch.tensor([2,0,3,1])
    for key in ("edge_cell_shift", "phys_edge_overlap"):
        packet[key] = packet[key][order]
    packet["edge_index"] = packet["edge_index"][:, order]
    packet["phys_edge_overlap"].requires_grad_()
    S.attach_overlap_packet(data, packet, width=4, basis_token="unit-basis")
    assert torch.equal(data["phys_edge_overlap"], expected)
    data["phys_edge_overlap"].sum().backward()
    assert torch.equal(packet["phys_edge_overlap"].grad, torch.ones(4,4))


@pytest.mark.parametrize("fault", ["geometry", "basis", "image", "duplicate", "nan", "layout", "missing", "batch"])
def test_overlap_packet_fail_closed(fault):
    data, packet = packet_fixture()
    if fault == "geometry": packet["pos"][0,0] += .01
    elif fault == "basis": packet["basis_token"] = "wrong"
    elif fault == "image": packet["edge_cell_shift"][2,0] = 2
    elif fault == "duplicate":
        packet["edge_index"][:,1] = packet["edge_index"][:,0]
        packet["edge_cell_shift"][1] = packet["edge_cell_shift"][0]
    elif fault == "nan": packet["phys_edge_overlap"][0,0] = float('nan')
    elif fault == "layout": packet["layout"] = "latents"
    elif fault == "missing": packet.pop("cell")
    elif fault == "batch": data["batch"] = torch.zeros(3,dtype=torch.long)
    with pytest.raises(ValueError):
        S.attach_overlap_packet(data, packet, width=4, basis_token="unit-basis")


def test_standard_overlap_snapshot_survives_latent_replacement_and_missing_fails():
    data, packet = packet_fixture()
    with pytest.raises(ValueError, match="physical S"):
        S.prepare_shift_overlap(data, 4)
    data["node_overlap"] = packet["phys_node_overlap"]
    data["edge_overlap"] = packet["phys_edge_overlap"]
    S.prepare_shift_overlap(data, 4, standard=True)
    data["edge_overlap"] = torch.randn(4, 64)
    assert torch.equal(data["phys_edge_overlap"], packet["phys_edge_overlap"])
    S.prepare_shift_overlap(data, 4)


def test_qeq_zero_initialization_first_readout_then_hardness_gradient():
    net = C.ResponseNetwork(3, 2, kind="qeq", hidden=12, dtype=torch.float64)
    x = torch.randn(3,3,dtype=torch.float64)
    types = torch.tensor([0,1,0])
    opt = torch.optim.Adam(net.parameters(),lr=.02)
    target = torch.tensor([[.2],[-.3],[.1]],dtype=torch.float64)
    v,_ = net(x,types,pos=structure())
    (v-target).square().mean().backward()
    assert net.readout.weight.grad.norm() > 1e-8
    assert torch.count_nonzero(net.hardness.weight.grad) == 0
    opt.step(); opt.zero_grad()
    v,_ = net(x,types,pos=structure())
    (v-target).square().mean().backward()
    assert net.hardness.weight.grad.norm() > 1e-10

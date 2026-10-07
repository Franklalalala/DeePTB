"""response.qeq_local: charge equilibration as a correction to a local readout (frozen-probe form).

The local readout carries element-dependent relative levels, while the charge-equilibration
term adds the nonlocal correction v = local readout + kappa * Gamma q.
Pure-torch tests of dptb.nn.charge_response (no e3nn needed) plus option validation with e3nn.
"""
import pytest
import torch

from dptb.nn.charge_response import ResponseNetwork, graph_mean, isolated_gamma


def _net(seed=0, **kw):
    torch.manual_seed(seed)
    return ResponseNetwork(6, 4, kind="qeq", dtype=torch.float64, **kw)


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])
    types = torch.tensor([0, 1, 1, 2, 0, 0, 3, 1, 1])
    x = torch.randn(len(batch), 6, generator=g, dtype=torch.float64)
    pos = torch.randn(len(batch), 3, generator=g, dtype=torch.float64) * 3
    return x, types, batch, pos


def _randomize(net, seed=5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.3)
    return net


def test_default_qeq_is_unchanged_by_the_switch():
    a, b = _net(), _net(qeq_local=False)
    assert list(a.state_dict()) == list(b.state_dict())
    assert not any(k.startswith(("chi_head", "chi_z", "j_z", "kappa")) for k in a.state_dict())
    for va, vb in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(va, vb)
    _randomize(a); _randomize(b)
    x, types, batch, pos = _batch()
    va, ea = a(x, types, batch, pos=pos)
    vb, eb = b(x, types, batch, pos=pos)
    assert torch.equal(va, vb) and torch.equal(ea["q"], eb["q"]) and torch.equal(ea["hardness"], eb["hardness"])


def test_qeq_local_zero_initialisation_and_first_gradients():
    net = _net(qeq_local=True, context="none")
    x, types, batch, pos = _batch()
    v, aux = net(x, types, batch, pos=pos)
    assert torch.equal(v, torch.zeros_like(v)) and torch.equal(aux["q"], torch.zeros_like(aux["q"]))
    # J = hardness_min + softplus(5) + softplus(0) at init, as in the probe.
    j0 = 5.0 + torch.nn.functional.softplus(torch.tensor(5.0, dtype=torch.float64)) + torch.log(torch.tensor(2.0, dtype=torch.float64))
    assert torch.allclose(aux["hardness"], j0.expand_as(aux["hardness"]))
    target = torch.randn(v.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    ((v - target).square().sum()).backward()
    assert net.readout.weight.grad.abs().sum() > 0          # local term learns from step 1
    assert net.chi_z.grad.abs().sum() > 0                    # element electronegativity differences
    assert net.chi_head.weight.grad.abs().sum() > 0
    # q = 0 at init: kappa and the hardness parameters get gradient only once charges form.
    assert float(net.kappa.grad) == 0.0 and net.j_z.grad.abs().sum() == 0


def test_qeq_local_equals_probe_formula():
    net = _randomize(_net(qeq_local=True, context="none"))
    x, types, batch, pos = _batch()
    v, aux = net(x, types, batch, pos=pos)
    with torch.no_grad():
        kappa = net.kappa.detach().clone()
        net.kappa.zero_()
        local, _ = net(x, types, batch, pos=pos)
        net.kappa.copy_(kappa)
    manual = torch.empty_like(v)
    for g in range(3):
        rows = torch.where(batch == g)[0]
        G = isolated_gamma(pos[rows], net.sigma)
        chi = aux["chi"][rows].reshape(-1)
        j = aux["hardness"][rows].reshape(-1)
        # lane-B probes.py F_Q/F_SQ solve, verbatim
        rhs = torch.stack((chi, torch.ones_like(chi)), dim=1)
        sol = torch.cholesky_solve(rhs, torch.linalg.cholesky(G + torch.diag(j)))
        q = -sol[:, 0] + sol[:, 1] * (sol[:, 0].sum() / sol[:, 1].sum())
        q = q - q.sum() / len(q)
        assert torch.allclose(q, aux["q"][rows].reshape(-1), atol=1e-12)
        manual[rows] = local[rows] + kappa * (G @ q).unsqueeze(-1)
    assert torch.allclose(v, manual, atol=1e-10)
    scaled = _randomize(_net(qeq_local=True, context="none", output_scale=0.1))
    vs, _ = scaled(x, types, batch, pos=pos)
    assert torch.allclose(vs, 0.1 * v, atol=1e-12)


def test_qeq_local_permutation_and_charge_conservation():
    net = _randomize(_net(qeq_local=True, context="none"))
    x, types, batch, pos = _batch()
    v, aux = net(x, types, batch, pos=pos)
    for g in range(3):
        assert abs(float(aux["q"][batch == g].sum())) < 1e-10
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(4))
    vp, _ = net(x[perm], types[perm], batch[perm], pos=pos[perm])
    assert torch.allclose(vp, v[perm], atol=1e-10)


def _offset_task(seed, graphs):
    """Isolated A/B clusters of varying size and geometry; target = centred per-element level offset
    (the element-constant component that dominates the diagnosed label levels, M1 R^2 = 0.86)."""
    g = torch.Generator().manual_seed(seed)
    batch, types, feats, pos, target = [], [], [], [], []
    for k in range(graphs):
        na, nb = (int(t) for t in torch.randint(1, 4, (2,), generator=g))
        n = na + nb
        p = torch.randn(n, 3, generator=g, dtype=torch.float64) * (1.5 + 0.8 * n ** (1 / 3))
        tk = torch.tensor([0] * na + [1] * nb)
        y = torch.where(tk == 0, 0.6, -0.4).double()
        y = y - y.mean()
        for i in range(n):
            batch.append(k); types.append(int(tk[i]))
            feats.append(torch.cat((torch.nn.functional.one_hot(tk[i], 2).double(),
                                    0.1 * torch.randn(4, generator=g, dtype=torch.float64))))
        pos.append(p); target.append(y)
    return (torch.stack(feats), torch.tensor(types), torch.tensor(batch),
            torch.cat(pos), torch.cat(target).unsqueeze(1))


def test_local_readout_is_what_carries_element_offsets():
    x, t, b, p, y = _offset_task(3, 48)
    xt, tt, bt, pt, yt = _offset_task(4, 24)

    def fit(qeq_local):
        torch.manual_seed(0)
        net = ResponseNetwork(6, 2, kind="qeq", qeq_local=qeq_local, hidden=32, element_dim=4,
                              dtype=torch.float64)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(300):
            v, _ = net(x, t, b, pos=p)
            v = v - graph_mean(v, b)[b]
            loss = (v - y).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v, _ = net(xt, tt, bt, pos=pt)
            v = v - graph_mean(v, bt)[bt]
            return float((v - yt).square().mean().sqrt())

    scale = float(yt.square().mean().sqrt())
    pure, local = fit(False), fit(True)
    assert local < 0.05 * scale, (local, scale)
    assert local < 0.2 * pure, (local, pure, scale)


def test_qeq_local_is_a_declared_config_key():
    """The run config goes through dargs in strict mode; an undeclared key aborts training (0928 smokes)."""
    pytest.importorskip("dargs")
    from dptb.utils.argcheck import model_options
    arg = model_options()
    data = arg.normalize_value({"shift_head": {"mode": "atom", "response": {"kind": "qeq", "qeq_local": True}}},
                               trim_pattern="_*")
    arg.check_value(data, strict=True)
    assert data["shift_head"]["response"]["qeq_local"] is True
    legacy = arg.normalize_value({"shift_head": {"mode": "atom", "response": {"kind": "qeq"}}}, trim_pattern="_*")
    arg.check_value(legacy, strict=True)
    assert legacy["shift_head"]["response"]["qeq_local"] is False


def test_qeq_local_option_validation():
    with pytest.raises(ValueError):
        ResponseNetwork(6, 4, kind="context", qeq_local=True)
    with pytest.raises(ValueError):
        ResponseNetwork(6, 4, kind="qeq", qeq_local=1)
    pytest.importorskip("e3nn")
    from dptb.nn.response_shift_head import normalize_response_options
    assert normalize_response_options({"kind": "qeq"})["qeq_local"] is False
    with pytest.raises(ValueError):
        normalize_response_options({"kind": "qeq", "qeq_local": True, "context": "sublattice"})
    for bad in ({"kind": "context", "qeq_local": True}, {"kind": "qeq", "qeq_local": "yes"}):
        with pytest.raises(ValueError):
            normalize_response_options(bad)

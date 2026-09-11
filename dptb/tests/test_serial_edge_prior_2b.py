import torch
from dptb.tests.test_lem_moe_v3_prior_2b import _build, _data, _has_grad, _params_equal
from dptb.nn.embedding.lem_moe_v3_edge import LemMoEV3EdgeH0


def build(stage):
    return _build(stage == 1, method='lem_moe_v3_edge_prior_2b', prior_kind='h0',
                  num_experts=4, top_k=2, num_shared_experts=1,
                  edge_router_prior_activate=False, edge_moe_compact_min_edges=0)


def data(model):
    d = _data(model)
    d['node_h0'] = d.pop('node_p23')
    d['edge_h0'] = d.pop('edge_p2')
    return d


def test_serial_edge_router_load_and_gradient_contract():
    s1 = build(1)
    assert isinstance(s1.embedding, LemMoEV3EdgeH0)
    out = s1(data(s1))
    (out['node_features'].square().mean() + out['edge_features'].square().mean()).backward()
    assert _has_grad(s1.embedding.two_b_out_node)
    assert not _has_grad(s1.embedding.layers[0])
    assert out['edge_moe_num_route_tokens'].item() == 2
    s2 = build(2)
    assert s1.embedding.router.net[0].weight.shape == s2.embedding.router.net[0].weight.shape
    s2.load_state_dict(s1.state_dict(), strict=True)
    assert _params_equal(s1.embedding.two_b_init, s2.embedding.h0_init.base_init)
    assert bool(s2.embedding.two_b_gnn_seeded)
    frozen = {n: p.detach().clone() for n, p in s2.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam(s2.parameters(), lr=1e-2)
    fixed_input = data(s2)
    def pairwise_output():
        with torch.no_grad():
            s2.embedding.only2b = True
            result = s2({k: v.clone() if torch.is_tensor(v) else v for k, v in fixed_input.items()})
            s2.embedding.only2b = False
            return result['node_features'].clone(), result['edge_features'].clone()
    before = pairwise_output()
    out = s2(data(s2))
    (out['node_features'].square().mean() + out['edge_features'].square().mean()).backward()
    assert _has_grad(s2.embedding.layers[0])
    assert _has_grad(s2.embedding.router)
    for module in s2.embedding._two_b_modules():
        assert all(p.grad is None and not p.requires_grad for p in module.parameters())
    optimizer.step()
    assert all(torch.equal(dict(s2.named_parameters())[n], p) for n, p in frozen.items())
    after = pairwise_output()
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_h0_is_required_and_changes_predictions():
    model = build(1)
    original = data(model)
    a = model(dict(original))['node_features'].detach()
    changed = dict(original, node_h0=original['node_h0'] + 1)
    b = model(changed)['node_features'].detach()
    assert not torch.allclose(a, b)
    missing = dict(original)
    missing.pop('node_h0')
    import pytest
    with pytest.raises((KeyError, RuntimeError, ValueError)):
        model(missing)

"""Small numerical and serial-integration checks for the retained top-1 gate."""
import pytest
import torch

from dptb.nn.top1_prior import Top1PriorRouter, Top1Route
from dptb.nn.tensor_product_moe_v3 import MOLELinear
from dptb.tests.test_lem_moe_v3_prior_2b import _build, _data


def test_router_probability_and_all_logit_gradients():
    router = Top1PriorRouter(4, 4)
    router.net = torch.nn.Identity()
    logits = torch.tensor([[.1, 1.5, -.4, .3]], requires_grad=True)
    route, monitor, _ = router(logits)
    assert route.topk_indices.tolist() == [[1]]
    probabilities = logits.detach().softmax(-1)
    gate = probabilities[:, 1:2]
    torch.testing.assert_close(route.topk_values, gate)
    gradient, = torch.autograd.grad(route.topk_values.sum(), logits)
    expected = -gate * probabilities
    expected[:, 1] += gate[:, 0]
    torch.testing.assert_close(gradient, expected)
    assert torch.all(gradient != 0)
    torch.testing.assert_close(monitor, gate.mean())
    router.eval()
    torch.testing.assert_close(router(logits)[0].topk_values, gate)
    torch.testing.assert_close(router(torch.zeros(2, 4))[0].topk_values, torch.full((2, 1), .25))
    empty, value, _ = router(torch.empty(0, 4))
    assert empty.topk_values.shape == (0, 1) and value == 0


@pytest.mark.parametrize('pair_axis', [False, True])
def test_selected_linear_output_and_gradients(pair_axis):
    torch.manual_seed(71)
    layer = MOLELinear(3, 5, num_experts=4, num_shared_experts=0, mole_linear_mode='indexed_ref')
    x = torch.randn((7, 2, 3) if pair_axis else (7, 3), requires_grad=True)
    logits = torch.randn(7, 4, requires_grad=True)
    ids = torch.tensor([[0], [3], [1], [3], [0], [2], [1]])
    gates = logits.softmax(-1).gather(1, ids)
    actual = layer(x, Top1Route(ids, gates))
    expected = torch.einsum('n...i,noi->n...o', x, layer.weight_experts[ids[:, 0]])
    shape = (7, 1, 5) if pair_axis else (7, 5)
    expected = (expected + layer.bias_experts[ids[:, 0]].reshape(shape))
    expected = expected * gates.reshape(7, *([1] * (x.ndim - 1)))
    torch.testing.assert_close(actual, expected)
    inputs = [x, logits, *layer.parameters()]
    a = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), inputs)
    for got, want in zip(a, b):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize('prior', ['h0', 'na_cf'])
def test_switch_serial_reuses_s1_and_trains_router(prior):
    options = dict(method='lem_moe_v3_edge_prior_2b', prior_kind=prior,
                   num_experts=4, top_k=1, num_shared_experts=0,
                   edge_router_prior_activate=True, edge_router_top1_mode='switch',
                   so2_fusion_mode='streamed_m_major_cueq',
                   edge_moe_compact_min_edges=0)
    s1, s2 = _build(True, **options), _build(False, **options)
    s2.load_state_dict(s1.state_dict(), strict=True)
    data = _data(s2)
    if prior == 'h0':
        data['node_h0'] = data.pop('node_p23')
        data['edge_h0'] = data.pop('edge_p2')
    frozen = {name: p.detach().clone() for name, p in s2.named_parameters() if not p.requires_grad}
    out = s2(data)
    (out['node_features'].square().mean() + out['edge_features'].square().mean()).backward()
    grad = s2.embedding.router.net[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    torch.optim.SGD(s2.parameters(), lr=.001).step()
    assert frozen and all(torch.equal(dict(s2.named_parameters())[name], value) for name, value in frozen.items())


def test_switch_rejects_shared_experts():
    with pytest.raises(ValueError, match='num_shared_experts=0'):
        _build(False, method='lem_moe_v3_edge_prior_2b', num_experts=4,
               top_k=1, num_shared_experts=1, edge_router_prior_activate=True,
               edge_router_top1_mode='switch')

"""A flow-conditioned S2 must never change the frozen physical S1 input."""
import pytest
import torch

from dptb.nnops.flow import HamiltonianCFM
from dptb.tests.test_lem_moe_v3_prior_2b import _build, _data


def clone(data):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}


@pytest.mark.parametrize('kind,keys', [('h0', ('node_h0', 'edge_h0')),
                                     ('na_cf', ('node_p23', 'edge_p2'))])
def test_strict_s1_reuse_time_dependence_and_nonoracle_sampling(kind, keys):
    kwargs = dict(method='lem_moe_v3_edge_prior_2b', prior_kind=kind,
                  num_experts=4, top_k=2, num_shared_experts=1,
                  edge_router_prior_activate=False, edge_moe_compact_min_edges=0)
    s1 = _build(True, **kwargs)
    s2 = _build(False, **kwargs, use_flow_time_embedding=True,
                flow_time_condition_edges=True, flow_time_allow_missing=False)
    s2.load_state_dict(s1.state_dict(), strict=True)
    data = _data(s1)
    if kind == 'h0':
        data['node_h0'] = data.pop('node_p23')
        data['edge_h0'] = data.pop('edge_p2')
    data.update(batch=torch.zeros(2, dtype=torch.long),
                node_features=torch.randn_like(data[keys[0]]),
                edge_features=torch.randn_like(data[keys[1]]))
    flow = HamiltonianCFM(dict(enabled=True, prior='te', te_prior_mode='typewise',
                               node_h0_key=keys[0], edge_h0_key=keys[1]), idp=s2.idp)
    state, reference, ctx = flow.prepare_batch(clone(data), clone(data), t=torch.tensor([.2]))
    for label, key in zip(('node', 'edge'), keys):
        assert torch.equal(state['serial_original_' + key], data[key])
        base, prior = getattr(ctx, label + '_base'), getattr(ctx, label + '_prior')
        target, t = getattr(ctx, label + '_target'), getattr(ctx, label + '_t')[:, None]
        torch.testing.assert_close(state[key], (1-t)*(base+prior)+t*target)
        assert torch.equal(reference[label + '_features'], data[label + '_features'])
    frozen = {name: value.detach().clone() for name, value in s2.named_parameters() if not value.requires_grad}
    captures = {'node': [], 'edge': []}
    hooks = [getattr(s2.embedding, 'two_b_out_' + key).register_forward_hook(
        lambda module, args, out, key=key: captures[key].append(out.detach().clone()))
        for key in captures]
    try:
        first = s2(clone(state))
        changed = clone(state); changed['flow_time'] = torch.tensor([.8])
        second = s2(changed)
        assert any(not torch.allclose(first[key+'_features'], second[key+'_features']) for key in captures)
        for values in captures.values():
            torch.testing.assert_close(values[0], values[1], rtol=1e-6, atol=1e-6)
        s2.embedding.only2b = True
        s2(clone(data))
        s2.embedding.only2b = False
        for values in captures.values():
            torch.testing.assert_close(values[0], values[2], rtol=1e-6, atol=1e-6)
    finally:
        for hook in hooks: hook.remove()
    loss = first['node_features'].square().mean() + first['edge_features'].square().mean()
    loss.backward()
    torch.optim.SGD(s2.parameters(), lr=.001).step()
    assert all(torch.equal(dict(s2.named_parameters())[name], old) for name, old in frozen.items())
    no_labels = clone(data)
    no_labels['node_features'].zero_(); no_labels['edge_features'].zero_()
    for steps in (1, 2):
        sampled = flow.sample(s2, no_labels, num_steps=steps)
        assert all(torch.isfinite(sampled[key+'_features']).all() for key in captures)
    with pytest.raises(KeyError, match='immutable input'):
        s2(clone(data))

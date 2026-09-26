"""Legacy edge prior routing: optional AO-product CG with unchanged default."""
import copy

import pytest
import torch
from e3nn import o3

from dptb.nn.build import build_model
from dptb.nn.embedding.lem_moe_v3_h0_helpers import (
    _build_uureal_cg_change_of_basis, _sorted_irrep_coordinate_index,
)
from dptb.tests.shift_head_helpers import config, batch


def make_model(**options):
    cfg = config(moe=True)
    cfg['model_options']['embedding'].update(num_experts=4, top_k=2, **options)
    torch.manual_seed(160)
    return build_model(**cfg).eval()


@pytest.mark.parametrize('coupled', [False, True])
@pytest.mark.parametrize('init_cg', [False, True])
def test_cg_descriptor_invariant_without_double_conversion(coupled, init_cg):
    model = make_model(edge_router_prior_cg=True, h0_ao_cg=init_cg)
    data = batch(model)
    emb = model.embedding
    c = _build_uureal_cg_change_of_basis(model.idp, dtype=torch.float64, device='cpu')
    irreps, idx = _sorted_irrep_coordinate_index(model.idp)
    if coupled:
        data['_h0_coupled_rme'] = torch.tensor([True])
        data['edge_h0'] = (data['edge_h0'].double() @ c.T).float()
    with torch.random.fork_rng():
        torch.manual_seed(221)
        d = irreps.D_from_matrix(o3.rand_matrix(dtype=torch.float64))
    x = data['edge_h0'].double()
    y = x if coupled else x @ c.T
    y = (y[:,idx] @ d.T)[:,idx.argsort()]
    rotated = dict(data, edge_h0=(y if coupled else y @ c).float())
    types = data['edge_type'].flatten()
    edges = torch.arange(len(types))
    def desc(b):
        return emb._gram_descriptor(emb._raw_prior_source(b,types,edges))
    a, b = desc(data), desc(rotated)
    assert (a-b).norm()/a.norm() < 1e-5
    if not coupled:
        emb.edge_router_prior_cg = False
        a, b = desc(data), desc(rotated)
        assert (a-b).norm()/a.norm() > .01
    emb.edge_router_prior_cg = True
    with pytest.raises(ValueError):
        desc(dict(data, _h0_coupled_rme=torch.tensor([True, False])))
    without_prior = dict(data)
    del without_prior['edge_h0']
    with pytest.raises(KeyError):
        desc(without_prior)


def test_onehot_routes_ignore_cg_and_true_preserves_state_keys():
    a = make_model(edge_router_input='onehot')
    b = make_model(edge_router_input='onehot', edge_router_prior_cg=True)
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items())
    data = batch(a)
    with torch.no_grad():
        pa, pb = a(copy.deepcopy(data)), b(copy.deepcopy(data))
    for key in ('node_features','edge_features'):
        assert torch.equal(pa[key],pb[key])

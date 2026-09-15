import copy
from types import SimpleNamespace
import pytest
import torch
from e3nn import o3
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.lem_moe_v3_h0_helpers import H0InitLayer
from dptb.nn.embedding.lem_moe_v3_prior_2b import _Prior2bMixin

def layer(legacy=False):
    mapper=OrbitalMapper({'C':'2s2p1d'},method='e3tb',has_soc=True,nextham_uureal_mask=True,full_soc_prediction=False)
    base=torch.nn.Module();base.idp=mapper;base.irreps_out=mapper.get_irreps().sort()[0].simplify()
    return H0InitLayer(base,h0_ao_cg=not legacy,dtype=torch.float64,device='cpu').double()

def test_default_cg_and_explicit_legacy_checkpoint():
    fixed=layer();legacy=layer(True)
    assert fixed.h0_ao_cg and fixed._h0_ao_cg_version==1
    old=legacy.state_dict();old.pop('h0_ao_cg_version')
    other=layer(True);other.load_state_dict(copy.deepcopy(old),strict=True)
    x=torch.randn(3,fixed.h0_dim,dtype=torch.float64)
    assert torch.equal(other._ao_product_to_sorted_irreps(x),x.index_select(1,other._h0_sort_index))
    with pytest.raises(RuntimeError,match='h0_ao_cg_version'):
        fixed.load_state_dict(copy.deepcopy(old),strict=True)
    with pytest.raises(RuntimeError,match='h0_ao_cg_version'):
        legacy.load_state_dict(fixed.state_dict(),strict=True)
    layer().load_state_dict(fixed.state_dict(),strict=True)

@pytest.mark.parametrize('branch',['two_b','gnn'])
def test_both_serial_projectors_use_cg(branch):
    h0=layer();n=2;e=3
    node=torch.randn(n,h0.h0_dim,dtype=torch.float64,requires_grad=True)
    edge=torch.randn(e,h0.h0_dim,dtype=torch.float64,requires_grad=True)
    stub=SimpleNamespace(h0_init=h0,prior_node_key='node_p23',prior_edge_key='edge_p2',dtype=torch.float64,device='cpu')
    np=h0.node_projector if branch=='gnn' else copy.deepcopy(h0.node_projector)
    ep=h0.edge_projector if branch=='gnn' else copy.deepcopy(h0.edge_projector)
    atom=torch.zeros(n,dtype=torch.long);bond=torch.zeros(e,dtype=torch.long);active=torch.arange(e)
    y,z=_Prior2bMixin._project_prior(stub,{'node_p23':node,'edge_p2':edge},atom,bond,active,n,e,np,ep)
    assert torch.equal(y,np(h0._ao_product_to_sorted_irreps(h0._mask_node_source(node,atom))))
    assert torch.equal(z,ep(h0._ao_product_to_sorted_irreps(h0._mask_edge_source(edge,bond))))
    (y.square().sum()+z.square().sum()).backward()
    assert torch.isfinite(node.grad).all() and torch.isfinite(edge.grad).all()


def test_dense_default_and_explicit_legacy(monkeypatch):
    from dptb.nn.tensor_product import SO2_Linear
    monkeypatch.delenv("DPTB_SO2_M_LINEAR_MODE",raising=False)
    kwargs=dict(irreps_in="2x0e+2x1o",irreps_out="2x0e+2x1o")
    default=SO2_Linear(**kwargs)
    assert default.so2_m_linear_mode=="indexed_sandwich_cuda_multi"
    assert SO2_Linear(**kwargs,so2_m_linear_mode="standard").so2_m_linear_mode=="standard"

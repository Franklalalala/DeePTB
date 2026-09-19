from types import SimpleNamespace
import numpy as np
import torch
import pytest
from dptb.data.transforms import OrbitalMapper
from dptb.nacf.assembly import NACFFeaturePlan


@pytest.mark.parametrize('mapping,backend,device', [('expanded','torch','cpu'), ('compact','torch','cpu'), ('compact','cuda','cuda')])
def test_scalar_edge_packing_with_soc_bank(mapping, backend, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    mapper=OrbitalMapper({'H':'1s1p','He':'1s'},has_soc=True,nextham_uureal_mask=True)
    bank=SimpleNamespace(soc=object(),p2=SimpleNamespace(species={
        'H':{'orbital_shells':[0,1]},'He':{'orbital_shells':[0]}}))
    plan=SimpleNamespace(bank=bank,spinor_input=False,symbols=('H','He'),
        width=4,positions=torch.zeros((2,3),dtype=torch.float64,device=device),edge_index=torch.tensor([[0,1],[1,0]],device=device))
    packer=NACFFeaturePlan(plan,mapper,mapping=mapping,packing_backend=backend)
    blocks=torch.zeros((2,4,4),dtype=torch.float64)
    blocks[0,:,0]=torch.tensor([1.,2.,3.,4.]);blocks[1]=blocks[0].T.clone()
    expected=np.zeros((2,mapper.reduced_matrix_element),dtype=np.float32)
    expected[:,mapper.orbpair_maps['1s-1s']]=1.
    expected[0,mapper.orbpair_maps['1p-1s']]=[-4.,2.,-3.]
    expected[1,mapper.orbpair_maps['1s-1p']]=[-4.,2.,-3.]
    blocks=blocks.to(device)
    np.testing.assert_array_equal(packer.pack_edges(blocks).cpu().numpy(),expected)
    nodes=torch.zeros((2,4,4),dtype=torch.float64,device=device)
    torch.testing.assert_close(packer.pack(nodes,blocks)[1],packer.pack_edges(blocks),rtol=0,atol=0)

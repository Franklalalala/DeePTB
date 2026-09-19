from types import SimpleNamespace
import pytest
import torch
from dptb.data.transforms import OrbitalMapper
from dptb.nacf.assembly import NACFFeaturePlan


def test_old_native_binary_reports_rebuild(monkeypatch):
    from dptb.nacf import _cuda
    monkeypatch.setattr(_cuda, 'check_device', lambda device: None)
    monkeypatch.setattr(_cuda, 'extension', lambda: SimpleNamespace())
    blocks = SimpleNamespace(requires_grad=False, is_cuda=True,
                             dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError, match='rebuild with python -m dptb.nacf.precompile'):
        _cuda.pack(blocks, None, None, None, None, torch.float32)


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('empty', [False, True])
def test_compact_repeated_types_empty_edges_and_ao_gradients(device, empty):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    e = 0 if empty else 513
    stub = SimpleNamespace(symbols=('H','H'), width=1,
        positions=torch.zeros((2,3),dtype=torch.float64,device=device),
        edge_index=torch.zeros((2,e),dtype=torch.long,device=device),
        bank=SimpleNamespace(p2=SimpleNamespace(species={'H':{'orbital_shells':[0]}})))
    backend = 'cuda' if device == 'cuda' else 'torch'
    plan = NACFFeaturePlan(stub,OrbitalMapper({'H':'1s'}),mapping='compact',packing_backend=backend,output_dtype=torch.float64)
    node = torch.tensor([1.,-2.],device=device,dtype=torch.float64).reshape(2,1,1)
    edge = torch.arange(e,device=device,dtype=torch.float32).reshape(e,1,1)
    n, v = plan.pack(node,edge)
    torch.testing.assert_close(n[:,0],node[:,0,0],rtol=0,atol=0)
    torch.testing.assert_close(v[:,0],edge[:,0,0].double(),rtol=0,atol=0)
    edge.requires_grad_()
    plan.pack_edges(edge).sum().backward()
    torch.testing.assert_close(edge.grad,torch.ones_like(edge))

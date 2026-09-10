import numpy as np
import pytest
import torch

from dptb.data.interfaces.p2_table import RadialBlockTable
from dptb.nacf.radial import TorchRadialBlockTable


def make_table(interpolation='cubic', shells=(0, 1, 2, 3, 4)):
    rng = np.random.default_rng(614)
    knots = np.array([0., .2, .55, 1.1, 1.8, 2.6, 3.])
    width = sum(2 * l + 1 for l in shells)
    values = rng.normal(size=(len(knots), width, width))
    values[:, 0, -1] = 0.
    return RadialBlockTable(knots, values, shells, shells, 3., interpolation)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('interpolation', ['linear', 'cubic'])
def test_radial_matches_independent_scipy_oracle(device, interpolation):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    table = make_table(interpolation)
    model = TorchRadialBlockTable(table, device=device)
    vectors = np.random.default_rng(19).normal(size=(31, 3))
    vectors = np.concatenate((vectors, [[0, 0, 0], [0, 0, -1], [0, 0, 1],
                                       [0, 0, 3], [0, 0, 3 - 5e-13],
                                       [1e-9, 0, -1], [0, 0, 4]]))
    expected = np.stack([table.evaluate(v) for v in vectors])
    actual = model(torch.tensor(vectors, device=device)).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=5e-11, rtol=5e-11)
    assert model(torch.empty((0, 3), device=device, dtype=torch.float64)).shape == (0, 25, 25)
    assert torch.count_nonzero(model(torch.tensor([[0., 0., 3.]], device=device, dtype=torch.float64))) == 0


def test_radial_gradients_away_from_poles_and_cutoff():
    model = TorchRadialBlockTable(make_table(shells=(0, 1, 2)))
    vectors = torch.tensor([[.4, .7, .9], [-.3, .5, -.8]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model, (vectors,), fast_mode=True, atol=1e-5)


def test_radial_buffers_roundtrip_and_float32():
    table = make_table(shells=(0, 1, 2))
    original = TorchRadialBlockTable(table)
    clone = TorchRadialBlockTable(table)
    clone.load_state_dict(original.state_dict(), strict=True)
    vectors = torch.tensor([[.4, .7, .9]], dtype=torch.float64)
    torch.testing.assert_close(original(vectors), clone(vectors), rtol=0, atol=0)
    torch.testing.assert_close(clone.float()(vectors.float()).double(), original(vectors), atol=2e-5, rtol=2e-5)
    poles = torch.tensor([[0., 0., -1.], [0., 0., 1.], [0., 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(clone(poles.float()).double(), original(poles), atol=2e-5, rtol=2e-5)


def test_radial_zero_table_and_two_knot_cubic_fallback():
    table = RadialBlockTable(np.array([0., 2.]), np.zeros((2, 1, 3)), (0,), (1,), 2.)
    model = TorchRadialBlockTable(table)
    assert model.active_columns.numel() == 0
    assert torch.count_nonzero(model(torch.tensor([[.3, .1, .9]], dtype=torch.float64))) == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_native_cuda_matches_torch_and_replays_graph(dtype):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    source = make_table(shells=(0,1,2,3,4))
    native = TorchRadialBlockTable(source,device='cuda',dtype=dtype,backend='cuda')
    reference = TorchRadialBlockTable(source,device='cuda',dtype=dtype,backend='torch')
    vectors = torch.tensor([[0,0,0],[0,0,-1],[0,0,1],[.2,.7,-.9],[0,0,3],
                            [1e-9,0,-1],[.001,0,-1],[0,0,4]],device='cuda',dtype=dtype)
    tolerance = 2e-4 if dtype == torch.float32 else 5e-11
    native._forward_torch = lambda *args: pytest.fail('native path fell back to torch')
    actual = native(vectors)
    torch.testing.assert_close(actual,reference(vectors),atol=tolerance,rtol=tolerance)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = native(vectors)
    vectors.add_(.03)
    graph.replay()
    torch.testing.assert_close(captured,reference(vectors),atol=tolerance,rtol=tolerance)
    assert native(vectors[:0]).shape == (0,25,25)


def test_auto_cuda_preserves_autograd_reference_and_zero_channels():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    auto = TorchRadialBlockTable(make_table(shells=(0,1)),device='cuda')
    v = torch.tensor([[.3,.7,.8]],device='cuda',dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(auto,(v,),fast_mode=True)
    source = RadialBlockTable(np.array([0.,2.]),np.zeros((2,1,3)),(0,),(1,),2.)
    native = TorchRadialBlockTable(source,device='cuda',backend='cuda')
    assert torch.count_nonzero(native(v.detach())) == 0

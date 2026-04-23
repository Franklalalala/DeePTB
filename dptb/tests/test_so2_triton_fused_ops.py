import pytest

torch = pytest.importorskip("torch")

import dptb.nn.so2_triton_fused_ops as ops


def _assert_close(a, b, atol=1e-10, rtol=1e-10):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.mark.parametrize('dtype', [torch.float64])
def test_pack_m0_forward_backward(dtype):
    torch.manual_seed(0)
    n, c, d, l = 5, 7, 5, 2
    x = torch.randn(n, c, d, dtype=dtype, requires_grad=True)
    rot = torch.randn(n, d, d, dtype=dtype, requires_grad=True)

    ref = torch.einsum('ncd,nd->nc', x, rot[:, :, l])
    out = ops.triton_pack_group_m0(x, rot, l, True)
    _assert_close(out, ref)

    gx_ref, gr_ref = torch.autograd.grad(ref.square().sum(), (x, rot), retain_graph=False)
    gx, gr = torch.autograd.grad(out.square().sum(), (x, rot), retain_graph=False)
    _assert_close(gx, gx_ref)
    _assert_close(gr, gr_ref)


@pytest.mark.parametrize('dtype', [torch.float64])
def test_pack_pair_forward_backward(dtype):
    torch.manual_seed(0)
    n, c, d, l, m = 5, 7, 5, 2, 1
    x = torch.randn(n, c, d, dtype=dtype, requires_grad=True)
    rot = torch.randn(n, d, d, dtype=dtype, requires_grad=True)
    cols = [l - m, l + m]

    ref = torch.einsum('ncd,ndp->npc', x, rot[:, :, cols])
    out = ops.triton_pack_group_pair(x, rot, l, m, True)
    _assert_close(out, ref)

    gx_ref, gr_ref = torch.autograd.grad(ref.square().sum(), (x, rot), retain_graph=False)
    gx, gr = torch.autograd.grad(out.square().sum(), (x, rot), retain_graph=False)
    _assert_close(gx, gx_ref)
    _assert_close(gr, gr_ref)


@pytest.mark.parametrize('dtype', [torch.float64])
def test_scatter_pair_pipeline_forward_backward(dtype):
    torch.manual_seed(0)
    n, c, d, l, m, o = 5, 7, 5, 2, 1, 9
    x = torch.randn(n, c, d, dtype=dtype, requires_grad=True)
    rot = torch.randn(n, d, d, dtype=dtype, requires_grad=True)
    w = torch.randn(2, c, o, dtype=dtype)
    cols = [l - m, l + m]

    z_ref = torch.einsum('ncd,ndp->npc', x, rot[:, :, cols])
    y_ref = torch.einsum('npc,pco->npo', z_ref, w)
    out_ref = torch.einsum('npo,ndp->nod', y_ref, rot[:, :, cols])

    z = ops.triton_pack_group_pair(x, rot, l, m, True)
    y = torch.einsum('npc,pco->npo', z, w)
    out = ops.triton_scatter_group_pair(y, rot, l, m, d, True)
    _assert_close(out, out_ref)

    gx_ref, gr_ref = torch.autograd.grad(out_ref.square().sum(), (x, rot), retain_graph=False)
    gx, gr = torch.autograd.grad(out.square().sum(), (x, rot), retain_graph=False)
    _assert_close(gx, gx_ref)
    _assert_close(gr, gr_ref)

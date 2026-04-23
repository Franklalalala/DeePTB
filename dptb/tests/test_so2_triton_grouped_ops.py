import pytest

torch = pytest.importorskip("torch")


def _make_rot_block(n: int, d: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    base = torch.randn(n, d, d, device=device, dtype=dtype)
    base = base / base.abs().amax(dim=(1, 2), keepdim=True)
    return base.detach().clone().requires_grad_(True)


@pytest.mark.parametrize("dtype", [torch.float64])
def test_grouped_pack_scatter_cpu_reference(dtype):
    pytest.importorskip("torch")
    from dptb.nn import so2_triton_grouped_ops as ops

    device = torch.device("cpu")
    input_groups = {
        1: torch.randn(4, 5, 3, device=device, dtype=dtype, requires_grad=True),
        2: torch.randn(4, 7, 5, device=device, dtype=dtype, requires_grad=True),
    }
    rot_blocks = {
        1: _make_rot_block(4, 3, dtype, device),
        2: _make_rot_block(4, 5, dtype, device),
    }

    packed = ops.grouped_pack_pair(input_groups, rot_blocks, [1, 2], m=1, rotate_in=True)
    ref = {
        1: torch.einsum("ncd,ndp->npc", input_groups[1], rot_blocks[1][:, :, [0, 2]]),
        2: torch.einsum("ncd,ndp->npc", input_groups[2], rot_blocks[2][:, :, [1, 3]]),
    }
    torch.testing.assert_close(packed[1], ref[1], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(packed[2], ref[2], atol=1e-10, rtol=1e-10)

    y_groups = {
        1: torch.randn(4, 2, 5, device=device, dtype=dtype, requires_grad=True),
        2: torch.randn(4, 2, 7, device=device, dtype=dtype, requires_grad=True),
    }
    scattered = ops.grouped_scatter_pair(y_groups, rot_blocks, [1, 2], m=1, rotate_out=True)
    ref_scatter = {
        1: torch.einsum("npc,ndp->ncd", y_groups[1], rot_blocks[1][:, :, [0, 2]]),
        2: torch.einsum("npc,ndp->ncd", y_groups[2], rot_blocks[2][:, :, [1, 3]]),
    }
    torch.testing.assert_close(scattered[1], ref_scatter[1], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(scattered[2], ref_scatter[2], atol=1e-10, rtol=1e-10)


def test_grouped_pack_scatter_cuda_fp32_if_available():
    if not torch.cuda.is_available():
        pytest.skip("grouped Triton kernels require CUDA")

    from dptb.nn import so2_triton_grouped_ops as ops

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.float32
    input_groups = {
        1: torch.randn(13, 17, 3, device=device, dtype=dtype, requires_grad=True),
        2: torch.randn(13, 23, 5, device=device, dtype=dtype, requires_grad=True),
        3: torch.randn(13, 31, 7, device=device, dtype=dtype, requires_grad=True),
    }
    rot_blocks = {
        l: torch.randn(13, 2 * l + 1, 2 * l + 1, device=device, dtype=dtype, requires_grad=True)
        for l in input_groups
    }

    packed = ops.grouped_pack_pair(input_groups, rot_blocks, [1, 2, 3], m=1, rotate_in=True)
    ref = {
        l: torch.einsum("ncd,ndp->npc", input_groups[l], rot_blocks[l][:, :, [l - 1, l + 1]])
        for l in input_groups
    }
    for l in input_groups:
        torch.testing.assert_close(packed[l], ref[l], atol=1e-5, rtol=1e-5)

    sum(y.square().mean() for y in packed.values()).backward(retain_graph=True)
    for tensor in tuple(input_groups.values()) + tuple(rot_blocks.values()):
        assert tensor.grad is not None

    y_groups = {
        1: torch.randn(13, 2, 17, device=device, dtype=dtype, requires_grad=True),
        2: torch.randn(13, 2, 23, device=device, dtype=dtype, requires_grad=True),
        3: torch.randn(13, 2, 31, device=device, dtype=dtype, requires_grad=True),
    }
    rot_blocks_2 = {
        l: rot_blocks[l].detach().clone().requires_grad_(True)
        for l in y_groups
    }
    scattered = ops.grouped_scatter_pair(y_groups, rot_blocks_2, [1, 2, 3], m=1, rotate_out=True)
    ref_scattered = {
        l: torch.einsum("npc,ndp->ncd", y_groups[l], rot_blocks_2[l][:, :, [l - 1, l + 1]])
        for l in y_groups
    }
    for l in y_groups:
        torch.testing.assert_close(scattered[l], ref_scattered[l], atol=1e-5, rtol=1e-5)

    sum(y.square().mean() for y in scattered.values()).backward()
    for tensor in tuple(y_groups.values()) + tuple(rot_blocks_2.values()):
        assert tensor.grad is not None

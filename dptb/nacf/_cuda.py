"""Precompiled native backend; one launch evaluates and rotates a table batch."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from .precompiled import load
    return load()

@lru_cache(maxsize=8)
def check_device(device):
    from .precompiled import verify
    verify(device)


def evaluate(table, vectors):
    check_device(vectors.device)
    return extension().radial(
        vectors.contiguous(), table.knots, table.coefficients,
        table.cuda_degrees, table.cuda_directions, table.cuda_inverse,
        table.cuda_scales, table.cuda_ptr, table.cuda_terms,
        table.cuda_canonical, table.support_bohr).reshape(-1, *table.shape)


def pack(blocks, rows, indices, signs, imaginary, output_dtype):
    import torch
    if blocks.requires_grad:
        raise ValueError('native packing is inference-only; use NACFFeaturePlan for the Torch gradient path')
    if not blocks.is_cuda:
        raise ValueError('CUDA packing requires CUDA AO blocks')
    supported = (torch.float32, torch.float64, torch.complex64, torch.complex128)
    if blocks.dtype not in supported or output_dtype not in supported:
        raise ValueError('CUDA packing supports float32/64 and complex64/128')
    check_device(blocks.device)
    output = torch.empty((blocks.shape[0], indices.shape[1]), dtype=output_dtype, device=blocks.device)
    extension().pack_out(blocks.contiguous(), rows, indices, signs, imaginary, output)
    return output

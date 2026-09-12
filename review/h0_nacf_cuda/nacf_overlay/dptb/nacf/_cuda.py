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

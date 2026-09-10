"""Lazy native backend; one launch evaluates and rotates a whole table batch."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    source = Path(__file__).parent / 'csrc'
    return load(name='dptb_nacf_radial',
                sources=[str(source / 'bindings.cpp'), str(source / 'radial.cu')],
                extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--fmad=false'])


def evaluate(table, vectors):
    return extension().radial(
        vectors.contiguous(), table.knots, table.coefficients,
        table.cuda_degrees, table.cuda_directions, table.cuda_inverse,
        table.cuda_scales, table.cuda_ptr, table.cuda_terms,
        table.cuda_canonical, table.support_bohr).reshape(-1, *table.shape)

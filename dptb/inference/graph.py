"""Model neighbour graph built on the device.

Same edge set as ``AtomicData.from_ase(atoms, r_max=..., pbc=True)``: directed image pairs (i, j, S) with
|x_j + S @ cell - x_i| < c_i + c_j and no (i, i, 0), where x are the float32 model positions, c is half the
per-species cutoff kept in float32 (as ``get_r_map`` stores it) and the distance is FP64. Edges are ordered by
(i, j, S); the model is invariant to the edge order up to summation order.
"""
import torch
from ase.data import atomic_numbers as Z_OF

from dptb.data import AtomicDataDict as K
from dptb.prior.common.neighbors import image_pairs
from dptb.utils.argcheck import get_cutoffs_from_model_options


def species_cutoffs(model):
    """r_max of the model as {Z: cutoff in Angstrom}; a single float applies to every species."""
    r_max, er_max, oer_max = get_cutoffs_from_model_options(model.model_options)
    if er_max is not None or oer_max is not None:
        raise ValueError('environment cutoffs are not used by the inference engine')
    if isinstance(r_max, dict):
        if any('-' in str(k) for k in r_max):
            raise ValueError('bond-wise r_max is not supported by the inference engine')
        return {Z_OF[k]: float(v) for k, v in r_max.items()}
    return float(r_max)


def build_graph(numbers, pos, cell, cutoffs):
    """AtomicDataDict of one periodic structure on the device of ``pos``.

    numbers [n] int64, pos [n, 3] and cell [3, 3] in Angstrom (FP64 device tensors).
    """
    dev = pos.device
    if isinstance(cutoffs, dict):
        table = torch.zeros(max(cutoffs) + 1, dtype=torch.float32, device=dev)
        for z, r in cutoffs.items():
            table[z] = r
        half = 0.5 * table[numbers]
        if bool((half == 0).any()):
            raise ValueError('species without a model cutoff')
    else:
        half = torch.full((len(numbers),), 0.5 * cutoffs, dtype=torch.float32, device=dev)
    dtype = torch.get_default_dtype()
    x = pos.to(dtype)
    i, j, S = image_pairs(x.double(), cell.double(), half, strict=True, limit_dtype=torch.float32)
    return {
        K.POSITIONS_KEY: x,
        K.EDGE_INDEX_KEY: torch.stack([i, j]),
        K.EDGE_CELL_SHIFT_KEY: S.to(dtype),
        K.CELL_KEY: cell.to(dtype),
        K.PBC_KEY: torch.ones(3, dtype=torch.bool, device=dev),
        K.ATOMIC_NUMBERS_KEY: numbers.to(torch.int64)[:, None],
    }

"""Small dense Hamiltonian/overlap repair utilities.

These are inference-time safety tools.  They do not impose idempotency on H;
they enforce Hermiticity, stabilize S, and support one global gauge shift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, MutableMapping, Tuple

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class GeneralizedSpectrum:
    eigvals: Tensor
    eigvecs: Tensor
    min_overlap_eig: Tensor
    condition_overlap: Tensor


def hermitian_part(x: Tensor) -> Tensor:
    return 0.5 * (x + x.mH)


def regularize_overlap(s: Tensor, eig_floor: float = 1e-8) -> Tensor:
    s = hermitian_part(s)
    evals, evecs = torch.linalg.eigh(s)
    evals = evals.clamp_min(eig_floor)
    return (evecs * evals.unsqueeze(-2)) @ evecs.mH


def generalized_eigh_safe(h: Tensor, s: Tensor, eig_floor: float = 1e-8) -> GeneralizedSpectrum:
    h = hermitian_part(h)
    s_h = hermitian_part(s)
    se, su = torch.linalg.eigh(s_h)
    min_eig = se.min()
    cond = se.max() / se.clamp_min(eig_floor).min()
    se_reg = se.clamp_min(eig_floor)
    inv_sqrt = (su * se_reg.rsqrt().unsqueeze(-2)) @ su.mH
    h_orth = hermitian_part(inv_sqrt.mH @ h @ inv_sqrt)
    eps, q = torch.linalg.eigh(h_orth)
    c = inv_sqrt @ q
    return GeneralizedSpectrum(eigvals=eps, eigvecs=c, min_overlap_eig=min_eig, condition_overlap=cond)


def global_energy_shift_hamiltonian(h: Tensor, s: Tensor, delta_e: Tensor | float) -> Tensor:
    """Apply physical global energy alignment H <- H - delta_e S."""
    return h - torch.as_tensor(delta_e, dtype=h.real.dtype, device=h.device) * s


def repair_pair_blocks(blocks: MutableMapping[Tuple[int, int, int, int, int], Tensor]) -> Dict[Tuple[int, int, int, int, int], Tensor]:
    """Hermitian pair repair for real-space blocks keyed by ``(i,j,Rx,Ry,Rz)``.

    For every pair ``H_ij(R)`` and ``H_ji(-R)``, replace them by the average
    pair satisfying ``H_ji(-R) = H_ij(R)ᴴ``. Onsite blocks are Hermitianized.
    Missing reverse pairs are created.
    """
    out: Dict[Tuple[int, int, int, int, int], Tensor] = dict(blocks)
    visited = set()
    for key, block in list(blocks.items()):
        if key in visited:
            continue
        i, j, rx, ry, rz = key
        rev = (j, i, -rx, -ry, -rz)
        if key == rev:
            out[key] = hermitian_part(block)
            visited.add(key)
            continue
        rev_block = blocks.get(rev)
        if rev_block is None:
            avg = block
        else:
            avg = 0.5 * (block + rev_block.mH)
        out[key] = avg
        out[rev] = avg.mH
        visited.add(key)
        visited.add(rev)
    return out

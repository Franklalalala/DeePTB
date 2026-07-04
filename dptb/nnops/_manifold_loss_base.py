# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared assembly layer for the manifold-alignment losses.

The occupied-projector (``grassmann.py``) and near-Fermi (``riemannian_alignment.py``)
losses share the same batched book-keeping: flatten a stack of dense ``[..., N, N]``
matrices, pick a (sub)set of k-points, and resolve a per-k ``n_occ``.  Keeping a single
copy here avoids the two implementations drifting apart in their k-selection boundary
or ``n_occ`` remapping (they previously did).  Framework-free (torch only).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import torch

Tensor = torch.Tensor


def flatten_dense_mats(x: Tensor) -> Tensor:
    """``[..., N, N] -> [K, N, N]`` (a bare ``[N, N]`` becomes ``[1, N, N]``)."""
    if x.ndim == 2:
        return x.reshape((1,) + tuple(x.shape))
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"expected dense matrices [..., n, n], got {tuple(x.shape)}")
    return x.reshape((-1,) + tuple(x.shape[-2:]))


def flatten_optional_overlap(s: Optional[Tensor], like: Tensor) -> Optional[Tensor]:
    """Flatten an optional overlap to match ``like`` ([K, N, N]); a single S is broadcast."""
    if s is None:
        return None
    if s.ndim == 2:
        return s.reshape((1,) + tuple(s.shape)).expand(like.shape)
    return s.reshape((-1,) + tuple(s.shape[-2:]))


def select_matrix_indices(
    n_mats: int,
    max_kpoints: Optional[int],
    random_kpoints: bool,
    device: torch.device,
) -> Tensor:
    """Indices of the k-points to evaluate.

    All ``n_mats`` are used unless ``0 < max_kpoints < n_mats``; then either a random
    subset (``random_kpoints``) or an evenly-spaced deterministic subset is taken.  This
    is the single canonical boundary shared by both losses (they only differ in their
    *default* ``random_kpoints``, which each loss passes in from its own config).
    """
    if max_kpoints is None or int(max_kpoints) <= 0 or n_mats <= int(max_kpoints):
        return torch.arange(n_mats, device=device, dtype=torch.long)
    k = int(max_kpoints)
    if random_kpoints:
        return torch.randperm(n_mats, device=device)[:k]
    return torch.linspace(0, n_mats - 1, steps=k, device=device).round().to(torch.long).unique()


def n_occ_for_flat_index(n_occ: Union[int, "list", "tuple", Tensor], i: int, n_items: int) -> int:
    """Resolve the occupied count for flat matrix index ``i``.

    Accepts a scalar (shared by all k), a per-item list/tuple, or a tensor.  When a
    tensor/list carries one entry *per structure* rather than *per k-point*, it is
    remapped by proportional indexing so a k-subsampled batch still lines up.
    """
    if torch.is_tensor(n_occ):
        flat = n_occ.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        if flat.numel() == 0:
            raise ValueError("empty n_occ tensor")
        if flat.numel() == 1:
            return int(flat[0].item())
        j = min(int(i), int(flat.numel()) - 1)
        if flat.numel() != n_items and n_items > 0:
            j = min(int(i * flat.numel() / n_items), int(flat.numel()) - 1)
        return int(flat[j].item())
    if isinstance(n_occ, (list, tuple)):
        if len(n_occ) == 0:
            raise ValueError("empty n_occ sequence")
        return int(n_occ[min(int(i), len(n_occ) - 1)])
    return int(n_occ)


def dict_get(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """``data[key]`` tolerant of AtomicData-like mappings that raise on missing keys."""
    try:
        return data[key]
    except Exception:
        return default


class ManifoldLossModule:
    """Mixin providing the small shared bits of the two loss ``nn.Module`` wrappers.

    Subclasses must set ``self.device`` / ``self.dtype`` (as the losses already do).
    """

    _dict_get = staticmethod(dict_get)

    def _zero_scalar(self, like: Optional[Tensor]) -> Tensor:
        if like is None:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        return like.real.new_zeros(())

    def _build_base_loss(self, base_loss_options, loss_cls, *, idp, basis, dtype, device, overlap, **kwargs):
        """Construct the optional element-wise anchor loss, or ``None``."""
        if not base_loss_options:
            return None
        opts = dict(base_loss_options)
        opts.pop("enabled", None)
        if "method" not in opts:
            return None
        return loss_cls(**opts, idp=idp, basis=basis, dtype=dtype, device=device, overlap=overlap, **kwargs)


__all__ = [
    "flatten_dense_mats",
    "flatten_optional_overlap",
    "select_matrix_indices",
    "n_occ_for_flat_index",
    "dict_get",
    "ManifoldLossModule",
]

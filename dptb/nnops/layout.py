from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def uureal_projection_mask(
    idp: Any,
    *,
    raw_width: int,
    target_width: int,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    if idp is None or int(raw_width) == int(target_width):
        return None
    raw_mask = getattr(idp, "mask_uureal", None)
    if raw_mask is None:
        return None
    if torch.is_tensor(raw_mask):
        raw_mask = raw_mask.to(device=device or raw_mask.device, dtype=torch.bool).reshape(-1)
    else:
        raw_mask = torch.as_tensor(raw_mask, device=device, dtype=torch.bool).reshape(-1)
    if raw_mask.numel() != int(raw_width):
        return None
    if int(raw_mask.sum().detach().cpu().item()) != int(target_width):
        return None
    return raw_mask


def project_uureal_to_like(
    idp: Any,
    tensor: torch.Tensor,
    like: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if (
        not torch.is_tensor(tensor)
        or not torch.is_tensor(like)
        or tensor.ndim < 2
        or like.ndim < 2
    ):
        return tensor, None
    raw_mask = uureal_projection_mask(
        idp,
        raw_width=int(tensor.shape[-1]),
        target_width=int(like.shape[-1]),
        device=tensor.device,
    )
    if raw_mask is None:
        return tensor, None
    return tensor[..., raw_mask], raw_mask


def normalize_idp_mask_layout(
    idp: Any,
    mask: torch.Tensor,
    like: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    mask = mask.to(device=like.device, dtype=torch.bool)
    if mask.ndim == 0:
        mask = mask.reshape(1, 1)
    elif mask.ndim == 1:
        mask = mask.reshape(-1, 1)
    elif mask.ndim > 2:
        mask = mask.reshape(mask.shape[0], -1)

    mask, _raw_mask = project_uureal_to_like(idp, mask, like)
    if mask.ndim >= 2 and mask.shape[0] == like.shape[0] and mask.shape[-1] == like.shape[-1]:
        return mask

    if mask.ndim >= 2 and mask.shape[-1] == 1 and mask.shape[0] in {1, like.shape[0]}:
        while mask.ndim < like.ndim:
            mask = mask.unsqueeze(-1)
        return mask.expand_as(like)

    if mask.ndim >= 2 and mask.shape[0] == like.shape[0]:
        compressed_for_raw = uureal_projection_mask(
            idp,
            raw_width=int(like.shape[-1]),
            target_width=int(mask.shape[-1]),
            device=like.device,
        )
        if compressed_for_raw is not None:
            return mask

    raise ValueError(
        f"{label} layout does not match prediction layout; "
        "check nextham_uureal_mask/mask_uureal propagation."
    )

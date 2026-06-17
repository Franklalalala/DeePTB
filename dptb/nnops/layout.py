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

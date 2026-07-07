from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def _uureal_mask_cache(idp: Any):
    raw_mask = getattr(idp, "mask_uureal", None)
    if raw_mask is None:
        return None
    source_id = id(raw_mask)
    cache = getattr(idp, "_dptb_uureal_mask_cache", None)
    if isinstance(cache, dict) and cache.get("source_id") == source_id:
        return cache

    if torch.is_tensor(raw_mask):
        mask_cpu = raw_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    else:
        mask_cpu = torch.as_tensor(raw_mask, device="cpu", dtype=torch.bool).reshape(-1)
    cache = {
        "source_id": source_id,
        "mask_cpu": mask_cpu,
        "numel": int(mask_cpu.numel()),
        "count": int(mask_cpu.sum().item()),
        "device_masks": {},
    }
    try:
        setattr(idp, "_dptb_uureal_mask_cache", cache)
    except Exception:
        pass
    return cache


def _nextham_full_soc_uureal_mask(
    idp: Any,
    *,
    raw_width: int,
    target_width: int,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    if (
        idp is None
        or not bool(getattr(idp, "nextham_uureal_mask", False))
        or not bool(getattr(idp, "has_soc", False))
        or int(target_width) <= 0
        or int(raw_width) <= int(target_width)
        or int(raw_width) % int(target_width) != 0
    ):
        return None

    factor = int(raw_width) // int(target_width)
    expected = 4 * (2 if bool(getattr(idp, "soc_complex_doubling", True)) else 1)
    if factor != expected:
        return None

    orbpair_maps = getattr(idp, "orbpair_maps", None)
    if orbpair_maps is None and callable(getattr(idp, "get_orbpair_maps", None)):
        try:
            orbpair_maps = idp.get_orbpair_maps()
        except Exception:
            return None
    if not isinstance(orbpair_maps, dict):
        return None

    slices = []
    for slc in orbpair_maps.values():
        start = int(getattr(slc, "start", 0))
        stop = int(getattr(slc, "stop", 0))
        if stop > start:
            slices.append((start, stop))
    if not slices:
        return None
    slices.sort()

    mask_cpu = torch.zeros(int(raw_width), dtype=torch.bool, device="cpu")
    raw_offset = 0
    compact_width = 0
    for _start, stop in slices:
        width = int(stop) - int(_start)
        if width <= 0:
            continue
        if raw_offset + width > int(raw_width):
            return None
        mask_cpu[raw_offset: raw_offset + width] = True
        raw_offset += width * factor
        compact_width += width

    if raw_offset != int(raw_width) or compact_width != int(target_width):
        return None
    if device is None:
        device = torch.device("cpu")
    return mask_cpu.to(device=torch.device(device))


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
        return _nextham_full_soc_uureal_mask(
            idp,
            raw_width=int(raw_width),
            target_width=int(target_width),
            device=device,
        )
    cache = _uureal_mask_cache(idp)
    if cache is None:
        return None
    if cache["numel"] != int(raw_width) or cache["count"] != int(target_width):
        return _nextham_full_soc_uureal_mask(
            idp,
            raw_width=int(raw_width),
            target_width=int(target_width),
            device=device,
        )

    if device is None:
        device = raw_mask.device if torch.is_tensor(raw_mask) else torch.device("cpu")
    device = torch.device(device)
    device_key = str(device)
    device_masks = cache.get("device_masks", {})
    mask = device_masks.get(device_key)
    if mask is None:
        mask = cache["mask_cpu"].to(device=device, dtype=torch.bool)
        device_masks[device_key] = mask
        cache["device_masks"] = device_masks
    return mask


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

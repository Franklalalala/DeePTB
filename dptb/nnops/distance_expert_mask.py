"""Shared onsite/hopping distance-expert masks.

Two independent copies of this policy used to live in MultiTrainer and
DistanceEnsembleWrapper. Keep them identical.

Default (clip_last_expert_range=False) matches the historical 2-GPU layout:
the last expert is open-ended (dist >= d_min). That is required so a hopping
expert with range [1e-6, 10] still owns edges longer than 10 A, as in the
original 2-expert job.

Set clip_last_expert_range=True when a job trains only the onsite expert
with a single range such as [0, 1e-6]. Without the clip, num_experts==1
makes that expert "last" and the mask becomes dist >= 0, i.e. all edges.
"""
from __future__ import annotations

import torch


def edge_mask_for_distance_expert(
    dist: torch.Tensor,
    d_min,
    d_max,
    *,
    is_last_expert: bool,
    clip_last_expert_range: bool = False,
) -> torch.Tensor:
    d_min = float(d_min)
    if is_last_expert and not clip_last_expert_range:
        return dist >= d_min
    return (dist >= d_min) & (dist < float(d_max))


def node_mask_for_distance_expert(num_nodes: int, d_min, *, device) -> torch.Tensor:
    mask = torch.ones(int(num_nodes), dtype=torch.bool, device=device)
    if float(d_min) > 0:
        mask.fill_(False)
    return mask


def clip_last_expert_range_from_options(train_options) -> bool:
    if not train_options:
        return False
    return bool(train_options.get("clip_last_expert_range", False))

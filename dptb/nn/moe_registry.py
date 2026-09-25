"""Process-local registry that links MoE routers with the optimizer.

Two couplings need it: the optimizer scales each routed expert's Muon update by that expert's recent load
(``HybridMuon(expert_update_scale="sqrt_load")``), and a router can make its load-balancing bias step follow the
learning-rate schedule (``MOLERouterV3(bias_schedule="follow_lr" | "freeze_decay")``).  Routers register themselves
at construction; the registry holds weak references only, so it never keeps a model alive.
"""
import logging
import weakref
from typing import List, Optional

import torch

log = logging.getLogger(__name__)

_ROUTERS: "weakref.WeakSet" = weakref.WeakSet()
_WARNED = set()


def register_router(router) -> None:
    _ROUTERS.add(router)


def routers() -> List[torch.nn.Module]:
    return list(_ROUTERS)


def publish_lr_scale(scale: float, step: Optional[int] = None) -> None:
    """Called by the optimizer after each committed step with current_lr / peak_lr and the step count."""
    for router in list(_ROUTERS):
        router.bias_lr_scale = float(scale)
        if step is not None:
            router.opt_step = int(step)


def expert_load(num_experts: int) -> Optional[torch.Tensor]:
    """EMA of the hard per-expert load of the training router(s) with ``num_experts`` experts, or None.

    Several training routers with the same expert count (rare: e.g. a model copy left in train mode) are averaged
    and reported once.
    """
    found = [r for r in list(_ROUTERS)
             if getattr(r, "training", False) and int(getattr(r, "num_experts", -1)) == int(num_experts)
             and getattr(r, "ema_load", None) is not None]
    if not found:
        return None
    if len(found) > 1 and num_experts not in _WARNED:
        _WARNED.add(num_experts)
        log.warning("moe_registry: %d training routers with %d experts; averaging their loads", len(found), num_experts)
    loads = [r.ema_load.detach().float() for r in found]
    return loads[0] if len(loads) == 1 else torch.stack(loads).mean(0)

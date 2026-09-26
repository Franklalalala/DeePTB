"""One training-time Bernoulli route decision per structure, reused by every layer."""
import logging
import math
import numbers

import torch

from dptb.data import _keys
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

log = logging.getLogger(__name__)


def validate_route_drop(p, scale):
    if isinstance(p, bool) or not isinstance(p, numbers.Real) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("edge_router_route_drop_p must be finite and in [0, 1]")
    if scale not in ("inverted", "none"):
        raise ValueError("edge_router_route_drop_scale must be inverted or none")


def sample_structure_routes(data, p):
    """Return boolean keeps for structures and ALL edges (before cutoff filtering).

    ptr retains empty structures, including trailing ones. Without ptr, batch
    ids define the structure count; an empty batch represents zero structures.
    Use the device's default torch generator, captured by training_state/Saver.
    No private seed or generator, and p=1 needs no random draw.
    """
    batch = data[_keys.BATCH_KEY].reshape(-1)
    edge_index = data[_keys.EDGE_INDEX_KEY]
    ptr = data.get(_keys.BATCH_PTR_KEY)
    n_structures = ptr.numel() - 1 if ptr is not None else (int(batch.max()) + 1 if batch.numel() else 0)
    if n_structures < 0 or (batch.numel() and (int(batch.min()) < 0 or int(batch.max()) >= n_structures)):
        raise ValueError("route dropout: batch ids are inconsistent with ptr")
    source = batch.index_select(0, edge_index[0])
    target = batch.index_select(0, edge_index[1])
    if not torch.equal(source, target):
        raise ValueError("route dropout does not support edges between different structures")
    keep = (torch.zeros(n_structures, dtype=torch.bool, device=batch.device) if p == 1.0 else
            torch.rand(n_structures, device=batch.device, dtype=torch.float32) >= p)
    return keep, keep.index_select(0, source)


def apply_structure_routes(route, active_keep, p, scale, top_k):
    """Scale routed coefficients only; a new globals object starts with clean caches."""
    factor = active_keep.to(dtype=route.coefficients.dtype).unsqueeze(-1)
    if scale == "inverted" and p < 1.0:
        factor = factor / (1.0 - p)
    coeff = route.coefficients * factor
    indices = route.topk_indices
    if indices is None and coeff.shape[0] == 0:
        # Legacy sparse empty routing has no slot metadata.
        indices = torch.empty((0, min(top_k, coeff.shape[1])), device=coeff.device, dtype=torch.long)
    if indices is None:
        raise RuntimeError("route dropout requires per-edge top-k dispatch metadata")
    values = coeff.gather(1, indices)
    return MOLEGlobals(coefficients=coeff, topk_indices=indices, topk_values=values,
                       activation_space=True, coefficients_sum_to_one=False)


def record_route_drop(data, keep, edge_keep, active_keep, *, step):
    """Report actual rates on every enabled training forward, including empty batches.

    A gradient accumulation step can contain several such forwards. Report counts
    too, so callers can pool these without averaging unequal batch fractions.
    Eval does not overwrite the last training snapshot or emit training logs.
    """
    stats = {}
    for name, mask in (("structure", keep), ("edge", edge_keep), ("active_edge", active_keep)):
        dropped = (~mask).sum().detach()
        stats[name + "_count"] = dropped.new_tensor(mask.numel())
        stats[name + "_dropped"] = dropped
        stats[name + "_fraction"] = dropped.float() / max(mask.numel(), 1)
    data.update({"edge_router_route_drop_" + k: v for k, v in stats.items()})
    if log.isEnabledFor(logging.INFO):
        log.info("route_drop step=%s structure_fraction=%.6f edge_fraction=%.6f active_edge_fraction=%.6f",
                 step, stats["structure_fraction"].item(), stats["edge_fraction"].item(),
                 stats["active_edge_fraction"].item())
    return stats

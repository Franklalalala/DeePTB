"""Optional SO2CUDA adapter; model parameters and reference math stay in DeePTB."""
import collections
import logging
import importlib
import os
import torch
import torch.nn.functional as F


log = logging.getLogger(__name__)
_logged_fallbacks = set()


def reference_fallback(reason):
    """Explain each unsupported acceleration condition once per process."""
    if reason not in _logged_fallbacks:
        _logged_fallbacks.add(reason)
        log.warning("SO2CUDA: using the PyTorch reference (%s).", reason)
    STATS.declines[reason] += 1
    return None


def backend():
    mode = os.environ.get("SO2_CUDA_BACKEND", "auto").lower()
    if mode in ("off", "none", "torch"):
        return reference_fallback("SO2_CUDA_BACKEND=" + mode)
    if mode != "auto":
        raise ValueError("SO2_CUDA_BACKEND must be auto or off")
    try:
        deeptb = importlib.import_module("so2_cuda_ops.deeptb")
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] != "so2_cuda_ops":
            raise
        return reference_fallback("optional so2_cuda_ops package is not installed")
    return deeptb


def _cuda_fp32(*tensors):
    return (all(t.is_cuda and t.dtype == torch.float32 and t.device == tensors[0].device for t in tensors)
            and not torch.is_autocast_enabled()
            and not getattr(torch._C, "_are_functorch_transforms_active", lambda: False)())


def grouped_gemm(x, ptr, weight, *, fast_tf32=None):
    """Apply each [out,in] weight to the rows delimited by its segment pointer."""
    if x.ndim != 2 or weight.ndim != 3 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("grouped_gemm expects x[rows,in] and weight[groups,out,in]")
    if ptr.ndim != 1 or ptr.numel() != weight.shape[0] + 1:
        raise ValueError("grouped_gemm needs one segment boundary per group plus the end")
    ops = backend()
    if ops is not None and _cuda_fp32(x, weight):
        return ops.grouped_gemm(x, ptr, weight, fast_tf32=fast_tf32)
    if ops is not None:
        reference_fallback("grouped GEMM requires CUDA FP32 without autocast or torch.func")
    boundaries = ptr.tolist()
    sizes = [b - a for a, b in zip(boundaries[:-1], boundaries[1:])]
    if boundaries[0] != 0 or boundaries[-1] != x.shape[0] or any(n < 0 for n in sizes):
        raise ValueError("grouped_gemm segment boundaries must partition all input rows")
    if not sizes:
        # Preserve input and parameter gradients even for a zero-group batch.
        return x.new_empty((0, weight.shape[1])) + (x.sum() + weight.sum()) * 0
    return torch.cat([F.linear(part, w) for part, w in zip(torch.split(x, sizes), weight)], dim=0)


def grouped_gemm_multi(xs, ptrs, weights, *, fast_tf32=None):
    if not (len(xs) == len(ptrs) == len(weights)):
        raise ValueError("grouped_gemm_multi requires equally sized problem lists")
    if not xs:
        return []
    ops = backend()
    if ops is not None and _cuda_fp32(*xs, *weights):
        return ops.grouped_gemm_multi(xs, ptrs, weights, fast_tf32=fast_tf32)
    return [grouped_gemm(x, p, w, fast_tf32=fast_tf32) for x, p, w in zip(xs, ptrs, weights)]


class RouteStats:
    def __init__(self):
        self.calls = collections.Counter()
        self.declines = collections.Counter()

    def snapshot(self):
        return dict(calls=dict(self.calls), declines={str(k): v for k, v in self.declines.items()})

    def reset(self):
        self.calls.clear()
        self.declines.clear()


STATS = RouteStats()


def _enabled(name):
    return os.environ.get(name, '1').lower() not in ('0', 'false', 'off', 'no')


def _prepare(module, x, R, wigner):
    ops = backend()
    if ops is None:
        return None
    if not _cuda_fp32(x):
        return reference_fallback("SO2 requires CUDA FP32 without autocast or torch.func")
    if torch.is_tensor(R) and R.requires_grad:
        return reference_fallback("rotation geometry requires gradients")
    wigner = module._ensure_wigner_rotation(R, wigner)
    values = tuple(wigner.blocks) if hasattr(wigner, 'blocks') else wigner
    packed = ops.prepare_wigner(x, values, l_max=module.l_max,
                                rotate=module.rotate_in or module.rotate_out)
    if packed is None:
        return reference_fallback("unsupported or differentiable Wigner layout")
    cache = getattr(module, '_so2_backend_layouts', None)
    if cache is None:
        module._so2_backend_layouts = cache = {}
    key = str(x.device)
    if key not in cache:
        entries = lambda plans: tuple((int(p.l), int(p.mul), int(p.slice_info.start)) for p in plans)
        cache[key] = ops.prepare_layout(entries(module._in_entry_plans), entries(module._out_entry_plans),
            m_max=module.m_max, l_max=module.l_max, out_dim=module.irreps_out.dim, device=x.device,
            rotate_in=module.rotate_in, rotate_out=module.rotate_out, front=module.front)
    return ops, cache[key], packed, wigner


def activation_forward(module, x, R, routes, latents=None, wigner_D_all=None, *, fused=True):
    if not fused or not _enabled('DPTB_SO2_ACTIVATION_FUSED_P0'):
        return reference_fallback("activation-space fusion disabled")
    if getattr(routes, 'top1_reference_so2', False):
        return None
    idx, val = routes.topk_indices, routes.topk_values
    if idx is None or val is None or idx.shape[0] != x.shape[0]:
        return reference_fallback("activation fusion requires per-row routing")
    prepared = _prepare(module, x, R, wigner_D_all)
    if prepared is None:
        return None
    ops, layout, wigner, original = prepared
    if any(not layer.is_mole and not isinstance(layer.fc, torch.nn.Linear) for layer in module.m_linear):
        return reference_fallback("nonlinear interpolation blocks require the reference SO2 route")
    branch = getattr(routes, 'branch', 'all')
    fold = bool(getattr(routes, 'coefficients_sum_to_one', False)) and branch == 'all'
    linears = []
    for m in range(module.m_max+1):
        fc = module.fc_m0 if m == 0 else module.m_linear[m-1].fc
        routed = m == 0 or module.m_linear[m-1].is_mole
        if routed:
            weight, bias = fc._routed_weight_and_bias(fold)
            if m and bias is not None:
                return None
            linears.append(ops.LinearWeights(weight, bias, fc.weight_shared, fc.bias_shared))
        else:
            linears.append(ops.LinearWeights(fc.weight, fc.bias, routed=False))
    radials = None
    if module.radial_emb:
        weights = module.radial_emb(latents)
        bounds = module.m_in_index
        sizes = ([bounds[1]-bounds[0], bounds[module.m_max+1]-bounds[1]]
                 if module.front and module.m_max >= 1
                 else [bounds[m+1]-bounds[m] for m in range(module.m_max+1)])
        radials = torch.split(weights, sizes, dim=-1)
    idx = idx.to(device=x.device, dtype=torch.long)
    val = val.to(device=x.device, dtype=x.dtype)
    slots = tuple(routes.expert_slot_layout(j, idx[:, j], module.fc_m0.num_experts) for j in range(idx.shape[1]))
    routing = ops.ActivationRouting(idx, val, slots, fold, branch)
    schedule = os.environ.get('DPTB_SO2_ACTIVATION_FUSED_P0_GEMM', 'per_slot')
    out = ops.activation_forward(x, layout, wigner, tuple(linears), radials, routing, schedule=schedule)
    if out is None:
        return reference_fallback("activation backend declined this shape or route")
    STATS.calls['fused_p0'] += 1
    return out, original


def dense_forward(module, x, R, routes, latents=None, wigner_D_all=None):
    prepared = _prepare(module, x, R, wigner_D_all)
    if prepared is None:
        return None
    ops, layout, wigner, original = prepared
    # Keep the production m=0 reference and its arithmetic order.
    weights = module.radial_emb(latents) if module.radial_emb else None
    parts = None
    if weights is not None:
        bounds = [int(b) for b in module.m_in_index]
        sizes = [b-a for a,b in zip(bounds[:-1], bounds[1:])]
        rest = weights.shape[-1] - bounds[-1]
        parts = torch.split(weights, sizes+[rest] if rest else sizes, dim=-1)[:len(sizes)]
    out = x.new_zeros((x.shape[0], module.irreps_out.dim))
    inp0 = module._direct_rotate_pack_m(x, 0, original)
    if module.front and parts is not None:
        y0 = module.fc_m0(inp0 * parts[0], routes)
    elif parts is not None:
        y0 = module.fc_m0(inp0, routes) * parts[0]
    else:
        y0 = module.fc_m0(inp0, routes)
    module._accumulate_m0_output(out, y0, original)
    from .tensor_product_moe_v3 import _mole_graph_index
    graph_index = _mole_graph_index(routes, x.shape[0], device=x.device)
    # Pair leading dimensions are [N,2]; metadata depends only on that shape.
    pair_template = x.new_empty((x.shape[0], 2, 0))
    order, inverse, sorted_index = routes.indexed_flat_permutation(graph_index, pair_template)
    mixed = [x.new_empty((1, 0, 0))]
    for linear in module.m_linear:
        if not hasattr(linear.fc, '_mix_expert_parameters'):
            return None
        weight, bias = linear.fc._mix_expert_parameters(routes)
        if bias is not None:
            return None
        mixed.append(weight)
    ptr = routes.indexed_segment_ptr(sorted_index, mixed[1].shape[0] if len(mixed)>1 else 1, prefer_cpu=True)
    routing = ops.DenseRouting(graph_index, ptr, order, inverse)
    contributions = ops.dense_pairs(x, layout, wigner, tuple(mixed), parts, routing)
    if contributions is None:
        return reference_fallback("dense backend declined this shape or route")
    for part in contributions:
        out.add_(part)
    STATS.calls['dense'] += 1
    return out.contiguous(), original


def _true_dense_decline(reason):
    return reference_fallback('true-dense: ' + reason)


def true_dense_forward(module, x, weights, wigner_D_all):
    """Preserve the true-dense m=0 reference and delegate pair arithmetic."""
    if (not x.is_cuda or x.dtype != torch.float32 or torch.is_autocast_enabled()
            or getattr(torch._C, '_are_functorch_transforms_active', lambda: False)()):
        return _true_dense_decline('device, dtype, autocast or torch.func')
    if module.irreps_out.lmax < 1 or not all(isinstance(m.fc, torch.nn.Linear) for m in module.m_linear):
        return _true_dense_decline('scalar-only or interpolation layer')
    for names, lower in (
        (('DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES', 'SO2_CUDA_MIN_EDGES'), True),
        (('DPTB_SO2_INDEXED_SANDWICH_CUDA_MAX_EDGES', 'SO2_CUDA_MAX_EDGES'), False),
    ):
        value = next((os.environ[name] for name in names if name in os.environ), '0')
        try:
            limit = int(value)
        except ValueError:
            limit = 0
        if limit > 0 and (x.shape[0] < limit if lower else x.shape[0] > limit):
            return _true_dense_decline('edge-count gate')
    ops = backend()
    if ops is None or not hasattr(ops, 'true_dense_pairs'):
        return _true_dense_decline('SO2CUDA true-dense API unavailable or disabled')
    values = tuple(wigner_D_all.blocks) if hasattr(wigner_D_all, 'blocks') else wigner_D_all
    wigner = ops.prepare_wigner(x, values, l_max=module.l_max,
                                rotate=module.rotate_in or module.rotate_out)
    if wigner is None:
        return _true_dense_decline('differentiable or unsupported Wigner data')
    cache = getattr(module, '_so2_true_dense_layouts', None)
    if cache is None:
        module._so2_true_dense_layouts = cache = {}
    key = str(x.device)
    if key not in cache:
        entries = lambda plans: tuple((int(l), int(mul), int(sl.start)) for l, mul, sl, _ in plans)
        cache[key] = ops.prepare_layout(entries(module._in_entries), entries(module._out_entries),
            m_max=module.irreps_out.lmax, l_max=module.l_max, out_dim=module.irreps_out.dim,
            device=x.device, rotate_in=module.rotate_in, rotate_out=module.rotate_out, front=module.front)
    parts = None
    if weights is not None:
        bounds = [int(b) for b in module.m_in_index]
        parts = tuple(weights[:, a:b] for a, b in zip(bounds[:-1], bounds[1:]))
    # Preserve m=0 before m>0, including the original grouped rotation math.
    out = module._forward_m0(x, None if parts is None else parts[0], wigner_D_all)
    linears = (ops.LinearWeights(module.fc_m0.weight, module.fc_m0.bias, routed=False),) + tuple(
        ops.LinearWeights(m.fc.weight, m.fc.bias, routed=False) for m in module.m_linear)
    contributions = ops.true_dense_pairs(x, cache[key], wigner, linears, parts)
    if contributions is None:
        return _true_dense_decline('unsupported pair layout')
    for part in contributions:
        out = out + part
    STATS.calls['true_dense'] += 1
    return out.contiguous(), wigner_D_all

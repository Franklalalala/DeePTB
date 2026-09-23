"""prior_activate (activation-space MoLE) on the fused-P0 SO2CUDA route.

``streamed_m_major_fused_p0`` with ``indexed_sandwich_multi`` packs each m once,
runs the m-linears of all m in one cuBLAS grouped GEMM over rows sorted by
route, and scatters the raw GEMM output straight into the rotated output.  Its
routes are tokens with one mixed weight each, which per-edge routing cannot
afford.  Here the grouped GEMM is segmented by expert id instead: every active
edge contributes one row per top-k slot, each expert multiplies with its own
weight (the shared expert folded in when the coefficients sum to one), and the
k slot outputs of an edge are summed with its routing coefficients before the
scatter.  No per-edge weight is built.

m0 joins the same grouped calls (its bias is per expert).  m>0 blocks that are
not MoLE linears (the interpolation blocks of the output layer) run their own
linear and the finished-output scatter.  The kernels are SO2CUDA's existing
pack, cuBLAS grouped GEMM and scatter kernels.

Switch top-1 routes (dptb.nn.top1_prior, 256/1/0) are the case k = 1 without a
shared expert: each edge's selected expert, scaled by its retained probability,
exactly as top1_prior.linear computes it.
"""
import os

import torch
import torch.nn.functional as F

CALLS = 0
TOP1_CALLS = 0
FALLBACKS = 0
LAST_ERROR = None
_DISABLED = False
_DECLINE_WARNED = set()


class _RouteDeclined(RuntimeError):
    """An explicitly unsupported input, not a CUDA/runtime computation failure."""


def enabled():
    """DPTB_SO2_ACTIVATION_FUSED_P0=0 sends prior_activate to the pack/scatter route."""
    return os.environ.get("DPTB_SO2_ACTIVATION_FUSED_P0", "1").strip().lower() not in ("0", "false", "off", "no")


def _gemm_schedule():
    """'per_slot' (default): one grouped call per top-k slot, each over that slot's
    sort order (the row grouping of MOLELinear._apply_activation_space).
    'expanded': one grouped call holds the rows of every slot.  H200, PA 24/2/1,
    72,274 edges, fwd+bwd: 1124 ms per_slot, 1149 ms expanded."""
    value = os.environ.get("DPTB_SO2_ACTIVATION_FUSED_P0_GEMM", "per_slot").strip().lower()
    if value not in ("expanded", "per_slot"):
        raise RuntimeError("DPTB_SO2_ACTIVATION_FUSED_P0_GEMM must be expanded or per_slot, got %r" % value)
    return value


def _pair_rows(index):
    """Row index over [n] -> row index over the [n, 2] real/imaginary pair rows."""
    two = torch.arange(2, device=index.device, dtype=index.dtype)
    return (index.unsqueeze(1) * 2 + two).reshape(-1)


def _inverse(order):
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device, dtype=order.dtype))
    return inverse


def _layouts(mole_globals, idx, num_experts, schedule):
    """Sorted row layouts, cached on mole_globals: every SO2 layer of one forward
    shares the routing.

    Each layout is (gather, inverse, ptr, sorted_expert) over edge rows plus the
    same three for pair rows.  'expanded' has one layout over the n*k rows
    r = e*k + j of the flattened top-k table; 'per_slot' has one per slot j.
    """
    try:
        version = int(idx._version)
    except RuntimeError:
        version = None  # inference / transformed tensor: no reliable mutation token
    source = getattr(mole_globals, "_activation_fused_p0_source", None)
    cache = getattr(mole_globals, "_activation_fused_p0_layouts", None)
    if (cache is None or version is None or source is None
            or source[0] is not idx or source[1] != version):
        cache = {}
        mole_globals._activation_fused_p0_layouts = cache
        mole_globals._activation_fused_p0_source = (idx, version)
    key = (schedule, str(idx.device), tuple(idx.shape), int(num_experts))
    hit = cache.get(key)
    if hit is not None:
        return hit
    n, k = idx.shape
    if schedule == "expanded":
        slots = [(idx.reshape(-1), k)]
    else:
        slots = [(idx[:, j], 1) for j in range(k)]
    hit = []
    for slot, (flat, width) in enumerate(slots):
        flat = flat.to(torch.long)
        if schedule == "per_slot" and hasattr(mole_globals, "expert_slot_layout"):
            # Share the same expert sort/CPU ptr already used by scalar MoLE
            # and by pack/scatter. Avoid a second sort and device-host sync.
            order, inverse, ptr, sorted_expert = mole_globals.expert_slot_layout(slot, flat, num_experts)
        else:
            order = torch.argsort(flat, stable=True)
            inverse = _inverse(order)
            counts = torch.bincount(flat, minlength=int(num_experts))
            ptr = torch.zeros(int(num_experts) + 1, dtype=torch.long, device="cpu")
            ptr[1:] = torch.cumsum(counts.cpu(), dim=0)
            sorted_expert = flat.index_select(0, order)
        gather = torch.div(order, width, rounding_mode="floor") if width > 1 else order
        hit.append({
            "gather": gather,
            "inverse": inverse,
            "ptr": ptr,
            "sorted_expert": sorted_expert,
            "pair_gather": _pair_rows(gather),
            "pair_inverse": _pair_rows(inverse),
            "pair_ptr": ptr * 2,
        })
    if version is not None:
        cache[key] = hit
    return hit


def cuda_forward(module, x, R, mole_globals, latents=None, wigner_D_all=None):
    global CALLS, TOP1_CALLS
    from so2_cuda_ops import tensor_product as ops
    from so2_cuda_ops.grouped_gemm import grouped_gemm_multi

    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise _RouteDeclined("activation fused P0 requires CUDA float32")
    if torch.is_tensor(R) and R.requires_grad:
        raise _RouteDeclined("activation fused P0 requires fixed geometry")
    idx = mole_globals.topk_indices.to(device=x.device, dtype=torch.long)
    val = mole_globals.topk_values.to(device=x.device, dtype=x.dtype)
    n, k = idx.shape
    fold = bool(getattr(mole_globals, "coefficients_sum_to_one", False))
    schedule = _gemm_schedule()

    wigner_D_all = module._ensure_wigner_rotation(R, wigner_D_all)
    if ops._wigner_requires_grad(wigner_D_all):
        raise _RouteDeclined("activation fused P0 does not differentiate Wigner matrices")
    info = ops._wigner_tensor_and_mode(module, wigner_D_all, x)
    if info is None:
        raise _RouteDeclined("unsupported Wigner layout for activation fused P0")
    wigner, compact_offsets, mode, stride = info
    weights = module.radial_emb(latents) if module.radial_emb else None
    layouts = _layouts(mole_globals, idx, module.fc_m0.num_experts, schedule)
    out_dim = module.irreps_out.dim
    x = x.contiguous()

    blocks = []
    for m in range(module.m_max + 1):
        ib, il, ob, ol, offsets = ops._pair_maps(module, m, x.device)
        common = (wigner, ib, il, offsets, compact_offsets)
        if m == 0:
            inp = ops._PackM0Function.apply(x, *common, module.rotate_in, mode, stride)
            fc = module.fc_m0
        else:
            inp = ops._PackPairFunction.apply(x, *common, m, module.rotate_in, mode, stride)
            fc = module.m_linear[m - 1].fc if module.m_linear[m - 1].is_mole else None
        radial = None
        if weights is not None:
            radial = weights[:, module.m_in_index[m]:module.m_in_index[m + 1]]
            if m:
                radial = radial.unsqueeze(1)
            if module.front:
                inp = inp * radial
        block = {"m": m, "inp": inp, "radial": radial, "maps": (ob, ol, offsets), "fc": fc}
        if fc is not None:
            if fc.num_experts != module.fc_m0.num_experts:
                raise _RouteDeclined("activation fused P0 needs one expert count per SO2 layer")
            block["weight"], block["bias"] = fc._routed_weight_and_bias(fold)
            if m and block["bias"] is not None:
                raise _RouteDeclined("activation fused P0 expects bias-free m>0 MoLE linears")
        blocks.append(block)

    routed = [b for b in blocks if b["fc"] is not None]
    mixed = {}
    for j, lay in enumerate(layouts):
        xs, ptrs = [], []
        for b in routed:
            flat = b["inp"].reshape(-1, b["inp"].shape[-1])
            xs.append(flat.index_select(0, lay["gather"] if b["m"] == 0 else lay["pair_gather"]))
            ptrs.append(lay["ptr"] if b["m"] == 0 else lay["pair_ptr"])
        ys = grouped_gemm_multi(xs, ptrs, [b["weight"] for b in routed]) if routed else []
        for b, y in zip(routed, ys):
            if b["bias"] is not None:
                y = y + b["bias"].index_select(0, lay["sorted_expert"])
            y = y.index_select(0, lay["inverse"] if b["m"] == 0 else lay["pair_inverse"])
            if schedule == "expanded":
                # rows in (edge, slot[, pair]) order; the same sum over slots as
                # MOLELinear._apply_activation_space
                y = y.reshape(n, k, *b["inp"].shape[1:-1], y.shape[-1])
                view = [n] + [1] * (y.dim() - 2)
                parts = [y[:, s] * val[:, s].reshape(view) for s in range(k)]
            else:
                y = y.reshape(*b["inp"].shape[:-1], y.shape[-1])
                parts = [y * val[:, j].reshape([n] + [1] * (y.dim() - 1))]
            for part in parts:
                mixed[b["m"]] = part if b["m"] not in mixed else mixed[b["m"]] + part

    out = None
    for b in blocks:
        m, inp, radial, fc = b["m"], b["inp"], b["radial"], b["fc"]
        ob, ol, offsets = b["maps"]
        common_out = (wigner, ob, ol, offsets, compact_offsets, out_dim)
        if fc is None:
            y = module.m_linear[m - 1](inp, mole_globals)
            if radial is not None and not module.front:
                y = y * radial
            part = ops._ScatterPairOutputFunction.apply(y.contiguous(), *common_out, m, module.rotate_out, mode, stride)
        else:
            y = mixed[m]
            if not fold and fc.num_shared_experts > 0:
                shared_bias = fc.bias_shared.sum(0) if fc.bias_shared is not None else None
                y = y + F.linear(inp, fc.weight_shared.sum(0), shared_bias)
            if m == 0:
                if radial is not None and not module.front:
                    y = y * radial
                part = ops._ScatterM0OutputFunction.apply(y.contiguous(), *common_out, module.rotate_out, mode, stride)
            elif radial is not None and not module.front:
                y = module.m_linear[m - 1]._finish_linear_output(y) * radial
                part = ops._ScatterPairOutputFunction.apply(y.contiguous(), *common_out, m, module.rotate_out, mode, stride)
            else:
                part = ops._ScatterRawPairOutputFunction.apply(y.contiguous(), *common_out, m, module.rotate_out, mode, stride)
        out = part if out is None else out + part
    CALLS += 1
    if getattr(mole_globals, "top1_independent", False):
        TOP1_CALLS += 1
    if CALLS == 1:
        print("SO2_ACTIVATION_FUSED_P0_ACTIVE pid=%s edges=%s top_k=%s m_max=%s gemm=%s mode=%s"
              % (os.getpid(), n, k, module.m_max, schedule, mode), flush=True)
    return out.contiguous(), wigner_D_all


def _routed(mole_globals, x):
    """The per-row top-k routing MOLELinear.forward would apply in activation space:
    prior_activate routes carry coefficients (without them MOLELinear averages the
    experts instead); Switch top-1 routes (top1_prior) carry the selected expert and
    its retained probability, applied as gate * (W_e x + b_e)."""
    idx = getattr(mole_globals, "topk_indices", None)
    val = getattr(mole_globals, "topk_values", None)
    if not torch.is_tensor(idx) or not torch.is_tensor(val):
        return False
    if (len(x.shape) != 2 or idx.dim() != 2 or val.shape != idx.shape
            or idx.shape[0] != x.shape[0] or idx.shape[1] == 0
            or idx.dtype not in (torch.int32, torch.int64)):
        return False
    if getattr(mole_globals, "top1_independent", False):
        return not getattr(mole_globals, "top1_reference_so2", False) and idx.shape[1] == 1
    # The adapter is also callable directly. Never reinterpret graph-token
    # weight-space routing as per-row activation-space routing.
    return (bool(getattr(mole_globals, "activation_space", False))
            and getattr(mole_globals, "coefficients", None) is not None)


def try_forward(module, x, R, mole_globals, latents=None, wigner_D_all=None):
    """Return (out, wigner) or None; the caller then takes the pack/scatter route.

    None when disabled, on CPU or float64, with differentiable geometry, without
    per-row top-k routing, or when SO2CUDA is not importable.  An
    explicitly unsupported input declines this call without disabling later
    compatible calls. Unexpected RuntimeError (including OOM/device faults) is
    propagated; neither this wrapper nor the fallback may retry a failed kernel.
    """
    global FALLBACKS, LAST_ERROR, _DISABLED
    if _DISABLED or not enabled():
        return None
    if x.device.type != "cuda" or x.dtype != torch.float32:
        return None
    if (torch.is_autocast_enabled()
            or getattr(torch._C, "_are_functorch_transforms_active", lambda: False)()):
        return None  # first-order float32 adapter, not an autocast/torch.func qualification
    if (torch.is_tensor(R) and R.requires_grad) or not _routed(mole_globals, x):
        return None
    if getattr(mole_globals, "top1_independent", False):
        linears = [module.fc_m0] + [block.fc for block in getattr(module, "m_linear", ())
                                   if getattr(block, "is_mole", False)]
        if any(fc.num_shared_experts != 0 for fc in linears):
            return None  # top1_prior.linear refuses sharing in every block, not only m0
    try:
        import so2_cuda_ops  # noqa: F401
    except ImportError as exc:
        _DISABLED = True
        LAST_ERROR = repr(exc)[:800]
        print("SO2_ACTIVATION_FUSED_P0_UNAVAILABLE (pack/scatter route): %s" % LAST_ERROR, flush=True)
        return None
    try:
        return cuda_forward(module, x, R, mole_globals, latents, wigner_D_all)
    except _RouteDeclined as exc:
        FALLBACKS += 1
        LAST_ERROR = repr(exc)[:800]
        if LAST_ERROR not in _DECLINE_WARNED:
            _DECLINE_WARNED.add(LAST_ERROR)
            print("SO2_ACTIVATION_FUSED_P0_DECLINED (this call only): %s" % LAST_ERROR, flush=True)
        return None

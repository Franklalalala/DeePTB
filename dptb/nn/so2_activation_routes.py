"""Activation-space MoLE layers of SO2_Linear on the SO2CUDA kernels.

``prior_activate`` routes every active edge to its top-k experts and sums their outputs
with the routing coefficients (``MOLELinear._apply_activation_space``); a Switch top-1
route (``dptb.nn.top1_prior``) scales the selected expert by its retained probability.
The weight-space SO2 routes build one mixed weight per route token, which per-edge
routing cannot afford, so these layers take one of two routes:

``fused_p0``
    (``so2_fusion_mode: streamed_m_major_fused_p0``) SO2CUDA packs m0 and, in one
    multi-m pack, every m>0 block.  Per top-k slot one cuBLAS grouped GEMM, segmented
    by expert id, covers m0 and every MoLE m>0 block (each expert with its own weight;
    the shared expert folded in when the coefficients sum to one).  The slot outputs of
    an edge are summed with its coefficients and one output-major scatter writes the
    rotated output of all blocks from the raw GEMM outputs.  m>0 blocks that are not
    MoLE linears (the interpolation blocks of an output layer) run their own linear.
    No per-edge weight is built.
``pack_scatter``
    The same SO2CUDA rotation, packing and scatter; the expert linears stay in
    ``MOLELinear.forward``.

``forward`` tries them in this order.  A route that does not support a call declines it
and the caller runs the grouped streaming route; nothing is retried after an error
inside a route.  Both routes are first-order CUDA float32 routes for fixed geometry.

``DPTB_SO2_ACTIVATION_FUSED_P0=0`` and ``DPTB_SO2_ACTIVATION_CUDA=0`` switch the two
routes off; ``DPTB_SO2_ACTIVATION_FUSED_P0_GEMM=expanded`` runs the rows of all slots
in one grouped call instead of one call per slot.
"""
import collections
import logging
import os

import torch
import torch.nn.functional as F

from .tensor_product_moe_v3 import permute_rows

log = logging.getLogger(__name__)

FUSED_P0 = "fused_p0"
PACK_SCATTER = "pack_scatter"


class RouteStats:
    """Per-process record of what the activation-space SO2 layers ran.

    ``calls[route]`` counts the layer forwards each route returned (``fused_p0_top1``
    the Switch share of ``fused_p0``); ``declines[(route, reason)]`` counts the calls a
    route declined, ``route`` being ``"all"`` for conditions both routes share."""

    def __init__(self):
        self.calls = collections.Counter()
        self.declines = collections.Counter()

    def snapshot(self):
        return {
            "calls": dict(self.calls),
            "declines": {"%s: %s" % key: count for key, count in self.declines.items()},
        }

    def reset(self):
        self.calls.clear()
        self.declines.clear()


STATS = RouteStats()
_LOGGED = set()
_OPS = {}


class RouteDeclined(Exception):
    """A call the route does not support; not a failure of the computation."""


def _switch_on(name):
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "off", "no")


def _gemm_schedule():
    """'per_slot' (default): one grouped call per top-k slot over that slot's expert
    sort, the row grouping of MOLELinear._apply_activation_space.  'expanded': one
    grouped call over the rows of every slot."""
    value = os.environ.get("DPTB_SO2_ACTIVATION_FUSED_P0_GEMM", "per_slot").strip().lower()
    if value not in ("expanded", "per_slot"):
        raise RuntimeError("DPTB_SO2_ACTIVATION_FUSED_P0_GEMM must be expanded or per_slot, got %r" % value)
    return value


def _log_once(key, message):
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(message, flush=True)


_DECLINE_LABEL = {"all": "CUDA", FUSED_P0: "FUSED_P0", PACK_SCATTER: "PACK_SCATTER"}


def _decline(route, reason, *, log_it=False):
    STATS.declines[(route, reason)] += 1
    if log_it:
        _log_once(("declined", route, reason),
                  "SO2_ACTIVATION_%s_DECLINED (this call only): %s" % (_DECLINE_LABEL[route], reason))
    return None


def _load_ops():
    """(so2_cuda_ops.tensor_product, grouped_gemm_multi), or None when not importable."""
    if "ops" not in _OPS:
        try:
            from so2_cuda_ops import tensor_product
            from so2_cuda_ops.grouped_gemm import grouped_gemm_multi
            _OPS["ops"] = (tensor_product, grouped_gemm_multi)
        except ImportError as exc:
            _OPS["ops"] = None
            _log_once("unavailable", "SO2_ACTIVATION_CUDA_UNAVAILABLE (streamed route): %s" % repr(exc)[:800])
    return _OPS["ops"]


def _functorch_active():
    return getattr(torch._C, "_are_functorch_transforms_active", lambda: False)()


def _preflight(x, R, mole_globals):
    """Why neither route can take the call, or None."""
    if not (getattr(mole_globals, "activation_space", False) or getattr(mole_globals, "top1_independent", False)):
        return "weight-space routing"  # graph-token routes are never reinterpreted per row
    if x.device.type != "cuda" or x.dtype != torch.float32:
        return "not CUDA float32"
    if torch.is_tensor(R) and R.requires_grad:
        return "differentiable geometry"
    if torch.is_autocast_enabled():
        return "autocast"
    if _functorch_active():
        return "torch.func transform"
    if getattr(mole_globals, "top1_independent", False) and getattr(mole_globals, "top1_reference_so2", False):
        return "top1_reference_so2"
    return None


def _per_row_routing(mole_globals, x):
    """The per-row top-k routing that MOLELinear.forward applies in activation space.

    prior_activate routes carry coefficients (without them MOLELinear averages the
    experts); Switch routes carry the selected expert and its retained probability."""
    idx = getattr(mole_globals, "topk_indices", None)
    val = getattr(mole_globals, "topk_values", None)
    if not torch.is_tensor(idx) or not torch.is_tensor(val):
        return False
    if (x.dim() != 2 or idx.dim() != 2 or val.shape != idx.shape
            or idx.shape[0] != x.shape[0] or idx.shape[1] == 0
            or idx.dtype not in (torch.int32, torch.int64)):
        return False
    if getattr(mole_globals, "top1_independent", False):
        return idx.shape[1] == 1
    return getattr(mole_globals, "coefficients", None) is not None


def _routed_linears(module, fold):
    """(fc, weight, bias) per m for the fused route; fc is None for a non-MoLE m block.

    Raises RouteDeclined for layer structures the grouped GEMM does not cover, before
    any kernel runs."""
    linears = []
    for m in range(module.m_max + 1):
        fc = module.fc_m0 if m == 0 else (
            module.m_linear[m - 1].fc if module.m_linear[m - 1].is_mole else None)
        weight = bias = None
        if fc is not None:
            if fc.num_experts != module.fc_m0.num_experts:
                raise RouteDeclined("one expert count per SO2 layer required")
            weight, bias = fc._routed_weight_and_bias(fold)
            if m and bias is not None:
                raise RouteDeclined("m>0 MoLE linears with a bias")
        linears.append((fc, weight, bias))
    return linears


def _wigner_layout(ops, module, x, R, wigner_D_all):
    wigner_D_all = module._ensure_wigner_rotation(R, wigner_D_all)
    if ops._wigner_requires_grad(wigner_D_all):
        raise RouteDeclined("differentiable Wigner matrices")
    info = ops._wigner_tensor_and_mode(module, wigner_D_all, x)
    if info is None:
        raise RouteDeclined("unsupported Wigner layout")
    return wigner_D_all, info


def _radial_blocks(module, latents):
    """radial_emb(latents) cut the way the layer applies it, or None without a radial embedding.

    A front layer scales its packed inputs: (m0 block, every m>0 block in pack order).
    Otherwise each output block has its own piece.  One split: its backward concatenates
    the piece gradients once, where a weights[:, a:b] slice per piece zero-fills a
    full-width gradient for every piece."""
    if not module.radial_emb:
        return None
    weights = module.radial_emb(latents)
    bounds = module.m_in_index
    if module.front and module.m_max >= 1:
        sizes = [bounds[1] - bounds[0], bounds[module.m_max + 1] - bounds[1]]
    else:
        sizes = [bounds[m + 1] - bounds[m] for m in range(module.m_max + 1)]
    return torch.split(weights, sizes, dim=-1)


def _entry_map(bases, levels, dim, device):
    """For every feature f < dim, the (block, channel, d, l) entries of the irrep
    channels that cover f, as the output-major scatter kernels take them:
    (entry_offsets[dim + 1], entry_m, entry_channel, entry_d, entry_l)."""
    per_feature = [[] for _ in range(dim)]
    for block, (base_t, l_t) in enumerate(zip(bases, levels)):
        for channel, (base, l) in enumerate(zip(base_t.tolist(), l_t.tolist())):
            for d in range(2 * l + 1):
                per_feature[base + d].append((block, channel, d, l))
    offsets = [0]
    for entries in per_feature:
        offsets.append(offsets[-1] + len(entries))
    columns = list(zip(*(e for entries in per_feature for e in entries))) or [(), (), (), ()]
    as_long = lambda values: torch.tensor(list(values), dtype=torch.long, device=device)  # noqa: E731
    return (as_long(offsets),) + tuple(as_long(column) for column in columns)


def _layer_layout(module, device, in_dim):
    """Pair maps of every m block and the multi-m layout of the m>0 blocks, cached on the layer.

    The pack and scatter functions save these tensors for backward, so they are built
    outside inference mode (an inference-mode warm-up would otherwise leave inference
    tensors for a later training call), with their own pair-map cache on the layer."""
    cache = getattr(module, "_so2_activation_layout", None)
    if cache is None:
        cache = {}
        module._so2_activation_layout = cache
    key = (str(device), int(in_dim))
    hit = cache.get(key)
    if hit is not None:
        return hit
    from so2_cuda_ops.so2_sandwich_common import so2_pair_maps

    with torch.inference_mode(False), torch.no_grad():
        maps = [so2_pair_maps(module, m, device, cache_attr="_so2_activation_pair_maps")
                for m in range(module.m_max + 1)]
        hit = {"m0": maps[0], "multi": None}
        if module.m_max >= 1:
            maps = maps[1:]
            in_bases = [p[0] for p in maps]
            in_ls = [p[1] for p in maps]
            out_bases = [p[2] for p in maps]
            out_ls = [p[3] for p in maps]
            cins = [int(b.numel()) for b in in_bases]

            def prefix(sizes):
                values = [0]
                for size in sizes:
                    values.append(values[-1] + size)
                return torch.tensor(values, dtype=torch.long, device=device)

            hit["multi"] = {
                "in_bases": in_bases,
                "in_ls": in_ls,
                "out_bases": out_bases,
                "out_ls": out_ls,
                "cins": cins,
                "cin_prefix": prefix(cins),
                "cout_prefix": prefix([int(b.numel()) for b in out_bases]),
                "m_values": torch.tensor(list(range(1, module.m_max + 1)), dtype=torch.long, device=device),
                "in_entries": _entry_map(in_bases, in_ls, int(in_dim), device),
                "out_entries": _entry_map(out_bases, out_ls, int(module.irreps_out.dim), device),
            }
    cache[key] = hit
    return hit


class _PackAll(torch.autograd.Function):
    """SO2CUDA packing of a layer input: the m0 block [n, cin0] and, by the multi-m
    pack, every m>0 block as pairs [n, 2, sum cin_m].

    Packing reads each packed value from its irrep block through the rotation, so the
    backward is the transposed rotation scattered into the input layout.  It is written
    output-major (the m0 scatter and the multi-m pair scatter through the input maps):
    one thread sums each input-gradient element, where the packs' own backwards add
    every block with atomics, several m blocks onto the same element."""

    @staticmethod
    def forward(ctx, x, ops, wigner, compact_offsets, mode, stride, rotate, m0_maps, lay):
        in_base, in_l, offsets = m0_maps
        inp0 = ops._pack_m0_cuda(x, wigner, in_base, in_l, offsets, compact_offsets, rotate, mode, stride)
        pairs = ops._pack_pairs_multi_cuda(
            x, wigner, lay["in_bases"], lay["in_ls"], offsets, compact_offsets,
            lay["cin_prefix"], lay["m_values"], rotate, mode, stride)
        ctx.save_for_backward(wigner)
        ctx.meta = (ops, compact_offsets, mode, stride, rotate, m0_maps, lay, int(x.shape[1]))
        return inp0, pairs

    @staticmethod
    def backward(ctx, grad_inp0, grad_pairs):
        (wigner,) = ctx.saved_tensors
        ops, compact_offsets, mode, stride, rotate, (in_base, in_l, offsets), lay, in_dim = ctx.meta
        grad_x = None
        if grad_inp0 is not None:
            grad_x = ops._scatter_m0_forward_cuda(
                grad_inp0.contiguous(), wigner, in_base, in_l, offsets, compact_offsets,
                in_dim, rotate, mode, stride)
        if grad_pairs is not None:
            blocks = [block.contiguous() for block in torch.split(grad_pairs, lay["cins"], dim=-1)]
            part = ops._scatter_pairs_multi_output_major_forward_cuda(
                blocks, wigner, offsets, compact_offsets, lay["cin_prefix"], lay["m_values"],
                *lay["in_entries"], in_dim, rotate, mode, stride)
            grad_x = part if grad_x is None else grad_x + part
        return grad_x, None, None, None, None, None, None, None, None


class _Packed:
    """The SO2CUDA packing of one layer call.

    ``_PackAll`` packs m0 and, in one multi-m pack, every m>0 block; its backward writes
    each input-gradient element once.  ``scatter`` writes the rotated output of all
    blocks in one output-major pass instead of a full-width output per block."""

    def __init__(self, ops, module, x, wigner_info, radials):
        self.ops = ops
        self.module = module
        self.wigner, self.compact_offsets, self.mode, self.stride = wigner_info
        x = x.contiguous()
        layout = _layer_layout(module, x.device, x.shape[1])
        ib, il, self.m0_out_base, self.m0_out_l, self.offsets = layout["m0"]
        rotate = bool(module.rotate_in)
        self.layout = layout["multi"]
        if self.layout is None:
            inp0 = ops._PackM0Function.apply(
                x, self.wigner, ib, il, self.offsets, self.compact_offsets, rotate, self.mode, self.stride)
            packed = None
        else:
            inp0, packed = _PackAll.apply(x, ops, self.wigner, self.compact_offsets, self.mode, self.stride,
                                          rotate, (ib, il, self.offsets), self.layout)
        front = radials is not None and module.front
        if front:
            inp0 = inp0 * radials[0]
        self.inputs = [inp0]
        if packed is not None:
            if front:
                packed = packed * radials[1].unsqueeze(1)
            # [n, 2, cin_m] views; each reshapes to [2n, cin_m] rows without a copy
            self.inputs.extend(torch.split(packed, self.layout["cins"], dim=-1))
        self.radials = None if radials is None or module.front else radials

    def finish_m0(self, y):
        """The m0 output with the radial weight of a layer that scales its outputs."""
        return y if self.radials is None else y * self.radials[0]

    def finish_raw(self, m, raw):
        """Raw pair output [n, 2, 2C] of block m with the output radial folded in:
        complex_pair_output(raw) * r equals complex_pair_output(raw * [r, r])."""
        if self.radials is None:
            return raw
        radial = self.radials[m]
        return raw * torch.cat((radial, radial), dim=-1).unsqueeze(1)

    def scatter(self, y0, raws):
        """Rotated layer output from the finished m0 block and the raw m>0 blocks."""
        module = self.module
        head = (y0.contiguous(), self.wigner, self.m0_out_base, self.m0_out_l, self.offsets, self.compact_offsets)
        rotate = (module.rotate_out, self.mode, self.stride)
        if self.layout is None:
            return self.ops._ScatterM0OutputFunction.apply(*head, module.irreps_out.dim, *rotate)
        lay = self.layout
        return self.ops._ScatterM0RawPairsMultiOutputMajorFunction.apply(
            *head, lay["cout_prefix"], lay["m_values"], *lay["out_entries"], module.irreps_out.dim, *rotate,
            len(raws), *(raw.contiguous() for raw in raws), *lay["out_bases"], *lay["out_ls"])


def _pair_rows(index):
    """Row index over [n] -> row index over the [n, 2] real/imaginary pair rows."""
    two = torch.arange(2, device=index.device, dtype=index.dtype)
    return (index.unsqueeze(1) * 2 + two).reshape(-1)


def _slot_layouts(mole_globals, idx, num_experts, schedule):
    """Expert-sorted row layouts of the grouped GEMM, cached on mole_globals (every SO2
    layer of one forward shares the routing).

    'per_slot' has one layout per slot j, the expert sort of MOLEGlobals.expert_slot_layout
    shared with MOLELinear and top1_prior.linear; 'expanded' one layout over the n*k rows
    r = e*k + j of the flattened top-k table.  The cache follows the storage and version
    of the routing tensor; inference tensors, without a version counter, are not cached."""
    try:
        version = int(idx._version)
    except RuntimeError:
        version = None
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
    slots = [(idx.reshape(-1), k)] if schedule == "expanded" else [(idx[:, j], 1) for j in range(k)]
    hit = []
    for slot, (flat, width) in enumerate(slots):
        flat = flat.to(torch.long)
        if schedule == "per_slot":
            order, inverse, ptr, _ = mole_globals.expert_slot_layout(slot, flat, num_experts)
        else:
            order = torch.argsort(flat, stable=True)
            inverse = torch.empty_like(order)
            inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device, dtype=order.dtype))
            ptr = torch.zeros(int(num_experts) + 1, dtype=torch.long, device="cpu")
            ptr[1:] = torch.cumsum(torch.bincount(flat, minlength=int(num_experts)).cpu(), dim=0)
        # 'expanded' gathers each edge row once per slot (order // k); only the
        # per-slot gathers and the inverse gathers are permutations.
        gather = torch.div(order, width, rounding_mode="floor") if width > 1 else order
        hit.append({
            "gather": gather,
            "order": order,
            "inverse": inverse,
            "ptr": ptr,
            "pair_gather": _pair_rows(gather),
            "pair_order": _pair_rows(order),
            "pair_inverse": _pair_rows(inverse),
            "pair_ptr": ptr * 2,
        })
    if version is not None:
        cache[key] = hit
    return hit


def _fused_p0(module, x, packed, mole_globals, linears, schedule, grouped_gemm_multi):
    idx = mole_globals.topk_indices.to(device=x.device, dtype=torch.long)
    val = mole_globals.topk_values.to(device=x.device, dtype=x.dtype)
    n, k = idx.shape
    branch = getattr(mole_globals, "branch", "all")
    fold = bool(getattr(mole_globals, "coefficients_sum_to_one", False)) and branch == "all"
    routed = [m for m, (fc, _, _) in enumerate(linears) if fc is not None]
    # an expert bias is one more weight column against a column of ones: the GEMM adds
    # it and its gradient is the per-expert row sum of the GEMM's weight gradient
    weights = [linears[m][1] if linears[m][2] is None
               else torch.cat((linears[m][1], linears[m][2].unsqueeze(-1)), dim=-1) for m in routed]
    mixed = {}
    slot_layouts = [] if branch == "shared" else _slot_layouts(mole_globals, idx, module.fc_m0.num_experts, schedule)
    for j, lay in enumerate(slot_layouts):
        xs, ptrs = [], []
        for m in routed:
            flat = packed.inputs[m].reshape(-1, packed.inputs[m].shape[-1])
            pre = "" if m == 0 else "pair_"
            if schedule == "per_slot":
                rows = permute_rows(flat, lay[pre + "order"], lay[pre + "inverse"])
            else:
                rows = flat.index_select(0, lay[pre + "gather"])
            if linears[m][2] is not None:
                rows = torch.cat((rows, rows.new_ones(rows.shape[0], 1)), dim=1)
            xs.append(rows)
            ptrs.append(lay[pre + "ptr"])
        ys = grouped_gemm_multi(xs, ptrs, weights) if routed else []
        for m, y in zip(routed, ys):
            pre = "" if m == 0 else "pair_"
            y = permute_rows(y, lay[pre + "inverse"], lay[pre + "order"])
            inp = packed.inputs[m]
            if schedule == "expanded":
                # rows in (edge, slot[, pair]) order: the sum over slots of
                # MOLELinear._apply_activation_space
                y = y.reshape(n, k, *inp.shape[1:-1], y.shape[-1])
                view = [n] + [1] * (y.dim() - 2)
                parts = [y[:, s] * val[:, s].reshape(view) for s in range(k)]
            else:
                y = y.reshape(*inp.shape[:-1], y.shape[-1])
                parts = [y * val[:, j].reshape([n] + [1] * (y.dim() - 1))]
            for part in parts:
                mixed[m] = part if m not in mixed else mixed[m] + part

    raws = []
    for m, (fc, _, _) in enumerate(linears):
        inp = packed.inputs[m]
        if fc is None:
            if branch == "routed":  # a non-MoLE block belongs to the shared branch
                raw = inp.new_zeros(*inp.shape[:-1], 2 * module.m_linear[m - 1].num_out_channel)
            else:
                raw = module.m_linear[m - 1].fc(inp)  # interpolation block
        else:
            raw = mixed.get(m)
            if raw is None:
                raw = inp.new_zeros(*inp.shape[:-1], fc.out_features)
            if branch != "routed" and not fold and fc.num_shared_experts > 0:
                shared_bias = fc.bias_shared.sum(0) if fc.bias_shared is not None else None
                raw = raw + F.linear(inp, fc.weight_shared.sum(0), shared_bias)
        raws.append(packed.finish_m0(raw) if m == 0 else packed.finish_raw(m, raw))
    return packed.scatter(raws[0], raws[1:])


def _pack_scatter(module, packed, mole_globals):
    y0 = packed.finish_m0(module.fc_m0(packed.inputs[0], mole_globals))
    raws = []
    for m in range(1, module.m_max + 1):
        linear = module.m_linear[m - 1]
        inp = packed.inputs[m]
        if linear.is_mole:
            raw = linear.fc(inp, mole_globals)
        elif getattr(mole_globals, "branch", "all") == "routed":  # a non-MoLE block belongs to the shared branch
            raw = inp.new_zeros(*inp.shape[:-1], 2 * linear.num_out_channel)
        else:
            raw = linear.fc(inp)
        raws.append(packed.finish_raw(m, raw))
    return packed.scatter(y0, raws)


def forward(module, x, R, mole_globals, latents=None, wigner_D_all=None, *, fused):
    """(out, wigner_D_all) of an activation-space SO2_Linear call, or None.

    ``fused`` asks for the fused-P0 route first.  None means that neither route takes
    the call; the caller then runs the grouped streaming route."""
    reason = _preflight(x, R, mole_globals)
    if reason is not None:
        return _decline("all", reason)
    ops = _load_ops()
    if ops is None:
        return _decline("all", "so2_cuda_ops not importable")
    tensor_product, grouped_gemm_multi = ops

    linears = schedule = None
    if fused:
        if not _switch_on("DPTB_SO2_ACTIVATION_FUSED_P0"):
            _decline(FUSED_P0, "DPTB_SO2_ACTIVATION_FUSED_P0=0")
        elif not _per_row_routing(mole_globals, x):
            _decline(FUSED_P0, "no per-row top-k routing", log_it=True)
        else:
            fold = (bool(getattr(mole_globals, "coefficients_sum_to_one", False))
                    and getattr(mole_globals, "branch", "all") == "all")
            switch = getattr(mole_globals, "top1_independent", False)
            try:
                schedule = _gemm_schedule()
                linears = _routed_linears(module, fold)
                if switch and any(fc is not None and fc.num_shared_experts for fc, _, _ in linears):
                    # top1_prior.linear refuses shared experts in every block
                    raise RouteDeclined("Switch route with shared experts")
            except RouteDeclined as exc:
                linears = _decline(FUSED_P0, str(exc), log_it=True)
    if linears is None and not _switch_on("DPTB_SO2_ACTIVATION_CUDA"):
        return _decline(PACK_SCATTER, "DPTB_SO2_ACTIVATION_CUDA=0")

    try:
        wigner_D_all, wigner_info = _wigner_layout(tensor_product, module, x, R, wigner_D_all)
    except RouteDeclined as exc:
        return _decline("all", str(exc), log_it=True)
    packed = _Packed(tensor_product, module, x, wigner_info, _radial_blocks(module, latents))

    if linears is not None:
        out = _fused_p0(module, x, packed, mole_globals, linears, schedule, grouped_gemm_multi)
        n, k = mole_globals.topk_indices.shape
        STATS.calls[FUSED_P0] += 1
        if getattr(mole_globals, "top1_independent", False):
            STATS.calls[FUSED_P0 + "_top1"] += 1
        _log_once(("active", FUSED_P0),
                  "SO2_ACTIVATION_FUSED_P0_ACTIVE pid=%s edges=%s top_k=%s m_max=%s gemm=%s mode=%s"
                  % (os.getpid(), n, k, module.m_max, schedule, packed.mode))
    else:
        out = _pack_scatter(module, packed, mole_globals)
        STATS.calls[PACK_SCATTER] += 1
        _log_once(("active", PACK_SCATTER),
                  "SO2_ACTIVATION_CUDA_ACTIVE pid=%s edges=%s m_max=%s mode=%s"
                  % (os.getpid(), x.shape[0], module.m_max, packed.mode))
    return out.contiguous(), wigner_D_all

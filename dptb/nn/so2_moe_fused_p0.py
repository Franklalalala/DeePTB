from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2WignerBlocks, _mole_graph_index


_EXT = None
_WARNED: set[str] = set()
_FALSE = {"", "0", "false", "False", "FALSE", "off", "OFF", "no", "No"}
_PAIR_SEGMENT_LAYOUT_CACHE: "OrderedDict[tuple, tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]]" = OrderedDict()
_PAIR_SEGMENT_LAYOUT_CACHE_MAX = 64


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in _FALSE


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    here = Path(__file__).resolve().parent
    build_dir = Path(
        os.environ.get(
            "DPTB_SO2_MOE_FUSED_P0_BUILD_DIR",
            Path.home() / ".cache" / "dptb_so2_moe_fused_p0",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    cflags = ["-O3"]
    cuda_flags = ["-O3", "--expt-relaxed-constexpr"]
    include_paths = []
    if _flag("DPTB_SO2_MOE_FUSED_P0_LINEINFO"):
        cuda_flags.append("-lineinfo")
    cutlass_root = os.environ.get("DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT")
    if cutlass_root:
        cutlass_root_path = Path(cutlass_root)
        include_paths.extend([
            str(cutlass_root_path / "include"),
            str(cutlass_root_path / "tools" / "util" / "include"),
        ])
        cflags.append("-DDPTB_SO2_MOE_FUSED_P0_CUTLASS=1")
        cuda_flags.append("-DDPTB_SO2_MOE_FUSED_P0_CUTLASS=1")

    _EXT = load(
        name="dptb_so2_moe_fused_p0",
        sources=[
            str(here / "csrc" / "so2_moe_fused_p0.cpp"),
            str(here / "csrc" / "so2_moe_fused_p0_kernel.cu"),
        ],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=include_paths,
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=_flag("DPTB_SO2_MOE_FUSED_P0_VERBOSE"),
    )
    return _EXT


def _wigner_requires_grad(wigner_D_all) -> bool:
    if torch.is_tensor(wigner_D_all):
        return bool(wigner_D_all.requires_grad)
    if isinstance(wigner_D_all, SO2WignerBlocks):
        return any(block.requires_grad for block in wigner_D_all.blocks)
    return False


def _empty_long(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.long, device=device)


def _wigner_tensor_and_mode(module, wigner_D_all, x: torch.Tensor):
    if not (module.rotate_in or module.rotate_out) or module.l_max == 0:
        return x.new_empty((0,)), _empty_long(x.device), 0, 0

    if torch.is_tensor(wigner_D_all):
        if wigner_D_all.device != x.device or wigner_D_all.dtype != x.dtype or wigner_D_all.dim() != 3:
            _warn_once("bad_dense_wigner_fallback", "streamed_m_major_fused_p0 requires dense CUDA fp32 Wigner [N,D,D].")
            return None
        if wigner_D_all.shape[0] != x.shape[0] or wigner_D_all.shape[1] != wigner_D_all.shape[2]:
            _warn_once("bad_dense_wigner_shape_fallback", "streamed_m_major_fused_p0 dense Wigner shape is incompatible.")
            return None
        return wigner_D_all.contiguous(), _empty_long(x.device), 1, int(wigner_D_all.shape[1])

    if isinstance(wigner_D_all, SO2WignerBlocks):
        pieces = []
        compact_offsets = []
        cursor = 0
        for l in range(module.l_max + 1):
            if l >= len(wigner_D_all.blocks):
                _warn_once("missing_compact_wigner_fallback", "streamed_m_major_fused_p0 compact Wigner blocks are incomplete.")
                return None
            dim = int(module.dims[l])
            block = wigner_D_all.block(l)
            if (
                block.device != x.device
                or block.dtype != x.dtype
                or block.dim() != 3
                or block.shape[0] != x.shape[0]
                or block.shape[-2:] != (dim, dim)
            ):
                _warn_once("bad_compact_wigner_fallback", "streamed_m_major_fused_p0 compact Wigner block shape is incompatible.")
                return None
            compact_offsets.append(cursor)
            cursor += dim * dim
            pieces.append(block.reshape(x.shape[0], dim * dim))
        packed = torch.cat(pieces, dim=1).contiguous() if pieces else x.new_empty((x.shape[0], 0))
        offsets = torch.tensor(compact_offsets, dtype=torch.long, device=x.device).contiguous()
        return packed, offsets, 2, int(cursor)

    _warn_once(
        "unknown_wigner_fallback",
        f"streamed_m_major_fused_p0 received unsupported Wigner type {type(wigner_D_all)!r}; falling back.",
    )
    return None


def _pair_maps(module, m: int, device: torch.device):
    cache = getattr(module, "_so2_moe_fused_p0_pair_maps", None)
    if cache is None:
        cache = {}
        setattr(module, "_so2_moe_fused_p0_pair_maps", cache)

    key = (int(m), str(device))
    cached = cache.get(key)
    if cached is not None:
        return cached

    in_base = []
    in_l = []
    for entry in module._in_entries_by_m[m]:
        dim = 2 * int(entry.l) + 1
        start = int(entry.slice_info.start)
        for idx in range(int(entry.mul)):
            in_base.append(start + idx * dim)
            in_l.append(int(entry.l))

    out_base = []
    out_l = []
    for entry in module._out_entries_by_m[m]:
        dim = 2 * int(entry.l) + 1
        start = int(entry.slice_info.start)
        for idx in range(int(entry.mul)):
            out_base.append(start + idx * dim)
            out_l.append(int(entry.l))

    offsets = [int(module.offsets[l]) for l in range(module.l_max + 1)]
    cached = (
        torch.tensor(in_base, dtype=torch.long, device=device).contiguous(),
        torch.tensor(in_l, dtype=torch.long, device=device).contiguous(),
        torch.tensor(out_base, dtype=torch.long, device=device).contiguous(),
        torch.tensor(out_l, dtype=torch.long, device=device).contiguous(),
        torch.tensor(offsets, dtype=torch.long, device=device).contiguous(),
    )
    cache[key] = cached
    return cached


def _wigner_block_from_flat(
    wigner: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    l: int,
    wigner_mode: int,
) -> torch.Tensor:
    dim = 2 * int(l) + 1
    if wigner_mode == 1:
        start = int(offsets[int(l)].item())
        return wigner[:, start:start + dim, start:start + dim]
    if wigner_mode == 2:
        start = int(compact_offsets[int(l)].item())
        return wigner[:, start:start + dim * dim].reshape(wigner.shape[0], dim, dim)
    raise RuntimeError(f"invalid wigner_mode={wigner_mode}")


def _index_groups_by_l(levels: torch.Tensor):
    groups = []
    if levels.numel() == 0:
        return groups
    for l in torch.unique(levels.detach().cpu(), sorted=True).tolist():
        idx = torch.nonzero(levels == int(l), as_tuple=False).reshape(-1)
        groups.append((int(l), idx))
    return groups


def _pair_segment_layout(
    graph_index: torch.Tensor,
    num_routes: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return cached pair-row permutation and cuBLAS segment ptr for [N, 2, C]."""

    graph_index = graph_index.reshape(-1).to(dtype=torch.long)
    assume_sorted = _flag("DPTB_SO2_MOE_FUSED_P0_ASSUME_SORTED")
    key = (
        graph_index.device.type,
        graph_index.device.index,
        int(graph_index.data_ptr()),
        int(graph_index.numel()),
        int(getattr(graph_index, "_version", 0)),
        int(num_routes),
        bool(assume_sorted),
    )
    cached = _PAIR_SEGMENT_LAYOUT_CACHE.get(key)
    if cached is not None:
        _PAIR_SEGMENT_LAYOUT_CACHE.move_to_end(key)
        return cached

    flat_graph = graph_index.reshape(-1, 1).expand(-1, 2).reshape(-1).contiguous()
    if assume_sorted or flat_graph.numel() <= 1:
        order = None
        unorder = None
        sorted_graph = flat_graph
    elif torch.all(flat_graph[1:] >= flat_graph[:-1]).item():
        order = None
        unorder = None
        sorted_graph = flat_graph
    else:
        order = torch.argsort(flat_graph, stable=True)
        sorted_graph = flat_graph.index_select(0, order)
        unorder = torch.empty_like(order)
        unorder.scatter_(0, order, torch.arange(order.numel(), device=order.device, dtype=order.dtype))

    counts = torch.bincount(sorted_graph, minlength=int(num_routes))
    ptr = torch.zeros(int(num_routes) + 1, dtype=torch.long, device=counts.device)
    ptr[1:] = torch.cumsum(counts, dim=0)
    cached = (order, unorder, sorted_graph, ptr.to(device="cpu", dtype=torch.long).contiguous())
    _PAIR_SEGMENT_LAYOUT_CACHE[key] = cached
    while len(_PAIR_SEGMENT_LAYOUT_CACHE) > _PAIR_SEGMENT_LAYOUT_CACHE_MAX:
        _PAIR_SEGMENT_LAYOUT_CACHE.popitem(last=False)
    return cached


def _pack_pair_torch(
    x: torch.Tensor,
    wigner: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    m: int,
    rotate_in: bool,
    wigner_mode: int,
) -> torch.Tensor:
    n = x.shape[0]
    cin = int(in_base.numel())
    pair = x.new_empty((n, 2, cin))
    for l, idx in _index_groups_by_l(in_l):
        bases = in_base.index_select(0, idx).detach().cpu().tolist()
        row0 = l - int(m)
        row1 = l + int(m)
        dim = 2 * l + 1
        x_l = torch.stack([x[:, int(base):int(base) + dim] for base in bases], dim=1)
        if rotate_in:
            block = _wigner_block_from_flat(wigner, offsets, compact_offsets, l, wigner_mode)
            pair_l = torch.einsum("ncd,ndp->npc", x_l, block[:, :, [row0, row1]])
        else:
            pair_l = x_l[:, :, [row0, row1]].transpose(1, 2).contiguous()
        pair.index_copy_(2, idx, pair_l)
    return pair


def _scatter_pair_grad_torch(
    grad_pair: torch.Tensor,
    wigner: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    in_dim: int,
    m: int,
    rotate_in: bool,
    wigner_mode: int,
) -> torch.Tensor:
    grad_x = grad_pair.new_zeros((grad_pair.shape[0], int(in_dim)))
    for l, idx in _index_groups_by_l(in_l):
        bases = in_base.index_select(0, idx).detach().cpu().tolist()
        row0 = l - int(m)
        row1 = l + int(m)
        dim = 2 * l + 1
        grad_l = grad_pair.index_select(2, idx)
        if rotate_in:
            block = _wigner_block_from_flat(wigner, offsets, compact_offsets, l, wigner_mode)
            grad_vals = torch.einsum("npc,ndp->ncd", grad_l, block[:, :, [row0, row1]])
        else:
            grad_vals = grad_pair.new_zeros((grad_pair.shape[0], len(bases), dim))
            grad_vals[:, :, row0] = grad_l[:, 0, :]
            grad_vals[:, :, row1] += grad_l[:, 1, :]
        for local, base in enumerate(bases):
            grad_x[:, int(base):int(base) + dim] += grad_vals[:, local, :]
    return grad_x


def _output_pair_grad_torch(
    grad_out: torch.Tensor,
    wigner: torch.Tensor,
    out_base: torch.Tensor,
    out_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    m: int,
    rotate_out: bool,
    wigner_mode: int,
) -> torch.Tensor:
    n = grad_out.shape[0]
    cout = int(out_base.numel())
    grad_pair = grad_out.new_empty((n, 2, cout))
    for l, idx in _index_groups_by_l(out_l):
        bases = out_base.index_select(0, idx).detach().cpu().tolist()
        row0 = l - int(m)
        row1 = l + int(m)
        dim = 2 * l + 1
        grad_l = torch.stack([grad_out[:, int(base):int(base) + dim] for base in bases], dim=1)
        if rotate_out:
            block = _wigner_block_from_flat(wigner, offsets, compact_offsets, l, wigner_mode)
            grad_pair_l = torch.einsum("ncd,ndp->npc", grad_l, block[:, :, [row0, row1]])
        else:
            grad_pair_l = grad_l[:, :, [row0, row1]].transpose(1, 2).contiguous()
        grad_pair.index_copy_(2, idx, grad_pair_l)
    return grad_pair


def _segmented_raw_linear_backward(
    x_pair: torch.Tensor,
    grad_raw: torch.Tensor,
    graph_index: torch.Tensor,
    mixed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_routes = int(mixed_weight.shape[0])
    cin = int(mixed_weight.shape[2])
    out2 = int(mixed_weight.shape[1])
    grad_x_pair = torch.empty_like(x_pair)
    grad_weight = torch.zeros_like(mixed_weight)
    flat_graph = graph_index.reshape(-1)
    for route in range(n_routes):
        idx = torch.nonzero(flat_graph == route, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        x_r = x_pair.index_select(0, idx).reshape(-1, cin)
        grad_r = grad_raw.index_select(0, idx).reshape(-1, out2)
        grad_weight[route] = grad_r.transpose(0, 1).matmul(x_r)
        grad_x_r = grad_r.matmul(mixed_weight[route]).reshape(idx.numel(), 2, cin)
        grad_x_pair.index_copy_(0, idx, grad_x_r)
    return grad_x_pair, grad_weight


def _cublas_segmented_raw_linear_backward(
    x_pair: torch.Tensor,
    grad_raw: torch.Tensor,
    graph_index: torch.Tensor,
    mixed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from dptb.nn.cublas_grouped_gemm import _load_extension as _load_cublas_grouped

    n_routes = int(mixed_weight.shape[0])
    cin = int(mixed_weight.shape[2])
    out2 = int(mixed_weight.shape[1])
    x_flat = x_pair.reshape(-1, cin).contiguous()
    grad_flat = grad_raw.reshape(-1, out2).contiguous()
    order, unorder, _sorted_graph, ptr_cpu = _pair_segment_layout(graph_index, n_routes)

    if order is not None:
        x_sorted = x_flat.index_select(0, order).contiguous()
        grad_sorted = grad_flat.index_select(0, order).contiguous()
    else:
        x_sorted = x_flat
        grad_sorted = grad_flat

    ext = _load_cublas_grouped()
    grad_x_sorted = ext.grouped_gemm_forward_fp32(
        grad_sorted,
        ptr_cpu,
        mixed_weight.transpose(1, 2).contiguous(),
        False,
    )
    if unorder is not None:
        grad_x_flat = grad_x_sorted.index_select(0, unorder)
    else:
        grad_x_flat = grad_x_sorted
    grad_weight = ext.grouped_gemm_backward_weight_fp32(
        grad_sorted,
        x_sorted,
        ptr_cpu,
        n_routes,
        False,
    )
    return grad_x_flat.reshape_as(x_pair), grad_weight


def _cutlass_segmented_raw_linear_backward(
    x_pair: torch.Tensor,
    grad_raw: torch.Tensor,
    graph_index: torch.Tensor,
    mixed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from dptb.nn.cutlass_grouped_gemm import grouped_gemm, grouped_gemm_backward_weight

    n_routes = int(mixed_weight.shape[0])
    cin = int(mixed_weight.shape[2])
    out2 = int(mixed_weight.shape[1])
    x_flat = x_pair.reshape(-1, cin).contiguous()
    grad_flat = grad_raw.reshape(-1, out2).contiguous()
    order, unorder, _sorted_graph, ptr_cpu = _pair_segment_layout(graph_index, n_routes)

    if order is not None:
        x_sorted = x_flat.index_select(0, order).contiguous()
        grad_sorted = grad_flat.index_select(0, order).contiguous()
    else:
        x_sorted = x_flat
        grad_sorted = grad_flat

    grad_x_sorted = grouped_gemm(
        grad_sorted,
        ptr_cpu,
        mixed_weight.transpose(1, 2).contiguous(),
    )
    if unorder is not None:
        grad_x_flat = grad_x_sorted.index_select(0, unorder)
    else:
        grad_x_flat = grad_x_sorted
    grad_weight = grouped_gemm_backward_weight(
        grad_sorted,
        x_sorted,
        ptr_cpu,
        n_routes,
    )
    return grad_x_flat.reshape_as(x_pair), grad_weight


def _segmented_raw_linear_forward(
    x_pair: torch.Tensor,
    graph_index: torch.Tensor,
    mixed_weight: torch.Tensor,
) -> torch.Tensor:
    n_routes = int(mixed_weight.shape[0])
    out2 = int(mixed_weight.shape[1])
    raw = x_pair.new_empty((x_pair.shape[0], 2, out2))
    flat_graph = graph_index.reshape(-1)
    for route in range(n_routes):
        idx = torch.nonzero(flat_graph == route, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        x_r = x_pair.index_select(0, idx).reshape(-1, x_pair.shape[-1])
        raw_r = x_r.matmul(mixed_weight[route].transpose(0, 1)).reshape(idx.numel(), 2, out2)
        raw.index_copy_(0, idx, raw_r)
    return raw


def _segmented_pair_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    wigner: torch.Tensor,
    graph_index: torch.Tensor,
    mixed_weight: torch.Tensor,
    radial: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    out_base: torch.Tensor,
    out_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    out_dim: int,
    m: int,
    rotate_in: bool,
    rotate_out: bool,
    radial_on_input: bool,
    wigner_mode: int,
    wigner_stride: int,
    linear_backend: str,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    del out_dim, wigner_stride
    cin = int(in_base.numel())
    cout = int(out_base.numel())
    has_radial = radial.numel() != 0
    x_pair_no_radial = _pack_pair_torch(
        x, wigner, in_base, in_l, offsets, compact_offsets, m, rotate_in, wigner_mode
    )
    if has_radial and radial_on_input:
        x_pair_eff = x_pair_no_radial * radial.unsqueeze(1)
    else:
        x_pair_eff = x_pair_no_radial

    grad_pair = _output_pair_grad_torch(
        grad_out, wigner, out_base, out_l, offsets, compact_offsets, m, rotate_out, wigner_mode
    )
    grad_radial = torch.zeros_like(radial) if has_radial else None
    if has_radial and not radial_on_input:
        raw = _segmented_raw_linear_forward(x_pair_eff, graph_index, mixed_weight)
        y0_pre = raw[:, 0, :cout] - raw[:, 1, cout:]
        y1_pre = raw[:, 1, :cout] + raw[:, 0, cout:]
        grad_radial = grad_pair[:, 0, :] * y0_pre + grad_pair[:, 1, :] * y1_pre
        grad_pair = grad_pair * radial.unsqueeze(1)

    grad_raw = grad_out.new_zeros((grad_out.shape[0], 2, 2 * cout))
    grad_raw[:, 0, :cout] = grad_pair[:, 0, :]
    grad_raw[:, 1, :cout] = grad_pair[:, 1, :]
    grad_raw[:, 0, cout:] = grad_pair[:, 1, :]
    grad_raw[:, 1, cout:] = -grad_pair[:, 0, :]

    if linear_backend == "cutlass_segmented":
        grad_x_pair_eff, grad_weight = _cutlass_segmented_raw_linear_backward(
            x_pair_eff, grad_raw, graph_index, mixed_weight
        )
    elif linear_backend == "cublas_segmented":
        grad_x_pair_eff, grad_weight = _cublas_segmented_raw_linear_backward(
            x_pair_eff, grad_raw, graph_index, mixed_weight
        )
    else:
        grad_x_pair_eff, grad_weight = _segmented_raw_linear_backward(
            x_pair_eff, grad_raw, graph_index, mixed_weight
        )
    if has_radial and radial_on_input:
        grad_radial = (grad_x_pair_eff * x_pair_no_radial).sum(dim=1)
        grad_x_pair = grad_x_pair_eff * radial.unsqueeze(1)
    else:
        grad_x_pair = grad_x_pair_eff

    grad_x = _scatter_pair_grad_torch(
        grad_x_pair, wigner, in_base, in_l, offsets, compact_offsets,
        x.shape[1], m, rotate_in, wigner_mode
    )
    return grad_x, grad_weight, grad_radial


class _FusedPairFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        wigner,
        graph_index,
        mixed_weight,
        radial,
        in_base,
        in_l,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        out_dim: int,
        m: int,
        rotate_in: bool,
        rotate_out: bool,
        radial_on_input: bool,
        wigner_mode: int,
        wigner_stride: int,
    ):
        out = _load_extension().fused_pair_forward_fp32(
            x,
            wigner,
            graph_index,
            mixed_weight,
            radial,
            in_base,
            in_l,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            int(out_dim),
            int(m),
            bool(rotate_in),
            bool(rotate_out),
            bool(radial_on_input),
            int(wigner_mode),
            int(wigner_stride),
        )
        ctx.save_for_backward(
            x,
            wigner,
            graph_index,
            mixed_weight,
            radial,
            in_base,
            in_l,
            out_base,
            out_l,
            offsets,
            compact_offsets,
        )
        ctx.meta = (
            int(out_dim),
            int(m),
            bool(rotate_in),
            bool(rotate_out),
            bool(radial_on_input),
            int(wigner_mode),
            int(wigner_stride),
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            x,
            wigner,
            graph_index,
            mixed_weight,
            radial,
            in_base,
            in_l,
            out_base,
            out_l,
            offsets,
            compact_offsets,
        ) = ctx.saved_tensors
        (
            out_dim,
            m,
            rotate_in,
            rotate_out,
            radial_on_input,
            wigner_mode,
            wigner_stride,
        ) = ctx.meta
        backward_mode = os.environ.get("DPTB_SO2_MOE_FUSED_P0_BACKWARD_MODE", "cublas_segmented")
        if backward_mode == "atomic":
            grad_x, grad_mixed_weight, grad_radial = _load_extension().fused_pair_backward_fp32(
                grad_out.contiguous(),
                x,
                wigner,
                graph_index,
                mixed_weight,
                radial,
                in_base,
                in_l,
                out_base,
                out_l,
                offsets,
                compact_offsets,
                int(out_dim),
                int(m),
                bool(rotate_in),
                bool(rotate_out),
                bool(radial_on_input),
                int(wigner_mode),
                int(wigner_stride),
            )
        else:
            grad_x, grad_mixed_weight, grad_radial = _segmented_pair_backward(
                grad_out.contiguous(),
                x,
                wigner,
                graph_index,
                mixed_weight,
                radial,
                in_base,
                in_l,
                out_base,
                out_l,
                offsets,
                compact_offsets,
                int(out_dim),
                int(m),
                bool(rotate_in),
                bool(rotate_out),
                bool(radial_on_input),
                int(wigner_mode),
                int(wigner_stride),
                backward_mode,
            )
        if radial.numel() == 0:
            grad_radial = None
        return (
            grad_x,
            None,
            None,
            grad_mixed_weight,
            grad_radial,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _fused_pair_contribution(
    module,
    m: int,
    x: torch.Tensor,
    wigner: torch.Tensor,
    compact_offsets: torch.Tensor,
    wigner_mode: int,
    wigner_stride: int,
    mole_globals: MOLEGlobals,
    radial_weight: Optional[torch.Tensor],
):
    fc = module.m_linear[m - 1].fc
    if not hasattr(fc, "_mix_expert_parameters"):
        _warn_once(
            "non_mole_linear_fallback",
            "streamed_m_major_fused_p0 currently fuses MOLELinear m>0 blocks; non-MoE SO2_m_Linear falls back.",
        )
        return None
    if getattr(fc, "bias_experts", None) is not None:
        _warn_once("pair_bias_fallback", "streamed_m_major_fused_p0 expects bias-free m>0 MoE linears; falling back.")
        return None

    mixed_weight, mixed_bias = fc._mix_expert_parameters(mole_globals)
    if mixed_bias is not None:
        _warn_once("mixed_bias_fallback", "streamed_m_major_fused_p0 does not handle m>0 bias; falling back.")
        return None

    graph_index = _mole_graph_index(mole_globals, x.shape[0], device=x.device)
    if graph_index.numel() != x.shape[0]:
        raise ValueError(
            f"MOLE graph_index has {graph_index.numel()} rows, but fused input has {x.shape[0]} rows."
        )

    in_base, in_l, out_base, out_l, offsets = _pair_maps(module, m, x.device)
    cin = int(in_base.numel())
    cout = int(out_base.numel())
    if mixed_weight.shape[-2:] != (2 * cout, cin):
        _warn_once(
            "weight_shape_fallback",
            "streamed_m_major_fused_p0 mixed weight shape does not match SO2 pair maps; falling back.",
        )
        return None

    if radial_weight is None:
        radial = x.new_empty((0,))
        radial_on_input = True
    else:
        radial = radial_weight.squeeze(1).contiguous()
        radial_on_input = bool(module.front)
        expected = cin if radial_on_input else cout
        if radial.shape != (x.shape[0], expected):
            _warn_once(
                "radial_shape_fallback",
                f"streamed_m_major_fused_p0 radial shape {tuple(radial.shape)} does not match expected {(x.shape[0], expected)}; falling back.",
            )
            return None

    return _FusedPairFunction.apply(
        x.contiguous(),
        wigner,
        graph_index.to(device=x.device, dtype=torch.long).contiguous(),
        mixed_weight.contiguous(),
        radial,
        in_base,
        in_l,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        int(module.irreps_out.dim),
        int(m),
        bool(module.rotate_in),
        bool(module.rotate_out),
        bool(radial_on_input),
        int(wigner_mode),
        int(wigner_stride),
    )


def try_forward_so2_moe_fused_p0(module, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
    """Return a trainable fused SO2/MoE P0 result or ``None`` to fall back.

    The fused CUDA op owns the m > 0 SO2 prologue/epilogue and has an explicit
    backward for x, mixed MoE weights, and radial weights. Wigner/R are treated
    as constants, which matches the Hamiltonian-only training target for this
    branch but is not valid for force or coordinate-gradient training.
    """

    if x.device.type != "cuda" or x.dtype != torch.float32:
        _warn_once("device_dtype_fallback", "streamed_m_major_fused_p0 requires CUDA fp32; falling back.")
        return None

    if torch.is_tensor(R) and R.requires_grad:
        _warn_once(
            "r_grad_ignored",
            "streamed_m_major_fused_p0 treats Wigner/R as constant and does not propagate coordinate gradients.",
        )

    if module.radial_emb and latents is None:
        raise ValueError("SO2 fused P0 path requires latents when radial_emb=True.")

    wigner_D_all = module._ensure_wigner_rotation(R, wigner_D_all)
    if _wigner_requires_grad(wigner_D_all):
        _warn_once(
            "wigner_grad_ignored",
            "streamed_m_major_fused_p0 treats Wigner as constant and does not propagate Wigner gradients.",
        )

    wigner_info = _wigner_tensor_and_mode(module, wigner_D_all, x)
    if wigner_info is None:
        return None
    wigner, compact_offsets, wigner_mode, wigner_stride = wigner_info

    weights = module.radial_emb(latents) if module.radial_emb else None
    out = torch.zeros((x.shape[0], module.irreps_out.dim), dtype=x.dtype, device=x.device)

    radial_m0 = weights[:, module.m_in_index[0]:module.m_in_index[1]].unsqueeze(1) if module.radial_emb else None
    inp0 = module._direct_rotate_pack_m(x, 0, wigner_D_all)
    if module.front and module.radial_emb:
        y0 = module.fc_m0(inp0 * radial_m0.squeeze(1), mole_globals)
    elif module.radial_emb:
        y0 = module.fc_m0(inp0, mole_globals) * radial_m0.squeeze(1)
    else:
        y0 = module.fc_m0(inp0, mole_globals)
    module._accumulate_m0_output(out, y0, wigner_D_all)

    for m in range(1, module.m_max + 1):
        radial_m = weights[:, module.m_in_index[m]:module.m_in_index[m + 1]].unsqueeze(1) if module.radial_emb else None
        contribution = _fused_pair_contribution(
            module,
            m,
            x,
            wigner,
            compact_offsets,
            wigner_mode,
            wigner_stride,
            mole_globals,
            radial_m,
        )
        if contribution is None:
            return None
        out.add_(contribution)

    if _flag("DPTB_SO2_MOE_FUSED_P0_LOG_ONCE"):
        _warn_once(
            "active_route",
            f"streamed_m_major_fused_p0 active: compact_or_dense_wigner_mode={wigner_mode}, m_max={module.m_max}.",
        )

    return out.contiguous(), wigner_D_all

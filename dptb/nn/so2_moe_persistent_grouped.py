"""Persistent grouped SO2/MoE fused P1.

This module is meant to be dropped on top of the DeePTB
``0521-cublas-grouped-moe`` branch.  It turns the forward-side SO2 MoE
work into a single persistent grouped CUDA schedule over ``(route, m)``
problems, and includes ``m=0`` as a first-class special case in the same
multi-m schedule.

P1 deliberately keeps strict safety gates:

* CUDA fp32 only.
* Wigner/R are treated as constants, matching the existing fused P0 path.
* Mixed bias is fused as an optional per-``m`` epilogue term.
* The forward CUDA op owns route/m scheduling, Wigner input prologue, raw
  linear dot, and Wigner output epilogue.  Its backward is trainable and reuses
  the existing segmented cuBLAS/CUDA SO2 projection kernels while the persistent
  backward is still experimental.

The default forward mainloop is warp-collective: a warp owns one row of a
route/m/output tile, lanes split the K dimension, and lane 0 runs the custom
SO2 epilogue. ``DPTB_SO2_MOE_PERSISTENT_P1_MAINLOOP=cute_tiled`` switches to a
CuTe-backed shared-memory tiled prototype: the tile mainloop reads raw
``x + compact Wigner + SO2 maps`` through a custom A-loader and keeps the SO2
output epilogue in-kernel. ``scalar`` keeps the older thread-per-output
prototype for debugging.
"""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

from dptb.nn.so2_moe_fused_p0 import _segmented_m0_backward, _segmented_pair_backward
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2WignerBlocks, _mole_graph_index

_EXT = None
_WARNED: set[str] = set()
_FALSE = {"", "0", "false", "False", "FALSE", "off", "OFF", "no", "No"}
_LAYOUT_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
_LAYOUT_CACHE_MAX = 32


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in _FALSE


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)


def _mainloop_kind(name: Optional[str] = None) -> int:
    mode = name or os.environ.get("DPTB_SO2_MOE_PERSISTENT_P1_MAINLOOP", "warp_collective")
    if mode in ("scalar", "thread", "thread_scalar"):
        return 0
    if mode in ("warp", "warp_collective", "collective"):
        return 1
    if mode in ("cute_tiled", "cutlass_cute_tiled", "tiled"):
        return 2
    if mode in ("cutlass_native", "cutlass_native_grouped", "native_cutlass", "cutlass_grouped_native"):
        return 3
    raise RuntimeError(f"unknown DPTB_SO2_MOE_PERSISTENT_P1_MAINLOOP={mode!r}")


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
            "DPTB_SO2_MOE_PERSISTENT_P1_BUILD_DIR",
            Path.home() / ".cache" / "dptb_so2_moe_persistent_grouped_p1",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    cflags = ["-O3"]
    cuda_flags = ["-O3", "--expt-relaxed-constexpr"]
    include_paths = []
    if _flag("DPTB_SO2_MOE_PERSISTENT_P1_LINEINFO"):
        cuda_flags.append("-lineinfo")
    cutlass_root = (
        os.environ.get("DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_ROOT")
        or os.environ.get("DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT")
        or os.environ.get("DPTB_CUTLASS_ROOT")
    )
    if cutlass_root:
        cutlass_root_path = Path(cutlass_root)
        include_paths.extend([
            str(cutlass_root_path / "include"),
            str(cutlass_root_path / "tools" / "util" / "include"),
        ])
        cflags.append("-DDPTB_SO2_MOE_PERSISTENT_P1_CUTE=1")
        cuda_flags.append("-DDPTB_SO2_MOE_PERSISTENT_P1_CUTE=1")
    _EXT = load(
        name="dptb_so2_moe_persistent_grouped_p1",
        sources=[
            str(here / "csrc" / "so2_moe_persistent_grouped.cpp"),
            str(here / "csrc" / "so2_moe_persistent_grouped_kernel.cu"),
        ],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=include_paths,
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=_flag("DPTB_SO2_MOE_PERSISTENT_P1_VERBOSE"),
    )
    return _EXT


def _empty_long(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.long, device=device)


def _wigner_requires_grad(wigner_D_all) -> bool:
    if torch.is_tensor(wigner_D_all):
        return bool(wigner_D_all.requires_grad)
    if isinstance(wigner_D_all, SO2WignerBlocks):
        return any(block.requires_grad for block in wigner_D_all.blocks)
    return False


def _wigner_tensor_and_mode(module, wigner_D_all, x: torch.Tensor):
    """Return ``(wigner, compact_offsets, mode, stride)`` or ``None``.

    This mirrors the fused P0 compact/dense Wigner contract.  Dense mode stores
    the full block diagonal tensor ``[E, D, D]``.  Compact mode stores per-l
    blocks flattened per edge and an offset table into that flat row.
    """
    if not (module.rotate_in or module.rotate_out) or module.l_max == 0:
        return x.new_empty((0,)), _empty_long(x.device), 0, 0

    if torch.is_tensor(wigner_D_all):
        if wigner_D_all.device != x.device or wigner_D_all.dtype != x.dtype or wigner_D_all.dim() != 3:
            _warn_once("bad_dense_wigner_fallback", "persistent_grouped_p1 requires dense CUDA fp32 Wigner [E,D,D].")
            return None
        if wigner_D_all.shape[0] != x.shape[0] or wigner_D_all.shape[1] != wigner_D_all.shape[2]:
            _warn_once("bad_dense_wigner_shape_fallback", "persistent_grouped_p1 dense Wigner shape is incompatible.")
            return None
        return wigner_D_all.contiguous(), _empty_long(x.device), 1, int(wigner_D_all.shape[1])

    if isinstance(wigner_D_all, SO2WignerBlocks):
        pieces = []
        compact_offsets = []
        cursor = 0
        for l in range(module.l_max + 1):
            if l >= len(wigner_D_all.blocks):
                _warn_once("missing_compact_wigner_fallback", "persistent_grouped_p1 compact Wigner blocks are incomplete.")
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
                _warn_once("bad_compact_wigner_fallback", "persistent_grouped_p1 compact Wigner block shape is incompatible.")
                return None
            compact_offsets.append(cursor)
            cursor += dim * dim
            pieces.append(block.reshape(x.shape[0], dim * dim))
        packed = torch.cat(pieces, dim=1).contiguous() if pieces else x.new_empty((x.shape[0], 0))
        offsets = torch.tensor(compact_offsets, dtype=torch.long, device=x.device).contiguous()
        return packed, offsets, 2, int(cursor)

    _warn_once("unknown_wigner_fallback", f"persistent_grouped_p1 received unsupported Wigner type {type(wigner_D_all)!r}; falling back.")
    return None


def _m_maps(module, m: int, device: torch.device):
    """Return map tensors for one SO2 m, including m=0.

    ``m=0`` is represented as a scalar problem: each logical input/output
    channel has one SO2 component at row ``l``.  ``m>0`` is represented as a
    complex pair problem: rows ``l-m`` and ``l+m``.
    """
    cache = getattr(module, "_so2_moe_persistent_grouped_p1_maps", None)
    if cache is None:
        cache = {}
        setattr(module, "_so2_moe_persistent_grouped_p1_maps", cache)
    key = (int(m), str(device))
    cached = cache.get(key)
    if cached is not None:
        return cached

    in_base = []
    in_l = []
    for entry in module._in_entries_by_m[int(m)]:
        dim = 2 * int(entry.l) + 1
        start = int(entry.slice_info.start)
        for idx in range(int(entry.mul)):
            in_base.append(start + idx * dim)
            in_l.append(int(entry.l))

    out_base = []
    out_l = []
    for entry in module._out_entries_by_m[int(m)]:
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


def _pack_all_m_metadata(module, device: torch.device, *, include_m0: bool):
    start_m = 0 if include_m0 else 1
    m_values = list(range(start_m, int(module.m_max) + 1))
    in_ptr = [0]
    out_ptr = [0]
    in_base_parts = []
    in_l_parts = []
    out_base_parts = []
    out_l_parts = []
    offsets = None
    for m in m_values:
        in_base, in_l, out_base, out_l, offsets_m = _m_maps(module, m, device)
        in_base_parts.append(in_base)
        in_l_parts.append(in_l)
        out_base_parts.append(out_base)
        out_l_parts.append(out_l)
        in_ptr.append(in_ptr[-1] + int(in_base.numel()))
        out_ptr.append(out_ptr[-1] + int(out_base.numel()))
        offsets = offsets_m

    cat = lambda xs: torch.cat(xs, dim=0).contiguous() if xs else _empty_long(device)
    return (
        torch.tensor(m_values, dtype=torch.long, device=device).contiguous(),
        torch.tensor(in_ptr, dtype=torch.long, device=device).contiguous(),
        cat(in_base_parts),
        cat(in_l_parts),
        torch.tensor(out_ptr, dtype=torch.long, device=device).contiguous(),
        cat(out_base_parts),
        cat(out_l_parts),
        offsets if offsets is not None else _empty_long(device),
    )


def _mixed_parameters_for_m(module, m: int, mole_globals: MOLEGlobals):
    if int(m) == 0:
        fc = module.fc_m0
    else:
        fc = module.m_linear[int(m) - 1].fc
    if not hasattr(fc, "_mix_expert_parameters"):
        _warn_once("non_mole_linear_fallback", "persistent_grouped_p1 currently fuses MOLELinear SO2 blocks only; falling back.")
        return None
    mixed_weight, mixed_bias = fc._mix_expert_parameters(mole_globals)
    return mixed_weight.contiguous(), None if mixed_bias is None else mixed_bias.contiguous()


def _pack_all_m_weights(
    module,
    mole_globals: MOLEGlobals,
    m_values: torch.Tensor,
    out_ptr: torch.Tensor,
    in_ptr: torch.Tensor,
):
    m_list = [int(v) for v in m_values.detach().cpu().tolist()]
    weight_offsets = []
    bias_offsets = []
    weight_parts = []
    bias_parts = []
    weight_cursor = 0
    bias_cursor = 0
    n_routes: Optional[int] = None
    for m_idx, m in enumerate(m_list):
        params = _mixed_parameters_for_m(module, m, mole_globals)
        if params is None:
            return None
        w, b = params
        if m > 0 and b is not None:
            _warn_once(
                "pair_bias_fallback",
                "persistent_grouped_p1 keeps m>0 bias unsupported to match fused P0 backward; falling back.",
            )
            return None
        if w.dtype != torch.float32 or w.device.type != "cuda":
            _warn_once("weight_dtype_fallback", "persistent_grouped_p1 requires CUDA fp32 mixed weights; falling back.")
            return None
        if b is not None and (b.dtype != torch.float32 or b.device.type != "cuda"):
            _warn_once("bias_dtype_fallback", "persistent_grouped_p1 requires CUDA fp32 mixed bias; falling back.")
            return None
        if n_routes is None:
            n_routes = int(w.shape[0])
        elif int(w.shape[0]) != n_routes:
            _warn_once("route_count_fallback", "persistent_grouped_p1 found inconsistent route counts across m blocks; falling back.")
            return None
        cin = int((in_ptr[m_idx + 1] - in_ptr[m_idx]).item())
        cout = int((out_ptr[m_idx + 1] - out_ptr[m_idx]).item())
        expected_rows = cout if m == 0 else 2 * cout
        if tuple(w.shape[-2:]) != (expected_rows, cin):
            _warn_once(
                "weight_shape_fallback",
                f"persistent_grouped_p1 m={m} mixed weight shape {tuple(w.shape)} does not match expected (*,{expected_rows},{cin}); falling back.",
            )
            return None
        if b is not None and tuple(b.shape[-1:]) != (expected_rows,):
            _warn_once(
                "bias_shape_fallback",
                f"persistent_grouped_p1 m={m} mixed bias shape {tuple(b.shape)} does not match expected (*,{expected_rows}); falling back.",
            )
            return None
        weight_offsets.append(weight_cursor)
        flat_w = w.reshape(-1).contiguous()
        weight_parts.append(flat_w)
        weight_cursor += int(flat_w.numel())
        if b is None:
            bias_offsets.append(-1)
        else:
            bias_offsets.append(bias_cursor)
            flat_b = b.reshape(-1).contiguous()
            bias_parts.append(flat_b)
            bias_cursor += int(flat_b.numel())
    weight_flat = torch.cat(weight_parts, dim=0).contiguous() if weight_parts else torch.empty((0,), dtype=torch.float32, device=out_ptr.device)
    bias_flat = torch.cat(bias_parts, dim=0).contiguous() if bias_parts else torch.empty((0,), dtype=torch.float32, device=out_ptr.device)
    return (
        weight_flat,
        torch.tensor(weight_offsets, dtype=torch.long, device=out_ptr.device).contiguous(),
        bias_flat,
        torch.tensor(bias_offsets, dtype=torch.long, device=out_ptr.device).contiguous(),
        int(n_routes or 0),
    )


def _prepare_route_layout(
    graph_index: torch.Tensor,
    n_routes: int,
    n_m: int,
    block_m: int,
    block_n: int,
    out_ptr: torch.Tensor,
    *,
    raw_pair_tiles: bool = False,
):
    graph_index = graph_index.reshape(-1).to(dtype=torch.long)
    assume_sorted = _flag("DPTB_SO2_MOE_PERSISTENT_P1_ASSUME_SORTED")
    nosync_layout = _flag("DPTB_SO2_MOE_PERSISTENT_P1_NOSYNC_LAYOUT", "0")
    if not nosync_layout:
        key = (
            str(graph_index.device),
            int(graph_index.data_ptr()),
            int(graph_index.numel()),
            int(getattr(graph_index, "_version", 0)),
            int(n_routes),
            int(n_m),
            int(block_m),
            int(block_n),
            tuple(int(v) for v in out_ptr.detach().cpu().tolist()),
            bool(raw_pair_tiles),
            bool(assume_sorted),
        )
        cached = _LAYOUT_CACHE.get(key)
        if cached is not None:
            _LAYOUT_CACHE.move_to_end(key)
            return cached

        if graph_index.numel() == 0:
            edge_order = torch.empty((0,), dtype=torch.long, device=graph_index.device)
            route_ptr = torch.zeros((n_routes + 1,), dtype=torch.long, device=graph_index.device)
            prefix = torch.zeros((n_routes * n_m + 1,), dtype=torch.long, device=graph_index.device)
        else:
            if assume_sorted or torch.all(graph_index[1:] >= graph_index[:-1]).item():
                edge_order = torch.arange(graph_index.numel(), dtype=torch.long, device=graph_index.device)
                sorted_graph = graph_index
            else:
                edge_order = torch.argsort(graph_index, stable=True).contiguous()
                sorted_graph = graph_index.index_select(0, edge_order).contiguous()
            counts = torch.bincount(sorted_graph, minlength=int(n_routes))
            route_ptr = torch.zeros((n_routes + 1,), dtype=torch.long, device=graph_index.device)
            route_ptr[1:] = torch.cumsum(counts, dim=0)

            counts_cpu = counts.detach().cpu().tolist()
            out_ptr_cpu = [int(v) for v in out_ptr.detach().cpu().tolist()]
            pref = [0]
            for r in range(int(n_routes)):
                rows = int(counts_cpu[r])
                for m_idx in range(int(n_m)):
                    cout = out_ptr_cpu[m_idx + 1] - out_ptr_cpu[m_idx]
                    col_extent = 2 * cout if raw_pair_tiles else cout
                    row_tiles = (rows + int(block_m) - 1) // int(block_m)
                    col_tiles = (col_extent + int(block_n) - 1) // int(block_n)
                    pref.append(pref[-1] + row_tiles * col_tiles)
            prefix = torch.tensor(pref, dtype=torch.long, device=graph_index.device).contiguous()

        cached = (edge_order.contiguous(), route_ptr.contiguous(), prefix.contiguous())
        _LAYOUT_CACHE[key] = cached
        while len(_LAYOUT_CACHE) > _LAYOUT_CACHE_MAX:
            _LAYOUT_CACHE.popitem(last=False)
        return cached

    key = (
        str(graph_index.device),
        int(graph_index.data_ptr()),
        int(graph_index.numel()),
        int(getattr(graph_index, "_version", 0)),
        int(n_routes),
        int(n_m),
        int(block_m),
        int(block_n),
        int(out_ptr.data_ptr()),
        int(out_ptr.numel()),
        int(getattr(out_ptr, "_version", 0)),
        bool(raw_pair_tiles),
        bool(assume_sorted),
    )
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None:
        _LAYOUT_CACHE.move_to_end(key)
        return cached

    if graph_index.numel() == 0:
        edge_order = torch.empty((0,), dtype=torch.long, device=graph_index.device)
        route_ptr = torch.zeros((n_routes + 1,), dtype=torch.long, device=graph_index.device)
        prefix = torch.zeros((n_routes * n_m + 1,), dtype=torch.long, device=graph_index.device)
    else:
        if assume_sorted:
            edge_order = torch.arange(graph_index.numel(), dtype=torch.long, device=graph_index.device)
            sorted_graph = graph_index
        else:
            edge_order = torch.argsort(graph_index, stable=True).contiguous()
            sorted_graph = graph_index.index_select(0, edge_order).contiguous()
        counts = torch.bincount(sorted_graph, minlength=int(n_routes))
        route_ptr = torch.empty((n_routes + 1,), dtype=torch.long, device=graph_index.device)
        route_ptr[0] = 0
        route_ptr[1:] = torch.cumsum(counts, dim=0)

        widths = (out_ptr[1:] - out_ptr[:-1]).to(dtype=torch.long)
        if raw_pair_tiles:
            widths = widths * 2
        row_tiles = torch.div(counts + int(block_m) - 1, int(block_m), rounding_mode="floor")
        col_tiles = torch.div(widths + int(block_n) - 1, int(block_n), rounding_mode="floor")
        problem_tiles = (row_tiles[:, None] * col_tiles[None, :]).reshape(-1)
        prefix = torch.empty((problem_tiles.numel() + 1,), dtype=torch.long, device=graph_index.device)
        prefix[0] = 0
        prefix[1:] = torch.cumsum(problem_tiles, dim=0)

    cached = (edge_order.contiguous(), route_ptr.contiguous(), prefix.contiguous())
    _LAYOUT_CACHE[key] = cached
    while len(_LAYOUT_CACHE) > _LAYOUT_CACHE_MAX:
        _LAYOUT_CACHE.popitem(last=False)
    return cached


class _PersistentGroupedP1Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        wigner,
        edge_order,
        route_ptr,
        problem_tile_prefix,
        graph_index,
        weight_flat,
        weight_offsets,
        bias_flat,
        bias_offsets,
        m_values,
        in_ptr,
        in_base,
        in_l,
        out_ptr,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        radial_all,
        m_in_index,
        out_dim: int,
        n_routes: int,
        rotate_in: bool,
        rotate_out: bool,
        radial_on_input: bool,
        wigner_mode: int,
        wigner_stride: int,
        mainloop_kind: int,
        block_m: int,
        block_n: int,
        active_blocks: int,
    ):
        ext = _load_extension()
        if int(mainloop_kind) == 3:
            forward = ext.persistent_grouped_forward_cutlass_native_fp32
        elif int(mainloop_kind) == 2:
            forward = ext.persistent_grouped_forward_cute_tiled_fp32
        elif int(mainloop_kind) == 1:
            forward = ext.persistent_grouped_forward_warp_fp32
        elif int(mainloop_kind) == 0:
            forward = ext.persistent_grouped_forward_fp32
        else:
            raise RuntimeError(f"unknown persistent grouped mainloop kind={mainloop_kind!r}")
        out = forward(
            x,
            wigner,
            edge_order,
            route_ptr,
            problem_tile_prefix,
            weight_flat,
            weight_offsets,
            bias_flat,
            bias_offsets,
            m_values,
            in_ptr,
            in_base,
            in_l,
            out_ptr,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            radial_all,
            m_in_index,
            int(out_dim),
            int(n_routes),
            bool(rotate_in),
            bool(rotate_out),
            bool(radial_on_input),
            int(wigner_mode),
            int(wigner_stride),
            int(block_m),
            int(block_n),
            int(active_blocks),
        )
        ctx.save_for_backward(
            x,
            wigner,
            graph_index,
            weight_flat,
            weight_offsets,
            bias_flat,
            bias_offsets,
            m_values,
            in_ptr,
            in_base,
            in_l,
            out_ptr,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            radial_all,
            m_in_index,
        )
        ctx.meta = (
            int(out_dim),
            int(n_routes),
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
            weight_flat,
            weight_offsets,
            bias_flat,
            bias_offsets,
            m_values,
            in_ptr,
            in_base,
            in_l,
            out_ptr,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            radial_all,
            m_in_index,
        ) = ctx.saved_tensors
        (
            out_dim,
            n_routes,
            rotate_in,
            rotate_out,
            radial_on_input,
            wigner_mode,
            wigner_stride,
        ) = ctx.meta

        backward_mode = os.environ.get(
            "DPTB_SO2_MOE_PERSISTENT_P1_BACKWARD_MODE",
            os.environ.get("DPTB_SO2_MOE_FUSED_P0_BACKWARD_MODE", "cuda_cublas_segmented"),
        )
        grad_x = torch.zeros_like(x)
        grad_weight_flat = torch.zeros_like(weight_flat)
        grad_bias_flat = torch.zeros_like(bias_flat) if bias_flat.numel() else None
        grad_radial_all = torch.zeros_like(radial_all) if radial_all.numel() else None

        for m_idx in range(int(m_values.numel())):
            m = int(m_values[m_idx].item())
            in_begin = int(in_ptr[m_idx].item())
            in_end = int(in_ptr[m_idx + 1].item())
            out_begin = int(out_ptr[m_idx].item())
            out_end = int(out_ptr[m_idx + 1].item())
            cin = in_end - in_begin
            cout = out_end - out_begin
            if cin <= 0 or cout <= 0:
                continue

            row_count = cout if m == 0 else 2 * cout
            w_off = int(weight_offsets[m_idx].item())
            w_num = int(n_routes) * row_count * cin
            mixed_weight = weight_flat.narrow(0, w_off, w_num).view(int(n_routes), row_count, cin)

            b_off = int(bias_offsets[m_idx].item())
            if b_off >= 0:
                b_num = int(n_routes) * row_count
                mixed_bias = bias_flat.narrow(0, b_off, b_num).view(int(n_routes), row_count)
            else:
                mixed_bias = bias_flat.new_empty((0,))

            in_base_m = in_base.narrow(0, in_begin, cin)
            in_l_m = in_l.narrow(0, in_begin, cin)
            out_base_m = out_base.narrow(0, out_begin, cout)
            out_l_m = out_l.narrow(0, out_begin, cout)

            if radial_all.numel():
                radial_begin = int(m_in_index[m].item())
                radial_end = int(m_in_index[m + 1].item())
                radial_m = radial_all[:, radial_begin:radial_end].contiguous()
            else:
                radial_begin = 0
                radial_end = 0
                radial_m = radial_all

            if m == 0:
                grad_x_m, grad_weight_m, grad_bias_m, grad_radial_m = _segmented_m0_backward(
                    grad_out.contiguous(),
                    x,
                    wigner,
                    graph_index,
                    mixed_weight,
                    mixed_bias,
                    radial_m,
                    in_base_m,
                    in_l_m,
                    out_base_m,
                    out_l_m,
                    offsets,
                    compact_offsets,
                    int(out_dim),
                    bool(rotate_in),
                    bool(rotate_out),
                    bool(radial_on_input),
                    int(wigner_mode),
                    int(wigner_stride),
                    backward_mode,
                )
                if grad_bias_flat is not None and grad_bias_m is not None:
                    grad_bias_flat.narrow(0, b_off, grad_bias_m.numel()).copy_(grad_bias_m.reshape(-1))
            else:
                grad_x_m, grad_weight_m, grad_radial_m = _segmented_pair_backward(
                    grad_out.contiguous(),
                    x,
                    wigner,
                    graph_index,
                    mixed_weight,
                    radial_m,
                    in_base_m,
                    in_l_m,
                    out_base_m,
                    out_l_m,
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

            grad_x.add_(grad_x_m)
            grad_weight_flat.narrow(0, w_off, grad_weight_m.numel()).copy_(grad_weight_m.reshape(-1))
            if grad_radial_all is not None and grad_radial_m is not None:
                grad_radial_all[:, radial_begin:radial_end].add_(grad_radial_m)

        return (
            grad_x,            # x
            None,              # wigner
            None,              # edge_order
            None,              # route_ptr
            None,              # problem_tile_prefix
            None,              # graph_index
            grad_weight_flat,   # weight_flat
            None,              # weight_offsets
            grad_bias_flat,     # bias_flat
            None,              # bias_offsets
            None,              # m_values
            None,              # in_ptr
            None,              # in_base
            None,              # in_l
            None,              # out_ptr
            None,              # out_base
            None,              # out_l
            None,              # offsets
            None,              # compact_offsets
            grad_radial_all,   # radial_all
            None,              # m_in_index
            None,              # out_dim
            None,              # n_routes
            None,              # rotate_in
            None,              # rotate_out
            None,              # radial_on_input
            None,              # wigner_mode
            None,              # wigner_stride
            None,              # mainloop_kind
            None,              # block_m
            None,              # block_n
            None,              # active_blocks
        )


def try_forward_so2_moe_persistent_grouped_p1(
    module,
    x: torch.Tensor,
    R,
    mole_globals: MOLEGlobals,
    latents=None,
    wigner_D_all=None,
    *,
    include_m0_override: Optional[bool] = None,
    mainloop_override: Optional[str] = None,
):
    """Return a fused persistent grouped SO2/MoE forward result or ``None``.

    The returned value follows the existing SO2 fast-path convention:
    ``(out, wigner_D_all)``.  Callers should fall back to the stable path when
    this function returns ``None``.
    """
    if x.device.type != "cuda" or x.dtype != torch.float32:
        _warn_once("device_dtype_fallback", "persistent_grouped_p1 requires CUDA fp32; falling back.")
        return None
    if module.radial_emb and latents is None:
        raise ValueError("persistent_grouped_p1 requires latents when radial_emb=True.")
    if torch.is_tensor(R) and R.requires_grad:
        _warn_once("r_grad_ignored", "persistent_grouped_p1 treats Wigner/R as constant and does not propagate coordinate gradients.")
    wigner_D_all = module._ensure_wigner_rotation(R, wigner_D_all)
    if _wigner_requires_grad(wigner_D_all):
        _warn_once("wigner_grad_ignored", "persistent_grouped_p1 treats Wigner as constant and does not propagate Wigner gradients.")

    wigner_info = _wigner_tensor_and_mode(module, wigner_D_all, x)
    if wigner_info is None:
        return None
    wigner, compact_offsets, wigner_mode, wigner_stride = wigner_info

    include_m0 = (
        bool(include_m0_override)
        if include_m0_override is not None
        else _flag("DPTB_SO2_MOE_PERSISTENT_P1_INCLUDE_M0", "1")
    )
    mainloop_name = mainloop_override or os.environ.get("DPTB_SO2_MOE_PERSISTENT_P1_MAINLOOP", "warp_collective")
    mainloop_kind = _mainloop_kind(mainloop_name)
    if (
        int(mainloop_kind) == 3
        and torch.is_grad_enabled()
        and _flag("DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_NATIVE_GUARD", "0")
        and not _flag("DPTB_SO2_MOE_PERSISTENT_P1_FORCE_CUTLASS_NATIVE")
    ):
        _warn_once(
            "cutlass_native_guarded",
            "persistent_grouped_p1 production guard skipped cutlass_native under grad-enabled execution; "
            "set DPTB_SO2_MOE_PERSISTENT_P1_FORCE_CUTLASS_NATIVE=1 to force it.",
        )
        return None
    if int(mainloop_kind) == 3 and include_m0:
        m0_policy = os.environ.get("DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_M0_POLICY", "fallback")
        if m0_policy == "warp_all":
            _warn_once(
                "cutlass_native_m0_warp_all",
                "persistent_grouped_p1 cutlass_native requested with m0; using all-m warp_collective policy.",
            )
            mainloop_name = "warp_collective"
            mainloop_kind = _mainloop_kind(mainloop_name)
        elif m0_policy in ("fallback", "split"):
            _warn_once(
                "cutlass_native_m0_fallback",
                "persistent_grouped_p1 cutlass_native currently fuses m>0 and leaves m=0 on the existing fallback.",
            )
            include_m0 = False
        else:
            _warn_once(
                "cutlass_native_m0_policy_fallback",
                f"unknown DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_M0_POLICY={m0_policy!r}; falling back.",
            )
            return None

    (
        m_values,
        in_ptr,
        in_base,
        in_l,
        out_ptr,
        out_base,
        out_l,
        offsets,
    ) = _pack_all_m_metadata(module, x.device, include_m0=include_m0)
    if int(m_values.numel()) == 0:
        return None

    packed_weights = _pack_all_m_weights(module, mole_globals, m_values, out_ptr, in_ptr)
    if packed_weights is None:
        return None
    weight_flat, weight_offsets, bias_flat, bias_offsets, n_routes = packed_weights

    graph_index = _mole_graph_index(mole_globals, x.shape[0], device=x.device)
    if graph_index.numel() != x.shape[0]:
        raise ValueError(f"MOLE graph_index has {graph_index.numel()} rows, but fused input has {x.shape[0]} rows.")
    validate_route_ids_default = "0" if _flag("DPTB_SO2_MOE_PERSISTENT_P1_NOSYNC_LAYOUT", "0") else "1"
    if (
        _flag("DPTB_SO2_MOE_PERSISTENT_P1_VALIDATE_ROUTE_IDS", validate_route_ids_default)
        and graph_index.numel()
        and (torch.any(graph_index < 0).item() or torch.any(graph_index >= int(n_routes)).item())
    ):
        _warn_once("route_id_fallback", "persistent_grouped_p1 graph_index contains a route id outside mixed weight route range; falling back.")
        return None

    if int(mainloop_kind) == 3:
        block_m = 64
        block_n = 32
    else:
        block_m = max(1, _int_env("DPTB_SO2_MOE_PERSISTENT_P1_BLOCK_M", 8))
        block_n = max(1, _int_env("DPTB_SO2_MOE_PERSISTENT_P1_BLOCK_N", 8))
    active_blocks = _int_env("DPTB_SO2_MOE_PERSISTENT_P1_ACTIVE_BLOCKS", 0)

    edge_order, route_ptr, problem_tile_prefix = _prepare_route_layout(
        graph_index,
        n_routes,
        int(m_values.numel()),
        block_m,
        block_n,
        out_ptr,
        raw_pair_tiles=(int(mainloop_kind) == 3),
    )

    radial_all = module.radial_emb(latents).contiguous() if module.radial_emb else x.new_empty((0,))
    if module.radial_emb:
        m_in_index = torch.as_tensor(module.m_in_index, dtype=torch.long, device=x.device).contiguous()
        if m_in_index.numel() < int(module.m_max) + 2:
            _warn_once("radial_index_fallback", "persistent_grouped_p1 radial m_in_index length is incompatible; falling back.")
            return None
    else:
        m_in_index = _empty_long(x.device)

    out = _PersistentGroupedP1Function.apply(
        x.contiguous(),
        wigner,
        edge_order,
        route_ptr,
        problem_tile_prefix,
        graph_index,
        weight_flat,
        weight_offsets,
        bias_flat,
        bias_offsets,
        m_values,
        in_ptr,
        in_base,
        in_l,
        out_ptr,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        radial_all,
        m_in_index,
        int(module.irreps_out.dim),
        int(n_routes),
        bool(module.rotate_in),
        bool(module.rotate_out),
        bool(module.front),
        int(wigner_mode),
        int(wigner_stride),
        int(mainloop_kind),
        int(block_m),
        int(block_n),
        int(active_blocks),
    )

    if not include_m0:
        radial_m0 = radial_all[:, module.m_in_index[0]:module.m_in_index[1]].unsqueeze(1) if module.radial_emb else None
        inp0 = module._direct_rotate_pack_m(x, 0, wigner_D_all)
        if module.front and module.radial_emb:
            y0 = module.fc_m0(inp0 * radial_m0.squeeze(1), mole_globals)
        elif module.radial_emb:
            y0 = module.fc_m0(inp0, mole_globals) * radial_m0.squeeze(1)
        else:
            y0 = module.fc_m0(inp0, mole_globals)
        m0_out = torch.zeros_like(out)
        module._accumulate_m0_output(m0_out, y0, wigner_D_all)
        out = out + m0_out

    if _flag("DPTB_SO2_MOE_PERSISTENT_P1_LOG_ONCE"):
        _warn_once(
            "active_route",
            f"persistent_grouped_p1 active: routes={n_routes}, m_values={tuple(int(v) for v in m_values.detach().cpu().tolist())}, "
            f"tiles={int(problem_tile_prefix[-1].item())}, block_m={block_m}, block_n={block_n}, "
            f"wigner_mode={wigner_mode}, include_m0={include_m0}, "
            f"mainloop={mainloop_name}.",
        )
    return out.contiguous(), wigner_D_all

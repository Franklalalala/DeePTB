from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

from dptb.nn.tensor_product_moe_v3 import (
    MOLEGlobals,
    SO2WignerBlocks,
    _mole_graph_index,
)


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

    return _repeated_segment_layout(graph_index, num_routes, repeat=2)


def _row_segment_layout(
    graph_index: torch.Tensor,
    num_routes: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return cached row permutation and cuBLAS segment ptr for [N, C]."""

    return _repeated_segment_layout(graph_index, num_routes, repeat=1)


def _repeated_segment_layout(
    graph_index: torch.Tensor,
    num_routes: int,
    *,
    repeat: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    graph_index = graph_index.reshape(-1).to(dtype=torch.long)
    assume_sorted = _flag("DPTB_SO2_MOE_FUSED_P0_ASSUME_SORTED")
    key = (
        int(repeat),
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

    if repeat == 1:
        flat_graph = graph_index.contiguous()
    else:
        flat_graph = graph_index.reshape(-1, 1).expand(-1, int(repeat)).reshape(-1).contiguous()
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


def _pack_pair_cuda(
    x: torch.Tensor,
    wigner: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    m: int,
    rotate_in: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().pack_pair_fp32(
        x.contiguous(),
        wigner,
        in_base,
        in_l,
        offsets,
        compact_offsets,
        int(m),
        bool(rotate_in),
        int(wigner_mode),
        int(wigner_stride),
    )

def _pack_pairs_multi_cuda(
    x: torch.Tensor,
    wigner: torch.Tensor,
    in_bases: list[torch.Tensor],
    in_ls: list[torch.Tensor],
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    cin_prefix: torch.Tensor,
    m_values: torch.Tensor,
    rotate_in: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().pack_pairs_multi_fp32(
        x.contiguous(),
        wigner,
        in_bases,
        in_ls,
        offsets,
        compact_offsets,
        cin_prefix,
        m_values,
        bool(rotate_in),
        int(wigner_mode),
        int(wigner_stride),
    )


def _pack_m0_torch(
    x: torch.Tensor,
    wigner: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    rotate_in: bool,
    wigner_mode: int,
) -> torch.Tensor:
    n = x.shape[0]
    cin = int(in_base.numel())
    packed = x.new_empty((n, cin))
    for l, idx in _index_groups_by_l(in_l):
        bases = in_base.index_select(0, idx).detach().cpu().tolist()
        dim = 2 * l + 1
        x_l = torch.stack([x[:, int(base):int(base) + dim] for base in bases], dim=1)
        if rotate_in:
            block = _wigner_block_from_flat(wigner, offsets, compact_offsets, l, wigner_mode)
            packed_l = torch.einsum("ncd,nd->nc", x_l, block[:, :, l])
        else:
            packed_l = x_l[:, :, l]
        packed.index_copy_(1, idx, packed_l)
    return packed


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


def _scatter_m0_grad_torch(
    grad_m0: torch.Tensor,
    wigner: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    in_dim: int,
    rotate_in: bool,
    wigner_mode: int,
) -> torch.Tensor:
    grad_x = grad_m0.new_zeros((grad_m0.shape[0], int(in_dim)))
    for l, idx in _index_groups_by_l(in_l):
        bases = in_base.index_select(0, idx).detach().cpu().tolist()
        dim = 2 * l + 1
        grad_l = grad_m0.index_select(1, idx)
        if rotate_in:
            block = _wigner_block_from_flat(wigner, offsets, compact_offsets, l, wigner_mode)
            grad_vals = grad_l.unsqueeze(-1) * block[:, :, l].unsqueeze(1)
        else:
            grad_vals = grad_m0.new_zeros((grad_m0.shape[0], len(bases), dim))
            grad_vals[:, :, l] = grad_l
        for local, base in enumerate(bases):
            grad_x[:, int(base):int(base) + dim] += grad_vals[:, local, :]
    return grad_x


def _scatter_pair_grad_cuda(
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
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().scatter_pair_grad_fp32(
        grad_pair.contiguous(),
        wigner,
        in_base,
        in_l,
        offsets,
        compact_offsets,
        int(in_dim),
        int(m),
        bool(rotate_in),
        int(wigner_mode),
        int(wigner_stride),
    )


def _scatter_pairs_multi_grad_cuda(
    grad_packed: torch.Tensor,
    wigner: torch.Tensor,
    in_base_all: torch.Tensor,
    in_l_all: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    cin_prefix: torch.Tensor,
    m_values: torch.Tensor,
    in_dim: int,
    rotate_in: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().scatter_pairs_multi_grad_fp32(
        grad_packed.contiguous(),
        wigner,
        in_base_all,
        in_l_all,
        offsets,
        compact_offsets,
        cin_prefix,
        m_values,
        int(in_dim),
        bool(rotate_in),
        int(wigner_mode),
        int(wigner_stride),
    )


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


def _scatter_pair_forward_torch(
    pair_out: torch.Tensor,
    wigner: torch.Tensor,
    out_base: torch.Tensor,
    out_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    out_dim: int,
    m: int,
    rotate_out: bool,
    wigner_mode: int,
) -> torch.Tensor:
    n = pair_out.shape[0]
    out = pair_out.new_zeros((n, int(out_dim)))
    for l, idx in _index_groups_by_l(out_l):
        bases = out_base.index_select(0, idx).detach().cpu().tolist()
        row0 = l - int(m)
        row1 = l + int(m)
        dim = 2 * l + 1
        pair_l = pair_out.index_select(2, idx)
        if rotate_out:
            block = _wigner_block_from_flat(wigner, offsets, compact_offsets, l, wigner_mode)
            vals = torch.einsum("npc,ndp->ncd", pair_l, block[:, :, [row0, row1]])
        else:
            vals = pair_out.new_zeros((n, len(bases), dim))
            vals[:, :, row0] = pair_l[:, 0, :]
            vals[:, :, row1] += pair_l[:, 1, :]
        for local, base in enumerate(bases):
            out[:, int(base):int(base) + dim] += vals[:, local, :]
    return out


def _output_m0_grad_torch(
    grad_out: torch.Tensor,
    wigner: torch.Tensor,
    out_base: torch.Tensor,
    out_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    rotate_out: bool,
    wigner_mode: int,
) -> torch.Tensor:
    n = grad_out.shape[0]
    cout = int(out_base.numel())
    grad_m0 = grad_out.new_empty((n, cout))
    for l, idx in _index_groups_by_l(out_l):
        bases = out_base.index_select(0, idx).detach().cpu().tolist()
        dim = 2 * l + 1
        grad_l = torch.stack([grad_out[:, int(base):int(base) + dim] for base in bases], dim=1)
        if rotate_out:
            block = _wigner_block_from_flat(wigner, offsets, compact_offsets, l, wigner_mode)
            grad_l = torch.einsum("ncd,nd->nc", grad_l, block[:, :, l])
        else:
            grad_l = grad_l[:, :, l]
        grad_m0.index_copy_(1, idx, grad_l)
    return grad_m0


def _output_pair_grad_cuda(
    grad_out: torch.Tensor,
    wigner: torch.Tensor,
    out_base: torch.Tensor,
    out_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    m: int,
    rotate_out: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().output_pair_grad_fp32(
        grad_out.contiguous(),
        wigner,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        int(m),
        bool(rotate_out),
        int(wigner_mode),
        int(wigner_stride),
    )


def _scatter_pair_forward_cuda(
    pair_out: torch.Tensor,
    wigner: torch.Tensor,
    out_base: torch.Tensor,
    out_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    out_dim: int,
    m: int,
    rotate_out: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().scatter_pair_forward_fp32(
        pair_out.contiguous(),
        wigner,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        int(out_dim),
        int(m),
        bool(rotate_out),
        int(wigner_mode),
        int(wigner_stride),
    )


def _scatter_raw_pair_forward_cuda(
    raw: torch.Tensor,
    wigner: torch.Tensor,
    out_base: torch.Tensor,
    out_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    out_dim: int,
    m: int,
    rotate_out: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().scatter_raw_pair_forward_fp32(
        raw.contiguous(),
        wigner,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        int(out_dim),
        int(m),
        bool(rotate_out),
        int(wigner_mode),
        int(wigner_stride),
    )

def _scatter_raw_pairs_multi_forward_cuda(
    raws: list[torch.Tensor],
    wigner: torch.Tensor,
    out_bases: list[torch.Tensor],
    out_ls: list[torch.Tensor],
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    cout_prefix: torch.Tensor,
    m_values: torch.Tensor,
    out_dim: int,
    rotate_out: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().scatter_raw_pairs_multi_forward_fp32(
        [raw.contiguous() for raw in raws],
        wigner,
        out_bases,
        out_ls,
        offsets,
        compact_offsets,
        cout_prefix,
        m_values,
        int(out_dim),
        bool(rotate_out),
        int(wigner_mode),
        int(wigner_stride),
    )

def _scatter_raw_pairs_multi_output_major_forward_cuda(
    raws: list[torch.Tensor],
    wigner: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    cout_prefix: torch.Tensor,
    m_values: torch.Tensor,
    entry_offsets: torch.Tensor,
    entry_m: torch.Tensor,
    entry_channel: torch.Tensor,
    entry_d: torch.Tensor,
    entry_l: torch.Tensor,
    out_dim: int,
    rotate_out: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().scatter_raw_pairs_multi_output_major_forward_fp32(
        [raw.contiguous() for raw in raws],
        wigner,
        offsets,
        compact_offsets,
        cout_prefix,
        m_values,
        entry_offsets,
        entry_m,
        entry_channel,
        entry_d,
        entry_l,
        int(out_dim),
        bool(rotate_out),
        int(wigner_mode),
        int(wigner_stride),
    )


def _raw_pair_output_grad_cuda(
    grad_out: torch.Tensor,
    wigner: torch.Tensor,
    out_base: torch.Tensor,
    out_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    m: int,
    rotate_out: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().raw_pair_output_grad_fp32(
        grad_out.contiguous(),
        wigner,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        int(m),
        bool(rotate_out),
        int(wigner_mode),
        int(wigner_stride),
    )


def _scatter_pair_grad_radial_input_cuda(
    grad_pair_eff: torch.Tensor,
    pair_no_radial: torch.Tensor,
    radial: torch.Tensor,
    wigner: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    in_dim: int,
    m: int,
    rotate_in: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x, grad_radial = _load_extension().scatter_pair_grad_radial_input_fp32(
        grad_pair_eff.contiguous(),
        pair_no_radial.contiguous(),
        radial.contiguous(),
        wigner,
        in_base,
        in_l,
        offsets,
        compact_offsets,
        int(in_dim),
        int(m),
        bool(rotate_in),
        int(wigner_mode),
        int(wigner_stride),
    )
    return grad_x, grad_radial


def _pack_m0_cuda(
    x: torch.Tensor,
    wigner: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    rotate_in: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().pack_m0_fp32(
        x.contiguous(),
        wigner,
        in_base,
        in_l,
        offsets,
        compact_offsets,
        bool(rotate_in),
        int(wigner_mode),
        int(wigner_stride),
    )


def _output_m0_grad_cuda(
    grad_out: torch.Tensor,
    wigner: torch.Tensor,
    out_base: torch.Tensor,
    out_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    rotate_out: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().output_m0_grad_fp32(
        grad_out.contiguous(),
        wigner,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        bool(rotate_out),
        int(wigner_mode),
        int(wigner_stride),
    )


def _scatter_m0_grad_cuda(
    grad_m0: torch.Tensor,
    wigner: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    in_dim: int,
    rotate_in: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> torch.Tensor:
    return _load_extension().scatter_m0_grad_fp32(
        grad_m0.contiguous(),
        wigner,
        in_base,
        in_l,
        offsets,
        compact_offsets,
        int(in_dim),
        bool(rotate_in),
        int(wigner_mode),
        int(wigner_stride),
    )


def _scatter_m0_grad_radial_input_cuda(
    grad_eff: torch.Tensor,
    m0_no_radial: torch.Tensor,
    radial: torch.Tensor,
    wigner: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    in_dim: int,
    rotate_in: bool,
    wigner_mode: int,
    wigner_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x, grad_radial = _load_extension().scatter_m0_grad_radial_input_fp32(
        grad_eff.contiguous(),
        m0_no_radial.contiguous(),
        radial.contiguous(),
        wigner,
        in_base,
        in_l,
        offsets,
        compact_offsets,
        int(in_dim),
        bool(rotate_in),
        int(wigner_mode),
        int(wigner_stride),
    )
    return grad_x, grad_radial


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


def _segmented_linear_backward_torch(
    x_in: torch.Tensor,
    grad_out: torch.Tensor,
    graph_index: torch.Tensor,
    mixed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_routes = int(mixed_weight.shape[0])
    cin = int(mixed_weight.shape[2])
    cout = int(mixed_weight.shape[1])
    grad_x = torch.empty_like(x_in)
    grad_weight = torch.zeros_like(mixed_weight)
    flat_graph = graph_index.reshape(-1)
    for route in range(n_routes):
        idx = torch.nonzero(flat_graph == route, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        x_r = x_in.index_select(0, idx).reshape(-1, cin)
        grad_r = grad_out.index_select(0, idx).reshape(-1, cout)
        grad_weight[route] = grad_r.transpose(0, 1).matmul(x_r)
        grad_x.index_copy_(0, idx, grad_r.matmul(mixed_weight[route]).reshape(idx.numel(), cin))
    return grad_x, grad_weight


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


def _cublas_segmented_linear_backward(
    x_in: torch.Tensor,
    grad_out: torch.Tensor,
    graph_index: torch.Tensor,
    mixed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from dptb.nn.cublas_grouped_gemm import _load_extension as _load_cublas_grouped

    n_routes = int(mixed_weight.shape[0])
    cin = int(mixed_weight.shape[2])
    cout = int(mixed_weight.shape[1])
    x_flat = x_in.reshape(-1, cin).contiguous()
    grad_flat = grad_out.reshape(-1, cout).contiguous()
    order, unorder, _sorted_graph, ptr_cpu = _row_segment_layout(graph_index, n_routes)

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
    return grad_x_flat.reshape_as(x_in), grad_weight


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


def _cutlass_segmented_linear_backward(
    x_in: torch.Tensor,
    grad_out: torch.Tensor,
    graph_index: torch.Tensor,
    mixed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from dptb.nn.cutlass_grouped_gemm import grouped_gemm, grouped_gemm_backward_weight

    n_routes = int(mixed_weight.shape[0])
    cin = int(mixed_weight.shape[2])
    cout = int(mixed_weight.shape[1])
    x_flat = x_in.reshape(-1, cin).contiguous()
    grad_flat = grad_out.reshape(-1, cout).contiguous()
    order, unorder, _sorted_graph, ptr_cpu = _row_segment_layout(graph_index, n_routes)

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
    return grad_x_flat.reshape_as(x_in), grad_weight


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


def _segmented_m0_linear_forward(
    x_in: torch.Tensor,
    graph_index: torch.Tensor,
    mixed_weight: torch.Tensor,
    mixed_bias: torch.Tensor,
) -> torch.Tensor:
    n_routes = int(mixed_weight.shape[0])
    cout = int(mixed_weight.shape[1])
    raw = x_in.new_empty((x_in.shape[0], cout))
    flat_graph = graph_index.reshape(-1)
    has_bias = mixed_bias.numel() != 0
    for route in range(n_routes):
        idx = torch.nonzero(flat_graph == route, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        raw_r = x_in.index_select(0, idx).matmul(mixed_weight[route].transpose(0, 1))
        if has_bias:
            raw_r = raw_r + mixed_bias[route]
        raw.index_copy_(0, idx, raw_r)
    return raw


def _route_bias_grad(grad_out: torch.Tensor, graph_index: torch.Tensor, n_routes: int) -> torch.Tensor:
    grad_bias = grad_out.new_zeros((int(n_routes), grad_out.shape[-1]))
    expanded = graph_index.reshape(-1, 1).expand(-1, grad_out.shape[-1])
    grad_bias.scatter_add_(0, expanded, grad_out)
    return grad_bias


def _segmented_m0_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    wigner: torch.Tensor,
    graph_index: torch.Tensor,
    mixed_weight: torch.Tensor,
    mixed_bias: torch.Tensor,
    radial: torch.Tensor,
    in_base: torch.Tensor,
    in_l: torch.Tensor,
    out_base: torch.Tensor,
    out_l: torch.Tensor,
    offsets: torch.Tensor,
    compact_offsets: torch.Tensor,
    out_dim: int,
    rotate_in: bool,
    rotate_out: bool,
    radial_on_input: bool,
    wigner_mode: int,
    wigner_stride: int,
    linear_backend: str,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    del out_dim
    has_radial = radial.numel() != 0
    has_bias = mixed_bias.numel() != 0
    use_cuda_epilogue = linear_backend.startswith("cuda_")
    raw_backend = linear_backend[5:] if use_cuda_epilogue else linear_backend
    if use_cuda_epilogue:
        x_m0_no_radial = _pack_m0_cuda(
            x, wigner, in_base, in_l, offsets, compact_offsets,
            rotate_in, wigner_mode, wigner_stride
        )
    else:
        x_m0_no_radial = _pack_m0_torch(
            x, wigner, in_base, in_l, offsets, compact_offsets, rotate_in, wigner_mode
        )
    if has_radial and radial_on_input:
        x_m0_eff = x_m0_no_radial * radial
    else:
        x_m0_eff = x_m0_no_radial

    if use_cuda_epilogue:
        grad_linear = _output_m0_grad_cuda(
            grad_out, wigner, out_base, out_l, offsets, compact_offsets,
            rotate_out, wigner_mode, wigner_stride
        )
    else:
        grad_linear = _output_m0_grad_torch(
            grad_out, wigner, out_base, out_l, offsets, compact_offsets, rotate_out, wigner_mode
        )

    grad_radial = torch.zeros_like(radial) if has_radial else None
    if has_radial and not radial_on_input:
        raw = _segmented_m0_linear_forward(x_m0_eff, graph_index, mixed_weight, mixed_bias)
        grad_radial = grad_linear * raw
        grad_linear = grad_linear * radial

    if raw_backend == "cutlass_segmented":
        grad_m0_eff, grad_weight = _cutlass_segmented_linear_backward(
            x_m0_eff, grad_linear, graph_index, mixed_weight
        )
    elif raw_backend == "cublas_segmented":
        grad_m0_eff, grad_weight = _cublas_segmented_linear_backward(
            x_m0_eff, grad_linear, graph_index, mixed_weight
        )
    else:
        grad_m0_eff, grad_weight = _segmented_linear_backward_torch(
            x_m0_eff, grad_linear, graph_index, mixed_weight
        )
    grad_bias = _route_bias_grad(grad_linear, graph_index, int(mixed_weight.shape[0])) if has_bias else None

    if has_radial and radial_on_input:
        if use_cuda_epilogue and _flag("DPTB_SO2_MOE_FUSED_P0_FUSE_M0_INPUT_RADIAL_SCATTER", "1"):
            grad_x, grad_radial = _scatter_m0_grad_radial_input_cuda(
                grad_m0_eff,
                x_m0_no_radial,
                radial,
                wigner,
                in_base,
                in_l,
                offsets,
                compact_offsets,
                x.shape[1],
                rotate_in,
                wigner_mode,
                wigner_stride,
            )
            return grad_x, grad_weight, grad_bias, grad_radial
        grad_radial = grad_m0_eff * x_m0_no_radial
        grad_m0 = grad_m0_eff * radial
    else:
        grad_m0 = grad_m0_eff

    if use_cuda_epilogue:
        grad_x = _scatter_m0_grad_cuda(
            grad_m0, wigner, in_base, in_l, offsets, compact_offsets,
            x.shape[1], rotate_in, wigner_mode, wigner_stride
        )
    else:
        grad_x = _scatter_m0_grad_torch(
            grad_m0, wigner, in_base, in_l, offsets, compact_offsets,
            x.shape[1], rotate_in, wigner_mode
        )
    return grad_x, grad_weight, grad_bias, grad_radial


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
    del out_dim
    cin = int(in_base.numel())
    cout = int(out_base.numel())
    has_radial = radial.numel() != 0
    use_cuda_epilogue = linear_backend.startswith("cuda_")
    raw_backend = linear_backend[5:] if use_cuda_epilogue else linear_backend
    if use_cuda_epilogue:
        x_pair_no_radial = _pack_pair_cuda(
            x, wigner, in_base, in_l, offsets, compact_offsets,
            m, rotate_in, wigner_mode, wigner_stride
        )
    else:
        x_pair_no_radial = _pack_pair_torch(
            x, wigner, in_base, in_l, offsets, compact_offsets, m, rotate_in, wigner_mode
        )
    if has_radial and radial_on_input:
        x_pair_eff = x_pair_no_radial * radial.unsqueeze(1)
    else:
        x_pair_eff = x_pair_no_radial

    if use_cuda_epilogue:
        grad_pair = _output_pair_grad_cuda(
            grad_out, wigner, out_base, out_l, offsets, compact_offsets,
            m, rotate_out, wigner_mode, wigner_stride
        )
    else:
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

    if raw_backend == "cutlass_segmented":
        grad_x_pair_eff, grad_weight = _cutlass_segmented_raw_linear_backward(
            x_pair_eff, grad_raw, graph_index, mixed_weight
        )
    elif raw_backend == "cublas_segmented":
        grad_x_pair_eff, grad_weight = _cublas_segmented_raw_linear_backward(
            x_pair_eff, grad_raw, graph_index, mixed_weight
        )
    else:
        grad_x_pair_eff, grad_weight = _segmented_raw_linear_backward(
            x_pair_eff, grad_raw, graph_index, mixed_weight
        )
    if has_radial and radial_on_input:
        if use_cuda_epilogue and _flag("DPTB_SO2_MOE_FUSED_P0_FUSE_INPUT_RADIAL_SCATTER", "1"):
            grad_x, grad_radial = _scatter_pair_grad_radial_input_cuda(
                grad_x_pair_eff,
                x_pair_no_radial,
                radial,
                wigner,
                in_base,
                in_l,
                offsets,
                compact_offsets,
                x.shape[1],
                m,
                rotate_in,
                wigner_mode,
                wigner_stride,
            )
            return grad_x, grad_weight, grad_radial
        grad_radial = (grad_x_pair_eff * x_pair_no_radial).sum(dim=1)
        grad_x_pair = grad_x_pair_eff * radial.unsqueeze(1)
    else:
        grad_x_pair = grad_x_pair_eff

    if use_cuda_epilogue:
        grad_x = _scatter_pair_grad_cuda(
            grad_x_pair, wigner, in_base, in_l, offsets, compact_offsets,
            x.shape[1], m, rotate_in, wigner_mode, wigner_stride
        )
    else:
        grad_x = _scatter_pair_grad_torch(
            grad_x_pair, wigner, in_base, in_l, offsets, compact_offsets,
            x.shape[1], m, rotate_in, wigner_mode
        )
    return grad_x, grad_weight, grad_radial


class _FusedM0Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        wigner,
        graph_index,
        mixed_weight,
        mixed_bias,
        radial,
        in_base,
        in_l,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        out_dim: int,
        rotate_in: bool,
        rotate_out: bool,
        radial_on_input: bool,
        wigner_mode: int,
        wigner_stride: int,
    ):
        out = _load_extension().fused_m0_forward_fp32(
            x,
            wigner,
            graph_index,
            mixed_weight,
            mixed_bias,
            radial,
            in_base,
            in_l,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            int(out_dim),
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
            mixed_bias,
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
            mixed_bias,
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
            rotate_in,
            rotate_out,
            radial_on_input,
            wigner_mode,
            wigner_stride,
        ) = ctx.meta
        backward_mode = os.environ.get("DPTB_SO2_MOE_FUSED_P0_BACKWARD_MODE", "cuda_cublas_segmented")
        grad_x, grad_mixed_weight, grad_mixed_bias, grad_radial = _segmented_m0_backward(
            grad_out.contiguous(),
            x,
            wigner,
            graph_index,
            mixed_weight,
            mixed_bias,
            radial,
            in_base,
            in_l,
            out_base,
            out_l,
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
        if mixed_bias.numel() == 0:
            grad_mixed_bias = None
        if radial.numel() == 0:
            grad_radial = None
        return (
            grad_x,
            None,
            None,
            grad_mixed_weight,
            grad_mixed_bias,
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
        )


class _PackPairFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        wigner,
        in_base,
        in_l,
        offsets,
        compact_offsets,
        m: int,
        rotate_in: bool,
        wigner_mode: int,
        wigner_stride: int,
    ):
        pair = _pack_pair_cuda(
            x,
            wigner,
            in_base,
            in_l,
            offsets,
            compact_offsets,
            int(m),
            bool(rotate_in),
            int(wigner_mode),
            int(wigner_stride),
        )
        ctx.save_for_backward(wigner, in_base, in_l, offsets, compact_offsets)
        ctx.meta = (
            int(x.shape[1]),
            int(m),
            bool(rotate_in),
            int(wigner_mode),
            int(wigner_stride),
        )
        return pair

    @staticmethod
    def backward(ctx, grad_pair):
        wigner, in_base, in_l, offsets, compact_offsets = ctx.saved_tensors
        in_dim, m, rotate_in, wigner_mode, wigner_stride = ctx.meta
        grad_x = _scatter_pair_grad_cuda(
            grad_pair.contiguous(),
            wigner,
            in_base,
            in_l,
            offsets,
            compact_offsets,
            int(in_dim),
            int(m),
            bool(rotate_in),
            int(wigner_mode),
            int(wigner_stride),
        )
        return grad_x, None, None, None, None, None, None, None, None, None


class _PackPairsMultiFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        wigner,
        in_bases,
        in_ls,
        offsets,
        compact_offsets,
        cin_prefix,
        m_values,
        rotate_in: bool,
        wigner_mode: int,
        wigner_stride: int,
    ):
        packed = _pack_pairs_multi_cuda(
            x,
            wigner,
            in_bases,
            in_ls,
            offsets,
            compact_offsets,
            cin_prefix,
            m_values,
            bool(rotate_in),
            int(wigner_mode),
            int(wigner_stride),
        )
        use_multi_backward = _flag("DPTB_SO2_MOE_FUSED_P0_PACK_MULTI_BACKWARD", "1")
        if use_multi_backward:
            in_base_all = torch.cat(tuple(t.contiguous() for t in in_bases), dim=0).contiguous()
            in_l_all = torch.cat(tuple(t.contiguous() for t in in_ls), dim=0).contiguous()
        else:
            in_base_all = torch.empty((0,), dtype=torch.long, device=x.device)
            in_l_all = torch.empty((0,), dtype=torch.long, device=x.device)
        ctx.in_count = len(in_bases)
        ctx.use_multi_backward = bool(use_multi_backward)
        ctx.save_for_backward(
            wigner,
            offsets,
            compact_offsets,
            cin_prefix,
            m_values,
            in_base_all,
            in_l_all,
            *in_bases,
            *in_ls,
        )
        ctx.meta = (int(x.shape[1]), bool(rotate_in), int(wigner_mode), int(wigner_stride))
        return packed

    @staticmethod
    def backward(ctx, grad_packed):
        tensors = ctx.saved_tensors
        wigner, offsets, compact_offsets, cin_prefix, m_values, in_base_all, in_l_all = tensors[:7]
        n = ctx.in_count
        in_bases = tensors[7:7 + n]
        in_ls = tensors[7 + n:7 + 2 * n]
        in_dim, rotate_in, wigner_mode, wigner_stride = ctx.meta
        if ctx.use_multi_backward:
            grad_x = _scatter_pairs_multi_grad_cuda(
                grad_packed,
                wigner,
                in_base_all,
                in_l_all,
                offsets,
                compact_offsets,
                cin_prefix,
                m_values,
                int(in_dim),
                bool(rotate_in),
                int(wigner_mode),
                int(wigner_stride),
            )
            return grad_x, None, None, None, None, None, None, None, None, None, None
        grad_x = None
        for i in range(n):
            start = int(cin_prefix[i].item())
            end = int(cin_prefix[i + 1].item())
            grad_pair = grad_packed[:, :, start:end].contiguous()
            part = _scatter_pair_grad_cuda(
                grad_pair,
                wigner,
                in_bases[i],
                in_ls[i],
                offsets,
                compact_offsets,
                int(in_dim),
                int(m_values[i].item()),
                bool(rotate_in),
                int(wigner_mode),
                int(wigner_stride),
            )
            grad_x = part if grad_x is None else grad_x + part
        return grad_x, None, None, None, None, None, None, None, None, None, None


class _ScatterPairOutputFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pair_out,
        wigner,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        out_dim: int,
        m: int,
        rotate_out: bool,
        wigner_mode: int,
        wigner_stride: int,
    ):
        out = _scatter_pair_forward_cuda(
            pair_out,
            wigner,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            int(out_dim),
            int(m),
            bool(rotate_out),
            int(wigner_mode),
            int(wigner_stride),
        )
        ctx.save_for_backward(wigner, out_base, out_l, offsets, compact_offsets)
        ctx.meta = (
            int(out_dim),
            int(m),
            bool(rotate_out),
            int(wigner_mode),
            int(wigner_stride),
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        wigner, out_base, out_l, offsets, compact_offsets = ctx.saved_tensors
        _out_dim, m, rotate_out, wigner_mode, wigner_stride = ctx.meta
        grad_pair = _output_pair_grad_cuda(
            grad_out.contiguous(),
            wigner,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            int(m),
            bool(rotate_out),
            int(wigner_mode),
            int(wigner_stride),
        )
        return grad_pair, None, None, None, None, None, None, None, None, None, None


class _ScatterRawPairOutputFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raw,
        wigner,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        out_dim: int,
        m: int,
        rotate_out: bool,
        wigner_mode: int,
        wigner_stride: int,
    ):
        out = _scatter_raw_pair_forward_cuda(
            raw,
            wigner,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            int(out_dim),
            int(m),
            bool(rotate_out),
            int(wigner_mode),
            int(wigner_stride),
        )
        ctx.save_for_backward(wigner, out_base, out_l, offsets, compact_offsets)
        ctx.meta = (
            int(out_dim),
            int(m),
            bool(rotate_out),
            int(wigner_mode),
            int(wigner_stride),
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        wigner, out_base, out_l, offsets, compact_offsets = ctx.saved_tensors
        _out_dim, m, rotate_out, wigner_mode, wigner_stride = ctx.meta
        grad_raw = _raw_pair_output_grad_cuda(
            grad_out.contiguous(),
            wigner,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            int(m),
            bool(rotate_out),
            int(wigner_mode),
            int(wigner_stride),
        )
        return grad_raw, None, None, None, None, None, None, None, None, None, None


class _ScatterRawPairsMultiOutputFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        wigner,
        offsets,
        compact_offsets,
        cout_prefix,
        m_values,
        out_dim: int,
        rotate_out: bool,
        wigner_mode: int,
        wigner_stride: int,
        raw_count: int,
        *raws_and_maps,
    ):
        raw_count = int(raw_count)
        raws = list(raws_and_maps[:raw_count])
        out_bases = list(raws_and_maps[raw_count:2 * raw_count])
        out_ls = list(raws_and_maps[2 * raw_count:3 * raw_count])
        out = _scatter_raw_pairs_multi_forward_cuda(
            raws,
            wigner,
            out_bases,
            out_ls,
            offsets,
            compact_offsets,
            cout_prefix,
            m_values,
            int(out_dim),
            bool(rotate_out),
            int(wigner_mode),
            int(wigner_stride),
        )
        ctx.raw_count = raw_count
        ctx.save_for_backward(wigner, offsets, compact_offsets, cout_prefix, m_values, *out_bases, *out_ls)
        ctx.meta = (int(out_dim), bool(rotate_out), int(wigner_mode), int(wigner_stride))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        tensors = ctx.saved_tensors
        wigner, offsets, compact_offsets, cout_prefix, m_values = tensors[:5]
        n = ctx.raw_count
        out_bases = tensors[5:5 + n]
        out_ls = tensors[5 + n:5 + 2 * n]
        _out_dim, rotate_out, wigner_mode, wigner_stride = ctx.meta
        grad_raws = []
        for i in range(n):
            grad_raws.append(
                _raw_pair_output_grad_cuda(
                    grad_out.contiguous(),
                    wigner,
                    out_bases[i],
                    out_ls[i],
                    offsets,
                    compact_offsets,
                    int(m_values[i].item()),
                    bool(rotate_out),
                    int(wigner_mode),
                    int(wigner_stride),
                )
            )
        return (
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
            *grad_raws,
            *([None] * (2 * n)),
        )


class _ScatterRawPairsMultiOutputMajorFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        wigner,
        offsets,
        compact_offsets,
        cout_prefix,
        m_values,
        entry_offsets,
        entry_m,
        entry_channel,
        entry_d,
        entry_l,
        out_dim: int,
        rotate_out: bool,
        wigner_mode: int,
        wigner_stride: int,
        raw_count: int,
        *raws_and_maps,
    ):
        raw_count = int(raw_count)
        raws = list(raws_and_maps[:raw_count])
        out_bases = list(raws_and_maps[raw_count:2 * raw_count])
        out_ls = list(raws_and_maps[2 * raw_count:3 * raw_count])
        out = _scatter_raw_pairs_multi_output_major_forward_cuda(
            raws,
            wigner,
            offsets,
            compact_offsets,
            cout_prefix,
            m_values,
            entry_offsets,
            entry_m,
            entry_channel,
            entry_d,
            entry_l,
            int(out_dim),
            bool(rotate_out),
            int(wigner_mode),
            int(wigner_stride),
        )
        ctx.raw_count = raw_count
        ctx.save_for_backward(wigner, offsets, compact_offsets, cout_prefix, m_values, *out_bases, *out_ls)
        ctx.meta = (int(out_dim), bool(rotate_out), int(wigner_mode), int(wigner_stride))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        tensors = ctx.saved_tensors
        wigner, offsets, compact_offsets, _cout_prefix, m_values = tensors[:5]
        n = ctx.raw_count
        out_bases = tensors[5:5 + n]
        out_ls = tensors[5 + n:5 + 2 * n]
        _out_dim, rotate_out, wigner_mode, wigner_stride = ctx.meta
        grad_raws = []
        for i in range(n):
            grad_raws.append(
                _raw_pair_output_grad_cuda(
                    grad_out.contiguous(),
                    wigner,
                    out_bases[i],
                    out_ls[i],
                    offsets,
                    compact_offsets,
                    int(m_values[i].item()),
                    bool(rotate_out),
                    int(wigner_mode),
                    int(wigner_stride),
                )
            )
        return (
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
            None,
            None,
            *grad_raws,
            *([None] * (2 * n)),
        )


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
        ext = _load_extension()
        forward_mode = os.environ.get("DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE", "scalar")
        forward_fn = ext.fused_pair_forward_fp32
        tiled_forward_fns = {
            "cutlass_tiled2": "fused_pair_forward_tiled2_fp32",
            "cute_tiled2": "fused_pair_forward_tiled2_fp32",
            "cutlass_tiled3": "fused_pair_forward_tiled3_fp32",
            "cute_tiled3": "fused_pair_forward_tiled3_fp32",
            "cutlass_tiled4": "fused_pair_forward_tiled4_fp32",
            "cute_tiled4": "fused_pair_forward_tiled4_fp32",
            "cutlass_tiled8": "fused_pair_forward_tiled8_fp32",
            "cute_tiled8": "fused_pair_forward_tiled8_fp32",
        }
        if forward_mode in tiled_forward_fns:
            forward_fn = getattr(ext, tiled_forward_fns[forward_mode], None)
        elif forward_mode not in ("", "scalar"):
            if _flag("DPTB_SO2_MOE_FUSED_P0_STRICT_FORWARD_MODE"):
                raise RuntimeError(f"unknown DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE={forward_mode!r}")
            _warn_once(
                "unknown_forward_mode",
                f"unknown DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE={forward_mode!r}; using scalar fused P0 forward.",
            )
        if forward_fn is None:
            if _flag("DPTB_SO2_MOE_FUSED_P0_STRICT_FORWARD_MODE"):
                raise RuntimeError(
                    f"DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE={forward_mode!r} requires "
                    "building the extension with DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT."
                )
            _warn_once(
                "missing_tiled_forward",
                f"DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE={forward_mode!r} is unavailable; using scalar fused P0 forward.",
            )
            forward_fn = ext.fused_pair_forward_fp32

        out = forward_fn(
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
        backward_mode = os.environ.get("DPTB_SO2_MOE_FUSED_P0_BACKWARD_MODE", "cuda_cublas_segmented")
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


def _fused_pair_indexed_sandwich(
    module,
    m: int,
    x: torch.Tensor,
    wigner: torch.Tensor,
    compact_offsets: torch.Tensor,
    wigner_mode: int,
    wigner_stride: int,
    mole_globals: MOLEGlobals,
    radial_weight: Optional[torch.Tensor],
    *,
    require_cueq: bool,
):
    fc = module.m_linear[m - 1].fc
    mode = getattr(fc, "mole_linear_mode", None)
    if require_cueq and mode != "cueq_indexed_linear":
        if _flag("DPTB_SO2_MOE_FUSED_P0_STRICT_FORWARD_MODE"):
            raise RuntimeError(
                "DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE=cueq_sandwich requires "
                "mole_linear_mode='cueq_indexed_linear'."
            )
        _warn_once(
            "cueq_sandwich_backend_fallback",
            "cueq_sandwich requested but MOLELinear is not cueq_indexed_linear; "
            "using the configured indexed linear backend inside the CUDA rotation sandwich.",
        )

    in_base, in_l, out_base, out_l, offsets = _pair_maps(module, m, x.device)
    cin = int(in_base.numel())
    cout = int(out_base.numel())
    if getattr(fc, "in_features", None) != cin or getattr(fc, "out_features", None) != 2 * cout:
        _warn_once(
            "indexed_sandwich_shape_fallback",
            "streamed_m_major_fused_p0 indexed_sandwich shape does not match SO2 pair maps; falling back.",
        )
        return None

    pair = _PackPairFunction.apply(
        x.contiguous(),
        wigner,
        in_base,
        in_l,
        offsets,
        compact_offsets,
        int(m),
        bool(module.rotate_in),
        int(wigner_mode),
        int(wigner_stride),
    )

    radial = None
    if radial_weight is not None:
        radial = radial_weight.contiguous()
        expected = cin if bool(module.front) else cout
        if radial.shape != (x.shape[0], 1, expected):
            _warn_once(
                "indexed_sandwich_radial_shape_fallback",
                f"streamed_m_major_fused_p0 indexed_sandwich radial shape {tuple(radial.shape)} "
                f"does not match expected {(x.shape[0], 1, expected)}; falling back.",
            )
            return None

    if radial is not None and bool(module.front):
        pair_for_linear = pair * radial
    else:
        pair_for_linear = pair

    raw = fc(pair_for_linear, mole_globals)
    if radial is None or bool(module.front):
        return _ScatterRawPairOutputFunction.apply(
            raw.contiguous(),
            wigner,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            int(module.irreps_out.dim),
            int(m),
            bool(module.rotate_out),
            int(wigner_mode),
            int(wigner_stride),
        )

    pair_out = module.m_linear[m - 1]._finish_linear_output(raw)
    if radial is not None:
        pair_out = pair_out * radial

    return _ScatterPairOutputFunction.apply(
        pair_out.contiguous(),
        wigner,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        int(module.irreps_out.dim),
        int(m),
        bool(module.rotate_out),
        int(wigner_mode),
        int(wigner_stride),
    )


def _multi_output_entry_map(
    module,
    m_values_host: list[int],
    out_bases: list[torch.Tensor],
    out_ls: list[torch.Tensor],
    out_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = getattr(module, "_fused_p0_multi_output_entry_cache", None)
    if cache is None:
        cache = {}
        setattr(module, "_fused_p0_multi_output_entry_cache", cache)
    key = (tuple(int(v) for v in m_values_host), int(out_dim), str(device))
    cached = cache.get(key)
    if cached is not None:
        return cached

    per_feature: list[list[tuple[int, int, int, int]]] = [[] for _ in range(int(out_dim))]
    for m_idx, (_m, base_t, l_t) in enumerate(zip(m_values_host, out_bases, out_ls)):
        base_values = [int(v) for v in base_t.detach().cpu().tolist()]
        l_values = [int(v) for v in l_t.detach().cpu().tolist()]
        for channel, (base, l_value) in enumerate(zip(base_values, l_values)):
            dim = 2 * l_value + 1
            for d in range(dim):
                feature = base + d
                if 0 <= feature < int(out_dim):
                    per_feature[feature].append((m_idx, channel, d, l_value))

    entry_offsets = [0]
    entry_m = []
    entry_channel = []
    entry_d = []
    entry_l = []
    for entries in per_feature:
        for m_idx, channel, d, l_value in entries:
            entry_m.append(m_idx)
            entry_channel.append(channel)
            entry_d.append(d)
            entry_l.append(l_value)
        entry_offsets.append(len(entry_m))

    cached = (
        torch.tensor(entry_offsets, dtype=torch.long, device=device).contiguous(),
        torch.tensor(entry_m, dtype=torch.long, device=device).contiguous(),
        torch.tensor(entry_channel, dtype=torch.long, device=device).contiguous(),
        torch.tensor(entry_d, dtype=torch.long, device=device).contiguous(),
        torch.tensor(entry_l, dtype=torch.long, device=device).contiguous(),
    )
    cache[key] = cached
    return cached


def _fused_pairs_indexed_sandwich_multi(
    module,
    x: torch.Tensor,
    wigner: torch.Tensor,
    compact_offsets: torch.Tensor,
    wigner_mode: int,
    wigner_stride: int,
    mole_globals: MOLEGlobals,
    weights: Optional[torch.Tensor],
):
    if module.m_max < 1:
        return []

    graph_index = _mole_graph_index(mole_globals, x.shape[0], device=x.device)
    if graph_index.numel() != x.shape[0]:
        raise ValueError(
            f"MOLE graph_index has {graph_index.numel()} rows, but fused multi-m input has {x.shape[0]} rows."
        )

    forward_mode = os.environ.get("DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE", "scalar")
    use_native_multi = forward_mode in ("indexed_sandwich_multi_grouped", "cublas_multi_sandwich_grouped")
    use_grouped_pack = use_native_multi or _flag("DPTB_SO2_MOE_FUSED_P0_MULTI_PACK", "0")
    use_grouped_epilogue = use_native_multi or _flag("DPTB_SO2_MOE_FUSED_P0_MULTI_EPILOGUE", "0")
    pair_inputs = []
    mixed_weights = []
    post_radials = []
    maps = []
    in_bases = []
    in_ls = []
    cin_values = []
    m_values_host = []
    out_bases = []
    out_ls = []
    cout_values = []
    route_count = None

    for m in range(1, module.m_max + 1):
        fc = module.m_linear[m - 1].fc
        if not hasattr(fc, "_mix_expert_parameters") or getattr(fc, "mole_linear_mode", None) != "cublas_grouped":
            if _flag("DPTB_SO2_MOE_FUSED_P0_STRICT_FORWARD_MODE"):
                raise RuntimeError(
                    "DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE=indexed_sandwich_multi "
                    "currently requires m>0 MOLELinear blocks with mole_linear_mode='cublas_grouped'."
                )
            _warn_once(
                "indexed_sandwich_multi_backend_fallback",
                "indexed_sandwich_multi currently requires m>0 cublas_grouped MOLELinear blocks; falling back.",
            )
            return None
        if getattr(fc, "bias_experts", None) is not None:
            _warn_once(
                "indexed_sandwich_multi_bias_fallback",
                "indexed_sandwich_multi expects bias-free m>0 MoE linears; falling back.",
            )
            return None

        in_base, in_l, out_base, out_l, offsets = _pair_maps(module, m, x.device)
        cin = int(in_base.numel())
        cout = int(out_base.numel())
        if getattr(fc, "in_features", None) != cin or getattr(fc, "out_features", None) != 2 * cout:
            _warn_once(
                "indexed_sandwich_multi_shape_fallback",
                "indexed_sandwich_multi shape does not match SO2 pair maps; falling back.",
            )
            return None

        radial = None
        if weights is not None:
            radial = weights[:, module.m_in_index[m]:module.m_in_index[m + 1]].unsqueeze(1).contiguous()
            expected = cin if bool(module.front) else cout
            if radial.shape != (x.shape[0], 1, expected):
                _warn_once(
                    "indexed_sandwich_multi_radial_shape_fallback",
                    f"indexed_sandwich_multi radial shape {tuple(radial.shape)} "
                    f"does not match expected {(x.shape[0], 1, expected)}; falling back.",
                )
                return None

        mixed_weight, mixed_bias = fc._mix_expert_parameters(mole_globals)
        if mixed_bias is not None:
            _warn_once(
                "indexed_sandwich_multi_mixed_bias_fallback",
                "indexed_sandwich_multi does not handle m>0 bias; falling back.",
            )
            return None
        if mixed_weight.shape[-2:] != (2 * cout, cin):
            _warn_once(
                "indexed_sandwich_multi_weight_shape_fallback",
                "indexed_sandwich_multi mixed weight shape does not match SO2 pair maps; falling back.",
            )
            return None
        if route_count is None:
            route_count = int(mixed_weight.shape[0])
        elif route_count != int(mixed_weight.shape[0]):
            _warn_once(
                "indexed_sandwich_multi_route_count_fallback",
                "indexed_sandwich_multi requires all m blocks to share the same route count; falling back.",
            )
            return None

        in_bases.append(in_base)
        in_ls.append(in_l)
        cin_values.append(cin)
        m_values_host.append(int(m))
        out_bases.append(out_base)
        out_ls.append(out_l)
        cout_values.append(cout)
        pair_inputs.append(None)
        mixed_weights.append(mixed_weight.contiguous())
        post_radials.append(None if radial is None or bool(module.front) else radial)
        maps.append((m, out_base, out_l, offsets))

    if not pair_inputs:
        return []

    if use_grouped_pack:
        cin_prefix = [0]
        for cin in cin_values:
            cin_prefix.append(cin_prefix[-1] + int(cin))
        cin_prefix_t = torch.tensor(cin_prefix, dtype=torch.long, device=x.device).contiguous()
        m_values_t = torch.tensor(m_values_host, dtype=torch.long, device=x.device).contiguous()
        packed_all = _PackPairsMultiFunction.apply(
            x.contiguous(),
            wigner,
            in_bases,
            in_ls,
            offsets,
            compact_offsets,
            cin_prefix_t,
            m_values_t,
            bool(module.rotate_in),
            int(wigner_mode),
            int(wigner_stride),
        )
        for i, (m, cin) in enumerate(zip(m_values_host, cin_values)):
            pair = packed_all[:, :, cin_prefix[i]:cin_prefix[i + 1]]
            radial = weights[:, module.m_in_index[m]:module.m_in_index[m + 1]].unsqueeze(1).contiguous() if weights is not None else None
            pair_inputs[i] = pair * radial if radial is not None and bool(module.front) else pair
    else:
        for i, m in enumerate(m_values_host):
            pair = _PackPairFunction.apply(
                x.contiguous(),
                wigner,
                in_bases[i],
                in_ls[i],
                offsets,
                compact_offsets,
                int(m),
                bool(module.rotate_in),
                int(wigner_mode),
                int(wigner_stride),
            )
            radial = weights[:, module.m_in_index[m]:module.m_in_index[m + 1]].unsqueeze(1).contiguous() if weights is not None else None
            pair_inputs[i] = pair * radial if radial is not None and bool(module.front) else pair

    permute_idx, unpermute_idx, sorted_graph_index = mole_globals.indexed_flat_permutation(graph_index, pair_inputs[0])
    ptr = mole_globals.indexed_segment_ptr(sorted_graph_index, int(route_count), prefer_cpu=True)
    if _flag("DPTB_SO2_MOE_FUSED_P0_LOG_SCHEDULE"):
        ptr_cpu = ptr.detach().cpu()
        route_rows = (ptr_cpu[1:] - ptr_cpu[:-1]).to(dtype=torch.long)
        route_rows_list = [int(v) for v in route_rows.tolist()]
        if route_rows_list:
            rows_min = min(route_rows_list)
            rows_max = max(route_rows_list)
            rows_total = sum(route_rows_list)
        else:
            rows_min = rows_max = rows_total = 0
        m_desc = []
        for m, cin, cout in zip(m_values_host, cin_values, cout_values):
            m_desc.append(f"m={m}:M=2*rows,N={2 * int(cout)},K={int(cin)}")
        _warn_once(
            "indexed_sandwich_multi_schedule",
            "indexed_sandwich_multi schedule tag: "
            f"routes={int(route_count)}, route_rows_min/max/total={rows_min}/{rows_max}/{rows_total}, "
            f"m_count={len(m_values_host)}, problems={int(route_count) * len(m_values_host)}, "
            f"descriptors=[{'; '.join(m_desc)}]. "
            "Effective per-route GEMM is A=[rows_r*2,K], B=[N,K], C=[rows_r*2,N].",
        )

    flat_inputs = []
    for pair in pair_inputs:
        flat = pair.reshape(-1, pair.shape[-1])
        if permute_idx is not None:
            flat = flat.index_select(0, permute_idx)
        flat_inputs.append(flat.contiguous())

    from dptb.nn.cublas_grouped_gemm import grouped_gemm_multi

    flat_raw_outputs = grouped_gemm_multi(flat_inputs, [ptr] * len(flat_inputs), mixed_weights)

    if use_grouped_epilogue and all(post_radial is None for post_radial in post_radials):
        raw_tensors = []
        for flat_raw in flat_raw_outputs:
            if unpermute_idx is not None:
                flat_raw = flat_raw.index_select(0, unpermute_idx)
            raw_tensors.append(flat_raw.reshape(x.shape[0], 2, -1).contiguous())
        cout_prefix = [0]
        for cout in cout_values:
            cout_prefix.append(cout_prefix[-1] + int(cout))
        cout_prefix_t = torch.tensor(cout_prefix, dtype=torch.long, device=x.device).contiguous()
        m_values_t = torch.tensor(m_values_host, dtype=torch.long, device=x.device).contiguous()
        epilogue_schedule = os.environ.get("DPTB_SO2_MOE_FUSED_P0_MULTI_EPILOGUE_SCHEDULE", "output_major")
        if epilogue_schedule == "output_major":
            entry_offsets, entry_m, entry_channel, entry_d, entry_l = _multi_output_entry_map(
                module,
                m_values_host,
                out_bases,
                out_ls,
                int(module.irreps_out.dim),
                x.device,
            )
            return [
                _ScatterRawPairsMultiOutputMajorFunction.apply(
                    wigner,
                    offsets,
                    compact_offsets,
                    cout_prefix_t,
                    m_values_t,
                    entry_offsets,
                    entry_m,
                    entry_channel,
                    entry_d,
                    entry_l,
                    int(module.irreps_out.dim),
                    bool(module.rotate_out),
                    int(wigner_mode),
                    int(wigner_stride),
                    len(raw_tensors),
                    *raw_tensors,
                    *out_bases,
                    *out_ls,
                )
            ]
        if epilogue_schedule != "atomic":
            _warn_once(
                "indexed_sandwich_multi_epilogue_schedule_fallback",
                f"Unknown grouped epilogue schedule {epilogue_schedule!r}; using atomic scatter.",
            )
        return [
            _ScatterRawPairsMultiOutputFunction.apply(
                wigner,
                offsets,
                compact_offsets,
                cout_prefix_t,
                m_values_t,
                int(module.irreps_out.dim),
                bool(module.rotate_out),
                int(wigner_mode),
                int(wigner_stride),
                len(raw_tensors),
                *raw_tensors,
                *out_bases,
                *out_ls,
            )
        ]

    contributions = []
    for flat_raw, pair, post_radial, (m, out_base, out_l, offsets) in zip(
        flat_raw_outputs, pair_inputs, post_radials, maps
    ):
        if unpermute_idx is not None:
            flat_raw = flat_raw.index_select(0, unpermute_idx)
        raw = flat_raw.reshape(x.shape[0], 2, -1)
        if post_radial is None:
            contribution = _ScatterRawPairOutputFunction.apply(
                raw.contiguous(),
                wigner,
                out_base,
                out_l,
                offsets,
                compact_offsets,
                int(module.irreps_out.dim),
                int(m),
                bool(module.rotate_out),
                int(wigner_mode),
                int(wigner_stride),
            )
        else:
            pair_out = module.m_linear[m - 1]._finish_linear_output(raw) * post_radial
            contribution = _ScatterPairOutputFunction.apply(
                pair_out.contiguous(),
                wigner,
                out_base,
                out_l,
                offsets,
                compact_offsets,
                int(module.irreps_out.dim),
                int(m),
                bool(module.rotate_out),
                int(wigner_mode),
                int(wigner_stride),
            )
        contributions.append(contribution)

    return contributions


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

    forward_mode = os.environ.get("DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE", "scalar")
    if forward_mode in ("indexed_sandwich", "cueq_sandwich", "cueq_compatible"):
        return _fused_pair_indexed_sandwich(
            module,
            m,
            x,
            wigner,
            compact_offsets,
            wigner_mode,
            wigner_stride,
            mole_globals,
            radial_weight,
            require_cueq=(forward_mode in ("cueq_sandwich", "cueq_compatible")),
        )

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


def _fused_m0_contribution(
    module,
    x: torch.Tensor,
    wigner: torch.Tensor,
    compact_offsets: torch.Tensor,
    wigner_mode: int,
    wigner_stride: int,
    mole_globals: MOLEGlobals,
    radial_weight: Optional[torch.Tensor],
):
    fc = module.fc_m0
    if not hasattr(fc, "_mix_expert_parameters"):
        return None

    mixed_weight, mixed_bias = fc._mix_expert_parameters(mole_globals)
    graph_index = _mole_graph_index(mole_globals, x.shape[0], device=x.device)
    if graph_index.numel() != x.shape[0]:
        raise ValueError(
            f"MOLE graph_index has {graph_index.numel()} rows, but fused m0 input has {x.shape[0]} rows."
        )

    in_base, in_l, out_base, out_l, offsets = _pair_maps(module, 0, x.device)
    cin = int(in_base.numel())
    cout = int(out_base.numel())
    if mixed_weight.shape[-2:] != (cout, cin):
        _warn_once(
            "m0_weight_shape_fallback",
            "streamed_m_major_fused_p0 m0 mixed weight shape does not match SO2 maps; falling back.",
        )
        return None

    if mixed_bias is None:
        mixed_bias_arg = x.new_empty((0,))
    else:
        if mixed_bias.shape != (mixed_weight.shape[0], cout):
            _warn_once(
                "m0_bias_shape_fallback",
                "streamed_m_major_fused_p0 m0 mixed bias shape does not match SO2 maps; falling back.",
            )
            return None
        mixed_bias_arg = mixed_bias.contiguous()

    if radial_weight is None:
        radial = x.new_empty((0,))
        radial_on_input = True
    else:
        radial = radial_weight.squeeze(1).contiguous()
        radial_on_input = bool(module.front)
        expected = cin if radial_on_input else cout
        if radial.shape != (x.shape[0], expected):
            _warn_once(
                "m0_radial_shape_fallback",
                f"streamed_m_major_fused_p0 m0 radial shape {tuple(radial.shape)} does not match expected {(x.shape[0], expected)}; falling back.",
            )
            return None

    return _FusedM0Function.apply(
        x.contiguous(),
        wigner,
        graph_index.to(device=x.device, dtype=torch.long).contiguous(),
        mixed_weight.contiguous(),
        mixed_bias_arg,
        radial,
        in_base,
        in_l,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        int(module.irreps_out.dim),
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

    forward_mode = os.environ.get("DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE", "scalar")
    if forward_mode in (
        "indexed_sandwich_multi_direct_warp",
        "route_m_direct_warp",
        "custom_a_loader_epilogue",
        "indexed_sandwich_multi_cute_tiled",
        "indexed_sandwich_multi_cutlass_native",
        "custom_a_loader_cutlass_epilogue",
    ):
        from dptb.nn.so2_moe_persistent_grouped import try_forward_so2_moe_persistent_grouped_p1

        mainloop_override = (
            "cute_tiled"
            if forward_mode in (
                "indexed_sandwich_multi_cute_tiled",
            )
            else "cutlass_native"
            if forward_mode in (
                "indexed_sandwich_multi_cutlass_native",
                "custom_a_loader_cutlass_epilogue",
            )
            else "warp_collective"
        )
        return try_forward_so2_moe_persistent_grouped_p1(
            module,
            x,
            R,
            mole_globals,
            latents,
            wigner_D_all,
            include_m0_override=False,
            mainloop_override=mainloop_override,
        )

    weights = module.radial_emb(latents) if module.radial_emb else None
    out = torch.zeros((x.shape[0], module.irreps_out.dim), dtype=x.dtype, device=x.device)

    radial_m0 = weights[:, module.m_in_index[0]:module.m_in_index[1]].unsqueeze(1) if module.radial_emb else None
    m0_contribution = None
    if _flag("DPTB_SO2_MOE_FUSED_P0_FUSE_M0", "0"):
        m0_contribution = _fused_m0_contribution(
            module,
            x,
            wigner,
            compact_offsets,
            wigner_mode,
            wigner_stride,
            mole_globals,
            radial_m0,
        )
    if m0_contribution is None:
        if _flag("DPTB_SO2_MOE_FUSED_P0_STRICT_M0"):
            raise RuntimeError("streamed_m_major_fused_p0 m0 fusion declined under strict mode.")
        inp0 = module._direct_rotate_pack_m(x, 0, wigner_D_all)
        if module.front and module.radial_emb:
            y0 = module.fc_m0(inp0 * radial_m0.squeeze(1), mole_globals)
        elif module.radial_emb:
            y0 = module.fc_m0(inp0, mole_globals) * radial_m0.squeeze(1)
        else:
            y0 = module.fc_m0(inp0, mole_globals)
        module._accumulate_m0_output(out, y0, wigner_D_all)
    else:
        out.add_(m0_contribution)

    if forward_mode in (
        "indexed_sandwich_multi",
        "cublas_multi_sandwich",
        "route_m_sandwich",
        "indexed_sandwich_multi_grouped",
        "cublas_multi_sandwich_grouped",
    ):
        contributions = _fused_pairs_indexed_sandwich_multi(
            module,
            x,
            wigner,
            compact_offsets,
            wigner_mode,
            wigner_stride,
            mole_globals,
            weights,
        )
        if contributions is None:
            return None
        for contribution in contributions:
            out.add_(contribution)
    else:
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

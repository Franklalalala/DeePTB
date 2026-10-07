
from e3nn.o3 import xyz_to_angles, Irreps
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import os
import torch.nn.functional as F
from collections import defaultdict
from .tensor_product import InterpolationBlock, RadialFunction, complex_pair_output
from .so2_parity import ParityWeightMixin, normalize_so2_parity, enforce_so2_parity, parity_masks
from dptb.utils.cuda_cache_memory import cuda_cache_memory_probe, record_cuda_cache_event

# Load helpers (Keep original logic)
try:
    _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"), weights_only=False)
    _idx_data = torch.load(os.path.join(os.path.dirname(__file__), "z_rot_indices_lmax12.pt"), weights_only=False)
except (FileNotFoundError, RuntimeError):
    # Fallback for dry-run or missing files
    _Jd = []
    _idx_data = {}

_WIGNER_STATIC_CACHE = {}
log = logging.getLogger(__name__)


def _ensure_torch_fx_symbolic_tracing_compat():
    try:
        import torch.fx._symbolic_trace as symbolic_trace
    except Exception:
        return
    if not hasattr(symbolic_trace, "is_fx_symbolic_tracing") and hasattr(symbolic_trace, "is_fx_tracing"):
        symbolic_trace.is_fx_symbolic_tracing = symbolic_trace.is_fx_tracing


_ensure_torch_fx_symbolic_tracing_compat()


def build_z_rot_multi(angle_stack, mask, freq, reversed_inds, offsets, d_total: int):
    """
    angle_stack: (3*N, )    # Input with alpha, beta, gamma stacked together
    l_max: int

    Returns: (Xa, Xb, Xc) # Each is of shape (N, D_total, D_total)
    """
    N_all = angle_stack.shape[0]
    N = N_all // 3

    # Step 1: Vectorized computation of sine and cosine values
    angle_expand = angle_stack[None, :, None]  # (1, 3N, 1)
    freq_expand = freq[:, None, :]  # (L, 1, Mmax)
    sin_val = torch.sin(freq_expand * angle_expand)  # (L, 3N, Mmax)
    cos_val = torch.cos(freq_expand * angle_expand)  # (L, 3N, Mmax)

    # Step 2: Construct the block-diagonal matrix
    M_total = angle_stack.new_zeros((N_all, d_total, d_total))
    idx_l, idx_row = torch.where(mask)  # (K,), (K,)
    idx_col_diag = idx_row
    idx_col_anti = reversed_inds[idx_l, idx_row]
    global_row = offsets[idx_l] + idx_row  # (K,)
    global_col_diag = offsets[idx_l] + idx_col_diag
    global_col_anti = offsets[idx_l] + idx_col_anti

    # Assign values to the diagonal
    M_total[:, global_row, global_col_diag] = cos_val[idx_l, :, idx_row].transpose(0, 1)
    # Assign values to non-overlapping anti-diagonals
    overlap_mask = (global_row == global_col_anti)
    M_total[:, global_row[~overlap_mask], global_col_anti[~overlap_mask]] = sin_val[idx_l[~overlap_mask], :,
                                                                            idx_row[~overlap_mask]].transpose(0, 1)

    # Step 3: Split into three components corresponding to alpha, beta, gamma
    Xa = M_total[:N]
    Xb = M_total[N:2 * N]
    Xc = M_total[2 * N:]

    return Xa, Xb, Xc


def _get_wigner_static(l_max: int, device: torch.device, dtype: torch.dtype):
    key = (int(l_max), str(device), dtype)
    cached = _WIGNER_STATIC_CACHE.get(key)
    if cached is not None:
        return cached

    metadata = {
        "l_max": int(l_max),
        "local_entries_before": len(_WIGNER_STATIC_CACHE),
    }
    with cuda_cache_memory_probe("wigner_static", key, device=device, metadata=metadata, logger=log):
        idx_data = {
            k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
            for k, v in _idx_data.items()
        }
        sizes = idx_data["sizes"][:l_max + 1]
        offsets = idx_data["offsets"][:l_max + 1]
        mask = idx_data["mask"][:l_max + 1]
        freq = idx_data["freq"][:l_max + 1]
        reversed_inds = idx_data["reversed_inds"][:l_max + 1]

        dims = [2 * l + 1 for l in range(l_max + 1)]
        d_total = sum(dims)
        J_full_small = torch.zeros(d_total, d_total, dtype=dtype, device=device)
        for l, dim in enumerate(dims):
            start = l * l
            J_full_small[start:start + dim, start:start + dim] = _Jd[l].to(dtype=dtype, device=device)

        cached = {
            "sizes": sizes,
            "offsets": offsets,
            "mask": mask,
            "freq": freq,
            "reversed_inds": reversed_inds,
            "J_full_small": J_full_small,
            "d_total": d_total,
        }
        _WIGNER_STATIC_CACHE[key] = cached
        metadata["local_entries_after"] = len(_WIGNER_STATIC_CACHE)
    return cached


def batch_wigner_D(l_max, alpha, beta, gamma, _Jd):
    """
    Compute Wigner D matrices for all L (from 0 to l_max) in a single batch.
    Returns a tensor of shape [N, D, D], where D = sum(2l+1 for l in 0..l_max).
    """
    device = alpha.device
    N = alpha.shape[0]
    static = _get_wigner_static(l_max, device, alpha.dtype)
    d_total = static["d_total"]

    offsets = static["offsets"]
    mask = static["mask"]
    freq = static["freq"]
    reversed_inds = static["reversed_inds"]
    J_full_small = static["J_full_small"]

    J_full = J_full_small.unsqueeze(0).expand(N, -1, -1)
    angle_stack = torch.cat([alpha, beta, gamma], dim=0)
    Xa, Xb, Xc = build_z_rot_multi(angle_stack, mask, freq, reversed_inds, offsets, d_total)

    return Xa @ J_full @ Xb @ J_full @ Xc


def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


class SO2WignerBlocks:
    """Per-l Wigner rotation blocks without materializing the full [N, D, D] matrix."""

    __slots__ = ("blocks",)

    def __init__(self, blocks):
        self.blocks = tuple(blocks)

    def block(self, l: int):
        return self.blocks[l]


def batch_wigner_D_blocks(l_max, alpha, beta, gamma, _Jd):
    """Compute Wigner D as compact per-l blocks instead of a dense block-diagonal matrix."""
    return SO2WignerBlocks(wigner_D(l, alpha, beta, gamma) for l in range(l_max + 1))


def _normalize_wigner_apply_mode(wigner_apply_mode: str) -> str:
    if wigner_apply_mode not in ("full_dense", "compact_blocks"):
        raise ValueError(
            "wigner_apply_mode must be 'full_dense' or 'compact_blocks', "
            f"got {wigner_apply_mode!r}"
        )
    return wigner_apply_mode


def _normalize_so2_fusion_mode(so2_fusion_mode: str) -> str:
    allowed = (
        "staged",
        "streamed_m_major_ref",
        "streamed_m_major_cueq",
        "streamed_m_major_fused_p0",
    )
    if so2_fusion_mode not in allowed:
        raise ValueError(
            f"so2_fusion_mode must be one of {allowed}, "
            f"got {so2_fusion_mode!r}"
        )
    return so2_fusion_mode


def _make_wigner_rotation(l_max, alpha, beta, gamma, wigner_apply_mode: str):
    if wigner_apply_mode == "compact_blocks":
        return batch_wigner_D_blocks(l_max, alpha, beta, gamma, _Jd)
    return batch_wigner_D(l_max, alpha, beta, gamma, _Jd)


def _select_wigner_block(wigner_D_all, l: int, offsets, dims):
    if wigner_D_all is None:
        raise ValueError(
            f"wigner_D_all is required to select Wigner block l={l}; "
            "enable rotation construction or pass a precomputed Wigner object."
        )
    dim = dims[l]
    if isinstance(wigner_D_all, SO2WignerBlocks):
        if l >= len(wigner_D_all.blocks):
            raise ValueError(
                f"wigner_D_all only has {len(wigner_D_all.blocks)} compact blocks, "
                f"but SO2 needs l={l}. Recompute Wigner D with a larger l_max."
            )
        block = wigner_D_all.block(l)
        if block.shape[-2:] != (dim, dim):
            raise ValueError(
                f"Wigner block l={l} has shape {tuple(block.shape[-2:])}, "
                f"expected {(dim, dim)}."
            )
        return block
    start = offsets[l]
    if wigner_D_all.shape[-2] < start + dim or wigner_D_all.shape[-1] < start + dim:
        raise ValueError(
            f"wigner_D_all dense shape {tuple(wigner_D_all.shape[-2:])} does not include "
            f"block l={l}. Recompute Wigner D with l_max large enough for SO2_Linear."
        )
    return wigner_D_all[:, start:start + dim, start:start + dim]


@dataclass(frozen=True)
class _SO2EntryPlan:
    l: int
    mul: int
    slice_info: slice
    group_start: int


@dataclass(frozen=True)
class _SO2LGroupPlan:
    l: int
    dims: int
    total_mul: int
    muls: Tuple[int, ...]
    slices: Tuple[slice, ...]


def _build_so2_layout_plans(irreps) -> Tuple[Tuple[_SO2EntryPlan, ...], Dict[int, _SO2LGroupPlan]]:
    running_by_l = defaultdict(int)
    specs_by_l: Dict[int, List[Tuple[int, slice]]] = defaultdict(list)
    entries: List[_SO2EntryPlan] = []

    for (mul, (l, _p)), slice_info in zip(irreps, irreps.slices()):
        group_start = running_by_l[l]
        entries.append(
            _SO2EntryPlan(
                l=l,
                mul=mul,
                slice_info=slice_info,
                group_start=group_start,
            )
        )
        running_by_l[l] += mul
        specs_by_l[l].append((mul, slice_info))

    groups: Dict[int, _SO2LGroupPlan] = {}
    for l, specs in specs_by_l.items():
        groups[l] = _SO2LGroupPlan(
            l=l,
            dims=2 * l + 1,
            total_mul=sum(mul for mul, _ in specs),
            muls=tuple(mul for mul, _ in specs),
            slices=tuple(slice_info for _, slice_info in specs),
        )
    return tuple(entries), groups


def _gather_so2_l_group(x: torch.Tensor, plan: _SO2LGroupPlan) -> torch.Tensor:
    n = x.shape[0]
    parts = [
        x[:, slice_info].reshape(n, mul, plan.dims)
        for mul, slice_info in zip(plan.muls, plan.slices)
    ]
    if len(parts) == 1:
        return parts[0].contiguous()
    return torch.cat(parts, dim=1).contiguous()


# ------------------------------------------------------------------------------
# MOLE COMPONENTS (Added)
# ------------------------------------------------------------------------------

from .pdq_moe import MOLEGlobals, MOLERouterV3, MOLELinear, PDQMoE, PDQMoELinear, PDQMoERouter, PDQMoERouting, ROUTER_REGULARIZER_KEYS, _route_layout_token, _RowPermutation, permute_rows, _functorch_plain, _mole_split_sizes, _mole_graph_index, _expand_graph_index_for_leading_dims, _expand_graph_index_cached, _expand_route_index_for_leading_dims, _normalize_mole_linear_mode, _expert_route_indices_from_globals, router_z_loss, write_router_regularizers


def _index_select_wigner_edges(wigner_D_all, index: torch.Tensor):
    if wigner_D_all is None:
        return None
    if isinstance(wigner_D_all, SO2WignerBlocks):
        return SO2WignerBlocks(block.index_select(0, index) for block in wigner_D_all.blocks)
    return wigner_D_all.index_select(0, index)


# ------------------------------------------------------------------------------


class SO2PostActivationExpertMixer(torch.nn.Module):
    """Hybrid nonlinear expert mixer for SO2 TP outputs."""

    def __init__(
            self,
            tp: "SO2_Linear",
            activation: torch.nn.Module,
            router_from_0e: torch.nn.Module,
            scalar_dim: int,
            route_chunk_size: Optional[int] = None,
        checkpoint_routes: bool = False,
    ):
        super().__init__()
        object.__setattr__(self, "tp", tp)
        object.__setattr__(self, "activation", activation)
        self.router_from_0e = router_from_0e
        self.scalar_dim = int(scalar_dim)
        self.route_chunk_size = None if route_chunk_size is None else int(route_chunk_size)
        self.checkpoint_routes = bool(checkpoint_routes)

    def _mix_route_chunk(self, x_routes, R_routes, flat_expert_index, latents_routes, wigner_routes, n_rows, k_routes):
        y_routes, _ = self.tp.forward_expert_routes(
            x_routes,
            R_routes,
            flat_expert_index,
            latents=latents_routes,
            wigner_D_all=wigner_routes,
        )
        y_routes = self.activation(y_routes)
        if self.scalar_dim <= 0 or y_routes.shape[-1] < self.scalar_dim:
            raise ValueError(
                f"scalar_dim={self.scalar_dim} is incompatible with activated output "
                f"dim={y_routes.shape[-1]}."
            )
        scores = self.router_from_0e(y_routes[:, :self.scalar_dim]).reshape(n_rows, k_routes)
        alpha = torch.softmax(scores, dim=1).reshape(n_rows, k_routes, 1)
        return (y_routes.reshape(n_rows, k_routes, -1) * alpha).sum(dim=1)

    def forward(self, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
        if mole_globals is None:
            raise ValueError("SO2PostActivationExpertMixer requires MOLEGlobals.")
        n_rows = int(x.shape[0])
        if n_rows == 0:
            empty, _ = self.tp.forward_expert_routes(
                x,
                R,
                x.new_empty((0,), dtype=torch.long),
                latents=latents,
                wigner_D_all=wigner_D_all,
            )
            return empty, wigner_D_all

        expert_indices = _expert_route_indices_from_globals(
            mole_globals,
            self.tp.num_experts,
            n_rows,
            device=x.device,
        )
        if expert_indices.ndim != 2:
            raise ValueError(f"expert route indices must be [n_rows, k], got {tuple(expert_indices.shape)}.")
        k_routes = int(expert_indices.shape[1])
        if k_routes <= 0:
            raise ValueError("SO2PostActivationExpertMixer requires at least one expert route per row.")

        chunk_size = self.route_chunk_size or n_rows
        if chunk_size <= 0:
            chunk_size = n_rows

        out_parts = []
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            row_index = torch.arange(start, end, device=x.device, dtype=torch.long)
            local_row_index = torch.arange(end - start, device=x.device, dtype=torch.long)
            flat_local_row_index = local_row_index.repeat_interleave(k_routes)
            flat_expert_index = expert_indices[start:end].reshape(-1)

            x_chunk = x.index_select(0, row_index)
            R_chunk = R.index_select(0, row_index) if R is not None else None
            latents_chunk = latents.index_select(0, row_index) if latents is not None else x_chunk.new_empty((0,))
            chunk_wigner = _index_select_wigner_edges(wigner_D_all, row_index)
            if chunk_wigner is None and R is not None:
                chunk_wigner = self.tp._ensure_wigner_rotation(R_chunk, None)

            def chunk_fn(
                x_chunk_arg,
                latents_chunk_arg,
                R_arg=R_chunk,
                expert_arg=flat_expert_index,
                wigner_arg=chunk_wigner,
                local_index_arg=flat_local_row_index,
                has_latents=latents is not None,
                chunk_rows=end - start,
            ):
                x_routes_arg = x_chunk_arg.index_select(0, local_index_arg)
                R_routes_arg = R_arg.index_select(0, local_index_arg) if R_arg is not None else None
                latents_routes_arg = (
                    latents_chunk_arg.index_select(0, local_index_arg)
                    if has_latents else None
                )
                wigner_routes_arg = _index_select_wigner_edges(wigner_arg, local_index_arg)
                return self._mix_route_chunk(
                    x_routes_arg,
                    R_routes_arg,
                    expert_arg,
                    latents_routes_arg,
                    wigner_routes_arg,
                    chunk_rows,
                    k_routes,
                )

            if self.checkpoint_routes and torch.is_grad_enabled():
                mixed = torch_checkpoint(chunk_fn, x_chunk, latents_chunk, use_reentrant=False)
            else:
                mixed = chunk_fn(x_chunk, latents_chunk)
            out_parts.append(mixed)

        return torch.cat(out_parts, dim=0), wigner_D_all


class SO2SlotPostActivationMixer(torch.nn.Module):
    """Nonlinear experts for per-row top-k routing: ``h' = sum_j g_j act(SO2_{e_j}(x))``.

    ``SO2_{e_j}`` is the selected SO2 operator of top-k slot ``j``: expert ``e_j`` with the shared affine parameters
    folded in (weights and biases; any non-MoLE blocks such as interpolation blocks are part of every slot), which is
    exact only when the coefficients sum to one; with ``act`` the identity the sum equals the pre-activation mix.
    State dicts are interchangeable with pre_activation, but loading pre_activation weights does not preserve the
    function (the activation moved inside the sum).  Each slot runs the layer's own
    activation-space route with a one-slot MOLEGlobals (coefficient 1), so the grouped GEMM stays segmented by
    expert id and no per-edge weight is built; the slot outputs are activated separately and then weighted by
    the router's coefficients ``g``, which keep their gradient.  ``act`` must be equivariant on the layer's output
    irreps (e3nn ``Gate``: gates computed from 0e scalars, one gate value for all m components of a gated irrep).
    Cost: k SO2 passes (rotation, GEMM, scatter) instead of one, and k activated outputs kept for backward.
    """

    def __init__(self, tp: "SO2_Linear", activation: torch.nn.Module):
        super().__init__()
        # not registered as submodules: they belong to the owning update block
        object.__setattr__(self, "tp", tp)
        object.__setattr__(self, "activation", activation)

    def _activated_dim(self) -> int:
        irreps_out = getattr(self.activation, "irreps_out", None)
        if irreps_out is not None:
            return int(irreps_out.dim)
        probe = torch.zeros(0, int(self.tp.irreps_out.dim))
        return int(self.activation(probe).shape[-1])

    def forward(self, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
        if x.shape[0] == 0:
            # no active rows: the router builds globals without top-k metadata; nothing to route or activate
            return x.new_zeros((0, self._activated_dim())), wigner_D_all
        idx = getattr(mole_globals, "topk_indices", None)
        val = getattr(mole_globals, "topk_values", None)
        if mole_globals is None or not getattr(mole_globals, "activation_space", False) or idx is None or val is None:
            raise ValueError("SO2SlotPostActivationMixer needs per-row activation-space top-k routing "
                             "(prior_activate); got %r." % (type(mole_globals).__name__,))
        if not getattr(mole_globals, "coefficients_sum_to_one", False):
            raise ValueError("SO2SlotPostActivationMixer folds the shared expert into every slot, which needs "
                             "coefficients that sum to one.")
        if idx.dim() != 2 or val.shape != idx.shape or idx.shape[0] != x.shape[0] or idx.shape[1] == 0:
            raise ValueError("top-k routing must be [n_rows, k] for %d rows; got indices %s, values %s."
                             % (x.shape[0], tuple(idx.shape), tuple(val.shape)))
        num_experts = int(self.tp.num_experts)
        out = None
        for j in range(idx.shape[1]):
            slot_idx = idx[:, j:j + 1].to(device=x.device, dtype=torch.long).contiguous()
            one = torch.ones(slot_idx.shape, dtype=x.dtype, device=x.device)
            coeff = torch.zeros(slot_idx.shape[0], num_experts, dtype=x.dtype, device=x.device).scatter_(1, slot_idx, one)
            slot_globals = MOLEGlobals(coefficients=coeff, sizes=None, topk_indices=slot_idx, topk_values=one,
                                       activation_space=True, coefficients_sum_to_one=True)
            y, wigner_D_all = self.tp(x, R, slot_globals, latents, wigner_D_all)
            part = self.activation(y) * val[:, j:j + 1].to(device=y.device, dtype=y.dtype)
            out = part if out is None else out + part
        return out, wigner_D_all


class SO2SharedPostActivationMixer(torch.nn.Module):
    """Nonlinear experts with a separate shared branch (DPA3-MoE, Liu et al., npj Artif. Intell. 2026, eq. 4):

    ``h' = act(SO2_sh(x)) + sum_j g_j [act(SO2_{e_j}(x)) - act(0)]``

    ``SO2_sh`` is the layer with the shared expert and every non-MoLE block (interpolation) and no routed expert;
    ``SO2_{e_j}`` is routed expert ``e_j`` of top-k slot ``j`` alone (no shared expert, no non-MoLE block).  Both
    are activated separately; the coefficients ``g`` need not sum to one (``edge_router_gate=full_softmax`` keeps the
    router's probability mass).  Subtracting ``act(0)`` (zero for the gate activation) makes a routed expert with
    zero weights contribute exactly nothing, so routed experts initialised at zero reproduce the dense layer.
    State dicts are interchangeable with pre_activation.  Cost: k + 1 SO2 passes instead of one.
    """

    def __init__(self, tp: "SO2_Linear", activation: torch.nn.Module):
        super().__init__()
        # not registered as submodules: they belong to the owning update block
        object.__setattr__(self, "tp", tp)
        object.__setattr__(self, "activation", activation)

    def _activated_dim(self) -> int:
        irreps_out = getattr(self.activation, "irreps_out", None)
        if irreps_out is not None:
            return int(irreps_out.dim)
        probe = torch.zeros(0, int(self.tp.irreps_out.dim))
        return int(self.activation(probe).shape[-1])

    def forward(self, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
        if x.shape[0] == 0:
            return x.new_zeros((0, self._activated_dim())), wigner_D_all
        idx = getattr(mole_globals, "topk_indices", None)
        val = getattr(mole_globals, "topk_values", None)
        if mole_globals is None or not getattr(mole_globals, "activation_space", False) or idx is None or val is None:
            raise ValueError("SO2SharedPostActivationMixer needs per-row activation-space top-k routing "
                             "(prior_activate); got %r." % (type(mole_globals).__name__,))
        if idx.dim() != 2 or val.shape != idx.shape or idx.shape[0] != x.shape[0] or idx.shape[1] == 0:
            raise ValueError("top-k routing must be [n_rows, k] for %d rows; got indices %s, values %s."
                             % (x.shape[0], tuple(idx.shape), tuple(val.shape)))
        num_experts = int(self.tp.num_experts)
        n = idx.shape[0]
        idx = idx.to(device=x.device, dtype=torch.long)
        zero_val = torch.zeros(n, 1, dtype=x.dtype, device=x.device)
        shared_globals = MOLEGlobals(coefficients=torch.zeros(n, num_experts, dtype=x.dtype, device=x.device),
                                     sizes=None, topk_indices=idx[:, :1].contiguous(), topk_values=zero_val,
                                     activation_space=True, coefficients_sum_to_one=False, branch="shared")
        y, wigner_D_all = self.tp(x, R, shared_globals, latents, wigner_D_all)
        out = self.activation(y)
        act0 = self.activation(y.new_zeros(1, y.shape[-1]))
        for j in range(idx.shape[1]):
            slot_idx = idx[:, j:j + 1].contiguous()
            one = torch.ones(slot_idx.shape, dtype=x.dtype, device=x.device)
            coeff = torch.zeros(n, num_experts, dtype=x.dtype, device=x.device).scatter_(1, slot_idx, one)
            slot_globals = MOLEGlobals(coefficients=coeff, sizes=None, topk_indices=slot_idx, topk_values=one,
                                       activation_space=True, coefficients_sum_to_one=False, branch="routed")
            y, wigner_D_all = self.tp(x, R, slot_globals, latents, wigner_D_all)
            out = out + (self.activation(y) - act0) * val[:, j:j + 1].to(device=y.device, dtype=y.dtype)
        return out, wigner_D_all


class SO2_Linear(torch.nn.Module):
    """
    SO(2) Convolutional layer with MoE and Rotate Control.
    """

    def __init__(
            self,
            irreps_in,
            irreps_out,
            radial_emb: bool = False,
            latent_dim: int = None,
            radial_channels: list = None,
            extra_m0_outsize: int = 0,
            use_interpolation: bool = False,
            # === MoE 参数 ===
            num_experts: int = 8,
            num_shared_experts: int = 1, # Added
            # === Rotation 控制参数 (Keep-in-Frame) ===
            rotate_in: bool = True,
            rotate_out: bool = True,
            wigner_apply_mode: str = "compact_blocks",
            mole_linear_mode=None,
            mole_expert_parameterization="full",
            mole_expert_rank=64,
            so2_fusion_mode: str = "staged",
            so2_parity: str = "none",
    ):
        super(SO2_Linear, self).__init__()

        self.irreps_in = Irreps(irreps_in).simplify()
        self.so2_parity = normalize_so2_parity(so2_parity)
        self.irreps_out = (Irreps(f"{extra_m0_outsize}x0e") + Irreps(irreps_out)).simplify()
        self.in_l_max = self.irreps_in.lmax
        self.out_l_max = self.irreps_out.lmax
        self.m_max = min(self.in_l_max, self.out_l_max)
        self.l_max = max(self.in_l_max, self.out_l_max)
        self.radial_emb = radial_emb
        self.latent_dim = latent_dim

        # 保存 flag
        self.rotate_in = rotate_in
        self.rotate_out = rotate_out
        self.wigner_apply_mode = _normalize_wigner_apply_mode(wigner_apply_mode)
        env_so2_fusion_mode = os.environ.get("DPTB_SO2_FUSION_MODE")
        if env_so2_fusion_mode is not None and so2_fusion_mode in (None, "staged"):
            so2_fusion_mode = env_so2_fusion_mode
        self.so2_fusion_mode = _normalize_so2_fusion_mode(so2_fusion_mode)
        self.num_experts = num_experts

        self.m_linear = nn.ModuleList()

        num_in_m0 = self.irreps_in.num_irreps
        num_out_m0 = self.irreps_out.num_irreps

        # MODIFICATION: Use MOLELinear for scalar projection (bias=True as per original)
        self.fc_m0 = MOLELinear(
            num_in_m0,
            num_out_m0,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            bias=True,
            mole_linear_mode=mole_linear_mode,
            mole_expert_parameterization=mole_expert_parameterization,
            mole_expert_rank=mole_expert_rank,
        )

        for m in range(1, self.m_max + 1):
            # 假设 SO2_m_Linear 已经支持 num_experts 参数
            self.m_linear.append(SO2_m_Linear(
                m,
                self.irreps_in,
                self.irreps_out,
                use_interpolation=use_interpolation,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                mole_linear_mode=mole_linear_mode,
                mole_expert_parameterization=mole_expert_parameterization,
                mole_expert_rank=mole_expert_rank,
            ))

        # --- Mask 和 Index 构建逻辑 (保持不变) ---
        m_in_mask = torch.zeros(self.m_max + 1, self.irreps_in.dim, dtype=torch.bool)
        m_out_mask = torch.zeros(self.m_max + 1, self.irreps_out.dim, dtype=torch.bool)
        front = self.irreps_in.dim <= self.irreps_out.dim
        self.m_in_num = [0] * (self.m_max + 1)
        offset = 0
        for mul, (l, p) in self.irreps_in:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(min(l, self.m_max) + 1):
                m_in_mask[m, start_id + l + m] = True
                m_in_mask[m, start_id + l - m] = True
                if front:
                    self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)
        offset = 0
        for mul, (l, p) in self.irreps_out:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(min(l, self.m_max) + 1):
                m_out_mask[m, start_id + l + m] = True
                m_out_mask[m, start_id + l - m] = True
                if not front:
                    self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)
        self.register_buffer("m_in_mask", m_in_mask)
        self.register_buffer("m_out_mask", m_out_mask)
        self.m_in_index = [0] + [int(v) for v in torch.cumsum(torch.tensor(self.m_in_num), dim=0).tolist()]
        if radial_emb:
            self.radial_emb = RadialFunction([latent_dim] + radial_channels + [self.m_in_index[-1]])
        self.front = front
        self.dims = {l: 2 * l + 1 for l in range(self.l_max + 1)}
        self.offsets = {}
        offset = 0
        for l in range(self.l_max + 1):
            self.offsets[l] = offset
            offset += self.dims[l]
        self._in_entry_plans, self._in_group_plans = _build_so2_layout_plans(self.irreps_in)
        self._out_entry_plans, self._out_group_plans = _build_so2_layout_plans(self.irreps_out)
        self._in_entries_by_m = {
            m: tuple(entry for entry in self._in_entry_plans if entry.l >= m)
            for m in range(self.m_max + 1)
        }
        self._out_entries_by_m = {
            m: tuple(entry for entry in self._out_entry_plans if entry.l >= m)
            for m in range(self.m_max + 1)
        }
        self._active_rot_l = tuple(sorted({
            entry.l for entry in self._in_entry_plans if entry.l > 0
        } | {
            entry.l for entry in self._out_entry_plans if entry.l > 0
        }))
        if self.so2_parity == "enforce":
            enforce_so2_parity(self)

    def forward(self, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
        """Rotate, apply the m-wise (MoE) linears, rotate back.

        Args:
            x: Input features
            R: Edge vectors (for rotation)
            mole_globals: MoE routing info
            latents: Latent features for radial embedding
            wigner_D_all: Precomputed Wigner D matrices (optional)

        Activation-space routes preserve the selected experts through their
        nonlinear activations. Weight-space routes may use fused P0. The
        optional backend declines unsupported inputs to the grouped Torch route.
        """
        if self.num_experts == 0:
            # Never pass the embedding's E-way expert ids into a shared-only layer.
            # Keep activation-space dispatch so fused P0 uses its existing shared
            # branch, including each output interpolation block exactly once.
            n = x.shape[0]
            mole_globals = MOLEGlobals(
                coefficients=x.new_zeros((n, 0)),
                topk_indices=torch.zeros((n, 1), device=x.device, dtype=torch.long),
                topk_values=x.new_zeros((n, 1)),
                activation_space=True, coefficients_sum_to_one=False, branch="shared",
            )
        mode = self.so2_fusion_mode
        if mode == "staged":
            return self._forward_staged(x, R, mole_globals, latents, wigner_D_all)
        if mode == "streamed_m_major_ref":
            return self._forward_streamed_m_major_ref(x, R, mole_globals, latents, wigner_D_all)
        if getattr(mole_globals, "activation_space", False) or getattr(mole_globals, "top1_independent", False):
            from .so2_backend import activation_forward

            result = activation_forward(
                self, x, R, mole_globals, latents, wigner_D_all,
                fused=mode == "streamed_m_major_fused_p0",
            )
            if result is not None:
                return result
        elif mode == "streamed_m_major_fused_p0":
            from .so2_backend import dense_forward

            result = dense_forward(self, x, R, mole_globals, latents, wigner_D_all)
            if result is not None:
                return result
        return self._forward_streamed_m_major_grouped(x, R, mole_globals, latents, wigner_D_all)

    def _forward_staged(self, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
        n, _ = x.shape
        if self.radial_emb:
            if latents is None:
                raise ValueError("SO2_Linear requires latents when radial_emb=True.")
            weights = self.radial_emb(latents)
        x_ = torch.zeros_like(x)

        # === 1. 旋转矩阵准备 (Rotate Control) ===
        if wigner_D_all is None:
            # 只有当需要 rotate_in 或者 rotate_out 时才必须计算 D
            if (self.rotate_in or self.rotate_out) and self.l_max > 0:
                angle = xyz_to_angles(R[:, [1, 2, 0]])
                wigner_D_all = _make_wigner_rotation(
                    self.l_max,
                    angle[0],
                    angle[1],
                    torch.zeros_like(angle[0]),
                    self.wigner_apply_mode,
                )

        # === 2. Rotate In (Global -> Local) ===
        groups = defaultdict(list)
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            groups[l].append((mul, slice_info))
            if l == 0:
                x_[:, slice_info] = x[:, slice_info]

        for l, group in groups.items():
            if l == 0 or not group:
                continue
            muls, slices = zip(*group)

            # --- Flag Check: 如果 rotate_in 为 False，直接复制不旋转 ---
            if not self.rotate_in:
                for mul, sl in group:
                    x_[:, sl] = x[:, sl]
                continue
            # ----------------------------------------------------

            x_parts = [x[:, sl].reshape(n, mul, 2 * l + 1) for mul, sl in group]
            x_combined = torch.cat(x_parts, dim=1)
            rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
            transformed = torch.bmm(x_combined, rot_mat)
            for part, slice_info, mul in zip(transformed.split(muls, dim=1), slices, muls):
                x_[:, slice_info] = part.flatten(1)

        # === 3. Convolution (Linear / MoE) ===
        out = torch.zeros(n, self.irreps_out.dim, dtype=x.dtype, device=x.device)
        for m in range(self.m_max + 1):
            radial_weight = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(
                1) if self.radial_emb else 1.

            if m == 0:
                # MoE Logic for m=0
                inp = x_[:, self.m_in_mask[m]]
                if self.front and self.radial_emb:
                    # mole_globals passed here
                    out[:, self.m_out_mask[m]] += self.fc_m0(inp * radial_weight.squeeze(1), mole_globals)
                elif self.radial_emb:
                    out[:, self.m_out_mask[m]] += self.fc_m0(inp, mole_globals) * radial_weight.squeeze(1)
                else:
                    out[:, self.m_out_mask[m]] += self.fc_m0(inp, mole_globals)
            else:
                # MoE Logic for m>0
                x_m_in = x_[:, self.m_in_mask[m]].unflatten(1, (-1, 2)).transpose(1, 2).contiguous()

                if self.front and self.radial_emb:
                    x_m_in.mul_(radial_weight)
                    # mole_globals passed here
                    linear_output = self.m_linear[m - 1](x_m_in, mole_globals)
                elif self.radial_emb:
                    linear_output = self.m_linear[m - 1](x_m_in, mole_globals)
                    linear_output.mul_(radial_weight)
                else:
                    linear_output = self.m_linear[m - 1](x_m_in, mole_globals)

                final_addition = linear_output.transpose(1, 2).contiguous().flatten(1)
                out[:, self.m_out_mask[m]] += final_addition

        # === 4. Rotate Out (Local -> Global) ===
        # --- Flag Check: 如果 rotate_out 为 False，直接返回 ---
        if not self.rotate_out:
            return out.contiguous(), wigner_D_all
        # --------------------------------------------------

        out_groups = defaultdict(list)
        for (mul, (l, p)), slice_info in zip(self.irreps_out, self.irreps_out.slices()):
            if l > 0:
                out_groups[l].append((mul, slice_info))

        for l, group in out_groups.items():
            muls, slices = zip(*group)
            out_parts = [out[:, sl].reshape(n, mul, self.dims[l]) for mul, sl in group]
            out_combined = torch.cat(out_parts, dim=1)
            rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
            rotated = torch.bmm(out_combined, rot_mat.transpose(1, 2))
            for part, slice_info, mul in zip(rotated.split(muls, dim=1), slices, muls):
                out[:, slice_info] = part.flatten(1)

        return out.contiguous(), wigner_D_all

    def _ensure_wigner_rotation(self, R, wigner_D_all):
        if wigner_D_all is not None:
            return wigner_D_all
        if (self.rotate_in or self.rotate_out) and self.l_max > 0:
            angle = xyz_to_angles(R[:, [1, 2, 0]])
            return _make_wigner_rotation(
                self.l_max,
                angle[0],
                angle[1],
                torch.zeros_like(angle[0]),
                self.wigner_apply_mode,
            )
        return None

    def _direct_rotate_pack_m(self, x, m: int, wigner_D_all):
        n = x.shape[0]
        # Split the input once. Repeated x[:, slice] views each allocate a
        # full-width zero gradient (including the redundant leading slice).
        specs = tuple(self.irreps_in)
        widths = [mul * (2 * ir.l + 1) for mul, ir in specs]
        blocks = torch.split(x, widths, dim=-1) if len(widths) > 1 else (x,)
        parts = []
        for (mul, (l, p)), block in zip(specs, blocks):
            if l < m:
                continue
            x_l = block.reshape(n, mul, 2 * l + 1)
            if m == 0:
                if l == 0 or not self.rotate_in:
                    parts.append(x_l.select(-1, l))
                else:
                    rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
                    parts.append(torch.einsum("ncd,nd->nc", x_l, rot_mat.select(-1, l)))
            else:
                local_rows = [l - m, l + m]
                if not self.rotate_in:
                    pair = x_l.index_select(-1, torch.tensor(local_rows, device=x.device))
                else:
                    rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
                    pair = torch.einsum("ncd,ndp->ncp", x_l, rot_mat[:, :, local_rows])
                parts.append(pair)
        if m == 0:
            return torch.cat(parts, dim=1)
        return torch.cat(parts, dim=1).transpose(1, 2).contiguous()

    def _accumulate_m0_output(self, out, y_m0, wigner_D_all):
        n = out.shape[0]
        specs = tuple(self.irreps_out)
        blocks = torch.split(y_m0, [mul for mul, _ in specs], dim=-1) if len(specs) > 1 else (y_m0,)
        parts = []
        for (mul, (l, p)), y_l in zip(specs, blocks):
            if l == 0 or not self.rotate_out:
                zero = torch.zeros_like(y_l)
                contribution = torch.stack([y_l if i == l else zero for i in range(2 * l + 1)], dim=-1)
            else:
                rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
                contribution = y_l.unsqueeze(-1) * rot_mat.select(-1, l).unsqueeze(1)
            parts.append(contribution.reshape(n, mul * (2 * l + 1)))
        # Keep the caller's in-place accumulation contract, but assemble once
        # instead of building a CopySlices chain on views of the output buffer.
        out.add_(torch.cat(parts, dim=-1))

    def _accumulate_m_output(self, out, y_m, m: int, wigner_D_all):
        n = out.shape[0]
        channel_start = 0
        for (mul, (l, p)), slice_info in zip(self.irreps_out, self.irreps_out.slices()):
            if l < m:
                continue
            y_l = y_m[:, :, channel_start:channel_start + mul]
            channel_start += mul
            local_rows = [l - m, l + m]
            out_l = out[:, slice_info].reshape(n, mul, 2 * l + 1)
            if not self.rotate_out:
                out_l[:, :, local_rows] += y_l.transpose(1, 2)
            else:
                rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
                out_l += torch.einsum("npm,ndp->nmd", y_l, rot_mat[:, :, local_rows])

    def _forward_streamed_m_major_ref(self, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
        wigner_D_all = self._ensure_wigner_rotation(R, wigner_D_all)
        n, _ = x.shape
        if self.radial_emb and latents is None:
            raise ValueError("SO2_Linear streamed path requires latents when radial_emb=True.")
        weights = self.radial_emb(latents) if self.radial_emb else None
        out = torch.zeros(n, self.irreps_out.dim, dtype=x.dtype, device=x.device)

        for m in range(self.m_max + 1):
            radial_weight = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(
                1) if self.radial_emb else 1.

            if m == 0:
                inp = self._direct_rotate_pack_m(x, m, wigner_D_all)
                if self.front and self.radial_emb:
                    y_m = self.fc_m0(inp * radial_weight.squeeze(1), mole_globals)
                elif self.radial_emb:
                    y_m = self.fc_m0(inp, mole_globals) * radial_weight.squeeze(1)
                else:
                    y_m = self.fc_m0(inp, mole_globals)
                self._accumulate_m0_output(out, y_m, wigner_D_all)
                continue

            x_m_in = self._direct_rotate_pack_m(x, m, wigner_D_all)
            if self.front and self.radial_emb:
                x_m_in = x_m_in * radial_weight
                linear_output = self.m_linear[m - 1](x_m_in, mole_globals)
            elif self.radial_emb:
                linear_output = self.m_linear[m - 1](x_m_in, mole_globals)
                linear_output = linear_output * radial_weight
            else:
                linear_output = self.m_linear[m - 1](x_m_in, mole_globals)

            self._accumulate_m_output(out, linear_output, m, wigner_D_all)

        return out.contiguous(), wigner_D_all


    def forward_expert_routes(self, x, R, expert_index: torch.Tensor, latents=None, wigner_D_all=None):
        """Evaluate SO2 TP rows with raw expert weights selected by expert_index.

        expert_index is the dispatch class for each row. It indexes routed
        experts directly and deliberately bypasses graph-level coefficient
        weight fusion.
        """
        expert_index = expert_index.to(device=x.device, dtype=torch.long).reshape(-1)
        n, _ = x.shape
        if expert_index.numel() != n:
            raise ValueError(f"expert_index has {expert_index.numel()} rows, but input has {n} rows.")
        if self.radial_emb and latents is None:
            raise ValueError("SO2_Linear expert-route path requires latents when radial_emb=True.")

        wigner_D_all = self._ensure_wigner_rotation(R, wigner_D_all)
        weights = self.radial_emb(latents) if self.radial_emb else None
        rot_blocks = self._make_wigner_block_cache(wigner_D_all)
        input_groups = self._gather_input_l_groups(x)
        out_groups = self._alloc_output_l_groups(n, dtype=x.dtype, device=x.device)

        for m in range(self.m_max + 1):
            radial_weight = (
                weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(1)
                if self.radial_emb else 1.
            )

            if m == 0:
                inp = self._assemble_grouped_m0_input(input_groups, rot_blocks, n, x)
                if self.front and self.radial_emb:
                    y_m = self.fc_m0.apply_experts(inp * radial_weight.squeeze(1), expert_index)
                elif self.radial_emb:
                    y_m = self.fc_m0.apply_experts(inp, expert_index) * radial_weight.squeeze(1)
                else:
                    y_m = self.fc_m0.apply_experts(inp, expert_index)
                self._accumulate_grouped_m0_output_(out_groups, y_m, rot_blocks)
                continue

            x_m_in = self._assemble_grouped_pair_input(input_groups, rot_blocks, m, n, x)
            if self.front and self.radial_emb:
                x_m_in = x_m_in * radial_weight
                linear_output = self.m_linear[m - 1].forward_experts(x_m_in, expert_index)
            elif self.radial_emb:
                linear_output = self.m_linear[m - 1].forward_experts(x_m_in, expert_index)
                linear_output = linear_output * radial_weight
            else:
                linear_output = self.m_linear[m - 1].forward_experts(x_m_in, expert_index)

            self._accumulate_grouped_pair_output_(out_groups, linear_output, rot_blocks, m)

        out = self._materialize_output_l_groups(out_groups, n=n, dtype=x.dtype, device=x.device)
        return out.contiguous(), wigner_D_all

    def _make_wigner_block_cache(self, wigner_D_all) -> Dict[int, torch.Tensor]:
        if wigner_D_all is None:
            return {}
        return {
            l: _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
            for l in self._active_rot_l
        }

    def _gather_input_l_groups(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        return {
            l: _gather_so2_l_group(x, plan)
            for l, plan in self._in_group_plans.items()
        }

    def _alloc_output_l_groups(self, n: int, *, dtype, device) -> Dict[int, torch.Tensor]:
        return {
            l: torch.zeros((n, plan.total_mul, plan.dims), dtype=dtype, device=device)
            for l, plan in self._out_group_plans.items()
        }

    def _materialize_output_l_groups(self, out_groups: Dict[int, torch.Tensor], *, n: int, dtype, device) -> torch.Tensor:
        out = torch.zeros((n, self.irreps_out.dim), dtype=dtype, device=device)
        for entry in self._out_entry_plans:
            group_view = out_groups[entry.l][:, entry.group_start:entry.group_start + entry.mul, :]
            # Spell the trailing extent out instead of inferring it: with n == 0
            # (an empty node/edge stream, e.g. a single-atom record whose active
            # set is empty for some l) reshape(0, -1) is ambiguous and raises.
            # mul * dims is exactly what -1 resolves to whenever n > 0.
            out[:, entry.slice_info] = group_view.reshape(
                n, int(entry.mul) * int(group_view.shape[-1])
            )
        return out

    def _pack_group_m0(self, x_group: torch.Tensor, l: int, rot_block: Optional[torch.Tensor]) -> torch.Tensor:
        if x_group.numel() == 0:
            return x_group.new_empty((x_group.shape[0], x_group.shape[1]))
        if l == 0 or not self.rotate_in or rot_block is None:
            return x_group[:, :, l]
        return torch.einsum("ncd,nd->nc", x_group, rot_block[:, :, l])

    def _pack_group_pair(self, x_group: torch.Tensor, l: int, m: int, rot_block: Optional[torch.Tensor]) -> torch.Tensor:
        if x_group.numel() == 0:
            return x_group.new_empty((x_group.shape[0], 2, x_group.shape[1]))
        rows = [l - m, l + m]
        if not self.rotate_in or rot_block is None:
            return x_group[:, :, rows].transpose(1, 2).contiguous()
        return torch.einsum("ncd,ndp->npc", x_group, rot_block[:, :, rows])

    def _accumulate_group_m0_(self, out_group: torch.Tensor, y_group: torch.Tensor, l: int, rot_block: Optional[torch.Tensor]) -> None:
        if y_group.numel() == 0:
            return
        if l == 0 or not self.rotate_out or rot_block is None:
            out_group[:, :, l] += y_group
            return
        out_group += y_group.unsqueeze(-1) * rot_block[:, :, l].unsqueeze(1)

    def _accumulate_group_pair_(self, out_group: torch.Tensor, y_group: torch.Tensor, l: int, m: int, rot_block: Optional[torch.Tensor]) -> None:
        if y_group.numel() == 0:
            return
        rows = [l - m, l + m]
        if not self.rotate_out or rot_block is None:
            out_group[:, :, rows] += y_group.transpose(1, 2)
            return
        out_group += torch.einsum("npc,ndp->ncd", y_group, rot_block[:, :, rows])

    def _assemble_grouped_m0_input(
            self,
            input_groups: Dict[int, torch.Tensor],
            rot_blocks: Dict[int, torch.Tensor],
            n: int,
            x_template: torch.Tensor,
    ) -> torch.Tensor:
        packed_by_l = {}
        for l, x_group in input_groups.items():
            packed_by_l[l] = self._pack_group_m0(x_group, l, rot_blocks.get(l))

        parts = [
            packed_by_l[entry.l][:, entry.group_start:entry.group_start + entry.mul]
            for entry in self._in_entries_by_m[0]
        ]
        if not parts:
            return x_template.new_empty((n, 0))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1)

    def _assemble_grouped_pair_input(
            self,
            input_groups: Dict[int, torch.Tensor],
            rot_blocks: Dict[int, torch.Tensor],
            m: int,
            n: int,
            x_template: torch.Tensor,
    ) -> torch.Tensor:
        packed_by_l = {}
        for l, x_group in input_groups.items():
            if l < m:
                continue
            packed_by_l[l] = self._pack_group_pair(x_group, l, m, rot_blocks.get(l))

        parts = [
            packed_by_l[entry.l][:, :, entry.group_start:entry.group_start + entry.mul]
            for entry in self._in_entries_by_m[m]
        ]
        if not parts:
            return x_template.new_empty((n, 2, 0))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=2)

    def _accumulate_grouped_m0_output_(
            self,
            out_groups: Dict[int, torch.Tensor],
            y_m0: torch.Tensor,
            rot_blocks: Dict[int, torch.Tensor],
    ) -> None:
        cursor = 0
        for entry in self._out_entries_by_m[0]:
            y_entry = y_m0[:, cursor:cursor + entry.mul]
            cursor += entry.mul
            out_view = out_groups[entry.l][:, entry.group_start:entry.group_start + entry.mul, :]
            self._accumulate_group_m0_(out_view, y_entry, entry.l, rot_blocks.get(entry.l))

    def _accumulate_grouped_pair_output_(
            self,
            out_groups: Dict[int, torch.Tensor],
            y_m: torch.Tensor,
            rot_blocks: Dict[int, torch.Tensor],
            m: int,
    ) -> None:
        cursor = 0
        for entry in self._out_entries_by_m[m]:
            y_entry = y_m[:, :, cursor:cursor + entry.mul]
            cursor += entry.mul
            out_view = out_groups[entry.l][:, entry.group_start:entry.group_start + entry.mul, :]
            self._accumulate_group_pair_(out_view, y_entry, entry.l, m, rot_blocks.get(entry.l))

    def _forward_streamed_m_major_grouped(self, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
        wigner_D_all = self._ensure_wigner_rotation(R, wigner_D_all)
        n, _ = x.shape
        if self.radial_emb and latents is None:
            raise ValueError("SO2_Linear grouped streamed path requires latents when radial_emb=True.")
        weights = self.radial_emb(latents) if self.radial_emb else None
        rot_blocks = self._make_wigner_block_cache(wigner_D_all)
        input_groups = self._gather_input_l_groups(x)
        out_groups = self._alloc_output_l_groups(n, dtype=x.dtype, device=x.device)

        for m in range(self.m_max + 1):
            radial_weight = (
                weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(1)
                if self.radial_emb else 1.
            )

            if m == 0:
                inp = self._assemble_grouped_m0_input(input_groups, rot_blocks, n, x)
                if self.front and self.radial_emb:
                    y_m = self.fc_m0(inp * radial_weight.squeeze(1), mole_globals)
                elif self.radial_emb:
                    y_m = self.fc_m0(inp, mole_globals) * radial_weight.squeeze(1)
                else:
                    y_m = self.fc_m0(inp, mole_globals)
                self._accumulate_grouped_m0_output_(out_groups, y_m, rot_blocks)
                continue

            x_m_in = self._assemble_grouped_pair_input(input_groups, rot_blocks, m, n, x)
            if self.front and self.radial_emb:
                x_m_in = x_m_in * radial_weight
                linear_output = self.m_linear[m - 1](x_m_in, mole_globals)
            elif self.radial_emb:
                linear_output = self.m_linear[m - 1](x_m_in, mole_globals)
                linear_output = linear_output * radial_weight
            else:
                linear_output = self.m_linear[m - 1](x_m_in, mole_globals)

            self._accumulate_grouped_pair_output_(out_groups, linear_output, rot_blocks, m)

        out = self._materialize_output_l_groups(out_groups, n=n, dtype=x.dtype, device=x.device)
        return out.contiguous(), wigner_D_all


class SO2_m_Linear(torch.nn.Module):
    """
    SO(2) Convolution for a specific order m > 0.
    """

    def __init__(
            self,
            m,
            irreps_in,
            irreps_out,
            use_interpolation: bool = False,
            num_experts: int = 8,  # Added
            num_shared_experts: int = 1, # Added
            mole_linear_mode=None,
            mole_expert_parameterization="full",
            mole_expert_rank=64,
            so2_parity: str = "none",
    ):
        super(SO2_m_Linear, self).__init__()
        self.m = m
        self.num_in_channel = sum(mul for mul, (l, p) in irreps_in if l >= m)
        self.num_out_channel = sum(mul for mul, (l, p) in irreps_out if l >= m)

        # MODIFICATION: MOLE Logic with bias=False (original was bias=False)
        if use_interpolation:
            self.fc = InterpolationBlock(self.num_in_channel, 2 * self.num_out_channel, bias=False)
            self.is_mole = False
        else:
            self.fc = MOLELinear(
                self.num_in_channel,
                2 * self.num_out_channel,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                bias=False,
                mole_linear_mode=mole_linear_mode,
                mole_expert_parameterization=mole_expert_parameterization,
                mole_expert_rank=mole_expert_rank,
            )
            if self.fc.num_experts:
                self.fc.scale_expert_weights_(1 / math.sqrt(2))
            self.is_mole = True

        if normalize_so2_parity(so2_parity) == "enforce":
            if not self.is_mole:
                raise ValueError("so2_parity='enforce' does not support interpolation m blocks")
            self.fc.set_parity_masks(*parity_masks(irreps_in, irreps_out, m))

    def forward(self, x_m, mole_globals: MOLEGlobals):  # Added mole_globals
        # x_m ~ [N, 2, n_channels]
        if self.is_mole:
            x_m = self.fc(x_m, mole_globals)
        elif getattr(mole_globals, "branch", "all") == "routed":
            # a non-MoLE block (interpolation) belongs to the shared branch
            x_m = x_m.new_zeros(*x_m.shape[:-1], 2 * self.num_out_channel)
        else:
            x_m = self.fc(x_m)

        return self._finish_linear_output(x_m)


    def forward_experts(self, x_m, expert_index: torch.Tensor):
        if not self.is_mole:
            raise RuntimeError("SO2_m_Linear.forward_experts requires a MOLELinear block, not interpolation.")
        x_m = self.fc.apply_experts(x_m, expert_index)
        return self._finish_linear_output(x_m)

    def _finish_linear_output(self, x_m):
        return complex_pair_output(x_m, self.num_out_channel)

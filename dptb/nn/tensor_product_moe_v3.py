
from e3nn.o3 import xyz_to_angles, Irreps
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from e3nn.o3 import Linear as e3nn_Linear
from torch.nn import Linear
import os
import torch.nn.functional as F
from collections import defaultdict
from .tensor_product import InterpolationBlock, RadialFunction, complex_pair_output
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
        "streamed_m_major_persistent_grouped_p1",
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

def _route_layout_token(tensor):
    """Host-only metadata for a cached integer routing view; no device sync.

    A retained source tensor prevents allocator address reuse. Inference tensors
    have no version counter; conservatively do not cache them. Mutation through
    .data or external raw pointers is outside this contract (as for autograd).
    """
    try:
        return (tensor.data_ptr(), int(tensor._version), tuple(tensor.shape),
                tuple(tensor.stride()), str(tensor.device), tensor.dtype)
    except RuntimeError:
        return None


class MOLEGlobals:
    """Stores routing information for the current forward pass."""

    def __init__(
            self,
            coefficients=None,
            sizes=None,
            split_sizes=None,
            graph_index=None,
            topk_indices=None,
            topk_values=None,
            activation_space=False,
            coefficients_sum_to_one=False,
            branch="all",
    ):
        # Activation-space dispatch: mix experts as sum_e c_e (x W_e) rather than
        # materialising one [out, in] weight per route token.  Set by the caller
        # when routing is per-edge, where the weight-space path cannot fit.
        self.activation_space = bool(activation_space)
        # Only set where the router contract guarantees it (softmax over the
        # gathered top-k logits).  It licenses folding the shared expert into the
        # routed weights, which is exact only when the coefficients sum to 1.
        self.coefficients_sum_to_one = bool(coefficients_sum_to_one)
        # Which part of every MoLE layer this pass computes (activation-space routing only):
        # "all" = routed experts + shared expert (+ the non-MoLE blocks of an SO2 layer);
        # "routed" = the routed experts only (no shared expert, no non-MoLE block);
        # "shared" = the shared expert and the non-MoLE blocks only (routed coefficients unused).
        # SO2SharedPostActivationMixer activates the shared branch and every routed slot separately.
        if branch not in ("all", "routed", "shared"):
            raise ValueError("MOLEGlobals.branch must be all, routed or shared; got %r" % (branch,))
        if branch != "all" and not self.activation_space:
            raise ValueError("MOLEGlobals.branch=%r needs activation-space (per-row) routing" % (branch,))
        self.branch = branch
        # Keyed on the slot index explicitly: the other caches on this object are
        # content-blind and would alias slot 1 onto slot 0's permutation.
        self._expert_slot_layout_cache = {}
        self._expert_slot_layout_sources = {}
        self.coefficients = coefficients  # [Batch, Num_Experts]
        self.topk_indices = topk_indices
        self.topk_values = topk_values
        self.sizes = sizes  # [Batch] (Edge counts per system)
        # Explicit split sizes preserve the original split-loop contract and
        # must stay authoritative across all MOLELinear backends.
        self.graph_index = None if split_sizes is not None else graph_index
        self._sizes_tensor = self._normalize_sizes_tensor(sizes, split_sizes)
        self.split_sizes = self._normalize_split_sizes(sizes, split_sizes)
        self._expanded_graph_index_cache = {}
        self._indexed_flat_permutation_cache = {}
        self._indexed_segment_ptr_cache = {}
        self._indexed_inputs_are_sorted = False

    def expert_slot_layout(self, slot: int, expert_index, num_experts: int):
        """Sort order, inverse, and segment pointer for one top-k slot.

        Derived once per forward and shared by every MOLELinear, since
        topk_indices[:, slot] does not vary across modules.
        """
        key = (int(slot), str(expert_index.device), int(expert_index.numel()),
               int(num_experts))
        cached = self._expert_slot_layout_cache.get(key)
        token = _route_layout_token(expert_index)
        sources = self._expert_slot_layout_sources
        source = sources.get(key)
        if cached is not None and token is not None and source is not None and source[0] == token:
            return cached
        eidx = expert_index.reshape(-1).to(dtype=torch.long)
        order = torch.argsort(eidx, stable=True)
        inverse = torch.empty_like(order)
        inverse.scatter_(
            0, order,
            torch.arange(order.numel(), device=order.device, dtype=order.dtype),
        )
        counts = torch.bincount(eidx, minlength=int(num_experts))
        ptr = torch.zeros(int(num_experts) + 1, dtype=torch.long, device="cpu")
        ptr[1:] = torch.cumsum(counts.to("cpu"), dim=0)
        cached = (order, inverse, ptr.contiguous(), eidx.index_select(0, order))
        if token is not None:
            self._expert_slot_layout_cache[key] = cached
            sources[key] = (token, expert_index)
        else:
            self._expert_slot_layout_cache.pop(key, None)
            sources.pop(key, None)
        return cached

    @staticmethod
    def _normalize_split_sizes(sizes, split_sizes):
        if split_sizes is not None:
            if torch.is_tensor(split_sizes):
                if split_sizes.device.type != "cpu":
                    return None
                return MOLEGlobals._tensor_to_split_tuple(split_sizes)
            return tuple(int(v) for v in split_sizes)
        if sizes is None:
            return None
        if torch.is_tensor(sizes):
            if sizes.device.type != "cpu":
                return None
            return MOLEGlobals._tensor_to_split_tuple(sizes)
        return tuple(int(v) for v in sizes)

    @staticmethod
    def _normalize_sizes_tensor(sizes, split_sizes):
        values = split_sizes if split_sizes is not None else sizes
        if values is None or not torch.is_tensor(values):
            return None
        return values.detach().reshape(-1).to(dtype=torch.long)

    @staticmethod
    def _tensor_to_split_tuple(values):
        try:
            # Fast path for plain host metadata (e.g. precomputed LEM split
            # sizes): a direct storage read is legal even inside torch.func
            # regions, whereas detach/reshape would first lift the tensor into
            # a storageless interpreter wrapper.
            if values.device.type == "cpu" and values.dim() <= 1 and not values.requires_grad:
                return tuple(int(v) for v in values.tolist())
        except RuntimeError:
            pass
        values = _functorch_plain(values).detach().reshape(-1)
        if values.device.type != "cpu":
            # Compatibility fallback for direct callers that still pass CUDA sizes.
            values = values.cpu()
        return tuple(int(v) for v in values.tolist())

    def indexed_flat_permutation(self, graph_index: torch.Tensor, x: torch.Tensor):
        flat_graph_index = _expand_graph_index_cached(graph_index, x, self).reshape(-1).to(dtype=torch.long)
        key = (
            str(flat_graph_index.device),
            str(flat_graph_index.dtype),
            int(flat_graph_index.numel()),
            tuple(int(v) for v in x.shape[1:-1]),
        )
        cached = self._indexed_flat_permutation_cache.get(key)
        if cached is None:
            if flat_graph_index.numel() <= 1 or _functorch_plain(torch.all(flat_graph_index[1:] >= flat_graph_index[:-1])).item():
                permute_idx = None
                unpermute_idx = None
                sorted_graph_index = flat_graph_index
            else:
                permute_idx = torch.argsort(flat_graph_index, stable=True)
                sorted_graph_index = flat_graph_index.index_select(0, permute_idx)
                unpermute_idx = torch.empty_like(permute_idx)
                unpermute_idx.scatter_(
                    0,
                    permute_idx,
                    torch.arange(permute_idx.numel(), device=permute_idx.device, dtype=permute_idx.dtype),
                )
            cached = (permute_idx, unpermute_idx, sorted_graph_index)
            self._indexed_flat_permutation_cache[key] = cached
        return cached

    def sorted_indexed_view(self, graph_index: torch.Tensor, x: torch.Tensor):
        permute_idx, unpermute_idx, sorted_graph_index = self.indexed_flat_permutation(graph_index, x)
        sorted_view = MOLEGlobals(
            coefficients=self.coefficients,
            topk_indices=self.topk_indices,
            topk_values=self.topk_values,
            sizes=None,
            split_sizes=None,
            graph_index=sorted_graph_index,
        )
        sorted_view._indexed_inputs_are_sorted = True
        return permute_idx, unpermute_idx, sorted_view

    def indexed_segment_ptr(
            self,
            sorted_graph_index: torch.Tensor,
            num_groups: int,
            *,
            prefer_cpu: bool = True,
    ) -> torch.Tensor:
        target_device = torch.device("cpu") if prefer_cpu else sorted_graph_index.device
        key = (
            str(sorted_graph_index.device),
            int(sorted_graph_index.numel()),
            int(num_groups),
            str(target_device),
        )
        cached = self._indexed_segment_ptr_cache.get(key)
        if cached is None:
            counts = torch.bincount(sorted_graph_index, minlength=num_groups)
            ptr = torch.zeros(num_groups + 1, dtype=torch.long, device=counts.device)
            ptr[1:] = torch.cumsum(counts, dim=0)
            cached = ptr.to(device=target_device, dtype=torch.long).contiguous()
            self._indexed_segment_ptr_cache[key] = cached
        return cached


class _RowPermutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, order, inverse):
        ctx.save_for_backward(order, inverse)
        return x.index_select(0, order)

    @staticmethod
    def backward(ctx, grad):
        order, inverse = ctx.saved_tensors
        return permute_rows(grad, inverse, order), None, None


def permute_rows(x: torch.Tensor, order: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """x[order] for a permutation ``order`` of the rows of x whose inverse is ``inverse``.

    The backward gathers the gradient with ``inverse``, where index_select's backward
    zero-fills a buffer and index_adds into it.  Values and gradients are those of
    index_select; only a bijection qualifies (a gather with repeated rows must sum).
    Under a torch.func transform this is index_select itself."""
    if getattr(torch._C, "_are_functorch_transforms_active", lambda: False)():
        return x.index_select(0, order)
    return _RowPermutation.apply(x, order, inverse)


def _functorch_plain(values):
    """Unwrap functorch-wrapped non-differentiable metadata tensors.

    Routing indices / split sizes are integer metadata with no tangent; under
    torch.func transforms (e.g. the pixel-meanflow jvp backend) they can still
    arrive as storageless interpreter wrappers, which break host-side
    conversions (.tolist()/.item()). Unwrapping is value-exact for such
    tensors; on failure the original tensor is returned unchanged.
    """
    try:
        from torch._C import _functorch as _ft
        while _ft.is_functorch_wrapped_tensor(values):
            values = _ft.get_unwrapped(values)
        return values.detach()
    except Exception:
        return values


def _mole_split_sizes(mole_globals, n_rows: int):
    split_sizes = getattr(mole_globals, "split_sizes", None)
    if split_sizes is None:
        sizes_tensor = getattr(mole_globals, "_sizes_tensor", None)
        if sizes_tensor is None:
            split_sizes = (n_rows,)
        else:
            split_sizes = MOLEGlobals._tensor_to_split_tuple(sizes_tensor)
    if sum(split_sizes) != n_rows:
        raise ValueError(
            f"MOLE split sizes sum to {sum(split_sizes)}, but input has {n_rows} rows."
        )
    return split_sizes


def _mole_graph_index(mole_globals, n_rows: int, *, device):
    """Return sorted graph ids per row, matching the existing split-loop semantics."""
    graph_index = getattr(mole_globals, "graph_index", None)
    if graph_index is not None:
        graph_index = graph_index.to(device=device, dtype=torch.long).reshape(-1)
        if graph_index.numel() == n_rows:
            return graph_index
        raise ValueError(
            f"MOLE graph_index has {graph_index.numel()} rows, but input has {n_rows} rows."
        )

    sizes_tensor = getattr(mole_globals, "_sizes_tensor", None)
    if sizes_tensor is not None:
        cache = getattr(mole_globals, "_graph_index_cache", None)
        if cache is None:
            cache = {}
            setattr(mole_globals, "_graph_index_cache", cache)
        key = (str(device), "tensor_sizes", int(sizes_tensor.numel()))
        graph_index = cache.get(key)
        if graph_index is None:
            sizes = sizes_tensor.to(device=device, dtype=torch.long)
            graph_index = torch.repeat_interleave(
                torch.arange(sizes.shape[0], dtype=torch.long, device=device),
                sizes,
                output_size=n_rows,
            )
            cache[key] = graph_index
        if graph_index.numel() == n_rows:
            return graph_index
        raise ValueError(
            f"MOLE sizes expand to {graph_index.numel()} rows, but input has {n_rows} rows."
        )

    split_sizes = _mole_split_sizes(mole_globals, n_rows)
    cache = getattr(mole_globals, "_graph_index_cache", None)
    if cache is None:
        cache = {}
        setattr(mole_globals, "_graph_index_cache", cache)

    key = (str(device), split_sizes)
    graph_index = cache.get(key)
    if graph_index is None:
        sizes = torch.tensor(split_sizes, dtype=torch.long, device=device)
        # cuEquivariance indexed_linear requires sorted indices; the split_sizes
        # contract means rows are graph-contiguous, matching the old split loop.
        graph_index = torch.repeat_interleave(
            torch.arange(len(split_sizes), dtype=torch.long, device=device),
            sizes,
            output_size=n_rows,
        )
        cache[key] = graph_index
    return graph_index


def _expand_graph_index_for_leading_dims(graph_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand [E] graph ids to match x.reshape(-1, in_features)."""
    if x.ndim == 2:
        return graph_index

    expand_shape = [graph_index.shape[0]] + list(x.shape[1:-1])
    return graph_index.reshape(-1, *([1] * (x.ndim - 2))).expand(expand_shape).reshape(-1)


def _expand_graph_index_cached(
        graph_index: torch.Tensor,
        x: torch.Tensor,
        mole_globals: MOLEGlobals,
) -> torch.Tensor:
    if x.ndim == 2:
        return graph_index

    cache = getattr(mole_globals, "_expanded_graph_index_cache", None)
    if cache is None:
        cache = {}
        setattr(mole_globals, "_expanded_graph_index_cache", cache)

    key = (
        str(x.device),
        str(graph_index.device),
        str(graph_index.dtype),
        int(graph_index.numel()),
        tuple(int(v) for v in x.shape[1:-1]),
    )
    cached = cache.get(key)
    if cached is None:
        cached = _expand_graph_index_for_leading_dims(graph_index, x)
        cache[key] = cached
    return cached



def _expand_route_index_for_leading_dims(route_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand one route/expert id per row to match x.reshape(-1, x.shape[-1])."""
    route_index = route_index.reshape(-1).to(device=x.device, dtype=torch.long)
    if x.ndim == 2:
        return route_index
    expand_shape = [route_index.shape[0]] + list(x.shape[1:-1])
    return route_index.reshape(-1, *([1] * (x.ndim - 2))).expand(expand_shape).reshape(-1)


def _index_select_wigner_edges(wigner_D_all, index: torch.Tensor):
    if wigner_D_all is None:
        return None
    if isinstance(wigner_D_all, SO2WignerBlocks):
        return SO2WignerBlocks(block.index_select(0, index) for block in wigner_D_all.blocks)
    return wigner_D_all.index_select(0, index)


def _normalize_mole_linear_mode(mode: str) -> str:
    allowed = {"split_loop", "indexed_ref", "cueq_indexed_linear", "cublas_grouped"}
    if mode not in allowed:
        raise ValueError(f"mole_linear_mode must be one of {sorted(allowed)}, got {mode!r}")
    return mode



def _expert_route_indices_from_globals(
        mole_globals: MOLEGlobals,
        num_experts: int,
        n_rows: int,
        *,
        device: torch.device,
) -> torch.Tensor:
    """Return candidate expert ids per row, shape [n_rows, k]."""
    graph_index = _mole_graph_index(mole_globals, n_rows, device=device)
    topk_indices = getattr(mole_globals, "topk_indices", None)
    if topk_indices is not None:
        topk_indices = topk_indices.to(device=device, dtype=torch.long)
        if topk_indices.ndim != 2:
            raise ValueError(f"MOLE topk_indices must be [num_graphs, k], got {tuple(topk_indices.shape)}.")
        return topk_indices.index_select(0, graph_index)

    all_experts = torch.arange(num_experts, device=device, dtype=torch.long)
    return all_experts.reshape(1, num_experts).expand(n_rows, num_experts)


def router_z_loss(logits: torch.Tensor, sizes: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ST-MoE router z-loss: mean over route tokens of logsumexp(logits)**2, weighted by ``sizes``.

    It penalises the logit scale that saturates the gate, and keeps its gradient; the training objective adds it only
    when ``loss_options.train.router_z_loss_coef`` is > 0.
    """
    if logits.shape[0] == 0:
        return logits.new_zeros((), dtype=torch.float32)
    z = torch.logsumexp(logits.float(), dim=-1).square()
    if sizes is None:
        return z.mean()
    w = sizes.to(z.dtype).reshape(-1)
    return (z * w).sum() / w.sum().clamp_min(1e-12)


ROUTER_REGULARIZER_KEYS = (("router_z_loss", "last_router_z_loss"), ("router_aux_loss", "last_router_aux_loss"))


def write_router_regularizers(router: Optional[nn.Module], data: dict) -> None:
    """Expose the regularisers of the router's last forward to the loss (``data['router_z_loss']`` etc.)."""
    for key, attr in ROUTER_REGULARIZER_KEYS:
        value = getattr(router, attr, None) if router is not None else None
        if value is not None:
            data[key] = value


class MOLERouterV3(nn.Module):
    """Top-k router with aux-loss-free load balancing.

    Defaults reproduce the production router: raw logits, selection on ``sigmoid(logits) + expert_bias`` in training
    and on ``sigmoid(logits)`` in eval, softmax mixing over the selected logits.  The options below are part of the
    model configuration, so a checkpoint carries them:

    - ``logit_kind="cosine"``: ``z_e = logit_scale * cos(h, w_e)`` with ``h`` the hidden layer after SiLU and ``w_e``
      the e-th row of the last layer (its bias is unused); bounded, so the sigmoid never reaches its float32 1.0
      plateau (at raw logits of hundreds every expert ties at 1.0 and the selection falls to the bias / index).
    - ``select="logit"``: rank on the logits instead of their sigmoid (no saturation ties whatever the logit scale).
    - ``bias_at_eval``: keep the frozen ``expert_bias`` in the eval selection, so train and eval select alike.
    - ``bias_schedule``: ``"const"`` fixed sign-rule step; ``"follow_lr"`` step times lr / peak lr; ``"freeze_decay"``
      no bias updates once the learning rate falls below its peak (the optimizer publishes the ratio through
      :mod:`dptb.nn.moe_registry`).
    - ``select_noise``: std of Gaussian noise added to the selection scores in training only (noisy top-k); the
      mixing weights stay noise-free.
    - ``mixing_temperature`` (0924-stable): ``g = softmax(z_selected / T)``; changes mixing hardness only.
    - ``bias_freeze_after_step``: no bias updates from this committed optimizer step on (0 = never); the frozen bias
      keeps acting in the selection.
    - ``gate``: ``"renorm"`` (default) mixes the selected experts with a softmax over their own logits (sums to one);
      ``"full_softmax"`` takes the softmax over every routed expert and keeps the selected entries without
      renormalising (DPA3-MoE, Liu et al., npj Artif. Intell. 2026, eqs. 5-6), so the routed branch carries the
      router's probability mass (< 1) next to the shared expert.  ``coefficients_sum_to_one`` tells the caller which.
    - ``type_support`` (> 0, per-edge routing only): every bond type may only use a fixed set of ``type_support``
      experts, drawn by a deterministic hash of (bond type, expert, ``type_support_seed``); the router then selects
      its top-k inside that set (the mask acts on the selection scores and, for ``full_softmax``, on the softmax).
      Chemistry fixes the candidate experts, the router input (the prior descriptor) chooses among them, so no test
      edge can land in a (bond type, expert) cell the expert never trained on.  ``forward`` then needs ``bond_type``.
    """

    _LOGIT_KINDS = ("raw", "cosine")
    _SELECTS = ("sigmoid", "logit")
    _BIAS_SCHEDULES = ("const", "follow_lr", "freeze_decay")
    _GATES = ("renorm", "full_softmax")

    def __init__(self, in_features, num_experts=48, top_k=6,
                 aux_loss_free=True,
                 bias_update_speed=0.005,
                 full_expert_fast_path: bool = True,
                 mixing_temperature: float = 1.0,
                 logit_kind: str = "raw",
                 logit_scale: float = 10.0,
                 select: str = "sigmoid",
                 bias_at_eval: bool = False,
                 bias_schedule: str = "const",
                 select_noise: float = 0.0,
                 bias_freeze_after_step: int = 0,
                 gate: str = "renorm",
                 type_support: int = 0,
                 type_support_seed: int = 0):  # 修改1: 固定 Bias 更新速度，不再衰减
        super().__init__()
        if logit_kind not in self._LOGIT_KINDS:
            raise ValueError(f"logit_kind must be one of {self._LOGIT_KINDS}; got {logit_kind!r}")
        if select not in self._SELECTS:
            raise ValueError(f"select must be one of {self._SELECTS}; got {select!r}")
        if bias_schedule not in self._BIAS_SCHEDULES:
            raise ValueError(f"bias_schedule must be one of {self._BIAS_SCHEDULES}; got {bias_schedule!r}")
        if not float(logit_scale) > 0.0:
            raise ValueError(f"logit_scale must be positive; got {logit_scale!r}")
        if float(select_noise) < 0.0:
            raise ValueError(f"select_noise must be >= 0; got {select_noise!r}")
        if int(bias_freeze_after_step) < 0:
            raise ValueError(f"bias_freeze_after_step must be >= 0; got {bias_freeze_after_step!r}")
        if gate not in self._GATES:
            raise ValueError(f"gate must be one of {self._GATES}; got {gate!r}")
        type_support = int(type_support)
        if type_support < 0 or (type_support > 0 and (top_k is None or not int(top_k) <= type_support <= num_experts)):
            raise ValueError(f"type_support must be 0 or between top_k and num_experts; got {type_support!r} "
                             f"(top_k={top_k!r}, num_experts={num_experts!r})")
        self.top_k = top_k
        self.num_experts = num_experts
        self.aux_loss_free = aux_loss_free
        self.full_expert_fast_path = full_expert_fast_path
        # Temperature of the softmax that mixes the selected experts (and of the full-expert gate): the weights are
        # softmax(logits / T).  T > 1 keeps the mixing soft for a given logit gap; selection (sigmoid scores + the
        # balancing bias) does not depend on it.  T = 1 is the original gate.
        self.mixing_temperature = float(mixing_temperature)
        if not math.isfinite(self.mixing_temperature) or self.mixing_temperature <= 0:
            raise ValueError("mixing_temperature must be a finite positive number, got %r" % (mixing_temperature,))
        self.logit_kind = logit_kind
        self.logit_scale = float(logit_scale)
        self.select = select
        self.bias_at_eval = bool(bias_at_eval)
        self.bias_schedule = bias_schedule
        self.select_noise = float(select_noise)
        self.bias_freeze_after_step = int(bias_freeze_after_step)
        self.gate = gate
        self.type_support = type_support
        self.type_support_seed = int(type_support_seed)
        self._last_allowed = None   # [N, E] candidate mask of the last forward (type_support > 0), for checks
        # current_lr / peak_lr and the committed optimizer-step count, published by the optimizer after every step
        # (moe_registry); opt_step stays 0 with optimizers that do not publish
        self.bias_lr_scale = 1.0
        self.opt_step = 0
        # read-only statistics of the last *training* forward (validation never touches them); off by default
        self.record_train_stats = False
        self.last_train_stats = None

        # 固定的惩罚力度
        self.bias_update_speed = bias_update_speed

        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts)
        )

        self.register_buffer('expert_bias', torch.zeros(num_experts))
        effective_top_k = num_experts if top_k is None else min(top_k, num_experts)
        self.register_buffer('ema_load', torch.ones(num_experts) * (effective_top_k / num_experts))
        self._last_topk_indices = None
        self._last_topk_values = None
        self.last_router_z_loss = None

        from dptb.nn import moe_registry
        moe_registry.register_router(self)

        # 修改1: 删除了 step_count 等用于衰减的 Buffer

    def router_config(self) -> dict:
        return dict(logit_kind=self.logit_kind, logit_scale=self.logit_scale, select=self.select,
                    bias_at_eval=self.bias_at_eval, bias_schedule=self.bias_schedule,
                    bias_update_speed=float(self.bias_update_speed), select_noise=self.select_noise,
                    mixing_temperature=self.mixing_temperature, bias_freeze_after_step=self.bias_freeze_after_step,
                    gate=self.gate, type_support=self.type_support, type_support_seed=self.type_support_seed)

    @property
    def coefficients_sum_to_one(self) -> bool:
        """True when the top-k coefficients sum to one (licenses folding the shared expert into every slot)."""
        return self.gate == "renorm" or self.top_k is None or self.top_k >= self.num_experts

    def type_support_mask(self, bond_type: torch.Tensor) -> torch.Tensor:
        """[N] bond-type indices -> [N, E] bool, True on the ``type_support`` experts a bond type may use.  A fixed
        integer hash of (type, expert, seed) ranks the experts of every type and the top ``type_support`` form its
        set, so the set is a pure function of the type (same in every batch, step, process and checkpoint)."""
        m = 2147483647
        e = torch.arange(self.num_experts, device=bond_type.device, dtype=torch.int64).unsqueeze(0)
        tt = bond_type.reshape(-1, 1).to(torch.int64)
        h = (tt * 1000003 + e * 7919 + (self.type_support_seed + 1) * 104729) % m
        h = (h * 48271) % m
        h = ((h ^ (h >> 13)) * 69621) % m
        h = (h * 48271 + e) % m
        idx = torch.topk(h.to(torch.float64), k=self.type_support, dim=1).indices
        return torch.zeros(h.shape, dtype=torch.bool, device=bond_type.device).scatter_(1, idx, True)

    def _logits(self, global_features):
        if self.logit_kind == "cosine":
            hidden = self.net[1](self.net[0](global_features))
            return self.logit_scale * (
                F.normalize(hidden, dim=-1) @ F.normalize(self.net[2].weight, dim=-1).t()
            )
        return self.net(global_features)

    def _bias_step(self) -> float:
        speed = float(self.bias_update_speed)
        if self.bias_freeze_after_step > 0 and int(self.opt_step) >= self.bias_freeze_after_step:
            return 0.0
        if self.bias_schedule == "follow_lr":
            return speed * min(max(float(self.bias_lr_scale), 0.0), 1.0)
        if self.bias_schedule == "freeze_decay":
            return speed if float(self.bias_lr_scale) >= 0.999 else 0.0
        return speed

    def forward(self, global_features, sizes=None, bond_type=None, regularizer_weights=None):
        # 修改1: 删除了 Jitter (探索噪声) 的注入逻辑，完全依赖网络的自然 Logits
        logits = self._logits(global_features)
        allowed = None
        if self.type_support > 0:
            if bond_type is None or bond_type.numel() != logits.shape[0]:
                raise ValueError("type_support needs one bond type per routed row")
            allowed = self.type_support_mask(bond_type)
        self._last_allowed = allowed
        scores = torch.sigmoid(logits)
        # A dropped structure contributes neither task nor z-loss gradients.
        # Keep selection/load balancing on the original routes (no budget gate).
        self.last_router_z_loss = router_z_loss(
            logits, sizes if regularizer_weights is None else regularizer_weights)

        if self.top_k is None or self.top_k >= self.num_experts:
            # Full soft mixture: canonical slots keep activation-space dispatch
            # available regardless of the legacy fast-path flag. Selection bias
            # and noise have no role when every expert participates.
            probs = torch.softmax(self._tempered(logits), dim=-1)
            indices = torch.arange(self.num_experts, device=logits.device).expand(logits.shape[0], -1)
            self._last_topk_indices = indices
            self._last_topk_values = probs  # keep the router gradient
            monitor_val = probs.max(dim=-1)[0].mean().detach() if logits.shape[0] else probs.new_zeros(())
            legacy_load = not self.full_expert_fast_path and self.top_k is not None
            if self.training and (self.record_train_stats or legacy_load):
                with torch.no_grad():
                    total = logits.shape[0] if sizes is None else sizes.sum()
                    hard_load = probs.new_ones(self.num_experts) * total
                    # Retain the optional slow router's historical load buffer,
                    # but never update selection bias for an all-expert route.
                    if legacy_load:
                        self.ema_load.mul_(0.9).add_(hard_load, alpha=0.1)
                    if self.record_train_stats:
                        self._record_train_stats(logits, indices, probs, hard_load, self.expert_bias.detach().clone())
            return probs, monitor_val, probs.new_zeros(())

        # 加上 Bias 用于选择 Top-K (Aux-loss-free 核心机制)
        selection_base = logits if self.select == "logit" else scores
        if self.aux_loss_free and (self.training or self.bias_at_eval):
            scores_for_selection = selection_base + self.expert_bias
        else:
            scores_for_selection = selection_base
        if self.training and self.select_noise > 0.0:
            scores_for_selection = scores_for_selection + self.select_noise * torch.randn_like(scores_for_selection)
        if allowed is not None:
            scores_for_selection = scores_for_selection.masked_fill(~allowed, float("-inf"))

        if self.top_k is not None:
            topk_scores_biased, topk_indices = torch.topk(scores_for_selection, k=self.top_k, dim=-1)

            with torch.no_grad():
                mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()

                # 计算负载 (保留了 V1 支持 sizes 的优秀特性)
                if sizes is not None:
                    weight = sizes.view(-1, 1, 1)
                    weighted_mask = mask * weight
                    current_load = weighted_mask.sum(dim=(0, 1))
                    target_load = (sizes.sum() * self.top_k) / self.num_experts
                else:
                    current_load = mask.sum(dim=(0, 1))
                    target_load = (scores.size(0) * self.top_k) / self.num_experts

                # 使用 EMA 平滑历史负载统计，使返回的 CV 指标极其稳定
                if self.training:
                    self.ema_load.mul_(0.9).add_(current_load, alpha=0.1)
                expert_load_cv = self.ema_load.std() / (self.ema_load.mean() + 1e-8)

            # 修改1: 使用恒定力度 (0.005) 更新 Bias，持续进行负载均衡
            bias_before = self.expert_bias.detach().clone() if (self.training and self.record_train_stats) else None
            bias_step = self._bias_step() if (self.aux_loss_free and self.training) else 0.0
            if bias_step > 0.0:
                with torch.no_grad():
                    error = current_load - target_load
                    self.expert_bias -= torch.sign(error) * bias_step
                    # 保持 Bias 整体均值为 0，防止激活值整体漂移
                    self.expert_bias -= self.expert_bias.mean()

            # 修改2: 强制 L1 归一化 (防止路由专家被共享专家 "饿死")
            # normfix arm: normalise in logit space, not on the sigmoid scores.
            # sigmoid(z) underflows to 0 in float32 below z ~ -104, and past that
            # the 1e-8 epsilon becomes the whole denominator, so the gate row sums
            # to ~1e-24 instead of 1 and the routed branch contributes nothing.
            # softmax over the gathered logits is scale-free, needs no epsilon,
            # sums to 1 by construction and keeps a usable gradient. Selection and
            # the load statistics above deliberately still use `scores`.
            if self.gate == "full_softmax":
                # softmax over every routed expert, the selected entries kept as they are (sum < 1)
                gl = self._tempered(logits)
                if allowed is not None:   # the router's probability mass is spread over the type's experts only
                    gl = gl.masked_fill(~allowed, float("-inf"))
                topk_probs = torch.gather(torch.softmax(gl, dim=-1), 1, topk_indices)
            else:
                topk_logits = torch.gather(logits, 1, topk_indices)
                topk_probs = torch.softmax(self._tempered(topk_logits), dim=-1)
            self._last_topk_indices = topk_indices
            self._last_topk_values = topk_probs
            if self.training and self.record_train_stats:
                self._record_train_stats(scores_for_selection, topk_indices, topk_probs, current_load, bias_before)

            # 构建稀疏输出系数
            coeffs = torch.zeros_like(scores)
            coeffs.scatter_(1, topk_indices, topk_probs)

            # 监控指标：计算最大概率的均值 (反映 Router 的置信度，比计算全部均值更有意义)
            monitor_val = topk_probs.max(dim=-1)[0].mean().detach()

            return coeffs, monitor_val, expert_load_cv.detach()

    def last_topk(self):
        return self._last_topk_indices, self._last_topk_values

    def _tempered(self, logits):
        return logits if self.mixing_temperature == 1.0 else logits / self.mixing_temperature

    @torch.no_grad()
    def _record_train_stats(self, selection_scores, topk_indices, topk_probs, hard_load, bias_before):
        k = int(topk_indices.shape[1])
        n = int(selection_scores.shape[0])
        soft = torch.zeros(self.num_experts, dtype=torch.float32, device=topk_probs.device)
        soft.index_add_(0, topk_indices.reshape(-1), topk_probs.reshape(-1).float())
        g2 = torch.zeros_like(soft)
        g2.index_add_(0, topk_indices.reshape(-1), topk_probs.reshape(-1).float().square())
        stats = dict(opt_step=int(self.opt_step), n_rows=n, top_k=k, hard_load=hard_load.detach().float().clone(),
                     soft_load=soft, soft_load_sq=g2, bias=bias_before,
                     mmp=topk_probs.max(dim=-1)[0].float().mean())
        if k == self.num_experts and n == 0:
            stats["mmp"] = soft.new_zeros(())
        if k < self.num_experts and n > 0:
            v = torch.topk(selection_scores.detach().float(), k=k + 1, dim=-1).values
            margin = v[:, k - 1] - v[:, k]             # last selected vs first rejected selection score
            stats["sel_margin_mean"] = margin.mean()
            stats["sel_margin_q10"] = torch.quantile(margin[: min(n, 65536)], 0.1)
            stats["sel_ties"] = (margin == 0).float().mean()
        self.last_train_stats = stats


class MOLELinear(nn.Module):
    """
    Graph-level MoE linear layer.

    Each graph owns a mixed expert weight matrix. Production uses
    cuEquivariance indexed_linear so each edge selects its graph weight without
    materializing an edge-sized weight tensor.
    """

    def __init__(
            self,
            in_features,
            out_features,
            num_experts=8,
            num_shared_experts=1,
            bias=True,
            mole_linear_mode=None,
            mole_expert_parameterization="full",
            mole_expert_rank=64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        if num_experts < 0 or num_shared_experts < 0 or num_experts + num_shared_experts == 0:
            raise ValueError("MOLELinear needs nonnegative expert counts and at least one expert")
        if mole_expert_parameterization not in ("full", "shared_core"):
            raise ValueError("mole_expert_parameterization must be full or shared_core")
        if mole_expert_parameterization == "shared_core" and (
                isinstance(mole_expert_rank, bool) or not isinstance(mole_expert_rank, int) or mole_expert_rank <= 0):
            raise ValueError("mole_expert_rank must be a positive integer for shared_core")
        self.mole_expert_parameterization = mole_expert_parameterization
        self.mole_expert_rank = min(mole_expert_rank, in_features, out_features)
        self.mole_linear_mode = _normalize_mole_linear_mode(
            mole_linear_mode or os.environ.get("DPTB_MOLE_LINEAR_MODE", "split_loop")
        )
        self._cueq_indexed_linear_cache = {}
        cueq_weight_order = os.environ.get("DPTB_CUEQ_WEIGHT_ORDER", "io_scaled")
        self._cueq_weight_order = None if cueq_weight_order in ("", "auto") else cueq_weight_order

        # 1. 路由专家权重
        if num_experts == 0:
            self.register_parameter("weight_experts", None)
        elif mole_expert_parameterization == "full":
            self.weight_experts = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        else:
            rank = self.mole_expert_rank
            self.basis_left = nn.Parameter(torch.empty(out_features, rank))
            self.basis_right = nn.Parameter(torch.empty(in_features, rank))
            self.core_experts = nn.Parameter(torch.empty(num_experts, rank, rank))
        if bias and num_experts:
            self.bias_experts = nn.Parameter(torch.empty(num_experts, out_features))
        else:
            self.register_parameter('bias_experts', None)

        # 2. 共享专家权重 (Shared Expert) 支持配置数量
        if self.num_shared_experts > 0:
            self.weight_shared = nn.Parameter(torch.empty(num_shared_experts, out_features, in_features))
            if bias:
                self.bias_shared = nn.Parameter(torch.empty(num_shared_experts, out_features))
            else:
                self.register_parameter('bias_shared', None)
        else:
            self.register_parameter('weight_shared', None)
            self.register_parameter('bias_shared', None)

        self.reset_parameters()

    def reset_parameters(self):
        k = math.sqrt(1.0 / self.in_features)
        if self.num_experts == 0:
            pass
        elif self.mole_expert_parameterization == "full":
            nn.init.uniform_(self.weight_experts, -k, k)
        else:
            device = self.core_experts.device
            devices = []
            if device.type == "cuda":
                devices = [device.index if device.index is not None else torch.cuda.current_device()]
            # Initialize the factors without shifting the legacy stream used
            # by biases, shared experts, radial blocks and later routers.
            # fork_rng also preserves CPU state when the parameters are CUDA.
            with torch.random.fork_rng(devices=devices):
                nn.init.orthogonal_(self.basis_left)
                nn.init.orthogonal_(self.basis_right)
                # E||P D Q.T||_F^2 = out/3, matching the full uniform bank.
                nn.init.normal_(self.core_experts,
                                std=math.sqrt(self.out_features / 3.0) / self.mole_expert_rank)
            # Consume exactly the original bank's draws on its own device and
            # dtype. This temporary is construction/reset cost only: it is
            # neither registered nor kept for forward or checkpointing.
            legacy_draws = torch.empty(self.num_experts, self.out_features, self.in_features,
                                       device=device, dtype=self.core_experts.dtype)
            nn.init.uniform_(legacy_draws, -k, k)
            del legacy_draws
        if self.bias_experts is not None:
            nn.init.uniform_(self.bias_experts, -k, k)

        if self.num_shared_experts > 0:
            nn.init.uniform_(self.weight_shared, -k, k)
            if self.bias_shared is not None:
                nn.init.uniform_(self.bias_shared, -k, k)

    def __getattr__(self, name):
        # Preserve the readable bank interface used by SO2/CUDA callers. Only
        # P/Q/D are leaves in shared_core; no full matrix is checkpointed.
        if name == "weight_experts" and "core_experts" in self.__dict__.get("_parameters", {}):
            return self._expert_weight_bank()
        return super().__getattr__(name)

    def _expert_weight_bank(self):
        if self.num_experts == 0 or self.mole_expert_parameterization == "full":
            return super().__getattr__("weight_experts")
        # Expert-sized, not edge-sized. Keep this differentiable and local to
        # the current forward: caching across optimizer steps would be stale.
        return (self.basis_left.unsqueeze(0) @ self.core_experts) @ self.basis_right.t()

    @torch.no_grad()
    def scale_expert_weights_(self, scale):
        """Scale the represented bank, including the SO2 m>0 initial scale."""
        if self.num_experts == 0:
            return self
        if self.mole_expert_parameterization == "full":
            self.weight_experts.mul_(scale)
        else:
            self.core_experts.mul_(scale)
        return self

    def _apply_indexed_ref(self, x, mixed_weights, mixed_bias, graph_index):
        flat_x = x.reshape(-1, self.in_features)
        flat_graph_index = _expand_graph_index_for_leading_dims(graph_index, x)
        flat_w = mixed_weights.index_select(0, flat_graph_index)
        flat_out = torch.bmm(flat_w, flat_x.unsqueeze(-1)).squeeze(-1)
        if mixed_bias is not None:
            flat_out = flat_out + mixed_bias.index_select(0, flat_graph_index)
        return flat_out.reshape(*x.shape[:-1], self.out_features)


    def _apply_expert_indexed_ref(self, x, expert_index):
        flat_x = x.reshape(-1, self.in_features)
        flat_expert_index = _expand_route_index_for_leading_dims(expert_index, x)
        flat_w = self.weight_experts.index_select(0, flat_expert_index)
        flat_out = torch.bmm(flat_w, flat_x.unsqueeze(-1)).squeeze(-1)
        if self.bias_experts is not None:
            flat_out = flat_out + self.bias_experts.index_select(0, flat_expert_index)
        return flat_out.reshape(*x.shape[:-1], self.out_features)

    def _apply_expert_cublas_grouped(self, x, expert_index):
        if x.device.type != "cuda":
            raise RuntimeError("expert cublas_grouped dispatch requires CUDA.")
        weight = self._expert_weight_bank()
        if x.dtype != torch.float32 or weight.dtype != torch.float32:
            raise RuntimeError(
                "expert cublas_grouped dispatch currently requires float32 tensors; "
                f"got x={x.dtype}, weight={weight.dtype}."
            )

        from dptb.nn.cublas_grouped_gemm import grouped_gemm

        flat_x = x.reshape(-1, self.in_features)
        flat_expert_index = _expand_route_index_for_leading_dims(expert_index, x)
        if flat_expert_index.numel() > 1 and not _functorch_plain(torch.all(flat_expert_index[1:] >= flat_expert_index[:-1])).item():
            permute_idx = torch.argsort(flat_expert_index, stable=True)
            sorted_expert_index = flat_expert_index.index_select(0, permute_idx)
            flat_x = flat_x.index_select(0, permute_idx)
            unpermute_idx = torch.empty_like(permute_idx)
            unpermute_idx.scatter_(
                0,
                permute_idx,
                torch.arange(permute_idx.numel(), device=permute_idx.device, dtype=permute_idx.dtype),
            )
        else:
            sorted_expert_index = flat_expert_index
            unpermute_idx = None

        counts = torch.bincount(sorted_expert_index, minlength=self.num_experts)
        ptr = torch.zeros(self.num_experts + 1, dtype=torch.long, device=counts.device)
        ptr[1:] = torch.cumsum(counts, dim=0)
        flat_out = grouped_gemm(
            flat_x.contiguous(),
            ptr.to(device="cpu", dtype=torch.long).contiguous(),
            weight.contiguous(),
        )
        if self.bias_experts is not None:
            flat_out = flat_out + self.bias_experts.index_select(0, sorted_expert_index)
        if unpermute_idx is not None:
            flat_out = flat_out.index_select(0, unpermute_idx)
        return flat_out.reshape(*x.shape[:-1], self.out_features)

    def apply_experts(self, x, expert_index, *, include_shared_experts: bool = False):
        """Apply raw expert weights selected by expert_index, without coefficient mixing.

        This is the nonlinear MoE building block: expert_index is an expert id,
        not a graph id for a pre-mixed weight class.
        """
        if self.num_experts == 0:
            raise ValueError("A shared-only MOLELinear has no routed experts to select")
        expert_index = expert_index.to(device=x.device, dtype=torch.long).reshape(-1)
        if expert_index.numel() != x.shape[0]:
            raise ValueError(
                f"expert_index has {expert_index.numel()} rows, but input has {x.shape[0]} rows."
            )
        if expert_index.numel() and (
                int(_functorch_plain(expert_index.min()).item()) < 0 or int(_functorch_plain(expert_index.max()).item()) >= self.num_experts
        ):
            raise ValueError(f"expert_index values must be in [0, {self.num_experts}).")

        if self.mole_linear_mode == "cublas_grouped":
            out = self._apply_expert_cublas_grouped(x, expert_index)
        else:
            out = self._apply_expert_indexed_ref(x, expert_index)

        if include_shared_experts and self.num_shared_experts > 0:
            shared_weight = self.weight_shared.sum(0)
            shared_bias = self.bias_shared.sum(0) if self.bias_shared is not None else None
            out = out + F.linear(x, shared_weight, shared_bias)
        return out

    def _routed_weight_and_bias(self, fold_shared: bool):
        """Expert weights, with the shared expert folded in when licensed.

        sum_j c_j (W_ej + W_sh) == sum_j c_j W_ej + W_sh requires sum_j c_j == 1.
        """
        if self.num_experts == 0:
            if fold_shared:
                raise ValueError("A shared-only MOLELinear cannot fold shared weights into routed slots")
            # The fused SO2 shared branch asks for the layout but never dispatches it.
            # These empty views are not parameters and never enter the state dict.
            return self.weight_shared[:0], None
        weight = self._expert_weight_bank()
        bias = self.bias_experts
        if fold_shared and self.num_shared_experts > 0:
            weight = weight + self.weight_shared.sum(0).unsqueeze(0)
            if self.bias_shared is not None:
                shared_b = self.bias_shared.sum(0).unsqueeze(0)
                bias = shared_b if bias is None else bias + shared_b
        return weight, bias

    def _apply_expert_with_layout(self, x, layout, weight, bias):
        """One expert pass using a precomputed sort order and segment pointer."""
        order, inverse, ptr, sorted_index = layout
        flat_x = x.reshape(-1, self.in_features)
        rows = flat_x.shape[0]
        if order.numel() != rows:
            # leading dims expand one route row into several matrix rows
            repeat = rows // order.numel()
            base = order.unsqueeze(1) * repeat + torch.arange(
                repeat, device=order.device, dtype=order.dtype).unsqueeze(0)
            order = base.reshape(-1)
            inverse = torch.empty_like(order)
            inverse.scatter_(
                0, order,
                torch.arange(rows, device=order.device, dtype=order.dtype))
            ptr = ptr * repeat
            sorted_index = sorted_index.repeat_interleave(repeat)
        # Everything below is in SORTED row order and the single index_select at
        # the end puts it back.  Mixing the two orders silently routes rows to
        # the wrong expert, so no intermediate may be indexed with anything but
        # a sorted-space index.
        xs = permute_rows(flat_x, order, inverse).contiguous()
        use_cublas = (
            self.mole_linear_mode == "cublas_grouped"
            and x.device.type == "cuda"
            and xs.dtype == torch.float32
            and weight.dtype == torch.float32
        )
        if use_cublas:
            from dptb.nn.cublas_grouped_gemm import grouped_gemm

            ys = grouped_gemm(xs, ptr, weight.contiguous())
        else:
            # ptr is a CPU tensor, so these bounds cost no device sync -- unlike
            # the num_experts nonzero() calls they replace.
            bounds = ptr.tolist()
            parts = [F.linear(xs[bounds[e]:bounds[e + 1]], weight[e], None)
                     for e in range(self.num_experts)
                     if bounds[e + 1] > bounds[e]]
            ys = (torch.cat(parts, 0) if parts
                  else xs.new_zeros(rows, self.out_features))
        if bias is not None:
            ys = ys + bias.index_select(0, sorted_index)
        return permute_rows(ys, inverse, order).reshape(
            *x.shape[:-1], self.out_features)

    def _apply_expert_sorted_loop(self, x, expert_index):
        """One matmul per expert over its own rows; no per-row weight is built.

        The reference indexed path gathers weight_experts per row, which costs
        n_rows * out * in -- the very thing activation space exists to avoid.
        """
        flat_x = x.reshape(-1, self.in_features)
        eidx = _expand_route_index_for_leading_dims(expert_index, x)
        weight = self._expert_weight_bank()
        out = None
        for e in range(self.num_experts):
            rows = (eidx == e).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            bias = self.bias_experts[e] if self.bias_experts is not None else None
            part = F.linear(flat_x.index_select(0, rows), weight[e], bias)
            if out is None:
                # Take dtype from the matmul, not from x: under autocast F.linear
                # returns bf16/fp16 while x stays fp32, and index_add requires
                # self and source to agree.
                out = part.new_zeros(flat_x.shape[0], self.out_features)
            out = out.index_add(0, rows, part)
        if out is None:
            out = flat_x.new_zeros(flat_x.shape[0], self.out_features)
        return out.reshape(*x.shape[:-1], self.out_features)

    def _apply_expert_no_materialize(self, x, expert_index):
        if (
            self.mole_linear_mode == "cublas_grouped"
            and x.device.type == "cuda"
            and x.dtype == torch.float32
            and self.weight_experts.dtype == torch.float32
        ):
            return self._apply_expert_cublas_grouped(x, expert_index)
        return self._apply_expert_sorted_loop(x, expert_index)

    def _apply_activation_space(self, x, mole_globals: MOLEGlobals):
        """x (sum_e c_e W_e) == sum_e c_e (x W_e).

        Same arithmetic as _mix_expert_parameters followed by a grouped GEMM,
        but the resident tensor is k copies of the OUTPUT instead of one
        [out, in] weight per route token: n_tokens * out * in becomes
        k * n_rows * out.  That is what makes per-edge routing affordable.
        Summation order differs, so results agree to float rounding, not bitwise.
        """
        idx = getattr(mole_globals, "topk_indices", None)
        val = getattr(mole_globals, "topk_values", None)
        if idx is None or val is None:
            raise ValueError(
                "activation-space MoLE needs top-k routing metadata, but "
                "router.last_topk() returned None; route the inputs before dispatch."
            )
        if idx.shape[0] != x.shape[0]:
            raise ValueError(
                f"activation-space MoLE got {idx.shape[0]} route rows for "
                f"{x.shape[0]} input rows; routing must be per-row here."
            )
        idx = idx.to(device=x.device, dtype=torch.long)
        val = val.to(device=x.device, dtype=x.dtype)
        branch = getattr(mole_globals, "branch", "all")
        # only a pass that computes both parts may fold the shared expert into the slots
        fold = bool(getattr(mole_globals, "coefficients_sum_to_one", False)) and branch == "all"
        out = None
        if branch != "shared":
            weight, bias = self._routed_weight_and_bias(fold)
            view = [idx.shape[0]] + [1] * (x.dim() - 1)
            for j in range(idx.shape[1]):
                layout = mole_globals.expert_slot_layout(j, idx[:, j], self.num_experts)
                part = self._apply_expert_with_layout(x, layout, weight, bias)
                part = part * val[:, j].reshape(view)
                out = part if out is None else out + part
        if out is None:
            out = x.new_zeros(*x.shape[:-1], self.out_features)
        if branch != "routed" and not fold and self.num_shared_experts > 0:
            shared_bias = self.bias_shared.sum(0) if self.bias_shared is not None else None
            out = out + F.linear(x, self.weight_shared.sum(0), shared_bias)
        return out

    def _mix_expert_parameters(self, mole_globals: MOLEGlobals):
        if getattr(mole_globals, "activation_space", False):
            raise RuntimeError(
                "activation-space MoLE reached the weight-space path, which "
                "would materialise one [out_features, in_features] weight per "
                "route token -- exactly what per-edge routing cannot afford. "
                "Reach every MOLELinear through MOLELinear.forward or the "
                "SO2CUDA routes of so2_activation_routes."
            )
        coefficients = mole_globals.coefficients
        topk_indices = getattr(mole_globals, "topk_indices", None)
        topk_values = getattr(mole_globals, "topk_values", None)
        weight = self._expert_weight_bank()

        if (
            topk_indices is not None
            and topk_values is not None
            and topk_indices.shape[0] == coefficients.shape[0]
        ):
            topk_indices = topk_indices.to(device=weight.device, dtype=torch.long)
            topk_values = topk_values.to(device=weight.device, dtype=weight.dtype)
            n_routes, k_routes = topk_indices.shape
            gathered_weights = weight.index_select(0, topk_indices.reshape(-1))
            gathered_weights = gathered_weights.reshape(
                n_routes,
                k_routes,
                self.out_features,
                self.in_features,
            )
            mixed_weights = (gathered_weights * topk_values.reshape(n_routes, k_routes, 1, 1)).sum(dim=1)

            mixed_bias = None
            if self.bias_experts is not None:
                gathered_bias = self.bias_experts.index_select(0, topk_indices.reshape(-1))
                gathered_bias = gathered_bias.reshape(n_routes, k_routes, self.out_features)
                mixed_bias = (gathered_bias * topk_values.reshape(n_routes, k_routes, 1)).sum(dim=1)
        else:
            mixed_weights = torch.einsum("be, eoi -> boi", coefficients, weight)
            mixed_bias = None
            if self.bias_experts is not None:
                mixed_bias = torch.einsum("be, eo -> bo", coefficients, self.bias_experts)

        if self.num_shared_experts > 0:
            mixed_weights = mixed_weights + self.weight_shared.sum(0).unsqueeze(0)
            if mixed_bias is not None and self.bias_shared is not None:
                mixed_bias = mixed_bias + self.bias_shared.sum(0).unsqueeze(0)
            elif mixed_bias is None and self.bias_shared is not None:
                mixed_bias = self.bias_shared.sum(0).unsqueeze(0).expand(mixed_weights.shape[0], -1)

        return mixed_weights, mixed_bias

    def _apply_cublas_grouped(self, x, mixed_weights, mixed_bias, graph_index, mole_globals: MOLEGlobals):
        if x.device.type != "cuda":
            raise RuntimeError("mole_linear_mode='cublas_grouped' requires CUDA.")
        if x.dtype != torch.float32 or mixed_weights.dtype != torch.float32:
            raise RuntimeError(
                "mole_linear_mode='cublas_grouped' currently requires float32 tensors; "
                f"got x={x.dtype}, weight={mixed_weights.dtype}."
            )

        from dptb.nn.cuda_ops.grouped_gemm import grouped_gemm

        flat_x = x.reshape(-1, self.in_features)
        if getattr(mole_globals, "_indexed_inputs_are_sorted", False):
            sorted_graph_index = _expand_graph_index_cached(graph_index, x, mole_globals).reshape(-1).to(dtype=torch.long)
            unpermute_idx = None
        else:
            permute_idx, unpermute_idx, sorted_graph_index = mole_globals.indexed_flat_permutation(graph_index, x)
            if permute_idx is not None:
                flat_x = flat_x.index_select(0, permute_idx)

        ptr = mole_globals.indexed_segment_ptr(
            sorted_graph_index,
            int(mixed_weights.shape[0]),
            prefer_cpu=True,
        )
        flat_out = grouped_gemm(flat_x.contiguous(), ptr, mixed_weights.contiguous())
        if mixed_bias is not None:
            flat_out = flat_out + mixed_bias.index_select(0, sorted_graph_index)
        if unpermute_idx is not None:
            flat_out = flat_out.index_select(0, unpermute_idx)
        return flat_out.reshape(*x.shape[:-1], self.out_features)

    def _cueq_flatten_weight(self, mixed_weights, order: str):
        scale = math.sqrt(self.in_features)
        if order == "io_scaled":
            flat = mixed_weights.transpose(1, 2).contiguous() * scale
        elif order == "oi_scaled":
            flat = mixed_weights.contiguous() * scale
        elif order == "io":
            flat = mixed_weights.transpose(1, 2).contiguous()
        elif order == "oi":
            flat = mixed_weights.contiguous()
        else:
            raise ValueError(f"unknown cueq weight order {order!r}")
        return flat.reshape(mixed_weights.shape[0], -1)

    def _get_cueq_indexed_linear(self, num_graphs: int, *, dtype, device):
        if device.type != "cuda":
            raise RuntimeError("cueq_indexed_linear requires CUDA; use split_loop or indexed_ref on CPU.")
        if dtype not in (torch.float32, torch.float64):
            raise RuntimeError(
                "cueq_indexed_linear is currently validated only for float32/float64. "
                "Disable AMP/autocast or use split_loop."
            )

        try:
            import cuequivariance as cue
            import cuequivariance_torch as cuet
        except ImportError as exc:
            raise ImportError(
                "mole_linear_mode='cueq_indexed_linear' requires cuequivariance and "
                "cuequivariance_torch."
            ) from exc

        key = (num_graphs, str(dtype), str(device), self.in_features, self.out_features)
        mod = self._cueq_indexed_linear_cache.get(key)
        metadata = {
            "num_graphs": int(num_graphs),
            "dtype": str(dtype),
            "device": str(device),
            "in_features": int(self.in_features),
            "out_features": int(self.out_features),
            "local_entries_before": len(self._cueq_indexed_linear_cache),
        }
        if mod is not None:
            metadata["local_entries_after"] = len(self._cueq_indexed_linear_cache)
            record_cuda_cache_event(
                "cueq_indexed_linear",
                key,
                "hit",
                metadata=metadata,
                logger=log,
            )
            return mod

        record_cuda_cache_event(
            "cueq_indexed_linear",
            key,
            "miss",
            metadata=metadata,
            logger=log,
        )
        with cuda_cache_memory_probe(
            "cueq_indexed_linear",
            key,
            device=device,
            metadata=metadata,
            logger=log,
        ):
            irreps_in = cue.Irreps(cue.O3, f"{self.in_features}x0e")
            irreps_out = cue.Irreps(cue.O3, f"{self.out_features}x0e")
            mod = cuet.Linear(
                irreps_in,
                irreps_out,
                shared_weights=True,
                internal_weights=False,
                weight_classes=num_graphs,
                layout=cue.ir_mul,
                device=device,
                dtype=dtype,
                method="indexed_linear",
            )
            self._cueq_indexed_linear_cache[key] = mod
            metadata["local_entries_after"] = len(self._cueq_indexed_linear_cache)
        if os.environ.get("DPTB_CUEQ_CACHE_DIAG", "0") not in ("", "0", "false", "False"):
            log.info(
                "Created cuEq indexed_linear cache entry: num_graphs=%s dtype=%s device=%s "
                "in=%s out=%s local_entries=%s",
                num_graphs,
                dtype,
                device,
                self.in_features,
                self.out_features,
                len(self._cueq_indexed_linear_cache),
            )
        return mod

    def _infer_cueq_weight_order(self, cue_lin, flat_x, mixed_weights, flat_graph_index):
        if self._cueq_weight_order is not None:
            return self._cueq_weight_order

        with torch.no_grad():
            n_probe = min(int(flat_x.shape[0]), 64)
            probe_x = flat_x[:n_probe]
            probe_idx = flat_graph_index[:n_probe]
            ref_w = mixed_weights.index_select(0, probe_idx)
            ref = torch.bmm(ref_w, probe_x.unsqueeze(-1)).squeeze(-1)

            best_order, best_err = None, None
            for order in ("io_scaled", "oi_scaled", "io", "oi"):
                try:
                    weight = self._cueq_flatten_weight(mixed_weights, order)
                    out = cue_lin(probe_x, weight=weight, weight_indices=probe_idx)
                    err_val = float((out - ref).abs().max().detach().cpu())
                except Exception:
                    continue
                if best_err is None or err_val < best_err:
                    best_order, best_err = order, err_val

        if best_order is None or best_err is None or best_err > 1e-4:
            raise RuntimeError(
                "Could not infer cuEquivariance scalar Linear weight order; "
                f"best_order={best_order}, best_err={best_err}."
            )

        self._cueq_weight_order = best_order
        return best_order

    def _apply_cueq_indexed_linear(self, x, mixed_weights, mixed_bias, graph_index, mole_globals: MOLEGlobals):
        num_graphs = int(mixed_weights.shape[0])
        if num_graphs == 1:
            # cuEq's indexed_linear only consumes weight_indices when weight_classes > 1.
            return F.linear(
                x,
                mixed_weights[0],
                mixed_bias[0] if mixed_bias is not None else None,
            )

        flat_x = x.reshape(-1, self.in_features)
        if getattr(mole_globals, "_indexed_inputs_are_sorted", False):
            sorted_graph_index = _expand_graph_index_cached(graph_index, x, mole_globals).reshape(-1).to(dtype=torch.long)
            unpermute_idx = None
        else:
            permute_idx, unpermute_idx, sorted_graph_index = mole_globals.indexed_flat_permutation(graph_index, x)
            if permute_idx is not None:
                flat_x = flat_x.index_select(0, permute_idx)
        cue_lin = self._get_cueq_indexed_linear(num_graphs, dtype=x.dtype, device=x.device)

        order = self._infer_cueq_weight_order(cue_lin, flat_x, mixed_weights, sorted_graph_index)
        flat_weight = self._cueq_flatten_weight(mixed_weights, order)
        flat_out = cue_lin(flat_x, weight=flat_weight, weight_indices=sorted_graph_index)
        if mixed_bias is not None:
            flat_out = flat_out + mixed_bias.index_select(0, sorted_graph_index)
        if unpermute_idx is not None:
            flat_out = flat_out.index_select(0, unpermute_idx)
        return flat_out.reshape(*x.shape[:-1], self.out_features)

    def _apply_split_loop_by_graph_index(self, x, mixed_weights, mixed_bias, mole_globals):
        """One F.linear per route over the rows graph_index assigns to it."""
        graph_index = _mole_graph_index(mole_globals, x.shape[0], device=x.device)
        if x.shape[0] == 0:
            # A zero-row indexed contraction allocates no per-row weights, but
            # retains the zero gradients to inputs, routing and expert parameters.
            return self._apply_indexed_ref(x, mixed_weights, mixed_bias, graph_index)
        order = torch.argsort(graph_index, stable=True)
        counts = torch.bincount(graph_index, minlength=mixed_weights.shape[0]).tolist()
        parts = []
        for route, x_route in enumerate(torch.split(x.index_select(0, order), counts, dim=0)):
            if x_route.shape[0]:
                bias = mixed_bias[route] if mixed_bias is not None else None
                parts.append(F.linear(x_route, mixed_weights[route], bias))
        if not parts:
            return x.new_zeros(*x.shape[:-1], self.out_features)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
        return torch.cat(parts, dim=0).index_select(0, inverse)

    def forward(self, x, mole_globals: MOLEGlobals):
        if self.num_experts == 0:
            if getattr(mole_globals, "branch", "shared") == "routed":
                raise ValueError("A shared-only MOLELinear cannot execute a routed branch")
            bias = self.bias_shared.sum(0) if self.bias_shared is not None else None
            return F.linear(x, self.weight_shared.sum(0), bias)
        if getattr(mole_globals, "top1_independent", False):
            from .top1_prior import linear
            return linear(self, x, mole_globals)
        # 安全回退
        if mole_globals is None or mole_globals.coefficients is None:
            w_avg = self.weight_experts.mean(0)
            if self.num_shared_experts > 0:
                w_avg = w_avg + self.weight_shared.sum(0)
            b_avg = None
            if self.bias_experts is not None:
                b_avg = self.bias_experts.mean(0)
                if self.num_shared_experts > 0 and self.bias_shared is not None:
                    b_avg = b_avg + self.bias_shared.sum(0)
            return F.linear(x, w_avg, b_avg)

        if getattr(mole_globals, "activation_space", False):
            return self._apply_activation_space(x, mole_globals)

        # === 核心逻辑: 权重融合 (Weight Merging) ===
        # 1. 混合路由专家权重
        # coefficients: [Batch, Num_Experts]
        # weight_experts: [Num_Experts, Out, In]
        # mixed_weights: [Batch, Out, In]
        mixed_weights, mixed_bias = self._mix_expert_parameters(mole_globals)

        # 2. 【关键】融合共享专家权重
        # 利用分配律: (W_routed + sum(W_shared)) * x

        # 3. 处理 Bias

        # 4. 执行线性变换
        # 根据系统大小拆分 Input，因为每个系统(Graph)对应一个混合后的权重
        mode = self.mole_linear_mode
        if mode != "split_loop":
            graph_index = _mole_graph_index(mole_globals, x.shape[0], device=x.device)
            if graph_index.numel() != x.shape[0]:
                raise ValueError(
                    f"MOLE graph_index has {graph_index.numel()} rows, but input has {x.shape[0]} rows."
                )
            if mode == "indexed_ref":
                return self._apply_indexed_ref(x, mixed_weights, mixed_bias, graph_index)
            if mode == "cueq_indexed_linear":
                return self._apply_cueq_indexed_linear(x, mixed_weights, mixed_bias, graph_index, mole_globals)
            if mode == "cublas_grouped":
                return self._apply_cublas_grouped(x, mixed_weights, mixed_bias, graph_index, mole_globals)
            raise AssertionError(f"unreachable mole_linear_mode={mode!r}")

        if (
            getattr(mole_globals, "graph_index", None) is not None
            and getattr(mole_globals, "split_sizes", None) is None
            and getattr(mole_globals, "_sizes_tensor", None) is None
        ):
            # Rows name their route only through graph_index (edge-MoE dispatch).
            # Contiguous splits would put every row on route 0.
            return self._apply_split_loop_by_graph_index(x, mixed_weights, mixed_bias, mole_globals)
        split_sizes = _mole_split_sizes(mole_globals, x.shape[0])
        x_split = torch.split(x, split_sizes, dim=0)
        out_parts = []

        # 循环执行 (虽然是 Python 循环，但通常 System 数量不多，开销可控)
        for i, x_sys in enumerate(x_split):
            w = mixed_weights[i]
            b = mixed_bias[i] if mixed_bias is not None else None
            out_parts.append(F.linear(x_sys, w, b))

        return torch.cat(out_parts, dim=0)
# ------------------------------------------------------------------------------

class SO2_Attention(torch.nn.Module):
    def __init__(self, node_irreps, latent_dim: int, use_so2_att_proj: bool = True):
        super().__init__()
        self.irreps_in = node_irreps.simplify()
        self.l_max = max((l for (_, (l, _)), _ in zip(self.irreps_in, self.irreps_in.slices()) if l > 0), default=0)
        self.dims = {l: 2 * l + 1 for l in range(self.l_max + 1)}
        self.offsets = {}
        offset = 0
        for l in range(self.l_max + 1):
            self.offsets[l] = offset
            offset += self.dims[l]

        self.lin_center = e3nn_Linear(node_irreps, node_irreps, shared_weights=True, internal_weights=True, biases=True)
        self.lin_neighbor = e3nn_Linear(node_irreps, node_irreps, shared_weights=True, internal_weights=True,
                                        biases=True)

        groups = defaultdict(list)
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            groups[l].append((mul, slice_info))
        self.groups = groups

        # --- 修改：为每个 l 建立输入维为 (total_mul * (2l+1)) 的线性映射
        self.sim_linears = nn.ModuleDict()
        for l, g in groups.items():
            total_mul = sum(m for m, _ in g)
            in_dim = total_mul * self.dims[l]  # m * d
            self.sim_linears[f"l{l}"] = nn.Sequential(
                nn.Linear(in_dim, latent_dim),
                nn.SiLU(),
            )

        # --- 修改：用一个 final_mlp 替代简单求和（把所有 l 的 latent_dim 串联后再做一次融合）
        num_l = len(groups)
        # final_mlp: (num_l * latent_dim) -> latent_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(num_l * latent_dim, 2 * latent_dim),
            nn.SiLU(),  # 平滑非线性（比 ReLU 更稳定）
            nn.Linear(2 * latent_dim, latent_dim)
        )

    def forward(self, node_features, active_edge_vector, active_edge_index, wigner_D_all=None):
        n, _ = node_features.shape
        # keep node features as-is (no per-edge rotation here)
        rot_n_feat_ = node_features.new_zeros(node_features.shape)

        if wigner_D_all is None and self.l_max > 0:
            angle = xyz_to_angles(active_edge_vector[:, [1, 2, 0]])
            wigner_D_all = batch_wigner_D(self.l_max, angle[0], angle[1], torch.zeros_like(angle[0]), _Jd)

        # keep scalar parts unchanged
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            if l == 0:
                rot_n_feat_[:, slice_info] = node_features[:, slice_info]

        # keep the raw (unrotated) node parts in rot_n_feat_ so linear layers can be applied
        for l, group in self.groups.items():
            if l == 0 or not group:
                continue
            for mul, sl in group:
                rot_n_feat_[:, sl] = node_features[:, sl]

        # apply linear maps (these are node-wise)
        rot_center_node_feat = self.lin_center(rot_n_feat_)
        rot_center_node_feat = rot_center_node_feat[active_edge_index[0]]  # shape: (n_edges, dim)

        rot_neighbor_node_feat = self.lin_neighbor(rot_n_feat_)
        rot_neighbor_node_feat = rot_neighbor_node_feat[active_edge_index[1]]  # shape: (n_edges, dim)

        latent_list = []
        # Now for each l, build per-edge (n_edges, total_mul, 2l+1) and apply per-edge rotation
        for l, group in self.groups.items():
            muls, slices = zip(*group)
            total_mul = sum(m for m, _ in group)
            # center/neighbor parts now have batch = n_edges
            # each part reshape -> (n_edges, mul, 2l+1)
            center_node_parts = [rot_center_node_feat[:, sl].reshape(-1, mul, self.dims[l]) for mul, sl in group]
            center_node_combined = torch.cat(center_node_parts, dim=1)  # (n_edges, total_mul, d)

            neighbor_node_parts = [rot_neighbor_node_feat[:, sl].reshape(-1, mul, self.dims[l]) for mul, sl in group]
            neighbor_node_combined = torch.cat(neighbor_node_parts, dim=1)  # (n_edges, total_mul, d)

            if l == 0:
                # l=0: dims[l] == 1, no rotation needed; keep consistent flow
                # center_node_combined, neighbor_node_combined have shape (e, total_mul, 1)
                center_rot = center_node_combined
                neighbor_rot = neighbor_node_combined
            else:
                # get per-edge rotation matrices: shape (n_edges, 2l+1, 2l+1)
                start = self.offsets[l]
                rot_mat = wigner_D_all[:, start:start + self.dims[l], start:start + self.dims[l]]

                # rotate center & neighbor per-edge:
                # center_combined: (e, m, d), rot_mat: (e, d, d) -> rotated_center: (e, m, d)
                center_rot = torch.einsum('emd,edq->emq', center_node_combined, rot_mat)
                neighbor_rot = torch.einsum('emd,edq->emq', neighbor_node_combined, rot_mat)

            # --- 修改：不对 d 求和，而是保留 (e, m, d)，做 elementwise 相乘后展平为 (e, m*d)
            # elementwise product as similarity per-component
            sim_tensor = center_rot * neighbor_rot  # (e, m, d)
            e = sim_tensor.shape[0]
            sim_flat = sim_tensor.reshape(e, -1)  # (e, m * d)

            # map flattened similarity (m*d) to latent_dim
            sim_mapped = self.sim_linears[f"l{l}"](sim_flat)  # (e, latent_dim)
            latent_list.append(sim_mapped)

        # latent_list: list of (e, latent_dim), one per l
        # stack along new l-dim -> (e, num_l, latent_dim)
        latent_stack = torch.stack(latent_list, dim=1)
        # flatten (e, num_l * latent_dim) and fuse via final_mlp
        e = latent_stack.shape[0]
        fused = latent_stack.reshape(e, -1)
        latent = self.final_mlp(fused)  # (e, latent_dim)

        return latent



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
    ):
        super(SO2_Linear, self).__init__()

        self.irreps_in = Irreps(irreps_in).simplify()
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
        if self.irreps_in.dim <= self.irreps_out.dim:
            front = True
            self.m_in_num = [0] * (self.m_max + 1)
        else:
            front = False
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

    def forward(self, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
        """Rotate, apply the m-wise (MoE) linears, rotate back.

        Args:
            x: Input features
            R: Edge vectors (for rotation)
            mole_globals: MoE routing info
            latents: Latent features for radial embedding
            wigner_D_all: Precomputed Wigner D matrices (optional)

        ``so2_fusion_mode`` selects the route.  Activation-space routing (prior_activate,
        Switch top-1) takes the SO2CUDA routes of ``so2_activation_routes`` in every
        grouped mode; weight-space routing takes the fused-P0 or persistent-P1 kernels
        when requested.  Whatever a route declines runs on the grouped streaming route.
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
            from . import so2_activation_routes

            result = so2_activation_routes.forward(
                self, x, R, mole_globals, latents, wigner_D_all,
                fused=mode == "streamed_m_major_fused_p0",
            )
            if result is not None:
                return result
        elif mode == "streamed_m_major_fused_p0":
            from dptb.nn.so2_moe_fused_p0 import try_forward_so2_moe_fused_p0

            result = try_forward_so2_moe_fused_p0(self, x, R, mole_globals, latents, wigner_D_all)
            if result is not None:
                return result
        elif mode == "streamed_m_major_persistent_grouped_p1":
            from dptb.nn.so2_moe_persistent_grouped import try_forward_so2_moe_persistent_grouped_p1

            result = try_forward_so2_moe_persistent_grouped_p1(self, x, R, mole_globals, latents, wigner_D_all)
            if result is not None:
                return result
            if os.environ.get("DPTB_SO2_MOE_PERSISTENT_P1_STRICT", "0") not in ("", "0", "false", "False", "FALSE"):
                raise RuntimeError("streamed_m_major_persistent_grouped_p1 declined; strict mode forbids cueq fallback.")
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
        n, _ = x.shape
        if m == 0:
            parts = []
            for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
                x_l = x[:, slice_info].reshape(n, mul, 2 * l + 1)
                if l == 0 or not self.rotate_in:
                    parts.append(x_l[:, :, l])
                else:
                    rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
                    parts.append(torch.einsum("ncd,nd->nc", x_l, rot_mat[:, :, l]))
            return torch.cat(parts, dim=1)

        parts = []
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            if l < m:
                continue
            x_l = x[:, slice_info].reshape(n, mul, 2 * l + 1)
            local_rows = [l - m, l + m]
            if not self.rotate_in:
                pair = x_l[:, :, local_rows]
            else:
                rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
                pair = torch.einsum("ncd,ndp->ncp", x_l, rot_mat[:, :, local_rows])
            parts.append(pair)
        return torch.cat(parts, dim=1).transpose(1, 2).contiguous()

    def _accumulate_m0_output(self, out, y_m0, wigner_D_all):
        n = out.shape[0]
        channel_start = 0
        for (mul, (l, p)), slice_info in zip(self.irreps_out, self.irreps_out.slices()):
            y_l = y_m0[:, channel_start:channel_start + mul]
            channel_start += mul
            out_l = out[:, slice_info].reshape(n, mul, 2 * l + 1)
            if l == 0 or not self.rotate_out:
                out_l[:, :, l] += y_l
            else:
                rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
                out_l += y_l.unsqueeze(-1) * rot_mat[:, :, l].unsqueeze(1)

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

    def _cueq_linear_is_enabled(self) -> bool:
        backends = []
        if isinstance(getattr(self, "fc_m0", None), MOLELinear):
            backends.append(self.fc_m0.mole_linear_mode)
        for module in self.m_linear:
            fc = getattr(module, "fc", None)
            if isinstance(fc, MOLELinear):
                backends.append(fc.mole_linear_mode)
        return bool(backends) and all(mode == "cueq_indexed_linear" for mode in backends)

    def _cublas_m_fusion_enabled(self) -> bool:
        if os.environ.get("DPTB_SO2_FUSE_M_CUBLAS", "0") in ("", "0", "false", "False"):
            return False
        if self.m_max < 2:
            return False
        for module in self.m_linear:
            fc = getattr(module, "fc", None)
            if not isinstance(fc, MOLELinear) or fc.mole_linear_mode != "cublas_grouped":
                return False
        return True

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

    def _apply_m_linears_cublas_multi(
            self,
            x_inputs: list[torch.Tensor],
            mole_globals: MOLEGlobals,
    ) -> list[torch.Tensor]:
        if not x_inputs:
            return []
        if x_inputs[0].device.type != "cuda" or x_inputs[0].dtype != torch.float32:
            raise RuntimeError("SO2 m-fused cuBLAS path requires CUDA float32 inputs.")

        from dptb.nn.cuda_ops.grouped_gemm import grouped_gemm_multi

        graph_index = _mole_graph_index(mole_globals, x_inputs[0].shape[0], device=x_inputs[0].device)
        if graph_index.numel() != x_inputs[0].shape[0]:
            raise ValueError(
                f"MOLE graph_index has {graph_index.numel()} rows, but SO2 input has {x_inputs[0].shape[0]} rows."
            )
        if getattr(mole_globals, "_indexed_inputs_are_sorted", False):
            sorted_graph_index = _expand_graph_index_cached(graph_index, x_inputs[0], mole_globals).reshape(-1).to(dtype=torch.long)
            unpermute_idx = None
            permute_idx = None
        else:
            permute_idx, unpermute_idx, sorted_graph_index = mole_globals.indexed_flat_permutation(graph_index, x_inputs[0])

        flat_inputs = []
        mixed_weights = []
        mixed_biases = []
        for x_m, module in zip(x_inputs, self.m_linear):
            fc = module.fc
            flat_x = x_m.reshape(-1, fc.in_features)
            if permute_idx is not None:
                flat_x = flat_x.index_select(0, permute_idx)
            weight, bias = fc._mix_expert_parameters(mole_globals)
            flat_inputs.append(flat_x.contiguous())
            mixed_weights.append(weight.contiguous())
            mixed_biases.append(bias)

        ptr = mole_globals.indexed_segment_ptr(
            sorted_graph_index,
            int(mixed_weights[0].shape[0]),
            prefer_cpu=True,
        )
        flat_outputs = grouped_gemm_multi(flat_inputs, [ptr] * len(flat_inputs), mixed_weights)

        outputs = []
        for flat_out, bias, x_m, module in zip(flat_outputs, mixed_biases, x_inputs, self.m_linear):
            if bias is not None:
                flat_out = flat_out + bias.index_select(0, sorted_graph_index)
            if unpermute_idx is not None:
                flat_out = flat_out.index_select(0, unpermute_idx)
            outputs.append(flat_out.reshape(*x_m.shape[:-1], module.fc.out_features))
        return outputs

    def _forward_cublas_fused_m_pairs_(
            self,
            input_groups: Dict[int, torch.Tensor],
            rot_blocks: Dict[int, torch.Tensor],
            n: int,
            x: torch.Tensor,
            weights: Optional[torch.Tensor],
            mole_globals: MOLEGlobals,
            out_groups: Dict[int, torch.Tensor],
    ) -> None:
        x_inputs = []
        post_radial_weights = []
        for m in range(1, self.m_max + 1):
            radial_weight = (
                weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(1)
                if self.radial_emb else None
            )
            x_m_in = self._assemble_grouped_pair_input(input_groups, rot_blocks, m, n, x)
            if self.front and self.radial_emb:
                x_m_in = x_m_in * radial_weight
                post_radial_weights.append(None)
            else:
                post_radial_weights.append(radial_weight)
            x_inputs.append(x_m_in)

        raw_outputs = self._apply_m_linears_cublas_multi(x_inputs, mole_globals)
        for m, raw_output, radial_weight in zip(range(1, self.m_max + 1), raw_outputs, post_radial_weights):
            linear_output = self.m_linear[m - 1]._finish_linear_output(raw_output)
            if radial_weight is not None:
                linear_output = linear_output * radial_weight
            self._accumulate_grouped_pair_output_(out_groups, linear_output, rot_blocks, m)

    def _forward_streamed_m_major_grouped(self, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
        wigner_D_all = self._ensure_wigner_rotation(R, wigner_D_all)
        wigner_D_return = wigner_D_all
        unpermute_idx = None
        graph_index = getattr(mole_globals, "graph_index", None)
        if (
            os.environ.get("DPTB_SO2_SORTED_EDGE_VIEW", "0") != "0"
            and graph_index is not None
            # sorted_indexed_view is a weight-space graph-token view. It does
            # not permute per-row top-k routes or preserve Switch semantics.
            and not getattr(mole_globals, "activation_space", False)
            and not getattr(mole_globals, "top1_independent", False)
            and self._cueq_linear_is_enabled()
            and not getattr(mole_globals, "_indexed_inputs_are_sorted", False)
        ):
            permute_idx, unpermute_idx, mole_globals = mole_globals.sorted_indexed_view(graph_index, x)
            if permute_idx is not None:
                x = x.index_select(0, permute_idx)
                if latents is not None:
                    latents = latents.index_select(0, permute_idx)
                wigner_D_all = _index_select_wigner_edges(wigner_D_all, permute_idx)
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

            if m == 1 and self._cublas_m_fusion_enabled():
                self._forward_cublas_fused_m_pairs_(
                    input_groups,
                    rot_blocks,
                    n,
                    x,
                    weights,
                    mole_globals,
                    out_groups,
                )
                break

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
        if unpermute_idx is not None:
            out = out.index_select(0, unpermute_idx)
        return out.contiguous(), wigner_D_return


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

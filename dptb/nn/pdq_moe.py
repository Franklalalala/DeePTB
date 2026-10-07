"""PDQ-MoE expert banks, top-k routing and per-forward routing metadata.

H0-routed shared-basis mixture of experts (PDQ-MoE) uses
``W(e) = W_s + sum_k g_k(e) P D_k Q.T``. The routing metadata also supports
legacy graph routing without a second copy of the expert implementation.
"""

import logging
import math
import os
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from dptb.utils.cuda_cache_memory import cuda_cache_memory_probe, record_cuda_cache_event

from .so2_parity import ParityWeightMixin


log = logging.getLogger(__name__)


def _grouped_gemm(x, ptr, weight, *, graph_routing=False):
    """Backend seam for the existing expert and graph grouped-GEMM routes."""
    from .so2_backend import grouped_gemm
    return grouped_gemm(x, ptr, weight)


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


class PDQMoERouting:
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
            raise ValueError("PDQMoERouting.branch must be all, routed or shared; got %r" % (branch,))
        if branch != "all" and not self.activation_space:
            raise ValueError("PDQMoERouting.branch=%r needs activation-space (per-row) routing" % (branch,))
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
        # must stay authoritative across all PDQMoE backends.
        self.graph_index = None if split_sizes is not None else graph_index
        self._sizes_tensor = self._normalize_sizes_tensor(sizes, split_sizes)
        self.split_sizes = self._normalize_split_sizes(sizes, split_sizes)
        self._expanded_graph_index_cache = {}
        self._indexed_flat_permutation_cache = {}
        self._indexed_segment_ptr_cache = {}
        self._indexed_inputs_are_sorted = False

    def expert_slot_layout(self, slot: int, expert_index, num_experts: int):
        """Sort order, inverse, and segment pointer for one top-k slot.

        Derived once per forward and shared by every PDQMoE, since
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
                return PDQMoERouting._tensor_to_split_tuple(split_sizes)
            return tuple(int(v) for v in split_sizes)
        if sizes is None:
            return None
        if torch.is_tensor(sizes):
            if sizes.device.type != "cpu":
                return None
            return PDQMoERouting._tensor_to_split_tuple(sizes)
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
        sorted_view = PDQMoERouting(
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
            split_sizes = PDQMoERouting._tensor_to_split_tuple(sizes_tensor)
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
        mole_globals: PDQMoERouting,
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


def _normalize_mole_linear_mode(mode: str) -> str:
    allowed = {"split_loop", "indexed_ref", "cueq_indexed_linear", "cublas_grouped"}
    if mode not in allowed:
        raise ValueError(f"mole_linear_mode must be one of {sorted(allowed)}, got {mode!r}")
    return mode



def _expert_route_indices_from_globals(
        mole_globals: PDQMoERouting,
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


class PDQMoERouter(nn.Module):
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
                 type_support_seed: int = 0):  # Fixed bias update speed unless a schedule is configured.
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

        # Base load-balancing bias step.
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

        # Optimizer progress is supplied by the router registry.

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
        # Selection starts from the network logits.
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

        # Add the balancing bias before top-k selection.
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

                # Count the load, optionally weighted by graph sizes.
                if sizes is not None:
                    weight = sizes.view(-1, 1, 1)
                    weighted_mask = mask * weight
                    current_load = weighted_mask.sum(dim=(0, 1))
                    target_load = (sizes.sum() * self.top_k) / self.num_experts
                else:
                    current_load = mask.sum(dim=(0, 1))
                    target_load = (scores.size(0) * self.top_k) / self.num_experts

                # Smooth load statistics with an exponential moving average.
                if self.training:
                    self.ema_load.mul_(0.9).add_(current_load, alpha=0.1)
                expert_load_cv = self.ema_load.std() / (self.ema_load.mean() + 1e-8)

            # Update the balancing bias at its configured step size.
            bias_before = self.expert_bias.detach().clone() if (self.training and self.record_train_stats) else None
            bias_step = self._bias_step() if (self.aux_loss_free and self.training) else 0.0
            if bias_step > 0.0:
                with torch.no_grad():
                    error = current_load - target_load
                    self.expert_bias -= torch.sign(error) * bias_step
                    # Center the bias to prevent a common offset from drifting.
                    self.expert_bias -= self.expert_bias.mean()

            # Normalize the selected logits for the renormalized gate.
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

            # Build sparse routing coefficients.
            coeffs = torch.zeros_like(scores)
            coeffs.scatter_(1, topk_indices, topk_probs)

            # Monitor the mean maximum selected probability.
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


class PDQMoE(ParityWeightMixin, nn.Module):
    """Shared-basis expert linear layer with edge or graph routing.

    The shared-core bank represents each expert as ``P @ D_k @ Q.T`` and
    adds full-rank shared weights. Full expert banks remain available for
    controlled comparisons. Routing, mixing order and checkpoint names are
    shared with the legacy linear layer.
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
            raise ValueError("PDQMoE needs nonnegative expert counts and at least one expert")
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

        # Routed expert weights.
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

        # Shared expert weights.
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
            nn.init.uniform_(self._parameters["weight_experts"], -k, k)
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
            nn.init.uniform_(self._parameters["bias_experts"], -k, k)

        if self.num_shared_experts > 0:
            nn.init.uniform_(self._parameters["weight_shared"], -k, k)
            if self.bias_shared is not None:
                nn.init.uniform_(self._parameters["bias_shared"], -k, k)

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
        weight = (self.basis_left.unsqueeze(0) @ self.core_experts) @ self.basis_right.t()
        return self._parity_value("weight_experts", weight)

    @torch.no_grad()
    def scale_expert_weights_(self, scale):
        """Scale the represented bank, including the SO2 m>0 initial scale."""
        if self.num_experts == 0:
            return self
        if self.mole_expert_parameterization == "full":
            self._parameters["weight_experts"].mul_(scale)
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
        weight = self._expert_weight_bank()

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
        flat_out = _grouped_gemm(
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
            raise ValueError("A shared-only PDQMoE has no routed experts to select")
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
                raise ValueError("A shared-only PDQMoE cannot fold shared weights into routed slots")
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
            ys = _grouped_gemm(xs, ptr, weight.contiguous())
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

    def _apply_activation_space(self, x, mole_globals: PDQMoERouting):
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

    def _mix_expert_parameters(self, mole_globals: PDQMoERouting):
        if getattr(mole_globals, "activation_space", False):
            raise RuntimeError(
                "activation-space MoLE reached the weight-space path, which "
                "would materialise one [out_features, in_features] weight per "
                "route token -- exactly what per-edge routing cannot afford. "
                "Reach every PDQMoE through PDQMoE.forward or the "
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

    def _apply_cublas_grouped(self, x, mixed_weights, mixed_bias, graph_index, mole_globals: PDQMoERouting):
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
        flat_out = _grouped_gemm(flat_x.contiguous(), ptr, mixed_weights.contiguous(), graph_routing=True)
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

    def _apply_cueq_indexed_linear(self, x, mixed_weights, mixed_bias, graph_index, mole_globals: PDQMoERouting):
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

    def forward(self, x, mole_globals: PDQMoERouting):
        if self.num_experts == 0:
            if getattr(mole_globals, "branch", "shared") == "routed":
                raise ValueError("A shared-only PDQMoE cannot execute a routed branch")
            bias = self.bias_shared.sum(0) if self.bias_shared is not None else None
            return F.linear(x, self.weight_shared.sum(0), bias)
        if getattr(mole_globals, "top1_independent", False):
            from .top1_prior import linear
            return linear(self, x, mole_globals)
        # Use the mean expert bank when no routing coefficients are supplied.
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

        # Merge graph-level expert parameters.
        # Mix the routed expert weights.
        # coefficients: [Batch, Num_Experts]
        # weight_experts: [Num_Experts, Out, In]
        # mixed_weights: [Batch, Out, In]
        mixed_weights, mixed_bias = self._mix_expert_parameters(mole_globals)

        # The parameter mixture includes shared experts.
        # (W_routed + sum(W_shared)) * x follows distributivity.

        # The mixed bias follows the same graph routing.

        # Apply the graph-specific linear transformation.
        # Split rows by graph because each graph owns one mixed weight.
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

        # Apply each graph weight to its contiguous rows.
        for i, x_sys in enumerate(x_split):
            w = mixed_weights[i]
            b = mixed_bias[i] if mixed_bias is not None else None
            out_parts.append(F.linear(x_sys, w, b))

        return torch.cat(out_parts, dim=0)

# Identity aliases preserve type checks and historical module-qualified imports.
MOLEGlobals = PDQMoERouting
MOLERouterV3 = PDQMoERouter
MOLELinear = PDQMoE
PDQMoELinear = PDQMoE

"""Shared scalar tensor products and expert mixing."""
from typing import Optional, List, Union, Dict, Tuple
import math
import os
import torch
from e3nn import o3
# Note: Modified SO2_Linear and MOLE classes imported here
from dptb.nn.tensor_product_moe_v3 import SO2_Linear, MOLEGlobals, SO2PostActivationExpertMixer
import math


import logging

log = logging.getLogger(__name__)


def _normalize_node_message_aggregation(mode: Optional[str]) -> str:
    mode = mode or "scatter"
    allowed = {"scatter"}
    if mode not in allowed:
        raise ValueError(
            "node_message_aggregation must be one of "
            f"{sorted(allowed)}, got {mode!r}."
        )
    return mode


def _normalize_edge_attention_key_source(source: Optional[str]) -> str:
    source = source or "message"
    allowed = {"message"}
    if source not in allowed:
        raise ValueError(
            "edge_attention_key_source must be one of "
            f"{sorted(allowed)}, got {source!r}."
        )
    return source


def _normalize_onehot_tp_mode(mode: Optional[str]) -> str:
    mode = mode or os.environ.get("DPTB_ONEHOT_TP_MODE", "scalar_fast")
    allowed = {"scalar_fast"}
    if mode not in allowed:
        raise ValueError(
            "0422-cueq-fastest only supports onehot_tp_mode='scalar_fast'. "
            f"Use the legacy cueq branch for e3nn/auto A/B, got {mode!r}."
        )
    return mode


def _normalize_stable_standard_compat_mode(name: str, mode: Optional[str]) -> str:
    if mode in (None, "", "standard"):
        return "standard"
    raise ValueError(
        f"0425-stable accepts only {name}=None or 'standard'. "
        f"Use the Triton experiment branch for {name}={mode!r}."
    )


def _normalize_so2_expert_mixing_mode(mode: Optional[str]) -> str:
    mode = mode or "pre_activation"
    allowed = {"pre_activation", "post_activation", "post_activation_slot", "post_activation_shared"}
    if mode not in allowed:
        raise ValueError(f"so2_expert_mixing_mode must be one of {sorted(allowed)}, got {mode!r}.")
    return mode


def _normalize_so2_moe_layers(layers, n_layers: int) -> Tuple[int, ...]:
    """Layer indices whose SO2 updates own routed experts; all is legacy behavior."""
    if layers is None or layers == "all":
        return tuple(range(n_layers))
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("so2_moe_layers must be 'all' or a nonempty list of layer indices")
    if any(type(i) is not int or not 0 <= i < n_layers for i in layers):
        raise ValueError("so2_moe_layers indices must be integers in [0, n_layers)")
    if len(set(layers)) != len(layers):
        raise ValueError("so2_moe_layers must not repeat a layer index")
    return tuple(sorted(layers))


def _normalize_cg_head_impl(mode: Optional[str]) -> str:
    """P1-3: reduction-path opt-in for the h_b0 late-CG head. ``legacy`` (default)
    reproduces the pre-fusion (0715-refactor) per-path accumulation bit-stably;
    ``fused`` opts in to the grouped-einsum reassociation (see
    ``late_block_expansion_cg.py`` for the full numerics discussion). Validated
    here so a bad value fails fast at embedding construction regardless of which
    output_route is selected -- the h_b0 head itself performs the same check."""
    mode = mode or "legacy"
    allowed = {"legacy", "fused"}
    if mode not in allowed:
        raise ValueError(f"cg_head_impl must be one of {sorted(allowed)}, got {mode!r}.")
    return mode


def _build_so2_post_activation_expert_mixer(
        tp: SO2_Linear,
        activation: torch.nn.Module,
        scalar_dim: int,
        router_hidden_dim: int,
        route_chunk_size: Optional[int],
        route_checkpoint: bool,
) -> SO2PostActivationExpertMixer:
    return SO2PostActivationExpertMixer(
        tp=tp,
        activation=activation,
        router_from_0e=torch.nn.Sequential(
            torch.nn.Linear(scalar_dim, router_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(router_hidden_dim, 1),
        ),
        scalar_dim=scalar_dim,
        route_chunk_size=route_chunk_size,
        checkpoint_routes=route_checkpoint,
    )


def _apply_so2_tp_or_post_activation_mixer(
        module: torch.nn.Module,
        tp_input: torch.Tensor,
        edge_vector: torch.Tensor,
        mole_globals: MOLEGlobals,
        latents: torch.Tensor,
        wigner_D_all,
):
    if module.post_activation_expert_mixer is None:
        features, wigner_D_all = module.tp(
            tp_input,
            edge_vector,
            mole_globals,
            latents,
            wigner_D_all,
        )
        features = module.activation(features)
        return features, wigner_D_all
    return module.post_activation_expert_mixer(
        tp_input,
        edge_vector,
        mole_globals,
        latents,
        wigner_D_all,
    )


def _instruction_get(instruction, name: str, index: int):
    if hasattr(instruction, name):
        return getattr(instruction, name)
    return instruction[index]


def _onehot_weight_shape(connection_mode: str, mul1: int, mul2: int, mul_out: int) -> Tuple[int, ...]:
    if connection_mode == "uvu":
        return (mul1, mul2)
    if connection_mode == "uvw":
        return (mul1, mul2, mul_out)
    raise ValueError(f"unsupported scalar onehot TP connection mode {connection_mode!r}")


def _split_last(t: torch.Tensor, widths: Tuple[int, ...]) -> Tuple[torch.Tensor, ...]:
    """Views of the consecutive blocks of the last dim, taken by one split.

    The split's backward concatenates the block gradients once.  Indexing
    ``t[..., a:b]`` per block instead gives each block a SliceBackward0 that
    zero-fills a gradient of the full width.
    """
    if len(widths) == 1:
        return (t,)
    return torch.split(t, widths, dim=-1)


class ScalarOnehotTP(torch.nn.Module):
    """Scalar-onehot tensor product without storing an e3nn TP module.

    Every instruction couples an irrep of input 1 with a scalar block of input 2 and
    keeps the irrep: ``uvu`` scales each channel by a gain computed from input 2,
    ``uvw`` mixes the channels.  The weight keeps e3nn's flat per-instruction layout,
    so ``from_e3nn`` and existing checkpoints load unchanged.

    The inputs are split once and the output is concatenated once, so the backward
    writes one gradient per input instead of one zero-filled full-width tensor
    (SliceBackward0) and one full-width copy (CopySlices) per instruction.  The
    ``uvw`` instructions of one irrep and one scalar block that couple every input
    irrep of that type with every output irrep of that type exactly once (the layout
    of FullyConnectedTensorProduct) run together: one matmul with input 2, one
    batched matmul with input 1.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        *,
        weight_shapes: Optional[List[Tuple[int, ...]]] = None,
        path_weights: Optional[List[float]] = None,
        weight: Optional[torch.Tensor] = None,
        _layout_only: bool = False,
    ):
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self._in1_widths = tuple(mul * ir.dim for mul, ir in self.irreps_in1)
        self._in2_widths = tuple(mul * ir.dim for mul, ir in self.irreps_in2)
        self._out_widths = tuple(mul * ir.dim for mul, ir in self.irreps_out)

        paths = []
        scales = []
        offset = 0
        for idx, instruction in enumerate(instructions):
            has_weight = _instruction_get(instruction, "has_weight", 4)
            if not has_weight:
                raise ValueError("ScalarOnehotTP requires weighted instructions")

            i_in1 = _instruction_get(instruction, "i_in1", 0)
            i_in2 = _instruction_get(instruction, "i_in2", 1)
            i_out = _instruction_get(instruction, "i_out", 2)
            connection_mode = _instruction_get(instruction, "connection_mode", 3)

            mul1, ir1 = self.irreps_in1[i_in1]
            mul2, ir2 = self.irreps_in2[i_in2]
            mul_out, ir_out = self.irreps_out[i_out]
            if ir2.l != 0 or ir2.p != 1 or ir1 != ir_out:
                raise ValueError("ScalarOnehotTP only supports scalar second input preserving irreps")
            if connection_mode == "uvu" and mul_out != mul1:
                raise ValueError("uvu ScalarOnehotTP instructions need equal input and output multiplicity")

            shape = (
                tuple(weight_shapes[idx])
                if weight_shapes is not None
                else _onehot_weight_shape(connection_mode, mul1, mul2, mul_out)
            )
            numel = math.prod(shape)
            path_weight = (
                float(path_weights[idx])
                if path_weights is not None
                else float(_instruction_get(instruction, "path_weight", 5))
            )
            scale = path_weight / math.sqrt(ir_out.dim)
            paths.append((i_in1, i_in2, i_out, connection_mode, offset, shape, mul1, mul2, mul_out, ir1.dim))
            scales.append(scale)
            offset += numel

        self._paths = tuple(paths)
        self._path_numels = tuple(math.prod(path[5]) for path in self._paths)
        self._uvu_same_input = (
            len(self._paths) > 1
            and all(path[3] == "uvu" for path in self._paths)
            and len({path[1] for path in self._paths}) == 1
            and len({path[7] for path in self._paths}) == 1
        )
        if self._uvu_same_input:
            uvu_gain_scales = []
            for idx, path in enumerate(self._paths):
                uvu_gain_scales.extend([scales[idx]] * path[6])
            self._uvu_total_mul = sum(path[6] for path in self._paths)
            self._uvu_mul2 = self._paths[0][7]
            self._uvu_muls = tuple(path[6] for path in self._paths)
        else:
            uvu_gain_scales = []
            self._uvu_total_mul = 0
            self._uvu_mul2 = 0
            self._uvu_muls = ()
        init_dtype = weight.dtype if weight is not None else torch.get_default_dtype()
        init_device = weight.device if weight is not None else None
        self.register_buffer(
            "_path_scales",
            torch.tensor(scales, dtype=init_dtype, device=init_device),
            persistent=False,
        )
        self.register_buffer(
            "_uvu_gain_scales",
            torch.tensor(uvu_gain_scales, dtype=init_dtype, device=init_device),
            persistent=False,
        )
        groups, group_maps = ([], []) if self._uvu_same_input else self._complete_uvw_groups(scales)
        self._uvw_groups = tuple(groups)
        for g, (index, group_scale) in enumerate(group_maps):
            # Not persistent: derived from the instructions, not part of the state_dict.
            self.register_buffer("_uvw_index_%d" % g, index.to(device=init_device), persistent=False)
            self.register_buffer(
                "_uvw_scale_%d" % g,
                group_scale.to(dtype=init_dtype, device=init_device),
                persistent=False,
            )
        grouped = {idx for group in self._uvw_groups for idx in group[0]}
        self._single_paths = (
            ()
            if self._uvu_same_input
            else tuple(idx for idx in range(len(self._paths)) if idx not in grouped)
        )
        if weight is not None and weight.numel() != offset:
            raise ValueError(f"ScalarOnehotTP weight has {weight.numel()} values, expected {offset}")
        if _layout_only:
            # External-weight adapters need metadata, never an unregistered copy
            # of the source module's Parameter. Safe to construct under torch.func.
            self.register_parameter("weight", None)
            return
        self.weight = torch.nn.Parameter(torch.empty(offset, dtype=init_dtype, device=init_device))
        if weight is None:
            torch.nn.init.normal_(self.weight, std=1.0 / math.sqrt(max(offset, 1)))
        else:
            if weight.numel() != offset:
                raise ValueError(f"ScalarOnehotTP weight has {weight.numel()} values, expected {offset}")
            with torch.no_grad():
                self.weight.copy_(weight.reshape(-1).to(dtype=self.weight.dtype))

    def _complete_uvw_groups(self, scales):
        """uvw paths sharing (irrep, scalar block) that form a complete input x output
        product, each group with its weight gather index and path scales.

        The group matrix is M[v, (u, w)] = scale_path * weight_path[u_local, v, w_local]
        over the stacked input channels u and output channels w of the group."""
        keyed = {}
        for idx, (i_in1, i_in2, _, mode, _, shape, mul1, mul2, mul_out, _) in enumerate(self._paths):
            if mode == "uvw" and tuple(shape) == (mul1, mul2, mul_out):
                keyed.setdefault((self.irreps_in1[i_in1].ir, i_in2), []).append(idx)
        groups, maps = [], []
        for (ir, i_in2), ids in keyed.items():
            in1 = sorted({self._paths[idx][0] for idx in ids})
            out = sorted({self._paths[idx][2] for idx in ids})
            pairs = {(self._paths[idx][0], self._paths[idx][2]) for idx in ids}
            if len(pairs) != len(ids) or len(pairs) != len(in1) * len(out):
                continue
            in1_muls = tuple(self.irreps_in1[i].mul for i in in1)
            out_muls = tuple(self.irreps_out[j].mul for j in out)
            n_u, n_v, n_w = sum(in1_muls), self.irreps_in2[i_in2].mul, sum(out_muls)
            u_start = dict(zip(in1, [sum(in1_muls[:k]) for k in range(len(in1))]))
            w_start = dict(zip(out, [sum(out_muls[:k]) for k in range(len(out))]))
            index = torch.empty((n_v, n_u, n_w), dtype=torch.long)
            group_scale = torch.empty((n_u, n_w), dtype=torch.float64)
            for idx in ids:
                i_in1, _, i_out, _, offset, _, mul1, mul2, mul_out, _ = self._paths[idx]
                flat = offset + torch.arange(mul1 * mul2 * mul_out, dtype=torch.long).reshape(mul1, mul2, mul_out)
                u0, w0 = u_start[i_in1], w_start[i_out]
                index[:, u0:u0 + mul1, w0:w0 + mul_out] = flat.permute(1, 0, 2)
                group_scale[u0:u0 + mul1, w0:w0 + mul_out] = scales[idx]
            groups.append((tuple(ids), i_in2, tuple(in1), in1_muls, tuple(out), out_muls, n_u, n_v, n_w, ir.dim))
            maps.append((index.reshape(-1), group_scale.reshape(-1)))
        return groups, maps

    @classmethod
    def from_e3nn(cls, tp: torch.nn.Module, *, _layout_only: bool = False) -> "ScalarOnehotTP":
        weight_views = list(tp.weight_views())
        path_weights = [float(_instruction_get(instruction, "path_weight", 5)) for instruction in tp.instructions]
        return cls(
            tp.irreps_in1,
            tp.irreps_in2,
            tp.irreps_out,
            tp.instructions,
            weight_shapes=[tuple(view.shape) for view in weight_views],
            path_weights=path_weights,
            weight=tp.weight.detach(),
            _layout_only=_layout_only,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._forward_with_weight(x, y, self.weight)

    def _forward_with_weight(self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.shape[:-1] != y.shape[:-1]:
            raise ValueError(f"onehot TP input shapes must share leading dims, got {x.shape} and {y.shape}")
        if x.shape[-1] != self.irreps_in1.dim or y.shape[-1] != self.irreps_in2.dim:
            raise ValueError(
                f"onehot TP input dims mismatch: got x={x.shape[-1]}, y={y.shape[-1]}, "
                f"expected x={self.irreps_in1.dim}, y={self.irreps_in2.dim}"
            )

        leading_shape = x.shape[:-1]
        # Explicit row count also handles zero-width representations; (-1, 0)
        # cannot infer a dimension from an empty tensor.
        n = math.prod(leading_shape)
        x_flat = x.reshape(n, self.irreps_in1.dim)
        y_flat = y.reshape(n, self.irreps_in2.dim)
        if not self._out_widths:
            return x_flat.new_zeros((*leading_shape, 0))
        dtype = x_flat.dtype
        x_parts = _split_last(x_flat, self._in1_widths)
        y_parts = _split_last(y_flat, self._in2_widths)
        blocks = [None] * len(self._out_widths)

        def accumulate(i_out, value):
            # in the dtype of x, like the former writes into an x-dtype output buffer
            value = value.to(dtype)
            blocks[i_out] = value if blocks[i_out] is None else blocks[i_out] + value

        if self._uvu_same_input:
            packed_weight = weight.reshape(self._uvu_total_mul, self._uvu_mul2)
            packed_weight = packed_weight * self._uvu_gain_scales.to(
                dtype=dtype, device=x_flat.device
            ).unsqueeze(-1)
            gains = torch.matmul(y_parts[self._paths[0][1]], packed_weight.transpose(0, 1))
            for path, gain in zip(self._paths, _split_last(gains, self._uvu_muls)):
                i_in1, _, i_out, _, _, _, mul1, _, _, ir_dim = path
                mixed = x_parts[i_in1].reshape(n, mul1, ir_dim) * gain.unsqueeze(-1)
                accumulate(i_out, mixed.reshape(n, mul1 * ir_dim))
        else:
            for g, group in enumerate(self._uvw_groups):
                _, i_in2, in1, in1_muls, out, out_muls, n_u, n_v, n_w, ir_dim = group
                index = getattr(self, "_uvw_index_%d" % g)
                group_scale = getattr(self, "_uvw_scale_%d" % g).to(dtype=weight.dtype)
                matrix = weight.index_select(0, index).reshape(n_v, n_u * n_w) * group_scale
                mix = torch.matmul(y_parts[i_in2], matrix).reshape(n, n_u, n_w)
                x_group = [x_parts[i].reshape(n, mul, ir_dim) for i, mul in zip(in1, in1_muls)]
                x_group = x_group[0] if len(x_group) == 1 else torch.cat(x_group, dim=1)
                # [n, w, u] @ [n, u, ir_dim]: the sum over the stacked input channels
                out_group = torch.bmm(mix.transpose(1, 2), x_group)
                parts = (out_group,) if len(out) == 1 else torch.split(out_group, out_muls, dim=1)
                for i_out, mul, part in zip(out, out_muls, parts):
                    accumulate(i_out, part.reshape(n, mul * ir_dim))
            if self._single_paths:
                path_scales = self._path_scales.to(dtype=dtype, device=x_flat.device)
                weight_parts = _split_last(weight, self._path_numels)
                for idx in self._single_paths:
                    i_in1, i_in2, i_out, connection_mode, _, shape, mul1, mul2, mul_out, ir_dim = self._paths[idx]
                    path_weight = weight_parts[idx].reshape(shape)
                    x_block = x_parts[i_in1].reshape(n, mul1, ir_dim)
                    y_block = y_parts[i_in2].reshape(n, mul2)
                    if connection_mode == "uvu":
                        mixed = x_block * torch.einsum("nv,uv->nu", y_block, path_weight).unsqueeze(-1)
                    elif connection_mode == "uvw":
                        mixed = torch.einsum("nui,nv,uvw->nwi", x_block, y_block, path_weight)
                    else:
                        raise ValueError(f"unsupported scalar onehot TP connection mode {connection_mode!r}")
                    accumulate(i_out, (mixed * path_scales[idx]).reshape(n, mul_out * ir_dim))

        for i_out, block in enumerate(blocks):
            if block is None:
                blocks[i_out] = x_flat.new_zeros((n, self._out_widths[i_out]))
        out = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)
        return out.reshape(*leading_shape, self.irreps_out.dim)


def _scalar_onehot_tp_fast(tp: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ScalarOnehotTP arithmetic for an e3nn TensorProduct module, on that module's weight.

    The layout (instructions, path scales, gather maps) is built once per device and
    dtype and kept outside the module's registered state, so its state_dict and
    parameters are unchanged.
    """
    # Version the key so old, parameter-bearing cached layouts cannot be reused.
    # TP instructions are immutable after construction; actual weight VALUES are
    # deliberately not cached (functional_call and optimizer updates stay live).
    key = (2, x.device, x.dtype, tp.weight.device, tp.weight.dtype)
    cached = tp.__dict__.get("_scalar_onehot_layout")
    transformed = getattr(torch._C, "_are_functorch_transforms_active", lambda: False)()
    if cached is None or cached[0] != key:
        # Inference-mode-created indices/scales cannot later be saved for
        # backward. Build normal constants even during an inference warm-up.
        with torch.inference_mode(False), torch.no_grad():
            layout = ScalarOnehotTP.from_e3nn(tp, _layout_only=True).to(device=x.device, dtype=x.dtype)
        cached = (key, layout)
        # A cold transform may wrap newly created constants. Use them for this
        # invocation only, rather than leaking interpreter tensors into tp.
        if not transformed:
            tp.__dict__["_scalar_onehot_layout"] = cached
    return cached[1]._forward_with_weight(x, y, tp.weight)


def _capture_shift_hidden(module, data, layer_index: int, node_features: torch.Tensor,
                          num_nodes_total: int, active_edges: torch.Tensor) -> None:
    """Opt-in response-head capture of the INPUT of the last interaction layer.

    Shared by every LEM MoE v3 forward (base, H0, edge, edge+H0). Rows of
    inactive trailing nodes are zero-padded exactly as the post-layer padding.
    Disabled (default) it does nothing, so off-mode outputs stay bitwise v9.
    """
    if not getattr(module, "capture_shift_hidden", False) or layer_index != len(module.layers) - 1:
        return
    captured = node_features
    if captured.shape[0] < num_nodes_total:
        captured = torch.cat((captured, captured.new_zeros(
            (num_nodes_total - captured.shape[0], captured.shape[1]))), dim=0)
    data["_shift_node_features"] = captured
    data["_shift_active_edges"] = active_edges


def _apply_onehot_tp(tp: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    mode = _normalize_onehot_tp_mode(mode)
    if isinstance(tp, ScalarOnehotTP):
        return tp(x, y)
    return _scalar_onehot_tp_fast(tp, x, y)



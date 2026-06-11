from __future__ import annotations

import logging
import math
import os
import time
from typing import Optional

import torch
from e3nn import o3
from torch import nn

try:
    import openequivariance as oeq
except ImportError:
    oeq = None

log = logging.getLogger(__name__)

_FALSE = {"", "0", "false", "False", "FALSE", "off", "OFF", "no", "No"}
_SUPPORTED_TP_MODES = {"uvw", "uvu", "uvv", "uuw", "uuu", "uvuv"}
_SCALAR_DIRECT_BACKENDS = {"scalar", "scalar_direct", "scalar_side"}


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in _FALSE


def _path_multiplicity(irreps_in1: o3.Irreps, irreps_in2: o3.Irreps, instruction: tuple[int, int, int, str, bool]) -> int:
    i_in1, i_in2, _, mode, _ = instruction
    if mode == "uvw":
        return int(irreps_in1[i_in1].mul * irreps_in2[i_in2].mul)
    if mode == "uvu":
        return int(irreps_in2[i_in2].mul)
    if mode in ("uvv", "uuw"):
        return int(irreps_in1[i_in1].mul)
    if mode in ("uuu", "uvuv"):
        return 1
    raise NotImplementedError(f"Unsupported TP mode: {mode}")


def _all_scalar_irreps(irreps: o3.Irreps) -> bool:
    return all(ir.l == 0 for _, ir in irreps)


def _scalar_side_weight_numel(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: list[tuple[int, int, int, str, bool, float]],
) -> int:
    numel = 0
    for i_in1, i_in2, i_out, mode, has_weight, _ in instructions:
        if not has_weight:
            continue
        if mode != "uvw":
            raise NotImplementedError(f"scalar-side direct TP only supports uvw, got {mode}")
        numel += int(irreps_in1[i_in1].mul * irreps_in2[i_in2].mul * irreps_out[i_out].mul)
    return numel


class ScalarSideTensorProduct(nn.Module):
    """Direct PyTorch implementation for TP where the second input is scalar-only.

    This avoids OpenEquivariance JIT for onehot/species-style tensor products.
    It is intentionally narrow: only weighted ``uvw`` paths are supported.
    """

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: list[tuple[int, int, int, str, bool, float]],
    ):
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.instructions = list(instructions)
        if not _all_scalar_irreps(self.irreps_in2):
            raise ValueError("ScalarSideTensorProduct requires scalar-only irreps_in2")
        if any(mode != "uvw" for _, _, _, mode, _, _ in self.instructions):
            raise NotImplementedError("ScalarSideTensorProduct only supports uvw instructions")

        self.in1_slices = self.irreps_in1.slices()
        self.in2_slices = self.irreps_in2.slices()
        self.out_slices = self.irreps_out.slices()
        self.weight_numel = _scalar_side_weight_numel(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self.instructions,
        )

    @staticmethod
    def can_run(irreps_in2: o3.Irreps, instructions: list[tuple[int, int, int, str, bool, float]]) -> bool:
        return _all_scalar_irreps(o3.Irreps(irreps_in2)) and all(mode == "uvw" for _, _, _, mode, _, _ in instructions)

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.weight_numel > 0 and weight is None:
            raise ValueError("ScalarSideTensorProduct requires an external weight tensor")
        if self.weight_numel == 0:
            weight = x.new_empty((0,))
        else:
            weight = weight.reshape(-1)
            if weight.numel() != self.weight_numel:
                raise ValueError(f"Expected {self.weight_numel} TP weights, got {weight.numel()}")

        out = x.new_zeros((*x.shape[:-1], self.irreps_out.dim))
        flat_x = x.reshape(-1, x.shape[-1])
        flat_y = y.reshape(-1, y.shape[-1])
        flat_out = out.reshape(-1, out.shape[-1])

        offset = 0
        for i_in1, i_in2, i_out, _, has_weight, path_weight in self.instructions:
            if not has_weight:
                continue
            mul_1, ir_in1 = self.irreps_in1[i_in1]
            mul_2, _ = self.irreps_in2[i_in2]
            mul_out, ir_out = self.irreps_out[i_out]
            if ir_in1.dim != ir_out.dim:
                raise ValueError("Scalar-side TP expects matching input/output irrep dimensions")

            block_numel = int(mul_1 * mul_2 * mul_out)
            w = weight[offset : offset + block_numel].reshape(mul_2, mul_1, mul_out)
            offset += block_numel

            xb = flat_x[:, self.in1_slices[i_in1]].reshape(-1, mul_1, ir_in1.dim)
            yb = flat_y[:, self.in2_slices[i_in2]].reshape(-1, mul_2)
            update = torch.einsum("nud,nv,vuw->nwd", xb, yb, w)
            norm = path_weight / math.sqrt(float(mul_1 * mul_2))
            if norm != 1.0:
                update = update * norm
            out_slice = self.out_slices[i_out]
            flat_out[:, out_slice] += update.reshape(-1, mul_out * ir_out.dim)

        return out


def get_feasible_tp(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    filter_irreps_out: o3.Irreps,
    tp_mode: str = "uvw",
    trainable: bool = True,
    *,
    path_normalization: str = "unit",
    sort_irreps: bool = False,
):
    """Generate OpenEquivariance-compatible irreps and instructions.

    ``path_normalization='unit'`` matches the existing EMolES wrappers.
    ``path_normalization='e3nn'`` matches the legacy LEM MoE OEQ wrapper.
    """

    if tp_mode not in _SUPPORTED_TP_MODES:
        raise NotImplementedError(f"Unsupported TP mode: {tp_mode}")
    if path_normalization not in {"unit", "e3nn"}:
        raise ValueError(f"Unsupported path_normalization={path_normalization!r}")

    irreps_in1 = o3.Irreps(irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    filter_irreps_out = o3.Irreps(filter_irreps_out)
    irreps_mid = []
    instructions = []

    for i, (mul_1, ir_in1) in enumerate(irreps_in1):
        for j, (mul_2, ir_in2) in enumerate(irreps_in2):
            if tp_mode in ("uuw", "uuu") and mul_1 != mul_2:
                continue

            for ir_out in ir_in1 * ir_in2:
                if ir_out not in filter_irreps_out:
                    continue

                if tp_mode == "uvw":
                    mul_out = filter_irreps_out.count(ir_out)
                elif tp_mode == "uvu":
                    mul_out = mul_1
                elif tp_mode == "uvv":
                    mul_out = mul_2
                elif tp_mode == "uuu":
                    mul_out = mul_1
                elif tp_mode == "uuw":
                    mul_out = filter_irreps_out.count(ir_out)
                elif tp_mode == "uvuv":
                    mul_out = mul_1 * mul_2
                else:
                    raise NotImplementedError(f"Unsupported TP mode: {tp_mode}")

                found_k = -1
                for k, (mul, ir) in enumerate(irreps_mid):
                    if ir == ir_out and mul == mul_out:
                        found_k = k
                        break
                if found_k == -1:
                    found_k = len(irreps_mid)
                    irreps_mid.append((mul_out, ir_out))
                instructions.append((i, j, found_k, tp_mode, trainable))

    irreps_mid_obj = o3.Irreps(irreps_mid)

    if path_normalization == "e3nn":
        path_weights = []
        for instruction in instructions:
            _, _, i_out, _, _ = instruction
            alpha = irreps_mid_obj[i_out].ir.dim
            path_count = sum(
                _path_multiplicity(irreps_in1, irreps_in2, candidate)
                for candidate in instructions
                if candidate[2] == i_out
            )
            path_weights.append(math.sqrt(alpha / path_count) if path_count > 0 else 1.0)
    else:
        path_weights = [1.0] * len(instructions)

    if sort_irreps:
        irreps_mid_obj, permutation, _ = irreps_mid_obj.sort()
    else:
        permutation = list(range(len(irreps_mid_obj)))

    final_instructions = []
    for instruction, path_weight in zip(instructions, path_weights):
        i_in1, i_in2, i_out, mode, train = instruction
        final_instructions.append((i_in1, i_in2, permutation[i_out], mode, train, path_weight))

    return irreps_mid_obj, final_instructions


class OEQTensorProduct(nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        tp_mode: str = "uvw",
        *,
        mode: Optional[str] = None,
        internal_weights: bool = True,
        shared_weights: bool = True,
        path_normalization: str = "unit",
        sort_irreps: bool = False,
        simplify_post_linear: bool = False,
        device=None,
        backend: Optional[str] = None,
    ):
        super().__init__()
        if mode is not None:
            tp_mode = mode
        backend = (backend or os.environ.get("DPTB_OEQ_TP_BACKEND", "oeq")).lower()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.internal_weights_flag = internal_weights
        self.device_hint = device

        self.irreps_mid, instructions = get_feasible_tp(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            tp_mode=tp_mode,
            trainable=True,
            path_normalization=path_normalization,
            sort_irreps=sort_irreps,
        )

        direct_requested = backend in _SCALAR_DIRECT_BACKENDS
        if direct_requested:
            if not ScalarSideTensorProduct.can_run(self.irreps_in2, instructions):
                raise NotImplementedError(
                    "DPTB_OEQ_TP_BACKEND=scalar_direct only supports scalar-side uvw tensor products"
                )
            self.problem = None
            self.tp = ScalarSideTensorProduct(self.irreps_in1, self.irreps_in2, self.irreps_mid, instructions)
            self.compile_seconds = 0.0
            self.weight_numel = self.tp.weight_numel
            if _truthy_env("DPTB_OEQ_TP_LOG_COMPILE"):
                log.warning(
                    "Scalar-side TensorProduct selected: %s x %s -> %s mode=%s weights=%d",
                    self.irreps_in1,
                    self.irreps_in2,
                    self.irreps_mid,
                    tp_mode,
                    self.weight_numel,
                )
        else:
            if backend != "oeq":
                raise ValueError(f"Unsupported OEQTensorProduct backend={backend!r}")
            if oeq is None:
                raise ImportError("OpenEquivariance not installed.")
            self.problem = oeq.TPProblem(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_mid,
                instructions,
                shared_weights=shared_weights,
                internal_weights=False,
            )
            t0 = time.perf_counter()
            self.tp = oeq.TensorProduct(self.problem, torch_op=True)
            self.compile_seconds = time.perf_counter() - t0
            self.weight_numel = self.problem.weight_numel
            if _truthy_env("DPTB_OEQ_TP_LOG_COMPILE"):
                log.warning(
                    "OEQ TensorProduct compiled in %.3fs: %s x %s -> %s mode=%s weights=%d",
                    self.compile_seconds,
                    self.irreps_in1,
                    self.irreps_in2,
                    self.irreps_mid,
                    tp_mode,
                    self.weight_numel,
                )

        if self.internal_weights_flag and self.weight_numel > 0:
            self.weights = nn.Parameter(torch.randn(self.weight_numel))
            with torch.no_grad():
                self.weights.div_(self.weight_numel ** 0.5)
        else:
            self.register_parameter("weights", None)

        if simplify_post_linear:
            needs_post_linear = self.irreps_mid.simplify() != self.irreps_out.simplify()
        else:
            needs_post_linear = self.irreps_mid != self.irreps_out
        self.post_linear = o3.Linear(self.irreps_mid, self.irreps_out) if needs_post_linear else nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None):
        w = self.weights if self.internal_weights_flag else weight
        if self.weight_numel > 0:
            out = self.tp(x, y, w)
        else:
            out = self.tp(x, y)
        return self.post_linear(out)

"""e3nn ``o3.Linear`` and ``nn.Gate`` with forwards that split their input once.

The generated forwards of both modules take every input irrep block by ``narrow``
(``Gate`` also copies the blocks into zero-filled outputs).  In the backward each such
view zero-fills a gradient of the full input width (``SliceBackward0``, ``CopySlices``),
once per irrep block.  These subclasses split the input once, so the backward
concatenates the block gradients once.  Parameters, buffers, instructions,
normalization and state_dict keys are e3nn's; outputs agree with e3nn's to float
rounding (the summation order of the generated einsums differs).
"""
import copy
from typing import List, Optional, Tuple

import torch
from e3nn import nn as e3nn_nn
from e3nn import o3
from e3nn.util.jit import compile_mode


def _split(x: torch.Tensor, widths: List[int]) -> Tuple[torch.Tensor, ...]:
    if len(widths) == 1:
        return (x,)
    return torch.split(x, widths, dim=-1)


@compile_mode(None)
class Linear(o3.Linear):
    """``e3nn.o3.Linear``; the forward with internal shared weights splits the input once.

    Explicit ``weight``/``bias`` arguments, external or per-sample weights and the
    ``f_in``/``f_out`` form run e3nn's generated forward."""

    def __init__(self, irreps_in, irreps_out, **kwargs) -> None:
        super().__init__(irreps_in, irreps_out, **kwargs)
        self._fast = (kwargs.get("f_in") is None and kwargs.get("f_out") is None
                      and self.internal_weights and self.shared_weights)
        self._in_widths = [mul_ir.dim for mul_ir in self.irreps_in]
        # the flat weight/bias layout of e3nn's codegen: instructions with an empty
        # path are dropped, weights then biases in instruction order
        paths = [ins for ins in self.instructions if 0 not in ins.path_shape]
        weights = [ins for ins in paths if ins.i_in != -1]
        biases = [ins for ins in paths if ins.i_in == -1]
        self._weight_sizes = [ins.path_shape[0] * ins.path_shape[1] for ins in weights]
        self._bias_sizes = [ins.path_shape[0] for ins in biases]
        outputs = []
        for i_out, mul_ir_out in enumerate(self.irreps_out):
            if mul_ir_out.mul == 0:
                continue
            terms = [
                (k, ins.i_in, ins.path_shape[0], float(ins.path_weight))
                for k, ins in enumerate(weights)
                if ins.i_out == i_out
            ]
            bias = [k for k, ins in enumerate(biases) if ins.i_out == i_out]
            outputs.append((mul_ir_out.mul, mul_ir_out.ir.dim, tuple(terms), bias[0] if bias else -1))
        self._outputs = tuple(outputs)

    def forward(self, features, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None):
        if not self._fast or weight is not None or bias is not None:
            return super().forward(features, weight, bias)
        lead = features.shape[:-1]
        n = 1
        for size in lead:
            n *= int(size)
        x = features.reshape(n, self.irreps_in.dim)  # explicit rows: reshape(-1, 0) is ambiguous
        if not self._outputs:
            return x.new_zeros(*lead, self.irreps_out.dim)
        xs = _split(x, self._in_widths)
        ws = _split(self.weight, self._weight_sizes) if self._weight_sizes else ()
        bs = _split(self.bias, self._bias_sizes) if self._bias_sizes else ()
        outs = []
        for mul_out, ir_dim, terms, i_bias in self._outputs:
            out = None
            for k, i_in, mul_in, path_weight in terms:
                # the path weight scales the weight, not the [n, width] output
                w = ws[k].reshape(mul_in, mul_out) * path_weight
                if ir_dim == 1:
                    term = torch.matmul(xs[i_in], w)
                else:
                    term = torch.einsum("uw,zui->zwi", w, xs[i_in].reshape(n, mul_in, ir_dim))
                    term = term.reshape(n, mul_out * ir_dim)
                out = term if out is None else out + term
            if i_bias >= 0:
                b = bs[i_bias].reshape(1, mul_out * ir_dim)
                out = b.expand(n, -1) if out is None else out + b
            if out is None:
                out = x.new_zeros(n, mul_out * ir_dim)
            outs.append(out)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        return out.reshape(*lead, self.irreps_out.dim)


@compile_mode(None)
class Gate(e3nn_nn.Gate):
    """``e3nn.nn.Gate``; the forward splits the input once and gates each block directly.

    The scalars and gates pass through e3nn's normalized activations
    (``self.act_scalars``, ``self.act_gates``) as before.  Each gated block is
    multiplied by its gate channel, which is what the elementwise tensor product
    computes: its path weight times the Wigner 3j entry of l x 0 -> l is 1 up to
    rounding (a constant that is not is applied).  The construction checks this against
    ``self.mul`` in float64 and keeps e3nn's forward if the product has any other form."""

    def __init__(self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated) -> None:
        super().__init__(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)
        cut = self.sc.cut
        self._in_widths = [mul_ir.dim for mul_ir in cut.irreps_in]
        self._scalar_blocks, self._gate_blocks, self._gated_blocks = (tuple(ins) for ins in cut.instructions)
        self._products = self._gated_products()

    def _gated_products(self):
        """(mul, ir_dim, constant) per gated block, in the order of self.irreps_gated, or
        None when self.mul is not the per-channel product this forward computes."""
        mul = self.mul
        in1, in2 = list(mul.irreps_in1), list(mul.irreps_in2)
        gated = [(m, ir) for m, ir in self.irreps_gated]
        if [(m, ir) for m, ir in in1] != gated or len(mul.instructions) != len(in1):
            return None
        products = []
        for i, ins in enumerate(mul.instructions):
            if (ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode) != (i, i, i, "uuu") or in2[i][1] != o3.Irrep("0e"):
                return None
            products.append((in1[i][0], in1[i][1].dim, 1.0))
        if not products:
            return ()
        # the constant of each block, read off one channel, and a check of the whole
        # product on random inputs
        with torch.no_grad():
            generator = torch.Generator().manual_seed(0)
            x1 = torch.randn(3, self.irreps_gated.dim, dtype=torch.float64, generator=generator)
            x2 = torch.randn(3, self.irreps_gates.dim, dtype=torch.float64, generator=generator)
            ref = copy.deepcopy(mul).to(torch.float64)(x1, x2)
            constants = []
            got = []
            offset1 = offset2 = 0
            for m, d, _ in products:
                a = x1[:, offset1:offset1 + m * d].reshape(3, m, d)
                g = x2[:, offset2:offset2 + m].unsqueeze(-1)
                r = ref[:, offset1:offset1 + m * d].reshape(3, m, d)
                c = float((r / (a * g))[0, 0, 0])
                constants.append(1.0 if abs(c - 1.0) < 1e-6 else c)
                got.append((a * g * c).reshape(3, m * d))
                offset1 += m * d
                offset2 += m
            if not torch.allclose(torch.cat(got, dim=-1), ref, rtol=1e-12, atol=1e-12):
                return None
        return tuple((m, d, c) for (m, d, _), c in zip(products, constants))

    @staticmethod
    def _take(blocks, indices):
        if len(indices) == 0:
            return None
        if len(indices) == 1:
            return blocks[indices[0]]
        return torch.cat([blocks[i] for i in indices], dim=-1)

    def forward(self, features):
        if self._products is None:
            return super().forward(features)
        blocks = _split(features, self._in_widths)
        scalars = self._take(blocks, self._scalar_blocks)
        outs = [] if scalars is None else [self.act_scalars(scalars)]
        if not self._gate_blocks:
            return outs[0] if outs else features.new_zeros(features.shape[:-1] + (0,))
        gates = self.act_gates(self._take(blocks, self._gate_blocks))
        lead = features.shape[:-1]
        gate_parts = _split(gates, [m for m, _, _ in self._products])
        for i_block, (m, d, c), gate in zip(self._gated_blocks, self._products, gate_parts):
            gated = blocks[i_block].reshape(*lead, m, d) * gate.unsqueeze(-1)
            if c != 1.0:
                gated = gated * c
            outs.append(gated.reshape(*lead, m * d))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)

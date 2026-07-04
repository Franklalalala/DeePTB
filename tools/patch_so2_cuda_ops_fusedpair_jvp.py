#!/usr/bin/env python
"""Add forward-mode AD (setup_context + jvp) to so2_cuda_ops _FusedPairFunction.

The production SO2CUDA fast line (so2_fusion_mode=streamed_m_major_fused_p0)
routes the SO2 tensor product through ``_FusedPairFunction`` in the *external*
``so2_cuda_ops`` package. To run the pixel-meanflow ``du_dt_backend=jvp``
backend on that line, this Function must expose a forward-mode rule.

The fused pair ``out = fused_pair(x, wigner, gi, mixed_weight, radial, ...)`` is
**trilinear**: linear in ``x``, in ``mixed_weight`` and in ``radial`` separately
(wigner / indices / metadata carry no tangent). Its jvp is therefore the sum,
over each differentiable input that carries a tangent, of the same forward with
that input replaced by its tangent -- exact, and it may call the CUDA kernel
because native ``torch.autograd.forward_ad`` dual tensors have real storage
(unlike functorch wrappers).

This patcher is idempotent and anchored on the exact current source; it backs
up the original to ``tensor_product.py.bak_jvp``. Point SO2_CUDA_OPS_SRC at the
editable so2_cuda_ops source tree (the package must be installed editable).

Usage:
    SO2_CUDA_OPS_SRC=/path/to/SO2CUDA/src/so2_cuda_ops \\
        python tools/patch_so2_cuda_ops_fusedpair_jvp.py
"""

from __future__ import annotations

import os
import py_compile
import shutil
import sys
from pathlib import Path


def main() -> int:
    root = os.environ.get("SO2_CUDA_OPS_SRC")
    if not root:
        try:
            import so2_cuda_ops

            root = str(Path(so2_cuda_ops.__file__).resolve().parent)
        except Exception:
            sys.stderr.write(
                "set SO2_CUDA_OPS_SRC to the so2_cuda_ops source dir "
                "(could not import so2_cuda_ops)\n"
            )
            return 2

    src = Path(root) / "tensor_product.py"
    if not src.exists():
        sys.stderr.write(f"not found: {src}\n")
        return 2

    marker = "# DeePTB-pMF-jvp-patch:"
    class_header = "class _FusedPairFunction(torch.autograd.Function):"

    def _class_block(text):
        # the _FusedPairFunction class body only, so idempotence/collision
        # checks don't trip on hooks belonging to a *different* Function.
        start = text.find(class_header)
        if start < 0:
            return ""
        nxt = text.find("\nclass ", start + 1)
        return text[start:] if nxt < 0 else text[start:nxt]

    txt = src.read_text()
    block = _class_block(txt)
    if not block:
        sys.stderr.write("_FusedPairFunction class not found (source drifted)\n")
        return 3
    if marker in block:
        print("ALREADY PATCHED")
        return 0
    if (
        "def setup_context(" in block
        or "def jvp(" in block
    ):
        sys.stderr.write(
            "_FusedPairFunction already has forward-AD hooks but not the DeePTB "
            "marker; refusing an ambiguous patch (source drifted upstream)\n"
        )
        return 3

    sig_old = (
        "class _FusedPairFunction(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(\n"
        "        ctx,\n"
        "        x,\n"
        "        wigner,"
    )
    sig_new = (
        "class _FusedPairFunction(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(\n"
        "        x,\n"
        "        wigner,"
    )
    if sig_old not in txt:
        sys.stderr.write("forward-signature anchor not found (source drifted)\n")
        return 3

    save_old = '''        ctx.save_for_backward(
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
    def backward(ctx, grad_out):'''

    save_new = '''        return out

    # DeePTB-pMF-jvp-patch: forward-AD hooks for the pixel-meanflow jvp backend
    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            x, wigner, graph_index, mixed_weight, radial,
            in_base, in_l, out_base, out_l, offsets, compact_offsets,
            out_dim, m, rotate_in, rotate_out, radial_on_input,
            wigner_mode, wigner_stride,
        ) = inputs
        _saved = (x, wigner, graph_index, mixed_weight, radial,
                  in_base, in_l, out_base, out_l, offsets, compact_offsets)
        ctx.save_for_backward(*_saved)
        ctx.save_for_forward(*_saved)
        ctx.meta = (
            int(out_dim), int(m), bool(rotate_in), bool(rotate_out),
            bool(radial_on_input), int(wigner_mode), int(wigner_stride),
        )

    @staticmethod
    def jvp(ctx, *tangents):
        # SO2 fused pair is trilinear: linear in x, mixed_weight, radial.
        # d_out = sum over each input carrying a tangent of forward(that input
        # replaced by its tangent). Non-differentiable inputs (wigner, indices,
        # metadata) have None tangents. Runs the primal CUDA kernel on the
        # tangents -- allowed for a jvp staticmethod (no vmap), and native
        # forward_ad dual tensors carry storage so the kernel accepts them.
        x_t = tangents[0]
        mixed_weight_t = tangents[3]
        radial_t = tangents[4]
        (x, wigner, graph_index, mixed_weight, radial,
         in_base, in_l, out_base, out_l, offsets, compact_offsets) = ctx.saved_tensors
        (out_dim, m, rotate_in, rotate_out, radial_on_input,
         wigner_mode, wigner_stride) = ctx.meta

        def _raw(xx, ww, rr):
            return _FusedPairFunction.forward(
                xx, wigner, graph_index, ww, rr,
                in_base, in_l, out_base, out_l, offsets, compact_offsets,
                out_dim, m, rotate_in, rotate_out, radial_on_input,
                wigner_mode, wigner_stride,
            )

        out_t = None
        if x_t is not None:
            out_t = _raw(x_t, mixed_weight, radial)
        if mixed_weight_t is not None:
            term = _raw(x, mixed_weight_t, radial)
            out_t = term if out_t is None else out_t + term
        if radial_t is not None and radial.numel() > 0:
            term = _raw(x, mixed_weight, radial_t)
            out_t = term if out_t is None else out_t + term
        return out_t

    @staticmethod
    def backward(ctx, grad_out):'''

    if save_old not in txt:
        sys.stderr.write("save/backward anchor not found (source drifted)\n")
        return 3

    # Verify all anchors, build in memory, confirm the marker landed in the
    # target class, then compile a temp file and atomically replace -- so a bad
    # replacement or compile failure never leaves a broken editable package.
    patched = txt.replace(sig_old, sig_new, 1).replace(save_old, save_new, 1)
    if marker not in _class_block(patched):
        sys.stderr.write("internal error: marker not inserted into _FusedPairFunction\n")
        return 3

    backup = Path(str(src) + ".bak_jvp")
    if not backup.exists():
        shutil.copy(src, backup)
    tmp = Path(str(src) + ".jvp_tmp.py")
    try:
        tmp.write_text(patched)
        py_compile.compile(str(tmp), doraise=True)
        os.replace(str(tmp), str(src))
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"PATCHED {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

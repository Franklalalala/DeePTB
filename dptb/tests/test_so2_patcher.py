"""CI-runnable tests for tools/patch_so2_cuda_ops_fusedpair_jvp.py.

These exercise the anchored/atomic behaviour of the so2_cuda_ops patcher
against a fake ``tensor_product.py`` (no CUDA / no real so2_cuda_ops needed),
covering review finding 4 (false no-op, broken-file-on-fail) and finding 8
(patcher path was untested).
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

PATCHER = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "patch_so2_cuda_ops_fusedpair_jvp.py"
)

# Minimal fake matching the patcher's exact anchors (forward signature + the
# save_for_backward/ctx.meta/return block), plus a benign sibling Function.
FAKE_SRC = textwrap.dedent(
    '''\
    import torch


    class _OtherFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, g):
            return g


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
            out = x
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
            return (None,) * 18
    '''
)


def _run(src_dir: Path):
    env = dict(os.environ)
    env["SO2_CUDA_OPS_SRC"] = str(src_dir)
    return subprocess.run(
        [sys.executable, str(PATCHER)],
        env=env,
        capture_output=True,
        text=True,
    )


def _write_fake(tmp_path: Path, body: str) -> Path:
    d = tmp_path / "so2_cuda_ops"
    d.mkdir()
    (d / "tensor_product.py").write_text(body)
    return d


def test_patcher_applies_and_is_idempotent(tmp_path):
    d = _write_fake(tmp_path, FAKE_SRC)
    tp = d / "tensor_product.py"

    r1 = _run(d)
    assert r1.returncode == 0, r1.stderr
    assert "PATCHED" in r1.stdout
    patched = tp.read_text()
    # forward-AD hooks landed inside _FusedPairFunction, forward lost its ctx arg
    assert "def setup_context(ctx, inputs, output):" in patched
    assert "def jvp(ctx, *tangents):" in patched
    assert "DeePTB-pMF-jvp-patch:" in patched
    # compiles
    compile(patched, str(tp), "exec")

    # second run is a no-op, not a double-apply
    r2 = _run(d)
    assert r2.returncode == 0
    assert "ALREADY PATCHED" in r2.stdout
    assert tp.read_text() == patched


def test_patcher_refuses_on_drifted_source_without_damage(tmp_path):
    # a source whose _FusedPairFunction no longer matches the save anchor
    drifted = FAKE_SRC.replace("        out = x\n", "        out = x + 0\n").replace(
        "            compact_offsets,\n        )", "            compact_offsets)"
    )
    d = _write_fake(tmp_path, drifted)
    tp = d / "tensor_product.py"
    before = tp.read_text()

    r = _run(d)
    assert r.returncode != 0
    assert "drift" in r.stderr.lower() or "anchor" in r.stderr.lower()
    # source is untouched and still valid python (no broken half-write)
    assert tp.read_text() == before
    compile(tp.read_text(), str(tp), "exec")


def test_patcher_refuses_foreign_hooks_without_marker(tmp_path):
    # _FusedPairFunction already has a setup_context but NOT the DeePTB marker
    # (methods are at 4-space class-body indent in the dedented fake).
    foreign = FAKE_SRC.replace(
        "    @staticmethod\n    def backward(ctx, grad_out):\n        return (None,) * 18",
        "    @staticmethod\n    def setup_context(ctx, inputs, output):\n"
        "        pass\n\n"
        "    @staticmethod\n    def backward(ctx, grad_out):\n        return (None,) * 18",
    )
    assert "def setup_context(" in foreign  # guard: the fixture actually mutated
    d = _write_fake(tmp_path, foreign)
    tp = d / "tensor_product.py"
    before = tp.read_text()

    r = _run(d)
    assert r.returncode != 0
    assert "marker" in r.stderr.lower() or "ambiguous" in r.stderr.lower()
    assert tp.read_text() == before

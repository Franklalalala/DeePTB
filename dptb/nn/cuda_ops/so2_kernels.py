"""Compatibility imports for SO2/MoE CUDA extension entry points."""

from __future__ import annotations


def load_fused_p0_extension():
    from dptb.nn.so2_moe_fused_p0 import _load_extension

    return _load_extension()


def load_persistent_grouped_p1_extension():
    from dptb.nn.so2_moe_persistent_grouped import _load_extension

    return _load_extension()

"""Utilities for evaluation contexts that must keep autograd enabled."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch


GRAD_INFERENCE_ENV = "DPTB_ENABLE_GRAD_INFERENCE"

STREAMED_SO2_FUSION_MODES = frozenset(
    {
        "streamed_m_major_ref",
        "streamed_m_major_cueq",
        "streamed_m_major_fused_p0",
        "streamed_m_major_persistent_grouped_p1",
    }
)
FAST_MOLE_LINEAR_MODES = frozenset({"cublas_grouped", "cueq_indexed_linear"})

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _iter_modules(module: Any) -> Iterator[Any]:
    modules = getattr(module, "modules", None)
    if callable(modules):
        yield from modules()
    elif module is not None:
        yield module


def _parse_env_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{GRAD_INFERENCE_ENV} must be one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {value!r}"
    )


def requires_grad_enabled_inference(module: Any) -> bool:
    """Return whether a model advertises a backend that should avoid no_grad eval.

    The helper intentionally keys off public module attributes instead of importing
    DeePTB model classes. That keeps it usable from standalone downstream scripts
    and avoids changing any model forward semantics.
    """

    for submodule in _iter_modules(module):
        so2_fusion_mode = getattr(submodule, "so2_fusion_mode", None)
        if so2_fusion_mode in STREAMED_SO2_FUSION_MODES:
            return True

        mole_linear_mode = getattr(submodule, "mole_linear_mode", None)
        if mole_linear_mode in FAST_MOLE_LINEAR_MODES:
            return True

    return False


@contextmanager
def grad_enabled_inference(
    module: Optional[Any] = None,
    *,
    enable_grad: Optional[bool] = None,
) -> Iterator[None]:
    """Use no_grad by default unless a DeePTB backend or env flag asks for grad.

    ``enable_grad`` is the preferred explicit control for downstream evaluators.
    If it is left as ``None``, ``DPTB_ENABLE_GRAD_INFERENCE`` takes precedence,
    then the module backend attributes are inspected.
    """

    if enable_grad is None:
        env_value = os.environ.get(GRAD_INFERENCE_ENV)
        if env_value is not None:
            enable_grad = _parse_env_flag(env_value)
        else:
            enable_grad = requires_grad_enabled_inference(module)

    context = torch.enable_grad() if enable_grad else torch.no_grad()
    with context:
        yield


__all__ = [
    "FAST_MOLE_LINEAR_MODES",
    "GRAD_INFERENCE_ENV",
    "STREAMED_SO2_FUSION_MODES",
    "grad_enabled_inference",
    "requires_grad_enabled_inference",
]

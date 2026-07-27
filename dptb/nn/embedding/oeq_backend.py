"""Shared optional import of ``openequivariance``.

The four OEQ embedding modules used to each print a bare warning at import
time, and dptb/nn/embedding/__init__.py imports all of them eagerly on the CLI
path — so the default (non-Linux/non-GPU) install printed the same line four
times before doing anything. The backend is resolved here once, silently, and
the warning is emitted lazily (at most once) by the routes that actually need
it, right before their existing fail-closed ImportError.
"""

from __future__ import annotations

import logging

try:
    import openequivariance as oeq
except ImportError:
    oeq = None

log = logging.getLogger(__name__)

OPENEQUIVARIANCE_MISSING_MESSAGE = (
    "openequivariance is not installed, so the OEQ embedding routes are "
    'unavailable. Install it with pip install ".[openequi]" (Linux GPU, '
    "Python >=3.10)."
)

_warned = False


def warn_openequivariance_missing() -> None:
    """Emit the missing-backend warning once per process."""

    global _warned
    if oeq is not None or _warned:
        return
    _warned = True
    log.warning(OPENEQUIVARIANCE_MISSING_MESSAGE)


__all__ = [
    "OPENEQUIVARIANCE_MISSING_MESSAGE",
    "oeq",
    "warn_openequivariance_missing",
]

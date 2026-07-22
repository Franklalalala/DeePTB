# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single source of truth for HamilBlockwiseNexTHamLoss's endpoint metric space.

Deliberately dependency-free (stdlib only: no torch, no ``dptb.data``, no
``dptb.nnops.loss``).  Two very different callers need the exact same formula
and must never let their copies drift apart:

* ``dptb.nnops.blockwise_nextham_loss.HamilBlockwiseNexTHamLoss`` (needs torch)
  computes it at construction time and again when rebuilding the endpoint loss
  from additive flow statistics.
* ``dptb.utils.argcheck.validate_block_ode_contract`` (a config-time validator)
  needs it to reject, before training starts, any blockwise loss configuration
  whose endpoint metric space would not be ``"block"`` while block-ODE is
  active.  Block-ODE flows only ever publish block-space endpoint statistics
  and validation refuses to fall back to a direct criterion recompute for
  block-ODE (see ``dptb.nnops.trainer``), so a mismatched metric space is not
  a soft warning -- it is a guaranteed crash on the first validation step.

``dptb.utils.argcheck`` must stay importable without pulling torch/model
modules in transitively: ``dptb.data``'s package ``__init__`` imports
``dptb.utils.argcheck`` (via ``dptb.data.build``), so ``argcheck`` importing
anything under ``dptb.data`` or ``dptb.nnops.loss`` -- both of which
``blockwise_nextham_loss.py`` imports -- would be a circular import.  This
module has none of those dependencies, so both sides can import it safely.
"""

from __future__ import annotations

# Optimization modes whose *gradient* loss is already computed in the
# old-RME-compatible feature space (see
# ``HamilBlockwiseNexTHamLoss.optimization``).
FEATURE_ENDPOINT_OPTIMIZATION_MODES = frozenset(
    {"feature", "feature_compatible", "compat"}
)


def endpoint_metric_space_for_options(
    *, log_feature_compatible: bool, optimization: str
) -> str:
    """Return the endpoint metric space ``HamilBlockwiseNexTHamLoss`` would log.

    Mirrors ``HamilBlockwiseNexTHamLoss.__init__`` and
    ``compatible_loss_from_stats``: the criterion logs the exact
    old-RME-compatible endpoint (``"rme"``) -- never the native AO-block one
    (``"block"``) -- whenever ``log_feature_compatible`` is set, or whenever
    the optimization loss itself is computed in that space
    (``optimization in FEATURE_ENDPOINT_OPTIMIZATION_MODES``).  Every call
    site (both loss-module copies and the block-ODE config validator) must go
    through this one function rather than reimplementing the ternary.
    """

    return (
        "rme"
        if bool(log_feature_compatible)
        or optimization in FEATURE_ENDPOINT_OPTIMIZATION_MODES
        else "block"
    )

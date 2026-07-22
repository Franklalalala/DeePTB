# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single, dependency-free source of truth for the tied-irrep Gaussian
prior's canonical ``tied_irrep_irreps`` string.

Deliberately dependency-free (stdlib only: no torch, no e3nn, no
``dptb.data``).  Two very different callers need the exact same literal and
must never let their copies drift apart (PR#31 review finding P2-2):

* ``dptb.nnops.tied_irrep_gaussian_prior`` (needs torch + e3nn) uses it as
  the runtime default/validation target for ``tied_irrep_irreps``
  (``validate_tied_irrep_options``, called from ``flow.py``'s constructor).
* ``dptb.utils.argcheck`` (a config-time validator) uses it as the schema
  default and to reject, before training starts, any ``tied_irrep_gaussian``
  config whose ``tied_irrep_irreps`` is not this exact shape
  (``validate_block_ode_contract``).

``dptb.utils.argcheck`` must stay importable without pulling torch/e3nn in
transitively: ``dptb.data``'s package ``__init__`` imports
``dptb.utils.argcheck`` (via ``dptb.data.build``), so ``argcheck`` importing
``dptb.nnops.tied_irrep_gaussian_prior`` at module scope -- which imports
torch and e3nn unconditionally -- would load that weight into every process
that merely imports ``dptb.data``, even ones that never touch
``tied_irrep_gaussian``.  This module has neither dependency, so both sides
may import the shared literal unconditionally; e3nn/torch itself is only
imported lazily, inside the specific validator branch that needs
``o3.Irreps`` equivalence, in ``dptb.utils.argcheck``.
"""

from __future__ import annotations

TIED_IRREP_CANONICAL_IRREPS = "3x0e + 2x1e + 1x2e"

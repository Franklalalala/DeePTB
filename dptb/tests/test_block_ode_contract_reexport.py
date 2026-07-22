# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pin PR1's argcheck/block_ode_contract split: a pure move, not a copy.

``validate_flow_loss_contract`` and ``validate_block_ode_contract`` moved
verbatim out of ``dptb.utils.argcheck`` into the new sibling module
``dptb.utils.block_ode_contract``; ``argcheck.py`` re-exports both names, by
name (``from dptb.utils.block_ode_contract import ...``), at their original
line position, immediately before the ``normalize()`` call site.

Every existing test in this suite does
``from dptb.utils.argcheck import validate_block_ode_contract`` (or the
_flow_loss_ sibling) and would keep passing even if a future edit
accidentally replaced the re-export with a second, independently-defined
copy of the function -- an ``==`` check on behavior would not catch that
regression, since two structurally-identical functions still compare unequal
under the default ``object.__eq__``, and two *different* functions that
happen to behave identically on every test's inputs would slip through
unnoticed. Only an identity (``is``) check pins "same function object,
imported, not redefined."
"""
from __future__ import annotations

import dptb.utils.argcheck as argcheck
import dptb.utils.block_ode_contract as block_ode_contract


def test_validate_block_ode_contract_is_reexported_by_identity():
    assert argcheck.validate_block_ode_contract is block_ode_contract.validate_block_ode_contract


def test_validate_flow_loss_contract_is_reexported_by_identity():
    assert argcheck.validate_flow_loss_contract is block_ode_contract.validate_flow_loss_contract


def test_direct_argcheck_import_matches_module_attribute_identity():
    """The common test-suite import style (``from dptb.utils.argcheck import
    validate_block_ode_contract``) must bind the identical object the module
    attribute exposes -- i.e. the re-export is a real module-level name, not
    something only reachable via ``getattr`` trickery or a ``__getattr__``
    shim.
    """
    from dptb.utils.argcheck import (
        validate_block_ode_contract as imported_block_ode_contract,
        validate_flow_loss_contract as imported_flow_loss_contract,
    )

    assert imported_block_ode_contract is block_ode_contract.validate_block_ode_contract
    assert imported_flow_loss_contract is block_ode_contract.validate_flow_loss_contract

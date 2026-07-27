"""flow_options.prior='dftbsk' must be rejected by a message that is true.

flow_priors.DFTBSK_NAMES was emptied so flow.py could stay byte-identical, but
flow.py's own rejection message still lists 'dftbsk' as supported. The retired
aliases are now caught during flow-option canonicalization, which runs first —
including inside HamiltonianCFM.__init__ — so flow.py is untouched.
"""

from __future__ import annotations

import pytest

from dptb.configuration import (
    RETIRED_DFTBSK_PRIOR_NAMES,
    canonicalize_flow_options,
)
from dptb.nnops.flow_priors import DFTBSK_NAMES


def test_flow_priors_registry_is_empty():
    assert DFTBSK_NAMES == frozenset()


@pytest.mark.parametrize("prior", sorted(RETIRED_DFTBSK_PRIOR_NAMES))
def test_every_retired_alias_is_rejected(prior):
    with pytest.raises(ValueError, match="removed together with the SK model route"):
        canonicalize_flow_options({"enabled": True, "prior": prior})


def test_rejection_message_does_not_advertise_dftbsk_as_supported():
    with pytest.raises(ValueError) as excinfo:
        canonicalize_flow_options({"enabled": True, "prior": "dftbsk"})
    message = str(excinfo.value)
    # the only occurrence of the name is the echoed input value
    assert message.count("dftbsk") == 1
    assert "prior='external'" in message


def test_alias_matching_is_case_insensitive():
    with pytest.raises(ValueError, match="removed together with the SK model route"):
        canonicalize_flow_options({"enabled": True, "prior": "DFTB_SK"})


@pytest.mark.parametrize(
    "prior", ["zero", "gaussian", "external", "basis_onsite", "overlap_huckel"]
)
def test_supported_priors_still_canonicalize(prior):
    assert canonicalize_flow_options({"enabled": True, "prior": prior})["prior"] == prior


def test_hamiltonian_cfm_construction_surfaces_the_clear_error():
    from dptb.nnops.flow import HamiltonianCFM

    with pytest.raises(ValueError, match="removed together with the SK model route"):
        HamiltonianCFM({"enabled": True, "prior": "dftbsk"})

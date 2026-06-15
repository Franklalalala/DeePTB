"""nnops package initialization.

Importing this package installs optional RMF support into the trainer-side flow
builder without changing the default CFM/Pixel MeanFlow code paths.  The patch is
kept here so the existing ``dptb.nnops.flow`` module remains source-compatible
with the 0615 CFM branch.
"""

from __future__ import annotations


def _install_rmf_flow_builder() -> None:
    from dptb.nnops import flow as _flow
    from dptb.nnops.rmf import build_hamiltonian_flow as _rmf_build_hamiltonian_flow

    if getattr(_flow.build_hamiltonian_flow, "__name__", "") != "build_hamiltonian_flow":
        return
    _flow.build_hamiltonian_flow = _rmf_build_hamiltonian_flow


_install_rmf_flow_builder()

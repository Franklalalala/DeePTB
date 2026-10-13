"""Explicit inputs for the qualified ABACUS SOC two-center reference route.

Apply once to the original species, then use the returned species for both
offline preparation and assembly with SOC_REFERENCE_DR_BOHR. This does not
change the legacy defaults or the local-potential/FFT calculation.
"""
from dataclasses import replace
from typing import Mapping

import numpy as np

from .models import SpeciesData


# USE_NEW_TWO_CENTER tabulates cutoff=2*rmax with int(rmax/.01)+1 points.
# .02 reproduces the qualified SOC reference; inspect the resulting table's
# actual spacing when applying this route to other orbital cutoffs/builds.
SOC_REFERENCE_DR_BOHR = 0.02


def align_soc_projectors(species_data: Mapping[str, SpeciesData]) -> dict[str, SpeciesData]:
    """Copy projector samples using ABACUS setupNonlocal's cutoff convention.

    ABACUS ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f,
    source/module_cell/setup_nonlocal.cpp:111-142, assigns ``cut_mesh = ir``
    at the final index with abs(beta)>1e-10, rounds upward to an odd count,
    then copies ``ir < cut_mesh``. Thus an odd final index is excluded.
    This reproduces that reference, not the UPF reader's inclusive support
    convention. Adding one before rounding changes the reference operator.

    Keep cutoff_radius as the original conservative support envelope: it also
    controls the universal two-center grid through max_projector_cutoff.
    It intentionally need not equal r[cutoff_index - 1] after alignment.
    Clamp the all-below-threshold fallback to the available mesh rather than
    retaining the reference's out-of-range count on an even mesh. Only this
    fallback may have an even count; it leaves the stored samples unchanged.
    Orbital and non-projector arrays are shared read-only; the input projector
    arrays and dataclasses are never modified.
    """
    aligned = {}
    for symbol, data in species_data.items():
        if len(data.upf.r) == 0:
            raise ValueError(f"{symbol}: SOC projector alignment requires a nonempty radial mesh")
        projectors = []
        for projector in data.upf.projectors:
            nz = np.flatnonzero(np.abs(projector.radial_u) > 1e-10)
            cut = int(nz[-1]) if len(nz) else len(data.upf.r)
            if cut % 2 == 0:
                cut += 1
            cut = min(cut, len(data.upf.r))
            radial_u = projector.radial_u.copy()
            radial_u[cut:] = 0
            projectors.append(replace(projector, radial_u=radial_u, cutoff_index=cut))
        aligned[symbol] = replace(data, upf=replace(data.upf, projectors=projectors))
    return aligned

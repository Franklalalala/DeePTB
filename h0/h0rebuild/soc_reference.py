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

    The final index with abs(beta)>1e-10 becomes the sample count, rounded
    upward to an odd count. In particular an odd final index is excluded;
    unconditionally adding one would reproduce the original mismatch.

    Keep cutoff_radius and all physical data unchanged, as in the qualified
    SOC reconstruction. Orbital and non-projector arrays are shared read-only;
    the input projector arrays and dataclasses are never modified.
    """
    aligned = {}
    for symbol, data in species_data.items():
        projectors = []
        for projector in data.upf.projectors:
            nz = np.flatnonzero(np.abs(projector.radial_u) > 1e-10)
            cut = int(nz[-1]) if len(nz) else len(data.upf.r)
            if cut % 2 == 0:
                cut += 1
            radial_u = projector.radial_u.copy()
            radial_u[cut:] = 0
            projectors.append(replace(projector, radial_u=radial_u, cutoff_index=cut))
        aligned[symbol] = replace(data, upf=replace(data.upf, projectors=projectors))
    return aligned

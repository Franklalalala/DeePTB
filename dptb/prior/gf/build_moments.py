"""Atomic moment rows (schema deeptb.atomic_moments/v1): M2_bohr2 of each species for the GF prior.

M2 = int r^2 q(r) dr (Simpson on the UPF rab mesh) with q = PP_RHOATOM (4 pi r^2 rho) scaled to Z_valence: the second
moment of the neutral valence density. The GF prior adds the G = 0 electrostatic shift 4 pi sum_a M2_a Ry / (3 Omega)
times S. Reproduces the deployed ``atomic_R1_prepared_v2.json`` M2_bohr2 of all 68 species to 4e-16 relative; its other
fields (X_eV, diagonal_average_eV) are not used by the prior and are not built.
UPF files are found by name in the ``pporb`` directories through the reference P2 species rows (SHA256 checked).
"""
import hashlib
import json
import time
from pathlib import Path

from dptb.prior.common.pporb import simpson_rab
from dptb.prior.common.store import TableStore
from dptb.prior.gf.build_p2 import readers, source_files
from dptb.prior.gf.build_soc import _p2_rows

SCHEMA = 'deeptb.atomic_moments/v1'
ORDERED = False
PER_SPECIES = True


def second_moment(upf):
    q = upf.rhoatom_q * upf.z_valence / simpson_rab(upf.rhoatom_q, upf.rab)
    return float(simpson_rab(upf.r ** 2 * q, upf.rab))


def build(species, out_dir, *, reference, pporb, species_files=None, device=None, log=print):
    """Moment rows of ``species`` into the moments root ``out_dir`` (created or extended, rows never replaced)."""
    rows, p2_sha = _p2_rows(reference)
    store = TableStore(out_dir, SCHEMA, length_unit='bohr', source_p2_manifest_sha256=p2_sha)
    gid = store.generator('dptb.prior.gf.build_moments')
    seconds = {}
    for s in species:
        if store.has('species', s):
            continue
        t0 = time.perf_counter()
        orb_path, upf_path = source_files([s], rows, pporb, species_files)[s]
        m2 = second_moment(readers()[1](upf_path))
        store.add('species', s, dict(M2_bohr2=m2, generator=gid,
                                     upf_sha256=hashlib.sha256(Path(upf_path).read_bytes()).hexdigest(),
                                     orbital_sha256=hashlib.sha256(Path(orb_path).read_bytes()).hexdigest()))
        seconds[s] = time.perf_counter() - t0
        log(f'MOMENTS_BUILT {s} M2={m2:.12g}')
    store.flush()
    return seconds


def species_of(reference):
    rows, _ = _p2_rows(reference)
    return sorted(rows)


def present(out_dir):
    path = Path(out_dir) / 'manifest.json'
    return set(json.loads(path.read_text()).get('species', {})) if path.exists() else set()

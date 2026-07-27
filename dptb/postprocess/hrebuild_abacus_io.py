"""ABACUS I/O layer for the ``restart_dh`` rebuild endpoint.

Everything about *talking to ABACUS* lives here: unit conventions, the
CSR read/write pair (DFTIO 0-indexed block dicts <-> ABACUS ``hrs1_nao.csr``
files, incl. the SOC spinor interleave), the ``override_full_h`` HDF5
packing, STRU/INPUT/KPT generation, and the subprocess executor. The
repair orchestration (guards, ``one_shot_repair``, ``repair_atomic_data``)
lives in ``dptb.postprocess.hrebuild``, which re-exports this module's
public names -- import from there unless you specifically want the raw
I/O primitives.

Unit discipline (verified empirically, see the WS4 report): ABACUS CSR
files are in Rydberg; ``write_blocks_to_abacus_csr`` assumes eV input and
divides by ``H_FACTOR`` internally. Callers declare their block unit via
``unit=`` on :func:`write_hr_csr` / :func:`read_hr_csr`.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag

from dptb.postprocess.write_abacus_csr_file import (
    abacus2dftio_matrices,
    write_blocks_to_abacus_csr,
    parse_basis_to_l_list,
    find_basis_for_Z_or_symbol,
    H_FACTOR,
)

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

# eV per unit, i.e. value_in_eV = value_in_unit * EV_PER_UNIT[unit].
# write_blocks_to_abacus_csr always assumes an eV input and divides by
# H_FACTOR (Ry->eV) internally, so pre-scaling to eV here is sufficient
# and keeps that shared writer untouched.
EV_PER_UNIT = {
    "eV": 1.0,
    "Ha": 27.211386245988,
    "Hartree": 27.211386245988,
    "Ry": H_FACTOR,
    "Rydberg": H_FACTOR,
}

KEY_RE = re.compile(r'^\s*(-?\d+)[ _](-?\d+)[ _](-?\d+)[ _](-?\d+)[ _](-?\d+)\s*$')


def _scale_blocks(blocks_dict: Dict[str, np.ndarray], factor: float) -> Dict[str, np.ndarray]:
    if factor == 1.0:
        return blocks_dict
    out = {}
    for k, v in blocks_dict.items():
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        out[k] = arr * factor
    return out


# ---------------------------------------------------------------------------
# CSR write (blocks -> hrs1_nao.csr)
# ---------------------------------------------------------------------------

def write_hr_csr(
    atomic_numbers,
    basis_dict: Dict,
    blocks_dict: Dict[str, np.ndarray],
    output_path: Union[str, Path],
    unit: str = "eV",
    step: int = 0,
):
    """Write a 0-indexed ``{i}_{j}_{Rx}_{Ry}_{Rz}`` DFTIO-ordered block dict
    (the output of ``dptb.data.interfaces.feature_to_block``) to an ABACUS
    ``hrs1_nao.csr`` file, in Ry, ABACUS AO ordering.

    ``unit`` declares the unit of the *input* ``blocks_dict`` values
    ("eV" for ABACUS-native/dftio-parsed models -- the default and the only
    verified route for this endpoint; "Ha"/"Ry" accepted for completeness
    but not exercised by the WS4-A acceptance run).
    """
    if unit not in EV_PER_UNIT:
        raise ValueError(f"Unknown unit {unit!r}; expected one of {sorted(EV_PER_UNIT)}")
    blocks_eV = _scale_blocks(blocks_dict, EV_PER_UNIT[unit])
    reassembled, norbits = write_blocks_to_abacus_csr(
        atomic_numbers=atomic_numbers,
        basis_dict=basis_dict,
        blocks_dict=blocks_eV,
        matrix_symbol="H",
        output_path=str(output_path),
        step=step,
        unfold_symmetry=True,
    )
    return reassembled, norbits


# ---------------------------------------------------------------------------
# CSR read (ABACUS output csr -> blocks, DFTIO ordering, 0-indexed)
# ---------------------------------------------------------------------------

def _abacus_to_dftio(mat: np.ndarray, l_lefts, l_rights, abacus2dftio) -> np.ndarray:
    """Inverse of ``write_abacus_csr_file.transform_2_ABACUS``.

    Reproduces ``dftio.io.abacus.abacus_parser.AbacusParser.transform`` verbatim
    (verified against the installed dftio package on natlan, 2026-07-03):
    ``block_diag(ABACUS2DFTIO[l_left]) @ mat @ block_diag(ABACUS2DFTIO[l_right]).T``.
    """
    left_mats = [abacus2dftio[l] for l in l_lefts]
    right_mats = [abacus2dftio[l] for l in l_rights]
    block_lefts = block_diag(*left_mats) if left_mats else np.eye(0)
    block_rights = block_diag(*right_mats) if right_mats else np.eye(0)
    return block_lefts @ mat @ block_rights.T


def read_hr_csr(
    csr_path: Union[str, Path],
    atomic_numbers,
    basis_dict: Dict,
    unit: str = "eV",
    is_soc: bool = False,
    zero_thr: float = 1e-10,
):
    """Read an ABACUS real-space matrix CSR file (``hrs1_nao.csr`` /
    ``data-HR-sparse_SPIN0.csr`` / ``data-HR0_SPIN0.csr`` / the ``S`` twin)
    back into a 0-indexed ``{i}_{j}_{Rx}_{Ry}_{Rz}`` DFTIO-ordered block
    dict, converted from Ry into ``unit``.

    Standalone reimplementation of
    ``dftio.io.abacus.abacus_parser.AbacusParser.parse_matrix`` that avoids
    depending on the heavyweight ``Parser``/``dpdata`` machinery, which
    expects a whole raw-data directory tree rather than a single CSR file.
    """
    abacus2dftio = abacus2dftio_matrices()

    if unit not in EV_PER_UNIT:
        raise ValueError(f"Unknown unit {unit!r}; expected one of {sorted(EV_PER_UNIT)}")
    # value_in_unit = value_in_Ry * H_FACTOR / EV_PER_UNIT[unit]  (Ry -> eV -> unit)
    ry_to_unit = H_FACTOR / EV_PER_UNIT[unit]

    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    element_l_lists = {}
    for z in np.unique(atomic_numbers):
        basis_str = find_basis_for_Z_or_symbol(basis_dict, int(z))
        ll = parse_basis_to_l_list(basis_str) if basis_str else [0]
        element_l_lists[int(z)] = ll if ll else [0]

    site_norbits_spatial = np.array(
        [sum(2 * l + 1 for l in element_l_lists[int(z)]) for z in atomic_numbers], dtype=int
    )
    site_norbits_physical = site_norbits_spatial * 2 if is_soc else site_norbits_spatial
    site_cumsum = np.cumsum(site_norbits_physical)
    nsites = len(atomic_numbers)

    with open(csr_path, "r") as f:
        lines = f.readlines()

    dim_line_idx = None
    for i, line in enumerate(lines):
        if "Matrix Dimension of" in line:
            dim_line_idx = i
            break
    if dim_line_idx is None:
        raise ValueError(f"Cannot find 'Matrix Dimension of' in {csr_path}")
    norbits = int(lines[dim_line_idx].split()[-1])
    if norbits != int(site_cumsum[-1]):
        raise ValueError(
            f"CSR norbits={norbits} does not match basis-derived norbits={int(site_cumsum[-1])}; "
            "basis_dict/atomic_numbers/is_soc likely mismatched with the CSR file."
        )

    blocks: Dict[str, np.ndarray] = {}
    i = dim_line_idx + 2
    while i < len(lines):
        header = lines[i].split()
        if len(header) < 4:
            i += 1
            continue
        Rx, Ry_, Rz, nnz = int(header[0]), int(header[1]), int(header[2]), int(header[3])
        if nnz == 0:
            i += 1
            continue
        data_line, col_line, ptr_line = lines[i + 1], lines[i + 2], lines[i + 3]
        if is_soc:
            raw = data_line.split()
            cleaned = [tok.strip("()").replace(",", "+") + "j" for tok in raw]
            # token like "(re,im)" -> "re+imj" (im already carries its own sign)
            cleaned = [re.sub(r"\+(-)", r"\1", tok) for tok in cleaned]
            data_vals = np.array(cleaned).astype(np.complex128)
        else:
            data_vals = np.array(data_line.split()).astype(np.float64)
        col_idx = np.array(col_line.split()).astype(int)
        indptr = np.array(ptr_line.split()).astype(int)
        mat_full = csr_matrix((data_vals, col_idx, indptr), shape=(norbits, norbits)).toarray()

        for si in range(nsites):
            for sj in range(nsites):
                i0 = int(site_cumsum[si] - site_norbits_physical[si])
                i1 = int(site_cumsum[si])
                j0 = int(site_cumsum[sj] - site_norbits_physical[sj])
                j1 = int(site_cumsum[sj])
                sub = mat_full[i0:i1, j0:j1]
                if np.abs(sub).max() < zero_thr:
                    continue
                l_lefts = element_l_lists[int(atomic_numbers[si])]
                l_rights = element_l_lists[int(atomic_numbers[sj])]
                if is_soc:
                    ni, nj = site_norbits_spatial[si], site_norbits_spatial[sj]
                    sub = sub.reshape(ni, 2, nj, 2).transpose(1, 0, 3, 2).reshape(2 * ni, 2 * nj)
                    dftio_block = _abacus_to_dftio(sub, l_lefts * 2, l_rights * 2, abacus2dftio)
                else:
                    dftio_block = _abacus_to_dftio(sub, l_lefts, l_rights, abacus2dftio)
                blocks[f"{si}_{sj}_{Rx}_{Ry_}_{Rz}"] = dftio_block * ry_to_unit
        i += 4

    return blocks, norbits


# ---------------------------------------------------------------------------
# HDF5 packing (matches dptb.postprocess.write_block.write_block's schema
# exactly, so it round-trips through Band.get_bands(kpath_kwargs={"override_full_h": ...}))
# ---------------------------------------------------------------------------

def blocks_to_override_h5(blocks_dict: Dict[str, np.ndarray], output_path: Union[str, Path]) -> None:
    import h5py

    with h5py.File(str(output_path), "w") as fid:
        grp = fid.create_group("0")
        for key, value in blocks_dict.items():
            arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            grp[key] = arr


# ---------------------------------------------------------------------------
# STRU / INPUT / KPT generation
# ---------------------------------------------------------------------------

@dataclass
class PPOrbSpec:
    """Per-element pseudopotential + numerical-orbital file names (relative
    to ``pp_orb_dir``), and the AO basis string driving ``basis_dict`` in
    :func:`write_hr_csr` / :func:`read_hr_csr`."""

    pseudo: str
    orbital: str
    basis: str
    mass: float = 1.0


@dataclass
class StructureSpec:
    symbols: Sequence[str]
    positions_angstrom: np.ndarray  # (natom, 3)
    cell_angstrom: np.ndarray       # (3, 3)
    pbc: Tuple[bool, bool, bool] = (True, True, True)


_ANG_TO_BOHR = 1.0 / 0.529177210903


def write_stru(structure: StructureSpec, pp_orb: Dict[str, PPOrbSpec], output_path: Union[str, Path]) -> None:
    species = list(dict.fromkeys(structure.symbols))
    lines = ["ATOMIC_SPECIES"]
    for sp in species:
        lines.append(f"{sp}  {pp_orb[sp].mass:.4f}  {pp_orb[sp].pseudo}")
    lines += ["", "NUMERICAL_ORBITAL"]
    for sp in species:
        lines.append(pp_orb[sp].orbital)
    lines += ["", "LATTICE_CONSTANT", f"{_ANG_TO_BOHR:.13f}", "", "LATTICE_VECTORS"]
    for row in structure.cell_angstrom:
        lines.append("  ".join(f"{x:.14f}" for x in row))
    lines += ["", "ATOMIC_POSITIONS", "Cartesian_angstrom"]
    for sp in species:
        idx = [i for i, s in enumerate(structure.symbols) if s == sp]
        lines += ["", sp, "0.0", str(len(idx))]
        for i in idx:
            x, y, z = structure.positions_angstrom[i]
            lines.append(f"{x:.10f} {y:.10f} {z:.10f} 1 1 1")
    Path(output_path).write_text("\n".join(lines) + "\n")


def write_kpt(output_path: Union[str, Path], kmesh: Tuple[int, int, int] = (1, 1, 1)) -> None:
    Path(output_path).write_text(
        "K_POINTS\n0\nGamma\n{} {} {} 0 0 0\n".format(*kmesh)
    )


_DEFAULT_INPUT_KEYS = dict(
    calculation="scf",
    basis_type="lcao",
    ks_solver="genelpa",
    smearing_method="gaussian",
    smearing_sigma=1e-3,
    mixing_type="pulay",
    mixing_beta=0.4,
    mixing_gg0=1.5,
    mixing_ndim=30,
    scf_thr=1e-8,
    symmetry=0,
    out_chg=0,
    out_mat_hs2=1,
)


def write_input(
    output_path: Union[str, Path],
    mode: str,
    ecutwfc: float = 100.0,
    nspin: int = 1,
    lspinorb: int = 0,
    nbands: Optional[int] = None,
    pp_orb_dir: str = "PP_ORB",
    extra: Optional[Dict] = None,
) -> None:
    """``mode='one_shot'``: ``scf_nmax 1`` (single-iteration repair, ``R(H)``).
    ``mode='full_scf'``: normal ``scf_nmax`` convergence starting from the
    supplied H (SCF-acceleration / full-repair use case)."""
    if mode not in ("one_shot", "full_scf"):
        raise ValueError(f"mode must be 'one_shot' or 'full_scf', got {mode!r}")
    keys = dict(_DEFAULT_INPUT_KEYS)
    keys["pseudo_dir"] = pp_orb_dir
    keys["orbital_dir"] = pp_orb_dir
    keys["ecutwfc"] = ecutwfc
    keys["nspin"] = nspin
    keys["lspinorb"] = lspinorb
    keys["scf_nmax"] = 1 if mode == "one_shot" else 100
    keys["init_chg"] = "hr"
    keys["read_file_dir"] = "./"
    if nbands is not None:
        keys["nbands"] = nbands
    if extra:
        keys.update(extra)
    lines = ["INPUT_PARAMETERS"]
    for k, v in keys.items():
        lines.append(f"{k}\t{v}")
    Path(output_path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# ABACUS process execution (local subprocess, or a pluggable remote
# executor e.g. SSH exec_command for natlan)
# ---------------------------------------------------------------------------

Executor = Callable[[str, str], Tuple[int, str, str]]  # (workdir, command) -> (returncode, stdout, stderr)


def local_executor(workdir: str, command: str) -> Tuple[int, str, str]:
    proc = subprocess.run(command, shell=True, cwd=workdir, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def run_abacus(
    workdir: Union[str, Path],
    abacus_bin: str,
    mpi_procs: int = 1,
    omp_threads: int = 1,
    executor: Executor = local_executor,
    timeout: Optional[float] = None,
) -> Tuple[int, str, str]:
    workdir = str(workdir)
    cmd = f"export OMP_NUM_THREADS={omp_threads}; mpirun -np {mpi_procs} {abacus_bin} > run.log 2>&1"
    return executor(workdir, cmd)


"""One-shot Hamiltonian rebuild endpoint on top of ABACUS ``restart_dh``.

WS4 (see ``F:\\claude\\0702_nextham_dm_plan\\02_llm_execution_plan.md``, section
"WS4 hrebuild"). Given a predicted real-space Hamiltonian block dict
(the output of ``dptb.data.interfaces.feature_to_block``), this module:

1. writes it out as an ABACUS ``hrs1_nao.csr`` file (:func:`write_hr_csr`),
2. drives an ABACUS binary built from the ``dyzheng/abacus-develop@restart_dh``
   branch with ``init_chg hr`` so it re-derives a Hamiltonian from the
   aufbau density of the supplied H (one SCF step for ``mode="one_shot"``,
   or full convergence for ``mode="full_scf"``) (:func:`run_abacus`),
3. reads the resulting ``data-HR-sparse_SPIN0.csr`` back into the same
   block-dict convention (:func:`read_hr_csr`),
4. packs it into the ``group "0"`` HDF5 layout that
   ``dptb.postprocess.elec_struc_cal.ElecStruCal._open_h5_first_block`` /
   ``Band.get_bands(..., kpath_kwargs={"override_full_h": ...})`` already
   consume, so the repaired Hamiltonian can be fed straight back into the
   existing band-structure / eigenvalue machinery without any new k-space
   code (:func:`blocks_to_override_h5`).

Unit discipline (verified empirically, not assumed -- see the WS4 report):
ABACUS CSR files are in Rydberg. ``write_blocks_to_abacus_csr`` in
``write_abacus_csr_file.py`` assumes its *input* is already in eV (it
divides by ``H_FACTOR = 13.605698`` to reach Ry). That assumption holds for
ABACUS-native NAO-basis models (dftio-parsed periodic crystal data, e.g. the
D3 / h0 production route). It does **not** hold for the QHFlow2/QH9-derived
molecular route (water N1 CFM model), whose Hamiltonians are in Hartree and
whose AO basis is a Gaussian basis with no corresponding ABACUS NAO/UPF
pair -- that route is out of scope for this endpoint (see WS4 report, "GTO
vs NAO" note in plan section 11). Callers must declare the unit their
blocks are in via ``unit=`` on :func:`write_hr_csr` / :func:`read_hr_csr`.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import eigh

# ABACUS I/O layer (units, CSR read/write, STRU/INPUT/KPT, executor).
# Re-exported here so existing `from dptb.postprocess import hrebuild as hb`
# call sites keep working unchanged.
from dptb.postprocess.hrebuild_abacus_io import (  # noqa: F401
    EV_PER_UNIT,
    H_FACTOR,
    KEY_RE,
    Executor,
    PPOrbSpec,
    StructureSpec,
    _ANG_TO_BOHR,
    _DEFAULT_INPUT_KEYS,
    _abacus_to_dftio,
    _scale_blocks,
    blocks_to_override_h5,
    find_basis_for_Z_or_symbol,
    local_executor,
    parse_basis_to_l_list,
    read_hr_csr,
    run_abacus,
    write_blocks_to_abacus_csr,
    write_hr_csr,
    write_input,
    write_kpt,
    write_stru,
)

# ---------------------------------------------------------------------------
# Gap-threshold guard (red-line #3 / risk table: refuse repair on
# metals / near-degenerate gaps, matching the toy result in plan section 2.5)
# ---------------------------------------------------------------------------

def estimate_gap_ev(h: np.ndarray, s: np.ndarray, n_occ: int) -> float:
    evals = eigh(h, s, eigvals_only=True)
    return float(evals[n_occ] - evals[n_occ - 1])


def gap_allows_repair(h: np.ndarray, s: np.ndarray, n_occ: int, gap_threshold_ev: float = 0.5) -> Tuple[bool, float]:
    gap = estimate_gap_ev(h, s, n_occ)
    return gap >= gap_threshold_ev, gap



# ---------------------------------------------------------------------------
# Self-consistency guard (plan §3.4-1, calibrated on the 2026-07-03
# production repair-line test, see 05 report §3):
#   ||R(H)-H|| block-residual mean over common keys, in eV --
#     case_0154 (healthy, repair floor-limited)  : 2.7e-4
#     case_0193 (sick but inside basin, helped)  : 1.9e-3
#     case_0008 (outside contraction basin, blew): 7.0e-2
#   |E_Harris - E_KohnSham| (eV, advisory only -- 4x separation vs the
#   residual's 40x): 0.22 / 80.5 / 337.9 for the same three cases.
# Two-sided verdict: below `min_gain_residual_ev` the prediction is already
# self-consistent within the one-shot repair floor (~39 meV band-level on
# case_0154) and repairing can only add map-difference noise; above
# `max_residual_mean_ev` the input is outside the contraction basin and the
# one-shot image is untrustworthy (and the residual itself is a label-free
# bad-prediction probe).
# ---------------------------------------------------------------------------

@dataclass
class SCGuardConfig:
    min_gain_residual_ev: float = 5.0e-4
    max_residual_mean_ev: float = 2.0e-2


def self_consistency_residual(
    blocks_in: Dict[str, np.ndarray],
    blocks_out: Dict[str, np.ndarray],
    unit: str = "eV",
) -> Dict[str, float]:
    """Block-wise ``|R(H) - H|`` statistics over common keys, reported in eV.

    ``unit`` declares the unit both block dicts are in (they must match)."""
    factor = EV_PER_UNIT[unit]
    common = sorted(set(blocks_in) & set(blocks_out))
    if not common:
        return {"n_common": 0, "residual_mean_ev": float("nan"), "residual_max_ev": float("nan")}
    means = []
    mx = 0.0
    for k in common:
        d = np.abs(np.asarray(blocks_in[k]) - np.asarray(blocks_out[k])) * factor
        means.append(float(d.mean()))
        mx = max(mx, float(d.max()))
    return {"n_common": len(common), "residual_mean_ev": float(np.mean(means)), "residual_max_ev": mx}


_ENERGY_ROW_RE = re.compile(r"^\s*(E_KohnSham|E_Harris)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.M)


def parse_scf_energies_ev(running_log_text: str) -> Dict[str, Optional[float]]:
    """Last ``E_KohnSham``/``E_Harris`` rows of an ABACUS running log, in eV.

    ABACUS prints these rows with two columns (Ry, eV) -- the eV column is
    taken. NOTE: the two columns of a single row are the SAME energy in two
    units, not a Harris-vs-KS pair; the Harris/KS gap must be taken across
    the two separate rows."""
    found: Dict[str, float] = {}
    for m in _ENERGY_ROW_RE.finditer(running_log_text):
        found[m.group(1)] = float(m.group(3))
    e_ks = found.get("E_KohnSham")
    e_ha = found.get("E_Harris")
    gap = abs(e_ha - e_ks) if (e_ks is not None and e_ha is not None) else None
    return {"e_kohnsham_ev": e_ks, "e_harris_ev": e_ha, "harris_ks_gap_ev": gap}


def evaluate_sc_guard(
    residual_mean_ev: float,
    config: SCGuardConfig,
) -> Tuple[bool, Optional[str]]:
    """Return ``(repair_trustworthy, reason)`` for a computed residual."""
    if not np.isfinite(residual_mean_ev):
        return False, "sc guard: residual undefined (no common blocks)"
    if residual_mean_ev < config.min_gain_residual_ev:
        return False, (
            f"sc guard: residual mean {residual_mean_ev:.3e} eV < gain floor "
            f"{config.min_gain_residual_ev:.3e} eV -- prediction already "
            "self-consistent within the repair floor; keep the original."
        )
    if residual_mean_ev > config.max_residual_mean_ev:
        return False, (
            f"sc guard: residual mean {residual_mean_ev:.3e} eV > basin ceiling "
            f"{config.max_residual_mean_ev:.3e} eV -- input is outside the "
            "self-consistency contraction basin; one-shot image untrusted "
            "(and the prediction itself should be treated as suspect)."
        )
    return True, None


def _coerce_sc_guard(sc_guard) -> Optional[SCGuardConfig]:
    if sc_guard is None or sc_guard is False:
        return None
    if sc_guard is True:
        return SCGuardConfig()
    if isinstance(sc_guard, SCGuardConfig):
        return sc_guard
    if isinstance(sc_guard, dict):
        return SCGuardConfig(**sc_guard)
    raise TypeError(f"sc_guard must be bool/dict/SCGuardConfig/None, got {type(sc_guard)!r}")


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    ok: bool
    mode: str
    repaired_blocks: Optional[Dict[str, np.ndarray]] = None
    repaired_h5: Optional[str] = None
    workdir: Optional[str] = None
    gap_ev: Optional[float] = None
    skipped_reason: Optional[str] = None
    abacus_returncode: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    # Self-consistency guard metrics (always attached when the rebuild ran;
    # `ok`/`repaired_blocks` are NOT affected by the guard -- callers such as
    # Band._apply_hrebuild_repair decide whether to *use* the repaired H).
    sc_residual_mean_ev: Optional[float] = None
    sc_residual_max_ev: Optional[float] = None
    sc_n_common: Optional[int] = None
    e_kohnsham_ev: Optional[float] = None
    e_harris_ev: Optional[float] = None
    harris_ks_gap_ev: Optional[float] = None
    repair_trustworthy: bool = True
    guard_reason: Optional[str] = None


def one_shot_repair(
    *,
    blocks_dict: Dict[str, np.ndarray],
    structure: StructureSpec,
    pp_orb: Dict[str, PPOrbSpec],
    workdir: Union[str, Path],
    abacus_bin: str,
    unit: str = "eV",
    mode: str = "one_shot",
    is_soc: bool = False,
    nspin: Optional[int] = None,
    lspinorb: int = 0,
    kmesh: Tuple[int, int, int] = (1, 1, 1),
    ecutwfc: float = 100.0,
    pp_orb_dir: str = "PP_ORB",
    n_occ: Optional[int] = None,
    overlap_blocks_dict: Optional[Dict[str, np.ndarray]] = None,
    overlap_unit: str = "eV",
    gap_threshold_ev: float = 0.5,
    executor: Executor = local_executor,
    mpi_procs: int = 1,
    keep_workdir: bool = True,
    input_extra: Optional[Dict] = None,
    sc_guard=True,
) -> RepairResult:
    """One-shot / full-SCF Hamiltonian repair via ABACUS ``restart_dh``.

    ``n_occ`` (number of doubly-occupied MOs) enables the gap-threshold
    guard (red line #3: refuse repair on near-metallic systems, see plan
    section 2.5/6). If ``n_occ`` is ``None`` the guard is skipped -- callers
    working with genuinely periodic/metallic systems should pass an
    explicit occupation count or handle the guard themselves upstream.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    atomic_numbers = _symbols_to_z(structure.symbols)
    basis_dict = {sp: pp_orb[sp].basis for sp in dict.fromkeys(structure.symbols)}
    nspin = nspin if nspin is not None else (4 if is_soc else 1)

    gap_ev = None
    if n_occ is not None and overlap_blocks_dict is not None:
        h_dense = _blocks_to_dense_onsite_gamma(blocks_dict, atomic_numbers, basis_dict, is_soc)
        s_dense = _blocks_to_dense_onsite_gamma(overlap_blocks_dict, atomic_numbers, basis_dict, is_soc)
        allowed, gap_ev = gap_allows_repair(h_dense, s_dense, n_occ, gap_threshold_ev)
        if not allowed:
            return RepairResult(
                ok=False, mode=mode, gap_ev=gap_ev,
                skipped_reason=f"gap {gap_ev:.4f} eV < threshold {gap_threshold_ev} eV; repair refused (red line #3)",
            )

    write_hr_csr(atomic_numbers, basis_dict, blocks_dict, workdir / "hrs1_nao.csr", unit=unit)
    write_stru(structure, pp_orb, workdir / "STRU")
    write_kpt(workdir / "KPT", kmesh=kmesh)
    write_input(workdir / "INPUT", mode=mode, ecutwfc=ecutwfc, nspin=nspin, lspinorb=lspinorb, pp_orb_dir=pp_orb_dir, extra=input_extra)

    # ABACUS names the out_mat_hs2 real-space H(R) CSR "hrs1_nao.csr" (no
    # "data-"/"_SPIN0" decoration) inside OUT.ABACUS -- verified empirically
    # against both the group's own restart_dh smoke tests and this client's
    # own runs (2026-07-03); *not* "data-HR-sparse_SPIN0.csr" (that name is
    # used by a different output path, e.g. out_mat_hr0).
    out_csr = workdir / "OUT.ABACUS" / "hrs1_nao.csr"

    # Fail closed against stale workdir reuse: a previous run may have left an
    # OUT.ABACUS tree behind, and a failed relaunch must not pass off the old
    # CSR as a fresh repair. Unlink errors are real errors -- do not swallow.
    out_csr.unlink(missing_ok=True)

    rc, stdout, stderr = run_abacus(workdir, abacus_bin, mpi_procs=mpi_procs, executor=executor)

    # run_abacus redirects the subprocess's stdout+stderr into workdir/run.log
    # ("> run.log 2>&1"), so the executor's own streams are normally empty --
    # the log tail is the only actionable diagnostic for a failed run.
    stdout_tail = stdout[-2000:] if stdout else ""
    if not stdout_tail:
        run_log = workdir / "run.log"
        if run_log.exists():
            try:
                stdout_tail = run_log.read_text(errors="ignore")[-2000:]
            except OSError:
                pass
    stderr_tail = stderr[-2000:] if stderr else ""

    if rc != 0:
        return RepairResult(
            ok=False, mode=mode, gap_ev=gap_ev, abacus_returncode=rc,
            stdout_tail=stdout_tail, stderr_tail=stderr_tail, workdir=str(workdir),
            skipped_reason=f"ABACUS exited with status {rc}; repair refused",
        )

    if not out_csr.exists():
        return RepairResult(
            ok=False, mode=mode, gap_ev=gap_ev, abacus_returncode=rc,
            stdout_tail=stdout_tail, stderr_tail=stderr_tail, workdir=str(workdir),
            skipped_reason=f"expected output CSR not found: {out_csr}",
        )

    repaired_blocks, _ = read_hr_csr(out_csr, atomic_numbers, basis_dict, unit=unit, is_soc=is_soc)
    h5_path = workdir / "repaired_H.h5"
    blocks_to_override_h5(repaired_blocks, h5_path)

    # Self-consistency guard metrics + verdict (never mutates ok/blocks).
    guard_cfg = _coerce_sc_guard(sc_guard)
    residual = self_consistency_residual(blocks_dict, repaired_blocks, unit=unit)
    energies: Dict[str, Optional[float]] = {"e_kohnsham_ev": None, "e_harris_ev": None, "harris_ks_gap_ev": None}
    log_path = workdir / "OUT.ABACUS" / "running_scf.log"
    if log_path.exists():
        try:
            energies = parse_scf_energies_ev(log_path.read_text(errors="ignore"))
        except Exception:  # noqa: BLE001 -- diagnostics must not break the repair
            pass
    trustworthy, reason = True, None
    if guard_cfg is not None:
        trustworthy, reason = evaluate_sc_guard(residual["residual_mean_ev"], guard_cfg)

    if not keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)

    return RepairResult(
        ok=True, mode=mode, repaired_blocks=repaired_blocks, repaired_h5=str(h5_path),
        workdir=str(workdir), gap_ev=gap_ev, abacus_returncode=rc,
        stdout_tail=stdout_tail, stderr_tail=stderr_tail,
        sc_residual_mean_ev=residual["residual_mean_ev"],
        sc_residual_max_ev=residual["residual_max_ev"],
        sc_n_common=residual["n_common"],
        e_kohnsham_ev=energies["e_kohnsham_ev"],
        e_harris_ev=energies["e_harris_ev"],
        harris_ks_gap_ev=energies["harris_ks_gap_ev"],
        repair_trustworthy=trustworthy,
        guard_reason=reason,
    )


def _symbols_to_z(symbols: Sequence[str]) -> np.ndarray:
    import ase.data
    return np.array([ase.data.atomic_numbers[s] for s in symbols], dtype=int)


def repair_atomic_data(
    *,
    data,
    idp,
    pp_orb: Dict[str, PPOrbSpec],
    workdir: Union[str, Path],
    abacus_bin: str,
    unit: str = "eV",
    mode: str = "one_shot",
    is_soc: bool = False,
    nspin: Optional[int] = None,
    lspinorb: int = 0,
    kmesh: Tuple[int, int, int] = (1, 1, 1),
    ecutwfc: float = 100.0,
    pp_orb_dir: str = "PP_ORB",
    n_occ: Optional[int] = None,
    gap_threshold_ev: float = 0.5,
    executor: Executor = local_executor,
    mpi_procs: int = 1,
    input_extra: Optional[Dict] = None,
    sc_guard=True,
) -> RepairResult:
    """:func:`one_shot_repair`, but taking a DeePTB ``AtomicDataDict`` (post
    model-forward, i.e. ``NODE_FEATURES_KEY``/``EDGE_FEATURES_KEY`` already
    hold the *predicted* Hamiltonian) plus its ``idp`` instead of a raw
    block dict + :class:`StructureSpec`. This is the entry point
    ``Band.get_bands(..., kpath_kwargs={"repair": {...}})`` uses (see
    ``dptb/postprocess/bandstructure/band.py``).

    Extracts ``atomic_numbers``/Cartesian positions (Angstrom)/cell from
    ``data`` via ase conversion, and predicted blocks via
    ``dptb.data.interfaces.feature_to_block`` (0-indexed, DFTIO ordering --
    matches this module's read/write convention exactly, see WS4 report).
    """
    from dptb.data import AtomicData
    from dptb.data.interfaces import feature_to_block

    blocks_dict = feature_to_block(data, idp, overlap=False)

    ase_atoms = AtomicData.from_AtomicDataDict(data).to("cpu").to_ase()
    if isinstance(ase_atoms, list):
        if len(ase_atoms) != 1:
            raise ValueError(
                f"repair_atomic_data expects a single structure, got a batch of {len(ase_atoms)}."
            )
        ase_atoms = ase_atoms[0]
    symbols = list(ase_atoms.get_chemical_symbols())
    structure = StructureSpec(
        symbols=symbols,
        positions_angstrom=np.asarray(ase_atoms.get_positions()),
        cell_angstrom=np.asarray(ase_atoms.get_cell()),
        pbc=tuple(bool(x) for x in ase_atoms.get_pbc()),
    )

    overlap_blocks_dict = None
    if n_occ is not None:
        try:
            overlap_blocks_dict = feature_to_block(data, idp, overlap=True)
        except Exception:
            overlap_blocks_dict = None

    return one_shot_repair(
        blocks_dict=blocks_dict,
        structure=structure,
        pp_orb=pp_orb,
        workdir=workdir,
        abacus_bin=abacus_bin,
        unit=unit,
        mode=mode,
        is_soc=is_soc,
        nspin=nspin,
        lspinorb=lspinorb,
        kmesh=kmesh,
        ecutwfc=ecutwfc,
        pp_orb_dir=pp_orb_dir,
        n_occ=n_occ,
        overlap_blocks_dict=overlap_blocks_dict,
        gap_threshold_ev=gap_threshold_ev,
        executor=executor,
        mpi_procs=mpi_procs,
        input_extra=input_extra,
        sc_guard=sc_guard,
    )


def _blocks_to_dense_onsite_gamma(blocks_dict, atomic_numbers, basis_dict, is_soc) -> np.ndarray:
    """Assemble the Gamma-point (R=0 only) dense H or S from a DFTIO-ordered
    0-indexed block dict, for the gap-threshold guard. Molecule-only helper
    (periodic systems should pass a pre-assembled dense matrix upstream)."""
    element_l_lists = {}
    for z in np.unique(atomic_numbers):
        basis_str = find_basis_for_Z_or_symbol(basis_dict, int(z))
        ll = parse_basis_to_l_list(basis_str) if basis_str else [0]
        element_l_lists[int(z)] = ll if ll else [0]
    spatial = np.array([sum(2 * l + 1 for l in element_l_lists[int(z)]) for z in atomic_numbers], dtype=int)
    dim = spatial * (2 if is_soc else 1)
    cumsum = np.cumsum(dim)
    norb = int(cumsum[-1])
    dense = np.zeros((norb, norb), dtype=complex if is_soc else float)
    for key, block in blocks_dict.items():
        m = KEY_RE.match(key)
        if not m:
            continue
        i, j, rx, ry, rz = (int(g) for g in m.groups())
        if (rx, ry, rz) != (0, 0, 0):
            continue
        i0, i1 = int(cumsum[i] - dim[i]), int(cumsum[i])
        j0, j1 = int(cumsum[j] - dim[j]), int(cumsum[j])
        arr = block.detach().cpu().numpy() if hasattr(block, "detach") else np.asarray(block)
        dense[i0:i1, j0:j1] = arr
    return dense

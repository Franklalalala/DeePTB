"""Offline unit tests for dptb.postprocess.hrebuild (WS4).

These do not invoke ABACUS -- the end-to-end fixed-point validation against
the real restart_dh binary (periodic non-SOC + SOC) lives in the WS4 report,
run on natlan where the binary is built. This file only locks down the
parts that are pure Python and must stay correct under refactors: the
CSR write/read round trip (both non-SOC and SOC), unit conversion, and the
gap-threshold guard.
"""
from pathlib import Path

import numpy as np
import pytest

# The CSR write/read pair needs the ABACUS<->DFTIO orbital-ordering matrices,
# and dftio is an optional source-only install (see DFTIO_MISSING_MESSAGE in
# dptb/postprocess/write_abacus_csr_file.py). Nine of the ten tests below go
# through that transform.
pytest.importorskip("dftio", reason="dftio supplies the ABACUS<->DFTIO transforms")

from dptb.postprocess import hrebuild as hb


def _toy_two_atom_blocks(is_soc: bool, seed: int = 0):
    """A tiny made-up 2-atom (s,p / s) system: enough orbitals to exercise
    the l=0,1 DFTIO<->ABACUS transform and Hermitian completion, small
    enough to be a fast, deterministic offline test."""
    rng = np.random.default_rng(seed)
    atomic_numbers = np.array([8, 1])  # O, H
    basis_dict = {"O": "1s1p", "H": "1s"}
    dim = {0: 4, 1: 1}  # O: 1s+1p = 1+3, H: 1s = 1

    def rand_block(ni, nj, cplx):
        m = rng.normal(size=(ni, nj))
        if cplx:
            m = m + 1j * rng.normal(size=(ni, nj))
        return m

    blocks = {}
    for i in (0, 1):
        for j in (0, 1):
            ni = dim[i] * (2 if is_soc else 1)
            nj = dim[j] * (2 if is_soc else 1)
            if i == j:
                m = rand_block(ni, nj, is_soc)
                m = 0.5 * (m + (m.conj().T if is_soc else m.T))  # onsite Hermitian
            else:
                m = rand_block(ni, nj, is_soc)
            blocks[f"{i}_{j}_0_0_0"] = m
    return atomic_numbers, basis_dict, blocks


@pytest.mark.parametrize("is_soc", [False, True])
@pytest.mark.parametrize("unit", ["eV", "Ha", "Ry"])
def test_write_read_csr_roundtrip(tmp_path, is_soc, unit):
    atomic_numbers, basis_dict, blocks = _toy_two_atom_blocks(is_soc)
    csr_path = tmp_path / "hrs1_nao.csr"

    hb.write_hr_csr(atomic_numbers, basis_dict, blocks, csr_path, unit=unit)
    read_back, norbits = hb.read_hr_csr(csr_path, atomic_numbers, basis_dict, unit=unit, is_soc=is_soc)

    expected_norbits = (4 + 1) * (2 if is_soc else 1)
    assert norbits == expected_norbits

    # write_blocks_to_abacus_csr Hermitian-completes missing reverse keys,
    # so read-back may contain extra keys (e.g. "1_0_0_0_0" filled in from
    # "0_1_0_0_0"); every key we wrote must round-trip to float32 precision.
    for key, original in blocks.items():
        assert key in read_back, f"missing key {key} after round trip"
        np.testing.assert_allclose(read_back[key], original, atol=1e-5, rtol=1e-5)


def test_unit_conversion_matches_hartree_to_rydberg_factor(tmp_path):
    """Regression guard for the eV-vs-Hartree unit bug this module exists to
    avoid: a value declared as 1.0 Ha must serialize to exactly 2.0 Ry in
    the CSR text (Ha = 2 Ry exactly), not 1.0/13.605698 Ry (which is what
    you'd get by mistakenly treating a Hartree-unit input as eV)."""
    atomic_numbers = np.array([1, 1])
    basis_dict = {"H": "1s"}
    blocks = {
        "0_0_0_0_0": np.array([[1.0]]),
        "1_1_0_0_0": np.array([[1.0]]),
        "0_1_0_0_0": np.array([[0.0]]),
    }
    csr_path = tmp_path / "h.csr"
    hb.write_hr_csr(atomic_numbers, basis_dict, blocks, csr_path, unit="Ha")
    data_line = csr_path.read_text().splitlines()[4]  # STEP/dim/count/"Rx Ry Rz nnz"/data
    values = [float(x) for x in data_line.split()]
    # 1.0 Ha must serialize to 2.0 Ry (float32 precision), not ~0.0735 Ry
    # (which is what a mistaken "treat Ha input as eV" bug would produce).
    for v in values:
        assert v == pytest.approx(2.0, abs=1e-5)
        assert v != pytest.approx(1.0 / hb.H_FACTOR, abs=1e-3)


def test_gap_guard_refuses_repair_below_threshold():
    h = np.diag([-1.0, -0.5, 0.05, 0.5])
    s = np.eye(4)
    # HOMO=-0.5 (idx1), LUMO=0.05 (idx2) -> gap = 0.55 eV
    allowed, gap = hb.gap_allows_repair(h, s, n_occ=2, gap_threshold_ev=0.6)
    assert not allowed
    assert gap == pytest.approx(0.55)

    allowed, gap = hb.gap_allows_repair(h, s, n_occ=2, gap_threshold_ev=0.1)
    assert allowed
    assert gap == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# one_shot_repair fail-closed behavior (fake executor, no real ABACUS)
# ---------------------------------------------------------------------------

def _repair_call_fixture(tmp_path):
    """Toy O-H inputs + a workdir pre-planted with a STALE output CSR, as a
    reused/dirty workdir would contain after an earlier successful run."""
    atomic_numbers, basis_dict, blocks = _toy_two_atom_blocks(is_soc=False)
    structure = hb.StructureSpec(
        symbols=["O", "H"],
        positions_angstrom=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]),
        cell_angstrom=np.eye(3) * 10.0,
    )
    pp_orb = {
        "O": hb.PPOrbSpec(pseudo="O.upf", orbital="O.orb", basis=basis_dict["O"], mass=15.999),
        "H": hb.PPOrbSpec(pseudo="H.upf", orbital="H.orb", basis=basis_dict["H"], mass=1.008),
    }
    workdir = tmp_path / "work"
    stale_csr = workdir / "OUT.ABACUS" / "hrs1_nao.csr"
    stale_csr.parent.mkdir(parents=True)
    hb.write_hr_csr(atomic_numbers, basis_dict, blocks, stale_csr, unit="eV")
    return blocks, structure, pp_orb, workdir, stale_csr


def test_one_shot_repair_fails_closed_on_nonzero_exit(tmp_path):
    """rc != 0 must refuse the repair even though a (stale) output CSR was
    lying around, and must surface run.log as the diagnostic tail (run_abacus
    redirects all ABACUS output there, so executor stdout/stderr are empty)."""
    blocks, structure, pp_orb, workdir, stale_csr = _repair_call_fixture(tmp_path)

    def failing_executor(wd, command):
        (Path(wd) / "run.log").write_text("ABACUS FATAL: mock crash\n")
        return 1, "", ""

    result = hb.one_shot_repair(
        blocks_dict=blocks, structure=structure, pp_orb=pp_orb,
        workdir=workdir, abacus_bin="abacus", executor=failing_executor,
        keep_workdir=True,
    )
    assert result.ok is False
    assert result.abacus_returncode == 1
    assert "exited with status 1" in result.skipped_reason
    assert "ABACUS FATAL" in result.stdout_tail
    assert not stale_csr.exists(), "stale CSR must be removed before launch"


def test_one_shot_repair_does_not_reuse_stale_csr(tmp_path):
    """rc == 0 but no fresh output written (e.g. misconfigured out_mat_hs2):
    the pre-existing stale CSR must NOT be read back as a 'successful' repair."""
    blocks, structure, pp_orb, workdir, stale_csr = _repair_call_fixture(tmp_path)

    def noop_executor(wd, command):
        return 0, "", ""

    result = hb.one_shot_repair(
        blocks_dict=blocks, structure=structure, pp_orb=pp_orb,
        workdir=workdir, abacus_bin="abacus", executor=noop_executor,
        keep_workdir=True,
    )
    assert result.ok is False
    assert "not found" in result.skipped_reason
    assert result.repaired_blocks is None

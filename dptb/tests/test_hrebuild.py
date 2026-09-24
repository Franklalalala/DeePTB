"""Offline unit tests for dptb.postprocess.hrebuild (WS4).

These do not invoke ABACUS -- the end-to-end fixed-point validation against
the real restart_dh binary (periodic non-SOC + SOC) lives in the WS4 report,
run on a host where the binary is built. This file locks down the parts that
are pure Python and must stay correct under refactors: the CSR write/read
round trip (both non-SOC and SOC), unit conversion, the gap-threshold guard,
one_shot_repair's fail-closed behavior, and (WS4 phase B) the persistent
server's JSON encode/decode and threaded-dispatch path -- the repair request
there targets a nonexistent ABACUS binary and must come back as a clean
``{"ok": False, ...}`` response rather than crashing the server or hanging.
"""
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dftio", reason="dptb.postprocess.hrebuild's CSR codec needs dftio")

from dptb.postprocess import hrebuild as hb  # noqa: E402
from dptb.postprocess import hrebuild_server as hs  # noqa: E402

# Unix domain sockets are the production transport (Linux), but AF_UNIX
# is unavailable in some Windows Python builds used for local dev (this repo's
# conda env among them) -- fall back to loopback TCP there so this test still
# exercises the exact same server/client/protocol code.
_HAS_AF_UNIX = hasattr(socket, "AF_UNIX")


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


# ---------------------------------------------------------------------------
# WS4 phase B: the persistent server (encode/decode, dispatch, fail-closed)
# ---------------------------------------------------------------------------

def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tcp_port_open(host, port) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


@pytest.fixture
def running_server(tmp_path):
    scratch_root = str(tmp_path / "scratch")
    if _HAS_AF_UNIX:
        # A short, dedicated directory keeps the socket path well under AF_UNIX's
        # sun_path limit (~107 bytes); pytest's own tmp_path can nest deep enough
        # to exceed it, which fails the bind with ENAMETOOLONG rather than skip.
        socket_path = os.path.join(tempfile.mkdtemp(prefix="hrb"), "s")
        kwargs = dict(scratch_root=scratch_root, socket_path=socket_path, max_workers=2)
        ready_check = lambda: os.path.exists(socket_path)
        endpoint = dict(socket_path=socket_path)
    else:
        port = _free_tcp_port()
        kwargs = dict(scratch_root=scratch_root, tcp_address=("127.0.0.1", port), max_workers=2)
        ready_check = lambda: _tcp_port_open("127.0.0.1", port)
        endpoint = dict(tcp_address=("127.0.0.1", port))

    thread = threading.Thread(target=hs.serve, kwargs=kwargs, daemon=True)
    thread.start()
    for _ in range(50):
        if ready_check():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("server endpoint never came up")
    time.sleep(0.1)
    yield endpoint


def test_encode_decode_blocks_roundtrip():
    blocks = {
        "0_0_0_0_0": np.random.default_rng(0).normal(size=(4, 4)),
        "0_1_1_0_0": (np.random.default_rng(1).normal(size=(4, 1))
                       + 1j * np.random.default_rng(2).normal(size=(4, 1))),
    }
    payload = hs.encode_blocks(blocks)
    decoded = hs.decode_blocks(payload)
    for k, v in blocks.items():
        np.testing.assert_allclose(decoded[k], v)


def _can_symlink(tmp_path) -> bool:
    """os.symlink needs a privilege on Windows; skip rather than fail there."""
    probe_target = tmp_path / "symlink_probe_target"
    probe_target.write_text("x")
    try:
        os.symlink(str(probe_target), str(tmp_path / "symlink_probe_link"))
        return True
    except OSError:
        return False


def test_pp_orb_cache_collision_raises(tmp_path):
    """Two different files staged under the same entry name must fail loudly
    (silently serving the first-staged pseudo to a request that meant another
    one is a wrong-physics bug); restaging the same source is a no-op."""
    if not _can_symlink(tmp_path):
        pytest.skip("os.symlink unavailable (Windows without privilege)")

    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    for d in (src_a, src_b):
        d.mkdir()
        (d / "H.upf").write_text(f"pseudo from {d.name}\n")

    cache = hs.PPOrbCache(str(tmp_path / "cache"))
    staged = cache.stage(str(src_a))
    assert os.path.samefile(os.path.join(staged, "H.upf"), src_a / "H.upf")

    # same source again: cached, no error
    assert cache.stage(str(src_a)) == staged

    with pytest.raises(FileExistsError, match="collision"):
        cache.stage(str(src_b))


def test_repair_response_carries_guard_diagnostics_and_passthrough(tmp_path, monkeypatch):
    """The service must pass input_extra/sc_guard through to one_shot_repair
    and return the self-consistency guard diagnostics (plus output tails on
    failure) so remote callers audit repairs with local semantics."""
    captured = {}

    def fake_repair(**kwargs):
        captured.update(kwargs)
        return hb.RepairResult(
            ok=False, mode=kwargs.get("mode", "one_shot"),
            skipped_reason="fake failure", abacus_returncode=3,
            stdout_tail="MOCK RUN LOG TAIL", stderr_tail="MOCK STDERR",
            workdir=str(kwargs.get("workdir")),
            sc_residual_mean_ev=1.9e-3, sc_residual_max_ev=2.5e-3, sc_n_common=4,
            e_kohnsham_ev=-3904.73, e_harris_ev=-3904.51, harris_ks_gap_ev=0.22,
            repair_trustworthy=True, guard_reason=None,
        )

    monkeypatch.setattr(hs.hb, "one_shot_repair", fake_repair)
    service = hs.HrebuildService(scratch_root=str(tmp_path / "scratch"), max_workers=1)
    try:
        resp = service.handle_request({
            "kind": "repair",
            "blocks_dict": hs.encode_blocks({"0_0_0_0_0": np.eye(1)}),
            "structure": {
                "symbols": ["H"],
                "positions_angstrom": [[0.0, 0.0, 0.0]],
                "cell_angstrom": np.eye(3).tolist(),
            },
            "pp_orb": {"H": {"pseudo": "H.upf", "orbital": "H.orb", "basis": "1s", "mass": 1.008}},
            "abacus_bin": "abacus",
            "input_extra": {"scf_thr": 1e-8},
            "sc_guard": {"max_residual_mean_ev": 0.5},
        })
    finally:
        service.shutdown()

    assert captured["input_extra"] == {"scf_thr": 1e-8}
    assert captured["sc_guard"] == {"max_residual_mean_ev": 0.5}

    assert resp["ok"] is False
    assert resp["sc_residual_mean_ev"] == pytest.approx(1.9e-3)
    assert resp["harris_ks_gap_ev"] == pytest.approx(0.22)
    assert resp["repair_trustworthy"] is True
    assert resp["stdout_tail"] == "MOCK RUN LOG TAIL"
    assert resp["stderr_tail"] == "MOCK STDERR"
    assert "repaired_blocks" not in resp


def test_repair_request_failure_is_reported_not_fatal(running_server):
    """A request pointing at a nonexistent ABACUS binary must come back as
    a clean error response, and the server/connection must stay usable
    afterwards (two ping round trips bracketing the failing call, which also
    covers the plain ping/pong contract)."""
    client = hs.Client(**running_server)
    try:
        assert client.ping()["ok"] is True

        structure = hb.StructureSpec(
            symbols=["H", "H"],
            positions_angstrom=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
            cell_angstrom=np.eye(3) * 20.0,
        )
        pp_orb = {"H": hb.PPOrbSpec(pseudo="H.upf", orbital="H.orb", basis="1s", mass=1.008)}
        blocks = {
            "0_0_0_0_0": np.eye(1) * -0.5,
            "1_1_0_0_0": np.eye(1) * -0.5,
            "0_1_0_0_0": np.eye(1) * -0.1,
        }
        resp = client.repair(
            blocks_dict=blocks,
            structure=structure,
            pp_orb=pp_orb,
            abacus_bin="/nonexistent/abacus/binary",
            mode="one_shot",
            timeout_sec=30,
        )
        assert resp["ok"] is False
        assert resp.get("error") or resp.get("skipped_reason")

        assert client.ping()["ok"] is True
    finally:
        client.close()

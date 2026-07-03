"""WS4 phase B smoke test: server starts, accepts a persistent connection,
round-trips ping + a repair request through the full JSON encode/decode and
threaded-dispatch path. Does not require a real ABACUS binary -- the repair
request targets a nonexistent binary and is expected to fail *inside* the
worker pool and come back as a clean ``{"ok": False, "error": ...}``
response rather than crashing the server or hanging the connection, which
is exactly the "server stays up under a bad/failing request" property this
test locks down. The real ABACUS-backed throughput/soak-test acceptance
criteria need a live binary and are out of scope here (see WS4 report).
"""
import os
import socket
import tempfile
import threading
import time

import numpy as np
import pytest

from dptb.postprocess import hrebuild as hb
from dptb.postprocess import hrebuild_server as hs


# Unix domain sockets are the production transport (natlan/Linux), but
# AF_UNIX is unavailable in some Windows Python builds used for local dev
# (this repo's conda env among them) -- fall back to loopback TCP there so
# this test still exercises the exact same server/client/protocol code.
_HAS_AF_UNIX = hasattr(socket, "AF_UNIX")


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_server(tmp_path):
    scratch_root = str(tmp_path / "scratch")
    if _HAS_AF_UNIX:
        socket_path = str(tmp_path / "hrebuild.sock")
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


def _tcp_port_open(host, port) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def test_ping(running_server):
    client = hs.Client(**running_server)
    try:
        resp = client.ping()
        assert resp == {"ok": True, "kind": "pong"}
    finally:
        client.close()


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
    afterwards (two ping round trips bracketing the failing call)."""
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

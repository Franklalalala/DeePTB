"""WS4 phase B: a resident service wrapping :mod:`dptb.postprocess.hrebuild`.

ABACUS itself has no persistent/server mode (the ``restart_dh`` patch only
adds ``init_chg hr`` + ``refresh(false)``; it is still a one-shot CLI binary
per invocation, and red line #5 forbids reworking its C++ internals here).
What this module amortizes across requests instead is everything *around*
each ABACUS invocation:

  - a warm worker pool (no Python/interpreter cold-start per request),
  - a shared, symlinked ``PP_ORB`` cache directory instead of a per-request
    copy of potentially large pseudopotential/orbital files,
  - concurrent in-flight ABACUS subprocesses (bounded by ``max_workers``)
    instead of one request blocking the next end-to-end,
  - a persistent unix-domain-socket connection so clients don't pay SSH/
    process-launch overhead per repair call.

Protocol: newline-delimited JSON over a Unix domain socket (default) or
TCP. One request = one line of JSON in, one line of JSON out.

Request fields mirror :func:`dptb.postprocess.hrebuild.one_shot_repair`
keyword arguments; ``blocks_dict`` values are serialized as
``{"__complex__": bool, "shape": [...], "data": [...]}`` (nested lists,
real/imag interleaved when complex) since JSON has no native array type.

This is scaffolding validated at the protocol/mechanism level (see
``dptb/tests/test_hrebuild_server.py``: start server, round-trip a
synthetic repair-less "echo" style request, shut down cleanly). The
``>=10x throughput vs. cold CLI`` and ``24h no leak`` Phase-B acceptance
criteria in the WS4 plan need a production benchmarking run against a real
ABACUS binary and real traffic, which is out of scope for this session --
see the WS4 report for what remains.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import socketserver
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from dptb.postprocess import hrebuild as hb

log = logging.getLogger(__name__)


def _encode_array(arr: np.ndarray) -> Dict:
    is_complex = np.iscomplexobj(arr)
    payload = {
        "__complex__": is_complex,
        "shape": list(arr.shape),
    }
    if is_complex:
        payload["real"] = arr.real.tolist()
        payload["imag"] = arr.imag.tolist()
    else:
        payload["data"] = arr.tolist()
    return payload


def _decode_array(payload: Dict) -> np.ndarray:
    shape = tuple(payload["shape"])
    if payload.get("__complex__"):
        real = np.asarray(payload["real"], dtype=np.float64).reshape(shape)
        imag = np.asarray(payload["imag"], dtype=np.float64).reshape(shape)
        return real + 1j * imag
    return np.asarray(payload["data"], dtype=np.float64).reshape(shape)


def encode_blocks(blocks: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    return {k: _encode_array(np.asarray(v)) for k, v in blocks.items()}


def decode_blocks(payload: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    return {k: _decode_array(v) for k, v in payload.items()}


class PPOrbCache:
    """Shared, symlinked PP_ORB staging directory. Each unique element ->
    pseudopotential/orbital file pair is symlinked once into
    ``cache_root``; per-request work dirs then symlink ``PP_ORB`` -> this
    shared directory instead of copying files, so a busy server doesn't
    re-copy multi-MB orbital files on every request."""

    def __init__(self, cache_root: str):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._staged: set = set()

    def stage(self, source_dir: str) -> str:
        source_dir = str(Path(source_dir).resolve())
        with self._lock:
            if source_dir in self._staged:
                return str(self.cache_root)
            for entry in os.listdir(source_dir):
                dst = self.cache_root / entry
                if not dst.exists():
                    os.symlink(os.path.join(source_dir, entry), dst)
            self._staged.add(source_dir)
        return str(self.cache_root)


class HrebuildService:
    """The actual request handler, independent of the socket transport so
    it can be unit-tested without opening a real socket."""

    def __init__(self, *, scratch_root: str, pp_orb_cache: Optional[PPOrbCache] = None, max_workers: int = 4):
        self.scratch_root = Path(scratch_root)
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self.pp_orb_cache = pp_orb_cache
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self._request_counter = 0
        self._counter_lock = threading.Lock()

    def _next_workdir(self) -> Path:
        with self._counter_lock:
            self._request_counter += 1
            n = self._request_counter
        return self.scratch_root / f"req_{n:08d}_{int(time.time())}"

    def handle_request(self, req: Dict) -> Dict:
        kind = req.get("kind")
        if kind == "ping":
            return {"ok": True, "kind": "pong"}
        if kind != "repair":
            return {"ok": False, "error": f"unknown request kind {kind!r}"}

        try:
            blocks_dict = decode_blocks(req["blocks_dict"])
            overlap_blocks_dict = (
                decode_blocks(req["overlap_blocks_dict"]) if req.get("overlap_blocks_dict") else None
            )
            structure = hb.StructureSpec(
                symbols=req["structure"]["symbols"],
                positions_angstrom=np.asarray(req["structure"]["positions_angstrom"]),
                cell_angstrom=np.asarray(req["structure"]["cell_angstrom"]),
                pbc=tuple(req["structure"].get("pbc", (True, True, True))),
            )
            pp_orb = {
                el: hb.PPOrbSpec(**spec) for el, spec in req["pp_orb"].items()
            }

            workdir = self._next_workdir()
            pp_orb_dir_arg = "PP_ORB"
            if self.pp_orb_cache is not None and req.get("pp_orb_source_dir"):
                staged = self.pp_orb_cache.stage(req["pp_orb_source_dir"])
                pp_orb_dir_arg = staged
            elif req.get("pp_orb_source_dir"):
                pp_orb_dir_arg = req["pp_orb_source_dir"]

            future = self.pool.submit(
                hb.one_shot_repair,
                blocks_dict=blocks_dict,
                structure=structure,
                pp_orb=pp_orb,
                workdir=workdir,
                abacus_bin=req["abacus_bin"],
                unit=req.get("unit", "eV"),
                mode=req.get("mode", "one_shot"),
                is_soc=req.get("is_soc", False),
                nspin=req.get("nspin"),
                lspinorb=req.get("lspinorb", 0),
                kmesh=tuple(req.get("kmesh", (1, 1, 1))),
                ecutwfc=req.get("ecutwfc", 100.0),
                pp_orb_dir=pp_orb_dir_arg,
                n_occ=req.get("n_occ"),
                overlap_blocks_dict=overlap_blocks_dict,
                gap_threshold_ev=req.get("gap_threshold_ev", 0.5),
                mpi_procs=req.get("mpi_procs", 1),
                keep_workdir=req.get("keep_workdir", False),
            )
            result = future.result(timeout=req.get("timeout_sec", 600))
        except Exception as exc:  # noqa: BLE001 - report to client, don't crash the server
            log.warning(f"[hrebuild-server] request failed: {exc}\n{traceback.format_exc()}")
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        resp = {
            "ok": result.ok,
            "mode": result.mode,
            "gap_ev": result.gap_ev,
            "skipped_reason": result.skipped_reason,
            "abacus_returncode": result.abacus_returncode,
        }
        if result.ok:
            resp["repaired_blocks"] = encode_blocks(result.repaired_blocks)
        return resp

    def shutdown(self):
        self.pool.shutdown(wait=True)


class _LineHandler(socketserver.StreamRequestHandler):
    def handle(self):
        service: HrebuildService = self.server.service  # type: ignore[attr-defined]
        for raw_line in self.rfile:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = service.handle_request(req)
            except Exception as exc:  # noqa: BLE001
                resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))


class TCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


# AF_UNIX / socketserver.UnixStreamServer is unavailable on some Windows
# Python builds (used for local dev of this repo); the production target
# (natlan/Linux) always has it. Define the unix-socket transport only when
# supported, so importing this module doesn't fail on Windows dev machines.
if hasattr(socket, "AF_UNIX") and hasattr(socketserver, "UnixStreamServer"):

    class UnixStreamServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True
        allow_reuse_address = True

else:
    UnixStreamServer = None


def serve(
    *,
    scratch_root: str,
    socket_path: Optional[str] = None,
    tcp_address: Optional[tuple] = None,
    pp_orb_cache_root: Optional[str] = None,
    max_workers: int = 4,
):
    """Blocking call: run the hrebuild server. Exactly one of
    ``socket_path`` (unix domain socket, default/recommended for
    single-host use) or ``tcp_address=(host, port)`` must be given."""
    if bool(socket_path) == bool(tcp_address):
        raise ValueError("pass exactly one of socket_path or tcp_address")

    pp_orb_cache = PPOrbCache(pp_orb_cache_root) if pp_orb_cache_root else None
    service = HrebuildService(scratch_root=scratch_root, pp_orb_cache=pp_orb_cache, max_workers=max_workers)

    if socket_path:
        if UnixStreamServer is None:
            raise RuntimeError("Unix domain sockets are unavailable on this Python/OS; pass tcp_address instead.")
        if os.path.exists(socket_path):
            os.remove(socket_path)
        server = UnixStreamServer(socket_path, _LineHandler)
    else:
        server = TCPServer(tcp_address, _LineHandler)
    server.service = service  # type: ignore[attr-defined]

    log.info(f"[hrebuild-server] listening on {socket_path or tcp_address}, max_workers={max_workers}")
    try:
        server.serve_forever()
    finally:
        service.shutdown()
        server.server_close()


class Client:
    """Thin synchronous client. Reuses one connection across calls (the
    throughput win vs. a cold CLI per repair call)."""

    def __init__(self, socket_path: Optional[str] = None, tcp_address: Optional[tuple] = None):
        if bool(socket_path) == bool(tcp_address):
            raise ValueError("pass exactly one of socket_path or tcp_address")
        if socket_path:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(socket_path)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(tcp_address)
        self._rfile = self.sock.makefile("rb")

    def _call(self, req: Dict, timeout: Optional[float] = None) -> Dict:
        if timeout is not None:
            self.sock.settimeout(timeout)
        self.sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
        line = self._rfile.readline()
        return json.loads(line.decode("utf-8"))

    def ping(self) -> Dict:
        return self._call({"kind": "ping"})

    def repair(self, **kwargs) -> Dict:
        req = dict(kwargs)
        req["kind"] = "repair"
        req["blocks_dict"] = encode_blocks(kwargs["blocks_dict"])
        if kwargs.get("overlap_blocks_dict"):
            req["overlap_blocks_dict"] = encode_blocks(kwargs["overlap_blocks_dict"])
        req["structure"] = {
            "symbols": list(kwargs["structure"].symbols),
            "positions_angstrom": np.asarray(kwargs["structure"].positions_angstrom).tolist(),
            "cell_angstrom": np.asarray(kwargs["structure"].cell_angstrom).tolist(),
            "pbc": list(kwargs["structure"].pbc),
        }
        req["pp_orb"] = {
            el: {"pseudo": spec.pseudo, "orbital": spec.orbital, "basis": spec.basis, "mass": spec.mass}
            for el, spec in kwargs["pp_orb"].items()
        }
        timeout = req.pop("timeout_sec", None)
        resp = self._call(req, timeout=timeout)
        if resp.get("ok") and "repaired_blocks" in resp:
            resp["repaired_blocks"] = decode_blocks(resp["repaired_blocks"])
        return resp

    def close(self):
        self.sock.close()

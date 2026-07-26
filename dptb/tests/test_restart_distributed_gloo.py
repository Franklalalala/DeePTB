"""2-rank gloo conformance harness for the distributed checkpoint paths.

Real 2-process ``torch.distributed`` coverage (backend=gloo, subprocess
workers, loopback TCP rendezvous) for the P0-3/P0-4 Saver semantics that
single-process tests can only mock:

(a) ``rank_states`` round-trip: each rank contributes its OWN RNG snapshot via
    the ``all_gather_object`` collective and ``resolve_rank_rng_state`` hands
    each rank back its own entry (not rank0's).
(b) publish-failure propagation: a rank0 ``_write_manifest`` failure raises on
    BOTH ranks through the ``_finalize_publish`` broadcast — no deadlock.
(c) save assembly-failure propagation: a rank0 ``_assemble_full_model_state``
    failure inside the real ``Saver._save`` distributed path raises on BOTH
    ranks BEFORE the post-save barrier — no deadlock.
(d) real manifest ``os.replace`` failure and rank-local prepare failure both
    propagate to BOTH ranks before any peer can enter an unmatched collective.

Every test drives the workers with a join timeout so a hang FAILS the test
instead of blocking the suite. If 2-rank gloo init is genuinely impossible on
the host, the worker records the real failed attempt and the test SKIPs.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

import torch.distributed as dist

from dptb.nnops.training_state import (
    CHECKPOINT_KIND_ITERATION,
    read_resume_metadata,
    resolve_rank_rng_state,
)
from dptb.plugins.saver import Saver

WORLD_SIZE = 2
# Generous: each spawned worker cold-imports torch + dptb before rendezvous.
PG_TIMEOUT_S = 30.0
JOIN_TIMEOUT_S = 120.0


# ---------------------------------------------------------------------------
# Tiny fixtures (importable at top level: mp.spawn pickles worker references)
# ---------------------------------------------------------------------------
class _TinyExpert(nn.Module):
    def __init__(self, fill):
        super().__init__()
        self.weight = nn.Parameter(torch.full((2,), float(fill)))


class _ExpertsModel(nn.Module):
    name = "probe"

    def __init__(self, num_experts):
        super().__init__()
        self.experts = nn.ModuleList(
            [_TinyExpert(i + 1) for i in range(num_experts)]
        )
        self.model_options = {"embedding": {}, "prediction": {}}


class _StubDistTrainer:
    """The minimal trainer surface Saver's distributed ``_save`` touches."""

    def __init__(self, rank, world_size):
        self.distributed_expert = True
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.is_main_process = rank == 0
        self.expert_data_parallel_size = 1
        self.local_expert_idx = int(rank)
        self.num_experts = int(world_size)
        self.model = _ExpertsModel(world_size)
        self.optimizers = [
            torch.optim.SGD(e.parameters(), lr=0.1) for e in self.model.experts
        ]
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
            for opt in self.optimizers
        ]
        self.iter = 5
        self.ep = 1
        self._batch_in_epoch = 5
        self.update_lr_per_iter = False
        self.stats = {}
        self.task = "hamiltonians"
        self.train_options = {"max_ckpt": 5}
        self.common_options = {"device": "cpu", "dtype": "float32"}
        self.device = "cpu"

    @staticmethod
    def _unwrap_expert_module(expert):
        return expert


def _write_result(result_dir, rank, text):
    path = os.path.join(result_dir, f"rank{rank}.txt")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _trace_stage(result_dir, rank, stage):
    """Leave a per-rank breadcrumb when a parent-side timeout kills workers."""
    path = os.path.join(result_dir, f"rank{rank}.stages")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(stage + "\n")
        fh.flush()


def _init_pg(rank, world_size, init_method, result_dir):
    """REAL 2-rank gloo init attempt; records SKIP on genuine impossibility."""
    try:
        _trace_stage(result_dir, rank, "init_pg_enter")
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=PG_TIMEOUT_S),
        )
        _trace_stage(result_dir, rank, "init_pg_ok")
        return True
    except Exception:
        _write_result(
            result_dir, rank, "SKIP:gloo init failed:\n" + traceback.format_exc()
        )
        return False


def _destroy_pg():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _make_saver(rank, world_size, ckpt_dir):
    trainer = _StubDistTrainer(rank, world_size)
    saver = Saver(interval=[(1, "iteration")])
    saver.trainer = trainer
    saver.checkpoint_path = str(ckpt_dir)
    saver.push = False
    return saver, trainer


# ---------------------------------------------------------------------------
# Workers (top-level functions: spawn pickles them by module reference)
# ---------------------------------------------------------------------------
def _worker_rank_states_roundtrip(rank, world_size, init_method, ckpt_dir, result_dir):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _trace_stage(result_dir, rank, "worker_start")
    try:
        if not _init_pg(rank, world_size, init_method, result_dir):
            return
        try:
            # Desynchronize the two ranks' torch RNG streams.
            torch.manual_seed(1000 + 17 * rank)
            _ = torch.rand(3 + rank)
            expected_rng = torch.get_rng_state()

            _trace_stage(result_dir, rank, "make_saver_enter")
            saver, trainer = _make_saver(rank, world_size, ckpt_dir)
            _trace_stage(result_dir, rank, "save_enter")
            saver._save(
                name="probe.iter5",
                model=trainer.model,
                model_options=trainer.model.model_options,
                common_options=trainer.common_options,
                train_options=trainer.train_options,
                kind=CHECKPOINT_KIND_ITERATION,
            )

            # _save ends in a barrier, so rank0's atomic write is complete.
            ckpt = torch.load(
                os.path.join(ckpt_dir, "probe.iter5.pth"),
                map_location="cpu",
                weights_only=False,
            )
            rank_states = ckpt["rank_states"]
            assert set(rank_states.keys()) == {"0", "1"}, (
                f"rank_states keys: {sorted(rank_states)}"
            )
            t0 = rank_states["0"]["rng_state"]["torch"]
            t1 = rank_states["1"]["rng_state"]["torch"]
            assert not torch.equal(t0, t1), (
                "per-rank RNG snapshots must differ across ranks"
            )

            resume = read_resume_metadata(ckpt)
            own = resolve_rank_rng_state(ckpt, resume)
            assert torch.equal(own["torch"], expected_rng), (
                "resolve_rank_rng_state must return THIS rank's own snapshot"
            )
            other = rank_states[str(1 - rank)]["rng_state"]["torch"]
            assert not torch.equal(own["torch"], other), (
                "resolve_rank_rng_state returned the other rank's snapshot"
            )

            # The assembled model state carries every expert exactly once.
            model_state = ckpt["model_state_dict"]
            for i in range(world_size):
                assert f"experts.{i}.weight" in model_state
            _write_result(result_dir, rank, "OK")
        except Exception:
            _write_result(result_dir, rank, "FAIL:\n" + traceback.format_exc())
    finally:
        _destroy_pg()


def _worker_publish_failure_propagates(rank, world_size, init_method, ckpt_dir, result_dir):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _trace_stage(result_dir, rank, "worker_start")
    try:
        if not _init_pg(rank, world_size, init_method, result_dir):
            return
        try:
            _trace_stage(result_dir, rank, "make_saver_enter")
            saver, _trainer = _make_saver(rank, world_size, ckpt_dir)
            _trace_stage(result_dir, rank, "finalize_publish_enter")
            if rank == 0:
                def _boom():
                    raise OSError("manifest disk full")

                saver._write_manifest = _boom

            def _publish():
                saver._write_manifest()

            try:
                saver._finalize_publish(_publish)
            except OSError as exc:
                if rank == 0 and "manifest disk full" in str(exc):
                    _write_result(result_dir, rank, "OK")
                    return
                raise
            except RuntimeError as exc:
                if rank != 0 and "aborting on all ranks" in str(exc):
                    _write_result(result_dir, rank, "OK")
                    return
                raise
            _write_result(
                result_dir, rank,
                f"FAIL:_finalize_publish did not raise on rank {rank}",
            )
        except Exception:
            _write_result(result_dir, rank, "FAIL:\n" + traceback.format_exc())
    finally:
        _destroy_pg()


def _worker_assembly_failure_propagates(rank, world_size, init_method, ckpt_dir, result_dir):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _trace_stage(result_dir, rank, "worker_start")
    try:
        if not _init_pg(rank, world_size, init_method, result_dir):
            return
        try:
            _trace_stage(result_dir, rank, "make_saver_enter")
            saver, trainer = _make_saver(rank, world_size, ckpt_dir)
            _trace_stage(result_dir, rank, "save_enter")
            if rank == 0:
                def _boom(expert_states):
                    raise RuntimeError("rank0 assemble failed")

                saver._assemble_full_model_state = _boom

            try:
                saver._save(
                    name="probe.iter1",
                    model=trainer.model,
                    model_options=trainer.model.model_options,
                    common_options=trainer.common_options,
                    train_options=trainer.train_options,
                    kind=CHECKPOINT_KIND_ITERATION,
                )
            except RuntimeError as exc:
                msg = str(exc)
                ok = (
                    (rank == 0 and "rank0 assemble failed" in msg)
                    or (rank != 0 and "aborting on all ranks" in msg)
                )
                if ok:
                    # Both ranks raised BEFORE the post-save barrier: had any
                    # rank entered it, it would hang and trip the join timeout.
                    assert not os.path.exists(
                        os.path.join(ckpt_dir, "probe.iter1.pth")
                    ), "failed save must not leave a checkpoint file"
                    _write_result(result_dir, rank, "OK")
                    return
                raise
            _write_result(
                result_dir, rank,
                f"FAIL:Saver._save did not raise on rank {rank}",
            )
        except Exception:
            _write_result(result_dir, rank, "FAIL:\n" + traceback.format_exc())
    finally:
        _destroy_pg()


def _worker_manifest_io_failure_propagates(rank, world_size, init_method, ckpt_dir, result_dir):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _trace_stage(result_dir, rank, "worker_start")
    try:
        if not _init_pg(rank, world_size, init_method, result_dir):
            return
        try:
            import dptb.plugins.saver as saver_mod

            saver, trainer = _make_saver(rank, world_size, ckpt_dir)
            real_replace = saver_mod.os.replace
            manifest_path = os.path.normcase(
                os.path.join(ckpt_dir, Saver.MANIFEST_NAME)
            )

            if rank == 0:
                def fail_manifest_replace(src, dst):
                    if os.path.normcase(os.fspath(dst)) == manifest_path:
                        raise OSError("real manifest replace failed")
                    return real_replace(src, dst)

                saver_mod.os.replace = fail_manifest_replace

            _trace_stage(result_dir, rank, "iteration_save_enter")
            try:
                saver.iteration(field="iteration")
            except OSError as exc:
                if rank == 0 and "real manifest replace failed" in str(exc):
                    _write_result(result_dir, rank, "OK")
                    return
                raise
            except RuntimeError as exc:
                if rank != 0 and "aborting on all ranks" in str(exc):
                    _write_result(result_dir, rank, "OK")
                    return
                raise
            _write_result(result_dir, rank, f"FAIL:manifest I/O did not raise on rank {rank}")
        except Exception:
            _write_result(result_dir, rank, "FAIL:\n" + traceback.format_exc())
    finally:
        _destroy_pg()


def _worker_rank1_prepare_failure_propagates(rank, world_size, init_method, ckpt_dir, result_dir):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _trace_stage(result_dir, rank, "worker_start")
    try:
        if not _init_pg(rank, world_size, init_method, result_dir):
            return
        try:
            saver, trainer = _make_saver(rank, world_size, ckpt_dir)
            if rank == 1:
                expert = trainer.model.experts[trainer.local_expert_idx]

                def fail_local_state_dict(*args, **kwargs):
                    raise RuntimeError("rank1 local state_dict failed")

                expert.state_dict = fail_local_state_dict

            _trace_stage(result_dir, rank, "prepare_failure_save_enter")
            try:
                saver._save(
                    name="probe.iter5",
                    model=trainer.model,
                    model_options=trainer.model.model_options,
                    common_options=trainer.common_options,
                    train_options=trainer.train_options,
                    kind=CHECKPOINT_KIND_ITERATION,
                )
            except RuntimeError as exc:
                message = str(exc)
                if (
                    "checkpoint local prepare failed on rank 1" in message
                    and "rank1 local state_dict failed" in message
                ):
                    _write_result(result_dir, rank, "OK")
                    return
                raise
            _write_result(result_dir, rank, f"FAIL:prepare failure did not raise on rank {rank}")
        except Exception:
            _write_result(result_dir, rank, "FAIL:\n" + traceback.format_exc())
    finally:
        _destroy_pg()


# ---------------------------------------------------------------------------
# Parent-side runner
# ---------------------------------------------------------------------------
def _terminate(processes):
    for p in processes:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass
    for p in processes:
        try:
            p.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
                p.wait(timeout=5.0)
            except Exception:
                pass


def _free_loopback_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker_command(worker, rank, init_method, ckpt_dir, result_dir):
    return [
        sys.executable,
        os.path.abspath(__file__),
        "--gloo-worker",
        worker.__name__,
        str(rank),
        str(WORLD_SIZE),
        init_method,
        str(ckpt_dir),
        str(result_dir),
    ]


def _run_two_rank(worker, tmp_path, monkeypatch):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available in this build")
    # Keep the subprocess workers CPU-only (fast startup, deterministic RNG keys);
    # monkeypatch restores the parent env after the test.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    if os.name == "nt" and not os.environ.get("GLOO_SOCKET_IFNAME"):
        # Gloo's Windows UV transport expects the adapter alias (the value
        # returned by socket.if_nameindex(), ``loopback_0``, is not accepted).
        monkeypatch.setenv(
            "GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1"
        )

    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    # ``mp.spawn`` + file:// was attempted first on the target Windows host,
    # but both ranks hung inside init_process_group until the parent timeout.
    # The BRIEF explicitly permits a subprocess fallback. A loopback TCP store
    # keeps this a real 2-rank gloo process group while avoiding that host's
    # broken FileStore rendezvous path.
    init_method = (
        f"tcp://127.0.0.1:{_free_loopback_port()}?use_libuv=0"
    )
    worker_env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[2])
    worker_env["PYTHONPATH"] = repo_root + (
        os.pathsep + worker_env["PYTHONPATH"]
        if worker_env.get("PYTHONPATH")
        else ""
    )
    processes = []
    logs = []
    try:
        for rank in range(WORLD_SIZE):
            log_path = result_dir / f"rank{rank}.subprocess.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            logs.append(log_fh)
            processes.append(
                subprocess.Popen(
                    _worker_command(
                        worker, rank, init_method, ckpt_dir, result_dir
                    ),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=worker_env,
                )
            )

        deadline = time.monotonic() + JOIN_TIMEOUT_S
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, JOIN_TIMEOUT_S)
            process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        _terminate(processes)
        pytest.fail(
            f"2-rank gloo workers did not finish within {JOIN_TIMEOUT_S}s "
            "(possible deadlock/hang)"
        )
    finally:
        for log_fh in logs:
            log_fh.close()

    crashed = {
        rank: process.returncode
        for rank, process in enumerate(processes)
        if process.returncode != 0
    }
    if crashed:
        details = {
            rank: (result_dir / f"rank{rank}.subprocess.log").read_text(
                encoding="utf-8"
            )
            for rank in crashed
        }
        pytest.fail(f"distributed worker process failures: {crashed}; {details}")

    results = {}
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank{rank}.txt"
        assert path.exists(), f"rank {rank} exited without writing a result"
        results[rank] = path.read_text(encoding="utf-8")

    skips = [r for r in results.values() if r.startswith("SKIP:")]
    if skips:
        pytest.skip(
            "2-rank gloo init not possible on this host after a real attempt: "
            + skips[0][:500]
        )
    failures = {k: v for k, v in results.items() if v != "OK"}
    assert not failures, f"worker failures: {failures}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_rank_states_roundtrip_two_rank_gloo(tmp_path, monkeypatch):
    _run_two_rank(_worker_rank_states_roundtrip, tmp_path, monkeypatch)


def test_publish_failure_propagates_to_all_ranks(tmp_path, monkeypatch):
    _run_two_rank(_worker_publish_failure_propagates, tmp_path, monkeypatch)


def test_save_assembly_failure_propagates_before_barrier(tmp_path, monkeypatch):
    _run_two_rank(_worker_assembly_failure_propagates, tmp_path, monkeypatch)


def test_real_manifest_io_failure_propagates_to_all_ranks(tmp_path, monkeypatch):
    _run_two_rank(_worker_manifest_io_failure_propagates, tmp_path, monkeypatch)


def test_rank1_prepare_failure_propagates_before_large_gather(tmp_path, monkeypatch):
    _run_two_rank(_worker_rank1_prepare_failure_propagates, tmp_path, monkeypatch)


if __name__ == "__main__":
    workers = {
        fn.__name__: fn
        for fn in (
            _worker_rank_states_roundtrip,
            _worker_publish_failure_propagates,
            _worker_assembly_failure_propagates,
            _worker_manifest_io_failure_propagates,
            _worker_rank1_prepare_failure_propagates,
        )
    }
    if len(sys.argv) != 8 or sys.argv[1] != "--gloo-worker":
        raise SystemExit("invalid gloo worker invocation")
    workers[sys.argv[2]](
        int(sys.argv[3]),
        int(sys.argv[4]),
        sys.argv[5],
        sys.argv[6],
        sys.argv[7],
    )

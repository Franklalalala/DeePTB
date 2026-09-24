"""Real 2-rank gloo coverage for the distributed-expert Saver and iteration paths.

Subprocess workers over a loopback TCP rendezvous (backend=gloo) exercise what
single-process tests can only mock:

(a) ``rank_states`` round-trip: each rank contributes its OWN RNG snapshot via
    the ``all_gather_object`` collective and ``resolve_rank_rng_state`` hands
    each rank back its own entry (not rank0's).
(b) a rank0-side or rank1-side save failure raises on BOTH ranks before any
    peer can enter an unmatched collective, and leaves no checkpoint file.
(c) a NaN/Inf/NaN-gradient loss on one rank is never partly committed by
    either rank; the next good batch updates both.

Every test drives the workers with a join timeout so a hang FAILS the test
instead of blocking the suite. If 2-rank gloo init is genuinely impossible on
the host, the worker records the real failed attempt and the test SKIPs.
"""

from __future__ import annotations

import copy
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import timedelta

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
from dptb.tests._trainer_probes import DistPathProbeTrainer, StubBatch, corrupt

pytestmark = pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with the gloo backend",
)

WORLD_SIZE = 2
# Generous: each spawned worker cold-imports torch + dptb before rendezvous.
PG_TIMEOUT_S = 30.0
JOIN_TIMEOUT_S = 120.0


# ---------------------------------------------------------------------------
# Fixtures shared by the Saver-failure workers (top-level: workers re-import
# this module by name, so anything they touch must be importable that way)
# ---------------------------------------------------------------------------
class _TinyExpert(nn.Module):
    def __init__(self, fill):
        super().__init__()
        self.weight = nn.Parameter(torch.full((2,), float(fill)))


class _ExpertsModel(nn.Module):
    name = "probe"

    def __init__(self, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([_TinyExpert(i + 1) for i in range(num_experts)])
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
        self.optimizers = [torch.optim.SGD(e.parameters(), lr=0.1) for e in self.model.experts]
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5) for opt in self.optimizers
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
            backend="gloo", init_method=init_method, rank=rank, world_size=world_size,
            timeout=timedelta(seconds=PG_TIMEOUT_S),
        )
        _trace_stage(result_dir, rank, "init_pg_ok")
        return True
    except Exception:
        _write_result(result_dir, rank, "SKIP:gloo init failed:\n" + traceback.format_exc())
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
# Worker (a): each rank's RNG snapshot round-trips through rank_states
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
                name="probe.iter5", model=trainer.model, model_options=trainer.model.model_options,
                common_options=trainer.common_options, train_options=trainer.train_options,
                kind=CHECKPOINT_KIND_ITERATION,
            )

            # _save ends in a barrier, so rank0's atomic write is complete.
            ckpt = torch.load(os.path.join(ckpt_dir, "probe.iter5.pth"), map_location="cpu",
                              weights_only=False)
            rank_states = ckpt["rank_states"]
            assert set(rank_states.keys()) == {"0", "1"}, f"rank_states keys: {sorted(rank_states)}"
            t0 = rank_states["0"]["rng_state"]["torch"]
            t1 = rank_states["1"]["rng_state"]["torch"]
            assert not torch.equal(t0, t1), "per-rank RNG snapshots must differ across ranks"

            resume = read_resume_metadata(ckpt)
            own = resolve_rank_rng_state(ckpt, resume)
            assert torch.equal(own["torch"], expected_rng), "must return THIS rank's own snapshot"
            other = rank_states[str(1 - rank)]["rng_state"]["torch"]
            assert not torch.equal(own["torch"], other), "returned the other rank's snapshot"

            # The assembled model state carries every expert exactly once.
            model_state = ckpt["model_state_dict"]
            for i in range(world_size):
                assert f"experts.{i}.weight" in model_state
            _write_result(result_dir, rank, "OK")
        except Exception:
            _write_result(result_dir, rank, "FAIL:\n" + traceback.format_exc())
    finally:
        _destroy_pg()


# ---------------------------------------------------------------------------
# Worker (b): a save failure on either rank aborts both ranks with no
# partial checkpoint, whichever stage the failure is injected at.
# ---------------------------------------------------------------------------
def _inject_publish_failure(saver, trainer, where, rank):
    if where == "rank0_assembly" and rank == 0:
        def _boom(expert_states):
            raise RuntimeError("rank0 assemble failed")

        saver._assemble_full_model_state = _boom
    elif where == "rank0_manifest_replace" and rank == 0:
        import dptb.plugins.saver as saver_mod

        real_replace = saver_mod.os.replace
        manifest_path = os.path.normcase(os.path.join(saver.checkpoint_path, Saver.MANIFEST_NAME))

        def fail_manifest_replace(src, dst):
            if os.path.normcase(os.fspath(dst)) == manifest_path:
                raise OSError("manifest replace failed")
            return real_replace(src, dst)

        saver_mod.os.replace = fail_manifest_replace
    elif where == "rank1_local_state_dict" and rank == 1:
        expert = trainer.model.experts[trainer.local_expert_idx]

        def fail_local_state_dict(*args, **kwargs):
            raise RuntimeError("rank1 local state_dict failed")

        expert.state_dict = fail_local_state_dict


def _publish_failure_worker(rank, world_size, init_method, ckpt_dir, result_dir, where):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _trace_stage(result_dir, rank, "worker_start")
    try:
        if not _init_pg(rank, world_size, init_method, result_dir):
            return
        try:
            saver, trainer = _make_saver(rank, world_size, ckpt_dir)
            _inject_publish_failure(saver, trainer, where, rank)
            _trace_stage(result_dir, rank, "save_enter")
            try:
                if where == "rank0_manifest_replace":
                    saver.iteration(field="iteration")
                else:
                    saver._save(
                        name="probe.iter5", model=trainer.model,
                        model_options=trainer.model.model_options,
                        common_options=trainer.common_options, train_options=trainer.train_options,
                        kind=CHECKPOINT_KIND_ITERATION,
                    )
            except Exception:
                # both ranks must raise before any of them could enter an
                # unmatched collective. A prepare/assembly failure happens
                # before any file write; a manifest-replace failure happens
                # after the checkpoint blob is already committed, so only the
                # former leaves no checkpoint file.
                if where != "rank0_manifest_replace":
                    assert not os.path.exists(os.path.join(ckpt_dir, "probe.iter5.pth")), \
                        "a failed publish must not leave a checkpoint file"
                _write_result(result_dir, rank, "OK")
                return
            _write_result(result_dir, rank, f"FAIL:no exception raised on rank {rank} ({where})")
        except Exception:
            _write_result(result_dir, rank, "FAIL:\n" + traceback.format_exc())
    finally:
        _destroy_pg()


def _worker_publish_failure_rank0_assembly(rank, world_size, init_method, ckpt_dir, result_dir):
    _publish_failure_worker(rank, world_size, init_method, ckpt_dir, result_dir, "rank0_assembly")


def _worker_publish_failure_rank0_manifest_replace(rank, world_size, init_method, ckpt_dir, result_dir):
    _publish_failure_worker(rank, world_size, init_method, ckpt_dir, result_dir, "rank0_manifest_replace")


def _worker_publish_failure_rank1_local_state_dict(rank, world_size, init_method, ckpt_dir, result_dir):
    _publish_failure_worker(rank, world_size, init_method, ckpt_dir, result_dir, "rank1_local_state_dict")


# ---------------------------------------------------------------------------
# Worker (c): a nonfinite loss on rank 1 is never partly committed on either
# rank; the next good batch updates both. ``mode="ddp"`` runs the same
# consensus through a real DistributedDataParallel-wrapped expert.
# ---------------------------------------------------------------------------
def _nonfinite_consensus_worker(rank, world_size, init_method, ckpt_dir, result_dir, mode):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _trace_stage(result_dir, rank, "worker_start")
    try:
        if not _init_pg(rank, world_size, init_method, result_dir):
            return
        try:
            trainer = DistPathProbeTrainer(num_experts=2 if mode == "experts" else 1)
            trainer.rank, trainer.world_size = rank, world_size
            trainer.local_expert_idx = rank if mode == "experts" else 0
            idx = trainer.local_expert_idx
            if mode == "ddp":
                trainer.model.experts[0] = torch.nn.parallel.DistributedDataParallel(
                    torch.nn.Linear(1, 1, bias=False)
                )
                trainer.expert_dp_backend = "ddp"
            trainer.optimizers[idx] = torch.optim.Adam(trainer.model.experts[idx].parameters(), lr=0.01)
            trainer.lr_schedulers[idx] = torch.optim.lr_scheduler.StepLR(trainer.optimizers[idx], 10)
            trainer.update_lr_per_iter = True
            trainer._should_flush_display_window_now = lambda _: False
            trainer.call_plugins = lambda **kwargs: None
            build = trainer._build_train_payload
            kind = None

            def payload(**kwargs):
                if mode == "ddp":
                    value = trainer.model.experts[0](torch.ones(1, 1)).square().sum()
                    result = {"expert_onsite": 0.1, "expert_hopping": 0.2, "active_nodes": 1.0,
                             "active_edges": 1.0, "onsite_weighted_sum": 0.1,
                             "hopping_weighted_sum": 0.2, "z_values": [], "load_cv_values": []}
                else:
                    result = build(**kwargs)
                    value = result["loss"]
                result["loss"] = corrupt(value, kind) if rank == 1 and kind else value
                return result

            trainer._build_train_payload = payload
            # Warm optimizer state, then confirm both fault kinds leave it and the weights intact.
            trainer.iteration(StubBatch(0))
            params = list(trainer.model.experts[idx].parameters())
            before = [p.detach().clone() for p in params]
            states = copy.deepcopy(trainer.optimizers[idx].state_dict())
            scheduler = copy.deepcopy(trainer.lr_schedulers[idx].state_dict())
            for kind in ("nan", "gradient"):
                assert trainer.iteration(StubBatch(1)) is None
                assert trainer.iter == 2
                assert all(torch.equal(p, b) and p.grad is None for p, b in zip(params, before))
                now = trainer.optimizers[idx].state_dict()
                for key, values in states["state"].items():
                    for name, value in values.items():
                        assert torch.equal(now["state"][key][name], value)
                assert trainer.lr_schedulers[idx].state_dict() == scheduler
            kind = None
            assert trainer.iteration(StubBatch(2)).isfinite()
            assert trainer.iter == 3 and trainer.training_state.batch_in_epoch == 4
            assert all(not torch.equal(p, b) for p, b in zip(params, before))
            _write_result(result_dir, rank, "OK")
        except Exception:
            _write_result(result_dir, rank, "FAIL:\n" + traceback.format_exc())
    finally:
        _destroy_pg()


def _worker_nonfinite_consensus_experts(rank, world_size, init_method, ckpt_dir, result_dir):
    _nonfinite_consensus_worker(rank, world_size, init_method, ckpt_dir, result_dir, mode="experts")


def _worker_nonfinite_consensus_ddp(rank, world_size, init_method, ckpt_dir, result_dir):
    _nonfinite_consensus_worker(rank, world_size, init_method, ckpt_dir, result_dir, mode="ddp")


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
        sys.executable, os.path.abspath(__file__), "--gloo-worker", worker.__name__,
        str(rank), str(WORLD_SIZE), init_method, str(ckpt_dir), str(result_dir),
    ]


def _run_two_rank(worker, tmp_path, monkeypatch):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available in this build")
    # Keep the subprocess workers CPU-only (fast startup, deterministic RNG keys);
    # monkeypatch restores the parent env after the test.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    if os.name == "nt" and not os.environ.get("GLOO_SOCKET_IFNAME"):
        # Gloo's Windows UV transport expects the adapter alias (the value
        # returned by socket.if_nameindex(), "loopback_0", is not accepted).
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")

    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    # A loopback TCP store keeps this a real 2-rank gloo process group while
    # avoiding the FileStore rendezvous path, which hangs on this host.
    init_method = f"tcp://127.0.0.1:{_free_loopback_port()}?use_libuv=0"
    processes = []
    logs = []
    try:
        for rank in range(WORLD_SIZE):
            log_path = result_dir / f"rank{rank}.subprocess.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            logs.append(log_fh)
            processes.append(subprocess.Popen(
                _worker_command(worker, rank, init_method, ckpt_dir, result_dir),
                stdout=log_fh, stderr=subprocess.STDOUT, env=os.environ.copy(),
            ))

        deadline = time.monotonic() + JOIN_TIMEOUT_S
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, JOIN_TIMEOUT_S)
            process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        _terminate(processes)
        pytest.fail(f"2-rank gloo workers did not finish within {JOIN_TIMEOUT_S}s (possible deadlock/hang)")
    finally:
        for log_fh in logs:
            log_fh.close()

    crashed = {rank: process.returncode for rank, process in enumerate(processes) if process.returncode != 0}
    if crashed:
        details = {rank: (result_dir / f"rank{rank}.subprocess.log").read_text(encoding="utf-8")
                  for rank in crashed}
        pytest.fail(f"distributed worker process failures: {crashed}; {details}")

    results = {}
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank{rank}.txt"
        assert path.exists(), f"rank {rank} exited without writing a result"
        results[rank] = path.read_text(encoding="utf-8")

    skips = [r for r in results.values() if r.startswith("SKIP:")]
    if skips:
        pytest.skip("2-rank gloo init not possible on this host after a real attempt: " + skips[0][:500])
    failures = {k: v for k, v in results.items() if v != "OK"}
    assert not failures, f"worker failures: {failures}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_rank_states_roundtrip_two_rank_gloo(tmp_path, monkeypatch):
    _run_two_rank(_worker_rank_states_roundtrip, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "where", ["rank0_assembly", "rank0_manifest_replace", "rank1_local_state_dict"],
)
def test_publish_failure_propagates_to_all_ranks(tmp_path, monkeypatch, where):
    worker = {
        "rank0_assembly": _worker_publish_failure_rank0_assembly,
        "rank0_manifest_replace": _worker_publish_failure_rank0_manifest_replace,
        "rank1_local_state_dict": _worker_publish_failure_rank1_local_state_dict,
    }[where]
    _run_two_rank(worker, tmp_path, monkeypatch)


@pytest.mark.parametrize("mode", ["experts", "ddp"])
def test_two_rank_nonfinite_consensus_and_next_good_batch(tmp_path, monkeypatch, mode):
    worker = {
        "experts": _worker_nonfinite_consensus_experts,
        "ddp": _worker_nonfinite_consensus_ddp,
    }[mode]
    _run_two_rank(worker, tmp_path, monkeypatch)


if __name__ == "__main__":
    workers = {
        fn.__name__: fn
        for fn in (
            _worker_rank_states_roundtrip,
            _worker_publish_failure_rank0_assembly,
            _worker_publish_failure_rank0_manifest_replace,
            _worker_publish_failure_rank1_local_state_dict,
            _worker_nonfinite_consensus_experts,
            _worker_nonfinite_consensus_ddp,
        )
    }
    if len(sys.argv) != 8 or sys.argv[1] != "--gloo-worker":
        raise SystemExit("invalid gloo worker invocation")
    workers[sys.argv[2]](int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7])

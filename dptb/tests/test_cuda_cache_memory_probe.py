import logging


def test_cuda_cache_memory_probe_is_noop_when_disabled(monkeypatch):
    from dptb.utils import cuda_cache_memory as probe

    probe.configure_cuda_cache_memory_monitor(enabled=False)

    def fail_snapshot(device=None):
        raise AssertionError("snapshot should not be called when probe is disabled")

    monkeypatch.setattr(probe, "snapshot_cuda_memory", fail_snapshot)

    with probe.cuda_cache_memory_probe("cueq_indexed_linear", ("key",), device="cuda:0"):
        pass


def test_cuda_cache_memory_probe_logs_context_and_deltas(monkeypatch, caplog):
    from dptb.utils import cuda_cache_memory as probe

    probe.configure_cuda_cache_memory_monitor(enabled=True, min_delta_mb=0.0)
    snapshots = iter(
        [
            {
                "allocated_mb": 10.0,
                "reserved_mb": 20.0,
                "peak_allocated_mb": 30.0,
                "peak_reserved_mb": 40.0,
                "free_mb": 1000.0,
                "total_mb": 2000.0,
            },
            {
                "allocated_mb": 15.5,
                "reserved_mb": 28.0,
                "peak_allocated_mb": 35.0,
                "peak_reserved_mb": 48.0,
                "free_mb": 990.0,
                "total_mb": 2000.0,
            },
        ]
    )
    monkeypatch.setattr(probe, "snapshot_cuda_memory", lambda device=None: next(snapshots))
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")

    with caplog.at_level(logging.INFO):
        with probe.cuda_cache_memory_context(iteration=42, stage="expert/model_forward", expert=7):
            with probe.cuda_cache_memory_probe(
                "cueq_indexed_linear",
                (16, "torch.float32", "cuda:1"),
                device="cuda:1",
                metadata={"local_entries": 2},
            ):
                pass

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "[CUDA_CACHE_MEMORY]" in messages
    assert "rank=3" in messages
    assert "local_rank=1" in messages
    assert "iter=42" in messages
    assert "stage=expert/model_forward" in messages
    assert "expert=7" in messages
    assert "cache=cueq_indexed_linear" in messages
    assert "allocated_delta_mb=5.5" in messages
    assert "reserved_delta_mb=8.0" in messages
    assert "free_delta_mb=-10.0" in messages

    probe.configure_cuda_cache_memory_monitor(enabled=False)

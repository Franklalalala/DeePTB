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


def test_cuda_cache_event_monitor_tracks_hits_without_cuda_snapshot(monkeypatch, caplog):
    from dptb.utils import cuda_cache_memory as probe

    probe.reset_cuda_cache_event_stats()
    probe.configure_cuda_cache_memory_monitor(
        enabled=False,
        event_enabled=True,
        event_summary_interval=2,
    )

    def fail_snapshot(device=None):
        raise AssertionError("event monitor should not query CUDA memory")

    monkeypatch.setattr(probe, "snapshot_cuda_memory", fail_snapshot)

    with caplog.at_level(logging.INFO):
        with probe.cuda_cache_memory_context(iteration=3, stage="expert/model_forward", expert=1):
            probe.record_cuda_cache_event(
                "cueq_indexed_linear",
                (16, "torch.float32", "cuda:0", 64, 64),
                "miss",
                metadata={"num_graphs": 16, "in_features": 64, "out_features": 64},
            )
            probe.record_cuda_cache_event(
                "cueq_indexed_linear",
                (16, "torch.float32", "cuda:0", 64, 64),
                "hit",
                metadata={"num_graphs": 16, "in_features": 64, "out_features": 64},
            )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "[CUDA_CACHE_EVENT]" in messages
    assert "event=miss" in messages
    assert "summary_total=2" in messages
    assert "summary_hits=1" in messages
    assert "summary_misses=1" in messages
    stats = probe.cuda_cache_event_stats_snapshot()
    assert stats["cueq_indexed_linear|num_graphs=16|in_features=64|out_features=64"]["total"] == 2

    probe.reset_cuda_cache_event_stats()
    probe.configure_cuda_cache_memory_monitor(enabled=False, event_enabled=False)

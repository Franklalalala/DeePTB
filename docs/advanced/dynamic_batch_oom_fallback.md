# Dynamic Batch OOM Fallback Notes

This note documents the current dynamic-batch OOM fallback contract and the
validation pitfalls that came up while hardening it.

## Runtime contract

- Fallback is skip-based. If the current dynamic batch raises CUDA OOM before
  optimizer step starts, the batch is dropped and the iteration returns
  `None`; it is not split and retried.
- The dynamic-batch runtime threshold is shrunk by `oom_shrink_factor`, and the
  cached dynamic-batch iterator is invalidated. The new threshold applies to
  the next loader iterator, not necessarily to already materialized batches.
- Single-process training can use the fallback.
- `distributed_expert` can use the fallback only when
  `expert_data_parallel_size <= 1`. Expert-DP replicas need a rank consensus to
  skip safely; this fallback intentionally does not add such synchronization.
- `distributed_rank0_prepare_batch` disables the fallback.
- Reference batches disable the fallback for that iteration.

The healthy path must not add `torch.cuda.synchronize`, `dist.barrier`, or new
per-iteration collectives. The only distributed special case is an OOM exactly
on a display boundary: the OOM rank joins the existing display-window flush so
peer ranks do not hang at the collectives they were already going to execute.

## Logs

Fallback logs use structured prefixes:

- `[DYNAMIC_BATCH_OOM_SHRINK]` records the old/new max cost and current batch
  metadata.
- `[DYNAMIC_BATCH_OOM_SKIP]` records iteration, rank, location, batch cost,
  `num_graphs`, and skipped counters.

Display-window state may include `dynamic_batch_oom_skipped_iters`.

## Validation traps

Do not catch and swallow a CUDA OOM inside a test helper or fake forward. The
trainer must see the `RuntimeError`/`torch.OutOfMemoryError`; otherwise the
fallback path is never exercised and the validation is a false negative.

For no-sync validation, monkeypatch `torch.cuda.synchronize` to raise before
running the OOM or cache-event path. A passing run then proves the tested path
did not call the Python-level synchronize API. This does not prove that CUDA
kernels are fully asynchronous, but it catches accidental explicit sync calls.

For distributed validation, use a real `torchrun` process group. In the
supported `expert_data_parallel_size <= 1` case, validate that an OOM on one
rank does not hang peers at the display-window collectives. In
`expert_data_parallel_size > 1`, validate that fallback is disabled rather than
trying to skip locally.

## CUEQ `num_graphs` cache events

`cueq_indexed_linear` caches modules by `(num_graphs, dtype, device,
in_features, out_features)`. Dynamic batch can therefore create a cache miss
when it first emits a new batch cardinality, for example `72` then `71`.

Use `monitor_cuda_cache_events=true` to record pure-Python hit/miss counters for
these keys. This event monitor does not query CUDA memory and does not
synchronize. It shows whether cardinality changes create cache churn; it does
not by itself prove that cache misses are the dominant speed bottleneck.

Keep `monitor_cuda_cache_memory_sync=false` unless a diagnostic run explicitly
needs synchronized allocator snapshots.

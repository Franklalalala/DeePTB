"""Lightweight CUDA memory probes for long-lived runtime caches.

The normal training monitor tells us which display window reached a new CUDA
high-water mark. This helper is intentionally narrower: it records before/after
memory only around cache misses that may allocate persistent CUDA state.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from typing import Any, Dict, Iterator, Mapping, Optional


log = logging.getLogger(__name__)

_FALSE_STRINGS = {"", "0", "false", "False", "no", "No", "off", "Off"}
_CONFIG = {
    "enabled": None,
    "sync": None,
    "min_delta_mb": 0.0,
    "event_enabled": None,
    "event_summary_interval": 0,
}
_CONTEXT = threading.local()
_EVENT_LOCK = threading.Lock()
_EVENT_STATS: Dict[str, Dict[str, Any]] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value not in _FALSE_STRINGS


def configure_cuda_cache_memory_monitor(
    *,
    enabled: Optional[bool] = None,
    sync: Optional[bool] = None,
    min_delta_mb: Optional[float] = None,
    event_enabled: Optional[bool] = None,
    event_summary_interval: Optional[int] = None,
) -> None:
    """Configure process-local cache memory probing.

    ``enabled=None`` leaves control to ``DPTB_CUDA_CACHE_MEMORY_DIAG``. This is
    useful for direct scripts, while train_options can still force an explicit
    value by passing ``True`` or ``False``.
    """

    _CONFIG["enabled"] = None if enabled is None else bool(enabled)
    _CONFIG["sync"] = None if sync is None else bool(sync)
    _CONFIG["event_enabled"] = None if event_enabled is None else bool(event_enabled)
    if min_delta_mb is not None:
        _CONFIG["min_delta_mb"] = float(min_delta_mb)
    if event_summary_interval is not None:
        _CONFIG["event_summary_interval"] = max(int(event_summary_interval), 0)


def cuda_cache_memory_monitor_enabled() -> bool:
    configured = _CONFIG.get("enabled")
    if configured is not None:
        return bool(configured)
    return _env_flag("DPTB_CUDA_CACHE_MEMORY_DIAG", False)


def cuda_cache_event_monitor_enabled() -> bool:
    configured = _CONFIG.get("event_enabled")
    if configured is not None:
        return bool(configured)
    return _env_flag("DPTB_CUDA_CACHE_EVENT_DIAG", False)


def _cuda_cache_memory_sync_enabled() -> bool:
    configured = _CONFIG.get("sync")
    if configured is not None:
        return bool(configured)
    return _env_flag("DPTB_CUDA_CACHE_MEMORY_SYNC", False)


def _current_context() -> Dict[str, Any]:
    ctx = getattr(_CONTEXT, "values", None)
    return dict(ctx) if isinstance(ctx, dict) else {}


@contextlib.contextmanager
def cuda_cache_memory_context(**values: Any) -> Iterator[None]:
    previous = _current_context()
    merged = dict(previous)
    for key, value in values.items():
        if value is not None:
            merged[key] = value
    _CONTEXT.values = merged
    try:
        yield
    finally:
        _CONTEXT.values = previous


def _import_torch():
    try:
        import torch  # noqa: WPS433 - optional, lazy runtime import
    except Exception:
        return None
    return torch


def _as_cuda_device(device: Any = None):
    torch = _import_torch()
    if torch is None:
        return None, None
    if device is None:
        if not torch.cuda.is_available():
            return torch, None
        try:
            return torch, torch.device("cuda", torch.cuda.current_device())
        except Exception:
            return torch, None
    try:
        dev = device if isinstance(device, torch.device) else torch.device(device)
    except Exception:
        return torch, None
    if dev.type != "cuda":
        return torch, None
    return torch, dev


def snapshot_cuda_memory(device: Any = None) -> Optional[Dict[str, float]]:
    """Return CUDA allocator and device-free memory in MiB, or ``None``."""

    torch, dev = _as_cuda_device(device)
    if torch is None or dev is None or not torch.cuda.is_available():
        return None
    try:
        if _cuda_cache_memory_sync_enabled():
            torch.cuda.synchronize(dev)
        mb = 1024 ** 2
        free, total = torch.cuda.mem_get_info(dev)
        peak_reserved = (
            torch.cuda.max_memory_reserved(dev)
            if hasattr(torch.cuda, "max_memory_reserved")
            else torch.cuda.memory_reserved(dev)
        )
        return {
            "allocated_mb": torch.cuda.memory_allocated(dev) / mb,
            "reserved_mb": torch.cuda.memory_reserved(dev) / mb,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(dev) / mb,
            "peak_reserved_mb": peak_reserved / mb,
            "free_mb": free / mb,
            "total_mb": total / mb,
        }
    except Exception:
        return None


def _json_value(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))
    except Exception:
        return json.dumps(str(value), separators=(",", ":"))


def _format_float(value: float) -> str:
    return f"{float(value):.1f}"


def reset_cuda_cache_event_stats() -> None:
    with _EVENT_LOCK:
        _EVENT_STATS.clear()


def cuda_cache_event_stats_snapshot() -> Dict[str, Dict[str, Any]]:
    with _EVENT_LOCK:
        return json.loads(json.dumps(_EVENT_STATS, default=str))


def _event_stats_key(cache_name: str, metadata: Optional[Mapping[str, Any]]) -> str:
    if not metadata:
        return str(cache_name)
    parts = [str(cache_name)]
    for key in ("num_graphs", "dtype", "device", "in_features", "out_features"):
        if key in metadata:
            parts.append(f"{key}={metadata[key]}")
    return "|".join(parts)


def record_cuda_cache_event(
    cache_name: str,
    cache_key: Any,
    event: str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, str]]:
    """Record a pure-Python cache hit/miss event without querying CUDA state."""

    if not cuda_cache_event_monitor_enabled():
        return None

    event = str(event)
    metadata = dict(metadata or {})
    row: Dict[str, str] = {
        "rank": os.environ.get("RANK", ""),
        "local_rank": os.environ.get("LOCAL_RANK", ""),
        "iter": "" if _current_context().get("iteration") is None else str(_current_context().get("iteration")),
        "stage": "" if _current_context().get("stage") is None else str(_current_context().get("stage")),
        "expert": "" if _current_context().get("expert") is None else str(_current_context().get("expert")),
        "cache": str(cache_name),
        "event": event,
        "key": _json_value(cache_key),
    }
    for key, value in metadata.items():
        row[f"meta_{key}"] = _json_value(value)

    summary_interval = int(_CONFIG.get("event_summary_interval") or 0)
    should_log = event != "hit"
    summary: Optional[Dict[str, Any]] = None
    with _EVENT_LOCK:
        stats_key = _event_stats_key(str(cache_name), metadata)
        stats = _EVENT_STATS.setdefault(
            stats_key,
            {
                "cache": str(cache_name),
                "total": 0,
                "hits": 0,
                "misses": 0,
                "num_graphs": {},
            },
        )
        stats["total"] += 1
        if event == "hit":
            stats["hits"] += 1
        elif event == "miss":
            stats["misses"] += 1
        if "num_graphs" in metadata:
            ng = str(metadata["num_graphs"])
            stats["num_graphs"][ng] = int(stats["num_graphs"].get(ng, 0)) + 1
        if summary_interval > 0 and stats["total"] % summary_interval == 0:
            should_log = True
            summary = dict(stats)
            summary["num_graphs"] = dict(stats["num_graphs"])

    if summary is not None:
        row["summary_total"] = str(summary["total"])
        row["summary_hits"] = str(summary["hits"])
        row["summary_misses"] = str(summary["misses"])
        row["summary_num_graphs"] = _json_value(summary["num_graphs"])

    if should_log:
        message = " ".join(f"{key}={value}" for key, value in row.items())
        (logger or log).info("[CUDA_CACHE_EVENT] %s", message)
    return row


def _cache_memory_row(
    cache_name: str,
    cache_key: Any,
    before: Mapping[str, float],
    after: Mapping[str, float],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    ctx = _current_context()
    row: Dict[str, str] = {
        "rank": os.environ.get("RANK", ""),
        "local_rank": os.environ.get("LOCAL_RANK", ""),
        "iter": "" if ctx.get("iteration") is None else str(ctx.get("iteration")),
        "stage": "" if ctx.get("stage") is None else str(ctx.get("stage")),
        "expert": "" if ctx.get("expert") is None else str(ctx.get("expert")),
        "cache": str(cache_name),
        "event": "miss",
        "key": _json_value(cache_key),
    }
    for name in (
        "allocated_mb",
        "reserved_mb",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "free_mb",
        "total_mb",
    ):
        b = float(before.get(name, 0.0))
        a = float(after.get(name, 0.0))
        short = name.removesuffix("_mb")
        row[f"{short}_before_mb"] = _format_float(b)
        row[f"{short}_after_mb"] = _format_float(a)
        row[f"{short}_delta_mb"] = _format_float(a - b)
    if metadata:
        for key, value in metadata.items():
            row[f"meta_{key}"] = _json_value(value)
    return row


def _should_log_row(row: Mapping[str, str]) -> bool:
    threshold = float(_CONFIG.get("min_delta_mb") or 0.0)
    if threshold <= 0.0:
        return True
    delta_keys = (
        "allocated_delta_mb",
        "reserved_delta_mb",
        "peak_allocated_delta_mb",
        "peak_reserved_delta_mb",
        "free_delta_mb",
    )
    for key in delta_keys:
        try:
            if abs(float(row.get(key, "0"))) >= threshold:
                return True
        except Exception:
            continue
    return False


def record_cuda_cache_memory_event(
    cache_name: str,
    cache_key: Any,
    before: Mapping[str, float],
    after: Mapping[str, float],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, str]]:
    row = _cache_memory_row(cache_name, cache_key, before, after, metadata=metadata)
    if not _should_log_row(row):
        return None
    message = " ".join(f"{key}={value}" for key, value in row.items())
    (logger or log).info("[CUDA_CACHE_MEMORY] %s", message)
    return row


@contextlib.contextmanager
def cuda_cache_memory_probe(
    cache_name: str,
    cache_key: Any,
    *,
    device: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Iterator[None]:
    """Log CUDA memory deltas for a cache miss block when probing is enabled."""

    if not cuda_cache_memory_monitor_enabled():
        yield
        return

    before = snapshot_cuda_memory(device)
    ok = False
    try:
        yield
        ok = True
    finally:
        if ok and before is not None:
            after = snapshot_cuda_memory(device)
            if after is not None:
                record_cuda_cache_memory_event(
                    cache_name,
                    cache_key,
                    before,
                    after,
                    metadata=metadata,
                    logger=logger,
                )

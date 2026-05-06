import logging
import math
import random
from collections import deque
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Sampler

from dptb.utils.torch_geometric import Batch, Data, Dataset


log = logging.getLogger(__name__)


DYNAMIC_BATCH_PART_KEYS = (
    "graph",
    "node",
    "edge",
    "env",
    "onsitenv",
    "kpoint",
    "eig_band_square",
)


def _data_keys(data: Any):
    if isinstance(data, dict):
        return data.keys()
    keys = getattr(data, "keys", None)
    return keys if isinstance(keys, list) else (keys() if callable(keys) else keys)


def _has_key(data: Any, key: str) -> bool:
    try:
        keys = _data_keys(data)
        if keys is not None:
            return key in keys
    except Exception:
        pass
    return hasattr(data, key)


def _get_value(data: Any, key: str, default=None):
    try:
        if isinstance(data, dict):
            return data.get(key, default)
        if _has_key(data, key):
            return data[key]
    except Exception:
        pass
    return getattr(data, key, default)


def _num_rows(x: Any) -> int:
    if x is None:
        return 0
    if torch.is_tensor(x):
        return int(x.shape[0]) if x.ndim >= 1 else 1
    if isinstance(x, (list, tuple)):
        return len(x)
    shape = getattr(x, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    return 0


def _num_index_items(x: Any) -> int:
    if x is None:
        return 0
    if torch.is_tensor(x):
        if x.ndim >= 2:
            return int(x.shape[-1])
        if x.ndim == 1:
            return int(x.shape[0])
        return 1
    if isinstance(x, (list, tuple)):
        return len(x)
    return 0


def _nested_tensor_cost(x: Any, fn) -> int:
    if x is None:
        return 0
    if torch.is_tensor(x):
        return int(fn(x))
    if isinstance(x, (list, tuple)):
        return int(sum(_nested_tensor_cost(v, fn) for v in x))
    if isinstance(x, dict):
        return int(sum(_nested_tensor_cost(v, fn) for v in x.values()))
    return 0


class AtomicDataCostEstimator:
    """Estimate graph cost before moving a batch to GPU."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "graph": 1.0,
        "node": 1.0,
        "edge": 1.0,
        "env": 1.0,
        "onsitenv": 1.0,
        "kpoint": 0.0,
        "eig_band_square": 0.0,
    }

    def __init__(
        self,
        mode: str = "cost",
        cost_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        valid_modes = {"cost", "node", "edge", "env", "onsitenv"}
        if mode not in valid_modes:
            raise ValueError(f"dynamic_batch.mode must be one of {sorted(valid_modes)}, got {mode!r}")
        self.mode = mode
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if cost_weights:
            self.weights.update({str(k): float(v) for k, v in cost_weights.items()})

    def parts(self, data: Data) -> Dict[str, int]:
        num_nodes = int(getattr(data, "num_nodes", 0) or 0)
        if num_nodes <= 0:
            for key in ("pos", "positions", "atomic_numbers", "atom_type", "atom_types", "x"):
                num_nodes = _num_rows(_get_value(data, key, None))
                if num_nodes > 0:
                    break
        num_nodes = max(int(num_nodes), 1)

        num_edges = int(getattr(data, "num_edges", 0) or 0)
        if num_edges <= 0:
            num_edges = _num_index_items(_get_value(data, "edge_index", None))

        num_env = _num_index_items(_get_value(data, "env_index", None))
        num_onsitenv = _num_index_items(_get_value(data, "onsitenv_index", None))
        num_kpoints = _nested_tensor_cost(
            _get_value(data, "kpoint", None),
            lambda t: t.shape[0] if t.ndim >= 1 else 1,
        )
        eig_band_square = _nested_tensor_cost(
            _get_value(data, "eigenvalue", None),
            lambda t: (t.shape[-2] * (t.shape[-1] ** 2)) if t.ndim >= 2 else t.numel(),
        )

        return {
            "graph": 1,
            "node": int(num_nodes),
            "edge": int(num_edges),
            "env": int(num_env),
            "onsitenv": int(num_onsitenv),
            "kpoint": int(num_kpoints),
            "eig_band_square": int(eig_band_square),
        }

    def __call__(self, data: Data) -> int:
        parts = self.parts(data)
        return self.from_parts(parts)

    def from_parts(self, parts: Dict[str, int]) -> int:
        parts = normalize_cost_parts(parts)
        if self.mode == "node":
            value = parts["node"]
        elif self.mode == "edge":
            value = parts["edge"]
        elif self.mode == "env":
            value = parts["env"]
        elif self.mode == "onsitenv":
            value = parts["onsitenv"]
        else:
            value = sum(self.weights.get(k, 0.0) * v for k, v in parts.items())
        return max(1, int(math.ceil(float(value))))


def normalize_cost_parts(parts: Optional[Dict[str, Any]]) -> Dict[str, int]:
    out = {key: 0 for key in DYNAMIC_BATCH_PART_KEYS}
    out["graph"] = 1
    if not parts:
        return out
    for key, value in dict(parts).items():
        key = str(key)
        if key not in out:
            continue
        try:
            out[key] = int(value)
        except (TypeError, ValueError):
            out[key] = 0
    out["graph"] = max(1, int(out.get("graph", 1)))
    return out


class _IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return int(idx), self.dataset[idx]


def _metadata_cost_parts(
    dataset,
    idx: int,
    estimator: AtomicDataCostEstimator,
    data=None,
    *,
    prefer_loaded_data: bool = False,
):
    if (
        prefer_loaded_data
        and data is not None
        and bool(getattr(dataset, "prefer_loaded_dynamic_batch_cost_parts", False))
    ):
        parts = normalize_cost_parts(estimator.parts(data))
        return estimator.from_parts(parts), parts

    get_parts = getattr(dataset, "get_dynamic_batch_cost_parts", None)
    if callable(get_parts):
        parts = normalize_cost_parts(get_parts(int(idx)))
        return estimator.from_parts(parts), parts

    get_cost = getattr(dataset, "get_dynamic_batch_cost", None)
    if callable(get_cost):
        cost = max(1, int(math.ceil(float(get_cost(int(idx))))))
        parts = normalize_cost_parts(estimator.parts(data) if data is not None else None)
        return cost, parts

    if data is None:
        data = dataset[int(idx)]
    parts = estimator.parts(data)
    parts = normalize_cost_parts(parts)
    return estimator.from_parts(parts), parts


def _attach_dynamic_metadata(
    batch: Batch,
    *,
    sample_indices: Optional[List[int]],
    item_costs: List[int],
    item_parts: List[Dict[str, int]],
    mode: str,
    cost_weights: Dict[str, float],
) -> Batch:
    batch.__dptb_sample_indices__ = list(sample_indices) if sample_indices is not None else None
    batch.__dptb_item_costs__ = [int(v) for v in item_costs]
    batch.__dptb_item_parts__ = [normalize_cost_parts(p) for p in item_parts]
    batch.__dptb_batch_cost__ = int(sum(item_costs))
    batch.__dptb_batch_num_graphs__ = int(len(item_costs))
    batch.__dptb_batch_max_item_cost__ = int(max(item_costs)) if item_costs else 0
    normalized_parts = batch.__dptb_item_parts__
    batch.__dptb_batch_num_nodes__ = int(sum(p.get("node", 0) for p in normalized_parts))
    batch.__dptb_batch_num_edges__ = int(sum(p.get("edge", 0) for p in normalized_parts))
    batch.__dptb_dynamic_batch_mode__ = str(mode)
    batch.__dptb_dynamic_batch_cost_weights__ = dict(cost_weights)
    return batch


def _collate_dynamic_data_list(
    data_list: List[Data],
    *,
    sample_indices: Optional[List[int]],
    cost_estimator: AtomicDataCostEstimator,
    exclude_keys: List[str],
    metadata_dataset=None,
) -> Batch:
    batch = Batch.from_data_list(data_list, exclude_keys=set(exclude_keys)).contiguous()
    if metadata_dataset is not None and sample_indices is not None:
        cost_and_parts = [
            _metadata_cost_parts(
                metadata_dataset,
                idx,
                cost_estimator,
                data=data,
                prefer_loaded_data=True,
            )
            for idx, data in zip(sample_indices, data_list)
        ]
        item_costs = [int(cost) for cost, _parts in cost_and_parts]
        item_parts = [dict(parts) for _cost, parts in cost_and_parts]
    else:
        item_costs = [cost_estimator(data) for data in data_list]
        item_parts = [cost_estimator.parts(data) for data in data_list]
    return _attach_dynamic_metadata(
        batch,
        sample_indices=sample_indices,
        item_costs=item_costs,
        item_parts=item_parts,
        mode=cost_estimator.mode,
        cost_weights=cost_estimator.weights,
    )


def split_batch_for_oom(batch: Batch, exclude_keys: Optional[List[str]] = None):
    """Split a collated dynamic batch into two microbatches for OOM fallback."""
    if getattr(batch, "num_graphs", 0) <= 1:
        raise RuntimeError("Cannot split an OOM batch with one or fewer graphs.")
    data_list = batch.to_data_list()
    mid = max(1, len(data_list) // 2)
    sample_indices = getattr(batch, "__dptb_sample_indices__", None)
    mode = getattr(batch, "__dptb_dynamic_batch_mode__", "cost")
    cost_weights = getattr(batch, "__dptb_dynamic_batch_cost_weights__", None)
    estimator = AtomicDataCostEstimator(mode=mode, cost_weights=cost_weights)
    exclude_keys = exclude_keys or []
    left_indices = sample_indices[:mid] if sample_indices is not None else None
    right_indices = sample_indices[mid:] if sample_indices is not None else None
    item_costs = getattr(batch, "__dptb_item_costs__", None)
    item_parts = getattr(batch, "__dptb_item_parts__", None)
    left = _collate_dynamic_data_list(
        data_list[:mid],
        sample_indices=left_indices,
        cost_estimator=estimator,
        exclude_keys=exclude_keys,
    )
    right = _collate_dynamic_data_list(
        data_list[mid:],
        sample_indices=right_indices,
        cost_estimator=estimator,
        exclude_keys=exclude_keys,
    )
    if item_costs is not None and item_parts is not None:
        _attach_dynamic_metadata(
            left,
            sample_indices=left_indices,
            item_costs=[int(v) for v in item_costs[:mid]],
            item_parts=[dict(p) for p in item_parts[:mid]],
            mode=mode,
            cost_weights=estimator.weights,
        )
        _attach_dynamic_metadata(
            right,
            sample_indices=right_indices,
            item_costs=[int(v) for v in item_costs[mid:]],
            item_parts=[dict(p) for p in item_parts[mid:]],
            mode=mode,
            cost_weights=estimator.weights,
        )
    return left, right


class DynamicCostBatchSampler(Sampler[List[int]]):
    """Yield dataset index batches capped by a DeePTB graph-cost budget."""

    def __init__(
        self,
        dataset,
        *,
        max_cost: int,
        mode: str = "cost",
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        drop_oversized: bool = False,
        bucket_size: int = 0,
        packing_strategy: Optional[str] = None,
        min_samples: Optional[int] = None,
        seed: int = 0,
        cost_weights: Optional[Dict[str, float]] = None,
        rank: int = 0,
        world_size: int = 1,
        num_steps: Optional[int] = None,
    ) -> None:
        if int(max_cost) <= 0:
            raise ValueError(f"dynamic_batch.max_cost must be positive, got {max_cost}")
        if max_samples is not None and int(max_samples) <= 0:
            raise ValueError(f"dynamic_batch.max_samples must be positive, got {max_samples}")
        if int(world_size) <= 0:
            raise ValueError(f"dynamic_batch.world_size must be positive, got {world_size}")
        if int(rank) < 0 or int(rank) >= int(world_size):
            raise ValueError(f"dynamic_batch.rank={rank} outside world_size={world_size}")

        self.dataset = dataset
        self.max_cost = int(max_cost)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.drop_oversized = bool(drop_oversized)
        self.bucket_size = int(bucket_size or 0)
        self.seed = int(seed or 0)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.num_steps = None if num_steps is None else int(num_steps)
        self.cost_estimator = AtomicDataCostEstimator(mode=mode, cost_weights=cost_weights)
        if packing_strategy is None:
            packing_strategy = "random_evict" if self.shuffle and self.max_samples is not None else "sequential"
        packing_strategy = str(packing_strategy)
        if packing_strategy not in {"sequential", "random_refill", "random_evict"}:
            raise ValueError(
                "dynamic_batch.packing_strategy must be 'sequential', 'random_refill', or 'random_evict', "
                f"got {packing_strategy!r}"
            )
        if packing_strategy == "random_refill":
            packing_strategy = "random_evict"
        self.packing_strategy = packing_strategy
        if min_samples is None:
            min_samples = 2 if self.packing_strategy == "random_evict" else 1
        if int(min_samples) <= 0:
            raise ValueError(f"dynamic_batch.min_samples must be positive, got {min_samples}")
        self.min_samples = int(min_samples)
        self.epoch = 0
        self._cost_cache: Dict[int, int] = {}
        self._warned_oversized = set()
        self._cached_batches_key = None
        self._cached_batches: Optional[List[List[int]]] = None
        self._cost_cache_signature = self._cost_signature()
        self.padding_events = deque(maxlen=1024)
        self.last_padding_stats: Optional[Dict[str, Any]] = None
        self.last_packing_stats: Optional[Dict[str, Any]] = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def step_epoch(self, epoch: int) -> None:
        self.set_epoch(epoch)

    def _dataset_cost_version(self):
        dataset_version = getattr(self.dataset, "dynamic_batch_cost_version", None)
        if callable(dataset_version):
            dataset_version = dataset_version()
        return dataset_version

    def _weights_key(self):
        return tuple(sorted((str(k), float(v)) for k, v in self.cost_estimator.weights.items()))

    def _cost_signature(self):
        return (
            self._dataset_cost_version(),
            self.cost_estimator.mode,
            self._weights_key(),
        )

    def _cache_key(self):
        return (
            self.epoch,
            len(self.dataset),
            self.max_cost,
            self.max_samples,
            self.shuffle,
            self.drop_last,
            self.drop_oversized,
            self.bucket_size,
            self.packing_strategy,
            self.min_samples,
            self.seed,
            self.rank,
            self.world_size,
            self.num_steps,
            self._cost_signature(),
        )

    def invalidate_cache(self, *, clear_costs: bool = False) -> None:
        self._cached_batches_key = None
        self._cached_batches = None
        if clear_costs:
            self._cost_cache.clear()
            self._cost_cache_signature = self._cost_signature()

    def _cost(self, idx: int) -> int:
        idx = int(idx)
        cost_signature = self._cost_signature()
        if cost_signature != self._cost_cache_signature:
            self.invalidate_cache(clear_costs=True)
        if idx not in self._cost_cache:
            self._cost_cache[idx] = _metadata_cost_parts(self.dataset, idx, self.cost_estimator)[0]
        return self._cost_cache[idx]

    def _ordered_indices(self) -> List[int]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        return indices

    def _warn_oversized_sample(self, idx: int, cost: int, *, action: str) -> None:
        if idx in self._warned_oversized:
            return
        verb = "drop" if action == "drop" else "yield oversized singleton"
        log.warning(
            "%s dynamic-batch sample idx=%s cost=%s > max_cost=%s",
            verb,
            idx,
            cost,
            self.max_cost,
        )
        self._warned_oversized.add(idx)

    def _record_packing_stats(
        self,
        batches: List[List[int]],
        *,
        strategy: str,
        direct_accepts: int = 0,
        refill_accepts: int = 0,
        evict_accepts: int = 0,
        putbacks: int = 0,
        oversized_dropped: int = 0,
        oversized_singletons: int = 0,
        tail_dropped: int = 0,
    ) -> None:
        batch_sizes = [len(batch) for batch in batches]
        batch_costs = [sum(self._cost(idx) for idx in batch) for batch in batches]
        total_batches = len(batches)
        avg_samples = float(sum(batch_sizes)) / max(total_batches, 1)
        avg_cost = float(sum(batch_costs)) / max(total_batches, 1)
        max_cost = max(batch_costs) if batch_costs else 0
        min_samples = min(batch_sizes) if batch_sizes else 0
        max_samples = max(batch_sizes) if batch_sizes else 0
        stats = {
            "strategy": str(strategy),
            "epoch": int(self.epoch),
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "batches": int(total_batches),
            "avg_samples": avg_samples,
            "min_samples": int(min_samples),
            "max_samples": int(max_samples),
            "avg_cost": avg_cost,
            "max_cost_observed": int(max_cost),
            "max_cost_limit": int(self.max_cost),
            "avg_cost_utilization": avg_cost / max(float(self.max_cost), 1.0),
            "direct_accepts": int(direct_accepts),
            "refill_accepts": int(refill_accepts),
            "evict_accepts": int(evict_accepts),
            "putbacks": int(putbacks),
            "oversized_dropped": int(oversized_dropped),
            "oversized_singletons": int(oversized_singletons),
            "tail_dropped": int(tail_dropped),
        }
        self.last_packing_stats = stats
        log.info(
            "dynamic_batch packing: strategy=%s epoch=%s rank=%s world_size=%s "
            "batches=%s avg_samples=%.3f min_samples=%s max_samples=%s "
            "avg_cost=%.1f max_cost_observed=%s max_cost_limit=%s util=%.3f "
            "direct_accepts=%s refill_accepts=%s evict_accepts=%s putbacks=%s "
            "oversized_dropped=%s oversized_singletons=%s tail_dropped=%s",
            stats["strategy"],
            stats["epoch"],
            stats["rank"],
            stats["world_size"],
            stats["batches"],
            stats["avg_samples"],
            stats["min_samples"],
            stats["max_samples"],
            stats["avg_cost"],
            stats["max_cost_observed"],
            stats["max_cost_limit"],
            stats["avg_cost_utilization"],
            stats["direct_accepts"],
            stats["refill_accepts"],
            stats["evict_accepts"],
            stats["putbacks"],
            stats["oversized_dropped"],
            stats["oversized_singletons"],
            stats["tail_dropped"],
        )

    def _padding_rng(self, reason: str) -> random.Random:
        reason_offset = 1009 if reason == "num_steps" else 2003
        return random.Random(self.seed + self.epoch * 104729 + reason_offset)

    def _pad_batches(self, batches: List[List[int]], target_len: int, *, reason: str) -> List[List[int]]:
        if len(batches) >= target_len or not batches:
            return batches

        original_len = len(batches)
        need = int(target_len) - original_len
        rng = self._padding_rng(reason)
        pad_sources: List[int] = []
        source_indices = list(range(original_len))

        while len(pad_sources) < need:
            shuffled = list(source_indices)
            rng.shuffle(shuffled)
            if original_len > 1 and not pad_sources and shuffled[0] == original_len - 1:
                shuffled.append(shuffled.pop(0))
            pad_sources.extend(shuffled)

        selected_sources = pad_sources[:need]
        for source_idx in selected_sources:
            batches.append(list(batches[source_idx]))

        stats = {
            "reason": str(reason),
            "original_len": int(original_len),
            "target_len": int(target_len),
            "added": int(need),
            "source_indices": list(selected_sources),
            "epoch": int(self.epoch),
            "rank": int(self.rank),
            "world_size": int(self.world_size),
        }
        self.last_padding_stats = stats
        self.padding_events.append(stats)

        source_preview = selected_sources[:16]
        source_suffix = "..." if len(selected_sources) > len(source_preview) else ""
        log.info(
            "dynamic_batch padded %s batches from %s to %s for %s; "
            "epoch=%s rank=%s world_size=%s source_indices=%s%s",
            need,
            original_len,
            target_len,
            reason,
            self.epoch,
            self.rank,
            self.world_size,
            source_preview,
            source_suffix,
        )
        return batches

    def _make_sequential_global_batches(self):
        batches: List[List[int]] = []
        cur: List[int] = []
        cur_cost = 0
        oversized_dropped = 0
        oversized_singletons = 0

        def flush(force_tail: bool = False):
            nonlocal cur, cur_cost
            if not cur:
                return
            if (not self.drop_last) or force_tail:
                batches.append(cur)
            elif self.max_samples is not None and len(cur) >= self.max_samples:
                batches.append(cur)
            cur = []
            cur_cost = 0

        for idx in self._ordered_indices():
            cost = self._cost(idx)
            if cost > self.max_cost:
                flush(force_tail=True)
                if self.drop_oversized:
                    self._warn_oversized_sample(idx, cost, action="drop")
                    oversized_dropped += 1
                    continue
                self._warn_oversized_sample(idx, cost, action="singleton")
                batches.append([idx])
                oversized_singletons += 1
                continue

            hit_cost = bool(cur) and (cur_cost + cost > self.max_cost)
            hit_count = bool(cur) and self.max_samples is not None and len(cur) >= self.max_samples
            if hit_cost or hit_count:
                batches.append(cur)
                cur = []
                cur_cost = 0

            cur.append(idx)
            cur_cost += cost

        flush(force_tail=False)
        return batches, {
            "strategy": "sequential",
            "oversized_dropped": oversized_dropped,
            "oversized_singletons": oversized_singletons,
        }

    def _return_deferred_to_pool(self, pool: deque, deferred: List[int], rng: random.Random) -> None:
        rng.shuffle(deferred)
        for idx in deferred:
            pool.insert(rng.randrange(len(pool) + 1), idx)

    def _append_batch_or_drop_tail(self, batches: List[List[int]], cur: List[int]) -> int:
        if not cur:
            return 0
        if len(cur) < self.min_samples:
            log.info(
                "dynamic_batch drop tail below min_samples: epoch=%s rank=%s size=%s min_samples=%s indices=%s",
                self.epoch,
                self.rank,
                len(cur),
                self.min_samples,
                list(cur),
            )
            return len(cur)
        if self.drop_last and self.max_samples is not None and len(cur) < self.max_samples:
            return len(cur)
        batches.append(list(cur))
        return 0

    def _make_random_evict_global_batches(self):
        if self.max_samples is None:
            return self._make_sequential_global_batches()

        batches: List[List[int]] = []
        pool = deque(self._ordered_indices())
        rng = random.Random(self.seed + self.epoch * 104729 + 7919)
        putbacks = 0
        direct_accepts = 0
        evict_accepts = 0
        oversized_dropped = 0
        oversized_singletons = 0
        tail_dropped = 0

        def fill_from_pool(cur: List[int]) -> int:
            nonlocal oversized_dropped
            cur_cost = sum(self._cost(idx) for idx in cur)
            while pool and len(cur) < self.max_samples:
                idx = pool.popleft()
                cost = self._cost(idx)
                if cost > self.max_cost:
                    if self.drop_oversized:
                        self._warn_oversized_sample(idx, cost, action="drop")
                        oversized_dropped += 1
                        continue
                    if not cur:
                        self._warn_oversized_sample(idx, cost, action="singleton")
                        cur.append(idx)
                        return cost
                    pool.appendleft(idx)
                    break
                cur.append(idx)
                cur_cost += cost
            return cur_cost

        while pool:
            cur: List[int] = []
            deferred: List[int] = []
            cur_cost = fill_from_pool(cur)
            if not cur:
                continue

            direct = cur_cost <= self.max_cost
            while cur_cost > self.max_cost and len(cur) > 1:
                remove_pos = rng.randrange(len(cur))
                removed = cur.pop(remove_pos)
                cur_cost -= self._cost(removed)
                deferred.append(removed)
                putbacks += 1

            if cur_cost > self.max_cost and len(cur) == 1:
                idx = cur[0]
                cost = self._cost(idx)
                if self.drop_oversized:
                    self._warn_oversized_sample(idx, cost, action="drop")
                    oversized_dropped += 1
                    cur = []
                else:
                    self._warn_oversized_sample(idx, cost, action="singleton")
                    oversized_singletons += 1

            if cur:
                before = len(batches)
                tail_dropped += self._append_batch_or_drop_tail(batches, cur)
                if len(batches) > before:
                    if direct:
                        direct_accepts += 1
                    else:
                        evict_accepts += 1

            self._return_deferred_to_pool(pool, deferred, rng)

        return batches, {
            "strategy": "random_evict",
            "direct_accepts": direct_accepts,
            "refill_accepts": 0,
            "evict_accepts": evict_accepts,
            "putbacks": putbacks,
            "oversized_dropped": oversized_dropped,
            "oversized_singletons": oversized_singletons,
            "tail_dropped": tail_dropped,
        }

    def _make_global_batches(self) -> List[List[int]]:
        if self.packing_strategy == "random_evict":
            batches, stats = self._make_random_evict_global_batches()
        else:
            batches, stats = self._make_sequential_global_batches()

        if self.num_steps is not None:
            target_len = self.num_steps * self.world_size
            batches = self._pad_batches(batches, target_len, reason="num_steps")
            batches = batches[:target_len]
        self._record_packing_stats(batches, **stats)
        return batches

    def _shard_for_rank(self, batches: List[List[int]]) -> List[List[int]]:
        if self.world_size == 1:
            return batches
        if not batches:
            return batches
        if self.drop_last:
            keep = (len(batches) // self.world_size) * self.world_size
            batches = batches[:keep]
        else:
            remainder = len(batches) % self.world_size
            if remainder:
                batches = self._pad_batches(
                    batches,
                    len(batches) + (self.world_size - remainder),
                    reason="world_size",
                )
        return batches[self.rank :: self.world_size]

    def _batches_for_epoch(self) -> List[List[int]]:
        key = self._cache_key()
        if self._cached_batches_key != key or self._cached_batches is None:
            self._cached_batches = self._shard_for_rank(self._make_global_batches())
            self._cached_batches_key = key
        return self._cached_batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._batches_for_epoch():
            yield list(batch)

    def __len__(self) -> int:
        return len(self._batches_for_epoch())


def resolve_dynamic_batch_options(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    dynamic_batch: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not dynamic_batch or not dynamic_batch.get("enabled", False):
        return None

    opts = dict(dynamic_batch)
    if opts.get("max_edge", None) is not None:
        opts["max_cost"] = opts["max_edge"]
        opts["mode"] = "edge"
        opts["cost_weights"] = None
    else:
        opts.setdefault("mode", "edge")
        opts.setdefault("cost_weights", None)
        if opts.get("mode") == "edge":
            opts["cost_weights"] = None
    opts.setdefault("max_samples", batch_size)
    opts.setdefault("shuffle", shuffle)
    opts.setdefault("drop_last", False)
    opts.setdefault("drop_oversized", False)
    opts.setdefault("bucket_size", 0)
    opts.setdefault(
        "packing_strategy",
        "random_evict" if bool(opts.get("shuffle", shuffle)) and opts.get("max_samples") is not None else "sequential",
    )
    if opts.get("packing_strategy") == "random_refill":
        opts["packing_strategy"] = "random_evict"
    opts.setdefault("min_samples", 2 if opts.get("packing_strategy") == "random_evict" else 1)
    if opts.get("seed", None) is None:
        opts["seed"] = 0
    opts.setdefault("calibrated", False)

    if opts.get("max_cost") is None:
        if not opts.get("calibrate", False):
            raise ValueError("dynamic_batch is enabled but max_cost is not set and calibrate is false.")
        estimator = AtomicDataCostEstimator(
            mode=opts.get("mode", "cost"),
            cost_weights=opts.get("cost_weights", None),
        )
        indices = list(range(len(dataset)))
        if bool(shuffle):
            generator = torch.Generator()
            generator.manual_seed(int(opts.get("seed", 0)))
            indices = torch.randperm(len(dataset), generator=generator).tolist()
        max_batches = int(opts.get("calibration_batches", 1000))
        batch_costs: List[int] = []
        fixed_batch_size = max(int(batch_size), 1)
        for start in range(0, len(indices), fixed_batch_size):
            if len(batch_costs) >= max_batches:
                break
            batch_indices = indices[start: start + fixed_batch_size]
            batch_costs.append(
                int(sum(_metadata_cost_parts(dataset, idx, estimator)[0] for idx in batch_indices))
            )
        if not batch_costs:
            raise ValueError("dynamic_batch calibration found no batches.")
        q = float(opts.get("calibration_quantile", 0.95))
        q = min(max(q, 0.0), 1.0)
        quantile = torch.quantile(torch.tensor(batch_costs, dtype=torch.float32), q)
        opts["max_cost"] = int(math.ceil(float(quantile.item())))
        if opts.get("mode") == "edge":
            opts["max_edge"] = opts["max_cost"]
        opts["calibrated"] = True
        opts["calibration_batch_costs"] = batch_costs
        log.info(
            "dynamic_batch calibration: mode=%s batches=%s quantile=%.3f max_cost=%s max_edge=%s max_samples=%s",
            opts.get("mode"),
            len(batch_costs),
            q,
            opts["max_cost"],
            opts.get("max_edge"),
            opts.get("max_samples"),
        )

    opts["max_cost"] = int(opts["max_cost"])
    if opts.get("mode") == "edge":
        opts["max_edge"] = int(opts.get("max_edge", opts["max_cost"]))
        opts["cost_weights"] = None
    opts["max_samples"] = None if opts.get("max_samples") is None else int(opts["max_samples"])
    opts["min_samples"] = int(opts.get("min_samples", 1))
    return opts


class Collater(object):
    """Collate a list of ``AtomicData``.

    Args:
        exclude_keys: keys to ignore in the input, not copying to the output
    """

    def __init__(
        self,
        dataset=None,
        exclude_keys: List[str] = [],
        cost_estimator: Optional[AtomicDataCostEstimator] = None,
    ):
        self.dataset = dataset
        self._exclude_keys = set(exclude_keys)
        self._cost_estimator = cost_estimator

    @classmethod
    def for_dataset(
        cls,
        dataset,
        exclude_keys: List[str] = [],
        cost_estimator: Optional[AtomicDataCostEstimator] = None,
    ):
        """Construct a collater appropriate to ``dataset``."""
        return cls(
            dataset=dataset,
            exclude_keys=exclude_keys,
            cost_estimator=cost_estimator,
        )

    def collate(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        sample_indices = None
        if batch and isinstance(batch[0], tuple) and len(batch[0]) == 2:
            sample_indices = [int(item[0]) for item in batch]
            batch = [item[1] for item in batch]
        if self._cost_estimator is not None:
            return _collate_dynamic_data_list(
                batch,
                sample_indices=sample_indices,
                cost_estimator=self._cost_estimator,
                exclude_keys=self.exclude_keys,
                metadata_dataset=getattr(self.dataset, "dataset", self.dataset),
            )
        return Batch.from_data_list(batch, exclude_keys=self._exclude_keys)

    def __call__(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        return self.collate(batch)

    @property
    def exclude_keys(self):
        return list(self._exclude_keys)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        exclude_keys: List[str] = [],
        dynamic_batch: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        resolved_dynamic_batch = resolve_dynamic_batch_options(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            dynamic_batch=dynamic_batch,
        )
        if resolved_dynamic_batch is not None:
            if kwargs.get("sampler", None) is not None:
                raise ValueError("dynamic_batch is incompatible with an explicit sampler.")
            if kwargs.get("batch_sampler", None) is not None:
                raise ValueError("dynamic_batch is incompatible with an explicit batch_sampler.")
            kwargs.pop("sampler", None)
            kwargs.pop("batch_sampler", None)
            kwargs.pop("drop_last", None)

            rank = int(resolved_dynamic_batch.get("rank", 0))
            world_size = int(resolved_dynamic_batch.get("world_size", 1))
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = int(resolved_dynamic_batch.get("rank", torch.distributed.get_rank()))
                world_size = int(resolved_dynamic_batch.get("world_size", torch.distributed.get_world_size()))

            batch_sampler = DynamicCostBatchSampler(
                dataset,
                max_cost=resolved_dynamic_batch["max_cost"],
                mode=resolved_dynamic_batch.get("mode", "cost"),
                max_samples=resolved_dynamic_batch.get("max_samples", batch_size),
                shuffle=resolved_dynamic_batch.get("shuffle", shuffle),
                drop_last=resolved_dynamic_batch.get("drop_last", False),
                drop_oversized=resolved_dynamic_batch.get("drop_oversized", False),
                bucket_size=resolved_dynamic_batch.get("bucket_size", 0),
                packing_strategy=resolved_dynamic_batch.get("packing_strategy", None),
                min_samples=resolved_dynamic_batch.get("min_samples", None),
                seed=resolved_dynamic_batch.get("seed", 0),
                cost_weights=resolved_dynamic_batch.get("cost_weights", None),
                rank=rank,
                world_size=world_size,
                num_steps=resolved_dynamic_batch.get("num_steps", None),
            )
            indexed_dataset = _IndexedDataset(dataset)
            super(DataLoader, self).__init__(
                indexed_dataset,
                batch_sampler=batch_sampler,
                collate_fn=Collater.for_dataset(
                    indexed_dataset,
                    exclude_keys=exclude_keys,
                    cost_estimator=batch_sampler.cost_estimator,
                ),
                **kwargs,
            )
            self.dynamic_batch_options = resolved_dynamic_batch
            self.dynamic_batch_sampler = batch_sampler
            return

        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater.for_dataset(dataset, exclude_keys=exclude_keys),
            **kwargs,
        )
        self.dynamic_batch_options = None
        self.dynamic_batch_sampler = None

    def invalidate_dynamic_batch_cache(self, *, clear_costs: bool = False) -> None:
        sampler = getattr(self, "dynamic_batch_sampler", None)
        if sampler is not None and hasattr(sampler, "invalidate_cache"):
            sampler.invalidate_cache(clear_costs=clear_costs)


class PartialSampler(Sampler[int]):
    r"""Samples elements without replacement, but divided across a number of calls to `__iter__`.

    To ensure deterministic reproducibility and restartability, dataset permutations are generated
    from a combination of the overall seed and the epoch number. As a result, the caller must
    tell this sampler the epoch number before each time `__iter__` is called by calling
    `my_partial_sampler.step_epoch(epoch_number_about_to_run)` each time.

    This sampler decouples epochs from the dataset size and cycles through the dataset over as
    many (partial) epochs as it may take. As a result, the _dataset_ epoch can change partway
    through a training epoch.

    Args:
        data_source (Dataset): dataset to sample from
        shuffle (bool): whether to shuffle the dataset each time the _entire_ dataset is consumed
        num_samples_per_epoch (int): number of samples to draw in each call to `__iter__`.
            If `None`, defaults to `len(data_source)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Dataset
    num_samples_per_epoch: int
    shuffle: bool
    _epoch: int
    _prev_epoch: int

    def __init__(
        self,
        data_source: Dataset,
        shuffle: bool = True,
        num_samples_per_epoch: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.shuffle = shuffle
        if num_samples_per_epoch is None:
            num_samples_per_epoch = self.num_samples_total
        self.num_samples_per_epoch = num_samples_per_epoch
        assert self.num_samples_per_epoch <= self.num_samples_total
        assert self.num_samples_per_epoch >= 1
        self.generator = generator
        self._epoch = None
        self._prev_epoch = None

    @property
    def num_samples_total(self) -> int:
        # dataset size might change at runtime
        return len(self.data_source)

    def step_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        assert self._epoch is not None
        assert (self._prev_epoch is None) or (self._epoch == self._prev_epoch + 1)
        assert self._epoch >= 0

        full_epoch_i, start_sample_i = divmod(
            # how much data we've already consumed:
            self._epoch * self.num_samples_per_epoch,
            # how much data there is the dataset:
            self.num_samples_total,
        )

        if self.shuffle:
            temp_rng = torch.Generator()
            # Get new randomness for each _full_ time through the dataset
            # This is deterministic w.r.t. the combination of dataset seed and epoch number
            # Both of which persist across restarts
            # (initial_seed() is restored by set_state())
            temp_rng.manual_seed(self.generator.initial_seed() + full_epoch_i)
            full_order_this = torch.randperm(self.num_samples_total, generator=temp_rng)
            # reseed the generator for the _next_ epoch to get the shuffled order of the
            # _next_ dataset epoch to pad out this one for completing any partial batches
            # at the end:
            temp_rng.manual_seed(self.generator.initial_seed() + full_epoch_i + 1)
            full_order_next = torch.randperm(self.num_samples_total, generator=temp_rng)
            del temp_rng
        else:
            full_order_this = torch.arange(self.num_samples_total)
            # without shuffling, the next epoch has the same sampling order as this one:
            full_order_next = full_order_this

        full_order = torch.cat((full_order_this, full_order_next), dim=0)
        del full_order_next, full_order_this

        this_segment_indexes = full_order[
            start_sample_i : start_sample_i + self.num_samples_per_epoch
        ]
        # because we cycle into indexes from the next dataset epoch,
        # we should _always_ be able to get num_samples_per_epoch
        assert len(this_segment_indexes) == self.num_samples_per_epoch
        yield from this_segment_indexes

        self._prev_epoch = self._epoch

    def __len__(self) -> int:
        return self.num_samples_per_epoch

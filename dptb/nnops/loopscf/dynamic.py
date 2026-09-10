"""Graph-cost packing and transactional forward/backward OOM retries.

Optimizer commits deliberately live outside this module: an optimizer OOM can
partially mutate its parameters and moments and must not be blindly retried.
"""
from dataclasses import dataclass, asdict
from collections import deque
import gc
import math
import torch


def is_cuda_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and any(s in str(error).lower() for s in
        ('cuda out of memory', 'cuda error: out of memory', 'cuda_error_out_of_memory'))
    )


@dataclass
class CostController:
    budget: float
    max_graphs: int = 16
    target_bytes: int = 68 * 1024**3
    cooldown: int = 0
    oom_count: int = 0

    def pack(self, costs):
        groups, group, cost = [], [], 0
        for i, value in enumerate(costs):
            if group and (len(group) >= self.max_graphs or cost + value > self.budget):
                groups.append(group)
                group, cost = [], 0
            group.append(i)
            cost += value
        if group:
            groups.append(group)
        return groups

    def shrink(self, failed_cost, failed_size):
        self.oom_count += 1
        if failed_size == 1:
            raise RuntimeError('A single structure cannot fit; sample is not skipped')
        # Positive per-graph costs ensure the failing group is split next time.
        self.budget = max(1, min(self.budget * .65, failed_cost * .65))
        self.cooldown = 5

    def observe(self, used_cost, peak_bytes, static_bytes):
        if peak_bytes <= 0 or used_cost <= 0:
            return
        ratio = max(1, self.target_bytes - static_bytes) / max(1, peak_bytes - static_bytes)
        if ratio < 1:
            self.budget = max(1, min(self.budget, used_cost * max(.65, .92 * ratio)))
            self.cooldown = 5
        elif self.cooldown:
            self.cooldown -= 1
        else:
            # Grow from observed cost, never invent large headroom from tiny tails.
            self.budget = max(self.budget, min(self.budget * 1.15, used_cost * .95 * ratio))

    def state_dict(self):
        return asdict(self)


def edge_budget_batches(loader, controller, cost_fn, initial_count=4, initialize=False):
    """Pack consecutive CPU records first; send an accepted batch directly to GPU.

The loader may prefetch fixed-size CPU chunks. Those chunks are not optimizer
batches. One lookahead record stays pending when the edge budget is reached;
after each yield the next pack observes the controller's updated budget.
"""
    source = iter(item for chunk in loader for item in chunk)
    pending = deque()
    if initialize:
        for _ in range(initial_count):
            try:
                item = next(source)
            except StopIteration:
                break
            pending.append((item, cost_fn(item)))
        if pending:
            controller.budget = sum(cost for _, cost in pending)
    exhausted = False
    while True:
        items, cost = [], 0
        while len(items) < controller.max_graphs:
            if not pending and not exhausted:
                try:
                    item = next(source)
                    pending.append((item, cost_fn(item)))
                except StopIteration:
                    exhausted = True
            if not pending:
                break
            item, value = pending[0]
            if items and cost + value > controller.budget:
                break
            pending.popleft()
            items.append(item)
            cost += value
        if not items:
            return
        yield items


def _attempt(fn, indices, weight):
    try:
        return fn(indices, weight), None
    except Exception as error:
        if not is_cuda_oom(error):
            raise
        # Return text, not the exception/traceback which retains CUDA tensors.
        return None, str(error)


def backward_with_retry(costs, controller, micro_backward, zero_grad,
                        capture_state, restore_state, cleanup=None, on_retry=None):
    """Accumulate one graph-mean objective, retrying the same logical batch.

``micro_backward(indices, weight)`` must finish backward before returning only
CPU diagnostics. It must never step the optimizer. All accumulated gradients
and mutable forward state are rolled back before a failed logical attempt.
"""
    if not costs or any(c < 1 or not math.isfinite(c) for c in costs):
        raise ValueError('positive finite graph costs of at least one edge required')
    state = capture_state()
    attempts = 0
    while True:
        zero_grad()
        restore_state(state)
        groups = controller.pack(costs)
        results = []
        failure = None
        for indices in groups:
            result, error = _attempt(micro_backward, indices, len(indices) / len(costs))
            if error is not None:
                failure = (indices, error)
                break
            results.append(result)
        if failure is None:
            return results, groups, attempts
        indices, message = failure
        results.clear()
        zero_grad()
        if cleanup is not None:
            cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        attempts += 1
        if on_retry is not None:
            on_retry(indices, message, attempts)
        if attempts > 20:
            raise RuntimeError('OOM retry limit exceeded; logical batch uncommitted')
        controller.shrink(sum(costs[i] for i in indices), len(indices))

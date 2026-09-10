"""CPU checks for ordered packing and transactional OOM gradient retries."""

import pytest
import torch
from dptb.nnops.loopscf.dynamic import edge_budget_batches

from dptb.nnops.loopscf.dynamic import CostController, backward_with_retry


def test_cost_packing_preserves_every_graph_and_order():
    costs = [2, 3, 9, 1, 1, 1, 4, 2]
    controller = CostController(budget=6, max_graphs=2)
    groups = controller.pack(costs)
    assert [i for group in groups for i in group] == list(range(len(costs)))
    assert all(len(group) <= 2 for group in groups)
    assert all(len(group) == 1 or sum(costs[i] for i in group) <= 6
               for group in groups)
    assert [2] in groups  # An oversized structure is attempted, never dropped.
    assert controller.pack([]) == []


def test_controller_shrinks_and_waits_before_bounded_growth():
    controller = CostController(budget=100, target_bytes=1000)
    controller.shrink(failed_cost=80, failed_size=4)
    assert 0 < controller.budget < 80
    assert controller.oom_count == 1
    after_oom = controller.budget
    for _ in range(5):
        controller.observe(used_cost=after_oom, peak_bytes=300, static_bytes=100)
        assert controller.budget == after_oom
    controller.observe(used_cost=after_oom, peak_bytes=300, static_bytes=100)
    assert after_oom < controller.budget <= after_oom * 1.15

    before_pressure = controller.budget
    controller.observe(used_cost=before_pressure, peak_bytes=1400, static_bytes=100)
    assert 0 < controller.budget < before_pressure
    assert controller.cooldown > 0

    # A tiny last microbatch should not inflate the next batch budget.
    controller.cooldown = 0
    before_tail = controller.budget
    controller.observe(used_cost=1, peak_bytes=300, static_bytes=100)
    assert controller.budget == before_tail
    restored = CostController(**controller.state_dict())
    assert restored.state_dict() == controller.state_dict()


def test_partial_backward_oom_replays_batch_and_matches_unsplit_gradient(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    x = torch.arange(1, 10, dtype=torch.float64)
    target = torch.linspace(-0.3, 1.7, 9, dtype=torch.float64)
    parameter = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    reference = parameter.detach().clone().requires_grad_()
    ((reference * x - target).square().mean()).backward()
    controller = CostController(budget=40, max_graphs=4)
    state = {"seen": []}
    calls, restores, retries, cleanups = [], [], [], []
    failed = False
    zero_calls = 0

    def zero_grad():
        nonlocal zero_calls
        zero_calls += 1
        parameter.grad = None

    def restore(snapshot):
        state["seen"] = list(snapshot)
        restores.append(list(state["seen"]))

    def micro_backward(indices, weight):
        nonlocal failed
        calls.append(list(indices))
        state["seen"].extend(indices)
        loss = (parameter * x[indices] - target[indices]).square().mean()
        (weight * loss).backward()
        if not failed and indices[0] == 4:
            # Simulate failure after both a prior microbatch and this one's
            # backward have left gradients and mutated forward state behind.
            assert parameter.grad is not None and parameter.grad.abs() > 0
            failed = True
            raise torch.cuda.OutOfMemoryError("injected CUDA allocation failure")
        return float(loss.detach())

    results, groups, attempts = backward_with_retry(
        [10] * len(x), controller, micro_backward, zero_grad,
        capture_state=lambda: list(state["seen"]), restore_state=restore,
        cleanup=lambda: cleanups.append(True),
        on_retry=lambda indices, message, count: retries.append((list(indices), count)),
    )
    assert calls[:2] == [list(range(4)), list(range(4, 8))]
    assert calls[2][0] == 0  # Restart the whole logical batch, not just failed part.
    assert attempts == 1 and retries == [(list(range(4, 8)), 1)]
    assert cleanups == [True] and zero_calls >= 3
    assert restores == [[], []]
    assert state["seen"] == list(range(len(x)))
    assert [i for group in groups for i in group] == list(range(len(x)))
    assert len({len(group) for group in groups}) > 1  # Weighted unequal tails.
    assert len(results) == len(groups)
    torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-12, rtol=1e-12)
    assert parameter.item() == 0.7  # Retry helper never commits an optimizer step.


def test_non_oom_runtime_error_propagates_without_retry(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    controller = CostController(budget=10)
    calls = []

    def broken(indices, weight):
        calls.append(list(indices))
        raise RuntimeError("CUDA device-side assert triggered")

    with pytest.raises(RuntimeError, match="device-side assert"):
        backward_with_retry([2, 2], controller, broken, lambda: None,
                            lambda: None, lambda state: None)
    assert calls == [[0, 1]]
    assert controller.oom_count == 0


def test_singleton_oom_fails_without_skipping_structure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    calls = []

    def zero_grad():
        parameter.grad = None

    def broken(indices, weight):
        calls.append(list(indices))
        (parameter.square() * weight).backward()
        raise torch.cuda.OutOfMemoryError("injected singleton OOM")

    with pytest.raises(RuntimeError, match="single structure.*not skipped"):
        backward_with_retry([100, 2], CostController(budget=10), broken,
                            zero_grad, lambda: None, lambda state: None)
    assert calls == [[0]]
    assert parameter.grad is None
    assert parameter.item() == 1.0


@pytest.mark.parametrize("costs", [[], [0], [0.5], [-1], [float("nan")], [float("inf")]])
def test_invalid_costs_are_rejected_before_forward(costs):
    def unexpected(*args):
        pytest.fail("Invalid costs must be rejected before invoking callbacks")

    with pytest.raises(ValueError, match="positive finite"):
        backward_with_retry(costs, CostController(budget=10), unexpected,
                            unexpected, unexpected, unexpected)


def test_edge_packer_crosses_cpu_chunk_boundaries_and_observes_new_budget():
    controller = CostController(budget=7, max_graphs=4)
    batches = edge_budget_batches([[1, 2, 3], [4, 5, 6]], controller, lambda x: x)
    assert next(batches) == [1, 2, 3]
    controller.budget = 15
    assert next(batches) == [4, 5, 6]
    assert list(batches) == []


def test_edge_packer_initial_budget_and_oversized_singleton_keep_every_record():
    controller = CostController(budget=1, max_graphs=3)
    batches = list(edge_budget_batches([[2, 3, 20, 1, 1]], controller,
                                      lambda x: x, initial_count=2, initialize=True))
    assert controller.budget == 5
    assert batches == [[2, 3], [20], [1, 1]]

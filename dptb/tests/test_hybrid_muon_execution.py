"""Execution modes preserve routing, per-parameter state and checkpoint reuse."""
import copy

import pytest
import torch

from dptb.utils.dpa4_optim import HybridMuon


def _parameters():
    shapes = [(4, 8), (4, 8), (8, 4), (2, 4, 8), (1, 4, 8), (32,), (7,), (4, 8)]
    names = ["weight", "weight2", "tall.weight", "core_experts", "singleton.weight",
             "flat.weight", "bias", "router.weight"]
    generator = torch.Generator().manual_seed(11)
    return [(name, torch.nn.Parameter(torch.randn(shape, generator=generator)))
            for name, shape in zip(names, shapes)]


def _assert_state_equal(left, right, exact=True):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], dict):
            _assert_state_equal(left[key], right[key], exact)
        elif torch.is_tensor(left[key]):
            if exact:
                assert torch.equal(left[key], right[key]), key
            else:
                torch.testing.assert_close(left[key], right[key], rtol=3e-5, atol=2e-6)
        else:
            assert left[key] == right[key], key


def test_cached_execution_matches_legacy_bitwise_and_restores_state():
    left = _parameters()
    right = [(name, torch.nn.Parameter(param.detach().clone())) for name, param in left]
    options = dict(lr=0.01, adamw_name_patterns=("*router*",), expert_update_scale="const",
                   expert_update_scale_const=0.4, expert_weight_decay_mult=0.5)
    reference = HybridMuon(left, **options)
    reference.execution_mode = "legacy"
    cached = HybridMuon(right, **options)
    cached.execution_mode = "cached"
    generator = torch.Generator().manual_seed(12)
    for step in range(20):
        for (_, a), (_, b) in zip(left, right):
            grad = torch.randn(a.shape, generator=generator)
            a.grad = grad
            b.grad = grad.clone()
        reference.step()
        cached.step()
        if step == 8:
            checkpoint = copy.deepcopy(reference.state_dict())
            reference.load_state_dict(copy.deepcopy(checkpoint))
            cached.load_state_dict(checkpoint)
        for (_, a), (_, b) in zip(left, right):
            assert torch.equal(a, b)
        _assert_state_equal(reference.state_dict()["state"], cached.state_dict()["state"])
        assert reference.get_diagnostics() == cached.get_diagnostics()
    assert reference.state_dict()["param_groups"] == cached.state_dict()["param_groups"]


def test_cached_routes_follow_group_parameter_and_shape_changes():
    params = _parameters()
    optimizer = HybridMuon(params)
    assert optimizer.route_counts == {"muon": 7, "adam": 1}
    group = optimizer.param_groups[0]
    group["adamw_name_patterns"] = ["*router*"]
    assert optimizer.route_counts == {"muon": 6, "adam": 2}
    group["muon_1d_route_mode"] = "off"
    assert optimizer.route_counts == {"muon": 5, "adam": 3}
    group["params"][0].data = torch.ones(7)
    assert optimizer.route_counts == {"muon": 4, "adam": 4}
    group["params"].pop()
    assert optimizer.route_counts == {"muon": 4, "adam": 3}
    optimizer.add_param_group({"params": [("new.weight", torch.nn.Parameter(torch.ones(16)))]})
    assert optimizer.route_counts == {"muon": 5, "adam": 3}


def test_sparse_gradient_is_rejected_and_closure_remains_differentiable():
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = HybridMuon([param])

    def closure():
        optimizer.zero_grad()
        loss = param.square().sum()
        loss.backward()
        return loss

    assert optimizer.step(closure).item() == 4.0
    param.grad = torch.sparse_coo_tensor([[0]], [1.0], (4,))
    with pytest.raises(RuntimeError, match="sparse"):
        optimizer.step()


@pytest.mark.parametrize("magma_lite", [False, True])
def test_batched_execution_preserves_per_parameter_states_and_diagnostics(magma_lite):
    left = _parameters()
    right = [(name, torch.nn.Parameter(param.detach().clone())) for name, param in left]
    options = dict(lr=0.01, magma_lite=magma_lite, adamw_name_patterns=("*router*",),
                   expert_update_scale="const", expert_update_scale_const=0.4)
    reference = HybridMuon(left, **options)
    reference.execution_mode = "legacy"
    batched = HybridMuon(right, **options)
    batched.execution_mode = "batched"
    generator = torch.Generator().manual_seed(13)
    for step in range(20):
        for index, ((_, a), (_, b)) in enumerate(zip(left, right)):
            grad = torch.randn(a.shape, generator=generator)
            if index == 3:
                grad[1].zero_()
            a.grad = None if step == 4 and index == 1 else grad
            b.grad = None if a.grad is None else grad.clone()
        reference.step()
        batched.step()
        for (_, a), (_, b) in zip(left, right):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
        _assert_state_equal(reference.state_dict()["state"], batched.state_dict()["state"], exact=False)
        a, b = reference.get_diagnostics(), batched.get_diagnostics()
        assert a.keys() == b.keys()
        for key in a:
            assert a[key] == pytest.approx(b[key], rel=3e-5, abs=2e-6), key

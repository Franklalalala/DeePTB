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
@pytest.mark.parametrize("execution_mode", ["batched", "foreach", "fast"])
def test_batched_execution_preserves_per_parameter_states_and_diagnostics(magma_lite, execution_mode):
    left = _parameters()
    right = [(name, torch.nn.Parameter(param.detach().clone())) for name, param in left]
    options = dict(lr=0.01, magma_lite=magma_lite, adamw_name_patterns=("*router*",),
                   expert_update_scale="const", expert_update_scale_const=0.4)
    reference = HybridMuon(left, **options)
    reference.execution_mode = "legacy"
    batched = HybridMuon(right, **options)
    batched.execution_mode = execution_mode
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


@pytest.mark.parametrize("duplicate", [False, True])
def test_batched_modes_preserve_aliased_parameter_updates(duplicate):
    a = torch.nn.Parameter(torch.arange(32.0).reshape(4, 8) / 32)
    b = torch.nn.Parameter(a.detach().clone())
    left = [a, a if duplicate else torch.nn.Parameter(a.detach())]
    right = [b, b if duplicate else torch.nn.Parameter(b.detach())]
    if duplicate:
        with pytest.warns(UserWarning, match="duplicate"):
            reference = HybridMuon(left)
        with pytest.warns(UserWarning, match="duplicate"):
            batched = HybridMuon(right)
    else:
        reference, batched = HybridMuon(left), HybridMuon(right)
    reference.execution_mode = "legacy"
    batched.execution_mode = "foreach"
    for p, q in zip(left, right):
        p.grad = torch.ones_like(p)
        q.grad = p.grad.clone()
    reference.step()
    batched.step()
    assert torch.equal(a, b)
    _assert_state_equal(reference.state_dict()["state"], batched.state_dict()["state"])


@pytest.mark.parametrize("execution_mode", ["batched", "foreach", "fast"])
@pytest.mark.parametrize("inactive_second", [False, True])
def test_batched_modes_preserve_gradients_aliasing_parameter_storage(execution_mode, inactive_second):
    initial = [torch.tensor([[1.0, 2.0], [3.0, 5.0]]), torch.tensor([[2.0, -1.0], [4.0, 3.0]])]
    left = [torch.nn.Parameter(value.clone()) for value in initial]
    right = [torch.nn.Parameter(value.clone()) for value in initial]
    options = dict(lr=0.1, weight_decay=0.1, magma_lite=False, muon_clip=False)
    reference, batched = HybridMuon(left, **options), HybridMuon(right, **options)
    reference.execution_mode = "legacy"
    batched.execution_mode = execution_mode
    for params in (left, right):
        params[0].grad = params[1].detach()
        params[1].grad = None if inactive_second else params[0].detach()
    for step in range(2):
        reference.step()
        batched.step()
        for a, b in zip(left, right):
            assert torch.equal(a, b)
        _assert_state_equal(reference.state_dict()["state"], batched.state_dict()["state"])
        assert reference.get_diagnostics() == batched.get_diagnostics()
        if step == 0:
            # Replacing aliasing gradients must make batching safe again.
            for a, b in zip(left, right):
                a.grad = torch.ones_like(a)
                b.grad = a.grad.clone()


def test_foreach_execution_supports_mixed_parameter_dtypes_and_group_options():
    left = _parameters()
    left[0][1].data = left[0][1].double()
    right = [(name, torch.nn.Parameter(param.detach().clone())) for name, param in left]
    reference, fast = HybridMuon(left), HybridMuon(right)
    reference.execution_mode = "legacy"
    fast.execution_mode = "foreach"
    for optimizer in (reference, fast):
        optimizer.param_groups[0].update(muon_clip_mode="fixed", magma_temperature=0.5)
    for (_, a), (_, b) in zip(left, right):
        a.grad = torch.ones_like(a)
        b.grad = a.grad.clone()
    reference.step()
    fast.step()
    for (_, a), (_, b) in zip(left, right):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
    _assert_state_equal(reference.state_dict()["state"], fast.state_dict()["state"], exact=False)


@pytest.mark.parametrize("temperature", [1e-20, 0.5, 2.0, 4.0])
def test_fast_magma_constants_match_tensor_formula(temperature):
    left = _parameters()
    right = [(name, torch.nn.Parameter(param.detach().clone())) for name, param in left]
    reference = HybridMuon(left, magma_temperature=temperature)
    fast = HybridMuon(right, magma_temperature=temperature)
    reference.execution_mode = "legacy"
    fast.execution_mode = "fast"
    for (_, a), (_, b) in zip(left, right):
        a.grad = torch.ones_like(a)
        b.grad = a.grad.clone()
    reference.step()
    fast.step()
    for (_, a), (_, b) in zip(left, right):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
    _assert_state_equal(reference.state_dict()["state"], fast.state_dict()["state"], exact=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fast_cuda_graph_preserves_clocks_and_checkpoint_continuation():
    left = [(name, torch.nn.Parameter(param.detach().cuda())) for name, param in _parameters()]
    right = [(name, torch.nn.Parameter(param.detach().clone())) for name, param in left]
    reference, fast = HybridMuon(left), HybridMuon(right)
    reference.execution_mode = "legacy"
    generator = torch.Generator().manual_seed(14)
    for step in range(8):
        for optimizer in (reference, fast):
            optimizer.param_groups[0]["lr"] = 0.01 / (step + 1)
            if step == 2:
                optimizer.param_groups[0].update(magma_temperature=0.7, muon_clip_auto_mult=2.0)
        for index, ((_, a), (_, b)) in enumerate(zip(left, right)):
            grad = torch.randn(a.shape, generator=generator).cuda()
            a.grad = None if step == 3 and index == 1 else grad
            b.grad = None if a.grad is None else grad.clone()
        reference.step()
        fast.step()
        for (_, a), (_, b) in zip(left, right):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
        _assert_state_equal(reference.state_dict()["state"], fast.state_dict()["state"], exact=False)
        if step == 4:
            fast.load_state_dict(copy.deepcopy(reference.state_dict()))
    for key, value in reference.get_diagnostics().items():
        assert value == pytest.approx(fast.get_diagnostics()[key], rel=3e-5, abs=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fast_cuda_buckets_keep_separate_outputs_for_equal_shapes_in_different_groups():
    generator = torch.Generator().manual_seed(15)
    left = [(f"weight{index}", torch.nn.Parameter(torch.randn(4, 8, generator=generator).cuda()))
            for index in range(4)]
    right = [(name, torch.nn.Parameter(param.detach().clone())) for name, param in left]
    reference = HybridMuon([{"params": left[:2]}, {"params": left[2:]}], lr=0.01)
    fast = HybridMuon([{"params": right[:2]}, {"params": right[2:]}], lr=0.01)
    reference.execution_mode = "legacy"
    for step in range(3):
        for index, ((_, a), (_, b)) in enumerate(zip(left, right)):
            a.grad = None if step == 1 and index == 0 else torch.full_like(a, index + 1 + step)
            b.grad = None if a.grad is None else a.grad.clone()
        reference.step()
        fast.step()
    for (_, a), (_, b) in zip(left, right):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
    _assert_state_equal(reference.state_dict()["state"], fast.state_dict()["state"], exact=False)
    for key, value in reference.get_diagnostics().items():
        assert value == pytest.approx(fast.get_diagnostics()[key], rel=3e-5, abs=2e-6)

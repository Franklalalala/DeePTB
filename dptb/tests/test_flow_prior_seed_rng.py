import pytest

torch = pytest.importorskip("torch")

from dptb.nnops.flow import _seeded_rng_scope  # noqa: E402


def test_seeded_scope_reproduces_manual_seed_and_restores_the_cpu_generator():
    state = {"node_features": torch.ones(2, 3), "note": "not a tensor"}
    torch.manual_seed(712)
    expected = torch.randn(8)
    torch.manual_seed(5)
    before = torch.random.get_rng_state()
    with _seeded_rng_scope(state, 712):
        drawn = torch.randn(8)
    assert torch.equal(drawn, expected)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="needs two CUDA devices")
def test_seeded_scope_leaves_cuda_devices_outside_the_state_untouched():
    torch.cuda.manual_seed_all(5)
    torch.randn(4, device="cuda:1")  # advance the other device's generator
    other_before = torch.cuda.get_rng_state(1)
    own_before = torch.cuda.get_rng_state(0)
    state = {"node_h0": torch.zeros(2, device="cuda:0")}

    with _seeded_rng_scope(state, 712):
        drawn = torch.randn(8, device="cuda:0")

    assert torch.equal(torch.cuda.get_rng_state(1), other_before)
    assert torch.equal(torch.cuda.get_rng_state(0), own_before)
    torch.cuda.manual_seed(712)
    assert torch.equal(drawn, torch.randn(8, device="cuda:0"))

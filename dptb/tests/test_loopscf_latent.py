"""Nontrivial equivariance, gradients, and active-edge cache behavior."""

import pytest
import torch
from torch import nn
from e3nn import o3
from dptb.data import AtomicDataDict as A
from dptb.nnops.loopscf.latent import (
    SharedLatentCore,
    ResidualReadout,
    _wrap_cached_embedding,
)


def core(irreps="2x0e + 1x1o + 1x1e + 1x2o", nonzero=True):
    torch.manual_seed(7)
    result = SharedLatentCore(irreps).double()
    if nonzero:
        with torch.no_grad():
            result.node_out.weight.normal_()
            result.edge_out.weight.normal_()
    return result


@pytest.mark.parametrize("inversion", [False, True])
def test_nonzero_core_equivariance(inversion):
    module = core()
    n = torch.randn(3, module.irreps.dim, dtype=torch.float64)
    e = torch.randn(4, module.irreps.dim, dtype=torch.float64)
    ei = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
    rotation = o3.angles_to_matrix(
        *[torch.tensor(v, dtype=torch.float64) for v in (0.31, 0.87, -0.63)]
    )
    if inversion:
        rotation = -rotation
    d = module.irreps.D_from_matrix(rotation)
    out = module(n, e, ei)
    rotated = module(n @ d.T, e @ d.T, ei)
    assert (out[0] - n).abs().max() > 1e-4
    for a, b in zip(out, rotated):
        torch.testing.assert_close(b, a @ d.T, atol=1e-7, rtol=1e-7)


def test_zero_init_learns_and_handles_no_edges():
    module = core(nonzero=False)
    n = torch.randn(3, module.irreps.dim, dtype=torch.float64)
    e = n.new_empty(0, module.irreps.dim)
    ei = torch.empty(2, 0, dtype=torch.long)
    out = module(n, e, ei)
    torch.testing.assert_close(out[0], n, atol=0, rtol=0)
    assert out[1].shape == e.shape
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    out[0].square().sum().backward()
    assert module.node_out.weight.grad.norm() > 0
    optimizer.step()
    optimizer.zero_grad()
    after = module(n, e, ei)[0]
    assert (after - n).abs().max() > 0
    after.square().sum().backward()
    assert module.node_self.weight.grad.norm() > 0


def test_representation_version_fails_closed():
    module = core()
    sd = module.state_dict()
    core().load_state_dict(sd, strict=True)
    sd["representation_version"] = torch.tensor(2)
    with pytest.raises(RuntimeError, match="representation version"):
        core().load_state_dict(sd, strict=True)


class TinyInit(nn.Module):
    def __init__(self, active):
        super().__init__()
        self.active = active
        self.calls = 0

    def forward(self, data):
        self.calls += 1
        return (
            None,
            data["node_input"],
            data["edge_input"][self.active],
            None,
            self.active,
        )


class TinyEmbedding(nn.Module):
    def __init__(self, active):
        super().__init__()
        self.init_layer = TinyInit(active)
        self.latent_core = core("2x0e")

    def _apply_rme_output_heads(self, n, e, noh, eoh):
        return n, e

    def forward(self, data):
        _, n, e, _, active = self.init_layer(data)
        n, e = self._apply_rme_output_heads(
            n, e, n.new_ones(len(n), 1), e.new_ones(len(e), 1)
        )
        return {
            **data,
            A.NODE_FEATURES_KEY: n,
            A.EDGE_FEATURES_KEY: e.new_zeros(
                data[A.EDGE_INDEX_KEY].shape[1], e.shape[1]
            ).index_copy(0, active, e),
        }


@pytest.mark.parametrize("strategy", ["bptt", "detach", "reset"])
@pytest.mark.parametrize("active", [[3, 1], [3, 2, 1, 0]])
@pytest.mark.parametrize("readout", ["frozen", "residual"])
def test_cache_uses_actual_edge_rows_and_encodes_once(strategy, active, readout):
    active = torch.tensor(active)
    emb = TinyEmbedding(active)
    data = {
        "node_input": torch.randn(3, 2, dtype=torch.float64),
        "edge_input": torch.randn(4, 2, dtype=torch.float64),
        A.EDGE_INDEX_KEY: torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]]),
    }
    original = {k: v.clone() for k, v in data.items()}
    initial = (data["node_input"], data["edge_input"][active])
    ei = data[A.EDGE_INDEX_KEY][:, active]
    expected1 = emb.latent_core(*initial, ei)
    expected2 = emb.latent_core(*(initial if strategy == "reset" else expected1), ei)
    if readout == "residual":
        emb.latent_readout = ResidualReadout("2x0e", "2x0e").double()
        with torch.no_grad():
            emb.latent_readout.node.weight.normal_()
            emb.latent_readout.edge.weight.normal_()
        corrections = emb.latent_readout(*expected2)
        expected2 = tuple(a + b for a, b in zip(initial, corrections))
    _wrap_cached_embedding(emb, readout)
    ctx = {"edge_index": data[A.EDGE_INDEX_KEY], "strategy": strategy}
    emb._latent_ctx = ctx
    first = emb(data)
    # Mutation by downstream AO conversion must not damage cached tensors.
    first[A.NODE_FEATURES_KEY] = first[A.NODE_FEATURES_KEY] + 100
    first["node_input"].add_(100)
    second = emb(data)
    assert emb.init_layer.calls == ctx["encoder_calls"] == 1
    torch.testing.assert_close(second[A.NODE_FEATURES_KEY], expected2[0])
    torch.testing.assert_close(second[A.EDGE_FEATURES_KEY][active], expected2[1])
    inactive = torch.ones(4, dtype=torch.bool)
    inactive[active] = False
    assert torch.count_nonzero(second[A.EDGE_FEATURES_KEY][inactive]) == 0
    for k, v in original.items():
        torch.testing.assert_close(data[k], v, atol=0, rtol=0)


def test_residual_readout_preserves_base_then_trains_recurrence():
    module = core(nonzero=False)
    readout = ResidualReadout(module.irreps, module.irreps).double()
    n = torch.randn(3, module.irreps.dim, dtype=torch.float64)
    e = torch.randn(3, module.irreps.dim, dtype=torch.float64)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    optimizer = torch.optim.SGD([*module.parameters(), *readout.parameters()], lr=0.01)
    z = module(*module(n, e, ei), ei)
    correction = readout(*z)
    assert all(torch.count_nonzero(v) == 0 for v in correction)
    loss = sum((x - 1).square().sum() for x in correction)
    loss.backward()
    assert readout.node.weight.grad.norm() > 0
    optimizer.step()
    optimizer.zero_grad()
    loss = sum((x - 1).square().sum() for x in readout(*module(*module(n, e, ei), ei)))
    loss.backward()
    assert module.node_out.weight.grad.norm() > 0

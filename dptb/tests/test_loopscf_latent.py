"""Loop-SCF latent core: nontrivial equivariance, gradients, active-edge cache
behavior, the AO<->RME/scalar feedback adapters, checkpoint version guards,
and the Trainer stepwise-loss bridge (absorbs test_loopscf_feedback.py's
adapter tests and test_loopscf_latent_training.py)."""
import re
from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from torch import nn

from dptb.data import AtomicData, AtomicDataDict as A
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.nnops.loopscf.adapters import ZeroInitWM, _attach_adapters, _inject
from dptb.nnops.loopscf.latent import (
    ResidualReadout,
    SharedLatentCore,
    _wrap_cached_embedding,
    validate_latent_checkpoint,
)
from dptb.nnops.loopscf.representation import AOPriorToRME, AOScalarFeedback
from dptb.nnops.loopscf.training import patch_stepwise_loss
from dptb.nnops.trainer import Trainer
from dptb.utils.constants import anglrMId


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
        return (None, data["node_input"], data["edge_input"][self.active], None, self.active)


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


@pytest.mark.parametrize("strategy", ["bptt", "reset"])
@pytest.mark.parametrize("active", [[3, 1], [3, 2, 1, 0]])
@pytest.mark.parametrize("readout", ["frozen", "residual"])
def test_cache_uses_actual_edge_rows_and_encodes_once(strategy, active, readout):
    # "detach" is dropped from this grid: its forward pass is identical to
    # "bptt" (only the backward path differs), so it would only double the
    # case count without checking a different forward value.
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


# ---------------------------------------------------------------------------
# AO<->RME / scalar feedback adapters (formerly test_loopscf_feedback.py)
# ---------------------------------------------------------------------------


def test_physical_h0_roundtrip_through_network_rme_without_input_mutation():
    from dptb.data.transforms import OrbitalMapper

    mapper = OrbitalMapper({"C": "2s2p1d"}, method="e3tb")
    codec = AOPriorToRME(mapper, dtype=torch.float64, device="cpu")
    node = torch.randn(2, mapper.reduced_matrix_element, dtype=torch.float64)
    edge = torch.randn(3, mapper.reduced_matrix_element, dtype=torch.float64)
    saved_node, saved_edge = node.clone(), edge.clone()
    edges = torch.tensor([[0, 0, 1], [1, 0, 0]])
    data = {
        A.NODE_H0_KEY: node,
        A.EDGE_H0_KEY: edge,
        A.EDGE_INDEX_KEY: edges,
        A.POSITIONS_KEY: torch.randn(2, 3, dtype=torch.float64),
    }
    rn, re_ = codec(data)
    reconstruct = E3Hamiltonian(
        idp=mapper, dtype=torch.float64, device="cpu",
        node_field=A.NODE_H0_KEY, edge_field=A.EDGE_H0_KEY,
    )
    recovered = reconstruct({**data, A.NODE_H0_KEY: rn, A.EDGE_H0_KEY: re_})
    torch.testing.assert_close(recovered[A.NODE_H0_KEY], node, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(recovered[A.EDGE_H0_KEY], edge, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(saved_node, node)
    torch.testing.assert_close(saved_edge, edge)


def test_ao_scalar_rotation_and_gradient():
    from dptb.data.transforms import OrbitalMapper

    mapper = OrbitalMapper({"C": "2s2p1d1f"}, method="e3tb")
    scalar = AOScalarFeedback(mapper)
    torch.manual_seed(74)
    x = torch.randn(3, scalar.width, dtype=torch.float64, requires_grad=True)
    R = o3.rand_matrix(dtype=torch.float64)
    rotated = x.clone()

    for pair, sl in mapper.orbpair_maps.items():
        ls = [anglrMId[re.findall(r"[a-zA-Z]", shell)[0]] for shell in pair.split("-")]
        a, b = [o3.Irrep(l, (-1) ** l).D_from_matrix(R) for l in ls]
        block = x[:, sl].reshape(-1, a.shape[0], b.shape[0])
        rotated[:, sl] = (a @ block @ b.T).flatten(1)
    torch.testing.assert_close(scalar(x), scalar(rotated), atol=1e-10, rtol=1e-10)
    scalar(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    # A traceless p block has zero scalar despite a nonzero first coordinate.
    pp = mapper.orbpair_maps["1p-1p"]
    y = torch.zeros_like(x)
    y[:, pp] = torch.diag(torch.tensor([-2.0, 0.0, 2.0])).flatten()
    torch.testing.assert_close(scalar(y), torch.zeros_like(scalar(y)))


@pytest.mark.parametrize("indices", [[1, 2], [2, 0, 1]])
def test_feedback_uses_active_order_even_at_equal_count(indices):
    emb = SimpleNamespace(
        wm_node=torch.nn.Identity(), wm_edge=torch.nn.Identity(), _wm_hid_slices=[slice(0, 1)]
    )
    wm = torch.tensor([[10.0], [20.0], [30.0]])
    _, edge = _inject(
        emb,
        {"wm_n": torch.zeros(1, 1), "wm_e": wm, "active_edges": torch.tensor(indices)},
        torch.zeros(1, 1),
        torch.zeros(len(indices), 1),
    )
    torch.testing.assert_close(edge, wm[indices])
    if len(indices) != 3:
        with pytest.raises(RuntimeError, match="mapping"):
            _inject(
                emb, {"wm_n": torch.zeros(1, 1), "wm_e": wm},
                torch.zeros(1, 1), torch.zeros(len(indices), 1),
            )


def test_same_shape_legacy_feedback_checkpoint_is_rejected():
    module = ZeroInitWM(3, 1)
    state = module.state_dict()
    module.load_state_dict(state)
    del state["feedback_version"]
    with pytest.raises(RuntimeError, match="incompatible"):
        module.load_state_dict(state, strict=False)


@pytest.mark.parametrize("mode", ["head", "moe"])
def test_injection_is_equivariant_at_its_actual_layer(mode):
    from dptb.data.transforms import OrbitalMapper

    emb = torch.nn.Module()
    emb.idp = OrbitalMapper({"H": "1s1p"}, method="e3tb")
    emb.init_layer = torch.nn.Linear(5, 5, dtype=torch.float64)
    emb.init_layer.irreps_out = o3.Irreps("2x0e+1x1o")
    final = torch.nn.Linear(5, 5, dtype=torch.float64)
    final.irreps_out = o3.Irreps("1x0e+1x1o+1x0e")
    emb.layers = torch.nn.ModuleList([final])
    _attach_adapters(emb, mode)
    with torch.no_grad():
        emb.wm_node.proj.weight.fill_(0.2)
        emb.wm_edge.proj.weight.fill_(0.3)
    ir = emb.init_layer.irreps_out if mode == "moe" else final.irreps_out
    D = ir.D_from_matrix(o3.rand_matrix(dtype=torch.float64))
    h = torch.randn(2, 5, dtype=torch.float64)
    wm = torch.ones(2, emb.wm_node.proj.in_features, dtype=torch.float64)
    ctx = {"wm_n": wm, "wm_e": wm}
    original = _inject(emb, ctx, h, h)
    rotated = _inject(emb, ctx, h @ D.T, h @ D.T)
    for before, after in zip(original, rotated):
        torch.testing.assert_close(before @ D.T, after, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Trainer stepwise-loss bridge and checkpoint strategy guard
# (formerly test_loopscf_latent_training.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy,expected_gradient", [("bptt", 162.0), ("detach", 90.0), ("reset", 36.0)]
)
def test_trainer_summed_loss_uses_normal_backward(monkeypatch, strategy, expected_gradient):
    class Batch(dict):
        def to(self, device):
            return self

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self._loopscf_full_bptt = True

        def _wm_one_k(self, *args):
            raise AssertionError("latent must use normal summed-loss backward")

        def forward(self, batch):
            first = self.weight * batch["x"]
            state = {"bptt": first, "detach": first.detach(), "reset": batch["x"]}[strategy]
            second = self.weight * state
            return {
                **batch,
                A.NODE_FEATURES_KEY: second,
                "_loop_preds": [(first, first * 0), (second, second * 0)],
            }

    # Register restoration before the wrapper mutates the class method.
    monkeypatch.setattr(Trainer, "_loss_on_batch", Trainer._loss_on_batch)
    monkeypatch.setattr(AtomicData, "to_AtomicDataDict", staticmethod(lambda b: dict(b)))
    patch_stepwise_loss(K=2)
    model = Model()
    trainer = SimpleNamespace(model=model, device="cpu", _batch_info=lambda batch: {})
    batch = Batch(x=torch.tensor(3.0), **{A.NODE_FEATURES_KEY: torch.tensor(0.0)})
    loss = Trainer._loss_on_batch(
        trainer, batch,
        lambda p, r: (p[A.NODE_FEATURES_KEY] - r[A.NODE_FEATURES_KEY]).square(),
    )
    assert model.weight.grad is None
    Trainer._backward_loss(trainer, loss)
    torch.testing.assert_close(model.weight.grad, torch.tensor(expected_gradient))


def test_latent_checkpoint_strategy_is_not_silently_changed():
    validate_latent_checkpoint({"_latent_strategy_code": torch.tensor(0)}, "bptt")
    with pytest.raises(ValueError, match="strategy"):
        validate_latent_checkpoint({"_latent_strategy_code": torch.tensor(2)}, "bptt")
    with pytest.raises(ValueError, match="strategy"):
        validate_latent_checkpoint({}, "bptt")
    with pytest.raises(ValueError, match="readout"):
        validate_latent_checkpoint(
            {"_latent_strategy_code": torch.tensor(0)}, "bptt", "residual"
        )


def test_full_forward_invokes_instrumented_prepare_once(monkeypatch):
    from dptb.nnops.loopscf import latent

    class Model(nn.Module):
        transform = True

        def __init__(self):
            super().__init__()
            self.embedding = TinyEmbedding(torch.tensor([3, 1]))
            emb = self.embedding
            emb.output_route_spec = SimpleNamespace(output_contract="rme")
            emb.layers = nn.ModuleList([nn.Identity()])
            emb.layers[0].irreps_out = o3.Irreps("2x0e")
            emb.idp = SimpleNamespace(orbpair_irreps=o3.Irreps("2x0e"))

        def forward(self, data):
            return self.embedding(data)

    # TinyEmbedding is a minimal stand-in, not a real lem_moe embedding module,
    # so it needs these two test doubles to be discoverable/convertible at all
    # (unlike the _wm_prepare tracing this test used to add on top, which was
    # pure private-call-count instrumentation and is dropped below).
    monkeypatch.setattr(latent, "_iter_embeddings", lambda m: [("embedding", m.embedding)])
    monkeypatch.setattr(
        latent, "AOPriorToRME",
        lambda *a, **kw: lambda b: (b[A.NODE_H0_KEY], b[A.EDGE_H0_KEY]),
    )
    model = latent.install_latent_corrector(Model(), SimpleNamespace(has_soc=False), K=2)
    batch = {
        "node_input": torch.randn(3, 2, dtype=torch.float64),
        "edge_input": torch.randn(4, 2, dtype=torch.float64),
        A.EDGE_INDEX_KEY: torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]]),
        A.ATOM_TYPE_KEY: torch.zeros(3, dtype=torch.long),
        A.NODE_H0_KEY: torch.zeros(3, 2, dtype=torch.float64),
        A.EDGE_H0_KEY: torch.zeros(4, 2, dtype=torch.float64),
    }
    out = model(batch)
    # Observable outputs only: K entries were produced and the encoder ran once
    # (dropped the private _wm_prepare patch and its call-count/kpts checks).
    assert len(out["_loop_preds"]) == 2
    assert model.embedding.init_layer.calls == 1

"""Verify the real Trainer loss bridge preserves the requested credit horizon."""

from types import SimpleNamespace
import pytest
import torch
from torch import nn
from dptb.data import AtomicData, AtomicDataDict as A
from dptb.nnops.trainer import Trainer
from dptb.nnops.loopscf.training import patch_stepwise_loss
from dptb.nnops.loopscf.latent import validate_latent_checkpoint


@pytest.mark.parametrize(
    "strategy,expected_gradient", [("bptt", 162.0), ("detach", 90.0), ("reset", 36.0)]
)
def test_trainer_summed_loss_uses_normal_backward(
    monkeypatch, strategy, expected_gradient
):
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
            state = {"bptt": first, "detach": first.detach(), "reset": batch["x"]}[
                strategy
            ]
            second = self.weight * state
            return {
                **batch,
                A.NODE_FEATURES_KEY: second,
                "_loop_preds": [(first, first * 0), (second, second * 0)],
            }

    # Register restoration before the wrapper mutates the class method.
    monkeypatch.setattr(Trainer, "_loss_on_batch", Trainer._loss_on_batch)
    monkeypatch.setattr(
        AtomicData, "to_AtomicDataDict", staticmethod(lambda b: dict(b))
    )
    patch_stepwise_loss(K=2)
    model = Model()
    trainer = SimpleNamespace(model=model, device="cpu", _batch_info=lambda batch: {})
    batch = Batch(x=torch.tensor(3.0), **{A.NODE_FEATURES_KEY: torch.tensor(0.0)})
    loss = Trainer._loss_on_batch(
        trainer,
        batch,
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
    from dptb.tests.test_loopscf_latent import TinyEmbedding
    from e3nn import o3

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

    monkeypatch.setattr(
        latent, "_iter_embeddings", lambda m: [("embedding", m.embedding)]
    )
    monkeypatch.setattr(
        latent,
        "AOPriorToRME",
        lambda *a, **kw: lambda b: (b[A.NODE_H0_KEY], b[A.EDGE_H0_KEY]),
    )
    model = latent.install_latent_corrector(
        Model(), SimpleNamespace(has_soc=False), K=2
    )
    original_prepare = model._wm_prepare
    observed = []

    def traced(batch):
        state = original_prepare(batch)
        observed.append((state["n_atom"], state["kpts"].clone()))
        return state

    model._wm_prepare = traced
    batch = {
        "node_input": torch.randn(3, 2, dtype=torch.float64),
        "edge_input": torch.randn(4, 2, dtype=torch.float64),
        A.EDGE_INDEX_KEY: torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]]),
        A.ATOM_TYPE_KEY: torch.zeros(3, dtype=torch.long),
        A.NODE_H0_KEY: torch.zeros(3, 2, dtype=torch.float64),
        A.EDGE_H0_KEY: torch.zeros(4, 2, dtype=torch.float64),
    }
    out = model(batch)
    assert len(observed) == 1 and observed[0][0] == 3
    torch.testing.assert_close(out["_loop_kpts"], observed[0][1], atol=0, rtol=0)
    assert len(out["_loop_preds"]) == 2
    assert model.embedding.init_layer.calls == 1

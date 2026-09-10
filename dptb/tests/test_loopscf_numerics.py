"""Public package regressions for electron counting and interleaved gradients."""

import types, unittest
import torch
import pytest
from dptb.nnops.loopscf.occupations import (
    factor_overlap_robust,
    compute_mulliken_fast,
    _eval_occupation_kpoints,
)
from dptb.nnops.loopscf.training import stepwise_train_loss
from dptb.nnops.trainer import Trainer

idp = types.SimpleNamespace(atom_norb=torch.tensor([1]))


def q(H, S, n, weights=None):
    args = (
        H,
        *factor_overlap_robust(S),
        n,
        idp,
        torch.zeros(H.shape[-1], dtype=torch.long),
    )
    return compute_mulliken_fast(
        *args, **({} if weights is None else {"k_weights": weights})
    )


def diag(x):
    return torch.diag_embed(torch.tensor(x, dtype=torch.float64))


class Physics(unittest.TestCase):
    def test_odd(self):
        torch.testing.assert_close(
            q(diag([[-1, 1]]), diag([[1, 1]]), 1), torch.tensor([1.0, 0.0])
        )

    def test_fractional(self):
        torch.testing.assert_close(
            q(diag([[-1, 1]]), diag([[1, 1]]), 2.5), torch.tensor([2.0, 0.5])
        )

    def test_metal_global(self):
        torch.testing.assert_close(
            q(diag([[-3, -2], [-1, 3]]), diag([[1, 1], [1, 1]]), 2),
            torch.tensor([1.0, 1.0]),
        )

    def test_weighted(self):
        torch.testing.assert_close(
            q(
                diag([[-3, -2], [-1, 3]]),
                diag([[1, 1], [1, 1]]),
                2,
                torch.tensor([0.25, 0.75]),
            ),
            torch.tensor([1.5, 0.5]),
        )

    def test_degenerate(self):
        torch.testing.assert_close(
            q(diag([[0, 0]]), diag([[1, 1]]), 1), torch.tensor([0.5, 0.5])
        )

    def test_k_permutation(self):
        h = diag([[-3, -2], [-1, 3]])
        s = diag([[1, 1], [1, 1]])
        w = torch.tensor([0.25, 0.75])
        torch.testing.assert_close(q(h, s, 2, w), q(h.flip(0), s, 2, w.flip(0)))

    def test_capacity(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            q(diag([[-1, 1]]), diag([[1, -0.405]]), 4)

    def test_empty_overlap(self):
        with self.assertRaisesRegex(ValueError, "subspace"):
            q(diag([[-1, 1]]), diag([[-1, -2]]), 1)

    def test_negative_electrons(self):
        with self.assertRaises(ValueError):
            q(diag([[-1, 1]]), diag([[1, 1]]), -1)

    def test_nonorthogonal_complex(self):
        s = torch.tensor([[[2, 0.2j], [-0.2j, 1.0]]], dtype=torch.complex128)
        h = torch.tensor([[[-1, 0.4j], [-0.4j, 2.0]]], dtype=torch.complex128)
        self.assertAlmostEqual(q(h, s, 1.5).sum().item(), 1.5, places=5)

    def test_empty_and_full(self):
        h = diag([[-1, 1]])
        s = diag([[1, 1]])
        torch.testing.assert_close(q(h, s, 0), torch.zeros(2))
        torch.testing.assert_close(q(h, s, 4), torch.ones(2) * 2)

    def test_atom_partition_mismatch(self):
        with self.assertRaisesRegex(ValueError, "orbital"):
            h = diag([[-1, 1]])
            compute_mulliken_fast(
                h, *factor_overlap_robust(diag([[1, 1]])), 2, idp, torch.tensor([0])
            )


def test_stepwise_gradients_and_trainer_hook():
    p = torch.nn.Parameter(torch.tensor(2.0))
    m = types.SimpleNamespace(
        _wm_K=2,
        _wm_prepare=lambda b: {},
        _wm_one_k=lambda *a: {"x": p * a[1]},
        _wm_update=lambda *a: (None, None),
    )
    backward = torch.Tensor.backward
    loss, _ = stepwise_train_loss(
        m, {}, {}, lambda out, ref: out["x"] ** 2, [0.25, 0.75]
    )
    Trainer._backward_loss(None, loss)
    assert p.grad.item() == 13.0
    assert loss.item() == 13.0
    assert torch.Tensor.backward is backward
    with pytest.raises(RuntimeError):
        Trainer._backward_loss(None, torch.tensor(0.0))


def test_eval_quadrature_repeats_without_touching_rng():
    state = torch.random.get_rng_state().clone()
    a = _eval_occupation_kpoints(2, 8, "cpu")
    torch.testing.assert_close(a, _eval_occupation_kpoints(2, 8, "cpu"))
    assert torch.equal(state, torch.random.get_rng_state())
    assert a.shape == (2, 8, 3)


def test_band_metric_rejects_ambiguous_electron_count(monkeypatch):
    from dptb.nnops.band_losses import FW10EigLoss
    from dptb.nnops.loopscf.spectral import patch_fw10_per_graph
    from dptb.data import AtomicDataDict as A

    original = FW10EigLoss._band_loss
    monkeypatch.setattr(FW10EigLoss, "_band_loss", original)
    patch_fw10_per_graph()
    obj = FW10EigLoss(basis={"H": "1s"})
    data = {
        A.ATOM_TYPE_KEY: torch.zeros(2, dtype=torch.long),
        A.EDGE_INDEX_KEY: torch.tensor([[0, 1], [0, 1]]),
        A.EDGE_CELL_SHIFT_KEY: torch.zeros(2, 3),
    }
    ref = {
        A.BATCH_KEY: torch.tensor([0, 1]),
        "ptr": torch.tensor([0, 1, 2]),
        "nelec": torch.tensor([1.0]),
    }
    with pytest.raises(ValueError, match="electron count per graph"):
        obj._band_loss(data, ref)

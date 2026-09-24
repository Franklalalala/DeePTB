"""Loop-SCF physics: Mulliken electron counting, Fermi levels, overlap factorization, k-space
assembly, band-error alignment and the stepwise training bridge."""

import importlib.util
import types
from pathlib import Path

import pytest
import torch

from dptb.data import AtomicDataDict as A
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.loopscf.kspace import assemble_flat, build_k_plan
from dptb.nnops.loopscf.metrics import fermi_level, mu_aligned_band_error
from dptb.nnops.loopscf.occupations import (
    _eval_occupation_kpoints,
    compute_mulliken_fast,
    factor_overlap_robust,
)
from dptb.nnops.loopscf.spectral import _fw10_one_graph, eigvals_from_factor
from dptb.nnops.loopscf.training import stepwise_train_loss
from dptb.nnops.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parents[2]
_IDP = types.SimpleNamespace(atom_norb=torch.tensor([1]))


def _diag(values):
    return torch.diag_embed(torch.tensor(values, dtype=torch.float64))


def _mulliken(H, S, nelec, weights=None, orbital_atoms=None):
    if orbital_atoms is None:
        orbital_atoms = torch.zeros(H.shape[-1], dtype=torch.long)
    kwargs = {} if weights is None else {"k_weights": weights}
    return compute_mulliken_fast(
        H, *factor_overlap_robust(S), nelec, _IDP, orbital_atoms, **kwargs
    )


_BAND = _diag([[-1, 1]])
_TWO_K = _diag([[-3, -2], [-1, 3]])
_UNIT = _diag([[1, 1]])
_UNIT_TWO_K = _diag([[1, 1], [1, 1]])
_K_WEIGHTS = torch.tensor([0.25, 0.75])


@pytest.mark.parametrize(
    ("H", "S", "nelec", "weights", "expected"),
    [
        (_BAND, _UNIT, 1, None, [1.0, 0.0]),
        (_BAND, _UNIT, 2.5, None, [2.0, 0.5]),
        (_TWO_K, _UNIT_TWO_K, 2, None, [1.0, 1.0]),
        (_TWO_K, _UNIT_TWO_K, 2, _K_WEIGHTS, [1.5, 0.5]),
        (_diag([[0, 0]]), _UNIT, 1, None, [0.5, 0.5]),
        (_BAND, _UNIT, 0, None, [0.0, 0.0]),
        (_BAND, _UNIT, 4, None, [2.0, 2.0]),
    ],
    ids=["odd", "fractional", "metal_global_fermi", "k_weighted", "degenerate", "empty", "full"],
)
def test_mulliken_occupations_match_hand_values(H, S, nelec, weights, expected):
    torch.testing.assert_close(
        _mulliken(H, S, nelec, weights), torch.tensor(expected, dtype=torch.float32)
    )


def test_mulliken_does_not_depend_on_k_point_order():
    torch.testing.assert_close(
        _mulliken(_TWO_K, _UNIT_TWO_K, 2, _K_WEIGHTS),
        _mulliken(_TWO_K.flip(0), _UNIT_TWO_K, 2, _K_WEIGHTS.flip(0)),
    )


def test_mulliken_conserves_electrons_with_complex_nonorthogonal_overlap():
    S = torch.tensor([[[2, 0.2j], [-0.2j, 1.0]]], dtype=torch.complex128)
    H = torch.tensor([[[-1, 0.4j], [-0.4j, 2.0]]], dtype=torch.complex128)
    assert _mulliken(H, S, 1.5).sum().item() == pytest.approx(1.5, abs=5e-6)


@pytest.mark.parametrize(
    ("S", "nelec", "orbital_atoms", "key"),
    [
        (_diag([[1, -0.405]]), 4, None, "capacity"),
        (_diag([[-1, -2]]), 1, None, "subspace"),
        (_UNIT, -1, None, "nelec"),
        (_UNIT, 2, torch.tensor([0]), "orbital"),
    ],
    ids=["over_capacity", "empty_overlap_subspace", "negative_electrons", "atom_partition_mismatch"],
)
def test_invalid_electron_counts_and_partitions_are_rejected(S, nelec, orbital_atoms, key):
    with pytest.raises(ValueError, match=key):
        _mulliken(_BAND, S, nelec, orbital_atoms=orbital_atoms)


def test_ill_conditioned_overlap_is_projected_and_reported():
    S = torch.diag(torch.tensor([1.0, 1e-12], dtype=torch.float64)).unsqueeze(0)
    diagnostics = {}
    factors = factor_overlap_robust(S, diagnostics=diagnostics)
    assert diagnostics["dropped_modes"].tolist() == [1]
    assert diagnostics["spin_degenerate_capacity"].tolist() == [2]
    assert diagnostics["condition_S"].item() == pytest.approx(1e12)
    H = torch.diag(torch.tensor([2.0, 1e-8], dtype=torch.float64)).unsqueeze(0)
    assert eigvals_from_factor(H, *factors)[0, 0].item() == pytest.approx(2.0)


@pytest.mark.parametrize("smearing", [0.0, 0.1])
def test_metal_fermi_level_and_shift_invariant_path_metric(smearing):
    bz = torch.tensor([[-3.0, -2.0], [-1.0, 3.0]], dtype=torch.float64)
    mu = fermi_level(bz, 2, smearing=smearing)
    assert mu.item() == pytest.approx(-1.5, abs=2e-5)
    shifted = fermi_level(bz + 7, 2, smearing=smearing)
    path = torch.tensor([[-4.0, 2.0], [-0.4, 0.8]])
    error, count = mu_aligned_band_error(path + 7, path, mu_pred=shifted, mu_ref=mu)
    assert count == 4
    assert error.item() == pytest.approx(0, abs=1e-6)


def test_fractional_degenerate_fermi_level_and_weighted_filling():
    assert fermi_level(torch.zeros(2, 2), 1.5).item() == 0
    bz = torch.tensor([[-3.0, -2.0], [-1.0, 3.0]])
    assert fermi_level(bz, 2, k_weights=torch.tensor([0.25, 0.75])).item() == -1


def test_isolated_atom_assembly_preserves_double_precision_and_gradient():
    mapper = OrbitalMapper({"H": "1s"}, method="e3tb")
    plan = build_k_plan(
        mapper,
        torch.tensor([0]),
        torch.empty(2, 0, dtype=torch.long),
        torch.tensor([0]),
        torch.tensor([0, 1]),
        "cpu",
    )
    x = torch.tensor([[1.000000001]], dtype=torch.float64, requires_grad=True)
    H = plan.block(
        assemble_flat(
            plan, x, x.new_empty(0, 1), torch.empty(1, 0, dtype=torch.complex128), torch.complex128
        ),
        0,
    )
    torch.testing.assert_close(H.real.reshape_as(x), x, atol=1e-14, rtol=0)
    H.real.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_band_error_aligns_over_the_full_path_not_per_chunk():
    # A k-dependent common shift: aligning each k-point separately would hide it.
    labels = torch.tensor([[-2.0, 2.0], [-1.0, 3.0], [0.0, 4.0]], dtype=torch.float64)
    prediction = labels + torch.tensor([[0.0], [1.0], [3.0]], dtype=torch.float64)
    global_error = _fw10_one_graph(prediction, labels, 2, 10)[0]
    local_error = torch.stack(
        [_fw10_one_graph(p[None], r[None], 2, 10)[0] for p, r in zip(prediction, labels)]
    ).mean()
    assert global_error > 1 and local_error == 0


def test_chunked_nonorthogonal_path_bands_match_the_full_path():
    path = REPO_ROOT / "examples" / "loopscf" / "evaluate.py"
    if not path.is_file():
        pytest.skip("needs the repository checkout (examples/loopscf/evaluate.py)")
    spec = importlib.util.spec_from_file_location("loopscf_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    idp = OrbitalMapper({"H": "1s"}, method="e3tb")
    ref = {
        A.ATOM_TYPE_KEY: torch.tensor([0, 0]),
        A.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]]),
        A.EDGE_CELL_SHIFT_KEY: torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64),
        A.NODE_OVERLAP_KEY: torch.tensor([[1.0], [1.3]], dtype=torch.float64),
        A.EDGE_OVERLAP_KEY: torch.tensor([[0.1], [0.1]], dtype=torch.float64),
        A.NODE_H0_KEY: torch.tensor([[-1.0], [2.0]], dtype=torch.float64),
        A.EDGE_H0_KEY: torch.tensor([[0.2], [0.2]], dtype=torch.float64),
    }
    kpoints = torch.tensor(
        [[0.0, 0.0, 0.0], [0.13, 0.0, 0.0], [0.3, 0.0, 0.0], [0.5, 0.0, 0.0], [0.8, 0.0, 0.0]],
        dtype=torch.float64,
    )
    zero = (torch.zeros(2, 1, dtype=torch.float64), torch.zeros(2, 1, dtype=torch.float64))
    outputs = {
        "label": zero,
        "prediction": (
            torch.tensor([[0.1], [0.3]], dtype=torch.float64),
            torch.tensor([[0.2], [0.2]], dtype=torch.float64),
        ),
    }
    full, full_diag = module.path_bands(ref, outputs, idp, kpoints, 5)
    chunks, chunk_diag = module.path_bands(ref, outputs, idp, kpoints, 2)
    assert full_diag == chunk_diag
    for name in outputs:
        torch.testing.assert_close(chunks[name], full[name], rtol=1e-12, atol=1e-12)


def test_stepwise_loss_gradient_and_trainer_backward_hook():
    weight = torch.nn.Parameter(torch.tensor(2.0))
    model = types.SimpleNamespace(
        _wm_K=2,
        _wm_prepare=lambda batch: {},
        _wm_one_k=lambda *args: {"x": weight * args[1]},
        _wm_update=lambda *args: (None, None),
    )
    backward = torch.Tensor.backward
    loss, _ = stepwise_train_loss(model, {}, {}, lambda out, ref: out["x"] ** 2, [0.25, 0.75])
    Trainer._backward_loss(None, loss)
    # 0.25 * (2 * 1)^2 + 0.75 * (2 * 2)^2 = 13, d/dw = 0.25 * 2w + 0.75 * 8w = 13 at w = 2
    assert weight.grad.item() == 13.0
    assert loss.item() == 13.0
    assert torch.Tensor.backward is backward
    with pytest.raises(RuntimeError):
        Trainer._backward_loss(None, torch.tensor(0.0))


def test_eval_quadrature_repeats_without_touching_the_rng():
    state = torch.random.get_rng_state().clone()
    kpoints = _eval_occupation_kpoints(2, 8, "cpu")
    torch.testing.assert_close(kpoints, _eval_occupation_kpoints(2, 8, "cpu"))
    assert torch.equal(state, torch.random.get_rng_state())
    assert kpoints.shape == (2, 8, 3)


def test_band_metric_rejects_ambiguous_electron_count(monkeypatch):
    from dptb.nnops.band_losses import FW10EigLoss
    from dptb.nnops.loopscf.spectral import patch_fw10_per_graph

    # Register restoration before the patch mutates the class method.
    monkeypatch.setattr(FW10EigLoss, "_band_loss", FW10EigLoss._band_loss)
    patch_fw10_per_graph()
    loss = FW10EigLoss(basis={"H": "1s"})
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
    with pytest.raises(ValueError, match="nelec"):
        loss._band_loss(data, ref)

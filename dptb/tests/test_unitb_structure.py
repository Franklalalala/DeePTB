"""Structure controls preserve calibration, checkpoint and low-rank contracts."""
import copy

import pytest
import torch
from torch.nn import functional as F

from dptb.data import _keys
from dptb.nn.build import build_model
from dptb.nn.embedding.unitb_options import unitb_options
from dptb.nn.embedding.unitb_structure import (
    StructureStats, fit_structure_stats, initialize_fresh, merged_linear,
)
from dptb.nn.pdq_moe import PDQMoELinear, PDQMoERouting


def _model(scope, execution="merged_core"):
    return build_model(
        common_options=dict(basis={"H": "1s", "O": "1s1p"}, overlap=False,
                            dtype="float32", device="cpu"),
        model_options=dict(embedding=dict(
            method="unitb", n_layers=1, n_radial_basis=4, r_max=4.0,
            irreps_hidden="3x0e+3x1o+3x2e", avg_num_neighbors=2.0,
            env_embed_multiplicity=2, latent_dim=8, latent_channels=[8],
            edge_one_hot_dim=4, tp_radial_channels=[4],
            use_layer_onehot_tp=False, use_out_onehot_tp=False,
            mole_linear_mode="split_loop", so2_fusion_mode="streamed_m_major_ref",
            mole_expert_rank=2, use_interpolation_out=False,
            structure_mole=dict(enabled=True, route_scope=scope, execution=execution),
        ), prediction=dict(method="e3tb", scale_type="no_scale")),
        train_options={}, no_check=True,
    )


def _data(model, scale=1.0):
    generator = torch.Generator().manual_seed(77)
    h, o = [model.idp.chemical_symbol_to_type[s] for s in ("H", "O")]
    dim = model.idp.reduced_matrix_element
    return {
        _keys.POSITIONS_KEY: torch.tensor([[0., 0., 0.], [scale, .2, -.1]]),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]]),
        _keys.ATOM_TYPE_KEY: torch.tensor([[h], [o]]),
        _keys.EDGE_TYPE_KEY: torch.tensor([model.idp.bond_to_type[s] for s in ("H-O", "O-H")]),
        _keys.NODE_H0_KEY: torch.randn(2, dim, generator=generator),
        _keys.EDGE_H0_KEY: torch.randn(2, dim, generator=generator),
    }


@pytest.mark.parametrize("scope,execution", [
    ("constant", "merged_core"), ("structure", "merged_core"), ("structure", "reference"),
])
def test_structure_checkpoint_and_gradients(scope, execution, monkeypatch):
    monkeypatch.delenv("DPTB_SO2_FUSION_MODE", raising=False)
    monkeypatch.delenv("DPTB_SO2_FUSE_M_CUBLAS", raising=False)
    torch.manual_seed(42)
    model = _model(scope, execution)
    data = _data(model)
    if scope == "structure":
        with pytest.raises(RuntimeError, match="uncalibrated"):
            model(copy.deepcopy(data))
        calibration = fit_structure_stats(model, [data, _data(model, 1.4)])
        assert calibration["split"] == "train"
        assert model.embedding.structure_stats.training_count.item() == 2
    else:
        assert model.embedding.router is None
        assert model.embedding.structure_stats is None
    restored = _model(scope, execution)
    restored.load_state_dict(model.state_dict(), strict=True)
    expected = model(copy.deepcopy(data))
    actual = restored(copy.deepcopy(data))
    for key in (_keys.NODE_FEATURES_KEY, _keys.EDGE_FEATURES_KEY):
        assert torch.equal(actual[key], expected[key])
    sum(expected[k].square().mean() for k in
        (_keys.NODE_FEATURES_KEY, _keys.EDGE_FEATURES_KEY)).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    cores = [m for m in model.modules() if isinstance(m, PDQMoELinear)]
    assert cores and any(m.core_experts.grad is not None and torch.count_nonzero(m.core_experts.grad)
                         for m in cores)
    if scope == "structure":
        assert model.embedding.router[0].weight.grad is not None
        assert torch.count_nonzero(model.embedding.router[0].weight.grad)


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("shape", [(5, 4), (5, 2, 4), (0, 4)])
def test_merged_core_matches_dense_matrix_reference(constant, shape):
    torch.manual_seed(43)
    count = 1 if constant else 4
    layer = PDQMoELinear(4, 3, num_experts=count, num_shared_experts=1,
                         mole_expert_parameterization="shared_core", mole_expert_rank=2,
                         mole_linear_mode="split_loop").double()
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    alpha = torch.ones(2, 1, dtype=torch.float64) if constant else torch.rand(2, 4, dtype=torch.float64)
    alpha = (alpha / alpha.sum(-1, keepdim=True)).requires_grad_()
    graph = torch.tensor([1, 0, 1, 0, 1])[:shape[0]]
    route = PDQMoERouting(coefficients=alpha, graph_index=graph)
    route.structure_execution = "constant" if constant else "merged_core"
    route.structure_rows = tuple(torch.where(graph == s)[0] for s in range(2))
    route.structure_inverse = torch.cat(route.structure_rows).argsort()
    actual = merged_linear(layer, x, route)
    weights = layer.weight_shared[0] + torch.einsum(
        "or,krs,is->koi", layer.basis_left, layer.core_experts, layer.basis_right
    )
    biases = layer.bias_shared[0] + layer.bias_experts
    expected = torch.zeros_like(actual)
    for k in range(count):
        coefficient = alpha[graph, k].reshape((-1,) + (1,) * (x.ndim - 1))
        expected = expected + coefficient * F.linear(x, weights[k], biases[k])
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.core_experts.grad is not None and torch.isfinite(layer.core_experts.grad).all()


def test_calibration_is_training_only_frozen_and_persistent():
    stats = StructureStats(species=2, gram_dim=3, rbf_rmax=4.)
    rows = [torch.arange(stats.width).float()[None], torch.arange(stats.width).float()[None] + 2]
    with pytest.raises(ValueError, match="train"):
        stats.fit(rows, split="test")
    stats.fit(rows, split="train")
    assert torch.equal(stats.mean, rows[0][0] + 1)
    assert torch.equal(stats.scale, torch.ones_like(stats.scale))
    with pytest.raises(ValueError, match="frozen"):
        stats.fit(rows, split="train")
    restored = StructureStats(species=2, gram_dim=3, rbf_rmax=4.)
    restored.load_state_dict(stats.state_dict(), strict=True)
    assert torch.equal(restored(rows[0]), stats(rows[0]))
    masked = restored(rows[0], prior_stats=False)
    assert not torch.count_nonzero(masked[:, -64:])


def test_dense_conversion_accepts_equivalent_historical_method(tmp_path):
    target = _model("constant")
    source_options = copy.deepcopy(target.model_options)
    source_options["embedding"] = unitb_options(source_options["embedding"])
    source_options["embedding"].update(
        method="lem_moe_v3_edge_h0", structure_mole={"enabled": False},
        num_experts=1, top_k=1, num_shared_experts=0,
        mole_expert_parameterization="full", use_flow_time_embedding=False,
    )
    source = build_model(
        common_options=dict(basis={"H": "1s", "O": "1s1p"}, overlap=False,
                            dtype="float32", device="cpu"),
        model_options=source_options, train_options={}, no_check=True,
    )
    checkpoint = tmp_path / "dense.pth"
    torch.save(dict(config={"model_options": source_options, "train_options": {}},
                    model_state_dict=source.state_dict()), checkpoint)
    initialize_fresh(target, {"init_from": str(checkpoint)})
    source_layers = dict(source.named_modules())
    for name, layer in target.named_modules():
        if isinstance(layer, PDQMoELinear):
            actual = layer.weight_shared[0] + layer.basis_left @ layer.core_experts[0] @ layer.basis_right.T
            torch.testing.assert_close(actual, source_layers[name].weight_experts[0], rtol=1e-6, atol=1e-7)

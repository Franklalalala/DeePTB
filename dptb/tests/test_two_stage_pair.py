"""Two-stage pair stream: streamed vs materialized refinement, equivariance, sensitivity, config
rejection, and its LemMoEV3H0 / LemPair / block-ODE integration."""
from __future__ import annotations

import copy

import pytest
import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nn.embedding.lem_pair import LemPair
from dptb.nn.embedding.two_stage_pair import NormFreePairRefineLayer, TwoStagePairStream
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

from dptb.tests.block_ode_fixtures import _rotate_canvas_blocks
from dptb.tests.pair_helpers import (
    ao_wigner,
    block_ode_model,
    clone_data,
    complete_directed_edges,
    deterministic_fp64,
    edge_block_drift,
    feature_wigner,
    fp64_default,
    model_options,
    molecule_data,
    prepared_flow_batch,
)

TWO_STAGE_OPTIONS = dict(
    two_stage_pair_enable=True,
    two_stage_pair_refine_layers=2,
    two_stage_pair_refine_rank=3,
    two_stage_pair_refine_radial_dim=3,
    two_stage_pair_refine_edge_chunk_size=2,
)


def _stream_case(*, n_refine_layers=0, tail_gate=False):
    torch.manual_seed(20260724)
    irreps = o3.Irreps("2x0e+1x1o+1x1e+1x2e")
    stream = TwoStagePairStream(
        num_types=1,
        node_irreps=irreps,
        edge_irreps=irreps,
        latent_dim=4,
        latent_channels=(4,),
        radial_channels=(4,),
        use_layer_onehot_tp=False,
        edge_one_hot_dim=2,
        so2_fusion_mode="streamed_m_major_ref",
        mole_linear_mode="indexed_ref",
        dtype=torch.float64,
        device="cpu",
        num_experts=1,
        num_shared_experts=1,
        n_refine_layers=n_refine_layers,
        refine_rank=3,
        refine_radial_dim=3,
        refine_edge_chunk_size=2,
        tail_gate=tail_gate,
    ).eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.2, -0.1], [-0.3, 0.8, 0.4]], dtype=torch.float64
    )
    edge_index = complete_directed_edges(positions.shape[0])
    src, dst = edge_index
    n_edges = edge_index.shape[1]
    inputs = dict(
        latents=torch.randn(n_edges, 4, dtype=torch.float64),
        node_features=torch.randn(positions.shape[0], irreps.dim, dtype=torch.float64),
        node_onehot=torch.ones(positions.shape[0], 1, dtype=torch.float64),
        edge_features=torch.randn(n_edges, irreps.dim, dtype=torch.float64),
        edge_index=edge_index,
        edge_vector=positions[dst] - positions[src],
        cutoff_coeffs=torch.ones(n_edges, dtype=torch.float64),
        active_edges=torch.arange(n_edges, dtype=torch.long),
        edge_one_hot=torch.randn(n_edges, 2, dtype=torch.float64),
        wigner_D_all=None,
        mole_globals=MOLEGlobals(
            coefficients=torch.ones(1, 1, dtype=torch.float64),
            sizes=torch.tensor([n_edges], dtype=torch.long),
            topk_indices=torch.zeros(1, 1, dtype=torch.long),
            topk_values=torch.ones(1, 1, dtype=torch.float64),
        ),
    )
    return stream, irreps, inputs


def _direct_model(*, enabled):
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    if enabled:
        options.update(TWO_STAGE_OPTIONS)
    else:
        options.update({**TWO_STAGE_OPTIONS, "two_stage_pair_enable": False})
    return LemMoEV3H0(**options).eval()


# --- the stream on its own ----------------------------------------------------


@pytest.mark.parametrize("tail_gate", [False, True])
@pytest.mark.parametrize(
    ("dtype", "tolerance"), [(torch.float64, 1.0e-12), (torch.float32, 1.0e-5)], ids=["fp64", "fp32"]
)
def test_streamed_refinement_matches_materialized_reference(dtype, tolerance, tail_gate):
    irreps = o3.Irreps("2x0e+2x1o+1x1e+1x2e")

    def refine_stack(seed):
        torch.manual_seed(seed)
        return torch.nn.ModuleList(
            NormFreePairRefineLayer(
                irreps, irreps, rank=3, radial_dim=2, edge_chunk_size=2,
                tail_gate=tail_gate, dtype=dtype, device="cpu",
            )
            for _ in range(2)
        ).eval()

    layers = refine_stack(20260725)
    node_features = torch.randn(4, irreps.dim, dtype=dtype)
    all_edges = complete_directed_edges(4)
    active_edges = torch.tensor([0, 2, 5, 7, 10], dtype=torch.long)  # ragged last chunk
    edge_vector = torch.randn(all_edges.shape[1], 3, dtype=dtype)
    edge_features = torch.randn(active_edges.numel(), irreps.dim, dtype=dtype)

    reference = candidate = edge_features
    for layer in layers:
        reference = layer._forward_materialized(
            node_features, reference, all_edges, edge_vector, active_edges
        )
        candidate = layer(node_features, candidate, all_edges, edge_vector, active_edges)
    torch.testing.assert_close(candidate, reference, rtol=0.0, atol=tolerance)
    if dtype != torch.float64:
        return

    restored = refine_stack(1)
    restored.load_state_dict(layers.state_dict(), strict=True)
    replay = edge_features
    for layer in restored:
        replay = layer(node_features, replay, all_edges, edge_vector, active_edges)
    assert torch.equal(replay, candidate)


@pytest.mark.parametrize(
    ("n_refine_layers", "tail_gate"),
    [(0, False), (2, False), (2, True)],
    ids=["no_refine", "refine", "refine_tail_gate"],
)
def test_stream_is_equivariant(n_refine_layers, tail_gate):
    with deterministic_fp64():
        stream, irreps, inputs = _stream_case(n_refine_layers=n_refine_layers, tail_gate=tail_gate)
        reference = stream(**inputs)
        torch.manual_seed(17)
        rotation = o3.rand_matrix(dtype=torch.float64)
        representation = feature_wigner(irreps, rotation)
        rotated = stream(
            **dict(
                inputs,
                node_features=inputs["node_features"] @ representation.T,
                edge_features=inputs["edge_features"] @ representation.T,
                edge_vector=inputs["edge_vector"] @ rotation.T,
            )
        )
    torch.testing.assert_close(rotated, reference @ representation.T, rtol=0.0, atol=1.0e-9)


@pytest.mark.parametrize("n_refine_layers", [0, 2])
def test_stream_row_responds_to_its_edge_state_and_trains_refine_layers(n_refine_layers):
    with deterministic_fp64():
        stream, _, inputs = _stream_case(n_refine_layers=n_refine_layers)
        row = 2
        seed = inputs["edge_features"].detach().clone().requires_grad_(True)
        output = stream(**dict(inputs, edge_features=seed))
        gradient = torch.autograd.grad(output[row].sum(), seed)[0]
        perturbed = seed.detach().clone()
        perturbed[row, 0] += 1.0e-3
        changed = stream(**dict(inputs, edge_features=perturbed))

        stream.zero_grad(set_to_none=True)
        stream(**inputs).square().mean().backward()

    assert torch.isfinite(output).all()
    assert gradient[row].norm().item() > 0.0
    assert (changed[row] - output[row].detach()).abs().max().item() > 0.0
    refine_gradients = {
        name: parameter.grad
        for name, parameter in stream.named_parameters()
        if name.startswith("refine_layers.")
    }
    assert bool(refine_gradients) is bool(n_refine_layers)
    for name, grad in refine_gradients.items():
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().max().item() > 0.0, name


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ({"n_refine_layers": -1}, "n_refine_layers"),
        ({"refine_condition": "vector"}, "refine_condition"),
        ({"refine_rank": 0}, "rank"),
        ({"refine_radial_dim": 0}, "radial_dim"),
        ({"refine_edge_chunk_size": 0}, "edge_chunk_size"),
    ],
)
def test_refinement_configuration_fails_closed(override, key):
    options = dict(
        num_types=1,
        node_irreps="1x0e+1x1o",
        edge_irreps="1x0e+1x1o",
        latent_dim=2,
        latent_channels=(2,),
        use_layer_onehot_tp=False,
        edge_one_hot_dim=1,
        so2_fusion_mode="streamed_m_major_ref",
        mole_linear_mode="indexed_ref",
        dtype=torch.float64,
        device="cpu",
        num_experts=1,
        num_shared_experts=1,
    )
    with fp64_default(), pytest.raises(ValueError, match=key):
        TwoStagePairStream(**{**options, **override})


# --- integration into the embeddings -----------------------------------------------


def test_disabled_two_stage_is_bit_exact_with_the_baseline_embedding():
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    with deterministic_fp64():
        torch.manual_seed(20260724)
        baseline = LemMoEV3H0(**options).eval()
        baseline_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20260724)
        disabled = _direct_model(enabled=False)
        assert torch.equal(baseline_rng, torch.random.get_rng_state())
        assert baseline.state_dict().keys() == disabled.state_dict().keys()
        for key, value in baseline.state_dict().items():
            assert torch.equal(value, disabled.state_dict()[key]), key
        reference = baseline(molecule_data(baseline))
        actual = disabled(molecule_data(disabled))
    for key in (_keys.NODE_HAMILTONIAN_KEY, _keys.EDGE_HAMILTONIAN_KEY, _keys.EDGE_OVERLAP_KEY):
        assert torch.equal(reference[key], actual[key]), key


def _assert_trained(model, prefix):
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.requires_grad
    ]
    assert gradients, prefix
    assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients), prefix
    assert any(grad.abs().max().item() > 0.0 for grad in gradients), prefix


def test_enabled_two_stage_embedding_is_equivariant_and_trains_new_parameters():
    with deterministic_fp64():
        torch.manual_seed(20260724)
        model = _direct_model(enabled=True)
        data = molecule_data(model)
        torch.manual_seed(73)
        drift, _ = edge_block_drift(model, data, o3.rand_matrix(dtype=torch.float64))
        output = model(clone_data(data))
        (
            output[_keys.NODE_HAMILTONIAN_KEY].square().mean()
            + output[_keys.EDGE_HAMILTONIAN_KEY].square().mean()
        ).backward()
    assert drift <= 1.0e-9
    _assert_trained(model, "two_stage_pair.")


@pytest.mark.parametrize("mp_cutoff", [None, 1.0], ids=["single_cutoff", "dual_cutoff"])
def test_lem_pair_endpoint_two_stage_with_refine_is_equivariant_and_trainable(mp_cutoff):
    """With a real MP/head split, Stage 2 and the refinement must use full-edge Wigner blocks."""
    options = model_options()
    options.update(
        mp_cutoff=mp_cutoff,
        condition_source="endpoints",
        hb0_hermitian_average=True,
        log_head_input_rms=True,
        pair_refine_enable=True,
        pair_refine_rank=3,
        pair_refine_weight_mode="per_path",
        pair_refine_identity_init=True,
        **TWO_STAGE_OPTIONS,
    )
    with deterministic_fp64():
        torch.manual_seed(20260724)
        model = LemPair(**options).eval()
        data = molecule_data(model)
        if mp_cutoff is not None:
            src, dst = data[_keys.EDGE_INDEX_KEY]
            positions = data[_keys.POSITIONS_KEY]
            mp_rows = int(((positions[src] - positions[dst]).norm(dim=-1) < mp_cutoff).sum())
            assert 0 < mp_rows < src.numel()
        torch.manual_seed(83)
        drift, reference = edge_block_drift(model, data, o3.rand_matrix(dtype=torch.float64))
        model.train()
        model(clone_data(data))[_keys.EDGE_HAMILTONIAN_KEY].sum().backward()
    assert "head_input_rms" in reference
    assert torch.isfinite(reference[_keys.EDGE_HAMILTONIAN_KEY]).all()
    assert drift <= 1.0e-9
    _assert_trained(model, "two_stage_pair.")
    _assert_trained(model, "pair_refine.")


def test_block_ode_two_stage_edge_state_sensitivity_gradient_and_equivariance():
    with deterministic_fp64():
        torch.manual_seed(20260724)
        model = block_ode_model("lem_moe_v3_h0", rme_fusion_init=0.2, **TWO_STAGE_OPTIONS)
        flow, raw, model_data = prepared_flow_batch(model)
        residual_key = _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY
        row = 0
        live = clone_data(model_data)
        seed = live[residual_key].requires_grad_(True)
        output = model(live)
        gradient = torch.autograd.grad(output[_keys.EDGE_HAMILTONIAN_KEY][row].sum(), seed)[0]
        perturbed = clone_data(model_data)
        perturbed[residual_key][row, 0, 0] += 1.0e-3
        changed = model(perturbed)[_keys.EDGE_HAMILTONIAN_KEY][row]

        # Whole-flow covariance on a nonzero, row-specific invariant s-s block state; general
        # irreps covariance is covered at the stream boundary by test_stream_is_equivariant.
        scalar_raw = copy.deepcopy(raw)
        for key in (
            _keys.NODE_H0_BLOCKS_KEY,
            _keys.EDGE_H0_BLOCKS_KEY,
            _keys.NODE_DELTA_HAMIL_BLOCKS_KEY,
            _keys.EDGE_DELTA_HAMIL_BLOCKS_KEY,
        ):
            scalar_raw[key].zero_()
            scalar_raw[key][:, 0, 0] = 0.1
        torch.manual_seed(89)
        rotation = o3.rand_matrix(dtype=torch.float64)
        rotated_raw = copy.deepcopy(scalar_raw)
        rotated_raw[_keys.POSITIONS_KEY] = scalar_raw[_keys.POSITIONS_KEY] @ rotation.T
        t = torch.tensor([0.41], dtype=torch.float64)
        outputs = []
        for record in (scalar_raw, rotated_raw):
            state, _, _ = flow.prepare_batch(copy.deepcopy(record), copy.deepcopy(record), t=t)
            outputs.append(model(state)[_keys.EDGE_HAMILTONIAN_KEY].detach())
        expected = _rotate_canvas_blocks(outputs[0], ao_wigner(model.embedding, rotation))

    assert gradient[row].norm().item() > 0.0
    assert (changed - output[_keys.EDGE_HAMILTONIAN_KEY][row].detach()).abs().max().item() > 0.0
    torch.testing.assert_close(outputs[1], expected, rtol=0.0, atol=1.0e-9)

from __future__ import annotations

from contextlib import contextmanager

import torch
import pytest
from e3nn import o3

from dptb.nn.embedding.two_stage_pair import TwoStagePairStream
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals


@contextmanager
def _deterministic_fp64():
    old_dtype = torch.get_default_dtype()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_deterministic)
        torch.set_default_dtype(old_dtype)


def _complete_directed_edges(count: int) -> torch.Tensor:
    return torch.tensor(
        [(i, j) for i in range(count) for j in range(count) if i != j],
        dtype=torch.long,
    ).T.contiguous()


def _case(*, n_refine_layers=0, tail_gate=False):
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
        [[0.0, 0.0, 0.0], [0.7, 0.2, -0.1], [-0.3, 0.8, 0.4]],
        dtype=torch.float64,
    )
    edge_index = _complete_directed_edges(positions.shape[0])
    src, dst = edge_index
    edge_vector = positions[dst] - positions[src]
    n_edges = edge_index.shape[1]
    inputs = dict(
        latents=torch.randn(n_edges, 4, dtype=torch.float64),
        node_features=torch.randn(positions.shape[0], irreps.dim, dtype=torch.float64),
        node_onehot=torch.ones(positions.shape[0], 1, dtype=torch.float64),
        edge_features=torch.randn(n_edges, irreps.dim, dtype=torch.float64),
        edge_index=edge_index,
        edge_vector=edge_vector,
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


def test_eq13_late_pair_construction_is_equivariant():
    with _deterministic_fp64():
        stream, irreps, inputs = _case()
        reference = stream(**inputs)

        torch.manual_seed(17)
        rotation = o3.rand_matrix(dtype=torch.float64)
        xyz_to_yzx = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        feature_rotation = xyz_to_yzx @ rotation @ xyz_to_yzx.T
        representation = irreps.D_from_matrix(feature_rotation)
        rotated_inputs = dict(inputs)
        rotated_inputs["node_features"] = inputs["node_features"] @ representation.T
        rotated_inputs["edge_features"] = inputs["edge_features"] @ representation.T
        rotated_inputs["edge_vector"] = inputs["edge_vector"] @ rotation.T
        rotated_inputs["wigner_D_all"] = None
        rotated = stream(**rotated_inputs)
        expected = reference @ representation.T
        drift = (rotated - expected).abs().max().item()
        print(f"two_stage_eq13_equivariance_max_abs={drift:.16e}")
        assert drift <= 1.0e-9


def test_eq13_retains_row_specific_edge_state_value_and_gradient():
    with _deterministic_fp64():
        stream, _, inputs = _case()
        row = 2
        seed = inputs["edge_features"].detach().clone().requires_grad_(True)
        differentiable_inputs = dict(inputs, edge_features=seed)
        output = stream(**differentiable_inputs)
        gradient = torch.autograd.grad(output[row].sum(), seed)[0]
        gradient_norm = gradient[row].norm().item()

        perturbed = seed.detach().clone()
        perturbed[row, 0] += 1.0e-3
        changed = stream(**dict(inputs, edge_features=perturbed))
        sensitivity = (changed[row] - output[row].detach()).abs().max().item()
        print(
            "two_stage_eq13_h_t_sensitivity "
            f"max_abs={sensitivity:.16e} grad_row_norm={gradient_norm:.16e}"
        )
        assert sensitivity > 0.0
        assert gradient_norm > 0.0


def test_eq14_two_layer_norm_free_tail_is_equivariant_sensitive_and_finite():
    with _deterministic_fp64():
        stream, irreps, inputs = _case(n_refine_layers=2)
        rms = []

        def capture_rms(_module, _inputs, output):
            rms.append(output.detach().square().mean().sqrt().item())

        handles = [
            layer.register_forward_hook(capture_rms)
            for layer in stream.refine_layers
        ]
        try:
            seed = inputs["edge_features"].detach().clone().requires_grad_(True)
            output = stream(**dict(inputs, edge_features=seed))
        finally:
            for handle in handles:
                handle.remove()
        assert torch.isfinite(output).all()

        row = 2
        gradient = torch.autograd.grad(
            output[row].sum(), seed, retain_graph=True
        )[0]
        gradient_norm = gradient[row].norm().item()
        perturbed = seed.detach().clone()
        perturbed[row, 0] += 1.0e-3
        changed = stream(**dict(inputs, edge_features=perturbed))
        sensitivity = (changed[row] - output[row].detach()).abs().max().item()

        torch.manual_seed(31)
        rotation = o3.rand_matrix(dtype=torch.float64)
        xyz_to_yzx = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        representation = irreps.D_from_matrix(
            xyz_to_yzx @ rotation @ xyz_to_yzx.T
        )
        rotated = stream(
            **dict(
                inputs,
                node_features=inputs["node_features"] @ representation.T,
                edge_features=inputs["edge_features"] @ representation.T,
                edge_vector=inputs["edge_vector"] @ rotation.T,
                wigner_D_all=None,
            )
        )
        expected = output.detach() @ representation.T
        drift = (rotated - expected).abs().max().item()

        stream.zero_grad(set_to_none=True)
        backward_output = stream(**inputs)
        loss = backward_output.square().mean()
        loss.backward()
        gradients = {
            name: parameter.grad
            for name, parameter in stream.named_parameters()
            if name.startswith("refine_layers.")
        }
        assert gradients
        assert all(value is not None for value in gradients.values())
        assert all(torch.isfinite(value).all() for value in gradients.values())
        assert all(value.abs().max().item() > 0.0 for value in gradients.values())

        dynamic_dof = [
            layer.dynamic_dof_per_edge for layer in stream.refine_layers
        ]
        weight_numel = [layer.weight_numel for layer in stream.refine_layers]
        print(
            "two_stage_eq14 "
            f"equivariance_max_abs={drift:.16e} "
            f"h_t_sensitivity={sensitivity:.16e} "
            f"h_t_grad_row_norm={gradient_norm:.16e} "
            f"rms={rms} dynamic_dof={dynamic_dof} "
            f"weight_numel={weight_numel}"
        )
        assert drift <= 1.0e-9
        assert sensitivity > 0.0
        assert gradient_norm > 0.0
        assert len(rms) == 2
        assert all(torch.isfinite(torch.tensor(value)) for value in rms)


def test_optional_tail_gate_is_equivariant_without_statistical_normalization():
    with _deterministic_fp64():
        stream, irreps, inputs = _case(n_refine_layers=2, tail_gate=True)
        reference = stream(**inputs)
        torch.manual_seed(47)
        rotation = o3.rand_matrix(dtype=torch.float64)
        xyz_to_yzx = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        representation = irreps.D_from_matrix(
            xyz_to_yzx @ rotation @ xyz_to_yzx.T
        )
        rotated = stream(
            **dict(
                inputs,
                node_features=inputs["node_features"] @ representation.T,
                edge_features=inputs["edge_features"] @ representation.T,
                edge_vector=inputs["edge_vector"] @ rotation.T,
                wigner_D_all=None,
            )
        )
        drift = (rotated - reference @ representation.T).abs().max().item()
        assert drift <= 1.0e-9
        assert all(
            "norm" not in name.lower()
            for name, _ in stream.refine_layers.named_modules()
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"n_refine_layers": -1}, "non-negative"),
        ({"refine_condition": "vector"}, "scalar_0e"),
        ({"refine_rank": 0}, "rank must be positive"),
        ({"refine_radial_dim": 0}, "radial_dim must be positive"),
        ({"refine_edge_chunk_size": 0}, "edge_chunk_size must be positive"),
    ],
)
def test_refinement_configuration_fails_closed(override, match):
    with _deterministic_fp64(), pytest.raises(ValueError, match=match):
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
        options.update(override)
        TwoStagePairStream(**options)

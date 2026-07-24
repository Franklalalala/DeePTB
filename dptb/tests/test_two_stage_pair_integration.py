from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.build import build_model
from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nn.embedding.lem_pair import LemPair

from test_lem_pair_common import (
    ao_wigner,
    clone_data,
    model_options,
    molecule_data,
    rotate_data,
)
from test_residual_ao_block_ode import (
    _b_flow,
    _b_record,
    _rotate_canvas_blocks,
    _shared_canvas_wigner_d,
)


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


def _direct_model(*, enabled: bool):
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    options.update(
        two_stage_pair_enable=enabled,
        two_stage_pair_refine_layers=2,
        two_stage_pair_refine_rank=3,
        two_stage_pair_refine_radial_dim=3,
        two_stage_pair_refine_edge_chunk_size=2,
    )
    return LemMoEV3H0(**options).eval()


def _flow_model():
    return build_model(
        common_options={
            "basis": {"H": "1s", "C": "1s1p"},
            "overlap": False,
            "dtype": "float64",
            "device": "cpu",
        },
        model_options={
            "embedding": {
                "method": "lem_moe_v3_h0",
                "output_route": "h_b0",
                "h0_init_scope": "both",
                "use_spatial_residual_block_input": True,
                "n_layers": 1,
                "avg_num_neighbors": 2.0,
                "r_max": 4.0,
                "irreps_hidden": "2x0e+2x1o+2x1e+2x2e",
                "env_embed_multiplicity": 2,
                "latent_dim": 6,
                "latent_channels": [6],
                "edge_one_hot_dim": 3,
                "num_experts": 1,
                "num_shared_experts": 1,
                "top_k": 1,
                "universal": True,
                "use_layer_onehot_tp": False,
                "use_out_onehot_tp": False,
                "use_interpolation_out": False,
                "tp_radial_emb": False,
                "mole_linear_mode": "indexed_ref",
                "so2_fusion_mode": "streamed_m_major_ref",
                "rme_fusion_rank": 3,
                "rme_fusion_init": 0.2,
                "use_flow_time_embedding": True,
                "flow_time_condition_edges": True,
                "flow_time_allow_missing": False,
                "require_full_block_edge_coverage": True,
                "two_stage_pair_enable": True,
                "two_stage_pair_refine_layers": 2,
                "two_stage_pair_refine_rank": 3,
                "two_stage_pair_refine_radial_dim": 3,
                "two_stage_pair_refine_edge_chunk_size": 2,
            },
            "prediction": {
                "method": "block_native",
                "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True,
                "scale_type": "no_scale",
            },
        },
        train_options={},
        no_check=False,
    ).to(dtype=torch.float64).eval()


def test_default_disabled_is_bit_exact_and_does_not_construct_a_module():
    with _deterministic_fp64():
        options = model_options()
        options.pop("mp_avg_num_neighbors")
        torch.manual_seed(20260724)
        baseline = LemMoEV3H0(**options).eval()
        baseline_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20260724)
        disabled = _direct_model(enabled=False)
        disabled_rng = torch.random.get_rng_state().clone()

        assert torch.equal(baseline_rng, disabled_rng)
        assert baseline.state_dict().keys() == disabled.state_dict().keys()
        assert all(
            torch.equal(baseline.state_dict()[name], disabled.state_dict()[name])
            for name in baseline.state_dict()
        )
        assert disabled.two_stage_pair is None
        assert all(
            not name.startswith("two_stage_pair.")
            for name, _ in disabled.named_modules()
        )

        reference = baseline(molecule_data(baseline))
        actual = disabled(molecule_data(disabled))
        for key in (
            _keys.NODE_HAMILTONIAN_KEY,
            _keys.EDGE_HAMILTONIAN_KEY,
            _keys.EDGE_OVERLAP_KEY,
        ):
            assert torch.equal(reference[key], actual[key])


def test_enabled_end_to_end_is_equivariant_and_all_new_parameters_receive_gradients():
    with _deterministic_fp64():
        torch.manual_seed(20260724)
        model = _direct_model(enabled=True)
        data = molecule_data(model)
        reference = model(clone_data(data))
        torch.manual_seed(73)
        rotation = o3.rand_matrix(dtype=torch.float64)
        rotated = model(rotate_data(data, rotation))
        d_ao = ao_wigner(model, rotation)
        expected = torch.einsum(
            "ij,njk,lk->nil",
            d_ao,
            reference[_keys.EDGE_HAMILTONIAN_KEY],
            d_ao,
        )
        drift = (
            rotated[_keys.EDGE_HAMILTONIAN_KEY] - expected
        ).abs().max().item()

        model.zero_grad(set_to_none=True)
        output = model(clone_data(data))
        loss = (
            output[_keys.NODE_HAMILTONIAN_KEY].square().mean()
            + output[_keys.EDGE_HAMILTONIAN_KEY].square().mean()
        )
        loss.backward()
        new_gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("two_stage_pair.") and parameter.requires_grad
        }
        assert new_gradients
        assert all(value is not None for value in new_gradients.values())
        assert all(torch.isfinite(value).all() for value in new_gradients.values())
        assert all(value.abs().max().item() > 0.0 for value in new_gradients.values())
        costs = [
            (layer.dynamic_dof_per_edge, layer.weight_numel)
            for layer in model.two_stage_pair.refine_layers
        ]
        print(
            "two_stage_end_to_end "
            f"equivariance_max_abs={drift:.16e} "
            f"refine_costs={costs}"
        )
        assert drift <= 1.0e-9


def test_lem_pair_two_stage_endpoint_combination_is_equivariant():
    with _deterministic_fp64():
        options = model_options()
        options.update(
            condition_source="endpoints",
            two_stage_pair_enable=True,
            two_stage_pair_refine_layers=2,
            two_stage_pair_refine_rank=3,
            two_stage_pair_refine_radial_dim=3,
            two_stage_pair_refine_edge_chunk_size=2,
        )
        torch.manual_seed(20260724)
        model = LemPair(**options).eval()
        data = molecule_data(model)
        reference = model(clone_data(data))
        torch.manual_seed(79)
        rotation = o3.rand_matrix(dtype=torch.float64)
        rotated = model(rotate_data(data, rotation))
        d_ao = ao_wigner(model, rotation)
        expected = torch.einsum(
            "ij,njk,lk->nil",
            d_ao,
            reference[_keys.EDGE_HAMILTONIAN_KEY],
            d_ao,
        )
        drift = (
            rotated[_keys.EDGE_HAMILTONIAN_KEY] - expected
        ).abs().max().item()
        print(f"two_stage_endpoint_equivariance_max_abs={drift:.16e}")
        assert drift <= 1.0e-9


def _prepared_flow_pair(model):
    flow = _b_flow(model.idp, dtype=torch.float64)
    raw, _, _ = _b_record(model.idp, dtype=torch.float64, seed=31)
    model_data, _, _ = flow.prepare_batch(
        copy.deepcopy(raw),
        copy.deepcopy(raw),
        t=torch.tensor([0.41], dtype=torch.float64),
    )
    return flow, raw, model_data


def test_block_ode_flow_edge_state_sensitivity_gradient_and_equivariance():
    with _deterministic_fp64():
        torch.manual_seed(20260724)
        model = _flow_model()
        flow, raw, model_data = _prepared_flow_pair(model)
        residual_key = _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY
        seed = model_data[residual_key].detach().clone().requires_grad_(True)
        live = copy.deepcopy(model_data)
        live[residual_key] = seed
        output = model(live)
        row = 0
        gradient = torch.autograd.grad(
            output[_keys.EDGE_HAMILTONIAN_KEY][row].sum(),
            seed,
        )[0]
        gradient_norm = gradient[row].norm().item()

        perturbed_seed = seed.detach().clone()
        perturbed_seed[row, 0, 0] += 1.0e-3
        perturbed = copy.deepcopy(model_data)
        perturbed[residual_key] = perturbed_seed
        changed = model(perturbed)
        sensitivity = (
            changed[_keys.EDGE_HAMILTONIAN_KEY][row]
            - output[_keys.EDGE_HAMILTONIAN_KEY][row].detach()
        ).abs().max().item()

        # Full flow equivariance uses a nonzero, row-specific invariant s-s
        # block state.  General irreps covariance is covered at the two-stage
        # boundary by test_eq14_two_layer_norm_free_tail_is_equivariant...
        scalar_raw = copy.deepcopy(raw)
        for key in (
            _keys.NODE_H0_BLOCKS_KEY,
            _keys.EDGE_H0_BLOCKS_KEY,
            _keys.NODE_DELTA_HAMIL_BLOCKS_KEY,
            _keys.EDGE_DELTA_HAMIL_BLOCKS_KEY,
        ):
            scalar_raw[key].zero_()
            scalar_raw[key][:, 0, 0] = 0.1
        scalar_data, _, _ = flow.prepare_batch(
            copy.deepcopy(scalar_raw),
            copy.deepcopy(scalar_raw),
            t=torch.tensor([0.41], dtype=torch.float64),
        )
        scalar_output = model(scalar_data)

        torch.manual_seed(89)
        rotation = o3.rand_matrix(dtype=torch.float64)
        xyz_to_yzx = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        d_ao = _shared_canvas_wigner_d(
            xyz_to_yzx @ rotation @ xyz_to_yzx.T
        )
        rotated_raw = copy.deepcopy(scalar_raw)
        rotated_raw[_keys.POSITIONS_KEY] = (
            scalar_raw[_keys.POSITIONS_KEY] @ rotation.T
        )
        rotated_data, _, _ = flow.prepare_batch(
            copy.deepcopy(rotated_raw),
            copy.deepcopy(rotated_raw),
            t=torch.tensor([0.41], dtype=torch.float64),
        )
        rotated_output = model(rotated_data)
        expected = _rotate_canvas_blocks(
            scalar_output[_keys.EDGE_HAMILTONIAN_KEY].detach(), d_ao
        )
        drift = (
            rotated_output[_keys.EDGE_HAMILTONIAN_KEY] - expected
        ).abs().max().item()
        print(
            "two_stage_block_ode "
            f"sensitivity={sensitivity:.16e} "
            f"gradient_row_norm={gradient_norm:.16e} "
            f"equivariance_max_abs={drift:.16e}"
        )
        assert sensitivity > 0.0
        assert gradient_norm > 0.0
        assert drift <= 1.0e-9
        assert flow._strict_certification_mode == "always"

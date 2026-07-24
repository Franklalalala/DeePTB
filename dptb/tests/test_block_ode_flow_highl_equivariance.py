from __future__ import annotations

import copy
from contextlib import contextmanager

import pytest
import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.build import build_model

from test_residual_ao_block_ode import (
    _b_flow,
    _b_record,
    _rotate_canvas_blocks,
    _shared_canvas_wigner_d,
)


@contextmanager
def _deterministic_fp64():
    previous_dtype = torch.get_default_dtype()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_default_dtype(previous_dtype)


def _flow_model(*, two_stage_pair_enable: bool):
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
                "two_stage_pair_enable": two_stage_pair_enable,
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


def _rotate_flow_record(raw, rotation, d_ao):
    rotated = copy.deepcopy(raw)
    rotated[_keys.POSITIONS_KEY] = raw[_keys.POSITIONS_KEY] @ rotation.T
    rotated[_keys.CELL_KEY] = raw[_keys.CELL_KEY] @ rotation.T
    for key in (
        _keys.NODE_H0_BLOCKS_KEY,
        _keys.EDGE_H0_BLOCKS_KEY,
        _keys.NODE_DELTA_HAMIL_BLOCKS_KEY,
        _keys.EDGE_DELTA_HAMIL_BLOCKS_KEY,
    ):
        rotated[key] = _rotate_canvas_blocks(raw[key], d_ao)
    return rotated


@pytest.mark.parametrize("two_stage_pair_enable", [False, True], ids=["off", "on"])
def test_block_ode_flow_highl_state_is_equivariant_when_ao_blocks_rotate(
    two_stage_pair_enable: bool,
):
    with _deterministic_fp64():
        torch.manual_seed(20260724)
        model = _flow_model(two_stage_pair_enable=two_stage_pair_enable)
        flow = _b_flow(model.idp, dtype=torch.float64)
        raw, _, _ = _b_record(model.idp, dtype=torch.float64, seed=37)
        time = torch.tensor([0.41], dtype=torch.float64)
        base_data, _, _ = flow.prepare_batch(
            copy.deepcopy(raw),
            copy.deepcopy(raw),
            t=time,
        )

        # C has a 1p shell, and these rows/columns are genuinely populated.
        # This rules out a scalar-only probe that would make D B D^T trivial.
        assert (
            base_data[_keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY][..., 1:, :]
            .abs()
            .max()
            .item()
            > 0.0
        )
        assert (
            base_data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY][..., 1:, :]
            .abs()
            .max()
            .item()
            > 0.0
        )
        init_layer = model.embedding.init_layer
        projector_inputs = {"node": [], "edge": []}
        handles = [
            init_layer.node_projector.register_forward_pre_hook(
                lambda _module, args: projector_inputs["node"].append(
                    args[0].detach().clone()
                )
            ),
            init_layer.edge_projector.register_forward_pre_hook(
                lambda _module, args: projector_inputs["edge"].append(
                    args[0].detach().clone()
                )
            ),
        ]
        try:
            base_output = model(base_data)
            base_projector_inputs = {
                key: values.pop() for key, values in projector_inputs.items()
            }

            torch.manual_seed(101)
            rotation = o3.rand_matrix(dtype=torch.float64)
            xyz_to_yzx = torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                dtype=torch.float64,
            )
            rme_rotation = xyz_to_yzx @ rotation @ xyz_to_yzx.T
            d_ao = _shared_canvas_wigner_d(rme_rotation)
            rotated_raw = _rotate_flow_record(raw, rotation, d_ao)
            rotated_data, _, _ = flow.prepare_batch(
                copy.deepcopy(rotated_raw),
                copy.deepcopy(rotated_raw),
                t=time,
            )
            rotated_output = model(rotated_data)
            rotated_projector_inputs = {
                key: values.pop() for key, values in projector_inputs.items()
            }
        finally:
            for handle in handles:
                handle.remove()

        # G-FIX2: the actual H0 tensors presented to the sorted-irrep linears
        # must already transform in that representation.  This is the boundary
        # that drifted by O(1) before the raw->sorted fix.
        d_h0 = init_layer.h0_irreps.D_from_matrix(rme_rotation)
        h0_boundary_drifts = {}
        for label in ("node", "edge"):
            expected = torch.einsum(
                "ij,nj->ni", d_h0, base_projector_inputs[label]
            )
            h0_boundary_drifts[label] = (
                rotated_projector_inputs[label] - expected
            ).abs().max().item()
        print(
            "block_ode_flow_highl_h0_boundary "
            f"two_stage_pair_enable={two_stage_pair_enable} "
            f"node_max_abs={h0_boundary_drifts['node']:.16e} "
            f"edge_max_abs={h0_boundary_drifts['edge']:.16e}"
        )
        assert max(h0_boundary_drifts.values()) <= 2.0e-15, h0_boundary_drifts

        drifts = {}
        for label, key in (
            ("node", _keys.NODE_HAMILTONIAN_KEY),
            ("edge", _keys.EDGE_HAMILTONIAN_KEY),
        ):
            expected = _rotate_canvas_blocks(base_output[key], d_ao)
            drifts[label] = (rotated_output[key] - expected).abs().max().item()
        print(
            "block_ode_flow_highl "
            f"two_stage_pair_enable={two_stage_pair_enable} "
            f"node_max_abs={drifts['node']:.16e} "
            f"edge_max_abs={drifts['edge']:.16e}"
        )
        assert max(drifts.values()) <= 1.0e-9, drifts

"""H0 input space of the H0 init layer: dataset batches carry AO-product H0 and get the
``h0_ao_cg`` conversion; the block-ODE flows write coupled RME and declare it with
``_keys.H0_COUPLED_RME_KEY``, which leaves only the irreps sort."""
from __future__ import annotations

import copy

import pytest
import torch

from dptb.data import _keys
from dptb.nn.build import build_model
from dptb.tests.block_ode_fixtures import _b_flow, _b_record


def _model():
    torch.manual_seed(20260924)
    return build_model(
        common_options={"basis": {"H": "1s", "C": "1s1p"}, "overlap": False, "dtype": "float64", "device": "cpu"},
        model_options={
            "embedding": {
                "method": "lem_moe_v3_h0", "output_route": "h_b0", "h0_init_scope": "both",
                "use_spatial_residual_block_input": True, "n_layers": 1, "avg_num_neighbors": 2.0,
                "r_max": 4.0, "irreps_hidden": "2x0e+2x1o+2x1e+2x2e", "env_embed_multiplicity": 2,
                "latent_dim": 6, "latent_channels": [6], "edge_one_hot_dim": 3, "num_experts": 1,
                "num_shared_experts": 1, "top_k": 1, "universal": True, "use_layer_onehot_tp": False,
                "use_out_onehot_tp": False, "use_interpolation_out": False, "tp_radial_emb": False,
                "mole_linear_mode": "indexed_ref", "so2_fusion_mode": "streamed_m_major_ref",
                "rme_fusion_rank": 3, "rme_fusion_init": 0.2, "use_flow_time_embedding": True,
                "flow_time_condition_edges": True, "flow_time_allow_missing": False,
                "require_full_block_edge_coverage": True,
            },
            "prediction": {"method": "block_native", "block_decoder": "expansion_cg",
                           "blockwise_hamiltonian": True, "scale_type": "no_scale"},
        },
        train_options={},
        no_check=False,
    ).to(dtype=torch.float64).eval()


def test_block_ode_flow_declares_coupled_h0():
    model = _model()
    flow = _b_flow(model.idp, dtype=torch.float64)
    raw, _, _ = _b_record(model.idp, dtype=torch.float64, seed=5)
    data, _, _ = flow.prepare_batch(copy.deepcopy(raw), copy.deepcopy(raw), t=torch.tensor([0.3], dtype=torch.float64))
    assert bool(data[_keys.H0_COUPLED_RME_KEY])


def test_ao_product_batch_equals_the_same_h0_supplied_coupled():
    """Bit for bit: an unmarked batch is converted exactly as before (AO product -> coupled -> sort),
    and supplying that coupled H0 with the marker gives the identical output."""
    model = _model()
    layer = model.embedding.init_layer
    assert layer.h0_ao_cg
    flow = _b_flow(model.idp, dtype=torch.float64)
    raw, _, _ = _b_record(model.idp, dtype=torch.float64, seed=5)
    data, _, _ = flow.prepare_batch(copy.deepcopy(raw), copy.deepcopy(raw), t=torch.tensor([0.3], dtype=torch.float64))
    generator = torch.Generator().manual_seed(11)
    atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
    bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
    node_ao = layer._mask_node_source(
        torch.randn(data[_keys.NODE_H0_KEY].shape, generator=generator, dtype=torch.float64), atom_type)
    edge_ao = layer._mask_edge_source(
        torch.randn(data[_keys.EDGE_H0_KEY].shape, generator=generator, dtype=torch.float64), bond_type)

    plain = copy.deepcopy(data)
    plain.pop(_keys.H0_COUPLED_RME_KEY)
    plain[_keys.NODE_H0_KEY], plain[_keys.EDGE_H0_KEY] = node_ao, edge_ao

    change_of_basis = layer._h0_cg_change_of_basis.to(dtype=torch.float64)
    coupled = copy.deepcopy(data)
    coupled[_keys.NODE_H0_KEY] = torch.einsum("kc,nc->nk", change_of_basis, node_ao)
    coupled[_keys.EDGE_H0_KEY] = torch.einsum("kc,nc->nk", change_of_basis, edge_ao)

    with torch.no_grad():
        out_plain = model(plain)
        out_coupled = model(coupled)
        misread = model(dict(plain, **{_keys.H0_COUPLED_RME_KEY: torch.ones((), dtype=torch.bool)}))
    for key in (_keys.NODE_HAMILTONIAN_KEY, _keys.EDGE_HAMILTONIAN_KEY):
        assert torch.equal(out_plain[key], out_coupled[key]), key
    assert not torch.equal(out_plain[_keys.NODE_HAMILTONIAN_KEY], misread[_keys.NODE_HAMILTONIAN_KEY])


def test_collated_flags_must_agree():
    from dptb.nn.embedding.lem_moe_v3_h0_helpers import _h0_is_coupled_rme

    key = _keys.H0_COUPLED_RME_KEY
    assert _h0_is_coupled_rme({key: torch.tensor([True, True])})
    assert not _h0_is_coupled_rme({key: torch.tensor([False, False])})
    with pytest.raises(ValueError):
        _h0_is_coupled_rme({key: torch.tensor([True, False])})


def test_ao_h0_written_for_model_input_drops_the_declaration():
    from dptb.postprocess.elec_struc_cal import ElecStruCal

    key = _keys.H0_COUPLED_RME_KEY
    data = {_keys.NODE_FEATURES_KEY: torch.zeros(2, 3), _keys.EDGE_FEATURES_KEY: torch.zeros(4, 3),
            key: torch.ones((), dtype=torch.bool)}
    ElecStruCal._copy_feature_h0_to_model_input(data)
    assert key not in data
    torch.testing.assert_close(data[_keys.NODE_H0_KEY], data[_keys.NODE_FEATURES_KEY])


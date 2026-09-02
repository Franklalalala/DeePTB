from __future__ import annotations

import pytest
import torch

from dptb.data import _keys
from dptb.nn.build import build_model
from dptb.nn.embedding.lem_moe_v3_prior_2b import resolve_prior_2b_keys


def _embedding(only2b: bool, **extra):
    cfg = {
        "method": "lem_moe_v3_prior_2b",
        "only2b": only2b,
        "prior_kind": "na_cf",
        "prior_merge_mode": "concat",
        "prior_init_scope": "both",
        "n_layers": 1,
        "avg_num_neighbors": 2.0,
        "r_max": 4.0,
        "irreps_hidden": "4x0e+4x1o+4x1e+4x2e",
        "env_embed_multiplicity": 2,
        "latent_dim": 8,
        "latent_channels": [8],
        "edge_one_hot_dim": 4,
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
        "equivariant_norm_type": "none",
    }
    cfg.update(extra)
    return cfg


def _build(only2b: bool, **extra):
    return build_model(
        common_options={
            "basis": {"H": "1s", "O": "1s1p"},
            "overlap": False,
            "dtype": "float32",
            "device": "cpu",
        },
        model_options={
            "embedding": _embedding(only2b, **extra),
            "prediction": {"method": "e3tb", "scale_type": "no_scale"},
        },
        train_options={},
        no_check=False,
    )


def _data(model):
    h = model.idp.chemical_symbol_to_type["H"]
    o = model.idp.chemical_symbol_to_type["O"]
    rme = int(model.idp.reduced_matrix_element)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]],
            dtype=torch.float32,
        ),
        _keys.EDGE_INDEX_KEY: edge_index,
        _keys.ATOM_TYPE_KEY: torch.tensor([[h], [o]], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.tensor(
            [model.idp.bond_to_type["H-O"], model.idp.bond_to_type["O-H"]],
            dtype=torch.long,
        ),
        _keys.NODE_P23_KEY: torch.randn(2, rme, dtype=torch.float32),
        _keys.EDGE_P2_KEY: torch.randn(2, rme, dtype=torch.float32),
    }


def test_resolve_na_cf_keys():
    assert resolve_prior_2b_keys("na_cf") == (
        _keys.NODE_P23_KEY,
        _keys.EDGE_P2_KEY,
    )


def test_stage1_concat_into_first_so2_layer_and_uses_p():
    model = _build(only2b=True)
    emb = model.embedding
    geo_dim = int(emb.h0_init.irreps_out.dim)
    assert emb.layers[0].irreps_in.dim == 2 * geo_dim
    assert emb.concat_irreps.dim == 2 * geo_dim

    data = _data(model)
    out = model(data)
    rme = int(model.idp.reduced_matrix_element)
    assert out[_keys.NODE_FEATURES_KEY].shape == (2, rme)
    assert out[_keys.EDGE_FEATURES_KEY].shape[1] == rme

    loss = out[_keys.NODE_FEATURES_KEY].abs().mean() + out[
        _keys.EDGE_FEATURES_KEY
    ].abs().mean()
    loss.backward()
    proj_grad = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in emb.h0_init.node_projector.parameters()
    )
    assert proj_grad, "stage 1 must train the P RME projector"
    layer_grad = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in emb.layers[0].parameters()
    )
    assert layer_grad, "stage 1 SO2 layer must see concat(P) and train"


def test_stage2_freezes_two_b_skip_but_still_uses_p():
    model = _build(only2b=False)
    emb = model.embedding
    assert all(not p.requires_grad for p in emb.two_b_out_node.parameters())
    assert all(not p.requires_grad for p in emb.h0_init.node_projector.parameters())
    assert all(not p.requires_grad for p in emb.h0_init.base_init.parameters())
    assert any(p.requires_grad for p in emb.layers[0].parameters())
    assert any(p.requires_grad for p in emb.out_node.parameters())

    data = _data(model)
    out = model(data)
    loss = out[_keys.NODE_FEATURES_KEY].square().mean()
    loss.backward()
    for p in emb.h0_init.node_projector.parameters():
        assert p.grad is None or float(p.grad.abs().sum()) == 0
    layer_grad = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in emb.layers[0].parameters()
    )
    assert layer_grad


def test_refuses_replace_merge():
    with pytest.raises(ValueError, match="concat"):
        _build(only2b=True, prior_merge_mode="replace")

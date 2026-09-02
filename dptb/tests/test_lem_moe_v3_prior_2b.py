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
        "n_layers": 2,
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
    torch.manual_seed(0)
    return {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]],
            dtype=torch.float32,
        ),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.ATOM_TYPE_KEY: torch.tensor([[h], [o]], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.tensor(
            [model.idp.bond_to_type["H-O"], model.idp.bond_to_type["O-H"]],
            dtype=torch.long,
        ),
        _keys.NODE_P23_KEY: torch.randn(2, rme, dtype=torch.float32),
        _keys.EDGE_P2_KEY: torch.randn(2, rme, dtype=torch.float32),
    }


def _has_grad(module):
    return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())


def _params_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_resolve_na_cf_keys():
    assert resolve_prior_2b_keys("na_cf") == (_keys.NODE_P23_KEY, _keys.EDGE_P2_KEY)


def test_first_layer_takes_concat_and_mirrors_base_ffn_rule():
    emb = _build(only2b=True, ffn_hidden_factor=2.0).embedding
    geo_dim = int(emb.h0_init.irreps_out.dim)
    assert emb.layers[0].irreps_in.dim == 2 * geo_dim
    # base rule: layer 0 of a 2-layer stack gets the node FFN when ffn_hidden_factor > 1
    assert emb.layers[0].node_ffn is not None
    assert emb.layers[1].node_ffn is None


def test_stage1_trains_only_the_two_b_branch():
    model = _build(only2b=True)
    emb = model.embedding
    out = model(_data(model))
    rme = int(model.idp.reduced_matrix_element)
    assert out[_keys.NODE_FEATURES_KEY].shape == (2, rme)
    assert out[_keys.EDGE_FEATURES_KEY].shape[1] == rme
    assert "mean_max_prob" in out
    (out[_keys.NODE_FEATURES_KEY].abs().mean() + out[_keys.EDGE_FEATURES_KEY].abs().mean()).backward()
    for module in emb._two_b_modules():
        assert _has_grad(module), "stage 1 must train the 2b branch"
    assert not _has_grad(emb.layers[0]), "stage 1 must not touch the GNN"
    assert not _has_grad(emb.h0_init.base_init)
    assert not _has_grad(emb.h0_init.node_projector)
    assert not _has_grad(emb.out_node)


def test_stage2_freezes_two_b_and_trains_gnn_init():
    model = _build(only2b=False)
    emb = model.embedding
    for module in emb._two_b_modules():
        assert all(not p.requires_grad for p in module.parameters())
    assert all(p.requires_grad for p in emb.h0_init.base_init.parameters())
    assert all(p.requires_grad for p in emb.h0_init.node_projector.parameters())
    out = model(_data(model))
    out[_keys.NODE_FEATURES_KEY].square().mean().backward()
    assert _has_grad(emb.layers[0])
    assert _has_grad(emb.h0_init.base_init), "stage 2 keeps the GNN InitLayer trainable"
    assert _has_grad(emb.h0_init.node_projector)
    for module in emb._two_b_modules():
        assert all(p.grad is None for p in module.parameters())


def test_stage2_adds_frozen_two_b_to_gnn_output():
    model = _build(only2b=False)
    emb = model.embedding
    data = _data(model)
    with torch.no_grad():
        full = model(dict(data))[_keys.NODE_FEATURES_KEY].clone()
        emb.only2b = True
        two_b = model(dict(data))[_keys.NODE_FEATURES_KEY].clone()
        emb.only2b = False
    assert not torch.allclose(full, two_b)


def test_stage2_seeds_gnn_init_from_stage1_checkpoint_once():
    stage1 = _build(only2b=True)
    e1 = stage1.embedding
    with torch.no_grad():
        for p in e1.two_b_init.parameters():
            p.add_(torch.randn_like(p))
        for p in e1.two_b_node_proj.parameters():
            p.add_(torch.randn_like(p))
    assert not _params_equal(e1.two_b_init, e1.h0_init.base_init)
    ckpt = stage1.state_dict()

    stage2 = _build(only2b=False)
    e2 = stage2.embedding
    stage2.load_state_dict(ckpt)
    assert bool(e2.two_b_gnn_seeded)
    assert _params_equal(e2.two_b_init, e2.h0_init.base_init)
    assert _params_equal(e2.two_b_node_proj, e2.h0_init.node_projector)
    assert _params_equal(e2.two_b_edge_proj, e2.h0_init.edge_projector)

    # a later stage-2 restart must not re-seed
    with torch.no_grad():
        for p in e2.h0_init.base_init.parameters():
            p.add_(1.0)
    ckpt2 = stage2.state_dict()
    stage3 = _build(only2b=False)
    stage3.load_state_dict(ckpt2)
    e3 = stage3.embedding
    assert bool(e3.two_b_gnn_seeded)
    assert not _params_equal(e3.two_b_init, e3.h0_init.base_init)


def test_refuses_replace_merge_and_partial_scope():
    with pytest.raises(ValueError, match="concat"):
        _build(only2b=True, prior_merge_mode="replace")
    with pytest.raises(ValueError, match="both"):
        _build(only2b=True, prior_init_scope="node")

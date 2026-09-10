"""Prior-activate must use an edge descriptor without changing legacy routing."""

import torch
from dptb.tests.test_lem_moe_v3_prior_2b import _build, _data


def test_prior2b_pa_uses_descriptor_and_backpropagates():
    m = _build(
        only2b=False,
        edge_router_prior_activate=True,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        so2_fusion_mode="staged",
    )
    e = m.embedding
    assert e.edge_router_prior_activate
    assert e.router.net[0].weight.shape[1] > 4
    out = m(_data(m))
    assert out["edge_moe_num_route_tokens"] > 0
    (
        out["node_features"].square().mean() + out["edge_features"].square().mean()
    ).backward()
    assert any(p.grad is not None for p in e.layers.parameters())


def test_pa_s1_s2_checkpoint_compatible():
    options = dict(
        edge_router_prior_activate=True,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        so2_fusion_mode="staged",
    )
    s1 = _build(True, **options)
    s2 = _build(False, **options)
    s2.load_state_dict(s1.state_dict(), strict=True)
    assert (
        s2.embedding.router.net[0].weight.shape
        == s1.embedding.router.net[0].weight.shape
    )
    assert s2.embedding.edge_router_prior_activate


def test_plain_route_retains_graph_router():
    m = _build(False)
    assert not getattr(m.embedding, "edge_router_prior_activate", False)
    m.load_state_dict(_build(False).state_dict(), strict=True)


def test_pa_rebuilt_layer_respects_environment_guard(monkeypatch):
    import os

    monkeypatch.setenv("DPTB_SO2_FUSION_MODE", "streamed_m_major_ref")
    m = _build(
        False,
        edge_router_prior_activate=True,
        num_experts=4,
        top_k=2,
        so2_fusion_mode="staged",
    )
    assert os.environ["DPTB_SO2_FUSION_MODE"] == "streamed_m_major_ref"
    assert all(
        getattr(mod, "so2_fusion_mode", "staged") in ("staged", "streamed_m_major_cueq")
        for mod in m.embedding.modules()
    )

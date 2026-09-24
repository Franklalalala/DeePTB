"""Serial prior-2b embeddings: stage 1 trains the two-body branch alone, stage 2 freezes it and trains the GNN
on top.  Checkpoint seeding, the H0 AO->irreps conversion, prior-activate routing, Switch top-1 and flow time."""
import copy
import os
from types import SimpleNamespace

import pytest
import torch

from dptb.data import _keys
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.lem_moe_v3_edge import LemMoEV3EdgeH0
from dptb.nn.embedding.lem_moe_v3_h0_helpers import H0InitLayer
from dptb.nn.embedding.lem_moe_v3_prior_2b import _Prior2bMixin
from dptb.tests.model_helpers import _build, _data, _has_grad, _params_equal

EDGE = dict(method="lem_moe_v3_edge_prior_2b", num_experts=4, top_k=2, num_shared_experts=1,
            edge_router_prior_activate=False, edge_moe_compact_min_edges=0)
PA = dict(edge_router_prior_activate=True, num_experts=4, num_shared_experts=1, top_k=2, so2_fusion_mode="staged")
SWITCH = dict(method="lem_moe_v3_edge_prior_2b", num_experts=4, top_k=1, num_shared_experts=0,
              edge_router_prior_activate=True, edge_router_top1_mode="switch", so2_fusion_mode="streamed_m_major_cueq",
              edge_moe_compact_min_edges=0)
PRIOR_KEYS = {"h0": ("node_h0", "edge_h0"), "na_cf": ("node_p23", "edge_p2")}


def _prior_data(model, kind):
    data = _data(model)
    if kind == "h0":
        data["node_h0"] = data.pop("node_p23")
        data["edge_h0"] = data.pop("edge_p2")
    return data


def _clone(data):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}


def _frozen(model):
    return {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}


def _unchanged(model, frozen):
    params = dict(model.named_parameters())
    return bool(frozen) and all(torch.equal(params[name], value) for name, value in frozen.items())


# ---------------------------------------------------------------------------
# graph-level router (lem_moe_v3_prior_2b)
# ---------------------------------------------------------------------------
def test_concat_merge_doubles_the_first_layer_input():
    emb = _build(only2b=True, ffn_hidden_factor=2.0).embedding
    assert emb.layers[0].irreps_in.dim == 2 * int(emb.h0_init.irreps_out.dim)
    # layer 0 of a 2-layer stack gets the node FFN when ffn_hidden_factor > 1, as in the base model
    assert emb.layers[0].node_ffn is not None and emb.layers[1].node_ffn is None


def test_stage1_trains_only_the_two_body_branch():
    model = _build(only2b=True)
    emb = model.embedding
    out = model(_data(model))
    rme = int(model.idp.reduced_matrix_element)
    assert out[_keys.NODE_FEATURES_KEY].shape == (2, rme) and out[_keys.EDGE_FEATURES_KEY].shape[1] == rme
    (out[_keys.NODE_FEATURES_KEY].abs().mean() + out[_keys.EDGE_FEATURES_KEY].abs().mean()).backward()
    assert all(_has_grad(module) for module in emb._two_b_modules())
    for gnn in (emb.layers[0], emb.h0_init.base_init, emb.h0_init.node_projector, emb.out_node):
        assert not _has_grad(gnn)


def test_stage2_freezes_the_two_body_branch_and_trains_the_gnn():
    model = _build(only2b=False)
    emb = model.embedding
    assert not any(p.requires_grad for module in emb._two_b_modules() for p in module.parameters())
    data = _data(model)
    with torch.no_grad():
        emb.only2b = True
        two_body = model(_clone(data))[_keys.NODE_FEATURES_KEY]
        emb.only2b = False
    out = model(_clone(data))
    assert not torch.allclose(out[_keys.NODE_FEATURES_KEY].detach(), two_body)
    out[_keys.NODE_FEATURES_KEY].square().mean().backward()
    assert _has_grad(emb.layers[0]) and _has_grad(emb.h0_init.base_init) and _has_grad(emb.h0_init.node_projector)
    assert all(p.grad is None for module in emb._two_b_modules() for p in module.parameters())


def test_stage1_checkpoint_seeds_the_stage2_gnn_init_once():
    stage1 = _build(only2b=True)
    with torch.no_grad():
        for p in [*stage1.embedding.two_b_init.parameters(), *stage1.embedding.two_b_node_proj.parameters()]:
            p.add_(torch.randn_like(p))
    assert not _params_equal(stage1.embedding.two_b_init, stage1.embedding.h0_init.base_init)
    stage2 = _build(only2b=False)
    stage2.load_state_dict(stage1.state_dict())
    e2 = stage2.embedding
    assert bool(e2.two_b_gnn_seeded)
    assert _params_equal(e2.two_b_init, e2.h0_init.base_init)
    assert _params_equal(e2.two_b_node_proj, e2.h0_init.node_projector)
    assert _params_equal(e2.two_b_edge_proj, e2.h0_init.edge_projector)
    # a stage-2 restart keeps its trained GNN init instead of re-seeding it
    with torch.no_grad():
        for p in e2.h0_init.base_init.parameters():
            p.add_(1.0)
    restart = _build(only2b=False)
    restart.load_state_dict(stage2.state_dict())
    assert bool(restart.embedding.two_b_gnn_seeded)
    assert not _params_equal(restart.embedding.two_b_init, restart.embedding.h0_init.base_init)


@pytest.mark.parametrize("option, value, key", [
    ("prior_merge_mode", "replace", "concat"),
    ("prior_init_scope", "node", "both"),
    ("prior_kind", "p3", "prior_kind"),
])
def test_rejects_unsupported_prior_options(option, value, key):
    with pytest.raises(ValueError, match=key):
        _build(only2b=True, **{option: value})


# ---------------------------------------------------------------------------
# edge router (lem_moe_v3_edge_prior_2b)
# ---------------------------------------------------------------------------
def test_edge_serial_stages_train_the_right_branch_and_keep_the_frozen_two_body_output():
    s1 = _build(True, prior_kind="h0", **EDGE)
    assert isinstance(s1.embedding, LemMoEV3EdgeH0)
    out = s1(_prior_data(s1, "h0"))
    (out["node_features"].square().mean() + out["edge_features"].square().mean()).backward()
    assert _has_grad(s1.embedding.two_b_out_node) and not _has_grad(s1.embedding.layers[0])
    assert out["edge_moe_num_route_tokens"].item() == 2

    s2 = _build(False, prior_kind="h0", **EDGE)
    s2.load_state_dict(s1.state_dict(), strict=True)
    assert _params_equal(s1.embedding.two_b_init, s2.embedding.h0_init.base_init)
    assert bool(s2.embedding.two_b_gnn_seeded)
    frozen, data = _frozen(s2), _prior_data(s2, "h0")

    def two_body_output():
        with torch.no_grad():
            s2.embedding.only2b = True
            result = s2(_clone(data))
            s2.embedding.only2b = False
        return result["node_features"].clone(), result["edge_features"].clone()

    before = two_body_output()
    out = s2(_clone(data))
    (out["node_features"].square().mean() + out["edge_features"].square().mean()).backward()
    assert _has_grad(s2.embedding.layers[0]) and _has_grad(s2.embedding.router)
    assert all(p.grad is None and not p.requires_grad for m in s2.embedding._two_b_modules() for p in m.parameters())
    torch.optim.Adam(s2.parameters(), lr=1e-2).step()
    assert _unchanged(s2, frozen)
    assert all(torch.equal(a, b) for a, b in zip(before, two_body_output()))


def test_edge_h0_prior_is_required_and_used():
    model = _build(True, prior_kind="h0", **EDGE)
    data = _prior_data(model, "h0")
    base = model(_clone(data))["node_features"].detach()
    shifted = model(dict(_clone(data), node_h0=data["node_h0"] + 1))["node_features"].detach()
    assert not torch.allclose(base, shifted)
    data.pop("node_h0")
    with pytest.raises((KeyError, RuntimeError, ValueError)):
        model(data)


# ---------------------------------------------------------------------------
# prior-activate routing and Switch top-1
# ---------------------------------------------------------------------------
def test_prior_activate_router_reads_the_edge_descriptor_and_backpropagates():
    model = _build(False, **PA)
    emb = model.embedding
    assert emb.router.net[0].weight.shape[1] > 4  # descriptor columns beyond the 4 edge one-hot inputs
    out = model(_data(model))
    assert out["edge_moe_num_route_tokens"] > 0
    (out["node_features"].square().mean() + out["edge_features"].square().mean()).backward()
    assert any(p.grad is not None for p in emb.layers.parameters())


def test_prior_activate_stage1_checkpoint_loads_strictly_into_stage2():
    s1, s2 = _build(True, **PA), _build(False, **PA)
    s2.load_state_dict(s1.state_dict(), strict=True)
    assert s2.embedding.edge_router_prior_activate


@pytest.mark.parametrize("route", ["staged", "streamed_m_major_cueq", "streamed_m_major_fused_p0"])
def test_prior_activate_keeps_its_route_despite_the_fusion_env(monkeypatch, route):
    monkeypatch.setenv("DPTB_SO2_FUSION_MODE", "streamed_m_major_ref")
    model = _build(False, **dict(PA, so2_fusion_mode=route))
    assert {m.so2_fusion_mode for m in model.embedding.modules() if hasattr(m, "so2_fusion_mode")} == {route}
    assert os.environ["DPTB_SO2_FUSION_MODE"] == "streamed_m_major_ref"


@pytest.mark.parametrize("route", ["streamed_m_major_ref", "streamed_m_major_persistent_grouped_p1"])
def test_prior_activate_rejects_weight_space_routes(route):
    with pytest.raises(ValueError, match="so2_fusion_mode"):
        _build(False, **dict(PA, so2_fusion_mode=route))


@pytest.mark.parametrize("kind", ["h0", "na_cf"])
def test_switch_stage2_reuses_stage1_and_trains_the_router(kind):
    s1, s2 = _build(True, prior_kind=kind, **SWITCH), _build(False, prior_kind=kind, **SWITCH)
    s2.load_state_dict(s1.state_dict(), strict=True)
    frozen = _frozen(s2)
    out = s2(_prior_data(s2, kind))
    (out["node_features"].square().mean() + out["edge_features"].square().mean()).backward()
    grad = s2.embedding.router.net[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    torch.optim.SGD(s2.parameters(), lr=.001).step()
    assert _unchanged(s2, frozen)


def test_switch_rejects_shared_experts():
    with pytest.raises(ValueError, match="num_shared_experts"):
        _build(False, **dict(SWITCH, num_shared_experts=1))


# ---------------------------------------------------------------------------
# flow-conditioned stage 2
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["h0", "na_cf"])
def test_flow_stage2_never_changes_the_frozen_physical_stage1_input(kind):
    from dptb.nnops.flow import HamiltonianCFM

    keys = PRIOR_KEYS[kind]
    s1 = _build(True, prior_kind=kind, **EDGE)
    s2 = _build(False, prior_kind=kind, **EDGE, use_flow_time_embedding=True, flow_time_condition_edges=True,
                flow_time_allow_missing=False)
    s2.load_state_dict(s1.state_dict(), strict=True)
    data = _prior_data(s1, kind)
    data.update(batch=torch.zeros(2, dtype=torch.long), node_features=torch.randn_like(data[keys[0]]),
                edge_features=torch.randn_like(data[keys[1]]))
    flow = HamiltonianCFM(dict(enabled=True, prior="te", te_prior_mode="typewise", node_h0_key=keys[0],
                               edge_h0_key=keys[1]), idp=s2.idp)
    state, reference, ctx = flow.prepare_batch(_clone(data), _clone(data), t=torch.tensor([.2]))
    for label, key in zip(("node", "edge"), keys):
        assert torch.equal(state["serial_original_" + key], data[key])
        base, prior = getattr(ctx, label + "_base"), getattr(ctx, label + "_prior")
        target, t = getattr(ctx, label + "_target"), getattr(ctx, label + "_t")[:, None]
        torch.testing.assert_close(state[key], (1 - t) * (base + prior) + t * target)
        assert torch.equal(reference[label + "_features"], data[label + "_features"])

    frozen = _frozen(s2)
    captures = {"node": [], "edge": []}
    hooks = [getattr(s2.embedding, "two_b_out_" + key).register_forward_hook(
        lambda module, args, out, key=key: captures[key].append(out.detach().clone())) for key in captures]
    try:
        first = s2(_clone(state))
        second = s2(dict(_clone(state), flow_time=torch.tensor([.8])))
        assert any(not torch.allclose(first[k + "_features"], second[k + "_features"]) for k in captures)
        s2.embedding.only2b = True
        s2(_clone(data))
        s2.embedding.only2b = False
        for values in captures.values():  # the two-body branch saw the same physical input all three times
            torch.testing.assert_close(values[1], values[0], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(values[2], values[0], rtol=1e-6, atol=1e-6)
    finally:
        for hook in hooks:
            hook.remove()
    (first["node_features"].square().mean() + first["edge_features"].square().mean()).backward()
    torch.optim.SGD(s2.parameters(), lr=.001).step()
    assert _unchanged(s2, frozen)
    no_labels = _clone(data)
    no_labels["node_features"].zero_()
    no_labels["edge_features"].zero_()
    for steps in (1, 2):
        sampled = flow.sample(s2, no_labels, num_steps=steps)
        assert all(torch.isfinite(sampled[k + "_features"]).all() for k in captures)
    with pytest.raises(KeyError, match="immutable input"):
        s2(_clone(data))


# ---------------------------------------------------------------------------
# H0 prior projection: AO products -> sorted irreps by Clebsch-Gordan (the default)
# ---------------------------------------------------------------------------
def _h0_layer(legacy=False):
    mapper = OrbitalMapper({"C": "2s2p1d"}, method="e3tb", has_soc=True, nextham_uureal_mask=True,
                           full_soc_prediction=False)
    base = torch.nn.Module()
    base.idp = mapper
    base.irreps_out = mapper.get_irreps().sort()[0].simplify()
    return H0InitLayer(base, h0_ao_cg=not legacy, dtype=torch.float64, device="cpu").double()


def test_h0_cg_is_default_and_legacy_checkpoints_restore_the_sort_projection():
    fixed, legacy = _h0_layer(), _h0_layer(legacy=True)
    assert fixed.h0_ao_cg
    old = legacy.state_dict()
    old.pop("h0_ao_cg_version")  # checkpoints from before the CG projection
    restored = _h0_layer(legacy=True)
    restored.load_state_dict(copy.deepcopy(old), strict=True)
    x = torch.randn(3, fixed.h0_dim, dtype=torch.float64)
    assert torch.equal(restored._ao_product_to_sorted_irreps(x), x.index_select(1, restored._h0_sort_index))
    for model, state in ((fixed, copy.deepcopy(old)), (legacy, fixed.state_dict())):
        with pytest.raises(RuntimeError, match="h0_ao_cg_version"):
            model.load_state_dict(state, strict=True)
    _h0_layer().load_state_dict(fixed.state_dict(), strict=True)


@pytest.mark.parametrize("branch", ["two_b", "gnn"])
def test_both_serial_projectors_use_the_cg_projection(branch):
    h0 = _h0_layer()
    node = torch.randn(2, h0.h0_dim, dtype=torch.float64, requires_grad=True)
    edge = torch.randn(3, h0.h0_dim, dtype=torch.float64, requires_grad=True)
    owner = SimpleNamespace(h0_init=h0, prior_node_key="node_p23", prior_edge_key="edge_p2", dtype=torch.float64,
                            device="cpu")
    node_proj = h0.node_projector if branch == "gnn" else copy.deepcopy(h0.node_projector)
    edge_proj = h0.edge_projector if branch == "gnn" else copy.deepcopy(h0.edge_projector)
    atom, bond = torch.zeros(2, dtype=torch.long), torch.zeros(3, dtype=torch.long)
    y, z = _Prior2bMixin._project_prior(owner, {"node_p23": node, "edge_p2": edge}, atom, bond, torch.arange(3),
                                        2, 3, node_proj, edge_proj)
    assert torch.equal(y, node_proj(h0._ao_product_to_sorted_irreps(h0._mask_node_source(node, atom))))
    assert torch.equal(z, edge_proj(h0._ao_product_to_sorted_irreps(h0._mask_edge_source(edge, bond))))
    (y.square().sum() + z.square().sum()).backward()
    assert torch.isfinite(node.grad).all() and torch.isfinite(edge.grad).all()

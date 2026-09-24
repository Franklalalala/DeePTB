"""Router regularisers of the routed MoE embeddings: the ST-MoE z-loss (MOLERouterV3, the Switch top-1 router), the
Switch balancing loss E * sum_e f_e * P_e, and how hamil_abs adds them to the training objective."""
import pytest
import torch

from dptb.data import AtomicDataDict, _keys
from dptb.nn.tensor_product_moe_v3 import MOLERouterV3, router_z_loss, write_router_regularizers
from dptb.nn.top1_prior import Top1PriorRouter
from dptb.nnops.loss import HamilLossAbs
from dptb.tests.model_helpers import _build, _data


def _lse2(logits, weights=None):
    z = torch.logsumexp(logits.double(), dim=-1) ** 2
    if weights is None:
        return z.mean()
    w = weights.double()
    return (z * w).sum() / w.sum()


def test_router_z_loss_is_the_mean_squared_logsumexp():
    torch.manual_seed(0)
    logits = torch.randn(9, 6) * 3
    torch.testing.assert_close(router_z_loss(logits).double(), _lse2(logits), rtol=1e-6, atol=0)
    sizes = torch.tensor([1., 4., 0., 2., 1., 1., 3., 5., 1.])
    torch.testing.assert_close(router_z_loss(logits, sizes).double(), _lse2(logits, sizes), rtol=1e-6, atol=0)
    assert router_z_loss(logits[:0]).item() == 0.0


def test_moe_router_v3_exposes_a_differentiable_z_loss():
    torch.manual_seed(1)
    router = MOLERouterV3(in_features=5, num_experts=8, top_k=2)
    x = torch.randn(40, 5)
    router(x)
    z = router.last_router_z_loss
    torch.testing.assert_close(z.double(), _lse2(router.net(x).detach()), rtol=1e-6, atol=0)
    z.backward()
    grads = [p.grad for p in router.net.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads) and grads[-1].abs().sum() > 0
    data = {}
    write_router_regularizers(router, data)
    assert data["router_z_loss"] is router.last_router_z_loss and "router_aux_loss" not in data


def test_mixing_temperature_softens_the_top_k_weights_without_changing_selection():
    torch.manual_seed(3)
    x = torch.randn(30, 5)
    base = MOLERouterV3(in_features=5, num_experts=8, top_k=2)
    hot = MOLERouterV3(in_features=5, num_experts=8, top_k=2, mixing_temperature=4.0)
    hot.load_state_dict(base.state_dict())
    base.eval(), hot.eval()
    c1, m1, _ = base(x)
    i1, w1 = base.last_topk()
    c4, m4, _ = hot(x)
    i4, w4 = hot.last_topk()
    assert torch.equal(i1, i4)
    logits = base.net(x).detach()
    torch.testing.assert_close(w4, torch.softmax(logits.gather(1, i1) / 4.0, dim=-1))
    torch.testing.assert_close(w1, torch.softmax(logits.gather(1, i1), dim=-1))
    assert m4 < m1
    full = MOLERouterV3(in_features=5, num_experts=8, top_k=8, mixing_temperature=2.0)
    full.load_state_dict(base.state_dict())
    probs, _, _ = full(x)
    torch.testing.assert_close(probs, torch.softmax(full.net(x).detach() / 2.0, dim=-1))
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            MOLERouterV3(in_features=5, num_experts=8, top_k=2, mixing_temperature=bad)


def test_edge_router_temperature_reaches_the_router_and_is_rejected_for_switch():
    pa = dict(method="lem_moe_v3_edge_prior_2b", num_experts=4, num_shared_experts=1, top_k=2,
              edge_router_prior_activate=True, so2_fusion_mode="staged", edge_moe_compact_min_edges=0)
    assert _build(only2b=False, edge_router_temperature=5.0, **pa).embedding.router.mixing_temperature == 5.0
    assert _build(only2b=False, **pa).embedding.router.mixing_temperature == 1.0
    with pytest.raises(ValueError):
        _build(only2b=False, method="lem_moe_v3_edge_prior_2b", num_experts=4, num_shared_experts=0, top_k=1,
               edge_router_prior_activate=True, edge_router_top1_mode="switch",
               so2_fusion_mode="streamed_m_major_cueq", edge_moe_compact_min_edges=0, edge_router_temperature=2.0)


def _switch(num_experts):
    router = Top1PriorRouter(num_experts, num_experts)
    router.net = torch.nn.Identity()
    return router


@pytest.mark.parametrize("spread", [True, False])
def test_switch_balancing_loss_is_one_when_balanced_and_e_when_collapsed(spread):
    e = 8
    router = _switch(e)
    target = torch.arange(32) % e if spread else torch.zeros(32, dtype=torch.long)
    logits = torch.nn.functional.one_hot(target, e).float() * 30.0
    router(logits)
    expected = 1.0 if spread else float(e)
    torch.testing.assert_close(router.last_router_aux_loss, torch.tensor(expected), rtol=1e-4, atol=1e-4)


def test_switch_balancing_loss_matches_the_reference_and_its_gradient():
    torch.manual_seed(2)
    e = 6
    router = _switch(e)
    logits = torch.randn(25, e, requires_grad=True)
    sizes = torch.randint(1, 4, (25,)).float()
    router(logits, sizes)
    aux = router.last_router_aux_loss
    probs = logits.softmax(-1)
    frac = torch.zeros(e).index_add_(0, probs.argmax(-1), sizes) / sizes.sum()
    ref = e * (frac * ((probs * sizes[:, None]).sum(0) / sizes.sum())).sum()
    torch.testing.assert_close(aux, ref)
    got, = torch.autograd.grad(aux, logits, retain_graph=True)
    want, = torch.autograd.grad(ref, logits)
    torch.testing.assert_close(got, want)
    assert got.abs().sum() > 0
    torch.testing.assert_close(router.last_router_z_loss.double(), _lse2(logits.detach(), sizes), rtol=1e-6, atol=0)
    router(torch.empty(0, e))
    assert router.last_router_aux_loss.item() == 0.0 and router.last_router_z_loss.item() == 0.0


def _hamil_abs_inputs():
    class FakeIdp:
        mask_to_nrme = torch.tensor([[True, True, False, False]])
        mask_to_erme = torch.tensor([[True, True, True, True]])

    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.NODE_FEATURES_KEY: torch.zeros((1, 4)),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.zeros((1, 4)),
    }
    ref = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([[0]]),
        AtomicDataDict.NODE_FEATURES_KEY: torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.tensor([[3.0, 3.0, 3.0, 3.0]]),
    }
    return FakeIdp(), data, ref


def test_hamil_abs_adds_the_weighted_router_terms_and_keeps_the_components():
    idp, data, ref = _hamil_abs_inputs()
    plain = HamilLossAbs(idp=idp)(dict(data), ref)
    z = torch.tensor(2.5, requires_grad=True)
    aux = torch.tensor(1.5, requires_grad=True)
    data_reg = dict(data, router_z_loss=z, router_aux_loss=aux)
    assert torch.equal(HamilLossAbs(idp=idp)(dict(data_reg), ref), plain)  # coefficients default to 0
    loss = HamilLossAbs(idp=idp, router_z_loss_coef=0.01, router_aux_loss_coef=0.1)
    total = loss(dict(data_reg), ref)
    torch.testing.assert_close(total, plain + 0.01 * 2.5 + 0.1 * 1.5)
    total.backward()
    torch.testing.assert_close(z.grad, torch.tensor(0.01))
    torch.testing.assert_close(aux.grad, torch.tensor(0.1))
    assert loss.last_router_z_loss.item() == 2.5 and loss.last_router_aux_loss.item() == 1.5
    torch.testing.assert_close(loss.last_onsite_loss, torch.tensor(1.0))
    torch.testing.assert_close(loss.last_hopping_loss, torch.tensor(3.0))


def test_hamil_abs_rejects_a_coefficient_without_its_term_and_bad_values():
    idp, data, ref = _hamil_abs_inputs()
    with pytest.raises(KeyError):
        HamilLossAbs(idp=idp, router_aux_loss_coef=1e-3)(dict(data), ref)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            HamilLossAbs(idp=idp, router_z_loss_coef=bad)


@pytest.mark.parametrize("routing", ["prior_activate", "switch"])
def test_routed_embeddings_hand_the_regularisers_to_the_loss(routing):
    kw = dict(num_experts=4, num_shared_experts=1, top_k=2, edge_router_prior_activate=True, so2_fusion_mode="staged")
    if routing == "switch":
        kw.update(num_shared_experts=0, top_k=1, edge_router_top1_mode="switch", so2_fusion_mode="streamed_m_major_cueq")
    model = _build(only2b=False, method="lem_moe_v3_edge_prior_2b", edge_moe_compact_min_edges=0, **kw)
    out = model(_data(model))
    keys = ["router_z_loss"] + (["router_aux_loss"] if routing == "switch" else [])
    assert all(out[k].requires_grad and torch.isfinite(out[k]) for k in keys)
    sum(out[k] for k in keys).backward()
    router_grads = [p.grad for p in model.embedding.router.parameters() if p.requires_grad]
    assert router_grads and all(g is not None for g in router_grads)
    assert out[_keys.EDGE_FEATURES_KEY].shape[0] > 0

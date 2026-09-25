"""Edge-router options (edge_router_*) and the routed-expert options of HybridMuon (expert_*, adamw_name_patterns).

Defaults must reproduce the pristine code bit for bit; the pristine classes are loaded from the base commit with git.
"""
from __future__ import annotations

import copy
import importlib.util
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from dptb.nn import moe_registry
from dptb.nn.tensor_product_moe_v3 import MOLERouterV3
from dptb.utils.dpa4_optim import HybridMuon, WarmupStableDecayLR

BASE_COMMIT = "4c6d0e4"   # 0924-stable + router regularisers + post_activation_slot, before these options
REPO = Path(__file__).resolve().parents[2]
S1C_OPT = dict(lr=1.0e-2, weight_decay=0.01, muon_beta=0.95, muon_scale=0.2, magma_lite=True, muon_clip=True,
               muon_clip_mode="auto", muon_clip_rms=0.2)


def _pristine(relpath, modname):
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:{relpath}"], check=True,
                             capture_output=True, text=True).stdout
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"pristine source unavailable: {exc}")
    mod = types.ModuleType(modname)
    mod.__file__ = f"<{BASE_COMMIT}:{relpath}>"
    mod.__package__ = ".".join(Path(relpath).with_suffix("").parts[:-1])   # relative imports resolve in the live package
    sys.modules[modname] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def _x(n=64, d=40, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def _router(seed=0, **kw):
    torch.manual_seed(seed)
    return MOLERouterV3(40, num_experts=8, top_k=2, **kw)


# ---------------------------------------------------------------- router

def test_default_router_matches_pristine():
    old_mod = _pristine("dptb/nn/tensor_product_moe_v3.py", "_pristine_tp_moe_v3")
    torch.manual_seed(0)
    old = old_mod.MOLERouterV3(40, num_experts=8, top_k=2)
    new = _router(0)
    new.load_state_dict(old.state_dict())
    x = _x() * 30.0                          # large logits: saturated sigmoid ties, the production regime
    for mode in ("train", "train", "train", "eval"):
        old.train(mode == "train"); new.train(mode == "train")
        co, mo, cvo = old(x); cn, mn, cvn = new(x)
        assert torch.equal(co, cn) and torch.equal(mo, mn) and torch.equal(cvo, cvn)
        assert torch.equal(old.last_topk()[0], new.last_topk()[0])
        assert torch.equal(old.last_topk()[1], new.last_topk()[1])
        assert torch.equal(old.expert_bias, new.expert_bias) and torch.equal(old.ema_load, new.ema_load)


def test_cosine_logit_router_selects_alike_in_train_and_eval():
    r = _router(1, logit_kind="cosine", logit_scale=10.0, select="logit", bias_at_eval=True, bias_update_speed=0.0)
    with torch.no_grad():
        r.expert_bias.copy_(torch.linspace(-1.0, 1.0, 8))
    x = _x() * 50.0
    r.train(); ct, _, _ = r(x); it = r.last_topk()[0].clone()
    r.eval(); ce, _, _ = r(x); ie = r.last_topk()[0].clone()
    assert torch.equal(it, ie) and torch.allclose(ct, ce)
    z = r._logits(x)
    assert float(z.detach().abs().max()) <= 10.0 + 1e-4
    assert torch.allclose(ce.sum(-1), torch.ones(64), atol=1e-6)
    # selection = top-2 of z + b, not of sigmoid(z) + b
    assert torch.equal(torch.sort(ie, -1).values, torch.sort(torch.topk(z + r.expert_bias, 2, -1).indices, -1).values)
    r.train(); r.zero_grad()
    c, _, _ = r(x)
    (c * torch.randn_like(c)).sum().backward()
    assert r.net[0].weight.grad.abs().sum() > 0 and r.net[2].weight.grad.abs().sum() > 0
    assert r.net[2].bias.grad is None        # unused by the cosine logits


def test_select_noise_is_train_only_and_leaves_mixing_noise_free():
    base = _router(2, select="logit")
    noisy = _router(2, select="logit", select_noise=1.0)
    noisy.load_state_dict(base.state_dict())
    x = _x()
    base.eval(); noisy.eval()
    assert torch.equal(base(x)[0], noisy(x)[0])
    base.train(); noisy.train()
    torch.manual_seed(5); base(x); ib = base.last_topk()[0]
    torch.manual_seed(5); noisy(x); idx, val = noisy.last_topk()
    assert (idx != ib).any()
    z = noisy._logits(x)
    assert torch.allclose(val, torch.softmax(torch.gather(z, 1, idx), -1), atol=1e-6)


def _bias_delta(r, x, scale):
    r.train(); r.bias_lr_scale = scale
    before = r.expert_bias.clone(); r(x)
    return r.expert_bias - before


def test_bias_schedules():
    x = _x() * 5.0
    const = _router(3); follow = _router(3, bias_schedule="follow_lr"); freeze = _router(3, bias_schedule="freeze_decay")
    follow.load_state_dict(const.state_dict()); freeze.load_state_dict(const.state_dict())
    d_const = _bias_delta(const, x, 0.25)
    d_follow = _bias_delta(follow, x, 0.25)
    assert torch.allclose(d_follow, 0.25 * d_const, atol=1e-7) and d_const.abs().max() > 0
    assert torch.equal(_bias_delta(freeze, x, 0.5), torch.zeros(8))
    assert _bias_delta(freeze, x, 1.0).abs().max() > 0


def test_router_rejects_bad_options():
    for kw in (dict(logit_kind="tanh"), dict(select="softmax"), dict(bias_schedule="cosine"), dict(logit_scale=0.0),
               dict(select_noise=-1.0)):
        with pytest.raises(ValueError):
            _router(**kw)


# ---------------------------------------------------------------- optimizer

def _named_params(seed=0, num_experts=8):
    torch.manual_seed(seed)
    return [
        ("embedding.router.net.0.weight", nn.Parameter(torch.randn(16, 40) * 0.1)),
        ("embedding.router.net.0.bias", nn.Parameter(torch.zeros(16))),
        ("layers.0.fc.weight_experts", nn.Parameter(torch.randn(num_experts, 12, 10) * 0.1)),
        ("layers.0.fc.bias_experts", nn.Parameter(torch.randn(num_experts, 12) * 0.1)),
        ("layers.0.fc.weight_shared", nn.Parameter(torch.randn(1, 12, 10) * 0.1)),
        ("layers.0.fc.bias_shared", nn.Parameter(torch.zeros(1, 12))),
    ]


def _run(opt_cls, named, steps=4, seed=9, **kw):
    opt = opt_cls(named, **{**S1C_OPT, **kw})
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        for _, p in named:
            p.grad = torch.randn(p.shape, generator=g)
        opt.step()
    return opt


def test_default_hybrid_muon_matches_pristine():
    old_mod = _pristine("dptb/utils/dpa4_optim.py", "_pristine_dpa4_optim")
    a, b = _named_params(), _named_params()
    _run(old_mod.HybridMuon, a); _run(HybridMuon, b)
    for (na, pa), (_, pb) in zip(a, b):
        assert torch.equal(pa, pb), na


def test_expert_update_scale_const_scales_only_expert_updates():
    ref, sc = _named_params(), _named_params()
    init = [p.detach().clone() for _, p in ref]
    _run(HybridMuon, ref, steps=1, weight_decay=0.0)
    _run(HybridMuon, sc, steps=1, weight_decay=0.0, expert_update_scale="const", expert_update_scale_const=0.5)
    for (name, pr), (_, ps), p0 in zip(ref, sc, init):
        dr, ds = pr.detach() - p0, ps.detach() - p0
        if "experts" in name:
            assert torch.allclose(ds, 0.5 * dr, atol=1e-7), name
        else:
            assert torch.equal(ds, dr), name


class _FakeRouter(nn.Module):
    def __init__(self, load):
        super().__init__()
        self.num_experts = len(load)
        self.register_buffer("ema_load", torch.tensor(load, dtype=torch.float32))
        moe_registry.register_router(self)


def test_expert_update_scale_sqrt_load():
    load = [4.0, 1.0, 1.0, 0.0, 2.0, 1.0]     # 6 experts: no other test builds a 6-expert router
    fake = _FakeRouter(load)                 # keep a reference: the registry is weak
    ref, sc = _named_params(num_experts=6), _named_params(num_experts=6)
    init = [p.detach().clone() for _, p in ref]
    _run(HybridMuon, ref, steps=1, weight_decay=0.0)
    opt = _run(HybridMuon, sc, steps=1, weight_decay=0.0, expert_update_scale="sqrt_load", expert_update_scale_min=0.1)
    lt = torch.tensor(load)
    want = (lt / lt.mean()).sqrt().clamp(0.1, 1.0)
    assert torch.allclose(opt.last_expert_scale, want)
    for (name, pr), (_, ps), p0 in zip(ref, sc, init):
        if "experts" not in name:
            continue
        dr, ds = pr.detach() - p0, ps.detach() - p0
        ratio = want.reshape(-1, *([1] * (dr.dim() - 1)))
        assert torch.allclose(ds, ratio * dr, atol=1e-7), name
    del fake


def test_adamw_force_moves_2d_params_off_muon_with_scaled_lr():
    named = _named_params()
    p0 = dict((n, p.detach().clone()) for n, p in named)
    opt = _run(HybridMuon, named, steps=1, weight_decay=0.0, adamw_name_patterns=["*bias_experts*"],
               adamw_pattern_lr_scale=0.1)
    pe = dict(named)["layers.0.fc.bias_experts"]
    assert "exp_avg" in opt.state[pe] and "momentum_buffer" not in opt.state[pe]
    assert "momentum_buffer" in opt.state[dict(named)["layers.0.fc.weight_experts"]]
    # first AdamW step: update = lr * m_hat / (sqrt(v_hat) + eps) = lr * sign(g) (eps 1e-20)
    step = (pe.detach() - p0["layers.0.fc.bias_experts"]).abs()
    assert torch.allclose(step, torch.full_like(step, 1.0e-3), rtol=1e-4)


def test_expert_weight_decay_mult():
    named = _named_params()
    p0 = dict((n, p.detach().clone()) for n, p in named)
    opt = HybridMuon(named, **{**S1C_OPT, "expert_weight_decay_mult": 3.0})
    for _, p in named:
        p.grad = torch.zeros_like(p)
    opt.step()
    w = dict(named)["layers.0.fc.weight_experts"]
    assert torch.allclose(w.detach(), p0["layers.0.fc.weight_experts"] * (1.0 - 1.0e-2 * 0.01 * 3.0), atol=1e-8)
    s = dict(named)["layers.0.fc.weight_shared"]
    assert torch.allclose(s.detach(), p0["layers.0.fc.weight_shared"] * (1.0 - 1.0e-2 * 0.01), atol=1e-8)


def test_optimizer_publishes_lr_ratio_and_keeps_peak_across_resume():
    r = _router(4, bias_schedule="freeze_decay")
    named = _named_params()
    opt = HybridMuon(named, **S1C_OPT)
    sched = WarmupStableDecayLR(opt, total_steps=20, warmup_steps=5, decay_ratio=0.5, min_lr=1e-4)
    seen = []
    for _ in range(18):
        for _, p in named:
            p.grad = torch.randn_like(p)
        opt.step(); seen.append(r.bias_lr_scale); sched.step()
    assert all(abs(s - 1.0) < 1e-12 for s in seen[:11])       # warmup + plateau
    assert seen[-1] < 1.0 and r._bias_step() == 0.0
    sd = copy.deepcopy(opt.state_dict())
    named2 = _named_params()
    opt2 = HybridMuon(named2, **S1C_OPT); opt2.load_state_dict(sd)
    assert opt2.param_groups[0]["peak_lr_seen"] == pytest.approx(1.0e-2)


# ---------------------------------------------------------------- embedding

from dptb.tests.test_lem_moe_v3_prior_2b import _build, _data  # noqa: E402

PA = dict(edge_router_prior_activate=True, num_experts=4, num_shared_experts=1, top_k=2, so2_fusion_mode="staged")


def test_router_options_reach_the_router_and_the_model_config():
    opts = dict(edge_router_logit="cosine", edge_router_logit_scale=8.0, edge_router_select="logit",
                edge_router_bias_at_eval=True, edge_router_bias_speed=0.001, edge_router_bias_schedule="follow_lr",
                edge_router_select_noise=0.5)
    model = _build(False, **dict(PA, **opts))
    cfg = model.embedding.router.router_config()
    assert cfg == dict(logit_kind="cosine", logit_scale=8.0, select="logit", bias_at_eval=True,
                       bias_schedule="follow_lr", bias_update_speed=0.001, select_noise=0.5,
                       mixing_temperature=1.0, bias_freeze_after_step=0)
    out = model(_data(model))
    (out["node_features"].square().mean() + out["edge_features"].square().mean()).backward()
    assert model.embedding.router.net[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["onehot_r", "onehot"])
def test_router_input_controls(kind):
    model = _build(False, **dict(PA, edge_router_input=kind, edge_router_rbf=6))
    emb = model.embedding
    want = emb.edge_one_hot_dim + (6 if kind == "onehot_r" else 0)
    assert emb.router.net[0].in_features == want == emb.edge_router_in_features
    out = model(_data(model))
    (out["node_features"].square().mean() + out["edge_features"].square().mean()).backward()
    g = emb.router.net[0].weight.grad
    assert g is not None and g[:, :emb.edge_one_hot_dim].abs().sum() > 0
    if kind == "onehot_r":
        assert g[:, emb.edge_one_hot_dim:].abs().sum() > 0   # zero-initialised radial columns still learn


def test_router_input_requires_per_edge_routing():
    # the edge class itself (lem_moe_v3_prior_2b without prior_activate builds the non-edge H0 class)
    with pytest.raises(ValueError, match="edge_router_input"):
        _build(False, **dict(PA, method="lem_moe_v3_edge_prior_2b", edge_router_prior_activate=False,
                             edge_router_input="onehot_r"))


def test_default_embedding_matches_pristine_router_shape():
    model = _build(False, **PA)
    emb = model.embedding
    assert emb.router.net[0].in_features == emb.edge_one_hot_dim + emb.edge_router_prior_dim
    assert emb.router.router_config()["logit_kind"] == "raw" and emb.router.router_config()["select"] == "sigmoid"


# ---------------------------------------------------------------- additions after astra round 3

def test_mixing_temperature_changes_mixing_only():
    a = _router(6, logit_kind="cosine", select="logit")
    b = _router(6, logit_kind="cosine", select="logit", mixing_temperature=2.0)
    b.load_state_dict(a.state_dict())
    x = _x()
    a.eval(); b.eval(); a(x); b(x)
    ia, va = a.last_topk(); ib, vb = b.last_topk()
    assert torch.equal(ia, ib)
    z = b._logits(x)
    assert torch.allclose(vb, torch.softmax(torch.gather(z, 1, ib) / 2.0, -1), atol=1e-6)
    assert (vb.max(-1).values <= va.max(-1).values + 1e-7).all()      # softer


def test_bias_freeze_after_step_uses_the_published_step():
    r = _router(7, bias_freeze_after_step=3)
    x = _x() * 5.0
    moe_registry.publish_lr_scale(1.0, step=2)
    assert _bias_delta(r, x, 1.0).abs().max() > 0
    moe_registry.publish_lr_scale(1.0, step=3)
    assert torch.equal(_bias_delta(r, x, 1.0), torch.zeros(8))


def test_optimizer_counts_committed_steps():
    r = _router(8)
    named = _named_params()
    opt = _run(HybridMuon, named, steps=5)
    assert opt.param_groups[0]["committed_steps"] == 5 and r.opt_step == 5


def test_sqrt_load_times_uniform_factor():
    load = [4.0, 1.0, 1.0, 0.0, 2.0, 1.0]
    fake = _FakeRouter(load)
    named = _named_params(num_experts=6)
    opt = _run(HybridMuon, named, steps=1, weight_decay=0.0, expert_update_scale="sqrt_load",
               expert_update_scale_const=0.3, expert_update_scale_min=0.1)
    lt = torch.tensor(load)
    assert torch.allclose(opt.last_expert_scale, 0.3 * (lt / lt.mean()).sqrt().clamp(0.1, 1.0))
    del fake


def test_expert_stats_recorded_only_on_request():
    named = _named_params()
    opt = HybridMuon(named, **S1C_OPT)
    for _, p in named:
        p.grad = torch.randn_like(p)
    opt.step()
    assert opt.last_expert_stats == {}
    opt.record_expert_stats = True
    opt.step()
    st = opt.last_expert_stats["layers.0.fc.weight_experts"]
    assert set(st) == {"grad", "momentum", "update_clipped", "update_applied", "scale", "decay", "weight"}
    assert st["grad"].shape == (8,) and torch.allclose(st["update_applied"], st["update_clipped"])
    assert "layers.0.fc.bias_experts" in opt.last_expert_stats and "layers.0.fc.weight_shared" not in opt.last_expert_stats


def test_train_stats_come_from_training_forwards_only():
    r = _router(9, logit_kind="cosine", select="logit")
    r.record_train_stats = True
    x = _x()
    r.eval(); r(x)
    assert r.last_train_stats is None
    r.train(); c, _, _ = r(x)
    st = r.last_train_stats
    assert st["n_rows"] == 64 and st["top_k"] == 2
    assert torch.allclose(st["hard_load"].sum(), torch.tensor(128.0))
    assert torch.allclose(st["soft_load"].sum(), torch.tensor(64.0), atol=1e-4)
    assert torch.allclose(st["soft_load"], c.sum(0).float(), atol=1e-5)
    assert st["sel_margin_mean"] >= 0
    snapshot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    r.eval(); r(x * 3.0)                                  # validation must not overwrite them
    assert all(torch.equal(snapshot[k], r.last_train_stats[k]) if torch.is_tensor(snapshot[k])
               else snapshot[k] == r.last_train_stats[k] for k in snapshot)


def test_new_optimizer_and_router_defaults_do_not_look_like_geometry_losses():
    # MultiTrainer refuses precompute_lem_cutoff_coeffs when any train_options key containing force/stress/virial is
    # truthy; these options once shipped an `adamw_force_lr_scale` default of 1.0 and every such run stopped at start.
    from dptb.nnops.multi_trainer import MultiTrainer
    from dptb.utils.argcheck import HybridMuon as hm_args, _edge_router_arguments
    opt_defaults = {a.name: a.default for a in hm_args()}
    assert not MultiTrainer._contains_geometry_gradient_option({"optimizer": opt_defaults})
    for a in list(hm_args()) + list(_edge_router_arguments()):
        assert not any(term in a.name.lower() for term in ("force", "stress", "virial")) or a.name == "muon_force_name_patterns", a.name

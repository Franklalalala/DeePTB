"""Worker alpha initialization, dense function preservation and checkpoint lineage."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from dptb.nn.build import build_model
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear, MOLERouterV3, SO2_Linear
from dptb.plugins.saver import Saver
from dptb.tests._requires import requires_so2_cuda
from dptb.tests.model_helpers import _build


@pytest.fixture(scope="module")
def worker_init():
    folder = Path(os.environ.get("R8_WORKER_DIR", Path(__file__).resolve().parents[3] / "worker"))
    path = folder / "task_worker_v21_init.py"
    if not path.is_file():
        pytest.skip("needs worker/task_worker_v21_init.py or R8_WORKER_DIR")
    spec = importlib.util.spec_from_file_location("r8_worker_init", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(**extra):
    emb = dict(mole_expert_parameterization="full", so2_expert_mixing_mode="pre_activation",
               edge_router_gate="renorm", num_experts=24, num_shared_experts=1, n_layers=3)
    emb.update(extra)
    return dict(model_options=dict(embedding=emb))


def _map(worker, state, dense, alpha=0.5, config=None, **extra):
    mi = dict(mode="dense_to_shared", **{"from": "S0_hp.pth"})
    if alpha is not None:
        mi["alpha"] = alpha
    return worker.initialize_state_dict(state, mi, source_checkpoint=dict(model_state_dict=dense, iteration=37500),
                                        target_config=config or _config(), **extra)


def _check(a, b, record_property, name):
    record_property(name, float((a.detach().double() - b.detach().double()).abs().max()) if a.numel() else 0.)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5, msg=lambda m: name + ": " + m)


@pytest.mark.parametrize("alpha", [0., 0.5, 1.])
def test_split_each_block_bias_shared_only_and_untouched(worker_init, alpha, record_property):
    torch.manual_seed(67)
    routed = MOLELinear(13, 17, num_experts=24, num_shared_experts=1)
    shared = MOLELinear(11, 19, num_experts=0, num_shared_experts=1)
    dense_a = MOLELinear(13, 17, num_experts=1, num_shared_experts=0)
    dense_b = MOLELinear(11, 19, num_experts=1, num_shared_experts=0)
    state = {"routed." + k: v for k, v in routed.state_dict().items()}
    state.update({"shared_only." + k: v for k, v in shared.state_dict().items()})
    dense = {"routed." + k: v for k, v in dense_a.state_dict().items()}
    dense.update({"shared_only." + k: v for k, v in dense_b.state_dict().items()})
    state.update(backbone=torch.randn(7, 5), counter=torch.tensor(11), router_weight=torch.randn(4, 9))
    dense.update(backbone=torch.randn(7, 5), counter=torch.tensor(19))
    before, source_before = copy.deepcopy(state), copy.deepcopy(dense)
    mapped, rec = _map(worker_init, state, dense, alpha)
    assert rec["ok"] and rec["alpha"] == alpha
    assert all(torch.equal(v, state[k]) for k, v in before.items())
    assert all(torch.equal(v, dense[k]) for k, v in source_before.items())
    for key in ("backbone", "counter"):
        assert torch.equal(mapped[key], dense[key])
    assert torch.equal(mapped["router_weight"], state["router_weight"])
    c = torch.randn(9, 24).softmax(-1)
    for kind in ("weight", "bias"):
        d = dense["routed." + kind + "_experts"][0]
        s = mapped["routed." + kind + "_shared"][0]
        e = mapped["routed." + kind + "_experts"]
        assert torch.equal(e, (d * alpha).expand_as(e))
        assert torch.equal(s, d * (1 - alpha))
        mixed = s + torch.einsum("ne,e...->n...", c, e)
        _check(mixed, d.expand_as(mixed), record_property, "fp32_block_" + kind)
        assert torch.equal(mapped["shared_only." + kind + "_shared"], dense["shared_only." + kind + "_experts"])
    if alpha == 0:
        legacy, _ = _map(worker_init, state, dense, None)
        assert all(torch.equal(v, legacy[k]) for k, v in mapped.items())


@pytest.mark.parametrize("change", [dict(mole_expert_parameterization="shared_core"),
                                   dict(so2_expert_mixing_mode="post_activation_shared"),
                                   dict(so2_expert_mixing_mode="post_activation_slot"),
                                   dict(so2_expert_mixing_mode="post_activation"),
                                   dict(edge_router_gate="full_softmax"),
                                   dict(edge_router_top1_mode="switch"),
                                   dict(num_shared_experts=0), dict(num_shared_experts=2)])
def test_alpha_rejects_unsupported_semantics(worker_init, change):
    with pytest.raises(worker_init.MoeInitError):
        _map(worker_init, {}, {}, config=_config(**change))


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_alpha_rejects_invalid_values(worker_init, alpha):
    with pytest.raises(worker_init.MoeInitError):
        _map(worker_init, {}, {}, alpha=alpha)


@pytest.mark.parametrize("mode", ["copy", "zero_routed"])
def test_alpha_rejects_other_initialization_modes(worker_init, mode):
    with pytest.raises(worker_init.MoeInitError):
        worker_init.initialize_state_dict({}, dict(mode=mode, alpha=0.5), target_config=_config())


@pytest.mark.parametrize("kind", ["unpaired", "two_shared", "shared_core", "no_config"])
def test_alpha_validates_actual_state_not_just_config(worker_init, kind):
    state = dict(weight_experts=torch.randn(24, 5, 7), weight_shared=torch.randn(1, 5, 7))
    cfg = _config()
    if kind == "unpaired":
        del state["weight_shared"]
    elif kind == "two_shared":
        state["weight_shared"] = torch.randn(2, 5, 7)
    elif kind == "shared_core":
        state["core_experts"] = torch.randn(24, 3, 3)
    else:
        cfg = None
    with pytest.raises(worker_init.MoeInitError):
        worker_init.initialize_state_dict(state, dict(mode="dense_to_shared", alpha=0.5), target_config=cfg)


# Actual bank dimensions in the saved S18c2b_hp module probe (first, hidden,
# output layers; both m=0 with bias and paired m>0). Inputs are synthetic.
@pytest.mark.parametrize("out_dim,in_dim,pair", [(416, 210, False), (384, 180, True), (64, 30, True),
                                               (416, 672, False), (384, 576, True), (416, 448, False),
                                               (277, 672, False), (277, 375, False)])
def test_production_shaped_module_batch_matches_dense(worker_init, out_dim, in_dim, pair, record_property):
    torch.manual_seed(831)
    dense = MOLELinear(in_dim, out_dim, num_experts=1, num_shared_experts=0, bias=not pair)
    target = MOLELinear(in_dim, out_dim, num_experts=24, num_shared_experts=1, bias=not pair,
                        mole_linear_mode="split_loop")
    sd, _ = _map(worker_init, target.state_dict(), dense.state_dict())
    target.load_state_dict(sd, strict=True)
    logits = torch.randn(37, 24)
    values, indices = logits.topk(2, -1)
    values = values.softmax(-1)
    coeff = torch.zeros_like(logits).scatter(1, indices, values)
    route = MOLEGlobals(coefficients=coeff, topk_indices=indices, topk_values=values,
                        activation_space=True, coefficients_sum_to_one=True)
    x = torch.randn((37, 2, in_dim) if pair else (37, in_dim))
    with torch.no_grad():
        a = F.linear(x, dense.weight_experts[0], None if pair else dense.bias_experts[0])
        b = target(x, route)
    _check(a, b, record_property, "fp32_production_module")


def _models():
    common = dict(so2_fusion_mode="staged", n_layers=3, tp_radial_emb=True, tp_radial_channels=[8])
    dense = _build(False, num_experts=1, num_shared_experts=0, **common)
    target = _build(False, num_experts=24, num_shared_experts=1, top_k=2,
                    edge_router_prior_activate=True, so2_expert_mixing_mode="pre_activation", **common)
    return dense, target


def _model_config(model):
    return dict(model_options=model.model_options,
                common_options=dict(basis={"H": "1s", "O": "1s1p"}, overlap=False, dtype="float32", device="cpu"),
                train_options={})


def _batch(model):
    # Two nontrivial graphs with distinct edge counts; actual AtomicDataDict and
    # prior/head dimensions obtained from the model, no mocked forward path.
    generator = torch.Generator().manual_seed(506)
    n = 11
    pos = torch.randn(n, 3, generator=generator) * 0.3
    batch = torch.tensor([0] * 5 + [1] * 6)
    edge = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j and batch[i] == batch[j]]).t()
    h, o = (model.idp.chemical_symbol_to_type[k] for k in ("H", "O"))
    symbols = ["H" if i % 3 else "O" for i in range(n)]
    types = torch.tensor([h if k == "H" else o for k in symbols]).reshape(-1, 1)
    bonds = torch.tensor([model.idp.bond_to_type[symbols[i] + "-" + symbols[j]] for i, j in edge.t().tolist()])
    rme = model.idp.reduced_matrix_element
    return dict(pos=pos, edge_index=edge, atom_types=types, edge_type=bonds, batch=batch,
                ptr=torch.tensor([0, 5, 11]), node_p23=torch.randn(n, rme, generator=generator),
                edge_p2=torch.randn(edge.shape[1], rme, generator=generator))


@pytest.mark.parametrize("alpha", [0., 0.5])
def test_full_model_blocks_modules_and_final_outputs(worker_init, alpha, record_property):
    torch.manual_seed(105)
    dense, target = _models()
    sd, rec = _map(worker_init, target.state_dict(), dense.state_dict(), alpha)
    if alpha == 0:
        old, _ = _map(worker_init, target.state_dict(), dense.state_dict(), None)
        assert all(torch.equal(v, old[k]) for k, v in sd.items())
    for key in rec["copied"]:
        assert torch.equal(sd[key], dense.state_dict()[key]), key
    for key in rec["kept_init"]:
        assert torch.equal(sd[key], target.state_dict()[key]), key
    c = torch.randn(3, 24).softmax(-1)
    for key in rec["mapped_shared"]:
        expert = key.replace("_shared", "_experts")
        mixed = sd[key][0] + torch.einsum("ne,e...->n...", c, sd[expert])
        _check(mixed, dense.state_dict()[expert][0].expand_as(mixed), record_property, "fp32_block_" + key)
    target.load_state_dict(sd, strict=True)
    dense.eval(), target.eval()
    da, ta, hooks = {}, {}, []
    for model, captured in ((dense, da), (target, ta)):
        for name, module in model.named_modules():
            if isinstance(module, MOLELinear):
                hooks.append(module.register_forward_hook(lambda m, a, out, key=name, store=captured: store.__setitem__(key, out.detach())))
    data = _batch(dense)
    try:
        with torch.no_grad():
            a, b = dense(copy.deepcopy(data)), target(copy.deepcopy(data))
    finally:
        for hook in hooks:
            hook.remove()
    assert da.keys() == ta.keys() and da
    for name in da:
        _check(da[name], ta[name], record_property, "fp32_module_" + name)
    for key in ("node_features", "edge_features"):
        _check(a[key], b[key], record_property, "fp32_final_" + key)


def test_selective_model_keeps_shared_only_layers_dense(worker_init, record_property):
    torch.manual_seed(197)
    dense, all_layers = _models()
    selective_config = copy.deepcopy(_model_config(all_layers))
    selective_config["model_options"]["embedding"]["so2_moe_layers"] = [1]
    selected = build_model(**selective_config)
    source = dict(model_state_dict=dense.state_dict(), iteration=37500, config=_model_config(dense))
    mi = dict(mode="dense_to_shared", alpha=0.5, **{"from": "S0_hp.pth"})
    sd_all, rec_all = worker_init.initialize_state_dict(all_layers.state_dict(), mi, source_checkpoint=source,
                                                       target_config=_model_config(all_layers))
    origin = dict(mode=mi["mode"], source=mi["from"], source_iteration=37500, alpha=0.5)
    reference = dict(model_state_dict=sd_all, iteration=0, epoch=0, config=_model_config(all_layers), moe_init=origin)
    rec_all["out"] = "all_layer_MOE_INIT.pth"
    sd, rec = worker_init.initialize_state_dict(selected.state_dict(), dict(mi, reference_init=rec_all["out"]),
                                               source_checkpoint=source, reference_checkpoint=reference,
                                               reference_receipt=rec_all, target_config=selective_config)
    shared_only = []
    for key in rec["mapped_shared"]:
        expert = key.replace("_shared", "_experts")
        if expert not in sd:
            shared_only.append(key)
            assert torch.equal(sd[key], source["model_state_dict"][expert])
        else:
            assert torch.equal(sd[key], source["model_state_dict"][expert] * 0.5)
    assert shared_only and rec["split_routed"]
    selected.load_state_dict(sd, strict=True)
    selected.eval(), dense.eval()
    data = _batch(dense)
    with torch.no_grad():
        a, b = dense(copy.deepcopy(data)), selected(copy.deepcopy(data))
    for key in ("node_features", "edge_features"):
        _check(a[key], b[key], record_property, "fp32_selective_final_" + key)


def test_checkpoint_metadata_strict_reload_and_resume_without_resplitting(worker_init, tmp_path, record_property):
    torch.manual_seed(132)
    dense, model = _models()
    sd, rec = _map(worker_init, model.state_dict(), dense.state_dict())
    model.load_state_dict(sd, strict=True)
    config = _model_config(model)
    origin = dict(mode="dense_to_shared", source=rec["source"], source_iteration=rec["source_iteration"], alpha=rec["alpha"])
    init_path = tmp_path / "MOE_INIT.pth"
    torch.save(dict(config=config, model_state_dict=sd, moe_init=origin, iteration=0, epoch=0), init_path)
    loaded = build_model(checkpoint=str(init_path))
    loaded.load_state_dict(sd, strict=True)
    assert loaded.moe_init_metadata == origin
    opt = torch.optim.SGD(loaded.parameters(), lr=1e-3, momentum=0.9)
    data = _batch(loaded)

    def step(m, o):
        o.zero_grad(set_to_none=True)
        out = m(copy.deepcopy(data))
        loss = out["node_features"].square().mean() + out["edge_features"].square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        o.step()
        return out

    step(loaded, opt)
    assert any(not torch.equal(sd[k], loaded.state_dict()[k]) for k in rec["split_routed"])
    saver = Saver()
    saver.trainer = SimpleNamespace(model=loaded, task="train", ep=1, iter=1, stats={})
    obj = saver._assemble_checkpoint_obj("step1", "iteration", config["model_options"], config["common_options"],
                                          {}, loaded.state_dict(), [dict(optimizer_state_dict=opt.state_dict())])
    assert obj["moe_init"] == origin and obj["moe_init"] is not loaded.moe_init_metadata
    path = tmp_path / "step1.pth"
    torch.save(obj, path)
    resumed = build_model(checkpoint=str(path))
    resumed.load_state_dict(obj["model_state_dict"], strict=True)
    assert resumed.moe_init_metadata == origin
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in loaded.state_dict().items())
    assert [n for n, _ in loaded.named_parameters()] == [n for n, _ in resumed.named_parameters()]
    resumed_opt = torch.optim.SGD(resumed.parameters(), lr=1e-3, momentum=0.9)
    resumed_opt.load_state_dict(torch.load(path, weights_only=False)["optimizer_state_dict"])
    for pa, pb in zip(loaded.parameters(), resumed.parameters()):
        for key, value in opt.state[pa].items():
            assert torch.equal(value, resumed_opt.state[pb][key]) if torch.is_tensor(value) else value == resumed_opt.state[pb][key]
    a, b = step(loaded, opt), step(resumed, resumed_opt)
    for (name, pa), (_, pb) in zip(loaded.named_parameters(), resumed.named_parameters()):
        if pa.grad is not None:
            assert pb.grad is not None
            record_property("fp32_resume_grad_" + name, float((pa.grad - pb.grad).abs().max()))
    for key in ("node_features", "edge_features"):
        assert torch.equal(a[key], b[key])
    for k, v in loaded.state_dict().items():
        _check(v, resumed.state_dict()[k], record_property, "fp32_resume_" + k)


def test_worker_dryrun_writes_alpha_receipt_checkpoint_and_resume_guard(worker_init, tmp_path):
    dense, target = _models()
    src = tmp_path / "S0_hp.pth"
    torch.save(dict(model_state_dict=dense.state_dict(), iteration=37500), src)
    (tmp_path / "target.json").write_text(json.dumps(_model_config(target)))
    # Stub only campaign path/log helpers. Execute the real worker, builder,
    # initialization helper, atomic checkpoint and receipt writes in a process.
    (tmp_path / "common_v1.py").write_text(
        "import json\nfrom pathlib import Path\nROOT=Path(__file__).parent\nD_CONFIGS=ROOT\nD_STATE=ROOT/'state'\n"
        "def stamp(): return 'test'\ndef write_json(path,obj): path.write_text(json.dumps(obj))\n")
    spec = dict(task_id="alpha", config="target.json", moe_init=dict(mode="dense_to_shared", alpha=0.5,
                                                                    **{"from": str(src)}))
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    env = dict(os.environ, NB_WORKER_BIN=str(tmp_path), NB_MOE_INIT_DRYRUN="1", CUDA_VISIBLE_DEVICES="")
    worker_path = Path(worker_init.__file__).with_name("task_worker_v21.py")
    cmd = [sys.executable, str(worker_path), str(path)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    folder = tmp_path / "state" / "alpha"
    rec = json.loads((folder / "MOE_INIT.json").read_text())
    checkpoint = torch.load(folder / "MOE_INIT.pth", weights_only=False)
    assert rec["ok"] and rec["alpha"] == checkpoint["moe_init"]["alpha"] == 0.5
    actual, _ = worker_init.initialize_state_dict(target.state_dict(), spec["moe_init"],
                                                 source_checkpoint=torch.load(src, weights_only=False),
                                                 target_config=_model_config(target))
    # The model seed differs, but every tensor copied/split from dense must match.
    for key in rec["copied"] + rec["mapped_shared"] + rec["split_routed"]:
        assert torch.equal(actual[key], checkpoint["model_state_dict"][key]), key
    generated = json.loads((folder / "SPEC_MOE_INIT.json").read_text())
    assert "moe_init" not in generated and generated["moe_init_file"] == str(folder / "MOE_INIT.pth")
    spec["resume_from"] = str(folder / "MOE_INIT.pth")
    path.write_text(json.dumps(spec))
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode != 0 and "moe_init cannot be combined" in result.stderr


@requires_so2_cuda
def test_alpha_gpu_fused_forward_and_all_gradients(worker_init, record_property, monkeypatch):
    from dptb.nn import so2_activation_routes as routes

    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "1")
    torch.manual_seed(82)
    opts = dict(irreps_in="4x0e+3x1o+2x2e", irreps_out="3x0e+2x1o+2x2e",
                mole_linear_mode="cublas_grouped", so2_fusion_mode="staged")
    dense = SO2_Linear(**opts, num_experts=1, num_shared_experts=0)
    staged = SO2_Linear(**opts, num_experts=24, num_shared_experts=1)
    sd, _ = _map(worker_init, staged.state_dict(), dense.state_dict())
    staged.load_state_dict(sd, strict=True)
    staged, dense = staged.cuda(), dense.cuda()
    fused = copy.deepcopy(staged)
    fused.so2_fusion_mode = "streamed_m_major_fused_p0"
    ra = MOLERouterV3(9, num_experts=24, top_k=2).cuda().eval()
    rb = copy.deepcopy(ra)
    xa = torch.randn(31, staged.irreps_in.dim, device="cuda", requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    ha = torch.randn(31, 9, device="cuda", requires_grad=True)
    hb = ha.detach().clone().requires_grad_(True)
    r = torch.randn(31, 3, device="cuda")

    def route(router, h):
        c, _, _ = router(h)
        idx, val = router.last_topk()
        return MOLEGlobals(coefficients=c, topk_indices=idx, topk_values=val,
                            activation_space=True, coefficients_sum_to_one=True)

    a = staged(xa, r, route(ra, ha))[0]
    with torch.no_grad():
        d = dense(xa, r, MOLEGlobals(coefficients=torch.ones(1, 1, device="cuda")))[0]
    _check(a, d, record_property, "gpu_fp32_alpha_vs_dense")
    before = routes.STATS.calls.get(routes.FUSED_P0, 0)
    b = fused(xb, r, route(rb, hb))[0]
    calls = routes.STATS.calls.get(routes.FUSED_P0, 0) - before
    assert calls > 0
    record_property("observed_fused_p0_calls", calls)
    record_property("gpu_fp32_forward", float((a - b).detach().abs().max()))
    torch.testing.assert_close(a, b, atol=3e-4, rtol=3e-4)
    ga = torch.autograd.grad(a.square().sum(), [xa, ha, *ra.parameters(), *staged.parameters()])
    gb = torch.autograd.grad(b.square().sum(), [xb, hb, *rb.parameters(), *fused.parameters()])
    for i, (aa, bb) in enumerate(zip(ga, gb)):
        record_property("gpu_fp32_grad_%d" % i, float((aa - bb).detach().abs().max()))
        torch.testing.assert_close(aa, bb, atol=2e-3, rtol=2e-3)

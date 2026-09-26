"""Real local LMDB single-head smoke; GPU script requires actual fused dispatch."""
import copy
import json
import os
from pathlib import Path
import time

import pytest
import torch

from dptb.data.dataloader import Collater
from dptb.nn.build import build_model
from dptb.nnops.loss import HamilLossAbs
from dptb.tests.atom_route_helpers import atom_config
from dptb.tests.shift_head_helpers import mini_dataset, close_dataset


@pytest.mark.parametrize("scope", ["onsite", "hopping"])
def test_real_single_head_training(scope, record_property):
    root = os.environ.get("R14C_MINI_ROOT")
    if not root:
        pytest.skip("Set R14C_MINI_ROOT to the real LMDB fixture")
    device = os.environ.get("R14C_TEST_DEVICE", "cpu")
    gpu = device.startswith("cuda")
    if gpu:
        assert torch.cuda.is_available()
    torch.manual_seed(1426)
    ds = mini_dataset(root, sidecar=False)
    try:
        data = Collater()([ds[i] for i in range(4)]).to_dict()
    finally:
        close_dataset(ds)
    cfg = atom_config(device=device)
    cfg["common_options"]["basis"] = json.loads((Path(root) / "basis.json").read_text())
    cfg["model_options"]["embedding"].update(
        irreps_hidden="2x0e+2x1o+2x2e+1x3o+1x4e+1x5o+1x6e", use_interpolation_out=True, latent_dim=4, latent_channels=[4],
        env_embed_multiplicity=1, r_max=8.,
        so2_fusion_mode="streamed_m_major_fused_p0" if gpu else "staged",
        mole_linear_mode="cublas_grouped" if gpu else "split_loop")
    cfg["train_options"] = dict(distance_ranges=[[0., 1e-6]] if scope == "onsite" else [[1e-6, 8.]],
                                clip_last_expert_range=True)
    model = build_model(**cfg)
    data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
    loss_fn = HamilLossAbs(idp=model.idp, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    rows = []
    routed_calls = []
    if gpu:
        from dptb.nn import so2_activation_routes as routes
        before = routes.STATS.calls.get(routes.FUSED_P0, 0)
        def before_routed(module, args):
            routed_calls.append(routes.STATS.calls.get(routes.FUSED_P0, 0))
        def after_routed(module, args, result):
            routed_calls[-1] = routes.STATS.calls.get(routes.FUSED_P0, 0) - routed_calls[-1]
        model.embedding.layers[1].register_forward_pre_hook(before_routed)
        model.embedding.layers[1].register_forward_hook(after_routed)
    for step in range(6):
        # The production HybridMuon publishes this automatically. Adam smoke
        # supplies its successful update count to the same logging field.
        model.embedding.router.opt_step = step
        if gpu:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(copy.deepcopy(data))
        em, nm = model._build_expert_masks(out, 0)
        assert bool(nm.any()) == (scope == "onsite")
        assert bool(em.any()) == (scope == "hopping")
        out.update(expert_edge_mask=em, expert_node_mask=nm)
        loss = loss_fn(out, data)
        assert torch.isfinite(loss)
        gn, ge = torch.autograd.grad(loss, [out["node_features"], out["edge_features"]],
                                     allow_unused=True, retain_graph=True)
        excluded, included = (ge, gn) if scope == "onsite" else (gn, ge)
        assert excluded is None or torch.count_nonzero(excluded) == 0
        assert included is not None and included.norm() > 0
        loss.backward()
        assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
        router_grad = model.embedding.router.net[0].weight.grad.norm().item()
        assert router_grad > 0
        opt.step()
        if gpu:
            torch.cuda.synchronize()
        rows.append(dict(step=step, loss=loss.item(), seconds=time.perf_counter() - start,
                         router_grad_norm=router_grad, atom_route=copy.deepcopy(model.embedding.last_atom_route_stats),
                         peak_cuda_bytes=torch.cuda.max_memory_allocated() if gpu else None))
    assert rows[-1]["loss"] < rows[0]["loss"]
    calls = routes.STATS.calls.get(routes.FUSED_P0, 0) - before if gpu else 0
    if gpu:
        assert calls > 0 and len(routed_calls) == 6 and min(routed_calls) > 0
    record_property("initial_loss", rows[0]["loss"])
    record_property("final_loss", rows[-1]["loss"])
    record_property("fused_p0_calls", calls)
    dest = os.environ.get("R14C_METRICS_DIR")
    if dest:
        Path(dest).mkdir(parents=True, exist_ok=True)
        Path(dest, f"train_{scope}_{device.replace(':', '_')}.json").write_text(json.dumps(
            dict(config=cfg, rows=rows, fused_p0_calls=calls, routed_layer_fused_calls=routed_calls,
                 nodes=data["pos"].shape[0],
                 edges=data["edge_index"].shape[1], parameters=sum(p.numel() for p in model.parameters())), indent=2))

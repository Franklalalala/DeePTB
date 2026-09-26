"""Opt-in real-record smoke for A, A-R, A-1 and U-A; no production metrics."""
import copy
import json
import os
from pathlib import Path

import pytest
import torch

from dptb.data.dataloader import Collater
from dptb.nn.build import build_model
from dptb.nn.structure_mole import fit_structure_stats, svd_split_state
from dptb.nnops.loss import HamilLossAbs
from dptb.tests.shift_head_helpers import mini_dataset, close_dataset
from dptb.tests.structure_mole_helpers import model_config


@pytest.mark.parametrize("arm", ["A", "A-R", "A-1", "U-A"])
def test_real_structure_mole_training(arm, record_property):
    root = os.environ.get("R14A_MINI_ROOT")
    if not root:
        pytest.skip("Set R14A_MINI_ROOT to the existing four-record local fixture")
    device = os.environ.get("R14A_TEST_DEVICE", "cpu")
    if device.startswith("cuda"):
        assert torch.cuda.is_available(), "requested GPU execution unavailable"
    ds = mini_dataset(root, sidecar=False)
    data = Collater()([ds[i] for i in range(4)]).to_dict()
    close_dataset(ds)
    cfg = model_config(scope="constant" if arm == "A-1" else "structure", prior=arm != "A-R", device=device,
                       n_layers=1, irreps_hidden="2x0e+2x1o+2x2e", latent_dim=4, latent_channels=[4],
                       edge_one_hot_dim=4, env_embed_multiplicity=1, r_max=8.,
                       mole_linear_mode="cublas_grouped" if device.startswith("cuda") else "split_loop")
    cfg["common_options"]["basis"] = json.loads((Path(root)/"basis.json").read_text())
    cfg["train_options"] = dict(distance_ranges=[[1e-6,8.]], clip_last_expert_range=False)
    model = build_model(**cfg)
    data = {k: v.to(device) if torch.is_tensor(v) else v for k,v in data.items()}
    if arm != "A-1":
        fit_structure_stats(model, [data], split="train")
    if arm == "U-A":
        dense_cfg = copy.deepcopy(cfg)
        dense_cfg["model_options"]["embedding"].update(num_experts=1, num_shared_experts=0, top_k=1,
                                                       mole_expert_parameterization="full", structure_mole={"enabled":False})
        dense = build_model(**dense_cfg)
        svd_split_state(model,dense.state_dict())
        da, db = dense(copy.deepcopy(data)),model(copy.deepcopy(data))
        for k in ("node_features","edge_features"):
            torch.testing.assert_close(da[k],db[k],atol=5e-5,rtol=5e-5)
    loss_fn = HamilLossAbs(idp=model.idp,device=device)
    opt = torch.optim.Adam(model.parameters(),lr=1e-4)
    losses=[]
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out=model(copy.deepcopy(data))
        em,nm=model._build_expert_masks(out,0)
        assert em.any() and not nm.any(), "hopping smoke must have no onsite loss"
        out.update(expert_edge_mask=em,expert_node_mask=nm)
        loss=loss_fn(out,data)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1]<losses[0],losses
    record_property("losses",json.dumps(losses))
    record_property("structure_metrics",json.dumps({k:v.tolist() for k,v in model.embedding.last_structure_metrics.items()}))

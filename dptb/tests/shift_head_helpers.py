"""Small SOC uu-real LEM fixtures shared by shift-head regressions."""
import copy
import torch
from dptb.nn.build import build_model


def config(mode=None, moe=False, device="cpu", scope="both"):
    embedding = dict(method="lem_moe_v3_edge_h0", n_layers=2, avg_num_neighbors=2.0,
        r_max=4.0, irreps_hidden="4x0e+2x1o+2x2e", env_embed_multiplicity=2,
        latent_dim=8, latent_channels=[8], edge_one_hot_dim=4,
        num_experts=2 if moe else 1, num_shared_experts=1 if moe else 0,
        top_k=2 if moe else 1, universal=True, use_layer_onehot_tp=False,
        use_out_onehot_tp=False, use_interpolation_out=False, tp_radial_emb=False,
        mole_linear_mode="split_loop", so2_fusion_mode="staged", equivariant_norm_type="none",
        edge_router_prior_activate=moe, edge_router_bias_speed=0.0)
    model_options = {"embedding": embedding, "prediction": {"method":"e3tb", "scale_type":"no_scale"}}
    if mode is not None:
        model_options["shift_head"] = {"mode": mode, "hidden": 8, "layers": 2}
    train_options = {}
    if scope != "both":
        train_options = {"distance_ranges": [[0., 1e-6]] if scope == "onsite" else [[1e-6, 2.0]],
                         "clip_last_expert_range": True}
    return {"model_options": model_options,
            "common_options": {"basis": {"H":"2s1p", "O":"2s1p"}, "has_soc": True,
                "nextham_uureal_mask": True, "overlap":False, "dtype":"float32", "device":device},
            "train_options": train_options}


def make_model(**kwargs):
    return build_model(**config(**kwargs))


def batch(model):
    device = model.device
    g = torch.Generator().manual_seed(23)
    r = model.idp.reduced_matrix_element
    data = {"pos": torch.tensor([[0.,0.,0.], [1.,.1,.2], [2.8,.2,.1]]),
            "edge_index":torch.tensor([[0,1,0,2,1,2],[1,0,2,0,2,1]]),
            "atomic_numbers":torch.tensor([[1],[8],[1]]),
            "node_h0":torch.randn(3,r,generator=g), "edge_h0":torch.randn(6,r,generator=g),
            "phys_node_overlap":torch.randn(3,r,generator=g), "phys_edge_overlap":torch.randn(6,r,generator=g),
            "node_features":torch.randn(3,r,generator=g), "edge_features":torch.randn(6,r,generator=g)}
    return model.idp({k:v.to(device) for k,v in data.items()})


def save_model(model, cfg, path):
    cfg = copy.deepcopy(cfg)
    cfg["model_options"] = copy.deepcopy(model.model_options)
    torch.save({"config":cfg,"model_state_dict":model.state_dict()},path)


def close_dataset(ds):
    for env in ds._lmdb_env_cache.values():
        env.close()
    ds._lmdb_env_cache = {}


def mini_dataset(root, sidecar=True, r_max=8.0):
    import json
    from pathlib import Path
    from dptb.data.build import DatasetBuilder
    root=Path(root)
    return DatasetBuilder()(root=str(root/'test'), r_max=r_max, type='LMDBDataset',prefix='data',
        get_Hamiltonian=True,get_H0=True,target_kind='h0res',basis=json.loads((root/'basis.json').read_text()),
        has_soc=True,nextham_uureal_mask=True,
        overlap_sidecar_root=str(root/'ovl_test') if sidecar else None)

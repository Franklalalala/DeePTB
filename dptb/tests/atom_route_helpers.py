"""Small late-routing models; no dependency on external task directories."""
import torch

from dptb.nn.build import build_model
from dptb.tests.shift_head_helpers import config, batch


def atom_config(parameterization="shared_core", device="cpu", **extra):
    cfg = config(moe=True, device=device)
    cfg["model_options"]["embedding"].update(
        n_layers=3, irreps_hidden="4x0e+2x1o+2x1e+2x2e", num_experts=4, top_k=4,
        so2_moe_layers=[1], edge_router_scope="atom_after_layer0",
        mole_expert_parameterization=parameterization, mole_expert_rank=3,
        **extra)
    return cfg


def atom_model(parameterization="shared_core", device="cpu", **extra):
    torch.manual_seed(1426)
    return build_model(**atom_config(parameterization, device, **extra))


def atom_batch(model):
    return batch(model)

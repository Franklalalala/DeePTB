"""Synthetic, label-independent structures for structure MoLE contracts."""
import copy

import torch

from dptb.nn.build import build_model
from dptb.nn.structure_mole import raw_statistics
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals
from dptb.tests.shift_head_helpers import config, batch


def model_config(scope="structure", execution="merged_core", prior=True, device="cpu", **extra):
    cfg = config(device=device)
    emb = cfg["model_options"]["embedding"]
    k = 1 if scope == "constant" else 4
    emb.update(num_experts=k, num_shared_experts=1, top_k=k,
               mole_expert_parameterization="shared_core", mole_expert_rank=3,
               so2_fusion_mode="streamed_m_major_fused_p0",
               structure_mole=dict(enabled=True, route_scope=scope, execution=execution, prior_stats=prior))
    emb.update(extra)
    return cfg


def structures(model):
    a = batch(model)
    b = copy.deepcopy(a)
    b["pos"] *= .73
    b["edge_h0"] *= 1.8
    b["node_h0"] *= .6
    c = copy.deepcopy(a)
    c["pos"] *= 1.1
    c["edge_h0"] *= -.4
    return [a, b, c]


def join(items, separate=True):
    node_offset = 0
    result = {}
    for i, item in enumerate(items):
        for key in ("pos", "atom_types", "node_h0", "edge_h0", "edge_type",
                    "node_features", "edge_features", "edge_index"):
            value = item[key]
            if key == "edge_index":
                value = value + node_offset
            if key == "pos":
                value = value + value.new_tensor([20. * i, 0., 0.])
            result.setdefault(key, []).append(value)
        result.setdefault("batch", []).append(torch.full_like(item["atom_types"].flatten(), i if separate else 0))
        node_offset += item["pos"].shape[0]
    return {k: torch.cat(v, 1 if k == "edge_index" else 0) for k, v in result.items()}


def calibrated_model(**kw):
    torch.manual_seed(140)
    model = build_model(**model_config(**kw))
    items = structures(model)
    if model.embedding.structure_stats is not None:
        model.embedding.structure_stats.fit((raw_statistics(model.embedding, x)[0] for x in items), split="train")
    return model, join(items)


def route(alpha, graph, execution):
    if execution == "reference":
        a = alpha[graph]
        return MOLEGlobals(coefficients=a, activation_space=True, coefficients_sum_to_one=True,
                           topk_indices=torch.arange(alpha.shape[1], device=a.device).expand(a.shape[0], -1),
                           topk_values=a)
    result = MOLEGlobals(coefficients=alpha, graph_index=graph, coefficients_sum_to_one=True)
    result.structure_execution = execution
    result.structure_rows = tuple(torch.where(graph == s)[0] for s in range(alpha.shape[0]))
    result.structure_inverse = torch.cat(result.structure_rows).argsort()
    return result

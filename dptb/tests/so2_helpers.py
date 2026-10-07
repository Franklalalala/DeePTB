"""Shared SO2 layer and activation builders for behavioral tests."""
import torch
from e3nn import o3

from dptb.nn.embedding.unitb_activations import build_gate_activation
from dptb.nn.tensor_product_moe_v3 import SO2_Linear

IRREPS_IN = "4x0e + 3x1o + 2x2e"
GATED_OUT = "3x0e + 2x1o + 2x2e"


def _layer(route, irreps_out, *, radial=False, interpolation=False, shared=1, dtype=torch.float64, device="cpu"):
    torch.manual_seed(20260925)
    layer = SO2_Linear(irreps_in=IRREPS_IN, irreps_out=irreps_out, radial_emb=radial,
                       latent_dim=8 if radial else None, radial_channels=[16] if radial else None,
                       use_interpolation=interpolation, num_experts=4, num_shared_experts=shared,
                       mole_linear_mode="indexed_ref", so2_fusion_mode=route)
    return layer.to(device=device, dtype=dtype)


def _inputs(layer, n=23, device="cpu", dtype=torch.float64, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, layer.irreps_in.dim, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    R = torch.randn(n, 3, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    lat = torch.randn(n, 8, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    return x.requires_grad_(True), R, lat


def _gate_layer(route="staged", dtype=torch.float64, device="cpu", **kw):
    gate = build_gate_activation(o3.Irreps(GATED_OUT))
    layer = _layer(route, str(gate.irreps_in), dtype=dtype, device=device, **kw)
    return layer, gate.to(device=device, dtype=dtype)


def _route_calls():
    from dptb.nn.so2_backend import STATS
    return STATS.calls["fused_p0"]

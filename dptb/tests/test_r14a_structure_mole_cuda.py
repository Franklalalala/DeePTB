"""Explicit GPU qualification; no skip or CPU fallback counts as a pass."""
import copy

import torch

from dptb.nn import so2_activation_routes as dispatch
from dptb.nn.structure_mole import svd_split_state
from dptb.nn.tensor_product_moe_v3 import SO2_Linear, MOLELinear
from dptb.nn.build import build_model
from dptb.tests.structure_mole_helpers import route, calibrated_model
from dptb.tests.shift_head_helpers import config


def test_gpu_fused_p0_structure_reference_and_merged_all_gradients(record_property):
    assert torch.cuda.is_available(), "GPU qualification requires visible CUDA"
    torch.manual_seed(1401)
    kwargs = dict(irreps_in="8x0e+6x1o+4x2e", irreps_out="6x0e+4x1o+4x2e",
                  num_experts=4, num_shared_experts=1, mole_expert_parameterization="shared_core",
                  mole_expert_rank=4, mole_linear_mode="cublas_grouped",
                  so2_fusion_mode="streamed_m_major_fused_p0")
    ref = SO2_Linear(**kwargs).cuda()
    merged = SO2_Linear(**kwargs).cuda()
    merged.load_state_dict(ref.state_dict(), strict=True)
    # Interleaved structure rows, complex m>0 layout, and nonuniform alpha.
    graph = torch.arange(91, device="cuda") % 7
    x = torch.randn(91, ref.irreps_in.dim, device="cuda", requires_grad=True)
    xm = x.detach().clone().requires_grad_(True)
    logits = torch.randn(7,4,device="cuda",requires_grad=True)
    lm = logits.detach().clone().requires_grad_(True)
    vectors = torch.randn(91,3,device="cuda")
    count = dispatch.STATS.calls.get(dispatch.FUSED_P0, 0)
    a = ref(x, vectors, route(logits.softmax(-1), graph, "reference"))[0]
    calls = dispatch.STATS.calls.get(dispatch.FUSED_P0, 0) - count
    assert calls > 0, "reference silently fell back from fused P0"
    record_property("observed_fused_p0_calls", calls)
    b = merged(xm, vectors, route(lm.softmax(-1), graph, "merged_core"))[0]
    assert dispatch.STATS.calls.get(dispatch.FUSED_P0, 0) == count + calls
    torch.testing.assert_close(a,b,atol=3e-4,rtol=3e-4)
    cotangent = torch.randn_like(a)
    ga = torch.autograd.grad((a*cotangent).sum(), [x,logits,*ref.parameters()])
    gb = torch.autograd.grad((b*cotangent).sum(), [xm,lm,*merged.parameters()])
    record_property("forward_max_error", float((a-b).abs().max()))
    record_property("gradient_max_error", max(float((aa-bb).abs().max()) for aa,bb in zip(ga,gb)))
    for aa,bb in zip(ga,gb):
        torch.testing.assert_close(aa,bb,atol=2e-3,rtol=2e-3)


def test_gpu_svd_dense_function_and_core_learning(record_property):
    assert torch.cuda.is_available(), "GPU qualification requires visible CUDA"
    dense = build_model(**config(device="cuda"))
    model, data = calibrated_model(device="cuda", mole_linear_mode="cublas_grouped")
    svd_split_state(model, dense.state_dict())
    a, b = dense(copy.deepcopy(data)), model(copy.deepcopy(data))
    for key in ("node_features", "edge_features"):
        torch.testing.assert_close(a[key], b[key], atol=5e-5, rtol=5e-5)
        record_property(key+"_max_error", float((a[key]-b[key]).abs().max()))
    loss = b["node_features"].square().sum()+b["edge_features"].square().sum()
    loss.backward()
    for layer in model.modules():
        if isinstance(layer,MOLELinear):
            assert torch.isfinite(layer.core_experts.grad).all()
    assert model.embedding.last_structure_alpha.std(0).max() > 1e-3

"""Offline wiring tests for the product_split_flow objective.

These exercise the parts that do NOT need the real e3tb model / idp / HR2HK: the
build_hamiltonian_flow dispatch, the model_in_loss contract, prepare_batch time
canonicalisation, and -- the novel bit -- the Euclidean H feature-space split-flow
consistency algebra (``step_r`` must equal the geodesic point ``H_r`` iff the model
predicts the clean endpoint).  The Grassmann-P term and a full model forward are covered
by the remote (natlan) smoke, which has a real idp + data.
"""
import torch

from dptb.nnops.flow import build_hamiltonian_flow, HamiltonianCFM, CFMContext
from dptb.nnops.product_split_flow import HamiltonianProductSplitFlow, ProductSplitContext

torch.manual_seed(0)


def _flow(**pm):
    opts = {"enabled": True, "objective": "product_split_flow", "product_manifold": pm}
    return build_hamiltonian_flow(opts, idp=None, dtype=torch.float64, device="cpu")


def test_build_dispatch_and_model_in_loss():
    obj = _flow(n_occ=5, lambda_h=1.0, lambda_p=0.0, mu_consistency=1.0)
    assert isinstance(obj, HamiltonianProductSplitFlow)
    assert isinstance(obj, HamiltonianCFM)
    assert obj.model_in_loss is True
    assert obj.h2k is None                     # no dense assembler when lambda_p == 0
    assert (obj.lambda_h, obj.mu_consistency, obj.lambda_p) == (1.0, 1.0, 0.0)


def test_build_dispatch_builds_hr2hk_when_p_on_but_needs_idp():
    # lambda_p>0 wants HR2HK, but idp=None -> stays None (guarded), no crash.
    obj = _flow(n_occ=5, lambda_p=0.1)
    assert obj.h2k is None                     # HR2HK needs idp; gracefully absent offline


def _node_context(n=3, f=4, t_cur=0.3, r_next=0.7):
    base = torch.randn(n, f, dtype=torch.float64)
    prior = torch.randn(n, f, dtype=torch.float64)
    target = torch.randn(n, f, dtype=torch.float64)      # the clean full-H RME endpoint
    res = target - base
    node_t = torch.full((n,), t_cur, dtype=torch.float64)
    node_current = base + (1.0 - t_cur) * prior + t_cur * res
    ctx_cfm = CFMContext(
        t=torch.tensor([t_cur], dtype=torch.float64),
        node_t=node_t, edge_t=None,
        node_base=base, edge_base=None,
        node_target=target, edge_target=None,
        node_current=node_current, edge_current=None,
        node_prior=prior, edge_prior=None,
    )
    psc = ProductSplitContext(cfm=ctx_cfm, t=torch.tensor([t_cur], dtype=torch.float64),
                              r=torch.tensor([r_next], dtype=torch.float64))
    return psc, target


def test_h_consistency_zero_at_clean_endpoint():
    obj = _flow(lambda_p=0.0)
    psc, target = _node_context()
    ref_data = {obj.node_target_key: target}
    pred_data = {obj.node_target_key: target.clone()}     # perfect clean-endpoint prediction
    loss = obj._h_consistency_loss(pred_data, ref_data, psc)
    assert float(loss) < 1e-18, float(loss)              # step_r == H_r exactly on the straight path


def test_h_consistency_positive_off_endpoint_and_differentiable():
    obj = _flow(lambda_p=0.0)
    psc, target = _node_context()
    ref_data = {obj.node_target_key: target}
    pred = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    loss = obj._h_consistency_loss({obj.node_target_key: pred}, ref_data, psc)
    assert float(loss) > 1e-6                             # off the clean endpoint -> nonzero
    loss.backward()
    assert pred.grad is not None and float(pred.grad.abs().max()) > 0.0   # trainable


def test_h_consistency_scales_like_extrapolation_factor():
    # For pred = target + d, step_r - H_r = (r-t)/(1-t) * d, so the loss = mean((c*d)^2)
    # with c = (r-t)/(1-t).  Check the closed form (node_weight=1).
    obj = _flow(lambda_p=0.0)
    t_cur, r_next = 0.25, 0.75
    psc, target = _node_context(t_cur=t_cur, r_next=r_next)
    d = 0.05 * torch.randn_like(target)
    ref_data = {obj.node_target_key: target}
    loss = obj._h_consistency_loss({obj.node_target_key: target + d}, ref_data, psc)
    c = (r_next - t_cur) / (1.0 - t_cur)
    expected = ((c * d) ** 2).mean()
    assert abs(float(loss) - float(expected)) < 1e-12


def test_prepare_batch_canonicalises_validation_times():
    # The trainer's model-in-loss validation passes (r=0, t=1); prepare_batch must
    # canonicalise to t_cur=0 <= r_next=1 and stamp the product time keys.
    obj = _flow(lambda_p=0.0)
    n, f = 2, 3
    target = torch.randn(n, f, dtype=torch.float64)
    base = torch.randn(n, f, dtype=torch.float64)          # H0 base (CFM residual mode needs it)
    data = {obj.node_target_key: target.clone(), obj.node_h0_key: base.clone()}
    ref_data = {obj.node_target_key: target.clone(), obj.node_h0_key: base.clone()}
    zero = torch.zeros(1, dtype=torch.float64)
    one = torch.ones(1, dtype=torch.float64)
    data_t, ref_t, ctx = obj.prepare_batch(data, ref_data, r=zero, t=one)
    assert isinstance(ctx, ProductSplitContext)
    assert float(ctx.t.max()) <= float(ctx.r.min()) + 1e-9      # t_cur <= r_next
    assert float(ctx.t.max()) < 1e-9 and float(ctx.r.min()) > 1.0 - 1e-9  # (0, 1)
    assert obj.flow_time_t_key in data_t and obj.flow_time_r_key in data_t


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fail = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except Exception as e:  # noqa
            fail += 1; print("FAIL", fn.__name__, ":", type(e).__name__, e)
    print(f"\n{len(fns)-fail}/{len(fns)} passed")
    sys.exit(1 if fail else 0)

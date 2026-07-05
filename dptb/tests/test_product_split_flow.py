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


def test_spinor_lift_overlap_is_noop_when_dims_agree():
    # non-SOC (or already-lifted) S: N==N -> returned unchanged (identity object).
    s = torch.randn(2, 4, 4, dtype=torch.complex128)
    h = torch.randn(2, 4, 4, dtype=torch.complex128)
    out = HamiltonianProductSplitFlow._spinor_lift_overlap(s, h)
    assert out is s


def test_spinor_lift_overlap_block_diag_matches_kron_I2():
    # SOC: H is 2N x 2N, S is N x N -> lift to block_diag(S,S) == I_2 (x) S, matching
    # HR2HK's block-diagonal spin-major layout (up block, then down block).
    n = 3
    s = torch.randn(2, n, n, dtype=torch.complex128)
    h = torch.zeros(2, 2 * n, 2 * n, dtype=torch.complex128)
    out = HamiltonianProductSplitFlow._spinor_lift_overlap(s, h)
    assert out.shape == (2, 2 * n, 2 * n)
    eye2 = torch.eye(2, dtype=torch.complex128)
    expected = torch.stack([torch.kron(eye2, s[k]) for k in range(2)], dim=0)
    assert torch.allclose(out, expected)
    # explicit block structure: top-left = bottom-right = S, off-diagonal spin blocks = 0.
    assert torch.allclose(out[:, :n, :n], s) and torch.allclose(out[:, n:, n:], s)
    assert torch.allclose(out[:, :n, n:], torch.zeros_like(s))
    assert torch.allclose(out[:, n:, :n], torch.zeros_like(s))


def test_spinor_lift_overlap_rejects_incompatible_dim():
    from dptb.nnops.grassmann import SkippableRecord
    s = torch.randn(5, 5, dtype=torch.complex128)          # 5 is neither == nor 2x of 7
    h = torch.zeros(7, 7, dtype=torch.complex128)
    try:
        HamiltonianProductSplitFlow._spinor_lift_overlap(s, h)
        raise AssertionError("expected SkippableRecord for incompatible S/H dims")
    except SkippableRecord:
        pass


def test_soc_lifted_overlap_gives_valid_occupied_projector():
    # The end-to-end point of the lift: occupied_projector(H_2N, S_2N, n_occ) must be
    # well-posed and S-orthonormal once S is lifted -- with a NON-orthogonal S and a
    # spin-coupled H (off-diagonal spin blocks) so the metric genuinely matters.
    from dptb.nnops._manifold_math import occupied_projector
    torch.manual_seed(1)
    n, n_occ = 4, 3
    a = torch.randn(n, n, dtype=torch.complex128)
    s_n = a @ a.mH + n * torch.eye(n, dtype=torch.complex128)     # HPD spatial overlap
    b = torch.randn(2 * n, 2 * n, dtype=torch.complex128)
    h_2n = b + b.mH                                                # Hermitian SOC H (spin-coupled)
    s_2n = HamiltonianProductSplitFlow._spinor_lift_overlap(s_n, h_2n)
    p, u, eps = occupied_projector(h_2n, s_2n, n_occ, return_frame=True)
    assert p.shape == (2 * n, 2 * n)
    assert torch.allclose(p, p.mH, atol=1e-9)                      # symmetric
    assert torch.allclose(p @ p, p, atol=1e-7)                     # idempotent
    # rank == n_occ (trace of an orthonormal projector counts occupied states)
    assert abs(float(p.diagonal(dim1=-2, dim2=-1).sum().real) - n_occ) < 1e-6
    # S-orthonormal occupied frame in the ORIGINAL metric: (S^{1/2} basis) C^H C == I.
    assert torch.allclose(u.mH @ u, torch.eye(n_occ, dtype=torch.complex128), atol=1e-7)
    # chordal self-distance is exactly zero (the loss floor).
    from dptb.nnops._manifold_math import chordal_distance_sq
    assert float(chordal_distance_sq(p, p)) < 1e-20


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

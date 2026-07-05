"""Tests for the flow-training layer: velocity head, endpoint alignment, trainable flow,
loss balance, projector consistency, and the conduction-window mechanism.

These go past the geometry (test_manifold_product) to the actual "接 flow" assembly:
* an endpoint-parameterised velocity head that is *exactly expressive* for the conditional
  geodesic, so the whole product-manifold flow can be **trained offline** (loss -> 0 by
  gradient descent through exp/log/proju/QR/distance) before the real e3tb head is wired in;
* the euler-1 endpoint-aligned validation score (the 58eb6b6/a2b7b7b fix);
* the balanced ``L = L_H + lambda*L_P + mu*consistency`` objective;
* the fixed-window conduction-band Euclidean penalty (the Part-2 "divide and conquer").
"""
import torch

from dptb.nnops.manifold_type import GrassmannManifold, EuclideanManifold, ProductManifold
from dptb.nnops.manifold_flow import (
    OrderedIntervalSampler,
    forward_flow,
    split_flow_loss,
    euler_sample,
    validation_endpoint_loss,
)
from dptb.nnops.manifold_flow_model import (
    EndpointProductHead,
    projector_consistency_loss,
    conduction_window_loss,
    product_flow_loss,
)

torch.manual_seed(0)


def _frame(n, k):
    q, _ = torch.linalg.qr(torch.randn(n, k, dtype=torch.float64))
    return q[:, :k]


def _nearby(u0, k, scale=0.3):
    u1 = _frame(u0.shape[0], k)
    if float(torch.linalg.svdvals(u0.mH @ u1).min()) < 0.2:
        u1, _ = torch.linalg.qr(u0 + scale * u1)
        u1 = u1[:, :k]
    return u1


def _product():
    return ProductManifold([EuclideanManifold(event_ndim=1), GrassmannManifold()])


def _endpoints(dim=5, n=8, k=2):
    x0 = (torch.randn(dim, dtype=torch.float64), _frame(n, k))
    x1 = (torch.randn(dim, dtype=torch.float64), _nearby(x0[1], k))
    return x0, x1


def _make_true_head(P, x1, dim=5, n=8, k=2):
    """A head whose predicted endpoint == the data endpoint x1 (the exact average velocity)."""
    head = EndpointProductHead(P, h_shape=(dim,), n=n, k=k)
    with torch.no_grad():
        head.h1.copy_(x1[0])
        head.raw_u.copy_(x1[1])       # already orthonormal; projx keeps the same projector
    return head


# --------------------------------------------------------------------------- #
# velocity head
# --------------------------------------------------------------------------- #
def test_head_endpoint_on_manifold_and_tangent_horizontal():
    P = _product()
    x0, x1 = _endpoints()
    head = EndpointProductHead(P, h_shape=(5,), n=8, k=2)
    he, ue = head.endpoint()
    assert torch.allclose(ue.mH @ ue, torch.eye(2, dtype=torch.float64), atol=1e-10)  # on Gr
    t = torch.tensor(0.3, dtype=torch.float64)
    s = torch.tensor(0.6, dtype=torch.float64)
    v = head(x0, t, s)
    # the Grassmann part of log(x0, endpoint) is horizontal at x0
    assert float((x0[1].mH @ v[1]).abs().max()) < 1e-9


# --------------------------------------------------------------------------- #
# endpoint alignment (euler-1 / euler-3 validation score)
# --------------------------------------------------------------------------- #
def test_validation_endpoint_zero_for_true_head():
    P = _product()
    x0, x1 = _endpoints()
    head = _make_true_head(P, x1)
    for ns in (1, 3):
        val = validation_endpoint_loss(P, head, x0, x1, num_steps=ns)
        assert float(val) < 1e-16, (ns, float(val))
    # one-shot euler sample lands exactly on the (data) endpoint
    reached = euler_sample(P, head, x0, 1)
    assert float(P.distance_sq(reached, x1)) < 1e-16


# --------------------------------------------------------------------------- #
# the actual "跑通": a velocity head TRAINS to the flow by gradient descent
# --------------------------------------------------------------------------- #
def test_split_flow_trains_the_head():
    torch.manual_seed(1)
    P = _product()
    x0, x1 = _endpoints()
    head = EndpointProductHead(P, h_shape=(5,), n=8, k=2, seed=3)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    gen = torch.Generator().manual_seed(7)

    def sample_tr():
        ab, _ = torch.sort(0.05 + 0.85 * torch.rand(2, generator=gen, dtype=torch.float64))
        return ab[0], ab[1]

    t0, r0 = sample_tr()
    init = float(split_flow_loss(P, head, x0, x1, t0, r0))
    for _ in range(600):
        t, r = sample_tr()
        loss = split_flow_loss(P, head, x0, x1, t, r)
        opt.zero_grad(); loss.backward(); opt.step()
    # endpoint recovered => loss on a fresh (t, r) is ~0; and the head endpoint matches x1
    final = float(split_flow_loss(P, head, x0, x1, torch.tensor(0.2, dtype=torch.float64),
                                  torch.tensor(0.7, dtype=torch.float64)))
    assert final < 1e-4, (init, final)
    he, ue = head.endpoint()
    assert float((he - x1[0]).abs().max()) < 1e-2
    assert float(P.manifolds[1].chordal_distance_sq(ue, x1[1])) < 1e-4


def test_meanflow_trains_the_head():
    torch.manual_seed(2)
    P = _product()
    x0, x1 = _endpoints()
    head = EndpointProductHead(P, h_shape=(5,), n=8, k=2, seed=5)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    gen = torch.Generator().manual_seed(11)

    def sample_ts():
        ab, _ = torch.sort(0.05 + 0.85 * torch.rand(2, generator=gen, dtype=torch.float64))
        return ab[0], ab[1]

    t0, s0 = sample_ts()
    init = float(product_flow_loss(P, head, x0, x1, t0, s0)[0])
    for _ in range(600):
        t, s = sample_ts()
        loss, _ = product_flow_loss(P, head, x0, x1, t, s)
        opt.zero_grad(); loss.backward(); opt.step()
    # MeanFlow identity residual -> ~0 and the endpoint is recovered
    final_val = float(validation_endpoint_loss(P, head, x0, x1, num_steps=1))
    assert final_val < 1e-3, (init, final_val)


# --------------------------------------------------------------------------- #
# loss balance L = L_H + lambda*L_P (+ mu*consistency)
# --------------------------------------------------------------------------- #
def test_product_flow_loss_lambda_weights_p_factor():
    P = _product()
    x0, x1 = _endpoints()
    head = EndpointProductHead(P, h_shape=(5,), n=8, k=2, seed=9)  # random => nonzero residual
    t = torch.tensor(0.3, dtype=torch.float64)
    s = torch.tensor(0.6, dtype=torch.float64)
    l1, parts1 = product_flow_loss(P, head, x0, x1, t, s, lam=1.0)
    l2, parts2 = product_flow_loss(P, head, x0, x1, t, s, lam=3.0)
    # L_H and L_P themselves are lambda-independent; only the combination changes by 2*L_P
    assert abs(float(parts1["L_H"]) - float(parts2["L_H"])) < 1e-12
    assert abs(float(parts1["L_P"]) - float(parts2["L_P"])) < 1e-12
    assert abs((float(l2) - float(l1)) - 2.0 * float(parts1["L_P"])) < 1e-10


# --------------------------------------------------------------------------- #
# mu term: physical consistency P vs occupied_projector(H)
# --------------------------------------------------------------------------- #
def test_projector_consistency_zero_when_frame_is_occupied_subspace():
    torch.manual_seed(4)
    N, n_occ = 6, 2
    a = torch.randn(N, N, dtype=torch.float64)
    h = a + a.T
    w, v = torch.linalg.eigh(h)
    u_occ = v[:, :n_occ]                       # the true occupied frame of h
    assert float(projector_consistency_loss(u_occ, h, n_occ=n_occ)) < 1e-10
    u_wrong = v[:, n_occ:2 * n_occ]            # a virtual frame => inconsistent
    assert float(projector_consistency_loss(u_wrong, h, n_occ=n_occ)) > 0.1


def test_product_flow_loss_includes_consistency_term():
    P = _product()
    # Euclidean factor as an N-vector here is just for the flow; the consistency_fn is exercised
    x0, x1 = _endpoints()
    head = _make_true_head(P, x1)
    t = torch.tensor(0.3, dtype=torch.float64)
    s = torch.tensor(0.6, dtype=torch.float64)
    called = {"n": 0}

    def cons_fn(x_t):
        called["n"] += 1
        return x_t[0].new_tensor(0.25)         # a fixed positive consistency

    loss, parts = product_flow_loss(P, head, x0, x1, t, s, mu=2.0, consistency_fn=cons_fn)
    assert called["n"] == 1
    assert abs(float(parts["consistency"]) - 0.25) < 1e-12
    # true head => flow residual ~0, so loss ~ mu * consistency = 0.5
    assert abs(float(loss) - 0.5) < 1e-6


# --------------------------------------------------------------------------- #
# conduction-window mechanism (fixed reference window, Part 2)
# --------------------------------------------------------------------------- #
def test_conduction_window_loss():
    torch.manual_seed(6)
    N = 8
    c = _frame(N, 4)                           # fixed reference window frame (occ + conduction)
    a = torch.randn(N, N, dtype=torch.float64)
    h_ref = a + a.T
    assert float(conduction_window_loss(h_ref, h_ref, c)) < 1e-20   # zero at the reference
    h_pred = h_ref + 0.1 * torch.randn(N, N, dtype=torch.float64)
    assert float(conduction_window_loss(h_pred, h_ref, c)) > 0.0
    # gradient descent on h_pred inside the window drives the penalty down
    hp = h_pred.clone().requires_grad_(True)
    opt = torch.optim.Adam([hp], lr=0.05)
    start = float(conduction_window_loss(hp, h_ref, c))
    for _ in range(200):
        loss = conduction_window_loss(hp, h_ref, c)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(conduction_window_loss(hp, h_ref, c)) < 0.05 * start


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

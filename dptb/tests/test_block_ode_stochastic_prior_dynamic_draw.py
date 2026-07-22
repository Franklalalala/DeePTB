"""RF3 characterization: stochastic prior re-resolves the draw engine per component.

The pre-refactor ``HamiltonianCFM`` residual/start-state draws spelled the
draw engine ``self._te_prior_like(...)`` (and the tied-irrep engine) once for
the node component and again for the edge component -- two live attribute
lookups off the instance.  The block-ode refactor's shared
``_draw_residual_component`` (``dptb.nnops.block_ode.stochastic_priors``)
captured the bound method once (``draw_fn = getattr(owner, name)``) and reused
it for both components, and likewise cached the prior-strength scale
(``te_prior_sigma`` / ``tied_irrep_sigma``) once for both assertion tuples.

That is RF3 (P2): the module docstring itself promises the engine is resolved
"dynamically ... at call time" so an instance-level monkeypatch / stateful
descriptor / subclass hook installed *during* the node draw is honoured for
the edge draw.  Capturing it once breaks that for any owner whose draw method
mutates between the node and edge calls.

This test drives a *real* ``HamiltonianCFM`` (built from the shared
``test_block_ode_flow`` fixtures, with a real ``BlockStateCodec`` and
``OrbitalMapper``) through the real ``_residual_te_eps`` delegator.  It
installs a swapping ``_te_prior_like`` on the live *instance* (isolated to the
one throwaway flow object -- no class-level state is touched) that records
which variant ran for each component and rebinds itself after the node draw,
then asserts node and edge resolved *different* live methods.  Each variant
delegates to the genuine draw engine so the downstream projection / belt is
exercised unchanged.  It FAILS on clean ``c7d097f`` (edge reuses the node's
captured method) and PASSES once the fix re-resolves the engine per component.
"""

from __future__ import annotations

from dptb.tests.test_block_ode_flow import _case, _flow, _fresh


def _projected_te_flow():
    idp, data, _codec, _h0 = _case()
    flow = _flow(
        idp,
        prior="projected_te",
        te_prior_mode="irrep",
        node_sigma=0.25,
        edge_sigma=0.25,
    )
    return flow, data


def test_te_draw_engine_reresolved_between_node_and_edge():
    flow, data = _projected_te_flow()
    node_like = data["node_h0"]
    edge_like = data["edge_h0"]

    genuine = flow._te_prior_like  # the real bound draw engine
    calls = []

    def second(*args, **kwargs):
        calls.append(("second", kwargs["label"]))
        return genuine(*args, **kwargs)

    def first(*args, **kwargs):
        calls.append(("first", kwargs["label"]))
        # A fault-injection hook / stateful owner may rebind the live method
        # after the node draw and expect the edge draw to re-resolve it.
        flow._te_prior_like = second
        return genuine(*args, **kwargs)

    # Instance-level shadow only; the flow is discarded at test end.
    flow._te_prior_like = first

    flow._residual_te_eps(
        _fresh(data),
        node_like,
        edge_like,
        generator=None,
        certify_image=False,
    )

    # Clean c7d097f captures ``draw_fn`` once, so the edge draw reuses ``first``
    # and this is ``[("first", "node"), ("first", "edge")]``.  The fix
    # re-resolves per component, so the edge draw sees the swapped ``second``.
    assert calls == [("first", "node"), ("second", "edge")]

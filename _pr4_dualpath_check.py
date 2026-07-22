"""Temporary PR4 dual-path differential check (block-ode stochastic_priors refactor).

Proves the NEW ``dptb.nnops.block_ode.stochastic_priors`` template functions
produce BITWISE-IDENTICAL (``torch.equal``) output to the OLD
``HamiltonianCFM`` method bodies they replace, across 5 fixed seeds x
{projected_te, tied_irrep_gaussian} for the pure-residual-epsilon draw (with
node_blocks/edge_blocks checked separately), plus a matching check for the
H0+noise start-state draw (``_block_initial_state``, both the seeded
projected_te path and the zero-prior early return), the
``_residual_stochastic_eps`` dispatcher, ``_strict_image_certification_due``,
and the two assertion belts (pass-through AND both raise paths, message text
included).

Run BEFORE deleting the old flow.py bodies -- this script calls both the
still-live old instance methods AND the new module functions independently,
passing the SAME live ``HamiltonianCFM`` instance as ``owner`` to the new
path. Once this passes and the check is committed into the branch's diff
history, the old bodies are deleted from flow.py in favor of delegators, per
the refactor plan section 3 (PR4 row).

Not a pytest test: standalone script, deleted before this PR's final commit
(dptb/tests/test_residual_ao_block_ode.py, test_tied_irrep_gaussian_prior.py,
and test_block_ode_flow.py provide the durable regression coverage).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dptb" / "tests"))

import torch

from dptb.nnops.block_ode import stochastic_priors
from test_residual_ao_block_ode import _b_flow, _b_record, _b_te_flow, _mapper
from test_tied_irrep_gaussian_prior import _b_tied_flow


SEEDS = [1000, 2000, 20260720, 314159, 999983]


def _check_residual_eps(label, flow_builder, old_method_name):
    mapper = _mapper()
    ok = 0
    for seed in SEEDS:
        flow = flow_builder(mapper)
        data, h0, _d1 = _b_record(mapper)
        node_base, edge_base = flow.block_codec.blocks_to_rme(copy.deepcopy(data), h0)

        old_method = getattr(flow, old_method_name)
        old_result = old_method(
            copy.deepcopy(data),
            node_base,
            edge_base,
            generator=flow._seeded_generator(node_base.device, seed),
            certify_image=True,
        )

        new_fn = getattr(stochastic_priors, old_method_name)
        new_result = new_fn(
            flow,
            copy.deepcopy(data),
            node_base,
            edge_base,
            generator=flow._seeded_generator(node_base.device, seed),
            certify_image=True,
        )

        assert torch.equal(old_result.node_blocks, new_result.node_blocks), (
            f"{label} seed={seed} node_blocks MISMATCH"
        )
        assert torch.equal(old_result.edge_blocks, new_result.edge_blocks), (
            f"{label} seed={seed} edge_blocks MISMATCH"
        )
        ok += 1
    print(f"[OK] {label}: {ok}/{len(SEEDS)} seeds, node+edge bitwise identical")


def _check_block_initial_state():
    mapper = _mapper()
    ok = 0
    for seed in SEEDS:
        flow = _b_te_flow(mapper)
        data, h0, _d1 = _b_record(mapper)

        old_start, old_node_res, old_edge_res = flow._block_initial_state(
            copy.deepcopy(data), h0, prior_seed=seed, certify_image=True
        )
        new_start, new_node_res, new_edge_res = stochastic_priors._block_initial_state(
            flow, copy.deepcopy(data), h0, prior_seed=seed, certify_image=True
        )
        assert torch.equal(old_start.node_blocks, new_start.node_blocks)
        assert torch.equal(old_start.edge_blocks, new_start.edge_blocks)
        assert torch.equal(old_node_res, new_node_res)
        assert torch.equal(old_edge_res, new_edge_res)
        ok += 1
    print(
        f"[OK] _block_initial_state (A-mode, projected_te): {ok}/{len(SEEDS)} "
        "seeds bitwise identical"
    )

    # zero-prior branch too (no randomness, but exercises the early-return path).
    flow_zero = _b_flow(mapper)  # prior='zero'
    data, h0, _d1 = _b_record(mapper)
    old_start, old_node_res, old_edge_res = flow_zero._block_initial_state(
        copy.deepcopy(data), h0
    )
    new_start, new_node_res, new_edge_res = stochastic_priors._block_initial_state(
        flow_zero, copy.deepcopy(data), h0
    )
    assert torch.equal(old_start.node_blocks, new_start.node_blocks)
    assert torch.equal(old_start.edge_blocks, new_start.edge_blocks)
    assert torch.equal(old_node_res, new_node_res)
    assert torch.equal(old_edge_res, new_edge_res)
    print("[OK] _block_initial_state (zero-prior early return) bitwise identical")


def _check_dispatcher_and_misc():
    mapper = _mapper()

    flow_te = _b_te_flow(mapper)
    data, h0, _d1 = _b_record(mapper)
    node_base, edge_base = flow_te.block_codec.blocks_to_rme(copy.deepcopy(data), h0)
    old = flow_te._residual_stochastic_eps(
        copy.deepcopy(data), node_base, edge_base,
        generator=flow_te._seeded_generator(node_base.device, SEEDS[0]), certify_image=True,
    )
    new = stochastic_priors._residual_stochastic_eps(
        flow_te, copy.deepcopy(data), node_base, edge_base,
        generator=flow_te._seeded_generator(node_base.device, SEEDS[0]), certify_image=True,
    )
    assert torch.equal(old.node_blocks, new.node_blocks)
    assert torch.equal(old.edge_blocks, new.edge_blocks)
    print("[OK] _residual_stochastic_eps dispatcher (te branch) bitwise identical")

    flow_tied = _b_tied_flow(mapper)
    old2 = flow_tied._residual_stochastic_eps(
        copy.deepcopy(data), node_base, edge_base,
        generator=flow_tied._seeded_generator(node_base.device, SEEDS[0]), certify_image=True,
    )
    new2 = stochastic_priors._residual_stochastic_eps(
        flow_tied, copy.deepcopy(data), node_base, edge_base,
        generator=flow_tied._seeded_generator(node_base.device, SEEDS[0]), certify_image=True,
    )
    assert torch.equal(old2.node_blocks, new2.node_blocks)
    assert torch.equal(old2.edge_blocks, new2.edge_blocks)
    print("[OK] _residual_stochastic_eps dispatcher (tied_irrep branch) bitwise identical")

    # _strict_image_certification_due: pure scheduling, no randomness.
    for mode, _period in [("always", 1), ("first_batch", 1), ("every_n(3)", 3)]:
        flow = _b_flow(mapper, strict_certification=mode)
        for batch in range(6):
            flow._strict_certification_batches = batch
            old_due = flow._strict_image_certification_due()
            new_due = stochastic_priors._strict_image_certification_due(flow)
            assert old_due == new_due, f"mode={mode} batch={batch}: {old_due} vs {new_due}"
    print("[OK] _strict_image_certification_due: all mode/batch combos match")

    # Assertion belts: exercise pass and both raise paths identically.
    zeros = torch.zeros(2, 4, 4, dtype=torch.float64)
    live = torch.randn(2, 4, 4, dtype=torch.float64)
    nan = zeros.clone()
    nan[0, 0, 0] = float("nan")

    for components in (
        (("node", live, 1.0),),
        (("node", zeros, 0.0),),
    ):
        flow_te._assert_projected_te_draw_finite_and_scaled(components)
        stochastic_priors._assert_projected_te_draw_finite_and_scaled(flow_te, components)

    for components, _match in (
        ((("node", zeros, 1.0),), "zero"),
        ((("edge", nan, 1.0),), "NaN"),
    ):
        old_exc = new_exc = None
        try:
            flow_te._assert_projected_te_draw_finite_and_scaled(components)
        except ValueError as exc:
            old_exc = str(exc)
        try:
            stochastic_priors._assert_projected_te_draw_finite_and_scaled(flow_te, components)
        except ValueError as exc:
            new_exc = str(exc)
        assert old_exc is not None and new_exc is not None, (old_exc, new_exc)
        assert old_exc == new_exc, f"message mismatch:\n  old={old_exc!r}\n  new={new_exc!r}"
    print("[OK] assertion belts: pass-through + raise messages bitwise/text identical")


if __name__ == "__main__":
    _check_residual_eps("_residual_te_eps", _b_te_flow, "_residual_te_eps")
    _check_residual_eps(
        "_residual_tied_irrep_gaussian_eps", _b_tied_flow, "_residual_tied_irrep_gaussian_eps"
    )
    _check_block_initial_state()
    _check_dispatcher_and_misc()
    print("\nALL PR4 DUAL-PATH CHECKS PASSED.")

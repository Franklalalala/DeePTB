# SPDX-License-Identifier: LGPL-3.0-or-later
"""Data-gated integration test for the `residual_hamiltonian` target switch.

With the switch off the block-native target is absolute H (mean|.| ~= 0.24 on the
water QHFlow2 set); with it on the target is dH = H - H0 (mean|.| ~= 0.0155).
Skips automatically when the reference LMDB is not present (e.g. CI).

Runnable directly (prints the numbers) or under pytest.
"""
import os

import numpy as np
import pytest
import torch

DATA = "/home/mingkang_nt/codex/qhflow_water_dptb_20260608/dptb_raw/water_qhflow2_seed42/train"
R_MAX = 7.408480947893776
BASIS = {"H": "2s1p", "O": "3s2p1d"}


def _build(residual):
    from dptb.data.build import DatasetBuilder

    return DatasetBuilder()(
        root=DATA,
        prefix="data",
        type="LMDBDataset",
        separator=".",
        r_max=R_MAX,
        basis=BASIS,
        get_Hamiltonian=True,
        residual_hamiltonian=residual,
    )


def _masked_mean_abs(data):
    from dptb.data.interfaces.blockwise_tensor import (
        NODE_DELTA_HAMIL_BLOCKS_KEY,
        EDGE_DELTA_HAMIL_BLOCKS_KEY,
        NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
        EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    )

    def _one(block_key, shape_key):
        blocks = torch.as_tensor(data[block_key]).abs().double()
        shapes = torch.as_tensor(data[shape_key]).long()
        tot, cnt = 0.0, 0
        for b, (r, c) in zip(blocks, shapes):
            tot += float(b[: int(r), : int(c)].sum())
            cnt += int(r) * int(c)
        return tot, cnt

    tn, cn = _one(NODE_DELTA_HAMIL_BLOCKS_KEY, NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY)
    te, ce = _one(EDGE_DELTA_HAMIL_BLOCKS_KEY, EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY)
    return {
        "onsite": tn / max(cn, 1),
        "hopping": te / max(ce, 1),
        "all": (tn + te) / max(cn + ce, 1),
    }


@pytest.mark.skipif(not os.path.isdir(DATA), reason="water QHFlow2 LMDB not available")
def test_residual_switch_changes_target_scale():
    off = _masked_mean_abs(_build(False).get(0))
    on = _masked_mean_abs(_build(True).get(0))
    print("OFF (absolute H):", {k: round(v, 5) for k, v in off.items()})
    print("ON  (residual dH):", {k: round(v, 5) for k, v in on.items()})
    # absolute-H target
    assert 0.20 < off["all"] < 0.30, off
    # residual dH target is ~16x smaller
    assert 0.010 < on["all"] < 0.025, on
    assert on["all"] < 0.15 * off["all"], (on, off)


def test_residual_guard_accepts_genuine_full_h():
    """Full-H slot: subtracting a close H0 shrinks the target -> guard passes."""
    from dptb.data.dataset.lmdb_dataset import assert_residual_target_shrinks

    rng = np.random.default_rng(0)
    h0 = {"1_1_0_0_0": rng.normal(0.0, 1.0, (5, 5)), "1_2_0_0_0": rng.normal(0.0, 0.5, (5, 5))}
    blocks = {k: v + rng.normal(0.0, 0.02, v.shape) for k, v in h0.items()}
    delta = {k: np.asarray(blocks[k]) - np.asarray(h0[k]) for k in blocks}
    assert_residual_target_shrinks(blocks, delta)


def test_residual_guard_rejects_delta_in_h_slot():
    """Delta-in-H-slot convention: subtracting H0 inflates -> guard raises."""
    from dptb.data.dataset.lmdb_dataset import assert_residual_target_shrinks

    rng = np.random.default_rng(1)
    # The "hamiltonian" slot already holds a small residual, as in the 0516
    # NexTHam crystal LMDBs; H0 is full-H scale.
    blocks = {"1_1_0_0_0": rng.normal(0.0, 0.015, (5, 5))}
    h0 = {"1_1_0_0_0": rng.normal(0.0, 1.0, (5, 5))}
    delta = {k: np.asarray(blocks[k]) - np.asarray(h0[k]) for k in blocks}
    with pytest.raises(RuntimeError, match="double-subtract"):
        assert_residual_target_shrinks(blocks, delta)


def test_residual_guard_rejects_zero_or_garbage_h0():
    """H0 ~ 0 leaves the target unshrunk -> guard raises (no silent no-op)."""
    from dptb.data.dataset.lmdb_dataset import assert_residual_target_shrinks

    rng = np.random.default_rng(2)
    blocks = {"1_1_0_0_0": rng.normal(0.0, 1.0, (5, 5))}
    h0 = {"1_1_0_0_0": np.zeros((5, 5))}
    delta = {k: np.asarray(blocks[k]) - np.asarray(h0[k]) for k in blocks}
    with pytest.raises(RuntimeError, match="double-subtract"):
        assert_residual_target_shrinks(blocks, delta)


def test_residual_guard_keeps_complex_magnitude():
    """Imaginary components must participate in the shrink decision."""
    from dptb.data.dataset.lmdb_dataset import assert_residual_target_shrinks

    blocks = {"1_1_0_0_0": np.array([[1.0 + 10.0j]])}
    delta = {"1_1_0_0_0": np.array([[0.1 + 9.0j]])}
    # Real-only magnitudes would incorrectly see a 10x shrink and pass. The
    # complex magnitude shrinks by only ~1.12x and must fail the 1.2x guard.
    with pytest.raises(RuntimeError, match="does not shrink"):
        assert_residual_target_shrinks(blocks, delta)


def test_residual_builder_rejects_prepacked_target_provenance():
    from dptb.data.dataset.lmdb_dataset import build_residual_hamiltonian_target_blocks

    data = {
        "hamiltonian_0": {"1_1_0_0_0": np.zeros((1, 1))},
        "node_delta_hamil_blocks": np.zeros((1, 1, 1)),
    }
    blocks = {"1_1_0_0_0": np.ones((1, 1))}
    with pytest.raises(ValueError, match="already contains prepacked"):
        build_residual_hamiltonian_target_blocks(data, blocks)


def test_residual_builder_validates_every_call_and_block_shape():
    from dptb.data.dataset.lmdb_dataset import build_residual_hamiltonian_target_blocks

    good_blocks = {"1_1_0_0_0": np.ones((2, 2))}
    good_data = {"hamiltonian_0": {"1_1_0_0_0": np.full((2, 2), 0.95)}}
    delta = build_residual_hamiltonian_target_blocks(good_data, good_blocks)
    assert np.allclose(delta["1_1_0_0_0"], 0.05)

    bad_data = {"hamiltonian_0": {"1_1_0_0_0": np.zeros((1, 2))}}
    with pytest.raises(ValueError, match="mismatched Hamiltonian/H0 shapes"):
        build_residual_hamiltonian_target_blocks(bad_data, good_blocks)


if __name__ == "__main__":
    off = _masked_mean_abs(_build(False).get(0))
    on = _masked_mean_abs(_build(True).get(0))
    print("OFF (absolute H) :", {k: round(v, 5) for k, v in off.items()})
    print("ON  (residual dH):", {k: round(v, 5) for k, v in on.items()})
    print("ratio all on/off :", round(on["all"] / off["all"], 4))

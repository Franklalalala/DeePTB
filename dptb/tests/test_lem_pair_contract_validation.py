from __future__ import annotations

import pytest
import torch

from dptb.nn.embedding.lem_pair import LemPair, _canonicalize_mp_cutoff

from test_lem_pair_common import fp64_default, model_options


def _build(**updates):
    options = model_options()
    options.update(updates)
    torch.manual_seed(20260723)
    return LemPair(**options)


def test_cutoff_and_degree_contracts_fail_closed():
    with fp64_default():
        with pytest.raises(ValueError, match="mp_cutoff must be finite"):
            _build(mp_cutoff=0.0)
        with pytest.raises(ValueError, match="must cover every basis species"):
            _build(mp_cutoff={"H": 1.0})
        with pytest.raises(ValueError, match="mp_avg_num_neighbors"):
            _build(mp_avg_num_neighbors=0.0)


def test_additive_residual_rejects_unused_learnable_ratio_parameters():
    with pytest.raises(ValueError, match="res_update_ratios_learnable=false"):
        _build(
            res_update_additive=True,
            res_update_ratios_learnable=True,
        )


class _TwoSpeciesMapper:
    basis = {"H": "1s", "C": "1s1p"}
    bond_to_type = {"H-H": 0, "H-C": 1, "C-H": 2, "C-C": 3}


@pytest.mark.parametrize(
    ("mp_cutoff", "r_max"),
    [
        (6.0, 5.0),
        ({"H": 5.0, "C": 5.0}, 5.0),
        (6.0, {"H": 4.0, "C": 6.0}),
        ({"H": 4.0, "C": 6.0}, {"H": 3.0, "C": 5.0}),
    ],
)
def test_redundant_mp_cutoff_canonicalizes_for_scalar_and_dict_pairs(
    mp_cutoff, r_max
):
    assert _canonicalize_mp_cutoff(mp_cutoff, r_max, _TwoSpeciesMapper()) is None


def test_mp_cutoff_stays_dual_when_redundancy_cannot_be_proved():
    cutoff = {"H": 4.0, "C": 6.0}
    assert (
        _canonicalize_mp_cutoff(
            cutoff,
            {"H": 3.0},
            _TwoSpeciesMapper(),
        )
        == cutoff
    )

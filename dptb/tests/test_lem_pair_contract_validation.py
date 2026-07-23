from __future__ import annotations

import pytest
import torch

from dptb.nn.embedding.lem_pair import LemPair

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

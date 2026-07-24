from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nn.embedding.lem_pair import (
    LemPair,
    _canonicalize_mp_cutoff,
    _get_mp_edge_mask,
    load_lem_h0_backbone,
)
from dptb.utils.argcheck import model_options as model_options_argcheck

from test_lem_pair_common import fp64_default, model_options


def _build(**updates):
    options = model_options()
    options.update(updates)
    torch.manual_seed(20260723)
    return LemPair(**options)


def test_cutoff_and_degree_contracts_fail_closed():
    with fp64_default():
        for cutoff in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="mp_cutoff must be finite"):
                _build(mp_cutoff=cutoff)
        with pytest.raises(TypeError, match="must not be boolean"):
            _build(mp_cutoff=True)
        with pytest.raises(ValueError, match="must cover every basis species"):
            _build(mp_cutoff={"H": 1.0})
        with pytest.raises(ValueError, match="unknown basis species"):
            _build(mp_cutoff={"O": 1.0, "Xe": 1.0})
        for degree in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="mp_avg_num_neighbors"):
                _build(mp_avg_num_neighbors=degree)
        with pytest.raises(TypeError, match="mp_avg_num_neighbors"):
            _build(mp_avg_num_neighbors=True)


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


def test_runtime_dict_mask_rejects_malformed_or_incomplete_bond_contracts():
    edge_length = torch.tensor([0.5])
    bond_type = torch.tensor([0])

    class Malformed:
        bond_to_type = {"H-C-extra": 0}

    with pytest.raises(ValueError, match="two elements"):
        _get_mp_edge_mask(
            edge_length,
            bond_type,
            Malformed(),
            {"H": 1.0, "C": 1.0},
        )

    class Missing:
        bond_to_type = {"H-C": 0}

    with pytest.raises(KeyError, match="missing an element"):
        _get_mp_edge_mask(
            edge_length,
            bond_type,
            Missing(),
            {"H": 1.0},
        )


def test_legacy_h0_method_rejects_pair_only_mp_cutoff_in_strict_argcheck():
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(
        (root / "configs" / "route_h_b0_late_block_expansion_cg.yaml").read_text()
    )["model_options"]
    payload["embedding"]["method"] = "lem_moe_v3_h0"
    payload["embedding"]["mp_cutoff"] = 1.0
    argument = model_options_argcheck()
    normalized = argument.normalize_value(payload)
    with pytest.raises(Exception, match="mp_cutoff"):
        argument.check_value(normalized, strict=True)


@pytest.mark.parametrize("method", ["lem_pair", "lem_cutoff"])
def test_only_actual_mp_cutoff_consumers_accept_the_field(method):
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(
        (root / "configs" / "route_h_b0_late_block_expansion_cg.yaml").read_text()
    )["model_options"]
    payload["embedding"]["method"] = method
    payload["embedding"]["mp_cutoff"] = 1.0
    argument = model_options_argcheck()
    normalized = argument.normalize_value(payload)
    argument.check_value(normalized, strict=True)


def test_legacy_h0_backbone_migration_is_allowlisted_and_fail_closed():
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    with fp64_default():
        torch.manual_seed(20260723)
        legacy = LemMoEV3H0(**options).eval()
        torch.manual_seed(20260723)
        dual = LemPair(
            **options,
            mp_cutoff=1.0,
            mp_avg_num_neighbors=1.5,
            pair_refine_enable=True,
            pair_refine_rank=4,
            pair_refine_init=0.1,
        ).eval()

    allowed = (
        "dual_cutoff_readout_normalization",
        "dual_cutoff_pair_readout.",
        "dual_cutoff_edge_context_projection.",
        "pair_refine.",
    )
    incompatible = load_lem_h0_backbone(
        dual,
        legacy.state_dict(),
        allowed_missing_prefixes=allowed,
    )
    missing = tuple(incompatible.missing_keys)
    assert missing
    assert all(
        any(
            key.startswith(prefix)
            or (
                prefix.startswith("dual_cutoff_")
                and f".{prefix}" in key
            )
            for prefix in allowed
        )
        for key in missing
    )
    assert all(
        any(
            key.startswith(prefix)
            or (
                prefix.startswith("dual_cutoff_")
                and f".{prefix}" in key
            )
            for key in missing
        )
        for prefix in allowed
    )

    # Keep nn.Module state_dict metadata: stripping it is now deliberately
    # rejected for affected high-l H0 checkpoints before key migration runs.
    tampered = legacy.state_dict()
    original_key = next(iter(tampered))
    tampered[original_key + "_typo"] = tampered.pop(original_key)
    with pytest.raises(RuntimeError, match="migration rejected"):
        load_lem_h0_backbone(
            dual,
            tampered,
            allowed_missing_prefixes=allowed,
        )

from pathlib import Path

import pytest
import yaml

from dptb.nn.embedding.output_routes import OFFICIAL_OUTPUT_ROUTES
from dptb.utils.argcheck import model_options


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "h_a0": "route_h_a0_late_rme_expansion_nocg.yaml",
    "h_a1": "route_h_a1_late_rme_cartesian_hybrid.yaml",
    "h_b0": "route_h_b0_late_block_expansion_cg.yaml",
    "h_b1": "route_h_b1_late_block_cartesian_projector.yaml",
    "p_b0": "route_p_b0_direct_ao_projector_wigner.yaml",
    "p_b1_ict": "route_p_b1_direct_ao_projector_ict_bank.yaml",
}


def test_six_canonical_route_configs_pass_strict_argcheck():
    assert tuple(CONFIGS) == OFFICIAL_OUTPUT_ROUTES
    model_arg = model_options()
    for route, filename in CONFIGS.items():
        payload = yaml.safe_load((ROOT / "configs" / filename).read_text())
        normalized = model_arg.normalize_value(payload["model_options"])
        model_arg.check_value(normalized, strict=True)
        assert normalized["embedding"]["output_route"] == route


def test_output_route_and_conflicting_legacy_alias_are_rejected_by_model(tmp_path):
    # dargs validates the two compatibility fields independently; semantic
    # conflict is deliberately centralized in the route registry/model build.
    from dptb.nn.embedding.output_routes import resolve_output_route

    with pytest.raises(ValueError, match="conflicts"):
        resolve_output_route(
            output_route="h_a0",
            legacy_mode="late_block_expansion_cg",
        )

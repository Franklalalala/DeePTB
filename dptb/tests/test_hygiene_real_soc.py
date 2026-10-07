"""Set DPTB_HYGIENE_SOC_CONFIG to runs/base_on.json plus its real LMDB/sidecar assets."""
import copy
import os
from pathlib import Path

import pytest
import torch

from dptb.nn.build import build_model
from dptb.tests.hygiene_soc_helpers import reference_config, reference_data


def test_real_soc_global_float64_construction_and_forward():
    path = os.environ.get("DPTB_HYGIENE_SOC_CONFIG")
    if not path or not Path(path).is_file():
        pytest.skip("requires DPTB_HYGIENE_SOC_CONFIG pointing to runs/base_on.json and its real SOC data")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        options = reference_config(path, "float64")
        for key in ("root", "overlap_sidecar_root"):
            asset = options["data_options"]["validation"].get(key)
            if asset and not Path(asset).is_dir():
                pytest.skip(f"real SOC validation asset missing: {key}={asset}")
        assert options["common_options"]["has_soc"]
        assert options["common_options"]["nextham_uureal_mask"]
        torch.manual_seed(42)
        model = build_model(model_options=copy.deepcopy(options["model_options"]),
                            common_options=copy.deepcopy(options["common_options"]),
                            train_options=copy.deepcopy(options["train_options"])).eval()
        constants = [v for m in model.modules() if hasattr(m, "soc_base_matrix")
                     for v in m.soc_base_matrix.values()]
        assert constants and all(v.dtype == torch.complex128 for v in constants)
        assert all(p.dtype == torch.float64 for p in model.parameters() if p.is_floating_point())
        data = reference_data(options)
        assert data["node_h0"].numel() > 0 and data["edge_h0"].numel() > 0
        with torch.no_grad():
            out = model(copy.deepcopy(data))
        for part in ("node_features", "edge_features"):
            assert out[part].numel() > 0
            assert out[part].dtype == torch.float64
            assert torch.isfinite(out[part]).all()
    finally:
        torch.set_default_dtype(previous)

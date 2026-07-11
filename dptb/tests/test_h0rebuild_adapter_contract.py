from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dptb.data.interfaces.h0rebuild_adapter import (
    PHYSICAL_H0_SIDECAR_SCHEMA,
    array_sha256,
    build_physical_h0_meta,
)
from dptb.utils.argcheck import flow_options
from tools.materialize_h0rebuild_lmdb import _validated_roots


def _record():
    return {
        "atomic_numbers": np.array([14, 14], dtype=np.int64),
        "pos": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        "cell": np.eye(3, dtype=np.float64) * 5.0,
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "edge_cell_shift": np.zeros((2, 3), dtype=np.float64),
        "node_physical_h0": np.zeros((2, 8), dtype=np.float64),
        "edge_physical_h0": np.zeros((2, 8), dtype=np.float64),
    }


def test_physical_h0_meta_is_order_sensitive_and_hashed():
    record = _record()
    meta = build_physical_h0_meta(record, energy_unit="eV")
    assert meta["schema"] == PHYSICAL_H0_SIDECAR_SCHEMA
    assert meta["node_sha256"] == array_sha256(record["node_physical_h0"])
    changed = _record()
    changed["edge_index"] = changed["edge_index"][:, ::-1]
    changed["edge_cell_shift"] = changed["edge_cell_shift"][::-1]
    changed_meta = build_physical_h0_meta(changed, energy_unit="eV")
    assert changed_meta["edge_signature"] != meta["edge_signature"]


def test_physical_h0_meta_rejects_raw_complex_features():
    record = _record()
    record["edge_physical_h0"] = record["edge_physical_h0"].astype(np.complex128)
    with pytest.raises(TypeError, match="real floating rank-2"):
        build_physical_h0_meta(record)


def test_materializer_rejects_nested_input_output_roots(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()

    with pytest.raises(ValueError, match="non-nested"):
        _validated_roots(str(input_root), str(input_root / "output"))
    with pytest.raises(ValueError, match="non-nested"):
        _validated_roots(str(input_root), str(tmp_path))


def test_physical_h0_overlay_uses_train_options_schema():
    repo_root = Path(__file__).resolve().parents[2]
    overlay = yaml.safe_load(
        (repo_root / "configs" / "physical_h0_flow_overlay.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert "flow_options" not in overlay
    flow = overlay["train_options"]["flow_options"]
    assert flow["node_h0_key"] == "node_physical_h0"
    assert flow["edge_h0_key"] == "edge_physical_h0"
    normalized = flow_options().normalize_value(flow)
    flow_options().check_value(normalized, strict=True)

    # P0 wiring fix: the H0-init embedding must read exactly the keys the flow
    # overwrites with the interpolated state x_t.  If the embedding is left at
    # the stored-h0 defaults (node_h0/edge_h0) while the flow points at the
    # physical keys, x_t never reaches the network and the prior is silently
    # deactivated.  The overlay must therefore repoint the embedding keys too,
    # and they must stay aligned with the flow keys.
    embedding = overlay["model_options"]["embedding"]
    assert embedding["h0_node_key"] == flow["node_h0_key"] == "node_physical_h0"
    assert embedding["h0_edge_key"] == flow["edge_h0_key"] == "edge_physical_h0"

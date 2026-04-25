import ast
from pathlib import Path


LEM_MOE_V3 = Path(__file__).resolve().parents[1] / "nn" / "embedding" / "lem_moe_v3.py"
LEM_MOE_V3_H0 = Path(__file__).resolve().parents[1] / "nn" / "embedding" / "lem_moe_v3_h0.py"


def _method_source(class_name: str, method_name: str) -> str:
    text = LEM_MOE_V3.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return "\n".join(lines[item.lineno - 1:item.end_lineno])

    raise AssertionError(f"{class_name}.{method_name} not found")


def test_init_layer_keeps_latents_active_only_until_public_scatter():
    source = _method_source("InitLayer", "forward")

    assert "edge_invariants = self.bessel(edge_length)" not in source
    assert "edge_length[active_edges]" in source
    assert "torch.zeros" not in source
    assert "torch.index_copy" not in source
    assert "latents[active_edges]" not in source


def test_update_layers_do_not_reslice_or_scatter_latents_by_active_edges():
    source = "\n".join(
        [
            _method_source("UpdateEdge", "forward"),
            _method_source("UpdateNode", "forward"),
        ]
    )

    assert "latents[active_edges]" not in source
    assert "torch.index_copy" not in source


def test_public_forwards_prepare_active_edge_metadata_once():
    source = _method_source("LemMoEV3", "forward")
    h0_source = LEM_MOE_V3_H0.read_text(encoding="utf-8")

    for text in (source, h0_source):
        assert "active_edge_center = edge_index[0][active_edges]" in text
        assert "active_edge_neighbor = edge_index[1][active_edges]" in text
        assert (
            "active_edge_vector = edge_vector[active_edges]" in text
            or "active_edge_vector = init_active_edge_vector" in text
        )
        assert "active_cutoff_coeffs = cutoff_coeffs[active_edges]" in text
        assert "_active_edge_tensor_to_full(latents, active_edges, edge_index.shape[1])" in text

    assert "edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY]" not in source
    assert "edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY]" not in h0_source
    assert "edge_sh = self.sh(init_active_edge_vector[:, [1, 2, 0]])" in source
    assert "edge_sh = self.sh(init_active_edge_vector[:, [1, 2, 0]])" in h0_source

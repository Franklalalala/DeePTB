from __future__ import annotations

import ast
from pathlib import Path


def test_emoles_uses_single_canonical_oeq_implementation():
    source = (
        Path(__file__).resolve().parents[1]
        / "nn"
        / "embedding"
        / "emoles.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    top_level_definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "get_feasible_tp" not in top_level_definitions
    assert "OEQTensorProduct" not in top_level_definitions

    imported = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "dptb.nn.embedding.oeq_tp":
            imported.update(alias.name for alias in node.names)
    assert imported.issuperset(
        {"get_feasible_tp", "OEQTensorProduct"}
    )

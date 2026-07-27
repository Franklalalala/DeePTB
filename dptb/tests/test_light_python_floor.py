"""No module may evaluate 3.10-only typing syntax at import time.

pyproject/README/Dockerfile all declare a Python 3.9 floor, but PEP 604 unions
(`X | Y`) in a *non-string* annotation are evaluated when the `def` executes.
dptb/nn/activation_recompute.py had 14 of them on the `dptb --help` import
path, so the CLI could not start on 3.9.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Names common enough in annotations that a `|` between two of them is a type
# union rather than an arithmetic/set/array OR.
_TYPE_ISH = {
    "None",
    "Any",
    "Optional",
    "Union",
    "Sequence",
    "Mapping",
    "Iterable",
    "Callable",
    "Tensor",
    "Path",
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "list",
    "dict",
    "tuple",
    "set",
    "type",
    "ndarray",
    "dtype",
    "device",
}


def _looks_like_a_type(node) -> bool:
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in _TYPE_ISH or node.id[:1].isupper()
    if isinstance(node, ast.Attribute):
        return node.attr in _TYPE_ISH or node.attr[:1].isupper()
    if isinstance(node, ast.Subscript):
        return _looks_like_a_type(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _looks_like_a_type(node.left) or _looks_like_a_type(node.right)
    return False


def _python_sources():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def test_no_runtime_evaluated_pep604_union_without_future_import():
    offenders = []
    for path in _python_sources():
        source = path.read_text(encoding="utf-8-sig", errors="replace")
        if "from __future__ import annotations" in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            pytest.fail(f"{path} does not parse")
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)):
                continue
            if _looks_like_a_type(node.left) and _looks_like_a_type(node.right):
                offenders.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} {ast.unparse(node)}"
                )

    assert not offenders, (
        "PEP 604 unions are evaluated at import time and break the declared "
        "Python 3.9 floor. Add `from __future__ import annotations` to:\n  "
        + "\n  ".join(offenders)
    )


def test_activation_recompute_declares_future_annotations():
    source = (PACKAGE_ROOT / "nn" / "activation_recompute.py").read_text(
        encoding="utf-8"
    )
    assert "from __future__ import annotations" in source

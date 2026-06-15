import ast
import logging
import sys
from pathlib import Path
from typing import Any


QHFLOW2_ESCN_PATH = (
    Path(__file__).resolve().parents[1] / "nn" / "embedding" / "qhflow2_escn.py"
)


def _load_import_helpers():
    source = QHFLOW2_ESCN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "_snapshot_root_logger_state",
        "_restore_root_logger_state",
        "_import_qhflow2_escn_backbone",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    found = {node.name for node in nodes}
    missing = wanted - found
    assert not missing, f"missing QHFlow2 import helper(s): {sorted(missing)}"

    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"__name__": "_qhflow2_escn_logging_test", "Any": Any}
    exec(compile(module, str(QHFLOW2_ESCN_PATH), "exec"), namespace)
    namespace["_ensure_qhflow2_src"] = lambda path=None: sys.path.insert(0, str(path))
    return namespace


def test_qhflow2_backbone_import_restores_root_logger_handlers(tmp_path):
    src = tmp_path / "qhflow2_src"
    module_dir = src / "models" / "modules"
    module_dir.mkdir(parents=True)
    (src / "models" / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "escn_backbone_v4.py").write_text(
        "\n".join(
            [
                "import logging",
                "root = logging.getLogger()",
                "for handler in root.handlers[:]:",
                "    root.removeHandler(handler)",
                "root.setLevel(logging.ERROR)",
                "root.propagate = False",
                "class eSCNMDBackbone_ham:",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    original_propagate = root.propagate
    sentinel = logging.NullHandler()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.addHandler(sentinel)
    root.setLevel(logging.WARNING)
    root.propagate = True
    for name in ["models.modules.escn_backbone_v4", "models.modules", "models"]:
        sys.modules.pop(name, None)

    namespace = _load_import_helpers()
    try:
        backbone = namespace["_import_qhflow2_escn_backbone"](str(src))

        assert backbone.__name__ == "eSCNMDBackbone_ham"
        assert root.handlers == [sentinel]
        assert root.level == logging.WARNING
        assert root.propagate is True
    finally:
        for name in ["models.modules.escn_backbone_v4", "models.modules", "models"]:
            sys.modules.pop(name, None)
        try:
            sys.path.remove(str(src))
        except ValueError:
            pass
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)
        root.propagate = original_propagate

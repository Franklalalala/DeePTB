import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _function_source(relpath: str, name: str) -> str:
    text = _read(relpath)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(text.splitlines()[node.lineno - 1: node.end_lineno])
    raise AssertionError(f"Could not find function {name} in {relpath}")


def test_trainer_dynamic_batch_state_is_not_blockwise_metric_patch_target():
    src = _function_source("nnops/trainer.py", "_dynamic_batch_state_from_batch")
    assert "loss_obj" not in src
    assert "last_block_loss" not in src


def test_multitrainer_flush_display_window_is_not_blockwise_metric_patch_target():
    src = _function_source("nnops/multi_trainer.py", "_flush_display_window")
    assert "loss_obj" not in src
    assert "last_block_loss" not in src


def test_argcheck_exposes_blockwise_loss_and_prediction_switch():
    src = _read("utils/argcheck.py")
    assert 'Argument("hamil_blockwise_nextham"' in src
    assert 'Argument("hamil_block_abs"' in src
    assert 'Argument("blockwise_hamiltonian"' in src


def test_nnenv_can_select_blockwise_hamiltonian_wrapper():
    src = _read("nn/deeptb.py")
    assert "BlockwiseE3Hamiltonian" in src
    assert 'prediction_copy.get("blockwise_hamiltonian"' in src


def test_single_train_component_monitors_update_each_iteration():
    src = _read("entrypoints/train.py")
    assert "TrainOnsiteLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')])" in src
    assert "TrainHoppingLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')])" in src
    assert "TrainOnsiteLossMonitor(interval=[(jdata[\"train_options\"][\"validation_freq\"], 'iteration')" not in src
    assert "TrainHoppingLossMonitor(interval=[(jdata[\"train_options\"][\"validation_freq\"], 'iteration')" not in src


def test_blockwise_train_metrics_are_registered_for_logging():
    single_src = _read("entrypoints/train.py")
    multi_src = _read("entrypoints/multi_train.py")
    for src in (single_src, multi_src):
        assert '"hamil_blockwise_nextham"' in src
        assert '"hamil_block_abs"' in src
        assert '"train_feature_compat_loss"' in src
        assert '"train_block_onsite_loss"' in src
        assert "ScalarFieldMonitor(stat_name=stat_name" in src

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
    assert 'Argument("hamil_blockwise"' in src
    assert 'Argument("hamil_block_abs"' in src
    assert 'Argument("blockwise_hamiltonian"' in src


def test_nnenv_can_select_blockwise_hamiltonian_wrapper():
    src = _read("nn/deeptb.py")
    assert "BlockwiseE3Hamiltonian" in src
    assert 'prediction_copy.get("blockwise_hamiltonian"' in src


def test_nnenv_can_select_direct_blockwise_hamiltonian_wrapper():
    src = _read("nn/deeptb.py")
    assert "DirectBlockwiseE3Hamiltonian" in src
    assert 'prediction_copy.get("direct_blockwise_hamiltonian"' in src


def test_nnenv_forwards_blockwise_hamiltonian_options():
    src = _read("nn/deeptb.py")
    assert "_blockwise_ham_kwargs" in src
    assert '"strict_complete_edges"' in src
    assert '"add_h0"' in src
    assert "**_blockwise_ham_kwargs" in src


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
        assert '"hamil_blockwise"' in src
        assert '"hamil_block_abs"' in src
        assert '"train_feature_compat_loss"' in src
        assert '"train_block_onsite_loss"' in src
        assert "ScalarFieldMonitor(stat_name=stat_name" in src


def test_argcheck_exposes_review_blockwise_options():
    src = _read("utils/argcheck.py")
    for token in (
        'Argument("complete_edges"',
        'Argument("strict_complete_edges"',
        'Argument("direct_blockwise_hamiltonian"',
        'Argument("add_h0"',
        'Argument("distributed_log_reduce"',
        'Argument("expose_component_sums"',
    ):
        assert token in src


def test_trainers_expose_raw_blockwise_component_stats():
    trainer_src = _read("nnops/trainer.py")
    multi_src = _read("nnops/multi_trainer.py")
    assert "last_component_stats" in trainer_src
    assert "last_component_stats" in multi_src


def test_lmdb_dataset_can_recover_h0_features_from_blockwise_tensors():
    src = _read("data/dataset/lmdb_dataset.py")
    assert "block_tensors_to_feature_tensors" in src
    assert "node_h0_blocks" in src
    assert "edge_h0_blocks" in src
    assert "uses_blockwise_targets" in src


def test_direct_blockwise_decoder_skips_feature_to_block_materializer():
    src = _read("nn/blockwise_hamiltonian.py")
    assert "class DirectAOBlockDecoder" in src
    assert "class DirectBlockwiseE3Hamiltonian" in src
    direct_forward = src.split("class DirectAOBlockDecoder", 1)[1].split("class DirectBlockwiseE3Hamiltonian", 1)[0]
    assert "feature_tensors_to_block_tensors" not in direct_forward


def test_structured_block_decoder_is_not_dense_learned_head():
    src = _read("nn/blockwise_hamiltonian.py")
    assert "class StructuredAOBlockDecoder" in src
    assert "class StructuredBlockwiseE3Hamiltonian" in src
    decoder_src = src.split("class StructuredAOBlockDecoder", 1)[1].split("class StructuredBlockwiseE3Hamiltonian", 1)[0]
    assert "nn.Linear" not in decoder_src
    assert "nn.Parameter" not in decoder_src
    assert "feature_tensors_to_block_tensors" not in decoder_src
    assert "_scatter_features_to_blocks" in decoder_src


def test_nnenv_can_select_structured_blockwise_hamiltonian_wrapper():
    src = _read("nn/deeptb.py")
    assert "StructuredBlockwiseE3Hamiltonian" in src
    assert 'prediction_copy.get("structured_blockwise_hamiltonian"' in src

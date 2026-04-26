import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_repo_text(rel_path):
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _method_source(text, name):
    marker = f"    def {name}("
    start = text.index(marker)
    next_method = text.find("\n    def ", start + len(marker))
    if next_method == -1:
        return text[start:]
    return text[start:next_method]


def _top_level_function_source(text, name):
    marker = f"def {name}("
    start = text.index(marker)
    next_function = text.find("\ndef ", start + len(marker))
    next_class = text.find("\nclass ", start + len(marker))
    candidates = [pos for pos in (next_function, next_class) if pos != -1]
    end = min(candidates) if candidates else len(text)
    return text[start:end]


def _assert_in_order(text, snippets):
    cursor = 0
    for snippet in snippets:
        pos = text.find(snippet, cursor)
        assert pos != -1, snippet
        cursor = pos + len(snippet)


def test_expert_parallel_layout_defaults_to_one_rank_per_expert():
    from dptb.nnops.expert_parallel_layout import (
        rank_to_expert_parallel,
        resolve_expert_parallel_layout,
    )

    layout = resolve_expert_parallel_layout(
        num_experts=2,
        world_size=2,
        train_options={},
    )

    assert layout.expert_data_parallel_size == 1
    assert layout.expected_world_size == 2

    rank0 = rank_to_expert_parallel(
        rank=0,
        num_experts=layout.num_experts,
        expert_data_parallel_size=layout.expert_data_parallel_size,
    )
    rank1 = rank_to_expert_parallel(
        rank=1,
        num_experts=layout.num_experts,
        expert_data_parallel_size=layout.expert_data_parallel_size,
    )

    assert rank0.local_expert_idx == 0
    assert rank0.expert_dp_rank == 0
    assert rank0.expert_group_ranks == [0]
    assert rank1.local_expert_idx == 1
    assert rank1.expert_dp_rank == 0
    assert rank1.expert_group_ranks == [1]


def test_expert_parallel_layout_maps_contiguous_replicas_to_same_expert():
    from dptb.nnops.expert_parallel_layout import (
        rank_to_expert_parallel,
        resolve_expert_parallel_layout,
    )

    layout = resolve_expert_parallel_layout(
        num_experts=2,
        world_size=4,
        train_options={"expert_data_parallel_size": 2},
    )

    assert layout.expert_data_parallel_size == 2
    assert layout.expected_world_size == 4

    rank0 = rank_to_expert_parallel(
        rank=0,
        num_experts=2,
        expert_data_parallel_size=2,
    )
    rank1 = rank_to_expert_parallel(
        rank=1,
        num_experts=2,
        expert_data_parallel_size=2,
    )
    rank2 = rank_to_expert_parallel(
        rank=2,
        num_experts=2,
        expert_data_parallel_size=2,
    )
    rank3 = rank_to_expert_parallel(
        rank=3,
        num_experts=2,
        expert_data_parallel_size=2,
    )

    assert rank0.local_expert_idx == 0
    assert rank0.expert_dp_rank == 0
    assert rank0.expert_group_ranks == [0, 1]
    assert rank1.local_expert_idx == 0
    assert rank1.expert_dp_rank == 1
    assert rank1.expert_group_ranks == [0, 1]
    assert rank2.local_expert_idx == 1
    assert rank2.expert_dp_rank == 0
    assert rank2.expert_group_ranks == [2, 3]
    assert rank3.local_expert_idx == 1
    assert rank3.expert_dp_rank == 1
    assert rank3.expert_group_ranks == [2, 3]


def test_expert_parallel_layout_rejects_world_size_mismatch():
    from dptb.nnops.expert_parallel_layout import resolve_expert_parallel_layout

    with pytest.raises(ValueError, match="world_size must equal num_experts \\* expert_data_parallel_size"):
        resolve_expert_parallel_layout(
            num_experts=2,
            world_size=3,
            train_options={"expert_data_parallel_size": 2},
        )


def test_expert_parallel_layout_accepts_short_expert_dp_alias():
    from dptb.nnops.expert_parallel_layout import resolve_expert_parallel_layout

    layout = resolve_expert_parallel_layout(
        num_experts=3,
        world_size=6,
        train_options={"expert_dp_size": 2},
    )

    assert layout.expert_data_parallel_size == 2
    assert layout.expected_world_size == 6


def test_expert_parallel_layout_rejects_conflicting_aliases():
    from dptb.nnops.expert_parallel_layout import resolve_expert_parallel_layout

    with pytest.raises(ValueError, match="conflicting expert data parallel settings"):
        resolve_expert_parallel_layout(
            num_experts=2,
            world_size=4,
            train_options={"expert_data_parallel_size": 2, "expert_dp_size": 3},
        )


def test_expert_dp_global_batch_semantics_divides_local_batch():
    text = _read_repo_text("dptb/nnops/multi_trainer.py")
    ns = {}
    exec(_top_level_function_source(text, "_resolve_local_expert_dp_batch_size"), ns)
    _resolve_local_expert_dp_batch_size = ns["_resolve_local_expert_dp_batch_size"]

    assert _resolve_local_expert_dp_batch_size(
        80,
        expert_data_parallel_size=2,
        semantics="global",
    ) == 40
    assert _resolve_local_expert_dp_batch_size(
        80,
        expert_data_parallel_size=2,
        semantics="local",
    ) == 80

    with pytest.raises(ValueError, match="must be divisible"):
        _resolve_local_expert_dp_batch_size(
            81,
            expert_data_parallel_size=2,
            semantics="global",
        )


def test_restart_merge_does_not_lock_expert_dp_topology():
    text = _read_repo_text("dptb/nnops/ddp_utils.py")
    start = text.index("RESTART_LOCKED_TRAIN_OPTION_KEYS = (")
    end = text.index("\n)", start)
    locked_keys = text[start:end]

    assert '"expert_data_parallel_size"' not in locked_keys
    assert '"expert_dp_size"' not in locked_keys
    assert '"distance_ranges"' in locked_keys


def test_multi_train_spawns_one_process_per_expert_replica():
    text = _read_repo_text("dptb/entrypoints/multi_train.py")

    assert "expert_dp_size = get_expert_data_parallel_size(train_opt)" in text
    assert "world_size = len(distance_ranges) * expert_dp_size" in text


def test_multi_trainer_uses_sharded_loaders_and_same_expert_grad_sync():
    text = _read_repo_text("dptb/nnops/multi_trainer.py")

    assert "DistributedSampler(" in text
    assert "num_replicas=self.expert_data_parallel_size" in text
    assert "rank=self.expert_dp_rank" in text
    assert "def _local_expert_dp_batch_size" in text
    assert "train_batch_size = self._local_expert_dp_batch_size(\"batch_size\")" in text
    assert "ref_batch_size = self._local_expert_dp_batch_size(\"ref_batch_size\")" in text
    assert "val_batch_size = self._local_expert_dp_batch_size(\"val_batch_size\")" in text
    assert "group=self.expert_dp_process_group" in text
    assert "grad_flags = torch.tensor(" in text
    assert "[1 if param.grad is not None else 0 for param in params]" in text
    assert "param.grad = torch.zeros_like(" in text
    assert 'self.expert_dp_grad_sync_mode = str(' in text
    assert 'self.expert_dp_grad_check_mode = str(' in text
    assert 'self.expert_dp_grad_bucket_mb = float(' in text
    assert 'self.expert_dp_batch_size_semantics = str(' in text
    assert "dist.all_reduce_coalesced(" in text
    assert "async_op=True" in text
    assert "flat = torch.cat([grad.contiguous().view(-1) for grad in bucket])" in text
    assert "grad_status = torch.tensor(" in text
    assert "missing_grads = grad_status[0]" in text
    assert "if int(missing_grads.item()) == 0:" in text
    assert "reduce_param_grads(params)" in text
    assert "scale = float(self.expert_data_parallel_size)" in text
    assert "grad.div_(scale)" in text
    assert "metric_tensor = self._mean_expert_dp_scalar(metric_tensor)" in text
    assert "metric = self._mean_expert_dp_scalar(metric)" in text
    assert 'self.sync_expert_dp_buffers = bool(self.train_options.get("sync_expert_dp_buffers", True))' in text
    assert "or not self.sync_expert_dp_buffers" in text
    assert "self._sync_local_expert_buffers(local_idx)" in text


def test_expert_dp_buffer_switch_only_guards_buffer_sync():
    text = _read_repo_text("dptb/nnops/multi_trainer.py")

    parameter_sync = _method_source(text, "_sync_local_expert_parameters")
    grad_sync = _method_source(text, "_sync_local_expert_grads")
    buffer_sync = _method_source(text, "_sync_local_expert_buffers")

    assert "sync_expert_dp_buffers" not in parameter_sync
    assert "sync_expert_dp_buffers" not in grad_sync
    assert "or not self.sync_expert_dp_buffers" in buffer_sync


def test_expert_dp_grad_sync_keeps_fast_path_and_missing_grad_fallback():
    text = _read_repo_text("dptb/nnops/multi_trainer.py")
    grad_sync = _method_source(text, "_sync_local_expert_grads")

    _assert_in_order(
        grad_sync,
        [
            "if self.expert_dp_grad_check_mode in",
            "assume_dense",
            "reduce_param_grads(params)",
            "return",
            "if self.expert_dp_grad_check_mode not in",
            "grad_status = torch.tensor(",
            "sum(1 for param in params if param.grad is None)",
            "sum(1 for param in params if param.grad is not None and param.grad.is_sparse)",
            "dist.all_reduce(",
            "grad_status,",
            "if int(grad_status[1].item()) != 0:",
            "raise RuntimeError(",
            "missing_grads = grad_status[0]",
            "if int(missing_grads.item()) == 0:",
            "reduce_param_grads(params)",
            "return",
            "grad_flags = torch.tensor(",
            "[1 if param.grad is not None else 0 for param in params]",
            "grad_flags,",
            "grad_flags.tolist()",
            "param.grad = torch.zeros_like(",
            "params_with_grads.append(param)",
            "reduce_param_grads(params_with_grads)",
        ],
    )


def test_expert_dp_grad_sync_keeps_bucketed_dense_and_sparse_branches():
    text = _read_repo_text("dptb/nnops/multi_trainer.py")
    grad_sync = _method_source(text, "_sync_local_expert_grads")

    _assert_in_order(
        grad_sync,
        [
            "bucket_cap_bytes = max(int(self.expert_dp_grad_bucket_mb * 1024 * 1024), 1)",
            "pending_reductions = []",
            "def enqueue_flat_bucket(bucket):",
            "if len(bucket) == 1:",
            "dist.all_reduce(",
            "async_op=True",
            "flat = torch.cat([grad.contiguous().view(-1) for grad in bucket])",
            "pending_reductions.append((work, bucket, flat))",
            "def enqueue_coalesced_bucket(bucket):",
            "dist.all_reduce_coalesced(",
            "async_op=True",
            "def finish_pending_reductions():",
            "scale = float(self.expert_data_parallel_size)",
            "flat.div_(scale)",
            "grad.copy_(flat[offset:offset + numel].view_as(grad))",
            "grad.div_(scale)",
            "def reduce_param_grads(reduce_params):",
            "grad_buckets = {}",
            "if grad.is_sparse:",
            "raise RuntimeError(",
            "grad_bytes = grad.numel() * grad.element_size()",
            "if grad_bytes >= bucket_cap_bytes:",
            "bucket_key = (grad.device, grad.dtype)",
            "if bucket and bucket_bytes + grad_bytes > bucket_cap_bytes:",
            "for bucket, _bucket_bytes in grad_buckets.values():",
        ],
    )
    assert "group=self.expert_dp_process_group" in grad_sync


def test_expert_dp_buffer_sync_uses_coalesced_float_buckets():
    text = _read_repo_text("dptb/nnops/multi_trainer.py")
    buffer_sync = _method_source(text, "_sync_local_expert_buffers")

    _assert_in_order(
        buffer_sync,
        [
            "bucket_cap_bytes = max(int(self.expert_dp_buffer_bucket_mb * 1024 * 1024), 1)",
            "def enqueue_float_buffer_bucket(bucket):",
            "dist.all_reduce_coalesced(",
            "async_op=True",
            "float_buckets = {}",
            "if buf.is_floating_point():",
            "if buf_bytes >= bucket_cap_bytes:",
            "dist.broadcast(",
            "for bucket, _bucket_bytes in float_buckets.values():",
            "enqueue_float_buffer_bucket(bucket)",
            "work.wait()",
            "buf.div_(scale)",
        ],
    )


def test_lmdb_dataset_reuses_worker_local_envs():
    text = _read_repo_text("dptb/data/dataset/lmdb_dataset.py")
    getstate = _method_source(text, "__getstate__")

    assert "self._lmdb_env_cache = {}" in text
    assert "self._lmdb_path_map += [lmdb_path] * txn.stat()['entries']" in text
    assert "def __getstate__(self):" in text
    assert "state[\"_lmdb_env_cache\"] = {}" in text
    assert "env.close()" not in getstate
    assert "def _get_lmdb_env(self, path: str):" in text
    assert "cache[path] = env" in text
    assert "def _load_data_dict(self, idx: int):" in text
    assert "with env.begin(buffers=True) as txn:" in text


def test_argcheck_exposes_expert_dp_sync_tuning_options():
    text = _read_repo_text("dptb/utils/argcheck.py")

    assert 'Argument("expert_dp_batch_size_semantics", str, optional=True, default="global"' in text
    assert 'Argument("expert_dp_grad_sync_mode", str, optional=True, default="coalesced"' in text
    assert 'Argument("expert_dp_grad_check_mode", str, optional=True, default="auto"' in text
    assert 'Argument("expert_dp_grad_bucket_mb", (int, float), optional=True, default=64' in text
    assert 'Argument("expert_dp_buffer_sync_mode", str, optional=True, default="coalesced"' in text
    assert 'Argument("expert_dp_buffer_bucket_mb", (int, float), optional=True, default=64' in text


def test_argcheck_exposes_expert_dp_ddp_backend_options():
    text = _read_repo_text("dptb/utils/argcheck.py")

    assert 'Argument("expert_dp_backend", str, optional=True, default="manual"' in text
    assert 'Argument("expert_dp_ddp_static_graph", bool, optional=True, default=False' in text
    assert 'Argument("expert_dp_ddp_gradient_as_bucket_view", bool, optional=True, default=False' in text
    assert 'Argument("expert_dp_ddp_find_unused_parameters", bool, optional=True, default=True' in text
    assert 'Argument("expert_dp_ddp_broadcast_buffers", bool, optional=True, default=False' in text
    assert 'Argument("expert_dp_ddp_bucket_cap_mb", (int, float), optional=True, default=None' in text


def test_argcheck_exposes_a_train_option_to_enable_expert_ddp_backend():
    text = _read_repo_text("dptb/utils/argcheck.py")

    has_backend_switch = 'Argument("expert_dp_backend", str, optional=True, default="manual"' in text
    has_bool_alias = 'Argument("expert_dp_use_ddp", bool, optional=True, default=False' in text

    assert has_backend_switch or has_bool_alias


def test_expert_dp_use_ddp_alias_is_declared_and_mapped_to_ddp_backend():
    argcheck_text = _read_repo_text("dptb/utils/argcheck.py")
    trainer_text = _read_repo_text("dptb/nnops/multi_trainer.py")

    assert 'Argument("expert_dp_use_ddp", bool, optional=True, default=False' in argcheck_text
    assert '"expert_dp_use_ddp"' in trainer_text
    assert 'self.expert_dp_backend = "ddp"' in trainer_text


def test_multi_trainer_wraps_local_expert_with_ddp_backend():
    text = _read_repo_text("dptb/nnops/multi_trainer.py")

    assert "from torch.nn.parallel import DistributedDataParallel" in text
    assert 'self.expert_dp_backend = str(self.train_options.get("expert_dp_backend", "manual")).lower()' in text
    assert "def _maybe_wrap_local_expert_ddp(self):" in text
    assert "DistributedDataParallel(" in text
    assert "process_group=self.expert_dp_process_group" in text
    assert "static_graph=self.expert_dp_ddp_static_graph" in text
    assert "gradient_as_bucket_view=self.expert_dp_ddp_gradient_as_bucket_view" in text
    assert "find_unused_parameters=self.expert_dp_ddp_find_unused_parameters" in text
    assert "broadcast_buffers=self.expert_dp_ddp_broadcast_buffers" in text
    assert "bucket_cap_mb=self.expert_dp_ddp_bucket_cap_mb" in text
    assert "self._maybe_wrap_local_expert_ddp()" in text


def test_expert_dp_ddp_backend_skips_manual_grad_sync():
    trainer_text = _read_repo_text("dptb/nnops/multi_trainer.py")
    grad_sync = _method_source(trainer_text, "_sync_local_expert_grads")

    _assert_in_order(
        grad_sync,
        [
            'if self.expert_dp_backend == "ddp":',
            "return",
            "if (",
            "not self._dist_ready()",
        ],
    )


def test_multi_trainer_expert_optimizer_uses_unwrapped_expert_parameters():
    trainer_text = _read_repo_text("dptb/nnops/multi_trainer.py")
    expert_parameters = _method_source(trainer_text, "_expert_parameters")

    assert "def _unwrap_expert_module" in trainer_text
    assert "return self._expert_module(expert_idx).parameters()" in expert_parameters
    assert "return get_optimizer(model_param=self._expert_parameters(i), **opt_cfg)" in trainer_text
    assert "opt = _make_opt_for_expert(idx)" in trainer_text


def test_saver_uses_unwrapped_local_expert_state_dict_for_ddp_wrapped_expert():
    saver_text = _read_repo_text("dptb/plugins/saver.py")
    gather_dist_states = _method_source(saver_text, "_gather_dist_states")

    assert (
        "self.trainer._unwrap_expert_module(self.trainer.model.experts[local_idx]).state_dict()"
        in gather_dist_states
        or "self.trainer._expert_module(local_idx).state_dict()" in gather_dist_states
    )


def test_saver_collapses_replica_states_to_one_state_per_expert():
    text = _read_repo_text("dptb/plugins/saver.py")

    assert "dp_size = int(getattr(self.trainer, \"expert_data_parallel_size\", 1))" in text
    assert "src_rank = expert_idx * dp_size" in text


def test_cueq_torch_fx_compat_shim_is_installed():
    text = _read_repo_text("dptb/nn/tensor_product_moe_v3.py")

    assert "def _ensure_torch_fx_symbolic_tracing_compat():" in text
    assert "symbolic_trace.is_fx_symbolic_tracing = symbolic_trace.is_fx_tracing" in text
    assert "_ensure_torch_fx_symbolic_tracing_compat()" in text

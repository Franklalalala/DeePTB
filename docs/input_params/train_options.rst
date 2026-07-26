Train Options
=============

Training, flow, distributed execution, monitoring, checkpoint, and loss controls. The strict schema in ``dptb.utils.argcheck.train_options`` is authoritative; this compact page avoids duplicating thousands of generated lines.

* ``num_epoch`` — Total number of training epochs. It is worth noted, if the model is reloaded with `-r` or `--restart` option, epoch which have been trained will counted from the time that the checkpoint is saved.

* ``distance_ranges``; default ``<_Flags.NONE: 0>`` — The ranges split for distance-based MoE / expert parallelism. Default: `[[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]]`

* ``parallel_multi``; default ``False`` — Set true to start parallel training on CUDA streams in single-process multi-expert mode. This option is automatically disabled when `use_ddp=True`.

* ``batch_size``; default ``1`` — The training batch size. In expert data parallel mode the default semantics are same-expert global batch, so the per-rank DataLoader batch is batch_size / expert_data_parallel_size. Default: `1`

* ``dynamic_batch``; default ``{'enabled': False}`` — Dynamic DeePTB block/edge batching. When enabled, batch_size remains the maximum number of samples per batch, while max_cost caps the total sample cost. The default mode is block, which uses raw Hamiltonian/H0 offsite block-key counts when

  Nested keys: ``enabled``, ``mode``, ``max_cost``, ``max_edge``, ``max_samples``, ``min_samples``, ``calibrate``, ``calibration_batches``, ``calibration_quantile``, ``bucket_size``, ``packing_strategy``, ``drop_last``, ``drop_oversized``, ``seed``, ``num_steps``, ``use_global_dist``, ``oom_fallback``, ``oom_shrink_factor``

* ``activation_recompute``; default ``{'enabled': False}`` — Train-time activation recomputation/checkpointing for memory hot paths. Supported targets are lem_moe_v3_tp and lem_non_linear_expert_block. The nonlinear target checkpoints gather/cat, full expert TP, expert activation, and 0e post-activat

  Nested keys: ``enabled``, ``targets``, ``checkpoint_node_tp``, ``checkpoint_edge_tp``, ``use_reentrant``, ``preserve_rng_state``

* ``ref_batch_size``; default ``1`` — The reference-data batch size. In expert data parallel mode the default semantics are local/per-rank so the common default value `1` remains valid when expert_data_parallel_size > 1. Default: `1`

* ``val_batch_size``; default ``1`` — The validation batch size. In expert data parallel mode the default semantics are local/per-rank so the common default value `1` remains valid when expert_data_parallel_size > 1. Default: `1`

* ``monitor_flag``; default ``False`` — Set true to start monitor.

* ``monitor_param_dynamics``; default ``False`` — Set true to enable lightweight parameter dynamics monitoring without forward/backward hooks. The monitor records sampled parameter update and gradient-flow metrics for key module groups.

* ``monitor_param_dynamics_freq``; default ``0`` — Parameter dynamics sampling interval in iterations. Use 0 to follow display_freq. Default: `0`.

* ``monitor_param_dynamics_tensorboard``; default ``None`` — Write parameter dynamics curves to TensorBoard when the monitor is enabled. Default follows use_tensorboard.

* ``monitor_param_dynamics_dead_patience``; default ``3`` — Number of consecutive no-gradient samples before marking a group as DEAD.

* ``monitor_param_dynamics_delta_eps``; default ``0.0`` — Absolute element-change threshold used for delta_nonzero_fraction.

* ``monitor_param_dynamics_grad_eps``; default ``0.0`` — Absolute gradient threshold used for grad_nonzero_fraction.

* ``monitor_param_dynamics_delta_norm_dead_threshold``; default ``1e-12`` — Deprecated compatibility option. DEAD detection is gradient-norm based; delta metrics are diagnostic only.

* ``monitor_param_dynamics_grad_norm_dead_threshold``; default ``1e-12`` — Gradient norm threshold used by parameter dynamics DEAD detection; groups below this value count as no-gradient.

* ``monitor_gated_edge_attention``; default ``False`` — Set true to record Fig.2-style diagnostics for gated edge aggregation: gate statistics, pre/post-gate sparsity, activation maxima, and top inbound-edge contribution share.

* ``monitor_gated_edge_attention_freq``; default ``0`` — Gated edge aggregation monitor sampling interval in iterations. Use 0 to follow display_freq. Default: `0`.

* ``monitor_gated_edge_attention_tensorboard``; default ``None`` — Write gated edge aggregation diagnostics to TensorBoard when the monitor is enabled. Default follows use_tensorboard.

* ``monitor_gated_edge_attention_heatmap``; default ``False`` — Set true to save Fig.2-like query-key heatmap PNG/NPZ snapshots for gated edge aggregation. Rows are target/query nodes, columns are source/key nodes, and colors are normalized edge-message contribution mass.

* ``monitor_gated_edge_attention_heatmap_size``; default ``64`` — Maximum number of query and key nodes shown in gated edge aggregation heatmaps. Default: `64`.

* ``clip_grad``; default ``1`` — Gradient clipping max norm.

* ``valid_fast``; default ``True`` — Set True to valid on the first batch of validation dataset, set False to valid the whole dataset. Default: `True`

* ``optimizer``; default ``{}`` — The optimizer setting for selecting the gradient optimizer of model training. Optimizer supported includes `Adam`, `AdamW`, `SGD` and `LBFGS` For more information about these optmization algorithm, we refer to: - `Adam`: [Adam: A Method for

  ``type`` choices: ``Adam``, ``AdamW``, ``HybridMuon``, ``SGD``, ``RMSprop``, ``LBFGS``

* ``lr_scheduler``; default ``{}`` — The learning rate scheduler tools settings, the lr scheduler is used to scales down the learning rate during the training process. Proper setting can make the training more stable and efficient. The supported lr schedular includes: `Exponen

  ``type`` choices: ``exp``, ``linear``, ``rop``, ``warmup_rop``, ``cos``, ``wsd``, ``cyclic``, ``qhflow_poly``

* ``update_lr_per_iter``; default ``False`` — Set true to update learning rate per-step. Default: `False`.

* ``sliding_win_size``; default ``50`` — Sliding window size for the average of the latest iterations' loss. Used for the reduce on plateau learning rate scheduler in case of the pairing of large dataset and small batch size. Default: `50`

* ``expert_lrs``; default ``[]`` — Optional per-expert initial learning rates. If provided, it must be a list of floats with length == num_experts (len(distance_ranges)). expert_lrs[i] will override optimizer.lr when building optimizer for expert i. Default: [] (disabled, us

* ``expert_optimizer_overrides``; default ``[]`` — Optional per-expert optimizer override dictionaries. If provided, it should be a list with length == num_experts (len(distance_ranges)); a single item is broadcast to all experts, and identical legacy entries collapse for a single expert. E

* ``expert_lr_scheduler_overrides``; default ``[]`` — Optional per-expert learning-rate scheduler override dictionaries. If provided, it should be a list with length == num_experts (len(distance_ranges)); a single item is broadcast to all experts, and identical legacy entries collapse for a si

* ``save_freq``; default ``10`` — Checkpoint save frequency in committed optimizer steps. Default: `10`.

* ``validation_freq``; default ``10`` — Frequency or every how many iteration to do model validation on validation datasets. Set 0 to disable iteration validation. Default: `10`

* ``validation_epoch_freq``; default ``1`` — Frequency or every how many epochs to do model validation on validation datasets. Set 0 to disable epoch validation. Default: `1`

* ``display_freq``; default ``1`` — Frequency, or every how many iteration to display the training log to screem. Default: `1`

* ``use_tensorboard``; default ``False`` — Set true to use tensorboard. It will record iteration error once every `25` iterations, epoch error once per epoch. There are tree types of error will be recorded. `train_loss_iter` is iteration loss, `train_loss_last` is the error of the l

* ``max_ckpt``; default ``4`` — The maximum number of saved checkpoints, Default: `4`

* ``max_epoch_ckpt``; default ``None`` — The maximum number of committed epoch checkpoints. When omitted, inherits max_ckpt.

* ``use_ddp``; default ``False`` — Set true to enable distributed expert-parallel training across multiple GPUs. When `distance_ranges` contains multiple experts, each rank will host one expert. Default: `False`

* ``ddp_backend``; default ``'nccl'`` — The backend used for distributed training. Usually `nccl` for GPUs and `gloo` for CPUs. Default: `nccl`

* ``ddp_master_addr``; default ``'127.0.0.1'`` — Master node address for distributed communication. Default: `127.0.0.1`

* ``ddp_master_port``; default ``29501`` — Master node port for distributed communication. Default: `29501`

* ``ddp_timeout_sec``; default ``1800`` — Timeout in seconds for distributed process group operations. Default: `1800`

* ``expert_data_parallel_size``; default ``1`` — Number of same-expert replicas in distributed expert-parallel training. With two `distance_ranges` and `expert_data_parallel_size=2`, ranks 0/1 train expert 0 and ranks 2/3 train expert 1, synchronizing gradients only inside each same-exper

* ``expert_dp_size``; default ``1`` — Number of same-expert replicas in distributed expert-parallel training. With two `distance_ranges` and `expert_data_parallel_size=2`, ranks 0/1 train expert 0 and ranks 2/3 train expert 1, synchronizing gradients only inside each same-exper

* ``train_num_workers``; default ``0`` — Number of DataLoader workers for train loader (implemented in MultiTrainer).

* ``ref_num_workers``; default ``0`` — Number of DataLoader workers for reference loader (implemented in MultiTrainer).

* ``val_num_workers``; default ``0`` — Number of DataLoader workers for validation loader (implemented in MultiTrainer).

* ``data_pin_memory``; default ``True`` — Enable pin_memory when rebuilding loaders in MultiTrainer.

* ``data_persistent_workers``; default ``True`` — Enable persistent_workers when rebuilding loaders in MultiTrainer.

* ``data_prefetch_factor``; default ``2`` — Prefetch factor when rebuilding loaders in MultiTrainer.

* ``distributed_rank0_prepare_batch``; default ``False`` — In distributed expert mode, only rank0 loads batch, performs CPU preprocessing + H2D + with_edge_vectors, then broadcasts packed GPU tensor groups to other ranks.

* ``precompute_lem_active_edges``; default ``True`` — Precompute LEM/MoE-v3 active edge indices on the CPU batch before moving tensors to GPU. This avoids the CUDA nonzero used by InitLayer when the cutoff configuration is fixed.

* ``precompute_lem_cutoff_coeffs``; default ``True`` — Also precompute LEM/MoE-v3 cutoff coefficients before model forward. Default true for fixed-geometry Hamiltonian training; set false for force/stress/virial or other geometry-gradient training.

* ``endpoint_loss_mode``; default ``'reduce'`` — Mode for reconstructing the canonical stitched endpoint loss. Use `reduce` for packed statistics or `full_forward` for a stitched validation forward. Default: `reduce`

* ``debug_tags``; default ``False`` — Set true to print stage-level timing logs for iteration, batch preparation, forward, backward, communication, scheduler and plugin stages. Useful for bottleneck diagnosis. Default: `False`

* ``debug_tag_freq``; default ``1`` — Print debug timing tags once every N iterations. Default: `1`

* ``debug_tag_cuda_mem``; default ``True`` — Set true to record CUDA allocated/reserved/peak memory in debug stage logs. Default: `True`

* ``debug_tag_cuda_sync``; default ``False`` — Set true to call `torch.cuda.synchronize()` before measuring each stage. This makes timing more accurate but will slow training, so use it only for debugging. Default: `False`

* ``debug_tag_reset_peak``; default ``None`` — Set true to reset CUDA peak counters at every debug tag boundary. When `monitor_cuda_memory=True`, the default is False so regular window-level peak memory remains valid. When `monitor_cuda_memory=False`, the default is True to preserve his

* ``debug_oom_dump``; default ``True`` — Set true to dump detailed CUDA memory summary on OOM. Default: `True`

* ``monitor_cuda_memory``; default ``True`` — Set true to record CUDA allocated/reserved and peak allocated/reserved memory in regular iteration/epoch logs and TensorBoard. In distributed expert mode, per-rank values are gathered as expert_i_cuda_*_mb fields and global cuda_*_mb fields

* ``monitor_cuda_cache_memory``; default ``None`` — Set true to log lightweight before/after CUDA memory deltas on persistent cache misses, including Wigner static tensors and cuEquivariance indexed_linear modules. This helps attribute stepwise memory jumps without enabling hook-heavy module

* ``monitor_cuda_cache_memory_sync``; default ``None`` — Set true to synchronize CUDA before cache-memory snapshots. More accurate but slower; default unset follows DPTB_CUDA_CACHE_MEMORY_SYNC.

* ``monitor_cuda_cache_memory_min_delta_mb``; default ``0.0`` — Only log cache-memory rows whose absolute allocated/reserved/peak/free delta is at least this many MiB. Default: `0`, log every probed cache miss.

* ``monitor_cuda_cache_events``; default ``None`` — Set true to log pure-Python persistent cache hit/miss events, including cuEq indexed_linear num_graphs keys. This does not query CUDA memory or synchronize. Default: unset, follows DPTB_CUDA_CACHE_EVENT_DIAG.

* ``monitor_cuda_cache_event_summary_interval``; default ``0`` — When cache event monitoring is enabled, log hit summaries every N events per cache key. Set 0 to log only misses. Default: `0`.

* ``monitor_cuda_module_memory``; default ``None`` — Set true to record CUDA memory snapshots around selected module forward/backward hooks. This is independent from monitor_flag, and currently targets SO2_Linear, MOLELinear, S2/FFN helpers, and non-TorchScript TensorProduct wrappers. Default

* ``monitor_cuda_module_memory_sync``; default ``False`` — Set true to synchronize CUDA before module-memory snapshots. More accurate but slower. Default: `False`

* ``monitor_cuda_module_memory_min_delta_mb``; default ``0.0`` — Only write module-memory rows whose allocated/reserved/current peak delta is at least this many MiB. Use a positive threshold for long production runs to avoid very large CSV files. Default: `0`

* ``sync_expert_dp_buffers``; default ``True`` — Set true to synchronize same-expert buffers after each expert data-parallel optimizer step. Disable only for throughput A/B when buffers are known not to affect training state. Default: `True`

* ``expert_dp_backend``; default ``'manual'`` — Same-expert data-parallel backend. `manual` uses DeePTB's explicit post-backward gradient sync; `ddp` wraps the local expert in torch.nn.parallel.DistributedDataParallel with the same-expert process group so gradient all-reduce can overlap

* ``expert_dp_use_ddp``; default ``False`` — Shortcut for setting expert_dp_backend to `ddp`. When true, expert_dp_backend is ignored. Default: `False`

* ``expert_dp_batch_size_semantics``; default ``'global'`` — Legacy default for training batch size interpretation when expert_data_parallel_size > 1. `global` means same-expert global batch and automatically divides local DataLoader batch by expert_data_parallel_size; `local` preserves per-rank sema

* ``expert_dp_train_batch_size_semantics``; default ``None`` — How batch_size is interpreted when expert_data_parallel_size > 1. Defaults to expert_dp_batch_size_semantics, normally `global`, to preserve fixed same-expert global batch.

* ``expert_dp_ref_batch_size_semantics``; default ``'local'`` — How ref_batch_size is interpreted when expert_data_parallel_size > 1. Default: `local`, so reference loaders keep their configured per-rank batch unless explicitly changed to `global`.

* ``expert_dp_val_batch_size_semantics``; default ``'local'`` — How val_batch_size is interpreted when expert_data_parallel_size > 1. Default: `local`, so validation loaders keep their configured per-rank batch unless explicitly changed to `global`.

* ``expert_dp_train_sampler_drop_last``; default ``False`` — Set true to make the corresponding same-expert DistributedSampler drop tail samples instead of padding duplicate indices. Default: `False` to preserve PyTorch DistributedSampler behavior.

* ``expert_dp_ref_sampler_drop_last``; default ``False`` — Set true to make the corresponding same-expert DistributedSampler drop tail samples instead of padding duplicate indices. Default: `False` to preserve PyTorch DistributedSampler behavior.

* ``expert_dp_val_sampler_drop_last``; default ``False`` — Set true to make the corresponding same-expert DistributedSampler drop tail samples instead of padding duplicate indices. Default: `False` to preserve PyTorch DistributedSampler behavior.

* ``expert_dp_ddp_static_graph``; default ``False`` — Set true when expert DDP graphs are static, enabling DDP static_graph optimization. Default: `False`

* ``expert_dp_ddp_gradient_as_bucket_view``; default ``False`` — Set true to let expert DDP gradients view all-reduce buckets and avoid extra bucket copies. Default: `False`

* ``expert_dp_ddp_find_unused_parameters``; default ``True`` — Set true if DDP-wrapped expert forward can leave trainable parameters unused. Default: `True`

* ``expert_dp_ddp_broadcast_buffers``; default ``False`` — Set true to let DDP broadcast expert buffers at forward start. Default: `False`; DeePTB keeps its post-step expert buffer sync path for manual parity. When true with `expert_dp_backend=ddp`, DDP owns buffer synchronization and DeePTB skips

* ``expert_dp_ddp_bucket_cap_mb``; default ``None`` — DDP bucket_cap_mb for same-expert DDP backend. Leave unset to use PyTorch's default bucket size.

* ``expert_dp_grad_sync_mode``; default ``'coalesced'`` — Same-expert data-parallel gradient synchronization implementation. `coalesced` uses torch.distributed.all_reduce_coalesced when available and falls back to flat buckets. `flat` always uses explicit flat buckets. Default: `coalesced`

* ``expert_dp_grad_check_mode``; default ``'auto'`` — Same-expert data-parallel missing-gradient check mode. `auto` performs a safe tiny collective before the dense bucket reductions; `assume_dense` skips that check for static dense expert graphs. Use `assume_dense` only for throughput A/B aft

* ``expert_dp_grad_bucket_mb``; default ``64`` — Target same-expert data-parallel gradient bucket size in MiB. Default: `64`

* ``expert_dp_buffer_sync_mode``; default ``'coalesced'`` — Same-expert data-parallel buffer synchronization implementation. `coalesced` uses coalesced float-buffer all-reduce when available. Default: `coalesced`

* ``expert_dp_buffer_bucket_mb``; default ``64`` — Target same-expert data-parallel float-buffer bucket size in MiB. Default: `64`

* ``debug_profile``; default ``False`` — Set true to enable PyTorch profiler for a selected iteration range and export Chrome trace json files. Useful for detailed CPU/CUDA/kernel timeline analysis. Default: `False`

* ``debug_profile_start_iter``; default ``5`` — The first iteration index to profile when `debug_profile=True`. Default: `5`

* ``debug_profile_end_iter``; default ``5`` — The last iteration index to profile when `debug_profile=True`. If equal to `debug_profile_start_iter`, only one iteration is profiled. Default: same as `debug_profile_start_iter`

* ``debug_profile_dir``; default ``''`` — Output directory for profiler Chrome trace json files. If not set, a default local profile directory will be used.

* ``ddp_debug_detail``; default ``False`` — Set true to enable `TORCH_DISTRIBUTED_DEBUG=DETAIL`, which prints more detailed distributed runtime diagnostics. Default: `False`

* ``nccl_debug``; default ``False`` — Set true to enable `NCCL_DEBUG`. Default: `False`

* ``nccl_debug_level``; default ``'INFO'`` — Debug level for NCCL when `nccl_debug=True`, e.g. `INFO` or `WARN`. Default: `INFO`

* ``cuda_launch_blocking``; default ``False`` — Set true to enable `CUDA_LAUNCH_BLOCKING=1` for easier debugging of asynchronous CUDA errors. This will significantly slow training and should NOT be used for performance benchmarking. Default: `False`

* ``nccl_async_error_handling``; default ``True`` — Set true to enable `NCCL_ASYNC_ERROR_HANDLING=1`. Recommended for distributed runs. Default: `True`

* ``cudnn_benchmark``; default ``False`` — Set true to enable `torch.backends.cudnn.benchmark`, which may improve performance when input shapes are stable. Default: `False`

* ``allow_tf32``; default ``True`` — Set true to allow TF32 on supported NVIDIA GPUs for faster matrix operations with possible tiny numerical differences. Default: `True`

* ``float32_matmul_precision``; default ``''`` — Precision policy for float32 matmul, passed to `torch.set_float32_matmul_precision`. Typical values are `highest`, `high`, `medium`. Empty string means keeping framework default.

* ``flow_options``; default ``{'enabled': False}`` — Trainer-side conditional flow matching for Hamiltonian prediction. When enabled, DeePTB replaces node_h0/edge_h0 by an interpolated Hamiltonian state H_t and trains the existing model to predict the clean target Hamiltonian, following a QHF

  Nested keys: ``enabled``, ``objective``, ``mode``, ``prior``, ``node_h0_key``, ``edge_h0_key``, ``node_target_key``, ``edge_target_key``, ``output_space``, ``state_space``, ``target_semantics``, ``block_input_adapter``, ``h0_condition_space``, ``block_export_final_full_h``, ``block_ode``, ``time_conditioning_required``, ``block_inverse_mode``, ``block_inverse_atol``, ``strict_certification``, ``node_output_key``, ``edge_output_key``, ``node_block_target_key``, ``edge_block_target_key``, ``node_block_shape_key``, ``edge_block_shape_key``, ``flow_time_key``, ``flow_time_r_key``, ``flow_time_t_key``, ``flow_time_h_key``, ``meanflow``, ``time_sampling``, ``t_min``, ``t_max``, ``t0_probability``, ``t_eps``, ``time_logit_mean``, ``time_logit_std``, ``node_sigma``, ``edge_sigma``, ``residual_sigma_floor``, ``te_prior_sigma``, ``te_prior_mode``, ``te_prior_per_graph``, ``te_prior_validation_seed``, ``tied_irrep_sigma``, ``tied_irrep_mode``, ``tied_irrep_irreps``, ``tied_irrep_validation_seed``, ``prior_node_key``, ``prior_edge_key``, ``prior_key_prefixes``, ``external_prior_strict``, ``allow_complex_prior_real_projection``, ``physical_prior_fallback``, ``basis_onsite_scale``, ``basis_onsite_missing_value``, ``basis_onsite_edge_value``, ``huckel_k``, ``huckel_node_overlap_key``, ``huckel_edge_overlap_key``, ``huckel_strict_overlap``, ``huckel_strict_basis``, ``huckel_edge_energy_fallback``, ``huckel_edge_length_decay``, ``huckel_energy_mode``, ``huckel_scale_mode``, ``huckel_scale_global``, ``huckel_edge_channel_scale``, ``prior_calibration``, ``basis_onsite_mode``, ``prior_node``, ``prior_edge``, ``haar_node_key``, ``haar_edge_key``, ``haar_candidate_index``, ``haar_dm_strict``, ``physical_prior_jitter_sigma``, ``physical_prior_jitter_reference_scale``, ``physical_prior_jitter_edge_decay``, ``loss_type``, ``node_weight``, ``edge_weight``, ``z_loss_coef``, ``endpoint_weight_power``, ``endpoint_weight_cap``, ``component_reduction``, ``validation_ode_steps``, ``apply_to_reference``, ``validation_flow_metrics``, ``overwrite_feature_keys``, ``detach_interpolated_h0``, ``missing_h0_policy``

* ``loss_options``

  Nested keys: ``train``, ``validation``, ``reference``, ``test``

* ``self_consistency``; default ``{'enabled': False}`` — WS4-C training-period self-consistency loss (see F:\claude\0702_nextham_dm_plan and dptb/nnops/self_consistency.py). Every `every_n_steps` steps, a `sample_frac` slice of the batch's predicted Hamiltonians is sent to an ABACUS restart_dh hr

  Nested keys: ``enabled``, ``endpoint``, ``sample_mode``, ``tensor_keys``, ``every_n_steps``, ``sample_frac``, ``weight``, ``warmup_epochs``, ``gap_threshold_ev``, ``staleness_steps``, ``consume_timeout``, ``max_workers``, ``retry_unfinished``, ``mode``, ``unlabeled_pool_weight``

* ``runtime``; default ``<_Flags.NONE: 0>`` — Optional grouping of runtime/performance and DataLoader knobs. Mirrors the identical flat train_options keys; normalized back to flat.

  Nested keys: ``cudnn_benchmark``, ``allow_tf32``, ``float32_matmul_precision``, ``precompute_lem_active_edges``, ``precompute_lem_cutoff_coeffs``, ``cuda_launch_blocking``, ``train_num_workers``, ``ref_num_workers``, ``val_num_workers``, ``data_pin_memory``, ``data_persistent_workers``, ``data_prefetch_factor``

* ``distributed``; default ``<_Flags.NONE: 0>`` — Optional grouping of DDP / expert-data-parallel and NCCL knobs. Mirrors the identical flat train_options keys; normalized back to flat.

  Nested keys: ``use_ddp``, ``ddp_backend``, ``ddp_master_addr``, ``ddp_master_port``, ``ddp_timeout_sec``, ``expert_data_parallel_size``, ``expert_dp_size``, ``parallel_multi``, ``distributed_rank0_prepare_batch``, ``sync_expert_dp_buffers``, ``expert_dp_backend``, ``expert_dp_use_ddp``, ``expert_dp_batch_size_semantics``, ``expert_dp_train_batch_size_semantics``, ``expert_dp_ref_batch_size_semantics``, ``expert_dp_val_batch_size_semantics``, ``expert_dp_train_sampler_drop_last``, ``expert_dp_ref_sampler_drop_last``, ``expert_dp_val_sampler_drop_last``, ``expert_dp_ddp_static_graph``, ``expert_dp_ddp_gradient_as_bucket_view``, ``expert_dp_ddp_find_unused_parameters``, ``expert_dp_ddp_broadcast_buffers``, ``expert_dp_ddp_bucket_cap_mb``, ``expert_dp_grad_sync_mode``, ``expert_dp_grad_check_mode``, ``expert_dp_grad_bucket_mb``, ``expert_dp_buffer_sync_mode``, ``expert_dp_buffer_bucket_mb``, ``ddp_debug_detail``, ``nccl_debug``, ``nccl_debug_level``, ``nccl_async_error_handling``

* ``checkpoint``; default ``<_Flags.NONE: 0>`` — Optional grouping of checkpoint save/retention knobs. Mirrors the identical flat train_options keys; normalized back to flat.

  Nested keys: ``save_freq``, ``max_ckpt``, ``max_epoch_ckpt``

* ``observers``; default ``<_Flags.NONE: 0>`` — Optional grouping of monitors, TensorBoard, debug tags and profiler knobs. Mirrors the identical flat train_options keys; normalized back to flat.

  Nested keys: ``monitor_flag``, ``monitor_param_dynamics``, ``monitor_param_dynamics_freq``, ``monitor_param_dynamics_tensorboard``, ``monitor_param_dynamics_dead_patience``, ``monitor_param_dynamics_delta_eps``, ``monitor_param_dynamics_grad_eps``, ``monitor_param_dynamics_delta_norm_dead_threshold``, ``monitor_param_dynamics_grad_norm_dead_threshold``, ``monitor_gated_edge_attention``, ``monitor_gated_edge_attention_freq``, ``monitor_gated_edge_attention_tensorboard``, ``monitor_gated_edge_attention_heatmap``, ``monitor_gated_edge_attention_heatmap_size``, ``use_tensorboard``, ``monitor_cuda_memory``, ``monitor_cuda_cache_memory``, ``monitor_cuda_cache_memory_sync``, ``monitor_cuda_cache_memory_min_delta_mb``, ``monitor_cuda_cache_events``, ``monitor_cuda_cache_event_summary_interval``, ``monitor_cuda_module_memory``, ``monitor_cuda_module_memory_sync``, ``monitor_cuda_module_memory_min_delta_mb``, ``debug_tags``, ``debug_tag_freq``, ``debug_tag_cuda_mem``, ``debug_tag_cuda_sync``, ``debug_tag_reset_peak``, ``debug_oom_dump``, ``debug_profile``, ``debug_profile_start_iter``, ``debug_profile_end_iter``, ``debug_profile_dir``

* ``physical_prior``; default ``<_Flags.NONE: 0>`` — Optional grouping of the physical-prior machinery (flow_options + self_consistency). Mirrors the identical flat train_options keys; normalized back to flat.

  Nested keys: ``flow_options``, ``self_consistency``

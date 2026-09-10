"""Experimental non-SOC LoopSCF with explicit numerical contracts.

Stable imports do not imply demonstrated downstream generalization.
"""

from .constants import (
    K_DEFAULT,
    CORRECTNESS_VERSION,
    WM_KEYS,
    ARM1_PATTERNS,
    ARM2_PATTERNS,
)
from .kspace import (
    fast_k_supported,
    _OrbpairPlan,
    _orbpair_scatter_plan,
    KBlockPlan,
    build_k_plan,
    bloch_phase,
    assemble_flat,
)
from .occupations import (
    factor_overlap_robust,
    _global_occupations,
    compute_mulliken_fast,
    _eval_occupation_kpoints,
)
from .adapters import (
    _irreps,
    _zeroe_slices,
    _gather_0e,
    _add_0e,
    _iter_embeddings,
    ZeroInitWM,
    _attach_adapters,
    _inject,
    _wrap_embedding,
    freeze_by_patterns,
)
from .loop import install_working_memory_true_diag
from .spectral import (
    _field,
    _batch_ptr,
    eigvals_from_factor,
    _fw10_one_graph,
    patch_fw10_per_graph,
)
from .training import (
    _format_loop_loss_log,
    stepwise_train_loss,
    patch_stepwise_loss,
    patch_smoke_exit,
)

__all__ = [
    "K_DEFAULT",
    "CORRECTNESS_VERSION",
    "WM_KEYS",
    "ARM1_PATTERNS",
    "ARM2_PATTERNS",
    "fast_k_supported",
    "_OrbpairPlan",
    "_orbpair_scatter_plan",
    "KBlockPlan",
    "build_k_plan",
    "bloch_phase",
    "assemble_flat",
    "factor_overlap_robust",
    "_global_occupations",
    "compute_mulliken_fast",
    "_eval_occupation_kpoints",
    "_irreps",
    "_zeroe_slices",
    "_gather_0e",
    "_add_0e",
    "_iter_embeddings",
    "ZeroInitWM",
    "_attach_adapters",
    "_inject",
    "_wrap_embedding",
    "freeze_by_patterns",
    "install_working_memory_true_diag",
    "_field",
    "_batch_ptr",
    "eigvals_from_factor",
    "_fw10_one_graph",
    "patch_fw10_per_graph",
    "_format_loop_loss_log",
    "stepwise_train_loss",
    "patch_stepwise_loss",
    "patch_smoke_exit",
]

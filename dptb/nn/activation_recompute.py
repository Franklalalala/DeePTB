# PEP 604 unions below are evaluated at def time without this; the declared
# floor is Python 3.9 and this module is on the `dptb --help` import path.
from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch
from torch.utils.checkpoint import checkpoint

from dptb.nn.tensor_product_moe_v3 import MOLEGlobals


log = logging.getLogger(__name__)

_FLAG_NAMES = (
    "_activation_recompute_enabled",
    "_activation_recompute_use_reentrant",
    "_activation_recompute_preserve_rng_state",
)
_DEFAULT_TARGETS = ("lem_moe_v3_tp", "lem_non_linear_expert_block")


def _split_tensor_args(args: tuple[Any, ...]):
    specs = []
    tensor_args = []
    for arg in args:
        if torch.is_tensor(arg):
            specs.append(("tensor", len(tensor_args)))
            tensor_args.append(arg)
        else:
            specs.append(("value", arg))
    return specs, tensor_args


def _restore_args(specs, tensor_args: tuple[torch.Tensor, ...]):
    restored = []
    for kind, value in specs:
        if kind == "tensor":
            restored.append(tensor_args[value])
        else:
            restored.append(value)
    return restored


def _should_checkpoint(enabled: bool, tensor_args) -> bool:
    return (
        enabled
        and torch.is_grad_enabled()
        and any(torch.is_tensor(arg) and arg.requires_grad for arg in tensor_args)
    )


def checkpoint_function_call(
        fn: Callable[..., Any],
        *args: Any,
        enabled: bool,
        use_reentrant: bool = False,
        preserve_rng_state: bool = False,
) -> Any:
    """Run a callable through non-reentrant activation recomputation.

    Non-tensor arguments are captured in the closure. This keeps cached SO2
    state, MOLEGlobals, and optional Wigner objects compatible with checkpoint.
    """
    if use_reentrant:
        raise ValueError(
            "activation_recompute.use_reentrant=True is not supported for "
            "LEM MoE TP. Use use_reentrant=False."
        )

    if not _should_checkpoint(enabled, args):
        return fn(*args)

    specs, tensor_args = _split_tensor_args(tuple(args))

    def _run(*flat_tensor_args):
        restored_args = _restore_args(specs, flat_tensor_args)
        return fn(*restored_args)

    return checkpoint(
        _run,
        *tensor_args,
        use_reentrant=False,
        preserve_rng_state=preserve_rng_state,
    )


def checkpoint_module_call(
        module: torch.nn.Module,
        *args: Any,
        enabled: bool,
        use_reentrant: bool = False,
        preserve_rng_state: bool = False,
) -> Any:
    if not module.training:
        return module(*args)
    return checkpoint_function_call(
        module,
        *args,
        enabled=enabled,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
    )


def checkpoint_so2_linear_call(
        module: torch.nn.Module,
        x: torch.Tensor,
        edge_vector: torch.Tensor,
        mole_globals: MOLEGlobals | None,
        latents: torch.Tensor | None = None,
        wigner_D_all: Any = None,
        *,
        enabled: bool,
        use_reentrant: bool = False,
        preserve_rng_state: bool = False,
) -> Any:
    """Checkpoint an MoLE SO2_Linear call while preserving router gradients."""
    if (
        not enabled
        or not module.training
        or not torch.is_grad_enabled()
        or not any(
            torch.is_tensor(arg) and arg.requires_grad
            for arg in (x, edge_vector, latents, wigner_D_all, getattr(mole_globals, "coefficients", None))
        )
    ):
        return module(x, edge_vector, mole_globals, latents, wigner_D_all)

    coefficients = getattr(mole_globals, "coefficients", None)
    sizes = getattr(mole_globals, "sizes", None)
    split_sizes = getattr(mole_globals, "split_sizes", None)
    graph_index = getattr(mole_globals, "graph_index", None)

    def _run(x_arg, edge_vector_arg, coefficients_arg, sizes_arg, graph_index_arg, latents_arg, wigner_D_all_arg):
        recompute_globals = clone_mole_globals_for_recompute(
            mole_globals,
            coefficients=coefficients_arg,
            sizes=sizes_arg,
            split_sizes=split_sizes,
            graph_index=graph_index_arg,
        )
        return module(
            x_arg,
            edge_vector_arg,
            recompute_globals,
            latents_arg,
            wigner_D_all_arg,
        )

    return checkpoint_function_call(
        _run,
        x,
        edge_vector,
        coefficients,
        sizes,
        graph_index,
        latents,
        wigner_D_all,
        enabled=True,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
    )


def clone_mole_globals_for_recompute(
        src: MOLEGlobals | None,
        *,
        coefficients: torch.Tensor | None,
        sizes: torch.Tensor | None,
        split_sizes: Any,
        graph_index: torch.Tensor | None,
) -> MOLEGlobals | None:
    if src is None:
        return None
    out = MOLEGlobals(
        coefficients=coefficients,
        sizes=sizes,
        split_sizes=split_sizes,
        graph_index=None if split_sizes is not None else graph_index,
    )
    if graph_index is not None:
        if _validate_recompute_graph_index() and not _graph_index_matches_split_sizes(
            graph_index, split_sizes
        ):
            raise ValueError(
                "MOLE graph_index is inconsistent with split_sizes during "
                "activation recompute."
            )
        out.graph_index = graph_index
    return out


def _validate_recompute_graph_index() -> bool:
    return os.environ.get(
        "DPTB_ACTIVATION_RECOMPUTE_VALIDATE_GRAPH_INDEX", "0"
    ) not in ("", "0", "false", "False")


def _graph_index_matches_split_sizes(graph_index: torch.Tensor, split_sizes: Any) -> bool:
    split_tuple = MOLEGlobals._normalize_split_sizes(None, split_sizes)
    if split_tuple is None:
        return False
    graph_index_flat = graph_index.detach().reshape(-1).to(dtype=torch.long)
    if graph_index_flat.numel() != sum(split_tuple):
        return False
    expected = torch.tensor(split_tuple, dtype=torch.long, device=graph_index_flat.device)
    expected_graph_index = torch.repeat_interleave(
        torch.arange(len(split_tuple), dtype=torch.long, device=graph_index_flat.device),
        expected,
    )
    return bool(torch.equal(graph_index_flat, expected_graph_index))


def _so2_linear_from_parts(
        module: torch.nn.Module,
        node_in: torch.Tensor,
        edge_in: torch.Tensor,
        edge_center: torch.Tensor,
        edge_neighbor: torch.Tensor | None,
        active_edges: torch.Tensor,
        edge_vector: torch.Tensor,
        mole_globals: MOLEGlobals | None,
        latents: torch.Tensor | None,
        wigner_D_all: Any,
) -> Any:
    edge_node_in = node_in[edge_center[active_edges]]
    parts = [edge_node_in, edge_in]
    if edge_neighbor is not None:
        parts.append(node_in[edge_neighbor[active_edges]])
    x = torch.cat(parts, dim=-1)
    active_edge_vector = edge_vector[active_edges]
    active_latents = latents[active_edges] if latents is not None else None
    return module(x, active_edge_vector, mole_globals, active_latents, wigner_D_all)


def checkpoint_so2_linear_from_parts(
        module: torch.nn.Module,
        node_in: torch.Tensor,
        edge_in: torch.Tensor,
        edge_center: torch.Tensor,
        active_edges: torch.Tensor,
        edge_vector: torch.Tensor,
        mole_globals: MOLEGlobals | None,
        latents: torch.Tensor | None = None,
        wigner_D_all: Any = None,
        *,
        edge_neighbor: torch.Tensor | None = None,
        enabled: bool,
        use_reentrant: bool = False,
        preserve_rng_state: bool = False,
) -> Any:
    """Checkpoint index/gather + cat + SO2_Linear as one recompute region."""
    if (
        not enabled
        or not module.training
        or not torch.is_grad_enabled()
        or not any(
            torch.is_tensor(arg) and arg.requires_grad
            for arg in (node_in, edge_in, edge_vector, latents, wigner_D_all, getattr(mole_globals, "coefficients", None))
        )
    ):
        return _so2_linear_from_parts(
            module,
            node_in,
            edge_in,
            edge_center,
            edge_neighbor,
            active_edges,
            edge_vector,
            mole_globals,
            latents,
            wigner_D_all,
        )

    coefficients = getattr(mole_globals, "coefficients", None)
    sizes = getattr(mole_globals, "sizes", None)
    split_sizes = getattr(mole_globals, "split_sizes", None)
    graph_index = getattr(mole_globals, "graph_index", None)

    def _run(
            node_in_arg,
            edge_in_arg,
            edge_center_arg,
            edge_neighbor_arg,
            active_edges_arg,
            edge_vector_arg,
            coefficients_arg,
            sizes_arg,
            graph_index_arg,
            latents_arg,
            wigner_D_all_arg,
    ):
        recompute_globals = clone_mole_globals_for_recompute(
            mole_globals,
            coefficients=coefficients_arg,
            sizes=sizes_arg,
            split_sizes=split_sizes,
            graph_index=graph_index_arg,
        )
        return _so2_linear_from_parts(
            module,
            node_in_arg,
            edge_in_arg,
            edge_center_arg,
            edge_neighbor_arg,
            active_edges_arg,
            edge_vector_arg,
            recompute_globals,
            latents_arg,
            wigner_D_all_arg,
        )

    return checkpoint_function_call(
        _run,
        node_in,
        edge_in,
        edge_center,
        edge_neighbor,
        active_edges,
        edge_vector,
        coefficients,
        sizes,
        graph_index,
        latents,
        wigner_D_all,
        enabled=True,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
    )


def _clear_activation_recompute_flags(model: torch.nn.Module) -> None:
    for module in model.modules():
        for name in _FLAG_NAMES:
            if hasattr(module, name):
                delattr(module, name)


def _set_activation_recompute_flags(
        module: torch.nn.Module,
        *,
        use_reentrant: bool,
        preserve_rng_state: bool,
) -> None:
    module._activation_recompute_enabled = True
    module._activation_recompute_use_reentrant = use_reentrant
    module._activation_recompute_preserve_rng_state = preserve_rng_state


def configure_activation_recompute(model: torch.nn.Module, options: dict[str, Any] | None) -> dict[str, int]:
    """Apply train-time activation recomputation flags without wrapping modules."""
    _clear_activation_recompute_flags(model)

    if not isinstance(options, dict) or not bool(options.get("enabled", False)):
        return {"enabled": 0, "node_tp": 0, "edge_tp": 0, "non_linear_node": 0, "non_linear_edge": 0}

    target_opt = options.get("targets", list(_DEFAULT_TARGETS))
    targets = {target_opt} if isinstance(target_opt, str) else set(target_opt)
    supported_targets = set(_DEFAULT_TARGETS)
    if not targets.intersection(supported_targets):
        log.warning("activation_recompute enabled, but no supported target is selected: %s", sorted(targets))
        return {"enabled": 1, "node_tp": 0, "edge_tp": 0, "non_linear_node": 0, "non_linear_edge": 0}

    checkpoint_node_tp = bool(options.get("checkpoint_node_tp", True))
    checkpoint_edge_tp = bool(options.get("checkpoint_edge_tp", True))
    use_reentrant = bool(options.get("use_reentrant", False))
    if use_reentrant:
        raise ValueError(
            "activation_recompute.use_reentrant=True is not supported for "
            "LEM MoE TP. Use use_reentrant=False."
        )
    preserve_rng_state = bool(options.get("preserve_rng_state", False))

    state = {"enabled": 1, "node_tp": 0, "edge_tp": 0, "non_linear_node": 0, "non_linear_edge": 0}
    for module in model.modules():
        class_name = module.__class__.__name__
        if "lem_moe_v3_tp" in targets:
            if class_name == "UpdateNode" and checkpoint_node_tp and hasattr(module, "tp"):
                _set_activation_recompute_flags(
                    module,
                    use_reentrant=use_reentrant,
                    preserve_rng_state=preserve_rng_state,
                )
                state["node_tp"] += 1
            elif class_name == "UpdateEdge" and checkpoint_edge_tp and hasattr(module, "tp"):
                _set_activation_recompute_flags(
                    module,
                    use_reentrant=use_reentrant,
                    preserve_rng_state=preserve_rng_state,
                )
                state["edge_tp"] += 1
        if "lem_non_linear_expert_block" in targets:
            if class_name == "NonLinearExpertUpdateNode" and checkpoint_node_tp and hasattr(module, "expert_tp"):
                _set_activation_recompute_flags(
                    module,
                    use_reentrant=use_reentrant,
                    preserve_rng_state=preserve_rng_state,
                )
                state["non_linear_node"] += 1
            elif class_name == "NonLinearExpertUpdateEdge" and checkpoint_edge_tp and hasattr(module, "expert_tp"):
                _set_activation_recompute_flags(
                    module,
                    use_reentrant=use_reentrant,
                    preserve_rng_state=preserve_rng_state,
                )
                state["non_linear_edge"] += 1

    log.info(
        "activation_recompute enabled: targets=%s node_tp=%s edge_tp=%s "
        "non_linear_node=%s non_linear_edge=%s use_reentrant=%s preserve_rng_state=%s",
        sorted(targets),
        state["node_tp"],
        state["edge_tp"],
        state["non_linear_node"],
        state["non_linear_edge"],
        use_reentrant,
        preserve_rng_state,
    )
    return state

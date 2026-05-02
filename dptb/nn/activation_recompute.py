import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from dptb.nn.tensor_product_moe_v3 import MOLEGlobals


log = logging.getLogger(__name__)


def _split_tensor_args(args: Tuple[Any, ...]) -> Tuple[List[Any], List[torch.Tensor]]:
    specs: List[Any] = []
    tensor_args: List[torch.Tensor] = []
    for arg in args:
        if torch.is_tensor(arg):
            specs.append(("tensor", len(tensor_args)))
            tensor_args.append(arg)
        else:
            specs.append(("value", arg))
    return specs, tensor_args


def _restore_args(specs: List[Any], tensor_args: Tuple[torch.Tensor, ...]) -> List[Any]:
    restored: List[Any] = []
    for kind, value in specs:
        if kind == "tensor":
            restored.append(tensor_args[value])
        else:
            restored.append(value)
    return restored


def checkpoint_module_call(
    module: torch.nn.Module,
    *args: Any,
    enabled: bool,
    use_reentrant: bool = False,
    preserve_rng_state: bool = False,
) -> Any:
    """Run a module call through activation recomputation when it is useful.

    Non-tensor arguments are captured in the closure instead of being passed to
    checkpoint directly. This keeps optional state such as cached Wigner blocks
    compatible with both PyTorch checkpoint implementations.
    """
    if use_reentrant:
        raise ValueError(
            "activation_recompute.use_reentrant=True is not supported for "
            "LEM MoE TP. Use use_reentrant=False."
        )

    if (
        not enabled
        or not module.training
        or not torch.is_grad_enabled()
        or not any(torch.is_tensor(arg) and arg.requires_grad for arg in args)
    ):
        return module(*args)

    specs, tensor_args = _split_tensor_args(tuple(args))

    def _run(*flat_tensor_args):
        restored_args = _restore_args(specs, flat_tensor_args)
        return module(*restored_args)

    return checkpoint(
        _run,
        *tensor_args,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
    )


def checkpoint_so2_linear_call(
    module: torch.nn.Module,
    x: torch.Tensor,
    edge_vector: torch.Tensor,
    mole_globals: Optional[MOLEGlobals],
    latents: Optional[torch.Tensor] = None,
    wigner_D_all: Optional[torch.Tensor] = None,
    *,
    enabled: bool,
    use_reentrant: bool = False,
    preserve_rng_state: bool = False,
) -> Any:
    """Checkpoint an MoLE SO2_Linear call while preserving router gradients."""
    if use_reentrant:
        raise ValueError(
            "activation_recompute.use_reentrant=True is not supported for "
            "LEM MoE TP. Use use_reentrant=False."
        )

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

    args = (x, edge_vector, coefficients, sizes, graph_index, latents, wigner_D_all)
    specs, tensor_args = _split_tensor_args(args)

    def _run(*flat_tensor_args):
        (
            x_arg,
            edge_vector_arg,
            coefficients_arg,
            sizes_arg,
            graph_index_arg,
            latents_arg,
            wigner_D_all_arg,
        ) = _restore_args(specs, flat_tensor_args)
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

    return checkpoint(
        _run,
        *tensor_args,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
    )


def clone_mole_globals_for_recompute(
    src: Optional[MOLEGlobals],
    *,
    coefficients: Optional[torch.Tensor],
    sizes: Optional[torch.Tensor],
    split_sizes: Any,
    graph_index: Optional[torch.Tensor],
) -> Optional[MOLEGlobals]:
    """Rebuild MOLEGlobals for checkpoint recompute without dropping graph_index."""
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
    edge_neighbor: Optional[torch.Tensor],
    active_edges: torch.Tensor,
    edge_vector: torch.Tensor,
    mole_globals: Optional[MOLEGlobals],
    latents: Optional[torch.Tensor],
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
    mole_globals: Optional[MOLEGlobals],
    latents: Optional[torch.Tensor] = None,
    wigner_D_all: Any = None,
    *,
    edge_neighbor: Optional[torch.Tensor] = None,
    enabled: bool,
    use_reentrant: bool = False,
    preserve_rng_state: bool = False,
) -> Any:
    """Checkpoint index/gather + cat + SO2_Linear as one recompute region."""
    if use_reentrant:
        raise ValueError(
            "activation_recompute.use_reentrant=True is not supported for "
            "LEM MoE TP. Use use_reentrant=False."
        )

    tensor_candidates = (
        node_in,
        edge_in,
        edge_vector,
        latents,
        wigner_D_all,
        getattr(mole_globals, "coefficients", None),
    )
    if (
        not enabled
        or not module.training
        or not torch.is_grad_enabled()
        or not any(torch.is_tensor(arg) and arg.requires_grad for arg in tensor_candidates)
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

    args = (
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
    )
    specs, tensor_args = _split_tensor_args(args)

    def _run(*flat_tensor_args):
        (
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
        ) = _restore_args(specs, flat_tensor_args)
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

    return checkpoint(
        _run,
        *tensor_args,
        use_reentrant=False,
        preserve_rng_state=preserve_rng_state,
    )


def _clear_activation_recompute_flags(model: torch.nn.Module) -> None:
    names = (
        "_activation_recompute_enabled",
        "_activation_recompute_use_reentrant",
        "_activation_recompute_preserve_rng_state",
    )
    for module in model.modules():
        for name in names:
            if hasattr(module, name):
                delattr(module, name)


def configure_activation_recompute(model: torch.nn.Module, options: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Apply train-time activation recomputation flags without wrapping modules."""
    _clear_activation_recompute_flags(model)

    if not isinstance(options, dict) or not bool(options.get("enabled", False)):
        return {"enabled": 0, "node_tp": 0, "edge_tp": 0}

    target_opt = options.get("targets", ["lem_moe_v3_tp"])
    targets = {target_opt} if isinstance(target_opt, str) else set(target_opt)
    if "lem_moe_v3_tp" not in targets:
        log.warning("activation_recompute enabled, but no supported target is selected: %s", sorted(targets))
        return {"enabled": 1, "node_tp": 0, "edge_tp": 0}

    checkpoint_node_tp = bool(options.get("checkpoint_node_tp", True))
    checkpoint_edge_tp = bool(options.get("checkpoint_edge_tp", True))
    use_reentrant = bool(options.get("use_reentrant", False))
    if use_reentrant:
        raise ValueError(
            "activation_recompute.use_reentrant=True is not supported for "
            "LEM MoE TP. Use use_reentrant=False."
        )
    preserve_rng_state = bool(options.get("preserve_rng_state", False))

    node_tp = 0
    edge_tp = 0
    for module in model.modules():
        class_name = module.__class__.__name__
        if class_name == "UpdateNode" and checkpoint_node_tp and hasattr(module, "tp"):
            module._activation_recompute_enabled = True
            module._activation_recompute_use_reentrant = use_reentrant
            module._activation_recompute_preserve_rng_state = preserve_rng_state
            node_tp += 1
        elif class_name == "UpdateEdge" and checkpoint_edge_tp and hasattr(module, "tp"):
            module._activation_recompute_enabled = True
            module._activation_recompute_use_reentrant = use_reentrant
            module._activation_recompute_preserve_rng_state = preserve_rng_state
            edge_tp += 1

    log.info(
        "activation_recompute enabled: target=lem_moe_v3_tp node_tp=%s edge_tp=%s "
        "use_reentrant=%s preserve_rng_state=%s",
        node_tp,
        edge_tp,
        use_reentrant,
        preserve_rng_state,
    )
    return {"enabled": 1, "node_tp": node_tp, "edge_tp": edge_tp}

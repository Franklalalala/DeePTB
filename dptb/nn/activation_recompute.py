import logging
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
    preserve_rng_state: bool = True,
) -> Any:
    """Run a module call through activation recomputation when it is useful.

    Non-tensor arguments are captured in the closure instead of being passed to
    checkpoint directly. This keeps optional state such as cached Wigner blocks
    compatible with both PyTorch checkpoint implementations.
    """
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
    preserve_rng_state: bool = True,
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
        recompute_globals = MOLEGlobals(
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


def configure_activation_recompute(model: torch.nn.Module, options: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Apply train-time activation recomputation flags without wrapping modules."""
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
    preserve_rng_state = bool(options.get("preserve_rng_state", True))

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

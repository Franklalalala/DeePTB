from __future__ import annotations

"""Small tensor diagnostics for Hamiltonian flow geometry.

These helpers deliberately stay model-agnostic.  They are meant for smoke
probes and validation notebooks that need scalar geometry checks without
threading another objective through the trainer.
"""

from typing import Dict, Iterable, Optional, Sequence

import torch


def _as_real_flat(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        x = torch.view_as_real(x)
    return x.reshape(-1)


def _flatten_real_tensors(tensors: Iterable[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    parts = [_as_real_flat(x) for x in tensors if torch.is_tensor(x)]
    if not parts:
        return None
    return torch.cat(parts)


def cosine_similarity_tensors(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Return cosine similarity after flattening real and complex tensors."""

    a = _as_real_flat(first)
    b = _as_real_flat(second).to(device=a.device, dtype=a.dtype)
    if a.numel() != b.numel():
        raise ValueError(f"cosine inputs must have the same element count, got {a.numel()} and {b.numel()}")
    denom = a.norm() * b.norm()
    if float(denom.detach().cpu()) <= eps:
        return a.new_full((), float("nan"))
    return torch.dot(a, b) / denom.clamp_min(eps)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return _as_real_flat(x).norm()


def norm_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    denom = _norm(denominator)
    if float(denom.detach().cpu()) <= eps:
        return denom.new_full((), float("nan"))
    return _norm(numerator) / denom.clamp_min(eps)


def grad_cosine(
    first_loss: torch.Tensor,
    second_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Cosine between two loss gradients over a shared parameter list."""

    params = [p for p in parameters if p.requires_grad]
    if not params:
        return first_loss.new_full((), float("nan"))
    first_grads = torch.autograd.grad(
        first_loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    second_grads = torch.autograd.grad(
        second_loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    first = _flatten_real_tensors(first_grads)
    second = _flatten_real_tensors(second_grads)
    if first is None or second is None:
        return first_loss.new_full((), float("nan"))
    return cosine_similarity_tensors(first, second, eps=eps)


def pixel_meanflow_du_dt_diagnostics(
    *,
    target_v: torch.Tensor,
    du_dt: torch.Tensor,
    flow_loss: Optional[torch.Tensor] = None,
    jvp_loss: Optional[torch.Tensor] = None,
    parameters: Optional[Sequence[torch.nn.Parameter]] = None,
    eps: float = 1.0e-12,
) -> Dict[str, torch.Tensor]:
    """Report pMF du/dt scale and optional flow/JVP gradient alignment."""

    state = {
        "du_dt_norm": _norm(du_dt).detach(),
        "target_v_norm": _norm(target_v).detach(),
        "du_dt_norm_over_target_v_norm": norm_ratio(du_dt, target_v, eps=eps).detach(),
    }
    if flow_loss is not None and jvp_loss is not None and parameters is not None:
        state["grad_cos_flow_jvp"] = grad_cosine(
            flow_loss,
            jvp_loss,
            parameters,
            eps=eps,
        ).detach()
    return state


def endpoint_velocity(
    *,
    current: torch.Tensor,
    endpoint: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    denom = (1.0 - t.to(device=current.device, dtype=current.dtype)).clamp_min(eps)
    while denom.ndim < current.ndim:
        denom = denom.unsqueeze(-1)
    return (endpoint - current) / denom


def _component_chord_velocity(
    *,
    current: Optional[torch.Tensor],
    target: Optional[torch.Tensor],
    endpoint: Optional[torch.Tensor],
    t: torch.Tensor,
    eps: float,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    if current is None or target is None or endpoint is None:
        return None
    v_theta = endpoint_velocity(current=current, endpoint=endpoint, t=t, eps=eps)
    chord = endpoint_velocity(current=current, endpoint=target, t=t, eps=eps)
    return v_theta, chord


def cfm_chord_cosine_diagnostics(
    *,
    t: torch.Tensor,
    node_t: Optional[torch.Tensor] = None,
    node_current: Optional[torch.Tensor] = None,
    node_target: Optional[torch.Tensor] = None,
    node_endpoint: Optional[torch.Tensor] = None,
    edge_t: Optional[torch.Tensor] = None,
    edge_current: Optional[torch.Tensor] = None,
    edge_target: Optional[torch.Tensor] = None,
    edge_endpoint: Optional[torch.Tensor] = None,
    eps: float = 1.0e-6,
) -> Dict[str, torch.Tensor]:
    """Compare endpoint-parameterized CFM velocity to the target chord."""

    state: Dict[str, torch.Tensor] = {}
    v_parts = []
    chord_parts = []
    node = _component_chord_velocity(
        current=node_current,
        target=node_target,
        endpoint=node_endpoint,
        t=t if node_t is None else node_t,
        eps=eps,
    )
    if node is not None:
        v_node, chord_node = node
        state["node_cos_v_theta_chord"] = cosine_similarity_tensors(
            v_node, chord_node, eps=eps
        ).detach()
        v_parts.append(v_node)
        chord_parts.append(chord_node)
    edge = _component_chord_velocity(
        current=edge_current,
        target=edge_target,
        endpoint=edge_endpoint,
        t=t if edge_t is None else edge_t,
        eps=eps,
    )
    if edge is not None:
        v_edge, chord_edge = edge
        state["edge_cos_v_theta_chord"] = cosine_similarity_tensors(
            v_edge, chord_edge, eps=eps
        ).detach()
        v_parts.append(v_edge)
        chord_parts.append(chord_edge)

    if not v_parts:
        ref = t if torch.is_tensor(t) else torch.tensor(0.0)
        state["cos_v_theta_chord"] = ref.new_full((), float("nan"))
        return state

    v_all = _flatten_real_tensors(v_parts)
    chord_all = _flatten_real_tensors(chord_parts)
    state["cos_v_theta_chord"] = cosine_similarity_tensors(v_all, chord_all, eps=eps).detach()
    return state

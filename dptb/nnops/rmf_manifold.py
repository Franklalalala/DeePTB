from __future__ import annotations

"""Torch-native manifold primitives for Hamiltonian RMF.

The RMF training helper only needs a small subset of manifold operations:
projection of ambient vectors to tangent spaces, exponential/logarithmic maps,
geodesic interpolation, and the path velocity.  This module intentionally keeps
the API independent of the official JAX RMF implementation so the default CFM
training path never acquires a JAX dependency.

The Euclidean manifold is the production-safe fallback.  It is also the exact
geometry of the current SOC ``uu_real`` residual target space, where the model
learns a residual correction relative to H0 rather than a full Hamiltonian.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping

import torch


def _time_view(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-item time tensor to ``like`` without copying data."""
    t = torch.as_tensor(t, device=like.device, dtype=like.dtype)
    if t.ndim == 0:
        return t.reshape((1,) + (1,) * (like.ndim - 1))
    return t.reshape((-1,) + (1,) * (like.ndim - 1))


class RiemannianManifold(ABC):
    """Minimal embedded-manifold interface used by Hamiltonian RMF.

    ``project(x, v)`` follows the Riemannian MeanFlow convention: it projects an
    ambient vector ``v`` to the tangent space at point ``x``.  The remaining
    methods are written in terms of points on the manifold and tangent vectors.
    """

    name: str = "abstract"

    @abstractmethod
    def project(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project ambient vector ``v`` to the tangent space at ``x``."""

    @abstractmethod
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Return ``exp_x(u)``."""

    @abstractmethod
    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``log_x(y)`` in the tangent space at ``x``."""

    def geodesic_interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate from ``x0`` to ``x1`` by geodesic fraction ``t``."""
        return self.expmap(x0, _time_view(t, x0) * self.logmap(x0, x1))

    @abstractmethod
    def tangent_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity of the interpolation path at time ``t``."""


class EuclideanManifold(RiemannianManifold):
    """Flat residual Hamiltonian manifold.

    This fallback has no external dependencies and preserves the existing CFM
    residual semantics: H_t = H0 + residual_t, and the model still predicts the
    real-H residual endpoint through the existing DeePTB output heads.
    """

    name = "euclidean"

    def project(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        del x
        return v

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - x

    def geodesic_interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_view = _time_view(t, x0)
        return (1.0 - t_view) * x0 + t_view * x1

    def tangent_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        del t
        return x1 - x0


def _manifold_name(config: Any) -> str:
    if config is None:
        return "euclidean"
    if isinstance(config, str):
        return config.lower()
    if isinstance(config, Mapping):
        return str(config.get("type", config.get("name", "euclidean"))).lower()
    raise TypeError(f"Unsupported RMF manifold config type: {type(config)!r}")


def build_rmf_manifold(config: Any = None) -> RiemannianManifold:
    """Build a Torch-native RMF manifold from a config value."""
    name = _manifold_name(config)
    aliases: Dict[str, str] = {
        "r": "euclidean",
        "r^n": "euclidean",
        "rn": "euclidean",
        "flat": "euclidean",
        "identity": "euclidean",
        "euclidean": "euclidean",
    }
    canonical = aliases.get(name, name)
    if canonical == "euclidean":
        return EuclideanManifold()
    raise ValueError(
        f"Unsupported RMF manifold {name!r}. The clean patch currently ships "
        "only the Torch Euclidean fallback; add curved manifolds behind this "
        "factory without changing default CFM behavior."
    )

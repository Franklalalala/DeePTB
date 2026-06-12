from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from dptb.data import _keys


def sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_positions: int = 2000,
) -> torch.Tensor:
    """QHFlow2-compatible sinusoidal embedding for normalized flow time."""
    timesteps = timesteps.reshape(-1)
    if embedding_dim < 2:
        return timesteps[:, None].expand(-1, embedding_dim)
    scaled = timesteps * max_positions
    half_dim = embedding_dim // 2
    exponent = math.log(max_positions) / max(half_dim - 1, 1)
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -exponent
    )
    angles = scaled.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding.to(dtype=timesteps.dtype)


class FlowTimeConditioner(torch.nn.Module):
    """Inject one time embedding per graph into node scalar channels."""

    def __init__(
        self,
        scalar_channels: int,
        flow_time_key: str = "flow_time",
        flow_time_keys: Optional[Sequence[str]] = None,
        max_positions: int = 2000,
        missing_time_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.scalar_channels = int(scalar_channels)
        self.flow_time_key = str(flow_time_key)
        self.flow_time_keys = tuple(str(key) for key in (flow_time_keys or ()))
        self.max_positions = int(max_positions)
        self.missing_time_value = float(missing_time_value)
        num_time_keys = max(len(self.flow_time_keys), 1)
        self.register_buffer(
            "_flow_time_key_weights",
            torch.arange(1, num_time_keys + 1, dtype=torch.float32),
            persistent=False,
        )

    def _graph_time(
        self,
        data: Dict[str, Any],
        key: str,
        *,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        graph_t = data.get(key, None)
        if graph_t is None:
            return torch.full(
                (num_graphs,),
                self.missing_time_value,
                device=device,
                dtype=dtype,
            )
        graph_t = torch.as_tensor(graph_t, device=device, dtype=dtype).reshape(-1)
        if graph_t.numel() == 1 and num_graphs == 1:
            return graph_t
        if graph_t.numel() != num_graphs:
            raise ValueError(
                f"`{key}` must contain one value per graph "
                f"({num_graphs}), got {graph_t.numel()}."
            )
        return graph_t

    def forward(self, node_features: torch.Tensor, data: Dict[str, Any]) -> torch.Tensor:
        batch = data.get(_keys.BATCH_KEY, None)
        if batch is None:
            batch = torch.zeros(node_features.shape[0], device=node_features.device, dtype=torch.long)
        else:
            batch = batch[: node_features.shape[0]].to(device=node_features.device, dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        time_keys = self.flow_time_keys or (self.flow_time_key,)
        graph_embedding = None
        for idx, key in enumerate(time_keys):
            graph_t = self._graph_time(
                data,
                key,
                num_graphs=num_graphs,
                device=node_features.device,
                dtype=node_features.dtype,
            )
            key_embedding = sinusoidal_time_embedding(
                graph_t,
                embedding_dim=self.scalar_channels,
                max_positions=self.max_positions,
            )
            if self.flow_time_keys:
                key_embedding = key_embedding * self._flow_time_key_weights[idx].to(
                    device=key_embedding.device,
                    dtype=key_embedding.dtype,
                )
            graph_embedding = key_embedding if graph_embedding is None else graph_embedding + key_embedding
        node_embedding = graph_embedding.index_select(0, batch)
        conditioned = node_features.clone()
        conditioned[:, : self.scalar_channels] = (
            conditioned[:, : self.scalar_channels] + node_embedding
        )
        return conditioned

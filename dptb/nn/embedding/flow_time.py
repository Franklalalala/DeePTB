from __future__ import annotations

import math
from typing import Any, Dict

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
        max_positions: int = 2000,
        missing_time_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.scalar_channels = int(scalar_channels)
        self.flow_time_key = str(flow_time_key)
        self.max_positions = int(max_positions)
        self.missing_time_value = float(missing_time_value)

    def forward(self, node_features: torch.Tensor, data: Dict[str, Any]) -> torch.Tensor:
        batch = data.get(_keys.BATCH_KEY, None)
        if batch is None:
            batch = torch.zeros(node_features.shape[0], device=node_features.device, dtype=torch.long)
        else:
            batch = batch[: node_features.shape[0]].to(device=node_features.device, dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        graph_t = data.get(self.flow_time_key, None)
        if graph_t is None:
            graph_t = torch.full(
                (num_graphs,),
                self.missing_time_value,
                device=node_features.device,
                dtype=node_features.dtype,
            )
        else:
            graph_t = torch.as_tensor(graph_t, device=node_features.device, dtype=node_features.dtype).reshape(-1)
            if graph_t.numel() == 1 and num_graphs == 1:
                pass
            elif graph_t.numel() != num_graphs:
                raise ValueError(
                    f"`{self.flow_time_key}` must contain one value per graph "
                    f"({num_graphs}), got {graph_t.numel()}."
                )
        graph_embedding = sinusoidal_time_embedding(
            graph_t,
            embedding_dim=self.scalar_channels,
            max_positions=self.max_positions,
        )
        node_embedding = graph_embedding.index_select(0, batch)
        conditioned = node_features.clone()
        conditioned[:, : self.scalar_channels] = (
            conditioned[:, : self.scalar_channels] + node_embedding
        )
        return conditioned

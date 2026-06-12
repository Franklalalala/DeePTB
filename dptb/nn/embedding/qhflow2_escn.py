from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import ase.data
from e3nn import o3
from torch import nn
from torch_scatter import scatter_mean

from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_batch, with_edge_vectors
from dptb.nn.embedding.emb import Embedding


def _default_qhflow2_src() -> str:
    here = Path(__file__).resolve()
    candidates = [
        # Repo-local vendor path for ordinary clones:
        #   DeePTB/vendor/QHFlow2/src
        here.parents[3] / "vendor" / "QHFlow2" / "src",
        # Current natlan experiment layout:
        #   qhflow2_aligned_dptb_20260609/vendor/QHFlow2/src
        here.parents[4] / "vendor" / "QHFlow2" / "src",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _ensure_qhflow2_src(path: str | None = None) -> None:
    src = path or os.environ.get("QHFLOW2_SRC") or _default_qhflow2_src()
    expected = Path(src) / "models" / "modules" / "escn_backbone_v4.py"
    if not expected.is_file():
        raise FileNotFoundError(
            "QHFlow2 source was not found. Set `qhflow2_src` or QHFLOW2_SRC to "
            f"the QHFlow2 `src` directory; expected {expected}."
        )
    if src and src not in sys.path:
        sys.path.insert(0, src)


class _QHFlowData(dict):
    def __init__(self, *args: Any, num_graphs: int, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._num_graphs = int(num_graphs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_num_graphs":
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __len__(self) -> int:
        return self._num_graphs


@Embedding.register("qhflow2_escn")
class QHFlow2ESCNEmbedding(nn.Module):
    """DPTB embedding wrapper around QHFlow2's eSCN Hamiltonian backbone.

    This is a diagnostic bridge: DPTB still owns LMDB loading, CFM target
    construction, and e3tb feature order, while QHFlow2 supplies the message
    passing backbone. The current DPTB node/edge features are pooled into a
    graph-level Hamiltonian context for the QHFlow2 matrix input path. The
    node/edge heads are dense adapters supervised in DPTB RME order; they do
    not by themselves prove an architecture-level m-order/equivariant contract.
    """

    def __init__(
        self,
        idp,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device("cpu"),
        hidden_size: int = 128,
        sh_lmax: int = 4,
        num_gnn_layers: int = 5,
        num_ham_gnn_layers: int = 2,
        esen_max_radius: float = 5.0,
        matrix_l: int = 6,
        context_hidden: int = 256,
        head_hidden: int = 256,
        ham_context_mode: str = "features",
        h0_node_key: str = _keys.NODE_H0_KEY,
        h0_edge_key: str = _keys.EDGE_H0_KEY,
        fallback_node_key: str = _keys.NODE_FEATURES_KEY,
        fallback_edge_key: str = _keys.EDGE_FEATURES_KEY,
        strict_ham_context_h0: bool = True,
        use_flow_time_embedding: bool = True,
        flow_time_key: str = "flow_time",
        allow_missing_flow_time: bool = False,
        qhflow2_src: str | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        _ensure_qhflow2_src(qhflow2_src)
        from models.modules.escn_backbone_v4 import eSCNMDBackbone_ham

        self.idp = idp
        self.dtype = dtype
        self.device = torch.device(device)
        self.hidden_size = int(hidden_size)
        self.sh_lmax = int(sh_lmax)
        self.matrix_l = int(matrix_l)
        self.ham_context_mode = str(ham_context_mode)
        if self.ham_context_mode not in {"features", "zero"}:
            raise ValueError(
                "QHFlow2ESCNEmbedding ham_context_mode must be 'features' or 'zero', "
                f"got {self.ham_context_mode!r}"
            )
        self.h0_node_key = str(h0_node_key)
        self.h0_edge_key = str(h0_edge_key)
        self.fallback_node_key = str(fallback_node_key)
        self.fallback_edge_key = str(fallback_edge_key)
        self.strict_ham_context_h0 = bool(strict_ham_context_h0)
        self.use_flow_time_embedding = bool(use_flow_time_embedding)
        if not self.use_flow_time_embedding:
            raise ValueError(
                "QHFlow2ESCNEmbedding requires use_flow_time_embedding=True; "
                "the vendored eSCN Hamiltonian backbone unconditionally consumes its time message."
            )
        self.flow_time_key = flow_time_key
        self.allow_missing_flow_time = bool(allow_missing_flow_time)

        self.idp.get_irreps(no_parity=False)
        self.idp.get_orbpair_maps()
        self.rme_dim = int(self.idp.orbpair_irreps.dim)
        self.latent_dim = self.hidden_size
        self._out_irreps = self.idp.orbpair_irreps
        type_to_z = [ase.data.atomic_numbers[sym] for sym in self.idp.type_names]
        self.register_buffer(
            "_type_to_z",
            torch.tensor(type_to_z, dtype=torch.long),
            persistent=False,
        )

        self.context_proj = nn.Sequential(
            nn.Linear(2 * self.rme_dim, context_hidden),
            nn.SiLU(),
            nn.Linear(context_hidden, self.matrix_l * self.hidden_size),
        )
        self.backbone = eSCNMDBackbone_ham(
            lmax=self.sh_lmax,
            mmax=self.sh_lmax,
            sphere_channels=self.hidden_size,
            hidden_channels=self.hidden_size,
            num_layers=int(num_gnn_layers),
            use_block_S=False,
            use_block_H=False,
            use_dataset_embedding=False,
            use_time_embedding=self.use_flow_time_embedding,
            num_ham_gnn_layers=int(num_ham_gnn_layers),
            cutoff=float(esen_max_radius),
            otf_graph=False,
        )
        sph_feature_size = int((self.sh_lmax + 1) ** 2)
        flat_dim = sph_feature_size * self.hidden_size
        self.node_head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, self.rme_dim),
        )
        self.edge_head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, self.rme_dim),
        )
        self.to(device=self.device, dtype=self.dtype)

    @property
    def out_node_dim(self) -> int:
        return self.rme_dim

    @property
    def out_edge_dim(self) -> int:
        return self.rme_dim

    @property
    def out_node_irreps(self):
        return self._out_irreps

    @property
    def out_edge_irreps(self):
        return self._out_irreps

    def _num_graphs(self, batch: torch.Tensor) -> int:
        if batch.numel() == 0:
            return 1
        return int(batch.max().item()) + 1

    def _flow_time(self, data: AtomicDataDict.Type, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        value = data.get(self.flow_time_key, None)
        if value is None:
            if self.allow_missing_flow_time:
                return torch.zeros(num_graphs, device=batch.device, dtype=self.dtype)
            raise KeyError(
                f"QHFlow2ESCNEmbedding requires one `{self.flow_time_key}` value per graph. "
                "Set allow_missing_flow_time=true only for an explicit t=0 evaluation."
            )
        value = torch.as_tensor(value, device=batch.device, dtype=self.dtype).reshape(-1)
        if value.numel() == 1:
            return value.expand(num_graphs)
        if value.numel() == num_graphs:
            return value
        if value.numel() == batch.numel():
            return scatter_mean(value, batch, dim=0, dim_size=num_graphs)
        raise ValueError(
            f"`{self.flow_time_key}` must be scalar, per-graph ({num_graphs}), or "
            f"per-node ({batch.numel()}); got {value.numel()} values."
        )

    def _context_value(
        self,
        data: AtomicDataDict.Type,
        h0_key: str,
        feature_key: str,
        *,
        label: str,
    ) -> torch.Tensor | None:
        value = data.get(h0_key, None)
        if value is None:
            if self.strict_ham_context_h0:
                raise KeyError(
                    "QHFlow2ESCNEmbedding ham_context_mode='features' requires "
                    f"`{h0_key}` for the {label} context. Disable "
                    "strict_ham_context_h0 only for an explicit target-fed ablation, "
                    "or use ham_context_mode='zero'."
                )
            value = data.get(feature_key, None)
        return value

    def _masked_graph_mean(
        self,
        value: torch.Tensor,
        graph_index: torch.Tensor,
        num_graphs: int,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        value = value.to(dtype=self.dtype)
        graph_index = graph_index.to(device=value.device, dtype=torch.long)
        if mask is not None:
            mask = torch.as_tensor(mask, device=value.device, dtype=torch.bool).reshape(-1)
            if mask.numel() != value.shape[0]:
                raise ValueError(
                    f"Expert mask length {mask.numel()} does not match context rows {value.shape[0]}."
                )
            value = value[mask]
            graph_index = graph_index[mask]
        if value.shape[0] == 0:
            return torch.zeros(
                num_graphs,
                self.rme_dim,
                device=graph_index.device,
                dtype=self.dtype,
            )
        return scatter_mean(value, graph_index, dim=0, dim_size=num_graphs)

    def _ham_context(
        self,
        data: AtomicDataDict.Type,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if self.ham_context_mode == "zero":
            return torch.zeros(
                num_graphs,
                self.matrix_l,
                self.hidden_size,
                device=batch.device,
                dtype=self.dtype,
            )

        node = self._context_value(
            data,
            self.h0_node_key,
            self.fallback_node_key,
            label="node",
        )
        if node is None:
            node = torch.zeros(batch.numel(), self.rme_dim, device=batch.device, dtype=self.dtype)
        node_ctx = self._masked_graph_mean(
            node,
            batch,
            num_graphs,
            data.get("expert_node_mask", None),
        )

        edge = self._context_value(
            data,
            self.h0_edge_key,
            self.fallback_edge_key,
            label="edge",
        )
        if edge is None:
            edge_ctx = torch.zeros(num_graphs, self.rme_dim, device=batch.device, dtype=self.dtype)
        elif edge.shape[0] == 0:
            edge_ctx = torch.zeros(num_graphs, self.rme_dim, device=batch.device, dtype=self.dtype)
        else:
            edge_batch = batch[edge_index[0]]
            edge_ctx = self._masked_graph_mean(
                edge,
                edge_batch,
                num_graphs,
                data.get("expert_edge_mask", None),
            )

        context = torch.cat([node_ctx, edge_ctx], dim=-1)
        return self.context_proj(context).reshape(num_graphs, self.matrix_l, self.hidden_size)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_batch(data)
        data = with_edge_vectors(data, with_lengths=True)
        pos = data[_keys.POSITIONS_KEY].to(dtype=self.dtype)
        batch = data[_keys.BATCH_KEY].long()
        edge_index = data[_keys.EDGE_INDEX_KEY].long()
        atomic_numbers_value = data.get(_keys.ATOMIC_NUMBERS_KEY, None)
        if atomic_numbers_value is None:
            atom_type = data[_keys.ATOM_TYPE_KEY].long().reshape(-1)
            atomic_numbers_value = self._type_to_z.to(atom_type.device)[atom_type]
        atomic_numbers = atomic_numbers_value.long().reshape(-1, 1)
        num_graphs = self._num_graphs(batch)

        # DPTB stores dst-src vectors; QHFlow2's non-OTF graph path expects src-dst.
        # Negating the PBC-aware DPTB vector preserves the QHFlow2 convention.
        edge_vec = -data[_keys.EDGE_VECTORS_KEY].to(dtype=self.dtype)
        edge_len = data[_keys.EDGE_LENGTH_KEY].to(dtype=self.dtype)
        ham_context = self._ham_context(data, batch, edge_index, num_graphs)

        qh_data = _QHFlowData(
            {
                "pos": pos,
                "batch": batch,
                "atomic_numbers": atomic_numbers,
                "edge_index": edge_index,
                "edge_distance": edge_len,
                "edge_distance_vec": edge_vec,
                "t": self._flow_time(data, batch, num_graphs),
            },
            num_graphs=num_graphs,
        )
        emb = self.backbone(qh_data, [None, ham_context, None, None])
        node_embedding = emb["node_embedding"].reshape(pos.shape[0], -1)
        edge_embedding = emb["xy_embedding"].reshape(edge_index.shape[1], -1)

        data[_keys.NODE_FEATURES_KEY] = self.node_head(node_embedding)
        data[_keys.EDGE_FEATURES_KEY] = self.edge_head(edge_embedding)
        return data

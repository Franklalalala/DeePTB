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
from dptb.data.AtomicDataDict import with_batch
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
    if src and src not in sys.path:
        sys.path.insert(0, src)


def _snapshot_root_logger_state() -> dict[str, Any]:
    import logging

    root = logging.getLogger()
    return {
        "handlers": list(root.handlers),
        "level": root.level,
        "propagate": root.propagate,
        "disabled": root.disabled,
        "filters": list(root.filters),
    }


def _restore_root_logger_state(state: dict[str, Any]) -> None:
    import logging

    root = logging.getLogger()
    original_handlers = list(state["handlers"])
    for handler in list(root.handlers):
        root.removeHandler(handler)
        if handler not in original_handlers:
            handler.close()
    for handler in original_handlers:
        root.addHandler(handler)

    root.setLevel(state["level"])
    root.propagate = state["propagate"]
    root.disabled = state["disabled"]
    root.filters[:] = list(state["filters"])


def _import_qhflow2_escn_backbone(qhflow2_src: str | None = None):
    state = _snapshot_root_logger_state()
    try:
        _ensure_qhflow2_src(qhflow2_src)
        from models.modules.escn_backbone_v4 import eSCNMDBackbone_ham

        return eSCNMDBackbone_ham
    finally:
        _restore_root_logger_state(state)


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
    graph-level Hamiltonian context for the QHFlow2 matrix input path.
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
        use_flow_time_embedding: bool = True,
        flow_time_key: str = "flow_time",
        qhflow2_src: str | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        eSCNMDBackbone_ham = _import_qhflow2_escn_backbone(qhflow2_src)

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
        self.use_flow_time_embedding = bool(use_flow_time_embedding)
        self.flow_time_key = flow_time_key

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
            return torch.zeros(num_graphs, device=batch.device, dtype=self.dtype)
        value = torch.as_tensor(value, device=batch.device, dtype=self.dtype).reshape(-1)
        if value.numel() == 1:
            return value.expand(num_graphs)
        if value.numel() == num_graphs:
            return value
        if value.numel() == batch.numel():
            return scatter_mean(value, batch, dim=0, dim_size=num_graphs)
        return value[:1].expand(num_graphs)

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

        node = data.get(_keys.NODE_FEATURES_KEY, None)
        if node is None:
            node = torch.zeros(batch.numel(), self.rme_dim, device=batch.device, dtype=self.dtype)
        node = node.to(dtype=self.dtype)
        node_ctx = scatter_mean(node, batch, dim=0, dim_size=num_graphs)

        edge = data.get(_keys.EDGE_FEATURES_KEY, None)
        if edge is None:
            edge_ctx = torch.zeros(num_graphs, self.rme_dim, device=batch.device, dtype=self.dtype)
        elif edge.shape[0] == 0:
            edge_ctx = torch.zeros(num_graphs, self.rme_dim, device=batch.device, dtype=self.dtype)
        else:
            edge = edge.to(dtype=self.dtype)
            edge_batch = batch[edge_index[0]]
            edge_ctx = scatter_mean(edge, edge_batch, dim=0, dim_size=num_graphs)

        context = torch.cat([node_ctx, edge_ctx], dim=-1)
        return self.context_proj(context).reshape(num_graphs, self.matrix_l, self.hidden_size)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_batch(data)
        pos = data[_keys.POSITIONS_KEY].to(dtype=self.dtype)
        batch = data[_keys.BATCH_KEY].long()
        edge_index = data[_keys.EDGE_INDEX_KEY].long()
        atomic_numbers_value = data.get(_keys.ATOMIC_NUMBERS_KEY, None)
        if atomic_numbers_value is None:
            atom_type = data[_keys.ATOM_TYPE_KEY].long().reshape(-1)
            atomic_numbers_value = self._type_to_z.to(atom_type.device)[atom_type]
        atomic_numbers = atomic_numbers_value.long().reshape(-1, 1)
        num_graphs = self._num_graphs(batch)

        src, dst = edge_index[0], edge_index[1]
        edge_vec = pos[src] - pos[dst]
        edge_len = torch.linalg.norm(edge_vec, dim=-1)
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

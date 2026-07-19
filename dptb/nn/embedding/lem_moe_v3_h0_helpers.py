from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple, Union

import torch
from e3nn import o3
from e3nn.o3 import Linear
from torch_runstats.scatter import scatter

from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicData import register_fields
from dptb.data.interfaces.blockwise_tensor import (
    edge_feature_slices,
    ensure_spatial_block_mapper,
    infer_block_shapes,
    is_soc_uureal_mapper,
    mapper_max_norb,
    onsite_feature_slices,
)


# Keep H0 tensors as first-class node/edge fields once the dataset starts
# emitting them. This is a no-op for the current fallback-only workflow.
register_fields(
    node_fields=[
        _keys.NODE_H0_KEY,
        _keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY,
        _keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
    ],
    edge_fields=[
        _keys.EDGE_H0_KEY,
        _keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY,
        _keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
    ],
)


log = logging.getLogger(__name__)


def _sorted_irrep_coordinate_index(idp, *, device=None) -> tuple[o3.Irreps, torch.Tensor]:
    """Map mapper/raw orbpair coordinates to H0Init's sorted irrep layout."""
    raw = idp.get_irreps()
    sorted_result = raw.sort()
    raw_slices = raw.slices()
    coordinate_order: list[int] = []
    for term_index in sorted_result.p:
        sli = raw_slices[int(term_index)]
        coordinate_order.extend(range(int(sli.start), int(sli.stop)))
    index = torch.tensor(coordinate_order, dtype=torch.long, device=device)
    sorted_irreps = sorted_result.irreps.simplify()
    if index.numel() != int(idp.reduced_matrix_element):
        raise RuntimeError(
            "uu_real mapper irrep permutation width disagrees with reduced_matrix_element: "
            f"permutation={index.numel()}, mapper={idp.reduced_matrix_element}."
        )
    return sorted_irreps, index


class DirectUuRealBlockProjector(torch.nn.Module):
    """Bias-free mapper-derived AO-block contraction into hidden irreps.

    The plans gather only mapper-valid shell coordinates from each padded block,
    apply the mapper's raw->sorted irrep permutation, and use separate node/edge
    equivariant linear maps. No canonical block inverse or H0/full-H block
    materialization occurs in the hot path.
    """

    def __init__(self, idp, irreps_out, *, dtype, device) -> None:
        super().__init__()
        ensure_spatial_block_mapper(idp)
        if not is_soc_uureal_mapper(idp):
            raise ValueError(
                "DirectUuRealBlockProjector requires the explicit SOC uu_real mapper contract."
            )
        self.idp = idp
        self.canvas = mapper_max_norb(idp)
        irreps_in, sort_index = _sorted_irrep_coordinate_index(idp, device=device)
        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.register_buffer("sort_index", sort_index, persistent=True)

        node_plan = torch.full(
            (len(idp.type_names), int(idp.reduced_matrix_element)),
            -1,
            dtype=torch.long,
            device=device,
        )
        for symbol, type_index in idp.chemical_symbol_to_type.items():
            for row, col, feat in onsite_feature_slices(idp, symbol):
                flat = [
                    r * self.canvas + c
                    for r in range(int(row.start), int(row.stop))
                    for c in range(int(col.start), int(col.stop))
                ]
                node_plan[int(type_index), int(feat.start):int(feat.stop)] = torch.tensor(
                    flat, dtype=torch.long, device=device
                )

        edge_plan = torch.full(
            (len(idp.bond_types), int(idp.reduced_matrix_element)),
            -1,
            dtype=torch.long,
            device=device,
        )
        for bond, type_index in idp.bond_to_type.items():
            left, right = bond.split("-")
            for row, col, feat in edge_feature_slices(idp, left, right):
                flat = [
                    r * self.canvas + c
                    for r in range(int(row.start), int(row.stop))
                    for c in range(int(col.start), int(col.stop))
                ]
                edge_plan[int(type_index), int(feat.start):int(feat.stop)] = torch.tensor(
                    flat, dtype=torch.long, device=device
                )
        self.register_buffer("node_plan", node_plan, persistent=True)
        self.register_buffer("edge_plan", edge_plan, persistent=True)
        self.node_linear = Linear(
            irreps_in, self.irreps_out, shared_weights=True,
            internal_weights=True, biases=False,
        ).to(dtype=dtype, device=device)
        self.edge_linear = Linear(
            irreps_in, self.irreps_out, shared_weights=True,
            internal_weights=True, biases=False,
        ).to(dtype=dtype, device=device)

    def _gather(self, blocks: torch.Tensor, types: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        if blocks.ndim != 3 or tuple(blocks.shape[-2:]) != (self.canvas, self.canvas):
            raise ValueError(
                "uu_real residual blocks must use the mapper canvas "
                f"{(self.canvas, self.canvas)}; got {tuple(blocks.shape)}."
            )
        selected = plan.index_select(0, types.flatten().to(device=plan.device, dtype=torch.long))
        valid = selected >= 0
        raw = blocks.flatten(1).gather(1, selected.clamp_min(0).to(blocks.device))
        raw = raw * valid.to(device=blocks.device, dtype=blocks.dtype)
        return raw.index_select(1, self.sort_index.to(blocks.device))

    def forward(self, data, atom_type, bond_type, active_edges):
        required = (
            _keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY,
            _keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY,
            _keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
            _keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"uureal_block_ode residual projector missing keys={missing}.")
        node_blocks = data[required[0]]
        edge_blocks = data[required[1]]
        expected_node, expected_edge = infer_block_shapes(data, self.idp, device=node_blocks.device)
        node_shapes = torch.as_tensor(data[required[2]], device=node_blocks.device, dtype=torch.long)
        edge_shapes = torch.as_tensor(data[required[3]], device=edge_blocks.device, dtype=torch.long)
        if not torch.equal(node_shapes, expected_node) or not torch.equal(edge_shapes, expected_edge):
            raise ValueError("uu_real residual block shapes disagree with mapper/species plans.")
        node_raw = self._gather(node_blocks, atom_type, self.node_plan)
        edge_raw = self._gather(edge_blocks, bond_type, self.edge_plan)
        return self.node_linear(node_raw), self.edge_linear(edge_raw)[active_edges]


def _take_on_index_device(
    tensor: torch.Tensor,
    index: Union[int, torch.Tensor],
    result_device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if torch.is_tensor(index):
        tensor = tensor.to(device=index.device)
        out = tensor[index]
    else:
        out = tensor[index]

    if result_device is not None:
        out = out.to(device=result_device)
    return out


def _prepare_source_tensor(
    tensor: Optional[torch.Tensor],
    expected_dim: int,
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
    key: str,
    label: str,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None

    if torch.is_complex(tensor):
        log.warning("%s source `%s` is complex; only the real part is used.", label, key)
        tensor = tensor.real

    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != expected_dim:
        log.warning(
            "Skip %s source `%s` because last dim %s != expected %s.",
            label,
            key,
            tensor.shape[-1],
            expected_dim,
        )
        return None
    return tensor


def _get_feature_source(
    data: AtomicDataDict.Type,
    candidate_keys: Sequence[str],
    expected_dim: int,
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
    label: str,
) -> Optional[torch.Tensor]:
    tensor, _key = _get_feature_source_with_key(
        data,
        candidate_keys,
        expected_dim=expected_dim,
        dtype=dtype,
        device=device,
        label=label,
    )
    return tensor


def _get_feature_source_with_key(
    data: AtomicDataDict.Type,
    candidate_keys: Sequence[str],
    expected_dim: int,
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
    label: str,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Like :func:`_get_feature_source`, but also reports which key resolved."""
    for key in candidate_keys:
        tensor = _prepare_source_tensor(
            data.get(key, None),
            expected_dim=expected_dim,
            dtype=dtype,
            device=device,
            key=key,
            label=label,
        )
        if tensor is not None:
            return tensor, key
    return None, None


class H0InitLayer(torch.nn.Module):
    def __init__(
        self,
        base_init: torch.nn.Module,
        h0_node_key: str = _keys.NODE_H0_KEY,
        h0_edge_key: str = _keys.EDGE_H0_KEY,
        use_h0_node_init: bool = True,
        use_h0_edge_init: bool = True,
        h0_node_mode: str = "direct",
        fallback_to_hamiltonian: bool = True,
        fallback_node_key: str = _keys.NODE_FEATURES_KEY,
        fallback_edge_key: str = _keys.EDGE_FEATURES_KEY,
        allow_target_fallback_in_training: bool = False,
        use_uureal_residual_block_input: bool = False,
        merge_mode: str = "replace",
        self_edge_tol: float = 1e-8,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        if h0_node_mode not in {"direct", "self_edge"}:
            raise ValueError(f"Unsupported h0_node_mode={h0_node_mode!r}")
        if merge_mode not in {"replace", "add"}:
            raise ValueError(f"Unsupported merge_mode={merge_mode!r}")

        self.base_init = base_init
        self.idp = base_init.idp
        self.irreps_out = base_init.irreps_out
        self.h0_irreps = self.idp.orbpair_irreps.sort()[0].simplify()
        self.h0_dim = self.h0_irreps.dim
        self.h0_node_key = h0_node_key
        self.h0_edge_key = h0_edge_key
        self.use_h0_node_init = bool(use_h0_node_init)
        self.use_h0_edge_init = bool(use_h0_edge_init)
        self.h0_node_mode = h0_node_mode
        self.fallback_to_hamiltonian = fallback_to_hamiltonian
        self.fallback_node_key = fallback_node_key
        self.fallback_edge_key = fallback_edge_key
        self.allow_target_fallback_in_training = allow_target_fallback_in_training
        self.use_uureal_residual_block_input = bool(use_uureal_residual_block_input)
        self.merge_mode = merge_mode
        self.self_edge_tol = self_edge_tol
        self.dtype = dtype
        self.device = device

        self.node_projector = Linear(
            irreps_in=self.h0_irreps,
            irreps_out=self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        self.edge_projector = Linear(
            irreps_in=self.h0_irreps,
            irreps_out=self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        self.residual_block_projector = (
            DirectUuRealBlockProjector(
                self.idp, self.irreps_out, dtype=dtype, device=device
            )
            if self.use_uureal_residual_block_input
            else None
        )
        if self.residual_block_projector is not None:
            if self.h0_irreps != self.residual_block_projector.irreps_in:
                raise RuntimeError(
                    "H0Init and uu_real residual projector irrep layouts disagree."
                )
            self.register_buffer(
                "_uureal_h0_sort_index",
                self.residual_block_projector.sort_index.detach().clone(),
                persistent=True,
            )

    def _candidate_keys(
        self,
        primary_key: str,
        feature_fallback_key: Optional[str],
        hamiltonian_key: Optional[str],
    ) -> list[str]:
        keys: list[str] = []
        for key in [primary_key]:
            if key and key not in keys:
                keys.append(key)

        if self.fallback_to_hamiltonian:
            for key in [hamiltonian_key, feature_fallback_key]:
                if key and key not in keys:
                    keys.append(key)

        return keys

    def _align_feature_rows(
        self,
        tensor: torch.Tensor,
        n_rows: int,
    ) -> torch.Tensor:
        if tensor.shape[0] == n_rows:
            return tensor
        if tensor.shape[0] > n_rows:
            return tensor[:n_rows]

        pad = torch.zeros(
            n_rows - tensor.shape[0],
            tensor.shape[1],
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=0)

    def _merge_features(
        self,
        base_features: torch.Tensor,
        h0_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.merge_mode == "replace":
            return h0_features

        n_rows = max(base_features.shape[0], h0_features.shape[0])
        base_features = self._align_feature_rows(base_features, n_rows)
        h0_features = self._align_feature_rows(h0_features, n_rows)
        return base_features + h0_features

    def _mask_node_source(
        self,
        node_source: torch.Tensor,
        atom_type: torch.Tensor,
    ) -> torch.Tensor:
        mask = _take_on_index_device(
            self.idp.mask_to_nrme,
            atom_type.flatten(),
            result_device=node_source.device,
        )
        return node_source * mask.to(dtype=node_source.dtype)

    def _mask_edge_source(
        self,
        edge_source: torch.Tensor,
        bond_type: torch.Tensor,
    ) -> torch.Tensor:
        mask = _take_on_index_device(
            self.idp.mask_to_erme,
            bond_type.flatten(),
            result_device=edge_source.device,
        )
        return edge_source * mask.to(dtype=edge_source.dtype)

    def _node_from_self_edge(
        self,
        edge_features: torch.Tensor,
        data: AtomicDataDict.Type,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        active_edges: torch.Tensor,
        atom_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        active_src = edge_index[0][active_edges]
        active_dst = edge_index[1][active_edges]
        active_len = edge_length[active_edges]
        self_mask = torch.logical_and(active_src == active_dst, active_len <= self.self_edge_tol)
        node_features = scatter(
            edge_features[self_mask],
            active_src[self_mask],
            dim=0,
            dim_size=atom_type.numel(),
        )
        return node_features, self_mask.any()

    def _guard_target_fallback(self, resolved_key: Optional[str], primary_key: str, label: str) -> None:
        """Refuse to train on the target Hamiltonian dressed up as an H0 input.

        The candidate chain falls back to ``node_features``/``edge_features`` /
        the ``*_hamiltonian`` blocks when the H0 keys are absent.  At inference
        that is a deliberate surrogate; during training it silently feeds the
        label to the model.  Fail closed unless explicitly opted in.
        """
        if (
            resolved_key is None
            or resolved_key == primary_key
            or not self.training
            or self.allow_target_fallback_in_training
        ):
            return
        raise RuntimeError(
            f"H0InitLayer resolved `{resolved_key}` as the {label} input because "
            f"`{primary_key}` is missing from the batch. Feeding the target "
            "Hamiltonian/features as a model input during training is a label "
            "leak. Provide the H0 keys (or point h0_node_key/h0_edge_key at, "
            "e.g., node_overlap/edge_overlap for an overlap-feature run), or set "
            "embedding option allow_target_fallback_in_training=true to opt in "
            "explicitly."
        )

    def _fallback_node_features(
        self,
        data: AtomicDataDict.Type,
        atom_type: torch.Tensor,
        base_node_features: torch.Tensor,
    ) -> torch.Tensor:
        node_source, node_source_key = _get_feature_source_with_key(
            data=data,
            candidate_keys=self._candidate_keys(
                self.h0_node_key,
                self.fallback_node_key,
                _keys.NODE_HAMILTONIAN_KEY,
            ),
            expected_dim=self.h0_dim,
            dtype=self.dtype,
            device=self.device,
            label="node H0",
        )
        self._guard_target_fallback(node_source_key, self.h0_node_key, "node H0")
        if node_source is None:
            log.warning(
                "No usable node H0 source found; falling back to the original node init."
            )
            return base_node_features

        node_source = self._mask_node_source(node_source, atom_type)
        if self.use_uureal_residual_block_input:
            node_source = node_source.index_select(1, self._uureal_h0_sort_index)
        return self._merge_features(base_node_features, self.node_projector(node_source))

    def forward(
        self,
        data: AtomicDataDict.Type,
        edge_index: torch.Tensor,
        atom_type: torch.Tensor,
        bond_type: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_length: torch.Tensor,
        edge_one_hot: torch.Tensor,
        active_edges: Optional[torch.Tensor] = None,
        cutoff_coeffs: Optional[torch.Tensor] = None,
    ):
        latents, base_node_features, base_edge_features, cutoff_coeffs, active_edges = self.base_init(
            edge_index,
            atom_type,
            bond_type,
            edge_sh,
            edge_length,
            edge_one_hot,
            active_edges,
            cutoff_coeffs,
        )

        edge_features_h0 = None
        edge_features = base_edge_features
        if self.use_h0_edge_init:
            edge_source, edge_source_key = _get_feature_source_with_key(
                data=data,
                candidate_keys=self._candidate_keys(
                    self.h0_edge_key,
                    self.fallback_edge_key,
                    _keys.EDGE_HAMILTONIAN_KEY,
                ),
                expected_dim=self.h0_dim,
                dtype=self.dtype,
                device=self.device,
                label="edge H0",
            )
            self._guard_target_fallback(edge_source_key, self.h0_edge_key, "edge H0")
            if edge_source is None:
                log.warning(
                    "No usable edge H0 source found; falling back to the original InitLayer output."
                )
                return latents, base_node_features, base_edge_features, cutoff_coeffs, active_edges
            edge_source = self._mask_edge_source(edge_source, bond_type)
            if self.use_uureal_residual_block_input:
                edge_source = edge_source.index_select(1, self._uureal_h0_sort_index)
            edge_features_h0 = self.edge_projector(edge_source[active_edges])
            edge_features = self._merge_features(base_edge_features, edge_features_h0)

        if not self.use_h0_node_init:
            return latents, base_node_features, edge_features, cutoff_coeffs, active_edges

        if self.h0_node_mode == "self_edge":
            if edge_features_h0 is None:
                raise ValueError(
                    "h0_node_mode='self_edge' requires use_h0_edge_init=true."
                )
            self_node_features_h0, has_self_edge = self._node_from_self_edge(
                edge_features=edge_features_h0,
                data=data,
                edge_index=edge_index,
                edge_length=edge_length,
                active_edges=active_edges,
                atom_type=atom_type,
            )
            self_node_features = self._merge_features(base_node_features, self_node_features_h0)
            fallback_node_features = self._fallback_node_features(data, atom_type, base_node_features)
            node_features = torch.where(
                has_self_edge.reshape(1, 1),
                self_node_features,
                fallback_node_features,
            )
        else:
            node_features = self._fallback_node_features(data, atom_type, base_node_features)

        if self.residual_block_projector is not None:
            residual_node, residual_edge = self.residual_block_projector(
                data, atom_type, bond_type, active_edges
            )
            node_features = node_features + residual_node
            edge_features = edge_features + residual_edge

        return latents, node_features, edge_features, cutoff_coeffs, active_edges

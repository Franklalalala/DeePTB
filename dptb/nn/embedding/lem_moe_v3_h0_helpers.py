from __future__ import annotations

import logging
import re
from typing import Optional, Sequence, Tuple, Union

import torch
from e3nn import o3
from e3nn.o3 import Linear, wigner_3j
from torch_runstats.scatter import scatter

from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicData import register_fields
from dptb.data.interfaces.blockwise_tensor import (
    edge_feature_slices,
    ensure_non_soc_mapper,
    ensure_spatial_block_mapper,
    infer_block_shapes,
    is_soc_uureal_mapper,
    mapper_max_norb,
    onsite_feature_slices,
)
from dptb.utils.constants import anglrMId


# Keep H0 tensors as first-class node/edge fields once the dataset starts
# emitting them. This is a no-op for the current fallback-only workflow.
register_fields(
    node_fields=[
        _keys.NODE_H0_KEY,
        _keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY,
        _keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
        _keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY,
        _keys.NODE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY,
    ],
    edge_fields=[
        _keys.EDGE_H0_KEY,
        _keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY,
        _keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY,
        _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY,
        _keys.EDGE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY,
    ],
)


log = logging.getLogger(__name__)


def _shell_angular_momentum(label: str) -> int:
    """Return the angular momentum ``l`` of an AO shell label such as ``2p``."""
    letters = re.findall(r"[A-Za-z]", str(label))
    if not letters:
        raise ValueError(f"Cannot parse angular momentum from orbital label {label!r}.")
    key = letters[-1].lower()
    if key not in anglrMId:
        raise ValueError(f"Unsupported AO shell letter {key!r} in label {label!r}.")
    return int(anglrMId[key])


def _build_uureal_cg_change_of_basis(idp, *, dtype, device) -> torch.Tensor:
    """Fixed AO-product -> coupled-RME change of basis in the raw orbpair layout.

    For every directed orbital-shell pair ``(l1,l2)`` the flattened AO sub-block
    (product basis, row-major ``i*w+j``) is contracted to coupled reduced matrix
    elements with the same ``wigner_3j(l1,l2,L)*sqrt(2L+1)`` convention that
    :class:`~dptb.nn.hamiltonian.E3Hamiltonian` uses for the *forward* RME->block
    expansion.  The result is one block-diagonal ``[rme, rme]`` matrix ``C`` with
    ``coupled = C @ product``; because the complete set of coupled angular momenta
    is an orthonormal basis of the product space, ``C`` is exactly the inverse of
    that forward expansion.  It is species-independent: the raw orbpair layout is
    shared by every atom/bond type (per-type validity is handled by the gather
    plans), so a single shared matrix suffices.
    """
    ensure_spatial_block_mapper(idp)
    rme = int(idp.reduced_matrix_element)
    change_of_basis = torch.zeros((rme, rme), dtype=dtype, device=device)
    for pair, feat in idp.orbpair_maps.items():
        left, _, right = str(pair).partition("-")
        l1 = _shell_angular_momentum(left)
        l2 = _shell_angular_momentum(right)
        h, w = 2 * l1 + 1, 2 * l2 + 1
        width = int(feat.stop - feat.start)
        if width != h * w:
            raise RuntimeError(
                f"orbpair {pair!r} feature width {width} != AO product dim {h * w}."
            )
        cg = torch.cat(
            [
                wigner_3j(l1, l2, ell, dtype=dtype, device=device)
                * float(2 * ell + 1) ** 0.5
                for ell in range(abs(l1 - l2), l1 + l2 + 1)
            ],
            dim=-1,
        )  # (h, w, n_rme)
        # coupled[r] = sum_{i,j} cg[i,j,r] * product[i*w + j]
        block = cg.reshape(h * w, width).transpose(0, 1)  # (n_rme, h*w): [r, i*w+j]
        change_of_basis[feat.start:feat.stop, feat.start:feat.stop] = block
    return change_of_basis


def _sorted_irrep_coordinate_index(idp, *, device=None) -> tuple[o3.Irreps, torch.Tensor]:
    """Map raw orbpair coordinates to H0Init's sorted irrep layout.

    Built from first principles so the returned permutation and the returned
    ``Irreps`` are guaranteed consistent: raw terms are stably sorted by irrep,
    the sorted term coordinate ranges are concatenated into the gather index, and
    the same sorted term order defines the (simplified) irreps.  ``raw[index]``
    therefore transforms exactly as ``sorted_irreps`` -- a prerequisite for the
    equivariant linear that consumes it.  (Deriving the index from
    ``Irreps.sort().p`` instead mixes up unequal irreps into shared multiplicity
    slots for multi-shell bases and breaks rotation covariance.)
    """
    raw = idp.get_irreps()
    raw_slices = raw.slices()
    order = sorted(range(len(raw)), key=lambda term: raw[term].ir)
    coordinate_order: list[int] = []
    sorted_terms: list[tuple[int, o3.Irrep]] = []
    for term_index in order:
        sli = raw_slices[term_index]
        coordinate_order.extend(range(int(sli.start), int(sli.stop)))
        sorted_terms.append((int(raw[term_index].mul), raw[term_index].ir))
    index = torch.tensor(coordinate_order, dtype=torch.long, device=device)
    sorted_irreps = o3.Irreps(sorted_terms).simplify()
    mapper_dim = int(idp.reduced_matrix_element)
    if index.numel() != mapper_dim:
        raise RuntimeError(
            "mapper irrep permutation width disagrees with reduced_matrix_element: "
            f"permutation={index.numel()}, mapper={idp.reduced_matrix_element}."
        )
    expected = torch.arange(mapper_dim, dtype=torch.long, device=index.device)
    if not torch.equal(index.sort().values, expected):
        raise RuntimeError(
            "mapper raw-to-sorted irrep coordinate index is not a permutation."
        )
    return sorted_irreps, index


class DirectUuRealBlockProjector(torch.nn.Module):
    """Bias-free AO-block Clebsch-Gordan contraction into hidden irreps.

    Each padded AO block is a *product-basis* tensor.  Per directed orbital-shell
    pair the plan gathers the mapper-valid ``(2l1+1)(2l2+1)`` sub-block entries,
    a fixed ``wigner_3j(l1,l2,L)*sqrt(2L+1)`` change of basis contracts them into
    *coupled* reduced matrix elements (the exact inverse of E3Hamiltonian's
    forward RME->block expansion), the mapper's raw->sorted irrep permutation is
    applied, and only then does a separate node/edge equivariant linear mix the
    multiplicities.  Feeding raw product coordinates straight into the equivariant
    linear -- as a plain flatten/gather would -- is not rotation covariant; the CG
    contraction is what restores covariance.  No canonical block inverse or
    H0/full-H block materialization occurs in the hot path.
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
        self.register_buffer(
            "cg_change_of_basis",
            _build_uureal_cg_change_of_basis(idp, dtype=dtype, device=device),
            persistent=True,
        )

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

    def _gather_product(
        self, blocks: torch.Tensor, types: torch.Tensor, plan: torch.Tensor
    ) -> torch.Tensor:
        """Gather mapper-valid AO-product entries into the raw orbpair layout."""
        if blocks.ndim != 3 or tuple(blocks.shape[-2:]) != (self.canvas, self.canvas):
            raise ValueError(
                "uu_real residual blocks must use the mapper canvas "
                f"{(self.canvas, self.canvas)}; got {tuple(blocks.shape)}."
            )
        selected = plan.index_select(0, types.flatten().to(device=plan.device, dtype=torch.long))
        valid = selected >= 0
        raw = blocks.flatten(1).gather(1, selected.clamp_min(0).to(blocks.device))
        return raw * valid.to(device=blocks.device, dtype=blocks.dtype)

    def _contract(self, blocks: torch.Tensor, types: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        """AO block -> coupled RME (raw orbpair layout) -> sorted irrep layout.

        Returns the exact input consumed by the equivariant linear, i.e. the
        coupled reduced matrix elements in the ``irreps_in`` (sorted) coordinate
        order.  A zero block maps to zero coordinates (the change of basis and the
        permutation are both linear and bias-free), so bit-level zero
        preservation is retained.
        """
        raw_product = self._gather_product(blocks, types, plan)
        change_of_basis = self.cg_change_of_basis.to(device=blocks.device, dtype=blocks.dtype)
        raw_coupled = torch.einsum("kc,nc->nk", change_of_basis, raw_product)
        return raw_coupled.index_select(1, self.sort_index.to(blocks.device))

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
        node_raw = self._contract(node_blocks, atom_type, self.node_plan)
        edge_raw = self._contract(edge_blocks, bond_type, self.edge_plan)
        return self.node_linear(node_raw), self.edge_linear(edge_raw)[active_edges]


class DirectSpatialResidualBlockProjector(torch.nn.Module):
    """Bias-free non-SOC AO-block Clebsch-Gordan contraction into hidden irreps.

    Structural sibling of :class:`DirectUuRealBlockProjector` for the plain
    non-SOC ``residual_ao_block_ode`` mode.  Each padded AO block is a
    *product-basis* tensor.  Per directed orbital-shell pair the plan gathers the
    mapper-valid ``(2l1+1)(2l2+1)`` sub-block entries, a fixed
    ``wigner_3j(l1,l2,L)*sqrt(2L+1)`` change of basis contracts them into
    *coupled* reduced matrix elements (the exact inverse of E3Hamiltonian's
    forward RME->block expansion), the mapper's raw->sorted irrep permutation is
    applied, and only then does a separate node/edge equivariant linear mix the
    multiplicities.  Feeding raw product coordinates straight into the equivariant
    linear -- as a plain flatten/gather would -- is not rotation covariant; the CG
    contraction is what restores covariance.  No canonical block inverse or
    H0/full-H block materialization occurs in the hot path.
    """

    def __init__(self, idp, irreps_out, *, dtype, device) -> None:
        super().__init__()
        # This projector is strictly non-SOC.  A reduced SOC uu_real mapper would
        # pass ensure_spatial_block_mapper, so reject it explicitly (and before
        # ensure_non_soc_mapper's NotImplementedError) with a ValueError so a
        # compact-uu contract can never masquerade as spatial residual input.
        if is_soc_uureal_mapper(idp):
            raise ValueError(
                "DirectSpatialResidualBlockProjector consumes plain non-SOC "
                "records; the SOC uu_real mapper may not masquerade as "
                "residual_ao_block_ode spatial residual input."
            )
        ensure_non_soc_mapper(idp)
        self.idp = idp
        self.canvas = mapper_max_norb(idp)
        irreps_in, sort_index = _sorted_irrep_coordinate_index(idp, device=device)
        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.register_buffer("sort_index", sort_index, persistent=True)
        self.register_buffer(
            "cg_change_of_basis",
            _build_uureal_cg_change_of_basis(idp, dtype=dtype, device=device),
            persistent=True,
        )

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

    def _gather_product(
        self, blocks: torch.Tensor, types: torch.Tensor, plan: torch.Tensor
    ) -> torch.Tensor:
        """Gather mapper-valid AO-product entries into the raw orbpair layout."""
        if blocks.ndim != 3 or tuple(blocks.shape[-2:]) != (self.canvas, self.canvas):
            raise ValueError(
                "spatial residual blocks must use the mapper canvas "
                f"{(self.canvas, self.canvas)}; got {tuple(blocks.shape)}."
            )
        selected = plan.index_select(0, types.flatten().to(device=plan.device, dtype=torch.long))
        valid = selected >= 0
        raw = blocks.flatten(1).gather(1, selected.clamp_min(0).to(blocks.device))
        return raw * valid.to(device=blocks.device, dtype=blocks.dtype)

    def _contract(self, blocks: torch.Tensor, types: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        """AO block -> coupled RME (raw orbpair layout) -> sorted irrep layout.

        Returns the exact input consumed by the equivariant linear, i.e. the
        coupled reduced matrix elements in the ``irreps_in`` (sorted) coordinate
        order.  A zero block maps to zero coordinates (the change of basis and the
        permutation are both linear and bias-free), so bit-level zero
        preservation is retained.
        """
        raw_product = self._gather_product(blocks, types, plan)
        change_of_basis = self.cg_change_of_basis.to(device=blocks.device, dtype=blocks.dtype)
        raw_coupled = torch.einsum("kc,nc->nk", change_of_basis, raw_product)
        return raw_coupled.index_select(1, self.sort_index.to(blocks.device))

    def forward(self, data, atom_type, bond_type, active_edges):
        required = (
            _keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY,
            _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY,
            _keys.NODE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY,
            _keys.EDGE_SPATIAL_RESIDUAL_BLOCK_SHAPE_KEY,
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"residual_ao_block_ode spatial projector missing keys={missing}.")
        node_blocks = data[required[0]]
        edge_blocks = data[required[1]]
        expected_node, expected_edge = infer_block_shapes(data, self.idp, device=node_blocks.device)
        node_shapes = torch.as_tensor(data[required[2]], device=node_blocks.device, dtype=torch.long)
        edge_shapes = torch.as_tensor(data[required[3]], device=edge_blocks.device, dtype=torch.long)
        if not torch.equal(node_shapes, expected_node) or not torch.equal(edge_shapes, expected_edge):
            raise ValueError("spatial residual block shapes disagree with mapper/species plans.")
        node_raw = self._contract(node_blocks, atom_type, self.node_plan)
        edge_raw = self._contract(edge_blocks, bond_type, self.edge_plan)
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
    # v2 fixes the generic non-uu_real H0 input contract: mapper/raw orbpair
    # coordinates are permuted into sorted-irrep coordinates before the e3nn
    # node/edge projectors.  Keep the marker in module metadata rather than as a
    # persistent tensor so parameter keys and initialization RNG stay unchanged.
    _version = 2

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
        use_spatial_residual_block_input: bool = False,
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
        if use_uureal_residual_block_input and use_spatial_residual_block_input:
            raise ValueError(
                "use_uureal_residual_block_input and "
                "use_spatial_residual_block_input are mutually exclusive: the SOC "
                "uu_real and non-SOC spatial residual projectors consume "
                "different block contracts."
            )

        self.base_init = base_init
        self.idp = base_init.idp
        self.irreps_out = base_init.irreps_out
        self.h0_irreps = self.idp.orbpair_irreps.sort()[0].simplify()
        derived_h0_irreps, h0_sort_index = _sorted_irrep_coordinate_index(
            self.idp, device=device
        )
        if self.h0_irreps != derived_h0_irreps:
            raise RuntimeError(
                "H0Init sorted irreps disagree with the mapper-derived "
                "raw-to-sorted coordinate permutation."
            )
        # H0/node/edge feature sources and their mapper masks use raw orbpair
        # order, while the two equivariant projectors below declare sorted
        # irreps.  Every H0 path must therefore mask in raw order and apply this
        # permutation immediately before projection.  Keep the generic buffer
        # non-persistent so legacy/default state_dict keys remain bit-exact.
        self.register_buffer(
            "_h0_sort_index", h0_sort_index, persistent=False
        )
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
        self.use_spatial_residual_block_input = bool(use_spatial_residual_block_input)
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
            if not torch.equal(
                self._h0_sort_index, self._uureal_h0_sort_index
            ):
                raise RuntimeError(
                    "H0Init and uu_real residual projector raw-to-sorted "
                    "coordinate permutations disagree."
                )
        # The non-SOC spatial residual projector is a separate conditioning
        # channel for residual_ao_block_ode.  It owns an equivalent mapper-derived
        # raw-to-sorted permutation for its AO-block contraction; the ordinary H0
        # channel still uses the generic permutation registered above.
        self.spatial_residual_block_projector = (
            DirectSpatialResidualBlockProjector(
                self.idp, self.irreps_out, dtype=dtype, device=device
            )
            if self.use_spatial_residual_block_input
            else None
        )
        if self.spatial_residual_block_projector is not None:
            if self.h0_irreps != self.spatial_residual_block_projector.irreps_in:
                raise RuntimeError(
                    "H0Init and spatial residual projector irrep layouts disagree."
                )
            if not torch.equal(
                self._h0_sort_index,
                self.spatial_residual_block_projector.sort_index,
            ):
                raise RuntimeError(
                    "H0Init and spatial residual projector raw-to-sorted "
                    "coordinate permutations disagree."
                )

    def _legacy_unsorted_h0_checkpoint_is_unsafe(self) -> bool:
        """Whether a pre-v2 checkpoint was trained against the broken layout."""

        sort_index = self._h0_sort_index.detach()
        identity = torch.arange(
            sort_index.numel(),
            dtype=sort_index.dtype,
            device=sort_index.device,
        )
        return bool(
            (self.use_h0_node_init or self.use_h0_edge_init)
            and not self.use_uureal_residual_block_input
            and not torch.equal(sort_index, identity)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version")
        module_name = prefix[:-1] if prefix.endswith(".") else prefix
        if self._legacy_unsorted_h0_checkpoint_is_unsafe():
            # Explicit pre-v2 markers are still fail-closed. Missing metadata is
            # the ensemble saver flattening state_dict into a plain dict; those
            # weights were trained under the current sorted-irrep contract.
            if version is not None and int(version) < self._version:
                error_msgs.append(
                    f"{module_name or '<root>'}: checkpoint H0 layout version "
                    f"{version!r} predates the H0 raw-to-sorted RME layout fix "
                    f"(required version {self._version}). This high-l non-uu_real "
                    "checkpoint cannot be resumed safely; retrain from initialization."
                )
            elif version is None:
                log.warning(
                    "%s: H0 layout version metadata missing; loading the current "
                    "sorted-irrep contract. Explicit version < %s still fails closed.",
                    module_name or "<root>",
                    self._version,
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
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
        node_source = node_source.index_select(1, self._h0_sort_index)
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
            edge_source = edge_source.index_select(1, self._h0_sort_index)
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

        if self.spatial_residual_block_projector is not None:
            spatial_residual_node, spatial_residual_edge = self.spatial_residual_block_projector(
                data, atom_type, bond_type, active_edges
            )
            node_features = node_features + spatial_residual_node
            edge_features = edge_features + spatial_residual_edge

        return latents, node_features, edge_features, cutoff_coeffs, active_edges

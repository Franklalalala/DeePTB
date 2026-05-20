# SPDX-License-Identifier: LGPL-3.0-or-later
"""Clean block-wise tensor utilities for non-SOC NexTHam-style DeePTB.

This module keeps one responsibility per function:

1. derive atom/edge metadata from an ``AtomicData``-like mapping;
2. convert DeePTB non-SOC feature tensors to padded AO blocks;
3. pack official ``feature_to_block`` dictionaries into ordered tensors;
4. Hermitian-complete hopping blocks from reverse edges;
5. compute block-level and feature-compatible metric components.

The feature-compatible metrics are computed by walking ``OrbitalMapper``'s
canonical orbital-pair slices, not by storing feature masks in LMDB.
SOC is intentionally disabled in this minimal package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

try:  # DeePTB runtime
    from dptb.data import _keys
except Exception:  # pragma: no cover - standalone tests
    class _FallbackKeys:
        NODE_FEATURES_KEY = "node_features"
        EDGE_FEATURES_KEY = "edge_features"
        NODE_H0_KEY = "node_h0"
        EDGE_H0_KEY = "edge_h0"
        ATOMIC_NUMBERS_KEY = "atomic_numbers"
        ATOM_TYPE_KEY = "atom_types"
        EDGE_TYPE_KEY = "edge_types"
        EDGE_INDEX_KEY = "edge_index"
        EDGE_CELL_SHIFT_KEY = "edge_cell_shift"

    _keys = _FallbackKeys()

try:
    from ase.data import chemical_symbols
except Exception:  # pragma: no cover - enough for unit tests
    chemical_symbols = ["X"] * 119
    chemical_symbols[1] = "H"
    chemical_symbols[3] = "Li"
    chemical_symbols[6] = "C"
    chemical_symbols[14] = "Si"
    chemical_symbols[50] = "Sn"


# Converted target labels: delta H in AO-block space.
NODE_DELTA_HAMIL_BLOCKS_KEY = "node_delta_hamil_blocks"
EDGE_DELTA_HAMIL_BLOCKS_KEY = "edge_delta_hamil_blocks"
NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY = "node_delta_hamil_block_shape"
EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY = "edge_delta_hamil_block_shape"

# Converted H0 initial/reference tensors.
NODE_H0_BLOCKS_KEY = "node_h0_blocks"
EDGE_H0_BLOCKS_KEY = "edge_h0_blocks"
NODE_H0_BLOCK_SHAPE_KEY = "node_h0_block_shape"
EDGE_H0_BLOCK_SHAPE_KEY = "edge_h0_block_shape"

# Model prediction keys used by the block loss.
NODE_PRED_HAMIL_BLOCKS_KEY = "node_hamil_blocks"
EDGE_PRED_HAMIL_BLOCKS_KEY = "edge_hamil_blocks"


@dataclass(frozen=True)
class BlockTensorResult:
    """Ordered padded block tensors plus per-block valid shapes."""

    node_blocks: Optional[torch.Tensor]
    edge_blocks: Optional[torch.Tensor]
    node_shapes: Optional[torch.Tensor]
    edge_shapes: Optional[torch.Tensor]


@dataclass(frozen=True)
class ComponentSums:
    """Global metric components; take square root only after summing counts."""

    abs_sum: torch.Tensor
    square_sum: torch.Tensor
    count: torch.Tensor


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def data_get(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return data[key]
    except Exception:
        return default


def key(name: str, fallback: str) -> str:
    return getattr(_keys, name, fallback)


def as_tensor(value: Any, *, device=None, dtype=None) -> torch.Tensor:
    out = value if torch.is_tensor(value) else torch.as_tensor(value)
    if device is not None or dtype is not None:
        out = out.to(
            device=device if device is not None else out.device,
            dtype=dtype if dtype is not None else out.dtype,
        )
    return out


def ensure_non_soc_mapper(idp: Any) -> Any:
    """Initialize lazy ``OrbitalMapper`` maps and fail fast for SOC."""
    if bool(getattr(idp, "has_soc", False)):
        raise NotImplementedError(
            "This clean block-wise package is for the current non-SOC workflow. "
            "SOC real/imag feature semantics need a separate validation pass."
        )
    if not getattr(idp, "norbs", None) and hasattr(idp, "get_orbital_maps"):
        idp.get_orbital_maps()
    if not getattr(idp, "orbpair_maps", None) and hasattr(idp, "get_orbpair_maps"):
        idp.get_orbpair_maps()
    return idp


def mapper_max_norb(idp: Any) -> int:
    ensure_non_soc_mapper(idp)
    return max(int(v) for v in idp.norbs.values())


def symbols_from_data(data: Mapping[str, Any], idp: Any) -> List[str]:
    """Return atom symbols in data order from atomic_numbers or atom_types."""
    atomic_numbers = data_get(data, key("ATOMIC_NUMBERS_KEY", "atomic_numbers"), None)
    if atomic_numbers is not None:
        z = as_tensor(atomic_numbers).detach().cpu().long().flatten().tolist()
        return [chemical_symbols[int(v)] for v in z]

    atom_types = data_get(data, key("ATOM_TYPE_KEY", "atom_types"), None)
    if atom_types is None:
        raise KeyError("Need atomic_numbers or atom_types to derive atom symbols.")
    atom_types = as_tensor(atom_types).detach().cpu().long().flatten().tolist()

    if hasattr(idp, "type_to_chemical_symbol"):
        return [idp.type_to_chemical_symbol[int(t)] for t in atom_types]
    rev = {int(v): s for s, v in getattr(idp, "chemical_symbol_to_type", {}).items()}
    if not rev:
        raise KeyError("Mapper lacks type_to_chemical_symbol/chemical_symbol_to_type.")
    return [rev[int(t)] for t in atom_types]


def atom_types_from_data(data: Mapping[str, Any], idp: Any, *, device=None) -> torch.Tensor:
    atom_types = data_get(data, key("ATOM_TYPE_KEY", "atom_types"), None)
    if atom_types is not None:
        return as_tensor(atom_types, device=device, dtype=torch.long).flatten()
    mapping = getattr(idp, "chemical_symbol_to_type", None)
    if mapping is None:
        type_names = getattr(idp, "type_names", None)
        if type_names is None:
            raise KeyError("Mapper lacks chemical_symbol_to_type/type_names.")
        mapping = {s: i for i, s in enumerate(type_names)}
    return torch.as_tensor([int(mapping[s]) for s in symbols_from_data(data, idp)], dtype=torch.long, device=device)


def edge_types_from_data(data: Mapping[str, Any], idp: Any, *, device=None) -> torch.Tensor:
    edge_types = data_get(data, key("EDGE_TYPE_KEY", "edge_types"), None)
    if edge_types is not None:
        return as_tensor(edge_types, device=device, dtype=torch.long).flatten()

    edge_index = edge_index_from_data(data)
    if hasattr(idp, "transform_bond"):
        atomic_numbers = data_get(data, key("ATOMIC_NUMBERS_KEY", "atomic_numbers"), None)
        if atomic_numbers is not None:
            z = as_tensor(atomic_numbers, device=edge_index.device, dtype=torch.long).flatten()
            return idp.transform_bond(z[edge_index[0]], z[edge_index[1]]).flatten().to(device=device, dtype=torch.long)

    symbols = symbols_from_data(data, idp)
    bond_to_type = getattr(idp, "bond_to_type", None)
    if bond_to_type is None and hasattr(idp, "bond_types"):
        bond_to_type = {b: i for i, b in enumerate(idp.bond_types)}
    if bond_to_type is None:
        raise KeyError("Mapper lacks transform_bond/bond_to_type.")
    vals = [int(bond_to_type[f"{symbols[int(u)]}-{symbols[int(v)]}"]) for u, v in edge_index.T.cpu().tolist()]
    return torch.as_tensor(vals, dtype=torch.long, device=device)


def edge_index_from_data(data: Mapping[str, Any]) -> torch.Tensor:
    edge_index = data_get(data, key("EDGE_INDEX_KEY", "edge_index"), None)
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    return as_tensor(edge_index, dtype=torch.long)


def edge_shift_from_data(data: Mapping[str, Any], *, n_edges: Optional[int] = None, device=None) -> torch.Tensor:
    shift = data_get(data, key("EDGE_CELL_SHIFT_KEY", "edge_cell_shift"), None)
    if shift is None:
        n = int(n_edges if n_edges is not None else edge_index_from_data(data).shape[1])
        return torch.zeros((n, 3), dtype=torch.long, device=device)
    # Some existing LMDBs store shifts as float32.  They are cell translations,
    # so round before casting to long.
    return as_tensor(shift, device=device).round().to(dtype=torch.long)


def matrix_shape_for_symbol(idp: Any, symbol: str) -> Tuple[int, int]:
    return int(idp.norbs[symbol]), int(idp.norbs[symbol])


def matrix_shape_for_bond(idp: Any, sym_i: str, sym_j: str) -> Tuple[int, int]:
    return int(idp.norbs[sym_i]), int(idp.norbs[sym_j])


def infer_block_shapes(data: Mapping[str, Any], idp: Any, *, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    ensure_non_soc_mapper(idp)
    symbols = symbols_from_data(data, idp)
    node_shapes = torch.as_tensor([matrix_shape_for_symbol(idp, s) for s in symbols], dtype=torch.long, device=device)
    edge_index = edge_index_from_data(data).detach().cpu()
    edge_shapes = torch.as_tensor(
        [matrix_shape_for_bond(idp, symbols[int(u)], symbols[int(v)]) for u, v in edge_index.T.tolist()],
        dtype=torch.long,
        device=device,
    )
    return node_shapes, edge_shapes


def block_mask_from_shapes(shapes: torch.Tensor, block_hw: Tuple[int, int]) -> torch.Tensor:
    shapes = as_tensor(shapes, dtype=torch.long)
    h, w = int(block_hw[0]), int(block_hw[1])
    rows = torch.arange(h, device=shapes.device).view(1, h, 1)
    cols = torch.arange(w, device=shapes.device).view(1, 1, w)
    return (rows < shapes[:, 0].view(-1, 1, 1)) & (cols < shapes[:, 1].view(-1, 1, 1))


def block_key(i: int, j: int, shift: Sequence[int], *, start_id: int = 0) -> str:
    return f"{int(i) + start_id}_{int(j) + start_id}_{int(shift[0])}_{int(shift[1])}_{int(shift[2])}"


def _zero_component_like(tensor: Optional[torch.Tensor], *, device=None) -> ComponentSums:
    if tensor is None:
        dtype = torch.float32
        dev = torch.device("cpu") if device is None else device
    else:
        dtype = tensor.real.dtype if tensor.is_complex() else tensor.dtype
        dev = tensor.device
    z = torch.zeros((), dtype=dtype, device=dev)
    return ComponentSums(z, z, z)


# -----------------------------------------------------------------------------
# Mapper feature slices
# -----------------------------------------------------------------------------


def onsite_feature_slices(idp: Any, symbol: str) -> Iterable[Tuple[slice, slice, slice]]:
    """Yield ``(row_slice, col_slice, feature_slice)`` for onsite features."""
    basis = idp.basis[symbol]
    for i, basis_i in enumerate(basis):
        row = idp.orbital_maps[symbol][basis_i]
        full_i = idp.basis_to_full_basis[symbol][basis_i]
        for basis_j in basis[i:]:
            col = idp.orbital_maps[symbol][basis_j]
            full_j = idp.basis_to_full_basis[symbol][basis_j]
            feat = idp.orbpair_maps.get(f"{full_i}-{full_j}")
            if feat is not None:
                yield row, col, feat


def edge_feature_slices(idp: Any, sym_i: str, sym_j: str) -> Iterable[Tuple[slice, slice, slice]]:
    """Yield canonical non-SOC edge slices matching DeePTB feature packing."""
    full_order = {b: i for i, b in enumerate(idp.full_basis)}
    for basis_i in idp.basis[sym_i]:
        row = idp.orbital_maps[sym_i][basis_i]
        full_i = idp.basis_to_full_basis[sym_i][basis_i]
        for basis_j in idp.basis[sym_j]:
            col = idp.orbital_maps[sym_j][basis_j]
            full_j = idp.basis_to_full_basis[sym_j][basis_j]
            if full_order[full_i] > full_order[full_j]:
                continue
            feat = idp.orbpair_maps.get(f"{full_i}-{full_j}")
            if feat is not None:
                yield row, col, feat


def _slice_hw(row: slice, col: slice) -> Tuple[int, int]:
    return int(row.stop - row.start), int(col.stop - col.start)


# -----------------------------------------------------------------------------
# Hermitian reverse-edge metadata and completion
# -----------------------------------------------------------------------------


def reverse_edge_index(data: Mapping[str, Any], *, device=None) -> torch.Tensor:
    """Return reverse edge positions for ``(i,j,R)->(j,i,-R)``; missing is -1."""
    edge_index = edge_index_from_data(data).detach().cpu()
    n_edges = int(edge_index.shape[1])
    shift = edge_shift_from_data(data, n_edges=n_edges).detach().cpu()
    lookup: Dict[Tuple[int, int, int, int, int], int] = {}
    for e, (u, v) in enumerate(edge_index.T.tolist()):
        r0, r1, r2 = [int(x) for x in shift[e].tolist()]
        lookup[(int(u), int(v), r0, r1, r2)] = int(e)
    rev = []
    for e, (u, v) in enumerate(edge_index.T.tolist()):
        r0, r1, r2 = [int(x) for x in shift[e].tolist()]
        rev.append(lookup.get((int(v), int(u), -r0, -r1, -r2), -1))
    return torch.as_tensor(rev, dtype=torch.long, device=device)


def edge_direct_feature_mask(
    data: Mapping[str, Any],
    idp: Any,
    *,
    pad_shape: Optional[Tuple[int, int]] = None,
    device=None,
) -> torch.Tensor:
    """AO entries directly represented by each edge feature vector."""
    ensure_non_soc_mapper(idp)
    symbols = symbols_from_data(data, idp)
    edge_index = edge_index_from_data(data).detach().cpu()
    h = w = mapper_max_norb(idp)
    if pad_shape is not None:
        h, w = int(pad_shape[0]), int(pad_shape[1])
    mask = torch.zeros((edge_index.shape[1], h, w), dtype=torch.bool, device=device)
    for e, (u, v) in enumerate(edge_index.T.tolist()):
        for row, col, _ in edge_feature_slices(idp, symbols[int(u)], symbols[int(v)]):
            mask[e, row, col] = True
    return mask


def complete_edge_blocks_from_reverse(
    data: Mapping[str, Any],
    idp: Any,
    edge_blocks: Optional[torch.Tensor],
    *,
    direct_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Fill missing non-canonical entries using ``H_ij(R)=H_ji(-R)^T``.

    Already represented canonical entries are never overwritten.  The function
    is differentiable with respect to ``edge_blocks``.
    """
    if edge_blocks is None:
        return None, None, reverse_edge_index(data)
    if edge_blocks.numel() == 0:
        return edge_blocks, direct_mask, reverse_edge_index(data, device=edge_blocks.device)
    ensure_non_soc_mapper(idp)
    if direct_mask is None:
        direct_mask = edge_direct_feature_mask(data, idp, pad_shape=tuple(edge_blocks.shape[-2:]), device=edge_blocks.device)
    else:
        direct_mask = direct_mask.to(device=edge_blocks.device, dtype=torch.bool)
    rev = reverse_edge_index(data, device=edge_blocks.device)
    if rev.numel() == 0:
        return edge_blocks, direct_mask, rev
    has_rev = rev >= 0
    safe_rev = rev.clamp_min(0)
    rev_blocks = edge_blocks.index_select(0, safe_rev).transpose(-1, -2)
    if rev_blocks.is_complex():
        rev_blocks = rev_blocks.conj()
    rev_mask = direct_mask.index_select(0, safe_rev).transpose(-1, -2)
    fill = (~direct_mask) & rev_mask & has_rev.view(-1, 1, 1)
    completed = torch.where(fill, rev_blocks, edge_blocks)
    return completed, direct_mask | fill, rev


# -----------------------------------------------------------------------------
# Converters / materializers
# -----------------------------------------------------------------------------


def feature_tensors_to_block_tensors(
    data: Mapping[str, Any],
    idp: Any,
    *,
    node_features: Optional[torch.Tensor] = None,
    edge_features: Optional[torch.Tensor] = None,
    node_pad_shape: Optional[Tuple[int, int]] = None,
    edge_pad_shape: Optional[Tuple[int, int]] = None,
    symmetrize_onsite: bool = True,
    complete_edges: bool = True,
) -> BlockTensorResult:
    """Differentiably materialize non-SOC feature tensors into AO blocks."""
    ensure_non_soc_mapper(idp)
    symbols = symbols_from_data(data, idp)
    default_dim = mapper_max_norb(idp)
    node_pad_shape = tuple(node_pad_shape or (default_dim, default_dim))
    edge_pad_shape = tuple(edge_pad_shape or (default_dim, default_dim))

    node_blocks = node_shapes = None
    if node_features is not None:
        node_features = as_tensor(node_features)
        device = node_features.device
        dtype = node_features.dtype
        n_nodes = len(symbols)
        node_blocks = torch.zeros((n_nodes, node_pad_shape[0], node_pad_shape[1]), dtype=dtype, device=device)
        node_shapes = torch.as_tensor([matrix_shape_for_symbol(idp, s) for s in symbols], dtype=torch.long, device=device)
        by_symbol: Dict[str, List[int]] = {}
        for i, s in enumerate(symbols):
            by_symbol.setdefault(s, []).append(i)
        for symbol, positions in by_symbol.items():
            idx = torch.as_tensor(positions, dtype=torch.long, device=device)
            sub_feat = node_features.index_select(0, idx)
            shape = matrix_shape_for_symbol(idp, symbol)
            sub_block = torch.zeros((idx.numel(), node_pad_shape[0], node_pad_shape[1]), dtype=dtype, device=device)
            for row, col, feat in onsite_feature_slices(idp, symbol):
                h, w = _slice_hw(row, col)
                part = sub_feat[:, feat].reshape(idx.numel(), h, w)
                sub_block[:, row, col] = part
                if symmetrize_onsite and (row.start != col.start or row.stop != col.stop):
                    sub_block[:, col, row] = part.transpose(-1, -2)
            node_blocks.index_copy_(0, idx, sub_block)

    edge_blocks = edge_shapes = None
    if edge_features is not None:
        edge_features = as_tensor(edge_features)
        device = edge_features.device
        dtype = edge_features.dtype
        edge_index = edge_index_from_data(data).detach().cpu()
        n_edges = int(edge_index.shape[1])
        edge_blocks = torch.zeros((n_edges, edge_pad_shape[0], edge_pad_shape[1]), dtype=dtype, device=device)
        edge_shapes_list: List[Tuple[int, int]] = []
        by_pair: Dict[Tuple[str, str], List[int]] = {}
        for e, (u, v) in enumerate(edge_index.T.tolist()):
            pair = (symbols[int(u)], symbols[int(v)])
            by_pair.setdefault(pair, []).append(e)
            edge_shapes_list.append(matrix_shape_for_bond(idp, *pair))
        edge_shapes = torch.as_tensor(edge_shapes_list, dtype=torch.long, device=device)
        for (sym_i, sym_j), positions in by_pair.items():
            idx = torch.as_tensor(positions, dtype=torch.long, device=device)
            sub_feat = edge_features.index_select(0, idx)
            sub_block = torch.zeros((idx.numel(), edge_pad_shape[0], edge_pad_shape[1]), dtype=dtype, device=device)
            for row, col, feat in edge_feature_slices(idp, sym_i, sym_j):
                h, w = _slice_hw(row, col)
                part = sub_feat[:, feat].reshape(idx.numel(), h, w)
                sub_block[:, row, col] = part
            edge_blocks.index_copy_(0, idx, sub_block)
        if complete_edges:
            edge_blocks, _, _ = complete_edge_blocks_from_reverse(data, idp, edge_blocks)

    return BlockTensorResult(node_blocks, edge_blocks, node_shapes, edge_shapes)


def _to_block_tensor(value: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    out = value if torch.is_tensor(value) else torch.as_tensor(value)
    return out.to(dtype=dtype) if dtype is not None else out


def _put_padded(dst: torch.Tensor, index: int, block: torch.Tensor, shape: Tuple[int, int]) -> None:
    rows, cols = shape
    dst[index, :rows, :cols] = block[:rows, :cols].to(device=dst.device, dtype=dst.dtype)


def block_dict_to_ordered_tensors(
    data: Mapping[str, Any],
    idp: Any,
    blocks: Mapping[str, Any],
    *,
    start_id: int = 0,
    node_pad_shape: Optional[Tuple[int, int]] = None,
    edge_pad_shape: Optional[Tuple[int, int]] = None,
    dtype: Optional[torch.dtype] = None,
    missing_edge_policy: str = "error",
    complete_edges: bool = True,
) -> BlockTensorResult:
    """Pack official ``feature_to_block`` dict output into LMDB-ready tensors."""
    ensure_non_soc_mapper(idp)
    symbols = symbols_from_data(data, idp)
    default_dim = mapper_max_norb(idp)
    node_pad_shape = tuple(node_pad_shape or (default_dim, default_dim))
    edge_pad_shape = tuple(edge_pad_shape or (default_dim, default_dim))
    if dtype is None:
        first = _to_block_tensor(next(iter(blocks.values())))
        dtype = torch.complex64 if first.is_complex() else torch.float32

    node_blocks = torch.zeros((len(symbols), node_pad_shape[0], node_pad_shape[1]), dtype=dtype)
    node_shapes = torch.zeros((len(symbols), 2), dtype=torch.long)
    for atom_i, symbol in enumerate(symbols):
        shape = matrix_shape_for_symbol(idp, symbol)
        node_shapes[atom_i] = torch.as_tensor(shape)
        k = block_key(atom_i, atom_i, (0, 0, 0), start_id=start_id)
        if k not in blocks:
            raise KeyError(f"Missing onsite block key {k}")
        _put_padded(node_blocks, atom_i, _to_block_tensor(blocks[k], dtype=dtype), shape)

    edge_index = edge_index_from_data(data).detach().cpu()
    n_edges = int(edge_index.shape[1])
    edge_shift = edge_shift_from_data(data, n_edges=n_edges).detach().cpu()
    edge_blocks = torch.zeros((n_edges, edge_pad_shape[0], edge_pad_shape[1]), dtype=dtype)
    edge_shapes = torch.zeros((n_edges, 2), dtype=torch.long)
    for e, (u, v) in enumerate(edge_index.T.tolist()):
        shift = [int(x) for x in edge_shift[e].tolist()]
        shape = matrix_shape_for_bond(idp, symbols[int(u)], symbols[int(v)])
        edge_shapes[e] = torch.as_tensor(shape)
        direct = block_key(int(u), int(v), shift, start_id=start_id)
        reverse = block_key(int(v), int(u), (-shift[0], -shift[1], -shift[2]), start_id=start_id)
        if direct in blocks:
            block = _to_block_tensor(blocks[direct], dtype=dtype)
        elif reverse in blocks:
            block = _to_block_tensor(blocks[reverse], dtype=dtype).transpose(-1, -2)
            if block.is_complex():
                block = block.conj()
        elif missing_edge_policy == "zero":
            block = torch.zeros(shape, dtype=dtype)
        else:
            raise KeyError(f"Missing edge block key {direct} and reverse {reverse}")
        _put_padded(edge_blocks, e, block, shape)
    if complete_edges:
        edge_blocks, _, _ = complete_edge_blocks_from_reverse(data, idp, edge_blocks)
    return BlockTensorResult(node_blocks, edge_blocks, node_shapes, edge_shapes)


def attach_block_tensors(result: MutableMapping[str, Any], packed: BlockTensorResult, *, prefix: str) -> MutableMapping[str, Any]:
    p = prefix.lower()
    if p in {"delta", "delta_h", "delta_hamil", "target"}:
        node_key, edge_key = NODE_DELTA_HAMIL_BLOCKS_KEY, EDGE_DELTA_HAMIL_BLOCKS_KEY
        node_shape_key, edge_shape_key = NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY, EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY
    elif p in {"h0", "h_0"}:
        node_key, edge_key = NODE_H0_BLOCKS_KEY, EDGE_H0_BLOCKS_KEY
        node_shape_key, edge_shape_key = NODE_H0_BLOCK_SHAPE_KEY, EDGE_H0_BLOCK_SHAPE_KEY
    else:
        node_key, edge_key = f"node_{prefix}_blocks", f"edge_{prefix}_blocks"
        node_shape_key, edge_shape_key = f"node_{prefix}_block_shape", f"edge_{prefix}_block_shape"
    if packed.node_blocks is not None:
        result[node_key] = packed.node_blocks
        result[node_shape_key] = packed.node_shapes
    if packed.edge_blocks is not None:
        result[edge_key] = packed.edge_blocks
        result[edge_shape_key] = packed.edge_shapes
    return result


def attach_prediction_block_tensors(
    result: MutableMapping[str, Any],
    packed: BlockTensorResult,
    *,
    node_key: str = NODE_PRED_HAMIL_BLOCKS_KEY,
    edge_key: str = EDGE_PRED_HAMIL_BLOCKS_KEY,
    node_shape_key: str = "node_hamil_block_shape",
    edge_shape_key: str = "edge_hamil_block_shape",
) -> MutableMapping[str, Any]:
    if packed.node_blocks is not None:
        result[node_key] = packed.node_blocks
        result[node_shape_key] = packed.node_shapes
    if packed.edge_blocks is not None:
        result[edge_key] = packed.edge_blocks
        result[edge_shape_key] = packed.edge_shapes
    return result


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def value_components(values: torch.Tensor, *, complex_reduction: str = "modulus") -> ComponentSums:
    if values.numel() == 0:
        return _zero_component_like(values)
    if values.is_complex() and complex_reduction == "real_imag":
        abs_sum = values.real.abs().sum() + values.imag.abs().sum()
        square_sum = values.real.square().sum() + values.imag.square().sum()
        count = torch.as_tensor(float(values.numel() * 2), dtype=values.real.dtype, device=values.device)
    else:
        mag = values.abs() if values.is_complex() else values.abs()
        abs_sum = mag.sum()
        square_sum = mag.square().sum()
        count = torch.as_tensor(float(values.numel()), dtype=mag.dtype, device=values.device)
    return ComponentSums(abs_sum, square_sum, count)


def add_components(a: ComponentSums, b: ComponentSums) -> ComponentSums:
    return ComponentSums(a.abs_sum + b.abs_sum, a.square_sum + b.square_sum, a.count + b.count)


def block_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    shapes: torch.Tensor,
    *,
    complex_reduction: str = "modulus",
) -> ComponentSums:
    target = target.to(device=pred.device, dtype=pred.dtype)
    mask = block_mask_from_shapes(shapes.to(device=pred.device), tuple(pred.shape[-2:]))
    diff = pred - target
    if diff.is_complex() and complex_reduction == "real_imag":
        abs_term = diff.real.abs() + diff.imag.abs()
        square_term = diff.real.square() + diff.imag.square()
        count_factor = 2.0
        dtype = diff.real.dtype
    else:
        abs_term = diff.abs() if diff.is_complex() else diff.abs()
        square_term = abs_term.square()
        count_factor = 1.0
        dtype = abs_term.dtype
    return ComponentSums(
        abs_term[mask].sum(),
        square_term[mask].sum(),
        mask.sum().to(device=pred.device, dtype=dtype) * count_factor,
    )


def feature_components_from_blocks(
    data: Mapping[str, Any],
    idp: Any,
    pred_node_blocks: Optional[torch.Tensor],
    target_node_blocks: Optional[torch.Tensor],
    pred_edge_blocks: Optional[torch.Tensor],
    target_edge_blocks: Optional[torch.Tensor],
    *,
    complex_reduction: str = "modulus",
) -> Tuple[ComponentSums, ComponentSums]:
    """Old hamil_abs-compatible components from AO-block diffs."""
    ensure_non_soc_mapper(idp)
    symbols = symbols_from_data(data, idp)
    device = None
    if pred_node_blocks is not None:
        device = pred_node_blocks.device
    elif pred_edge_blocks is not None:
        device = pred_edge_blocks.device
    node_sum = _zero_component_like(pred_node_blocks, device=device)
    edge_sum = _zero_component_like(pred_edge_blocks, device=device)

    if pred_node_blocks is not None and target_node_blocks is not None:
        diff = pred_node_blocks - target_node_blocks.to(device=pred_node_blocks.device, dtype=pred_node_blocks.dtype)
        for symbol in sorted(set(symbols)):
            idx = torch.as_tensor([i for i, s in enumerate(symbols) if s == symbol], dtype=torch.long, device=diff.device)
            sub = diff.index_select(0, idx)
            for row, col, _ in onsite_feature_slices(idp, symbol):
                node_sum = add_components(node_sum, value_components(sub[:, row, col], complex_reduction=complex_reduction))

    if pred_edge_blocks is not None and target_edge_blocks is not None:
        edge_index = edge_index_from_data(data).detach().cpu()
        if edge_index.shape[1] > 0:
            diff = pred_edge_blocks - target_edge_blocks.to(device=pred_edge_blocks.device, dtype=pred_edge_blocks.dtype)
            by_pair: Dict[Tuple[str, str], List[int]] = {}
            for e, (u, v) in enumerate(edge_index.T.tolist()):
                by_pair.setdefault((symbols[int(u)], symbols[int(v)]), []).append(e)
            for (sym_i, sym_j), positions in by_pair.items():
                idx = torch.as_tensor(positions, dtype=torch.long, device=diff.device)
                sub = diff.index_select(0, idx)
                for row, col, _ in edge_feature_slices(idp, sym_i, sym_j):
                    edge_sum = add_components(edge_sum, value_components(sub[:, row, col], complex_reduction=complex_reduction))
    return node_sum, edge_sum


# Backward-compatible alias used by earlier package/tests.
feature_compatible_components_from_blocks = feature_components_from_blocks


def mae_from_components(comp: ComponentSums) -> torch.Tensor:
    return comp.abs_sum / comp.count.clamp_min(1.0)


def l1_rmse_from_components(comp: ComponentSums, *, eps: float = 1e-12) -> torch.Tensor:
    safe = comp.count.clamp_min(1.0)
    valid = (comp.count > 0.5).to(dtype=comp.abs_sum.dtype, device=comp.abs_sum.device)
    return 0.5 * (comp.abs_sum / safe + torch.sqrt(comp.square_sum / safe + eps)) * valid

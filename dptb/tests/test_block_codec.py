"""Block <-> coupled-RME codec, block-state projector and strict graph topology checks."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_mask_from_shapes,
    canonical_block_tensors_to_feature_tensors,
    feature_tensors_to_block_tensors,
    infer_block_shapes,
    strict_reverse_edge_index,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nn.hamiltonian import E3Hamiltonian, _contract_cg_rme, _inverse_contract_cg_hr
from dptb.nnops.block_flow_codec import BlockStateCodec, _projector_topology_fingerprint, project_block_state
from dptb.tests.block_ode_fixtures import FP64_ATOL, _case, _uureal_mapper

FP32_CODEC_ATOL = 2.0e-5
SUPPORTED_L = ("s", "p", "d", "f", "g", "h")
SUPPORTED_PAIRTYPES = tuple(
    f"{left}-{right}" for index, left in enumerate(SUPPORTED_L) for right in SUPPORTED_L[index:]
)


def _tol(x, dtype):
    if dtype == torch.float64:
        return max(FP64_ATOL, 1e-12 * float(x.abs().max().item()))
    return max(2e-5, 2e-5 * float(x.abs().max().item()))


def _single_atom_reverse_pair(idp, dtype):
    """One atom with a periodic self-image edge pair along x."""
    edge_type = idp.bond_to_type["H-H"]
    return {
        "pos": torch.zeros((1, 3), dtype=dtype),
        "cell": torch.eye(3, dtype=dtype).unsqueeze(0),
        "pbc": torch.tensor([True, False, False]),
        "batch": torch.zeros((1,), dtype=torch.long),
        "atom_types": torch.tensor([idp.chemical_symbol_to_type["H"]], dtype=torch.long),
        "edge_type": torch.tensor([edge_type, edge_type], dtype=torch.long),
        "edge_index": torch.tensor([[0, 0], [0, 0]], dtype=torch.long),
        "edge_cell_shift": torch.tensor([[1, 0, 0], [-1, 0, 0]], dtype=dtype),
    }


def _mixed_case(dtype=torch.float64):
    """Two periodic H 1s / C 2p dimers: C's compact p shell starts at 0 while the union canvas starts with s."""
    idp = OrbitalMapper({"H": ["1s"], "C": ["2p"]}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    t_h, t_c = idp.chemical_symbol_to_type["H"], idp.chemical_symbol_to_type["C"]
    data = {
        "pos": torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 1.0, 0.0]], dtype=dtype),
        "cell": torch.stack([torch.eye(3, dtype=dtype) * 5.0] * 2),
        "pbc": torch.tensor([[True, False, False], [False, True, False]]),
        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
        "edge_cell_shift": torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=dtype),
        "atom_types": torch.tensor([t_h, t_c, t_h, t_c], dtype=torch.long),
        "edge_type": torch.tensor(
            [idp.bond_to_type["H-C"], idp.bond_to_type["C-H"], idp.bond_to_type["H-C"], idp.bond_to_type["C-H"]],
            dtype=torch.long,
        ),
    }
    return idp, data


def _middle_shell_case(dtype=torch.float64):
    """Si 3s3d / C 2p dimer: Si's compact frame skips the union p shell between its s and d."""
    idp = OrbitalMapper({"Si": ["3s", "3d"], "C": ["2p"]}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    idp.get_irreps(no_parity=False)
    data = {
        "pos": torch.zeros(2, 3, dtype=dtype),
        "atom_types": torch.tensor([idp.chemical_symbol_to_type["Si"], idp.chemical_symbol_to_type["C"]]),
        "edge_type": torch.tensor([idp.bond_to_type["Si-C"], idp.bond_to_type["C-Si"]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_cell_shift": torch.zeros(2, 3, dtype=dtype),
    }
    return idp, data


def _masked_random_rme(idp, data, dtype):
    generator = torch.Generator().manual_seed(718)
    width = idp.reduced_matrix_element
    node = torch.randn(len(data["atom_types"]), width, generator=generator, dtype=dtype)
    edge = torch.randn(data["edge_index"].shape[1], width, generator=generator, dtype=dtype)
    return node * idp.mask_to_nrme[data["atom_types"]].to(dtype), edge * idp.mask_to_erme[data["edge_type"]].to(dtype)


def _random_state(dtype=torch.float64, requires_grad=False, seed=718):
    """An arbitrary (off-image) block state shaped for ``_mixed_case``."""
    generator = torch.Generator().manual_seed(seed)
    return BlockTensorResult(
        torch.randn(4, 3, 3, generator=generator, dtype=dtype, requires_grad=requires_grad),
        torch.randn(4, 3, 3, generator=generator, dtype=dtype, requires_grad=requires_grad),
        torch.tensor([[1, 1], [3, 3], [1, 1], [3, 3]], dtype=torch.long),
        torch.tensor([[1, 3], [3, 1], [1, 3], [3, 1]], dtype=torch.long),
    )


def _padding_is_zero(state):
    node_mask = block_mask_from_shapes(state.node_shapes, tuple(state.node_blocks.shape[-2:]))
    edge_mask = block_mask_from_shapes(state.edge_shapes, tuple(state.edge_blocks.shape[-2:]))
    return (
        torch.count_nonzero(state.node_blocks[~node_mask]) == 0
        and torch.count_nonzero(state.edge_blocks[~edge_mask]) == 0
    )


# ---------------------------------------------------------------------------
# Clebsch-Gordan bases and the inverse contraction
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def all_l_mapper():
    idp = OrbitalMapper({"H": ["1s", "2p", "3d", "4f", "5g", "6h"]}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    return idp


@pytest.fixture(scope="module")
def all_l_cgbasis(all_l_mapper):
    bases = {dtype: E3Hamiltonian(idp=all_l_mapper, dtype=dtype).cgbasis for dtype in (torch.float32, torch.float64)}
    assert set(bases[torch.float32]) == set(SUPPORTED_PAIRTYPES)
    return bases


@pytest.mark.parametrize("pairtype", SUPPORTED_PAIRTYPES)
def test_cg_basis_is_orthogonal_and_inverse_cg_recovers_every_coupled_coordinate(all_l_cgbasis, pairtype):
    """fp64: each s..h pair-type basis is square, full rank and orthogonal; fp32: the inverse CG maps
    every product-space basis column (built by hand, not by the forward helper) back to its coordinate."""
    basis64 = all_l_cgbasis[torch.float64][pairtype]
    matrix = basis64.reshape(-1, basis64.shape[-1])
    size = matrix.shape[0]
    assert matrix.shape == (size, size)
    identity = torch.eye(size, dtype=torch.float64)
    assert torch.linalg.matrix_rank(matrix, atol=1e-12, rtol=1e-12) == size
    assert (matrix.T @ matrix - identity).abs().max().item() <= FP64_ATOL
    assert (matrix @ matrix.T - identity).abs().max().item() <= FP64_ATOL
    assert torch.linalg.cond(matrix).item() <= 1.0 + FP64_ATOL

    basis32 = all_l_cgbasis[torch.float32][pairtype]
    product = basis32.reshape(-1, size).transpose(0, 1).reshape(size, 1, basis32.shape[0], basis32.shape[1])
    restored = _inverse_contract_cg_hr(basis32, product)
    torch.testing.assert_close(restored.squeeze(-1), torch.eye(size, dtype=torch.float32), rtol=0.0, atol=2.0e-6)


def test_inverse_cg_solves_nonorthogonal_bases_and_rejects_incomplete_ones():
    nonorthogonal = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]], dtype=torch.float64)
    rme = torch.tensor([[[1.25], [-0.75]]], dtype=torch.float64)
    restored = _inverse_contract_cg_hr(nonorthogonal, _contract_cg_rme(nonorthogonal, rme))
    torch.testing.assert_close(restored, rme, rtol=0.0, atol=1e-12)

    rectangular = torch.zeros((2, 3, 2), dtype=torch.float64)
    with pytest.raises(RuntimeError):
        _inverse_contract_cg_hr(rectangular, torch.zeros((1, 1, 2, 3), dtype=torch.float64))
    singular = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    with pytest.raises(RuntimeError):
        _inverse_contract_cg_hr(singular, torch.ones((1, 1, 1, 2), dtype=torch.float64))


# ---------------------------------------------------------------------------
# Pack / gather slot alignment and codec round trips
# ---------------------------------------------------------------------------
def test_repeated_shell_slots_are_packed_and_gathered_in_mapper_order():
    """Sentinel features cross repeated s/s and p/p chunks; both the direct write and the transpose
    completion of the reverse edge land in the mapper's slots, and the strict gather inverts the pack."""
    idp = OrbitalMapper({"H": "3s2p1d"}, method="e3tb", device="cpu")
    idp.get_orbital_maps()
    data = _single_atom_reverse_pair(idp, torch.float64)
    width = int(idp.reduced_matrix_element)
    node_features = torch.arange(1, width + 1, dtype=torch.float64).unsqueeze(0)
    edge_features = torch.stack(
        (torch.arange(1001, 1001 + width, dtype=torch.float64), torch.arange(2001, 2001 + width, dtype=torch.float64))
    )
    packed = feature_tensors_to_block_tensors(
        data,
        idp,
        node_features=node_features,
        edge_features=edge_features,
        symmetrize_onsite=True,
        complete_edges=True,
        strict_complete_edges=True,
    )
    exact = dict(rtol=0.0, atol=0.0)
    for shell_i, shell_j in (("1s", "2s"), ("2s", "3s"), ("3s", "1p"), ("1p", "2p"), ("2p", "1d")):
        row, col = idp.orbital_maps["H"][shell_i], idp.orbital_maps["H"][shell_j]
        feature = idp.orbpair_maps[f"{shell_i}-{shell_j}"]
        shape = (row.stop - row.start, col.stop - col.start)
        expected_node = node_features[0, feature].reshape(shape)
        expected_edge = [edge_features[k, feature].reshape(shape) for k in (0, 1)]
        torch.testing.assert_close(packed.node_blocks[0, row, col], expected_node, **exact)
        torch.testing.assert_close(packed.node_blocks[0, col, row], expected_node.T, **exact)
        torch.testing.assert_close(packed.edge_blocks[0, row, col], expected_edge[0], **exact)
        torch.testing.assert_close(packed.edge_blocks[1, row, col], expected_edge[1], **exact)
        torch.testing.assert_close(packed.edge_blocks[0, col, row], expected_edge[1].T, **exact)
        torch.testing.assert_close(packed.edge_blocks[1, col, row], expected_edge[0].T, **exact)

    gathered = canonical_block_tensors_to_feature_tensors(
        data,
        idp,
        node_blocks=packed.node_blocks,
        edge_blocks=packed.edge_blocks,
        node_shapes=packed.node_shapes,
        edge_shapes=packed.edge_shapes,
        mode="strict",
        atol=0.0,
    )
    torch.testing.assert_close(gathered.node_features, node_features, **exact)
    torch.testing.assert_close(gathered.edge_features, edge_features, **exact)


def test_s_through_h_float32_codec_round_trip(all_l_mapper):
    idp = all_l_mapper
    data = _single_atom_reverse_pair(idp, torch.float32)
    codec = BlockStateCodec(idp, dtype=torch.float32, inverse_mode="strict", atol=FP32_CODEC_ATOL)
    norb = int(idp.norbs["H"])
    generator = torch.Generator().manual_seed(20260718)
    raw_node = torch.randn(norb, norb, dtype=torch.float32, generator=generator)
    node = (0.5 * (raw_node + raw_node.T)).unsqueeze(0)
    edge_0 = torch.randn(norb, norb, dtype=torch.float32, generator=generator)
    edge = torch.stack((edge_0, edge_0.T))
    state = BlockTensorResult(
        node, edge, torch.tensor([[norb, norb]]), torch.tensor([[norb, norb], [norb, norb]])
    )
    rebuilt = codec.rme_to_blocks(data, *codec.blocks_to_rme(data, state))
    torch.testing.assert_close(rebuilt.node_blocks, node, rtol=0.0, atol=FP32_CODEC_ATOL)
    torch.testing.assert_close(rebuilt.edge_blocks, edge, rtol=0.0, atol=FP32_CODEC_ATOL)


_SPECIES_COMPACT_CASES = {
    # builder, union basis, (species, shell, compact start), directed edge shapes
    "nonprefix": (_mixed_case, ["1s", "1p"], ("C", "2p", 0), [[1, 3], [3, 1], [1, 3], [3, 1]]),
    "middle_shell": (_middle_shell_case, ["1s", "1p", "1d"], ("Si", "3d", 1), [[6, 3], [3, 6]]),
}


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("case", sorted(_SPECIES_COMPACT_CASES))
def test_species_compact_codec_round_trip(case, dtype):
    """Species-compact blocks and canonical RME are mutual inverses when a species' shells are not a
    prefix of the union basis (the canonical edge coordinates may be split over both directed rows);
    rectangular edge blocks keep zero padding."""
    build, union_basis, (species, shell, start), edge_shapes = _SPECIES_COMPACT_CASES[case]
    idp, data = build(dtype)
    assert idp.full_basis == union_basis
    assert idp.orbital_maps[species][shell].start == start
    codec = BlockStateCodec(idp, dtype=dtype)
    node_in, edge_in = _masked_random_rme(idp, data, dtype)
    first = codec.rme_to_blocks(data, node_in, edge_in)
    node, edge = codec.blocks_to_rme(data, first)
    assert (node - node_in).abs().max().item() <= _tol(node_in, dtype)  # every masked onsite coordinate survives
    blocks = codec.rme_to_blocks(data, node, edge)
    node2, edge2 = codec.blocks_to_rme(data, blocks)
    assert (blocks.node_blocks - first.node_blocks).abs().max().item() <= _tol(first.node_blocks, dtype)
    assert (blocks.edge_blocks - first.edge_blocks).abs().max().item() <= _tol(first.edge_blocks, dtype)
    assert (node2 - node).abs().max().item() <= _tol(node, dtype)
    assert (edge2 - edge).abs().max().item() <= _tol(edge, dtype)
    assert blocks.edge_shapes.tolist() == edge_shapes
    assert _padding_is_zero(blocks)


def test_independently_projected_physical_blocks_round_trip():
    idp, data = _mixed_case()
    codec = BlockStateCodec(idp, dtype=torch.float64)
    allowed = project_block_state(data, idp, _random_state())
    rebuilt = codec.rme_to_blocks(data, *codec.blocks_to_rme(data, allowed))
    assert (rebuilt.node_blocks - allowed.node_blocks).abs().max().item() <= FP64_ATOL
    assert (rebuilt.edge_blocks - allowed.edge_blocks).abs().max().item() <= FP64_ATOL


def test_codec_ignores_non_tensor_trainer_batch_metadata():
    _, data, codec, h0 = _case()
    with_metadata = {
        **data,
        "__slices__": {"edge_index": [0, int(data["edge_index"].shape[1])]},
        "__data_class__": object,
    }
    packed = codec.rme_to_blocks(with_metadata, data["node_h0"], data["edge_h0"], project=True)
    node_rme, edge_rme = codec.blocks_to_rme(with_metadata, packed)
    torch.testing.assert_close(node_rme, data["node_h0"], rtol=0, atol=FP64_ATOL)
    torch.testing.assert_close(edge_rme, data["edge_h0"], rtol=0, atol=FP64_ATOL)
    torch.testing.assert_close(packed.node_blocks, h0.node_blocks, rtol=0, atol=FP64_ATOL)
    torch.testing.assert_close(packed.edge_blocks, h0.edge_blocks, rtol=0, atol=FP64_ATOL)


def test_strict_gather_rejects_off_image_blocks_and_project_mode_reports_residual():
    idp = OrbitalMapper({"H": ["1s"], "C": ["2s", "2p"]}, method="e3tb", device="cpu")
    idp.get_irreps(no_parity=False)
    data = {
        "atom_types": torch.tensor([idp.chemical_symbol_to_type["H"], idp.chemical_symbol_to_type["C"]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_cell_shift": torch.zeros(2, 3),
    }
    shapes = dict(node_shapes=torch.tensor([[1, 1], [4, 4]]), edge_shapes=torch.tensor([[1, 4], [4, 1]]))
    node = torch.zeros(2, 4, 4, dtype=torch.float64)
    edge = torch.zeros(2, 4, 4, dtype=torch.float64)
    node[1, 1, 0] = 1e-3  # non-canonical transpose side
    node[0, 3, 3] = 1e-3  # padding
    with pytest.raises(ValueError, match="packer image"):
        canonical_block_tensors_to_feature_tensors(
            data, idp, node_blocks=node, edge_blocks=edge, mode="strict", atol=FP64_ATOL, **shapes
        )
    result = canonical_block_tensors_to_feature_tensors(
        data, idp, node_blocks=node, edge_blocks=edge, mode="project", atol=FP64_ATOL, **shapes
    )
    assert result.node_projection_residual.max().item() >= 2.5e-4
    assert result.projected_node_blocks[0, 3, 3].item() == 0.0
    zeros = dict(node_blocks=torch.zeros_like(node), edge_blocks=torch.zeros_like(edge))
    with pytest.raises(ValueError):
        canonical_block_tensors_to_feature_tensors(data, idp, atol=float("nan"), **zeros, **shapes)


def test_codec_certification_skip_requires_the_internal_token():
    _, data, codec, h0 = _case()
    with pytest.raises(RuntimeError):
        codec.blocks_to_rme(data, h0, certify_image=False)


def test_exact_codec_rejects_compact_uureal_mapper():
    with pytest.raises(NotImplementedError):
        BlockStateCodec(_uureal_mapper(), dtype=torch.float64)


# ---------------------------------------------------------------------------
# Block-state projector
# ---------------------------------------------------------------------------
def test_projector_is_idempotent_and_enforces_all_invariants():
    idp, data = _mixed_case()
    raw = _random_state()
    once = project_block_state(data, idp, raw)
    twice = project_block_state(data, idp, once)
    assert max(
        (once.node_blocks - raw.node_blocks).abs().max().item(),
        (once.edge_blocks - raw.edge_blocks).abs().max().item(),
    ) > 1e-3
    assert (twice.node_blocks - once.node_blocks).abs().max().item() <= FP64_ATOL
    assert (twice.edge_blocks - once.edge_blocks).abs().max().item() <= FP64_ATOL
    assert (once.node_blocks - once.node_blocks.transpose(-1, -2)).abs().max().item() <= FP64_ATOL
    rev = strict_reverse_edge_index(data)
    assert (once.edge_blocks - once.edge_blocks.index_select(0, rev).transpose(-1, -2)).abs().max().item() <= FP64_ATOL
    assert _padding_is_zero(once)

    fractional = BlockTensorResult(
        raw.node_blocks, raw.edge_blocks, raw.node_shapes.to(torch.float64) + 0.5, raw.edge_shapes.to(torch.float64) + 0.5
    )
    with pytest.raises(ValueError, match="shapes"):
        project_block_state(data, idp, fractional)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_projector_is_self_adjoint_and_fixes_every_invariant_state(dtype):
    """<P x, y> == <x, P y>, and a hand-built state with symmetric onsite blocks, transpose-paired
    reverse edges and zero padding is left unchanged: P is the orthogonal projector onto that subspace."""
    idp, data = _mixed_case(dtype)
    x = _random_state(dtype, seed=1)
    y = _random_state(dtype, seed=2)
    px = project_block_state(data, idp, x)
    py = project_block_state(data, idp, y)
    lhs = (px.node_blocks * y.node_blocks).sum() + (px.edge_blocks * y.edge_blocks).sum()
    rhs = (x.node_blocks * py.node_blocks).sum() + (x.edge_blocks * py.edge_blocks).sum()
    assert (lhs - rhs).abs().item() <= _tol(lhs, dtype)

    node_mask = block_mask_from_shapes(x.node_shapes, (3, 3)).to(dtype)
    edge_mask = block_mask_from_shapes(x.edge_shapes, (3, 3)).to(dtype)
    node = (x.node_blocks + x.node_blocks.transpose(-1, -2)) * node_mask
    forward = y.edge_blocks[0::2] * edge_mask[0::2]
    edge = torch.stack((forward[0], forward[0].T, forward[1], forward[1].T))
    invariant = BlockTensorResult(node, edge, x.node_shapes, x.edge_shapes)
    fixed = project_block_state(data, idp, invariant)
    assert (fixed.node_blocks - node).abs().max().item() <= _tol(node, dtype)
    assert (fixed.edge_blocks - edge).abs().max().item() <= _tol(edge, dtype)


def test_projector_topology_cache_is_content_addressed():
    """A cached projection of one topology does not serve a same-shaped graph with different edges."""
    idp, data = _mixed_case()
    state = _random_state()
    project_block_state(data, idp, state)
    cloned = {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}
    project_block_state(cloned, idp, state)
    cloned["edge_index"] = cloned["edge_index"].clone()
    cloned["edge_index"][0, 0] = 1
    with pytest.raises(ValueError):
        project_block_state(cloned, idp, state)

    # Large numpy topology values are hashed by content, not by their (elided) repr.
    base_shift = np.zeros((1200, 3), dtype=np.float32)
    variant_shift = base_shift.copy()
    variant_shift[600, 1] = 1.0
    assert repr(base_shift) == repr(variant_shift)
    fingerprint = _projector_topology_fingerprint
    assert fingerprint({"edge_cell_shift": base_shift}, idp) != fingerprint({"edge_cell_shift": variant_shift}, idp)
    assert fingerprint({"edge_cell_shift": base_shift}, idp) == fingerprint({"edge_cell_shift": base_shift.copy()}, idp)


def test_projector_gradcheck():
    idp, data = _mixed_case()
    state = _random_state(requires_grad=True)

    def fn(node, edge):
        projected = project_block_state(
            data, idp, BlockTensorResult(node, edge, state.node_shapes, state.edge_shapes)
        )
        return projected.node_blocks, projected.edge_blocks

    assert torch.autograd.gradcheck(fn, (state.node_blocks, state.edge_blocks), eps=1e-6, atol=1e-5)


# ---------------------------------------------------------------------------
# Strict reverse-edge topology
# ---------------------------------------------------------------------------
def test_strict_reverse_edge_index_pairs_batched_edges_and_rejects_missing_or_duplicate():
    _, data = _mixed_case()
    assert strict_reverse_edge_index(data).tolist() == [1, 0, 3, 2]

    missing = dict(data)
    missing["edge_index"] = data["edge_index"][:, :3]
    missing["edge_cell_shift"] = data["edge_cell_shift"][:3]
    missing["edge_type"] = data["edge_type"][:3]
    with pytest.raises(ValueError, match="reverse"):
        strict_reverse_edge_index(missing)

    duplicate = dict(data)
    duplicate["edge_index"] = torch.cat([data["edge_index"], data["edge_index"][:, :1]], dim=1)
    duplicate["edge_cell_shift"] = torch.cat([data["edge_cell_shift"], data["edge_cell_shift"][:1]])
    duplicate["edge_type"] = torch.cat([data["edge_type"], data["edge_type"][:1]])
    with pytest.raises(ValueError, match="Duplicate"):
        strict_reverse_edge_index(duplicate)


def _graph_mapper():
    mapper = OrbitalMapper({"H": ["1s"], "C": ["2s"]}, method="e3tb", device="cpu")
    mapper.get_orbital_maps()
    mapper.get_irreps(no_parity=False)
    return mapper


def _two_atom_graph(*, periodic=False):
    return {
        _keys.ATOMIC_NUMBERS_KEY: torch.tensor([1, 6], dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor([0, 0], dtype=torch.long),
        _keys.CELL_KEY: torch.eye(3).unsqueeze(0),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.tensor(
            [[1, 0, 0], [-1, 0, 0]] if periodic else [[0, 0, 0], [0, 0, 0]], dtype=torch.long
        ),
        _keys.PBC_KEY: torch.tensor([periodic, False, False]),
    }


def _with(**updates):
    def mutate(data):
        for key, value in updates.items():
            if value is None:
                data.pop(key)
            else:
                data[key] = value
    return mutate


_CORRUPT_GRAPHS = {
    # name: (periodic, mutation, offending key)
    "fractional_edge_index": (False, _with(edge_index=torch.tensor([[0.5, 1.0], [1.0, 0.5]])), "edge_index"),
    "nan_edge_index": (False, _with(edge_index=torch.tensor([[float("nan"), 1.0], [1.0, 0.0]])), "edge_index"),
    "negative_edge_index": (False, _with(edge_index=torch.tensor([[-1, 0], [0, -1]])), "edge_index"),
    "out_of_range_edge_index": (False, _with(edge_index=torch.tensor([[0, 2], [2, 0]])), "edge_index"),
    "periodic_without_shift": (True, _with(edge_cell_shift=None), "edge_cell_shift"),
    "shift_on_nonperiodic_axis": (True, _with(edge_cell_shift=torch.tensor([[0, 1, 0], [0, -1, 0]])), "edge_cell_shift"),
    "uint64_shift_overflow": (
        False, _with(edge_cell_shift=torch.tensor([[2**64 - 1, 0, 0], [0, 0, 0]], dtype=torch.uint64)), "edge_cell_shift"
    ),
    "periodic_without_cell": (True, _with(cell=None), "cell"),
    "periodic_without_pbc": (True, _with(pbc=None), "pbc"),
    "cross_batch_edge": (
        False, _with(batch=torch.tensor([0, 1]), pbc=torch.zeros((2, 3), dtype=torch.bool)), "batch"
    ),
    "zero_row_pbc": (False, _with(pbc=torch.empty((0, 3), dtype=torch.bool)), "pbc"),
    "zero_row_pbc_no_edges": (
        False,
        _with(
            pbc=torch.empty((0, 3), dtype=torch.bool),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_cell_shift=torch.empty((0, 3), dtype=torch.long),
        ),
        "pbc",
    ),
}


@pytest.mark.parametrize("name", sorted(_CORRUPT_GRAPHS))
def test_strict_reverse_edge_index_rejects_corrupt_graph(name):
    periodic, mutate, key = _CORRUPT_GRAPHS[name]
    data = _two_atom_graph(periodic=periodic)
    mutate(data)
    with pytest.raises(ValueError, match=key):
        strict_reverse_edge_index(data)


def test_strict_reverse_edge_index_validates_edge_type_against_mapper():
    mapper = _graph_mapper()
    data = _two_atom_graph()
    data[_keys.EDGE_TYPE_KEY] = torch.tensor([mapper.bond_to_type["H-C"], mapper.bond_to_type["C-H"]])
    assert strict_reverse_edge_index(data, idp=mapper).tolist() == [1, 0]
    for bad in (data[_keys.EDGE_TYPE_KEY].flip(0), torch.tensor([0.5, 1.0])):
        data[_keys.EDGE_TYPE_KEY] = bad
        with pytest.raises(ValueError, match="edge_type"):
            strict_reverse_edge_index(data, idp=mapper)


def test_empty_edge_graph_keeps_rank_two_shapes_through_strict_codec():
    mapper = _graph_mapper()
    width = int(mapper.reduced_matrix_element)
    data = {
        _keys.ATOMIC_NUMBERS_KEY: torch.tensor([1], dtype=torch.long),
        _keys.BATCH_KEY: torch.tensor([0], dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.empty((2, 0), dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.empty((0, 3), dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.empty((0,), dtype=torch.long),
        _keys.PBC_KEY: torch.tensor([False, False, False]),
    }
    node_shapes, edge_shapes = infer_block_shapes(data, mapper)
    assert node_shapes.shape == (1, 2)
    assert edge_shapes.shape == (0, 2)
    assert strict_reverse_edge_index(data, idp=mapper).shape == (0,)
    packed = feature_tensors_to_block_tensors(
        data,
        mapper,
        node_features=torch.zeros((1, width), dtype=torch.float64),
        edge_features=torch.zeros((0, width), dtype=torch.float64),
        strict_complete_edges=True,
    )
    assert packed.edge_blocks.shape[0] == 0
    assert packed.edge_shapes.shape == (0, 2)
    inverse = canonical_block_tensors_to_feature_tensors(
        data,
        mapper,
        node_blocks=packed.node_blocks,
        edge_blocks=packed.edge_blocks,
        node_shapes=packed.node_shapes,
        edge_shapes=packed.edge_shapes,
        mode="strict",
        atol=FP64_ATOL,
    )
    assert inverse.edge_features.shape == (0, width)
    assert inverse.edge_projection_residual.shape == (0,)


def test_with_edge_vectors_accepts_integer_cell_shifts_with_float_cell():
    data = {
        _keys.POSITIONS_KEY: torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.5, 0.0]], dtype=torch.float64),
        _keys.CELL_KEY: torch.eye(3, dtype=torch.float64) * 2.0,
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.EDGE_CELL_SHIFT_KEY: torch.tensor([[1, 0, 0], [-1, 0, 0]], dtype=torch.long),
    }
    out = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
    assert out[_keys.EDGE_VECTORS_KEY].dtype == torch.float64
    assert out[_keys.EDGE_LENGTH_KEY].dtype == torch.float64
    torch.testing.assert_close(
        out[_keys.EDGE_VECTORS_KEY], torch.tensor([[2.25, 0.5, 0.0], [-2.25, -0.5, 0.0]], dtype=torch.float64)
    )

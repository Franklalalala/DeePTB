from __future__ import annotations

import copy

import pytest
import torch
from e3nn import o3

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import (
    BlockTensorResult,
    block_tensors_to_feature_tensors,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.lem_moe_v3_h0_helpers import DirectUuRealBlockProjector
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.nnops.block_flow_codec import BlockStateCodec
from dptb.nnops.flow import HamiltonianCFM
from dptb.utils.argcheck import validate_block_ode_contract


def _mapper():
    mapper = OrbitalMapper(
        {"H": "1s", "C": "1s1p"},
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=True,
        full_soc_prediction=False,
    )
    mapper.get_irreps()
    return mapper


def _uureal_mapper(basis, *, dtype=torch.float64):
    mapper = OrbitalMapper(
        basis,
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=True,
        full_soc_prediction=False,
    )
    mapper.get_irreps()
    return mapper


def _nonscalar_norm(irreps: o3.Irreps, coords: torch.Tensor) -> torch.Tensor:
    """L2 norm of every non-``0e`` coordinate of a coupled-RME vector."""
    pieces = []
    offset = 0
    for mul, ir in irreps:
        width = mul * ir.dim
        if ir.l != 0:
            pieces.append(coords[offset:offset + width])
        offset += width
    if not pieces:
        return coords.new_zeros(())
    return torch.cat(pieces).norm()


def _record(mapper):
    data = {
        "atomic_numbers": torch.tensor([1, 6]),
        "atom_types": torch.tensor([[mapper.chemical_symbol_to_type["H"]], [mapper.chemical_symbol_to_type["C"]]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.zeros(2, 3),
        "edge_type": torch.tensor([[mapper.bond_to_type["H-C"]], [mapper.bond_to_type["C-H"]]]),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "cell": torch.eye(3) * 8.0,
        "pbc": torch.tensor([False, False, False]),
        "batch": torch.zeros(2, dtype=torch.long),
    }
    node = torch.zeros(2, 4, 4)
    node[0, 0, 0] = 0.25
    c = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 100.0
    node[1] = 0.5 * (c + c.T)
    edge = torch.zeros(2, 4, 4)
    edge[0, :1, :4] = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    edge[1, :4, :1] = edge[0, :1, :4].T
    packed = BlockTensorResult(node, edge, torch.tensor([[1, 1], [4, 4]]), torch.tensor([[1, 4], [4, 1]]))
    node_h0, edge_h0 = block_tensors_to_feature_tensors(
        data, mapper, node_blocks=node * 0.5, edge_blocks=edge * 0.5
    )
    keep = int(mapper.reduced_matrix_element)
    data.update(
        {
            "node_h0": node_h0,
            "edge_h0": edge_h0,
            "node_delta_hamil_blocks": packed.node_blocks,
            "edge_delta_hamil_blocks": packed.edge_blocks,
            "node_delta_hamil_block_shape": packed.node_shapes,
            "edge_delta_hamil_block_shape": packed.edge_shapes,
            "blockwise_spatial_schema": "deeptb.blockwise_spatial/v1",
            "blockwise_target_mode": "already-delta",
            "blockwise_source_target_feature_width": keep,
            "blockwise_source_h0_feature_width": keep,
            "soc_uureal_compact": True,
            "soc_uureal_full_rme": keep * 8,
            "soc_uureal_keep": keep,
        }
    )
    return data


def _flow(mapper, **overrides):
    options = {
        "enabled": True,
        "mode": "residual",
        "prior": "zero",
        "output_space": "uureal_block_ode",
        "block_ode": True,
        "state_space": "residual_ao_block",
        "target_semantics": "residual_dh",
        "block_input_adapter": "direct_cg",
        "h0_condition_space": "compact_uureal_rme",
        "block_export_final_full_h": False,
        "prediction_add_h0": False,
        "time_conditioning_required": True,
        "node_block_target_key": "node_delta_hamil_blocks",
        "edge_block_target_key": "edge_delta_hamil_blocks",
        "node_block_shape_key": "node_delta_hamil_block_shape",
        "edge_block_shape_key": "edge_delta_hamil_block_shape",
        "validation_ode_steps": [1, 3],
    }
    options.update(overrides)
    return HamiltonianCFM(options, idp=mapper, dtype=torch.float32)


def test_direct_projector_inverts_e3hamiltonian_forward_cgbasis():
    """The AO-block contraction is the exact inverse of E3Hamiltonian forward CG.

    Independent oracle: E3Hamiltonian owns the production RME->block forward CG
    (``cgbasis[pairtype] = wigner_3j(l1,l2,L)*sqrt(2L+1)``).  Expanding a random
    coupled-RME vector with that oracle and contracting the resulting AO block
    with the projector must recover the same RME (in the projector's sorted irrep
    order) to fp64 machine precision -- proving the projector performs a genuine
    product->coupled CG decomposition, not a flatten/gather.
    """
    mapper = _uureal_mapper({"C": "1p"})
    irreps_in = mapper.get_irreps().sort()[0].simplify()
    projector = DirectUuRealBlockProjector(mapper, irreps_in, dtype=torch.float64, device="cpu")
    atom_types = torch.tensor([[mapper.chemical_symbol_to_type["C"]]])
    p_slice = mapper.orbital_maps["C"]["1p"]

    oracle = E3Hamiltonian(
        idp=OrbitalMapper({"C": "1p"}, method="e3tb"),
        decompose=False,
        dtype=torch.float64,
        device="cpu",
    )
    cg = oracle.cgbasis["p-p"].to(torch.float64)  # (3, 3, 9) forward CG
    torch.manual_seed(0)
    rme = torch.randn(int(mapper.reduced_matrix_element), dtype=torch.float64)
    forward_block = torch.zeros(1, projector.canvas, projector.canvas, dtype=torch.float64)
    forward_block[0, p_slice, p_slice] = torch.einsum("ijr,r->ij", cg, rme)

    recovered = projector._contract(forward_block, atom_types, projector.node_plan)[0]
    expected = rme.index_select(0, projector.sort_index)
    assert torch.allclose(recovered, expected, atol=1e-10, rtol=0.0)


def test_direct_projector_pp_onsite_scalar_has_zero_nonscalar_channels():
    """Review counterexample: a pure-scalar p-p block has zero non-scalar RME.

    A ``p-p`` onsite block equal to the 3x3 identity is a pure rotational scalar
    (the ``0e`` trace channel).  A correct inverse-CG must therefore leave every
    ``1e``/``2e`` coordinate exactly zero.  The pre-fix flatten/gather produced a
    non-scalar norm of ~1.414 here; the CG contraction produces zero.
    """
    mapper = _uureal_mapper({"C": "1p"})
    irreps_in = mapper.get_irreps().sort()[0].simplify()
    projector = DirectUuRealBlockProjector(mapper, irreps_in, dtype=torch.float64, device="cpu")
    atom_types = torch.tensor([[mapper.chemical_symbol_to_type["C"]]])
    p_slice = mapper.orbital_maps["C"]["1p"]

    block = torch.zeros(1, projector.canvas, projector.canvas, dtype=torch.float64)
    block[0, p_slice, p_slice] = torch.eye(3, dtype=torch.float64)
    coupled = projector._contract(block, atom_types, projector.node_plan)[0]

    assert _nonscalar_norm(irreps_in, coupled) <= 1e-10
    # Non-vacuity: the scalar (trace) channel is genuinely populated.
    assert coupled.norm() > 1.0


def test_direct_projector_is_so3_rotation_covariant():
    """Rotating the AO block equals rotating the coupled output by Wigner-D.

    Independent oracle: e3nn Wigner-D of the AO shells (``1x0e+1x1o`` canvas) and
    of the coupled ``irreps_in``.  For a random proper rotation the two paths --
    contract(D_AO . B . D_AO^T) and D_irreps . contract(B) -- must agree to fp64
    machine precision.  A flatten/gather over product coordinates fails this.
    """
    mapper = _uureal_mapper({"C": "1s1p"})
    irreps_in = mapper.get_irreps().sort()[0].simplify()
    projector = DirectUuRealBlockProjector(mapper, irreps_in, dtype=torch.float64, device="cpu")
    atom_types = torch.tensor([[mapper.chemical_symbol_to_type["C"]]])

    # e3nn's D_from_matrix routes angle intermediates through the *default* dtype,
    # so fp64 covariance requires a fp64 default (else it caps out near 1e-7).
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        rotation = o3.rand_matrix(dtype=torch.float64)  # proper SO(3), machine-eps orthogonal
        d_ao = o3.Irreps("1x0e+1x1o").D_from_matrix(rotation)  # canvas 4x4
        d_irreps = irreps_in.D_from_matrix(rotation)

        block = torch.randn(1, projector.canvas, projector.canvas, dtype=torch.float64)
        rotated = (d_ao @ block[0] @ d_ao.transpose(-1, -2)).unsqueeze(0)

        lhs = projector._contract(rotated, atom_types, projector.node_plan)[0]
        rhs = d_irreps @ projector._contract(block, atom_types, projector.node_plan)[0]
        assert torch.allclose(lhs, rhs, atol=1e-10, rtol=0.0)
    finally:
        torch.set_default_dtype(previous_default)


def test_direct_projector_zero_is_bit_exact():
    mapper = _mapper()
    data = _record(mapper)
    projector = DirectUuRealBlockProjector(
        mapper, mapper.get_irreps().sort()[0].simplify(), dtype=torch.float32, device="cpu"
    )
    zero = copy.deepcopy(data)
    zero[_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY] = torch.zeros_like(data["node_delta_hamil_blocks"])
    zero[_keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY] = torch.zeros_like(data["edge_delta_hamil_blocks"])
    zero[_keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY] = data["node_delta_hamil_block_shape"]
    zero[_keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY] = data["edge_delta_hamil_block_shape"]
    node_hidden, edge_hidden = projector(
        zero, zero["atom_types"], zero["edge_type"], torch.arange(2)
    )
    assert torch.count_nonzero(node_hidden) == 0
    assert torch.count_nonzero(edge_hidden) == 0


def test_prepare_is_exact_scalar_bridge_and_does_not_require_h0_blocks():
    mapper = _mapper()
    data = _record(mapper)
    flow = _flow(mapper)
    model_data, ref, ctx = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.25])
    )
    assert torch.equal(
        model_data[_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY],
        0.25 * data["node_delta_hamil_blocks"],
    )
    assert torch.equal(
        model_data[_keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY],
        0.25 * data["edge_delta_hamil_blocks"],
    )
    assert "node_h0_blocks" not in data and "edge_h0_blocks" not in data
    assert ctx.block_target_semantics == "residual_dh"
    assert ref["blockwise_target_mode"] == "already-delta"


@pytest.mark.parametrize(
    "key,bad",
    [
        ("blockwise_spatial_schema", "wrong/v0"),
        ("blockwise_target_mode", "absolute"),
        ("blockwise_source_target_feature_width", 15),
        ("blockwise_source_h0_feature_width", 15),
        ("soc_uureal_compact", False),
        ("soc_uureal_full_rme", 16),
        ("soc_uureal_keep", 15),
    ],
)
def test_data_gate_fails_closed_per_metadata_field(key, bad):
    mapper = _mapper()
    data = _record(mapper)
    data[key] = bad
    with pytest.raises(ValueError, match=key):
        _flow(mapper).prepare_batch(copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.5]))


def test_flow_gate_accepts_converter_recorded_source_width():
    """Contract fix: keep < source_width is the NORMAL converter product.

    The official convert_feature_lmdb_to_blockwise.py records the ORIGINAL
    source feature width (keep*8 for a full-SOC source projected to compact
    uu_real).  The gate must accept it: the stored tensors are keep-wide and
    keep matches the mapper; the source width is provenance, not a storage
    width.  Anything below keep (or non-integer) still fails closed.
    """
    mapper = _mapper()
    keep = int(mapper.reduced_matrix_element)
    data = _record(mapper)
    data["blockwise_source_target_feature_width"] = keep * 8
    data["blockwise_source_h0_feature_width"] = keep * 8
    _flow(mapper).prepare_batch(copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.5]))

    below = _record(mapper)
    below["blockwise_source_target_feature_width"] = keep - 1
    with pytest.raises(ValueError, match="blockwise_source_target_feature_width"):
        _flow(mapper).prepare_batch(copy.deepcopy(below), copy.deepcopy(below), t=torch.tensor([0.5]))

    non_integer = _record(mapper)
    non_integer["blockwise_source_h0_feature_width"] = "wide"
    with pytest.raises(ValueError, match="blockwise_source_h0_feature_width"):
        _flow(mapper).prepare_batch(
            copy.deepcopy(non_integer), copy.deepcopy(non_integer), t=torch.tensor([0.5])
        )


def _full_soc_source_record(mapper, full_width: int):
    """A full-SOC-width feature record shaped like the official converter input."""
    import numpy as np

    data = _record(mapper)
    generator = torch.Generator().manual_seed(0)
    n_nodes = int(data["atomic_numbers"].shape[0])
    n_edges = int(data["edge_index"].shape[1])
    record = {
        "atomic_numbers": data["atomic_numbers"].numpy().astype("int64"),
        "pos": data["pos"].numpy().astype("float32"),
        "cell": data["cell"].numpy().astype("float32"),
        "pbc": data["pbc"].numpy(),
        "edge_index": data["edge_index"].numpy().astype("int64"),
        "edge_cell_shift": data["edge_cell_shift"].numpy().astype("float32"),
        "node_features": torch.randn(n_nodes, full_width, generator=generator).numpy().astype("float32"),
        "edge_features": torch.randn(n_edges, full_width, generator=generator).numpy().astype("float32"),
        "node_h0": torch.randn(n_nodes, full_width, generator=generator).numpy().astype("float32"),
        "edge_h0": torch.randn(n_edges, full_width, generator=generator).numpy().astype("float32"),
        "hamiltonian_semantics": "delta (H - H0), uu_real",
        "soc_real_channel_order": np.asarray(
            ["uu_re", "uu_im", "ud_re", "ud_im", "du_re", "du_im", "dd_re", "dd_im"]
        ),
        "full_soc_feature_width": full_width,
        "idx": 0,
        "nf": 1,
    }
    return record


def test_official_converter_product_passes_loader_gate(tmp_path):
    """converter -> loader chain: keep=16, source=128 must load (review P1-2).

    Runs the official convert_feature_lmdb_to_blockwise.py on a synthetic
    full-SOC source LMDB (feature width keep*8=128) and loads the genuine
    product through LMDBDataset with require_uureal_block_ode=True.  The
    product records blockwise_source_*_feature_width=128 with soc_uureal_keep=16;
    the pre-fix gate demanded source_width==keep and rejected every real
    converter product at startup.  A tampered product whose recorded source
    width falls below keep must still fail closed.
    """
    import json
    import pickle
    import shutil

    import lmdb

    from dptb.data.dataset.lmdb_dataset import LMDBDataset
    from tools.convert_feature_lmdb_to_blockwise import convert_root

    mapper = _mapper()
    keep = int(mapper.reduced_matrix_element)
    full_width = keep * 8
    record = _full_soc_source_record(mapper, full_width)

    source_root = tmp_path / "source"
    (source_root / "data.0000.lmdb").mkdir(parents=True)
    env = lmdb.open(str(source_root / "data.0000.lmdb"), map_size=1 << 24, subdir=True)
    with env.begin(write=True) as txn:
        txn.put((0).to_bytes(4, "big"), pickle.dumps(record, protocol=4))
    env.sync()
    env.close()

    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps(
            {
                "common_options": {
                    "basis": {"H": "1s", "C": "1s1p"},
                    "has_soc": True,
                    "nextham_uureal_mask": True,
                    "full_soc_prediction": False,
                }
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "blockwise"
    summary = convert_root(
        input_root=source_root,
        output_root=output_root,
        input_json=input_json,
        target_mode="already-delta",
    )
    assert summary["entries"] == 1

    produced = None
    env = lmdb.open(
        str(output_root / "data.0000.lmdb"), readonly=True, lock=False, subdir=True
    )
    with env.begin() as txn:
        produced = pickle.loads(txn.get((0).to_bytes(4, "big")))
    env.close()
    # The official product records the ORIGINAL source width, not keep.
    assert int(produced["blockwise_source_target_feature_width"]) == full_width
    assert int(produced["blockwise_source_h0_feature_width"]) == full_width
    assert int(produced["soc_uureal_keep"]) == keep

    info_files = {
        "data.0000.lmdb": {
            "r_max": 2.0,
            "er_max": None,
            "oer_max": None,
            "wave_align": False,
            "train_w_homo_lumo_gap": False,
            "train_w_eps": False,
            "train_w_charge": False,
            "train_dip": False,
            "train_polar": False,
        }
    }
    dataset = LMDBDataset(
        root=str(output_root),
        info_files=info_files,
        type_mapper=mapper,
        get_Hamiltonian=True,
        get_H0=True,
        prefer_precomputed_h0=True,
        residual_hamiltonian=False,
        require_uureal_block_ode=True,
    )
    sample = dataset.get(0)
    assert sample is not None

    # Negative: a recorded source width below keep still fails closed.
    tampered_root = tmp_path / "tampered"
    shutil.copytree(output_root, tampered_root)
    bad = dict(produced)
    bad["blockwise_source_target_feature_width"] = keep - 1
    env = lmdb.open(str(tampered_root / "data.0000.lmdb"), map_size=1 << 24, subdir=True)
    with env.begin(write=True) as txn:
        txn.put((0).to_bytes(4, "big"), pickle.dumps(bad, protocol=4), overwrite=True)
    env.sync()
    env.close()
    tampered = LMDBDataset(
        root=str(tampered_root),
        info_files=info_files,
        type_mapper=mapper,
        get_Hamiltonian=True,
        get_H0=True,
        prefer_precomputed_h0=True,
        residual_hamiltonian=False,
        require_uureal_block_ode=True,
    )
    with pytest.raises(ValueError, match="blockwise_source_target_feature_width"):
        tampered.get(0)


def test_data_gate_accepts_identical_collated_metadata_and_rejects_mixed_values():
    mapper = _mapper()
    data = _record(mapper)
    data["blockwise_spatial_schema"] = ["deeptb.blockwise_spatial/v1"]
    data["blockwise_target_mode"] = ["already-delta"]
    for key in (
        "blockwise_source_target_feature_width",
        "blockwise_source_h0_feature_width",
        "soc_uureal_compact",
        "soc_uureal_full_rme",
        "soc_uureal_keep",
    ):
        data[key] = torch.as_tensor([data[key]])
    _flow(mapper).prepare_batch(copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.5]))

    data["blockwise_spatial_schema"] = ["deeptb.blockwise_spatial/v1", "wrong/v0"]
    with pytest.raises(ValueError, match="batched metadata values"):
        _flow(mapper).prepare_batch(copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.5]))


class _EndpointSpy(torch.nn.Module):
    def __init__(self, endpoints):
        super().__init__()
        self.endpoints = endpoints
        self.inputs = []

    def forward(self, data):
        self.inputs.append(
            (
                data[_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY].clone(),
                data[_keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY].clone(),
            )
        )
        node, edge = self.endpoints[len(self.inputs) - 1]
        out = data.copy()
        out[_keys.NODE_PRED_HAMIL_BLOCKS_KEY] = node.clone()
        out[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY] = edge.clone()
        return out


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_sampler_zero_start_spy_closed_loop_and_manual_blend(steps):
    mapper = _mapper()
    data = _record(mapper)
    node = data["node_delta_hamil_blocks"]
    edge = data["edge_delta_hamil_blocks"]
    endpoints = [(node * (i + 1), edge * (i + 1)) for i in range(steps)]
    model = _EndpointSpy(endpoints)
    result = _flow(mapper).sample(model, copy.deepcopy(data), num_steps=steps)
    assert torch.count_nonzero(model.inputs[0][0]) == 0
    assert torch.count_nonzero(model.inputs[0][1]) == 0
    if steps >= 2:
        torch.testing.assert_close(model.inputs[1][0], node / steps, rtol=0.0, atol=1e-8)
        torch.testing.assert_close(model.inputs[1][1], edge / steps, rtol=0.0, atol=1e-8)
    assert torch.equal(result[_keys.NODE_PRED_HAMIL_BLOCKS_KEY], endpoints[-1][0])
    assert torch.equal(result[_keys.EDGE_PRED_HAMIL_BLOCKS_KEY], endpoints[-1][1])


def test_directed_compatible_metric_coexists_with_canonical_and_matches_hand_reduction():
    """Review P1-5: historical directed-coordinate metric alongside canonical.

    canonical (train_onsite/hopping_loss): each independent freedom once
    (onsite upper triangle + one canonical edge per Hermitian pair).
    compatible_directed (train_compatible_directed_*): every stored directed
    coordinate -- all valid onsite entries + ALL directed edges -- the same
    population the historical SOC uu-real RME losses averaged over.  Both are
    locked here against an independent hand-written reduction on a constructed
    sample where the two must disagree on hopping (2 directed edges vs 1
    canonical) and agree on onsite only through their different masks.
    """
    from dptb.data.interfaces.blockwise_tensor import (
        block_mask_from_shapes,
        strict_reverse_edge_index,
    )

    mapper = _mapper()
    data = _record(mapper)
    flow = _flow(mapper)

    _model_data, ref, ctx = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=torch.tensor([0.0])
    )
    pred = dict(ref)
    pred[flow.node_output_key] = 2.0 * torch.as_tensor(ref[flow.node_block_target_key])
    pred[flow.edge_output_key] = 2.0 * torch.as_tensor(ref[flow.edge_block_target_key])
    loss, state = flow.loss_on_sample(pred, ref, ctx)

    for key in (
        "train_compatible_directed_onsite_loss",
        "train_compatible_directed_hopping_loss",
        "train_compatible_directed_loss",
    ):
        assert key in state, key

    # Independent hand reduction: diff == target blocks (pred = 2*target).
    node_diff = torch.as_tensor(ref[flow.node_block_target_key], dtype=torch.float32)
    edge_diff = torch.as_tensor(ref[flow.edge_block_target_key], dtype=torch.float32)
    node_valid = block_mask_from_shapes(
        torch.as_tensor(ref[flow.node_block_shape_key]), tuple(node_diff.shape[-2:])
    )
    edge_valid = block_mask_from_shapes(
        torch.as_tensor(ref[flow.edge_block_shape_key]), tuple(edge_diff.shape[-2:])
    )
    upper = torch.triu(torch.ones(tuple(node_diff.shape[-2:]), dtype=torch.bool))
    rev = strict_reverse_edge_index(ref, idp=mapper)
    canonical_rows = torch.arange(edge_diff.shape[0]) <= rev

    assert flow.loss_type == "mse"  # hand reduction below assumes the default

    def hand_metric(diff, mask):
        values = diff[mask]
        return values.square().sum() / float(values.numel())

    canonical_onsite = hand_metric(node_diff, node_valid & upper.unsqueeze(0))
    canonical_hopping = hand_metric(edge_diff, edge_valid & canonical_rows.view(-1, 1, 1))
    directed_onsite = hand_metric(node_diff, node_valid)
    directed_hopping = hand_metric(edge_diff, edge_valid)

    torch.testing.assert_close(state["train_onsite_loss"], canonical_onsite, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(state["train_hopping_loss"], canonical_hopping, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        state["train_compatible_directed_onsite_loss"], directed_onsite, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        state["train_compatible_directed_hopping_loss"], directed_hopping, rtol=1e-6, atol=1e-7
    )
    # The two reductions genuinely differ on this sample's coordinate counts:
    # 2 directed edges vs 1 canonical edge (H-C / C-H Hermitian pair).
    assert int((edge_valid & canonical_rows.view(-1, 1, 1)).sum()) < int(edge_valid.sum())


def test_compatible_scorer_scores_uureal_residual_without_block_codec():
    """Review P1-4 repro: first N=1 validation with euler logging disabled.

    MultiTrainer picks flow.compatible_loss_on_sample when
    log_validation_flow_euler_loss=false (multi_trainer._validation path).  For
    uureal_block_ode the rollout returns residual-dH blocks and block_codec is
    intentionally None, so the pre-fix Full-H scoring path crashed with
    AttributeError ('NoneType'.rme_to_blocks).  The compatible scorer must
    route the mode to the residual-dH scorer and agree with loss_on_sample;
    the Full-H path (block_codec present, non-uureal) is untouched.
    """
    mapper = _mapper()
    data = _record(mapper)
    flow = _flow(
        mapper, log_validation_flow_euler_loss=False, validation_ode_steps=[1]
    )
    assert flow.block_codec is None

    node = data["node_delta_hamil_blocks"]
    edge = data["edge_delta_hamil_blocks"]
    sampled = flow.sample(_EndpointSpy([(node, edge)]), copy.deepcopy(data), num_steps=1)

    # Reconstruct the exact t=0 flow context MultiTrainer builds before scoring.
    zero_t = torch.zeros(1)
    _flow_batch, flow_ref, flow_ctx = flow.prepare_batch(
        copy.deepcopy(data), copy.deepcopy(data), t=zero_t
    )
    loss, state = flow.compatible_loss_on_sample(sampled, flow_ref, flow_ctx)
    assert torch.isfinite(loss)
    reference_loss, _ = flow.loss_on_sample(sampled, flow_ref, flow_ctx)
    torch.testing.assert_close(loss, reference_loss, rtol=0.0, atol=0.0)


def test_t0_injection_bypasses_t_min_clamp():
    """Review repro: t_min=0.5 + t0_probability=1 must yield exact zeros.

    t0_probability exists to train the t=0, D=0 inference boundary; before the
    fix the injected zeros were re-clamped to t_min (0.5), silently deleting
    the boundary training mass.
    """
    mapper = _mapper()
    flow = _flow(mapper, t_min=0.5, t0_probability=1.0)
    t = flow._sample_t(num_graphs=64, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(t, torch.zeros_like(t))

    # Partial injection: zeros survive AND every non-zero sample honours t_min.
    flow = _flow(mapper, t_min=0.5, t0_probability=0.5)
    generator = torch.Generator().manual_seed(0)
    t = flow._sample_t(
        num_graphs=512, device=torch.device("cpu"), dtype=torch.float32,
        generator=generator,
    )
    zero = t == 0.0
    assert bool(zero.any()) and bool((~zero).any())
    assert bool((t[~zero] >= 0.5).all())


def test_uureal_t0_probability_defaults_positive_and_rejects_explicit_zero():
    """uureal_block_ode must guarantee t=0 training mass (review P1-3b)."""
    mapper = _mapper()
    assert _flow(mapper).t0_probability == pytest.approx(0.15)
    assert _flow(mapper, t0_probability=0.25).t0_probability == pytest.approx(0.25)
    with pytest.raises(ValueError, match="t0_probability"):
        _flow(mapper, t0_probability=0.0)
    with pytest.raises(ValueError, match="t0_probability"):
        _flow(mapper, t0_probability=-0.1)


def test_argcheck_uureal_exception_is_explicit_and_mutually_bound():
    config = {
        "common_options": {
            "dtype": "float32", "has_soc": True,
            "nextham_uureal_mask": True, "full_soc_prediction": False,
        },
        "train_options": {
            "flow_options": {
                "enabled": True, "mode": "residual", "prior": "zero",
                "output_space": "uureal_block_ode", "block_ode": True,
                "state_space": "residual_ao_block", "target_semantics": "residual_dh",
                "block_input_adapter": "direct_cg",
                "h0_condition_space": "compact_uureal_rme",
                "prediction_add_h0": False, "time_conditioning_required": True,
                "node_block_target_key": "node_delta_hamil_blocks",
                "edge_block_target_key": "edge_delta_hamil_blocks",
                "node_block_shape_key": "node_delta_hamil_block_shape",
                "edge_block_shape_key": "edge_delta_hamil_block_shape",
                "validation_ode_steps": [1, 3],
            }
        },
        "model_options": {
            "embedding": {
                "method": "lem_moe_v3_h0", "output_route": "h_b0",
                "require_full_block_edge_coverage": True,
                "use_uureal_residual_block_input": True,
                "use_flow_time_embedding": True, "flow_time_condition_edges": True,
                "flow_time_allow_missing": False, "h0_merge_mode": "replace",
                "use_h0_node_init": True, "use_h0_edge_init": True,
            },
            "prediction": {
                "method": "block_native", "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True, "add_h0": False,
            },
        },
        "data_options": {
            "train": {
                "type": "LMDBDataset", "get_Hamiltonian": True, "get_H0": True,
                "residual_hamiltonian": False, "require_full_h_target": False,
                "require_residual_h_target": False, "require_uureal_block_ode": True,
            }
        },
    }
    validate_block_ode_contract(config)

    # t=0 training-mass interlock: explicit non-positive t0_probability fails
    # closed; a positive value or omission (runtime default 0.15) validates.
    config["train_options"]["flow_options"]["t0_probability"] = 0.0
    with pytest.raises(ValueError, match="t0_probability"):
        validate_block_ode_contract(config)
    config["train_options"]["flow_options"]["t0_probability"] = 0.2
    validate_block_ode_contract(config)
    del config["train_options"]["flow_options"]["t0_probability"]

    config["common_options"]["has_soc"] = False
    with pytest.raises(ValueError, match="has_soc=true"):
        validate_block_ode_contract(config)


def test_direct_projector_zero_preservation_is_structural_hard_gate():
    """Bias-free projector guarantees D_0=0 -> exactly zero hidden.

    Elevates the zero-preservation contract to a structural + bit-level hard
    gate: (1) neither node/edge equivariant linear owns a learnable bias, so the
    map is a pure ``W.x`` that cannot shift a zero residual; (2) a zero residual
    yields *exactly* zero hidden (bit-level torch.equal); (3) the packed
    non-zero residual yields non-zero hidden, so the zero gate is not vacuously
    satisfied by degenerate weights.
    """
    mapper = _mapper()
    data = _record(mapper)
    projector = DirectUuRealBlockProjector(
        mapper, mapper.get_irreps().sort()[0].simplify(), dtype=torch.float32, device="cpu"
    )

    # (1) structural: no learnable bias parameter on either equivariant linear.
    assert [n for n, _ in projector.node_linear.named_parameters() if "bias" in n] == []
    assert [n for n, _ in projector.edge_linear.named_parameters() if "bias" in n] == []

    # (3) non-triviality: the packed (non-zero) residual must move the hidden.
    live = copy.deepcopy(data)
    live[_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY] = data["node_delta_hamil_blocks"].clone()
    live[_keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY] = data["edge_delta_hamil_blocks"].clone()
    live[_keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY] = data["node_delta_hamil_block_shape"]
    live[_keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY] = data["edge_delta_hamil_block_shape"]
    node_live, edge_live = projector(
        live, live["atom_types"], live["edge_type"], torch.arange(2)
    )
    assert torch.count_nonzero(node_live) > 0
    assert torch.count_nonzero(edge_live) > 0

    # (2) bit-level zero-preservation: zero residual -> exactly zero hidden.
    zero = copy.deepcopy(data)
    zero[_keys.NODE_UUREAL_RESIDUAL_BLOCKS_KEY] = torch.zeros_like(data["node_delta_hamil_blocks"])
    zero[_keys.EDGE_UUREAL_RESIDUAL_BLOCKS_KEY] = torch.zeros_like(data["edge_delta_hamil_blocks"])
    zero[_keys.NODE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY] = data["node_delta_hamil_block_shape"]
    zero[_keys.EDGE_UUREAL_RESIDUAL_BLOCK_SHAPE_KEY] = data["edge_delta_hamil_block_shape"]
    node_hidden, edge_hidden = projector(
        zero, zero["atom_types"], zero["edge_type"], torch.arange(2)
    )
    assert torch.equal(node_hidden, torch.zeros_like(node_hidden))
    assert torch.equal(edge_hidden, torch.zeros_like(edge_hidden))


def test_argcheck_accepts_v2_contract_aliases():
    """The authoritative V2 dataset-contract option names validate as aliases.

    state_space=nextham_uureal_delta_block and h0_condition_space=
    nextham_uureal_rme are the V2 vocabulary for the F4 canonical
    residual_ao_block / compact_uureal_rme markers; both must validate.
    """
    config = {
        "common_options": {
            "dtype": "float32", "has_soc": True,
            "nextham_uureal_mask": True, "full_soc_prediction": False,
        },
        "train_options": {
            "flow_options": {
                "enabled": True, "mode": "residual", "prior": "zero",
                "output_space": "uureal_block_ode", "block_ode": True,
                "state_space": "nextham_uureal_delta_block",
                "target_semantics": "residual_dh",
                "block_input_adapter": "direct_cg",
                "h0_condition_space": "nextham_uureal_rme",
                "prediction_add_h0": False, "time_conditioning_required": True,
                "node_block_target_key": "node_delta_hamil_blocks",
                "edge_block_target_key": "edge_delta_hamil_blocks",
                "node_block_shape_key": "node_delta_hamil_block_shape",
                "edge_block_shape_key": "edge_delta_hamil_block_shape",
                "validation_ode_steps": [1, 3],
            }
        },
        "model_options": {
            "embedding": {
                "method": "lem_moe_v3_h0", "output_route": "h_b0",
                "require_full_block_edge_coverage": True,
                "use_uureal_residual_block_input": True,
                "use_flow_time_embedding": True, "flow_time_condition_edges": True,
                "flow_time_allow_missing": False, "h0_merge_mode": "replace",
                "use_h0_node_init": True, "use_h0_edge_init": True,
            },
            "prediction": {
                "method": "block_native", "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True, "add_h0": False,
            },
        },
        "data_options": {
            "train": {
                "type": "LMDBDataset", "get_Hamiltonian": True, "get_H0": True,
                "residual_hamiltonian": False, "require_full_h_target": False,
                "require_residual_h_target": False, "require_uureal_block_ode": True,
            }
        },
    }
    validate_block_ode_contract(config)
    # A name outside both vocabularies must still fail closed.
    config["train_options"]["flow_options"]["state_space"] = "not_a_marker"
    with pytest.raises(ValueError, match="state_space"):
        validate_block_ode_contract(config)


def test_block_state_codec_three_way_distinguishes_uureal_from_full_spinor():
    """The exact_rme codec routes compact uu_real to direct_spatial, not full spinor.

    Both remain fail-closed, but the messages must be distinct so a compact
    uu_real misconfiguration is directed to the residual projector while full
    spinor SOC stays an unconditional rejection (preserving the historical
    'does not support SOC' boundary).
    """
    uureal = _mapper()
    with pytest.raises(NotImplementedError, match="direct_spatial residual"):
        BlockStateCodec(uureal, dtype=torch.float64)

    full_spinor = OrbitalMapper({"C": ["2p"]}, method="e3tb", has_soc=True)
    with pytest.raises(NotImplementedError, match="does not support SOC"):
        BlockStateCodec(full_spinor, dtype=torch.float64)

"""Shared LemPair / pair-model test builders (not collected; import from dptb.tests.pair_helpers)."""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.embedding.ao_projector_bank import shell_l
from dptb.nn.embedding.lem_pair import LemPair

_XYZ_TO_YZX = torch.tensor(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64
)
_MOLECULE_POSITIONS = (
    (0.0, 0.0, 0.0),
    (0.70, 0.20, 0.10),
    (2.10, 0.30, -0.40),
    (0.20, 2.40, 0.50),
)


@contextmanager
def fp64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextmanager
def deterministic_fp64():
    """fp64 default dtype plus deterministic algorithms, both restored on exit."""
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True)
    try:
        with fp64_default():
            yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def model(
    *,
    seed=20260723,
    mp_cutoff=None,
    pair_refine_enable=False,
    res_update_additive=False,
    latents_layernorm=True,
    **overrides,
):
    torch.manual_seed(seed)
    options = model_options()
    options.update(
        mp_cutoff=mp_cutoff,
        pair_refine_enable=pair_refine_enable,
        pair_refine_rank=4,
        pair_refine_init=0.1,
        res_update_additive=res_update_additive,
        latents_layernorm=latents_layernorm,
    )
    options.update(overrides)
    return LemPair(**options).eval()


def model_options():
    return dict(
        basis={"O": "1s1p"},
        n_layers=2,
        n_radial_basis=4,
        r_max=5.0,
        irreps_hidden="2x0e+2x1o+2x1e+2x2e",
        avg_num_neighbors=3.0,
        mp_avg_num_neighbors=1.5,
        env_embed_multiplicity=2,
        latent_dim=4,
        latent_channels=[4],
        edge_one_hot_dim=2,
        num_experts=1,
        num_shared_experts=1,
        top_k=1,
        use_layer_onehot_tp=False,
        use_out_onehot_tp=False,
        use_interpolation_out=False,
        tp_radial_emb=False,
        mole_linear_mode="indexed_ref",
        so2_fusion_mode="streamed_m_major_ref",
        equivariant_norm_type="merged_rms",
        output_route="h_b0",
        rme_fusion_rank=2,
        rme_fusion_init=0.2,
        use_h0_init=True,
        fallback_to_hamiltonian=False,
        require_full_block_edge_coverage=True,
        dtype=torch.float64,
        device="cpu",
    )


def complete_directed_edges(count: int) -> torch.Tensor:
    rows = [(i, j) for i in range(count) for j in range(count) if i != j]
    return torch.tensor(rows, dtype=torch.long).T.contiguous()


def molecule_data(model, positions=None):
    """An all-oxygen molecule with complete directed edges and zero H0."""
    positions = torch.as_tensor(
        _MOLECULE_POSITIONS if positions is None else positions, dtype=torch.float64
    )
    n_atoms = int(positions.shape[0])
    edge_index = complete_directed_edges(n_atoms)
    n_edges = edge_index.shape[1]
    h0_dim = model.idp.reduced_matrix_element
    return {
        _keys.POSITIONS_KEY: positions,
        _keys.EDGE_INDEX_KEY: edge_index,
        _keys.ATOM_TYPE_KEY: torch.zeros((n_atoms, 1), dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.full(
            (n_edges,), model.idp.bond_to_type["O-O"], dtype=torch.long
        ),
        _keys.NODE_H0_KEY: torch.zeros((n_atoms, h0_dim), dtype=torch.float64),
        _keys.EDGE_H0_KEY: torch.zeros((n_edges, h0_dim), dtype=torch.float64),
    }


def batch_graphs(graphs):
    """Concatenate molecule_data graphs into one batch with offset edge indices."""
    keys = (
        _keys.POSITIONS_KEY,
        _keys.ATOM_TYPE_KEY,
        _keys.EDGE_TYPE_KEY,
        _keys.NODE_H0_KEY,
        _keys.EDGE_H0_KEY,
    )
    batch = {key: torch.cat([graph[key] for graph in graphs]) for key in keys}
    edge_indices, batches, offset = [], [], 0
    for index, graph in enumerate(graphs):
        n_nodes = int(graph[_keys.POSITIONS_KEY].shape[0])
        edge_indices.append(graph[_keys.EDGE_INDEX_KEY] + offset)
        batches.append(torch.full((n_nodes,), index, dtype=torch.long))
        offset += n_nodes
    batch[_keys.EDGE_INDEX_KEY] = torch.cat(edge_indices, dim=1)
    batch[_keys.BATCH_KEY] = torch.cat(batches)
    return batch


def clone_data(data):
    return {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in data.items()
    }


def rotate_data(data, rotation):
    rotated = clone_data(data)
    rotated[_keys.POSITIONS_KEY] = data[_keys.POSITIONS_KEY] @ rotation.T
    if _keys.CELL_KEY in data:
        rotated[_keys.CELL_KEY] = data[_keys.CELL_KEY] @ rotation.T
    return rotated


def feature_wigner(irreps, rotation):
    """Representation of a Cartesian rotation on e3nn features (l=1 stored as y, z, x)."""
    return o3.Irreps(irreps).D_from_matrix(_XYZ_TO_YZX @ rotation @ _XYZ_TO_YZX.T)


def ao_wigner(model, rotation):
    ao_irreps = o3.Irreps(
        [
            (1, (shell_l(shell), (-1) ** shell_l(shell)))
            for shell in model.idp.full_basis
        ]
    )
    return feature_wigner(ao_irreps, rotation)


def edge_block_drift(model, data, rotation):
    """Max |H_e(R x) - D H_e(x) D^T| over edge blocks, and the unrotated output."""
    with torch.no_grad():
        reference = model(clone_data(data))
        rotated = model(rotate_data(data, rotation))
    d_ao = ao_wigner(model, rotation)
    expected = torch.einsum(
        "ij,njk,lk->nil", d_ao, reference[_keys.EDGE_HAMILTONIAN_KEY], d_ao
    )
    drift = (rotated[_keys.EDGE_HAMILTONIAN_KEY] - expected).abs().max().item()
    return drift, reference


def block_ode_model(method="lem_pair", **embedding_overrides):
    """A small fp64 H-C block-ODE model (block-native head, flow-time conditioning)."""
    from dptb.nn.build import build_model

    embedding = {
        "method": method,
        "output_route": "h_b0",
        "h0_init_scope": "both",
        "use_spatial_residual_block_input": True,
        "n_layers": 1,
        "avg_num_neighbors": 2.0,
        "r_max": 4.0,
        "irreps_hidden": "2x0e+2x1o+2x1e+2x2e",
        "env_embed_multiplicity": 2,
        "latent_dim": 6,
        "latent_channels": [6],
        "edge_one_hot_dim": 3,
        "num_experts": 1,
        "num_shared_experts": 1,
        "top_k": 1,
        "universal": True,
        "use_layer_onehot_tp": False,
        "use_out_onehot_tp": False,
        "use_interpolation_out": False,
        "tp_radial_emb": False,
        "mole_linear_mode": "indexed_ref",
        "so2_fusion_mode": "streamed_m_major_ref",
        "rme_fusion_rank": 3,
        "rme_fusion_init": 0.0,
        "use_flow_time_embedding": True,
        "flow_time_condition_edges": True,
        "flow_time_allow_missing": False,
        "require_full_block_edge_coverage": True,
    }
    embedding.update(embedding_overrides)
    return build_model(
        common_options={
            "basis": {"H": "1s", "C": "1s1p"},
            "overlap": False,
            "dtype": "float64",
            "device": "cpu",
        },
        model_options={
            "embedding": embedding,
            "prediction": {
                "method": "block_native",
                "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True,
                "scale_type": "no_scale",
            },
        },
        train_options={},
        no_check=False,
    ).to(dtype=torch.float64).eval()


def prepared_flow_batch(model, *, seed=31, t=0.41):
    """(flow, raw record, model input at time t) for a block-ODE model's mapper."""
    from dptb.tests.block_ode_fixtures import _b_flow, _b_record

    flow = _b_flow(model.idp, dtype=torch.float64)
    raw, _, _ = _b_record(model.idp, dtype=torch.float64, seed=seed)
    model_data, _, _ = flow.prepare_batch(
        copy.deepcopy(raw),
        copy.deepcopy(raw),
        t=torch.tensor([t], dtype=torch.float64),
    )
    return flow, raw, model_data

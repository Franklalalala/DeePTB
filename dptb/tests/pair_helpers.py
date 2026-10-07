"""Shared graph-H0 model, input, and rotation helpers."""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.embedding.ao_projector_bank import shell_l

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

from __future__ import annotations

from contextlib import contextmanager

import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.embedding.ao_projector_bank import shell_l
from dptb.nn.embedding.lem_pair import LemPair


@contextmanager
def fp64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def model(
    *,
    seed=20260723,
    mp_cutoff=None,
    pair_refine_enable=False,
    res_update_additive=False,
    latents_layernorm=True,
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


def molecule_data(model):
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.70, 0.20, 0.10],
            [2.10, 0.30, -0.40],
            [0.20, 2.40, 0.50],
        ],
        dtype=torch.float64,
    )
    edge_index = complete_directed_edges(positions.shape[0])
    n_edges = edge_index.shape[1]
    h0_dim = model.idp.reduced_matrix_element
    return {
        _keys.POSITIONS_KEY: positions,
        _keys.EDGE_INDEX_KEY: edge_index,
        _keys.ATOM_TYPE_KEY: torch.zeros((4, 1), dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.full(
            (n_edges,), model.idp.bond_to_type["O-O"], dtype=torch.long
        ),
        _keys.NODE_H0_KEY: torch.zeros((4, h0_dim), dtype=torch.float64),
        _keys.EDGE_H0_KEY: torch.zeros((n_edges, h0_dim), dtype=torch.float64),
    }


def clone_data(data):
    return {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def rotate_data(data, rotation):
    rotated = clone_data(data)
    rotated[_keys.POSITIONS_KEY] = data[_keys.POSITIONS_KEY] @ rotation.T
    if _keys.CELL_KEY in data:
        rotated[_keys.CELL_KEY] = data[_keys.CELL_KEY] @ rotation.T
    return rotated


def ao_wigner(model, rotation):
    ao_irreps = o3.Irreps(
        [
            (1, (shell_l(shell), (-1) ** shell_l(shell)))
            for shell in model.idp.full_basis
        ]
    )
    xyz_to_yzx = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    return ao_irreps.D_from_matrix(xyz_to_yzx @ rotation @ xyz_to_yzx.T)

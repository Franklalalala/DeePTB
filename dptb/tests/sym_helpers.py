"""Shared fixtures for the symmetry tests: a small float64 lem_moe_v3_edge model, crystal graphs,
rigid rotations and the rotation of packed AO-product blocks.  Not collected."""
import copy

import numpy as np
import torch
from e3nn import o3
from ase import Atoms

from dptb.data import AtomicData, _keys
from dptb.nn.build import build_model

A = 4.376698970794678  # AlMg3, L1_2 (Pm-3m), the reference crystal


YZX = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], dtype=torch.float64)
TOL = 1e-9


def almg3(a=A):
    return Atoms("AlMg3", scaled_positions=[[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
                 cell=np.eye(3) * a, pbc=True)


def p1_structure():
    rng = np.random.default_rng(7)
    at = almg3()
    at.positions = at.positions + rng.normal(scale=0.08, size=at.positions.shape)
    at.cell = at.cell.array + rng.normal(scale=0.05, size=(3, 3))
    return at


def config(lmax=4, **extra):
    irreps = "+".join("4x%d%s" % (l, "e" if l % 2 == 0 else "o") for l in range(lmax + 1))
    emb = dict(method="lem_moe_v3_edge", n_layers=2, avg_num_neighbors=12.0, r_max=4.5,
               irreps_hidden=irreps, env_embed_multiplicity=2, latent_dim=8, latent_channels=[8],
               edge_one_hot_dim=4, num_experts=1, num_shared_experts=0, top_k=1, universal=True,
               use_layer_onehot_tp=False, use_out_onehot_tp=False, use_interpolation_out=False,
               tp_radial_emb=False, mole_linear_mode="split_loop", so2_fusion_mode="staged",
               equivariant_norm_type="none")
    emb.update(extra)
    return dict(common_options={"basis": {"Al": "1s1p1d", "Mg": "1s1p"}, "overlap": False,
                                "dtype": "float64", "device": "cpu"},
                model_options={"embedding": emb, "prediction": {"method": "e3tb", "scale_type": "no_scale"}},
                train_options={}, no_check=False)


def make_model(seed=11, **extra):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    model = build_model(**config(**extra))
    return model.eval()


def graph(model, atoms):
    data = AtomicData.to_AtomicDataDict(AtomicData.from_ase(atoms, r_max=4.5))
    data = {k: (v.to(torch.float64) if torch.is_floating_point(v) else v) for k, v in data.items()}
    return model.idp(data)


def rotate(data, r):
    out = {k: v.clone() for k, v in data.items()}
    out[_keys.POSITIONS_KEY] = data[_keys.POSITIONS_KEY] @ r.T
    out[_keys.CELL_KEY] = data[_keys.CELL_KEY] @ r.T
    for k in (_keys.EDGE_VECTORS_KEY, _keys.EDGE_LENGTH_KEY):
        out.pop(k, None)
    return out


def run(model, data):
    with torch.no_grad():
        out = model(copy.deepcopy(data))
    return out[_keys.NODE_FEATURES_KEY].clone(), out[_keys.EDGE_FEATURES_KEY].clone()


def wigner(model, r):
    """Rotation of the packed AO-product blocks: every orbital-pair slice A -> D_la A D_lb^T."""
    m = YZX @ torch.as_tensor(r, dtype=torch.float64) @ YZX.T
    dim = int(model.idp.reduced_matrix_element)
    big = torch.zeros(dim, dim, dtype=torch.float64)
    for name, sl in model.idp.orbpair_maps.items():
        la, lb = ("spdfgh".index(s.strip()[-1]) for s in name.split("-"))
        da = o3.Irrep(la, (-1) ** la).D_from_matrix(m)
        db = o3.Irrep(lb, (-1) ** lb).D_from_matrix(m)
        big[sl, sl] = torch.kron(da, db)
    return big


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))





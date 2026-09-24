"""Shared model builders: the small lem_moe_v3_prior_2b model and batch, and a stub-expert
DistanceEnsembleWrapper.  Not collected; import from dptb.tests.model_helpers."""
from __future__ import annotations

import torch
import torch.nn as nn

from dptb.data import _keys
from dptb.nn.build import DistanceEnsembleWrapper, build_model


def _embedding(only2b: bool, **extra):
    cfg = {
        "method": "lem_moe_v3_prior_2b",
        "only2b": only2b,
        "prior_kind": "na_cf",
        "prior_merge_mode": "concat",
        "prior_init_scope": "both",
        "n_layers": 2,
        "avg_num_neighbors": 2.0,
        "r_max": 4.0,
        "irreps_hidden": "4x0e+4x1o+4x1e+4x2e",
        "env_embed_multiplicity": 2,
        "latent_dim": 8,
        "latent_channels": [8],
        "edge_one_hot_dim": 4,
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
        "equivariant_norm_type": "none",
    }
    cfg.update(extra)
    return cfg


def _build(only2b: bool, **extra):
    return build_model(
        common_options={
            "basis": {"H": "1s", "O": "1s1p"},
            "overlap": False,
            "dtype": "float32",
            "device": "cpu",
        },
        model_options={
            "embedding": _embedding(only2b, **extra),
            "prediction": {"method": "e3tb", "scale_type": "no_scale"},
        },
        train_options={},
        no_check=False,
    )


def _data(model):
    h = model.idp.chemical_symbol_to_type["H"]
    o = model.idp.chemical_symbol_to_type["O"]
    rme = int(model.idp.reduced_matrix_element)
    torch.manual_seed(0)
    return {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]],
            dtype=torch.float32,
        ),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.ATOM_TYPE_KEY: torch.tensor([[h], [o]], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.tensor(
            [model.idp.bond_to_type["H-O"], model.idp.bond_to_type["O-H"]],
            dtype=torch.long,
        ),
        _keys.NODE_P23_KEY: torch.randn(2, rme, dtype=torch.float32),
        _keys.EDGE_P2_KEY: torch.randn(2, rme, dtype=torch.float32),
    }


def _has_grad(module):
    return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())


def _params_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


class _StubExpert(nn.Module):
    """A stand-in expert that returns a fixed (cloned) output dict.

    Mirrors the real experts, whose ``forward`` returns the plain
    ``AtomicDataDict`` (``Dict[str, Tensor]``).  Also exposes the attributes the
    wrapper's ``__init__`` reads from ``experts[0]``.
    """

    def __init__(self, outputs=None):
        super().__init__()
        self._outputs = outputs or {}
        self.name = "stub_expert"
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model_options = {}

    def forward(self, batch):
        return {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in self._outputs.items()
        }


def _make_wrapper(strict=False, num_experts=2, expert_outputs=None):
    if expert_outputs is None:
        experts = [_StubExpert() for _ in range(num_experts)]
    else:
        experts = [_StubExpert(o) for o in expert_outputs]
    ranges = [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)][:num_experts]
    return DistanceEnsembleWrapper(
        experts=experts, distance_ranges=ranges, strict_output_spec=strict
    )

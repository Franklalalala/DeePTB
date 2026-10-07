"""LEM message weighting, activation recompute, and cutoff metadata."""

import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.activation_recompute import checkpoint_function_call
from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
from dptb.nn.embedding.lem_moe_v3 import (
    UpdateNode,
    _cosine_cutoff_per_edge,
    _polynomial_cutoff_per_edge,
)
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.utils.argcheck import activation_recompute_options


def test_update_node_can_bypass_env_message_weighting():
    class DummyTP(torch.nn.Module):
        def forward(self, x, r, mole_globals, latents=None, wigner_D_all=None):
            return torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device), wigner_D_all

    class DummyUpdate:
        def __init__(self):
            self.irreps_in = o3.Irreps("1x0e")
            self.irreps_out = o3.Irreps("1x0e")
            self.edge_irreps_in = o3.Irreps("1x0e")
            self.tp = DummyTP()
            self.activation = torch.nn.Identity()
            self.lin_post = torch.nn.Identity()
            self.post_activation_expert_mixer = None
            self.node_norm = None
            self.edge_norm = None
            self.env_sum_normalizations = torch.tensor(1.0)
            self.res_update = False
            self.use_layer_onehot_tp = False
            self.edge_message_env_weight = False
            self.env_embed_mlps = self._unexpected_env_weight
            self._env_weighter = self._unexpected_env_weight

        def _unexpected_env_weight(self, *args, **kwargs):
            raise AssertionError("env message weighting should be bypassed")

    dummy = DummyUpdate()
    out = UpdateNode.forward(
        dummy, latents=torch.zeros(2, 1), node_features=torch.zeros(2, 1), edge_features=torch.zeros(2, 1),
        atom_type=torch.zeros(2, 1, dtype=torch.long), node_onehot=torch.zeros(2, 1),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), edge_vector=torch.zeros(2, 3),
        cutoff_coeffs=torch.ones(2), active_edges=torch.tensor([0, 1], dtype=torch.long),
        wigner_D_all=None, mole_globals=None,
    )

    assert torch.allclose(out, torch.ones(2, 1))


# ---------------------------------------------------------------------------
# Activation recompute
# ---------------------------------------------------------------------------


def test_checkpoint_function_call_recomputes_during_backward():
    calls = {"n": 0}

    def block(x):
        calls["n"] += 1
        return torch.sin(x).square()

    x = torch.randn(8, requires_grad=True)
    y = checkpoint_function_call(block, x, enabled=True, use_reentrant=False)

    assert calls["n"] == 1
    y.sum().backward()
    assert calls["n"] == 2
    assert x.grad is not None


def test_activation_recompute_argcheck_accepts_moe_target():
    normalized = activation_recompute_options().normalize_value({
        "enabled": True, "targets": ["lem_moe_v3_tp"], "checkpoint_node_tp": True,
        "checkpoint_edge_tp": False, "use_reentrant": False, "preserve_rng_state": False,
    })

    assert normalized["enabled"] is True
    assert normalized["targets"] == ["lem_moe_v3_tp"]
    assert normalized["checkpoint_edge_tp"] is False


# ---------------------------------------------------------------------------
# MultiTrainer active-edge split-size bookkeeping, and per-edge cutoff helpers
# ---------------------------------------------------------------------------

def test_lem_active_edge_split_sizes_use_cpu_edge_slices():
    batch = {"__slices__": {_keys.EDGE_INDEX_KEY: torch.tensor([0, 3, 5, 9])}}
    active_edges = torch.tensor([0, 2, 4, 5, 8], dtype=torch.long)
    assert MultiTrainer._lem_active_edge_split_sizes(batch, active_edges) == (2, 1, 2)

    class BatchLike:
        __slices__ = {_keys.EDGE_INDEX_KEY: [0, 3, 5, 9]}

    assert MultiTrainer._lem_active_edge_split_sizes(BatchLike(), [0, 2, 4, 5, 8]) == (2, 1, 2)


def test_lem_split_sizes_are_reattached_as_cpu_tensor_after_to_dict():
    batch_dict = {}
    cpu_batch = {_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY: torch.tensor([2, 0, 3], dtype=torch.long)}

    MultiTrainer._attach_lem_cpu_split_sizes(batch_dict, cpu_batch)

    split_sizes = batch_dict[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY]
    assert torch.is_tensor(split_sizes)
    assert split_sizes.device.type == "cpu"
    assert split_sizes.dtype == torch.long
    torch.testing.assert_close(split_sizes, torch.tensor([2, 0, 3], dtype=torch.long))


def test_lem_precompute_metadata_is_cleared_before_reuse():
    batch = {
        _keys.LEM_ACTIVE_EDGES_KEY: torch.tensor([0], dtype=torch.long),
        _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY: torch.tensor([1], dtype=torch.long),
        _keys.LEM_CUTOFF_COEFFS_KEY: torch.tensor([1.0]),
    }

    MultiTrainer._clear_lem_precompute_metadata(batch)

    assert _keys.LEM_ACTIVE_EDGES_KEY not in batch
    assert _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY not in batch
    assert _keys.LEM_CUTOFF_COEFFS_KEY not in batch


def test_per_edge_cutoffs_match_old_loop_semantics():
    edge_length = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
    bond_r_max = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    valid = torch.tensor([True, True, False, True])

    polynomial = _polynomial_cutoff_per_edge(edge_length, bond_r_max, p=6.0)
    cosine = _cosine_cutoff_per_edge(edge_length, bond_r_max, r_start_cos_ratio=0.8)

    polynomial_ref = torch.stack([polynomial_cutoff(edge_length[i:i + 1], bond_r_max[i:i + 1], p=6.0).flatten()[0]
                                   for i in range(edge_length.numel())])
    cosine_ref = torch.stack([cosine_cutoff(edge_length[i:i + 1], bond_r_max[i:i + 1], r_start_cos_ratio=0.8).flatten()[0]
                               for i in range(edge_length.numel())])

    torch.testing.assert_close(polynomial * valid, polynomial_ref * valid)
    torch.testing.assert_close(cosine * valid, cosine_ref * valid)

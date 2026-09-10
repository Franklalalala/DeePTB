"""Focused behavioral checks for whole-stack LoopSCF recurrence."""

from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from torch import nn

from dptb.data import AtomicDataDict as A
from dptb.nnops.loopscf.stack import (
    GraphExitGate,
    StackBridge,
    _wrap_stack,
    adaptive_objective,
    exit_distribution,
    graph_hamiltonian_losses,
    predict_until_exit,
    validate_stack_checkpoint,
)


def test_true_early_exit_stops_computation_and_checkpoint_marker():
    calls = []
    model = SimpleNamespace(training=False)
    model._wm_prepare = lambda batch: {
        "contexts": [{"encoder_calls": 1, "stack_calls": 0}]
    }

    def one(batch, k, n, e, state):
        calls.append(k)
        state["contexts"][0]["stack_calls"] += 1
        return {"_stack_logit": torch.zeros(1), "prediction": k}

    model._wm_one_k = one
    out = predict_until_exit(
        model,
        {A.BATCH_KEY: torch.zeros(2, dtype=torch.long)},
        max_steps=3,
        quantile=0.5,
    )
    assert (
        calls == [1, 2] and out["_exit_step"] == 2 and out["_stack_counts"] == [(1, 2)]
    )
    good = {"_stack_version": torch.tensor(1), "_stack_strategy_code": torch.tensor(0)}
    validate_stack_checkpoint(good)
    with pytest.raises(ValueError):
        validate_stack_checkpoint(good, "detach")
    with pytest.raises(ValueError):
        validate_stack_checkpoint({"_stack_version": torch.tensor(2)})


def test_stack_bridge_is_equivariant_and_zero_init_gets_gradients():
    irreps = o3.Irreps("2x0e + 1x1o + 1x2e")
    bridge = StackBridge(irreps, irreps, latent_dim=3).double()
    torch.manual_seed(20260910)
    li = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    lo = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    ni = torch.randn(4, irreps.dim, dtype=torch.float64, requires_grad=True)
    ei = torch.randn(5, irreps.dim, dtype=torch.float64, requires_grad=True)
    no = torch.randn(4, irreps.dim, dtype=torch.float64, requires_grad=True)
    eo = torch.randn(5, irreps.dim, dtype=torch.float64, requires_grad=True)

    zl, zn, ze = bridge((li, ni, ei), (lo, no, eo, None))
    torch.testing.assert_close(zl, li, atol=0, rtol=0)
    torch.testing.assert_close(zn, ni, atol=0, rtol=0)
    torch.testing.assert_close(ze, ei, atol=0, rtol=0)

    loss = (
        (zl - 1.3).square().sum()
        + (zn + 0.7).square().sum()
        + (ze - 0.2).square().sum()
    )
    loss.backward()
    assert bridge.latent_mix.grad.norm() > 0
    assert bridge.node.weight.grad.norm() > 0
    assert bridge.edge.weight.grad.norm() > 0

    with torch.no_grad():
        bridge.latent_mix.normal_()
        bridge.node.weight.normal_()
        bridge.edge.weight.normal_()
    rotation = o3.angles_to_matrix(
        *[torch.tensor(v, dtype=torch.float64) for v in (0.17, -0.43, 0.91)]
    )
    D = irreps.D_from_matrix(rotation)
    base = bridge(
        (li.detach(), ni.detach(), ei.detach()),
        (lo.detach(), no.detach(), eo.detach(), None),
    )
    rotated = bridge(
        (li.detach(), ni.detach() @ D.T, ei.detach() @ D.T),
        (lo.detach(), no.detach() @ D.T, eo.detach() @ D.T, None),
    )
    torch.testing.assert_close(rotated[0], base[0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(rotated[1], base[1] @ D.T, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(rotated[2], base[2] @ D.T, atol=1e-10, rtol=1e-10)


def test_exit_distribution_uniform_mass_and_joint_gate_gradient():
    torch.manual_seed(9)
    K = 4
    logits = torch.zeros(3, K, dtype=torch.float64, requires_grad=True)
    probs = exit_distribution(logits)
    torch.testing.assert_close(probs.sum(-1), torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(probs, torch.full_like(probs, 1.0 / K))

    gate = GraphExitGate("2x0e + 1x1o").double()
    node = torch.randn(5, gate.irreps.dim, dtype=torch.float64)
    batch = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    gate_logits = torch.stack([gate(node + 0.03 * k, batch) for k in range(K)], dim=-1)
    gate_probs = exit_distribution(gate_logits)
    losses = torch.tensor(
        [[0.7, 0.3, 0.6, 0.8], [1.1, 0.9, 0.4, 0.5], [0.2, 0.5, 0.9, 1.3]],
        dtype=torch.float64,
    )
    objective, entropy = adaptive_objective(losses, gate_probs, beta=0.02)
    assert entropy.shape == (3,)
    objective.backward()
    final = gate.net[-1]
    assert final.weight.grad is not None and torch.isfinite(final.weight.grad).all()
    assert final.weight.grad.norm() > 0


def test_adaptive_objective_rejects_shape_and_negative_beta():
    losses = torch.ones(2, 3)
    probs = torch.full((2, 3), 1 / 3)
    with pytest.raises(ValueError, match="matching"):
        adaptive_objective(losses[:, :2], probs, beta=0.0)
    with pytest.raises(ValueError, match="beta"):
        adaptive_objective(losses, probs, beta=-1e-3)


def _fake_idp(width=3):
    return SimpleNamespace(
        mask_to_nrme=torch.tensor([[True, True, False]], dtype=torch.bool),
        mask_to_erme=torch.tensor([[True, False, True]], dtype=torch.bool),
        has_soc=False,
        nextham_uureal_mask=False,
    )


def _graph_slice(ref, pred_node, pred_edge, graph):
    batch = ref[A.BATCH_KEY]
    edge_index = ref[A.EDGE_INDEX_KEY]
    node_mask = batch == graph
    edge_mask = batch[edge_index[0]] == graph
    node_ids = node_mask.nonzero(as_tuple=False).flatten()
    remap = torch.full((batch.numel(),), -1, dtype=torch.long)
    remap[node_ids] = torch.arange(node_ids.numel())
    local_edge_index = remap[edge_index[:, edge_mask]]
    data = {
        A.ATOM_TYPE_KEY: ref[A.ATOM_TYPE_KEY][node_mask],
        A.EDGE_TYPE_KEY: ref[A.EDGE_TYPE_KEY][edge_mask],
        A.NODE_FEATURES_KEY: pred_node[node_mask],
        A.EDGE_FEATURES_KEY: pred_edge[edge_mask],
        A.EDGE_INDEX_KEY: local_edge_index,
    }
    target = {
        A.ATOM_TYPE_KEY: ref[A.ATOM_TYPE_KEY][node_mask],
        A.EDGE_TYPE_KEY: ref[A.EDGE_TYPE_KEY][edge_mask],
        A.NODE_FEATURES_KEY: ref[A.NODE_FEATURES_KEY][node_mask],
        A.EDGE_FEATURES_KEY: ref[A.EDGE_FEATURES_KEY][edge_mask],
        A.EDGE_INDEX_KEY: local_edge_index,
    }
    return data, target


def test_graph_hamiltonian_losses_match_individual_hamil_loss_abs_per_graph():
    from dptb.nnops.loss import HamilLossAbs

    idp = _fake_idp()
    ref = {
        A.BATCH_KEY: torch.tensor([0, 0, 1], dtype=torch.long),
        A.ATOM_TYPE_KEY: torch.zeros(3, dtype=torch.long),
        A.EDGE_TYPE_KEY: torch.zeros(4, dtype=torch.long),
        A.EDGE_INDEX_KEY: torch.tensor([[0, 1, 2, 2], [1, 0, 2, 2]], dtype=torch.long),
        A.NODE_FEATURES_KEY: torch.tensor(
            [[0.0, 1.0, 5.0], [2.0, -1.0, 7.0], [1.5, 0.5, 9.0]],
            dtype=torch.float64,
        ),
        A.EDGE_FEATURES_KEY: torch.tensor(
            [[1.0, 4.0, 0.0], [0.5, 6.0, -0.5], [2.0, 8.0, 1.0], [1.5, 3.0, -1.0]],
            dtype=torch.float64,
        ),
    }
    pred_node = ref[A.NODE_FEATURES_KEY] + torch.tensor(
        [[0.2, -0.1, 100.0], [-0.3, 0.4, 200.0], [0.6, -0.5, 300.0]],
        dtype=torch.float64,
    )
    pred_edge = ref[A.EDGE_FEATURES_KEY] + torch.tensor(
        [
            [0.7, 100.0, -0.2],
            [-0.4, 200.0, 0.8],
            [0.1, 300.0, -0.6],
            [-0.5, 400.0, 0.3],
        ],
        dtype=torch.float64,
    )

    actual = graph_hamiltonian_losses(pred_node, pred_edge, ref, idp)
    loss = HamilLossAbs(idp=idp)
    expected = []
    for graph in (0, 1):
        data_g, ref_g = _graph_slice(ref, pred_node, pred_edge, graph)
        expected.append(loss(data_g, ref_g))
    torch.testing.assert_close(actual, torch.stack(expected))


class _StackTestLayer(nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps_in = irreps
        self.irreps_out = irreps
        self.latent_bias = nn.Parameter(
            torch.tensor([0.13, -0.07], dtype=torch.float64)
        )
        self.node_bias = nn.Parameter(torch.tensor([0.21, -0.11], dtype=torch.float64))
        self.edge_bias = nn.Parameter(torch.tensor([-0.17, 0.19], dtype=torch.float64))
        self.calls = 0

    def forward(
        self,
        latents,
        node_features,
        edge_features,
        node_onehot,
        edge_index,
        edge_vector,
        atom_type,
        cutoff_coeffs,
        active_edges,
        edge_one_hot,
        wigner_D_all,
        mole_globals,
        node_batch=None,
    ):
        self.calls += 1
        return (
            latents + self.latent_bias,
            node_features + self.node_bias + 0.1 * latents.mean(0),
            edge_features + self.edge_bias + 0.1 * latents.mean(0),
            wigner_D_all,
        )


class _StackTestEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.irreps = o3.Irreps("2x0e")
        self.layers = nn.ModuleList([_StackTestLayer(self.irreps)])
        self.stack_bridge = StackBridge(self.irreps, self.irreps, latent_dim=2).double()
        self.stack_exit = GraphExitGate(self.irreps).double()
        self.encoder_calls = 0

    def _apply_rme_output_heads(
        self, node_features, edge_features, node_one_hot, edge_one_hot
    ):
        if node_features.shape[0] != node_one_hot.shape[0]:
            raise AssertionError(
                "stack readout must pad hidden nodes before the full-node head"
            )
        return (
            node_features + 0.01 * node_one_hot[:, :2],
            edge_features + 0.02 * edge_one_hot[:, :2],
        )

    def forward(self, data):
        self.encoder_calls += 1
        active = data["active_edges"]
        latents = data["latent_input"]
        node_features = data["node_input"]
        edge_features = data["edge_input"].index_select(0, active)
        node_one_hot = data[A.NODE_ATTRS_KEY]
        edge_one_hot = data["edge_one_hot"].index_select(0, active)
        edge_index = data[A.EDGE_INDEX_KEY]
        edge_vector = data["edge_vector"]
        atom_type = data[A.ATOM_TYPE_KEY]
        cutoff_coeffs = data["cutoff_coeffs"]
        node_batch = data[A.BATCH_KEY][: node_features.shape[0]]

        data[A.EDGE_OVERLAP_KEY] = latents
        wigner = None
        for layer in self.layers:
            latents, node_features, edge_features, wigner = layer(
                latents,
                node_features,
                edge_features,
                node_one_hot[: node_features.shape[0]],
                edge_index,
                edge_vector,
                atom_type,
                cutoff_coeffs,
                active,
                edge_one_hot,
                wigner,
                SimpleNamespace(),
                node_batch,
            )
        if node_features.shape[0] < node_one_hot.shape[0]:
            pad = node_features.new_zeros(
                node_one_hot.shape[0] - node_features.shape[0], node_features.shape[1]
            )
            node_features = torch.cat([node_features, pad], dim=0)
        out_node, out_edge = self._apply_rme_output_heads(
            node_features, edge_features, node_one_hot, edge_one_hot
        )
        return {
            **data,
            A.NODE_FEATURES_KEY: out_node,
            A.EDGE_FEATURES_KEY: out_edge.new_zeros(
                edge_index.shape[1], out_edge.shape[-1]
            ).index_copy(0, active, out_edge),
            "latest_latents": latents,
        }


def _stack_test_batch():
    return {
        "latent_input": torch.tensor(
            [[0.2, -0.1], [0.4, 0.3], [-0.5, 0.7]],
            dtype=torch.float64,
            requires_grad=True,
        ),
        "node_input": torch.tensor(
            [[1.0, -2.0], [0.5, 0.25]], dtype=torch.float64, requires_grad=True
        ),
        "edge_input": torch.tensor(
            [[0.3, -0.7], [0.8, 0.1], [-0.2, 0.6]],
            dtype=torch.float64,
            requires_grad=True,
        ),
        A.NODE_ATTRS_KEY: torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64
        ),
        "edge_one_hot": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]], dtype=torch.float64
        ),
        A.BATCH_KEY: torch.tensor([0, 0, 0], dtype=torch.long),
        A.ATOM_TYPE_KEY: torch.zeros(3, dtype=torch.long),
        A.EDGE_INDEX_KEY: torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.long),
        "edge_vector": torch.randn(3, 3, dtype=torch.float64),
        "cutoff_coeffs": torch.ones(3, dtype=torch.float64),
        "active_edges": torch.tensor([2, 0], dtype=torch.long),
        A.NODE_H0_KEY: torch.zeros(3, 2, dtype=torch.float64),
        A.EDGE_H0_KEY: torch.zeros(3, 2, dtype=torch.float64),
    }


def test_wrap_stack_clones_initial_latents_and_zero_init_preserves_k1_to_k3():
    emb = _StackTestEmbedding()
    _wrap_stack(emb)
    emb._stack_ctx = {"strategy": "bptt"}
    batch = _stack_test_batch()

    first = emb(batch)
    expected_node = first[A.NODE_FEATURES_KEY].detach().clone()
    expected_edge = first[A.EDGE_FEATURES_KEY].detach().clone()
    first[A.EDGE_OVERLAP_KEY].add_(1000.0)
    first[A.NODE_FEATURES_KEY].add_(1000.0)

    second = emb(batch)
    third = emb(batch)
    assert emb.encoder_calls == 1
    assert emb.layers[0].calls == 3
    assert emb._stack_ctx["encoder_calls"] == 1
    assert emb._stack_ctx["stack_calls"] == 3
    assert third[A.NODE_FEATURES_KEY].shape == expected_node.shape
    torch.testing.assert_close(second[A.NODE_FEATURES_KEY], expected_node)
    torch.testing.assert_close(second[A.EDGE_FEATURES_KEY], expected_edge)
    torch.testing.assert_close(third[A.NODE_FEATURES_KEY], expected_node)
    torch.testing.assert_close(third[A.EDGE_FEATURES_KEY], expected_edge)

    loss = (
        third[A.NODE_FEATURES_KEY].square().sum()
        + third[A.EDGE_FEATURES_KEY].square().sum()
    )
    loss.backward()
    assert (
        batch["latent_input"].grad is not None
        and torch.isfinite(batch["latent_input"].grad).all()
    )
    assert (
        batch["node_input"].grad is not None
        and torch.isfinite(batch["node_input"].grad).all()
    )
    assert (
        batch["edge_input"].grad is not None
        and torch.isfinite(batch["edge_input"].grad).all()
    )

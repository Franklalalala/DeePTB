from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from e3nn import o3

from dptb.data import _keys

from test_lem_pair_common import (
    ao_wigner,
    complete_directed_edges,
    fp64_default,
    model,
)
from test_lem_pair_flow_contract import _flow_model
from test_residual_ao_block_ode import (
    _b_flow,
    _b_record,
    _rotate_canvas_blocks,
)


_ROW_OUTPUT_KEYS = (
    _keys.NODE_HAMILTONIAN_KEY,
    _keys.EDGE_HAMILTONIAN_KEY,
    _keys.EDGE_OVERLAP_KEY,
)


@contextmanager
def deterministic_fp64():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    with fp64_default():
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(previous)


def _o_graph(pair_model, positions):
    positions = torch.as_tensor(positions, dtype=torch.float64)
    edge_index = complete_directed_edges(int(positions.shape[0]))
    n_edges = int(edge_index.shape[1])
    h0_dim = pair_model.idp.reduced_matrix_element
    return {
        _keys.POSITIONS_KEY: positions,
        _keys.EDGE_INDEX_KEY: edge_index,
        _keys.ATOM_TYPE_KEY: torch.zeros(
            (positions.shape[0], 1), dtype=torch.long
        ),
        _keys.EDGE_TYPE_KEY: torch.full(
            (n_edges,),
            pair_model.idp.bond_to_type["O-O"],
            dtype=torch.long,
        ),
        _keys.NODE_H0_KEY: torch.zeros(
            (positions.shape[0], h0_dim), dtype=torch.float64
        ),
        _keys.EDGE_H0_KEY: torch.zeros(
            (n_edges, h0_dim), dtype=torch.float64
        ),
    }


def _batch_graphs(graphs):
    positions = []
    edge_indices = []
    atom_types = []
    edge_types = []
    node_h0 = []
    edge_h0 = []
    batches = []
    node_offset = 0
    for graph_index, graph in enumerate(graphs):
        n_nodes = int(graph[_keys.POSITIONS_KEY].shape[0])
        positions.append(graph[_keys.POSITIONS_KEY])
        edge_indices.append(graph[_keys.EDGE_INDEX_KEY] + node_offset)
        atom_types.append(graph[_keys.ATOM_TYPE_KEY])
        edge_types.append(graph[_keys.EDGE_TYPE_KEY])
        node_h0.append(graph[_keys.NODE_H0_KEY])
        edge_h0.append(graph[_keys.EDGE_H0_KEY])
        batches.append(torch.full((n_nodes,), graph_index, dtype=torch.long))
        node_offset += n_nodes
    return {
        _keys.POSITIONS_KEY: torch.cat(positions),
        _keys.EDGE_INDEX_KEY: torch.cat(edge_indices, dim=1),
        _keys.ATOM_TYPE_KEY: torch.cat(atom_types),
        _keys.EDGE_TYPE_KEY: torch.cat(edge_types),
        _keys.NODE_H0_KEY: torch.cat(node_h0),
        _keys.EDGE_H0_KEY: torch.cat(edge_h0),
        _keys.BATCH_KEY: torch.cat(batches),
    }


def _clone_tensors(data):
    return {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in data.items()
    }


def _assert_equal_or_fp64_floor(reference, actual):
    if torch.equal(reference, actual):
        return 0.0
    drift = (reference - actual).abs().max().item()
    assert drift <= 1.0e-12
    return drift


def test_g1_batch_partition_invariance_for_all_active_and_real_split_graphs():
    with deterministic_fp64():
        pair_model = model(mp_cutoff=1.0)
        graph_a = _o_graph(
            pair_model,
            [[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.0, 0.35, 0.0]],
        )
        graph_b = _o_graph(
            pair_model,
            [[0.0, 0.0, 0.0], [0.70, 0.0, 0.0], [2.10, 0.2, 0.0]],
        )
        with torch.no_grad():
            standalone = pair_model(_clone_tensors(graph_a))
            ab = pair_model(_batch_graphs([graph_a, graph_b]))
            ba = pair_model(_batch_graphs([graph_b, graph_a]))

        n_a = int(graph_a[_keys.POSITIONS_KEY].shape[0])
        e_a = int(graph_a[_keys.EDGE_INDEX_KEY].shape[1])
        n_b = int(graph_b[_keys.POSITIONS_KEY].shape[0])
        e_b = int(graph_b[_keys.EDGE_INDEX_KEY].shape[1])
        drifts = []
        drifts.append(
            _assert_equal_or_fp64_floor(
                standalone[_keys.NODE_HAMILTONIAN_KEY],
                ab[_keys.NODE_HAMILTONIAN_KEY][:n_a],
            )
        )
        drifts.append(
            _assert_equal_or_fp64_floor(
                standalone[_keys.NODE_HAMILTONIAN_KEY],
                ba[_keys.NODE_HAMILTONIAN_KEY][n_b : n_b + n_a],
            )
        )
        for key in (_keys.EDGE_HAMILTONIAN_KEY, _keys.EDGE_OVERLAP_KEY):
            drifts.append(
                _assert_equal_or_fp64_floor(
                    standalone[key],
                    ab[key][:e_a],
                )
            )
            drifts.append(
                _assert_equal_or_fp64_floor(
                    standalone[key],
                    ba[key][e_b : e_b + e_a],
                )
            )
        print(
            "lem_pair_g1_batch_partition "
            f"max_abs={max(drifts):.16e} "
            f"bit_exact={all(drift == 0.0 for drift in drifts)}"
        )


def test_g2_dual_block_ode_is_sensitive_to_non_mp_h0_and_residual_state():
    with deterministic_fp64():
        torch.manual_seed(20260724)
        pair_model = _flow_model(
            mp_cutoff=0.5,
            pair_refine_enable=False,
        )
        flow = _b_flow(pair_model.idp, dtype=torch.float64)
        raw, _, _ = _b_record(pair_model.idp, dtype=torch.float64, seed=31)
        model_data, _, _ = flow.prepare_batch(
            copy.deepcopy(raw),
            copy.deepcopy(raw),
            t=torch.tensor([0.41], dtype=torch.float64),
        )
        edge_row = 0
        edge_index = model_data[_keys.EDGE_INDEX_KEY]
        edge_length = (
            model_data[_keys.POSITIONS_KEY][edge_index[0, edge_row]]
            - model_data[_keys.POSITIONS_KEY][edge_index[1, edge_row]]
        ).norm()
        assert edge_length.item() > pair_model.embedding.mp_cutoff

        with torch.no_grad():
            reference = pair_model(_clone_tensors(model_data))
            h0_data = _clone_tensors(model_data)
            h0_data[_keys.EDGE_H0_KEY][edge_row] += torch.linspace(
                0.01,
                0.01 * h0_data[_keys.EDGE_H0_KEY].shape[-1],
                h0_data[_keys.EDGE_H0_KEY].shape[-1],
                dtype=torch.float64,
            )
            h0_output = pair_model(h0_data)

            residual_data = _clone_tensors(model_data)
            residual_data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY][edge_row] += (
                torch.linspace(
                    0.01,
                    0.16,
                    16,
                    dtype=torch.float64,
                ).reshape(4, 4)
            )
            residual_output = pair_model(residual_data)

        h0_delta = (
            h0_output[_keys.EDGE_HAMILTONIAN_KEY][edge_row]
            - reference[_keys.EDGE_HAMILTONIAN_KEY][edge_row]
        ).abs().max().item()
        residual_delta = (
            residual_output[_keys.EDGE_HAMILTONIAN_KEY][edge_row]
            - reference[_keys.EDGE_HAMILTONIAN_KEY][edge_row]
        ).abs().max().item()
        assert h0_delta > 0.0
        assert residual_delta > 0.0

        grad_data = _clone_tensors(model_data)
        edge_h0 = grad_data[_keys.EDGE_H0_KEY].requires_grad_(True)
        edge_residual = grad_data[
            _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY
        ].requires_grad_(True)
        grad_data[_keys.EDGE_H0_KEY] = edge_h0
        grad_data[_keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY] = edge_residual
        grad_output = pair_model(grad_data)
        torch.manual_seed(71)
        probe = torch.randn_like(
            grad_output[_keys.EDGE_HAMILTONIAN_KEY][edge_row]
        )
        objective = (
            grad_output[_keys.EDGE_HAMILTONIAN_KEY][edge_row] * probe
        ).sum()
        grad_h0, grad_residual = torch.autograd.grad(
            objective, (edge_h0, edge_residual)
        )
        h0_grad_norm = grad_h0[edge_row].norm().item()
        residual_grad_norm = grad_residual[edge_row].norm().item()
        assert h0_grad_norm > 0.0
        assert residual_grad_norm > 0.0

        torch.manual_seed(17)
        rotation = o3.rand_matrix(dtype=torch.float64)
        d_ao = ao_wigner(pair_model.embedding, rotation)
        # Isolate the residual block-state covariance with H0 held at the
        # valid zero tensor. H0 row sensitivity is asserted independently
        # above; its mapper-order RME rotation is covered by codec/projector
        # tests in test_residual_ao_block_ode.py.
        equivariant_data = _clone_tensors(model_data)
        equivariant_data[_keys.NODE_H0_KEY].zero_()
        equivariant_data[_keys.EDGE_H0_KEY].zero_()
        with torch.no_grad():
            equivariant_reference = pair_model(
                _clone_tensors(equivariant_data)
            )
        rotated_data = _clone_tensors(equivariant_data)
        rotated_data[_keys.POSITIONS_KEY] = (
            equivariant_data[_keys.POSITIONS_KEY]
            @ rotation.transpose(-1, -2)
        )
        for key in (
            _keys.NODE_SPATIAL_RESIDUAL_BLOCKS_KEY,
            _keys.EDGE_SPATIAL_RESIDUAL_BLOCKS_KEY,
        ):
            rotated_data[key] = _rotate_canvas_blocks(
                equivariant_data[key], d_ao
            )
        with torch.no_grad():
            rotated_output = pair_model(rotated_data)
        expected = _rotate_canvas_blocks(
            equivariant_reference[_keys.EDGE_HAMILTONIAN_KEY], d_ao
        )
        equivariance_drift = (
            rotated_output[_keys.EDGE_HAMILTONIAN_KEY] - expected
        ).abs().max().item()
        print(
            "lem_pair_g2_state_contract "
            f"h0_delta={h0_delta:.16e} "
            f"residual_delta={residual_delta:.16e} "
            f"h0_grad={h0_grad_norm:.16e} "
            f"residual_grad={residual_grad_norm:.16e} "
            f"equivariance={equivariance_drift:.16e}"
        )
        assert equivariance_drift <= 1.0e-9


def test_g3_all_trainable_parameters_participate_in_both_batch_regimes():
    with deterministic_fp64():
        pair_model = model(mp_cutoff=1.0)
        all_active = _o_graph(
            pair_model,
            [[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.0, 0.35, 0.0]],
        )
        real_split = _o_graph(
            pair_model,
            [[0.0, 0.0, 0.0], [0.70, 0.0, 0.0], [2.10, 0.2, 0.0]],
        )
        for label, data in (("all_active", all_active), ("real_split", real_split)):
            pair_model.zero_grad(set_to_none=True)
            output = pair_model(_clone_tensors(data))
            loss = sum(output[key].square().sum() for key in _ROW_OUTPUT_KEYS)
            loss.backward()
            missing = [
                name
                for name, parameter in pair_model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            assert not missing, f"{label} missing gradients: {missing}"
            projection_grads = [
                parameter.grad
                for name, parameter in pair_model.named_parameters()
                if "dual_cutoff_edge_context_projection" in name
            ]
            readout_grads = [
                parameter.grad
                for name, parameter in pair_model.named_parameters()
                if "dual_cutoff_pair_readout" in name
                and parameter.requires_grad
            ]
            assert projection_grads and all(grad is not None for grad in projection_grads)
            assert readout_grads and all(grad is not None for grad in readout_grads)


def test_g4_empty_mp_subgraphs_are_finite_and_backward_safe():
    with deterministic_fp64():
        pair_model = model(mp_cutoff=0.2)
        zero_graph = _o_graph(
            pair_model,
            [[0.0, 0.0, 0.0], [1.20, 0.0, 0.0], [0.0, 1.30, 0.0]],
        )
        active_graph = _o_graph(
            pair_model,
            [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.0, 0.09, 0.0]],
        )
        cases = (
            ("single_zero", zero_graph, (0,)),
            ("mixed", _batch_graphs([zero_graph, active_graph]), (0, 6)),
            ("all_zero", _batch_graphs([zero_graph, zero_graph]), (0, 0)),
        )
        observed_splits = []

        def capture(_module, inputs):
            observed_splits.append(tuple(inputs[-1].split_sizes))

        handle = pair_model.layers[0].edge_update.register_forward_pre_hook(capture)
        try:
            for label, data, expected_splits in cases:
                pair_model.zero_grad(set_to_none=True)
                output = pair_model(_clone_tensors(data))
                for key in _ROW_OUTPUT_KEYS:
                    assert torch.isfinite(output[key]).all(), label
                loss = sum(output[key].square().sum() for key in _ROW_OUTPUT_KEYS)
                loss.backward()
                assert observed_splits[-1] == expected_splits
        finally:
            handle.remove()

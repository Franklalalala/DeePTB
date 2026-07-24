from __future__ import annotations

import torch

from dptb.data import _keys
from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0
from dptb.nn.embedding.lem_pair import LemPair

from test_lem_pair_common import fp64_default, model_options, molecule_data


def _h0_model(*, seed: int = 20260724, **overrides) -> LemMoEV3H0:
    options = model_options()
    options.pop("mp_avg_num_neighbors")
    options.update(overrides)
    torch.manual_seed(seed)
    return LemMoEV3H0(**options).eval()


def _manual_slice_rms(features, irreps):
    values = []
    for term_slice in irreps.slices():
        block = features[..., term_slice]
        values.append((block.square().sum() / block.numel()).sqrt())
    return torch.stack(values)


def test_head_input_rms_default_off_has_no_key_and_is_bit_exact():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        implicit = _h0_model()
        implicit_rng = torch.random.get_rng_state().clone()
        explicit = _h0_model(log_head_input_rms=False)
        explicit_rng = torch.random.get_rng_state().clone()

        assert torch.equal(implicit_rng, explicit_rng)
        assert implicit.state_dict().keys() == explicit.state_dict().keys()
        assert all(
            torch.equal(implicit.state_dict()[key], explicit.state_dict()[key])
            for key in implicit.state_dict()
        )
        implicit_out = implicit(molecule_data(implicit))
        explicit_out = explicit(molecule_data(explicit))
        assert "head_input_rms" not in implicit_out
        assert "head_input_rms" not in explicit_out
        for key in (
            _keys.NODE_HAMILTONIAN_KEY,
            _keys.EDGE_HAMILTONIAN_KEY,
            _keys.EDGE_OVERLAP_KEY,
        ):
            assert torch.equal(implicit_out[key], explicit_out[key])


def test_head_input_rms_enabled_matches_manual_irreps_slice_calculation():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        model = _h0_model(log_head_input_rms=True)
        captured = {}

        def capture_node(_module, inputs):
            captured["node"] = inputs[0].detach().clone()

        def capture_edge(_module, inputs):
            captured["edge"] = inputs[0].detach().clone()

        node_handle = model.out_node.register_forward_pre_hook(capture_node)
        edge_handle = model.out_edge.register_forward_pre_hook(capture_edge)
        try:
            output = model(molecule_data(model))
        finally:
            node_handle.remove()
            edge_handle.remove()

        telemetry = output["head_input_rms"]
        expected_node = _manual_slice_rms(
            captured["node"], model.out_node.irreps_in
        )
        expected_edge = _manual_slice_rms(
            captured["edge"], model.out_edge.irreps_in
        )
        torch.testing.assert_close(
            telemetry["node"], expected_node, rtol=0.0, atol=1.0e-15
        )
        torch.testing.assert_close(
            telemetry["edge"], expected_edge, rtol=0.0, atol=1.0e-15
        )
        assert torch.equal(
            telemetry["node_l"],
            torch.tensor(
                [ir.l for _, ir in model.out_node.irreps_in], dtype=torch.long
            ),
        )
        assert torch.equal(
            telemetry["edge_l"],
            torch.tensor(
                [ir.l for _, ir in model.out_edge.irreps_in], dtype=torch.long
            ),
        )
        assert not telemetry["node"].requires_grad
        assert not telemetry["edge"].requires_grad
        print(
            "hb0_head_input_rms "
            f"node={telemetry['node'].tolist()} "
            f"edge={telemetry['edge'].tolist()}"
        )


def test_lem_pair_endpoint_head_input_rms_inherits_through_super():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        options = model_options()
        options.update(condition_source="endpoints", log_head_input_rms=True)
        torch.manual_seed(53)
        model = LemPair(**options).eval()
        output = model(molecule_data(model))
        assert "head_input_rms" in output
        assert torch.isfinite(output["head_input_rms"]["node"]).all()
        assert torch.isfinite(output["head_input_rms"]["edge"]).all()


def test_lem_pair_refine_head_input_rms_forwards_three_tuple_through_mro():
    with fp64_default():
        torch.use_deterministic_algorithms(True)
        options = model_options()
        options.update(
            pair_refine_enable=True,
            pair_refine_rank=4,
            pair_refine_identity_init=True,
            log_head_input_rms=True,
        )
        torch.manual_seed(59)
        model = LemPair(**options).eval()
        output = model(molecule_data(model))
        assert "head_input_rms" in output
        assert torch.isfinite(output["head_input_rms"]["node"]).all()
        assert torch.isfinite(output["head_input_rms"]["edge"]).all()

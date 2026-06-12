import torch
from e3nn.o3 import Linear

from dptb.nn.embedding.lem_moe_v3 import _eval_safe_e3nn_linear


def test_eval_safe_e3nn_linear_matches_e3nn_and_keeps_graph_transient(monkeypatch):
    monkeypatch.setenv("DPTB_E3NN_LINEAR_EVAL_REENTRANT_SAFE", "1")
    linear = Linear(
        "2x0e + 1x1o + 1x2e",
        "3x0e + 1x1o + 1x2e",
        internal_weights=True,
        shared_weights=True,
        biases=True,
    ).eval()
    features = torch.randn(7, linear.irreps_in.dim)

    with torch.no_grad():
        expected = linear(features)
        actual = _eval_safe_e3nn_linear(linear, features)
        actual_again = _eval_safe_e3nn_linear(linear, features)

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual_again, expected)
    assert "_dptb_eval_safe_graph" in linear.__dict__
    assert "_dptb_eval_safe_graph" not in linear._modules
    assert "_dptb_eval_safe_graph" not in linear.state_dict()

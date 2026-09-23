import types

import pytest

torch = pytest.importorskip("torch")

from dptb.nn.embedding.lem_moe_v3_edge import LemMoEV3Edge  # noqa: E402
from dptb.nn.tensor_product_moe_v3 import MOLELinear  # noqa: E402


class _EchoRouter:
    """Routing coefficients = the router input; dense routing, no top-k metadata."""

    def __call__(self, features, sizes=None):
        return features, features.new_zeros(()), features.new_zeros(())

    def last_topk(self):
        return None, None


def _edge_route(coefficients, bond_type, *, unique_types, compact_min_edges):
    owner = types.SimpleNamespace(
        edge_router_in_features=coefficients.shape[1],
        edge_router_top1_mode="legacy",
        edge_router_prior_activate=False,
        edge_router_unique_types=unique_types,
        edge_moe_compact_dispatch=True,
        edge_moe_compact_min_edges=compact_min_edges,
        num_experts=coefficients.shape[1],
        router=_EchoRouter(),
    )
    return LemMoEV3Edge._make_edge_moe_globals(owner, coefficients, bond_type)[0]


@pytest.mark.parametrize("mode", ["split_loop", "indexed_ref"])
@pytest.mark.parametrize(
    "unique_types,compact_min_edges",
    [(False, 16384), (True, 16384), (True, 0)],
    ids=["per-edge", "unique-expanded", "unique-compact"],
)
def test_non_pa_edge_routes_mix_each_edge_with_its_own_coefficients(mode, unique_types, compact_min_edges):
    torch.manual_seed(20260923)
    dtype = torch.float64
    layer = MOLELinear(3, 2, num_experts=4, num_shared_experts=1, bias=True, mole_linear_mode=mode).to(dtype)
    bond_type = torch.tensor([2, 0, 2, 1, 0])
    per_type = torch.softmax(torch.randn(3, 4, dtype=dtype), dim=-1)
    coefficients = per_type.index_select(0, bond_type)  # equal rows within a bond type
    x = torch.randn(5, 3, dtype=dtype)

    got = layer(x, _edge_route(coefficients, bond_type, unique_types=unique_types,
                               compact_min_edges=compact_min_edges))

    weight = torch.einsum("ne,eoi->noi", coefficients, layer.weight_experts) + layer.weight_shared.sum(0)
    bias = coefficients @ layer.bias_experts + layer.bias_shared.sum(0)
    torch.testing.assert_close(got, torch.einsum("noi,ni->no", weight, x) + bias)

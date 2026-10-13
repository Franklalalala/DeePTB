"""Embedding registry: built-in methods, extensions and unknown names."""
import pytest

import dptb.nn.embedding as embedding

BUILTIN_METHODS = {
    "none", "se2", "baseline", "mpnn", "deeph-e3", "e3baseline_6",
    "e3baseline_nonlocal", "trinity", "lem", "slem", "lem_prior", "slem_prior",
    "unitb", "lem_moe_v3", "lem_moe_v3_edge", "lem_moe_v3_edge_h0",
    "lem_moe_v3_h0", "lem_moe_v3_prior", "lem_moe_v3_prior_2b",
    "lem_moe_v3_edge_prior_2b",
}


def test_identity_registry_constructs_and_returns_the_input():
    model = embedding.Embedding(method="none", dtype="float64", device="cpu")
    data = {"sentinel": object()}

    assert isinstance(model, embedding.Identity)
    assert model(data) is data
    assert list(model.parameters()) == []


def test_package_exports_and_builtin_methods():
    for name in embedding.__all__:
        assert hasattr(embedding, name), name
    assert set(embedding.Embedding._register.keys()) == BUILTIN_METHODS


def test_extension_registration_and_unknown_method(monkeypatch):
    monkeypatch.setattr(embedding.Embedding, "_register", type(embedding.Embedding._register)())

    @embedding.Embedding.register("example_extension")
    class Extension:
        def __init__(self, value):
            self.value = value

    assert embedding.Embedding("example_extension", value=3).value == 3
    with pytest.raises(ValueError, match="not registered"):
        embedding.Embedding("missing_method")

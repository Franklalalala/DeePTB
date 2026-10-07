"""Public embedding registry and package exports."""

import dptb.nn.embedding as embedding


def test_identity_registry_constructs_and_returns_the_input():
    model = embedding.Embedding(method="none", dtype="float64", device="cpu")
    data = {"sentinel": object()}

    assert isinstance(model, embedding.Identity)
    assert model(data) is data
    assert list(model.parameters()) == []


def test_package_exports_resolve_and_upstream_methods_remain_registered():
    for name in embedding.__all__:
        assert hasattr(embedding, name), name

    expected = {
        "baseline", "deeph-e3", "e3baseline_6", "e3baseline_nonlocal",
        "none", "mpnn", "se2", "trinity", "lem", "slem",
    }
    assert expected <= set(embedding.Embedding._register.keys())

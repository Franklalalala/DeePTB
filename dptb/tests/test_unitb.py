"""UniTB construction, historical checkpoints, and equivariance with AO priors."""
import copy
import io
import logging

import pytest
import torch
from e3nn import o3
from torch.nn import functional as F

from dptb.data import _keys
from dptb.nn.build import build_model
from dptb.nn.embedding.prior_common import H0InitLayer
from dptb.nn.embedding.unitb import UniTB
from dptb.nn.embedding.unitb_options import unitb_options
from dptb.nn.pdq_moe import PDQMoE, PDQMoERouting
from dptb.tests._requires import requires_cuda, requires_module
from dptb.tests.sym_helpers import wigner


@pytest.fixture(autouse=True)
def reference_backend(monkeypatch):
    """These CPU tests select their backend independently of deployment flags."""
    monkeypatch.delenv("DPTB_SO2_FUSION_MODE", raising=False)
    monkeypatch.delenv("DPTB_SO2_FUSE_M_CUBLAS", raising=False)


@pytest.fixture
def default_dtype(request):
    previous = torch.get_default_dtype()
    dtype = getattr(request, "param", torch.float64)
    torch.set_default_dtype(dtype)
    yield dtype
    torch.set_default_dtype(previous)


def _options(kind, **overrides):
    options = dict(
        n_layers=2, n_radial_basis=4, r_max=4.0,
        irreps_hidden="4x0e+4x1o+4x2e", avg_num_neighbors=2.0,
        # Retain the production scalar span used by time conditioning.
        env_embed_multiplicity=10, latent_dim=8, latent_channels=[8],
        edge_one_hot_dim=4, tp_radial_channels=[4],
        use_layer_onehot_tp=False, use_out_onehot_tp=False,
        mole_linear_mode="split_loop", so2_fusion_mode="staged",
    )
    # Expert counts, routing, shared cores and time conditioning otherwise use
    # the public model defaults. A dense model changes only the expert count.
    if kind == "dense":
        options["num_experts"] = 1
    options.update(overrides)
    return options


def _model(options, *, method="unitb", dtype="float32"):
    return build_model(
        common_options=dict(basis={"H": "1s", "O": "1s1p"}, overlap=False,
                            dtype=dtype, device="cpu"),
        model_options=dict(embedding=dict(method=method, **options),
                           prediction=dict(method="e3tb", scale_type="no_scale")),
        train_options={}, no_check=False,
    )


def _batch(model, dtype=torch.float32):
    generator = torch.Generator().manual_seed(91)
    h = model.idp.chemical_symbol_to_type["H"]
    o = model.idp.chemical_symbol_to_type["O"]
    dim = model.idp.reduced_matrix_element
    return {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [-0.3, 0.9, 0.4]], dtype=dtype),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        _keys.ATOM_TYPE_KEY: torch.tensor([[h], [o], [o]]),
        _keys.EDGE_TYPE_KEY: torch.tensor([
            model.idp.bond_to_type[key] for key in ["H-O", "O-H", "H-O", "O-H", "O-O", "O-O"]]),
        _keys.NODE_H0_KEY: torch.randn(3, dim, dtype=dtype, generator=generator),
        _keys.EDGE_H0_KEY: torch.randn(6, dim, dtype=dtype, generator=generator),
    }


@pytest.mark.parametrize("kind", ["dense", "x1"])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64], indirect=True)
def test_minimal_models_use_production_defaults_and_train(kind, default_dtype):
    dtype = default_dtype
    torch.manual_seed(81)
    model = _model(_options(kind), dtype=str(dtype).removeprefix("torch."))
    embedding = model.embedding
    assert isinstance(embedding, UniTB)
    assert isinstance(embedding.init_layer, H0InitLayer)
    linears = [module for module in embedding.modules() if isinstance(module, PDQMoE)]
    assert linears
    if kind == "dense":
        assert embedding.num_experts == 1 and embedding.router.top_k == 1
        assert all(layer.num_shared_experts == 0 for layer in linears)
        assert all(layer.mole_expert_parameterization == "full" for layer in linears)
        assert not embedding.edge_router_prior_activate
    else:
        assert embedding.num_experts == 4 and embedding.router.top_k == 2
        assert all(layer.num_shared_experts == 1 for layer in linears)
        assert all(layer.mole_expert_parameterization == "shared_core" for layer in linears)
        assert embedding.edge_router_prior_activate and embedding.edge_router_prior_cg
        assert embedding.router.logit_kind == "cosine"
        assert embedding.use_flow_time_embedding
    output = model(_batch(model, dtype))
    values = [output[key] for key in (_keys.NODE_FEATURES_KEY, _keys.EDGE_FEATURES_KEY)]
    assert all(torch.isfinite(value).all() for value in values)
    sum(value.square().mean() for value in values).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    if kind == "x1":
        assert any(layer.core_experts.grad is not None and torch.count_nonzero(layer.core_experts.grad)
                   for layer in linears)
        assert torch.count_nonzero(embedding.router.net[0].weight.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
@pytest.mark.parametrize("route", ["graph", "edge", "selected_experts"])
def test_grouped_pdq_cpu_routes_preserve_math(
        monkeypatch, dtype, parameterization, route):
    """Graph/expert dispatch accepts reference backends; edge fallback is exact."""
    from dptb.nn import pdq_moe

    calls = []

    def reference_gemm(x, ptr, weight, *, graph_routing=False):
        calls.append((x.device.type, x.dtype))
        bounds = ptr.tolist()
        return torch.cat([F.linear(x[start:end], w)
                          for start, end, w in zip(bounds[:-1], bounds[1:], weight)], dim=0)

    monkeypatch.setattr(pdq_moe, "_grouped_gemm", reference_gemm)
    torch.manual_seed(84)
    linear = PDQMoE(7, 5, num_experts=4, num_shared_experts=1,
                   mole_expert_parameterization=parameterization,
                   mole_expert_rank=3, mole_linear_mode="cublas_grouped").to(dtype)
    x = torch.randn(6, 2, 7, dtype=dtype, requires_grad=True)
    # Interleaved routes and an unused expert exercise sorting and empty groups.
    indices = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 1], [0, 2], [1, 0]])
    logits = torch.randn(6, 2, dtype=dtype, requires_grad=True)
    values = logits.softmax(-1)
    coefficients = values.new_zeros(6, 4).scatter(1, indices, values)
    if parameterization == "shared_core":
        bank = torch.einsum("or,ers,is->eoi", linear.basis_left,
                            linear.core_experts, linear.basis_right)
    else:
        bank = linear.weight_experts

    if route == "selected_experts":
        got = linear.apply_experts(x, indices[:, 0], include_shared_experts=True)
        want = torch.einsum("n...i,noi->n...o", x, bank[indices[:, 0]])
        want = want + linear.bias_experts[indices[:, 0]].unsqueeze(1)
    else:
        graph_index = torch.tensor([2, 0, 5, 3, 1, 4]) if route == "graph" else torch.arange(6)
        routing = PDQMoERouting(coefficients=coefficients, topk_indices=indices,
                               topk_values=values, graph_index=graph_index,
                               activation_space=route == "edge", coefficients_sum_to_one=True)
        got = linear(x, routing)
        mixed = torch.einsum("ne,eoi->noi", coefficients, bank)
        bias = coefficients @ linear.bias_experts
        want = torch.einsum("n...i,noi->n...o", x, mixed[graph_index])
        want = want + bias[graph_index].unsqueeze(1)
    want = want + F.linear(x, linear.weight_shared.sum(0), linear.bias_shared.sum(0))
    tolerance = 3e-6 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)
    if route != "edge":
        assert calls and all(call == ("cpu", dtype) for call in calls)
    inputs = [x, *linear.parameters()]
    if route != "selected_experts":
        inputs.append(logits)
    probe = torch.randn_like(got)
    actual_gradients = torch.autograd.grad((got * probe).sum(), inputs, retain_graph=True)
    expected_gradients = torch.autograd.grad((want * probe).sum(), inputs)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@requires_cuda
@requires_module("cuequivariance")
@requires_module("cuequivariance_torch")
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64], indirect=True)
def test_pdq_cueq_two_graph_cache_preserves_outputs_and_gradients(monkeypatch, default_dtype):
    """Both a new cuEq kernel and its cached reuse preserve graph routing."""
    # cuEq creates its normalization coefficients with the default dtype.
    dtype = default_dtype
    monkeypatch.setenv("DPTB_CUEQ_CACHE_DIAG", "1")
    monkeypatch.delenv("DPTB_MOLE_LINEAR_MODE", raising=False)
    torch.manual_seed(85)
    linear = PDQMoE(7, 5, num_experts=4, num_shared_experts=1,
                   mole_linear_mode="cueq_indexed_linear").to(device="cuda", dtype=dtype)
    state = {key: value.clone() for key, value in linear.state_dict().items()}
    x = torch.randn(6, 2, 7, device="cuda", dtype=dtype, requires_grad=True)
    coefficients = torch.randn(2, 4, device="cuda", dtype=dtype).softmax(-1).requires_grad_()
    graph_index = torch.tensor([1, 0, 1, 0, 0, 1], device="cuda")
    routing = PDQMoERouting(coefficients=coefficients, graph_index=graph_index)
    probe = torch.randn(6, 2, 5, device="cuda", dtype=dtype)
    inputs = [x, coefficients, *linear.parameters()]
    tolerance = 2e-4 if dtype == torch.float32 else 1e-10
    assert not linear._cueq_indexed_linear_cache
    cached = None
    for _ in range(2):
        actual = linear(x, routing)
        weight = torch.einsum("ge,eoi->goi", coefficients, linear.weight_experts)
        weight = weight + linear.weight_shared.sum(0)
        bias = coefficients @ linear.bias_experts + linear.bias_shared.sum(0)
        expected = torch.einsum("noi,nki->nko", weight[graph_index], x)
        expected = expected + bias[graph_index].unsqueeze(1)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        actual_grads = torch.autograd.grad((actual * probe).sum(), inputs)
        expected_grads = torch.autograd.grad((expected * probe).sum(), inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)
        assert len(linear._cueq_indexed_linear_cache) == 1
        kernel = next(iter(linear._cueq_indexed_linear_cache.values()))
        if cached is not None:
            assert kernel is cached
        cached = kernel
    assert linear.state_dict().keys() == state.keys()
    for key, value in linear.state_dict().items():
        assert torch.equal(value, state[key])


@pytest.mark.parametrize("kind", ["dense", "x1"])
def test_legacy_method_and_option_aliases_restore_checkpoint_exactly(kind, caplog):
    options = _options(kind, expert_parameterization="full" if kind == "dense" else "pdq_moe",
                       expert_rank=3, router_input="onehot_prior", router_gate="renorm")
    historical = unitb_options(options)
    caplog.set_level(logging.INFO)
    torch.manual_seed(82)
    old = _model(historical, method="lem_moe_v3_edge_h0").eval()
    new = _model(options).eval()
    assert old.state_dict().keys() == new.state_dict().keys()
    buffer = io.BytesIO()
    torch.save(old.state_dict(), buffer)
    buffer.seek(0)
    new.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items())
    data = _batch(old)
    with torch.no_grad():
        expected = old(copy.deepcopy(data))
        actual = new(copy.deepcopy(data))
    for key in (_keys.NODE_FEATURES_KEY, _keys.EDGE_FEATURES_KEY):
        assert torch.equal(actual[key], expected[key])
    assert any(record.levelno == logging.INFO and "lem_moe_v3_edge_h0" in record.message
               for record in caplog.records)


def test_old_h0_marker_mapping_is_strict_and_preserves_legacy_math():
    old = _model(_options("dense", h0_ao_cg=False)).eval()
    state = old.state_dict()
    del state["embedding.init_layer.h0_ao_cg_version"]
    restored = _model(_options("dense", h0_ao_cg=False)).eval()
    restored.load_state_dict(state, strict=True)
    data = _batch(old)
    with torch.no_grad():
        expected, actual = old(copy.deepcopy(data)), restored(copy.deepcopy(data))
    for key in (_keys.NODE_FEATURES_KEY, _keys.EDGE_FEATURES_KEY):
        assert torch.equal(actual[key], expected[key])
    with pytest.raises(RuntimeError):
        _model(_options("dense", h0_ao_cg=True)).load_state_dict(state, strict=True)


@pytest.mark.parametrize("kind", ["dense", "x1"])
def test_float64_rotation_with_ao_priors(default_dtype, kind):
    torch.manual_seed(83)
    model = _model(_options(kind), dtype="float64").eval()
    data = _batch(model, torch.float64)
    if kind == "x1":
        # Give the H0 descriptor an active role; its columns start at zero when
        # initializing a new router for historical checkpoint compatibility.
        with torch.no_grad():
            model.embedding.router.net[0].weight[:, model.embedding.edge_one_hot_dim:].normal_(std=0.1)
    rotation = o3.rand_matrix(dtype=torch.float64)
    representation = wigner(model, rotation)
    rotated = copy.deepcopy(data)
    rotated[_keys.POSITIONS_KEY] = data[_keys.POSITIONS_KEY] @ rotation.T
    for key in (_keys.NODE_H0_KEY, _keys.EDGE_H0_KEY):
        rotated[key] = data[key] @ representation.T
    with torch.no_grad():
        expected, actual = model(copy.deepcopy(data)), model(rotated)
    for key in (_keys.NODE_FEATURES_KEY, _keys.EDGE_FEATURES_KEY):
        reference = expected[key] @ representation.T
        torch.testing.assert_close(actual[key], reference, rtol=0.0, atol=1e-10)

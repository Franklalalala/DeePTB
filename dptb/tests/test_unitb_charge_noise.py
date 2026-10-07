"""Charge conservation, strict legacy restoration and supervised prior draws."""
import copy

import pytest
import torch

from dptb.data.transforms import OrbitalMapper
from dptb.nn.charge_head import ChargeHead, normalize_charge_options
from dptb.nn.charge_response import isolated_gamma
from dptb.nn.response_shift_head import ResponseShiftHead
from dptb.nn.shift_head import normalize_shift_options
from dptb.nnops.prior_noise import PriorNoiseAugmentation
from dptb.tests._requires import requires_cuda
from dptb.tests.prior_noise_helpers import FakeIDP, make_batch


def assert_bytes_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.detach().contiguous().reshape(-1).view(torch.uint8),
                       expected.detach().contiguous().reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("method", ["unitb", "lem_moe_v3_edge_h0"])
def test_charge_model_shares_mapper_with_embedding(method):
    from dptb.nn.build import build_model
    from dptb.tests.shift_head_helpers import config

    cfg = config(mode="atom")
    cfg["model_options"]["embedding"]["method"] = method
    cfg["model_options"]["shift_head"]["response"] = {
        "kind": "qeq", "qeq_local": True, "canonical_onsite": False,
        "auxiliary_weight": 0.0, "output_scale": 0.1,
    }
    model = build_model(**cfg)
    # Lazy orbital-map initialization must be visible to both the embedding
    # and the charge head, which assemble features in the same orbital layout.
    assert model.embedding.idp is model.idp
    assert model.shift_head.idp is model.idp
    model.embedding.idp.get_orbital_maps()
    assert model.shift_head.idp.orbital_maps is model.embedding.idp.orbital_maps


def charge_fixture(response):
    idp = OrbitalMapper({"H": "1s", "He": "1s"}, method="e3tb", has_soc=True,
                       nextham_uureal_mask=True)
    options = normalize_charge_options({"response": response})
    head = ChargeHead(idp, "3x0e+1x1o", options, dtype=torch.float64)
    with torch.no_grad():
        for param in head.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    return head, options


@pytest.mark.parametrize("local", [False, True])
def test_qeq_forms_are_neutral_and_match_their_potential_definition(local):
    head, _ = charge_fixture({"kind": "qeq", "qeq_local": local})
    net = head.response_net
    x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    types = torch.tensor([0, 1, 0])
    pos = torch.tensor([[0., 0., 0.], [1., .2, .3], [2., .1, .4]], dtype=x.dtype)
    potential, aux = net(x, types, pos=pos)
    assert float(aux["q"].sum().abs()) < 1e-14
    electrostatic = isolated_gamma(pos) @ aux["q"]
    if local:
        with torch.no_grad():
            net.kappa.zero_()
        local_potential, local_aux = net(x, types, pos=pos)
        with torch.no_grad():
            net.kappa.fill_(2.)
        potential, aux = net(x, types, pos=pos)
        assert_bytes_equal(aux["q"], local_aux["q"])
        torch.testing.assert_close(potential - local_potential,
                                   2. * net.output_scale * electrostatic, atol=1e-14, rtol=1e-12)
    else:
        torch.testing.assert_close(potential, net.output_scale * electrostatic, atol=0., rtol=0.)
    potential.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.parametrize("response", [
    {"kind": "qeq", "qeq_local": True},
    {"kind": "qeq", "qeq_local": False},
    {"kind": "context", "local_only": True},
])
def test_charge_head_strict_legacy_restore_preserves_state_and_outputs(response):
    head, options = charge_fixture(response)
    legacy = ResponseShiftHead(head.idp, "3x0e+1x1o", normalize_shift_options(options),
                               dtype=torch.float64)
    legacy.load_state_dict(head.state_dict(), strict=True)
    restored = ChargeHead(head.idp, "3x0e+1x1o", options, dtype=torch.float64)
    restored.load_state_dict(legacy.state_dict(), strict=True)
    width = head.idp.reduced_matrix_element
    data = {
        "_shift_node_features": torch.randn(3, 6, dtype=torch.float64),
        "_shift_active_edges": torch.arange(4),
        "atom_types": torch.tensor([0, 1, 0]),
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "pos": torch.tensor([[0., 0., 0.], [1., .2, .3], [2., .1, .4]], dtype=torch.float64),
        "phys_node_overlap": torch.ones(3, width, dtype=torch.float64),
        "phys_edge_overlap": torch.ones(4, width, dtype=torch.float64),
        "node_features": torch.randn(3, width, dtype=torch.float64),
        "edge_features": torch.randn(4, width, dtype=torch.float64),
    }
    old_output = legacy(copy.deepcopy(data))
    output = restored(copy.deepcopy(data))
    for key in ("node_features", "edge_features", "node_shift_delta", "edge_shift_delta"):
        assert_bytes_equal(output[key], old_output[key])
    if response["kind"] == "context":
        changed = copy.deepcopy(data)
        changed["pos"] *= 7.
        assert_bytes_equal(restored(changed)["node_features"], output["node_features"])


@pytest.mark.parametrize("response", [
    {"kind": "qeq", "context": "sublattice"},
    {"kind": "context", "local_only": False},
    {"canonical_onsite": True}, {"auxiliary_weight": 0.1}, {"detach_features": True},
])
def test_charge_entry_rejects_active_archive_response_options(response):
    with pytest.raises(ValueError):
        normalize_charge_options({"response": response})


def check_prior_draw(device, dtype, scope, reference):
    device = torch.device(device)
    options = dict(te_prior_mode="typewise", te_prior_scale_reference=reference,
                   te_prior_sigma=0.5, node_sigma=1., edge_sigma=5.,
                   te_prior_per_graph=True, t_min=0., t_max=0., t0_probability=0.,
                   detach_interpolated_h0=True)
    new = PriorNoiseAugmentation(dict(prior="te", **options),
                                 idp=FakeIDP(device=device), dtype=dtype, device=device)
    data, labels = make_batch(device=device, dtype=dtype)
    data.update(labels)
    data["expert_node_mask"] = torch.full((3,), scope != "edge", device=device, dtype=torch.bool)
    data["expert_edge_mask"] = torch.full((4,), scope != "node", device=device, dtype=torch.bool)
    rng = lambda: torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    torch.manual_seed(63)
    actual = new(data, training=True)
    assert_bytes_equal(actual["flow_time"], torch.zeros(2, dtype=dtype, device=device))
    assert set(actual) - set(data) == {"flow_time"}
    for key in set(data) - {"node_h0", "edge_h0"}:
        assert actual[key] is data[key]
    if scope != "both":
        inactive = "edge_h0" if scope == "node" else "node_h0"
        assert_bytes_equal(actual[inactive], data[inactive])
    before = rng()
    assert new(data, training=False) is data
    assert_bytes_equal(rng(), before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scope", ["both", "node", "edge"])
@pytest.mark.parametrize("reference", ["target", "residual"])
def test_prior_noise_preserves_labels_scope_and_evaluation_rng(dtype, scope, reference):
    check_prior_draw("cpu", dtype, scope, reference)


@requires_cuda
def test_prior_noise_cuda_preserves_labels_scope_and_evaluation_rng():
    check_prior_draw("cuda", torch.float32, "node", "target")

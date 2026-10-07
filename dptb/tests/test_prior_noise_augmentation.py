"""Training prior perturbations preserve supervised labels and orbital masks."""
import pytest
import torch
from dptb.nnops.prior_noise import PriorNoiseAugmentation, assert_prior_noise_keys_reach_model
from dptb.tests.prior_noise_helpers import FakeIDP, make_batch


def sampler(**options):
    return PriorNoiseAugmentation(dict(prior="te", te_prior_mode="typewise", **options),
                                  idp=FakeIDP(device="cpu"))


def test_noise_keeps_labels_and_invalid_orbitals_clean():
    data, labels = make_batch(device="cpu", dtype=torch.float32)
    data.update(labels)
    noise = sampler()
    torch.manual_seed(7)
    out = noise(data, training=True)
    for label, mask in (("node", noise.idp.mask_to_nrme[data["atom_types"]]),
                        ("edge", noise.idp.mask_to_erme[data["edge_type"]])):
        assert out[label + "_features"] is data[label + "_features"]
        assert torch.equal(out[label + "_h0"][~mask], data[label + "_h0"][~mask])
        assert not torch.equal(out[label + "_h0"][mask], data[label + "_h0"][mask])


@pytest.mark.parametrize("options", [dict(t_max=.1), dict(mode="full"),
                                      dict(te_prior_mode="block"), dict(node_h0_key="node_features")])
def test_invalid_noise_contract_rejected(options):
    with pytest.raises(ValueError):
        defaults = dict(prior="te", te_prior_mode="typewise")
        defaults.update(options)
        PriorNoiseAugmentation(defaults, idp=FakeIDP(device="cpu"))


def test_noise_input_keys_must_reach_the_embedding():
    class Consumer(torch.nn.Module):
        h0_node_key = "node_h0"
        h0_edge_key = "edge_h0"
    assert_prior_noise_keys_reach_model(sampler(), Consumer())
    with pytest.raises(ValueError):
        assert_prior_noise_keys_reach_model(sampler(node_h0_key="unused_prior"), Consumer())

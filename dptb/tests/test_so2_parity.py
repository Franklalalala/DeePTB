"""CPU float64 O(3) regression tests, including non-natural output parity."""
import copy
import json
import os

import pytest
import torch
from e3nn import o3

from dptb.nn.so2_parity import parity_masks
from dptb.nn.tensor_product import SO2LinearCached as DenseSO2
from dptb.nn.tensor_product_moe_v3 import SO2_Linear, MOLEGlobals
from dptb.tests.sym_helpers import almg3, p1_structure, make_model, graph, rotate, run, wigner, rel, YZX


@pytest.fixture(autouse=True)
def float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.mark.parametrize("atoms", [almg3, p1_structure])
@pytest.mark.parametrize("norm", ["none", "merged_rms"])
def test_background_o3(atoms, norm):
    model = make_model(so2_parity="enforce", equivariant_norm_type=norm,
                       use_layer_onehot_tp=True, use_out_onehot_tp=True)
    data = graph(model, atoms())
    base = run(model, data)
    for name, r in [("rotation", o3.rand_matrix()), ("inversion", -torch.eye(3)),
                    ("mirror", torch.diag(torch.tensor([-1., 1., 1.])) )]:
        actual = run(model, rotate(data, r))
        d = wigner(model, r)
        for kind, a, b in zip(("node", "edge"), actual, base):
            expected = b @ d.T
            absolute = float((a - expected).abs().max())
            relative = rel(a, expected)
            print(f"{atoms.__name__}/{norm}/{name}/{kind}: rel={relative:.3e} max={absolute:.3e}")
            assert relative < 1e-9 and absolute < 1e-9


@pytest.mark.parametrize("backend", ["staged", "streamed_m_major_ref", "streamed_m_major_cueq",
                                     "streamed_m_major_fused_p0", "dense"])
def test_mixed_parity_so2(backend):
    torch.manual_seed(73)
    irreps = o3.Irreps("2x0e+1x0o+2x1o+1x1e+2x2e+1x2o")
    x, edges = torch.randn(7, irreps.dim), torch.randn(7, 3)
    if backend == "dense":
        layer = DenseSO2(irreps, irreps, so2_parity="enforce", so2_m_linear_mode="standard")
        forward = lambda a, b: layer(a, b)[0]
    else:
        layer = SO2_Linear(irreps, irreps, num_experts=2, num_shared_experts=1,
                           so2_parity="enforce", mole_linear_mode="split_loop", so2_fusion_mode=backend)
        routing = MOLEGlobals(coefficients=torch.tensor([[.3, .7]]),
                              split_sizes=(7,))
        forward = lambda a, b: layer(a, b, routing)[0]
    base = forward(x, edges)
    for r in (o3.rand_matrix(), -torch.eye(3), torch.diag(torch.tensor([-1., 1., 1.]))):
        d = irreps.D_from_matrix(YZX @ r @ YZX.T)
        actual, expected = forward(x @ d.T, edges @ r.T), base @ d.T
        assert rel(actual, expected) < 1e-9
        assert (actual - expected).abs().max() < 1e-9


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
def test_all_bank_readers_mask_and_optimizer(parameterization):
    irreps = o3.Irreps("2x0e+1x0o+2x1o+1x1e")
    layer = SO2_Linear(irreps, irreps, num_experts=2, num_shared_experts=1,
                       so2_parity="enforce", mole_expert_parameterization=parameterization,
                       mole_expert_rank=2, mole_linear_mode="split_loop")
    routing = MOLEGlobals(coefficients=torch.tensor([[.2, .8]]))
    opt = torch.optim.AdamW(layer.parameters(), lr=.01)
    for _ in range(2):
        opt.zero_grad()
        loss = 0
        for m, fc in [(0, layer.fc_m0), (1, layer.m_linear[0].fc)]:
            mask, bias_mask = parity_masks(irreps, irreps, m)
            banks = [fc.weight_experts, fc.weight_shared, fc._expert_weight_bank(),
                     fc._mix_expert_parameters(routing)[0], fc._routed_weight_and_bias(True)[0]]
            for bank in banks:
                assert torch.count_nonzero(bank[..., ~mask]) == 0
            x = torch.randn(3, fc.in_features)
            # Direct expert evaluation is shared by post-activation dispatch.
            loss = loss + fc.apply_experts(x, torch.tensor([0, 1, 0]), include_shared_experts=True).square().sum()
            if bias_mask is not None:
                for bias in (fc.bias_experts, fc.bias_shared, fc._mix_expert_parameters(routing)[1]):
                    assert torch.count_nonzero(bias[..., ~bias_mask]) == 0
        loss.backward()
        if parameterization == "full":
            for fc in (layer.fc_m0, layer.m_linear[0].fc):
                assert torch.count_nonzero(fc._parameters["weight_experts"].grad[..., ~fc._parity_weight_mask]) == 0
        opt.step()
    # Non-persistent masks never alter checkpoint names; strict round trips work.
    other = SO2_Linear(irreps, irreps, num_experts=2, num_shared_experts=1,
                       mole_expert_parameterization=parameterization, mole_expert_rank=2)
    other.load_state_dict(layer.state_dict(), strict=True)
    layer.load_state_dict(other.state_dict(), strict=True)


def test_default_none_and_state_compatibility():
    default = make_model()
    explicit = make_model(so2_parity="none")
    enforced = make_model(so2_parity="enforce")
    original = default.state_dict()
    assert original.keys() == enforced.state_dict().keys()
    for k, v in original.items():
        assert torch.equal(v, explicit.state_dict()[k])
        assert torch.equal(v, enforced.state_dict()[k])
    enforced.load_state_dict(original, strict=True)
    data = graph(default, p1_structure())
    for a, b in zip(run(default, data), run(explicit, data)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("option", [{"so2_parity": "invalid"},
                                    {"use_interpolation_out": True},
                                    {"hidden_edge_activation_type": "swiglu_s2"},
                                    {"ffn_hidden_factor": 2.0},
                                    {"edge_router_prior_activate": True, "num_experts": 2, "top_k": 2}])
def test_reject_incompatible_configuration(option):
    with pytest.raises(ValueError):
        make_model(**dict({"so2_parity": "enforce"}, **option))


def test_h0_legacy_contract_rejected():
    model = make_model(method="lem_moe_v3_edge_h0", so2_parity="enforce", h0_ao_cg=False)
    with pytest.raises(ValueError, match="h0_ao_cg"):
        run(model, graph(model, p1_structure()))


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_reset_edits_raw_parameters_without_losing_mask(kind):
    irreps = o3.Irreps("2x0e+2x1o")
    layer = (DenseSO2(irreps, irreps, so2_parity="enforce") if kind == "dense" else
             SO2_Linear(irreps, irreps, num_experts=2, num_shared_experts=1, so2_parity="enforce"))
    fc = layer.m_linear[0].fc
    key = "weight" if kind == "dense" else "weight_experts"
    previous = fc._parameters[key].clone()
    fc.reset_parameters()
    assert not torch.equal(previous, fc._parameters[key])
    assert torch.count_nonzero(getattr(fc, key)[..., ~fc._parity_weight_mask]) == 0
    if kind == "moe":
        previous = fc._parameters[key].clone()
        fc.scale_expert_weights_(.5)
        assert torch.equal(fc._parameters[key], previous * .5)


@pytest.mark.skipif(not os.environ.get("DPTB_PARITY_REAL_CONFIG"),
                    reason="set DPTB_PARITY_REAL_CONFIG to a real SOC validation config")
def test_real_soc_background_o3():
    """Uses the actual full basis, H0 and dataset loader; no random toy H0."""
    from dptb.data import AtomicData
    from dptb.data.build import build_dataset
    from dptb.data.dataloader import Collater
    from dptb.nn.build import build_model
    from dptb.utils.argcheck import normalize, collect_cutoffs

    with open(os.environ["DPTB_PARITY_REAL_CONFIG"], encoding="utf-8") as f:
        config = normalize(json.load(f))
    config["common_options"].update(device="cpu", dtype="float64")
    config["model_options"]["embedding"].update(so2_parity="enforce", so2_fusion_mode="staged",
                                                mole_linear_mode="split_loop")
    assert config["common_options"]["has_soc"]
    torch.manual_seed(917)
    model = build_model(model_options=config["model_options"], common_options=config["common_options"],
                        train_options=config["train_options"]).eval()
    ds = build_dataset(**collect_cutoffs(config), **config["data_options"]["validation"],
                       **config["common_options"])
    indices = [int(i) for i in os.environ.get("DPTB_PARITY_REAL_INDICES", "8,35,40,45").split(",")]
    for i in indices:
        data = AtomicData.to_AtomicDataDict(Collater()([ds[i]]))
        data = {k: v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v
                for k, v in data.items()}
        model.idp(data)
        assert "node_h0" in data and "edge_h0" in data
        base = run(model, data)
        for name, r in [("rotation", o3.rand_matrix()), ("inversion", -torch.eye(3)),
                        ("mirror", torch.diag(torch.tensor([-1., 1., 1.])) )]:
            rotated = copy.deepcopy(data)
            rotated["pos"] = data["pos"] @ r.T
            rotated["cell"] = data["cell"] @ r.T
            rotated.pop("edge_vectors", None)
            rotated.pop("edge_lengths", None)
            d = wigner(model, r)
            # Dataset AO blocks, including H0, must transform with the geometry.
            for key, value in data.items():
                if torch.is_tensor(value) and value.ndim == 2 and value.shape[-1] == d.shape[0] and value.is_floating_point():
                    rotated[key] = value @ d.T
            for kind, actual, b in zip(("node", "edge"), run(model, rotated), base):
                expected = b @ d.T
                relative, absolute = rel(actual, expected), float((actual - expected).abs().max())
                print(f"SOC/{i}/{name}/{kind}: rel={relative:.3e} max={absolute:.3e}")
                assert relative < 1e-9 and absolute < 1e-8

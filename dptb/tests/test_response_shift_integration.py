"""Full v9 integration: run in the production DeePTB CPU-capable environment.

This module intentionally skips rather than substitutes fake e3nn/data objects
when production dependencies/source modules are absent.
"""
import copy

import pytest
import torch

pytest.importorskip("e3nn", reason="full DeePTB/e3nn environment required")
pytest.importorskip("ase", reason="full DeePTB data environment required")

from dptb.nn.build import build_model
from dptb.nn.shift_head import PotentialShiftHead
from dptb.tests.shift_head_helpers import config, batch
from dptb.tests.test_r12b_shift_head import bitwise


def response_config(kind="context", scope="onsite"):
    cfg = config(mode="atom", scope=scope)
    cfg["model_options"]["shift_head"]["response"] = {
        "kind": kind, "hidden": 12, "canonical_onsite": False,
        "auxiliary_weight": 0.0, "local_only": kind == "context",
        "qeq_local": kind == "qeq", "output_scale": 0.1,
    }
    return cfg


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("scope", ["both", "onsite", "hopping"])
def test_disabled_outputs_gradients_and_rng_bitwise(moe, scope):
    cfg = config(moe=moe, scope=scope)
    torch.manual_seed(53); a = build_model(**copy.deepcopy(cfg)); arng = torch.get_rng_state().clone()
    off = copy.deepcopy(cfg)
    off["model_options"]["shift_head"] = {"mode": "off", "response": None}
    torch.manual_seed(53); b = build_model(**off); brng = torch.get_rng_state().clone()
    bitwise(arng, brng)
    assert a.state_dict().keys() == b.state_dict().keys()
    for key in a.state_dict(): bitwise(a.state_dict()[key], b.state_dict()[key])
    data = batch(a)
    x, y = a(copy.deepcopy(data)), b(copy.deepcopy(data))
    for key in ("node_features", "edge_features"): bitwise(x[key], y[key])
    for out in (x, y):
        (out["node_features"].square().sum() + out["edge_features"].square().sum()).backward()
    ap = dict(a.named_parameters())
    for name, p in b.named_parameters():
        if ap[name].grad is None: assert p.grad is None
        else: bitwise(ap[name].grad, p.grad)


def test_even_scalar_parity_guard():
    model = build_model(**response_config())
    head = PotentialShiftHead(model.idp, "2x0e+3x0o+1x1o", {"mode":"atom", "response":{"kind":"qeq"}})
    assert head.scalar_indices.tolist() == [0,1]


def test_response_assembly_rotation_equivariance_in_ao_space():
    # Real assembly method, independently encoded/decoded AO blocks, no mocked head.
    from dptb.data.interfaces.blockwise_tensor import edge_feature_slices, onsite_feature_slices
    model = build_model(**response_config(scope="both"))
    data = batch(model)
    from dptb.data.AtomicDataDict import with_edge_vectors
    data = with_edge_vectors(data, with_lengths=True)
    head = model.shift_head
    r, _ = torch.linalg.qr(torch.randn(3,3))
    if torch.linalg.det(r) < 0: r[:,0] *= -1
    u = torch.block_diag(torch.eye(2), r)  # H 2s1p: s,s,px,py,pz
    def encode(mats, node=False):
        out = torch.zeros(len(mats), model.idp.reduced_matrix_element)
        iterator = onsite_feature_slices(model.idp,"H") if node else edge_feature_slices(model.idp,"H","H")
        for si,sj,sl in iterator: out[:,sl] = mats[:,si,sj].reshape(len(mats),-1)
        return out
    nm, em = torch.randn(3,5,5), torch.randn(6,5,5)
    nm = (nm + nm.transpose(-1,-2)) / 2
    for a,b in ((0,1),(2,3),(4,5)): em[b] = em[a].T
    v = torch.tensor([[.3],[-.7],[.2]])
    data.update(phys_node_overlap=encode(nm, True), phys_edge_overlap=encode(em))
    dn,de = head.assemble(data,v)
    data.update(phys_node_overlap=encode(u@nm@u.T,True), phys_edge_overlap=encode(u@em@u.T))
    rn,re = head.assemble(data,v)
    i,j = data["edge_index"]
    torch.testing.assert_close(rn, encode(u@(v[:,:,None]*nm)@u.T,True), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(re, encode(u@(((v[i]+v[j])*.5)[:,:,None]*em)@u.T), atol=2e-6, rtol=2e-6)


def test_standard_overlap_inference_matches_physical_path_without_labels():
    cfg=response_config()
    cfg["model_options"]["shift_head"]["overlap_input"]="standard"
    model=build_model(**cfg).eval()
    data=batch(model)
    data.pop("node_features");data.pop("edge_features")
    physical=model(copy.deepcopy(data))
    standard=copy.deepcopy(data)
    standard["node_overlap"]=standard.pop("phys_node_overlap")
    standard["edge_overlap"]=standard.pop("phys_edge_overlap")
    result=model(standard)
    for key in ("node_features","edge_features"):
        bitwise(physical[key],result[key])
    # LEM's standard field is now a latent; the physical snapshot survives.
    bitwise(result["phys_edge_overlap"],data["phys_edge_overlap"])

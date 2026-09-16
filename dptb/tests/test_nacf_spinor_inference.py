"""SOC completion using real mapper layouts and deliberately mutating models."""
import copy
import json

import pytest
import torch

from dptb.data.transforms import OrbitalMapper
from dptb.nacf.spinor_completion import SOCUURealCompletion
from dptb.nacf.spinor_inference import (PreparedSOCInference, SOCResidualPair,
                                       _training_contract_config, _residual_contract)


class MutatingArm(torch.nn.Module):
    def __init__(self, mapper, node_delta, edge_delta):
        super().__init__()
        self.idp = mapper
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.node_delta, self.edge_delta = node_delta, edge_delta

    def forward(self, data):
        assert 'node_features' not in data and 'edge_features' not in data
        assert torch.count_nonzero(data['node_p23']) > 0
        # Mutate conditioning and geometry as an adversarial model would. The
        # other arm, original prior, output S and reusable graph must survive.
        data['node_p23'].zero_()
        data['pos'].add_(10)
        data['node_overlap'].zero_()
        data['node_features'] = torch.full_like(data['node_p23'], self.node_delta)
        data['edge_features'] = torch.full_like(data['edge_p2'], self.edge_delta)
        return data


@pytest.mark.parametrize('doubling', [True, False])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_paired_residual_full_soc_preserves_input_prior_overlap_and_repeated_calls(doubling,device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    compact = OrbitalMapper({'H':'1s', 'C':'1s1p'}, method='e3tb',
        has_soc=True, nextham_uureal_mask=True)
    full = OrbitalMapper(copy.deepcopy(compact.basis), method='e3tb',
        has_soc=True, full_soc_prediction=True, nextham_uureal_mask=False,
        soc_complex_doubling=doubling)
    completion = SOCUURealCompletion(compact, full,device=device)
    # Deliberately wrong unused heads reveal selecting the wrong arm.
    pair = SOCResidualPair(MutatingArm(compact, 2., -77.),
                           MutatingArm(compact, -88., 3.)).to(device)
    width = full.reduced_matrix_element
    prior = torch.arange(1, 2*width+1, dtype=torch.float64,device=device).reshape(2, width)
    if not doubling:
        prior = prior + 1j*(prior+100)
    features = {'node_p23':prior.clone(), 'edge_p2':prior.flip(0).clone(),
                'node_overlap':prior.clone()*.01, 'edge_overlap':prior.clone()*.02}
    geometry = {'pos':torch.tensor([[0.,0.,0.],[1.,0.,0.]], dtype=torch.float64),
                'edge_index':torch.tensor([[0,1],[1,0]]),
                'batch':torch.zeros((2,1),dtype=torch.long),
                'node_features':torch.full((2,compact.reduced_matrix_element),999.)}
    geometry = {k:v.to(device) for k,v in geometry.items()}
    pristine = {k:v.clone() for k,v in features.items()}
    original_geometry = {k:v.clone() for k,v in geometry.items()}
    prepared = PreparedSOCInference(pair, lambda:features, geometry, completion,
                                    ('node_p23','edge_p2'))
    expected_node, expected_edge = prior.clone(), prior.flip(0).clone()
    factor = 8 if doubling else 4
    # Derive expected slices from the real mapper, independently of the module
    # index buffers. Every untouched spin/imaginary entry is checked exactly.
    for span in full.orbpair_maps.values():
        n = (span.stop-span.start)//factor
        for offset in (0, 3*n):
            expected_node[:,span.start+offset:span.start+offset+n] += 2.
            expected_edge[:,span.start+offset:span.start+offset+n] += 3.
    for _ in range(2):
        actual = prepared()
        torch.testing.assert_close(actual['node_features'],expected_node,rtol=0,atol=0)
        torch.testing.assert_close(actual['edge_features'],expected_edge,rtol=0,atol=0)
        for field in ('node_overlap','edge_overlap'):
            torch.testing.assert_close(actual[field],pristine[field],rtol=0,atol=0)
        torch.testing.assert_close(actual['pos'],original_geometry['pos'],rtol=0,atol=0)
        for key in pristine:
            torch.testing.assert_close(features[key],pristine[key],rtol=0,atol=0)
        for key in original_geometry:
            torch.testing.assert_close(geometry[key],original_geometry[key],rtol=0,atol=0)


def test_paired_arms_reject_different_species_type_order():
    first = OrbitalMapper({'H':'1s','C':'1s1p'}, method='e3tb',
        has_soc=True,nextham_uureal_mask=True)
    second = copy.deepcopy(first)
    second.chemical_symbol_to_type = {'C':0,'H':1}
    if second.chemical_symbol_to_type == first.chemical_symbol_to_type:
        second.chemical_symbol_to_type = {'C':1,'H':0}
    with pytest.raises(ValueError,match='chemical_symbol_to_type'):
        SOCResidualPair(MutatingArm(first,1.,1.), MutatingArm(second,1.,1.))


def test_checkpoint_without_dataset_settings_requires_matching_nacf_sidecar(tmp_path):
    embedded = {'common_options':{'basis':{'H':'1s'},'has_soc':True,
                                  'nextham_uureal_mask':True,'full_soc_prediction':False},
                'model_options':{'embedding':{'method':'lem_moe_v3_edge_h0',
                                  'h0_node_key':'node_p23','h0_edge_key':'edge_p2'}}}
    with pytest.raises(ValueError,match='sidecar'):
        _training_contract_config(embedded,None)
    sidecar = copy.deepcopy(embedded)
    sidecar['data_options'] = {'train':{'prior_kind':'na_cf','target_kind':'nacfres','get_P2':True}}
    path = tmp_path/'train_config.json'
    path.write_text(json.dumps(sidecar))
    resolved = _training_contract_config(embedded,path)
    assert _residual_contract(resolved) == ('node_p23','edge_p2')
    sidecar['data_options']['train']['target_kind'] = 'h0res'
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError,match='Full-H minus NACF'):
        _residual_contract(_training_contract_config(embedded,path))
    sidecar['model_options']['embedding']['h0_node_key'] = 'node_h0'
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError,match='h0_node_key'):
        _training_contract_config(embedded,path)

"""Exercise the loader boundary: table and learned-model backends are independent."""
import copy
from types import SimpleNamespace

import pytest
import torch

from dptb.nacf import inference


@pytest.mark.parametrize('model_backend', ['checkpoint', 'reference'])
def test_loader_preserves_native_model_backend_unless_reference_is_explicit(tmp_path, monkeypatch, model_backend):
    options = {'embedding': {'method':'lem_moe_v3_prior_2b', 'prior_kind':'na_cf',
        'so2_fusion_mode':'streamed_m_major_fused_p0', 'mole_linear_mode':'cublas_grouped',
        'only2b':False, 'prior_init_scope':'both'}}
    checkpoint = tmp_path / 'model.pth'
    torch.save({'config':{'model_options':options}, 'model_state_dict':{'weight':torch.ones(1)}}, checkpoint)
    captured = {}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.idp = SimpleNamespace(has_soc=False)

    def build_model(**kwargs):
        captured['options'] = copy.deepcopy(kwargs['model_options'])
        return Model()

    def table_bank(*args, **kwargs):
        captured['table_backend'] = kwargs['backend']
        return SimpleNamespace(p2_manifest_sha256='pinned', soc=None,
                               _anchor=torch.empty(0, device=kwargs['device']))

    import dptb.nn
    monkeypatch.setattr(dptb.nn, 'build_model', build_model)
    monkeypatch.setattr(inference, 'P2TableStore', lambda path: object())
    monkeypatch.setattr(inference, 'P23VNAFactorTableStore', lambda path: object())
    monkeypatch.setattr(inference, 'OverlapTableStore', lambda path: object())
    monkeypatch.setattr(inference, 'NACFTableBank', table_bank)
    monkeypatch.setattr(inference, 'get_cutoffs_from_model_options', lambda opts:(3.,3.,3.))
    # Omitting model_backend exercises the public default, rather than spelling it out.
    kwargs = {} if model_backend == 'checkpoint' else {'model_backend':'reference'}
    predictor = inference.load_predictor(checkpoint, 'p2','p23','overlap','pinned',
                                         device='cpu', backend='torch', **kwargs)
    expected = copy.deepcopy(options)
    overrides = {}
    if model_backend == 'reference':
        overrides = {'so2_fusion_mode':'streamed_m_major_ref', 'mole_linear_mode':'split_loop'}
        expected['embedding'].update(overrides)
    assert captured['options'] == expected
    assert captured['table_backend'] == 'torch'
    assert predictor.runtime_model_overrides == overrides
    assert not predictor.model.training


def test_loader_rejects_unknown_model_backend_before_loading_checkpoint():
    with pytest.raises(ValueError, match='model_backend'):
        inference.load_predictor('absent.pth','p2','p23','overlap','pinned',model_backend='automatic_fallback')

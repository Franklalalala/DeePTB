"""The geometry CLI must complete both SOC arms, never use the compact scalar path."""
from types import SimpleNamespace
import pytest
from dptb.nacf import cli, spinor_inference


def options(**extra):
    args=dict(checkpoint='onsite.pth',hopping_checkpoint='hopping.pth',soc='soc',
        p2='p2',p23='p23',overlap='overlap',expected_p2_sha256='pin',device='cpu',backend='torch',
        model_backend='checkpoint',onsite_config='onsite.json',hopping_config='hopping.json',
        soc_ry_to_ev=13.605693122994,p23_missing_policy='error',expected_p23_sha256=None)
    args.update(extra)
    return SimpleNamespace(**args)


def test_cli_routes_soc_pair_and_training_contract_to_completion_loader(monkeypatch):
    captured={}
    sentinel=object()
    def paired(*args,**kwargs):
        captured.update(args=args,kwargs=kwargs)
        return sentinel
    def compact(*args,**kwargs):
        pytest.fail('SOC must not use the scalar loader')
    monkeypatch.setattr(spinor_inference,'load_soc_predictor',paired)
    monkeypatch.setattr(cli,'load_predictor',compact)
    assert cli.load_cli_predictor(options()) is sentinel
    assert captured['args']==('onsite.pth','hopping.pth')
    assert captured['kwargs']['onsite_config']=='onsite.json'
    assert captured['kwargs']['hopping_config']=='hopping.json'
    assert captured['kwargs']['soc']=='soc'
    assert captured['kwargs']['ry_to_ev']==13.605693122994


@pytest.mark.parametrize('changes,message',[
    ({'hopping_checkpoint':None},'hopping-checkpoint'),
    ({'soc':None},'require --soc'),
    ({'model_backend':'reference'},'checkpoint model backend'),
])
def test_cli_refuses_incomplete_or_changed_soc_contract_before_loading(changes,message):
    with pytest.raises(ValueError,match=message):cli.load_cli_predictor(options(**changes))

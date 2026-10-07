"""train_options.init_seed changes the initial parameters; 0 keeps them bitwise."""
import copy, json, os
import pytest
import torch

CFG = os.environ.get('DPTB_INIT_SEED_TEST_CONFIG', '')
pytestmark = pytest.mark.skipif(not os.path.exists(CFG), reason='set DPTB_INIT_SEED_TEST_CONFIG to a reference model config')


def build(init_seed):
    from dptb.utils.argcheck import normalize
    from dptb.nn.build import build_model
    cfg = normalize(json.load(open(CFG)))
    co = copy.deepcopy(cfg['common_options'])
    co.update(device='cpu', dtype='float32')
    mo = copy.deepcopy(cfg['model_options'])
    mo['embedding'].update(so2_fusion_mode='staged', mole_linear_mode='split_loop')
    tr = copy.deepcopy(cfg['train_options'])
    if init_seed is not None:
        tr['init_seed'] = init_seed
    return build_model(checkpoint=None, model_options=mo, common_options=co, train_options=tr).state_dict()


def test_init_seed_default_is_bitwise_and_nonzero_changes_parameters():
    ref, zero, one = build(None), build(0), build(1)
    floats = [k for k, v in ref.items() if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 1]
    assert all(torch.equal(ref[k], zero[k]) for k in ref)
    changed = [k for k in floats if not torch.equal(ref[k], one[k])]
    assert len(changed) > len(floats) // 2, (len(changed), len(floats))

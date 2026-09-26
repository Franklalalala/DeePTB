"""Small model and byte comparison helpers for chemical readout tests."""
import os
import torch
from dptb.tests.shift_head_helpers import config


def chemical_config(mode='chemical_core', device=None, moe=False, scope='both'):
    cfg = config(device=device or os.environ.get('R14B_TEST_DEVICE', 'cpu'), moe=moe, scope=scope)
    cfg['model_options']['embedding']['node_readout'] = mode
    return cfg


def bitwise(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8), b.detach().contiguous().reshape(-1).view(torch.uint8))


def support(model):
    from dptb.nn.chemical_readout import ChemicalCoreReadout
    for h in model.modules():
        if isinstance(h, ChemicalCoreReadout):
            h.set_counts(torch.arange(1, len(h.n_g)+1, device=h.n_g.device)*100)

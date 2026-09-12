"""Physical translation invariance of cached third-centre neighbourhoods."""
import numpy as np
import torch

from dptb.nacf.assembly import NACFTableBank
from dptb.tests.test_nacf_gpu import stores


def test_triclinic_partial_periodic_atom_wrapping_preserves_prior_and_overlap():
    p2, p23 = stores()
    bank = NACFTableBank(p2, p23, device='cpu')
    symbols = ['X','Y']
    pos = np.array([[.2,-.1,.3],[1.4,.2,-.1]])
    cell = np.array([[3.2,0.,0.],[1.3,3.1,0.],[0.,0.,0.]])
    edges = np.array([[0,1,0,1],[1,0,1,0]])
    shifts = np.array([[0,0,0],[0,0,0],[-1,1,0],[1,-1,0]])
    pbc = (True,True,False)
    original = bank.prepare(symbols,pos,cell,edges,shifts,pbc=pbc)()
    wraps = np.array([[2,-1,0],[-1,2,0]])
    changed_pos = pos + wraps @ cell
    changed_shifts = shifts + wraps[edges[0]] - wraps[edges[1]]
    wrapped = bank.prepare(symbols,changed_pos,cell,edges,changed_shifts,pbc=pbc)()
    for name in ('node_p23_ao_ev','edge_p2_ao_ev','node_overlap_ao','edge_overlap_ao'):
        torch.testing.assert_close(wrapped[name],original[name],atol=1e-12,rtol=1e-12)


def test_training_ry_conversion_scales_p2_but_not_vna_or_overlap():
    p2,p23 = stores()
    geometry = (['X','Y'],[[0.,0.,0.],[1.3,.2,.1]],np.eye(3)*10,
                [[0,1],[1,0]],np.zeros((2,3),dtype=int))
    once = NACFTableBank(p2,p23,device='cpu',ry_to_ev=1.).prepare(*geometry)()
    twice = NACFTableBank(p2,p23,device='cpu',ry_to_ev=2.).prepare(*geometry)()
    p23.epsilon = lambda symbol: np.zeros(1)
    p2_only = NACFTableBank(p2,p23,device='cpu',ry_to_ev=1.).prepare(*geometry)()
    for name in ('node_p23_ao_ev','edge_p2_ao_ev'):
        torch.testing.assert_close(twice[name]-once[name],p2_only[name],atol=1e-12,rtol=1e-12)
    for name in ('node_overlap_ao','edge_overlap_ao'):
        torch.testing.assert_close(twice[name],once[name],atol=0,rtol=0)

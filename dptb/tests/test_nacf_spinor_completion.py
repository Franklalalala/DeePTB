from types import SimpleNamespace

import pytest
import torch

from dptb.nacf.spinor_completion import SOCUURealCompletion


@pytest.mark.parametrize('doubling', [False, True])
def test_uureal_completion_retains_unlearned_soc_and_adds_residual_once(doubling):
    # Unequal orbital-pair sizes and different dict insertion order catch a
    # global reshape or dict-order assumption. The expected entries are explicit.
    factor = 8 if doubling else 4
    compact = SimpleNamespace(has_soc=True, nextham_uureal_mask=True,
        basis={'H':['1s','1p']}, reduced_matrix_element=4,
        orbpair_maps={'1s-1p':slice(1,4),'1s-1s':slice(0,1)}, get_orbpair_maps=lambda:None)
    full = SimpleNamespace(has_soc=True, nextham_uureal_mask=False,
        basis=compact.basis, reduced_matrix_element=4*factor,
        soc_complex_doubling=doubling,
        orbpair_maps={'1s-1s':slice(0,factor),'1s-1p':slice(factor,4*factor)},
        get_orbpair_maps=lambda:None)
    mapping=SOCUURealCompletion(compact,full)
    prior=torch.arange(8*factor,dtype=torch.float64).reshape(2,-1)
    if not doubling: prior=prior + 1j*(prior+100)
    original=prior.clone()
    delta=torch.tensor([[.5,1.,1.5,2.],[3.,4.,5.,6.]],dtype=torch.float64)
    uu=[0,factor,factor+1,factor+2]
    dd=[3,factor+9,factor+10,factor+11]
    expected=prior.clone()
    expected[:,uu]+=delta
    expected[:,dd]+=delta
    torch.testing.assert_close(mapping.extract_prior(prior),prior[:,uu].real)
    actual=mapping(prior,delta)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(prior,original,rtol=0,atol=0)
    untouched=[i for i in range(4*factor) if i not in uu+dd]
    torch.testing.assert_close(actual[:,untouched],prior[:,untouched],rtol=0,atol=0)
    if not doubling: torch.testing.assert_close(actual.imag,prior.imag,rtol=0,atol=0)
    with pytest.raises(ValueError,match='shape'):
        mapping(prior,delta[:,:3])
    with pytest.raises(ValueError,match='real dtype'):
        mapping(prior,delta.to(torch.complex128))

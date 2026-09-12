from dataclasses import replace
import numpy as np
from h0rebuild.models import UPFData
from h0rebuild.field_inputs import prepare_field_upf
from h0rebuild.radial_quadrature import simpson_rab

def test_abacus_field_cutoff_and_species_normalization():
    r=np.arange(202)*.1;rho=r*r*np.exp(-r)
    upf=UPFData('X',3.,r,np.full_like(r,.1),-2/(r+1),rho,np.zeros((0,0)),[],False,nlcc=np.exp(-r))
    got=prepare_field_upf(upf,15.)
    # First r>15 at index151: ABACUS forces the mesh count 152 down to151.
    assert len(got.r)==151 and got.r[-1]==15.
    assert len(got.nlcc)==151 and len(got.vloc_ry)==151
    np.testing.assert_allclose(simpson_rab(got.rhoatom_q,got.rab),3.,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(upf.rhoatom_q,rho)
    other=prepare_field_upf(replace(upf,z_valence=7.),15.)
    np.testing.assert_allclose(other.rhoatom_q,got.rhoatom_q*7/3,rtol=1e-15,atol=1e-15)

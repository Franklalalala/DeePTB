"""Independent built-in ABACUS vectors and spin-field branch regressions."""
from pathlib import Path
import numpy as np
import pytest

DATA=Path(__file__).with_name('data')


def test_builtin_spin_reference_vectors():
    pytest.importorskip('pylibxc')
    from h0rebuild.builtin_pbe import local,spin_correction
    a=np.loadtxt(DATA/'builtin_spin_reference.txt');rho,z,g=a[:,:3].T
    n=np.stack([rho*(1+z)/2,rho*(1-z)/2],-1)
    sigma=np.stack([g*g,.5*g*g,.25*g*g],-1)
    clipped=np.clip(z,-1,1)
    lda=local(np.stack([rho*(1+clipped)/2,rho*(1-clipped)/2],-1),True)
    xr,xs,cr,cs=spin_correction(n,sigma)
    np.testing.assert_allclose(lda,a[:,3:5],rtol=2e-12,atol=5e-15)
    np.testing.assert_allclose(xr,a[:,5:7],rtol=1e-12,atol=5e-15)
    np.testing.assert_allclose(2*xs[:,[0,2]],a[:,7:9],rtol=1e-12,atol=5e-15)
    # Smooth correlation constants differ slightly between implementations;
    # this is a bounded derivative check, not a claim of exact equivalence.
    corr=np.column_stack([cr,2*cs[:,0]])
    np.testing.assert_allclose(corr,a[:,9:12],rtol=1e-5,atol=2e-7)
    # Test branch zeros from the reference guard, not accidental rounded
    # cancellation to zero in an active formula (observed at 1e-22).
    inactive=(rho<=1e-6)|(np.abs(z)-1>1e-10)|(1.5*np.abs(g)<=1e-10)
    np.testing.assert_array_equal(corr[inactive],0.)


def test_builtin_scalar_reference_vectors():
    pytest.importorskip('pylibxc')
    from h0rebuild.builtin_pbe import local,scalar
    a=np.loadtxt(DATA/'builtin_scalar_reference.txt');rho,sigma=a[:,:2].T
    lda=local(rho);vr,vs=scalar(rho,sigma)
    np.testing.assert_allclose(lda,a[:,2],rtol=2e-12,atol=5e-15)
    corr=np.column_stack([vr-lda,2*vs])
    np.testing.assert_allclose(corr,a[:,3:],rtol=1e-5,atol=2e-7)
    np.testing.assert_array_equal(corr[a[:,3:]==0],0.)
    # Below the old 1e-6 cutoff a uniform density still has local PW.
    assert np.all(vr[(rho>1e-10)&(rho<1e-6)&(sigma==0)]<0)


@pytest.mark.parametrize('backend',['numpy','torch'])
def test_periodic_pbe_preserves_negative_fft_density(monkeypatch,backend):
    """A truncated positive atomic spike rings below zero on the FFT grid.

    Compare both field builders to ABACUS's abs-density XC convention,
    retaining the signed-density gradient in the GGA divergence.
    """
    pytest.importorskip('pylibxc')
    from types import SimpleNamespace
    from h0rebuild import reciprocal as rec
    from h0rebuild.models import Atom,Structure
    from h0rebuild.xc import pbe_vxc_ry
    st=Structure(np.eye(3)*10,[Atom('X',np.zeros(3))])
    upf=SimpleNamespace(r=np.arange(3.),rab=np.ones(3),rhoatom_q=np.ones(3),
                        vloc_ry=np.zeros(3),z_valence=1.,nlcc=None)
    sd={'X':SimpleNamespace(upf=upf)}
    monkeypatch.setattr(rec,'radial_charge_transform',lambda q,r,rab,g:np.ones_like(g))
    monkeypatch.setattr(rec,'radial_vloc_transform',lambda v,r,rab,z,g,volume,**kw:np.zeros_like(g))
    extra={}
    if backend=='torch':
        t=pytest.importorskip('torch')
        from h0rebuild import torch_fields as tf
        monkeypatch.setattr(tf,'radial_transforms_torch',lambda upf,g,volume,**kw:(t.ones_like(g),t.zeros_like(g),t.zeros_like(g)))
        monkeypatch.setattr(tf,'library_structure_factor',lambda pos,shape,**kw:(t.ones(shape,dtype=t.complex128),{}))
        extra=dict(compute_device='cpu',structure_factor_backend='torch_gaussian')
    got=rec.build_periodic_field(st,sd,ecutrho_ry=.5,fft_shape=(5,5,5),xc='PBE',
                                include_nlcc=False,field_backend=backend,**extra)
    shape=got.shape;g,g2=rec.reciprocal_grid(st.cell_bohr,shape);mask=g2<=.5
    # Analytic Fourier series for the seven retained plane waves.
    n=(1+2*sum(np.cos(2*np.pi*np.indices(shape)[axis]/5) for axis in range(3)))/st.volume
    grad=tuple(-4*np.pi/10*np.sin(2*np.pi*np.indices(shape)[axis]/5)/st.volume for axis in range(3))
    assert n.min() < -1e-6
    np.testing.assert_allclose(got.rho,n,rtol=1e-13,atol=1e-16)
    expected=pbe_vxc_ry(np.abs(n),grad,g,pw_mask=mask)
    np.testing.assert_allclose(got.vxc_ry,expected,rtol=1e-11,atol=1e-11)


def test_separate_spin_rotation_cutoffs(monkeypatch):
    from h0rebuild.spin_fields import collinear_pbe
    import h0rebuild.builtin_pbe as pbe
    monkeypatch.setattr(pbe,'local',lambda n,*a:np.broadcast_to([2.,1.],n.shape))
    monkeypatch.setattr(pbe,'spin_correction',lambda n,s:(np.broadcast_to([3.,1.],n.shape),np.zeros_like(s),np.zeros_like(n),np.zeros_like(s)))
    mag=np.array([0.,.5e-12,2e-12,.5e-10,2e-10]).reshape(5,1,1)
    rho=np.ones_like(mag);g=np.zeros(mag.shape+(3,));mask=np.ones_like(mag,dtype=bool)
    _,z=collinear_pbe(rho,rho*0,mag,g,mask,1.)
    np.testing.assert_array_equal(z.ravel(),[0,0,2,2,3])


@pytest.mark.parametrize('core_value',[0.,.05])
def test_signed_valence_channels_before_core(monkeypatch,core_value):
    from h0rebuild.spin_fields import collinear_pbe,quantization_axis_z
    import h0rebuild.builtin_pbe as pbe
    mag=np.array([-.2,-.01,0,.01,.2]).reshape(5,1,1)
    rho=np.full_like(mag,.1);core=np.full_like(mag,core_value);captured=[]
    monkeypatch.setattr(pbe,'local',lambda n,*a:np.zeros_like(n))
    def capture(n,s):
        captured.append(n.copy());return np.zeros_like(n),np.zeros_like(s),np.zeros_like(n),np.zeros_like(s)
    monkeypatch.setattr(pbe,'spin_correction',capture)
    g=np.zeros(mag.shape+(3,));mask=np.ones_like(mag,dtype=bool)
    axis=quantization_axis_z([0.,-2.,1.]);assert axis==-1
    assert quantization_axis_z([.001,0])==0
    collinear_pbe(rho,core,mag,g,mask,axis)
    expected=np.stack([(rho-mag+core)/2,(rho+mag+core)/2],-1)
    np.testing.assert_allclose(captured[0],expected,rtol=0,atol=2e-17)
    assert captured[0].min()<0  # overpolarized valence was not clipped

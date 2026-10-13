"""ABACUS built-in PBE local/correction gates, in Hartree atomic units.

Reference: ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f, wrapper_xc,
wrapper_gcxc and gradcorr. LibXC supplies smooth derivatives; zero-gradient
subtraction removes its own local contribution before ABACUS gates apply.
The reference local PW uses LDA_C_PW, not PBE's modified PW constants.
Small remaining smooth-formula differences are subject to numerical tests.
"""
import numpy as np


def _evaluate(name, rho, sigma=None, polarized=False):
    import pylibxc
    rho=np.asarray(rho,dtype=float)
    f=pylibxc.LibXCFunctional(name,'polarized' if polarized else 'unpolarized')
    # The explicit reference gates below own density cutoffs. This internal
    # floor only protects the derivative calculator on inactive dummy points.
    f.set_dens_threshold(1e-30)
    inp={'rho':rho.ravel()}
    if sigma is not None:inp['sigma']=np.asarray(sigma,dtype=float).ravel()
    result=f.compute(inp,do_vxc=True)
    vr=np.asarray(result['vrho']).reshape(rho.shape)
    vs=None if sigma is None else np.asarray(result['vsigma']).reshape(sigma.shape)
    return vr,vs


def local(rho, polarized=False):
    """Slater + original PW, with the caller's physical density channels."""
    n=np.asarray(rho,dtype=float)
    total=n.sum(-1) if polarized else n
    active=total>1e-10
    safe=np.maximum(n,0.)
    vr=sum(_evaluate(name,safe,polarized=polarized)[0] for name in ('LDA_X','LDA_C_PW'))
    return np.where(active[...,None] if polarized else active,vr,0.)


def _correction(name,n,sigma,polarized=False):
    vr,vs=_evaluate(name,n,sigma,polarized)
    zero,_=_evaluate(name,n,np.zeros_like(sigma),polarized)
    return vr-zero,vs


def scalar(rho,sigma):
    n=np.abs(np.asarray(rho,dtype=float));s=np.asarray(sigma,dtype=float)
    active=(n>1e-6)&(s>=1e-10)
    safe=np.where(active,n,1.);sg=np.where(active,s,0.)
    vr=local(n);vs=np.zeros_like(n)
    for name in ('GGA_X_PBE','GGA_C_PBE'):
        r,v=_correction(name,safe,sg)
        vr+=np.where(active,r,0.);vs+=np.where(active,v,0.)
    return vr,vs


def spin_correction(channels,sigma):
    """Separate exchange and correlation corrections, without local PW.

    Returns (x_vrho, x_vsigma, c_vrho, c_vsigma). Sigma is uu, ud, dd;
    multiplying 2*vsigma_uu by grad(up) forms its same-spin flux.
    """
    n=np.asarray(channels,dtype=float);s=np.asarray(sigma,dtype=float)
    total=n.sum(-1)
    active_x=(total[...,None]>1e-10)&(n>1e-10)&(np.sqrt(np.abs(s[...,[0,2]]))>1e-10)
    # Spin scaling: .5*E_x(2*n,4*sigma). vrho is unchanged and vsigma doubles.
    safe=np.where(active_x,2*n,1.);sg=np.where(active_x,4*s[...,[0,2]],0.)
    xr,xs=_correction('GGA_X_PBE',safe,sg)
    xr=np.where(active_x,xr,0.);xs=np.where(active_x,2*xs,0.)
    xv=np.stack([xs[...,0],np.zeros_like(total),xs[...,1]],-1)
    z=np.divide(n[...,0]-n[...,1],total,out=np.zeros_like(total),where=total!=0)
    sg_total=s[...,0]+2*s[...,1]+s[...,2]
    active_c=(total>1e-6)&(np.abs(z)-1<=1e-10)&(np.sqrt(np.abs(sg_total))>1e-10)
    clipped=np.clip(z,-1+1e-6,1-1e-6);safe_total=np.where(active_c,total,1.)
    cn=np.stack([safe_total*(1+clipped)/2,safe_total*(1-clipped)/2],-1)
    # PBE correlation depends only on the total-density gradient; choose an
    # equivalent split with all cross terms explicit and nonnegative.
    cs=np.repeat((np.where(active_c,np.maximum(sg_total,0.),0.)/4)[...,None],3,-1)
    cr,cv=_correction('GGA_C_PBE',cn,cs,True)
    cr=np.where(active_c[...,None],cr,0.);cv=np.where(active_c[...,None],cv,0.)
    return xr,xv,cr,cv

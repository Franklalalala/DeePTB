"""Float64 torch periodic fields, sharing the original physical conventions.

CUDA path: cuFINUFFT (or explicit torch Gaussian), radial quadrature,
Hartree, FFTs, and LDA stay on device. Return NumPy PeriodicField arrays once
at the boundary to the existing CPU AO/pseudopotential assembly. For PBE,
LibXC remains an explicit CPU stage; gradients/divergence still use torch FFT.
No local AO, KB projector, or eigensolver acceleration is claimed here.
"""
from __future__ import annotations
from math import prod
import numpy as np
from scipy.special import erf
from .constants import FOUR_PI, E2_RY_BOHR
from .radial_quadrature import simpson_rab_weights
from .nufft_backends import resolve_torch_device, library_structure_factor, to_numpy, _budget


def radial_transforms_torch(upf, g, volume, *, keep_g0_alpha, include_nlcc,
                            chunk_size=256):
    """Same PP_RAB Simpson/trailing trapezoid weights as the NumPy oracle.

    Returns Q(G), Vloc(G), core Q(G); no interpolation of radial transforms.
    Work O(number_of_G_magnitudes * fixed_radial_mesh_size), bounded chunks.
    """
    import torch as t
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    def tensor(a):
        return t.as_tensor(np.asarray(a),dtype=t.float64,device=g.device)
    r=tensor(upf.r);w=tensor(simpson_rab_weights(upf.rab));q=tensor(upf.rhoatom_q)
    vloc=tensor(upf.vloc_ry)
    if r.ndim!=1 or q.shape!=r.shape or vloc.shape!=r.shape:
        raise ValueError("UPF radial arrays must have identical one-dimensional shapes")
    # Shared CPU erf evaluation only on the fixed species radial mesh.
    ze=upf.z_valence*E2_RY_BOHR
    comp=r*vloc+ze*tensor(erf(upf.r))
    core=None
    if include_nlcc and upf.nlcc is not None:
        core=tensor(upf.nlcc)
        if core.shape!=r.shape:
            raise ValueError("UPF NLCC and radial mesh shapes disagree")
        if upf.nlcc_is_radial_density:
            core=FOUR_PI*r**2*core
    qg=t.empty_like(g);vg=t.empty_like(g);cg=t.zeros_like(g)
    # Compute alpha once with precisely the baseline integrand and weights.
    alpha=t.sum(r*(r*vloc+ze)*w)*FOUR_PI/volume if keep_g0_alpha else 0.0
    for start in range(0,g.numel(),chunk_size):
        stop=min(start+chunk_size,g.numel()); gg=g[start:stop]
        sinc=t.sinc(gg[:,None]*r[None,:]/np.pi)
        qg[start:stop]=t.sum(q[None,:]*sinc*w[None,:],dim=1)
        if core is not None:
            cg[start:stop]=t.sum(core[None,:]*sinc*w[None,:],dim=1)
        zero=gg<1e-13
        safe=t.where(zero,t.ones_like(gg),gg)
        integ=t.sum(comp[None,:]*t.sin(safe[:,None]*r[None,:])/safe[:,None]*w[None,:],dim=1)
        vals=FOUR_PI*(integ-ze*t.exp(-safe**2/4)/safe**2)/volume
        vg[start:stop]=t.where(zero,t.as_tensor(alpha,dtype=t.float64,device=g.device),vals)
    return qg,vg,cg


def lda_pz81_torch(rho):
    """Unpolarized PZ81, constants/formulas matched to xc.lda_pz81_vxc_ry."""
    import torch as t
    active=rho>1e-20
    n=t.where(active,rho,t.ones_like(rho))
    rs=(3/(4*np.pi*n))**(1/3)
    vx=-((3/np.pi)**(1/3))*n**(1/3)
    A,B,C,D=.0311,-.048,.0020,-.0116
    eps=A*t.log(rs)+B+C*rs*t.log(rs)+D*rs
    deps=A/rs+C*(t.log(rs)+1)+D
    high=eps-rs*deps/3
    gamma,beta1,beta2=-.1423,1.0529,.3334
    den=1+beta1*t.sqrt(rs)+beta2*rs
    eps=gamma/den
    deps=-gamma*(beta1/(2*t.sqrt(rs))+beta2)/den**2
    low=eps-rs*deps/3
    return t.where(active,2*(vx+t.where(rs<1,high,low)),t.zeros_like(rho))


def pbe_torch_hybrid(rho,grad,gvec,*,density_threshold,pw_mask=None):
    """LibXC CPU derivatives, torch spectral divergence; explicit transfers."""
    import torch as t
    from .xc import pbe_libxc_derivatives
    sigma=sum(g*g for g in grad)
    vrho,vsigma=pbe_libxc_derivatives(to_numpy(rho),to_numpy(sigma),density_threshold=density_threshold)
    vrho=t.as_tensor(vrho,dtype=t.float64,device=rho.device)
    vsigma=t.as_tensor(vsigma,dtype=t.float64,device=rho.device)
    from .pw_derivatives import derivative
    mask=t.ones_like(rho,dtype=t.bool) if pw_mask is None else pw_mask
    div=sum(derivative(2*vsigma*grad[axis],gvec,mask,axis) for axis in range(3))
    return 2*(vrho-div)


def build_periodic_field_torch(structure,species_data,*,ecutrho_ry,fft_shape,
        xc="LDA_PZ81",hartree_g0="zero",local_g0="alpha",include_nlcc=True,
        total_electrons=None,radial_g_round_decimals=None,pbe_density_threshold=1e-6,
        structure_factor_backend="cufinufft",structure_factor_eps=1e-12,
        structure_factor_max_work_mb=512.,compute_device="cuda:0",
        field_max_work_mb=1024.):
    from .reciprocal import PeriodicField,reciprocal_grid,_unique_magnitudes
    t,device=resolve_torch_device(compute_device)
    shape=tuple(fft_shape);xc_upper=xc.upper()
    if structure_factor_backend not in {"cufinufft","torch_gaussian"}:
        raise ValueError("torch fields require structure_factor_backend='cufinufft' or 'torch_gaussian'")
    if structure_factor_backend=="cufinufft" and device.type!="cuda":
        raise ValueError("cufinufft requires compute_device='cuda:<index>'")
    if xc_upper not in {"LDA","LDA_PZ81","PZ81","PBE","GGA_PBE","NONE","OFF"}:
        raise ValueError(f"Unsupported XC option: {xc}")
    present={a.species for a in structure.atoms}
    radial_max=max((len(species_data[s].upf.r) for s in present),default=0)
    # Field tensors, outputs and radial chunks, plus copies at final CPU boundary.
    # This budget is separate from the simultaneously live NUFFT plan workspace.
    estimate=320*prod(shape)+64*256*radial_max
    _budget(estimate,field_max_work_mb)
    volume=structure.volume
    positions=np.asarray([a.frac for a in structure.atoms],dtype=float)
    labels=np.asarray([a.species for a in structure.atoms],dtype=object)
    with t.no_grad():
        gvec_np,g2_np=reciprocal_grid(structure.cell_bohr,shape)
        gm_np,inverse_np=_unique_magnitudes(g2_np,round_decimals=radial_g_round_decimals)
        def tensor(a,dtype=t.float64):
            return t.as_tensor(a,dtype=dtype,device=device)
        gvec=tensor(gvec_np);g2=tensor(g2_np);gm=tensor(gm_np)
        inverse=tensor(inverse_np,t.int64)
        mask=g2<=float(ecutrho_ry)+1e-12
        rg=t.zeros(shape,dtype=t.complex128,device=device)
        cg=t.zeros_like(rg);vg=t.zeros_like(rg);sf_stats={}
        for symbol in sorted(species_data):
            ids=np.where(labels==symbol)[0]
            if not ids.size:
                continue
            sf,sf_stats[symbol]=library_structure_factor(positions[ids],shape,
                backend=structure_factor_backend,eps=structure_factor_eps,
                device=str(device),max_work_mb=structure_factor_max_work_mb)
            q,v,c=radial_transforms_torch(species_data[symbol].upf,gm,volume,
                keep_g0_alpha=local_g0.lower()=="alpha",include_nlcc=include_nlcc)
            rg+=sf*q[inverse].reshape(shape)/volume
            vg+=sf*v[inverse].reshape(shape)
            cg+=sf*c[inverse].reshape(shape)/volume
        rg=t.where(mask,rg,0.);cg=t.where(mask,cg,0.)
        if total_electrons is None:
            total_electrons=float(sum(species_data[a.species].upf.z_valence for a in structure.atoms))
        raw=float(rg[0,0,0].real*volume)
        if raw<=0 or not np.isfinite(raw):
            raise ValueError(f"Superposed PP_RHOATOM has non-positive/non-finite electron count {raw}")
        scale=float(total_electrons)/raw;rg*=scale
        vg=t.where(mask,vg,0.)
        if local_g0.lower()=="zero":
            vg[0,0,0]=0.
        elif local_g0.lower()!="alpha":
            raise ValueError("local_g0 must be 'alpha' or 'zero'")
        hg=t.zeros_like(rg);nz=g2>1e-14
        hg[nz]=FOUR_PI*E2_RY_BOHR*rg[nz]/g2[nz]
        if isinstance(hartree_g0,str):
            if hartree_g0.lower()!="zero":
                raise ValueError("hartree_g0 must be 'zero' or a numeric constant in Ry")
        else:
            hg[0,0,0]=float(hartree_g0)
        def ifft(c):
            return (t.fft.ifftn(c)*prod(shape)).real
        rho=ifft(rg);rho_raw=ifft(rg+cg);rho_xc=t.clamp_min(rho_raw,1e-30)
        vloc=ifft(vg);vh=ifft(hg)
        if xc_upper in {"LDA","LDA_PZ81","PZ81"}:
            vxc=lda_pz81_torch(rho_xc);xc_backend="torch_lda_pz81"
        elif xc_upper in {"PBE","GGA_PBE"}:
            grad=tuple(ifft(1j*gvec[...,axis]*(rg+cg)) for axis in range(3))
            vxc=pbe_torch_hybrid(rho_xc,grad,gvec,density_threshold=pbe_density_threshold,pw_mask=mask)
            xc_backend="libxc_cpu_derivatives_torch_spectral_divergence"
        else:
            vxc=t.zeros_like(rho);xc_backend="none"
        total=vloc+vh+vxc
        for a in (rho,rg,total):
            if not bool(t.isfinite(a).all()):
                raise FloatingPointError("Non-finite torch periodic field")
        arrays=[to_numpy(a) for a in (total,rho,vloc,vh,vxc,rg)]
        metadata=dict(structure_factor_backend=structure_factor_backend,
            structure_factor_eps=float(structure_factor_eps),structure_factor_stats=sf_stats,
            field_backend="torch",compute_device=str(device),torch_version=t.__version__,
            field_dtype="float64/complex128",field_output_location="cpu_numpy",
            field_estimated_work_bytes=estimate,field_max_work_mb=field_max_work_mb,
            field_estimate_excludes_nufft_workspace=True,field_memory_estimate_is_hard_limit=False,
            field_transfer_policy="geometry/radial input upload; final field download; PBE additionally transfers rho/sigma and derivatives",
            xc_evaluation_backend=xc_backend,cuda_runtime=t.version.cuda,
            ecutrho_ry=float(ecutrho_ry),fft_shape=list(shape),fft_grid_origin_fractional=[0.,0.,0.],
            radial_quadrature="ABACUS composite Simpson with PP_RAB",xc=xc,
            hartree_g0=hartree_g0,local_g0=local_g0,include_nlcc=include_nlcc,
            target_electrons=float(total_electrons),raw_superposed_electrons=raw,
            density_normalization_scale=scale,radial_g_round_decimals=radial_g_round_decimals,
            pbe_density_threshold_electrons_per_bohr3=float(pbe_density_threshold) if xc_upper in {"PBE","GGA_PBE"} else None,
            minimum_valence_density_on_fft_grid=float(t.min(rho)),
            minimum_xc_density_before_floor=float(t.min(rho_raw)))
    return PeriodicField(np.asarray(structure.cell_bohr),*arrays,gvec_np,metadata)

"""ABACUS collinear initial magnetization with full spinor SOC projectors.

Uses separate local PW and gradient corrections for built-in ABACUS PBE,
including its fixed-axis signed channels. No Hamiltonian labels are consumed. Noncollinear transverse
initial magnetization is rejected until independently validated.
"""
from dataclasses import replace
from .pw_derivatives import derivative
import numpy as np
from .nufft_backends import Type1NUFFTPlan,to_numpy
from .reciprocal import _unique_magnitudes,radial_charge_transform,radial_density_transform

def quantization_axis_z(moments):
    # cal_ux chooses the first atom with squared moment > 1e-6. All supported
    # input moments are collinear, so judge_parallel subsequently stays true.
    eligible=np.asarray(moments)[np.asarray(moments)**2>1e-6]
    return float(np.sign(eligible[0])) if eligible.size else 0.


def collinear_pbe(rho,core,mag,gvec,pw_mask,axis_z):
    from .builtin_pbe import local,spin_correction
    use_cuda=type(rho).__module__.startswith('torch')
    if use_cuda:
        import torch as xp
        def array(x):return xp.as_tensor(x,dtype=xp.float64,device=rho.device)
        def stack(x):return xp.stack(x,dim=-1)
    else:
        xp=np
        def array(x):return np.asarray(x)
        def stack(x):return np.stack(x,axis=-1)
    total=xp.abs(rho+core);amp=xp.minimum(xp.abs(mag),total)
    lda=array(local(to_numpy(stack([(total+amp)/2,(total-amp)/2])),True))
    neg=xp.where(mag*axis_z>0,1.,-1.) if axis_z else xp.ones_like(mag)
    # Split VALENCE before adding half core, without abs/clipping. Clipping
    # these channels before FFT changes the derivative at magnetic zeroes.
    channels=stack([(rho+neg*xp.abs(mag)+core)/2,(rho-neg*xp.abs(mag)+core)/2])
    grad=[[derivative(channels[...,s],gvec,pw_mask,a) for a in range(3)] for s in range(2)]
    sigma=stack([sum(x*x for x in grad[0]),sum(x*y for x,y in zip(*grad)),sum(x*x for x in grad[1])])
    xr,xs,cr,cs=spin_correction(to_numpy(channels),to_numpy(sigma))
    vr=array(xr+cr);vs=array(xs+cs)
    correction=[]
    for s in range(2):
        div=sum(derivative(2*vs[...,0 if s==0 else 2]*grad[s][a]+vs[...,1]*grad[1-s][a],gvec,pw_mask,a) for a in range(3))
        correction.append(2*(vr[...,s]-div))
    mean=lda[...,0]+lda[...,1]+(correction[0]+correction[1])/2
    local_z=(lda[...,0]-lda[...,1])*xp.where(xp.abs(mag)>1e-10,xp.sign(mag),0.)
    grad_z=(correction[0]-correction[1])/2*neg*xp.where(xp.abs(mag)>1e-12,xp.sign(mag),0.)
    return mean,local_z+grad_z

def add_collinear_spin_field(field,structure,species_data,moments_z,*,backend,device,eps,nthreads=1,max_work_mb=512.,threshold=1e-6):
    if threshold!=1e-6:raise ValueError('Built-in PBE uses fixed reference gates; custom threshold is unsupported')
    if field.metadata['xc'].upper() not in ['PBE','GGA_PBE']:
        raise NotImplementedError('Initial magnetic H0 is currently validated for PBE only')
    moments=np.asarray(moments_z,float)
    if moments.shape!=(len(structure.atoms),) or not np.isfinite(moments).all():
        raise ValueError('initial_moments_z must be a finite moment in Bohr magnetons for every atom')
    shape=field.shape;ng=int(np.prod(shape));g2=(field.gvec**2).sum(-1)
    gm,inv=_unique_magnitudes(g2,round_decimals=field.metadata['radial_g_round_decimals'])
    labels=np.array([a.species for a in structure.atoms]);pos=np.array([a.frac for a in structure.atoms])
    use_cuda=str(device).startswith('cuda')
    if use_cuda:
        import torch as t
        from .torch_fields import radial_transforms_torch
        def array(x,dtype=t.float64):return t.as_tensor(x,dtype=dtype,device=device)
        invd=array(inv,t.int64);gmd=array(gm);gv=array(field.gvec)
        mg=t.zeros(shape,dtype=t.complex128,device=device);cg=t.zeros_like(mg)
        fft=t.fft.fftn
        def ifft(x):return (t.fft.ifftn(x)*ng).real
    else:
        gv=field.gvec;mg=np.zeros(shape,complex);cg=np.zeros_like(mg)
        fft=np.fft.fftn
        def ifft(x):return (np.fft.ifftn(x)*ng).real
    for symbol in sorted(species_data):
        ids=np.where(labels==symbol)[0]
        if not len(ids):continue
        upf=species_data[symbol].upf;weights=moments[ids]/upf.z_valence
        if backend=='direct':
            phase=np.exp(-1j*(field.gvec.reshape(-1,3)@structure.cart_positions[ids].T))
            sf=phase.sum(1).reshape(shape);sm=(phase@weights).reshape(shape)
        else:
            with Type1NUFFTPlan(pos[ids],shape,backend=backend,device=device,eps=eps,nthreads=nthreads,max_work_mb=max_work_mb) as plan:
                sf=plan.execute();sm=plan.execute(weights)
        if use_cuda:
            q,_,core=radial_transforms_torch(upf,gmd,structure.volume,keep_g0_alpha=True,include_nlcc=field.metadata['include_nlcc'])
            mg+=sm*q[invd].reshape(shape)/structure.volume
            cg+=sf*core[invd].reshape(shape)/structure.volume
        else:
            q=radial_charge_transform(upf.rhoatom_q,upf.r,upf.rab,gm)[inv].reshape(shape)
            mg+=sm*q/structure.volume
            if field.metadata['include_nlcc'] and upf.nlcc is not None:
                fn=radial_density_transform if upf.nlcc_is_radial_density else radial_charge_transform
                core=fn(upf.nlcc,upf.r,upf.rab,gm)[inv].reshape(shape)
                cg+=sf*core/structure.volume
    mask=g2<=field.metadata['ecutrho_ry']+1e-12
    if use_cuda:
        mg=t.where(array(mask,t.bool),mg,0.)*field.metadata['density_normalization_scale']
        cg=t.where(array(mask,t.bool),cg,0.)
        mag=ifft(mg)
    else:
        mg=np.where(mask,mg,0.)*field.metadata['density_normalization_scale'];cg=np.where(mask,cg,0.)
        mag=ifft(mg)
    pw_mask=array(mask,t.bool) if use_cuda else mask
    axis_z=quantization_axis_z(moments)
    mean,delta=collinear_pbe(array(field.rho) if use_cuda else field.rho,ifft(cg),mag,gv,pw_mask,axis_z)
    if use_cuda:
        mean,delta,mag=to_numpy(mean),to_numpy(delta),to_numpy(mag)
    metadata={**field.metadata,'initial_moments_z':moments.tolist(),'initial_magnetization_integral':float(mag.sum()*field.grid_weight),
      'spin_field_contract':'built-in ABACUS PBE: original local PW; separate correction gates; signed valence channels then NLCC; separate spin rotations',
      'quantization_axis_z':axis_z,
      'xc_reference_dispatch':'ABACUS PBE use_libxc=false',
      'spin_field_backend':'torch_fft_cuda_libxc_cpu' if use_cuda else 'numpy_fft_libxc_cpu'}
    return replace(field,values_ry=field.vloc_ry+field.vh_ry+mean,vxc_ry=mean,metadata=metadata),delta

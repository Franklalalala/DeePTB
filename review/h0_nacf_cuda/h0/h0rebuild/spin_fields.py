"""ABACUS collinear initial magnetization with full spinor SOC projectors.

Matches Charge::atomic_rho and the LibXC nspin=4 conversion. No Hamiltonian
labels are consumed. Noncollinear transverse initial magnetization is rejected
by the public interface until independently validated.
"""
from dataclasses import replace
import numpy as np
from .nufft_backends import Type1NUFFTPlan,to_numpy
from .reciprocal import _unique_magnitudes,radial_charge_transform,radial_density_transform

def _derivatives(rho,sigma,threshold):
    import pylibxc
    shape=rho.shape[:-1];n=rho.reshape(-1,2);s=sigma.reshape(-1,3)
    active=n.sum(axis=1)>=threshold
    ne=np.where(active[:,None],n,threshold/2)
    se=np.where(active[:,None],s,0.)
    vr=np.zeros_like(n);vs=np.zeros_like(s)
    for name in ['GGA_X_PBE','GGA_C_PBE']:
        answer=pylibxc.LibXCFunctional(name,'polarized').compute({'rho':ne.ravel(),'sigma':se.ravel()},do_vxc=True)
        r=np.asarray(answer['vrho']).reshape(-1,2)
        v=np.asarray(answer['vsigma']).reshape(-1,3)
        if name=='GGA_C_PBE':
            masks=(n>=threshold)&(np.sqrt(abs(s[:,[0,2]]))>=1e-10)
            r=r*masks
            v=v*np.column_stack([masks[:,0],masks[:,0]&masks[:,1],masks[:,1]])
        vr+=r;vs+=v
    vr[~active]=0;vs[~active]=0
    return vr.reshape(*shape,2),vs.reshape(*shape,3)

def add_collinear_spin_field(field,structure,species_data,moments_z,*,backend,device,eps,nthreads=1,max_work_mb=512.,threshold=1e-6):
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
        mag=ifft(mg);total=t.abs(array(field.rho)+ifft(cg));amp=t.minimum(t.abs(mag),total)
        channels=t.stack([(total+amp)/2,(total-amp)/2],dim=-1)
    else:
        mg=np.where(mask,mg,0.)*field.metadata['density_normalization_scale'];cg=np.where(mask,cg,0.)
        mag=ifft(mg);total=abs(field.rho+ifft(cg));amp=np.minimum(abs(mag),total)
        channels=np.stack([(total+amp)/2,(total-amp)/2],axis=-1)
    grad=[]
    for s in range(2):
        coeff=fft(channels[...,s])/ng
        grad.append([ifft(1j*gv[...,axis]*coeff) for axis in range(3)])
    sigma=[sum(x*x for x in grad[0]),sum(x*y for x,y in zip(*grad)),sum(x*x for x in grad[1])]
    if use_cuda:
        sigma=t.stack(sigma,dim=-1)
        vr,vs=_derivatives(to_numpy(channels),to_numpy(sigma),threshold);vr=array(vr);vs=array(vs)
    else:vr,vs=_derivatives(channels,np.stack(sigma,axis=-1),threshold)
    potentials=[]
    for s in range(2):
        divergence=0
        for axis in range(3):
            flux=2*vs[...,0 if s==0 else 2]*grad[s][axis]+vs[...,1]*grad[1-s][axis]
            divergence=divergence+ifft(1j*gv[...,axis]*fft(flux)/ng)
        potentials.append(2*(vr[...,s]-divergence))
    mean=(potentials[0]+potentials[1])/2
    delta=(potentials[0]-potentials[1])/2
    if use_cuda:
        delta=delta*t.where(t.abs(mag)>1e-10,t.sign(mag),0.)
        mean,delta,mag=to_numpy(mean),to_numpy(delta),to_numpy(mag)
    else:delta=delta*np.where(abs(mag)>1e-10,np.sign(mag),0.)
    metadata={**field.metadata,'initial_moments_z':moments.tolist(),'initial_magnetization_integral':float(mag.sum()*field.grid_weight),
      'spin_field_contract':'ABACUS collinear nspin=4 atomic density and polarized LibXC PBE, including NLCC and correlation threshold masks',
      'spin_field_backend':'torch_fft_cuda_libxc_cpu' if use_cuda else 'numpy_fft_libxc_cpu'}
    return replace(field,values_ry=field.vloc_ry+field.vh_ry+mean,vxc_ry=mean,metadata=metadata),delta

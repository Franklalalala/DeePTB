"""ABACUS complete_default + Charge::atomic_rho radial field contract."""
from dataclasses import replace
import numpy as np
from .radial_quadrature import simpson_rab

def prepare_field_upf(upf, cutoff_bohr=15.0):
    if not np.isfinite(cutoff_bohr) or cutoff_bohr<=0:raise ValueError('pseudo_rcut_bohr must be finite and positive')
    # UPF v2 reader first drops the last sample of even meshes. This is
    # a field-only view: full projector inputs remain separate in assemble_h0.
    reader_n=len(upf.r)-(len(upf.r)%2 == 0)
    above=np.flatnonzero(upf.r[:reader_n]>cutoff_bohr)
    n=min(reader_n, 2*((int(above[0])+2)//2)-1) if len(above) else reader_n
    if n<3:raise ValueError('Pseudopotential field cutoff leaves fewer than 3 samples')
    r=upf.r[:n];rab=upf.rab[:n];q=upf.rhoatom_q[:n].copy()
    charge=float(simpson_rab(q,rab))
    if not np.isfinite(charge) or charge<=0:raise ValueError('Non-positive atomic valence density integral')
    # Match the radial-density origin extrapolation before returning to q(r).
    if r[0]!=0:
        rho=q/(4*np.pi*r*r)
        ratio=(rho[2]/rho[1])**(r[1]/(r[2]-r[1]))
        rho0=rho[1] if ratio<1e-12 else rho[1]/ratio
        q[0]=rho0*4*np.pi*r[0]**2
    q*=upf.z_valence/charge
    return replace(upf,r=r,rab=rab,rhoatom_q=q,vloc_ry=upf.vloc_ry[:n],
        nlcc=None if upf.nlcc is None else upf.nlcc[:n],metadata={**upf.metadata,
        'field_radial_preparation':{'algorithm':'ABACUS msh odd cutoff + per-species atomic charge normalization',
        'pseudo_rcut_bohr':cutoff_bohr,'msh':n,'raw_charge':charge,'scale':upf.z_valence/charge}})

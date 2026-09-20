"""Geometry-only density neighborhoods and caller-defined probe reductions.

Lengths are Bohr. No density definition, probe weights, or species exceptions
are selected here. The origin density is separate from the environment.
"""
import numpy as np
import torch
from .topology import build_edge_topology


def density_sum_cuda(points, neighbors, density_bank):
    """Sum accepted clamped cubic density channels on onsite/probe points.

    Each density object exposes knots and coeff (a list of [4,K-1] tensors).
    Channel positivity and support clipping exactly follow AtomicDensity.
    Caller supplies the ordered species/neighbor lists, including endpoints
    for onsite and excluding them for environmental probes.
    """
    from ._cuda import extension,check_device
    if points.requires_grad:raise ValueError('density CUDA path is inference-only')
    check_device(points.device)
    rho=points.new_zeros(len(points))
    for s,positions in neighbors.items():
        density=density_bank[s]
        coeff=torch.stack(density.coeff).contiguous()
        p=torch.as_tensor(positions,device=points.device,dtype=points.dtype).contiguous()
        extension().density_add(points.contiguous(),p,density.knots,coeff,rho)
    return rho


def density_topology(positions, cell, pbc, query_radii, density_supports,
                     edge_index=None, edge_cell_shift=None, **options):
    """Return atom and edge CSR lists, excluding (i,0) and edge (j,R).

    ``queries[q] = (i,k,sx,sy,sz)`` represents neighbour ``pos[k]+s@cell``.
    query_radii[i] must enclose every probe of every edge starting at i.
    Bounds are radius[i]+support[k]; periodic endpoint images are retained.
    Edge rows need not have reverses. Inputs describe one structure; call once
    per structure for a heterogeneous batch (no cross-structure neighbours).
    """
    edges=np.empty((2,0),dtype=np.int64) if edge_index is None else np.asarray(edge_index)
    shifts=np.empty((0,3),dtype=np.int64) if edge_cell_shift is None else np.asarray(edge_cell_shift)
    if np.asarray(positions).shape==(0,3):
        if edges.shape!=(2,0) or shifts.shape!=(0,3) or len(query_radii) or len(density_supports):
            raise ValueError('empty density geometry must have empty queries and edges')
        return {'queries':np.empty((0,5),dtype=np.int64),'terms':np.empty((0,3),dtype=np.int64),
                'reverse':np.empty(0,dtype=np.int64),'atom_ptr':np.array([0]),'edge_ptr':np.array([0]),
                'edge_queries':np.empty(0,dtype=np.int64),'broad_pairs':0,'search_s':0.,'join_s':0.}
    out=build_edge_topology(positions,cell,pbc,query_radii,density_supports,edges,shifts,mode='density',**options)
    q,t=out['queries'],out['terms']
    out['atom_ptr']=np.r_[0,np.cumsum(np.bincount(q[:,0],minlength=len(positions)))]
    out['edge_ptr']=np.r_[0,np.cumsum(np.bincount(t[:,0],minlength=edges.shape[1]))]
    out['edge_queries']=t[:,1].copy()
    return out


def onsite_density_neighbors(g, query_radii, density_supports, *, legacy_radius=None, **options):
    """Build every onsite list in one native search, add origin explicitly.

    legacy_radius preserves the accepted finite-radius truncation and neighbour
    ordering, including zero-density entries and their reduction chunk layout.
    Omit it to use exact density-support bounds for a new caller.
    """
    pos=np.asarray(g['positions_bohr'],dtype=np.float64);cell=np.asarray(g['cell_bohr'],dtype=np.float64)
    symbols=g['symbols'];radii=np.asarray(query_radii,dtype=np.float64)
    supports=np.asarray([density_supports[s] for s in symbols],dtype=np.float64)
    if legacy_radius is not None:
        radii=np.full(len(pos),legacy_radius,dtype=np.float64);supports=np.zeros(len(pos))
    topo=density_topology(pos,cell,g.get('pbc',(True,)*3),radii,supports,**options)
    result=[]
    for i in range(len(pos)):
        q=topo['queries'][topo['atom_ptr'][i]:topo['atom_ptr'][i+1]]
        q=np.concatenate((q,np.array([[i,i,0,0,0]],dtype=np.int64)))
        q=q[np.lexsort((q[:,4],q[:,3],q[:,2],q[:,1]))]
        delta=pos[q[:,1]]+q[:,2:]@cell-pos[i]
        if legacy_radius is not None:
            active=np.linalg.norm(delta,axis=1)<legacy_radius;q=q[active];delta=delta[active]
        grouped={}
        for s in dict.fromkeys(symbols):
            active=np.array([symbols[k]==s for k in q[:,1]])
            if active.any():grouped[s]=delta[active]
        result.append(grouped)
    return result


def probe_environment_density(probes, neighbor_positions, neighbor_species,
                              edge_ptr, density_bank, *, weights=None, chunk=256):
    """Torch reference F4: [E,P,3] -> [E] using caller-supplied density splines.

    Neighbour positions are absolute coordinates in each edge's image frame,
    grouped in CSR edge_ptr; endpoints must already be excluded. density_bank
    maps species identifiers to vectorized 1-D spline callables. Equal probe
    weights are used if absent; supplied [E,P] weights are normalized per edge.
    This API does not choose D1/D2 probes or a density/NLCC convention.
    """
    if probes.ndim!=3 or probes.shape[-1]!=3 or probes.shape[1]==0:
        raise ValueError('probes must be [edges, positive probes, 3]')
    ptr=np.asarray(edge_ptr,dtype=np.int64)
    if ptr.shape!=(len(probes)+1,) or ptr[0]!=0 or ptr[-1]!=len(neighbor_positions) or np.any(np.diff(ptr)<0):
        raise ValueError('invalid edge CSR offsets')
    if len(neighbor_species)!=len(neighbor_positions) or chunk<=0:
        raise ValueError('invalid neighbor species or chunk')
    if weights is not None and (weights.shape!=probes.shape[:2] or not torch.isfinite(weights).all() or (weights<0).any() or (weights.sum(-1)<=0).any()):
        raise ValueError('weights must be finite nonnegative with positive edge sum')
    positions=torch.as_tensor(neighbor_positions,device=probes.device,dtype=probes.dtype)
    result=probes.new_zeros(len(probes))
    for e in range(len(probes)):
        rho=probes.new_zeros(probes.shape[1])
        for s in dict.fromkeys(neighbor_species[ptr[e]:ptr[e+1]]):
            ids=[k for k in range(ptr[e],ptr[e+1]) if neighbor_species[k]==s]
            for start in range(0,len(ids),chunk):
                d=torch.linalg.vector_norm(probes[e,:,None,:]-positions[ids[start:start+chunk]][None,:,:],dim=-1)
                rho+=density_bank[s](d).sum(-1)
        result[e]=rho.mean() if weights is None else (rho*weights[e]).sum()/weights[e].sum()
    return result

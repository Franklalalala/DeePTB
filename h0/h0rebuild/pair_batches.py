"""Bounded S/T batches using the existing validated native kernel unchanged."""
from itertools import islice
import numpy as np


def scalar_pair_batches(two_center, pairs, atom_species, batch_size):
    """Yield original edges in order, with independent, unpadded CPU blocks.

    Only S/T evaluations are grouped. No symmetry reconstruction, changed
    displacements, candidate reduction, or full-geometry output staging.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('cuda_pair_batch_size must be a positive integer')
    iterator = iter(pairs)
    while batch := list(islice(iterator, batch_size)):
        if batch_size == 1:
            i,j,R,ci,cj = batch[0]
            s,t = two_center.scalar_pair(atom_species[i],atom_species[j],ci,cj)
            yield batch[0],s,t
            continue
        symbols = [(atom_species[i],atom_species[j]) for i,j,_,_,_ in batch]
        displacements = np.asarray([np.asarray(cj,dtype=np.float64)-np.asarray(ci,dtype=np.float64)
                                    for _,_,_,ci,cj in batch])
        gs,gt = two_center.eval_two_center_batch(symbols,displacements)
        # One transfer per tensor per batch; the public result does not retain
        # padding or the entire batch through a small NumPy view.
        ss,tt = gs.cpu().numpy(),gt.cpu().numpy()
        del gs,gt
        for edge,(si,sj),s,t in zip(batch,symbols,ss,tt):
            ni,nj = two_center.norb_per_species[si],two_center.norb_per_species[sj]
            yield edge,s[:ni,:nj].copy(),t[:ni,:nj].copy()


def with_local_chunks(evaluated_pairs, cache, atom_species, evaluators, positions, batch_size):
    iterator=iter(evaluated_pairs)
    while chunk := list(islice(iterator,batch_size)):
        local=cache.contract_chunk([edge for edge,_,_ in chunk],atom_species,evaluators,positions)
        for (edge,s,t),(v,z) in zip(chunk,local):
            yield edge,s,t,v,z

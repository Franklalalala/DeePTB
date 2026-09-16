"""CPU/SciPy reference for full spinor NACF and dimensionless overlap.

The historical scalar P2 assembler supplies its original geometry enumeration,
batched SciPy interpolation and cache-budget fallback. This adapter lifts its
real factors to spin space and retains complex D, including the fallback path.
It is a full-SOC extension of that CPU algorithm, not a historical SOC timing.
"""
import copy

import numpy as np
import torch

from dptb.data.interfaces.p2_table import P2TableAssembler
from dptb.data.interfaces.p23_table import P23VNAFactorAssembler


def _lift(value):
    return np.kron(np.eye(2), value)


class _SpinRadial:
    def __init__(self, source):
        self.source = source

    def evaluate(self, vector):
        return _lift(self.source.evaluate(vector))

    def evaluate_many(self, vectors):
        return _lift(self.source.evaluate_many(vectors))


class _SpinStore:
    def __init__(self, p2, soc):
        self.p2, self.soc = p2, soc
        self.species = copy.deepcopy(p2.species)
        for meta in self.species.values():
            for field in ('orbital_norb', 'projector_norb'):
                meta[field] *= 2
            for field in ('orbital_shells', 'projector_shells', 'projector_cutoffs_bohr'):
                meta[field] = list(meta[field])*2

    def onsite_component(self, symbol, kind):
        return _lift(self.p2.onsite_component(symbol, kind))

    def base_component(self, left, right, kind):
        return _SpinRadial(self.p2.base_component(left, right, kind))

    def projector(self, left, right):
        return _SpinRadial(self.p2.projector(left, right))

    def d_eff(self, symbol):
        return self.soc.d_spinor(symbol)


class CPUScalarAlgorithmSOCAssembler(P2TableAssembler):
    """Use the legacy algorithm with real factors and complex spinor output."""
    def __init__(self, p2, soc, **kwargs):
        super().__init__(_SpinStore(p2, soc), **kwargs)

    @staticmethod
    def _zero_block(geometry):
        return np.zeros((geometry.ni, geometry.nj), dtype=np.complex128)

    def _assemble_block_from_geometry(self, geometry, *, context=None):
        # The parent's scalar cache-budget fallback casts the final H to real.
        base = (self._zero_block(geometry) if geometry.outside_support else
                self._base_component_block(geometry, 'p2_base'))
        result = base + self._assemble_vnl(geometry, context=context)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite CPU SOC prior')
        return result


def cpu_soc_blocks(bank, symbols, positions_bohr, cell_bohr, edges, shifts):
    """Return padded spin-major AO H (eV) and S using CPU arithmetic only."""
    symbols = tuple(symbols)
    edges, shifts = np.asarray(edges), np.asarray(shifts, dtype=int)
    positions, cell = np.asarray(positions_bohr), np.asarray(cell_bohr)
    sizes = [int(bank.p2.species[s]['orbital_norb']) for s in symbols]
    n, width = len(symbols), max(sizes)
    keys = [(i,i,0,0,0) for i in range(n)] + [
        (int(i),int(j),*map(int,shift)) for (i,j),shift in zip(edges.T,shifts)]
    assembled = CPUScalarAlgorithmSOCAssembler(bank.p2,bank.soc).assemble_sparse_blocks(
        symbols=symbols,positions_bohr=positions,cell_bohr=cell,block_keys=keys)
    prior = np.zeros((len(keys),2*width,2*width),dtype=np.complex128)
    overlap = np.zeros_like(prior)
    for row,(i,j,*shift) in enumerate(keys):
        ii = np.r_[np.arange(sizes[i]),width+np.arange(sizes[i])]
        jj = np.r_[np.arange(sizes[j]),width+np.arange(sizes[j])]
        prior[row][np.ix_(ii,jj)] = assembled[keys[row]]*bank.ry_to_ev
        scalar_s = (bank.overlap.onsite_component(symbols[i],'overlap') if row<n else
            bank.overlap.base_component(symbols[i],symbols[j],'overlap').evaluate(
                positions[j]+np.asarray(shift)@cell-positions[i]))
        overlap[row][np.ix_(ii,jj)] = _lift(scalar_s)
    use_p23,_ = bank.p23_composition(symbols)
    if use_p23:
        addition,_,_ = P23VNAFactorAssembler(bank.p23,factor_dtype=np.float64).assemble_graph_addition(
            symbols=symbols,positions_bohr=positions,cell_bohr=cell,
            edge_index=np.empty((2,0),dtype=int),edge_cell_shift=np.empty((0,3),dtype=int),
            node_shapes=np.array([[s,s] for s in sizes]),edge_shapes=np.empty((0,2),dtype=int),
            node_pad_shape=(width,width),edge_pad_shape=(width,width))
        prior[:n] += _lift(addition)
    rows = {key:row for row,key in enumerate(keys[n:])}
    reverse = [rows[(j,i,-x,-y,-z)] for i,j,x,y,z in keys[n:]]
    output = {}
    for array,node_key,edge_key in ((prior,'node_p23_ao_ev','edge_p2_ao_ev'),
                                  (overlap,'node_overlap_ao','edge_overlap_ao')):
        output[node_key] = (array[:n]+array[:n].transpose(0,2,1).conj())*.5
        output[edge_key] = (array[n:]+array[n:][reverse].transpose(0,2,1).conj())*.5
    return output


def cpu_soc_features(predictor, atoms, prepared):
    """Include CPU gauge/RME packing and host-to-device transfer in timing."""
    from dptb.data.transforms import OrbitalMapper
    from dptb.data.interfaces.abacus import OrbAbacus2DeepTB
    from dptb.data.interfaces.ham_to_feature import block_to_feature
    from dptb.utils.constants import Bohr2Ang
    if not np.all(atoms.pbc):
        raise ValueError('the legacy CPU SOC benchmark requires fully periodic cells')
    if not hasattr(predictor,'cpu_full_mapper'):
        mapper = predictor.full_mapper
        predictor.cpu_full_mapper = OrbitalMapper(copy.deepcopy(mapper.basis),method='e3tb',
            chemical_symbol_to_type=mapper.chemical_symbol_to_type,has_soc=True,
            full_soc_prediction=True,nextham_uureal_mask=False,soc_complex_doubling=True,device='cpu')
    idp = predictor.cpu_full_mapper
    edges = prepared.geometry['edge_index'].cpu().numpy()
    shifts = prepared.geometry['edge_cell_shift'].cpu().numpy().astype(int)
    symbols = atoms.get_chemical_symbols()
    blocks = cpu_soc_blocks(predictor.bank,symbols,atoms.positions/Bohr2Ang,
                            atoms.cell.array/Bohr2Ang,edges,shifts)
    sizes = [int(predictor.bank.p2.species[s]['orbital_norb']) for s in symbols]
    width = max(sizes)
    keys = [(i,i,0,0,0) for i in range(len(atoms))] + [
        (int(i),int(j),*map(int,shift)) for (i,j),shift in zip(edges.T,shifts)]
    converter, result = OrbAbacus2DeepTB(), {}
    for node,edge,out_node,out_edge in (
            ('node_p23_ao_ev','edge_p2_ao_ev','node_p23','edge_p2'),
            ('node_overlap_ao','edge_overlap_ao','node_overlap','edge_overlap')):
        arrays = np.concatenate((blocks[node],blocks[edge]))
        converted = {}
        for row,(i,j,*shift) in enumerate(keys):
            ii = np.r_[np.arange(sizes[i]),width+np.arange(sizes[i])]
            jj = np.r_[np.arange(sizes[j]),width+np.arange(sizes[j])]
            converted['_'.join(map(str,keys[row]))] = converter.transform(
                arrays[row][np.ix_(ii,jj)],
                list(predictor.bank.p2.species[symbols[i]]['orbital_shells'])*2,
                list(predictor.bank.p2.species[symbols[j]]['orbital_shells'])*2)
        data = {'atomic_numbers':torch.tensor(atoms.numbers[:,None]),
                'edge_index':torch.tensor(edges),'edge_cell_shift':torch.tensor(shifts)}
        idp(data)
        block_to_feature(data,idp,converted,missing_block_policy='error',output_dtype=predictor.dtype)
        result[out_node] = data['node_features'].to(predictor.device)
        result[out_edge] = data['edge_features'].to(predictor.device)
    return result

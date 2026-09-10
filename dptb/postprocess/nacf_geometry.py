"""Geometry-only inference for non-SOC models trained on Full-H minus NACF.

The public input is ASE geometry (species, positions, cell and PBC), never H0,
DFT labels or precomputed structure priors. Preparation currently runs neighbour
enumeration on CPU. Numerical prior assembly and the model execute on device.
"""
from __future__ import annotations

import torch

from dptb.data import AtomicData
from dptb.data.dataloader import Collater
from dptb.data.interfaces.nacf_gpu import NACFFeaturePlan, NACFBatchAssemblyPlan
from dptb.utils.argcheck import get_cutoffs_from_model_options
from dptb.utils.constants import Bohr2Ang


class NACFGeometryPredictor:
    def __init__(self, model, table_bank, model_options, *, target,
                 expected_p2_source_fingerprint):
        if target != 'full_h_minus_nacf':
            raise ValueError('geometry NACF inference requires explicit full_h_minus_nacf target')
        if not expected_p2_source_fingerprint or expected_p2_source_fingerprint != table_bank.p2_manifest_sha256:
            raise ValueError('table bank does not match the checkpoint training P2 fingerprint')
        embedding = model_options.get('embedding', {})
        if embedding.get('method') not in ('lem_moe_v3_prior', 'lem_moe_v3_prior_2b') or embedding.get('prior_kind') != 'na_cf':
            raise ValueError('model is not an NACF-conditioned residual model')
        self.model = model.eval()
        self.bank = table_bank
        self.idp = model.hamiltonian.idp if hasattr(model, 'hamiltonian') else model.idp
        if self.idp.has_soc:
            raise ValueError('spinor SOC requires a separate assembly and packing contract')
        self.cutoffs = get_cutoffs_from_model_options(model_options)
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        if self.device != table_bank._anchor.device:
            raise ValueError('model and table bank must share a device')

    def prepare(self, structures):
        """Prepare a geometry or list of geometries; call again after any edit."""
        if hasattr(structures, 'get_positions'):
            structures = [structures]
        structures = list(structures)
        if not structures:
            raise ValueError('empty structure batch')
        assemblies, data = [], []
        rmax, ermax, oermax = self.cutoffs
        for atoms in structures:
            # from_points avoids importing any calculator or arbitrary arrays
            # attached to ASE Atoms; this is deliberately geometry-only input.
            graph = AtomicData.from_points(pos=atoms.get_positions(), cell=atoms.cell.array,
                                           pbc=atoms.pbc, atomic_numbers=atoms.get_atomic_numbers(),
                                           r_max=rmax, er_max=ermax, oer_max=oermax)
            assembly = self.bank.prepare(atoms.get_chemical_symbols(), atoms.get_positions() / Bohr2Ang,
                                         atoms.cell.array / Bohr2Ang, graph['edge_index'].numpy(),
                                         graph['edge_cell_shift'].numpy(), pbc=atoms.pbc)
            assemblies.append(assembly)
            data.append(self.idp(graph))
        batch = Collater()(data).to(self.device)
        geometry = AtomicData.to_AtomicDataDict(batch)
        assembly = assemblies[0] if len(assemblies) == 1 else NACFBatchAssemblyPlan(assemblies)
        plans = [NACFFeaturePlan(assembly, self.idp, output_dtype=self.dtype)]
        return PreparedNACFInference(self.model, plans, geometry)

    def __call__(self, structures):
        return self.prepare(structures)()


class PreparedNACFInference:
    """Reusable only for the identical geometry used in prepare().

    Repeated calls recompute priors on GPU, which allows honest warm timing.
    Returned H is absolute Full-H in eV and S is dimensionless, both in the
    checkpoint's triangular non-SOC RME layout. ``ptr`` separates structures.
    """
    def __init__(self, model, plans, geometry):
        self.model, self.plans, self.geometry = model, plans, geometry

    @torch.inference_mode()
    def __call__(self):
        features = [plan() for plan in self.plans]
        inputs = {key: value.clone() for key, value in self.geometry.items()}
        for key in ('node_p23', 'edge_p2', 'node_overlap', 'edge_overlap'):
            inputs[key] = torch.cat([item[key] for item in features], dim=0)
        # Preserve an independent prior before a model that mutates its dict.
        node_prior, edge_prior = inputs['node_p23'].clone(), inputs['edge_p2'].clone()
        node_s, edge_s = inputs['node_overlap'].clone(), inputs['edge_overlap'].clone()
        prediction = self.model(inputs)
        output = {key: value.clone() for key, value in self.geometry.items()}
        output.update(node_features=prediction['node_features'] + node_prior,
                      edge_features=prediction['edge_features'] + edge_prior,
                      node_overlap=node_s, edge_overlap=edge_s)
        return output


__all__ = ['NACFGeometryPredictor', 'PreparedNACFInference']

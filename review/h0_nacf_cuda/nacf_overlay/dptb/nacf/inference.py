"""Geometry-only inference for models trained on Full-H minus NACF.

The public input is ASE geometry (species, positions, cell and PBC), never H0,
DFT labels or precomputed structure priors. Preparation currently runs neighbour
enumeration on CPU. Numerical prior assembly and the model execute on device.
"""
from __future__ import annotations

import torch
import copy
from pathlib import Path

from dptb.data import AtomicData
from dptb.data.dataloader import Collater
from dptb.nacf.assembly import NACFFeaturePlan, NACFBatchAssemblyPlan
from dptb.utils.argcheck import get_cutoffs_from_model_options
from dptb.utils.constants import Bohr2Ang
from dptb.data.interfaces.p2_table import P2TableStore
from dptb.data.interfaces.p23_table import P23VNAFactorTableStore
from .overlap import OverlapTableStore
from .assembly import NACFTableBank


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
        if self.idp.has_soc and not self.idp.nextham_uureal_mask and table_bank.soc is None:
            raise ValueError('full SOC requires a source-bound spinor projector store')
        if not self.idp.has_soc and table_bank.soc is not None:
            raise ValueError('SOC projector store cannot be used with a non-SOC checkpoint')
        self.cutoffs = get_cutoffs_from_model_options(model_options)
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        if self.device != table_bank._anchor.device:
            raise ValueError('model and table bank must share a device')

    def prepare(self, structures):
        """Prepare a geometry or list of geometries; call again after any edit."""
        plan, geometry = prepare_geometry(self.bank, self.idp, structures, self.cutoffs, output_dtype=self.dtype)
        return PreparedNACFInference(self.model, plan, geometry)

    def __call__(self, structures):
        return self.prepare(structures)()


def prepare_geometry(bank, idp, structures, cutoffs, *, output_dtype=torch.float32):
    """Prepare NACF/S inputs independently of a learned model, including full SOC.

    Returns a callable feature plan and the geometry graph on the bank's device.
    Only species, coordinates, cell and PBC are read from the ASE objects.
    """
    if hasattr(structures, 'get_positions'):
        structures = [structures]
    structures = list(structures)
    if not structures:
        raise ValueError('empty structure batch')
    assemblies, data = [], []
    rmax, ermax, oermax = cutoffs
    for atoms in structures:
        graph = AtomicData.from_points(pos=atoms.get_positions(), cell=atoms.cell.array,
                                       pbc=atoms.pbc, atomic_numbers=atoms.get_atomic_numbers(),
                                       r_max=rmax, er_max=ermax, oer_max=oermax)
        assembly = bank.prepare(atoms.get_chemical_symbols(), atoms.get_positions() / Bohr2Ang,
                                atoms.cell.array / Bohr2Ang, graph['edge_index'].numpy(),
                                graph['edge_cell_shift'].numpy(), pbc=atoms.pbc)
        assemblies.append(assembly)
        data.append(idp(graph))
    batch = Collater()(data).to(bank._anchor.device)
    geometry = AtomicData.to_AtomicDataDict(batch)
    assembly = assemblies[0] if len(assemblies) == 1 else NACFBatchAssemblyPlan(assemblies)
    return NACFFeaturePlan(assembly, idp, output_dtype=output_dtype), geometry


class PreparedNACFInference:
    """Reusable only for the identical geometry used in prepare().

    Repeated calls recompute priors on GPU, which allows honest warm timing.
    Returned H is absolute Full-H in eV and S is dimensionless, both in the
    checkpoint's declared RME layout. ``ptr`` separates structures. Full SOC
    retains all four spin blocks and real/imaginary channels; uu-real models
    return only their explicitly reduced target and cannot predict full SOC H.
    """
    def __init__(self, model, plan, geometry):
        self.model, self.plan, self.geometry = model, plan, geometry

    @torch.inference_mode()
    def __call__(self):
        features = self.plan()
        inputs = {key: value.clone() for key, value in self.geometry.items()}
        for key in ('node_p23', 'edge_p2', 'node_overlap', 'edge_overlap'):
            inputs[key] = features[key]
        # Preserve an independent prior before a model that mutates its dict.
        node_prior, edge_prior = inputs['node_p23'].clone(), inputs['edge_p2'].clone()
        node_s, edge_s = inputs['node_overlap'].clone(), inputs['edge_overlap'].clone()
        prediction = self.model(inputs)
        output = {key: value.clone() for key, value in self.geometry.items()}
        output.update(node_features=prediction['node_features'] + node_prior,
                      edge_features=prediction['edge_features'] + edge_prior,
                      node_overlap=node_s, edge_overlap=edge_s)
        return output


__all__ = ['NACFGeometryPredictor', 'PreparedNACFInference', 'prepare_geometry', 'load_predictor']


def load_predictor(checkpoint, p2, p23, overlap, expected_p2_sha256, device='cuda', backend='auto', soc=None,
                   *, model_backend='checkpoint', p23_missing_policy='error', expected_p23_sha256=None):
    """Load the trained model backend unless an explicit reference run is requested."""
    if model_backend not in ('checkpoint', 'reference'):
        raise ValueError("model_backend must be 'checkpoint' or 'reference'")
    from dptb.nn import build_model
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload['model_state_dict']
    for name, tensor in state.items():
        if torch.is_tensor(tensor) and (tensor.is_floating_point() or tensor.is_complex()):
            if not torch.isfinite(tensor).all():
                raise ValueError(f'nonfinite checkpoint model tensor: {name}')
    options = copy.deepcopy(payload['config']['model_options'])
    del payload, state
    overrides = ({'so2_fusion_mode':'streamed_m_major_ref', 'mole_linear_mode':'split_loop'}
                 if model_backend == 'reference' else {})
    options['embedding'].update(overrides)
    from .soc import SOCProjectorStore
    p2_store = P2TableStore(p2)
    soc_store = None if soc is None else SOCProjectorStore(soc, p2_store)
    bank = NACFTableBank(p2_store, P23VNAFactorTableStore(p23),
                         overlap_store=OverlapTableStore(overlap), soc_store=soc_store, device=device, backend=backend,
                         p23_missing_policy=p23_missing_policy, expected_p23_sha256=expected_p23_sha256)
    if bank.p2_manifest_sha256 != expected_p2_sha256:
        raise ValueError('P2 manifest does not match supplied training fingerprint')
    model = build_model(checkpoint=str(checkpoint), model_options=options,
                        common_options={}).eval().to(device)
    predictor = NACFGeometryPredictor(model, bank, options, target='full_h_minus_nacf',
                                      expected_p2_source_fingerprint=expected_p2_sha256)
    predictor.runtime_model_overrides = overrides
    return predictor

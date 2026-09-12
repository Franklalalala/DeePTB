"""Geometry-only SOC inference with a compact uu-real residual checkpoint."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import torch

from dptb.data.transforms import OrbitalMapper
from dptb.utils.argcheck import get_cutoffs_from_model_options
from .inference import prepare_geometry
from .spinor_completion import SOCUURealCompletion


def _model_mapper(model):
    return model.hamiltonian.idp if hasattr(model, 'hamiltonian') else model.idp


class SOCResidualPair(torch.nn.Module):
    """Evaluate independent onsite/hopping arms on independent model inputs.

    This composes their predictions without merging state dictionaries. Both
    complete forwards are part of the AI latency. A DeePTB forward may mutate
    its input dictionary and tensors, so each arm receives its own copy.
    """

    def __init__(self, onsite_model, hopping_model):
        super().__init__()
        onsite_mapper, hopping_mapper = map(_model_mapper, (onsite_model, hopping_model))
        for mapper in (onsite_mapper, hopping_mapper):
            if not (mapper.has_soc and mapper.nextham_uureal_mask):
                raise ValueError('paired SOC arms must both predict compact uu-real residuals')
        for name in ('basis', 'chemical_symbol_to_type', 'reduced_matrix_element'):
            if getattr(onsite_mapper, name) != getattr(hopping_mapper, name):
                raise ValueError('paired SOC arms differ in '+name)
        for mapper in (onsite_mapper, hopping_mapper):
            mapper.get_orbpair_maps()
        if onsite_mapper.orbpair_maps != hopping_mapper.orbpair_maps:
            raise ValueError('paired SOC arms differ in orbital-pair order')
        first, second = next(onsite_model.parameters()), next(hopping_model.parameters())
        if first.device != second.device or first.dtype != second.dtype:
            raise ValueError('paired SOC arms must share device and dtype')
        self.onsite_model = onsite_model.eval()
        self.hopping_model = hopping_model.eval()

    @property
    def idp(self):
        return _model_mapper(self.onsite_model)

    def prepare_arm_inputs(self, inputs):
        onsite_inputs = {key:value.clone() for key,value in inputs.items()}
        hopping_inputs = {key:value.clone() for key,value in inputs.items()}
        return onsite_inputs, hopping_inputs

    def forward_arms(self, inputs):
        """Both neural forwards; independent input copies are prepared outside."""
        onsite_inputs, hopping_inputs = inputs
        onsite = self.onsite_model(onsite_inputs)
        hopping = self.hopping_model(hopping_inputs)
        return {'node_features':onsite['node_features'],
                'edge_features':hopping['edge_features']}

    def forward(self, inputs):
        return self.forward_arms(self.prepare_arm_inputs(inputs))


class SOCGeometryPredictor:
    """Complete all SOC channels from NACF plus a learned real residual.

    ``prior_kind`` and ``target_kind`` must be verified against the checkpoint's
    actual training data contract by the caller. Historical ``h0`` field names
    alone neither establish nor rule out NACF conditioning.
    """

    def __init__(self, model, bank, model_options, *, prior_kind, target_kind,
                 expected_p2_source_fingerprint, conditioning_keys):
        if prior_kind != 'na_cf' or target_kind != 'full_h_minus_nacf':
            raise ValueError('SOC completion requires NACF input and Full-H minus NACF residual')
        if not expected_p2_source_fingerprint or bank.p2_manifest_sha256 != expected_p2_source_fingerprint:
            raise ValueError('SOC table source differs from the declared training prior')
        if bank.soc is None or bank.overlap is None:
            raise ValueError('complete SOC inference requires spinor NACF and overlap tables')
        if tuple(conditioning_keys) not in (('node_h0','edge_h0'),('node_p23','edge_p2')):
            raise ValueError('unsupported model conditioning keys')
        self.model = model.eval()
        self.bank = bank
        self.compact_mapper = _model_mapper(model)
        if not (self.compact_mapper.has_soc and self.compact_mapper.nextham_uureal_mask):
            raise ValueError('expected a compact SOC uu-real residual model')
        parameter = next(model.parameters())
        self.device,self.dtype = parameter.device,parameter.dtype
        if self.dtype.is_complex or self.device != bank._anchor.device:
            raise ValueError('real residual model and table bank must share a device')
        self.full_mapper = OrbitalMapper(copy.deepcopy(self.compact_mapper.basis),
            chemical_symbol_to_type=self.compact_mapper.chemical_symbol_to_type,
            method='e3tb',has_soc=True,full_soc_prediction=True,
            nextham_uureal_mask=False,soc_complex_doubling=True,device=self.device)
        self.completion = SOCUURealCompletion(self.compact_mapper,self.full_mapper,device=self.device)
        self.conditioning_keys = tuple(conditioning_keys)
        self.cutoffs = get_cutoffs_from_model_options(model_options)

    def prepare(self, structures):
        plan,geometry = prepare_geometry(self.bank,self.full_mapper,structures,self.cutoffs,
                                         output_dtype=self.dtype)
        return PreparedSOCInference(self.model,plan,geometry,self.completion,self.conditioning_keys)

    def __call__(self, structures):
        return self.prepare(structures)()


class PreparedSOCInference:
    def __init__(self, model, plan, geometry, completion, conditioning_keys):
        self.model,self.plan,self.geometry = model,plan,geometry
        self.completion,self.conditioning_keys = completion,conditioning_keys

    def model_inputs(self, full_features):
        """Independent resident inputs; prepare outside pure-forward timing."""
        inputs = {key:value.clone() for key,value in self.geometry.items()}
        for field in ('node_features','edge_features'):
            inputs.pop(field,None)
        for output,source in zip(self.conditioning_keys,('node_p23','edge_p2')):
            inputs[output] = self.completion.extract_prior(full_features[source]).clone()
        for key in ('node_overlap','edge_overlap'):
            inputs[key] = self.completion.extract_prior(full_features[key]).clone()
        return inputs

    @torch.inference_mode()
    def __call__(self):
        features = self.plan()
        prediction = self.model(self.model_inputs(features))
        result = {key:value.clone() for key,value in self.geometry.items()}
        result.update(node_features=self.completion(features['node_p23'],prediction['node_features']),
                      edge_features=self.completion(features['edge_p2'],prediction['edge_features']),
                      node_overlap=features['node_overlap'],edge_overlap=features['edge_overlap'])
        return result


def _residual_contract(config):
    """Read NACF semantics from training configuration, never the file name."""
    common = config['common_options']
    training = config['data_options']['train']
    embedding = config['model_options']['embedding']
    if not (common.get('has_soc') and common.get('nextham_uureal_mask')
            and not common.get('full_soc_prediction', False)):
        raise ValueError('checkpoint is not a compact SOC uu-real model')
    if training.get('prior_kind') != 'na_cf' or training.get('target_kind') not in (
            'nacfres', 'full_h_minus_nacf'):
        raise ValueError('checkpoint must train NACF input and Full-H minus NACF residual')
    if not training.get('get_P2'):
        raise ValueError('checkpoint training did not request NACF prior features')
    if config.get('train_options', {}).get('flow_options', {}).get('enabled', False):
        raise ValueError('flow checkpoint requires its own inference integrator')
    keys = (embedding.get('h0_node_key'), embedding.get('h0_edge_key'))
    if keys not in (('node_p23','edge_p2'), ('node_h0','edge_h0')):
        raise ValueError('checkpoint has unsupported NACF conditioning keys')
    return keys


def _training_contract_config(embedded, sidecar_path):
    """Resolve dataset semantics omitted by some production checkpoints."""
    if sidecar_path is None:
        if 'data_options' not in embedded:
            raise ValueError('checkpoint omits data_options; supply its verified training config sidecar')
        return embedded
    sidecar = json.loads(Path(sidecar_path).read_text(encoding='utf-8-sig'))
    for name in ('basis','has_soc','nextham_uureal_mask','full_soc_prediction'):
        if embedded['common_options'].get(name,False) != sidecar['common_options'].get(name,False):
            raise ValueError('training sidecar disagrees with checkpoint '+name)
    for name in ('method','h0_node_key','h0_edge_key','h0_merge_mode','h0_node_mode'):
        if embedded['model_options']['embedding'].get(name) != sidecar['model_options']['embedding'].get(name):
            raise ValueError('training sidecar disagrees with checkpoint embedding '+name)
    if 'data_options' in embedded and _residual_contract(embedded) != _residual_contract(sidecar):
        raise ValueError('training sidecar disagrees with embedded dataset contract')
    return sidecar


def load_soc_predictor(onsite_checkpoint, hopping_checkpoint, *, p2, p23, overlap,
                       soc, expected_p2_sha256, device='cuda:0', backend='cuda',
                       ry_to_ev=13.605693122994, onsite_config=None, hopping_config=None,
                       p23_missing_policy='error', expected_p23_sha256=None):
    """Load two separately trained NACF residual arms and complete spinor H/S.

    ``expected_p2_sha256`` must come from a verified training-source audit. The
    stores also check that P23, overlap and SOC sidecars bind this P2 manifest.
    The conversion default matches the SOC29303 join generator. For another
    dataset, supply its verified conversion explicitly. Checkpoint loading,
    finite scans and table setup are outside warm timing.

    If a checkpoint omits dataset configuration, its same-run training sidecar
    is required. The caller must establish checkpoint/sidecar provenance; common
    basis and conditioning semantics are cross-checked here before model load.
    """
    from dptb.nn import build_model
    from dptb.data.interfaces.p2_table import P2TableStore
    from dptb.data.interfaces.p23_table import P23VNAFactorTableStore
    from .assembly import NACFTableBank
    from .overlap import OverlapTableStore
    from .soc import SOCProjectorStore

    models, configs, conditioning = [], [], []
    for checkpoint,sidecar in zip((onsite_checkpoint, hopping_checkpoint),(onsite_config,hopping_config)):
        path = Path(checkpoint)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        config = copy.deepcopy(payload['config'])
        training_config = _training_contract_config(config,sidecar)
        conditioning.append(_residual_contract(training_config))
        for name, tensor in payload['model_state_dict'].items():
            if torch.is_tensor(tensor) and (tensor.is_floating_point() or tensor.is_complex()):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f'nonfinite checkpoint tensor in {path.name}: {name}')
        del payload
        options = copy.deepcopy(config['model_options'])
        # Preserve the trained routing backend. In particular, prior-activated
        # edge routing cannot use the generic streamed_m_major_ref override.
        common = copy.deepcopy(config['common_options'])
        common['device'] = str(device)
        models.append(build_model(checkpoint=str(path), model_options=options,
                                   common_options=common).eval().to(device))
        configs.append(config)
    if conditioning[0] != conditioning[1]:
        raise ValueError('paired checkpoints use different prior input keys')
    if get_cutoffs_from_model_options(configs[0]['model_options']) != get_cutoffs_from_model_options(configs[1]['model_options']):
        raise ValueError('paired checkpoints require different geometry graphs')
    pair = SOCResidualPair(*models)
    source = P2TableStore(p2)
    bank = NACFTableBank(source, P23VNAFactorTableStore(p23),
        overlap_store=OverlapTableStore(overlap), soc_store=SOCProjectorStore(soc,source),
        device=device, backend=backend, ry_to_ev=ry_to_ev,
        p23_missing_policy=p23_missing_policy, expected_p23_sha256=expected_p23_sha256)
    predictor = SOCGeometryPredictor(pair, bank, configs[0]['model_options'],
        prior_kind='na_cf', target_kind='full_h_minus_nacf',
        expected_p2_source_fingerprint=expected_p2_sha256,
        conditioning_keys=conditioning[0])
    predictor.checkpoint_paths = tuple(str(Path(p).resolve()) for p in (onsite_checkpoint,hopping_checkpoint))
    predictor.runtime_model_overrides = {}
    return predictor

"""Head pairs of a route: a node ('on') and an edge ('hp') model reading the same prior features.

Each checkpoint runs the SO2 route it was trained with (LX1: activation-space ``streamed_m_major_fused_p0`` with
``cublas_grouped`` expert linears). With SO2CUDA installed these layers call its fused kernels and grouped GEMM;
without it ``dptb.nn.so2_backend`` falls back to the grouped PyTorch reference, the route the production pipeline
evaluates. float32, eval mode, no TF32. The model input carries the prior features in the feature slots and in the
named prior slots, the physical overlap packet for the charge-balance head, and flow time 0 for the noise shell.
"""
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from dptb.data import AtomicDataDict as K
from dptb.nn import so2_backend
from dptb.nn.build import build_model
from dptb.nn.checkpoint import load_checkpoint
from dptb.nn.heads.charge_response import PeriodicGammaCache, periodic_gamma_scope


def flow_time_keys(model):
    keys = []
    for m in model.modules():
        c = getattr(m, 'flow_time_conditioner', None)
        if c is not None:
            keys += list(c.flow_time_keys or (c.flow_time_key,))
    return sorted(set(keys))


def load_head(path, device):
    """(model, meta) of one checkpoint."""
    path = Path(path)
    ck = load_checkpoint(str(path))
    emb = ck['config']['model_options']['embedding']
    common = copy.deepcopy(ck['config']['common_options'])
    common.update(device=str(device), dtype='float32')
    model = build_model(checkpoint=str(path), common_options=common)
    model.eval()
    meta = dict(nk=emb.get('h0_node_key', K.NODE_H0_KEY), ek=emb.get('h0_edge_key', K.EDGE_H0_KEY),
                shift_on=(model.model_options.get('shift_head') or {}).get('mode', 'off') != 'off',
                flow_time_keys=flow_time_keys(model))
    return model, meta


class HeadPair:
    def __init__(self, on, hp, device='cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        self.heads = {'on': load_head(on, self.device), 'hp': load_head(hp, self.device)}
        from dptb.utils.argcheck import get_cutoffs_from_model_options
        cuts = [get_cutoffs_from_model_options(m.model_options) for m, _ in self.heads.values()]
        if cuts[0] != cuts[1]:
            raise ValueError('head graph cutoffs differ')
        self.model = self.heads['on'][0]
        self.idp = self.model.idp
        # the two heads are independent: the edge head runs in a worker thread on its own stream, so its kernels
        # overlap the node head's host-side launch and routing work (small structures are launch-bound)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dptb-head') \
            if self.device.type == 'cuda' else None
        self._stream = torch.cuda.Stream(self.device) if self._pool else None
        self.share_qeq_geometry = os.environ.get('DPTB_INFERENCE_SHARE_QEQ', '1') == '1'

    @property
    def so2(self):
        """'so2cuda' when the optional SO2CUDA package serves the SO2 layers, else 'reference'."""
        return 'so2cuda' if so2_backend.backend() is not None else 'reference'

    def forward(self, base, prior_node, prior_edge, s_node, s_edge):
        """Compact deltas (node [n, width], edge [E, width], float32) of both heads for one structure."""
        # Scope the shared geometry to this fresh call, including concurrent heads.
        gamma_cache = PeriodicGammaCache() if self.share_qeq_geometry else None
        args = (base, prior_node, prior_edge, s_node, s_edge, gamma_cache)
        if self._pool is None:
            return self._head('on', *args), self._head('hp', *args)
        caller = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(caller)

        def edge_head():
            with torch.cuda.stream(self._stream):
                out = self._head('hp', *args)
                done = torch.cuda.Event()
                done.record(self._stream)
            return out, done

        job = self._pool.submit(edge_head)
        node = self._head('on', *args)
        edge, done = job.result()
        caller.wait_event(done)
        edge.record_stream(caller)
        return node, edge

    def _head(self, head, base, prior_node, prior_edge, s_node, s_edge, gamma_cache=None):
        model, meta = self.heads[head]
        data = self.idp({k: v.clone() for k, v in base.items()})
        for k in (K.NODE_FEATURES_KEY, 'node_h0', 'node_p23', K.NODE_H0_KEY, meta['nk']):
            data[k] = prior_node.clone()
        for k in (K.EDGE_FEATURES_KEY, 'edge_h0', 'edge_p2', K.EDGE_H0_KEY, meta['ek']):
            data[k] = prior_edge.clone()
        if meta['shift_on']:
            if K.ATOMIC_NUMBERS_KEY not in data:
                data[K.ATOMIC_NUMBERS_KEY] = self.idp.untransform(data[K.ATOM_TYPE_KEY].reshape(-1)).long()
            data['phys_node_overlap'] = s_node.to(torch.float32)
            data['phys_edge_overlap'] = s_edge.to(torch.float32)
        if meta['flow_time_keys'] and K.BATCH_KEY not in data:
            data[K.BATCH_KEY] = torch.zeros(len(data[K.POSITIONS_KEY]), dtype=torch.long, device=self.device)
        for k in meta['flow_time_keys']:
            data[k] = torch.zeros(1, dtype=torch.float32, device=self.device)
        with torch.inference_mode(), periodic_gamma_scope(gamma_cache):
            pred = model(data)
        return pred[K.NODE_FEATURES_KEY if head == 'on' else K.EDGE_FEATURES_KEY].detach().float()

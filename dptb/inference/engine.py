"""Structure -> H, S (and the residual) as device block tables, for both routes.

    graph    model neighbour graph on the device (graph.py)
    prior    GF (Flash) or H0 (UniTB) prior blocks over its support, DeePTB gauge, eV, FP64 (dptb.prior.gf / .h0)
    model    prior features of the model graph -> node and edge heads -> compact delta (heads.py, features.py)
    finish   delta blocks, HR2HK symmetrisation, H = prior + delta, complex64 / float32 tables

Outputs follow the W1 block contract: H complex64 full SOC (up orbitals, then down), S float32, the residual (UniTB)
float32 uu-real H - H0; keys i_j_Rx_Ry_Rz over the prior support. Nothing leaves the device unless asked for.
"""
import time

import torch

from dptb.inference.features import FeatureMaps
from dptb.inference.graph import build_graph, species_cutoffs
from dptb.inference.heads import HeadPair
from dptb.nn.so2_backend import STATS as SO2_STATS
from dptb.prior.common.blocks import BlockTable, KeyIndex, symmetrise


class Result:
    def __init__(self, H, S, residual, nelec, timing, info):
        self.H, self.S, self.residual, self.nelec, self.timing, self.info = H, S, residual, nelec, timing, info


class InferenceEngine:
    """route 'flash' (GF prior) or 'unitb' (H0 prior); ``prior`` is the prior object of that route."""

    def __init__(self, route, prior, on, hp, device='cuda'):
        if route not in ('flash', 'unitb'):
            raise ValueError('route is flash or unitb')
        self.route = route
        self.device = torch.device(device)
        self.prior = prior
        self.heads = HeadPair(on, hp, self.device)
        self.cutoffs = species_cutoffs(self.heads.model)
        self.maps = FeatureMaps(self.heads.idp, self.device)

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def run(self, numbers, positions, cell, prefetch=None):
        """numbers [n], positions [n, 3] and cell [3, 3] in Angstrom (arrays or tensors) of one periodic structure.

        prefetch: atomic numbers of the next structure; its prior tables are read in the background while this one
        runs (cold compositions then cost no table I/O on the critical path).
        """
        tm = {}
        dev = self.device
        t0 = time.perf_counter()
        if prefetch is not None:
            self.prior.prefetch(prefetch)
        z = torch.as_tensor(numbers, dtype=torch.int64, device=dev)
        pos = torch.as_tensor(positions, dtype=torch.float64, device=dev)
        cell = torch.as_tensor(cell, dtype=torch.float64, device=dev)
        n = len(z)
        base = build_graph(z, pos, cell, self.cutoffs)
        ei, es = base['edge_index'], base['edge_cell_shift'].round().to(torch.int64)
        self._sync(); tm['graph_s'] = time.perf_counter() - t0

        t = time.perf_counter()
        prior = self.prior(z, pos, cell, ei, es)
        self._sync(); tm['prior_s'] = time.perf_counter() - t
        tm['prior_detail'] = prior.timing

        t = time.perf_counter()
        index = KeyIndex(prior.keys, n)
        rev = index.reverse(prior.keys)
        if bool((rev < 0).any()):
            raise ValueError('prior support is not closed under (i, j, R) -> (j, i, -R)')
        H0 = symmetrise(prior.H, rev, True)
        S0 = symmetrise(prior.S, rev, False)
        node_keys = torch.zeros((n, 5), dtype=torch.int64, device=dev)
        node_keys[:, 0] = node_keys[:, 1] = torch.arange(n, device=dev)
        edge_keys = torch.cat([ei.T, es], 1)
        mkeys = torch.cat([node_keys, edge_keys])
        mrows = index.find(mkeys)
        if bool((mrows < 0).any()):
            raise ValueError('model graph block outside the prior support')
        hf, sf = self.maps.encode(prior.species, prior.spec, mkeys, mrows, H0, S0)
        self._sync(); tm['features_s'] = time.perf_counter() - t

        t = time.perf_counter()
        SO2_STATS.reset()
        dn, de = self.heads.forward(base, hf[:n], hf[n:], sf[:n], sf[n:])
        self._sync(); tm['model_s'] = time.perf_counter() - t
        so2 = dict(SO2_STATS.snapshot(), backend=self.heads.so2)

        t = time.perf_counter()
        wh = S0.shape[-1]
        D = self.maps.decode(prior.species, prior.spec, mkeys, torch.cat([dn, de]), wh)
        mindex = KeyIndex(mkeys, n, index.span)
        mrev = mindex.reverse(mkeys)
        if bool((mrev < 0).any()):
            raise ValueError('model graph is not closed under (i, j, R) -> (j, i, -R)')
        D = symmetrise(D, mrev, True)
        Hb = H0.to(torch.complex64)
        H = Hb.clone()
        H[mrows] = H[mrows] + D
        rows = prior.norb[prior.keys[:, 0]]
        cols = prior.norb[prior.keys[:, 1]]
        residual = None
        if self.route == 'unitb':
            # uu block [:ni, :nj] of the contiguous spinor layout (up orbitals, then down)
            r = torch.arange(wh, device=dev)
            uu = (r[None, :, None] < rows[:, None, None]) & (r[None, None, :] < cols[:, None, None])
            res = (H.real[:, :wh, :wh] - Hb.real[:, :wh, :wh]) * uu
            residual = BlockTable(prior.keys, rows, cols, res)
        out = Result(H=BlockTable(prior.keys, 2 * rows, 2 * cols, H),
                     S=BlockTable(prior.keys, rows, cols, S0.to(torch.float32)),
                     residual=residual, nelec=prior.nelec, timing=tm, info=dict(prior.info, so2=so2))
        self._sync(); tm['assemble_s'] = time.perf_counter() - t
        tm['total_s'] = time.perf_counter() - t0
        return out

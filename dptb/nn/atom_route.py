"""Invariant, differentiable late atom routing without per-edge weight banks."""
import json
import logging

import torch

from dptb.data import _keys
from dptb.data.interfaces.blockwise_tensor import strict_reverse_edge_index
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

log = logging.getLogger(__name__)


def invariant_width(irreps):
    return sum(mul * (2 if ir.l == 0 and ir.p == 1 else 1) for mul, ir in irreps)


def node_invariants(features, irreps):
    """Signed 0e channels plus smooth per-copy norms of every irrep.

    No cross-parity contractions, no detach. The epsilon makes the derivative
    finite at zero; subtracting sqrt(eps) gives isolated zero features zero norms.
    """
    parts = []
    for (mul, ir), sl in zip(irreps, irreps.slices()):
        block = features[:, sl].reshape(features.shape[0], mul, ir.dim)
        if ir.l == 0 and ir.p == 1:
            parts.append(block[..., 0])
        parts.append((block.square().sum(-1) + 1e-8).sqrt() - 1e-4)
    return torch.cat(parts, -1)


def pool_prior(data, active_edges, descriptor, cutoff, n_nodes):
    """Pair reverse descriptors, then pool outgoing edges with smooth cutoff.

    Strict (i,j,R)<->(j,i,-R) metadata also validates graph separation. Missing
    partners and asymmetric active sets are errors. Zero-neighbor atoms get zero.
    A small denominator floor defines the zero-neighbor case continuously.
    """
    reverse = strict_reverse_edge_index(data, device=active_edges.device)
    local = torch.full_like(reverse, -1)
    local[active_edges] = torch.arange(active_edges.numel(), device=active_edges.device)
    rev = local[reverse[active_edges]]
    if (rev < 0).any():
        raise ValueError("atom routing requires reverse-paired active edges")
    symmetric = (descriptor + descriptor[rev]) * 0.5
    weight = (cutoff[active_edges] + cutoff[reverse[active_edges]]) * 0.5
    src = data[_keys.EDGE_INDEX_KEY][0, active_edges]
    sums = descriptor.new_zeros(n_nodes, descriptor.shape[1]).index_add(0, src, symmetric * weight[:, None])
    counts = descriptor.new_zeros(n_nodes).index_add(0, src, weight)
    return sums / counts.clamp_min(1e-8)[:, None]


def edge_coefficients(alpha, edge_index):
    coeff = (alpha[edge_index[0]] + alpha[edge_index[1]]) * 0.5
    indices = torch.arange(alpha.shape[1], device=alpha.device).expand(coeff.shape[0], -1)
    return MOLEGlobals(coefficients=coeff, topk_indices=indices, topk_values=coeff,
                       activation_space=True, coefficients_sum_to_one=True)


@torch.no_grad()
def record_atom_routes(alpha, data, layer, step, training, elements):
    """Per local batch: soft mass and Kish ESS, including graph-mean-weight ESS.

    Detached lists can be logged/serialized without retaining an autograd graph.
    Every routed forward is logged; opt_step comes from the optimizer registry
    (plain optimizers can leave it zero). layer and mode disambiguate eval/retries.
    """
    a = alpha.detach().float()
    batch = data[_keys.BATCH_KEY].reshape(-1).to(a.device)
    elements = elements.reshape(-1).to(a.device)
    _, inverse, counts = torch.unique(batch, sorted=True, return_inverse=True, return_counts=True)
    graph_mean = a.new_zeros(counts.numel(), a.shape[1]).index_add(0, inverse, a)
    graph_mean = graph_mean / counts.clamp_min(1)[:, None]

    def ess(x):
        return (x.sum(0).square() / x.square().sum(0).clamp_min(1e-30)).cpu().tolist()

    stats = dict(opt_step=int(step), layer=int(layer), training=bool(training),
                 n_atoms=a.shape[0], n_structures=counts.numel(),
                 alpha_mean=(a.mean(0) if len(a) else a.new_zeros(a.shape[1])).cpu().tolist(),
                 alpha_std=(a.std(0, unbiased=False) if len(a) else a.new_zeros(a.shape[1])).cpu().tolist(),
                 element_mean={str(int(z)): a[elements == z].mean(0).cpu().tolist()
                               for z in torch.unique(elements)},
                 atom_mass=a.sum(0).cpu().tolist(), structure_mass=graph_mean.sum(0).cpu().tolist(),
                 effective_atoms=ess(a), effective_structures=ess(graph_mean))
    log.info("atom_route %s", json.dumps(stats, sort_keys=True))
    return stats

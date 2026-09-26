"""Structure-conditioned MoLE. No labels, sample identities, or fitted shifts.

The reference route uses the existing all-soft activation-space implementation.
The merged route combines only r x r cores, then applies Q, D_s, P to grouped
activations. Neither route constructs an edge-sized full weight tensor.
"""
import copy
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

from dptb.data import AtomicDataDict, _keys
from dptb.data.interfaces.blockwise_tensor import strict_reverse_edge_index

log = logging.getLogger(__name__)


def options(value):
    cfg = dict(value or {})
    allowed = {"enabled", "route_scope", "prior_stats", "execution", "hidden", "rbf_rmax",
               "stats_path", "init_from"}
    if cfg.keys() - allowed:
        raise ValueError(f"Unknown structure_mole options: {sorted(cfg.keys() - allowed)}")
    result = dict(enabled=False, route_scope="structure", prior_stats=True,
                  execution="merged_core", hidden=64, rbf_rmax=10., stats_path="", init_from="")
    result.update(cfg)
    if type(result["enabled"]) is not bool or type(result["prior_stats"]) is not bool:
        raise ValueError("enabled/prior_stats must be boolean")
    if result["route_scope"] not in ("structure", "constant"):
        raise ValueError("route_scope must be structure or constant")
    if result["execution"] not in ("reference", "merged_core"):
        raise ValueError("execution must be reference or merged_core")
    if not isinstance(result["hidden"], int) or result["hidden"] < 1:
        raise ValueError("hidden must be positive")
    if not math.isfinite(result["rbf_rmax"]) or result["rbf_rmax"] <= 0:
        raise ValueError("rbf_rmax must be finite and positive")
    return result


def validate_embedding(cfg, kw):
    k = 1 if cfg["route_scope"] == "constant" else 4
    required = dict(num_experts=k, top_k=k, num_shared_experts=1,
                    mole_expert_parameterization="shared_core",
                    so2_expert_mixing_mode="pre_activation")
    for key, expected in required.items():
        if kw.get(key, expected) != expected:
            raise ValueError(f"structure_mole requires {key}={expected!r}")
        kw[key] = expected
    # These older routing contracts are intentionally disjoint.
    if kw.get("edge_router_prior_activate", False) or kw.get("edge_router_route_drop_p", 0) != 0:
        raise ValueError("structure_mole requires edge_router_prior_activate=false and route_drop_p=0")
    if kw.get("edge_router_bias_speed", 0) != 0 or kw.get("edge_router_top1_mode", "legacy") != "legacy":
        raise ValueError("structure_mole requires bias_speed=0 and no Switch")
    kw["edge_router_bias_speed"] = 0.
    if kw.get("edge_router_prior_stats", "") or kw.get("edge_router_input", "onehot_prior") != "onehot_prior":
        raise ValueError("structure_mole uses its own persistent training statistics")
    if cfg["route_scope"] == "constant" and cfg["execution"] != "merged_core":
        raise ValueError("constant scope uses the explicit single-core branch")


class StructureStats(nn.Module):
    """Intensive descriptors and frozen, equal-structure training calibration."""

    def __init__(self, species, gram_dim, rbf_rmax):
        super().__init__()
        self.species = species
        self.bonds = species * (species + 1) // 2
        self.width = species + self.bonds + 16 + 64
        self.rbf_rmax = rbf_rmax
        generator = torch.Generator(device="cpu").manual_seed(0)
        self.register_buffer("projection", torch.randn(gram_dim, 32, generator=generator, dtype=torch.float32) / math.sqrt(32))
        pairs = torch.empty(species, species, dtype=torch.long)
        cursor = 0
        for i in range(species):
            for j in range(i, species):
                pairs[i, j] = pairs[j, i] = cursor
                cursor += 1
        self.register_buffer("pair_index", pairs)
        self.register_buffer("mean", torch.zeros(self.width))
        self.register_buffer("scale", torch.ones(self.width))
        self.register_buffer("training_count", torch.zeros((), dtype=torch.long))

    def raw(self, data, gram, cutoff):
        types = data[_keys.ATOM_TYPE_KEY].flatten().long()
        batch = data.get(_keys.BATCH_KEY, torch.zeros_like(types)).flatten().long()
        # IDs are used exclusively to segment the batch, never as features.
        _, batch = torch.unique(batch, sorted=True, return_inverse=True)
        n = int(batch.max()) + 1 if batch.numel() else 0
        if n == 0:
            raise ValueError("structure statistics require at least one atom")
        edges = data[_keys.EDGE_INDEX_KEY]
        reverse = strict_reverse_edge_index(data, device=types.device)
        edge_batch = batch[edges[0]]
        # Average descriptors before taking moments, then count each pair once.
        projected = gram @ self.projection
        projected = .5 * (projected + projected[reverse])
        weights = .5 * (cutoff + cutoff[reverse])
        keep = torch.arange(reverse.numel(), device=reverse.device) <= reverse
        pair_ids = self.pair_index[types[edges[0]], types[edges[1]]]
        length = data[_keys.EDGE_VECTORS_KEY].norm(dim=-1)
        centers = torch.linspace(0., self.rbf_rmax, 16, device=gram.device, dtype=gram.dtype)
        rbf = torch.exp(-.5 * ((length[:, None] - centers) / (self.rbf_rmax / 15)) ** 2)
        rbf = .5 * (rbf + rbf[reverse])
        edge_batch, weights = edge_batch[keep], weights[keep].clamp_min(0)
        projected, rbf, pair_ids = projected[keep], rbf[keep], pair_ids[keep]
        # Use scatter counts, not an E x number-of-bond-types one-hot matrix.
        elem = gram.new_zeros(n * self.species).index_add(
            0, batch * self.species + types, gram.new_ones(types.numel())).reshape(n, self.species)
        elem = elem / elem.sum(-1, keepdim=True).clamp_min(1)
        denom = gram.new_zeros(n).index_add(0, edge_batch, weights).clamp_min(1e-20)
        bond = gram.new_zeros(n * self.bonds).index_add(
            0, edge_batch * self.bonds + pair_ids, weights).reshape(n, self.bonds) / denom[:, None]

        def pool(x):
            return x.new_zeros(n, x.shape[1]).index_add(0, edge_batch, weights[:, None] * x) / denom[:, None]

        mean = pool(projected)
        # Centered second moment avoids cancellation for identical copies.
        var = pool((projected - mean[edge_batch]).square())
        # Exact zero with a finite derivative at zero variance.
        std = torch.where(var > 0, var.clamp_min(1e-24).sqrt(), torch.zeros_like(var))
        return torch.cat((elem, bond, pool(rbf), mean, std), -1), batch

    def forward(self, raw, prior_stats=True):
        if self.training_count.item() == 0:
            raise RuntimeError("structure_mole statistics are uncalibrated; fit on the training split first")
        z = (raw - self.mean) / self.scale
        if not prior_stats:
            z = torch.cat((z[:, :-64], torch.zeros_like(z[:, -64:])), -1)
        return z

    @torch.no_grad()
    def fit(self, rows, *, split):
        if split != "train":
            raise ValueError("structure_mole calibration accepts only split='train'")
        if self.training_count.item():
            raise ValueError("statistics already frozen; construct a fresh model to recalibrate")
        count = 0
        mean = torch.zeros_like(self.mean, dtype=torch.float64)
        m2 = torch.zeros_like(mean)
        for row in rows:
            row = row.detach().to(device=mean.device, dtype=mean.dtype)
            if row.ndim != 2 or row.shape[1] != self.width or not torch.isfinite(row).all():
                raise ValueError("invalid structure statistics")
            size = row.shape[0]
            if not size:
                continue
            delta = row.mean(0) - mean
            m2 += (row - row.mean(0)).square().sum(0) + delta.square() * (count * size / (count + size))
            mean += delta * (size / (count + size))
            count += size
        if count < 2:
            raise ValueError("calibration needs at least two training structures")
        std = (m2 / count).sqrt()
        self.mean.copy_(mean)
        # Constant channels stay on unit scale rather than amplifying roundoff.
        self.scale.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))
        self.training_count.fill_(count)


def raw_statistics(embedding, data):
    prior_key = getattr(getattr(embedding, "init_layer", None), "h0_edge_key", None)
    if prior_key not in ("edge_h0", "edge_p2"):
        raise ValueError("structure_mole accepts only explicit edge_h0/edge_p2 prior fields")
    # Labels and arbitrary sample metadata do not even enter geometry/mapping.
    fields = ("pos", "edge_index", "edge_cell_shift", "cell", "pbc", "atom_types",
              "atomic_numbers", "edge_type", "batch", "ptr", "_h0_coupled_rme", prior_key)
    data = {key: data[key] for key in fields if key in data}
    if "_h0_coupled_rme" in data and not torch.is_tensor(data["_h0_coupled_rme"]):
        data["_h0_coupled_rme"] = torch.as_tensor(data["_h0_coupled_rme"], device=data["pos"].device)
    data = embedding.idp(data)
    # Recompute from current geometry; do not trust cached edge vectors.
    data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
    types = data[_keys.EDGE_TYPE_KEY].flatten()
    # Reuse the edge router's Gram construction, in its required coupled-RME
    # representation. Legacy edge routing is intentionally unchanged: its
    # raw-source helper only sorts, even when the H0 input is AO-product.
    from dptb.nn.embedding.lem_moe_v3_h0_helpers import _h0_is_coupled_rme
    source = data[prior_key].to(dtype=embedding.dtype)
    source = source * embedding.idp.mask_to_erme.to(source.device)[types].to(source.dtype)
    init = embedding.init_layer
    coupled = _h0_is_coupled_rme(data)
    if not coupled and not getattr(init, "h0_ao_cg", False):
        raise ValueError("structure_mole needs coupled RME or h0_ao_cg=true for AO-product priors")
    source = init._ao_product_to_sorted_irreps(source, coupled=coupled)
    gram = embedding._gram_descriptor(source)
    base = getattr(embedding.init_layer, "base_init", embedding.init_layer)
    cutoff = base.cutoff_coefficients(data[_keys.EDGE_LENGTH_KEY], types)
    return embedding.structure_stats.raw(data, gram, cutoff)


@torch.no_grad()
def coefficient_metrics(alpha):
    a = alpha.detach().double()
    centered = a - a.mean(0)
    eig = torch.linalg.eigvalsh(centered.T @ centered / max(a.shape[0], 1)).clamp_min(0)
    mass = eig.sum()
    p = eig / mass.clamp_min(1e-30)
    rank = torch.where(mass > 1e-24, torch.exp(-(p * p.clamp_min(1e-30).log()).sum()), mass * 0)
    return dict(alpha_mean=a.mean(0), alpha_std=a.std(0, unbiased=False), covariance_effective_rank=rank,
                n_eff=a.sum(0).square() / a.square().sum(0).clamp_min(1e-30))


def make_route(embedding, data, active_edges):
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals
    cfg = embedding.structure_mole_options
    if cfg["route_scope"] == "constant":
        types = data[_keys.ATOM_TYPE_KEY].flatten()
        batch = data.get(_keys.BATCH_KEY, torch.zeros_like(types)).flatten()
        n = torch.unique(batch).numel()
        alpha = torch.ones(n, 1, device=active_edges.device, dtype=embedding.dtype)
        route = MOLEGlobals()
        route.structure_execution = "constant"
    else:
        raw, batch = raw_statistics(embedding, data)
        z = embedding.structure_stats(raw, cfg["prior_stats"])
        alpha = embedding.router(z).softmax(-1)
        graph = batch[data[_keys.EDGE_INDEX_KEY][0, active_edges]]
        if cfg["execution"] == "reference":
            coeff = alpha[graph]
            route = MOLEGlobals(coefficients=coeff, activation_space=True, coefficients_sum_to_one=True,
                                topk_indices=torch.arange(4, device=alpha.device).expand(coeff.shape[0], -1),
                                topk_values=coeff)
        else:
            route = MOLEGlobals(coefficients=alpha, graph_index=graph, coefficients_sum_to_one=True)
            route.structure_execution = "merged_core"
            # One grouping per forward, reused by every block in every layer.
            route.structure_rows = tuple(torch.where(graph == s)[0] for s in range(alpha.shape[0]))
            route.structure_inverse = torch.cat(route.structure_rows).argsort()
        embedding.last_structure_z = z.detach()
    embedding.last_structure_alpha = alpha.detach()
    embedding.last_structure_metrics = coefficient_metrics(alpha)
    for key, value in embedding.last_structure_metrics.items():
        data["structure_mole_" + key] = value
    if embedding.training:
        log.info("structure_mole %s", {k: v.tolist() for k, v in embedding.last_structure_metrics.items()})
    return route, alpha.max(-1).values.mean(), alpha.new_zeros(()), alpha.new_tensor(alpha.shape[0])


def merged_linear(layer, x, route):
    if layer.mole_expert_parameterization != "shared_core" or layer.num_shared_experts != 1:
        raise ValueError("structure core execution requires shared_core and one shared matrix")
    shared_bias = None if layer.bias_shared is None else layer.bias_shared[0]
    shared = F.linear(x, layer.weight_shared[0], shared_bias)

    def apply(rows, core, bias):
        # This also preserves the [edge, complex_pair, channel] m>0 layout.
        y = F.linear(F.linear(rows, layer.basis_right.T), core)
        return F.linear(y, layer.basis_left, bias)

    if route.structure_execution == "constant":
        if layer.num_experts != 1:
            raise ValueError("constant execution requires exactly one core")
        bias = None if layer.bias_experts is None else layer.bias_experts[0]
        return shared + apply(x, layer.core_experts[0], bias)
    cores = torch.einsum("sk,kij->sij", route.coefficients, layer.core_experts)
    biases = None if layer.bias_experts is None else route.coefficients @ layer.bias_experts
    # Shared activation computed once, residuals partitioned by structure.
    parts = []
    for s, rows in enumerate(route.structure_rows):
        if rows.numel():
            parts.append(apply(x.index_select(0, rows), cores[s], None if biases is None else biases[s]))
    # Empty edge sets must keep the routed parameters in the backward graph.
    if x.shape[0] == 0:
        residual = apply(x, cores.sum(0), None if biases is None else biases.sum(0))
    else:
        residual = torch.cat(parts, 0).index_select(0, route.structure_inverse)
    return shared + residual


@torch.no_grad()
def svd_split_state(model, source):
    """Strict conversion of legal SO2 parameter blocks; no AO-matrix SVD.

    Dense in this LEM family is the legacy one-expert, zero-shared layout.
    For m>0, the [real; imaginary] output stack is factored intact and retains
    the existing complex_pair_output assembly, with no bias in those blocks.
    """
    from dptb.nn.tensor_product_moe_v3 import MOLELinear
    target = copy.deepcopy(model.state_dict())
    consumed, generated = set(), set()
    for name, layer in model.named_modules():
        if not isinstance(layer, MOLELinear):
            continue
        if layer.mole_expert_parameterization != "shared_core" or layer.num_shared_experts != 1:
            raise ValueError(f"SVD target block is not shared_core: {name}")
        prefix = name + "." if name else ""
        key = prefix + "weight_experts"
        weight = source[key]
        if weight.shape != (1, layer.out_features, layer.in_features):
            raise ValueError(f"source is not a same-shape dense block: {key}")
        consumed.add(key)
        w = weight[0].to(device=layer.weight_shared.device)
        u, d, vh = torch.linalg.svd(w.double(), full_matrices=False)
        r = layer.mole_expert_rank
        p, q, core = u[:, :r].to(w), vh[:r].T.to(w), torch.diag(d[:r]).to(w)
        values = dict(basis_left=p, basis_right=q,
                      core_experts=(.5 * core).expand(layer.num_experts, -1, -1).clone(),
                      weight_shared=(w - .5 * (p @ core @ q.T)).unsqueeze(0))
        if layer.bias_shared is not None:
            bkey = prefix + "bias_experts"
            if source[bkey].shape != layer.bias_shared.shape:
                raise ValueError(f"dense bias shape mismatch: {bkey}")
            values.update(bias_shared=source[bkey].clone(), bias_experts=torch.zeros_like(layer.bias_experts))
            consumed.add(bkey)
        for suffix, value in values.items():
            target[prefix + suffix] = value
            generated.add(prefix + suffix)
    def new_route(key):
        return key.startswith(("router.", "structure_stats.")) or ".router." in key or ".structure_stats." in key
    for key in target:
        if key in generated or new_route(key):
            continue
        if key not in source or source[key].shape != target[key].shape:
            raise ValueError(f"dense backbone key/shape mismatch: {key}")
        target[key] = source[key].clone()
        consumed.add(key)
    extras = {key for key in source if key not in consumed and not new_route(key)}
    if extras:
        raise ValueError(f"unexpected dense state: {sorted(extras)}")
    model.load_state_dict(target, strict=True)


def initialize_fresh(model, cfg):
    """Fresh build only; checkpoint reload does not consult either source file."""
    if cfg.get("init_from"):
        ckpt = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
        source_cfg = ckpt["config"]["model_options"]
        source_emb = source_cfg["embedding"]
        if (source_emb.get("num_experts") != 1 or source_emb.get("num_shared_experts", 0) != 0
                or (source_cfg.get("shift_head") or {}).get("mode", "off") != "off"
                or (source_emb.get("structure_mole") or {}).get("enabled", False)):
            raise ValueError("structure_mole.init_from requires a dense checkpoint without fitted shift heads")
        target_emb = model.model_options["embedding"]
        route_keys = {"structure_mole", "num_experts", "num_shared_experts", "top_k",
                      "mole_expert_parameterization", "mole_expert_rank", "so2_fusion_mode",
                      "mole_linear_mode", "mole_linear_m0_mode", "so2_m_linear_mode",
                      "mole_full_expert_fast_path", "so2_wigner_apply_mode"}
        for key in source_emb.keys() & target_emb.keys():
            if key not in route_keys and not key.startswith("edge_router_") and source_emb[key] != target_emb[key]:
                raise ValueError(f"dense conversion changes backbone semantics: {key}")
        if source_cfg.get("prediction") != model.model_options.get("prediction"):
            raise ValueError("dense conversion requires the same prediction head")
        source_train = ckpt["config"].get("train_options", {})
        source_ranges = source_train.get("distance_ranges") or None
        target_ranges = getattr(model, "distance_ranges", None)
        if source_ranges is not None:
            source_ranges = [list(r) for r in source_ranges]
        if target_ranges is not None:
            target_ranges = [list(r) for r in target_ranges]
        if source_ranges != target_ranges:
            raise ValueError("dense conversion requires the same distance wrapper")
        svd_split_state(model, ckpt["model_state_dict"])
    if cfg.get("stats_path"):
        bundle = torch.load(cfg["stats_path"], map_location="cpu", weights_only=False)
        if bundle.get("split") != "train":
            raise ValueError("statistics must come from the training split")
        if bundle.get("descriptor_contract") != descriptor_contract(model):
            raise ValueError("statistics basis/cutoff/prior contract does not match model")
        modules = {name: m for name, m in model.named_modules() if isinstance(m, StructureStats)}
        if modules.keys() != bundle["states"].keys():
            raise ValueError("statistics topology does not match model")
        for name, mod in modules.items():
            saved = bundle["states"][name]
            if not torch.equal(mod.projection.cpu(), saved["projection"].cpu()):
                raise ValueError("statistics projection mismatch")
            mod.load_state_dict(saved, strict=True)
            if (mod.training_count.item() < 2 or not torch.isfinite(mod.mean).all()
                    or not torch.isfinite(mod.scale).all() or (mod.scale <= 0).any()):
                raise ValueError("invalid training statistics")


def descriptor_contract(model):
    """Inputs that define the descriptor, independent of router/core parameters."""
    emb = model.model_options["embedding"]
    return dict(version=2, basis=model.idp.basis,
                r_max=emb.get("r_max", 5.), cutoff_type=emb.get("cutoff_type", "polynomial"),
                PolynomialCutoff_p=emb.get("PolynomialCutoff_p", 6),
                r_start_cos_ratio=emb.get("r_start_cos_ratio", .8),
                h0_edge_key=emb.get("h0_edge_key", "edge_h0"),
                h0_ao_cg=emb.get("h0_ao_cg", True),
                rbf_rmax=options(emb.get("structure_mole"))["rbf_rmax"], projection_seed=0)


@torch.no_grad()
def fit_structure_stats(model, batches, *, split="train"):
    """Read each training structure once, with equal weight, and freeze scales.

    Batches can be graph dictionaries or AtomicData/PyG batches. No model
    forward, trainable embedding or target tensor enters the calibration.
    """
    if split != "train":
        raise ValueError("calibration accepts only the training split")
    embeddings = [(name, m) for name, m in model.named_modules()
                  if getattr(m, "structure_mole_enabled", False) and m.structure_stats is not None]
    if not embeddings:
        raise ValueError("no structure-scoped router to calibrate")
    # Distance heads share the same input descriptor. Check it once and copy
    # the final immutable statistics, rather than re-iterating the dataset.
    first = embeddings[0][1]
    def rows():
        for data in batches:
            if hasattr(data, "to_dict"):
                data = data.to_dict()
            data = {k: v.to(device=first.device) if torch.is_tensor(v) else v for k, v in data.items()}
            yield raw_statistics(first, data)[0]
    first.structure_stats.fit(rows(), split=split)
    for _, emb in embeddings[1:]:
        emb.structure_stats.load_state_dict(first.structure_stats.state_dict(), strict=True)
    return dict(split="train", descriptor_contract=descriptor_contract(model),
                states={name + ".structure_stats": {k: v.cpu().clone() for k, v in emb.structure_stats.state_dict().items()}
                        for name, emb in embeddings})

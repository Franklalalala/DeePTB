"""Distance-expert composition, routing and output stitching.

This layer operates on constructed modules; model selection and checkpoint
loading belong to :mod:`dptb.nn.build`.
"""

import copy
import logging
import random
import warnings

import numpy as np
import torch
from torch import nn

from dptb.data import AtomicDataDict
from dptb.data.AtomicDataDict import with_edge_vectors
from dptb.nn.output_spec import default_output_spec, ModelOutputSpecError
from dptb.nnops.distance_expert_mask import edge_mask_for_distance_expert, node_mask_for_distance_expert

# Keep the public logging category used by existing training filters.
log = logging.getLogger("dptb.nn.build")


class DeterministicExpertSeed:
    """
    Temporarily seed one expert without changing subsequent random draws.
    Restore Python, NumPy and Torch random states on exit.
    """

    def __init__(self, seed_val: int):
        self.seed_val = seed_val

    def __enter__(self):
        self.py_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.torch_cuda_state = torch.cuda.get_rng_state_all()

        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_val)
            torch.cuda.manual_seed_all(self.seed_val)

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.torch_cuda_state)


# ======================================================================
# Distance ensemble
# ======================================================================
class DistanceEnsembleWrapper(nn.Module):
    """Distance-range multi-expert ensemble with spec-driven output stitching.

    Routing (ownership) policy
    --------------------------
    ``_build_expert_masks`` implements the DEFAULT, pre-existing routing
    semantics: each expert owns the edges whose length falls inside its
    ``(d_min, d_max)`` range, while ALL nodes are assigned to the expert(s)
    whose ``d_min == 0`` (in practice expert 0); every other expert receives an
    all-False node mask.  The :class:`~dptb.nn.output_spec.ModelOutputSpec`
    contract fixed the *stitch capability* -- node-aligned outputs of experts
    ``1..N`` are merged under their node masks instead of being silently
    dropped -- but it did NOT change this ownership policy: under default
    routing experts ``1..N`` still own no nodes, so their node outputs are
    (correctly) never merged.  Custom node-ownership schemes should override
    ``_build_expert_masks``; the stitch path honors whatever masks it returns.
    """

    def __init__(self, experts, distance_ranges, strict_output_spec=False, clip_last_expert_range=False):
        super().__init__()
        assert len(experts) == len(distance_ranges), \
            f"len(experts) != len(distance_ranges): {len(experts)} vs {len(distance_ranges)}"

        self.distance_ranges = distance_ranges
        self.num_experts = len(distance_ranges)
        self.clip_last_expert_range = bool(clip_last_expert_range)
        self.experts = nn.ModuleList(experts)
        if len(experts) > 1 and any(
            getattr(getattr(expert, "shift_head", None), "options", {}).get("response") is not None
            for expert in experts
        ):
            raise ValueError("response release supports one distance expert per model; multi-expert auxiliary stitching needs an explicit ownership contract")
        for i, expert in enumerate(experts):
            head = getattr(expert, "shift_head", None)
            if head is not None:
                head.distance_policy = (*distance_ranges[i], i == len(experts) - 1,
                                        self.clip_last_expert_range)

        base_model = self.experts[0]
        self.name = getattr(base_model, "name", "distance_ensemble")
        self.device = getattr(base_model, "device", torch.device("cpu"))
        self.dtype = getattr(base_model, "dtype", torch.float32)
        self.model_options = copy.deepcopy(getattr(base_model, "model_options", {}))
        # Output-stitching contract.  Default is PERMISSIVE (warn + skip unknown
        # keys, preserving legacy behaviour) but WITH the node-aligned merge path
        # active.  ``strict_output_spec=True`` opts into fail-closed validation.
        self._output_spec = default_output_spec(strict=bool(strict_output_spec))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            modules = object.__getattribute__(self, "_modules")
            experts = modules.get("experts", None) if modules is not None else None
            if experts is not None and len(experts) > 0:
                base_model = experts[0]
                if hasattr(base_model, name):
                    return getattr(base_model, name)
            raise e

    def _get_safe_num_nodes(self, batch):
        """Read the node count without indexing a nested tensor."""
        for key in ["ATOM_TYPE_KEY", "POSITIONS_KEY"]:
            actual_key = getattr(AtomicDataDict, key, key.lower().replace("_key", ""))
            if actual_key in batch:
                t = batch[actual_key]
                if torch.is_tensor(t) and not getattr(t, "is_nested", False) and t.ndim > 0:
                    return t.shape[0]

        # Fall back to the largest node index.
        edge_index_key = getattr(AtomicDataDict, "EDGE_INDEX_KEY", "edge_index")
        if edge_index_key in batch:
            t = batch[edge_index_key]
            if torch.is_tensor(t) and not getattr(t, "is_nested", False) and t.numel() > 0:
                return int(t.max().item()) + 1
        return 1

    def _build_expert_masks(self, batch, expert_idx):
        dist = batch["edge_lengths"]
        d_min, d_max = self.distance_ranges[expert_idx]
        edge_mask = edge_mask_for_distance_expert(
            dist, d_min, d_max,
            is_last_expert=(expert_idx == self.num_experts - 1),
            clip_last_expert_range=self.clip_last_expert_range,
        )
        num_nodes = self._get_safe_num_nodes(batch)
        node_mask = node_mask_for_distance_expert(
            num_nodes, d_min, device=dist.device,
        )
        return edge_mask, node_mask

    @staticmethod
    def _apply_merge(merge, dst, src, mask):
        """Dispatch a single merge op, mutating ``dst`` in place over ``mask``."""
        if merge == "keep_first":
            return
        if merge == "masked_replace":
            dst[mask] = src[mask]
        elif merge == "sum":
            dst[mask] = dst[mask] + src[mask]
        elif merge == "mean":
            dst[mask] = 0.5 * (dst[mask] + src[mask])
        else:
            raise ModelOutputSpecError(f"Unknown merge op '{merge}'.")

    def _stitch_outputs(self, res, res_i, edge_mask, node_mask):
        """Merge one expert's outputs (``res_i``) into the running result ``res``.

        Iterates the *spec* fields (not the raw ``res_i`` keys) so alignment is
        looked up, not guessed: edge-aligned fields use ``edge_mask``,
        node-aligned fields use ``node_mask``, graph/scalar fields keep expert
        0's value.  This is the core fix -- node-aligned outputs of experts
        ``1..N`` were previously discarded because only an edge path existed.

        In strict mode, undeclared keys and shape-alignment mismatches raise
        :class:`ModelOutputSpecError`; in permissive mode they warn and skip
        (expert-0 value preserved).

        The ``num_nodes == num_edges`` ambiguity only matters where alignment
        must be GUESSED from a tensor's leading dimension, i.e. for keys the
        spec does not declare.  Declared fields carry a declared alignment and
        need no shape disambiguation, so they are stitched normally even when
        the two counts coincide.
        """
        spec = self._output_spec
        strict = bool(getattr(spec, "strict", False))

        n_edge = int(edge_mask.shape[0])
        n_node = int(node_mask.shape[0])
        # Shape-based node/edge inference is impossible when the two counts
        # coincide.  This is only consulted for UNDECLARED keys below.
        ambiguous = n_edge == n_node

        # Pass 1: handle output keys the expert produced that the spec does not
        # declare. In strict mode this is an error (alignment cannot be
        # inferred). In permissive mode, fall back to the LEGACY shape-based
        # edge stitch for backward compatibility: any tensor whose leading dim
        # equals num_edges is edge-aligned and merged via edge_mask, exactly as
        # the pre-spec implementation did -- otherwise an undeclared edge-aligned
        # output (e.g. a Full-H reconstruction field not in _keys.py) would
        # silently keep only expert 0's contribution. Non-edge-shaped undeclared
        # tensors keep expert-0's value, matching the old behaviour.
        for key, src in res_i.items():
            if key in spec.fields:
                continue
            if not torch.is_tensor(src):
                continue
            if getattr(src, "is_nested", False) or getattr(src, "is_sparse", False):
                continue
            if strict:
                msg = (
                    f"DistanceEnsembleWrapper: output key '{key}' is not declared in "
                    f"ModelOutputSpec; cannot infer its node/edge alignment."
                )
                if ambiguous:
                    msg += (
                        f" Ambiguous shapes make inference impossible: num_nodes "
                        f"({n_node}) == num_edges ({n_edge})."
                    )
                raise ModelOutputSpecError(msg)
            dst = res.get(key)
            if (
                torch.is_tensor(dst)
                and src.ndim > 0
                and int(src.shape[0]) == n_edge
                and src.shape == dst.shape
                and not getattr(dst, "is_nested", False)
                and not getattr(dst, "is_sparse", False)
            ):
                # Legacy edge-aligned stitch for an undeclared key.
                if ambiguous:
                    log.warning(
                        f"DistanceEnsembleWrapper: undeclared output key '{key}' "
                        f"stitched as edge-aligned by the legacy shape guess, but "
                        f"num_nodes == num_edges ({n_node}) makes that guess "
                        f"ambiguous. Declare the key in ModelOutputSpec to fix "
                        f"its alignment.")
                dst[edge_mask] = src[edge_mask]
            else:
                log.warning(
                    f"DistanceEnsembleWrapper: undeclared output key '{key}' kept as "
                    f"expert-0 value (permissive; not edge-aligned by shape).")

        # Pass 2: merge declared, masked-aligned fields.
        masks = {"edge": edge_mask, "node": node_mask}
        for field, fspec in spec.fields.items():
            if field not in res_i:
                continue
            mask = masks.get(fspec.alignment)  # None for graph/scalar -> keep_first
            if mask is None:
                continue
            if field not in res:
                # Expert 0 produced no baseline tensor; a masked index-merge has
                # no destination to write into.  Preserve legacy skip behaviour.
                continue

            src = res_i[field]
            dst = res[field]
            if not torch.is_tensor(src) or not torch.is_tensor(dst):
                continue
            # Keep the existing NestedTensor / SparseTensor guards.
            if getattr(src, "is_nested", False) or getattr(dst, "is_nested", False):
                continue
            if getattr(src, "is_sparse", False) or getattr(dst, "is_sparse", False):
                continue

            # Alignment assertion: the leading dim must match the mask length
            # (and dst must match src so the index-assignment is well defined).
            if src.ndim == 0 or int(src.shape[0]) != int(mask.shape[0]) or src.shape != dst.shape:
                msg = (f"DistanceEnsembleWrapper: field '{field}' declared alignment "
                       f"'{fspec.alignment}' but src shape {tuple(src.shape)} / dst shape "
                       f"{tuple(dst.shape)} is incompatible with mask length {int(mask.shape[0])}.")
                if strict:
                    raise ModelOutputSpecError(msg)
                log.warning(msg + " Skipping (permissive mode).")
                continue

            try:
                self._apply_merge(fspec.merge, dst, src, mask)
            except ModelOutputSpecError:
                raise
            except Exception as e:
                # Terminal defense: never let an unexpected tensor crash the
                # permissive inference path.
                if strict:
                    raise
                log.warning(
                    f"DistanceEnsembleWrapper safely skipped stitching key '{field}' "
                    f"due to internal tensor error: {e}")
                continue

    def forward(self, batch):
        # Prepare edge geometry once for every expert's batch copy.
        if "edge_lengths" not in batch:
            batch = with_edge_vectors(batch, with_lengths=True)

        # Complete the graph bookkeeping needed for single-graph inference.
        batch_key = getattr(AtomicDataDict, "BATCH_KEY", "batch")
        if batch_key not in batch:
            num_nodes = self._get_safe_num_nodes(batch)
            device = batch.get("edge_lengths", torch.tensor([])).device

            # Every node belongs to graph zero.
            batch[batch_key] = torch.zeros(num_nodes, dtype=torch.long, device=device)
            # Graph boundaries for aggregation and scatter operations.
            ptr_key = getattr(AtomicDataDict, "PTR_KEY", "ptr")
            batch[ptr_key] = torch.tensor([0, num_nodes], dtype=torch.long, device=device)

        expert_idx = batch.get("expert_idx", None)

        # Single-expert training or evaluation.
        if expert_idx is not None:
            if torch.is_tensor(expert_idx):
                expert_idx_val = int(expert_idx.detach().item())
            else:
                expert_idx_val = int(expert_idx)

            clean_batch = {k: v for k, v in batch.items() if k != "expert_idx"}

            if "expert_edge_mask" not in clean_batch or "expert_node_mask" not in clean_batch:
                edge_mask, node_mask = self._build_expert_masks(clean_batch, expert_idx_val)
                clean_batch["expert_edge_mask"] = edge_mask
                clean_batch["expert_node_mask"] = node_mask

            return self.experts[expert_idx_val](clean_batch)

        # Full ensemble inference.
        base_batch = batch.copy()

        # Expert zero supplies the initial node features.
        batch_0 = base_batch.copy()
        edge_mask_0, node_mask_0 = self._build_expert_masks(base_batch, 0)
        batch_0["expert_edge_mask"] = edge_mask_0
        batch_0["expert_node_mask"] = node_mask_0

        res = self.experts[0](batch_0)

        # Evaluate and stitch the remaining experts.
        for i in range(1, self.num_experts):
            edge_mask_i, node_mask_i = self._build_expert_masks(base_batch, i)

            if not bool(edge_mask_i.any().item()):
                continue

            batch_i = base_batch.copy()
            batch_i["expert_edge_mask"] = edge_mask_i
            batch_i["expert_node_mask"] = node_mask_i

            res_i = self.experts[i](batch_i)
            # Pass BOTH masks so node-aligned outputs of experts 1..N are merged
            # (edge-only stitching silently dropped every node_* output before).
            self._stitch_outputs(res, res_i, edge_mask_i, node_mask_i)

        # Remove the masks added for expert evaluation.
        if "expert_edge_mask" in res:
            del res["expert_edge_mask"]
        if "expert_node_mask" in res:
            del res["expert_node_mask"]

        return res

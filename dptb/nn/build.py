from dptb.checkpoint_config import merge_checkpoint_common_options
from dptb.nn.deeptb import NNENV
from dptb.configuration import migrate_legacy_checkpoint_model_options
import logging
import torch
import torch.nn as nn
from dptb.data import AtomicDataDict
from dptb.data.AtomicDataDict import with_edge_vectors
from dptb.nn.output_spec import default_output_spec, ModelOutputSpecError
import copy
import random
import numpy as np

log = logging.getLogger(__name__)


# ======================================================================
# [独立扩展模块] Deterministic Seed Context Manager
# ======================================================================
class DeterministicExpertSeed:
    """
    上下文管理器：精准为当前的 Expert 设定固定的初始化种子（保证 Windows/Linux 完美对齐）。
    退出时自动恢复之前的全局随机状态，避免污染 DataLoader 等后续流程。
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
# [独立扩展模块] Distance Ensemble Wrapper
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

    def __init__(self, experts, distance_ranges, strict_output_spec=False):
        super().__init__()
        assert len(experts) == len(distance_ranges), \
            f"len(experts) != len(distance_ranges): {len(experts)} vs {len(distance_ranges)}"

        self.distance_ranges = distance_ranges
        self.num_experts = len(distance_ranges)
        self.experts = nn.ModuleList(experts)

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
        """安全地提取节点数，严格避开 NestedTensor 带来的底层报错"""
        for key in ["ATOM_TYPE_KEY", "POSITIONS_KEY"]:
            actual_key = getattr(AtomicDataDict, key, key.lower().replace("_key", ""))
            if actual_key in batch:
                t = batch[actual_key]
                if torch.is_tensor(t) and not getattr(t, "is_nested", False) and t.ndim > 0:
                    return t.shape[0]

        # 降级策略：根据 edge_index 的最大值推断
        edge_index_key = getattr(AtomicDataDict, "EDGE_INDEX_KEY", "edge_index")
        if edge_index_key in batch:
            t = batch[edge_index_key]
            if torch.is_tensor(t) and not getattr(t, "is_nested", False) and t.numel() > 0:
                return int(t.max().item()) + 1
        return 1

    def _build_expert_masks(self, batch, expert_idx):
        dist = batch["edge_lengths"]

        d_min, d_max = self.distance_ranges[expert_idx]
        if expert_idx == self.num_experts - 1:
            edge_mask = (dist >= d_min)
        else:
            edge_mask = (dist >= d_min) & (dist < d_max)

        num_nodes = self._get_safe_num_nodes(batch)

        node_mask = torch.ones(num_nodes, dtype=torch.bool, device=dist.device)
        if d_min > 0:
            node_mask.fill_(False)

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
        # 【防御 1】入口统一预处理：保证后续所有 batch copy 都携带 edge_lengths 和 vec
        if "edge_lengths" not in batch:
            batch = with_edge_vectors(batch, with_lengths=True)

        # 【防御 2】自动为单图推理补齐 PyG 依赖的所有结构键
        batch_key = getattr(AtomicDataDict, "BATCH_KEY", "batch")
        if batch_key not in batch:
            num_nodes = self._get_safe_num_nodes(batch)
            device = batch.get("edge_lengths", torch.tensor([])).device

            # 补齐 batch 数组 (全是 0，代表图 0)
            batch[batch_key] = torch.zeros(num_nodes, dtype=torch.long, device=device)
            # 补齐 ptr 游标数组 (PyG 聚合和 Scatter 常用)
            ptr_key = getattr(AtomicDataDict, "PTR_KEY", "ptr")
            batch[ptr_key] = torch.tensor([0, num_nodes], dtype=torch.long, device=device)

        expert_idx = batch.get("expert_idx", None)

        # ==================== 单专家分支 (Trainer 调用 或 单模块 Eval) ====================
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

        # ==================== 多专家全景推理分支 (Inference Ensemble) ====================
        base_batch = batch.copy()

        # Expert 0: 获取骨架节点特征
        batch_0 = base_batch.copy()
        edge_mask_0, node_mask_0 = self._build_expert_masks(base_batch, 0)
        batch_0["expert_edge_mask"] = edge_mask_0
        batch_0["expert_node_mask"] = node_mask_0

        res = self.experts[0](batch_0)

        # Expert 1~N: 增量处理并 Stitch
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

        # 擦除注入的掩码，保证传出纯净的数据流
        if "expert_edge_mask" in res:
            del res["expert_edge_mask"]
        if "expert_node_mask" in res:
            del res["expert_node_mask"]

        return res


def _construct_single_model(model_options, common_options):
    return NNENV(**model_options, **common_options)


def _construct_single_model_from_reference(
    checkpoint, model_options, common_options
):
    return NNENV.from_reference(checkpoint, **model_options, **common_options)


def _replicate_prototype_to_ensemble(
    prototype_model, distance_ranges, model_options, common_options
):
    proto_state = prototype_model.state_dict()
    experts = [prototype_model]
    for i in range(1, len(distance_ranges)):
        with DeterministicExpertSeed(i + 1):
            m = _construct_single_model(model_options, common_options)
        m.load_state_dict(proto_state, strict=True)
        experts.append(m)
    return DistanceEnsembleWrapper(experts=experts, distance_ranges=distance_ranges)


def _count_experts_in_state_dict(state_dict: dict):
    if state_dict is None: return 0
    ids = {int(k.split(".")[1]) for k in state_dict.keys() if
           k.startswith("experts.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()}
    return len(ids)


def _is_multi_expert_state_dict(state_dict: dict):
    return _count_experts_in_state_dict(state_dict) > 0


def _has_legacy_swiglu_s2_state(state_dict: dict) -> bool:
    if not state_dict:
        return False
    return any(".activation.mul." in key for key in state_dict.keys())


def _maybe_enable_legacy_swiglu_s2_compat(model_options: dict, state_dict: dict):
    if not state_dict or not model_options:
        return model_options

    embedding = model_options.get("embedding", None)
    if not isinstance(embedding, dict):
        return model_options
    if embedding.get("method") not in {
        "lem_moe_v3",
        "lem_moe_v3_h0",
        "lem_pair",
        "lem_moe_v3_prior",
    }:
        return model_options
    if embedding.get("swiglu_s2_compat_mode", "modern") != "modern":
        return model_options
    if not _has_legacy_swiglu_s2_state(state_dict):
        return model_options

    patched = copy.deepcopy(model_options)
    patched.setdefault("embedding", {})
    patched["embedding"]["swiglu_s2_compat_mode"] = "legacy_uniform_only"
    log.warning(
        "Detected legacy flat SwiGLU-S2 checkpoint layout; forcing "
        "embedding.swiglu_s2_compat_mode='legacy_uniform_only' for compatibility."
    )
    return patched


def _build_ensemble_from_wrapper_state(
    wrapper_state_dict, distance_ranges, model_options, common_options
):
    ckpt_num_experts = _count_experts_in_state_dict(wrapper_state_dict)
    if ckpt_num_experts != len(distance_ranges):
        raise ValueError(f"Checkpoint has {ckpt_num_experts} experts, but requires {len(distance_ranges)}.")
    experts = []
    for i in range(len(distance_ranges)):
        with DeterministicExpertSeed(i + 1):
            m = _construct_single_model(model_options, common_options)
        experts.append(m)
    model = DistanceEnsembleWrapper(experts=experts, distance_ranges=distance_ranges)
    model.load_state_dict(wrapper_state_dict, strict=True)
    return model


JSON_MODEL_RETIRED_MESSAGE = (
    "json model files belonged to the retired SK route; pass a .pth checkpoint."
)


def _reject_retired_json_model(path):
    """The json branch never worked: it parsed the config and then handed the
    same path to torch.load, which raised UnpicklingError."""
    if isinstance(path, str) and path.split(".")[-1].lower() == "json":
        raise ValueError(f"{JSON_MODEL_RETIRED_MESSAGE} Got: {path}")


# ======================================================================
# [核心主程序] 原版 build_model
# ======================================================================
def build_model(
        checkpoint: str = None,
        model_options: dict = None,
        common_options: dict = None,
        train_options: dict = None,
        no_check: bool = False,
        device: str = None,
        explicit_common_options: dict = None,
        weights_inferred_common_options: dict = None,
):
    # Keys the *user* actually wrote. Anything else in ``common_options`` is a
    # schema default and must lose to the checkpoint's own architecture.
    explicit_common_options = copy.deepcopy(explicit_common_options or {})
    weights_inferred_common_options = copy.deepcopy(weights_inferred_common_options or {})
    model_options = copy.deepcopy(model_options or {})
    common_options = copy.deepcopy(common_options or {})
    train_options = copy.deepcopy(train_options or {})

    if checkpoint is not None:
        from_scratch = False
    else:
        from_scratch = True
        if not all((model_options, common_options)):
            logging.error(
                "You need to provide model_options and common_options when you are initializing a model from scratch.")
            raise ValueError(
                "You need to provide model_options and common_options when you are initializing a model from scratch.")

    ckpt_state_dict = None

    if not from_scratch:
        _reject_retired_json_model(checkpoint)
        f = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ckptconfig = f['config']
        ckpt_state_dict = f.get("model_state_dict", None)
        del f

        checkpoint_model_options = migrate_legacy_checkpoint_model_options(
            ckptconfig["model_options"]
        )
        if len(model_options) == 0 or model_options == ckptconfig["model_options"]:
            model_options = checkpoint_model_options

        common_options = merge_checkpoint_common_options(
            common_options,
            ckptconfig.get("common_options", {}),
            explicit_common_options,
            preserve_runtime_defaults=True,
            weights_inferred_overrides=weights_inferred_common_options,
        )

        if len(train_options) == 0:
            train_options = ckptconfig.get("train_options", {})

        del ckptconfig

    model_options = _maybe_enable_legacy_swiglu_s2_compat(model_options, ckpt_state_dict)

    retired = [key for key in ("nnsk", "dftbsk") if model_options.get(key)]
    if retired:
        raise ValueError(
            "0726-light removed the retired SK/DFTB model routes: "
            + ", ".join(retired)
        )
    if not all((model_options.get("embedding"), model_options.get("prediction"))):
        raise ValueError("model_options must define both embedding and prediction.")
    if model_options["prediction"].get("method") not in {"e3tb", "block_native"}:
        raise ValueError(
            "0726-light supports prediction.method='e3tb' or 'block_native'."
        )
    if device:
        common_options.update({"device": device})

    distance_ranges = train_options.get("distance_ranges", None)
    use_distance_ensemble = distance_ranges is not None

    if use_distance_ensemble:
        log.info(f"Wrapping model with DistanceEnsembleWrapper ({len(distance_ranges)} experts)")
        if from_scratch:
            with DeterministicExpertSeed(1):
                prototype_model = _construct_single_model(
                    model_options, common_options
                )
            model = _replicate_prototype_to_ensemble(
                prototype_model, distance_ranges, model_options, common_options
            )
        else:
            if ckpt_state_dict is None:
                with DeterministicExpertSeed(1):
                    prototype_model = _construct_single_model(
                        model_options, common_options
                    )
                model = _replicate_prototype_to_ensemble(
                    prototype_model, distance_ranges, model_options, common_options
                )
            elif _is_multi_expert_state_dict(ckpt_state_dict):
                model = _build_ensemble_from_wrapper_state(
                    ckpt_state_dict,
                    distance_ranges,
                    model_options,
                    common_options,
                )
            else:
                with DeterministicExpertSeed(1):
                    prototype_model = _construct_single_model_from_reference(
                        checkpoint, model_options, common_options
                    )
                model = _replicate_prototype_to_ensemble(
                    prototype_model, distance_ranges, model_options, common_options
                )

    else:
        with DeterministicExpertSeed(1):
            if from_scratch:
                model = NNENV(**model_options, **common_options)
            else:
                model = NNENV.from_reference(
                    checkpoint, **model_options, **common_options
                )

    if not no_check:
        for k, v in model.model_options.items():
            if k not in model_options:
                log.warning(f"The model options {k} is not defined in input model_options, set to {v}.")
            else:
                deep_dict_difference(k, v, model_options)

    model.to(model.device)

    return model


def deep_dict_difference(base_key, expected_value, model_options):
    target_dict = copy.deepcopy(model_options)
    if isinstance(expected_value, dict):
        for subk, subv in expected_value.items():
            if subk not in target_dict.get(base_key, {}):
                log.warning(
                    f"The model option {subk} in {base_key} is not defined in input model_options, set to {subv}.")
            else:
                target2 = copy.deepcopy(target_dict[base_key])
                deep_dict_difference(f"{subk}", subv, target2)
    else:
        if expected_value != target_dict[base_key]:
            log.warning(
                f"The model option {base_key} is set to {expected_value}, but in input it is {target_dict[base_key]}, make sure it it correct!")

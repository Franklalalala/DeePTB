from __future__ import annotations

"""Physical prior families for :class:`dptb.nnops.flow.HamiltonianCFM`.

Each ``PriorFamily`` owns exactly one physical initial-guess mechanism.  A
family parses only its own ``flow_options`` keys in :meth:`from_options` (same
keys and defaults the trainer accepted before this module existed) and builds an
*absolute* prior Hamiltonian in :meth:`absolute_prior_like`.  Low-level numeric
helpers that several families share -- the projection/mask/coerce utilities, the
cached basis-onsite table, the Huckel edge-energy assembly and the Haar-DM
candidate selection -- stay on the ``HamiltonianCFM`` instance and are handed to
the family through the small :class:`PriorContext` view, so the families never
reach back into private trainer state directly.
"""

from typing import Any, Dict, Optional, Tuple

import logging

import torch

from dptb.data import AtomicDataDict, _keys

log = logging.getLogger(__name__)


# Name aliases per family.  ``HamiltonianCFM`` imports these so the registry and
# the trainer's ``allowed_priors`` set share one source of truth.
BASIS_ONSITE_NAMES = frozenset({"basis_onsite", "fixed_onsite", "atomic_onsite"})
OVERLAP_HUCKEL_NAMES = frozenset(
    {"overlap_huckel", "huckel_overlap", "extended_huckel", "eht"}
)
HAAR_DM_NAMES = frozenset({"haar_dm", "haar_density", "haar_projector"})
EXTERNAL_NAMES = frozenset(
    {"external", "dftb", "dftb_xtb", "xtb", "physical", "sk", "nnsk"}
)
DFTBSK_NAMES = frozenset(
    {"dftbsk", "dftb_sk", "dftb_scf0", "dftb_on_the_fly", "skf", "skfile"}
)


class PriorContext:
    """Minimal, explicit view of the shared helpers a family may call.

    Holds the owning ``HamiltonianCFM`` so ``idp`` (which tests reassign) and the
    numeric helpers are read live rather than snapshotted at construction time.
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    @property
    def idp(self) -> Any:
        return self._owner.idp

    def prior_mask(self, data, like, label):
        return self._owner._prior_mask(data, like, label)

    def coerce_prior_source(self, source, like, *, key, label):
        return self._owner._coerce_prior_source(source, like, key=key, label=label)

    def basis_onsite_table(self, like):
        return self._owner._basis_onsite_table(like)

    def huckel_edge_energy_like(self, like, *, data):
        return self._owner._huckel_edge_energy_like(like, data=data)

    def huckel_pair_energy_table(self, like):
        return self._owner._huckel_pair_energy_table(like)

    def calibration_edge_scale(self, like):
        return self._owner._calibration_table(like, "edge_scale")

    def calibration_node_table(self, like):
        return self._owner._calibration_table(like, "node_table")

    def select_haar_candidate(self, source, like, *, key, label, candidate_idx=None):
        return self._owner._select_haar_candidate(
            source, like, key=key, label=label, candidate_idx=candidate_idx
        )

    def num_graphs(self, data) -> int:
        return self._owner._num_graphs(data)


class PriorFamily:
    """Base class for a physical prior mechanism."""

    NAMES: frozenset = frozenset()

    @classmethod
    def from_options(cls, options: Dict[str, Any], idp: Any) -> "PriorFamily":
        raise NotImplementedError

    def absolute_prior_like(
        self,
        like: torch.Tensor,
        *,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
        ctx: PriorContext,
    ) -> Optional[torch.Tensor]:
        raise NotImplementedError


class BasisOnsiteFamily(PriorFamily):
    NAMES = BASIS_ONSITE_NAMES

    def __init__(
        self,
        *,
        scale: float,
        missing_value: float,
        edge_value: float,
        mode: str = "table",
    ) -> None:
        self.basis_onsite_scale = scale
        self.basis_onsite_missing_value = missing_value
        self.basis_onsite_edge_value = edge_value
        self.basis_onsite_mode = mode

    @classmethod
    def from_options(cls, options: Dict[str, Any], idp: Any) -> "BasisOnsiteFamily":
        mode = str(options.get("basis_onsite_mode", "table")).lower().replace("-", "_")
        if mode not in {"table", "calibrated"}:
            raise ValueError(
                "flow_options.basis_onsite_mode must be 'table' (free-atom onsite DB) "
                "or 'calibrated' (per-type onsite rows from a prior_calibration artifact)."
            )
        return cls(
            scale=float(options.get("basis_onsite_scale", 1.0)),
            missing_value=float(options.get("basis_onsite_missing_value", 0.0)),
            edge_value=float(options.get("basis_onsite_edge_value", 0.0)),
            mode=mode,
        )

    def absolute_prior_like(self, like, *, data, label, ctx):
        if label == "edge":
            prior = torch.full_like(like, float(self.basis_onsite_edge_value))
            return prior * ctx.prior_mask(data, like, label).to(dtype=like.dtype)
        if label != "node":
            return torch.zeros_like(like)
        if data is None or AtomicDataDict.ATOM_TYPE_KEY not in data:
            return None
        if self.basis_onsite_mode == "calibrated":
            table = ctx.calibration_node_table(like)
            if table is None:
                raise ValueError(
                    "flow_options.basis_onsite_mode='calibrated' requires "
                    "flow_options.prior_calibration pointing to an artifact with a "
                    "`node_table` (tools/calibrate_huckel_scales.py)."
                )
        else:
            table = ctx.basis_onsite_table(like)
        if table is None:
            return None
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].to(
            device=like.device,
            dtype=torch.long,
        ).reshape(-1)
        prior = torch.zeros_like(like)
        take = min(int(atom_types.numel()), int(like.shape[0]))
        if take > 0 and table.shape[0] > 0:
            raw_types = atom_types[:take]
            valid = (raw_types >= 0) & (raw_types < table.shape[0])
            rows = torch.arange(take, device=like.device, dtype=torch.long)[valid]
            if rows.numel() > 0:
                prior[rows] = table.index_select(0, raw_types[valid])
        return prior * ctx.prior_mask(data, like, label).to(dtype=like.dtype)


class OverlapHuckelFamily(PriorFamily):
    NAMES = OVERLAP_HUCKEL_NAMES

    ENERGY_MODES = frozenset({"type_mean", "orbital_pair"})
    SCALE_MODES = frozenset({"none", "global", "pair_block"})

    def __init__(
        self,
        *,
        huckel_k: float,
        node_overlap_key: str,
        edge_overlap_key: str,
        strict_overlap: bool,
        strict_basis: bool,
        edge_energy_fallback: float,
        edge_length_decay: float,
        basis: BasisOnsiteFamily,
        energy_mode: str = "type_mean",
        scale_mode: str = "none",
        scale_global: float = 1.0,
        channel_scale: Optional[Any] = None,
    ) -> None:
        self.huckel_k = huckel_k
        self.huckel_node_overlap_key = node_overlap_key
        self.huckel_edge_overlap_key = edge_overlap_key
        self.huckel_strict_overlap = strict_overlap
        self.huckel_strict_basis = strict_basis
        self.huckel_edge_energy_fallback = edge_energy_fallback
        self.huckel_edge_length_decay = edge_length_decay
        self.huckel_energy_mode = energy_mode
        self.huckel_scale_mode = scale_mode
        self.huckel_scale_global = scale_global
        self.huckel_edge_channel_scale = channel_scale
        self._basis = basis

    @classmethod
    def from_options(cls, options: Dict[str, Any], idp: Any) -> "OverlapHuckelFamily":
        edge_value = float(options.get("basis_onsite_edge_value", 0.0))
        energy_mode = str(options.get("huckel_energy_mode", "type_mean")).lower().replace("-", "_")
        if energy_mode not in cls.ENERGY_MODES:
            raise ValueError(
                "flow_options.huckel_energy_mode must be 'type_mean' (legacy: one "
                "scalar per edge) or 'orbital_pair' (Wolfsberg-Helmholz "
                "0.5*(eps_mu+eps_nu) per orbpair slice)."
            )
        scale_mode = str(options.get("huckel_scale_mode", "none")).lower().replace("-", "_")
        if scale_mode not in cls.SCALE_MODES:
            raise ValueError(
                "flow_options.huckel_scale_mode must be 'none', 'global' "
                "(huckel_scale_global scalar), or 'pair_block' (per bond-type x "
                "orbpair-slice table from flow_options.prior_calibration)."
            )
        return cls(
            huckel_k=float(options.get("huckel_k", options.get("overlap_huckel_k", 1.75))),
            node_overlap_key=str(
                options.get("huckel_node_overlap_key", _keys.NODE_OVERLAP_KEY)
            ),
            edge_overlap_key=str(
                options.get("huckel_edge_overlap_key", _keys.EDGE_OVERLAP_KEY)
            ),
            strict_overlap=bool(options.get("huckel_strict_overlap", True)),
            strict_basis=bool(options.get("huckel_strict_basis", True)),
            edge_energy_fallback=float(
                options.get("huckel_edge_energy_fallback", edge_value)
            ),
            edge_length_decay=float(options.get("huckel_edge_length_decay", 0.0)),
            basis=BasisOnsiteFamily.from_options(options, idp),
            energy_mode=energy_mode,
            scale_mode=scale_mode,
            scale_global=float(options.get("huckel_scale_global", 1.0)),
            channel_scale=cls._parse_channel_scale(
                options.get(
                    "huckel_edge_channel_scale",
                    options.get("overlap_huckel_edge_channel_scale", None),
                )
            ),
        )

    @staticmethod
    def _parse_channel_scale(value: Optional[Any]) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, str):
            if not value.strip():
                return None
            parts = [part.strip() for part in value.split(",") if part.strip()]
            if not parts:
                raise ValueError("flow_options.huckel_edge_channel_scale must not be empty.")
            return [float(part) for part in parts]
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1).tolist()
            if not flat:
                raise ValueError("flow_options.huckel_edge_channel_scale must not be empty.")
            return flat
        return value

    def _channel_scale_like(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        values = self.huckel_edge_channel_scale
        if values is None:
            return None
        scale = torch.as_tensor(values, device=like.device, dtype=like.dtype).reshape(-1)
        if int(scale.numel()) == 0:
            raise ValueError("flow_options.huckel_edge_channel_scale must not be empty.")
        if int(scale.numel()) == 1:
            return scale.reshape(())
        if like.ndim < 1 or int(scale.numel()) != int(like.shape[-1]):
            raise ValueError(
                "flow_options.huckel_edge_channel_scale must be a scalar or have one "
                f"value per edge feature channel ({int(like.shape[-1])}), got "
                f"{int(scale.numel())}."
            )
        return scale.reshape((1,) * (like.ndim - 1) + (int(scale.numel()),))

    def _edge_types(self, data, like, *, purpose: str) -> Optional[torch.Tensor]:
        edge_types = None if data is None else data.get(AtomicDataDict.EDGE_TYPE_KEY, None)
        if edge_types is None:
            if self.huckel_strict_basis:
                raise KeyError(
                    f"flow_options.prior='overlap_huckel' with {purpose} requires "
                    "`edge_type` in the batch to index per-bond-type tables."
                )
            return None
        edge_types = edge_types.to(device=like.device, dtype=torch.long).reshape(-1)
        if int(edge_types.numel()) != int(like.shape[0]):
            if self.huckel_strict_basis:
                raise ValueError(
                    f"flow_options.prior='overlap_huckel' with {purpose}: edge_type has "
                    f"{int(edge_types.numel())} rows but the edge overlap tensor has "
                    f"{int(like.shape[0])} rows."
                )
            return None
        return edge_types

    def absolute_prior_like(self, like, *, data, label, ctx):
        if label == "node":
            prior = self._basis.absolute_prior_like(like, data=data, label=label, ctx=ctx)
            if prior is None and self.huckel_strict_basis:
                raise ValueError(
                    "flow_options.prior='overlap_huckel' requires an OrbitalMapper-like idp "
                    "with basis_to_full_basis/orbpair_maps for node onsite initialization."
                )
            return prior
        if label != "edge":
            return torch.zeros_like(like)
        if data is None:
            if self.huckel_strict_overlap:
                raise KeyError(
                    f"flow_options.prior='overlap_huckel' requires `{self.huckel_edge_overlap_key}` "
                    "in the batch."
                )
            return None
        overlap_source = data.get(self.huckel_edge_overlap_key, None)
        if overlap_source is None:
            if self.huckel_strict_overlap:
                raise KeyError(
                    f"flow_options.prior='overlap_huckel' requires `{self.huckel_edge_overlap_key}` "
                    "in the batch. Precompute ABACUS/get_s overlap or enable dataset get_overlap."
                )
            return None
        overlap = ctx.coerce_prior_source(
            overlap_source,
            like,
            key=self.huckel_edge_overlap_key,
            label=label,
        )
        energy = None
        if self.huckel_energy_mode == "orbital_pair":
            pair_table = ctx.huckel_pair_energy_table(like)
            edge_types = self._edge_types(data, like, purpose="huckel_energy_mode='orbital_pair'")
            if pair_table is None:
                if self.huckel_strict_basis:
                    raise ValueError(
                        "flow_options.huckel_energy_mode='orbital_pair' requires an "
                        "OrbitalMapper-like idp with bond_to_type/basis_to_full_basis/"
                        "orbpair_maps to build the per-orbital-pair energy table."
                    )
            elif edge_types is not None:
                energy = pair_table.index_select(0, edge_types)
        if energy is None:
            energy = ctx.huckel_edge_energy_like(like, data=data)
        prior = float(self.huckel_k) * overlap * energy
        if self.huckel_scale_mode == "global":
            prior = prior * float(self.huckel_scale_global)
        elif self.huckel_scale_mode == "pair_block":
            scale_table = ctx.calibration_edge_scale(like)
            if scale_table is None:
                raise ValueError(
                    "flow_options.huckel_scale_mode='pair_block' requires "
                    "flow_options.prior_calibration pointing to an artifact with an "
                    "`edge_scale` table (tools/calibrate_huckel_scales.py)."
                )
            edge_types = self._edge_types(data, like, purpose="huckel_scale_mode='pair_block'")
            if edge_types is None:
                raise KeyError(
                    "flow_options.huckel_scale_mode='pair_block' requires `edge_type` "
                    "rows aligned with the edge overlap tensor."
                )
            prior = prior * scale_table.index_select(0, edge_types)
        channel_scale = self._channel_scale_like(prior)
        if channel_scale is not None:
            prior = prior * channel_scale
        if (
            self.huckel_edge_length_decay > 0.0
            and data is not None
            and _keys.EDGE_LENGTH_KEY in data
        ):
            edge_lengths = data[_keys.EDGE_LENGTH_KEY].to(device=like.device, dtype=like.dtype)
            edge_lengths = edge_lengths.reshape(-1)
            if int(edge_lengths.shape[0]) == int(like.shape[0]):
                view_shape = (edge_lengths.shape[0],) + (1,) * (like.ndim - 1)
                prior = prior * torch.exp(
                    -edge_lengths.reshape(view_shape) / self.huckel_edge_length_decay
                )
        return prior * ctx.prior_mask(data, like, label).to(dtype=like.dtype)


class HaarDMFamily(PriorFamily):
    NAMES = HAAR_DM_NAMES

    def __init__(
        self,
        *,
        node_key: str,
        edge_key: str,
        candidate_index: int,
        strict: bool,
    ) -> None:
        self.haar_node_key = node_key
        self.haar_edge_key = edge_key
        self.haar_candidate_index = candidate_index
        self.haar_dm_strict = strict

    @classmethod
    def from_options(cls, options: Dict[str, Any], idp: Any) -> "HaarDMFamily":
        return cls(
            node_key=str(options.get("haar_node_key", _keys.HAAR_NODE_FEATURES_KEY)),
            edge_key=str(options.get("haar_edge_key", _keys.HAAR_EDGE_FEATURES_KEY)),
            candidate_index=int(options.get("haar_candidate_index", -1)),
            strict=bool(options.get("haar_dm_strict", True)),
        )

    def absolute_prior_like(self, like, *, data, label, ctx, candidate_idx=None):
        if label == "node":
            key = self.haar_node_key
        elif label == "edge":
            key = self.haar_edge_key
        else:
            return torch.zeros_like(like)
        source = None if data is None else data.get(key, None)
        if source is None:
            if self.haar_dm_strict:
                raise KeyError(
                    f"flow_options.prior='haar_dm' requires precomputed `{key}` "
                    f"for the {label} prior. Precompute RME(D_haar) offline or "
                    "change the experiment label; no nonphysical random-matrix fallback is used."
                )
            return None
        prior = ctx.select_haar_candidate(
            source, like, key=key, label=label, candidate_idx=candidate_idx
        )
        if prior.shape != like.shape:
            raise ValueError(
                f"Haar-DM {label} prior `{key}` selected shape {tuple(prior.shape)} "
                f"does not match target shape {tuple(like.shape)}."
            )
        if not torch.isfinite(prior).all():
            raise ValueError(f"Haar-DM {label} prior `{key}` contains non-finite values.")
        return prior


class ExternalFamily(PriorFamily):
    NAMES = EXTERNAL_NAMES

    def __init__(
        self,
        *,
        prior: str,
        node_key: str,
        edge_key: str,
        key_prefixes: Tuple[str, ...],
        strict: bool,
    ) -> None:
        self.prior = prior
        self.prior_node_key = node_key
        self.prior_edge_key = edge_key
        self.prior_key_prefixes = key_prefixes
        self.external_prior_strict = strict

    @classmethod
    def from_options(cls, options: Dict[str, Any], idp: Any) -> "ExternalFamily":
        raw_prefixes = options.get("prior_key_prefixes", ())
        if isinstance(raw_prefixes, str):
            raw_prefixes = [raw_prefixes]
        key_prefixes = tuple(
            str(prefix).strip() for prefix in raw_prefixes if str(prefix).strip()
        )
        return cls(
            prior=str(options.get("prior", "zero")).lower().replace("-", "_"),
            node_key=str(options.get("prior_node_key", "") or ""),
            edge_key=str(options.get("prior_edge_key", "") or ""),
            key_prefixes=key_prefixes,
            strict=bool(options.get("external_prior_strict", True)),
        )

    def _prefixes(self) -> Tuple[str, ...]:
        if self.prior_key_prefixes:
            return self.prior_key_prefixes
        if self.prior == "dftb":
            return ("dftb", "dftbsk", "sk", "nnsk")
        if self.prior == "xtb":
            return ("xtb", "gfn", "gfn1", "gfn2")
        if self.prior == "sk":
            return ("sk", "dftbsk", "nnsk")
        if self.prior == "nnsk":
            return ("nnsk", "sk")
        if self.prior in {"dftb_xtb", "physical"}:
            return ("dftb", "xtb", "dftbsk", "sk", "nnsk", "gfn", "gfn1", "gfn2")
        return ("prior", "external")

    def candidate_keys(self, label: Optional[str]) -> Tuple[str, ...]:
        if label == "node":
            explicit = self.prior_node_key
        elif label == "edge":
            explicit = self.prior_edge_key
        else:
            explicit = ""

        keys = []
        if explicit:
            keys.append(explicit)
        # Explicit keys win; otherwise only the short, documented per-prefix set.
        for prefix in self._prefixes():
            keys.extend(
                [
                    f"{prefix}_{label}_h0",
                    f"{label}_{prefix}_h0",
                    f"{prefix}_{label}_features",
                ]
            )
        out = []
        seen = set()
        for key in keys:
            key = str(key or "")
            if key and key not in seen:
                out.append(key)
                seen.add(key)
        return tuple(out)

    def absolute_prior_like(self, like, *, data, label, ctx):
        if data is None:
            return None
        for key in self.candidate_keys(label):
            source = data.get(key, None)
            if source is None:
                continue
            return ctx.coerce_prior_source(source, like, key=key, label=label)
        return None


class DFTBSKFamily(PriorFamily):
    NAMES = DFTBSK_NAMES

    def __init__(
        self,
        *,
        prior: str,
        skdata: str,
        overlap: bool,
        strict: bool,
        require_geometry: bool,
    ) -> None:
        self.prior = prior
        self.prior_skdata = skdata
        self.dftb_prior_overlap = overlap
        self.dftb_prior_strict = strict
        self.dftb_prior_require_geometry = require_geometry
        self._model_cache: Dict[Tuple[str, torch.dtype], Any] = {}
        self._last: Optional[
            Tuple[
                Tuple[int, int, int, str, torch.dtype],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
            ]
        ] = None

    @classmethod
    def from_options(cls, options: Dict[str, Any], idp: Any) -> "DFTBSKFamily":
        skdata = str(
            options.get(
                "prior_skdata", options.get("dftb_skdata", options.get("skdata", ""))
            )
            or ""
        )
        return cls(
            prior=str(options.get("prior", "zero")).lower().replace("-", "_"),
            skdata=skdata,
            overlap=bool(options.get("dftb_prior_overlap", False)),
            strict=bool(options.get("dftb_prior_strict", True)),
            require_geometry=bool(options.get("dftb_prior_require_geometry", True)),
        )

    def should_try(self) -> bool:
        if self.prior in self.NAMES:
            return True
        return bool(self.prior_skdata) and self.prior in {
            "dftb",
            "dftb_xtb",
            "physical",
            "sk",
        }

    def _basis(self, ctx: PriorContext) -> Optional[Dict[str, Any]]:
        basis = getattr(ctx.idp, "basis", None)
        if isinstance(basis, dict):
            return basis
        return None

    def _model(self, ctx: PriorContext, *, device: torch.device, dtype: torch.dtype) -> Any:
        if not self.prior_skdata:
            raise ValueError(
                "On-the-fly DFTB-SK prior requires flow_options.prior_skdata "
                "or flow_options.dftb_skdata."
            )
        basis = self._basis(ctx)
        if basis is None:
            raise ValueError(
                "On-the-fly DFTB-SK prior requires an idp with a basis dictionary."
            )
        key = (str(device), dtype)
        model = self._model_cache.get(key)
        if model is None:
            from dptb.nn.dftbsk import DFTBSK

            model = DFTBSK(
                basis=basis,
                skdata=self.prior_skdata,
                overlap=self.dftb_prior_overlap,
                dtype=dtype,
                device=device,
                transform=True,
            )
            model.eval()
            self._model_cache[key] = model
        return model

    def _outputs(
        self,
        data: Optional[AtomicDataDict.Type],
        ctx: PriorContext,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if data is None:
            return None, None
        cache_key = (
            id(data),
            id(data.get(AtomicDataDict.ATOM_TYPE_KEY, None)),
            id(data.get(_keys.EDGE_INDEX_KEY, None)),
            str(device),
            dtype,
        )
        if self._last is not None:
            last_key, last_node, last_edge = self._last
            if last_key == cache_key:
                return last_node, last_edge

        runtime_data = data.copy()
        if self.dftb_prior_require_geometry and (
            _keys.EDGE_VECTORS_KEY not in runtime_data
            and (
                _keys.POSITIONS_KEY not in runtime_data
                or _keys.EDGE_INDEX_KEY not in runtime_data
            )
        ):
            raise KeyError(
                "On-the-fly DFTB-SK prior requires edge_vectors or pos+edge_index "
                "so hopping features can be rotated into the DeePTB RME layout."
            )
        if _keys.PBC_KEY not in runtime_data:
            num_graphs = ctx.num_graphs(runtime_data)
            runtime_data[_keys.PBC_KEY] = torch.zeros(
                num_graphs,
                3,
                device=device,
                dtype=torch.bool,
            )
        model = self._model(ctx, device=device, dtype=dtype)
        with torch.no_grad():
            out = model(runtime_data)
        node = out.get(_keys.NODE_FEATURES_KEY, None)
        edge = out.get(_keys.EDGE_FEATURES_KEY, None)
        self._last = (cache_key, node, edge)
        return node, edge

    def absolute_prior_like(self, like, *, data, label, ctx):
        if not self.should_try():
            return None
        if not self.prior_skdata:
            if self.prior in self.NAMES:
                raise ValueError(
                    "flow_options.prior='dftbsk' requires prior_skdata/dftb_skdata."
                )
            return None
        try:
            node, edge = self._outputs(
                data,
                ctx,
                device=like.device,
                dtype=like.dtype,
            )
            source = node if label == "node" else edge if label == "edge" else None
            if source is None:
                return None
            return ctx.coerce_prior_source(
                source,
                like,
                key=f"on_the_fly_dftbsk:{label}",
                label=label,
            )
        except Exception as exc:
            if self.prior in self.NAMES or self.dftb_prior_strict:
                raise RuntimeError(
                    "On-the-fly DFTB-SK prior failed. Check prior_skdata, basis, "
                    "edge geometry, and target feature layout."
                ) from exc
            log.warning("On-the-fly DFTB-SK prior failed; falling back: %s", exc)
            return None


class SplitPriorFamily(PriorFamily):
    """Compose one family for the node (onsite) prior and another for the edge
    (hopping) prior -- e.g. the hybrid oracle ``prior_node='basis_onsite'`` +
    ``prior_edge='external'`` (real H0 hoppings), or an overlap-only stack with
    different node/edge mechanisms.

    Only absolute physical families are composable (basis_onsite /
    overlap_huckel / external / dftbsk).  Haar-DM is excluded: its node and edge
    candidates must share one coherent draw, which a per-label split would break.
    Fail-closed: a side whose family produces no prior raises instead of
    degrading to zeros.
    """

    SPLIT_ALLOWED = (BasisOnsiteFamily, OverlapHuckelFamily, ExternalFamily, DFTBSKFamily)

    def __init__(self, *, node_family: PriorFamily, edge_family: PriorFamily,
                 node_name: str, edge_name: str) -> None:
        self.node_family = node_family
        self.edge_family = edge_family
        self.node_name = node_name
        self.edge_name = edge_name

    @classmethod
    def resolve_side(cls, name: str, families: Dict[type, PriorFamily]) -> PriorFamily:
        name = str(name).lower().replace("-", "_")
        for fam_cls in cls.SPLIT_ALLOWED:
            if name in fam_cls.NAMES:
                return families[fam_cls]
        allowed = sorted(set().union(*(f.NAMES for f in cls.SPLIT_ALLOWED)))
        raise ValueError(
            f"flow_options.prior_node/prior_edge={name!r} is not a splittable "
            f"physical prior family; use one of {allowed}. 'haar_dm' and the "
            "noise priors (te/gaussian/zero) cannot be split per label."
        )

    def absolute_prior_like(self, like, *, data, label, ctx):
        if label == "node":
            family, name = self.node_family, self.node_name
        elif label == "edge":
            family, name = self.edge_family, self.edge_name
        else:
            return torch.zeros_like(like)
        prior = family.absolute_prior_like(like, data=data, label=label, ctx=ctx)
        if prior is None:
            raise KeyError(
                f"Split prior: the {label} family {name!r} produced no absolute "
                f"prior (missing fields or idp support). The split prior is "
                "fail-closed; provide the required dataset fields or choose a "
                "different prior_node/prior_edge."
            )
        return prior


# prior-name -> family class.  Instantiated once per HamiltonianCFM.
PRIOR_FAMILY_CLASSES = (
    BasisOnsiteFamily,
    OverlapHuckelFamily,
    HaarDMFamily,
    ExternalFamily,
    DFTBSKFamily,
)


def build_prior_families(
    options: Dict[str, Any], idp: Any
) -> Dict[type, PriorFamily]:
    """Instantiate every family once (each parses only its own option keys)."""
    return {cls: cls.from_options(options, idp) for cls in PRIOR_FAMILY_CLASSES}

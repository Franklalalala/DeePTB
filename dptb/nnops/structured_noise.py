"""Shared structured TE draws, orbital masks and graph-indexed RNG streams.

Random draws follow raw mapper spans, node before edge, without changing
dtype, device or the training RNG stream.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from dptb.data import AtomicDataDict, _keys
from dptb.nnops.layout import project_uureal_to_like

log = logging.getLogger(__name__)

class StructuredNoise:
    """Reusable sampler methods; callers supply prior options and the mapper."""


    @staticmethod
    def _num_graphs(data: AtomicDataDict.Type) -> int:
        batch = data.get(_keys.BATCH_KEY, None)
        if batch is None or batch.numel() == 0:
            return 1
        return int(batch.max().item()) + 1

    def _base_like(self, data: AtomicDataDict.Type, target: torch.Tensor, h0_key: str, label: str) -> torch.Tensor:
        base = data.get(h0_key, None)
        if base is None:
            if self.strict_h0:
                raise KeyError(
                    f"Prior noise requires `{h0_key}` for the {label} base; "
                    "set missing_h0_policy='zero' only for an explicit zero-base experiment."
                )
            if self.warn_missing_h0:
                log.warning(
                    "Prior noise did not find `%s`; falling back to zeros for %s base. "
                    "For NextHAM-style training, make sure the dataset emits node_h0/edge_h0.",
                    h0_key,
                    label,
                )
            base = torch.zeros_like(target)
        else:
            base = base.to(device=target.device, dtype=target.dtype)
            if base.shape != target.shape:
                if self.strict_h0:
                    raise ValueError(
                        f"Prior noise {label} base `{h0_key}` shape {tuple(base.shape)} "
                        f"!= target shape {tuple(target.shape)}."
                    )
                if self.warn_missing_h0:
                    log.warning(
                        "Prior noise %s base `%s` shape %s != target shape %s; using zeros.",
                        label,
                        h0_key,
                        tuple(base.shape),
                        tuple(target.shape),
                    )
                base = torch.zeros_like(target)
        return base

    @staticmethod
    def _align_bool_mask(mask: torch.Tensor, like: torch.Tensor, *, pad_value: bool = False) -> torch.Tensor:
        mask = mask.to(device=like.device, dtype=torch.bool)
        if mask.ndim == 0:
            mask = mask.reshape(1, 1)
        elif mask.ndim == 1:
            mask = mask.reshape(-1, 1)
        elif mask.ndim > 2:
            mask = mask.reshape(mask.shape[0], -1)

        fill = bool(pad_value)
        if mask.shape[0] < like.shape[0]:
            pad = torch.full(
                (like.shape[0] - mask.shape[0], mask.shape[1]),
                fill_value=fill,
                device=like.device,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, pad], dim=0)
        elif mask.shape[0] > like.shape[0]:
            mask = mask[: like.shape[0]]

        if mask.shape[-1] == 1:
            while mask.ndim < like.ndim:
                mask = mask.unsqueeze(-1)
            return mask.expand_as(like)
        if mask.shape[-1] < like.shape[-1]:
            pad = torch.full(
                (mask.shape[0], like.shape[-1] - mask.shape[-1]),
                fill_value=fill,
                device=like.device,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, pad], dim=-1)
        elif mask.shape[-1] > like.shape[-1]:
            mask = mask[:, : like.shape[-1]]
        while mask.ndim < like.ndim:
            mask = mask.unsqueeze(-1)
        return mask.expand_as(like)

    def _project_raw_feature_table(self, table: torch.Tensor, feature_dim: int) -> torch.Tensor:
        if table.ndim < 2 or table.shape[-1] == int(feature_dim) or self.idp is None:
            return table
        like = table.new_empty((table.shape[0], int(feature_dim)))
        table, _raw_mask = project_uureal_to_like(self.idp, table, like)
        return table

    def _prior_mask(
        self,
        data: Optional[AtomicDataDict.Type],
        like: torch.Tensor,
        label: Optional[str],
    ) -> torch.Tensor:
        mask = torch.ones_like(like, dtype=torch.bool, device=like.device)
        if data is None or self.idp is None or like.ndim < 2:
            return mask

        if label == "node":
            type_key = AtomicDataDict.ATOM_TYPE_KEY
            mask_table = getattr(self.idp, "mask_to_nrme", None)
            expert_key = "expert_node_mask"
        elif label == "edge":
            type_key = AtomicDataDict.EDGE_TYPE_KEY
            mask_table = getattr(self.idp, "mask_to_erme", None)
            expert_key = "expert_edge_mask"
        else:
            return mask

        types = data.get(type_key, None)
        if types is not None and mask_table is not None:
            table = mask_table.to(device=like.device, dtype=torch.bool)
            if table.ndim == 0:
                table = table.reshape(1, 1)
            elif table.ndim == 1:
                table = table.reshape(-1, 1)
            elif table.ndim > 2:
                table = table.reshape(table.shape[0], -1)
            table = self._project_raw_feature_table(table, like.shape[-1])

            type_mask = torch.zeros(
                like.shape[0],
                table.shape[1],
                device=like.device,
                dtype=torch.bool,
            )
            if table.shape[0] > 0:
                row_types = types.to(device=like.device, dtype=torch.long).reshape(-1)
                take = min(int(row_types.numel()), int(like.shape[0]))
                if take > 0:
                    raw_types = row_types[:take]
                    valid = (raw_types >= 0) & (raw_types < table.shape[0])
                    valid_rows = torch.arange(take, device=like.device, dtype=torch.long)[valid]
                    if valid_rows.numel() > 0:
                        type_mask[valid_rows] = table.index_select(0, raw_types[valid])
            mask = mask & self._align_bool_mask(type_mask, like)

        expert_mask = data.get(expert_key, None)
        if expert_mask is not None:
            mask = mask & self._align_bool_mask(expert_mask, like)
        return mask

    def _row_graph_index(
        self,
        data: Optional[AtomicDataDict.Type],
        count: int,
        label: Optional[str],
        device: torch.device,
    ) -> torch.Tensor:
        if count <= 0 or data is None:
            return torch.zeros(count, device=device, dtype=torch.long)

        batch = data.get(_keys.BATCH_KEY, None)
        if batch is None or batch.numel() == 0:
            return torch.zeros(count, device=device, dtype=torch.long)
        batch = batch.to(device=device, dtype=torch.long).reshape(-1)

        if label == "node":
            if batch.numel() < count:
                batch = torch.cat([batch, batch.new_zeros(count - batch.numel())], dim=0)
            return batch[:count]

        if label == "edge":
            edge_index = data.get(_keys.EDGE_INDEX_KEY, None)
            if edge_index is None or edge_index.numel() == 0:
                return torch.zeros(count, device=device, dtype=torch.long)
            centers = edge_index[0].to(device=device, dtype=torch.long).reshape(-1)
            if centers.numel() < count:
                centers = torch.cat([centers, centers.new_zeros(count - centers.numel())], dim=0)
            centers = centers[:count].clamp(min=0, max=max(batch.numel() - 1, 0))
            return batch.index_select(0, centers)

        return torch.zeros(count, device=device, dtype=torch.long)

    def _row_type_index(
        self,
        data: Optional[AtomicDataDict.Type],
        count: int,
        label: Optional[str],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if count <= 0 or data is None:
            return None
        if label == "node":
            key = AtomicDataDict.ATOM_TYPE_KEY
        elif label == "edge":
            key = AtomicDataDict.EDGE_TYPE_KEY
        else:
            return None
        values = data.get(key, None)
        if values is None:
            return None
        values = values.to(device=device, dtype=torch.long).reshape(-1)
        if values.numel() < count:
            values = torch.cat([values, values.new_zeros(count - values.numel())], dim=0)
        return values[:count]

    def _te_irrep_slices(self, feature_dim: int) -> Optional[Tuple[Tuple[int, int, int], ...]]:
        feature_dim = int(feature_dim)
        cache = getattr(self, "_te_irrep_slices_cache", None)
        if isinstance(cache, dict) and feature_dim in cache:
            return cache[feature_dim]

        def _remember(value):
            if isinstance(cache, dict):
                cache[feature_dim] = value
            return value

        if self.idp is None:
            return _remember(None)
        irreps = getattr(self.idp, "orbpair_irreps", None)
        if irreps is None:
            get_irreps = getattr(self.idp, "get_irreps", None)
            if not callable(get_irreps):
                return _remember(None)
            try:
                irreps = get_irreps()
            except Exception:
                return _remember(None)
            if irreps is None:
                return _remember(None)
        # Feature rows follow OrbitalMapper/orbpair_maps order. Sorting irreps
        # changes contiguous feature spans and breaks mask/typewise raw-slice priors.
        raw_irreps = irreps

        slices = []
        offset = 0
        try:
            for mul, ir in raw_irreps:
                degree = int(getattr(ir, "l", 0))
                width = int(getattr(ir, "dim", 2 * degree + 1))
                for _ in range(int(mul)):
                    slices.append((offset, offset + width, degree))
                    offset += width
        except Exception:
            return _remember(None)
        if offset != feature_dim:
            raw_mask = getattr(self.idp, "mask_uureal", None)
            if raw_mask is None:
                return _remember(None)
            raw_mask = raw_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
            if raw_mask.numel() != offset:
                return _remember(None)
            if int(raw_mask.sum().item()) != feature_dim:
                return _remember(None)
            projected_slices = []
            compressed_offset = 0
            for start, end, degree in slices:
                kept = int(raw_mask[start:end].sum().item())
                if kept <= 0:
                    continue
                projected_slices.append((compressed_offset, compressed_offset + kept, degree))
                compressed_offset += kept
            if compressed_offset != feature_dim:
                return _remember(None)
            return _remember(tuple(projected_slices))
        return _remember(tuple(slices))

    def _te_radius(
        self,
        row_count: int,
        active_dim: torch.Tensor,
        graph_index: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_graphs: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.te_prior_per_graph and graph_index is not None and graph_index.numel() == row_count:
            if num_graphs is None:
                num_graphs = int(graph_index.max().detach().item()) + 1 if row_count > 0 else 1
            radius = torch.randn(
                num_graphs,
                1,
                device=device,
                dtype=dtype,
                generator=generator,
            ).index_select(0, graph_index)
        else:
            radius = torch.randn(
                row_count,
                1,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        return radius * active_dim.sqrt()


    def _apply_typewise_residual_scale(
        self,
        noise: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor,
        data: Optional[AtomicDataDict.Type],
        label: Optional[str],
        slices: Tuple[Tuple[int, int, int], ...],
        *,
        reference_scale: bool,
    ) -> torch.Tensor:
        if self.te_prior_mode != "typewise" or not reference_scale:
            return noise
        type_index = self._row_type_index(data, noise.shape[0], label, noise.device)
        if type_index is None or noise.ndim < 2:
            return noise

        out = noise.clone()
        ref = reference.detach().to(device=noise.device, dtype=noise.dtype)
        mask_f = mask.to(device=noise.device, dtype=noise.dtype)
        _types, inverse = torch.unique(type_index, sorted=True, return_inverse=True)
        num_types = int(_types.numel())
        if num_types == 0:
            return out

        for start, end, _degree in slices:
            seg_mask = mask_f[:, start:end]
            row_count = seg_mask.sum(dim=-1)
            row_square_sum = (ref[:, start:end].square() * seg_mask).sum(dim=-1)

            type_count = torch.zeros(
                num_types, device=noise.device, dtype=noise.dtype
            ).scatter_add_(0, inverse, row_count)
            type_square_sum = torch.zeros_like(type_count).scatter_add_(
                0, inverse, row_square_sum
            )

            rms = torch.sqrt(type_square_sum / type_count.clamp_min(1.0))
            valid = (type_count > 0) & torch.isfinite(rms)
            scale_by_type = torch.where(
                valid,
                rms.clamp_min(self.residual_sigma_floor),
                torch.ones_like(rms),
            )
            out[:, start:end] = (
                out[:, start:end] * scale_by_type.index_select(0, inverse).unsqueeze(-1)
            )
        return out

    def _te_prior_like(
        self,
        like: torch.Tensor,
        sigma: float,
        *,
        data: Optional[AtomicDataDict.Type] = None,
        label: Optional[str] = None,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        if like.numel() == 0:
            return torch.zeros_like(like)
        if like.ndim < 2:
            raise ValueError("Prior noise requires RME feature rows with rank >= 2")
        mask = self._prior_mask(data, like, label)
        slices = self._te_irrep_slices(like.shape[-1])
        if slices is None:
            raise ValueError("Prior noise requires mapper irrep spans matching the feature width")
        noise = torch.zeros_like(like)
        graph_index = self._row_graph_index(data, like.shape[0], label, like.device)
        for start, end, _degree in slices:
            seg_mask = mask[:, start:end].to(device=like.device, dtype=like.dtype)
            raw = torch.randn(
                like.shape[0],
                end - start,
                device=like.device,
                dtype=like.dtype,
            )
            raw = raw * seg_mask
            norm = raw.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-8)
            direction = raw / norm
            active_dim = seg_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            radius = self._te_radius(
                like.shape[0],
                active_dim,
                graph_index,
                device=like.device,
                dtype=like.dtype,
                num_graphs=num_graphs,
            )
            noise[:, start:end] = direction * radius * seg_mask

        noise = self._apply_typewise_residual_scale(
            noise,
            like,
            mask,
            data,
            label,
            slices,
            reference_scale=True,
        )
        return noise * (float(sigma) * self.te_prior_sigma)

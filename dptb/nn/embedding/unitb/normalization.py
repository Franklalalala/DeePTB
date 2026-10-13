from __future__ import annotations

from typing import Union

import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import FromS2Grid, ToS2Grid


_GRID_MAT_CACHE: dict[tuple[int, int, str, tuple[int, int]], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_grid_mats(
    lmax: int,
    mmax: int,
    normalization: str,
    resolution: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Grid matrices are generated under the ambient default dtype; keying the
    # cache on it prevents a float32 build from leaking float32 constants into
    # a later float64 build in the same process.
    key = (lmax, mmax, normalization, resolution, torch.get_default_dtype())
    mats = _GRID_MAT_CACHE.get(key)
    if mats is not None:
        return mats

    to_grid = ToS2Grid(
        lmax,
        resolution,
        normalization=normalization,
        device="cpu",
    )
    to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
    to_grid_mat = to_grid_mat.flatten(0, 1).contiguous()

    from_grid = FromS2Grid(
        resolution,
        lmax,
        normalization=normalization,
        device="cpu",
    )
    from_grid_mat = torch.einsum("am, mbi -> bai", from_grid.sha, from_grid.shb).detach()
    from_grid_mat = from_grid_mat.flatten(0, 1).permute(1, 0).contiguous()

    mats = (to_grid_mat, from_grid_mat)
    _GRID_MAT_CACHE[key] = mats
    return mats


class EquivariantMergedRMSNormFlat(nn.Module):
    def __init__(
        self,
        irreps: Union[str, o3.Irreps],
        eps: float = 1e-6,
        affine: bool = True,
        normalization: str = "component",
        std_balance_degrees: bool = True,
        center_0e: bool = True,
        treat_0o_as_scalar: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()

        self.irreps = o3.Irreps(irreps).simplify()
        self.eps = eps
        self.affine = affine
        self.normalization = normalization
        self.std_balance_degrees = std_balance_degrees
        self.center_0e = center_0e
        self.treat_0o_as_scalar = treat_0o_as_scalar

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if isinstance(device, str):
            device = torch.device(device)

        if normalization not in ("component", "norm"):
            raise ValueError(f"Unsupported normalization={normalization!r}")

        dim_to_group = []
        group_to_degree = []
        group_inv_dims = []
        scalar_dim_idx = []
        degree_counts = {}

        offset = 0
        group_id = 0
        for mul, ir in self.irreps:
            for _ in range(mul):
                dim_to_group.extend([group_id] * ir.dim)
                group_to_degree.append(ir.l)
                group_inv_dims.append(1.0 / ir.dim)
                if ir.l == 0 and (ir.p == 1 or self.treat_0o_as_scalar):
                    scalar_dim_idx.append(offset)
                degree_counts[ir.l] = degree_counts.get(ir.l, 0) + 1
                offset += ir.dim
                group_id += 1

        self.dim = offset
        self.num_groups = group_id
        self.num_scalar = len(scalar_dim_idx)
        self.num_degrees = len(degree_counts)

        degree_ids = sorted(degree_counts.keys())
        degree_map = {degree: idx for idx, degree in enumerate(degree_ids)}
        group_to_degree = [degree_map[degree] for degree in group_to_degree]
        degree_inv_group_counts = [1.0 / degree_counts[degree] for degree in degree_ids]

        self.register_buffer("dim_to_group", torch.tensor(dim_to_group, dtype=torch.long, device=device))
        self.register_buffer(
            "group_to_degree",
            torch.tensor(group_to_degree, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "group_inv_dims",
            torch.tensor(group_inv_dims, dtype=dtype, device=device).unsqueeze(0),
        )
        self.register_buffer(
            "degree_inv_group_counts",
            torch.tensor(degree_inv_group_counts, dtype=dtype, device=device).unsqueeze(0),
        )
        self.register_buffer(
            "scalar_dim_idx",
            torch.tensor(scalar_dim_idx, dtype=torch.long, device=device),
        )

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, self.num_groups, dtype=dtype, device=device))
            if self.center_0e and self.num_scalar > 0:
                self.affine_bias = nn.Parameter(torch.zeros(1, self.num_scalar, dtype=dtype, device=device))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        # per-block plan of the forward: (mul, dim, degree index, is a centred scalar)
        self._blocks = tuple(
            (mul, ir.dim, degree_map[ir.l], ir.l == 0 and (ir.p == 1 or self.treat_0o_as_scalar))
            for mul, ir in self.irreps
        )
        self._widths = [mul * dim for mul, dim, _, _ in self._blocks]
        self._group_widths = [mul for mul, _, _, _ in self._blocks]
        self._scalar_widths = [mul for mul, _, _, scalar in self._blocks if scalar]
        self._degree_inv_counts = tuple(degree_inv_group_counts)

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Centre the 0e channels, scale every channel by the inverse RMS merged over
        degrees, apply the affine weight and bias.

        The input is split into its irrep blocks once and the output concatenated once;
        the reductions run per block, so neither direction scatters or gathers over the
        full feature width."""
        if x.ndim != 2:
            raise ValueError(f"Expected [N, dim], got shape={tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[-1]}")

        orig_dtype = x.dtype
        # Upcast only low-precision inputs; float64 must stay float64 or the
        # norm silently truncates activations/gradients to float32 accuracy.
        compute_dtype = (
            torch.float32
            if orig_dtype in (torch.float16, torch.bfloat16)
            else orig_dtype
        )
        y = x.to(compute_dtype)
        n = y.shape[0]
        parts = torch.split(y, self._widths, dim=1) if len(self._widths) > 1 else (y,)
        views = [part if dim == 1 else part.reshape(n, mul, dim)
                 for part, (mul, dim, _, _) in zip(parts, self._blocks)]

        scalar_blocks = [i for i, block in enumerate(self._blocks) if block[3]]
        if self.center_0e and scalar_blocks:
            total = views[scalar_blocks[0]].sum(dim=1, keepdim=True)
            for i in scalar_blocks[1:]:
                total = total + views[i].sum(dim=1, keepdim=True)
            mean = total / self.num_scalar
            for i in scalar_blocks:
                views[i] = views[i] - mean

        # mean square of every group (one irrep channel), summed per block
        block_ms = []
        for view, (mul, dim, _, _) in zip(views, self._blocks):
            group_ms = view.square() if dim == 1 else view.square().sum(dim=-1)
            if self.normalization == "component" and dim > 1:
                group_ms = group_ms * (1.0 / dim)
            block_ms.append(group_ms.sum(dim=1, keepdim=True))
        if self.std_balance_degrees:
            degree_sums = [None] * self.num_degrees
            for ms, (_, _, degree, _) in zip(block_ms, self._blocks):
                degree_sums[degree] = ms if degree_sums[degree] is None else degree_sums[degree] + ms
            merged_ms = degree_sums[0] * self._degree_inv_counts[0]
            for degree in range(1, self.num_degrees):
                merged_ms = merged_ms + degree_sums[degree] * self._degree_inv_counts[degree]
            merged_ms = merged_ms / self.num_degrees
        else:
            merged_ms = block_ms[0]
            for ms in block_ms[1:]:
                merged_ms = merged_ms + ms
            merged_ms = merged_ms / self.num_groups

        scale = torch.rsqrt(merged_ms + self.eps)
        weights = None
        if self.affine:
            weights = (torch.split(self.affine_weight, self._group_widths, dim=1)
                       if len(self._group_widths) > 1 else (self.affine_weight,))
        biases = None
        if self.affine and self.affine_bias is not None and self._scalar_widths:
            biases = (torch.split(self.affine_bias, self._scalar_widths, dim=1)
                      if len(self._scalar_widths) > 1 else (self.affine_bias,))
        out = []
        k = 0
        for i, (view, (mul, dim, _, scalar)) in enumerate(zip(views, self._blocks)):
            block_scale = scale * weights[i] if weights is not None else scale
            block = view * block_scale if dim == 1 else view * block_scale.unsqueeze(-1)
            block = block.reshape(n, mul * dim)
            if biases is not None and scalar:
                block = block + biases[k]
                k += 1
            out.append(block)
        y = out[0] if len(out) == 1 else torch.cat(out, dim=1)
        return y.to(orig_dtype)


def build_equivariant_norm(
    norm_type: str,
    irreps: o3.Irreps,
    norm_eps: float,
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
):
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "merged_rms":
        return EquivariantMergedRMSNormFlat(
            irreps,
            eps=norm_eps,
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"Unsupported equivariant_norm_type={norm_type!r}")

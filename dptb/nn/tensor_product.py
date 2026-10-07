from e3nn.o3 import xyz_to_angles, Irreps
import math
import torch
import torch.nn as nn
from torch.nn import Linear
import os
from collections import defaultdict
from .so2_backend import true_dense_forward
from .so2_parity import ParityLinear, normalize_so2_parity, enforce_so2_parity, parity_masks


def complex_pair_output(x_m: torch.Tensor, num_out_channel: int) -> torch.Tensor:
    """Complex product of an SO2 m block from its real linear output.

    x_m is [N, 2, 2C]: pair row 0 holds (W_r x_r, W_i x_r) and row 1 holds
    (W_r x_i, W_i x_i).  The result [N, 2, C] is (W_r x_r - W_i x_i, W_r x_i + W_i x_r).
    unbind/stack give the backward one gradient buffer per level, where narrow()
    views zero-fill a full-size gradient each.
    """
    row_0, row_1 = x_m.unflatten(-1, (2, num_out_channel)).unbind(1)
    r_0, i_0 = row_0.unbind(-2)
    r_1, i_1 = row_1.unbind(-2)
    return torch.stack((r_0 - i_1, r_1 + i_0), dim=1)

_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"), weights_only=False)
_idx_data = torch.load(os.path.join(os.path.dirname(__file__), "z_rot_indices_lmax12.pt"), weights_only=False)


def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc

def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


def build_z_rot_multi(angle_stack, mask, freq, reversed_inds, offsets, sizes):
    """
    angle_stack: (3*N, )    # Input with alpha, beta, gamma stacked together
    Returns: (Xa, Xb, Xc) # Each is of shape (N, D_total, D_total)
    """
    N_all = angle_stack.shape[0]
    N = N_all // 3

    D_total = sizes.sum().item()

    # Step 1: Vectorized computation of sine and cosine values
    angle_expand = angle_stack[None, :, None]  # (1, 3N, 1)
    freq_expand = freq[:, None, :]  # (L, 1, Mmax)
    sin_val = torch.sin(freq_expand * angle_expand)  # (L, 3N, Mmax)
    cos_val = torch.cos(freq_expand * angle_expand)  # (L, 3N, Mmax)

    # Step 2: Construct the block-diagonal matrix
    M_total = angle_stack.new_zeros((N_all, D_total, D_total))
    idx_l, idx_row = torch.where(mask)  # (K,), (K,)
    idx_col_diag = idx_row
    idx_col_anti = reversed_inds[idx_l, idx_row]
    global_row = offsets[idx_l] + idx_row  # (K,)
    global_col_diag = offsets[idx_l] + idx_col_diag
    global_col_anti = offsets[idx_l] + idx_col_anti

    # Assign values to the diagonal
    M_total[:, global_row, global_col_diag] = cos_val[idx_l, :, idx_row].transpose(0, 1)
    # Assign values to non-overlapping anti-diagonals
    overlap_mask = (global_row == global_col_anti)
    M_total[:, global_row[~overlap_mask], global_col_anti[~overlap_mask]] = sin_val[idx_l[~overlap_mask], :,
                                                                            idx_row[~overlap_mask]].transpose(0, 1)

    # Step 3: Split into three components corresponding to alpha, beta, gamma
    Xa = M_total[:N]
    Xb = M_total[N:2 * N]
    Xc = M_total[2 * N:]

    return Xa, Xb, Xc


def batch_wigner_D(l_max, alpha, beta, gamma, _Jd):
    """
    Compute Wigner D matrices for all L (from 0 to l_max) in a single batch.
    Returns a tensor of shape [N, D, D], where D = sum(2l+1 for l in 0..l_max).
    """
    device = alpha.device
    N = alpha.shape[0]
    idx_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in _idx_data.items()}

    # Load static data
    sizes = idx_data["sizes"][:l_max + 1]
    offsets = idx_data["offsets"][:l_max + 1]
    mask = idx_data["mask"][:l_max + 1]
    freq = idx_data["freq"][:l_max + 1]
    reversed_inds = idx_data["reversed_inds"][:l_max + 1]

    # Precompute block structure information
    dims = [2 * l + 1 for l in range(l_max + 1)]
    D_total = sum(dims)

    # Construct block-diagonal J matrix
    J_full_small = torch.zeros(D_total, D_total, device=device)
    for l in range(l_max + 1):
        start = offsets[l]
        J_full_small[start:start + 2 * l + 1, start:start + 2 * l + 1] = _Jd[l]

    J_full = J_full_small.unsqueeze(0).expand(N, -1, -1)
    angle_stack = torch.cat([alpha, beta, gamma], dim=0)
    Xa, Xb, Xc = build_z_rot_multi(angle_stack, mask, freq, reversed_inds, offsets, sizes)

    return Xa @ J_full @ Xb @ J_full @ Xc


class InterpolationBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.out_features = out_features
        hidden_features1 = max(1, int(in_features * 2 / 3 + out_features * 1 / 3))
        hidden_features2 = max(1, int(in_features * 1 / 3 + out_features * 2 / 3))
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features1, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_features1, hidden_features2, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_features2, out_features, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class SO2LinearCached(torch.nn.Module):
    """SO(2) convolution returning ``(features, wigner_cache)`` explicitly."""

    def __init__(
            self,
            irreps_in,
            irreps_out,
            radial_emb: bool = False,
            latent_dim: int = None,
            radial_channels: list = None,
            extra_m0_outsize: int = 0,
            use_interpolation: bool = False,
            # === 新增参数 ===
            rotate_in: bool = True,
            rotate_out: bool = True,
            so2_m_linear_mode: str = None,
            so2_parity: str = "none",
    ):
        super().__init__()

        self.irreps_in = Irreps(irreps_in).simplify()
        self.so2_parity = normalize_so2_parity(so2_parity)
        self.irreps_out = (Irreps(f"{extra_m0_outsize}x0e") + Irreps(irreps_out)).simplify()
        self.radial_emb = radial_emb
        self.latent_dim = latent_dim
        self.m_linear = nn.ModuleList()
        self.so2_m_linear_mode = so2_m_linear_mode or os.environ.get("DPTB_SO2_M_LINEAR_MODE", "indexed_sandwich_cuda_multi")
        if self.so2_m_linear_mode == "cuda_pack_scatter_multi":
            self.so2_m_linear_mode = "indexed_sandwich_cuda_multi"
        if self.so2_m_linear_mode not in ("standard", "indexed_sandwich_cuda_multi"):
            raise ValueError(
                "so2_m_linear_mode must be 'standard' or 'indexed_sandwich_cuda_multi', "
                f"got {self.so2_m_linear_mode!r}"
            )

        # 保存 flag
        self.rotate_in = rotate_in
        self.rotate_out = rotate_out

        num_in_m0 = self.irreps_in.num_irreps
        num_out_m0 = self.irreps_out.num_irreps

        self.fc_m0 = Linear(num_in_m0, num_out_m0, bias=True)

        for m in range(1, self.irreps_out.lmax + 1):
            self.m_linear.append(SO2_m_Linear(m, self.irreps_in, self.irreps_out, use_interpolation=use_interpolation))

        if self.so2_parity == "enforce":
            enforce_so2_parity(self)

        self.m_in_mask = torch.zeros(self.irreps_in.lmax + 1, self.irreps_in.dim, dtype=torch.bool)
        self.m_out_mask = torch.zeros(self.irreps_in.lmax + 1, self.irreps_out.dim, dtype=torch.bool)
        if self.irreps_in.dim <= self.irreps_out.dim:
            front = True
            self.m_in_num = [0] * (self.irreps_in.lmax + 1)
        else:
            front = False
            self.m_in_num = [0] * (self.irreps_out.lmax + 1)
        offset = 0
        for mul, (l, p) in self.irreps_in:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(l + 1):
                self.m_in_mask[m, start_id + l + m] = True
                self.m_in_mask[m, start_id + l - m] = True
                if front:
                    self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)
        offset = 0
        for mul, (l, p) in self.irreps_out:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(l + 1):
                if m <= self.irreps_in.lmax:
                    self.m_out_mask[m, start_id + l + m] = True
                    self.m_out_mask[m, start_id + l - m] = True
                    if not front:
                        self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)
        self.m_in_index = [0] + list(torch.cumsum(torch.tensor(self.m_in_num), dim=0))
        if radial_emb:
            self.radial_emb = RadialFunction([latent_dim] + radial_channels + [self.m_in_index[-1]])
        self.front = front
        self.l_max = max((l for (_, (l, _)), _ in zip(self.irreps_in, self.irreps_in.slices()) if l > 0), default=0)
        self.dims = {l: 2 * l + 1 for l in range(self.l_max + 1)}
        self.offsets = {}
        offset = 0
        for l in range(self.l_max + 1):
            self.offsets[l] = offset
            offset += self.dims[l]
        self._in_entries, self._in_groups = self._build_layout_plans(self.irreps_in)
        self._out_entries, self._out_groups = self._build_layout_plans(self.irreps_out)

    def forward(self, x, R, latents=None, wigner_D_all=None):
        weights = self.radial_emb(latents) if self.radial_emb else None
        if wigner_D_all is None and (self.rotate_in or self.rotate_out) and self.l_max > 0:
            angle = xyz_to_angles(R[:, [1, 2, 0]])
            wigner_D_all = batch_wigner_D(self.l_max, angle[0], angle[1], torch.zeros_like(angle[0]), _Jd)
        if self.so2_m_linear_mode != "standard":
            result = true_dense_forward(self, x, weights, wigner_D_all)
            if result is not None:
                return result
        return self._forward_standard(x, weights, wigner_D_all)

    def _forward_m0(self, x, radial, wigner_D_all):
        n = x.shape[0]
        rot_blocks = {}
        if self.rotate_in or self.rotate_out:
            rot_blocks = {l: self._select_wigner_block(wigner_D_all, l) for l in range(self.l_max + 1)}
        input_groups = {l: self._gather_l_group(x, l) for l in self._in_groups}
        out_groups = self._alloc_output_l_groups(n, dtype=x.dtype, device=x.device)
        inp = self._assemble_grouped_m0_input(input_groups, rot_blocks, n, x)
        if self.front and radial is not None:
            y_m0 = self.fc_m0(inp * radial)
        elif radial is not None:
            y_m0 = self.fc_m0(inp) * radial
        else:
            y_m0 = self.fc_m0(inp)
        self._accumulate_grouped_m0_output_(out_groups, y_m0, rot_blocks)
        return self._materialize_output_l_groups(out_groups, n=n, dtype=x.dtype, device=x.device)

    def _forward_standard(self, x, weights, wigner_D_all):
        n, _ = x.shape
        x_ = torch.zeros_like(x)

        groups = defaultdict(list)
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            groups[l].append((mul, slice_info))
            if l == 0:
                x_[:, slice_info] = x[:, slice_info]

        for l, group in groups.items():
            if l == 0 or not group:
                continue
            muls, slices = zip(*group)

            # === 如果 rotate_in 为 False，直接复制不旋转 ===
            if not self.rotate_in:
                for mul, sl in group:
                    x_[:, sl] = x[:, sl]
                continue
            # ============================================

            x_parts = [x[:, sl].reshape(n, mul, 2 * l + 1) for mul, sl in group]
            x_combined = torch.cat(x_parts, dim=1)
            start = self.offsets[l]
            rot_mat = self._select_wigner_block(wigner_D_all, l)
            transformed = torch.bmm(x_combined, rot_mat)
            for part, slice_info, mul in zip(transformed.split(muls, dim=1), slices, muls):
                x_[:, slice_info] = part.reshape(n, -1)

        out = torch.zeros(n, self.irreps_out.dim, dtype=x.dtype, device=x.device)
        for m in range(self.irreps_out.lmax + 1):
            radial_weight = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(
                1) if self.radial_emb else 1.
            if m == 0:
                if self.front and self.radial_emb:
                    out[:, self.m_out_mask[m]] += self.fc_m0(x_[:, self.m_in_mask[m]] * radial_weight.squeeze(1))
                elif self.radial_emb:
                    out[:, self.m_out_mask[m]] += self.fc_m0(x_[:, self.m_in_mask[m]]) * radial_weight.squeeze(1)
                else:
                    out[:, self.m_out_mask[m]] += self.fc_m0(x_[:, self.m_in_mask[m]])
            else:
                x_m_in = x_[:, self.m_in_mask[m]].reshape(n, -1, 2).transpose(1, 2).contiguous()
                if self.front and self.radial_emb:
                    x_m_in.mul_(radial_weight)
                    linear_output = self.m_linear[m - 1](x_m_in)
                elif self.radial_emb:
                    linear_output = self.m_linear[m - 1](x_m_in)
                    linear_output.mul_(radial_weight)
                else:
                    linear_output = self.m_linear[m - 1](x_m_in)
                final_addition = linear_output.transpose(1, 2).contiguous().reshape(n, -1)
                out[:, self.m_out_mask[m]] += final_addition

        # === 如果 rotate_out 为 False，直接返回 out，不旋转回 global ===
        if not self.rotate_out:
            return out.contiguous(), wigner_D_all
        # =========================================================

        for (mul, (l, p)), slice_in in zip(self.irreps_out, self.irreps_out.slices()):
            if l > 0:
                start = self.offsets[l]
                rot_mat = self._select_wigner_block(wigner_D_all, l)
                x_slice = out[:, slice_in].clone().reshape(n, mul, -1)
                rotated = torch.einsum('nij,nmj->nmi', rot_mat, x_slice)
                out[:, slice_in] = rotated.reshape(n, -1)
        return out.contiguous(), wigner_D_all


    @staticmethod
    def _build_layout_plans(irreps):
        running_by_l = defaultdict(int)
        groups = defaultdict(list)
        entries = []
        for (mul, (l, _p)), slice_info in zip(irreps, irreps.slices()):
            group_start = running_by_l[l]
            entries.append((l, mul, slice_info, group_start))
            running_by_l[l] += mul
            groups[l].append((mul, slice_info))
        return tuple(entries), {
            l: (
                tuple(mul for mul, _ in specs),
                tuple(slice_info for _, slice_info in specs),
                sum(mul for mul, _ in specs),
                2 * l + 1,
            )
            for l, specs in groups.items()
        }


    def _select_wigner_block(self, wigner_D_all, l):
        if hasattr(wigner_D_all, "block") and hasattr(wigner_D_all, "blocks"):
            block = wigner_D_all.block(l)
            expected = (self.dims[l], self.dims[l])
            if block.shape[-2:] != expected:
                raise ValueError(f"compact Wigner block l={l} has shape {tuple(block.shape[-2:])}, expected {expected}")
            return block
        start = self.offsets[l]
        dim = self.dims[l]
        return wigner_D_all[:, start:start + dim, start:start + dim]


    def _gather_l_group(self, x, l):
        muls, slices, _total_mul, dims = self._in_groups[l]
        n = x.shape[0]
        parts = [
            x[:, slice_info].reshape(n, mul, dims)
            for mul, slice_info in zip(muls, slices)
        ]
        if len(parts) == 1:
            return parts[0].contiguous()
        return torch.cat(parts, dim=1).contiguous()


    def _pack_group_m0(self, x_group, l, rot_block):
        if x_group.numel() == 0:
            return x_group.new_empty((x_group.shape[0], x_group.shape[1]))
        if l == 0 or not self.rotate_in or rot_block is None:
            return x_group[:, :, l]
        return torch.einsum("ncd,nd->nc", x_group, rot_block[:, :, l])


    def _alloc_output_l_groups(self, n, *, dtype, device):
        return {
            l: torch.zeros((n, total_mul, dims), dtype=dtype, device=device)
            for l, (_muls, _slices, total_mul, dims) in self._out_groups.items()
        }


    def _assemble_grouped_m0_input(self, input_groups, rot_blocks, n, x_template):
        packed_by_l = {
            l: self._pack_group_m0(x_group, l, rot_blocks.get(l))
            for l, x_group in input_groups.items()
        }
        parts = [
            packed_by_l[l][:, group_start:group_start + mul]
            for l, mul, _slice_info, group_start in self._in_entries
        ]
        if not parts:
            return x_template.new_empty((n, 0))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1)


    def _accumulate_group_m0_(self, out_group, y_group, l, rot_block):
        if y_group.numel() == 0:
            return
        if l == 0 or not self.rotate_out or rot_block is None:
            out_group[:, :, l] += y_group
            return
        out_group += y_group.unsqueeze(-1) * rot_block[:, :, l].unsqueeze(1)


    def _accumulate_grouped_m0_output_(self, out_groups, y_m0, rot_blocks):
        cursor = 0
        for l, mul, _slice_info, group_start in self._out_entries:
            y_entry = y_m0[:, cursor:cursor + mul]
            cursor += mul
            out_view = out_groups[l][:, group_start:group_start + mul, :]
            self._accumulate_group_m0_(out_view, y_entry, l, rot_blocks.get(l))


    def _materialize_output_l_groups(self, out_groups, *, n, dtype, device):
        out = torch.zeros((n, self.irreps_out.dim), dtype=dtype, device=device)
        for l, mul, slice_info, group_start in self._out_entries:
            group_view = out_groups[l][:, group_start:group_start + mul, :]
            out[:, slice_info] = group_view.reshape(n, -1)
        return out


class SO2_Linear(SO2LinearCached):
    """SO(2) convolution with the upstream constructor and tensor return value."""

    def __init__(self, irreps_in, irreps_out, radial_emb=False, latent_dim=None,
                 radial_channels=None, extra_m0_outsize=0):
        super().__init__(irreps_in, irreps_out, radial_emb, latent_dim,
                         radial_channels, extra_m0_outsize)

    def forward(self, x, R, latents=None):
        output, _ = super().forward(x, R, latents)
        return output


class SO2_m_Linear(torch.nn.Module):
    def __init__(
            self,
            m,
            irreps_in,
            irreps_out,
            use_interpolation: bool = False,
            so2_parity: str = "none",
    ):
        super(SO2_m_Linear, self).__init__()
        self.m = m
        self.num_in_channel = sum(mul for mul, (l, p) in irreps_in if l >= m)
        self.num_out_channel = sum(mul for mul, (l, p) in irreps_out if l >= m)

        if use_interpolation:
            self.fc = InterpolationBlock(self.num_in_channel, 2 * self.num_out_channel, bias=False)
        else:
            self.fc = Linear(self.num_in_channel, 2 * self.num_out_channel, bias=False)
            self.fc.weight.data.mul_(1 / math.sqrt(2))

        if normalize_so2_parity(so2_parity) == "enforce":
            if use_interpolation:
                raise ValueError("so2_parity='enforce' does not support interpolation m blocks")
            self.fc.__class__ = ParityLinear
            self.fc.set_parity_masks(*parity_masks(irreps_in, irreps_out, m))

    def forward(self, x_m):
        # x_m ~ [N, 2, n_channels]
        x_m = self.fc(x_m)
        return self._finish_linear_output(x_m)

    def _finish_linear_output(self, x_m):
        return complex_pair_output(x_m, self.num_out_channel)


class RadialFunction(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(1, len(channels_list)):
            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]
            if i < len(channels_list) - 1:
                modules.append(nn.LayerNorm(channels_list[i]))
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.net(inputs)

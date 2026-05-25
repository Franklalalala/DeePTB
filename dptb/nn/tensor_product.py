from e3nn.o3 import xyz_to_angles, Irreps
import math
import torch
import torch.nn as nn
from e3nn.o3 import Linear as e3nn_Linear
from torch.nn import Linear
import os
import torch.nn.functional as F
from collections import defaultdict

# 你可能已有的静态数据加载（保持不变）
_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"), weights_only=False)
_idx_data = torch.load(os.path.join(os.path.dirname(__file__), "z_rot_indices_lmax12.pt"), weights_only=False)


class SO2_Attention(torch.nn.Module):
    def __init__(self, node_irreps, latent_dim: int, use_so2_att_proj: bool = True):
        super().__init__()
        self.irreps_in = node_irreps.simplify()
        self.l_max = max((l for (_, (l, _)), _ in zip(self.irreps_in, self.irreps_in.slices()) if l > 0), default=0)
        self.dims = {l: 2 * l + 1 for l in range(self.l_max + 1)}
        self.offsets = {}
        offset = 0
        for l in range(self.l_max + 1):
            self.offsets[l] = offset
            offset += self.dims[l]

        self.lin_center = e3nn_Linear(node_irreps, node_irreps, shared_weights=True, internal_weights=True, biases=True)
        self.lin_neighbor = e3nn_Linear(node_irreps, node_irreps, shared_weights=True, internal_weights=True,
                                        biases=True)

        groups = defaultdict(list)
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            groups[l].append((mul, slice_info))
        self.groups = groups

        # --- 修改：为每个 l 建立输入维为 (total_mul * (2l+1)) 的线性映射
        self.sim_linears = nn.ModuleDict()
        for l, g in groups.items():
            total_mul = sum(m for m, _ in g)
            in_dim = total_mul * self.dims[l]  # m * d
            self.sim_linears[f"l{l}"] = nn.Sequential(
                nn.Linear(in_dim, latent_dim),
                nn.SiLU(),
            )

        # --- 修改：用一个 final_mlp 替代简单求和（把所有 l 的 latent_dim 串联后再做一次融合）
        num_l = len(groups)
        # final_mlp: (num_l * latent_dim) -> latent_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(num_l * latent_dim, 2 * latent_dim),
            nn.SiLU(),  # 平滑非线性（比 ReLU 更稳定）
            nn.Linear(2 * latent_dim, latent_dim)
        )

    def forward(self, node_features, active_edge_vector, active_edge_index, wigner_D_all=None):
        n, _ = node_features.shape
        # keep node features as-is (no per-edge rotation here)
        rot_n_feat_ = node_features.new_zeros(node_features.shape)

        if wigner_D_all is None and self.l_max > 0:
            angle = xyz_to_angles(active_edge_vector[:, [1, 2, 0]])
            wigner_D_all = batch_wigner_D(self.l_max, angle[0], angle[1], torch.zeros_like(angle[0]), _Jd)

        # keep scalar parts unchanged
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            if l == 0:
                rot_n_feat_[:, slice_info] = node_features[:, slice_info]

        # keep the raw (unrotated) node parts in rot_n_feat_ so linear layers can be applied
        for l, group in self.groups.items():
            if l == 0 or not group:
                continue
            for mul, sl in group:
                rot_n_feat_[:, sl] = node_features[:, sl]

        # apply linear maps (these are node-wise)
        rot_center_node_feat = self.lin_center(rot_n_feat_)
        rot_center_node_feat = rot_center_node_feat[active_edge_index[0]]  # shape: (n_edges, dim)

        rot_neighbor_node_feat = self.lin_neighbor(rot_n_feat_)
        rot_neighbor_node_feat = rot_neighbor_node_feat[active_edge_index[1]]  # shape: (n_edges, dim)

        latent_list = []
        # Now for each l, build per-edge (n_edges, total_mul, 2l+1) and apply per-edge rotation
        for l, group in self.groups.items():
            muls, slices = zip(*group)
            total_mul = sum(m for m, _ in group)
            # center/neighbor parts now have batch = n_edges
            # each part reshape -> (n_edges, mul, 2l+1)
            center_node_parts = [rot_center_node_feat[:, sl].reshape(-1, mul, self.dims[l]) for mul, sl in group]
            center_node_combined = torch.cat(center_node_parts, dim=1)  # (n_edges, total_mul, d)

            neighbor_node_parts = [rot_neighbor_node_feat[:, sl].reshape(-1, mul, self.dims[l]) for mul, sl in group]
            neighbor_node_combined = torch.cat(neighbor_node_parts, dim=1)  # (n_edges, total_mul, d)

            if l == 0:
                # l=0: dims[l] == 1, no rotation needed; keep consistent flow
                # center_node_combined, neighbor_node_combined have shape (e, total_mul, 1)
                center_rot = center_node_combined
                neighbor_rot = neighbor_node_combined
            else:
                # get per-edge rotation matrices: shape (n_edges, 2l+1, 2l+1)
                start = self.offsets[l]
                rot_mat = wigner_D_all[:, start:start + self.dims[l], start:start + self.dims[l]]

                # rotate center & neighbor per-edge:
                # center_combined: (e, m, d), rot_mat: (e, d, d) -> rotated_center: (e, m, d)
                center_rot = torch.einsum('emd,edq->emq', center_node_combined, rot_mat)
                neighbor_rot = torch.einsum('emd,edq->emq', neighbor_node_combined, rot_mat)

            # --- 修改：不对 d 求和，而是保留 (e, m, d)，做 elementwise 相乘后展平为 (e, m*d)
            # elementwise product as similarity per-component
            sim_tensor = center_rot * neighbor_rot  # (e, m, d)
            e = sim_tensor.shape[0]
            sim_flat = sim_tensor.reshape(e, -1)  # (e, m * d)

            # map flattened similarity (m*d) to latent_dim
            sim_mapped = self.sim_linears[f"l{l}"](sim_flat)  # (e, latent_dim)
            latent_list.append(sim_mapped)

        # latent_list: list of (e, latent_dim), one per l
        # stack along new l-dim -> (e, num_l, latent_dim)
        latent_stack = torch.stack(latent_list, dim=1)
        # flatten (e, num_l * latent_dim) and fuse via final_mlp
        e = latent_stack.shape[0]
        fused = latent_stack.reshape(e, -1)
        latent = self.final_mlp(fused)  # (e, latent_dim)

        return latent


# --- 若文件中已有 build_z_rot_multi / batch_wigner_D / wigner_D / _z_rot_mat，保留不变 ---
# 这里假设上面的函数在原文件中已有实现（和你贴出的片段一致）
# 为了简洁，这里不重复这些辅助函数的实现（如果你的文件中没有，请把原实现拷回）

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


def rotate_vector(x, irreps, wigner_D_all, back=False):
    """
    辅助函数：手动旋转向量。
    back=False: Global -> Local (x @ R)
    back=True:  Local -> Global (x @ R.T)
    """
    n, _ = x.shape
    x_out = torch.zeros_like(x)
    irreps = Irreps(irreps).simplify()

    # 预计算 offset
    l_max = max((l for (_, (l, _)), _ in zip(irreps, irreps.slices()) if l > 0), default=0)
    dims = {l: 2 * l + 1 for l in range(l_max + 1)}
    offsets = {}
    offset = 0
    for l in range(l_max + 1):
        offsets[l] = offset
        offset += dims[l]

    groups = defaultdict(list)
    for (mul, (l, p)), slice_info in zip(irreps, irreps.slices()):
        groups[l].append((mul, slice_info))

    # 复制 scalar
    for (mul, (l, p)), slice_info in zip(irreps, irreps.slices()):
        if l == 0:
            x_out[:, slice_info] = x[:, slice_info]

    # 旋转 vector
    for l, group in groups.items():
        if l == 0 or not group:
            continue
        muls, slices = zip(*group)
        x_parts = [x[:, sl].reshape(n, mul, 2 * l + 1) for mul, sl in group]
        x_combined = torch.cat(x_parts, dim=1)  # (N, total_mul, 2l+1)

        start = offsets[l]
        rot_mat = wigner_D_all[:, start:start + dims[l], start:start + dims[l]]

        if not back:
            # Global -> Local
            transformed = torch.bmm(x_combined, rot_mat)
        else:
            # Local -> Global (multiply by transpose)
            transformed = torch.bmm(x_combined, rot_mat.transpose(1, 2))

        for part, slice_info in zip(transformed.split(muls, dim=1), slices):
            x_out[:, slice_info] = part.reshape(n, -1)

    return x_out


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


class SO2_Linear(torch.nn.Module):
    """
    SO(2) Convolutional layer.
    """

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
    ):
        super(SO2_Linear, self).__init__()

        self.irreps_in = Irreps(irreps_in).simplify()
        self.irreps_out = (Irreps(f"{extra_m0_outsize}x0e") + Irreps(irreps_out)).simplify()
        self.radial_emb = radial_emb
        self.latent_dim = latent_dim
        self.m_linear = nn.ModuleList()
        self.so2_m_linear_mode = so2_m_linear_mode or os.environ.get("DPTB_SO2_M_LINEAR_MODE", "standard")
        if self.so2_m_linear_mode == "cublas_grouped":
            self.so2_m_linear_mode = "indexed_sandwich_multi"
        if self.so2_m_linear_mode == "cuda_pack_scatter":
            self.so2_m_linear_mode = "indexed_sandwich_cuda"
        if self.so2_m_linear_mode == "cuda_pack_scatter_multi":
            self.so2_m_linear_mode = "indexed_sandwich_cuda_multi"
        if self.so2_m_linear_mode in ("scheduled_sandwich", "cuda_scheduled_sandwich"):
            self.so2_m_linear_mode = "indexed_sandwich_scheduled"
        if self.so2_m_linear_mode in ("materialized_sandwich", "cuda_materialized_sandwich"):
            self.so2_m_linear_mode = "indexed_sandwich_materialized"
        if self.so2_m_linear_mode in (
            "materialized_cuda_scheduler",
            "materialized_scheduled_sandwich",
            "cuda_materialized_scheduled_sandwich",
        ):
            self.so2_m_linear_mode = "indexed_sandwich_materialized_scheduled"
        if self.so2_m_linear_mode not in (
            "standard",
            "indexed_sandwich_multi",
            "indexed_sandwich_cuda",
            "indexed_sandwich_cuda_multi",
            "indexed_sandwich_scheduled",
            "indexed_sandwich_materialized",
            "indexed_sandwich_materialized_scheduled",
        ):
            raise ValueError(
                "so2_m_linear_mode must be 'standard', 'indexed_sandwich_multi', "
                "'indexed_sandwich_cuda', 'indexed_sandwich_cuda_multi', "
                "'indexed_sandwich_scheduled', 'indexed_sandwich_materialized', "
                "or 'indexed_sandwich_materialized_scheduled', "
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
        self._in_entries_by_m = {
            m: tuple(entry for entry in self._in_entries if entry[0] >= m)
            for m in range(self.irreps_out.lmax + 1)
        }
        self._out_entries_by_m = {
            m: tuple(entry for entry in self._out_entries if entry[0] >= m)
            for m in range(self.irreps_out.lmax + 1)
        }

    def forward(self, x, R, latents=None, wigner_D_all=None):
        weights = self.radial_emb(latents) if self.radial_emb else None

        # 旋转逻辑：需要旋转才计算 Wigner D (或者如果外部传进来了就用)
        if wigner_D_all is None:
            # 只有当需要 rotate_in 或者 rotate_out 时才必须计算 D
            if (self.rotate_in or self.rotate_out) and self.l_max > 0:
                angle = xyz_to_angles(R[:, [1, 2, 0]])
                wigner_D_all = batch_wigner_D(self.l_max, angle[0], angle[1], torch.zeros_like(angle[0]), _Jd)

        if self._use_indexed_sandwich_materialized_path(x):
            from dptb.nn.so2_materialized_sandwich import try_forward_so2_materialized_sandwich

            result = try_forward_so2_materialized_sandwich(self, x, weights, wigner_D_all)
            if result is not None:
                return result

        if self._use_indexed_sandwich_materialized_scheduled_path(x):
            from dptb.nn.so2_scheduled_sandwich import try_forward_so2_materialized_scheduled_sandwich

            result = try_forward_so2_materialized_scheduled_sandwich(self, x, weights, wigner_D_all)
            if result is not None:
                return result

        if self._use_indexed_sandwich_scheduled_path(x):
            from dptb.nn.so2_scheduled_sandwich import try_forward_so2_scheduled_sandwich

            result = try_forward_so2_scheduled_sandwich(self, x, weights, wigner_D_all)
            if result is not None:
                return result

        if self._use_indexed_sandwich_cuda_path(x):
            if self.so2_m_linear_mode == "indexed_sandwich_cuda_multi":
                result = self._forward_indexed_sandwich_cuda_multi(x, weights, wigner_D_all)
            else:
                result = self._forward_indexed_sandwich_cuda(x, weights, wigner_D_all)
            if result is not None:
                return result

        if self._use_indexed_sandwich_multi_path(x):
            return self._forward_indexed_sandwich_multi(x, weights, wigner_D_all)

        return self._forward_standard(x, weights, wigner_D_all)

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
            rot_mat = wigner_D_all[:, start:start + self.dims[l], start:start + self.dims[l]]
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
                rot_mat = wigner_D_all[:, start:start + self.dims[l], start:start + self.dims[l]]
                x_slice = out[:, slice_in].reshape(n, mul, -1)
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

    def _use_indexed_sandwich_multi_path(self, x):
        if self.so2_m_linear_mode != "indexed_sandwich_multi":
            return False
        if x.device.type != "cuda" or x.dtype != torch.float32:
            return False
        if self.irreps_out.lmax < 1:
            return False
        return all(isinstance(module.fc, nn.Linear) for module in self.m_linear)

    @staticmethod
    def _int_env(name, default):
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _int_env_any(names, default):
        for name in names:
            value = os.environ.get(name)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)
        return int(default)

    @staticmethod
    def _str_env_any(names, default):
        for name in names:
            value = os.environ.get(name)
            if value is not None:
                return str(value)
        return str(default)

    def _indexed_sandwich_cuda_multi_epilogue_schedule(self):
        return self._str_env_any(
            (
                "DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_EPILOGUE_SCHEDULE",
                "SO2_CUDA_EPILOGUE_SCHEDULE",
            ),
            "output_major",
        ).lower()

    def _indexed_sandwich_cuda_multi_gemm_layout(self):
        value = self._str_env_any(
            (
                "DPTB_SO2_INDEXED_SANDWICH_CUDA_MULTI_GEMM_LAYOUT",
                "SO2_CUDA_GEMM_LAYOUT",
                "SO2_CUDA_GEMM_STRATEGY",
            ),
            "raw",
        ).lower()
        aliases = {
            "raw_output_major_v2_cached": "raw_cached",
            "output_major_v2_cached": "raw_cached",
            "v2_cached": "raw_cached",
            "cached_raw": "raw_cached",
            "raw_output_major_v2_grouped": "grouped_raw_v2",
            "output_major_v2_grouped": "grouped_raw_v2",
            "v2_grouped": "grouped_raw_v2",
            "grouped_v2": "grouped_raw_v2",
            "raw_output_major_v3_pack": "raw_pack_v2",
            "raw_output_major_v3_pack_v2": "raw_pack_v2",
            "pack_v2": "raw_pack_v2",
            "desc_pack": "raw_pack_v2",
            "raw_output_major_v3_pack_m0_cuda": "raw_pack_v2_m0_cuda",
            "raw_output_major_v3_pack_v2_m0_cuda": "raw_pack_v2_m0_cuda",
            "pack_v2_m0_cuda": "raw_pack_v2_m0_cuda",
            "m0_cuda_pack_v2": "raw_pack_v2_m0_cuda",
        }
        return aliases.get(value, value)

    def _indexed_sandwich_cuda_multi_execution_tag(self):
        layout = self._indexed_sandwich_cuda_multi_gemm_layout()
        if layout == "raw_cached":
            return "raw_output_major_v2_cached"
        if layout == "grouped_raw_v2":
            return "raw_output_major_v2_grouped"
        if layout == "raw_pack_v2":
            return "raw_output_major_v3_pack_v2"
        if layout == "raw_pack_v2_m0_cuda":
            return "raw_output_major_v3_pack_v2_m0_cuda"
        if layout == "raw":
            return "raw_output_major_v1"
        return layout

    def _use_indexed_sandwich_cuda_path(self, x):
        if self.so2_m_linear_mode not in ("indexed_sandwich_cuda", "indexed_sandwich_cuda_multi"):
            return False
        if x.device.type != "cuda" or x.dtype != torch.float32:
            return False
        if self.irreps_out.lmax < 1:
            return False
        min_edges = self._int_env_any(
            ("DPTB_SO2_INDEXED_SANDWICH_CUDA_MIN_EDGES", "SO2_CUDA_MIN_EDGES"), 0
        )
        max_edges = self._int_env_any(
            ("DPTB_SO2_INDEXED_SANDWICH_CUDA_MAX_EDGES", "SO2_CUDA_MAX_EDGES"), 0
        )
        if min_edges > 0 and int(x.shape[0]) < min_edges:
            return False
        if max_edges > 0 and int(x.shape[0]) > max_edges:
            return False
        return all(isinstance(module.fc, nn.Linear) for module in self.m_linear)

    def _use_indexed_sandwich_scheduled_path(self, x):
        if self.so2_m_linear_mode != "indexed_sandwich_scheduled":
            return False
        if x.device.type != "cuda" or x.dtype != torch.float32:
            return False
        if self.irreps_out.lmax < 1:
            return False
        min_edges = self._int_env("DPTB_SO2_SCHEDULED_SANDWICH_MIN_EDGES", 0)
        max_edges = self._int_env("DPTB_SO2_SCHEDULED_SANDWICH_MAX_EDGES", 0)
        if min_edges > 0 and int(x.shape[0]) < min_edges:
            return False
        if max_edges > 0 and int(x.shape[0]) > max_edges:
            return False
        return all(isinstance(module.fc, nn.Linear) for module in self.m_linear)

    def _use_indexed_sandwich_materialized_path(self, x):
        if self.so2_m_linear_mode != "indexed_sandwich_materialized":
            return False
        if x.device.type != "cuda" or x.dtype != torch.float32:
            return False
        if self.irreps_out.lmax < 1:
            return False
        if self.radial_emb and not bool(self.front):
            return False
        min_edges = self._int_env_any(
            ("DPTB_SO2_MATERIALIZED_MIN_EDGES", "SO2_CUDA_MATERIALIZED_MIN_EDGES"), 0
        )
        max_edges = self._int_env_any(
            ("DPTB_SO2_MATERIALIZED_MAX_EDGES", "SO2_CUDA_MATERIALIZED_MAX_EDGES"), 0
        )
        if min_edges > 0 and int(x.shape[0]) < min_edges:
            return False
        if max_edges > 0 and int(x.shape[0]) > max_edges:
            return False
        return all(isinstance(module.fc, nn.Linear) for module in self.m_linear)

    def _use_indexed_sandwich_materialized_scheduled_path(self, x):
        if self.so2_m_linear_mode != "indexed_sandwich_materialized_scheduled":
            return False
        if x.device.type != "cuda" or x.dtype != torch.float32:
            return False
        if self.irreps_out.lmax < 1:
            return False
        min_edges = self._int_env_any(
            (
                "DPTB_SO2_MATERIALIZED_SCHEDULED_MIN_EDGES",
                "SO2_CUDA_MATERIALIZED_MIN_EDGES",
            ),
            0,
        )
        max_edges = self._int_env_any(
            (
                "DPTB_SO2_MATERIALIZED_SCHEDULED_MAX_EDGES",
                "SO2_CUDA_MATERIALIZED_MAX_EDGES",
            ),
            0,
        )
        if min_edges > 0 and int(x.shape[0]) < min_edges:
            return False
        if max_edges > 0 and int(x.shape[0]) > max_edges:
            return False
        return all(isinstance(module.fc, nn.Linear) for module in self.m_linear)

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

    def _pack_group_pair(self, x_group, l, m, rot_block):
        if x_group.numel() == 0:
            return x_group.new_empty((x_group.shape[0], 2, x_group.shape[1]))
        rows = [l - m, l + m]
        if not self.rotate_in or rot_block is None:
            return x_group[:, :, rows].transpose(1, 2).contiguous()
        return torch.einsum("ncd,ndp->npc", x_group, rot_block[:, :, rows])

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
            for l, mul, _slice_info, group_start in self._in_entries_by_m[0]
        ]
        if not parts:
            return x_template.new_empty((n, 0))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1)

    def _assemble_grouped_pair_input(self, input_groups, rot_blocks, m, n, x_template):
        packed_by_l = {}
        for l, x_group in input_groups.items():
            if l >= m:
                packed_by_l[l] = self._pack_group_pair(x_group, l, m, rot_blocks.get(l))
        parts = [
            packed_by_l[l][:, :, group_start:group_start + mul]
            for l, mul, _slice_info, group_start in self._in_entries_by_m[m]
            if l in packed_by_l
        ]
        if not parts:
            return x_template.new_empty((n, 2, 0))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=2)

    def _accumulate_group_m0_(self, out_group, y_group, l, rot_block):
        if y_group.numel() == 0:
            return
        if l == 0 or not self.rotate_out or rot_block is None:
            out_group[:, :, l] += y_group
            return
        out_group += y_group.unsqueeze(-1) * rot_block[:, :, l].unsqueeze(1)

    def _accumulate_group_pair_(self, out_group, y_group, l, m, rot_block):
        if y_group.numel() == 0:
            return
        rows = [l - m, l + m]
        if not self.rotate_out or rot_block is None:
            out_group[:, :, rows] += y_group.transpose(1, 2)
            return
        out_group += torch.einsum("npc,ndp->ncd", y_group, rot_block[:, :, rows])

    def _accumulate_grouped_m0_output_(self, out_groups, y_m0, rot_blocks):
        cursor = 0
        for l, mul, _slice_info, group_start in self._out_entries_by_m[0]:
            y_entry = y_m0[:, cursor:cursor + mul]
            cursor += mul
            out_view = out_groups[l][:, group_start:group_start + mul, :]
            self._accumulate_group_m0_(out_view, y_entry, l, rot_blocks.get(l))

    def _accumulate_grouped_pair_output_(self, out_groups, y_m, rot_blocks, m):
        cursor = 0
        for l, mul, _slice_info, group_start in self._out_entries_by_m[m]:
            y_entry = y_m[:, :, cursor:cursor + mul]
            cursor += mul
            out_view = out_groups[l][:, group_start:group_start + mul, :]
            self._accumulate_group_pair_(out_view, y_entry, l, m, rot_blocks.get(l))

    def _materialize_output_l_groups(self, out_groups, *, n, dtype, device):
        out = torch.zeros((n, self.irreps_out.dim), dtype=dtype, device=device)
        for l, mul, slice_info, group_start in self._out_entries:
            group_view = out_groups[l][:, group_start:group_start + mul, :]
            out[:, slice_info] = group_view.reshape(n, -1)
        return out

    def _apply_indexed_sandwich_multi_gemm(self, x_inputs):
        from dptb.nn.cuda_ops.grouped_gemm import indexed_sandwich_multi_gemm

        weights = [module.fc.weight.unsqueeze(0).contiguous() for module in self.m_linear[:len(x_inputs)]]
        ptrs = [
            torch.tensor([0, x_m.reshape(-1, x_m.shape[-1]).shape[0]], dtype=torch.long, device="cpu")
            for x_m in x_inputs
        ]
        return indexed_sandwich_multi_gemm(x_inputs, ptrs, weights)

    def _indexed_sandwich_cuda_pair_maps(self, m, device):
        from dptb.nn.so2_sandwich_common import so2_pair_maps

        return so2_pair_maps(self, m, device, cache_attr="_indexed_sandwich_cuda_pair_maps_cache")

    def _indexed_sandwich_cuda_multi_metadata(self, device):
        cache = getattr(self, "_indexed_sandwich_cuda_multi_metadata_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_indexed_sandwich_cuda_multi_metadata_cache", cache)

        key = str(device)
        cached = cache.get(key)
        if cached is not None:
            m_values_host, _, _, _, _, cin_values, cout_values, _, _, _, _, _, _ = cached
            for i, m in enumerate(m_values_host):
                fc = getattr(self.m_linear[m - 1], "fc", None)
                if (
                    not isinstance(fc, nn.Linear)
                    or fc.in_features != cin_values[i]
                    or fc.out_features != 2 * cout_values[i]
                ):
                    return None
            return cached

        in_bases = []
        in_ls = []
        out_bases = []
        out_ls = []
        cin_values = []
        cout_values = []
        m_values_host = []
        offsets = None
        cin_prefix = [0]
        cout_prefix = [0]
        for m, module in zip(range(1, self.irreps_out.lmax + 1), self.m_linear):
            fc = getattr(module, "fc", None)
            if not isinstance(fc, nn.Linear):
                return None
            in_base, in_l, out_base, out_l, offsets_m = self._indexed_sandwich_cuda_pair_maps(m, device)
            cin = int(in_base.numel())
            cout = int(out_base.numel())
            if cin == 0 or cout == 0:
                continue
            if fc.in_features != cin or fc.out_features != 2 * cout:
                return None
            offsets = offsets_m
            in_bases.append(in_base)
            in_ls.append(in_l)
            out_bases.append(out_base)
            out_ls.append(out_l)
            cin_values.append(cin)
            cout_values.append(cout)
            m_values_host.append(int(m))
            cin_prefix.append(cin_prefix[-1] + cin)
            cout_prefix.append(cout_prefix[-1] + cout)

        cached = (
            tuple(m_values_host),
            tuple(in_bases),
            tuple(in_ls),
            tuple(out_bases),
            tuple(out_ls),
            tuple(cin_values),
            tuple(cout_values),
            torch.tensor(cin_prefix, dtype=torch.long, device=device).contiguous(),
            torch.tensor(cout_prefix, dtype=torch.long, device=device).contiguous(),
            torch.tensor(m_values_host, dtype=torch.long, device=device).contiguous(),
            offsets,
            tuple(int(v) for v in cin_prefix),
            tuple(int(v) for v in cout_prefix),
        )
        cache[key] = cached
        return cached

    def _indexed_sandwich_cuda_multi_output_entry_map(
        self,
        m_values_host,
        out_bases,
        out_ls,
        device,
        entry_map_fn,
    ):
        cache = getattr(self, "_indexed_sandwich_cuda_multi_output_entry_map_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_indexed_sandwich_cuda_multi_output_entry_map_cache", cache)

        key = (str(device), int(self.irreps_out.dim), tuple(int(m) for m in m_values_host))
        cached = cache.get(key)
        if cached is None:
            cached = entry_map_fn(
                self,
                list(m_values_host),
                list(out_bases),
                list(out_ls),
                int(self.irreps_out.dim),
                device,
            )
            cache[key] = cached
        return cached

    def _indexed_sandwich_cuda_multi_call_plan(
        self,
        metadata,
        device,
        entry_map_fn,
        epilogue_schedule,
    ):
        cache = getattr(self, "_indexed_sandwich_cuda_multi_call_plan_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_indexed_sandwich_cuda_multi_call_plan_cache", cache)

        (
            m_values_host,
            in_bases,
            in_ls,
            out_bases,
            out_ls,
            _cin_values,
            _cout_values,
            cin_prefix_t,
            cout_prefix_t,
            m_values_t,
            offsets,
            cin_prefix,
            cout_prefix,
        ) = metadata
        key = (
            str(device),
            int(self.irreps_out.dim),
            tuple(int(v) for v in m_values_host),
            tuple(int(v) for v in cin_prefix),
            tuple(int(v) for v in cout_prefix),
            tuple(
                id(getattr(self.m_linear[int(m) - 1], "fc", None))
                for m in m_values_host
            ),
            str(epilogue_schedule),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached

        plan = {
            "m_values_host": tuple(int(v) for v in m_values_host),
            "in_bases": tuple(in_bases),
            "in_ls": tuple(in_ls),
            "out_bases": tuple(out_bases),
            "out_ls": tuple(out_ls),
            "m_modules": tuple(self.m_linear[int(m) - 1] for m in m_values_host),
            "cin_prefix_t": cin_prefix_t,
            "cout_prefix_t": cout_prefix_t,
            "m_values_t": m_values_t,
            "offsets": offsets,
            "cin_prefix": tuple(int(v) for v in cin_prefix),
            "cout_prefix": tuple(int(v) for v in cout_prefix),
            "raw_row_ptr_by_n": {},
            "output_entry_map": None,
        }
        total_cin = int(plan["cin_prefix"][-1])
        pack_desc = torch.empty((total_cin, 3), dtype=torch.long, device=device)
        cursor = 0
        for m, in_base, in_l in zip(plan["m_values_host"], plan["in_bases"], plan["in_ls"]):
            end = cursor + int(in_base.numel())
            pack_desc[cursor:end, 0].copy_(in_base)
            pack_desc[cursor:end, 1].copy_(in_l)
            pack_desc[cursor:end, 2].fill_(int(m))
            cursor = end
        plan["pack_desc"] = pack_desc.contiguous()
        plan["in_base_all"] = torch.cat(tuple(t.contiguous() for t in plan["in_bases"]), dim=0).contiguous()
        plan["in_l_all"] = torch.cat(tuple(t.contiguous() for t in plan["in_ls"]), dim=0).contiguous()
        plan["m0_maps"] = self._indexed_sandwich_cuda_pair_maps(0, device)
        if epilogue_schedule == "output_major":
            plan["output_entry_map"] = self._indexed_sandwich_cuda_multi_output_entry_map(
                plan["m_values_host"],
                plan["out_bases"],
                plan["out_ls"],
                device,
                entry_map_fn,
            )
        cache[key] = plan
        return plan

    @staticmethod
    def _indexed_sandwich_cuda_multi_raw_row_ptr(plan, n, rows_per_edge):
        row_count = int(n) * int(rows_per_edge)
        cache = plan.get("raw_row_ptr_by_n")
        if cache is None:
            cache = {}
            plan["raw_row_ptr_by_n"] = cache
        cached = cache.get(row_count)
        if cached is None:
            cached = torch.tensor([0, row_count], dtype=torch.long, device="cpu")
            if len(cache) >= 16:
                cache.pop(next(iter(cache)))
            cache[row_count] = cached
        return cached

    def _forward_indexed_sandwich_cuda(self, x, weights, wigner_D_all):
        strict = os.environ.get("DPTB_SO2_INDEXED_SANDWICH_CUDA_STRICT", "0").lower() in ("1", "true", "yes", "on")
        try:
            from dptb.nn.so2_moe_fused_p0 import (
                _PackPairFunction,
                _ScatterPairOutputFunction,
                _ScatterRawPairOutputFunction,
                _wigner_tensor_and_mode,
            )

            wigner_info = _wigner_tensor_and_mode(self, wigner_D_all, x)
            if wigner_info is None:
                return None
            wigner, compact_offsets, wigner_mode, wigner_stride = wigner_info

            n, _ = x.shape
            rot_blocks = {
                l: self._select_wigner_block(wigner_D_all, l)
                for l in range(self.l_max + 1)
            }
            input_groups = {
                l: self._gather_l_group(x, l)
                for l in self._in_groups
            }
            out_groups = self._alloc_output_l_groups(n, dtype=x.dtype, device=x.device)

            radial_weight = weights[:, self.m_in_index[0]:self.m_in_index[1]].unsqueeze(1) if self.radial_emb else 1.
            inp = self._assemble_grouped_m0_input(input_groups, rot_blocks, n, x)
            if self.front and self.radial_emb:
                y_m0 = self.fc_m0(inp * radial_weight.squeeze(1))
            elif self.radial_emb:
                y_m0 = self.fc_m0(inp) * radial_weight.squeeze(1)
            else:
                y_m0 = self.fc_m0(inp)
            self._accumulate_grouped_m0_output_(out_groups, y_m0, rot_blocks)
            out = self._materialize_output_l_groups(out_groups, n=n, dtype=x.dtype, device=x.device)

            for m, module in zip(range(1, self.irreps_out.lmax + 1), self.m_linear):
                in_base, in_l, out_base, out_l, offsets = self._indexed_sandwich_cuda_pair_maps(m, x.device)
                cin = int(in_base.numel())
                cout = int(out_base.numel())
                if cin == 0 or cout == 0 or module.fc.in_features != cin or module.fc.out_features != 2 * cout:
                    return None

                pair = _PackPairFunction.apply(
                    x.contiguous(),
                    wigner,
                    in_base,
                    in_l,
                    offsets,
                    compact_offsets,
                    int(m),
                    bool(self.rotate_in),
                    int(wigner_mode),
                    int(wigner_stride),
                )

                radial = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(1) if self.radial_emb else None
                pair_for_linear = pair * radial if radial is not None and bool(self.front) else pair
                raw = module.fc(pair_for_linear)
                if radial is None or bool(self.front):
                    contribution = _ScatterRawPairOutputFunction.apply(
                        raw.contiguous(),
                        wigner,
                        out_base,
                        out_l,
                        offsets,
                        compact_offsets,
                        int(self.irreps_out.dim),
                        int(m),
                        bool(self.rotate_out),
                        int(wigner_mode),
                        int(wigner_stride),
                    )
                else:
                    pair_out = module._finish_linear_output(raw) * radial
                    contribution = _ScatterPairOutputFunction.apply(
                        pair_out.contiguous(),
                        wigner,
                        out_base,
                        out_l,
                        offsets,
                        compact_offsets,
                        int(self.irreps_out.dim),
                        int(m),
                        bool(self.rotate_out),
                        int(wigner_mode),
                        int(wigner_stride),
                    )
                out = out + contribution
            return out.contiguous(), wigner_D_all
        except Exception:
            if strict:
                raise
            return None

    def _forward_indexed_sandwich_cuda_multi(self, x, weights, wigner_D_all):
        strict = os.environ.get("DPTB_SO2_INDEXED_SANDWICH_CUDA_STRICT", "0").lower() in ("1", "true", "yes", "on")
        try:
            from so2_cuda_ops.profiler import (
                cuda_span_end,
                cuda_span_start,
                maybe_print_profile,
                record_cuda_span,
                record_host_span,
            )
        except Exception:
            cuda_span_start = lambda *_args, **_kwargs: None
            cuda_span_end = lambda *_args, **_kwargs: None
            maybe_print_profile = lambda *_args, **_kwargs: None
            record_cuda_span = lambda _label, _device, fn: fn()
            record_host_span = lambda _label, fn: fn()

        profile_token = cuda_span_start("so2.forward_total", x)
        profile_done = False

        def _finish_profile(value):
            nonlocal profile_done
            if not profile_done:
                cuda_span_end(profile_token)
                maybe_print_profile(self._indexed_sandwich_cuda_multi_execution_tag())
                profile_done = True
            return value

        try:
            from dptb.nn.so2_moe_fused_p0 import (
                _PackM0Function,
                _PackPairsMultiFunction,
                _PackPairsMultiDescFunction,
                _ScatterRawPairOutputFunction,
                _ScatterM0OutputFunction,
                _ScatterPairOutputFunction,
                _ScatterPairsMultiOutputMajorFunction,
                _ScatterRawPairsMultiOutputFunction,
                _ScatterRawPairsMultiOutputMajorFunction,
                _multi_output_entry_map,
                _wigner_tensor_and_mode,
            )

            if self.radial_emb and not bool(self.front):
                return _finish_profile(self._forward_indexed_sandwich_cuda(x, weights, wigner_D_all))

            wigner_info = _wigner_tensor_and_mode(self, wigner_D_all, x)
            if wigner_info is None:
                return _finish_profile(None)
            wigner, compact_offsets, wigner_mode, wigner_stride = wigner_info

            n, _ = x.shape
            gemm_layout = self._indexed_sandwich_cuda_multi_gemm_layout()

            def _run_m0_path():
                radial_weight = weights[:, self.m_in_index[0]:self.m_in_index[1]].unsqueeze(1) if self.radial_emb else None
                if gemm_layout == "raw_pack_v2_m0_cuda":
                    in_base, in_l, out_base, out_l, m0_offsets = self._indexed_sandwich_cuda_pair_maps(0, x.device)
                    if (
                        isinstance(self.fc_m0, nn.Linear)
                        and int(in_base.numel()) == int(self.fc_m0.in_features)
                        and int(out_base.numel()) == int(self.fc_m0.out_features)
                    ):
                        inp = _PackM0Function.apply(
                            x.contiguous(),
                            wigner,
                            in_base,
                            in_l,
                            m0_offsets,
                            compact_offsets,
                            bool(self.rotate_in),
                            int(wigner_mode),
                            int(wigner_stride),
                        )
                        radial_flat = radial_weight.squeeze(1) if radial_weight is not None else None
                        if self.front and self.radial_emb:
                            y_m0 = self.fc_m0(inp * radial_flat)
                        elif self.radial_emb:
                            y_m0 = self.fc_m0(inp) * radial_flat
                        else:
                            y_m0 = self.fc_m0(inp)
                        return _ScatterM0OutputFunction.apply(
                            y_m0.contiguous(),
                            wigner,
                            out_base,
                            out_l,
                            m0_offsets,
                            compact_offsets,
                            int(self.irreps_out.dim),
                            bool(self.rotate_out),
                            int(wigner_mode),
                            int(wigner_stride),
                        )

                rot_blocks = {
                    l: self._select_wigner_block(wigner_D_all, l)
                    for l in range(self.l_max + 1)
                }
                input_groups = {
                    l: self._gather_l_group(x, l)
                    for l in self._in_groups
                }
                out_groups = self._alloc_output_l_groups(n, dtype=x.dtype, device=x.device)
                radial_weight = radial_weight if radial_weight is not None else 1.
                inp = self._assemble_grouped_m0_input(input_groups, rot_blocks, n, x)
                if self.front and self.radial_emb:
                    y_m0 = self.fc_m0(inp * radial_weight.squeeze(1))
                elif self.radial_emb:
                    y_m0 = self.fc_m0(inp) * radial_weight.squeeze(1)
                else:
                    y_m0 = self.fc_m0(inp)
                self._accumulate_grouped_m0_output_(out_groups, y_m0, rot_blocks)
                return self._materialize_output_l_groups(out_groups, n=n, dtype=x.dtype, device=x.device)

            out = record_cuda_span("so2.forward.m0_path", x, _run_m0_path)

            metadata = record_host_span(
                "so2.host.metadata",
                lambda: self._indexed_sandwich_cuda_multi_metadata(x.device),
            )
            if metadata is None:
                return _finish_profile(None)
            (
                m_values_host,
                in_bases,
                in_ls,
                out_bases,
                out_ls,
                _cin_values,
                _cout_values,
                cin_prefix_t,
                cout_prefix_t,
                m_values_t,
                offsets,
                cin_prefix,
                _cout_prefix,
            ) = metadata

            if not m_values_host:
                return _finish_profile((out.contiguous(), wigner_D_all))
            epilogue_schedule = self._indexed_sandwich_cuda_multi_epilogue_schedule()
            use_call_plan = gemm_layout in ("raw_cached", "grouped_raw_v2", "raw_pack_v2", "raw_pack_v2_m0_cuda")
            call_plan = None
            if use_call_plan:
                call_plan = record_host_span(
                    "so2.host.call_plan",
                    lambda: self._indexed_sandwich_cuda_multi_call_plan(
                        metadata,
                        x.device,
                        _multi_output_entry_map,
                        epilogue_schedule,
                    ),
                )
                m_values_host = call_plan["m_values_host"]
                in_bases = call_plan["in_bases"]
                in_ls = call_plan["in_ls"]
                out_bases = call_plan["out_bases"]
                out_ls = call_plan["out_ls"]
                m_modules = call_plan["m_modules"]
                cin_prefix_t = call_plan["cin_prefix_t"]
                cout_prefix_t = call_plan["cout_prefix_t"]
                m_values_t = call_plan["m_values_t"]
                offsets = call_plan["offsets"]
                cin_prefix = call_plan["cin_prefix"]
                cout_prefix = call_plan["cout_prefix"]
                pack_desc = call_plan.get("pack_desc")
                in_base_all = call_plan.get("in_base_all")
                in_l_all = call_plan.get("in_l_all")
            else:
                m_values_host = list(m_values_host)
                in_bases = list(in_bases)
                in_ls = list(in_ls)
                out_bases = list(out_bases)
                out_ls = list(out_ls)
                m_modules = [self.m_linear[m - 1] for m in m_values_host]
                pack_desc = None
                in_base_all = None
                in_l_all = None
            if gemm_layout in ("raw_pack_v2", "raw_pack_v2_m0_cuda") and pack_desc is not None:
                packed_all = _PackPairsMultiDescFunction.apply(
                    x.contiguous(),
                    wigner,
                    pack_desc,
                    in_base_all,
                    in_l_all,
                    offsets,
                    compact_offsets,
                    cin_prefix_t,
                    m_values_t,
                    bool(self.rotate_in),
                    int(wigner_mode),
                    int(wigner_stride),
                )
            else:
                packed_all = _PackPairsMultiFunction.apply(
                    x.contiguous(),
                    wigner,
                    in_bases,
                    in_ls,
                    offsets,
                    compact_offsets,
                    cin_prefix_t,
                    m_values_t,
                    bool(self.rotate_in),
                    int(wigner_mode),
                    int(wigner_stride),
                )

            if gemm_layout in (
                "block",
                "block_complex",
                "fairchem_block",
                "block_direct",
                "direct_block_complex",
                "compact_block",
            ):
                from dptb.nn.cuda_ops.grouped_gemm import (
                    indexed_sandwich_multi_block_direct_gemm,
                    indexed_sandwich_multi_block_gemm,
                )

                pair_inputs = []
                post_radials = []
                block_weights = []
                ptrs = []
                for i, (m, module) in enumerate(zip(m_values_host, m_modules)):
                    pair = packed_all[:, :, cin_prefix[i]:cin_prefix[i + 1]]
                    radial = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(1) if self.radial_emb else None
                    if radial is not None and bool(self.front):
                        pair = pair * radial
                        post_radials.append(None)
                    else:
                        post_radials.append(radial)
                    pair_inputs.append(pair)
                    block_weights.append(module.fc.weight.unsqueeze(0).contiguous())
                    ptrs.append(torch.tensor([0, int(n)], dtype=torch.long, device="cpu"))

                if gemm_layout in ("block_direct", "direct_block_complex", "compact_block"):
                    pair_outputs = indexed_sandwich_multi_block_direct_gemm(pair_inputs, block_weights)
                else:
                    pair_outputs = indexed_sandwich_multi_block_gemm(pair_inputs, ptrs, block_weights)
                for i, radial in enumerate(post_radials):
                    if radial is not None:
                        pair_outputs[i] = pair_outputs[i] * radial

                if epilogue_schedule == "output_major":
                    entry_offsets, entry_m, entry_channel, entry_d, entry_l = (
                        self._indexed_sandwich_cuda_multi_output_entry_map(
                            m_values_host,
                            out_bases,
                            out_ls,
                            x.device,
                            _multi_output_entry_map,
                        )
                    )
                    contribution = _ScatterPairsMultiOutputMajorFunction.apply(
                        wigner,
                        offsets,
                        compact_offsets,
                        cout_prefix_t,
                        m_values_t,
                        entry_offsets,
                        entry_m,
                        entry_channel,
                        entry_d,
                        entry_l,
                        int(self.irreps_out.dim),
                        bool(self.rotate_out),
                        int(wigner_mode),
                        int(wigner_stride),
                        len(pair_outputs),
                        *pair_outputs,
                        *out_bases,
                        *out_ls,
                    )
                else:
                    contribution = None
                    for pair_out, m, out_base, out_l in zip(pair_outputs, m_values_host, out_bases, out_ls):
                        part = _ScatterPairOutputFunction.apply(
                            pair_out.contiguous(),
                            wigner,
                            out_base,
                            out_l,
                            offsets,
                            compact_offsets,
                            int(self.irreps_out.dim),
                            int(m),
                            bool(self.rotate_out),
                            int(wigner_mode),
                            int(wigner_stride),
                        )
                        contribution = part if contribution is None else contribution + part
                return _finish_profile(((out + contribution).contiguous(), wigner_D_all))

            raw_tensors = []
            if gemm_layout in ("grouped_raw", "grouped_raw_v2", "cublas_grouped", "grouped_gemm"):
                from dptb.nn.cuda_ops.grouped_gemm import indexed_sandwich_multi_gemm

                pair_inputs = []
                raw_weights = []
                if gemm_layout == "grouped_raw_v2" and call_plan is not None:
                    ptr = self._indexed_sandwich_cuda_multi_raw_row_ptr(call_plan, n, rows_per_edge=2)
                else:
                    ptr = torch.tensor([0, int(2 * n)], dtype=torch.long, device="cpu")
                for i, (m, module) in enumerate(zip(m_values_host, m_modules)):
                    pair = packed_all[:, :, cin_prefix[i]:cin_prefix[i + 1]]
                    radial = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(1) if self.radial_emb else None
                    pair_inputs.append(pair * radial if radial is not None and bool(self.front) else pair)
                    raw_weights.append(module.fc.weight.unsqueeze(0).contiguous())
                raw_tensors = record_cuda_span(
                    "so2.forward.grouped_raw_linear",
                    packed_all,
                    lambda: indexed_sandwich_multi_gemm(pair_inputs, ptr, raw_weights),
                )
            else:
                for i, (m, module) in enumerate(zip(m_values_host, m_modules)):
                    pair = packed_all[:, :, cin_prefix[i]:cin_prefix[i + 1]]
                    radial = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(1) if self.radial_emb else None
                    pair_for_linear = pair * radial if radial is not None and bool(self.front) else pair
                    raw_tensors.append(
                        record_cuda_span(
                            "so2.forward.raw_linear_per_m",
                            pair_for_linear,
                            lambda: module.fc(pair_for_linear).contiguous(),
                        )
                    )

            if epilogue_schedule == "output_major":
                if call_plan is not None and call_plan.get("output_entry_map") is not None:
                    entry_offsets, entry_m, entry_channel, entry_d, entry_l = call_plan["output_entry_map"]
                else:
                    entry_offsets, entry_m, entry_channel, entry_d, entry_l = record_host_span(
                        "so2.host.output_entry_map",
                        lambda: self._indexed_sandwich_cuda_multi_output_entry_map(
                            m_values_host,
                            out_bases,
                            out_ls,
                            x.device,
                            _multi_output_entry_map,
                        )
                    )
                contribution = _ScatterRawPairsMultiOutputMajorFunction.apply(
                    wigner,
                    offsets,
                    compact_offsets,
                    cout_prefix_t,
                    m_values_t,
                    entry_offsets,
                    entry_m,
                    entry_channel,
                    entry_d,
                    entry_l,
                    int(self.irreps_out.dim),
                    bool(self.rotate_out),
                    int(wigner_mode),
                    int(wigner_stride),
                    len(raw_tensors),
                    *raw_tensors,
                    *out_bases,
                    *out_ls,
                )
            elif epilogue_schedule == "per_m":
                contribution = None
                for raw, m, out_base, out_l in zip(raw_tensors, m_values_host, out_bases, out_ls):
                    part = _ScatterRawPairOutputFunction.apply(
                        raw,
                        wigner,
                        out_base,
                        out_l,
                        offsets,
                        compact_offsets,
                        int(self.irreps_out.dim),
                        int(m),
                        bool(self.rotate_out),
                        int(wigner_mode),
                        int(wigner_stride),
                    )
                    contribution = part if contribution is None else contribution + part
            else:
                contribution = _ScatterRawPairsMultiOutputFunction.apply(
                    wigner,
                    offsets,
                    compact_offsets,
                    cout_prefix_t,
                    m_values_t,
                    int(self.irreps_out.dim),
                    bool(self.rotate_out),
                    int(wigner_mode),
                    int(wigner_stride),
                    len(raw_tensors),
                    *raw_tensors,
                    *out_bases,
                    *out_ls,
                )
            return _finish_profile(((out + contribution).contiguous(), wigner_D_all))
        except Exception:
            _finish_profile(None)
            if strict:
                raise
            return None

    def _forward_indexed_sandwich_multi(self, x, weights, wigner_D_all):
        n, _ = x.shape
        rot_blocks = {
            l: self._select_wigner_block(wigner_D_all, l)
            for l in range(self.l_max + 1)
        }
        input_groups = {
            l: self._gather_l_group(x, l)
            for l in self._in_groups
        }
        out_groups = self._alloc_output_l_groups(n, dtype=x.dtype, device=x.device)

        radial_weight = weights[:, self.m_in_index[0]:self.m_in_index[1]].unsqueeze(1) if self.radial_emb else 1.
        inp = self._assemble_grouped_m0_input(input_groups, rot_blocks, n, x)
        if self.front and self.radial_emb:
            y_m0 = self.fc_m0(inp * radial_weight.squeeze(1))
        elif self.radial_emb:
            y_m0 = self.fc_m0(inp) * radial_weight.squeeze(1)
        else:
            y_m0 = self.fc_m0(inp)
        self._accumulate_grouped_m0_output_(out_groups, y_m0, rot_blocks)

        x_inputs = []
        post_radial_weights = []
        for m in range(1, self.irreps_out.lmax + 1):
            radial_weight = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(1) if self.radial_emb else None
            x_m_in = self._assemble_grouped_pair_input(input_groups, rot_blocks, m, n, x)
            if self.front and self.radial_emb:
                x_m_in = x_m_in * radial_weight
                post_radial_weights.append(None)
            else:
                post_radial_weights.append(radial_weight)
            x_inputs.append(x_m_in)

        raw_outputs = self._apply_indexed_sandwich_multi_gemm(x_inputs)
        for m, raw_output, radial_weight, module in zip(range(1, self.irreps_out.lmax + 1), raw_outputs, post_radial_weights, self.m_linear):
            linear_output = module._finish_linear_output(raw_output)
            if radial_weight is not None:
                linear_output = linear_output * radial_weight
            self._accumulate_grouped_pair_output_(out_groups, linear_output, rot_blocks, m)

        out = self._materialize_output_l_groups(out_groups, n=n, dtype=x.dtype, device=x.device)
        return out.contiguous(), wigner_D_all


class SO2_m_Linear(torch.nn.Module):
    def __init__(
            self,
            m,
            irreps_in,
            irreps_out,
            use_interpolation: bool = False,
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

    def forward(self, x_m):
        # x_m ~ [N, 2, n_channels]
        x_m = self.fc(x_m)
        return self._finish_linear_output(x_m)

    def _finish_linear_output(self, x_m):
        x_r = x_m.narrow(2, 0, self.num_out_channel)
        x_i = x_m.narrow(2, self.num_out_channel, self.num_out_channel)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)
        return torch.cat((x_m_r, x_m_i), dim=1)


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

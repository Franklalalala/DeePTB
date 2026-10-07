# H0 重建

`h0` 是 `1006-stable` 的独立物理 H0 子系统；NACF 在 `dptb/nacf` 维护。H0 对应原子初始电荷的 `T + Vnl + Vion + VH[rho0] + Vxc[rho0]`，需要与相同物理输入的初始算符比较。

## 安装与离线准备

配置相容的 Linux Python、PyTorch、CUDA、pyabacus、NumPy、SciPy 和势场后端，从仓库根目录执行：

```bash
source h0/env.sh
export ABACUS_SOURCE_DIR=/path/to/matching/abacus/source
export H0_EXTRA_INCLUDE_DIRS=/path/to/extra/include
bash h0/build_once.sh
python h0/precompile.py --check
```

构建从当前导入的 pyabacus 发现 ModuleNAO 库，并使用当前 Python 环境的 include/lib；`CUDA_HOME` 和动态库搜索路径需要匹配。原生构建由用户显式执行，正常推理不调用编译器。仓库只提供源码；二进制、离线表及安装 manifest 需要按实际环境建立，不能给旧 manifest 改名。

`h0/env.sh` 选择当前 checkout，并将临时文件与缓存放在 `H0_WORK_ROOT`，默认 `h0/work`。它不选择 GPU 或激活环境；执行前显式配置 `CUDA_VISIBLE_DEVICES`。

`prepare_tables.py RAW STORE` 和 `h0rebuild.offline` API 准备物种及两中心数据。源文件、自旋、径向网格、依赖及二进制身份必须匹配；改变身份后在新目录显式准备。无损共享径向表见 [README_SHARED_RADIAL.md](README_SHARED_RADIAL.md)。

## 组装接口

`h0rebuild.assemble.assemble_h0` 接收结构、物种、物理 cutoff、精确 FFT 网格、自旋和初始磁矩。`h0rebuild.deeptb` 验证 AO block 来源与特征打包约定。`production_io.py` 可读取 ABACUS 参考结果用于验证；实际推理可以直接提供结构和物理输入。

完整非局域 block 支持须显式选择 `pair_support='nonlocal_complete'`。默认 `orbital_overlap` 仅输出轨道支持重叠的原子对，可能遗漏非零的第三中心非局域贡献。`strict_reproduction=True` 不改变此支持选择。运行和内存边界见 [热路径说明](README_HOT_PATH.md)。

`output_atom_cell_shifts` 指定输出坐标的整数晶格偏移。若 `r_out_i = r_in_i + q_i @ cell`，H0/S 及分项 block 的键按 `R_out = R_in + q_i - q_j` 变换，同时更新序列化坐标和 fingerprint；矩阵值与周期势场保持一致。默认保留输入坐标。`h0rebuild.cell_gauge` 提供整数偏移校验和结果 rebasing；参考输入读取器使用明确记录的原子坐标，拒绝真实位移或原子顺序变化。

## 显式 SOC 参考对齐

`h0rebuild.soc_reference.align_soc_projectors` 与 `SOC_REFERENCE_DR_BOHR` 对应 ABACUS 的 `USE_NEW_TWO_CENTER` 参考规则。参考两中心网格采用 `cutoff=2*rmax`、`nr=int(rmax/0.01)+1`，约为 0.02 Bohr 间距，并采用奇数投影子 sample-count 规则。

参考为 [ABACUS ee99e3ca](https://github.com/deepmodeling/abacus-develop/tree/ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f)。[投影子截断](https://github.com/deepmodeling/abacus-develop/blob/ee99e3ca7f64f7c3b33cc6b68bc0bb99ef16599f/source/module_cell/setup_nonlocal.cpp#L111-L142) 将偶数 `cut_mesh` 向上取奇数，只复制 `ir < cut_mesh` 的样本。它与 UPF reader 的 inclusive support-count 约定不同。

常量 `SOC_REFERENCE_DR_BOHR=0.02` 不保证所有 cutoff 的实际网格一致。参考间距是 `2*rmax/(nr-1)`，CUDA 表生成器采用 `nr=ceil(2*rmax/requested_dr)+1`。例如 `rmax=9` 时均为 901 点；`rmax=5.005` 时参考为 501 点，0.02 请求产生 502 点。每个新 cutoff 都要核对实际网格。

辅助函数保留原 `cutoff_radius` 作为保守支持和网格边界，转换后不必等于 `r[cutoff_index-1]`；全低于阈值的投影子按可用 mesh 限定 sample-count。对齐只对原始物种应用一次，并在表准备和组装中使用相同物种与间距：

```python
from h0rebuild.soc_reference import align_soc_projectors, SOC_REFERENCE_DR_BOHR
from h0rebuild.offline import prepared_two_center
from h0rebuild.assemble import assemble_h0

assert physics_options['nspin'] == 4
species = align_soc_projectors(original_species)
table = prepared_two_center(
    species, store=new_store, dr_bohr=SOC_REFERENCE_DR_BOHR,
    nspin=4, device='cuda:0', prepare=True,
)
print(table.metadata)
del table
result = assemble_h0(
    structure, species, **physics_options,
    two_center_backend='pyabacus', two_center_dr_bohr=SOC_REFERENCE_DR_BOHR,
    offline_table_dir=str(new_store), **runtime_options,
)
```

使用新 store，保留原 CUDA/FFT 运行参数。旧表键对应原投影子样本与间距；默认行为、通用表准备 CLI 和已冻结 store 不自动切换。保留原始物种，避免重复应用转换。

## 验证

修改边界时运行 `tests_h0fast` 中对应的行为测试。`acceptance.py` 是显式真实数据验收入口；`new100.py` 提供共享 case worker 和兼容 CLI，`new100_v3.py` 委托同一验收器。参考输入需外部提供，不是仓库内资产。

数值比较覆盖 H0、S 及各物理分项、block 键、形状、dtype 和晶胞规范；报告实际后端、误差和计时范围。H0 以 Ry 存储时先转换为 eV 再报告能量误差。参考网格对齐是特定参考契约，不能据此推断全数据集精度、积分收敛、训练先验误差或端到端模型加速。

# 给 online GPT 的审查入口

请先读本文件、`SCALE68_REVIEW.md`，再看 `source/` 和 `evidence/`。这是供独立审查的小包，不是可直接安装的完整 GPU 产品。

## 任务与冻结状态

审查修复相对父提交 `72f3cb6f51fdee54143847982665d203194eea0d` 的正确性，并判断如何把 H0 CUDA 路线扩展到全量 68 元素、约 3 万结构。最新用户要求：挂起远端较长测试，先交 online GPT 审核。不得恢复远端测试、启动新训练、重编译或运行包内历史调度脚本。

计算源码提交为 `7292b54`；后续提交仅更新审查文档。本包包含最终 Git commit、修复 diff、源码清单和 72 文件本地/部署哈希对照。

旧新版长测此前在用户下班时暂停于 20/100（19 PASS、1 Mn NUMERICAL_FAIL），没有完成。旧调度器把暂停时间算入超时，已核对并退役六个旧进程，未恢复它们。当前 v4 使用独立 attempt、身份校验和暂停计时，不混入旧 v3 记录。

**当前 v4：76/100 已结束，73 PASS、2 NUMERICAL_FAIL、1 ERROR。另有 2 个在途 worker 已 SIGSTOP，22 个尚未开始。调度器也已 SIGSTOP。** `evidence/PAUSED_REVIEW.json` 保存 PID、argv、start_ticks；`acceptance_audit.json` 再次确认三进程暂停及源码身份未变。原 dispatcher status.json 保留 RUNNING 是暂停现场，不代表进程仍在计算；总状态见 `STATUS.json`。暂停不会释放进程已有的 GPU 内存。

75 个数值终态均通过 attempt 身份、return code、终态回执联合核验；余下 1 个工程 ERROR 被拒收。暂停期间源码仍冻结。待用户允许恢复时，须重新核对 PID/argv/start_ticks，保持当前版本；不要启动另一 dispatcher，也不要恢复 retired v3。

## 修复与证据

F02–F12 已实现针对性修复：参考 UPF 偶数 mesh 与 cutoff 次序；自旋/标量 PBE 梯度和散度 PW 投影、显式 LibXC 密度阈值；两中心 native 设备/布局/形状/索引保护；H0 生成依赖契约、私有 resident 返回值、锁和 CUDA event；attempt 隔离及暂停预算；magmom 与严格输入拒绝；NACF 完整缓存 key 和自校验原子发布；local-grid 使用保存 AO 样条。

25 项独立 H0 针对性用例、30 项 NACF 集成用例通过。原始 focused 日志 23 项；extended 日志 3 项中包含 1 项重跑和 2 项新增，不能算 26 项。含真实双 GPU ambient-device、非默认 stream、PyAbacus 积分对照、缓存污染与真实 SIGSTOP/SIGCONT。NACF 集成复用原 binary，没有重编译；合成案例不等于 3 万条物理标签全部重验。

100/100 输入完成离线制表，59 个独立 UPF/ORB 输入、零制表错误；这不等于 100/100 数值验收。两中心扩展安装时显式重编译一次；local-grid 复用未改动 binary。sm80/86/89/90 机器码存在，但只有 L40S/sm89 做了本轮数值验证。运行时禁止编译、UPF/ORB 解析与双中心制表。

## 未关闭问题

| 项目 | 证据与状态 |
|---|---|
| F01 / Mn | SOC_mp-561353，Mn 原子3、第四 s AO，Hmax=13.2446639 meV，Smax=1.90217e-8；仍超过既定 Hmax<5 meV。 |
| Ca | SOC_mp-19824，Ca 原子0、第四 s AO，Hmax=7.5399653 meV，Smax=2.48597e-8；不能误归因给结构内 Mn。 |
| 内存预算 | SOC_mp-23435 在场构建前估算 5143.45 MiB > 配置 4096 MiB，属于主动预算保护，未发生实测 CUDA OOM；尚未重试。 |
| local-grid native | support_kernel.cu 有 r_vals[16]，descriptor 可导致越界，任意原生 tensor 形状/设备/索引防护仍不完整；本 cohort 最大9通道不触发该边界。 |
| 扩展与性能 | 全量 68 元素 H0 表、3 万结构、跨机发行、真正跨结构 GPU batch、端到端吞吐都未验收。 |

Mn 误差主要在共同自旋 s 块；近 rank-one 且与原点 AO 外积近似成比例只是线索，不能断定 Vlocal 根因。独立 ABACUS 局域 AO 插值诊断仍给 13.236165 meV，未合并生产算法。T+Vlocal+Vnl 闭合只验证本方记账，不能分别证明其相对参考正确。

下一假设是原子恰好落在 FFT 网格上时，PBE correlation 的每自旋 gradient<1e-10 mask 是否受数值噪声影响。**尚未实验：诊断在等 GPU 锁，按用户暂停要求已取消，未生成结果。** 不能声称其解释了误差。需要同一参考 ABACUS v3.10.1 / ee99e3ca7 的分项场或 T/Vlocal/Vnl 输出；固定参考源码节选在 reference/。不得拟合矩阵、调势移、放宽门限或替换失败结构。

## 请审核并返回

1. 逐项复核 F02–F12，给出已解决/部分解决/仍有反例，不以测试通过替代代码推理；特别复核缓存输入身份、原子发布、跨 stream 私有拷贝和暂停/孤儿进程处理。
2. 对 F01/Ca 提出最多三项有判别力的分项实验，明确支持/排除什么假设；不要给未经参考分项验证的确定根因。
3. 根据 SCALE68_REVIEW.md 审查：哪些只需扩离线数据，哪些需改 native 再显式预编译一次；元素对缓存如何兼顾复用、内存预算与数值身份；3万结构验收与训练数据标签契约如何隔离。
4. 按严重程度给出可执行修复顺序，每项附源码路径、函数或行号、触发条件、证据级别与验收条件。可以提出新缺陷，但不要把假设写成已证实的 CUDA 故障。

## 在普通 CPU 环境可做的验证

`python -B replay_evidence.py`（仅依赖 NumPy）校验所有包文件 SHA256 并重放保存的 Mn/Ca onsite 误差；它不运行新 H0，也不能复现未包含的全结构 H/S。包内 GPU 测试日志是既有执行证据。

source/h0/tests_h0fast 保留当前测试实现供审查；有些依赖 Linux、Torch/CUDA、PyAbacus 或原 NACF 父环境。环境缺失应报告 skipped/unavailable，禁止把 CPU stand-in 冒称真实 GPU 验证。不要执行 check_offline.py 的过时 warm-is-disk 断言。包内不含完整数据、9GB离线表、二进制、凭据；相应清单保留供核对。

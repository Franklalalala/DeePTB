请 review 这个 H0/NACF CUDA 交接包。先读 README.md、SNAPSHOT_SCOPE.md 和 SOURCE_MANIFEST.json，再核对 SOURCE_MANIFEST/源码和证据。
当前全部计算任务由用户要求挂起；本次授权是独立 review，不是恢复计算。不要 SIGCONT、启动测试队列、重新编译或测试旧版本。
重点审查：SOC Mn 13.245 meV 残差；隔夜暂停与1200秒超时/幂等恢复；H0/NACF缓存失效、并发和复用；真实预编译加载；最终物理输入约定；已有测试能支持哪些结论。
要求按严重程度给出 file:line、触发条件、影响和最小修改建议，区分已确认问题、待证假设、完成证据和未完成工作。不要把19/20通过写成100结构成功。

# H0 原生二进制

在相容环境中显式执行 `bash h0/build_once.sh`，再用 `python h0/precompile.py --check` 检查安装。环境、头文件和库路径配置见 [README.md](README.md)。

预编译二进制与 manifest 必须对应实际源码、依赖和目标设备。正常推理不启动编译器，也不把其他环境的安装记录作为当前身份；需要新构建时保留原不可变安装，建立新的输出与记录。

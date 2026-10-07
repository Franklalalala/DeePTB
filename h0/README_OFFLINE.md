# H0 离线表

安装和组装入口见 [README.md](README.md)。物种与两中心径向表由 `prepare_tables.py RAW STORE` 或 `h0rebuild.offline` 显式准备。

离线表需要绑定源文件、自旋、径向网格、生成器、依赖及二进制身份。推理只读取匹配的已准备表；缺失或不相容的表需要在新目录重新准备，不能改写旧 manifest。共享压缩格式见 [README_SHARED_RADIAL.md](README_SHARED_RADIAL.md)。

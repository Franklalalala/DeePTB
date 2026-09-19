# Tonari acknowledgement

The neighbor geometry, pair policy, CPU enumeration and thread-pool sources in
`csrc/vendor/tonari/` are copied without changes from
[songfeitong/tonari](https://github.com/songfeitong/tonari), commit
`803aec8a89ce7269f0b80f66ddfc4f408a99d88a`, under the MIT license.
The original copyright and full license are retained in that directory.
We thank Tonari's authors and contributors for this implementation.

DeePTB uses the standalone C++ core. It does not import or require Tonari's Python
package, Torch bindings or CUDA provider. No Python/Torch version upgrade is
required by this integration. Compilation requires a C++20 compiler.

`csrc/topology.cpp`, `topology.py` and `edge_vna.py` implement our NACF-specific
integer-image intersection, strict directional support, exact edge-row mapping,
factor-query reuse and VNA contractions. The full-neighbor versus half-output
policy, separation of topology from numerical work, grouped execution, and
explicit bounded work buffers draw on Tonari's design. This acknowledgement does
not imply that Tonari supplies the VNA physics or that its CUDA provider is used.

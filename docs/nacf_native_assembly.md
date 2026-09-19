# Native preparation of P23 onsite and P2 hopping

`NACFTableBank.prepare(..., topology='native')` moves the geometry-dependent
projector and onsite VNA neighbourhood joins to the vendored C++ core. Build
`tools/build_nacf_topology.py --output /path/libnacf_topology.so` explicitly and
set `DPTB_NACF_TOPOLOGY_LIBRARY`, or pass `library=` to `prepare`. The default
remains `topology='python'`; inference never starts a compiler.

The selected backend changes preparation only. Both paths evaluate the same
radial tables, projector matrices, P23 factors, overlap and AO gauge, using
the bank's chosen Torch or fused CUDA numerical backend. Original directed
edge rows and integer image shifts are preserved. The native plan supports
molecules, partial PBC, skew cells, periodic self images and scalar or SOC
projector contractions. `max_terms` bounds the combined projector and onsite
VNA contraction count; it is not a total memory bound.

Nonlocal projector contributions include endpoint centres and the origin
self projector. Onsite VNA excludes only its own origin centre. Periodic
self images remain active. Projector support uses the existing inclusive
tolerance; VNA uses the existing strict support convention. No edge VNA is
added by this option. Missing P23 coverage follows the existing explicit
composition policy; missing S raises rather than substituting an identity.

An overlap-only store binds geometry-generated S to legacy P2 tables:

```python
import torch
from dptb.nacf.assembly import NACFTableBank
from dptb.nacf.overlap import OverlapTableStore

bank = NACFTableBank(p2, p23, overlap_store=OverlapTableStore(overlap_root),
                     device='cuda', dtype=torch.float64, backend='cuda')
plan = bank.prepare(symbols, positions_bohr, cell_bohr, edge_index,
                    edge_cell_shift, pbc=pbc, topology='native')
blocks = plan()
```

The overlap manifest must bind the exact P2 manifest. A manifest marked
complete does not establish dataset coverage or payload availability: check
the required species/pairs and file hashes. Numerical agreement with source
S is a separate gate. Building S neither modifies P2/P23 nor fills missing
P23 factor pairs.

The algorithm enumerates local centre neighbours once and joins their integer
identities, rather than scanning every atomic centre for every output block.
For bounded neighbourhoods, joins scale with the requested sparse graph.
Large cutoffs, dense graphs and many periodic images still increase work.
Report fresh preparation separately from radial table loading and from
forward, packing, input IO and writing; kernel latency is not production
throughput.

Tests: `dptb/tests/test_nacf_assembly_native.py` checks native identities against
independent lattice enumeration and scalar/SOC numerical outputs against the
existing preparation path. `test_nacf_edge_vna_native.py` covers the unchanged
edge VNA API sharing the same native library.

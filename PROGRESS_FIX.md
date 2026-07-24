# PARTIAL — Stage 0 passed; implementation has not started

## Stage 0 — reproduction and complete H0 consumer audit

Timestamp: 2026-07-24 10:28:25 +08:00  
Status: PASS

### Failure reproduced

Command:

```text
set PYTHONPATH=E:\deeptb\wt_0724_merge
C:\Users\16608\.conda\envs\dptb\python.exe -m pytest dptb/tests/test_block_ode_flow_highl_equivariance.py -q -s
```

Result: **2 failed, 4 warnings in 35.92 s**, matching the upstream finding.

| two_stage_pair_enable | node max abs drift | edge max abs drift |
|---|---:|---:|
| false | 8.1091233946705366e-01 | 1.9660634910835956e-01 |
| true | 8.1091233946705366e-01 | 2.8586735079904640e-01 |

### Every `self.node_projector` / `self.edge_projector` consumer

There are exactly two direct call sites in
`dptb/nn/embedding/lem_moe_v3_h0_helpers.py`.

| Call site | Reachable modes | Tensor source at projector boundary | Layout before the fix |
|---|---|---|---|
| `_fallback_node_features`: `self.node_projector(node_source)` | `h0_node_mode="direct"`; also the fallback computed by `h0_node_mode="self_edge"` | First width-compatible candidate among `h0_node_key` (normally `node_h0`), `node_hamiltonian`, and `fallback_node_key` (normally `node_features`); then `_mask_node_source` | **raw orbpair RME**. Dataset/block codec and inverse-CG producers preserve mapper/orbpair feature slices. The mask is also indexed in raw `orbpair_maps` order. Only uu_real previously applied raw→sorted. |
| `forward`: `self.edge_projector(edge_source[active_edges])` | every `use_h0_edge_init=true` path, including direct/self-edge node modes, plain frozen H0, spatial residual, and uu_real | First width-compatible candidate among `h0_edge_key` (normally `edge_h0`), `edge_hamiltonian`, and `fallback_edge_key` (normally `edge_features`); then `_mask_edge_source` | **raw orbpair RME**. The mask is raw. Only uu_real previously applied raw→sorted; `active_edges` is a row selection and does not alter coordinate order. |

`h0_node_mode="self_edge"` has no hidden third projector call. Its primary node
feature is scattered from `edge_features_h0`, which is already the output of
`self.edge_projector`. It still eagerly computes `_fallback_node_features`, so
that fallback is covered by the node call above and must obey the same sort
contract.

The two direct residual AO-block projectors use their own
`node_linear`/`edge_linear`, not `self.node_projector`/`self.edge_projector`.
Their `_contract` methods already perform:

```text
AO product block -> raw coupled RME -> index_select(sort_index) -> sorted-irrep Linear
```

### Layout evidence

- `OrbitalMapper.get_irreps()` constructs terms in raw `orbpairtype_maps`
  iteration order.
- `mask_to_nrme` / `mask_to_erme` are filled with raw `orbpair_maps` slices.
- `record_pipeline.decode_h0()` either passes precomputed `node_h0`/`edge_h0`
  through or obtains them from the normal block-to-feature transform; neither
  sorts irrep coordinates.
- `BlockStateCodec.blocks_to_rme()` gathers canonical product features and
  applies inverse CG in each raw `orbpairtype_maps` slice, returning those raw
  features unchanged in coordinate order.
- `H0InitLayer.node_projector` and `edge_projector`, in contrast, declare
  `irreps_in=self.idp.orbpair_irreps.sort()[0].simplify()`.

Therefore every accepted H0/fallback feature source that reaches these two
projectors is raw and must be sorted after the raw mask and before projection.

### Sort-index consistency

Runtime probe, basis `{"H": "1s", "C": "1s1p"}`:

```text
SP_HELPER_EQUAL=True
UU_HELPER_EQUAL=True
SP_UU_EQUAL=False
WIDTHS=13,16
spatial index=[0,4,1,2,3,5,6,7,8,9,10,11,12]
```

The apparent cross-mode inequality is expected and explained: the spatial
projector uses the 13-coordinate non-SOC upper-triangular mapper, whereas the
uu_real projector uses the 16-coordinate directed SOC-uu_real mapper. The two
projector classes deliberately reject each other's mapper contract and cannot
coexist in one `H0InitLayer`. Within each mapper, the projector buffer is
bitwise equal to `_sorted_irrep_coordinate_index(mapper)`, and
`_uureal_h0_sort_index` is a clone of the uu_real projector buffer. Thus both
paths use the same mapper-derived raw→sorted algorithm; their numerical
permutations must not be compared across different mapper layouts.


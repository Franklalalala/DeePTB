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

## Stage 1 — minimal raw→sorted repair

Timestamp: 2026-07-24 10:30:27 +08:00
Status: PASS

Implementation:

- `H0InitLayer` now derives one mapper-specific `_h0_sort_index` for every H0
  projector configuration.
- The generic buffer is non-persistent, preserving the legacy/default
  state_dict key set and cross-tree golden contract.
- Both node and edge sources now follow the single correct contract:
  raw-layout mask first, then `index_select(_h0_sort_index)`, then the
  sorted-irrep projector.
- The existing persistent uu_real buffer remains present and is checked against
  the generic mapper-derived index, preserving its checkpoint and runtime
  behavior.
- The spatial projector's own index and irreps are checked against the H0
  projector contract.
- The incorrect comment claiming that frozen/spatial H0 should stay raw at the
  projector boundary was replaced with the explicit raw-source/sorted-projector
  contract.

Verification:

```text
C:\Users\16608\.conda\envs\dptb\python.exe -m py_compile \
  dptb\nn\embedding\lem_moe_v3_h0_helpers.py
```

Result: **PASS**. `git diff --check`: **PASS**.

## Stage 2 — G-FIX1–4 verification

Timestamp: 2026-07-24 10:36:24 +08:00
Status: PASS

### G-FIX1 — l>0 flow equivariance

```text
C:\Users\16608\.conda\envs\dptb\python.exe -m pytest \
  dptb/tests/test_block_ode_flow_highl_equivariance.py -q -s
```

Result: **2 passed, 4 warnings in 29.14 s**.

| two-stage | node output max abs drift | edge output max abs drift |
|---|---:|---:|
| off | 6.6613381477509392e-16 | 2.7755575615628914e-16 |
| on | 6.6613381477509392e-16 | 3.6776137690708310e-16 |

Both are below the required `1e-9` fp64 gate.

### G-FIX2 — actual sorted-projector input boundary

The high-l test now installs forward pre-hooks on the real
`H0InitLayer.node_projector` and `edge_projector`. It compares the tensors
actually presented to those sorted-irrep linears under the correct Wigner
rotation, without any post-hoc sorting in the test.

| two-stage | node projector-input drift | edge projector-input drift |
|---|---:|---:|
| off | 1.7763568394002505e-15 | 5.5511151231257827e-17 |
| on | 1.7763568394002505e-15 | 5.5511151231257827e-17 |

The node value is the same `1.78e-15` fp64 roundoff reported by the upstream
boundary scan after its diagnostic sort; it replaces the former O(1) drift and
the production path now performs that sort itself.

### G-FIX3 — uu_real bit identity

`test_uureal_projection_is_bit_exact_with_legacy_sort_path` constructs the
directed SOC-uu_real configuration and compares the fixed generic path against
the preserved legacy `_uureal_h0_sort_index` path:

- sorted node tensors: `torch.equal`
- sorted edge tensors: `torch.equal`
- node projector outputs: `torch.equal`
- edge projector outputs: `torch.equal`
- legacy persistent `_uureal_h0_sort_index` remains in state_dict
- generic `_h0_sort_index` is non-persistent

### G-FIX4 — zero and scalar H0 bit identity

`test_zero_and_scalar_only_h0_are_bit_exact_under_sort` verifies:

- non-scalar mapper with all-zero H0: sorted tensor and biased projector output
  are `torch.equal` to the pre-fix raw input path;
- pure `1s`/l=0 mapper: sort index is exactly the identity, and random H0 plus
  projector output are `torch.equal`.

```text
C:\Users\16608\.conda\envs\dptb\python.exe -m pytest \
  dptb/tests/test_h0_rme_sort_contract.py -q -s
```

Result: **2 passed in 25.01 s**.

Both changed test files pass `py_compile`; `git diff --check`: **PASS**.

## Stage 3 — cross-tree golden, regression, and breaking-change report

Timestamp: 2026-07-24 10:44:47 +08:00
Status: PASS

### Cross-tree golden

The script's historical default head points at `wt_0724_contract`, so the
fixed merge tree was passed explicitly:

```text
C:\Users\16608\.conda\envs\dptb\python.exe \
  scripts\crosstree_golden_lem_h0.py \
  --base E:\deeptb\wt_0724_base \
  --head E:\deeptb\wt_0724_merge \
  --report F:\claude\0724_pair_iter2\results\crosstree_golden_FIX.md
```

Result:

```text
float32=PASS state_tensors=184 max_delta=0.0000000000000000e+00
float64=PASS state_tensors=184 max_delta=0.0000000000000000e+00
```

### Regression

`pytest dptb/tests -q` was attempted first and stopped at two collection
errors:

- `ModuleNotFoundError: so2_cuda_ops`: the brief's explicit environment
  exemption.
- `test_dpa4_focus_attention.py` imports the absent
  `EdgeMessageValueGate`. Git history proves this is pre-existing:
  `cb624ac` removed the class while leaving the test; neither `dcacda5` nor
  pre-fix `ed50321` contains the class.

The task-authorized alternative gate then covered the complete A/B/C/D test
files plus all named block-ODE/H0/configuration regressions, including the new
high-l and H0 sort-contract tests:

```text
361 passed, 162 warnings in 167.71 s (0:02:47)
```

No old test failed due to a nonzero-l raw-order expectation; no old assertion
was changed. The only updated pre-existing test is the high-l regression that
originally exposed the bug, now strengthened to inspect the actual projector
input boundary.

### Breaking impact

Full report:
`F:\claude\0724_pair_iter2\results\h0_equivariance_fix_report.md`.

RUN06/B_HB0 (`residual_ao_block_ode`) is in the affected spatial-residual
path. Existing checkpoints were trained against the non-equivariant raw-layout
bug and are semantically incompatible with the corrected conditioning layout.
The cross-irrep mismatch cannot be repaired by a simple learned-weight
permutation; retraining is required for the equivariance benefit. uu_real,
zero-H0, and pure-l=0 behavior remain bit-exact.

## Stage 4 — merge Stage 5 closeout

Timestamp: 2026-07-24 10:47:47 +08:00
Status: PASS

### All-default bit-exact summary

Focused command:

```text
pytest \
  test_lem_pair_refine.py::test_default_off_is_bit_exact_clean_superset_of_lem_moe_v3_h0 \
  test_hb0_endpoint_condition.py::test_condition_source_default_and_explicit_edge_0e_are_bit_exact \
  test_hb0_hermiticity.py::test_hb0_hermitian_average_default_false_is_bit_exact_and_keeps_node_blocks \
  test_hb0_head_input_rms.py::test_head_input_rms_default_off_has_no_key_and_is_bit_exact \
  test_two_stage_pair_integration.py::test_default_disabled_is_bit_exact_and_does_not_construct_a_module
```

Result: **5 passed in 45.14 s**.

The combined proof is transitive:

1. Cross-tree golden: old `wt_0724_base` `LemMoEV3H0` equals current merge
   `LemMoEV3H0`, fp32/fp64, 184 state tensors, `max|Δ|=0`.
2. Current default `LemPair` equals current `LemMoEV3H0` in RNG state,
   state_dict keys/tensors, and outputs when pair refinement/topology options
   are disabled.
3. Default/explicit legacy values for endpoint conditioning, Hermitian
   averaging, RMS telemetry, and two-stage pair have equal RNG/state/output;
   disabled pair-refine and two-stage modules are not constructed.

Thus the single merged configuration with all opt-in feature families left at
their defaults is bit-exact to the old `LemMoEV3H0`.

### Flag/config matrix for natlan ablations

| config name | default | owner | effect | lane |
|---|---|---|---|---|
| `mp_cutoff` | `None` | `LemPair` | Enables the fixed-architecture dual-cutoff MP subgraph; `None` keeps the legacy topology. | A |
| `mp_avg_num_neighbors` | `None` | `LemPair` | Optional MP-subgraph normalization override used only with dual cutoff. | A |
| `res_update_additive` | `false` | `LemPair` | Opts into unscaled `x + delta` residual updates. | A |
| `latents_layernorm` | `true` | `LemPair` | Applies the legacy LayerNorm before pair-backbone latent updates. | A |
| `pair_refine_enable` | `false` | `LemPair` | Constructs and applies the final ordered-edge SO(3) pair refinement. | C |
| `pair_refine_rank` | `16` | `LemPair` | Conditioner bottleneck rank for pair refinement. | C |
| `pair_refine_condition` | `scalar_0e` | `LemPair` | Selects the invariant condition passed to dynamic refinement weights. | C |
| `pair_refine_internal_weights` | `true` | `LemPair` | Enables the refinement TP's shared/internal static weights. | C |
| `pair_refine_init` | `0.0` | `LemPair` | Initializes dynamic refinement amplitude. | C |
| `pair_refine_weight_mode` | `full` | `LemPair` | Chooses legacy full per-edge TP weights or low-cost `per_path`. | C |
| `pair_refine_max_weight_numel` | `None` | `LemPair` | Optional fail-closed cap on full FCTP weight count. | C |
| `pair_refine_identity_init` | `false` | `LemPair` | When true, zero-initializes dynamic/static refinement for identity output. | C |
| `condition_source` | `edge_0e` | `LemMoEV3H0` | Keeps the legacy H-B0 edge conditioner; `endpoints` opts into endpoint conditioning. | B |
| `hb0_hermitian_average` | `false` | `LemMoEV3H0` | Opt-in reverse-edge transpose averaging at the H-B0 output boundary. | B |
| `log_head_input_rms` | `false` | `LemMoEV3H0` | Attaches detached per-irrep node/edge head-input RMS telemetry. | B |
| `two_stage_pair_enable` | `false` | `LemMoEV3H0` | Constructs the QHFlow-style late pair stream and norm-free refinement tail. | D |
| `two_stage_pair_refine_layers` | `2` | `LemMoEV3H0` | Number of norm-free two-stage pair tail layers. | D |
| `two_stage_pair_tail_gate` | `false` | `LemMoEV3H0` | Opts into gating the two-stage refinement tail. | D |
| `two_stage_pair_refine_rank` | `16` | `LemMoEV3H0` | Dynamic conditioner rank in the two-stage tail. | D |
| `two_stage_pair_refine_condition` | `scalar_0e` | `LemMoEV3H0` | Invariant source for two-stage dynamic refinement. | D |
| `two_stage_pair_refine_radial_dim` | `4` | `LemMoEV3H0` | Radial embedding width used by the late pair stream. | D |
| `two_stage_pair_refine_edge_chunk_size` | `64` | `LemMoEV3H0` | Ordered-edge chunk size for two-stage refinement memory control. | D |

Final hard gates at this stage:

- cross-tree golden fp32/fp64: **PASS**, 184 tensors, `max|Δ|=0`;
- A/B/C/D + named block-ODE/H0 regression: **361 passed**;
- focused all-default bit identity: **5 passed**;
- flag matrix: **complete**.

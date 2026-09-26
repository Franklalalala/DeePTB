# Atomic potential shift head

`model_options.shift_head` is optional (`null` or `mode: off` keeps the original
model). Enabled modes require direct transformed e3tb output with a compact SOC
uu-real mapper. Supported embeddings are `lem`, `lem_moe_v3`,
`lem_moe_v3_h0`, `lem_moe_v3_edge`, and `lem_moe_v3_edge_h0`. Block-native,
untransformed RME, full spinor, and non-SOC compressed outputs are rejected.

```json
{
  "mode": "atom",
  "hidden": 64,
  "layers": 2,
  "element_dim": 0,
  "freeze_backbone": false,
  "init_from": null
}
```

The MLP reads all l=0 channels immediately before the embedding's node output
head. `layers` counts Linear layers, with SiLU between them. A positive
`element_dim` appends a learned element embedding. The final Linear weight and
bias start at zero. `atom` predicts one potential per atom; `shell` predicts
one per radial super-basis shell (1s and 2s are separate). Absent element shells
are unused in valid AO slots and excluded from shell potential statistics.

Physical dimensionless S must be supplied in `phys_node_overlap` and
`phys_edge_overlap`, in the same compact slots as the labels. The embedding's
`node_overlap`/`edge_overlap` fields are not physical S. In compact space the
correction is `v_i S_ii` and `(v_i+v_j)/2 S_ij`; shell blocks use
`(v_i,s+v_j,t)/2 S_ij[s,t]`, including onsite cross-shell blocks. Corrections are
lifted into native output slots with the inverse of `uureal_projection_mask`
after the Hamiltonian transform. All other native slots are unchanged, and
loss/evaluation continue to use the existing projection. Zero correction is an
exact identity, including signed zero, with the ordinary addition gradient.

Each distance expert receives its existing ownership policy at construction,
so direct MultiTrainer expert calls and ensemble inference mask the correction
identically. The embedding active-edge subset and explicit expert masks also
apply. Transpose consistency follows from reciprocal S and reciprocal masks.

`freeze_backbone` disables gradients for all non-head parameters; Trainer and
MultiTrainer pass only trainable head parameters to the optimizer in that mode.
The off/unfrozen optimizer parameter groups retain their legacy ordering and
membership. Frozen backbones still run in the caller's train/eval mode.

`init_from` is applied only by a fresh `build_model` call, after constructing
the complete distance wrapper. It requires a dense (num_experts=1), head-free
checkpoint with exactly matching backbone state keys and shapes, including the
wrapper topology. Missing/extra/shape-mismatched keys fail. The head stays zero.
It initializes weights, not optimizer/scheduler state. Ordinary resume strictly
loads the entire new checkpoint, preserves freezing, and never reopens the
initialization source path. Do not combine dense resume with head initialization.

Set `data_options.train.overlap_sidecar_root` and
`data_options.validation.overlap_sidecar_root` to their respective split
folders. Each sidecar shard must have the same basename and four-byte key as
its main shard. Every read checks idx (also against the key), graph fingerprint,
record UID if present in both, shape, float32 dtype, and finite S. Missing or
invalid entries fail closed. Main target features and the stored edge graph
are required. Current prepacked loading preserves all stored edge rows; a
changed final graph fails rather than silently misaligning S. Fields are
registered for row-wise Collater concatenation.

Sidecars use `LMDBDataset._get_lmdb_env`, including its worker cache and the
same `lmdb.open` boundary used by the Hopper pread adapter. Dataset pickling
clears both main and sidecar handles via the existing `__getstate__` mechanism.
The disabled option performs no sidecar I/O and adds no physical fields.

Training forwards log `v_mean`, population `v_std`, `v_absmax`, and node/edge
correction RMS. HamilLossAbs additionally logs orbital/distance-masked
`node_delta_mae/rms`, `edge_delta_mae/rms`, and signed `task_loss_change`
(corrected minus reconstructed uncorrected L1+RMSE, excluding router penalties).
These are per-call, per-rank INFO diagnostics, with no DDP aggregation or
TensorBoard reduction. Logging synchronizes scalar values on GPU. These
potentials are learned correction coordinates, not measured DFT potentials.

K-expert potential heads and prior Gram routing are not implemented. Unknown
head configuration keys fail instead of being ignored.

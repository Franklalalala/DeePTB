# 0425 FlashTP-Style SO2 Route

The 0425 stable production route remains:

```text
so2_wigner_apply_mode = compact_blocks
so2_fusion_mode       = streamed_m_major_cueq
mole_linear_mode      = cueq_indexed_linear
onehot_tp_mode        = scalar_fast
```

This branch adds a tunable SO2 path-aggregation mode:

```text
DPTB_SO2_FLASH_AGGREGATE=input   # default: rotate input l-groups once
DPTB_SO2_FLASH_AGGREGATE=output  # aggregate local outputs, rotate each l once
DPTB_SO2_FLASH_AGGREGATE=hybrid  # input aggregation plus high-l output aggregation
DPTB_SO2_FLASH_AGGREGATE=1       # full input+output aggregate
DPTB_SO2_FLASH_AGGREGATE=0       # direct-output fallback
```

It also adds sparse graph-level MoE weight mixing:

```text
DPTB_MOLE_SPARSE_MIX=1           # default: mix only router top-k experts
DPTB_MOLE_SPARSE_MIX=0           # fallback: dense all-expert einsum
```

FlashTP's public kernel targets `e3nn.o3.TensorProduct` channelwise `uvu`
paths with edge scatter/reduce. DeePTB's current hot path is different:
`SO2_Linear` decomposes irreps by SO(2) order `m`, applies MoE linear maps, and
rotates with Wigner blocks. A direct FlashTP call does not match this interface
without rewriting the layer into a standard CG tensor product.

The transferable FlashTP idea is path aggregation:

1. rotate each input `l` group once into the local frame;
2. serve all `m` paths from local-frame slices;
3. optionally accumulate local outputs by `l`;
4. optionally rotate each output `l` group once back to the global frame.

The default is input-side aggregation only, so the route still writes `m`
contributions directly into the final output tensor and avoids the per-`l`
output group buffer. Output aggregation is opt-in because it can reduce repeated
small rotation work but reintroduces the grouped output buffer. Hybrid mode is
the intermediate benchmark target: it keeps direct output for low `l` channels
and only aggregates output groups with `l >= DPTB_SO2_FLASH_HYBRID_L_MIN`
(default `2`).

## Validation

The SO2 pytest coverage compares aggregate modes against the direct fallback for
forward values and gradients, including mixed `rotate_in`/`rotate_out` settings.
End-to-end CUDA speed and peak memory should still be decided by Liyue A/B runs
because the change mainly affects kernel launch and tensor materialization
patterns.

For MoE, `MOLERouterV3` keeps the dense coefficient tensor for compatibility but
also returns the selected `topk_indices` and normalized `topk_probs`.
`MOLELinear` consumes that metadata when available, gathers only the active
expert weights, and falls back to the old dense `einsum` path when top-k covers
all experts or when metadata is absent. This follows the same FlashTP pressure
point as the SO2 route: skip inactive paths instead of materializing work for
every possible path.

The LEM v3 embedding now also keeps latent tensors active-edge-only through the
stack. The public output contract is unchanged: `EDGE_OVERLAP_KEY` and
`EDGE_FEATURES_KEY` are scattered back to full-edge shape at the API boundary.
Inside the hot path, bessel inputs, spherical harmonics, latent residual updates,
and per-layer SO2 calls are restricted to active edges, which removes repeated
`latents[active_edges]` slices and full-edge latent `index_copy` updates.

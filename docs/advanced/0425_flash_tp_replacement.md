# 0425 FlashTP Replacement

## Scope

This branch replaces the current SO2 tensor-product route with a direct
FlashTP-compatible tensor product path:

```text
so2_fusion_mode = flash_tp
```

It is not a streamed SO2 optimization branch. The old `staged`,
`streamed_m_major_ref`, and `streamed_m_major_cueq` routes remain available as
explicit fallback and comparison modes, but they are no longer the default on
this branch.

## Implementation Route

`SO2_Linear(..., so2_fusion_mode="flash_tp")` now skips construction of the old
SO2 `fc_m0/m_linear` modules and builds an e3nn `TensorProduct` configured with
channelwise `uvu` instructions. When the existing LEM activation contract asks
for output multiplicities that do not exactly match a FlashTP `uvu` path, the
FlashTP tensor product writes to a compatible intermediate irrep layout and a
small e3nn `Linear` projects back to the original requested output irreps.

On CUDA, the forward path attempts to use `flashTP_e3nn.uvu_TP`; when that
module is not importable it falls back to the equivalent e3nn `TensorProduct`
for smoke tests.

Set:

```text
DPTB_FLASH_TP_REQUIRE=1
```

to fail fast if the official FlashTP backend is missing. This is the expected
mode for GPU validation runs that are meant to exercise the FlashTP kernels.

## Validation Policy

All liyue smoke tests for this branch should be wrapped in a 600 second timeout.
Accuracy is monitored only for gross divergence; the optimization target is peak
memory or speed improvement from replacing the SO2 route.

## Background

The 0425 stable branch intentionally used:

```text
compact_blocks + streamed_m_major_cueq + cueq_indexed_linear + scalar_fast
```

This branch intentionally moves away from that route to test official FlashTP as
a direct replacement for the SO2 tensor product layer.

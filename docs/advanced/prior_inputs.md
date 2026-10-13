# Prior inputs and checkpoints

Prior inputs distinguish three spaces: AO products, coupled CG irreps, and feature ordering. H0 and other prior-aware models accepting this layout transform AO-product fields according to `h0_ao_cg` before the sorted-irrep permutation and e3nn Linear projection. Input producers and models must use the same convention.

`h0_ao_cg` defaults to enabled at the constructor and configuration boundaries. Checkpoints store `h0_ao_cg_version`; explicitly setting `h0_ao_cg=false` preserves the original sorted-AO behavior of historical unmarked weights. Version mismatches must raise rather than silently change the scientific meaning of inputs to existing weights. Prior-table rotation conventions are a separate producer contract and require their own validation.

`skip_nonfinite_batch` defaults to enabled. All experts and ranks are checked before the optimizer update. A nonfinite batch clears gradients, records the consumed data cursor, and skips successful-step metrics and per-step scheduler advancement. It does not repair corrupted optimizer state or roll back mutations made during forward. Explicitly setting it to `false` selects the configured alternative behavior.

The identity used to add a prior back does not establish data or checkpoint compatibility. Validation must cover real-input/target alignment, prior conversion, checkpoint versions, two-stage initialization and freezing, finite gradients, and nonfinite-batch handling. Do not use test weights to initialize production training.

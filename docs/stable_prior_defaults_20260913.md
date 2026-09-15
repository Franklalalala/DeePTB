# Stable prior initialization and training defaults

New H0 and prior-conditioned models convert packed AO-product fields to coupled CG irreps before the sorted-irrep permutation and e3nn Linear. This applies to physical H0, NACF, and both serial prior2b branches. h0_ao_cg defaults to true at constructor and schema boundaries. The stable-southpole NACF table rotation is a separate producer fix.

Checkpoints persist h0_ao_cg_version. Unmarked historical weights can be loaded with explicit h0_ao_cg=false, preserving the sorted-AO behavior. A marker mismatch fails with an actionable error; do not silently change an existing checkpoint's scientific input contract. Already corrected H0-CG v1 checkpoints remain compatible.

skip_nonfinite_batch defaults to true, including when an older config omits it. Check all experts/ranks before optimizer updates, log the bad batch and consumed cursor, clear gradients, and skip successful-step metrics and per-step scheduler advancement. This does not repair an already corrupted optimizer or roll back mutations inside forward. Explicit false selects the old handling.

Backend defaults follow actual layer type. Supported edge-MoE constructors and normalized configs default to streamed_m_major_fused_p0. The activation-space prior router keeps its guard and compatible grouped route. Dense SO2_Linear defaults to indexed_sandwich_cuda_multi (output_major/raw), with its existing CPU/dtype/shape fallback; explicit standard preserves the legacy backend. A one-expert MoE is still a MoE layer, not a dense SO2 layer. The optional block_complex experiment requires an unavailable external export and is outside this release qualification.

Qualification uses real SOC29303 input/target fields, both initializer branches, S1-to-S2 checkpoint seed/freeze, finite training, observed CUDA dispatch, and independent nonfinite/DDP tests. Test source configs omit stable switches; saved effective configs record resolved values. Smoke weights must never initialize a new production lineage. Legacy H0-CG baselines may resume their matching checkpoints with preserved optimizer/scheduler state.

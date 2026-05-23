# Indexed Sandwich Materialized Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-MoE SO2 backend that spends extra GPU memory to materialize regular GEMM inputs and benchmark two GEMM strategies.

**Architecture:** The new `indexed_sandwich_materialized` path reuses the existing CUDA Wigner prologue/epilogue kernels, materializes all `m>0` pair inputs into one contiguous buffer, and then applies either cuBLAS grouped GEMM or a single block-dense GEMM. It stays separate from `indexed_sandwich_cuda_multi` so production A/B can compare the memory-heavy route without changing the current winner.

**Tech Stack:** PyTorch autograd, existing DeePTB SO2 CUDA pack/scatter ops, existing cuBLAS grouped GEMM wrapper, pytest, natlan CUDA smoke tests.

---

### Task 1: Tests

**Files:**
- Modify: `dptb/tests/test_so2_non_moe_cublas.py`

- [ ] **Step 1: Add alias and correctness tests**

Add a test that `materialized_sandwich` maps to `indexed_sandwich_materialized`, and a CUDA correctness test parametrized over `grouped` and `block_dense`.

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest -q dptb/tests/test_so2_non_moe_cublas.py -k "materialized" --tb=short
```

Expected: fail because `indexed_sandwich_materialized` is not accepted yet.

### Task 2: Backend

**Files:**
- Create: `dptb/nn/so2_materialized_sandwich.py`
- Modify: `dptb/nn/so2_sandwich_common.py`

- [ ] **Step 1: Extract reusable m0 helper**

Move the existing m0 output helper shape into `so2_sandwich_common.py` so scheduled and materialized backends share it.

- [ ] **Step 2: Implement materialized grouped GEMM**

Use `_PackPairsMultiFunction` to produce `[E, 2, sum(Cin_m)]`, split views by `m`, apply `indexed_sandwich_multi_gemm`, then scatter with the existing raw-pair CUDA epilogue.

- [ ] **Step 3: Implement block-dense GEMM**

Build a block-diagonal dense weight matrix from the per-`m` weights and run one large matmul on the fully materialized pair buffer. Split raw outputs by `m` and reuse the same epilogue.

### Task 3: Dispatch And Bench

**Files:**
- Modify: `dptb/nn/tensor_product.py`
- Modify: `tools/bench_so2_non_moe_cublas.py`

- [ ] **Step 1: Wire aliases and gate**

Accept `materialized_sandwich` and `cuda_materialized_sandwich`, map them to `indexed_sandwich_materialized`, and use env gates `DPTB_SO2_MATERIALIZED_MIN_EDGES` / `DPTB_SO2_MATERIALIZED_MAX_EDGES`.

- [ ] **Step 2: Extend benchmark**

Add `--materialized-strategy` and report `speedup_vs_standard_indexed_sandwich_materialized`.

### Task 4: Verification

**Files:**
- No source edits.

- [ ] **Step 1: Compile locally**

Run:

```powershell
python -m py_compile dptb\nn\tensor_product.py dptb\nn\so2_sandwich_common.py dptb\nn\so2_materialized_sandwich.py dptb\tests\test_so2_non_moe_cublas.py tools\bench_so2_non_moe_cublas.py
```

- [ ] **Step 2: Run natlan CUDA tests**

Upload the changed files to the natlan worktree and run:

```bash
python -m pytest -q dptb/tests/test_so2_non_moe_cublas.py -k "materialized" --tb=short
```

- [ ] **Step 3: Run module A/B**

Run n=2048 and n=4096 for `grouped` and `block_dense` strategies, then keep only strategies with plausible production upside.

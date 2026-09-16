"""Comprehensive unit, regression, and benchmark test suite for CUDA Two-Center & Nonlocal accelerator.

Verifies:
1. Recompilation invariance (binary sha256 unchanged across multiple species sets)
2. Numerical agreement against upstream PyAbacusTwoCenter on scalar (nspin=1) and SOC (nspin=4)
3. Near-zero and symmetry/Hermiticity properties (S(R) = S(-R)^T)
4. Batched GPU evaluation consistency
5. Diagnostic timing profile on 24-atom case nonSOC_db_seq_id_11083
"""
import hashlib
import json
import time
import unittest
from pathlib import Path
import numpy as np
import torch

from production_io import read_abacus_orb, read_upf, SpeciesData, Structure, Atom
from h0rebuild.scalar_upf import scalarize_upf
from h0rebuild.pyabacus_integrals import PyAbacusTwoCenter
from h0rebuild.cuda_two_center import CUDATwoCenter
import build_two_center
from h0rebuild.precompiled import verify


def load_simple_structure(case_dir: Path, scalarize: bool = False):
    lines = [x.split("#")[0].strip() for x in (case_dir / "STRU").read_text().splitlines() if x.strip()]
    ia, io = lines.index("ATOMIC_SPECIES"), lines.index("NUMERICAL_ORBITAL")
    il, iv, ip = [lines.index(x) for x in ["LATTICE_CONSTANT", "LATTICE_VECTORS", "ATOMIC_POSITIONS"]]
    spec = [x.split() for x in lines[ia + 1 : io]]
    orbs = lines[io + 1 : il]
    lat0 = float(lines[il + 1])
    cell = np.array([[float(v) for v in x.split()] for x in lines[iv + 1 : iv + 4]]) * lat0
    pos = ip + 2
    atoms = []
    for symbol, _, _ in spec:
        n = int(lines[pos + 2])
        for row in lines[pos + 3 : pos + 3 + n]:
            xyz = np.array([float(v) for v in row.split()[:3]])
            atoms.append(Atom(symbol, xyz))
        pos += 3 + n
    sd = {}
    for entry, orb in zip(spec, orbs):
        raw_upf = read_upf(case_dir / "PP_ORB" / entry[2])
        upf = scalarize_upf(raw_upf) if (scalarize and raw_upf.has_so) else raw_upf
        sd[entry[0]] = SpeciesData(read_abacus_orb(case_dir / "PP_ORB" / orb), upf)
    return Structure(cell, atoms), sd


class TestCUDATwoCenter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.case_10667_scalar = Path("/home/mingkang_nt/codex/h0_flash_20260912/production_reference/db_seq_id_10667")
        cls.case_10282_scalar = Path("/home/mingkang_nt/codex/h0_flash_20260912/production_reference/db_seq_id_10282")
        cls.case_10667_soc = Path("/home/mingkang_nt/codex/h0_cuda_benchmark_20260912/oracle/db_seq_id_10667_nspin4")
        cls.case_10282_soc = Path("/home/mingkang_nt/codex/h0_cuda_benchmark_20260912/oracle/db_seq_id_10282_nspin4")
        cls.case_11083_24atom = Path("/home/mingkang_nt/codex/h0_cuda_random100_20260912/raw/nonSOC_db_seq_id_11083")

    def test_01_recompilation_invariance(self):
        """Verify extension is compiled once: binary hash unchanged across multiple element sets."""
        so_path = self.repo_root / "h0rebuild" / "_cuda_two_center.so"
        self.assertTrue(so_path.exists(), f"Missing binary: {so_path}")
        initial_sha = build_two_center.compute_file_sha256(so_path)

        # Load first element set (Carbon)
        struct1, sd1 = load_simple_structure(self.case_10667_scalar, scalarize=True)
        backend1 = CUDATwoCenter(sd1, nspin=1)
        self.assertEqual(backend1.species, ["C"])

        sha_after_1 = build_two_center.compute_file_sha256(so_path)
        self.assertEqual(initial_sha, sha_after_1, "Binary modified during backend 1 initialization!")

        # Load second element set (Rb, Br)
        struct2, sd2 = load_simple_structure(self.case_10282_scalar, scalarize=True)
        backend2 = CUDATwoCenter(sd2, nspin=1)
        self.assertEqual(backend2.species, ["Rb", "Br"])

        sha_after_2 = build_two_center.compute_file_sha256(so_path)
        self.assertEqual(initial_sha, sha_after_2, "Binary modified during backend 2 initialization!")

        # Verify build check passes without recompiling
        self.assertEqual(verify('_cuda_two_center')['binary_sha256'], initial_sha)

    def test_02_scalar_pair_against_pyabacus(self):
        """Compare S and T evaluation between PyAbacusTwoCenter and CUDATwoCenter on scalar case."""
        struct, sd = load_simple_structure(self.case_10667_scalar, scalarize=True)
        ref = PyAbacusTwoCenter(sd, nspin=1)
        cuda = CUDATwoCenter(sd, nspin=1)

        test_displacements = [
            [0.5, 0.0, 0.0],
            [1.2, 0.8, -0.5],
            [-0.7, 1.4, 0.3],
            [2.1, -1.5, 0.9],
            [0.0, 2.5, -1.0],
        ]

        for disp in test_displacements:
            ref_S, ref_T = ref.scalar_pair("C", "C", [0, 0, 0], disp)
            cuda_S, cuda_T = cuda.scalar_pair("C", "C", [0, 0, 0], disp)

            diff_S = np.max(np.abs(ref_S - cuda_S))
            diff_T = np.max(np.abs(ref_T - cuda_T))

            self.assertLess(diff_S, 1e-6, f"S mismatch for disp {disp}: {diff_S:.3e}")
            self.assertLess(diff_T, (1e-7 / 13.605693122994), f"T mismatch for disp {disp}: {diff_T:.3e}")

    def test_03_projector_overlap_against_pyabacus(self):
        """Compare projector-orbital overlap Q between PyAbacusTwoCenter and CUDATwoCenter."""
        struct, sd = load_simple_structure(self.case_10667_scalar, scalarize=True)
        ref = PyAbacusTwoCenter(sd, nspin=1)
        cuda = CUDATwoCenter(sd, nspin=1)

        test_displacements = [
            [0.3, 0.2, -0.1],
            [0.9, -0.6, 0.4],
            [1.5, 0.5, -0.8],
        ]

        for disp in test_displacements:
            ref_Q = ref.overlap("proj", "C", "orb", "C", disp, "S")
            cuda_Q = cuda.overlap("proj", "C", "orb", "C", disp, "S")

            diff_Q = np.max(np.abs(ref_Q - cuda_Q))
            self.assertLess(diff_Q, 1e-6, f"Q mismatch for disp {disp}: {diff_Q:.3e}")

    def test_04_nonlocal_block_scalar(self):
        """Compare nonlocal V_nl assembly for scalar nspin=1."""
        struct, sd = load_simple_structure(self.case_10667_scalar, scalarize=True)
        ref = PyAbacusTwoCenter(sd, nspin=1)
        cuda = CUDATwoCenter(sd, nspin=1)

        c1 = [0.0, 0.0, 0.0]
        c2 = [1.2, 0.8, -0.5]
        candidates = [(0, np.array([0.5, 0.4, -0.2])), (1, np.array([0.8, 0.5, -0.1]))]

        ref_vnl = ref.nonlocal_block(struct, "C", "C", c1, c2, candidates)
        cuda_vnl = cuda.nonlocal_block(struct, "C", "C", c1, c2, candidates)

        diff_vnl = np.max(np.abs(ref_vnl - cuda_vnl))
        self.assertLess(diff_vnl, (1e-7 / 13.605693122994), f"Nonlocal scalar V_nl mismatch: {diff_vnl:.3e}")

    def test_05_soc_against_pyabacus_10667(self):
        """Compare scalar_pair and nonlocal_block for SOC (nspin=4) on db_seq_id_10667."""
        struct, sd = load_simple_structure(self.case_10667_soc)
        ref = PyAbacusTwoCenter(sd, nspin=4)
        cuda = CUDATwoCenter(sd, nspin=4)

        # Test scalar_pair
        disp = [1.1, -0.7, 0.4]
        ref_S, ref_T = ref.scalar_pair("C", "C", [0, 0, 0], disp)
        cuda_S, cuda_T = cuda.scalar_pair("C", "C", [0, 0, 0], disp)
        self.assertLess(np.max(np.abs(ref_S - cuda_S)), 1e-6)
        self.assertLess(np.max(np.abs(ref_T - cuda_T)), (1e-7 / 13.605693122994))

        # Test nonlocal_block with complex spinor D
        c1 = [0.0, 0.0, 0.0]
        c2 = [1.0, 0.8, -0.4]
        candidates = [(0, np.array([0.4, 0.3, -0.1]))]

        ref_vnl = ref.nonlocal_block(struct, "C", "C", c1, c2, candidates)
        cuda_vnl = cuda.nonlocal_block(struct, "C", "C", c1, c2, candidates)

        self.assertTrue(np.iscomplexobj(cuda_vnl), "SOC V_nl must be complex")
        diff_vnl = np.max(np.abs(ref_vnl - cuda_vnl))
        self.assertLess(diff_vnl, (1e-7 / 13.605693122994), f"SOC V_nl mismatch: {diff_vnl:.3e}")

    def test_06_soc_against_pyabacus_10282(self):
        """Compare scalar_pair and nonlocal_block for SOC (nspin=4) on binary RbBr (db_seq_id_10282)."""
        struct, sd = load_simple_structure(self.case_10282_soc)
        ref = PyAbacusTwoCenter(sd, nspin=4)
        cuda = CUDATwoCenter(sd, nspin=4)

        # Test heteronuclear pair (Rb, Br)
        disp = [2.0, 1.5, -1.0]
        ref_S, ref_T = ref.scalar_pair("Rb", "Br", [0, 0, 0], disp)
        cuda_S, cuda_T = cuda.scalar_pair("Rb", "Br", [0, 0, 0], disp)
        self.assertLess(np.max(np.abs(ref_S - cuda_S)), 1e-6)
        self.assertLess(np.max(np.abs(ref_T - cuda_T)), (1e-7 / 13.605693122994))

        # Test nonlocal_block
        c1 = [0.0, 0.0, 0.0]
        c2 = [2.0, 1.5, -1.0]
        candidates = [(0, np.array([0.8, 0.6, -0.4])), (1, np.array([1.2, 0.9, -0.6]))]

        ref_vnl = ref.nonlocal_block(struct, "Rb", "Br", c1, c2, candidates)
        cuda_vnl = cuda.nonlocal_block(struct, "Rb", "Br", c1, c2, candidates)

        diff_vnl = np.max(np.abs(ref_vnl - cuda_vnl))
        self.assertLess(diff_vnl, (1e-7 / 13.605693122994), f"SOC binary V_nl mismatch: {diff_vnl:.3e}")

    def test_07_near_zero_and_symmetry(self):
        """Test numerical stability at near-zero displacement and Hermiticity symmetry."""
        struct, sd = load_simple_structure(self.case_10667_scalar, scalarize=True)
        cuda = CUDATwoCenter(sd, nspin=1)

        # Test R -> 0 stability
        disp_zero = [0.0, 0.0, 0.0]
        s_zero, t_zero = cuda.scalar_pair("C", "C", [0, 0, 0], disp_zero)
        self.assertFalse(np.isnan(s_zero).any(), "NaN in S at R=0")
        self.assertFalse(np.isnan(t_zero).any(), "NaN in T at R=0")

        # S at R=0 must be symmetric
        np.testing.assert_allclose(s_zero, s_zero.T, atol=1e-10, err_msg="S(0) not symmetric")

        # Test infinitesimal R
        disp_tiny = [1e-8, 1e-8, 1e-8]
        s_tiny, t_tiny = cuda.scalar_pair("C", "C", [0, 0, 0], disp_tiny)
        self.assertFalse(np.isnan(s_tiny).any(), "NaN in S at tiny R")
        np.testing.assert_allclose(s_zero, s_tiny, atol=1e-5, err_msg="Discontinuous limit at R->0")

        # Test Hermiticity / symmetry: S(R) = S(-R)^T, T(R) = T(-R)^T
        disp = [1.5, -0.8, 0.6]
        disp_neg = [-1.5, 0.8, -0.6]
        s_pos, t_pos = cuda.scalar_pair("C", "C", [0, 0, 0], disp)
        s_neg, t_neg = cuda.scalar_pair("C", "C", [0, 0, 0], disp_neg)

        np.testing.assert_allclose(s_pos, s_neg.T, atol=1e-10, err_msg="S(R) != S(-R)^T")
        np.testing.assert_allclose(t_pos, t_neg.T, atol=1e-10, err_msg="T(R) != T(-R)^T")

    def test_08_batched_evaluation_consistency(self):
        """Test that eval_two_center_batch gives identical results to single-pair evaluations."""
        struct, sd = load_simple_structure(self.case_10282_scalar, scalarize=True)
        cuda = CUDATwoCenter(sd, nspin=1)

        rng = np.random.default_rng(42)
        n_pairs = 30
        pair_symbols = []
        displacements = rng.uniform(-3.0, 3.0, size=(n_pairs, 3))

        species_list = ["Rb", "Br"]
        for i in range(n_pairs):
            s1 = species_list[i % 2]
            s2 = species_list[(i // 2) % 2]
            pair_symbols.append((s1, s2))

        # Batched evaluation on GPU
        out_S, out_T = cuda.eval_two_center_batch(pair_symbols, displacements)

        # Compare each pair against single-pair adapter
        for k in range(n_pairs):
            s1, s2 = pair_symbols[k]
            single_S, single_T = cuda.scalar_pair(s1, s2, [0, 0, 0], displacements[k])

            n1 = cuda.norb_per_species[s1]
            n2 = cuda.norb_per_species[s2]

            batch_S_k = out_S[k, :n1, :n2].cpu().numpy()
            batch_T_k = out_T[k, :n1, :n2].cpu().numpy()

            np.testing.assert_allclose(batch_S_k, single_S, atol=1e-14, err_msg=f"Batch S mismatch at pair {k}")
            np.testing.assert_allclose(batch_T_k, single_T, atol=1e-14, err_msg=f"Batch T mismatch at pair {k}")

    def test_09_diagnostic_profile_24atom_11083(self):
        """Diagnostic timing benchmark on 24-atom case nonSOC_db_seq_id_11083."""
        self.assertTrue(self.case_11083_24atom.exists(), f"Missing case: {self.case_11083_24atom}")
        struct, sd = load_simple_structure(self.case_11083_24atom, scalarize=True)

        # 1. Measure data-preparation time for CUDATwoCenter
        t0 = time.perf_counter()
        cuda = CUDATwoCenter(sd, nspin=1)
        cuda_prep_time = cuda.timing["prep_seconds"]
        print(f"\n[24-atom benchmark] CUDATwoCenter table prep time: {cuda_prep_time:.4f} s")

        # 2. Setup a benchmark batch of 100 orbital pairs
        rng = np.random.default_rng(123)
        n_pairs = 100
        atoms = struct.atoms
        pair_symbols = []
        displacements = rng.uniform(-4.0, 4.0, size=(n_pairs, 3))
        for k in range(n_pairs):
            a1 = atoms[k % len(atoms)]
            a2 = atoms[(k + 1) % len(atoms)]
            pair_symbols.append((a1.species, a2.species))

        # 3. Time batched CUDA evaluation (warmup + timed)
        cuda.eval_two_center_batch(pair_symbols[:5], displacements[:5])  # warmup
        torch.cuda.synchronize()

        t0_cuda = time.perf_counter()
        out_S, out_T = cuda.eval_two_center_batch(pair_symbols, displacements)
        torch.cuda.synchronize()
        cuda_kernel_time = time.perf_counter() - t0_cuda
        print(f"[24-atom benchmark] CUDA kernel time for {n_pairs} pairs: {cuda_kernel_time * 1000:.2f} ms ({cuda_kernel_time / n_pairs * 1000:.3f} ms/pair)")

        # 4. Time PyAbacusTwoCenter baseline for the same pairs
        t0_ref_init = time.perf_counter()
        ref = PyAbacusTwoCenter(sd, nspin=1)
        ref_prep_time = time.perf_counter() - t0_ref_init

        t0_ref = time.perf_counter()
        for k in range(n_pairs):
            s1, s2 = pair_symbols[k]
            ref.scalar_pair(s1, s2, [0, 0, 0], displacements[k])
        ref_loop_time = time.perf_counter() - t0_ref
        print(f"[24-atom benchmark] PyAbacus loop time for {n_pairs} pairs: {ref_loop_time:.4f} s ({ref_loop_time / n_pairs * 1000:.2f} ms/pair)")

        speedup = ref_loop_time / cuda_kernel_time
        print(f"[24-atom benchmark] Speedup over Python orbital loop: {speedup:.1f}x")

        # Assert significant speedup (> 10x)
        self.assertGreater(speedup, 10.0, f"CUDA kernel speedup was only {speedup:.1f}x (expected > 10x)")


if __name__ == "__main__":
    unittest.main()

"""Comprehensive test suite for CUDA local grid and potential acceleration lane."""
import unittest
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import CubicSpline

import h0rebuild._cuda_local_grid as clg
from h0rebuild.cuda_local_grid import (
    CudaSpeciesCache,
    CudaGeometrySupports,
    CudaPeriodicFFTGridAOCache,
)
from h0rebuild.radial import OrbitalEvaluator
from h0rebuild.grid_collocation import FFTGridAOCache, _cartesian_box_index_bounds
from h0rebuild.periodic_collocation import PeriodicFFTGridAOCache
from h0rebuild.assemble import build_periodic_field, iter_pair_images
from h0rebuild.models import BlockKey
from production_io import load_case

RAW_ROOT = Path("/home/mingkang_nt/codex/h0_cuda_random100_cell_gauge_v2_20260912/raw")


class TestCudaLocalGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert cls.device.type == "cuda", "CUDA must be available for local grid tests"

    def test_01_atom_support_exactness(self):
        """Test GPU support generation matches CPU FFTGridAOCache exact nodes and AO values."""
        st, sd, opts, _ = load_case(RAW_ROOT / "nonSOC_db_seq_id_15674")
        field = build_periodic_field(
            st, sd, ecutrho_ry=opts["ecutrho_ry"], fft_shape=opts["fft_shape"],
            xc="LDA_PZ81", include_nlcc=False, field_backend="numpy", compute_device="cpu"
        )
        atom = st.atoms[0]
        ev = OrbitalEvaluator(sd[atom.species].orb)
        center = atom.frac @ st.cell_bohr

        # CPU support
        cpu_builder = FFTGridAOCache(field, max_bytes=0)
        cpu_sup = cpu_builder.support(ev, center)
        cpu_indices = cpu_sup.integer_indices
        lo_cpu = cpu_indices.min(axis=0)
        hi_cpu = cpu_indices.max(axis=0)
        lookup_cpu = np.full(tuple(hi_cpu - lo_cpu + 1), -1, dtype=np.int32)
        rel = cpu_indices - lo_cpu
        lookup_cpu[rel[:, 0], rel[:, 1], rel[:, 2]] = np.arange(cpu_sup.npoints, dtype=np.int32)

        # GPU support
        spec_cache = CudaSpeciesCache(sd, device=self.device)
        sp = spec_cache.get(atom.species)
        nmin, nmax = _cartesian_box_index_bounds(field, center - sp["rcut"], center + sp["rcut"])
        counts = nmax - nmin + 1

        center_t = torch.tensor(center, device=self.device, dtype=torch.float64)
        nmin_t = torch.tensor(nmin, device=self.device, dtype=torch.int64)
        counts_t = torch.tensor(counts, device=self.device, dtype=torch.int32)
        shape_t = torch.tensor(field.shape, device=self.device, dtype=torch.int64)
        cell_t = torch.tensor(field.cell_bohr, device=self.device, dtype=torch.float64)
        field_t = torch.tensor(field.values_ry, device=self.device, dtype=torch.float64)
        dummy_spin = torch.empty(0, device=self.device, dtype=torch.float64)

        res = clg.build_atom_support_cuda(
            center_t, sp["rcut"], nmin_t, counts_t, shape_t, cell_t,
            sp["dr"], sp["n_intervals"], sp["num_channels"], sp["spline_coeffs"],
            sp["norb"], sp["descriptors"], field_t, False, dummy_spin
        )

        gpu_indices = res[0].cpu().numpy()
        gpu_values = res[1].cpu().numpy()
        lo_gpu = res[4].cpu().numpy()
        hi_gpu = res[5].cpu().numpy()
        lookup_gpu = res[6].cpu().numpy()

        self.assertEqual(cpu_sup.npoints, len(gpu_indices))
        np.testing.assert_array_equal(lo_cpu, lo_gpu)
        np.testing.assert_array_equal(hi_cpu, hi_gpu)
        np.testing.assert_array_equal(cpu_indices, gpu_indices)
        np.testing.assert_array_equal(lookup_cpu, lookup_gpu)

        max_ao_diff = np.max(np.abs(cpu_sup.values - gpu_values))
        self.assertLess(max_ao_diff, 1e-10)

    def test_02_boundary_atom_outside_home_cell(self):
        """Preserve support for atoms slightly outside the home cell (the prior bug)."""
        st, sd, opts, _ = load_case(RAW_ROOT / "nonSOC_db_seq_id_15674")
        field = build_periodic_field(
            st, sd, ecutrho_ry=opts["ecutrho_ry"], fft_shape=opts["fft_shape"],
            xc="LDA_PZ81", include_nlcc=False, field_backend="numpy", compute_device="cpu"
        )
        atom = st.atoms[0]
        ev = OrbitalEvaluator(sd[atom.species].orb)
        center_outside = (np.array([-1e-7, 0.25, 0.45])) @ st.cell_bohr

        # CPU support
        cpu_builder = FFTGridAOCache(field, max_bytes=0)
        cpu_sup = cpu_builder.support(ev, center_outside)

        # GPU support
        spec_cache = CudaSpeciesCache(sd, device=self.device)
        sp = spec_cache.get(atom.species)
        nmin, nmax = _cartesian_box_index_bounds(field, center_outside - sp["rcut"], center_outside + sp["rcut"])
        counts = nmax - nmin + 1

        center_t = torch.tensor(center_outside, device=self.device, dtype=torch.float64)
        nmin_t = torch.tensor(nmin, device=self.device, dtype=torch.int64)
        counts_t = torch.tensor(counts, device=self.device, dtype=torch.int32)
        shape_t = torch.tensor(field.shape, device=self.device, dtype=torch.int64)
        cell_t = torch.tensor(field.cell_bohr, device=self.device, dtype=torch.float64)
        field_t = torch.tensor(field.values_ry, device=self.device, dtype=torch.float64)
        dummy_spin = torch.empty(0, device=self.device, dtype=torch.float64)

        res = clg.build_atom_support_cuda(
            center_t, sp["rcut"], nmin_t, counts_t, shape_t, cell_t,
            sp["dr"], sp["n_intervals"], sp["num_channels"], sp["spline_coeffs"],
            sp["norb"], sp["descriptors"], field_t, False, dummy_spin
        )

        gpu_indices = res[0].cpu().numpy()
        gpu_values = res[1].cpu().numpy()
        self.assertEqual(cpu_sup.npoints, len(gpu_indices))
        np.testing.assert_array_equal(cpu_sup.integer_indices, gpu_indices)
        self.assertLess(np.max(np.abs(cpu_sup.values - gpu_values)), 1e-10)

    def test_03_small_nonsoc_full_blocks_regression(self):
        """Validate local blocks on small nonSOC system (nonSOC_db_seq_id_15674)."""
        st, sd, opts, _ = load_case(RAW_ROOT / "nonSOC_db_seq_id_15674")
        field = build_periodic_field(
            st, sd, ecutrho_ry=opts["ecutrho_ry"], fft_shape=opts["fft_shape"],
            xc=opts["xc"], include_nlcc=opts["include_nlcc"],
            field_backend="numpy", compute_device="cpu"
        )
        evaluators = {s: OrbitalEvaluator(sd[s].orb) for s in sd}
        orbital_cutoffs = np.asarray([evaluators[a.species].basis.rcut for a in st.atoms])

        pairs = list(iter_pair_images(st, orbital_cutoffs))[:50]
        cpu_cache = PeriodicFFTGridAOCache(field, max_bytes=None, device="cpu")
        cuda_cache = CudaPeriodicFFTGridAOCache(field, max_bytes=None, device=self.device)

        home_pos = st.cart_positions
        species_list = [a.species for a in st.atoms]

        v_gpu_blocks, _ = cuda_cache.contract_pairs_batch(pairs, species_list, evaluators, home_pos)

        diffs = []
        for i, j, R, ci, cj in pairs:
            v_cpu = cpu_cache.contract_pair(evaluators[st.atoms[i].species], evaluators[st.atoms[j].species], ci, home_pos[j], R)
            key = BlockKey(i, j, tuple(int(x) for x in R))
            v_gpu = v_gpu_blocks[key]
            diff = np.max(np.abs(v_cpu - v_gpu))
            diffs.append(diff)

        max_err_ev = max(diffs) * 13.605693122994
        print(f"\n[test_03_small_nonsoc] Pairs: {len(pairs)}, Max error: {max_err_ev:.3e} eV")
        self.assertLess(max_err_ev, 1e-7, "Local blocks must match CPU within 1e-7 eV")

    def test_04_small_soc_collinear_field_regression(self):
        """Validate local blocks on small SOC system (SOC_mp-31055) with zero-moment / spin-z potential."""
        from dataclasses import replace
        from h0rebuild.spin_fields import add_collinear_spin_field

        st, sd, opts, _ = load_case(RAW_ROOT / "SOC_mp-31055")
        field = build_periodic_field(
            st, sd, ecutrho_ry=opts["ecutrho_ry"], fft_shape=opts["fft_shape"],
            xc=opts["xc"], include_nlcc=opts["include_nlcc"],
            field_backend="numpy", compute_device="cpu"
        )
        moments_z = [0.0] * len(st.atoms)
        field_mod, spin_z = add_collinear_spin_field(
            field, st, sd, moments_z,
            backend="direct", device="cpu", eps=1e-12
        )
        spin_z_field = replace(field_mod, values_ry=spin_z)

        evaluators = {s: OrbitalEvaluator(sd[s].orb) for s in sd}
        orbital_cutoffs = np.asarray([evaluators[a.species].basis.rcut for a in st.atoms])

        pairs = list(iter_pair_images(st, orbital_cutoffs))[:30]

        cpu_cache_v = PeriodicFFTGridAOCache(field_mod, max_bytes=None, device="cpu")
        cpu_cache_vz = PeriodicFFTGridAOCache(spin_z_field, max_bytes=None, device="cpu")

        cuda_cache = CudaPeriodicFFTGridAOCache(field_mod, spin_z_field=spin_z_field, max_bytes=None, device=self.device)
        home_pos = st.cart_positions
        species_list = [a.species for a in st.atoms]

        v_gpu, vz_gpu = cuda_cache.contract_pairs_batch(pairs, species_list, evaluators, home_pos)

        diffs_v = []
        diffs_vz = []
        for i, j, R, ci, cj in pairs:
            ev_i = evaluators[st.atoms[i].species]
            ev_j = evaluators[st.atoms[j].species]
            vc = cpu_cache_v.contract_pair(ev_i, ev_j, ci, home_pos[j], R)
            vzc = cpu_cache_vz.contract_pair(ev_i, ev_j, ci, home_pos[j], R)
            key = BlockKey(i, j, tuple(int(x) for x in R))

            diffs_v.append(np.max(np.abs(vc - v_gpu[key])))
            diffs_vz.append(np.max(np.abs(vzc - vz_gpu[key])))

        max_err_v = max(diffs_v) * 13.605693122994
        max_err_vz = max(diffs_vz) * 13.605693122994
        print(f"\n[test_04_small_soc] Pairs: {len(pairs)}, Max error V: {max_err_v:.3e} eV, Vz: {max_err_vz:.3e} eV")
        self.assertLess(max_err_v, 1e-7)
        self.assertLess(max_err_vz, 1e-7)

    def test_05_bounded_slow24_case_regression(self):
        """Validate local blocks on bounded portion (first 30 pairs) of 24-atom Ba/Mo/N case."""
        st, sd, opts, _ = load_case(RAW_ROOT / "nonSOC_db_seq_id_11083")
        field = build_periodic_field(
            st, sd, ecutrho_ry=opts["ecutrho_ry"], fft_shape=opts["fft_shape"],
            xc=opts["xc"], include_nlcc=opts["include_nlcc"],
            field_backend="torch", compute_device="cuda:0", structure_factor_backend="cufinufft",
            structure_factor_max_work_mb=2048.0, field_max_work_mb=2048.0
        )
        evaluators = {s: OrbitalEvaluator(sd[s].orb) for s in sd}
        orbital_cutoffs = np.asarray([evaluators[a.species].basis.rcut for a in st.atoms])

        pairs = list(iter_pair_images(st, orbital_cutoffs))[:30]

        cpu_cache = PeriodicFFTGridAOCache(field, max_bytes=None, device="cpu")
        cuda_cache = CudaPeriodicFFTGridAOCache(field, max_bytes=None, device=self.device)
        home_pos = st.cart_positions
        species_list = [a.species for a in st.atoms]

        v_gpu, _ = cuda_cache.contract_pairs_batch(pairs, species_list, evaluators, home_pos)

        diffs = []
        for i, j, R, ci, cj in pairs:
            ev_i = evaluators[st.atoms[i].species]
            ev_j = evaluators[st.atoms[j].species]
            vc = cpu_cache.contract_pair(ev_i, ev_j, ci, home_pos[j], R)
            key = BlockKey(i, j, tuple(int(x) for x in R))
            diffs.append(np.max(np.abs(vc - v_gpu[key])))

        max_err = max(diffs) * 13.605693122994
        print(f"\n[test_05_slow24] Pairs: {len(pairs)}, Max error: {max_err:.3e} eV")
        self.assertLess(max_err, 1e-7)

    def test_06_cache_lifetime_and_budget(self):
        """Validate CudaSpeciesCache retention and CudaGeometrySupports memory budget."""
        st, sd, opts, _ = load_case(RAW_ROOT / "nonSOC_db_seq_id_15674")
        spec_cache = CudaSpeciesCache(sd, device=self.device)

        field = build_periodic_field(
            st, sd, ecutrho_ry=opts["ecutrho_ry"], fft_shape=opts["fft_shape"],
            xc="LDA_PZ81", include_nlcc=False, field_backend="numpy", compute_device="cpu"
        )
        geom1 = CudaGeometrySupports(st, spec_cache, field, max_bytes=1024 * 1024**2, device=self.device)
        self.assertGreater(geom1.total_bytes, 0)
        bytes1 = geom1.total_bytes

        # Geometry changes: clear and rebuild
        geom1.clear()
        self.assertEqual(geom1.total_bytes, 0)
        self.assertEqual(len(geom1.supports), 0)

        # Rebuild again using same spec_cache
        geom2 = CudaGeometrySupports(st, spec_cache, field, max_bytes=1024 * 1024**2, device=self.device)
        self.assertEqual(geom2.total_bytes, bytes1)

        # Verify exceeding memory budget raises MemoryError
        with self.assertRaises(MemoryError):
            CudaGeometrySupports(st, spec_cache, field, max_bytes=1024, device=self.device)


if __name__ == "__main__":
    unittest.main()

"""Reusable CUDA acceleration for AO-grid and local potential contractions."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence, Any, Optional

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from .models import Structure, SpeciesData, OrbitalBasis, BlockKey
from .reciprocal import PeriodicField
from .grid_collocation import _cartesian_box_index_bounds
from .radial import OrbitalEvaluator

from .precompiled import load, verify
clg = load('_cuda_local_grid')


class CudaSpeciesCache:
    """Reusable species-level tables on GPU across multiple structures and geometries.
    
    Computes uniform cubic spline coefficients once per orbital channel using SciPy,
    and uploads coefficients and descriptor metadata to GPU. Retained across geometry changes.
    """
    def __init__(self, species_data: Mapping[str, SpeciesData], device: str | torch.device = "cuda:0"):
        self.device = torch.device(device) if isinstance(device, str) else device
        with torch.cuda.device(self.device):
            verify('_cuda_local_grid', check_device=True)
        self.species_data = species_data
        self._tables: dict[str, dict[str, Any]] = {}
        self._build_tables()

    def _build_tables(self) -> None:
        for symbol, data in self.species_data.items():
            orb: OrbitalBasis = data.orb
            num_ch = len(orb.channels)
            n_int = len(orb.r) - 1
            dr = float(orb.dr)
            rcut = float(orb.rcut)

            c_arr = np.zeros((num_ch, 4, n_int), dtype=np.float64)
            for ic, ch in enumerate(orb.channels):
                cs = CubicSpline(orb.r, ch.radial, bc_type="not-a-knot", extrapolate=False)
                c_arr[ic] = cs.c

            descriptors = orb.descriptors()
            descs = np.array([[d.channel_index, d.l, d.m] for d in descriptors], dtype=np.int32)

            self._tables[symbol] = {
                "spline_coeffs": torch.tensor(c_arr, device=self.device, dtype=torch.float64),
                "descriptors": torch.tensor(descs, device=self.device, dtype=torch.int32),
                "dr": dr,
                "rcut": rcut,
                "n_intervals": n_int,
                "num_channels": num_ch,
                "norb": orb.norb,
            }

    def get(self, symbol: str) -> dict[str, Any]:
        return self._tables[symbol]

    def clear(self) -> None:
        self._tables.clear()


@dataclass
class CudaAtomSupport:
    """Non-zero orbital support and 3D lookup tensor for one atom on GPU."""
    atom_index: int
    species: str
    center: torch.Tensor          # [3] float64 on GPU
    integer_indices: torch.Tensor # [N, 3] int64 on GPU
    values: torch.Tensor          # [N, norb] float64 on GPU
    potential: torch.Tensor       # [N] float64 on GPU
    spin_z_potential: Optional[torch.Tensor] # [N] float64 on GPU (or None)
    lo: torch.Tensor              # [3] int64 on GPU
    hi: torch.Tensor              # [3] int64 on GPU
    lookup: torch.Tensor          # [D0, D1, D2] int32 on GPU
    npoints: int
    norb: int
    bytes_allocated: int


class CudaGeometrySupports:
    """Manages GPU atom supports for a specific structure and periodic grid.
    
    Invalidated and rebuilt on geometry change, while species tables are retained.
    Correctly supports atoms slightly outside the home cell by computing unwrapped
    Cartesian bounds and applying periodic reduction modulo the FFT shape.
    """
    def __init__(
        self,
        structure: Structure,
        species_cache: CudaSpeciesCache,
        field: PeriodicField,
        spin_z_field: Optional[PeriodicField] = None,
        *,
        max_bytes: Optional[int] = None,
        device: str | torch.device = "cuda:0",
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.structure = structure
        self.species_cache = species_cache
        self.field = field
        self.spin_z_field = spin_z_field
        self.max_bytes = max_bytes
        self.grid_shape = tuple(int(x) for x in field.shape)
        self.grid_weight = float(field.grid_weight)

        self.shape_t = torch.tensor(self.grid_shape, device=self.device, dtype=torch.int64)
        self.cell_t = torch.tensor(field.cell_bohr, device=self.device, dtype=torch.float64)
        self.field_values_t = torch.tensor(field.values_ry, device=self.device, dtype=torch.float64)
        self.has_spin_z = (spin_z_field is not None)
        self.spin_z_values_t = (
            torch.tensor(spin_z_field.values_ry, device=self.device, dtype=torch.float64)
            if self.has_spin_z else torch.empty(0, device=self.device, dtype=torch.float64)
        )

        self.supports: list[CudaAtomSupport] = []
        self.total_bytes = 0
        self.home_positions = structure.cart_positions
        self._build_all_supports()

    def _build_all_supports(self) -> None:
        if clg is None:
            raise RuntimeError("C++/CUDA extension _cuda_local_grid is not compiled/available.")

        dummy_spin = torch.empty(0, device=self.device, dtype=torch.float64)
        for i, atom in enumerate(self.structure.atoms):
            sp = self.species_cache.get(atom.species)
            center = self.home_positions[i]
            nmin, nmax = _cartesian_box_index_bounds(self.field, center - sp["rcut"], center + sp["rcut"])
            counts = nmax - nmin + 1

            center_t = torch.tensor(center, device=self.device, dtype=torch.float64)
            nmin_t = torch.tensor(nmin, device=self.device, dtype=torch.int64)
            counts_t = torch.tensor(counts, device=self.device, dtype=torch.int32)

            res = clg.build_atom_support_cuda(
                center_t, sp["rcut"], nmin_t, counts_t, self.shape_t, self.cell_t,
                sp["dr"], sp["n_intervals"], sp["num_channels"], sp["spline_coeffs"],
                sp["norb"], sp["descriptors"], self.field_values_t,
                self.has_spin_z, self.spin_z_values_t if self.has_spin_z else dummy_spin
            )

            integer_indices = res[0]
            values = res[1]
            potential = res[2]
            spin_z_pot = res[3] if self.has_spin_z else None
            lo = res[4]
            hi = res[5]
            lookup = res[6]

            npts = integer_indices.size(0)
            sz = (
                integer_indices.nbytes + values.nbytes + potential.nbytes +
                lookup.nbytes + (spin_z_pot.nbytes if spin_z_pot is not None else 0)
            )

            if self.max_bytes is not None and (self.total_bytes + sz > self.max_bytes):
                raise MemoryError(
                    f"CudaGeometrySupports exceeded memory budget of {self.max_bytes / (1024**2):.1f} MB"
                )

            support = CudaAtomSupport(
                atom_index=i,
                species=atom.species,
                center=center_t,
                integer_indices=integer_indices,
                values=values,
                potential=potential,
                spin_z_potential=spin_z_pot,
                lo=lo,
                hi=hi,
                lookup=lookup,
                npoints=npts,
                norb=sp["norb"],
                bytes_allocated=sz,
            )
            self.supports.append(support)
            self.total_bytes += sz

    def clear(self) -> None:
        self.supports.clear()
        self.total_bytes = 0


class CudaPeriodicFFTGridAOCache:
    """Drop-in CUDA compatibility adapter matching the PeriodicFFTGridAOCache contract.
    
    Provides single-pair and batched contraction with exact numerical agreement,
    reusing GPU-resident species and geometry supports.
    """
    def __init__(
        self,
        field: PeriodicField,
        *,
        spin_z_field: Optional[PeriodicField] = None,
        max_bytes: Optional[int] = 4096 * 1024**2,
        chunk_size: int = 100_000,
        device: str | torch.device = "cuda:0",
    ):
        self.field = field
        self.spin_z_field = spin_z_field
        self.max_bytes = max_bytes
        self.chunk_size = chunk_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.torch = torch

        self.shape_t = torch.tensor(field.shape, device=self.device, dtype=torch.int64)
        self.cell_t = torch.tensor(field.cell_bohr, device=self.device, dtype=torch.float64)
        self.field_values_t = torch.tensor(field.values_ry, device=self.device, dtype=torch.float64)
        self.has_spin_z = (spin_z_field is not None)
        self.spin_z_values_t = (
            torch.tensor(spin_z_field.values_ry, device=self.device, dtype=torch.float64)
            if self.has_spin_z else torch.empty(0, device=self.device, dtype=torch.float64)
        )

        self._species_cache: dict[int, dict[str, Any]] = {}
        self._anchors: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self.resident = 0
        self._stats = {
            "anchor_builds": 0,
            "anchor_hits": 0,
            "evictions": 0,
            "peak_bytes": 0,
            "anchor_seconds": 0.0,
            "contraction_seconds": 0.0,
            "pairs": 0,
            "points": 0,
        }

    def _get_species_table(self, evaluator: OrbitalEvaluator) -> dict[str, Any]:
        key = id(evaluator.basis)
        if key in self._species_cache:
            return self._species_cache[key]

        orb = evaluator.basis
        num_ch = len(orb.channels)
        n_int = len(orb.r) - 1
        dr = float(orb.dr)
        rcut = float(orb.rcut)

        stored = orb.metadata.get('offline_spline_coefficients')
        if stored is None:
            if 'offline_source_sha256' in orb.metadata:
                raise ValueError('Offline AO spline coefficients missing; explicitly prepare species')
            c_arr = np.stack([CubicSpline(orb.r, ch.radial, bc_type="not-a-knot", extrapolate=False).c for ch in orb.channels])
        else:
            c_arr = np.asarray(stored, dtype=np.float64)
            if c_arr.shape != (num_ch, 4, n_int) or not np.isfinite(c_arr).all():
                raise ValueError('Invalid offline AO spline coefficients')

        descriptors = orb.descriptors()
        descs = np.array([[d.channel_index, d.l, d.m] for d in descriptors], dtype=np.int32)

        entry = {
            "spline_coeffs": torch.tensor(c_arr, device=self.device, dtype=torch.float64),
            "descriptors": torch.tensor(descs, device=self.device, dtype=torch.int32),
            "dr": dr,
            "rcut": rcut,
            "n_intervals": n_int,
            "num_channels": num_ch,
            "norb": orb.norb,
        }
        self._species_cache[key] = entry
        return entry

    def _anchor(self, evaluator: OrbitalEvaluator, center: np.ndarray) -> dict[str, Any]:
        key = (id(evaluator), np.asarray(center, dtype=np.float64).tobytes())
        if key in self._anchors:
            self._stats["anchor_hits"] += 1
            self._anchors.move_to_end(key)
            return self._anchors[key]

        start = time.perf_counter()
        sp = self._get_species_table(evaluator)
        center_np = np.asarray(center, dtype=np.float64)
        nmin, nmax = _cartesian_box_index_bounds(self.field, center_np - sp["rcut"], center_np + sp["rcut"])
        counts = nmax - nmin + 1

        center_t = torch.tensor(center_np, device=self.device, dtype=torch.float64)
        nmin_t = torch.tensor(nmin, device=self.device, dtype=torch.int64)
        counts_t = torch.tensor(counts, device=self.device, dtype=torch.int32)
        dummy_spin = torch.empty(0, device=self.device, dtype=torch.float64)

        res = clg.build_atom_support_cuda(
            center_t, sp["rcut"], nmin_t, counts_t, self.shape_t, self.cell_t,
            sp["dr"], sp["n_intervals"], sp["num_channels"], sp["spline_coeffs"],
            sp["norb"], sp["descriptors"], self.field_values_t,
            self.has_spin_z, self.spin_z_values_t if self.has_spin_z else dummy_spin
        )

        integer_indices = res[0]
        values = res[1]
        potential = res[2]
        spin_z_pot = res[3] if self.has_spin_z else None
        lo = res[4]
        hi = res[5]
        lookup = res[6]

        npts = integer_indices.size(0)
        size = int(
            integer_indices.nbytes + values.nbytes + potential.nbytes +
            lookup.nbytes + (spin_z_pot.nbytes if spin_z_pot is not None else 0)
        )

        entry = {
            "lo": lo,
            "hi": hi,
            "lookup": lookup,
            "values": values,
            "potential": potential,
            "spin_z_potential": spin_z_pot,
            "bytes": size,
            "npoints": npts,
        }

        self._stats["anchor_builds"] += 1
        self._stats["anchor_seconds"] += time.perf_counter() - start

        if self.max_bytes is None or size <= self.max_bytes:
            while self._anchors and self.max_bytes is not None and self.resident + size > self.max_bytes:
                _, old = self._anchors.popitem(last=False)
                self.resident -= old["bytes"]
                self._stats["evictions"] += 1
            self._anchors[key] = entry
            self.resident += size
            self._stats["peak_bytes"] = max(self._stats["peak_bytes"], self.resident)

        return entry

    def contract_pair(self, left, right, center_i, center_j_home, image) -> np.ndarray:
        """Exact GPU contract_pair matching PeriodicFFTGridAOCache."""
        image = np.asarray(image)
        if image.shape != (3,) or not np.array_equal(image, np.rint(image)):
            raise ValueError("Explicit integer lattice image required")
        center_j = np.asarray(center_j_home) + image @ self.field.cell_bohr
        self._stats["pairs"] += 1

        if np.linalg.norm(np.asarray(center_i) - center_j) > left.basis.rcut + right.basis.rcut:
            return np.zeros((left.norb, right.norb))

        a = self._anchor(left, center_i)
        b = self._anchor(right, center_j_home)

        start = time.perf_counter()
        shift = torch.tensor(image.astype(np.int64), device=self.device) * self.shape_t
        dummy_spin = torch.empty(0, device=self.device, dtype=torch.float64)

        res = clg.contract_single_pair_cuda(
            a["lo"], a["hi"], a["lookup"], a["values"], a["potential"],
            False, dummy_spin,
            b["lo"], b["hi"], b["lookup"], b["values"],
            shift, float(self.field.grid_weight), left.norb, right.norb
        )
        out = res[0].cpu().numpy()
        self._stats["contraction_seconds"] += time.perf_counter() - start
        return out

    def contract_pairs_batch(
        self,
        pairs: Sequence[tuple[int, int, tuple[int, int, int], Any, Any]],
        atoms_species: Sequence[str],
        evaluators: Mapping[str, OrbitalEvaluator],
        home_positions: np.ndarray,
    ) -> tuple[dict[BlockKey, np.ndarray], Optional[dict[BlockKey, np.ndarray]]]:
        """Contract all pairs in batch entirely on GPU without per-pair Python overhead."""
        if not pairs:
            return {}, {} if self.has_spin_z else None

        # Ensure all home anchors are built
        anchors = [self._anchor(evaluators[sp], home_positions[idx]) for idx, sp in enumerate(atoms_species)]

        anchors_lo = [a["lo"] for a in anchors]
        anchors_hi = [a["hi"] for a in anchors]
        anchors_lookup = [a["lookup"] for a in anchors]
        anchors_values = [a["values"] for a in anchors]
        anchors_potential = [a["potential"] for a in anchors]
        dummy_spin = torch.empty(0, device=self.device, dtype=torch.float64)
        anchors_spin_z = [a["spin_z_potential"] if self.has_spin_z else dummy_spin for a in anchors]

        pair_i = torch.tensor([p[0] for p in pairs], device=self.device, dtype=torch.int32)
        pair_j = torch.tensor([p[1] for p in pairs], device=self.device, dtype=torch.int32)
        pair_R = torch.tensor([p[2] for p in pairs], device=self.device, dtype=torch.int64)
        atom_norbs = [evaluators[sp].norb for sp in atoms_species]

        start = time.perf_counter()
        batch_res = clg.contract_pairs_batch_cuda(
            anchors_lo, anchors_hi, anchors_lookup, anchors_values, anchors_potential,
            self.has_spin_z, anchors_spin_z,
            pair_i, pair_j, pair_R, self.shape_t, float(self.field.grid_weight), atom_norbs
        )
        self._stats["contraction_seconds"] += time.perf_counter() - start
        self._stats["pairs"] += len(pairs)

        v_blocks: dict[BlockKey, np.ndarray] = {}
        vz_blocks: dict[BlockKey, np.ndarray] = {} if self.has_spin_z else None

        v_tensors = batch_res[0]
        vz_tensors = batch_res[1] if self.has_spin_z else []

        for p_idx, p in enumerate(pairs):
            key = BlockKey(p[0], p[1], tuple(int(x) for x in p[2]))
            v_blocks[key] = v_tensors[p_idx].cpu().numpy()
            if self.has_spin_z:
                vz_blocks[key] = vz_tensors[p_idx].cpu().numpy()

        return v_blocks, vz_blocks

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "resident_bytes": self.resident,
            "device": str(self.device),
            "backend": "cuda_local_grid_csrc",
            "identity": "canonical atom plus supplied integer lattice image; no rounded centers",
        }

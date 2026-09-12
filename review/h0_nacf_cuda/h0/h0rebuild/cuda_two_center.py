"""Compiled generic CUDA accelerated Two-Center and Nonlocal Integrator backend.

Eliminates per-orbital Python loops in S, T, and Nonlocal KB assembly.
Uses official ABACUS radial collections and spherical Bessel transforms to tabulate
radial tables on CPU once (~0.2s), then moves Hermite cubic polynomial coefficients
and Gaunt coefficients to GPU for batched parallel evaluation.
"""
from collections import Counter
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .models import abacus_m_order

from .precompiled import load, verify
_C = load('_cuda_two_center')


class CUDATwoCenter:
    """Compiled CUDA two-center and nonlocal assembly backend with immutable species identity."""

    def __init__(
        self,
        species_data: dict,
        *,
        dr_bohr: float = 0.01,
        cache_dir: Optional[str] = None,
        nspin: int = 1,
        device: str = "cuda",
    ):
        if dr_bohr <= 0:
            raise ValueError("two-center grid step must be positive")
        if nspin not in (1, 4):
            raise ValueError(f"nspin must be 1 or 4, got {nspin}")
        if nspin == 1 and any(d.upf.has_so for d in species_data.values()):
            raise ValueError("The scalar two-center path requires scalarized UPFs")

        if any(c.l > 4 for d in species_data.values() for c in d.orb.channels) or any(p.l > 4 for d in species_data.values() for p in d.upf.projectors):
            raise NotImplementedError("CUDA two-center supports orbital/projector l <= 4; unsupported channels must not be silently truncated")
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            verify('_cuda_two_center', check_device=True)
        self.nspin = nspin
        self.species = list(species_data.keys())
        self.species_tuple = tuple(self.species)
        self.index = {s: i for i, s in enumerate(self.species)}
        self.sd = species_data

        t0_prep = time.perf_counter()

        from pyabacus import ModuleNAO as nao
        from pyabacus import ModuleBase as base
        import pyabacus

        self.desc = {"orb": {}, "proj": {}}
        self.collections = {}
        self.d = {}
        self.sbt = base.SphericalBesselTransformer()

        # Build radial collections using temporary directory for layout files
        with tempfile.TemporaryDirectory(prefix="h0flash-cuda-two-center-", dir=cache_dir) as tmp_dir:
            tmp_path = Path(tmp_dir)
            files = []
            for symbol, data in species_data.items():
                if data.orb.source is None:
                    raise ValueError("Official ORB source required for pyabacus collection layout")
                files.append(str(data.orb.source))
                self.desc["orb"][symbol] = [
                    (x.l, x.zeta, m) for x in data.orb.channels for m in abacus_m_order(x.l)
                ]
            orb = nao.RadialCollection()
            orb.build(len(files), files, "o")
            for symbol, data in species_data.items():
                for c in data.orb.channels:
                    orb(self.index[symbol], c.l, c.zeta).build(
                        c.l, True, len(data.orb.r), data.orb.r, c.radial, 0, c.zeta, symbol, self.index[symbol], False
                    )
            self.collections["orb"] = orb

            files = []
            for symbol, data in species_data.items():
                upf = data.upf
                counts = Counter(p.l for p in upf.projectors)
                if not counts:
                    raise NotImplementedError("pyabacus KB layout requires at least one projector per species")
                lmax = max(counts)
                cutoff = upf.max_projector_cutoff
                n = max(5, int(np.ceil(cutoff / 0.01)) + 1)
                step = cutoff / (n - 1)
                rg = np.arange(n) * step
                header = [
                    "Element " + symbol,
                    "Energy Cutoff(Ry) 100",
                    f"Radius Cutoff(a.u.) {cutoff:.17g}",
                    f"Lmax {lmax}",
                ]
                header += [f'Number of {"SPDFGH"[l]}orbital--> {counts[l]}' for l in range(lmax + 1)]
                header += ["SUMMARY END", f"Mesh {n}", f"dr {step:.17g}"]
                for l in range(lmax + 1):
                    for z in range(counts[l]):
                        header += [
                            "Type L N",
                            f"0 {l} {z}",
                            " ".join(f"{x:.17g}" for x in rg**l * np.exp(-rg) * (1 - rg / cutoff) ** 2),
                        ]
                layout_path = tmp_path / f"{symbol}.projector-layout.orb"
                layout_path.write_text("\n".join(header) + "\n")
                files.append(str(layout_path))

            proj = nao.RadialCollection()
            proj.build(len(files), files, "o")
            for symbol, data in species_data.items():
                upf = data.upf
                seen = Counter()
                slots = {}
                descriptors = []
                for p in upf.projectors:
                    z = seen[p.l]
                    seen[p.l] += 1
                    slots[p.index] = (p.l, z)
                    radial_u = p.radial_u.copy()
                    radial_u[p.cutoff_index:] = 0.0
                    proj(self.index[symbol], p.l, z).build(
                        p.l, True, len(upf.r), upf.r, radial_u, 1, z, symbol, self.index[symbol], False
                    )
                for l in sorted(seen):
                    for z in range(seen[l]):
                        descriptors += [(l, z, m) for m in abacus_m_order(l)]
                self.desc["proj"][symbol] = descriptors

                d = np.zeros((len(descriptors), len(descriptors)), dtype=np.float64)
                desc_to_idx = {x: i for i, x in enumerate(descriptors)}
                for p in upf.projectors:
                    for q in upf.projectors:
                        if p.l != q.l:
                            continue
                        for m in abacus_m_order(p.l):
                            i = desc_to_idx[(*slots[p.index], m)]
                            j = desc_to_idx[(*slots[q.index], m)]
                            d[i, j] = upf.dij_ry[p.index, q.index]
                if nspin == 4:
                    from .pyabacus_integrals import spinor_projector_matrix
                    self.d[symbol] = spinor_projector_matrix(upf, slots, descriptors)
                else:
                    self.d[symbol] = d
            self.collections["proj"] = proj

        # Grid setup and tabulations
        self.cutoff = 2.0 * max(max(d.orb.rcut, d.upf.max_projector_cutoff) for d in species_data.values())
        self.nr = int(np.ceil(self.cutoff / dr_bohr)) + 1
        self.dr = self.cutoff / (self.nr - 1)
        for collection in self.collections.values():
            collection.set_transformer(self.sbt)
            collection.set_uniform_grid(True, self.nr, self.cutoff, "i", True)

        self.integrators = {}
        for left, right, op in [("orb", "orb", "S"), ("orb", "orb", "T"), ("proj", "orb", "S")]:
            integrator = nao.TwoCenterIntegrator()
            integrator.tabulate(self.collections[left], self.collections[right], op, self.nr, self.cutoff)
            self.integrators[(left, right, op)] = integrator

        # Extract tables and coefficients into PyTorch tensors
        s_data = _C.extract_radial_table(self.integrators[("orb", "orb", "S")])
        t_data = _C.extract_radial_table(self.integrators[("orb", "orb", "T")])
        q_data = _C.extract_radial_table(self.integrators[("proj", "orb", "S")])

        lmax_orb = max(c.l for d in species_data.values() for c in d.orb.channels)
        lmax_proj = max(p.l for d in species_data.values() for p in d.upf.projectors)
        self.lmax = max(lmax_orb, lmax_proj, 4)
        gaunt_cpu = _C.extract_gaunt_table(self.lmax)

        # Prepare GPU tables
        self.S_coeffs = s_data["coeffs"].to(device=self.device, non_blocking=True)
        self.T_coeffs = t_data["coeffs"].to(device=self.device, non_blocking=True)
        self.Q_coeffs = q_data["coeffs"].to(device=self.device, non_blocking=True)
        self.S_index_map = s_data["index_map"].to(device=self.device, non_blocking=True)
        self.T_index_map = t_data["index_map"].to(device=self.device, non_blocking=True)
        self.Q_index_map = q_data["index_map"].to(device=self.device, non_blocking=True)
        self.gaunt_table = gaunt_cpu.to(device=self.device, non_blocking=True)

        self.index_map_strides = list(self.S_index_map.stride())
        self.S_index_map_strides = list(self.S_index_map.stride())
        self.Q_index_map_strides = list(self.Q_index_map.stride())
        self.gaunt_dims = list(self.gaunt_table.size())

        # Flattened descriptor arrays for orbitals
        orb_l_list, orb_zeta_list, orb_m_list = [], [], []
        species_orb_offsets = [0]
        for s in self.species:
            descs = self.desc["orb"][s]
            for l, z, m in descs:
                orb_l_list.append(l)
                orb_zeta_list.append(z)
                orb_m_list.append(m)
            species_orb_offsets.append(len(orb_l_list))

        self.orb_l = torch.tensor(orb_l_list, dtype=torch.int32, device=self.device)
        self.orb_zeta = torch.tensor(orb_zeta_list, dtype=torch.int32, device=self.device)
        self.orb_m = torch.tensor(orb_m_list, dtype=torch.int32, device=self.device)
        self.species_orb_offsets = torch.tensor(species_orb_offsets, dtype=torch.int32, device=self.device)

        # Flattened descriptor arrays for projectors
        proj_l_list, proj_zeta_list, proj_m_list = [], [], []
        species_proj_offsets = [0]
        for s in self.species:
            descs = self.desc["proj"][s]
            for l, z, m in descs:
                proj_l_list.append(l)
                proj_zeta_list.append(z)
                proj_m_list.append(m)
            species_proj_offsets.append(len(proj_l_list))

        self.proj_l = torch.tensor(proj_l_list, dtype=torch.int32, device=self.device)
        self.proj_zeta = torch.tensor(proj_zeta_list, dtype=torch.int32, device=self.device)
        self.proj_m = torch.tensor(proj_m_list, dtype=torch.int32, device=self.device)
        self.species_proj_offsets = torch.tensor(species_proj_offsets, dtype=torch.int32, device=self.device)

        self.max_norb = max(len(self.desc["orb"][s]) for s in self.species)
        self.max_nproj = max(len(self.desc["proj"][s]) for s in self.species)
        self.norb_per_species = {s: len(self.desc["orb"][s]) for s in self.species}
        self.nproj_per_species = {s: len(self.desc["proj"][s]) for s in self.species}

        # Padded D matrices for GPU assembly
        d_dim = 2 * self.max_nproj if nspin == 4 else self.max_nproj
        d_dtype = torch.complex128 if nspin == 4 else torch.float64
        d_padded = torch.zeros((len(self.species), d_dim, d_dim), dtype=d_dtype)
        for s, idx in self.index.items():
            np_s = self.nproj_per_species[s]
            d_mat = self.d[s]
            if nspin == 1:
                d_padded[idx, :np_s, :np_s] = torch.from_numpy(d_mat)
            else:
                # d_mat is [2 * np_s, 2 * np_s] with up in [:np_s] and down in [np_s:]
                # We place:
                # up-up in [:np_s, :np_s]
                # up-down in [:np_s, max_nproj:max_nproj+np_s]
                # down-up in [max_nproj:max_nproj+np_s, :np_s]
                # down-down in [max_nproj:max_nproj+np_s, max_nproj:max_nproj+np_s]
                d_t = torch.from_numpy(d_mat)
                d_padded[idx, :np_s, :np_s] = d_t[:np_s, :np_s]
                d_padded[idx, :np_s, self.max_nproj : self.max_nproj + np_s] = d_t[:np_s, np_s:]
                d_padded[idx, self.max_nproj : self.max_nproj + np_s, :np_s] = d_t[np_s:, :np_s]
                d_padded[idx, self.max_nproj : self.max_nproj + np_s, self.max_nproj : self.max_nproj + np_s] = d_t[np_s:, np_s:]

        self.D_padded_cuda = d_padded.to(device=self.device, non_blocking=True)

        t_prep_end = time.perf_counter()
        self.timing = {
            "prep_seconds": t_prep_end - t0_prep,
            "kernel_two_center_seconds": 0.0,
            "kernel_nonlocal_seconds": 0.0,
            "kernel_two_center_calls": 0,
            "kernel_nonlocal_calls": 0,
        }

        self.metadata = {
            "backend": "CUDA Two-Center / Nonlocal Accelerator",
            "device": str(self.device),
            "dr_bohr": self.dr,
            "table_radius_bohr": self.cutoff,
            "radial_points": self.nr,
            "lmax": self.lmax,
            "setup_seconds": self.timing["prep_seconds"],
            "nspin": self.nspin,
        }

    def _check_species(self, symbol: str) -> int:
        if symbol not in self.index:
            raise KeyError(f"Species '{symbol}' not recognized by prepared CUDATwoCenter backend (known: {self.species})")
        return self.index[symbol]

    def eval_two_center_batch(
        self,
        pair_symbols: List[Tuple[str, str]],
        displacements: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched GPU evaluation of two-center overlap S and kinetic T.

        Args:
            pair_symbols: list of (symbol_1, symbol_2) tuples of length N_pairs.
            displacements: [N_pairs, 3] array or tensor of displacement vectors (R = c2 - c1).

        Returns:
            (out_S, out_T): PyTorch tensors of shape [N_pairs, max_norb, max_norb] on GPU.
        """
        n_pairs = len(pair_symbols)
        if n_pairs == 0:
            empty = torch.empty((0, self.max_norb, self.max_norb), dtype=torch.float64, device=self.device)
            return empty, empty

        if not isinstance(displacements, torch.Tensor):
            disp_tensor = torch.tensor(displacements, dtype=torch.float64, device=self.device)
        else:
            disp_tensor = displacements.to(dtype=torch.float64, device=self.device)

        s1_indices = [self._check_species(s1) for s1, s2 in pair_symbols]
        s2_indices = [self._check_species(s2) for s1, s2 in pair_symbols]
        pair_s1 = torch.tensor(s1_indices, dtype=torch.int32, device=self.device)
        pair_s2 = torch.tensor(s2_indices, dtype=torch.int32, device=self.device)

        t0 = time.perf_counter()
        out_S, out_T = _C.eval_two_center_batch(
            disp_tensor,
            pair_s1,
            pair_s2,
            self.S_coeffs,
            self.T_coeffs,
            self.S_index_map,
            self.T_index_map,
            self.index_map_strides,
            self.gaunt_table,
            self.gaunt_dims,
            self.orb_l,
            self.orb_zeta,
            self.orb_m,
            self.species_orb_offsets,
            self.dr,
            self.cutoff,
            self.nr,
            self.max_norb,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        self.timing["kernel_two_center_seconds"] += dt
        self.timing["kernel_two_center_calls"] += 1

        return out_S, out_T

    def scalar_pair(
        self,
        symbol_i: str,
        symbol_j: str,
        ci: Union[List[float], np.ndarray],
        cj: Union[List[float], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-case adapter matching PyAbacusTwoCenter.scalar_pair(symbol_i, symbol_j, ci, cj)."""
        disp = np.asarray(cj, dtype=np.float64) - np.asarray(ci, dtype=np.float64)
        disp_tensor = torch.from_numpy(disp).reshape(1, 3).to(self.device)
        out_S, out_T = self.eval_two_center_batch([(symbol_i, symbol_j)], disp_tensor)
        ni = self.norb_per_species[symbol_i]
        nj = self.norb_per_species[symbol_j]
        s_np = out_S[0, :ni, :nj].cpu().numpy()
        t_np = out_T[0, :ni, :nj].cpu().numpy()
        return s_np, t_np

    def overlap(
        self,
        left: str,
        symbol_left: str,
        right: str,
        symbol_right: str,
        r: Union[List[float], np.ndarray],
        op: str = "S",
    ) -> np.ndarray:
        """Single-case adapter matching PyAbacusTwoCenter.overlap."""
        disp = np.asarray(r, dtype=np.float64)
        disp_tensor = torch.from_numpy(disp).reshape(1, 3).to(self.device)

        if left == "orb" and right == "orb":
            out_S, out_T = self.eval_two_center_batch([(symbol_left, symbol_right)], disp_tensor)
            ni = self.norb_per_species[symbol_left]
            nj = self.norb_per_species[symbol_right]
            res = out_S if op == "S" else out_T
            return res[0, :ni, :nj].cpu().numpy()
        elif left == "proj" and right == "orb" and op == "S":
            sp = self._check_species(symbol_left)
            so = self._check_species(symbol_right)
            p_tensor = torch.tensor([sp], dtype=torch.int32, device=self.device)
            o_tensor = torch.tensor([so], dtype=torch.int32, device=self.device)
            out_Q = _C.eval_projector_overlap_batch(
                disp_tensor,
                p_tensor,
                o_tensor,
                self.Q_coeffs,
                self.Q_index_map,
                self.Q_index_map_strides,
                self.gaunt_table,
                self.gaunt_dims,
                self.proj_l,
                self.proj_zeta,
                self.proj_m,
                self.species_proj_offsets,
                self.orb_l,
                self.orb_zeta,
                self.orb_m,
                self.species_orb_offsets,
                self.dr,
                self.cutoff,
                self.nr,
                self.max_nproj,
                self.max_norb,
            )
            np_p = self.nproj_per_species[symbol_left]
            norb_o = self.norb_per_species[symbol_right]
            return out_Q[0, :np_p, :norb_o].cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported overlap combination: left={left}, right={right}, op={op}")

    def nonlocal_block(
        self,
        structure,
        symbol_i: str,
        symbol_j: str,
        ci: Union[List[float], np.ndarray],
        cj: Union[List[float], np.ndarray],
        candidates: List[Tuple[int, np.ndarray]],
    ) -> np.ndarray:
        """Single-case adapter matching PyAbacusTwoCenter.nonlocal_block."""
        ni = self.norb_per_species[symbol_i]
        nj = self.norb_per_species[symbol_j]
        mult = 2 if self.nspin == 4 else 1

        ci_arr = np.asarray(ci, dtype=np.float64)
        cj_arr = np.asarray(cj, dtype=np.float64)
        rcut_i = self.sd[symbol_i].orb.rcut
        rcut_j = self.sd[symbol_j].orb.rcut

        # Filter candidate projectors by cutoff radius
        active_candidates = []
        for atom_index, pcenter in candidates:
            p_arr = np.asarray(pcenter, dtype=np.float64)
            symbol_p = structure.atoms[atom_index].species
            cutoff_p = self.sd[symbol_p].upf.max_projector_cutoff
            di = np.linalg.norm(ci_arr - p_arr)
            if di > cutoff_p + rcut_i:
                continue
            dj = np.linalg.norm(cj_arr - p_arr)
            if dj > cutoff_p + rcut_j:
                continue
            active_candidates.append((atom_index, symbol_p, p_arr))

        n_active = len(active_candidates)
        if n_active == 0:
            dtype = np.complex128 if self.nspin == 4 else np.float64
            return np.zeros((mult * ni, mult * nj), dtype=dtype)

        # Prepare batched projector overlaps Q_i and Q_j
        disp_i = np.empty((n_active, 3), dtype=np.float64)
        disp_j = np.empty((n_active, 3), dtype=np.float64)
        proj_species = []
        orb_species_i = [self._check_species(symbol_i)] * n_active
        orb_species_j = [self._check_species(symbol_j)] * n_active

        for idx, (atom_index, symbol_p, p_arr) in enumerate(active_candidates):
            disp_i[idx] = ci_arr - p_arr
            disp_j[idx] = cj_arr - p_arr
            proj_species.append(self._check_species(symbol_p))

        disp_i_t = torch.from_numpy(disp_i).to(self.device)
        disp_j_t = torch.from_numpy(disp_j).to(self.device)
        proj_sp_t = torch.tensor(proj_species, dtype=torch.int32, device=self.device)
        orb_sp_i_t = torch.tensor(orb_species_i, dtype=torch.int32, device=self.device)
        orb_sp_j_t = torch.tensor(orb_species_j, dtype=torch.int32, device=self.device)

        t0 = time.perf_counter()
        Q_i = _C.eval_projector_overlap_batch(
            disp_i_t,
            proj_sp_t,
            orb_sp_i_t,
            self.Q_coeffs,
            self.Q_index_map,
            self.Q_index_map_strides,
            self.gaunt_table,
            self.gaunt_dims,
            self.proj_l,
            self.proj_zeta,
            self.proj_m,
            self.species_proj_offsets,
            self.orb_l,
            self.orb_zeta,
            self.orb_m,
            self.species_orb_offsets,
            self.dr,
            self.cutoff,
            self.nr,
            self.max_nproj,
            self.max_norb,
        )

        Q_j = _C.eval_projector_overlap_batch(
            disp_j_t,
            proj_sp_t,
            orb_sp_j_t,
            self.Q_coeffs,
            self.Q_index_map,
            self.Q_index_map_strides,
            self.gaunt_table,
            self.gaunt_dims,
            self.proj_l,
            self.proj_zeta,
            self.proj_m,
            self.species_proj_offsets,
            self.orb_l,
            self.orb_zeta,
            self.orb_m,
            self.species_orb_offsets,
            self.dr,
            self.cutoff,
            self.nr,
            self.max_nproj,
            self.max_norb,
        )

        # Slice Q_i and Q_j to relevant orbital counts
        Q_i_sliced = Q_i[:, :, :ni].contiguous()
        Q_j_sliced = Q_j[:, :, :nj].contiguous()

        # Gather D matrices for candidates
        D_cand = self.D_padded_cuda[proj_sp_t]
        cand_active = torch.ones(n_active, dtype=torch.bool, device=self.device)
        cand_nproj = torch.tensor([self.nproj_per_species[self.species[sp]] for sp in proj_species], dtype=torch.int32, device=self.device)

        out_Vnl = _C.assemble_nonlocal_candidates(
            Q_i_sliced,
            Q_j_sliced,
            D_cand,
            cand_active,
            cand_nproj,
            ni,
            nj,
            self.nspin,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        self.timing["kernel_nonlocal_seconds"] += dt
        self.timing["kernel_nonlocal_calls"] += 1

        return out_Vnl.cpu().numpy()

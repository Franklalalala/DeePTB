from __future__ import annotations

"""Transparent operation-count model for offline ABACUS-style H[rho0] rebuilds.

This module intentionally estimates *work and memory*, not wall-clock time.  The
same operation count can differ by orders of magnitude between pure Python,
vectorized BLAS, CPU C++, and GPU implementations.  The model is meant for
route selection and profiler planning; production decisions must use measured
stage timings on the exact Si/Ca inputs and FFT shape.

The exact FFT-grid point density should be supplied through ``fft_grid_points``
and ``cell_volume_bohr3`` whenever possible.  The fallback
``h ~= pi/sqrt(ecutrho_ry)`` is only a Nyquist planning estimate for a density
cutoff expressed in Rydberg atomic units.
"""

from dataclasses import asdict, dataclass
import json
import math
from typing import Any


_FLOAT64_BYTES = 8
_INT64_XYZ_BYTES = 3 * 8


def human_bytes(value: float) -> str:
    value = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.3g} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def human_count(value: float) -> str:
    value = float(value)
    for scale, suffix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.3g}"


@dataclass(frozen=True)
class H0CostInputs:
    """Inputs for an auditable, hardware-independent cost estimate.

    Counts refer to *spatial* orbitals.  In a non-magnetic nspin=4 calculation,
    scalar local/S/T blocks can be formed spatially and duplicated in spin;
    projector-level SOC is accounted for separately through the effective
    projector-channel count.
    """

    n_atoms: int
    directed_h_blocks_per_atom: float
    spatial_orbitals_per_atom: int
    ao_cutoff_bohr: float

    # Prefer exact FFT density: prod(fft_shape) / cell volume.
    fft_grid_points: int | None = None
    cell_volume_bohr3: float | None = None
    ecutrho_ry: float | None = 100.0

    # Fraction of one AO support sphere contained in a typical AO-pair
    # intersection.  Equal 7-bohr spheres separated by ~4.4 bohr give ~0.54.
    pair_overlap_fraction: float = 0.50
    cached_support_images_per_atom: float = 2.0

    # Radial-table planning.
    n_species: int = 2
    radial_channels_per_species: int = 7
    radial_distance_points: int = 1500
    mean_angular_couplings_per_radial_pair: float = 3.0

    # KB projector planning.
    radial_projectors_per_center: int = 8
    mean_projector_angular_rows: float = 3.0
    projector_centers_per_basis_atom: float = 12.0
    projector_overlap_fraction: float = 0.20
    current_projector_overlap_reuse_factor: float = 8.0
    third_centers_per_h_block: float = 4.0

    # Pair-product low-rank planning.
    low_rank: int = 40

    # Representative 3c table dimensions.  These are not Hotcent internals;
    # they expose the generic N_r1 x N_r2 x N_theta storage law.
    three_center_radial_points: int = 160
    three_center_theta_points: int = 24
    three_center_scalar_channels_per_species_triplet: int = 100

    def validate(self) -> None:
        positive = {
            "n_atoms": self.n_atoms,
            "directed_h_blocks_per_atom": self.directed_h_blocks_per_atom,
            "spatial_orbitals_per_atom": self.spatial_orbitals_per_atom,
            "ao_cutoff_bohr": self.ao_cutoff_bohr,
            "cached_support_images_per_atom": self.cached_support_images_per_atom,
            "n_species": self.n_species,
            "radial_channels_per_species": self.radial_channels_per_species,
            "radial_distance_points": self.radial_distance_points,
            "mean_angular_couplings_per_radial_pair": self.mean_angular_couplings_per_radial_pair,
            "radial_projectors_per_center": self.radial_projectors_per_center,
            "mean_projector_angular_rows": self.mean_projector_angular_rows,
            "projector_centers_per_basis_atom": self.projector_centers_per_basis_atom,
            "current_projector_overlap_reuse_factor": self.current_projector_overlap_reuse_factor,
            "third_centers_per_h_block": self.third_centers_per_h_block,
            "three_center_radial_points": self.three_center_radial_points,
            "three_center_theta_points": self.three_center_theta_points,
            "three_center_scalar_channels_per_species_triplet": self.three_center_scalar_channels_per_species_triplet,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        for name, value in (
            ("pair_overlap_fraction", self.pair_overlap_fraction),
            ("projector_overlap_fraction", self.projector_overlap_fraction),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must satisfy 0 < value <= 1")
        if self.low_rank < 0:
            raise ValueError("low_rank must be non-negative")
        exact_grid = self.fft_grid_points is not None or self.cell_volume_bohr3 is not None
        if exact_grid:
            if self.fft_grid_points is None or self.cell_volume_bohr3 is None:
                raise ValueError(
                    "fft_grid_points and cell_volume_bohr3 must be supplied together"
                )
            if self.fft_grid_points <= 0 or self.cell_volume_bohr3 <= 0.0:
                raise ValueError("exact FFT grid inputs must be positive")
        elif self.ecutrho_ry is None or self.ecutrho_ry <= 0.0:
            raise ValueError(
                "supply exact FFT grid density or a positive ecutrho_ry planning fallback"
            )


@dataclass(frozen=True)
class H0CostEstimate:
    inputs: H0CostInputs
    grid_density_per_bohr3: float
    approximate_grid_spacing_bohr: float
    ao_support_grid_points: float
    pair_intersection_grid_points: float
    directed_h_blocks: float

    exact_local_cache_bytes: float
    exact_local_no_cache_ao_samples: float
    exact_local_cached_ao_samples: float
    exact_local_contraction_fmas: float

    direct_st_function_samples: float
    direct_st_contraction_fmas: float
    two_center_st_runtime_channel_values: float
    two_center_st_table_bytes: float

    unique_ao_projector_pairs: float
    ao_projector_grid_points: float
    current_direct_projector_dot_fmas: float
    cached_direct_projector_dot_fmas: float
    two_center_projector_runtime_channel_values: float
    two_center_projector_table_bytes: float
    nonlocal_algebra_fmas: float

    low_rank_reference_setup_flops: float
    low_rank_per_field_fmas: float
    low_rank_reference_crossover_fields: float | None
    low_rank_factor_bytes: float

    three_center_table_nodes: float
    three_center_table_bytes: float

    @property
    def cache_ao_sample_reduction(self) -> float:
        denom = self.exact_local_cached_ao_samples
        return self.exact_local_no_cache_ao_samples / denom if denom else math.inf

    @property
    def projector_overlap_reduction_from_cache(self) -> float:
        denom = self.cached_direct_projector_dot_fmas
        return self.current_direct_projector_dot_fmas / denom if denom else math.inf

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cache_ao_sample_reduction"] = self.cache_ao_sample_reduction
        out["projector_overlap_reduction_from_cache"] = (
            self.projector_overlap_reduction_from_cache
        )
        return out

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        crossover = (
            "not reached (per-field low-rank work >= exact)"
            if self.low_rank_reference_crossover_fields is None
            else f"~{self.low_rank_reference_crossover_fields:.1f} fields on the same geometry"
        )
        return "\n".join(
            [
                "# H0 rebuild planning estimate",
                "",
                "> Operation counts are hardware-independent planning quantities, not wall-clock predictions.",
                "",
                "## Geometry and grid",
                "",
                f"- Directed H blocks: **{self.directed_h_blocks:,.0f}**",
                f"- Approximate grid spacing: **{self.approximate_grid_spacing_bohr:.4f} bohr**",
                f"- AO support points: **{self.ao_support_grid_points:,.0f}** per centred support",
                f"- Typical AO-pair intersection: **{self.pair_intersection_grid_points:,.0f}** points",
                "",
                "## Exact full-local-potential contraction",
                "",
                f"- AO support cache: **{human_bytes(self.exact_local_cache_bytes)}**",
                f"- AO value samples without cache: **{human_count(self.exact_local_no_cache_ao_samples)}**",
                f"- AO value samples with cache: **{human_count(self.exact_local_cached_ao_samples)}**",
                f"- Ideal AO-evaluation reuse factor: **{self.cache_ao_sample_reduction:.2f}x**",
                f"- Dense local contraction: **{human_count(self.exact_local_contraction_fmas)} FMA**",
                "",
                "## S/T",
                "",
                f"- Current direct midpoint function samples: **{human_count(self.direct_st_function_samples)}**",
                f"- Current direct midpoint contractions: **{human_count(self.direct_st_contraction_fmas)} FMA**",
                f"- Two-centre runtime table values: **{human_count(self.two_center_st_runtime_channel_values)}**",
                f"- Estimated S/T radial-table storage: **{human_bytes(self.two_center_st_table_bytes)}**",
                "",
                "## Fully relativistic KB nonlocal term",
                "",
                f"- Unique AO-centre/projector-centre pairs: **{self.unique_ao_projector_pairs:,.0f}**",
                f"- Current repeated projector dot products: **{human_count(self.current_direct_projector_dot_fmas)} FMA**",
                f"- After exact overlap caching: **{human_count(self.cached_direct_projector_dot_fmas)} FMA**",
                f"- Reuse factor exposed by caching: **{self.projector_overlap_reduction_from_cache:.2f}x**",
                f"- Two-centre projector runtime values: **{human_count(self.two_center_projector_runtime_channel_values)}**",
                f"- Estimated AO-projector radial-table storage: **{human_bytes(self.two_center_projector_table_bytes)}**",
                f"- Remaining B†DB algebra: **{human_count(self.nonlocal_algebra_fmas)} FMA**",
                "",
                "## Pair-product low rank",
                "",
                f"- Reference per-pair SVD setup: **{human_count(self.low_rank_reference_setup_flops)} FLOP**",
                f"- Low-rank work per additional field: **{human_count(self.low_rank_per_field_fmas)} FMA**",
                f"- Reference-SVD crossover: **{crossover}**",
                f"- Per-geometry low-rank factors: **{human_bytes(self.low_rank_factor_bytes)}**",
                "",
                "## Generic 3c table storage",
                "",
                f"- Scalar table nodes: **{human_count(self.three_center_table_nodes)}**",
                f"- Float64 storage: **{human_bytes(self.three_center_table_bytes)}**",
                "",
                "The 3c number is a generic N_r1×N_r2×N_theta×channel estimate; it is not a claim about Hotcent's exact compressed layout.",
            ]
        )


def estimate_h0_cost(inputs: H0CostInputs) -> H0CostEstimate:
    inputs.validate()

    if inputs.fft_grid_points is not None:
        density = float(inputs.fft_grid_points) / float(inputs.cell_volume_bohr3)
        spacing = density ** (-1.0 / 3.0)
    else:
        spacing = math.pi / math.sqrt(float(inputs.ecutrho_ry))
        density = spacing ** -3

    support_volume = 4.0 * math.pi * float(inputs.ao_cutoff_bohr) ** 3 / 3.0
    support_points = support_volume * density
    pair_points = support_points * float(inputs.pair_overlap_fraction)
    blocks = float(inputs.n_atoms) * float(inputs.directed_h_blocks_per_atom)
    n = float(inputs.spatial_orbitals_per_atom)
    n2 = n * n
    support_count = float(inputs.n_atoms) * float(inputs.cached_support_images_per_atom)

    cache_bytes = support_count * support_points * (
        n * _FLOAT64_BYTES + _INT64_XYZ_BYTES
    )
    no_cache_samples = 2.0 * blocks * pair_points * n
    cached_samples = support_count * support_points * n
    local_fmas = blocks * pair_points * n2

    # Direct S: one matrix contraction.  Symmetric kinetic reference: two more.
    direct_st_samples = 4.0 * blocks * pair_points * n
    direct_st_fmas = 3.0 * blocks * pair_points * n2

    species_pairs = float(inputs.n_species * (inputs.n_species + 1) // 2)
    radial_pairs = float(inputs.radial_channels_per_species**2)
    st_table_bytes = (
        species_pairs
        * radial_pairs
        * float(inputs.mean_angular_couplings_per_radial_pair)
        * float(inputs.radial_distance_points)
        * 2.0  # S and T
        * _FLOAT64_BYTES
    )
    st_runtime_values = blocks * n2

    unique_ao_projector_pairs = (
        float(inputs.n_atoms) * float(inputs.projector_centers_per_basis_atom)
    )
    projector_points = support_points * float(inputs.projector_overlap_fraction)
    p = float(inputs.radial_projectors_per_center)
    mbar = float(inputs.mean_projector_angular_rows)
    unique_projector_fmas = unique_ao_projector_pairs * p * mbar * projector_points * n
    current_projector_fmas = (
        unique_projector_fmas * float(inputs.current_projector_overlap_reuse_factor)
    )
    projector_runtime_values = unique_ao_projector_pairs * p * mbar * n
    projector_table_bytes = (
        float(inputs.n_species**2)
        * float(inputs.radial_channels_per_species)
        * p
        * mbar
        * float(inputs.radial_distance_points)
        * _FLOAT64_BYTES
    )

    # Effective spinor projector rows P and spinor orbital dimension 2n.
    effective_p = p * mbar
    spin_n = 2.0 * n
    # D @ B_j plus B_i^† @ (...), per third centre and H block.
    nonlocal_fmas = (
        blocks
        * float(inputs.third_centers_per_h_block)
        * (effective_p * effective_p * spin_n + effective_p * spin_n * spin_n)
    )

    rank = float(min(inputs.low_rank, int(n2)))
    # Current reference builds a dense g x n^2 pair-product matrix and calls a
    # thin SVD.  4*g*m^2 + 8*m^3/3 is a standard leading-order planning model.
    m = n2
    svd_setup_per_pair = 4.0 * pair_points * m * m + (8.0 / 3.0) * m**3
    low_rank_setup_flops = blocks * svd_setup_per_pair
    low_rank_per_field_fmas = blocks * (pair_points * rank + rank * m)
    exact_per_field_flops = 2.0 * local_fmas
    low_per_field_flops = 2.0 * low_rank_per_field_fmas
    saving = exact_per_field_flops - low_per_field_flops
    crossover = low_rank_setup_flops / saving if saving > 0.0 else None
    low_rank_bytes = blocks * (
        pair_points * rank + rank + rank * m + pair_points * 3.0
    ) * _FLOAT64_BYTES

    # Ordered species triplets because centre roles are not interchangeable.
    species_triplets = float(inputs.n_species**3)
    three_center_nodes = (
        species_triplets
        * float(inputs.three_center_scalar_channels_per_species_triplet)
        * float(inputs.three_center_radial_points) ** 2
        * float(inputs.three_center_theta_points)
    )
    three_center_bytes = three_center_nodes * _FLOAT64_BYTES

    return H0CostEstimate(
        inputs=inputs,
        grid_density_per_bohr3=density,
        approximate_grid_spacing_bohr=spacing,
        ao_support_grid_points=support_points,
        pair_intersection_grid_points=pair_points,
        directed_h_blocks=blocks,
        exact_local_cache_bytes=cache_bytes,
        exact_local_no_cache_ao_samples=no_cache_samples,
        exact_local_cached_ao_samples=cached_samples,
        exact_local_contraction_fmas=local_fmas,
        direct_st_function_samples=direct_st_samples,
        direct_st_contraction_fmas=direct_st_fmas,
        two_center_st_runtime_channel_values=st_runtime_values,
        two_center_st_table_bytes=st_table_bytes,
        unique_ao_projector_pairs=unique_ao_projector_pairs,
        ao_projector_grid_points=projector_points,
        current_direct_projector_dot_fmas=current_projector_fmas,
        cached_direct_projector_dot_fmas=unique_projector_fmas,
        two_center_projector_runtime_channel_values=projector_runtime_values,
        two_center_projector_table_bytes=projector_table_bytes,
        nonlocal_algebra_fmas=nonlocal_fmas,
        low_rank_reference_setup_flops=low_rank_setup_flops,
        low_rank_per_field_fmas=low_rank_per_field_fmas,
        low_rank_reference_crossover_fields=crossover,
        low_rank_factor_bytes=low_rank_bytes,
        three_center_table_nodes=three_center_nodes,
        three_center_table_bytes=three_center_bytes,
    )

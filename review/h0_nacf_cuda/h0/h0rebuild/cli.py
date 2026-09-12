from __future__ import annotations

import argparse
import json

from .assemble import assemble_h0
from .config import load_config
from .io import save_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline reconstruction of ABACUS-style H[rho0]")
    parser.add_argument("config", help="JSON configuration file")
    parser.add_argument("-o", "--output", default=None, help="Output .npz path")
    parser.add_argument("--energy-unit", choices=["eV", "Ry", "Ha"], default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    structure, species_data, numerics, physics, output, _ = load_config(args.config)
    kwargs = {
        "ecutrho_ry": numerics.get("ecutrho_ry"),
        "fft_shape": tuple(numerics["fft_shape"]) if "fft_shape" in numerics else None,
        "pair_grid_step_bohr": numerics.get("pair_grid_step_bohr", 0.30),
        "local_integration": numerics.get("local_integration", "fft_grid"),
        "local_cache_max_mb": numerics.get("local_cache_max_mb", 512.0),
        "local_cache_chunk_size": numerics.get("local_cache_chunk_size", 100_000),
        "local_cache_key_decimals": numerics.get("local_cache_key_decimals"),
        "projector_grid_step_bohr": numerics.get("projector_grid_step_bohr"),
        "radial_g_round_decimals": numerics.get("radial_g_round_decimals"),
        "pbe_density_threshold": numerics.get("pbe_density_threshold", 1.0e-6),
        "prune_frobenius_ry": numerics.get("prune_frobenius_ry", 0.0),
        "strict_reproduction": numerics.get("strict_reproduction", True),
        "hermitize": numerics.get("hermitize", True),
        "spatial_backend": numerics.get("spatial_backend", "reference"),
        "structure_factor_backend": numerics.get("structure_factor_backend", "direct"),
        "structure_factor_nthreads": numerics.get("structure_factor_nthreads", 1),
        "field_backend": numerics.get("field_backend", "numpy"),
        "compute_device": numerics.get("compute_device", "cpu"),
        "field_max_work_mb": numerics.get("field_max_work_mb", 1024.0),
        "structure_factor_eps": numerics.get("structure_factor_eps", 1.0e-12),
        "structure_factor_max_work_mb": numerics.get("structure_factor_max_work_mb", 512.0),
        "pair_support": numerics.get("pair_support", "orbital_overlap"),
        "xc": physics.get("xc", "LDA_PZ81"),
        "include_nonlocal": physics.get("include_nonlocal", True),
        "include_nlcc": physics.get("include_nlcc", True),
        "hartree_g0": physics.get("hartree_g0", "zero"),
        "local_g0": physics.get("local_g0", "alpha"),
        "total_electrons": physics.get("total_electrons"),
    }
    result = assemble_h0(structure, species_data, **kwargs)
    out = args.output or output.get("path", "H0_blocks.npz")
    unit = args.energy_unit or output.get("energy_unit", "eV")
    npz, metadata = save_result(result, out, unit)
    print(json.dumps({"npz": str(npz), "metadata": str(metadata), "blocks": len(result.h_blocks_ry)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

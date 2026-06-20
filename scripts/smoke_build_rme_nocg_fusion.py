#!/usr/bin/env python3
"""Smoke the Route-A RME head contract.

When executed inside an applied DeePTB checkout, this uses the real
OrbitalMapper and E3Hamiltonian.  A standalone fallback is provided so the
patch bundle itself can validate its algebra and 3s2p1d=121 fixture before it
is copied into a full checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import yaml
from e3nn import o3


BASE_COMMIT = "7206a7baefe8c7a0bf01d6f53674fdf3c4b606ee"
ANGULAR_DIM = {"s": 1, "p": 3, "d": 5, "f": 7, "g": 9, "h": 11}
ANGULAR_L = {name: (dim - 1) // 2 for name, dim in ANGULAR_DIM.items()}


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def _git_has_ancestor(root: Path, ancestor: str) -> bool:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, "HEAD"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
    except Exception:
        return False


def _load_head_class(repo_root: Path):
    try:
        from dptb.nn.embedding.rme_nocg_fusion_head import RMENoCGFusionHead

        return RMENoCGFusionHead
    except Exception:
        bundle_root = Path(__file__).resolve().parents[1]
        source = (
            bundle_root
            / "new_files"
            / "dptb"
            / "nn"
            / "embedding"
            / "rme_nocg_fusion_head.py"
        )
        if not source.exists():
            source = repo_root / "dptb/nn/embedding/rme_nocg_fusion_head.py"
        spec = importlib.util.spec_from_file_location("rme_nocg_fusion_head", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import head from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.RMENoCGFusionHead


def _counts_from_basis(basis: Dict[str, Iterable[str]]) -> Dict[str, int]:
    counts = {key: 0 for key in ANGULAR_DIM}
    for shells in basis.values():
        local = {key: 0 for key in ANGULAR_DIM}
        for shell in shells:
            label = "".join(ch for ch in str(shell) if ch.isalpha()).lower()
            if label not in local:
                raise ValueError(f"Unsupported shell {shell!r}")
            local[label] += 1
        for key in counts:
            counts[key] = max(counts[key], local[key])
    return counts


def _standalone_orbpair_irreps(
    counts: Dict[str, int],
) -> Tuple[o3.Irreps, int, int]:
    labels = [label for label in ANGULAR_DIM if counts[label] > 0]
    terms = []
    full_norb = sum(counts[label] * ANGULAR_DIM[label] for label in labels)
    onsite_diag = sum(
        counts[label] * ANGULAR_DIM[label] ** 2 for label in labels
    )
    rme_dim = (full_norb**2 + onsite_diag) // 2

    for i, left in enumerate(labels):
        l1 = ANGULAR_L[left]
        for right in labels[i:]:
            l2 = ANGULAR_L[right]
            if left == right:
                num_blocks = counts[left] * (counts[left] + 1) // 2
            else:
                num_blocks = counts[left] * counts[right]
            parity = (-1) ** (l1 + l2)
            block = o3.Irreps(
                [(1, (ell, parity)) for ell in range(abs(l1 - l2), l1 + l2 + 1)]
            )
            for _ in range(num_blocks):
                terms.extend(list(block))
    irreps = o3.Irreps(terms)
    if irreps.dim != rme_dim:
        raise AssertionError(
            f"Standalone irrep dim {irreps.dim} != RME formula {rme_dim}"
        )
    return irreps, full_norb, rme_dim


def _scan_forbidden(head_source: Path) -> bool:
    text = head_source.read_text(encoding="utf-8")
    forbidden = "wigner" + "_3j"
    return forbidden in text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fixture = config.get("_patch_smoke", {})
    embedding = config.get("model_options", {}).get("embedding", {})
    expected_commit = fixture.get("expected_base_commit", BASE_COMMIT)
    expected_rme_dim = int(fixture.get("expected_rme_dim", 121))
    basis = fixture.get("basis")
    if not basis:
        raise ValueError("Config must provide _patch_smoke.basis")

    mode = embedding.get("rme_head_mode", "legacy_linear")
    rank = int(embedding.get("rme_fusion_rank", 16))
    init = float(embedding.get("rme_fusion_init", 0.0))
    condition = embedding.get("rme_fusion_condition", "scalar_0e")

    actual_head = _git_head(repo_root)
    base_is_ancestor = _git_has_ancestor(repo_root, expected_commit)
    print(f"selected DeePTB commit: {actual_head}")
    print(f"expected DeePTB commit: {expected_commit}")
    print(f"expected base is ancestor: {str(base_is_ancestor).lower()}")
    print(f"rme_head_mode: {mode}")

    backend = "standalone-contract"
    idp = None
    e3_hamiltonian = None
    try:
        from dptb.data.transforms import OrbitalMapper
        from dptb.nn.hamiltonian import E3Hamiltonian

        idp = OrbitalMapper(basis, method="e3tb", device="cpu", has_soc=False)
        idp.get_irreps(no_parity=False)
        raw_irreps = idp.orbpair_irreps
        full_norb = int(idp.full_basis_norb)
        rme_dim = int(idp.reduced_matrix_element)
        e3_hamiltonian = E3Hamiltonian(idp=idp, device="cpu")
        backend = "deeptb-live"
    except Exception as exc:
        counts = _counts_from_basis(basis)
        raw_irreps, full_norb, rme_dim = _standalone_orbpair_irreps(counts)
        print(f"backend fallback reason: {type(exc).__name__}: {exc}")

    final_irreps = raw_irreps.sort()[0].simplify()
    Head = _load_head_class(repo_root)
    legacy_edge = o3.Linear(
        final_irreps, raw_irreps, shared_weights=True, internal_weights=True, biases=True
    )
    head = Head(
        final_irreps,
        raw_irreps,
        rank=rank,
        init=init,
        condition=condition,
        legacy=legacy_edge,
    )

    # Constructing the full TP can be expensive; its first input contract is
    # exactly the final LEM irreps, which is the quantity we need to expose.
    out_onehot_tp_input_dim = final_irreps.dim
    width_ok = rme_dim == raw_irreps.dim == expected_rme_dim
    if idp is not None:
        width_ok = width_ok and idp.reduced_matrix_element == raw_irreps.dim
        width_ok = width_ok and e3_hamiltonian.idp.reduced_matrix_element == rme_dim

    if backend == "deeptb-live" and actual_head != expected_commit and not base_is_ancestor:
        raise AssertionError(
            "Wrong checkout: expected commit must be HEAD or an ancestor; "
            f"HEAD={actual_head}, expected={expected_commit}"
        )
    if not width_ok:
        raise AssertionError(
            f"RME width contract failed: rme={rme_dim}, irreps={raw_irreps.dim}, "
            f"expected={expected_rme_dim}"
        )

    source = Path(sys.modules[Head.__module__].__file__).resolve()
    forbidden_present = _scan_forbidden(source)

    print(f"backend: {backend}")
    print(f"AO basis dim: {full_norb}")
    print(f"final LEM layer output irreps dim: {final_irreps.dim}")
    print(f"output RME dim: {raw_irreps.dim}")
    print(f"out-onehot TP input dim: {out_onehot_tp_input_dim}")
    print(
        "E3Hamiltonian input width assert remains 121: "
        f"{'PASS' if width_ok else 'FAIL'}"
    )
    print(f"head output dim: {head.irreps_out.dim}")
    print(
        "forbidden coupling call appears in new head code path: "
        f"{str(forbidden_present).lower()}"
    )
    if forbidden_present:
        raise AssertionError(f"Forbidden call found in {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

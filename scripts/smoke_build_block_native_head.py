#!/usr/bin/env python3
"""Smoke the experimental Plan-B block-native output contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/n2_nocfm_block_native_linear_snippet.yaml"),
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo.resolve()
    sys.path.insert(0, str(repo_root))

    from dptb.nn.build import build_model
    from dptb.nn.embedding.block_native_head import BlockNativeLinearHead

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = build_model(
        model_options=config["model_options"],
        common_options=config["common_options"],
        train_options=config.get("train_options", {}),
        no_check=True,
    )
    embedding = model.embedding

    print(f"repo: {repo_root}")
    print(f"prediction method: {model.method}")
    print(f"embedding type: {type(embedding).__name__}")
    print(f"output mode: {embedding.rme_head_mode}")
    print(f"uses E3Hamiltonian: {hasattr(model, 'hamiltonian')}")
    print(f"out_node type: {type(embedding.out_node).__name__}")
    print(f"out_edge type: {type(embedding.out_edge).__name__}")
    print(f"max AO basis dim: {embedding.out_node.max_norb}")
    print(f"final LEM irreps dim: {embedding.layers[-1].irreps_out.dim}")
    print(f"legacy RME dim: {model.idp.reduced_matrix_element}")

    if model.method != "block_native":
        raise AssertionError(f"Expected block_native method, got {model.method!r}")
    if hasattr(model, "hamiltonian"):
        raise AssertionError("block_native path must bypass E3Hamiltonian")
    if embedding.rme_head_mode != "block_native_linear":
        raise AssertionError(f"Wrong output mode {embedding.rme_head_mode!r}")
    if not isinstance(embedding.out_node, BlockNativeLinearHead):
        raise AssertionError("out_node is not BlockNativeLinearHead")
    if not isinstance(embedding.out_edge, BlockNativeLinearHead):
        raise AssertionError("out_edge is not BlockNativeLinearHead")
    if int(embedding.out_node.max_norb) != 14:
        raise AssertionError(f"Expected max_norb=14, got {embedding.out_node.max_norb}")
    if int(embedding.layers[-1].irreps_out.dim) == int(model.idp.reduced_matrix_element):
        raise AssertionError("block_native should keep final LEM hidden, not final RME width")

    print("BLOCK_NATIVE_CONTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

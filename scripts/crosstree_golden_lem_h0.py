"""Bit-exact cross-tree golden for the legacy LemMoEV3H0 model.

The parent process launches this same file as a child with each repository on
PYTHONPATH. Children import LemMoEV3H0 directly, never through LemPair.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Dict, List

import numpy as np


DEFAULT_BASE = Path(r"E:\deeptb\wt_0724_base")
DEFAULT_HEAD = Path(r"E:\deeptb\wt_0724_contract")
DEFAULT_REPORT = Path(
    r"F:\claude\0724_pair_iter2\results\crosstree_golden_A.md"
)
PYTHON = Path(r"C:\Users\16608\.conda\envs\dptb\python.exe")
SEED = 20260724


def _child_dump(dtype_name: str, output: Path) -> None:
    import torch

    from dptb.data import _keys
    from dptb.nn.embedding.lem_moe_v3_h0 import LemMoEV3H0

    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    previous_dtype = torch.get_default_dtype()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(dtype)
    torch.use_deterministic_algorithms(True)
    try:
        options = dict(
            basis={"O": "1s1p"},
            n_layers=2,
            n_radial_basis=4,
            r_max=5.0,
            irreps_hidden="2x0e+2x1o+2x1e+2x2e",
            avg_num_neighbors=3.0,
            env_embed_multiplicity=2,
            latent_dim=4,
            latent_channels=[4],
            edge_one_hot_dim=2,
            num_experts=1,
            num_shared_experts=1,
            top_k=1,
            use_layer_onehot_tp=False,
            use_out_onehot_tp=False,
            use_interpolation_out=False,
            tp_radial_emb=False,
            mole_linear_mode="indexed_ref",
            so2_fusion_mode="streamed_m_major_ref",
            equivariant_norm_type="merged_rms",
            output_route="h_b0",
            rme_fusion_rank=2,
            rme_fusion_init=0.2,
            use_h0_init=True,
            fallback_to_hamiltonian=False,
            require_full_block_edge_coverage=True,
            dtype=dtype,
            device="cpu",
        )
        torch.manual_seed(SEED)
        model = LemMoEV3H0(**options).eval()
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.70, 0.20, 0.10],
                [2.10, 0.30, -0.40],
                [0.20, 2.40, 0.50],
            ],
            dtype=dtype,
        )
        rows = [
            (i, j)
            for i in range(int(positions.shape[0]))
            for j in range(int(positions.shape[0]))
            if i != j
        ]
        edge_index = torch.tensor(rows, dtype=torch.long).T.contiguous()
        n_edges = int(edge_index.shape[1])
        h0_dim = int(model.idp.reduced_matrix_element)
        data = {
            _keys.POSITIONS_KEY: positions,
            _keys.EDGE_INDEX_KEY: edge_index,
            _keys.ATOM_TYPE_KEY: torch.zeros((4, 1), dtype=torch.long),
            _keys.EDGE_TYPE_KEY: torch.full(
                (n_edges,),
                model.idp.bond_to_type["O-O"],
                dtype=torch.long,
            ),
            _keys.NODE_H0_KEY: torch.zeros((4, h0_dim), dtype=dtype),
            _keys.EDGE_H0_KEY: torch.zeros((n_edges, h0_dim), dtype=dtype),
        }
        with torch.no_grad():
            result = model(data)

        state = model.state_dict()
        arrays: Dict[str, np.ndarray] = {
            "state_key_order": np.asarray(list(state), dtype=np.str_),
            "source_file": np.asarray(
                [str(Path(inspect.getfile(LemMoEV3H0)).resolve())],
                dtype=np.str_,
            ),
            "output_node": result[_keys.NODE_HAMILTONIAN_KEY]
            .detach()
            .cpu()
            .numpy(),
            "output_edge": result[_keys.EDGE_HAMILTONIAN_KEY]
            .detach()
            .cpu()
            .numpy(),
            "output_overlap": result[_keys.EDGE_OVERLAP_KEY]
            .detach()
            .cpu()
            .numpy(),
        }
        for index, tensor in enumerate(state.values()):
            arrays[f"state_{index:04d}"] = tensor.detach().cpu().numpy()
        np.savez(output, **arrays)
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.use_deterministic_algorithms(previous_deterministic)


def _run_child(tree: Path, dtype_name: str, output: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tree)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    command = [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "--child",
        "--dtype",
        dtype_name,
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=tree,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"{tree} {dtype_name} child failed with "
            f"exit={completed.returncode}:\n{tail}"
        )


def _tensor_equal(left: np.ndarray, right: np.ndarray) -> bool:
    import torch

    return torch.equal(torch.from_numpy(left), torch.from_numpy(right))


def _max_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    if not (
        np.issubdtype(left.dtype, np.number)
        and np.issubdtype(right.dtype, np.number)
    ):
        return 0.0 if np.array_equal(left, right) else float("inf")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _compare(dtype_name: str, base_file: Path, head_file: Path) -> Dict[str, object]:
    base = np.load(base_file, allow_pickle=False)
    head = np.load(head_file, allow_pickle=False)
    base_keys = base["state_key_order"].tolist()
    head_keys = head["state_key_order"].tolist()
    result: Dict[str, object] = {
        "dtype": dtype_name,
        "passed": True,
        "state_count": len(base_keys),
        "first_mismatch": "",
        "max_delta": 0.0,
        "base_source": base["source_file"].tolist()[0],
        "head_source": head["source_file"].tolist()[0],
    }
    if base_keys != head_keys:
        result["passed"] = False
        limit = min(len(base_keys), len(head_keys))
        index = next(
            (i for i in range(limit) if base_keys[i] != head_keys[i]),
            limit,
        )
        base_key = base_keys[index] if index < len(base_keys) else "<missing>"
        head_key = head_keys[index] if index < len(head_keys) else "<missing>"
        result["first_mismatch"] = (
            f"state key order index {index}: base={base_key!r}, head={head_key!r}"
        )
        return result

    for index, key in enumerate(base_keys):
        array_key = f"state_{index:04d}"
        left = base[array_key]
        right = head[array_key]
        delta = _max_delta(left, right)
        result["max_delta"] = max(float(result["max_delta"]), delta)
        if not _tensor_equal(left, right):
            result["passed"] = False
            result["first_mismatch"] = (
                f"state tensor {key!r}: base shape/dtype={left.shape}/{left.dtype}, "
                f"head shape/dtype={right.shape}/{right.dtype}, max|delta|={delta}"
            )
            return result

    for output_name in ("output_node", "output_edge", "output_overlap"):
        left = base[output_name]
        right = head[output_name]
        delta = _max_delta(left, right)
        result["max_delta"] = max(float(result["max_delta"]), delta)
        if not _tensor_equal(left, right):
            result["passed"] = False
            result["first_mismatch"] = (
                f"{output_name}: base shape/dtype={left.shape}/{left.dtype}, "
                f"head shape/dtype={right.shape}/{right.dtype}, max|delta|={delta}"
            )
            return result
    return result


def _write_report(
    report: Path,
    base: Path,
    head: Path,
    results: List[Dict[str, object]],
) -> None:
    passed = all(bool(result["passed"]) for result in results)
    lines = [
        "# Lane A cross-tree legacy LemMoEV3H0 golden",
        "",
        f"Overall: **{'PASS' if passed else 'FAIL'}**",
        "",
        f"- Base tree: `{base}`",
        f"- Head tree: `{head}`",
        f"- Seed: `{SEED}`",
        "- Oracle: direct `LemMoEV3H0` construction in each child; LemPair is not imported.",
        "",
        "| dtype | result | state tensors | max abs delta | first mismatch |",
        "|---|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| {dtype} | {status} | {state_count} | {max_delta:.16e} | {mismatch} |".format(
                dtype=result["dtype"],
                status="PASS" if result["passed"] else "FAIL",
                state_count=result["state_count"],
                max_delta=float(result["max_delta"]),
                mismatch=str(result["first_mismatch"]).replace("|", "\\|") or "none",
            )
        )
    lines.extend(["", "## Imported sources", ""])
    for result in results:
        lines.extend(
            [
                f"- {result['dtype']} base: `{result['base_source']}`",
                f"- {result['dtype']} head: `{result['head_source']}`",
            ]
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.child:
        if args.dtype is None or args.output is None:
            parser.error("--child requires --dtype and --output")
        _child_dump(args.dtype, args.output)
        return 0

    results: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="lem_h0_crosstree_") as temporary:
        temporary_path = Path(temporary)
        for dtype_name in ("float32", "float64"):
            base_file = temporary_path / f"base_{dtype_name}.npz"
            head_file = temporary_path / f"head_{dtype_name}.npz"
            _run_child(args.base, dtype_name, base_file)
            _run_child(args.head, dtype_name, head_file)
            results.append(_compare(dtype_name, base_file, head_file))
    _write_report(args.report, args.base, args.head, results)
    for result in results:
        print(
            f"{result['dtype']}={'PASS' if result['passed'] else 'FAIL'} "
            f"state_tensors={result['state_count']} "
            f"max_delta={float(result['max_delta']):.16e}"
        )
        if result["first_mismatch"]:
            print(f"FIRST_MISMATCH={result['first_mismatch']}")
    print(f"REPORT={args.report}")
    return 0 if all(bool(result["passed"]) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

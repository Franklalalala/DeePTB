#!/usr/bin/env python
"""WS0 mechanism diagnosis: kappa(S), Gauge_MAE, PP/QQ/PQ energy split, ghost
states, Hermitian/pair-consistency probe, on a single checkpoint/run.

See F:\\claude\\0702_nextham_dm_plan\\02_llm_execution_plan.md WS0 for spec.
Overlap S and ground-truth dense H come from a precomputed npz sidecar keyed
by the LMDB record's source idx (this dataset does not carry S in the LMDB
itself) -- see --npz. Model/dataset loading mirrors
dptb_downstream_block_eval_0621cfm_grad.py (proven plumbing for this N1 water
route); this script adds the WS0-specific diagnostics on top and drops the
AO-block MAE split machinery that script needed (not required here).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

try:
    import numpy.core as _np_core
    import numpy.core.numeric as _np_core_numeric

    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.numeric", _np_core_numeric)
except Exception:
    pass

HARTREE_TO_EV = 27.211386245988


def _normalize_current_config(raw: Dict[str, Any]):
    from dptb.utils.argcheck import normalize
    import re

    def _drop_legacy_keys(obj):
        if isinstance(obj, dict):
            obj.pop("has_soc", None)
            for value in obj.values():
                _drop_legacy_keys(value)
        elif isinstance(obj, list):
            for value in obj:
                _drop_legacy_keys(value)
        return obj

    def _remove_path_key(obj, location, key):
        cur = obj
        if location:
            for part in location.split("/"):
                if not part:
                    continue
                if isinstance(cur, dict):
                    cur = cur.get(part)
                else:
                    return False
        if isinstance(cur, dict) and key in cur:
            cur.pop(key, None)
            return True
        return False

    cleaned = _drop_legacy_keys(copy.deepcopy(raw))
    removed_unknown = []
    last_exc = None
    for _ in range(64):
        try:
            out = normalize(copy.deepcopy(cleaned))
            if isinstance(out, dict):
                out.setdefault("_eval_config_compat_removed_keys", removed_unknown)
            return out
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            m = re.search(r"\[at location `([^`]*)`\] undefined key `([^`]*)`", msg)
            if not m:
                raise
            location, key = m.group(1), m.group(2)
            if not _remove_path_key(cleaned, location, key):
                raise
            removed_unknown.append(f"{location}/{key}" if location else key)
    raise last_exc


def _to_device_dict(data: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for key, value in data.items():
        out[key] = value.to(device=device) if torch.is_tensor(value) else value
    return out


def _batch_info(batch) -> Dict[str, Any]:
    return {
        "__slices__": batch.__slices__,
        "__cumsum__": batch.__cumsum__,
        "__cat_dims__": batch.__cat_dims__,
        "__num_nodes_list__": batch.__num_nodes_list__,
        "__data_class__": batch.__data_class__,
    }


def _count_graphs(batch_dict: Dict[str, Any]) -> int:
    batch = batch_dict.get("batch")
    if torch.is_tensor(batch) and batch.numel() > 0:
        return int(batch.detach().max().cpu().item()) + 1
    ptr = batch_dict.get("ptr")
    if torch.is_tensor(ptr) and ptr.numel() > 1:
        return int(ptr.numel() - 1)
    return 1


def _read_source_idx(dataset, dataset_index: int) -> int:
    if hasattr(dataset, "_load_data_dict"):
        record = dataset._load_data_dict(int(dataset_index))
        if "idx" not in record:
            raise KeyError(f"LMDB record {dataset_index} does not contain source idx")
        return int(record["idx"])
    raise TypeError("This evaluator expects LMDBDataset or a dataset exposing _load_data_dict().")


def _dense_from_features(graph_data, h2k, device: str) -> np.ndarray:
    from dptb.data import AtomicData, AtomicDataDict

    work = AtomicData.to_AtomicDataDict(graph_data)
    work = _to_device_dict(work, device)
    out = h2k(work)
    return out[AtomicDataDict.HAMILTONIAN_KEY].detach().cpu().numpy().astype(np.float64)


def _make_prediction_view(model, flow, original: Dict[str, Any], num_steps: int, flow_enabled: bool):
    from dptb.data import AtomicDataDict

    if flow_enabled:
        sampled = flow.sample(model, original.copy(), num_steps=num_steps)
        pred_view = original.copy()
        pred_view.update(sampled)
        return pred_view
    model_input = original.copy()
    model_input.pop(AtomicDataDict.NODE_FEATURES_KEY, None)
    model_input.pop(AtomicDataDict.EDGE_FEATURES_KEY, None)
    model_input.pop(AtomicDataDict.HAMILTONIAN_KEY, None)
    pred = model(model_input)
    pred_view = original.copy()
    pred_view.update(pred)
    return pred_view


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# WS0 diagnostics
# ---------------------------------------------------------------------------


def _generalized_eigh(h: np.ndarray, s: np.ndarray):
    """Solve H C = S C E via Cholesky(S), return (evals ascending, C with C^T S C = I)."""
    h_t = torch.as_tensor(h, dtype=torch.float64)
    s_t = torch.as_tensor(s, dtype=torch.float64)
    l = torch.linalg.cholesky(s_t)
    linv = torch.linalg.inv(l)
    hp = linv @ h_t @ linv.T
    evals, vecs_p = torch.linalg.eigh(hp)
    coeff = linv.T @ vecs_p
    return evals, coeff


def _condition_number(s: np.ndarray):
    s_t = torch.as_tensor(s, dtype=torch.float64)
    evals = torch.linalg.eigvalsh(s_t)
    lam_min = float(evals.min())
    lam_max = float(evals.max())
    kappa = lam_max / lam_min if lam_min > 0 else float("inf")
    return kappa, lam_min, lam_max


def _gauge_mu(pred_h: np.ndarray, gt_h: np.ndarray, s: np.ndarray) -> float:
    """Closed-form mu* minimizing ||pred_h - gt_h - mu*S||_F^2 (real, R-space)."""
    diff = pred_h - gt_h
    denom = float(np.sum(s * s))
    if denom <= 0:
        return 0.0
    return float(np.sum(diff * s) / denom)


def _pq_energy_split(pred_h: np.ndarray, gt_evals: torch.Tensor, gt_coeff: torch.Tensor, n_p: int):
    """Transform pred_h into the ground-truth eigenbasis and split the error
    (pred - gt, both expressed in gt eigenbasis where gt is diagonal) into
    PP/QQ/PQ squared-Frobenius shares. n_p = size of the P (occupied+cutoff)
    subspace."""
    pred_t = torch.as_tensor(pred_h, dtype=torch.float64)
    h_tilde_pred = gt_coeff.T @ pred_t @ gt_coeff
    h_tilde_gt = torch.diag(gt_evals)
    diff = h_tilde_pred - h_tilde_gt
    n = diff.shape[0]
    n_p = max(0, min(n_p, n))
    pp = diff[:n_p, :n_p]
    qq = diff[n_p:, n_p:]
    pq = diff[:n_p, n_p:]
    qp = diff[n_p:, :n_p]
    err_pp = float((pp * pp).sum())
    err_qq = float((qq * qq).sum())
    err_pq = float((pq * pq).sum() + (qp * qp).sum())
    total = err_pp + err_qq + err_pq
    # sanity: gt PQ block of gt-in-own-basis must be ~0 (diagonal by construction)
    gt_pq_selfcheck = float((h_tilde_gt[:n_p, n_p:] ** 2).sum())
    return {
        "err_pp": err_pp,
        "err_qq": err_qq,
        "err_pq": err_pq,
        "err_total": total,
        "frac_pp": err_pp / total if total > 0 else 0.0,
        "frac_qq": err_qq / total if total > 0 else 0.0,
        "frac_pq": err_pq / total if total > 0 else 0.0,
        "n_p": n_p,
        "n_q": n - n_p,
        "gt_pq_selfcheck": gt_pq_selfcheck,
    }


def _ghost_states(pred_evals: torch.Tensor, gt_homo: float, gt_lumo: float) -> int:
    vals = pred_evals.detach().cpu().numpy()
    return int(np.sum((vals > gt_homo) & (vals < gt_lumo)))


def _hermitian_error(pred_h: np.ndarray) -> float:
    """Symmetry violation of the *assembled dense* prediction. HR2HK_Gamma_Only
    forces block = block + block.T so this is expected to be exactly 0 --
    a bug detector, not a diagnostic of model quality."""
    return float(np.max(np.abs(pred_h - pred_h.T)))


def evaluate_run(
    *,
    root: Path,
    repo: Path,
    run_name: str,
    test_root: Path,
    npz_path: Path,
    ckpt_name: str,
    ckpt_abs_path: Optional[Path],
    batch_size: int,
    max_samples: Optional[int],
    device: str,
    seed: int,
    progress_every: int,
    e_cut_ev: float,
    nocc: int,
    window_virtual: int,
) -> Dict[str, Any]:
    t_start = time.time()
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from dptb.data import AtomicData, AtomicDataDict, DataLoader
    from dptb.data.build import build_dataset
    from dptb.nn.build import build_model
    from dptb.nn.hr2hk import HR2HK_Gamma_Only
    from dptb.nnops.flow import build_hamiltonian_flow
    from dptb.utils.argcheck import collect_cutoffs
    from dptb.utils.torch_geometric.batch import Batch

    cfg_path = root / "runs" / run_name / "train_config.json"
    if ckpt_abs_path is not None:
        ckpt_path = ckpt_abs_path
    else:
        ckpt_path = root / "runs" / run_name / "checkpoint" / ckpt_name
        if not ckpt_path.exists() and ckpt_name == "nnenv.latest.pth":
            ckpt_path = _first_existing(
                sorted((root / "runs" / run_name / "checkpoint").glob("nnenv.iter*.pth"), reverse=True)
            ) or ckpt_path
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    with cfg_path.open("r", encoding="utf-8") as fp:
        jdata = _normalize_current_config(json.load(fp))

    dtype_name = str(jdata["common_options"].get("dtype", "float32"))
    torch_dtype = getattr(torch, dtype_name)
    torch.set_default_dtype(torch_dtype)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    jdata["common_options"]["device"] = device
    jdata["train_options"]["use_ddp"] = False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cutoff_options = collect_cutoffs(jdata)
    data_test = dict(jdata["data_options"]["validation"])
    data_test["root"] = str(test_root)
    data_test["prefix"] = "data"
    data_test["prefer_precomputed_h0"] = False
    dataset = build_dataset(
        **cutoff_options,
        **data_test,
        **jdata["common_options"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, dynamic_batch=None, num_workers=0)

    model = build_model(
        checkpoint=str(ckpt_path),
        model_options=jdata["model_options"],
        common_options=jdata["common_options"],
        train_options=jdata["train_options"],
        no_check=True,
        device=device,
    )
    model.eval()
    idp = model.hamiltonian.idp
    if hasattr(idp, "get_orbital_maps"):
        idp.get_orbital_maps()
    if hasattr(idp, "get_orbpair_maps"):
        idp.get_orbpair_maps()

    h2k = HR2HK_Gamma_Only(
        idp=idp,
        edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.HAMILTONIAN_KEY,
        dtype=torch_dtype,
        device=device,
    )

    flow_options = jdata["train_options"].get("flow_options", {})
    flow_enabled = bool(flow_options.get("enabled", False))
    flow = None
    num_steps = 0
    if flow_enabled:
        flow = build_hamiltonian_flow(flow_options, idp=idp, dtype=torch_dtype, device=torch.device(device))
        validation_steps = tuple(getattr(flow, "validation_ode_steps", (1,)))
        num_steps = int(validation_steps[0]) if validation_steps else 1

    e_cut_ha = e_cut_ev / HARTREE_TO_EV

    npz = np.load(str(npz_path), allow_pickle=False)
    rows = []
    dataset_offset = 0
    n_batches = 0

    with torch.enable_grad():
        for batch in loader:
            if max_samples is not None and dataset_offset >= max_samples:
                break
            batch = batch.to(device)
            original = AtomicData.to_AtomicDataDict(batch)
            pred_view = _make_prediction_view(model, flow, original.copy(), num_steps, flow_enabled)

            pred_view_with_info = pred_view.copy()
            pred_view_with_info.update(_batch_info(batch))
            pred_graphs = Batch.from_dict(pred_view_with_info).to_data_list()
            batch_graphs = _count_graphs(original)

            for local_idx, pred_graph in enumerate(pred_graphs):
                dataset_index = dataset_offset + local_idx
                if max_samples is not None and dataset_index >= max_samples:
                    break
                source_idx = _read_source_idx(dataset, dataset_index)
                pred_h = _dense_from_features(pred_graph, h2k, device)
                gt_h = np.asarray(npz["hamiltonian"][source_idx], dtype=np.float64)
                overlap = np.asarray(npz["overlap"][source_idx], dtype=np.float64)

                # -- real-space --
                r_diff = pred_h - gt_h
                mae_r = float(np.mean(np.abs(r_diff)))
                mu_shift = _gauge_mu(pred_h, gt_h, overlap)
                gauge_mae_r = float(np.mean(np.abs(pred_h - mu_shift * overlap - gt_h)))
                hermitian_error = _hermitian_error(pred_h)

                # -- overlap conditioning --
                kappa_s, min_eig_s, max_eig_s = _condition_number(overlap)

                # -- generalized eigenproblem (labels' basis) --
                gt_evals, gt_coeff = _generalized_eigh(gt_h, overlap)
                pred_evals, _ = _generalized_eigh(pred_h, overlap)
                gt_homo = float(gt_evals[nocc - 1])
                gt_lumo = float(gt_evals[nocc])
                pred_homo = float(pred_evals[nocc - 1])
                pred_lumo = float(pred_evals[nocc])
                e_fermi = 0.5 * (gt_homo + gt_lumo)

                w_hi = min(nocc + window_virtual, gt_evals.shape[0])
                band_mae = float((pred_evals[:w_hi] - gt_evals[:w_hi]).abs().mean())
                gap_mae = float(abs((pred_lumo - pred_homo) - (gt_lumo - gt_homo)))
                ghost_states = _ghost_states(pred_evals, gt_homo, gt_lumo)

                # -- PP/QQ/PQ split (P = gt energy <= E_F + e_cut) --
                n_p = int((gt_evals <= (e_fermi + e_cut_ha)).sum().item())
                pq = _pq_energy_split(pred_h, gt_evals, gt_coeff, n_p)

                rows.append(
                    {
                        "dataset_index": int(dataset_index),
                        "source_idx": int(source_idx),
                        "hamiltonian_mae": mae_r,
                        "gauge_mae_r": gauge_mae_r,
                        "mu_shift": mu_shift,
                        "hermitian_error": hermitian_error,
                        "kappa_s": kappa_s,
                        "min_eig_s": min_eig_s,
                        "max_eig_s": max_eig_s,
                        "band_mae_occ_plus5": band_mae,
                        "homo_mae": abs(pred_homo - gt_homo),
                        "lumo_mae": abs(pred_lumo - gt_lumo),
                        "gap_mae": gap_mae,
                        "gt_gap": gt_lumo - gt_homo,
                        "ghost_states": ghost_states,
                        "n_p": pq["n_p"],
                        "n_q": pq["n_q"],
                        "pq_err_pp": pq["err_pp"],
                        "pq_err_qq": pq["err_qq"],
                        "pq_err_pq": pq["err_pq"],
                        "pq_err_total": pq["err_total"],
                        "pq_frac_pp": pq["frac_pp"],
                        "pq_frac_qq": pq["frac_qq"],
                        "pq_frac_pq": pq["frac_pq"],
                        "pq_gt_selfcheck": pq["gt_pq_selfcheck"],
                    }
                )

            dataset_offset += batch_graphs
            n_batches += 1
            if progress_every and n_batches % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "progress": run_name,
                            "batches": n_batches,
                            "graphs": len(rows),
                            "elapsed_s": time.time() - t_start,
                        }
                    ),
                    flush=True,
                )

    # NOTE on the H_ij-vs-H_ji^T "pair consistency" probe originally planned here:
    # verified (on both predictions and, as a control, on ground-truth labels --
    # the control gave the *same* nonzero mismatch, proving it is not a model bug)
    # that dptb.data.interfaces.feature_to_block() intentionally returns each
    # directed edge's block filled only in the upper-triangle-in-full-basis-order
    # half (see ham_to_feature.py ~L592-593, `is_upper` gate); the complementary
    # half is supplied by the *reverse* edge's block transposed, and only their
    # SUM (as HR2HK_Gamma_Only performs) is a complete, meaningful AO block. So a
    # per-direction feature_to_block() comparison is checking a false premise and
    # was dropped; hermitian_error below (on the assembled dense H, where the
    # sum-then-symmetrize has already happened) is the correct probe and is
    # confirmed exactly 0 for this e3tb/RME route, matching the plan's expectation.

    def _pct(vals, q):
        if not vals:
            return 0.0
        return float(np.percentile(np.asarray(vals, dtype=np.float64), q))

    def _summ(key):
        vals = [r[key] for r in rows]
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "median": _pct(vals, 50),
            "p95": _pct(vals, 95),
            "max": float(arr.max()) if arr.size else 0.0,
            "min": float(arr.min()) if arr.size else 0.0,
        }

    band_mae_arr = np.asarray([r["band_mae_occ_plus5"] for r in rows], dtype=np.float64)
    pq_pq_norm_arr = np.asarray([math.sqrt(max(r["pq_err_pq"], 0.0)) for r in rows], dtype=np.float64)
    mae_r_arr = np.asarray([r["hamiltonian_mae"] for r in rows], dtype=np.float64)

    def _corr(a, b):
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    corr_band_pq = _corr(band_mae_arr, pq_pq_norm_arr)
    corr_band_maer = _corr(band_mae_arr, mae_r_arr)

    ghost_rate = float(np.mean([1.0 if r["ghost_states"] > 0 else 0.0 for r in rows])) if rows else 0.0
    ghost_mean_count = float(np.mean([r["ghost_states"] for r in rows])) if rows else 0.0

    summary = {
        "run": run_name,
        "checkpoint": str(ckpt_path),
        "config": str(cfg_path),
        "repo": str(repo),
        "repo_head": os.popen("git rev-parse HEAD").read().strip(),
        "device": device,
        "flow_enabled": flow_enabled,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "num_graphs": len(rows),
        "e_cut_ev": e_cut_ev,
        "nocc": nocc,
        "window_virtual": window_virtual,
        "elapsed_sec": time.time() - t_start,
        "kappa_s": _summ("kappa_s"),
        "min_eig_s": _summ("min_eig_s"),
        "hamiltonian_mae_r": _summ("hamiltonian_mae"),
        "gauge_mae_r": _summ("gauge_mae_r"),
        "mu_shift": _summ("mu_shift"),
        "hermitian_error": _summ("hermitian_error"),
        "band_mae": _summ("band_mae_occ_plus5"),
        "gap_mae": _summ("gap_mae"),
        "homo_mae": _summ("homo_mae"),
        "lumo_mae": _summ("lumo_mae"),
        "pq_frac_pp": _summ("pq_frac_pp"),
        "pq_frac_qq": _summ("pq_frac_qq"),
        "pq_frac_pq": _summ("pq_frac_pq"),
        "pq_gt_selfcheck_max": float(max((r["pq_gt_selfcheck"] for r in rows), default=0.0)),
        "ghost_state_rate": ghost_rate,
        "ghost_state_mean_count": ghost_mean_count,
        "pair_consistency_note": (
            "per-direction feature_to_block() blocks are upper-triangle-only by "
            "design (see comment above); dropped as a false-premise probe. "
            "hermitian_error (assembled dense H) is the valid probe here."
        ),
        "corr_band_mae_vs_pq_norm": corr_band_pq,
        "corr_band_mae_vs_mae_r": corr_band_maer,
    }
    return {"summary": summary, "rows": rows}


def _write_outputs(out: Path, payload: Dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload["summary_only"], indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".rows.csv")
    rows = payload.get("rows", [])
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--ckpt", default="nnenv.latest.pth")
    parser.add_argument("--ckpt-abs-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--e-cut-ev", type=float, default=10.0)
    parser.add_argument("--nocc", type=int, default=5)
    parser.add_argument("--window-virtual", type=int, default=5)
    args = parser.parse_args()

    max_samples = args.max_samples if args.max_samples > 0 else None
    result = {"run": args.run, "error": None}
    try:
        out = evaluate_run(
            root=args.root,
            repo=args.repo,
            run_name=args.run,
            test_root=args.test_root,
            npz_path=args.npz,
            ckpt_name=args.ckpt,
            ckpt_abs_path=args.ckpt_abs_path,
            batch_size=args.batch_size,
            max_samples=max_samples,
            device=args.device,
            seed=args.seed,
            progress_every=args.progress_every,
            e_cut_ev=args.e_cut_ev,
            nocc=args.nocc,
            window_virtual=args.window_virtual,
        )
        payload = {"summary_only": out["summary"], "rows": out["rows"]}
        _write_outputs(args.out, payload)
        print(json.dumps(out["summary"], indent=2))
    except Exception as exc:
        payload = {
            "summary_only": {"run": args.run, "error": repr(exc), "traceback": traceback.format_exc()},
            "rows": [],
        }
        _write_outputs(args.out, payload)
        print(json.dumps(payload["summary_only"], indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

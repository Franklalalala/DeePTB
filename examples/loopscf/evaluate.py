"""Per-structure development evaluation with chunked path spectra.

Stored targets must be residual AO dH; label+H0 closure is checked per structure.
FW10 is evaluated after concatenating the complete path (one global VBM), never
averaged from independently aligned chunks. Occupations use the wrapper's BZ
sample, independently of this plotting path.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from dptb.data import AtomicData, AtomicDataDict as A
from dptb.data.build import build_dataset
from dptb.data.dataloader import DataLoader
from dptb.data.transforms import OrbitalMapper
from dptb.nn.build import build_model
from dptb.nnops.loss import _nrme_mask, _erme_mask
from dptb.nnops.loopscf import install_working_memory_true_diag, CORRECTNESS_VERSION
from dptb.nnops.loopscf.kspace import build_k_plan, bloch_phase, assemble_flat
from dptb.nnops.loopscf.occupations import factor_overlap_robust
from dptb.nnops.loopscf.spectral import eigvals_from_factor, _fw10_one_graph
from dptb.utils.argcheck import normalize


def plain(t):
    if getattr(t, "is_nested", False):
        t = t[0]
    return t[0] if t.ndim == 3 else t


def clone(d):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in d.items()}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotate_ao(feat, idp, rotation):
    import re
    from e3nn import o3
    from dptb.utils.constants import anglrMId

    rotation = rotation.cpu()
    yzx = rotation.new_tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    basis_rotation = yzx @ rotation @ yzx.T
    result = feat.clone()
    for pair, sl in idp.orbpair_maps.items():
        ls = [anglrMId[re.findall(r"[a-zA-Z]", t)[0]] for t in pair.split("-")]
        dl, dr = [
            o3.Irrep(l, (-1) ** l).D_from_matrix(basis_rotation).to(feat) for l in ls
        ]
        block = feat[:, sl].reshape(-1, dl.shape[0], dr.shape[0])
        result[:, sl] = (dl @ block @ dr.T).flatten(1)
    return result


def rotation_check(model, ref, outputs, idp, steps, label):
    from e3nn import o3

    # Fixed proper rotation; all checkpoints see exactly the same transform.
    rotation = o3.angles_to_matrix(
        torch.tensor(0.37), torch.tensor(1.13), torch.tensor(-0.61)
    )
    rotated = clone(ref)
    for key in (A.POSITIONS_KEY, A.CELL_KEY, A.EDGE_VECTORS_KEY):
        if key in rotated:
            rotated[key] = rotated[key] @ rotation.to(rotated[key]).T
    for key in (
        A.NODE_FEATURES_KEY,
        A.EDGE_FEATURES_KEY,
        A.NODE_H0_KEY,
        A.EDGE_H0_KEY,
        A.NODE_OVERLAP_KEY,
        A.EDGE_OVERLAP_KEY,
    ):
        rotated[key] = rotate_ao(rotated[key], idp, rotation)
    state = model._wm_prepare(rotated)
    wn = we = None
    errors = {}
    for k in range(1, max(steps) + 1):
        pred = model._wm_one_k(rotated, k, wn, we, state)
        wn, we = model._wm_update(pred, state)
        if k in steps:
            key = label + "_K%d" % k
            errors[key] = {}
            for field, value in zip(
                (A.NODE_FEATURES_KEY, A.EDGE_FEATURES_KEY), outputs[key]
            ):
                expected = rotate_ao(value, idp, rotation)
                relative = float(
                    torch.linalg.vector_norm(pred[field] - expected)
                    / torch.linalg.vector_norm(expected).clamp_min(1e-12)
                )
                errors[key][field] = relative
                if not relative < 1e-3:
                    raise ValueError(
                        "rotation tolerance exceeded: %s %s %g" % (key, field, relative)
                    )
    return errors


def overlap_summary(diags):
    return {
        "min_eig_S": min(float(d["min_eig_S"].min()) for d in diags),
        "max_condition_S": max(float(d["condition_S"].max()) for d in diags),
        "projected_kpoints": sum(int((d["dropped_modes"] > 0).sum()) for d in diags),
        "dropped_modes": sum(int(d["dropped_modes"].sum()) for d in diags),
        "min_retained_rank": min(int(d["retained_rank"].min()) for d in diags),
    }


def path_bands(ref, outputs, idp, kpts, chunk_size):
    at, ei = ref[A.ATOM_TYPE_KEY].reshape(-1), ref[A.EDGE_INDEX_KEY]
    bi = torch.zeros_like(at)
    plan = build_k_plan(
        idp, at, ei, bi, torch.tensor([0, len(at)], device=at.device), at.device
    )
    spectra = {name: [] for name in outputs}
    diags = []
    for kp in kpts.split(chunk_size):
        phase = bloch_phase(kp.unsqueeze(0), ref[A.EDGE_CELL_SHIFT_KEY], bi[ei[0]])
        sb = assemble_flat(
            plan,
            ref[A.NODE_OVERLAP_KEY],
            ref[A.EDGE_OVERLAP_KEY],
            phase,
            torch.complex128,
        )
        diag = {}
        factors = factor_overlap_robust(plan.block(sb, 0), diagnostics=diag)
        diags.append(diag)
        for name, (node, edge) in outputs.items():
            hb = assemble_flat(
                plan,
                node + ref[A.NODE_H0_KEY],
                edge + ref[A.EDGE_H0_KEY],
                phase,
                torch.complex128,
            )
            spectra[name].append(eigvals_from_factor(plan.block(hb, 0), *factors).cpu())
    return {name: torch.cat(parts) for name, parts in spectra.items()}, overlap_summary(
        diags
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-model")
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", choices=["train", "validation"], default="validation")
    ap.add_argument("--label", default="corrected")
    ap.add_argument("--mode", choices=["head", "moe"], default="head")
    ap.add_argument(
        "--architecture", choices=["scalar", "latent", "stack"], default="scalar"
    )
    ap.add_argument(
        "--latent-strategy", choices=["bptt", "detach", "reset"], default="bptt"
    )
    ap.add_argument("--K", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument(
        "--latent-readout", choices=["frozen", "residual"], default="residual"
    )
    ap.add_argument("--compare-original", action="store_true")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--indices", type=int, nargs="+")
    ap.add_argument("--k-chunk", type=int, default=8)
    ap.add_argument("--rotation-indices", type=int, nargs="*", default=[])
    ap.add_argument("--exit-quantile", type=float, default=0.5)
    args = ap.parse_args()
    if args.architecture == "latent" and (args.no_feedback or args.mode != "head"):
        ap.error("latent uses --mode head and --latent-strategy, not --no-feedback")
    if min(args.K) < 1 or args.k_chunk < 1:
        ap.error("K and k-chunk must be positive")
    torch.set_num_threads(6)
    torch.manual_seed(20260910)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = normalize(json.loads(Path(args.input).read_text()))
    common = dict(cfg["common_options"], device="cuda:0")
    do = cfg["data_options"]
    ds = build_dataset(
        **do[args.split],
        r_max=do.get("r_max"),
        er_max=do.get("er_max"),
        oer_max=do.get("oer_max"),
        **common
    )
    indices = list(range(len(ds))) if args.indices is None else args.indices
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(ds) for i in indices):
        ap.error("indices must be unique and in range")
    if not set(args.rotation_indices).issubset(indices):
        ap.error("rotation indices must be included in evaluated indices")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stack_train_K = raw.get("stack_protocol", {}).get("K", max(args.K))
    sd = raw.get("model_state_dict", raw)
    is_latent = any("latent_core" in k for k in sd)
    is_stack = "_stack_version" in sd
    has_wm = is_stack or is_latent or any("wm_node" in k or "wm_edge" in k for k in sd)
    saved_architecture = "stack" if is_stack else "latent" if is_latent else "scalar"
    if has_wm and saved_architecture != args.architecture:
        ap.error("saved adapters belong to a different LoopSCF architecture")
    if is_latent:
        from dptb.nnops.loopscf.latent import validate_latent_checkpoint

        validate_latent_checkpoint(sd, args.latent_strategy, args.latent_readout)
    if has_wm and (not args.base_model or args.compare_original):
        ap.error("WM checkpoints require --base-model and cannot --compare-original")
    del raw
    source = args.base_model if has_wm else args.checkpoint

    def build():
        return (
            build_model(
                checkpoint=source,
                model_options=cfg["model_options"],
                common_options=common,
            )
            .cuda()
            .eval()
        )

    original = build() if args.compare_original else None
    model = build()
    idp = OrbitalMapper(common["basis"], method="e3tb", device="cuda:0")
    if args.architecture == "stack":
        from dptb.nnops.loopscf.stack import install_stack_loop

        if args.latent_strategy == "reset":
            ap.error("stack does not use reset strategy")
        install_stack_loop(model, idp, K=max(args.K), strategy=args.latent_strategy)
    elif args.architecture == "latent":
        from dptb.nnops.loopscf.latent import install_latent_corrector

        install_latent_corrector(
            model,
            idp,
            K=max(args.K),
            strategy=args.latent_strategy,
            readout=args.latent_readout,
        )
    else:
        install_working_memory_true_diag(
            model,
            args.mode,
            idp,
            K=max(args.K),
            feedback=not args.no_feedback,
            collect_diagnostics=True,
        )
    if has_wm:
        model.load_state_dict(sd, strict=True)
    del sd
    model.eval()
    report = {
        "protocol": {
            **vars(args),
            "version": CORRECTNESS_VERSION,
            "dataset_size": len(ds),
            "dataset_root": do[args.split]["root"],
            "config_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "target": "residual_ao_dH",
            "band_metric": "legacy_vbm_aligned_fw10_complete_path",
            "matrix_metric": "valid_packed_ao_elements_no_shift",
            "tf32": False,
            "timing": "synchronized cumulative wrapper prepare/forward/update, excludes path metrics; new shapes can include compilation; shared GPU diagnostic, not a formal benchmark",
        },
        "indices": indices,
        "results": [],
        "failures": [],
        "closure_mismatches": [],
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
        tmp.replace(dest)

    with torch.no_grad():
        for index in indices:
            try:
                item = ds[index]
                kpts = plain(item[A.KPOINT_KEY]).cuda()
                eig_ref = plain(item[A.ENERGY_EIGENVALUE_KEY]).cpu()
                batch = next(
                    iter(
                        DataLoader(
                            dataset=[item],
                            batch_size=1,
                            exclude_keys=[A.KPOINT_KEY, A.ENERGY_EIGENVALUE_KEY],
                        )
                    )
                )
                ref = AtomicData.to_AtomicDataDict(batch.cuda())
                outputs = {
                    "label": (ref[A.NODE_FEATURES_KEY], ref[A.EDGE_FEATURES_KEY])
                }
                times = {}
                if original is not None:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    pred = original(clone(ref))
                    torch.cuda.synchronize()
                    times["original_K1"] = time.perf_counter() - start
                    outputs["original_K1"] = (
                        pred[A.NODE_FEATURES_KEY],
                        pred[A.EDGE_FEATURES_KEY],
                    )
                torch.cuda.synchronize()
                start = time.perf_counter()
                state = model._wm_prepare(ref)
                wn = we = None
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                for k in range(1, max(args.K) + 1):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    pred = model._wm_one_k(ref, k, wn, we, state)
                    wn, we = model._wm_update(pred, state)
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter() - start
                    if k in args.K:
                        key = args.label + "_K%d" % k
                        outputs[key] = (
                            pred[A.NODE_FEATURES_KEY],
                            pred[A.EDGE_FEATURES_KEY],
                        )
                        times[key] = elapsed
                adaptive_exit = None
                if args.architecture == "stack" and is_stack:
                    from dptb.nnops.loopscf.stack import predict_until_exit

                    torch.cuda.synchronize()
                    exit_start = time.perf_counter()
                    stopped = predict_until_exit(
                        model, ref, max_steps=stack_train_K, quantile=args.exit_quantile
                    )
                    torch.cuda.synchronize()
                    key = args.label + "_exit"
                    outputs[key] = (
                        stopped[A.NODE_FEATURES_KEY],
                        stopped[A.EDGE_FEATURES_KEY],
                    )
                    times[key] = time.perf_counter() - exit_start
                    adaptive_exit = {
                        "step": stopped["_exit_step"],
                        "cdf": stopped["_exit_cdf"],
                        "quantile": args.exit_quantile,
                        "max_steps": stack_train_K,
                        "counts": stopped["_stack_counts"],
                    }
                spectra, diag = path_bands(ref, outputs, idp, kpts, args.k_chunk)
                ne = float(ref["nelec"].reshape(-1)[0])
                closure = float(
                    _fw10_one_graph(spectra.pop("label"), eig_ref, ne, 10.0)[0]
                )
                outputs.pop("label")
                at = ref[A.ATOM_TYPE_KEY].reshape(-1)
                masks = [
                    _nrme_mask(idp, at, result_device=at.device),
                    _erme_mask(
                        idp, ref[A.EDGE_TYPE_KEY].flatten(), result_device=at.device
                    ),
                ]
                row = {
                    "index": index,
                    "n_atoms": len(at),
                    "nelec": ne,
                    "n_k": len(kpts),
                    "label_closure_fw10_ev": closure,
                    "label_closure_pass": closure < 1e-3,
                    "path_overlap": diag,
                    "occupation_overlap": (
                        overlap_summary(state["overlap_diagnostics"])
                        if "overlap_diagnostics" in state
                        else None
                    ),
                    "max_electron_error": (
                        max(abs(float(q.sum()) - ne) for q in state["q_history"])
                        if "q_history" in state
                        else None
                    ),
                    "latent_encoder_calls": (
                        [c["encoder_calls"] for c in state["contexts"]]
                        if "contexts" in state
                        else None
                    ),
                    "variants": {},
                    "adaptive_exit": adaptive_exit,
                }
                if index in args.rotation_indices:
                    row["rotation_relative_errors"] = rotation_check(
                        model, ref, outputs, idp, args.K, args.label
                    )
                for key, pair in outputs.items():
                    errors = [
                        (p - ref[f]).abs()[mask]
                        for p, f, mask in zip(
                            pair, [A.NODE_FEATURES_KEY, A.EDGE_FEATURES_KEY], masks
                        )
                    ]
                    row["variants"][key] = {
                        "onsite_mae_ev": float(errors[0].mean()),
                        "hopping_mae_ev": float(errors[1].mean()),
                        "packed_valid_element_mae_ev": float(torch.cat(errors).mean()),
                        "n_onsite_elements": errors[0].numel(),
                        "n_hopping_elements": errors[1].numel(),
                        "legacy_vbm_aligned_fw10_ev": float(
                            _fw10_one_graph(spectra[key], eig_ref, ne, 10.0)[0]
                        ),
                        "forward_seconds": times[key],
                    }
                # Validate the row before committing it to the incremental report.
                json.dumps(row, allow_nan=False)
                report["results"].append(row)
                if not row["label_closure_pass"]:
                    report["closure_mismatches"].append(
                        {"index": index, "label_closure_fw10_ev": closure}
                    )
                print(json.dumps(row), flush=True)
                del outputs, spectra, state, pred, ref, batch
            except Exception as exc:
                report["failures"].append(
                    {"index": index, "error": str(exc), "type": type(exc).__name__}
                )
                print("EVALUATION_FAILURE", index, repr(exc), flush=True)
            save()
    print(
        "EVALUATION_COMPLETE",
        len(report["results"]),
        "failed",
        len(report["failures"]),
        "closure_mismatches",
        len(report["closure_mismatches"]),
        flush=True,
    )
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

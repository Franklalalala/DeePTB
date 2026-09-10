"""Same-weight K1 diagnostic; physical H0 is added exactly once."""

from pathlib import Path
import argparse, json, sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--indices", type=int, nargs="+", default=[0, 11])
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from dptb.data import AtomicData, AtomicDataDict as A
from dptb.data.build import build_dataset
from dptb.data.dataloader import DataLoader
from dptb.data.transforms import OrbitalMapper
from dptb.nn.build import build_model
from dptb.nnops.loss import _nrme_mask, _erme_mask
from dptb.nnops.loopscf.representation import AOPriorToRME
from dptb.nnops.loopscf.kspace import build_k_plan, bloch_phase, assemble_flat
from dptb.nnops.loopscf.occupations import factor_overlap_robust
from dptb.nnops.loopscf.spectral import eigvals_from_factor, _fw10_one_graph
from dptb.utils.argcheck import normalize

torch.set_num_threads(6)
torch.manual_seed(20260910)
cfg = normalize(json.loads(Path(args.input).read_text()))
common = dict(cfg["common_options"])
common["device"] = "cuda:0"
do = cfg["data_options"]
ds = build_dataset(
    **do["train"],
    r_max=do.get("r_max"),
    er_max=do.get("er_max"),
    oer_max=do.get("oer_max"),
    **common
)
model = (
    build_model(
        checkpoint=args.checkpoint,
        model_options=cfg["model_options"],
        common_options=common,
    )
    .cuda()
    .eval()
)
idp = OrbitalMapper(common["basis"], method="e3tb", device="cuda:0")
convert = AOPriorToRME(idp, dtype=next(model.parameters()).dtype, device="cuda:0")


def copy_dict(d):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in d.items()}


def plain(t):
    if getattr(t, "is_nested", False):
        t = t[0]
    if t.ndim == 3:
        t = t[0]
    return t


results = []
with torch.no_grad():
    for index in args.indices:
        item = ds[index]
        kpts = plain(item[A.KPOINT_KEY]).cuda()
        eig_ref = plain(item[A.ENERGY_EIGENVALUE_KEY]).cuda()
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
        at = ref[A.ATOM_TYPE_KEY].reshape(-1)
        ei = ref[A.EDGE_INDEX_KEY]
        bi = torch.zeros_like(at)
        ptr = torch.tensor([0, len(at)], device=at.device)
        plan = build_k_plan(idp, at, ei, bi, ptr, at.device)
        phase = bloch_phase(kpts.unsqueeze(0), ref[A.EDGE_CELL_SHIFT_KEY], bi[ei[0]])
        sb = assemble_flat(
            plan,
            ref[A.NODE_OVERLAP_KEY],
            ref[A.EDGE_OVERLAP_KEY],
            phase,
            torch.complex128,
        )
        diag = {}
        factors = factor_overlap_robust(plan.block(sb, 0), diagnostics=diag)

        def bands(n, e):
            hb = assemble_flat(plan, n, e, phase, torch.complex128)
            return eigvals_from_factor(plan.block(hb, 0), *factors)

        ne = float(ref["nelec"].reshape(-1)[0])
        label_check = float(
            _fw10_one_graph(
                bands(
                    ref[A.NODE_FEATURES_KEY] + ref[A.NODE_H0_KEY],
                    ref[A.EDGE_FEATURES_KEY] + ref[A.EDGE_H0_KEY],
                ),
                eig_ref,
                ne,
                10.0,
            )[0]
        )
        assert label_check < 1e-3, (
            "label plus H0 does not reproduce stored bands",
            label_check,
        )
        row = {
            "index": index,
            "nelec": ne,
            "n_k": len(kpts),
            "label_plus_h0_vs_stored_bands_fw10_ev": label_check,
            "overlap": {
                k: v.tolist() if torch.is_tensor(v) else v for k, v in diag.items()
            },
        }
        for mode in ["original_ao_input", "corrected_rme_input"]:
            inp = copy_dict(ref)
            if mode == "corrected_rme_input":
                inp[A.NODE_H0_KEY], inp[A.EDGE_H0_KEY] = convert(inp)
            pred = model(inp)
            masks = [
                _nrme_mask(idp, at, result_device=at.device),
                _erme_mask(
                    idp, ref[A.EDGE_TYPE_KEY].flatten(), result_device=at.device
                ),
            ]
            physical = []
            errors = []
            for fk, hk, mask in zip(
                [A.NODE_FEATURES_KEY, A.EDGE_FEATURES_KEY],
                [A.NODE_H0_KEY, A.EDGE_H0_KEY],
                masks,
            ):
                assert pred[fk].shape == ref[fk].shape
                phys = pred[fk] + ref[hk]
                physical.append(phys)
                errors.append((pred[fk] - ref[fk]).abs()[mask])
            fw = float(_fw10_one_graph(bands(*physical), eig_ref, ne, 10.0)[0])
            row[mode] = {
                "onsite_mae_ev": float(errors[0].mean()),
                "hopping_mae_ev": float(errors[1].mean()),
                "packed_valid_element_mae_ev": float(torch.cat(errors).mean()),
                "legacy_vbm_aligned_fw10_ev": fw,
                "n_onsite_elements": errors[0].numel(),
                "n_hopping_elements": errors[1].numel(),
            }
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "overlap"}), flush=True)
report = {
    "checkpoint": args.checkpoint,
    "config": args.input,
    "device": torch.cuda.get_device_name(),
    "indices": args.indices,
    "results": results,
    "protocol": "Same original weights, no WM, eval mode, no training; corrected input matches v3 K1. Stored labels are residual AO dH, confirmed by label+H0 spectrum against stored DFT bands. Matrix MAE over valid packed AO elements, no energy shift, identical for dH errors and reconstructed H errors with the same H0. Bands use stored DFT path and legacy VBM-aligned +/-10 eV reference window; complex128 solver for both arms. Two training-set diagnostic structures, not a held-out accuracy benchmark.",
}
Path(args.output).write_text(json.dumps(report, indent=2))
print("PAIRED_MAE_PASS", flush=True)

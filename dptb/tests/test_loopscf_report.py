"""Synthetic unittest for weighting, missing-index, and closure-policy behavior."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "examples/loopscf"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import summarize_experiment as se


def _variant(onsite, n_on, hopping, n_hop, fw10, seconds):
    packed = (onsite * n_on + hopping * n_hop) / (n_on + n_hop)
    return {
        "onsite_mae_ev": onsite,
        "hopping_mae_ev": hopping,
        "packed_valid_element_mae_ev": packed,
        "legacy_vbm_aligned_fw10_ev": fw10,
        "n_onsite_elements": n_on,
        "n_hopping_elements": n_hop,
        "forward_seconds": seconds,
    }


def _payload():
    return {
        "protocol": {
            "name": "synthetic-loopscf",
            "checkpoint": "base.pth",
            "note": "element-weighted MAE vs unweighted structure mean",
        },
        "results": [
            {
                "index": 0,
                "label_closure_fw10_ev": 1e-5,
                "label_closure_pass": True,
                "path_overlap": {"min_eig_S": 0.01},
                "occupation_overlap": {"nelec": 88.0},
                "rotation_relative_errors": {"node": 1e-6, "edge": 2e-6},
                "variants": {
                    "baseline": _variant(0.2, 10, 0.1, 90, 1.0, 1.0),
                    "control": _variant(0.1, 10, 0.05, 90, 0.8, 1.2),
                    "loop": _variant(0.15, 10, 0.08, 90, 0.9, 2.0),
                },
            },
            {
                "index": 1,
                "label_closure_fw10_ev": 0.002,
                "label_closure_pass": False,
                "path_overlap": {"min_eig_S": 0.0},
                "occupation_overlap": {"nelec": 45.0},
                "rotation_relative_errors": {"node": 3e-6, "edge": 4e-6},
                "variants": {
                    "baseline": _variant(1.0, 100, 0.5, 100, 2.0, 3.0),
                    "control": _variant(0.8, 100, 0.4, 100, 3.0, 3.5),
                },
            },
        ],
        "failures": [{"index": 1, "variant": "loop", "reason": "evaluator timeout"}],
    }


def _run(payload, tmp: Path, closure_only: bool):
    inp = tmp / ("input_closure.json" if closure_only else "input_default.json")
    out = tmp / ("report_closure.json" if closure_only else "report_default.json")
    inp.write_text(json.dumps(payload), encoding="utf-8")
    argv = ["--inputs", str(inp), "--output", str(out)]
    if closure_only:
        argv.append("--closure-only")
    rc = se.main(argv)
    if rc != 0:
        raise AssertionError(f"main returned {rc}")
    return str(inp), json.loads(out.read_text(encoding="utf-8"))


class SummarizeExperimentSyntheticTest(unittest.TestCase):
    """Matched-experiment example: weighting, missing IDs, and closure policy."""

    def test_weighting_missing_index_and_closure_policy(self):
        payload = _payload()
        with tempfile.TemporaryDirectory(prefix="summarize_experiment_") as td:
            tmp = Path(td)
            inp, report = _run(payload, tmp, closure_only=False)
            _, closed = _run(payload, tmp, closure_only=True)

        self.assertEqual(report["inputs"][0]["protocol"]["name"], "synthetic-loopscf")
        self.assertEqual(report["inputs"][0]["path"], inp)
        self.assertEqual(report["failure_counts"]["total"], 1)
        self.assertEqual(report["failures"][0]["index"], 1)
        self.assertEqual(report["failures"][0]["variant"], "loop")
        self.assertEqual(report["units"]["onsite_mae_ev"], "eV")
        self.assertEqual(report["units"]["forward_seconds"], "s")
        self.assertIn("positive difference = worse", report["difference_convention"])

        baseline = report["variants"]["baseline"]
        self.assertEqual(baseline["count"], 2)
        self.assertEqual(baseline["indices"], [0, 1])
        packed_mean = baseline["metrics"]["packed_valid_element_mae_ev"]["mean"]
        weighted = baseline["weighted_mae_ev"]["packed_valid_element"]
        on_abs = 0.2 * 10 + 1.0 * 100
        hop_abs = 0.1 * 90 + 0.5 * 100
        n_on, n_hop = 110, 190
        self.assertAlmostEqual(packed_mean, (0.11 + 0.75) / 2.0)
        self.assertAlmostEqual(weighted, (on_abs + hop_abs) / (n_on + n_hop))
        self.assertNotAlmostEqual(weighted, packed_mean)
        self.assertAlmostEqual(baseline["weighted_mae_ev"]["onsite"], on_abs / n_on)
        self.assertAlmostEqual(baseline["weighted_mae_ev"]["hopping"], hop_abs / n_hop)
        self.assertEqual(baseline["weighted_mae_ev"]["n_valid_elements"], 300)
        self.assertEqual(baseline["forward_seconds"]["count"], 2)

        pairs = {
            (row["reference"], row["new"]): row for row in report["paired_differences"]
        }
        complete = pairs[("baseline", "control")]
        self.assertTrue(complete["identical_index_set"])
        fw = complete["metrics"]["legacy_vbm_aligned_fw10_ev"]
        self.assertAlmostEqual(fw["per_structure"][0]["delta"], -0.2)
        self.assertAlmostEqual(fw["per_structure"][1]["delta"], 1.0)
        self.assertEqual(fw["improvement_count"], 1)
        self.assertAlmostEqual(fw["worst_increase"], 1.0)
        self.assertAlmostEqual(fw["mean"], 0.4)
        self.assertEqual(complete["metrics"]["onsite_mae_ev"]["improvement_count"], 2)
        self.assertNotIn("forward_seconds", complete["metrics"])

        missing = pairs[("baseline", "loop")]
        self.assertFalse(missing["identical_index_set"])
        self.assertEqual(missing["missing_in_new"], [1])
        self.assertIsNone(missing["metrics"])
        self.assertEqual(report["variants"]["loop"]["indices"], [0])
        self.assertEqual(len(report["failures"]), 1)
        self.assertAlmostEqual(fw["p90"], se.percentile([-0.2, 1.0], 0.90))

        self.assertEqual(report["closure"]["policy"], "all")
        self.assertEqual(report["closure"]["mismatch_count"], 1)
        self.assertEqual(report["closure"]["mismatch_indices"], [1])
        self.assertEqual(report["closure"]["excluded_ids_by_input"][0]["ids"], [])
        self.assertEqual(report["inputs"][0]["n_excluded_by_closure"], 0)

        diags = {
            (row["index"], row["variant"]): row
            for row in report["per_structure_diagnostics"]
        }
        self.assertEqual(diags[(0, "baseline")]["path_overlap"]["min_eig_S"], 0.01)
        self.assertEqual(diags[(0, "baseline")]["occupation_overlap"]["nelec"], 88.0)
        self.assertEqual(diags[(0, "loop")]["rotation_relative_errors"]["node"], 1e-6)
        self.assertTrue(diags[(0, "baseline")]["label_closure_pass"])
        self.assertFalse(diags[(1, "baseline")]["label_closure_pass"])
        self.assertAlmostEqual(diags[(1, "control")]["label_closure_fw10_ev"], 0.002)
        self.assertTrue(diags[(1, "baseline")]["included_in_aggregate"])
        self.assertTrue(
            any(row["index"] == 1 for row in baseline["per_structure_diagnostics"])
        )

        self.assertEqual(closed["closure"]["policy"], "closure_only")
        self.assertEqual(closed["closure"]["mismatch_indices"], [1])
        self.assertEqual(closed["closure"]["excluded_ids_by_input"][0]["ids"], [1])
        self.assertEqual(closed["inputs"][0]["n_excluded_by_closure"], 1)
        self.assertEqual(closed["failure_counts"]["total"], 1)
        self.assertEqual(closed["failures"][0]["index"], 1)
        self.assertEqual(closed["failures"][0]["reason"], "evaluator timeout")
        self.assertEqual(closed["variants"]["baseline"]["indices"], [0])
        self.assertEqual(closed["variants"]["control"]["indices"], [0])
        self.assertAlmostEqual(
            closed["variants"]["baseline"]["weighted_mae_ev"]["packed_valid_element"],
            0.11,
        )
        closed_pairs = {
            (row["reference"], row["new"]): row for row in closed["paired_differences"]
        }
        self.assertTrue(closed_pairs[("baseline", "control")]["identical_index_set"])
        self.assertEqual(closed_pairs[("baseline", "control")]["indices"], [0])
        self.assertTrue(closed_pairs[("baseline", "loop")]["identical_index_set"])
        closed_diags = {
            (row["index"], row["variant"]): row
            for row in closed["per_structure_diagnostics"]
        }
        self.assertFalse(closed_diags[(1, "baseline")]["included_in_aggregate"])
        self.assertTrue(closed_diags[(0, "baseline")]["included_in_aggregate"])
        self.assertEqual(closed_diags[(1, "control")]["path_overlap"]["min_eig_S"], 0.0)
        self.assertTrue(math.isfinite(weighted))


if __name__ == "__main__":
    unittest.main()

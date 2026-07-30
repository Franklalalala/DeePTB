import csv
from collections import Counter
from pathlib import Path
import time

from benchmarks.conformance.figures import FIGURE_FILENAMES, generate_figures
from benchmarks.conformance.runner import run_conformance


def test_c1_conformance_smoke_end_to_end(tmp_path: Path):
    started = time.perf_counter()
    summary = run_conformance(
        n_cases=200,
        seed=730,
        output_dir=tmp_path,
        shard_size=100,
    )
    figure_paths = generate_figures(
        tmp_path / "cases.csv", tmp_path / "figs"
    )
    elapsed = time.perf_counter() - started

    assert summary["n_cases"] == 200
    assert summary["verdict_mismatch_count"] == 0
    assert summary["expected_reject_accepted_count"] == 0
    assert summary["matrix_shards"] == 2
    assert elapsed < 60.0

    with (tmp_path / "cases.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200
    assert {row["operator"] for row in rows} == {
        "fixed_mu",
        "qeq",
        "scf",
    }
    assert all(row["matrix_path"] and row["matrix_prefix"] for row in rows)
    assert any(row["expected_status"] == "accept" for row in rows)
    assert any(row["expected_status"] == "reject" for row in rows)
    mutation_rows = [row for row in rows if row["family"] == "mutation"]
    assert Counter(row["mutation_kind"] for row in mutation_rows) == {
        "wrong_shape": 4,
        "nan_inf": 4,
        "non_hermitian": 4,
        "non_spd": 4,
        "truncated_eigenbasis": 4,
        "wrong_request": 4,
        "serialized_field_rewrite": 3,
        "self_reported_loose_tolerance": 3,
    }
    assert {
        row["family"]
        for row in rows
        if row["operator"] == "scf" and row["family"] != "mutation"
    } == {
        "one_level",
        "symmetric_dimer",
        "asymmetric_dimer",
        "random_small",
    }
    assert [path.name for path in figure_paths] == list(FIGURE_FILENAMES)
    assert all(path.is_file() and path.stat().st_size > 0 for path in figure_paths)

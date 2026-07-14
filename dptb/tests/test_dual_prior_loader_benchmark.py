from __future__ import annotations

from pathlib import Path

from tools import benchmark_dual_prior_loader as benchmark_module
from tools.build_nonsoc_dual_prior_ablation_configs import (
    DEFAULT_REFERENCE,
    build_configs,
)


class _FakeValidatedDataset:
    def __init__(self, records: int = 3):
        self._records = records
        self._validated_record_contracts = {}
        self._lmdb_env_cache = {}

    def __len__(self):
        return self._records

    def get(self, index: int):
        self._last_lmdb_pickle_bytes = (1 << 20) + int(index)
        self._last_lmdb_record_identity = ("/fake/data.lmdb", int(index))
        key = (("/fake/data.lmdb",), int(index))
        self._validated_record_contracts[key] = (None, None)
        return {"index": index}


def test_benchmarks_four_unique_loader_contracts_and_gate_cache(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "configs"
    build_configs(
        reference=DEFAULT_REFERENCE,
        dual_full_root=tmp_path / "dual",
        output_dir=config_dir,
        expected_p2_source_fingerprint="2" * 64,
        expected_p23_source_fingerprint="3" * 64,
    )
    builds = []

    def fake_build_dataset(**kwargs):
        builds.append(kwargs)
        return _FakeValidatedDataset()

    monkeypatch.setattr(benchmark_module, "build_dataset", fake_build_dataset)
    report = benchmark_module.benchmark_matrix(
        config_dir=config_dir,
        split="train",
        warm_repeats=2,
        max_records=2,
    )

    assert report["schema"] == benchmark_module.SCHEMA
    assert set(report["arms"]) == {
        "p2_residual",
        "p2_direct",
        "p23_residual",
        "p23_direct",
    }
    # Two fresh datasets per arm: probe/warm and sequential first pass.
    assert len(builds) == 8
    for name, result in report["arms"].items():
        cold = result["cold_first_get"]
        warm = result["warm_same_record"]
        sequential = result["sequential_first_pass"]
        assert cold["first_gate_executed"] is True
        assert cold["validation_cache_entries_before"] == 0
        assert cold["validation_cache_entries_after"] == 1
        assert cold["pickle_bytes"] == 1 << 20
        assert warm["records"] == 2
        assert warm["validation_cache_entries_before"] == 1
        assert warm["validation_cache_entries_after"] == 1
        assert warm["pickle_bytes"] == 2 * (1 << 20)
        assert sequential["records"] == 2
        assert sequential["pickle_bytes"] == 2 * (1 << 20) + 1
        assert sequential["validation_cache_entries_before"] == 0
        assert sequential["validation_cache_entries_after"] == 2
        assert sequential["complete_split"] is False
        assert sequential["records_per_second"] > 0.0
        assert sequential["mib_per_second"] > 0.0
        assert result["require_prior_ao_blocks"] is name.endswith("residual")

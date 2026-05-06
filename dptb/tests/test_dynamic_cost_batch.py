import logging
from contextlib import contextmanager
import pytest
import torch

from dptb.data.dataset._base_datasets import AtomicInMemoryDataset
from dptb.data.dataset.lmdb_dataset import LMDBDataset
from dptb.data import AtomicDataDict
from dptb.data.dataloader import (
    AtomicDataCostEstimator,
    DataLoader,
    DynamicCostBatchSampler,
    resolve_dynamic_batch_options,
    split_batch_for_oom,
)
from dptb.nnops.multi_trainer import MultiTrainer, _base_train_options_for_multitrainer
from dptb.utils.argcheck import dynamic_batch_options
from dptb.utils.torch_geometric import Batch, Data


class ToyDataset:
    def __init__(self, node_counts):
        self.node_counts = list(node_counts)

    def __len__(self):
        return len(self.node_counts)

    def __getitem__(self, idx):
        n = int(self.node_counts[idx])
        return Data(
            pos=torch.zeros((n, 3), dtype=torch.float32),
            edge_index=torch.zeros((2, n + 1), dtype=torch.long),
            env_index=torch.zeros((2, n + 2), dtype=torch.long),
            onsitenv_index=torch.zeros((2, n + 3), dtype=torch.long),
            kpoint=torch.zeros((2, 3), dtype=torch.float32),
            eigenvalue=torch.zeros((1, 4), dtype=torch.float32),
        )


class MetadataCostDataset(ToyDataset):
    def __init__(self, node_counts):
        super().__init__(node_counts)
        self.item_reads = 0
        self.metadata_reads = 0
        self.dynamic_batch_cost_version = 0

    def __getitem__(self, idx):
        self.item_reads += 1
        return super().__getitem__(idx)

    def get_dynamic_batch_cost_parts(self, idx):
        self.metadata_reads += 1
        return {"graph": 1, "node": int(self.node_counts[idx])}


class PreferLoadedCostDataset(MetadataCostDataset):
    prefer_loaded_dynamic_batch_cost_parts = True


class EdgeMetadataDataset(ToyDataset):
    def __init__(self, edge_counts):
        super().__init__([1 for _ in edge_counts])
        self.edge_counts = list(edge_counts)

    def get_dynamic_batch_cost_parts(self, idx):
        return {"graph": 1, "node": 1, "edge": int(self.edge_counts[idx])}


class CallableVersionDataset(MetadataCostDataset):
    def __init__(self, node_counts):
        super().__init__(node_counts)
        del self.dynamic_batch_cost_version
        self.version_value = 0

    def dynamic_batch_cost_version(self):
        return self.version_value


def test_cost_estimator_uses_deeptb_graph_terms():
    data = ToyDataset([4])[0]
    estimator = AtomicDataCostEstimator(
        mode="cost",
        cost_weights={
            "graph": 10.0,
            "node": 2.0,
            "edge": 3.0,
            "env": 5.0,
            "onsitenv": 7.0,
            "kpoint": 11.0,
            "eig_band_square": 0.0,
        },
    )

    parts = estimator.parts(data)

    assert parts["graph"] == 1
    assert parts["node"] == 4
    assert parts["edge"] == 5
    assert parts["env"] == 6
    assert parts["onsitenv"] == 7
    assert parts["kpoint"] == 2
    assert estimator(data) == 10 + 2 * 4 + 3 * 5 + 5 * 6 + 7 * 7 + 11 * 2


def test_dynamic_loader_caps_cost_and_keeps_batch_size_as_max_samples():
    dataset = ToyDataset([4, 5, 12, 3])
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "mode": "node",
            "max_cost": 10,
        },
    )

    batches = list(loader)

    assert [b.__dptb_sample_indices__ for b in batches] == [[0, 1], [2], [3]]
    assert [b.num_graphs for b in batches] == [2, 1, 1]
    assert batches[0].__dptb_batch_cost__ == 9
    assert batches[0].__dptb_batch_num_nodes__ == 9
    assert batches[0].__dptb_batch_max_item_cost__ == 5
    assert batches[1].__dptb_batch_cost__ == 12


def test_dynamic_sampler_caches_epoch_batches_for_len_and_iter():
    dataset = ToyDataset([1, 1, 1, 1])
    sampler = DynamicCostBatchSampler(
        dataset,
        max_cost=2,
        mode="node",
        max_samples=2,
        shuffle=False,
    )
    calls = {"make": 0}
    make_global_batches = sampler._make_global_batches

    def _counted_make_global_batches():
        calls["make"] += 1
        return make_global_batches()

    sampler._make_global_batches = _counted_make_global_batches

    assert len(sampler) == 2
    assert list(sampler) == [[0, 1], [2, 3]]
    assert len(sampler) == 2
    assert calls["make"] == 1

    sampler.max_cost = 3
    assert len(sampler) == 2
    assert calls["make"] == 2

    sampler.set_epoch(1)

    assert len(sampler) == 2
    assert calls["make"] == 3

    dataset.node_counts.append(1)
    assert len(sampler) == 3
    assert calls["make"] == 4


def test_dynamic_sampler_uses_metadata_costs_and_manual_invalidation():
    dataset = MetadataCostDataset([2, 2])
    sampler = DynamicCostBatchSampler(
        dataset,
        max_cost=10,
        mode="node",
        shuffle=False,
    )

    assert list(sampler) == [[0, 1]]
    assert dataset.item_reads == 0

    dataset.node_counts[0] = 20
    assert list(sampler) == [[0, 1]]

    sampler.invalidate_cache(clear_costs=True)

    assert list(sampler) == [[0], [1]]
    assert dataset.item_reads == 0


def test_dynamic_sampler_random_evict_does_not_refill_from_future_pool():
    dataset = EdgeMetadataDataset([9, 9, 9, 1, 1, 1, 1, 1])
    sampler = DynamicCostBatchSampler(
        dataset,
        max_cost=12,
        mode="edge",
        max_samples=4,
        shuffle=False,
        packing_strategy="random_evict",
        seed=123,
    )

    batches = list(sampler)
    first_batch = batches[0]
    flattened = [idx for batch in batches for idx in batch]
    edge_sums = [sum(dataset.edge_counts[idx] for idx in batch) for batch in batches]

    assert set(first_batch).issubset({0, 1, 2, 3})
    assert sorted(flattened) == list(range(len(dataset)))
    assert all(total <= 12 for total in edge_sums)
    assert sampler.last_packing_stats["strategy"] == "random_evict"
    assert sampler.last_packing_stats["evict_accepts"] > 0
    assert sampler.last_packing_stats["refill_accepts"] == 0
    assert sampler.last_packing_stats["putbacks"] > 0


def test_dynamic_sampler_random_evict_drops_singleton_tail():
    dataset = EdgeMetadataDataset([4, 4, 4, 4, 4])
    sampler = DynamicCostBatchSampler(
        dataset,
        max_cost=100,
        mode="edge",
        max_samples=2,
        shuffle=False,
        packing_strategy="random_evict",
        seed=123,
    )

    assert list(sampler) == [[0, 1], [2, 3]]
    assert sampler.last_packing_stats["tail_dropped"] == 1
    assert sampler.last_packing_stats["min_samples"] == 2


def test_dynamic_metadata_parts_are_normalized_when_dataset_returns_partial_parts():
    dataset = MetadataCostDataset([2, 2])
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "mode": "env",
            "max_cost": 100,
        },
    )

    batch = next(iter(loader))

    assert batch.__dptb_item_parts__[0] == {
        "graph": 1,
        "node": 2,
        "edge": 0,
        "env": 0,
        "onsitenv": 0,
        "kpoint": 0,
        "eig_band_square": 0,
    }
    assert batch.__dptb_item_costs__ == [1, 1]


def test_dynamic_sampler_dataset_cost_version_invalidates_cost_cache():
    dataset = MetadataCostDataset([2, 2])
    sampler = DynamicCostBatchSampler(
        dataset,
        max_cost=10,
        mode="node",
        shuffle=False,
    )

    assert list(sampler) == [[0, 1]]

    dataset.node_counts[0] = 20
    dataset.dynamic_batch_cost_version += 1

    assert list(sampler) == [[0], [1]]
    assert dataset.item_reads == 0


def test_dynamic_sampler_callable_cost_version_invalidates_cost_cache():
    dataset = CallableVersionDataset([2, 2])
    sampler = DynamicCostBatchSampler(
        dataset,
        max_cost=10,
        mode="node",
        shuffle=False,
    )

    assert list(sampler) == [[0, 1]]

    dataset.node_counts[0] = 20
    dataset.version_value += 1

    assert list(sampler) == [[0], [1]]
    assert dataset.item_reads == 0


def test_dynamic_sampler_cost_cache_invalidates_when_estimator_signature_changes():
    dataset = MetadataCostDataset([2, 2])
    sampler = DynamicCostBatchSampler(
        dataset,
        max_cost=10,
        mode="node",
        shuffle=False,
        cost_weights={"node": 1.0},
    )

    assert list(sampler) == [[0, 1]]

    sampler.cost_estimator.mode = "cost"
    sampler.cost_estimator.weights["node"] = 10.0

    assert list(sampler) == [[0], [1]]


def test_dynamic_loader_metadata_costs_drive_batch_metadata():
    dataset = MetadataCostDataset([2, 2])
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch={"enabled": True, "mode": "node", "max_cost": 100},
    )

    batch = next(iter(loader))

    assert dataset.item_reads == 2
    assert dataset.metadata_reads == 4
    assert batch.__dptb_item_costs__ == [2, 2]
    assert batch.__dptb_batch_cost__ == 4

    dataset.node_counts[0] = 20
    dataset.dynamic_batch_cost_version += 1
    loader.invalidate_dynamic_batch_cache(clear_costs=True)
    batch = next(iter(loader))

    assert dataset.metadata_reads == 8
    assert batch.__dptb_item_costs__[0] == 20
    assert batch.__dptb_batch_cost__ == 22


def test_dynamic_loader_prefers_loaded_data_when_dataset_opts_in():
    dataset = PreferLoadedCostDataset([2, 2])
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch={"enabled": True, "mode": "node", "max_cost": 100},
    )

    batch = next(iter(loader))

    assert dataset.item_reads == 2
    assert dataset.metadata_reads == 2
    assert batch.__dptb_item_costs__ == [2, 2]
    assert batch.__dptb_batch_cost__ == 4


def test_dynamic_batch_calibration_uses_metadata_cost_getter():
    dataset = MetadataCostDataset([2, 20, 4, 6])

    opts = resolve_dynamic_batch_options(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "mode": "node",
            "calibrate": True,
            "calibration_batches": 10,
            "calibration_quantile": 1.0,
        },
    )

    assert opts["calibration_batch_costs"] == [22, 10]
    assert opts["max_cost"] == 22
    assert dataset.item_reads == 0
    assert dataset.metadata_reads == 4


def test_dynamic_batch_resolved_calibration_does_not_rescan_dataset():
    dataset = MetadataCostDataset([2, 20, 4, 6])

    opts = resolve_dynamic_batch_options(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "mode": "node",
            "calibrate": True,
            "calibration_batches": 10,
            "calibration_quantile": 1.0,
        },
    )
    item_reads = dataset.item_reads
    metadata_reads = dataset.metadata_reads
    dataset.node_counts = [1000, 1000, 1000, 1000]

    reused = resolve_dynamic_batch_options(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch=opts,
    )

    assert opts["calibrated"] is True
    assert opts["max_cost"] == 22
    assert reused["max_cost"] == 22
    assert dataset.item_reads == item_reads
    assert dataset.metadata_reads == metadata_reads


def test_dynamic_batch_max_edge_alias_uses_edge_only_calibration():
    dataset = EdgeMetadataDataset([10, 20, 30, 40])

    opts = resolve_dynamic_batch_options(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "max_edge": 45,
            "cost_weights": {"node": 999.0, "edge": 1.0},
        },
    )

    assert opts["mode"] == "edge"
    assert opts["max_cost"] == 45
    assert opts["max_edge"] == 45
    assert opts["cost_weights"] is None

    opts = resolve_dynamic_batch_options(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "calibrate": True,
            "calibration_batches": 10,
            "calibration_quantile": 1.0,
            "cost_weights": {"node": 999.0, "edge": 1.0},
        },
    )

    assert opts["mode"] == "edge"
    assert opts["max_edge"] == 70
    assert opts["max_cost"] == 70
    assert opts["calibration_batch_costs"] == [30, 70]


def test_dynamic_batch_argcheck_accepts_edge_packing_options():
    arg = dynamic_batch_options()
    cfg = {
        "enabled": True,
        "mode": "edge",
        "max_edge": 128,
        "max_samples": 32,
        "min_samples": 2,
        "packing_strategy": "random_evict",
        "oom_fallback": True,
    }

    normalized = arg.normalize_value(cfg)
    arg.check_value(normalized, strict=True)

    assert normalized["max_edge"] == 128
    assert normalized["min_samples"] == 2
    assert normalized["calibration_batches"] == 1000
    assert normalized["calibration_quantile"] == 0.95
    assert normalized["packing_strategy"] == "random_evict"


def test_dynamic_loader_forwards_invalidate_dynamic_batch_cache():
    dataset = MetadataCostDataset([2, 2])
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch={"enabled": True, "mode": "node", "max_cost": 10},
    )

    assert list(loader.dynamic_batch_sampler) == [[0, 1]]
    dataset.node_counts[0] = 20
    assert list(loader.dynamic_batch_sampler) == [[0, 1]]

    loader.invalidate_dynamic_batch_cache(clear_costs=True)

    assert list(loader.dynamic_batch_sampler) == [[0], [1]]


def test_dynamic_sampler_padding_uses_deterministic_non_last_sources(caplog):
    dataset = ToyDataset([1, 1, 1])
    caplog.set_level(logging.INFO, logger="dptb.data.dataloader")
    sampler = DynamicCostBatchSampler(
        dataset,
        max_cost=1,
        mode="node",
        max_samples=1,
        shuffle=False,
        seed=123,
        num_steps=6,
    )

    batches = list(sampler)

    assert len(batches) == 6
    assert batches[:3] == [[0], [1], [2]]
    assert batches[3:] != [[2], [2], [2]]
    assert len({tuple(batch) for batch in batches[3:]}) > 1
    assert list(
        DynamicCostBatchSampler(
            dataset,
            max_cost=1,
            mode="node",
            max_samples=1,
            shuffle=False,
            seed=123,
            num_steps=6,
        )
    ) == batches
    assert any("num_steps" in record.message for record in caplog.records)
    assert sampler.last_padding_stats["reason"] == "num_steps"
    assert sampler.last_padding_stats["added"] == 3
    assert sampler.padding_events[-1] == sampler.last_padding_stats


def test_dynamic_sampler_padding_events_are_bounded():
    dataset = ToyDataset([1])
    sampler = DynamicCostBatchSampler(
        dataset,
        max_cost=1,
        mode="node",
        max_samples=1,
        shuffle=False,
        seed=123,
    )

    for _ in range(1030):
        sampler._pad_batches([[0]], 2, reason="num_steps")

    assert len(sampler.padding_events) == 1024
    assert sampler.padding_events[-1] == sampler.last_padding_stats


def test_dynamic_sampler_world_size_padding_keeps_equal_rank_lengths():
    dataset = ToyDataset([1, 1, 1])
    common = dict(
        dataset=dataset,
        max_cost=1,
        mode="node",
        max_samples=1,
        shuffle=False,
        seed=123,
        world_size=2,
    )

    rank0 = list(DynamicCostBatchSampler(rank=0, **common))
    rank1 = list(DynamicCostBatchSampler(rank=1, **common))

    assert len(rank0) == len(rank1) == 2
    assert rank0 != rank1
    assert {(0,), (1,), (2,)}.issubset(
        {tuple(batch) for batch in rank0 + rank1}
    )


def test_calibration_derives_quantile_from_fixed_batch_totals():
    dataset = ToyDataset([1, 2, 10, 20, 5])

    opts = resolve_dynamic_batch_options(
        dataset,
        batch_size=2,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "mode": "node",
            "calibrate": True,
            "calibration_batches": 10,
            "calibration_quantile": 0.5,
        },
    )

    assert opts["max_cost"] == 5
    assert opts["max_samples"] == 2
    assert opts["calibrated"] is True
    assert opts["calibration_batch_costs"] == [3, 30, 5]


def test_split_batch_for_oom_bisects_and_preserves_metadata():
    dataset = ToyDataset([2, 3, 4, 5])
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        dynamic_batch={
            "enabled": True,
            "mode": "node",
            "max_cost": 100,
        },
    )
    batch = next(iter(loader))

    left, right = split_batch_for_oom(batch)

    assert left.__dptb_sample_indices__ == [0, 1]
    assert right.__dptb_sample_indices__ == [2, 3]
    assert left.__dptb_batch_cost__ == 5
    assert right.__dptb_batch_cost__ == 9
    assert left.__dptb_item_costs__ == [2, 3]
    assert right.__dptb_item_costs__ == [4, 5]
    assert left.num_graphs == 2
    assert right.num_graphs == 2


def test_atomic_inmemory_dataset_exposes_dynamic_batch_cost_parts_without_get_example():
    data_list = [
        Data(
            pos=torch.zeros((2, 3)),
            edge_index=torch.zeros((2, 3), dtype=torch.long),
            env_index=torch.zeros((2, 4), dtype=torch.long),
            onsitenv_index=torch.zeros((2, 5), dtype=torch.long),
            kpoint=torch.zeros((2, 3)),
            eigenvalue=torch.zeros((2, 3)),
        ),
        Data(
            pos=torch.zeros((4, 3)),
            edge_index=torch.zeros((2, 6), dtype=torch.long),
            env_index=torch.zeros((2, 7), dtype=torch.long),
            onsitenv_index=torch.zeros((2, 8), dtype=torch.long),
            kpoint=torch.zeros((1, 3)),
            eigenvalue=torch.zeros((1, 3)),
        ),
    ]
    dataset = AtomicInMemoryDataset.__new__(AtomicInMemoryDataset)
    dataset.data = Batch.from_data_list(data_list)
    dataset._indices = None

    assert dataset.get_dynamic_batch_cost_parts(1) == {
        "graph": 1,
        "node": 4,
        "edge": 6,
        "env": 7,
        "onsitenv": 8,
        "kpoint": 1,
        "eig_band_square": 9,
    }


def test_lmdb_dataset_exposes_dynamic_batch_cost_parts_from_entry_metadata():
    dataset = LMDBDataset.__new__(LMDBDataset)
    dataset._indices = None
    dataset.num_graphs = 1
    data_dict = {
        AtomicDataDict.POSITIONS_KEY: torch.zeros((3, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.ones(3, dtype=torch.long),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.zeros((5, 4)),
        AtomicDataDict.KPOINT_KEY: torch.zeros((2, 3)),
        AtomicDataDict.ENERGY_EIGENVALUE_KEY: torch.zeros((2, 6)),
    }
    dataset._load_data_dict = lambda idx: data_dict

    assert dataset.get_dynamic_batch_cost_parts(0) == {
        "graph": 1,
        "node": 3,
        "edge": 5,
        "env": 0,
        "onsitenv": 0,
        "kpoint": 2,
        "eig_band_square": 72,
    }


def test_lmdb_dataset_uses_raw_block_keys_for_dynamic_edge_cost_without_get():
    dataset = LMDBDataset.__new__(LMDBDataset)
    dataset._indices = None
    dataset.num_graphs = 1
    dataset.h0_key = "hamiltonian_0"
    data_dict = {
        AtomicDataDict.POSITIONS_KEY: torch.zeros((3, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.ones(3, dtype=torch.long),
        "hamiltonian": {
            "0_0_0_0_0": torch.zeros((1, 1)),
            "1_1_0_0_0": torch.zeros((1, 1)),
            "0_1_0_0_0": torch.zeros((1, 1)),
            "1_0_0_0_0": torch.zeros((1, 1)),
            "0_2_1_0_0": torch.zeros((1, 1)),
        },
    }
    dataset._load_data_dict = lambda idx: data_dict

    def _unexpected_get(idx):
        raise AssertionError("dynamic batch edge-cost metadata must not materialize LMDB samples")

    dataset.get = _unexpected_get

    assert dataset.get_dynamic_batch_cost_parts(0) == {
        "graph": 1,
        "node": 3,
        "edge": 3,
        "env": 0,
        "onsitenv": 0,
        "kpoint": 0,
        "eig_band_square": 0,
    }


def test_lmdb_dataset_uses_raw_h0_block_keys_for_dynamic_edge_cost():
    dataset = LMDBDataset.__new__(LMDBDataset)
    dataset._indices = None
    dataset.num_graphs = 1
    dataset.h0_key = "hamiltonian_0"
    data_dict = {
        AtomicDataDict.POSITIONS_KEY: torch.zeros((2, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.ones(2, dtype=torch.long),
        "hamiltonian_0": {
            "0_0_0_0_0": torch.zeros((1, 1)),
            "1_1_0_0_0": torch.zeros((1, 1)),
            "0_1_0_0_0": torch.zeros((1, 1)),
            "1_0_0_0_0": torch.zeros((1, 1)),
        },
    }
    dataset._load_data_dict = lambda idx: data_dict

    assert dataset.get_dynamic_batch_cost_parts(0)["edge"] == 2


def test_lmdb_dataset_dynamic_cost_parts_are_cached():
    dataset = LMDBDataset.__new__(LMDBDataset)
    dataset._indices = None
    dataset.num_graphs = 1
    dataset.h0_key = "hamiltonian_0"
    loads = {"count": 0}
    data_dict = {
        AtomicDataDict.POSITIONS_KEY: torch.zeros((2, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.ones(2, dtype=torch.long),
        "hamiltonian": {
            "0_0_0_0_0": torch.zeros((1, 1)),
            "0_1_0_0_0": torch.zeros((1, 1)),
        },
    }

    def _load_data_dict(idx):
        loads["count"] += 1
        return data_dict

    dataset._load_data_dict = _load_data_dict

    assert dataset.get_dynamic_batch_cost_parts(0)["edge"] == 1
    assert dataset.get_dynamic_batch_cost_parts(0)["edge"] == 1
    assert loads["count"] == 1

    dataset.invalidate_dynamic_batch_costs()
    assert dataset.get_dynamic_batch_cost_parts(0)["edge"] == 1
    assert loads["count"] == 2


def test_lmdb_dataset_h0_raw_conversion_can_override_precomputed_h0(monkeypatch):
    dataset = LMDBDataset.__new__(LMDBDataset)
    dataset._indices = None
    dataset.get_Hamiltonian = False
    dataset.get_H0 = True
    dataset.get_overlap = False
    dataset.get_DM = False
    dataset.get_eigenvalues = False
    dataset.orthogonal = False
    dataset.h0_key = "hamiltonian_0"
    dataset.prefer_precomputed_h0 = False
    dataset.transform = object()
    dataset.file_map = ["data.0000.lmdb"]
    dataset.info_files = {
        "data.0000.lmdb": {
            "r_max": 1.0,
            "er_max": None,
            "oer_max": None,
            "wave_align": False,
            "train_w_homo_lumo_gap": False,
            "train_w_eps": False,
            "train_w_charge": False,
            "train_dip": False,
            "train_polar": False,
        }
    }
    h0_blocks = {"0_0_0_0_0": torch.zeros((1, 1))}
    data_dict = {
        AtomicDataDict.CELL_KEY: torch.eye(3),
        AtomicDataDict.POSITIONS_KEY: torch.zeros((2, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.ones(2, dtype=torch.long),
        AtomicDataDict.PBC_KEY: torch.zeros(3, dtype=torch.bool),
        "hamiltonian_0": h0_blocks,
        AtomicDataDict.NODE_H0_KEY: torch.full((2, 3), 9.0),
        AtomicDataDict.EDGE_H0_KEY: torch.full((4, 3), 9.0),
    }
    dataset._load_data_dict = lambda idx: data_dict

    def _fake_from_points(**kwargs):
        return Data(
            pos=torch.as_tensor(kwargs["pos"]),
            edge_index=torch.zeros((2, 4), dtype=torch.long),
            edge_cell_shift=torch.zeros((4, 3), dtype=torch.long),
            atomic_numbers=torch.as_tensor(kwargs["atomic_numbers"]),
        )

    calls = []

    def _fake_block_to_feature(atomicdata, type_mapper, blocks, overlap, orthogonal, **kwargs):
        calls.append((blocks, kwargs))
        atomicdata[kwargs["node_field"]] = torch.full((2, 3), 1.0)
        atomicdata[kwargs["edge_field"]] = torch.full((4, 3), 2.0)

    monkeypatch.setattr("dptb.data.dataset.lmdb_dataset.AtomicData.from_points", _fake_from_points)
    monkeypatch.setattr("dptb.data.dataset.lmdb_dataset.block_to_feature", _fake_block_to_feature)

    out = dataset.get(0)

    assert calls == [(h0_blocks, {"node_field": AtomicDataDict.NODE_H0_KEY, "edge_field": AtomicDataDict.EDGE_H0_KEY})]
    assert torch.equal(out[AtomicDataDict.NODE_H0_KEY], torch.full((2, 3), 1.0))
    assert torch.equal(out[AtomicDataDict.EDGE_H0_KEY], torch.full((4, 3), 2.0))


def test_multitrainer_base_options_disable_dynamic_batch_before_rebuild():
    train_options = {
        "batch_size": 4,
        "dynamic_batch": {"enabled": True, "calibrate": True},
    }

    base_options = _base_train_options_for_multitrainer(train_options)

    assert base_options["dynamic_batch"]["enabled"] is False
    assert train_options["dynamic_batch"]["enabled"] is True
    assert base_options["dynamic_batch"]["calibrate"] is True


class _DummyTagger:
    @contextmanager
    def tag(self, *args, **kwargs):
        yield

    def dump_cuda_mem_summary(self, where):
        return None


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


def _make_oom_skip_fallback_trainer(param, opt, observed_states):
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.distributed_expert = False
    trainer.distributed_rank0_prepare_batch = False
    trainer.ref_batch = None
    trainer.reference_batch = None
    trainer.distance_ranges = [(0.0, 1.0)]
    trainer.num_experts = 1
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer.iter = 7
    trainer.display_sync_freq = 1000
    trainer.clip_grad_norm = 1000.0
    trainer.optimizers = [opt]
    trainer.train_lossfunc = object()
    trainer.dynamic_batch_oom_shrink_factor = 0.8
    trainer.model = torch.nn.Linear(1, 1)
    trainer._tagger = _DummyTagger()
    trainer._maybe_profile_iteration = lambda iteration: _DummyTagger().tag()
    trainer._t_last_iter_end = None
    trainer.debug_tags = False
    trainer.debug_tag_freq = 1
    trainer.train_loader = type(
        "Loader",
        (),
        {
            "batch_sampler": type("Sampler", (), {"max_cost": 100})(),
            "dynamic_batch_options": {"max_cost": 100},
        },
    )()
    trainer._is_cuda_device = lambda: False
    trainer._reset_cuda_memory_peak = lambda: None
    trainer._expert_parameters = lambda expert_idx: [param]
    trainer._to_float_scalar = lambda x: float(x.detach().item() if torch.is_tensor(x) else x)
    trainer._to_int_scalar = lambda x: int(x.detach().item() if torch.is_tensor(x) else x)
    trainer._compute_stitched_loss_by_reduce = lambda payloads, criterion: None
    trainer._local_scheduler_step = lambda metric: None
    trainer._add_cuda_memory_state = lambda state, metrics: None
    trainer._gather_cuda_memory_metrics = lambda: {}
    trainer.call_plugins = lambda queue_name, time, **state: observed_states.append(state)
    return trainer


def _build_cost_payload(param, batch_dict, batch_info, expert_idx, range_dis):
    loss = param * batch_dict["cost"]
    return {
        "loss": loss,
        "expert_onsite": loss.detach(),
        "expert_hopping": loss.detach(),
        "onsite_weighted_sum": loss.detach(),
        "hopping_weighted_sum": loss.detach(),
        "active_nodes": torch.tensor(1.0),
        "active_edges": torch.tensor(1.0),
        "onsite_l1_sum": None,
        "onsite_mse_sum": None,
        "onsite_cnt": None,
        "hopping_l1_sum": None,
        "hopping_mse_sum": None,
        "hopping_cnt": None,
        "z_values": [],
        "load_cv_values": [],
    }


def test_multitrainer_iteration_oom_fallback_skips_batch_without_step():
    dataset = ToyDataset([2, 3, 4, 5])
    batch = next(iter(DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        dynamic_batch={"enabled": True, "mode": "node", "max_cost": 100},
    )))

    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = _CountingSGD([param])
    observed_states = []
    prepare_calls = {"count": 0}
    trainer = _make_oom_skip_fallback_trainer(param, opt, observed_states)
    trainer.dynamic_batch_enabled = True
    trainer.dynamic_batch_oom_fallback = True

    def _prepare_batch_bundle(batch, with_lengths=True):
        prepare_calls["count"] += 1
        if prepare_calls["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        return {"cost": torch.tensor(float(batch.__dptb_batch_cost__))}, {}

    trainer._prepare_batch_bundle = _prepare_batch_bundle

    trainer._build_train_payload = lambda batch_dict, batch_info, expert_idx, range_dis, **kwargs: (
        _build_cost_payload(param, batch_dict, batch_info, expert_idx, range_dis)
    )

    loss = trainer.iteration(batch)

    assert loss is None
    assert prepare_calls["count"] == 1
    assert opt.step_calls == 0
    assert trainer.iter == 7
    assert trainer.train_loader.batch_sampler.max_cost == 80
    assert observed_states == []
    assert trainer.dynamic_batch_oom_skipped_iters == 1


def test_multitrainer_iteration_oom_fallback_disabled_reraises():
    dataset = ToyDataset([2, 3, 4, 5])
    batch = next(iter(DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        dynamic_batch={"enabled": True, "mode": "node", "max_cost": 100},
    )))
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = _CountingSGD([param])
    trainer = _make_oom_skip_fallback_trainer(param, opt, [])
    trainer.dynamic_batch_enabled = True
    trainer.dynamic_batch_oom_fallback = False
    trainer._prepare_batch_bundle = lambda batch, with_lengths=True: (
        (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        trainer.iteration(batch)

    assert opt.step_calls == 0


def test_multitrainer_iteration_oom_after_optimizer_step_does_not_retry():
    dataset = ToyDataset([2, 3, 4, 5])
    batch = next(iter(DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        dynamic_batch={"enabled": True, "mode": "node", "max_cost": 100},
    )))
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = _CountingSGD([param])
    trainer = _make_oom_skip_fallback_trainer(param, opt, [])
    trainer.dynamic_batch_enabled = True
    trainer.dynamic_batch_oom_fallback = True
    trainer._prepare_batch_bundle = lambda batch, with_lengths=True: (
        {"cost": torch.tensor(float(batch.__dptb_batch_cost__))},
        {},
    )
    trainer._build_train_payload = lambda batch_dict, batch_info, expert_idx, range_dis, **kwargs: (
        _build_cost_payload(param, batch_dict, batch_info, expert_idx, range_dis)
    )
    trainer._local_scheduler_step = lambda metric: (
        (_ for _ in ()).throw(RuntimeError("CUDA out of memory after optimizer"))
    )

    with pytest.raises(RuntimeError, match="out of memory after optimizer"):
        trainer.iteration(batch)

    assert opt.step_calls == 1


def test_multitrainer_distributed_oom_fallback_does_not_sync_without_local_oom(monkeypatch):
    dataset = ToyDataset([2, 3, 4, 5])
    batch = next(iter(DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        dynamic_batch={"enabled": True, "mode": "node", "max_cost": 100},
    )))
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = _CountingSGD([param])
    trainer = _make_oom_skip_fallback_trainer(param, opt, [])
    trainer.distributed_expert = True
    trainer.dynamic_batch_enabled = True
    trainer.dynamic_batch_oom_fallback = True
    trainer.local_expert_idx = 0
    trainer.rank = 0
    trainer.world_size = 2
    trainer._dist_ready = lambda: True

    def _unexpected_all_reduce(*args, **kwargs):
        raise AssertionError("normal OOM fallback path must not call all_reduce")

    monkeypatch.setattr("dptb.nnops.multi_trainer.dist.all_reduce", _unexpected_all_reduce)

    skipped = trainer._maybe_skip_dynamic_batch_after_oom(
        batch,
        local_oom=False,
        where="probe",
    )

    assert skipped is False
    assert opt.step_calls == 0
    assert trainer.iter == 7
    assert trainer.train_loader.batch_sampler.max_cost == 100


def test_multitrainer_distributed_oom_fallback_disabled_for_expert_dp_replicas():
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = _CountingSGD([param])
    trainer = _make_oom_skip_fallback_trainer(param, opt, [])
    trainer.distributed_expert = True
    trainer.expert_data_parallel_size = 2
    trainer.dynamic_batch_enabled = True
    trainer.dynamic_batch_oom_fallback = True

    assert trainer._can_skip_dynamic_batch_after_oom() is False


def test_multitrainer_distributed_oom_fallback_skips_after_local_oom(monkeypatch):
    dataset = ToyDataset([2, 3, 4, 5])
    batch = next(iter(DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        dynamic_batch={"enabled": True, "mode": "node", "max_cost": 100},
    )))
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = _CountingSGD([param])
    trainer = _make_oom_skip_fallback_trainer(param, opt, [])
    trainer.distributed_expert = True
    trainer.dynamic_batch_enabled = True
    trainer.dynamic_batch_oom_fallback = True
    trainer.local_expert_idx = 0
    trainer.rank = 0
    trainer.world_size = 2
    trainer.iter = 2
    trainer.display_sync_freq = 1000
    trainer._dist_ready = lambda: True
    trainer._prepare_batch_bundle = lambda batch, with_lengths=True: (
        {"cost": torch.tensor(float(batch.__dptb_batch_cost__))},
        {},
    )
    trainer._build_train_payload = lambda **kwargs: (
        (_ for _ in ()).throw(RuntimeError("CUDA out of memory during forward"))
    )

    def _unexpected_all_reduce(*args, **kwargs):
        raise AssertionError("OOM fallback must not add distributed synchronization")

    monkeypatch.setattr("dptb.nnops.multi_trainer.dist.all_reduce", _unexpected_all_reduce)

    loss = trainer.iteration(batch)

    assert loss is None
    assert opt.step_calls == 0
    assert trainer.iter == 3
    assert trainer.train_loader.batch_sampler.max_cost == 80
    assert trainer.dynamic_batch_oom_skipped_iters == 1


def _dynamic_batch_cfg_probe(**overrides):
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.dynamic_batch_enabled = overrides.pop("dynamic_batch_enabled", True)
    trainer.dynamic_batch_options = {
        "enabled": True,
        "max_cost": 100,
        "rank": 99,
        "world_size": 99,
    }
    trainer.dynamic_batch_options.update(overrides.pop("dynamic_batch_options", {}))
    trainer.distributed_expert = overrides.pop("distributed_expert", False)
    trainer.distributed_rank0_prepare_batch = overrides.pop(
        "distributed_rank0_prepare_batch", False
    )
    trainer.expert_data_parallel_size = overrides.pop("expert_data_parallel_size", 1)
    trainer.expert_dp_rank = overrides.pop("expert_dp_rank", 0)
    assert not overrides
    return trainer._dynamic_batch_cfg_for_train_loader()


def test_multitrainer_dynamic_batch_cfg_keeps_expert_parallel_unsharded():
    cfg = _dynamic_batch_cfg_probe(
        distributed_expert=True,
        expert_data_parallel_size=1,
    )

    assert cfg["rank"] == 0
    assert cfg["world_size"] == 1


def test_multitrainer_dynamic_batch_cfg_shards_expert_dp_replicas():
    cfg = _dynamic_batch_cfg_probe(
        distributed_expert=True,
        expert_data_parallel_size=2,
        expert_dp_rank=1,
    )

    assert cfg["rank"] == 1
    assert cfg["world_size"] == 2


def test_multitrainer_dynamic_batch_cfg_keeps_rank0_prepare_unsharded():
    cfg = _dynamic_batch_cfg_probe(
        distributed_expert=True,
        distributed_rank0_prepare_batch=True,
        expert_data_parallel_size=2,
        expert_dp_rank=1,
    )

    assert cfg["rank"] == 0
    assert cfg["world_size"] == 1


def test_multitrainer_dynamic_batch_cfg_disabled_returns_none():
    assert _dynamic_batch_cfg_probe(dynamic_batch_enabled=False) is None


def test_multitrainer_dynamic_batch_cfg_ignores_global_dist_by_default(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="dptb.nnops.multi_trainer")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

    cfg = _dynamic_batch_cfg_probe(distributed_expert=False)

    assert cfg["rank"] == 0
    assert cfg["world_size"] == 1
    assert any("ordinary DDP should set dynamic_batch.use_global_dist=true" in record.message for record in caplog.records)


def test_multitrainer_dynamic_batch_cfg_can_opt_into_global_dist(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

    cfg = _dynamic_batch_cfg_probe(
        distributed_expert=False,
        dynamic_batch_options={"use_global_dist": True},
    )

    assert cfg["rank"] == 2
    assert cfg["world_size"] == 4

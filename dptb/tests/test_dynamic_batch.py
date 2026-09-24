"""Dynamic-batch sampler/loader: cost-capped packing, cache invalidation,
calibration, deterministic padding, equal rank lengths, and per-dataset
cost-part extraction (in-memory and LMDB).
"""
import pytest
import torch

from dptb.data import AtomicDataDict
from dptb.data.dataloader import (
    AtomicDataCostEstimator,
    DataLoader,
    DynamicCostBatchSampler,
    resolve_dynamic_batch_options,
    split_batch_for_oom,
)
from dptb.data.dataset._base_datasets import AtomicInMemoryDataset
from dptb.data.dataset.lmdb_dataset import LMDBDataset
from dptb.nnops.multi_trainer import _base_train_options_for_multitrainer
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
        return {"block": int(self.node_counts[idx])}


class PreferLoadedCostDataset(MetadataCostDataset):
    prefer_loaded_dynamic_batch_cost_parts = True


class EdgeMetadataDataset(ToyDataset):
    def __init__(self, edge_counts):
        super().__init__([1 for _ in edge_counts])
        self.edge_counts = list(edge_counts)

    def get_dynamic_batch_cost_parts(self, idx):
        return {"edge": int(self.edge_counts[idx])}


class CallableVersionDataset(MetadataCostDataset):
    def __init__(self, node_counts):
        super().__init__(node_counts)
        del self.dynamic_batch_cost_version
        self.version_value = 0

    def dynamic_batch_cost_version(self):
        return self.version_value


class BlockEdgeMetadataDataset(ToyDataset):
    def __init__(self, block_counts, edge_counts):
        super().__init__([1 for _ in block_counts])
        self.block_counts = list(block_counts)
        self.edge_counts = list(edge_counts)

    def get_dynamic_batch_cost_parts(self, idx):
        return {"block": int(self.block_counts[idx]), "edge": int(self.edge_counts[idx])}


# ---------------------------------------------------------------------------
# cost estimation and packing
# ---------------------------------------------------------------------------
def test_cost_estimator_uses_block_or_edge_counts_only():
    data = ToyDataset([4])[0]
    edge_estimator = AtomicDataCostEstimator(mode="edge")
    block_estimator = AtomicDataCostEstimator(mode="block")

    parts = edge_estimator.parts(data)

    assert parts["block"] == 0
    assert parts["edge"] == 5
    assert edge_estimator(data) == 5
    assert block_estimator(data) == 5
    assert block_estimator.from_parts({"block": 7, "edge": 5}) == 7
    assert edge_estimator.from_parts({"block": 7, "edge": 5}) == 5


def test_dynamic_loader_caps_cost_and_keeps_batch_size_as_max_samples():
    dataset = MetadataCostDataset([4, 5, 12, 3])
    loader = DataLoader(dataset, batch_size=3, shuffle=False,
                        dynamic_batch={"enabled": True, "mode": "block", "max_cost": 10})

    batches = list(loader)

    assert [b.__dptb_sample_indices__ for b in batches] == [[0, 1], [2], [3]]
    assert [b.num_graphs for b in batches] == [2, 1, 1]
    assert batches[0].__dptb_batch_cost__ == 9
    assert batches[0].__dptb_batch_num_nodes__ == 9
    assert batches[0].__dptb_batch_max_item_cost__ == 5
    assert batches[1].__dptb_batch_cost__ == 12


def test_dynamic_sampler_random_evict_does_not_refill_from_future_pool():
    dataset = EdgeMetadataDataset([9, 9, 9, 1, 1, 1, 1, 1])
    sampler = DynamicCostBatchSampler(dataset, max_cost=12, mode="edge", max_samples=4, shuffle=False,
                                      packing_strategy="random_evict", seed=123)

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
    sampler = DynamicCostBatchSampler(dataset, max_cost=100, mode="edge", max_samples=2, shuffle=False,
                                      packing_strategy="random_evict", seed=123)

    assert list(sampler) == [[0, 1], [2, 3]]
    assert sampler.last_packing_stats["tail_dropped"] == 1
    assert sampler.last_packing_stats["min_samples"] == 2


def test_dynamic_metadata_parts_are_normalized_when_dataset_returns_partial_parts():
    dataset = MetadataCostDataset([2, 2])
    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        dynamic_batch={"enabled": True, "mode": "block", "max_cost": 100})

    batch = next(iter(loader))

    assert batch.__dptb_item_parts__[0] == {"block": 2, "edge": 0}
    assert batch.__dptb_item_costs__ == [2, 2]


def test_dynamic_loader_metadata_costs_drive_batch_metadata():
    dataset = MetadataCostDataset([2, 2])
    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        dynamic_batch={"enabled": True, "mode": "block", "max_cost": 100})

    batch = next(iter(loader))

    assert dataset.item_reads == 2
    assert batch.__dptb_item_costs__ == [2, 2]
    assert batch.__dptb_batch_cost__ == 4

    dataset.node_counts[0] = 20
    dataset.dynamic_batch_cost_version += 1
    loader.invalidate_dynamic_batch_cache(clear_costs=True)
    batch = next(iter(loader))

    assert batch.__dptb_item_costs__[0] == 20
    assert batch.__dptb_batch_cost__ == 22


def test_dynamic_loader_prefers_loaded_data_when_dataset_opts_in():
    dataset = PreferLoadedCostDataset([2, 2])
    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        dynamic_batch={"enabled": True, "mode": "block", "max_cost": 100})

    batch = next(iter(loader))

    assert dataset.item_reads == 2
    assert dataset.metadata_reads == 2
    assert batch.__dptb_item_costs__ == [3, 3]
    assert batch.__dptb_batch_cost__ == 6


# ---------------------------------------------------------------------------
# cache invalidation: only a real trigger recomputes the epoch's batches
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "trigger",
    ["set_epoch_or_length_change", "manual_invalidate", "dataset_cost_version_int",
     "dataset_cost_version_callable", "estimator_mode_change", "loader_invalidate"],
)
def test_dynamic_sampler_cache_recomputes_only_after_a_trigger(trigger):
    if trigger == "set_epoch_or_length_change":
        dataset = MetadataCostDataset([1, 1, 1, 1])
        sampler = DynamicCostBatchSampler(dataset, max_cost=2, mode="block", max_samples=2, shuffle=False)
        assert len(sampler) == 2 and list(sampler) == [[0, 1], [2, 3]]
        sampler.max_cost = 3  # still 2 batches at this size, but the cache must not go stale
        assert len(sampler) == 2
        sampler.set_epoch(1)
        assert len(sampler) == 2
        dataset.node_counts.append(1)
        assert len(sampler) == 3  # dataset length change is always visible
    elif trigger == "manual_invalidate":
        dataset = MetadataCostDataset([2, 2])
        sampler = DynamicCostBatchSampler(dataset, max_cost=10, mode="block", shuffle=False)
        assert list(sampler) == [[0, 1]]
        dataset.node_counts[0] = 20
        assert list(sampler) == [[0, 1]]  # stale cache: no trigger yet
        sampler.invalidate_cache(clear_costs=True)
        assert list(sampler) == [[0], [1]]
        assert dataset.item_reads == 0
    elif trigger == "dataset_cost_version_int":
        dataset = MetadataCostDataset([2, 2])
        sampler = DynamicCostBatchSampler(dataset, max_cost=10, mode="block", shuffle=False)
        assert list(sampler) == [[0, 1]]
        dataset.node_counts[0] = 20
        dataset.dynamic_batch_cost_version += 1
        assert list(sampler) == [[0], [1]]
        assert dataset.item_reads == 0
    elif trigger == "dataset_cost_version_callable":
        dataset = CallableVersionDataset([2, 2])
        sampler = DynamicCostBatchSampler(dataset, max_cost=10, mode="block", shuffle=False)
        assert list(sampler) == [[0, 1]]
        dataset.node_counts[0] = 20
        dataset.version_value += 1
        assert list(sampler) == [[0], [1]]
        assert dataset.item_reads == 0
    elif trigger == "estimator_mode_change":
        dataset = BlockEdgeMetadataDataset([2, 2], [20, 2])
        sampler = DynamicCostBatchSampler(dataset, max_cost=10, mode="block", shuffle=False)
        assert list(sampler) == [[0, 1]]
        sampler.cost_estimator.mode = "edge"
        assert list(sampler) == [[0], [1]]
    elif trigger == "loader_invalidate":
        dataset = MetadataCostDataset([2, 2])
        loader = DataLoader(dataset, batch_size=2, shuffle=False,
                            dynamic_batch={"enabled": True, "mode": "block", "max_cost": 10})
        assert list(loader.dynamic_batch_sampler) == [[0, 1]]
        dataset.node_counts[0] = 20
        assert list(loader.dynamic_batch_sampler) == [[0, 1]]
        loader.invalidate_dynamic_batch_cache(clear_costs=True)
        assert list(loader.dynamic_batch_sampler) == [[0], [1]]


# ---------------------------------------------------------------------------
# calibration: max_cost derives from a quantile of sampled batch totals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("quantile", "expected_max_cost", "expected_costs"),
    [(1.0, 22, [22, 10]), (0.5, 5, [3, 30, 5])],
    ids=["quantile_1_0", "quantile_0_5"],
)
def test_dynamic_batch_calibration_derives_max_cost_from_quantile(quantile, expected_max_cost, expected_costs):
    node_counts = [2, 20, 4, 6] if quantile == 1.0 else [1, 2, 10, 20, 5]
    dataset = MetadataCostDataset(node_counts)

    opts = resolve_dynamic_batch_options(
        dataset, batch_size=2, shuffle=False,
        dynamic_batch={"enabled": True, "mode": "block", "calibrate": True,
                       "calibration_batches": 10, "calibration_quantile": quantile},
    )

    assert opts["calibration_batch_costs"] == expected_costs
    assert opts["max_cost"] == expected_max_cost
    assert opts["calibrated"] is True
    assert dataset.item_reads == 0


def test_dynamic_batch_resolved_calibration_does_not_rescan_dataset():
    dataset = MetadataCostDataset([2, 20, 4, 6])

    opts = resolve_dynamic_batch_options(
        dataset, batch_size=2, shuffle=False,
        dynamic_batch={"enabled": True, "mode": "block", "calibrate": True,
                       "calibration_batches": 10, "calibration_quantile": 1.0},
    )
    item_reads = dataset.item_reads
    metadata_reads = dataset.metadata_reads
    dataset.node_counts = [1000, 1000, 1000, 1000]

    reused = resolve_dynamic_batch_options(dataset, batch_size=2, shuffle=False, dynamic_batch=opts)

    assert opts["calibrated"] is True
    assert opts["max_cost"] == 22
    assert reused["max_cost"] == 22
    assert dataset.item_reads == item_reads
    assert dataset.metadata_reads == metadata_reads


def test_dynamic_batch_max_edge_alias_keeps_default_block_mode():
    dataset = EdgeMetadataDataset([10, 20, 30, 40])

    opts = resolve_dynamic_batch_options(dataset, batch_size=2, shuffle=False,
                                         dynamic_batch={"enabled": True, "max_edge": 45})
    assert opts["mode"] == "block"
    assert opts["max_cost"] == 45
    assert opts["max_edge"] == 45

    opts = resolve_dynamic_batch_options(
        dataset, batch_size=2, shuffle=False,
        dynamic_batch={"enabled": True, "calibrate": True, "calibration_batches": 10, "calibration_quantile": 1.0},
    )
    assert opts["mode"] == "block"
    assert opts["max_edge"] == 70
    assert opts["max_cost"] == 70
    assert opts["calibration_batch_costs"] == [30, 70]


def test_dynamic_batch_argcheck_accepts_edge_packing_options():
    arg = dynamic_batch_options()
    cfg = {"enabled": True, "mode": "edge", "max_edge": 128, "max_samples": 32, "min_samples": 2,
          "packing_strategy": "random_evict", "oom_fallback": True}

    normalized = arg.normalize_value(cfg)
    arg.check_value(normalized, strict=True)

    assert normalized["max_edge"] == 128
    assert normalized["min_samples"] == 2
    assert normalized["calibration_batches"] == 1000
    assert normalized["calibration_quantile"] == 0.95
    assert normalized["packing_strategy"] == "random_evict"


@pytest.mark.parametrize("extra, warns", [({}, False), ({"max_samples": 2}, False), ({"max_samples": 96}, True)])
def test_dynamic_batch_warns_when_explicit_max_samples_differs_from_batch_size(caplog, extra, warns):
    dataset = MetadataCostDataset([2, 20, 4, 6])
    with caplog.at_level("WARNING", logger="dptb.data.dataloader"):
        opts = resolve_dynamic_batch_options(dataset, batch_size=2, shuffle=False,
                                             dynamic_batch={"enabled": True, "mode": "block", "max_cost": 50, **extra})
    assert opts["max_samples"] == extra.get("max_samples", 2)
    assert any("max_samples" in r.getMessage() for r in caplog.records) is warns


def test_multitrainer_base_options_disable_dynamic_batch_before_rebuild():
    train_options = {"batch_size": 4, "dynamic_batch": {"enabled": True, "calibrate": True}}

    base_options = _base_train_options_for_multitrainer(train_options)

    assert base_options["dynamic_batch"]["enabled"] is False
    assert train_options["dynamic_batch"]["enabled"] is True
    assert base_options["dynamic_batch"]["calibrate"] is True


# ---------------------------------------------------------------------------
# padding: deterministic non-last sources, bounded event history, equal ranks
# ---------------------------------------------------------------------------
def test_dynamic_sampler_padding_uses_deterministic_non_last_sources():
    dataset = ToyDataset([1, 1, 1])
    sampler = DynamicCostBatchSampler(dataset, max_cost=1, mode="block", max_samples=1, shuffle=False,
                                      seed=123, num_steps=6)

    batches = list(sampler)

    assert len(batches) == 6
    assert batches[:3] == [[0], [1], [2]]
    assert batches[3:] != [[2], [2], [2]]
    assert len({tuple(batch) for batch in batches[3:]}) > 1
    assert list(DynamicCostBatchSampler(dataset, max_cost=1, mode="block", max_samples=1, shuffle=False,
                                        seed=123, num_steps=6)) == batches
    assert sampler.last_padding_stats["reason"] == "num_steps"
    assert sampler.last_padding_stats["added"] == 3
    assert sampler.padding_events[-1] == sampler.last_padding_stats


def test_dynamic_sampler_padding_events_are_bounded():
    dataset = ToyDataset([1])
    sampler = DynamicCostBatchSampler(dataset, max_cost=1, mode="block", max_samples=1, shuffle=False, seed=123)

    for _ in range(1030):
        sampler._pad_batches([[0]], 2, reason="num_steps")

    assert len(sampler.padding_events) <= 1024
    assert sampler.padding_events[-1] == sampler.last_padding_stats


def test_dynamic_sampler_world_size_padding_keeps_equal_rank_lengths():
    dataset = ToyDataset([1, 1, 1])
    common = dict(dataset=dataset, max_cost=1, mode="block", max_samples=1, shuffle=False, seed=123, world_size=2)

    rank0 = list(DynamicCostBatchSampler(rank=0, **common))
    rank1 = list(DynamicCostBatchSampler(rank=1, **common))

    assert len(rank0) == len(rank1) == 2
    assert rank0 != rank1
    assert {(0,), (1,), (2,)}.issubset({tuple(batch) for batch in rank0 + rank1})


def test_split_batch_for_oom_bisects_and_preserves_metadata():
    dataset = MetadataCostDataset([2, 3, 4, 5])
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        dynamic_batch={"enabled": True, "mode": "block", "max_cost": 100})
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


# ---------------------------------------------------------------------------
# per-dataset cost-part extraction
# ---------------------------------------------------------------------------
def test_atomic_inmemory_dataset_exposes_dynamic_batch_cost_parts_without_get_example():
    data_list = [
        Data(pos=torch.zeros((2, 3)), edge_index=torch.zeros((2, 3), dtype=torch.long),
            env_index=torch.zeros((2, 4), dtype=torch.long), onsitenv_index=torch.zeros((2, 5), dtype=torch.long),
            kpoint=torch.zeros((2, 3)), eigenvalue=torch.zeros((2, 3))),
        Data(pos=torch.zeros((4, 3)), edge_index=torch.zeros((2, 6), dtype=torch.long),
            env_index=torch.zeros((2, 7), dtype=torch.long), onsitenv_index=torch.zeros((2, 8), dtype=torch.long),
            kpoint=torch.zeros((1, 3)), eigenvalue=torch.zeros((1, 3))),
    ]
    dataset = AtomicInMemoryDataset.__new__(AtomicInMemoryDataset)
    dataset.data = Batch.from_data_list(data_list)
    dataset._indices = None

    assert dataset.get_dynamic_batch_cost_parts(1) == {"block": 0, "edge": 6}


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

    assert dataset.get_dynamic_batch_cost_parts(0) == {"block": 0, "edge": 5}


@pytest.mark.parametrize(
    ("raw_key", "block_count"),
    [("hamiltonian", 3), ("hamiltonian_0", 2)],
    ids=["raw_hamiltonian_keys", "raw_h0_keys"],
)
def test_lmdb_dataset_raw_block_cost_uses_stored_keys_without_loading_and_caches(raw_key, block_count):
    dataset = LMDBDataset.__new__(LMDBDataset)
    dataset._indices = None
    dataset.num_graphs = 1
    dataset.h0_key = "hamiltonian_0"
    blocks = {
        "0_0_0_0_0": torch.zeros((1, 1)), "1_1_0_0_0": torch.zeros((1, 1)),
        "0_1_0_0_0": torch.zeros((1, 1)), "1_0_0_0_0": torch.zeros((1, 1)),
    }
    num_nodes = 2
    if raw_key == "hamiltonian":
        blocks["0_2_1_0_0"] = torch.zeros((1, 1))  # a 3rd atom type pair -> 3 unique nodes
        num_nodes = 3
    data_dict = {
        AtomicDataDict.POSITIONS_KEY: torch.zeros((num_nodes, 3)),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.ones(num_nodes, dtype=torch.long),
        raw_key: blocks,
    }
    loads = {"count": 0}

    def _load_data_dict(idx):
        loads["count"] += 1
        return data_dict

    dataset._load_data_dict = _load_data_dict
    if raw_key == "hamiltonian":
        def _unexpected_get(idx):
            raise AssertionError("dynamic batch block-cost metadata must not materialize LMDB samples")

        dataset.get = _unexpected_get

    assert dataset.get_dynamic_batch_cost_parts(0)["block"] == block_count
    # A second call is cached; only invalidate_dynamic_batch_costs() forces a reload.
    assert dataset.get_dynamic_batch_cost_parts(0)["block"] == block_count
    assert loads["count"] == 1

    dataset.invalidate_dynamic_batch_costs()
    assert dataset.get_dynamic_batch_cost_parts(0)["block"] == block_count
    assert loads["count"] == 2

import pytest

pytest.importorskip("torch")

import json
import torch

from dptb.plugins.saver import Saver


class DummyModel:
    name = "nnenv"
    model_options = {}


class DummyTrainer:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.rank = 3
        self.is_main_process = True
        self.model = DummyModel()


def _patch_cuda_cache_hooks(monkeypatch, calls):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: calls.append(("sync", str(device))))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty_cache", None)))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 2 * 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 5 * 1024 ** 2)


def test_saver_does_not_clear_cuda_cache_by_default(monkeypatch):
    saver = Saver()
    saver.trainer = DummyTrainer()
    calls = []
    _patch_cuda_cache_hooks(monkeypatch, calls)

    saver._clear_cuda_cache_after_iteration_save("nnenv.iter1000")

    assert ("empty_cache", None) not in calls


def test_saver_clears_cuda_cache_when_enabled(monkeypatch):
    saver = Saver()
    saver.trainer = DummyTrainer()
    calls = []
    _patch_cuda_cache_hooks(monkeypatch, calls)
    monkeypatch.setenv("DPTB_SAVER_CLEAR_CUDA_CACHE_AFTER_ITER_SAVE", "1")

    saver._clear_cuda_cache_after_iteration_save("nnenv.iter1000")

    assert ("empty_cache", None) in calls


class RaisingExpert:
    def state_dict(self):
        raise AssertionError("non-canonical expert state_dict should not be materialized")


class DummyOpt:
    def state_dict(self):
        raise AssertionError("non-canonical optimizer state_dict should not be materialized")


class DummySch:
    def state_dict(self):
        raise AssertionError("non-canonical scheduler state_dict should not be materialized")


class DummyDistExpertModel:
    def __init__(self):
        self.experts = [RaisingExpert(), RaisingExpert()]


class DummyDistExpertTrainer:
    def __init__(self):
        self.rank = 1
        self.world_size = 4
        self.is_main_process = False
        self.distributed_expert = True
        self.local_expert_idx = 0
        self.expert_dp_rank = 1
        self.expert_data_parallel_size = 2
        self.num_experts = 2
        self.model = DummyDistExpertModel()
        self.optimizers = [DummyOpt(), DummyOpt()]
        self.lr_schedulers = [DummySch(), DummySch()]

    def _unwrap_expert_module(self, module):
        return module


def test_noncanonical_expert_dp_rank_sends_empty_checkpoint_payload(monkeypatch):
    saver = Saver()
    saver.trainer = DummyDistExpertTrainer()
    saver.checkpoint_path = "."

    gathered = []

    monkeypatch.setattr("dptb.plugins.saver.dist.is_available", lambda: True)
    monkeypatch.setattr("dptb.plugins.saver.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "dptb.plugins.saver.dist.gather_object",
        lambda obj, object_gather_list=None, dst=0: gathered.append(obj),
    )

    assert saver._gather_dist_states() == (None, None, None)
    assert gathered == [None, None, None]


def test_save_profile_writes_jsonl(monkeypatch, tmp_path):
    saver = Saver()
    saver.trainer = DummyTrainer()
    saver.checkpoint_path = str(tmp_path)
    saver._profile_save_name = "nnenv.iter1"
    monkeypatch.setenv("DPTB_SAVE_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    saver._profile_record("unit_test", value=7)

    rows = (tmp_path / "save_profile_rank3.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["event"] == "unit_test"
    assert row["rank"] == 3
    assert row["save_name"] == "nnenv.iter1"
    assert row["value"] == 7

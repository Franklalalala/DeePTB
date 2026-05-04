import pytest

pytest.importorskip("torch")

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


def test_saver_clears_cuda_cache_after_iteration_save(monkeypatch):
    saver = Saver()
    saver.trainer = DummyTrainer()

    calls = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: calls.append(("sync", str(device))))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty_cache", None)))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 2 * 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 5 * 1024 ** 2)

    saver._clear_cuda_cache_after_iteration_save("nnenv.iter1000")

    assert ("empty_cache", None) in calls

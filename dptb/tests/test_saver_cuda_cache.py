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

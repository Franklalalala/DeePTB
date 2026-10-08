"""Single-GPU Trainer loader workers keep the in-process batch stream."""
from types import SimpleNamespace

import torch

from dptb.data.dataloader import DataLoader
from dptb.nnops.trainer import Trainer
from dptb.utils.torch_geometric import Data


class DeterministicItems:
    deterministic_items = True

    def __init__(self, node_counts):
        self.node_counts = list(node_counts)

    def __len__(self):
        return len(self.node_counts)

    def __getitem__(self, idx):
        n = int(self.node_counts[idx])
        generator = torch.Generator().manual_seed(1000 + int(idx))
        return Data(
            pos=torch.randn((n, 3), generator=generator),
            edge_index=torch.zeros((2, n + 1), dtype=torch.long),
            env_index=torch.zeros((2, n + 2), dtype=torch.long),
            onsitenv_index=torch.zeros((2, n + 3), dtype=torch.long),
            kpoint=torch.zeros((2, 3), dtype=torch.float32),
            eigenvalue=torch.zeros((1, 4), dtype=torch.float32),
        )

    def get_dynamic_batch_cost_parts(self, idx):
        return {"block": int(self.node_counts[idx])}


def _stream(workers):
    options = {"train_num_workers": workers, "data_pin_memory": False, "data_persistent_workers": False}
    loader = DataLoader(
        DeterministicItems([4, 5, 12, 3, 7, 2, 6, 9]), batch_size=3, shuffle=True,
        dynamic_batch={"enabled": True, "mode": "block", "max_cost": 14, "seed": 3},
        **Trainer._train_loader_worker_kwargs(options),
    )
    loader.batch_sampler.set_epoch(2)
    return loader, [(list(batch.__dptb_sample_indices__), batch.pos.clone()) for batch in loader]


def test_worker_options_follow_train_options():
    assert Trainer._train_loader_worker_kwargs({}) == {}
    assert Trainer._train_loader_worker_kwargs({"train_num_workers": 0, "data_pin_memory": True}) == {}
    options = {"train_num_workers": 3, "data_pin_memory": False,
               "data_persistent_workers": True, "data_prefetch_factor": 4}
    assert Trainer._train_loader_worker_kwargs(options) == {
        "num_workers": 3, "pin_memory": False, "persistent_workers": True, "prefetch_factor": 4,
    }


def test_workers_reproduce_the_in_process_batch_stream():
    _, serial = _stream(0)
    loader, parallel = _stream(2)
    assert [indices for indices, _ in serial] == [indices for indices, _ in parallel]
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(serial, parallel))
    exact, reason = Trainer._loader_exact_replay_status(loader)
    assert exact, reason


def test_workers_stay_inexact_without_deterministic_items():
    loader = SimpleNamespace(num_workers=2, dataset=object(),
                             batch_sampler=SimpleNamespace(set_epoch=lambda epoch: None))
    exact, reason = Trainer._loader_exact_replay_status(loader)
    assert not exact and "num_workers" in reason

"""Single-GPU Trainer loader workers keep the in-process batch stream."""
from types import SimpleNamespace

import pytest
import torch

import dptb.nnops.trainer as trainer_mod
from dptb.data.dataloader import DataLoader
from dptb.nnops.trainer import Trainer
from dptb.plugins.saver import Saver
from dptb.tests._trainer_probes import ProbeModel, ProbeTrainer
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


class DeterministicIntegers:
    deterministic_items = True

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return range(5)[index]


class WorkerRestartTrainer(ProbeTrainer):
    """Real restart/epoch/Saver flow with stochastic SGD and real workers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        options = self._train_loader_worker_kwargs(self.train_options)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_datasets, batch_size=1,
            shuffle=self.train_options.get("_test_shuffle", False), **options,
        )
        if self.use_reference:
            self.reference_loader = torch.utils.data.DataLoader(
                kwargs["reference_datasets"], batch_size=1, **options,
            )
        self.optimizer.param_groups[0]["momentum"] = 0.9
        self.loss_trace = []

    def iteration(self, ibatch, ref_batch=None):
        target = ibatch.to(torch.float32) + torch.rand(1)
        if ref_batch is not None:
            target = target + ref_batch.to(torch.float32)
        self.optimizer.zero_grad()
        loss = (self.model.weight - target).square().sum()
        loss.backward()
        self.loss_trace.append(float(loss.detach()))
        return super().iteration(ibatch, ref_batch)


@pytest.mark.parametrize("workers,persistent", [(0, False), (2, False), (2, True)])
@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("checkpoint_kind", ["epoch", "iteration"])
def test_worker_checkpoint_replays_stochastic_training(
    tmp_path, monkeypatch, workers, persistent, with_reference, checkpoint_kind,
):
    _check_worker_restart(
        tmp_path, monkeypatch, workers, persistent, with_reference, checkpoint_kind,
    )


def test_persistent_worker_epoch_restart_preserves_shuffled_batches(tmp_path, monkeypatch):
    _check_worker_restart(tmp_path, monkeypatch, 2, True, True, "epoch", shuffle=True)


def _check_worker_restart(
    tmp_path, monkeypatch, workers, persistent, with_reference, checkpoint_kind,
    shuffle=False,
):
    dataset = DeterministicIntegers()
    options = {
        "max_ckpt": 50, "update_lr_per_iter": False,
        "train_num_workers": workers, "data_pin_memory": False,
        "data_persistent_workers": persistent, "_test_shuffle": shuffle,
    }
    common = {"device": "cpu", "dtype": "float32"}
    reference = dataset if with_reference else None
    torch.manual_seed(20261008)
    original = WorkerRestartTrainer(
        model=ProbeModel(), train_datasets=dataset, reference_datasets=reference,
        train_options=options, common_options=common,
    )
    original.register_plugin(
        Saver(interval=[(1, checkpoint_kind)]), checkpoint_path=str(tmp_path),
    )
    original.run(epochs=3)
    checkpoint = tmp_path / ("probe.ep1.pth" if checkpoint_kind == "epoch" else "probe.iter3.pth")
    committed = len(dataset) if checkpoint_kind == "epoch" else 3

    def load_probe(path, *args, **kwargs):
        model = ProbeModel()
        model.load_state_dict(torch.load(path, weights_only=False)["model_state_dict"], strict=True)
        return model

    monkeypatch.setattr(trainer_mod, "build_model", load_probe)
    torch.manual_seed(999)  # process startup must not determine resumed training
    resumed = WorkerRestartTrainer.restart(
        str(checkpoint), train_datasets=dataset, reference_datasets=reference,
        train_options=options, common_options=common,
    )
    resumed.run(epochs=3)
    assert resumed.processed == original.processed[committed:]
    assert resumed.ref_seen == original.ref_seen[committed:]
    assert resumed.rng_trace == original.rng_trace[committed:]
    assert resumed.loss_trace == original.loss_trace[committed:]
    assert torch.equal(resumed.model.weight, original.model.weight)
    assert torch.equal(
        resumed.optimizer.state[resumed.model.weight]["momentum_buffer"],
        original.optimizer.state[original.model.weight]["momentum_buffer"],
    )
    assert resumed.lr_scheduler.state_dict() == original.lr_scheduler.state_dict()

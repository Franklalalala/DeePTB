import pytest

torch = pytest.importorskip("torch")
from torch import nn

from dptb.nnops import trainer as trainer_mod
from dptb.nnops.trainer import Trainer


class FakeBatch:
    def __init__(self, name, x, batch_cost=None):
        self.name = name
        self.x = torch.tensor(float(x))
        self.__slices__ = {}
        self.__cumsum__ = {}
        self.__cat_dims__ = {}
        self.__num_nodes_list__ = []
        self.__data_class__ = FakeBatch
        if batch_cost is not None:
            self.__dptb_batch_cost__ = batch_cost
            self.__dptb_batch_num_graphs__ = 1
            self.__dptb_batch_num_nodes__ = 2
            self.__dptb_batch_num_edges__ = 3
            self.__dptb_batch_max_item_cost__ = batch_cost

    def to(self, device):
        return self


class TinyModel(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.events = events

    def forward(self, batch):
        self.events.append(f"forward:{batch['name']}")
        batch = dict(batch)
        batch["pred"] = self.weight * batch["x"]
        return batch


class RecordingLoss:
    def __init__(self, name, events, model):
        self.name = name
        self.events = events
        self.model = model
        self.calls = []

    def __call__(self, batch, batch_for_loss):
        self.calls.append(batch["name"])
        self.events.append(f"{self.name}_loss:{batch['name']}")
        if self.name == "reference":
            assert self.model.weight.grad is not None
            assert "train_backward:train" in self.events
        loss = batch["pred"].sum()
        loss.register_hook(
            lambda grad, name=self.name, batch_name=batch["name"]: self.events.append(
                f"{name}_backward:{batch_name}"
            )
        )
        return loss


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


def _batch_to_dict(batch):
    return {"name": batch.name, "x": batch.x}


def _fake_trainer(monkeypatch):
    monkeypatch.setattr(trainer_mod.AtomicData, "to_AtomicDataDict", _batch_to_dict)
    events = []
    model = TinyModel(events)
    optimizer = CountingSGD(model.parameters())
    observed_states = []

    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.device = torch.device("cpu")
    trainer.clip_grad_norm = 1000.0
    trainer.update_lr_per_iter = False
    trainer.iter = 5
    trainer.stats = {"train_loss": {"latest_avg_iter_loss": torch.tensor(0.0)}}
    trainer.train_lossfunc = RecordingLoss("train", events, model)
    trainer.reference_lossfunc = RecordingLoss("reference", events, model)
    trainer.call_plugins = (
        lambda queue_name, time, **state: observed_states.append(
            (queue_name, time, state)
        )
    )
    return trainer, events, observed_states


def test_iteration_uses_reference_loss_and_backwards_train_before_reference(monkeypatch):
    trainer, events, observed_states = _fake_trainer(monkeypatch)

    loss = trainer.iteration(
        FakeBatch("train", 2.0, batch_cost=7),
        FakeBatch("reference", 3.0),
    )

    assert trainer.train_lossfunc.calls == ["train"]
    assert trainer.reference_lossfunc.calls == ["reference"]
    assert events.index("train_backward:train") < events.index(
        "reference_loss:reference"
    )
    assert events.index("reference_loss:reference") < events.index(
        "reference_backward:reference"
    )
    assert trainer.optimizer.step_calls == 1
    assert loss.item() == pytest.approx(5.0)
    assert observed_states[0][2]["batch_cost"] == 7
    assert observed_states[0][2]["batch_num_nodes"] == 2


def test_iteration_without_reference_keeps_single_train_backward(monkeypatch):
    trainer, events, observed_states = _fake_trainer(monkeypatch)

    loss = trainer.iteration(FakeBatch("train", 4.0, batch_cost=11))

    assert trainer.train_lossfunc.calls == ["train"]
    assert trainer.reference_lossfunc.calls == []
    assert events == ["forward:train", "train_loss:train", "train_backward:train"]
    assert trainer.optimizer.step_calls == 1
    assert loss.item() == pytest.approx(4.0)
    assert observed_states[0][2]["batch_cost"] == 11


class CountingIterable:
    def __init__(self, items):
        self.items = list(items)
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(self.items)


def test_epoch_cycles_persistent_reference_iterator():
    trainer = Trainer.__new__(Trainer)
    trainer.ep = 3
    trainer.use_reference = True
    trainer.train_loader = CountingIterable(["train0", "train1", "train2", "train3", "train4"])
    trainer.reference_loader = CountingIterable(["ref0", "ref1"])
    seen = []
    trainer.iteration = lambda train, ref=None: seen.append((train, ref))

    trainer.epoch()

    assert seen == [
        ("train0", "ref0"),
        ("train1", "ref1"),
        ("train2", "ref0"),
        ("train3", "ref1"),
        ("train4", "ref0"),
    ]
    assert trainer.reference_loader.iter_calls == 3


def test_epoch_without_reference_loader_calls_iteration_with_train_batches_only():
    trainer = Trainer.__new__(Trainer)
    trainer.ep = 3
    trainer.use_reference = False
    trainer.train_loader = CountingIterable(["train0", "train1"])
    seen = []
    trainer.iteration = lambda train, ref=None: seen.append((train, ref))

    trainer.epoch()

    assert seen == [("train0", None), ("train1", None)]
